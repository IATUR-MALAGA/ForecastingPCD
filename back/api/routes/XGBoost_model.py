from __future__ import annotations

from typing import Any, Optional, Dict, List
import re
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from front.utils.utils import create_dataframe_based_on_selection
from front.utils.utils import _safe_alias

from back.models.XGBoost.xgboost_model import (
    best_xgboost_params,
    create_xgboost_model,
)
from back.models.XGBoost.xgboost_statistics import compute_metrics

router = APIRouter(prefix="/models/xgboost", tags=["XGBoost_model"])


# ------------ Schemas ------------

class FilterSelection(BaseModel):
    table: str
    col: str
    values: list[str]


class XGBoostRunRequest(BaseModel):
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None

    train_ratio: float = Field(0.70, gt=0.0, lt=1.0)

    # Hiperparámetros
    auto_params: bool = True
    xgb_params: Optional[Dict[str, Any]] = None

    # Forecasting (lags)
    use_target_lags: bool = True
    max_lag: int = Field(12, ge=0)
    recursive_forecast: bool = True  # multi-step sin leakage

    return_df: bool = True


class XGBoostRunResponse(BaseModel):
    y_col: str
    predictors: list[str]
    feature_cols: list[str]

    n: int
    n_train: int
    n_test: int

    xgb_params: Dict[str, Any]

    mape: float
    rmse: float
    mae: float

    y_pred: list[float]
    df: Optional[list[dict[str, Any]]] = None


# ------------ Helpers (lags + forecast recursivo) ------------

_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<k>\d+)$")


def _add_target_lags(df: pd.DataFrame, y_col: str, max_lag: int) -> pd.DataFrame:
    if max_lag <= 0:
        return df
    out = df.copy()
    for k in range(1, max_lag + 1):
        out[f"{y_col}_lag{k}"] = out[y_col].shift(k)
    return out


def _recursive_predict(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    max_lag: int,
) -> pd.Series:
    """
    Predicción multi-step recursiva:
    - Los lags de y se alimentan con valores predichos conforme avanzamos.
    - Las features no-lag se toman de test (exógenas, calendario, etc.).
    """
    if max_lag <= 0:
        raise ValueError("recursive_forecast=True requiere max_lag > 0.")

    # historia inicial con los últimos valores reales del train
    hist = list(train[y_col].to_numpy())
    if len(hist) < max_lag:
        raise ValueError(f"No hay suficiente histórico para max_lag={max_lag} (train tiene {len(hist)} filas).")

    preds = []
    idxs = list(test.index)

    for i in range(len(test)):
        row = {}
        for col in feature_cols:
            m = _LAG_RE.match(col)
            if m and m.group("base") == y_col:
                k = int(m.group("k"))
                row[col] = hist[-k]
            else:
                # feature exógena/categoría/calendario/etc. (debe existir en test)
                if col not in test.columns:
                    raise ValueError(f"Feature '{col}' no está en test. Revisa tu dataframe/selección.")
                row[col] = test.iloc[i][col]

        X_step = pd.DataFrame([row], index=[idxs[i]])
        y_hat = float(model.predict(X_step)[0])
        preds.append(y_hat)
        hist.append(y_hat)

    return pd.Series(preds, index=test.index, name=y_col)


# ------------ Endpoint ------------

@router.post("/run", response_model=XGBoostRunResponse)
def xgboost_run(req: XGBoostRunRequest):
    try:
        df = create_dataframe_based_on_selection(
            target_var=req.target_var,
            predictors=req.predictors,
            filters_by_var={k: [f.model_dump() for f in v] for k, v in (req.filters_by_var or {}).items()},
        )
        if df is None or len(df) == 0:
            raise HTTPException(status_code=422, detail="El dataframe resultante está vacío")

        # alias “seguros”
        y_col = _safe_alias(req.target_var)
        predictors = [_safe_alias(c) for c in (req.predictors or [])]

        if y_col not in df.columns:
            raise HTTPException(status_code=422, detail=f"No existe la columna objetivo '{y_col}' en el dataframe")

        # 1) Lags de y (recomendado para forecasting)
        if req.use_target_lags and req.max_lag > 0:
            df = _add_target_lags(df, y_col=y_col, max_lag=req.max_lag)
            df = df.dropna(axis=0)  # quita las primeras filas por shift()

        if len(df) < 3:
            raise HTTPException(status_code=422, detail="Tras crear lags/dropna, quedan muy pocas filas para entrenar")

        # 2) split
        n = len(df)
        n_train = int(n * req.train_ratio)
        n_test = n - n_train

        if n_train <= 0 or n_test <= 0:
            raise HTTPException(
                status_code=422,
                detail=f"Split inválido: n={n}, n_train={n_train}, n_test={n_test}. Ajusta train_ratio."
            )
        train = df.iloc[:n_train]
        test = df.iloc[n_train:n_train + n_test]

        # 3) Features reales usadas por XGBoost = todo menos y
        feature_cols = [c for c in df.columns if c != y_col]
        if len(feature_cols) == 0:
            raise HTTPException(
                status_code=422,
                detail="No hay features (solo existe y). Activa use_target_lags o añade predictors."
            )
        # 4) Hiperparámetros
        if req.auto_params:
            best_params = best_xgboost_params(
                df=df,
                exog_cols=feature_cols,          # truco: forzamos a usar drop(y) siempre
                column_y=y_col,
                periodos_a_predecir=n_test,
                random_state=42,
            )
            # “base” + best_params
            xgb_params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": 42,
                **best_params,
            }
        else:
            xgb_params = req.xgb_params or {
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": 42,
            }
        # 5) Entrena
        model_fit = create_xgboost_model(
            train=train,
            exog_cols=feature_cols,   # usamos features reales
            column_y=y_col,
            xgb_params=xgb_params,
        )
        # 6) Predice
        if req.recursive_forecast and req.use_target_lags and req.max_lag > 0:
            pred_test = _recursive_predict(
                model_fit, train=train, test=test,
                y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag
            )
        else:
            # predicción “directa” usando columnas de test tal cual (ojo: puede filtrar y futuro si los lags vienen de y real)
            X_test = test[feature_cols]
            preds = model_fit.predict(X_test)
            pred_test = pd.Series(preds, index=test.index, name=y_col)

        # 7) métricas
        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        return XGBoostRunResponse(
            y_col=y_col,
            predictors=predictors,
            feature_cols=feature_cols,
            n=n,
            n_train=n_train,
            n_test=n_test,
            xgb_params={k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in xgb_params.items()},
            mape=float(mape),
            rmse=float(rmse),
            mae=float(mae),
            y_pred=[float(x) for x in list(pred_test.values)],
            df=df.to_dict(orient="records") if req.return_df else None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error XGBoost run: {e}")
