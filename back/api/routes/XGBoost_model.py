from __future__ import annotations

from typing import Any, Optional, Dict, List
import re
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from back.api.routes.SARIMAX_model import FilterSelection
from front.utils.utils import create_dataframe_based_on_selection
from front.utils.utils import _safe_alias

from back.models.XGBoost.xgboost_model import (
    best_xgboost_params,
    create_xgboost_model,
)
from back.models.XGBoost.xgboost_statistics import compute_metrics

router = APIRouter(prefix="/models/xgboost", tags=["XGBoost_model"])


# ------------ Schemas ------------

class XGBoostRunRequest(BaseModel):
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None

    train_ratio: float = Field(0.70, gt=0.0, lt=1.0)

    auto_params: bool = True
    xgb_params: Optional[Dict[str, Any]] = None

    use_target_lags: bool = True
    max_lag: int = Field(12, ge=0)
    recursive_forecast: bool = True

    horizon: int = 1               # <-- nuevo (como SARIMAX)
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
    y_forecast: list[float]        # <-- nuevo
    horizon: int                   # <-- nuevo
    n_obs: int                     # <-- nuevo (len(df_hist))
    df: Optional[list[dict[str, Any]]] = None



# ------------ Helpers (lags + forecast recursivo) ------------

_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<k>\d+)$")


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


from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import HTTPException

from front.utils.utils import _safe_alias, create_dataframe_based_on_selection, _find_col, add_fourier_annual_terms
from back.models.XGBoost.xgboost_model import best_xgboost_params, create_xgboost_model
from back.models.XGBoost.xgboost_statistics import compute_metrics


def _raise_422(detail: str) -> None:
    raise HTTPException(status_code=422, detail=detail)


def _detect_calendar_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    dia_col = _find_col(df, "dia", _safe_alias("dia"))
    mes_col = _find_col(df, "mes", _safe_alias("mes"))
    ano_col = _find_col(df, "anio", "año", "ano", _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano"))
    return dia_col, mes_col, ano_col


def _sort_temporally(df: pd.DataFrame, ano_col: Optional[str], mes_col: Optional[str], dia_col: Optional[str]) -> pd.DataFrame:
    sort_cols = [c for c in [ano_col, mes_col, dia_col] if c and c in df.columns]
    if sort_cols:
        # mergesort = estable/determinista
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def _force_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _split_hist_future(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_hist = df[y_col].notna()
    df_hist = df.loc[mask_hist].copy()
    df_future_all = df.loc[~mask_hist].copy()
    if len(df_hist) < 3:
        _raise_422("Histórico insuficiente para entrenar XGBoost")
    return df_hist, df_future_all


def _get_horizon(req, df_future_all: pd.DataFrame) -> int:
    horizon = int(getattr(req, "horizon", 1) or 1)
    if horizon < 1:
        _raise_422("horizon debe ser >= 1")
    if len(df_future_all) < horizon:
        _raise_422(
            f"No hay suficientes filas futuras para horizon={horizon}. "
            f"Filas futuras disponibles: {len(df_future_all)}."
        )
    return horizon


def _add_target_lags_hist(df_hist: pd.DataFrame, y_col: str, max_lag: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Crea lags SOLO en histórico (y conocida). No toca el futuro.
    """
    if max_lag <= 0:
        return df_hist.copy(), []

    out = df_hist.copy()
    lag_cols = []
    for k in range(1, max_lag + 1):
        c = f"{y_col}_lag{k}"
        out[c] = out[y_col].shift(k)
        lag_cols.append(c)

    # elimina solo las filas iniciales sin lags (no afecta al futuro)
    out = out.dropna(subset=lag_cols).copy()
    if len(out) < 3:
        _raise_422("Tras crear lags en histórico, quedan muy pocas filas para entrenar")
    return out, lag_cols


@router.post("/run", response_model=XGBoostRunResponse)
def xgboost_run(req: XGBoostRunRequest):
    try:
        df = create_dataframe_based_on_selection(
            target_var=req.target_var,
            predictors=req.predictors,
            filters_by_var={k: [f.model_dump() for f in v] for k, v in (req.filters_by_var or {}).items()},
        )
        if df is None or df.empty:
            _raise_422("El dataframe resultante está vacío")

        # alias “seguros”
        y_col = _safe_alias(req.target_var)
        predictors = [_safe_alias(c) for c in (req.predictors or [])]

        if y_col not in df.columns:
            _raise_422(f"No existe la columna objetivo '{y_col}' en el dataframe")

        # (opcional) orden temporal como SARIMAX
        dia_col, mes_col, ano_col = _detect_calendar_cols(df)
        df = _sort_temporally(df, ano_col, mes_col, dia_col)

        # fuerza numérico en y y predictores (si son numéricos)
        df = _force_numeric(df, [y_col] + predictors)

        # split histórico/futuro como SARIMAX
        df_hist, df_future_all = _split_hist_future(df, y_col)
        horizon = _get_horizon(req, df_future_all)
        df_future = df_future_all.iloc[:horizon].copy()

        # valida predictores en df
        missing_pred = [c for c in predictors if c not in df.columns]
        if missing_pred:
            _raise_422(f"Faltan columnas predictoras en df: {missing_pred}")

        # policy similar a SARIMAX:
        # - histórico: rellena NaNs en exógenas (evita problemas/variabilidad)
        # - futuro: si hay NaNs en exógenas, fallar (horizonte no usable)
        if predictors:
            df_hist.loc[:, predictors] = df_hist[predictors].fillna(0.0)
            if df_future[predictors].isna().any().any():
                _raise_422("Hay NaNs en predictores futuros: no se puede predecir")

        # crea dataset supervisado SOLO en histórico (lags)
        df_hist_sup, lag_cols = (df_hist.copy(), [])
        if req.use_target_lags and req.max_lag > 0:
            df_hist_sup, lag_cols = _add_target_lags_hist(df_hist, y_col=y_col, max_lag=req.max_lag)

        # features = predictores + lags (explícito, sin “todo menos y”)
        feature_cols = predictors + lag_cols
        if not feature_cols:
            _raise_422("No hay features. Activa use_target_lags o añade predictors.")

        # split train/test SOLO histórico supervisado
        n = len(df_hist_sup)
        n_train = int(n * req.train_ratio)
        n_test = n - n_train
        if n_train <= 0 or n_test <= 0:
            _raise_422(f"Split inválido (histórico): n={n}, n_train={n_train}, n_test={n_test}. Ajusta train_ratio.")

        train = df_hist_sup.iloc[:n_train].copy()
        test = df_hist_sup.iloc[n_train:n_train + n_test].copy()

        # hiperparámetros (ojo: usar SOLO histórico supervisado)
        if req.auto_params:
            best_params = best_xgboost_params(
                df=df_hist_sup,
                exog_cols=feature_cols,
                column_y=y_col,
                periodos_a_predecir=n_test,
                random_state=42,
            )
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

        # fit para métricas (train)
        model_fit = create_xgboost_model(
            train=train,
            exog_cols=feature_cols,
            column_y=y_col,
            xgb_params=xgb_params,
        )

        # predicción sobre test (métricas)
        if req.recursive_forecast and req.use_target_lags and req.max_lag > 0:
            pred_test = _recursive_predict(
                model_fit, train=train, test=test,
                y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag
            )
        else:
            X_test = test[feature_cols]
            pred_test = pd.Series(model_fit.predict(X_test), index=test.index, name=y_col)

        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        # fit full histórico (supervisado) + forecast futuro (como SARIMAX)
        model_fit_full = create_xgboost_model(
            train=df_hist_sup,
            exog_cols=feature_cols,
            column_y=y_col,
            xgb_params=xgb_params,
        )

        if req.use_target_lags and req.max_lag > 0:
            # forecast recursivo usando df_hist real como historia y df_future como exógenas
            y_forecast = _recursive_predict(
                model_fit_full,
                train=df_hist,      # historia real de y
                test=df_future,     # exógenas de futuro (y NaN no molesta)
                y_col=y_col,
                feature_cols=feature_cols,
                max_lag=req.max_lag,
            )
            y_forecast_list = [float(x) for x in y_forecast.values]
        else:
            # sin lags: forecast directo (requiere que df_future tenga todas las features)
            X_future = df_future[feature_cols]
            y_forecast_list = [float(x) for x in model_fit_full.predict(X_future)]
        print (f"y_forecast_list: {y_forecast_list}")
        return XGBoostRunResponse(
            y_col=y_col,
            predictors=predictors,
            feature_cols=feature_cols,
            n=len(df),
            n_train=n_train,
            n_test=n_test,
            xgb_params={k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in xgb_params.items()},
            mape=float(mape),
            rmse=float(rmse),
            mae=float(mae),
            y_pred=[float(x) for x in list(pred_test.values)],
            y_forecast=y_forecast_list,
            horizon=horizon,
            n_obs=len(df_hist),
            df=df.to_dict(orient="records") if req.return_df else None,
        )
    

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error XGBoost run: {e}")
