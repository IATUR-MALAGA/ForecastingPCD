from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from back.api.routes.SARIMAX_model import FilterSelection
from back.config import settings
from back.models.XGBoost.xgboost_model import best_xgboost_params, create_xgboost_model
from back.models.XGBoost.xgboost_statistics import compute_metrics
from front.utils.utils import _find_col, _safe_alias, create_dataframe_based_on_selection


COMMON_CFG = settings.get("models.common", {})
XGBOOST_CFG = settings.get("models.xgboost", {})
XGB_ENGINE_CFG = XGBOOST_CFG.get("engine", {}) if isinstance(XGBOOST_CFG, dict) else {}
XGB_DEFAULT_PARAMS_CFG = XGBOOST_CFG.get("default_params", {}) if isinstance(XGBOOST_CFG, dict) else {}


router = APIRouter(
    prefix=settings.get("server.routes.xgboost_prefix", "/models/xgboost"),
    tags=["XGBoost_model"],
)


class ScenarioOverride(BaseModel):
    var: str
    op: Literal["set", "add", "mul", "pct"]
    value: float
    start: Optional[str] = None
    end: Optional[str] = None


class ScenarioFutureValue(BaseModel):
    var: str
    date: str
    value: float


class ScenarioWindow(BaseModel):
    start: str
    end: str


class XGBoostRunRequest(BaseModel):
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None

    train_ratio: float = Field(float(COMMON_CFG.get("train_ratio", 0.70)), gt=0.0, lt=1.0)

    auto_params: bool = bool(XGBOOST_CFG.get("auto_params", True))
    xgb_params: Optional[Dict[str, Any]] = None

    use_target_lags: bool = bool(XGBOOST_CFG.get("use_target_lags", True))
    max_lag: int = Field(int(XGBOOST_CFG.get("max_lag", 12)), ge=0)
    recursive_forecast: bool = bool(XGBOOST_CFG.get("recursive_forecast", True))

    horizon: int = int(COMMON_CFG.get("horizon", 1))
    return_df: bool = bool(COMMON_CFG.get("return_df", True))

    scenario_mode: Optional[Literal["future", "past"]] = None
    scenario_overrides: list[ScenarioOverride] = Field(default_factory=list)
    scenario_future_values: list[ScenarioFutureValue] = Field(default_factory=list)
    scenario_window: Optional[ScenarioWindow] = None


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
    y_forecast: list[float]
    horizon: int
    n_obs: int
    y_true: Optional[list[float]] = None
    window: Optional[dict[str, str]] = None
    df: Optional[list[dict[str, Any]]] = None


_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<k>\d+)$")


def _raise_422(detail: str) -> None:
    raise HTTPException(status_code=422, detail=detail)


def _recursive_predict(model, train: pd.DataFrame, test: pd.DataFrame, y_col: str, feature_cols: list[str], max_lag: int) -> pd.Series:
    if max_lag <= 0:
        raise ValueError("recursive_forecast=True requiere max_lag > 0.")
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
                row[col] = hist[-int(m.group("k"))]
            else:
                if col not in test.columns:
                    raise ValueError(f"Feature '{col}' no está en test. Revisa tu dataframe/selección.")
                row[col] = test.iloc[i][col]
        X_step = pd.DataFrame([row], index=[idxs[i]])
        y_hat = float(model.predict(X_step)[0])
        preds.append(y_hat)
        hist.append(y_hat)
    return pd.Series(preds, index=test.index, name=y_col)


def _build_time_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    dia_col = _find_col(df, "dia", _safe_alias("dia"))
    mes_col = _find_col(df, "mes", _safe_alias("mes"))
    ano_col = _find_col(df, "anio", "año", "ano", _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano"))

    out = df.copy()
    if ano_col and mes_col and ano_col in out.columns and mes_col in out.columns:
        day = out[dia_col] if dia_col and dia_col in out.columns else 1
        out["__dt"] = pd.to_datetime(dict(year=pd.to_numeric(out[ano_col], errors="coerce"), month=pd.to_numeric(out[mes_col], errors="coerce"), day=pd.to_numeric(day, errors="coerce")), errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["__dt"] = pd.to_datetime(out.index)
    elif "fecha" in out.columns:
        out["__dt"] = pd.to_datetime(out["fecha"], errors="coerce")
    else:
        _raise_422("No fue posible construir un time_index robusto")

    out = out.dropna(subset=["__dt"]).sort_values("__dt", kind="mergesort").reset_index(drop=True)
    return out, dia_col, mes_col


def _apply_overrides(df: pd.DataFrame, overrides: list[ScenarioOverride], predictors: list[str]) -> pd.DataFrame:
    out = df.copy()
    for ov in overrides:
        var = _safe_alias(ov.var)
        if var not in predictors or var not in out.columns:
            continue
        mask = pd.Series(True, index=out.index)
        if ov.start:
            mask &= pd.to_datetime(out["__dt"]) >= pd.to_datetime(ov.start)
        if ov.end:
            mask &= pd.to_datetime(out["__dt"]) <= pd.to_datetime(ov.end)
        if ov.op == "set":
            out.loc[mask, var] = float(ov.value)
        elif ov.op == "add":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") + float(ov.value)
        elif ov.op == "mul":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") * float(ov.value)
        elif ov.op == "pct":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") * (1 + (float(ov.value) / 100.0))
    return out


def _add_target_lags_hist(df_hist: pd.DataFrame, y_col: str, max_lag: int) -> Tuple[pd.DataFrame, List[str]]:
    if max_lag <= 0:
        return df_hist.copy(), []
    out = df_hist.copy()
    lag_cols = []
    for k in range(1, max_lag + 1):
        c = f"{y_col}_lag{k}"
        out[c] = out[y_col].shift(k)
        lag_cols.append(c)
    out = out.dropna(subset=lag_cols).copy()
    if len(out) < int(COMMON_CFG.get("min_historical_rows", 3)):
        _raise_422("Tras crear lags en histórico, quedan muy pocas filas para entrenar")
    return out, lag_cols


def _infer_future_index(df_hist: pd.DataFrame, horizon: int, monthly_hint: bool) -> pd.DatetimeIndex:
    last_dt = pd.to_datetime(df_hist["__dt"]).max()
    if monthly_hint:
        return pd.date_range(start=(last_dt + pd.offsets.MonthBegin(1)).normalize(), periods=horizon, freq="MS")
    return pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=horizon, freq="D")


def _validate_missing_future(df_future: pd.DataFrame, predictors: list[str]) -> None:
    missing = []
    for var in predictors:
        miss_rows = df_future[df_future[var].isna()]["__dt"] if var in df_future.columns else []
        for dt in miss_rows:
            missing.append({"var": var, "date": pd.to_datetime(dt).strftime("%Y-%m-%d")})
    if missing:
        raise HTTPException(status_code=400, detail={"detail": "Missing future exogenous values", "missing": missing})


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

        y_col = _safe_alias(req.target_var)
        predictors = [_safe_alias(c) for c in (req.predictors or [])]
        if y_col not in df.columns:
            _raise_422(f"No existe la columna objetivo '{y_col}' en el dataframe")

        df, dia_col, mes_col = _build_time_index(df)
        for c in [y_col] + predictors:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)

        mask_hist = df[y_col].notna()
        df_hist = df.loc[mask_hist].copy()
        if len(df_hist) < int(COMMON_CFG.get("min_historical_rows", 3)):
            _raise_422("Histórico insuficiente para entrenar XGBoost")

        missing_pred = [c for c in predictors if c not in df.columns]
        if missing_pred:
            _raise_422(f"Faltan columnas predictoras en df: {missing_pred}")

        if req.scenario_mode == "past":
            if req.scenario_window is None:
                _raise_422("scenario_window es obligatorio cuando scenario_mode='past'")
            ws = pd.to_datetime(req.scenario_window.start)
            we = pd.to_datetime(req.scenario_window.end)
            train = df_hist[pd.to_datetime(df_hist["__dt"]) < ws].copy()
            test = df_hist[(pd.to_datetime(df_hist["__dt"]) >= ws) & (pd.to_datetime(df_hist["__dt"]) <= we)].copy()
            if train.empty or test.empty:
                _raise_422("No hay datos suficientes para la ventana solicitada")
            if predictors:
                train.loc[:, predictors] = train[predictors].fillna(0.0)
                test = _apply_overrides(test, req.scenario_overrides, predictors)
                if test[predictors].isna().any().any():
                    _raise_422("Hay NaNs en predictores de la ventana de escenario")

            df_hist_sup, lag_cols = (train.copy(), [])
            if req.use_target_lags and req.max_lag > 0:
                df_hist_sup, lag_cols = _add_target_lags_hist(train, y_col=y_col, max_lag=req.max_lag)
            feature_cols = predictors + lag_cols
            if not feature_cols:
                _raise_422("No hay features. Activa use_target_lags o añade predictors.")

            xgb_params = req.xgb_params or {
                **XGB_DEFAULT_PARAMS_CFG,
                "objective": XGB_ENGINE_CFG.get("objective", "reg:squarederror"),
                "tree_method": XGB_ENGINE_CFG.get("tree_method", "hist"),
                "n_jobs": int(XGB_ENGINE_CFG.get("n_jobs", 1)),
                "random_state": int(XGBOOST_CFG.get("random_state", 42)),
            }
            model_fit = create_xgboost_model(train=df_hist_sup, exog_cols=feature_cols, column_y=y_col, xgb_params=xgb_params)
            if req.recursive_forecast and req.use_target_lags and req.max_lag > 0:
                y_forecast = _recursive_predict(model_fit, train=train, test=test, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
            else:
                y_forecast = pd.Series(model_fit.predict(test[feature_cols]), index=test.index, name=y_col)
            mape, rmse, mae = compute_metrics(pred=y_forecast, df_test=test, indicador=y_col)
            return XGBoostRunResponse(
                y_col=y_col,
                predictors=predictors,
                feature_cols=feature_cols,
                n=len(df),
                n_train=len(train),
                n_test=len(test),
                xgb_params={k: (float(v) if isinstance(v, np.floating) else v) for k, v in xgb_params.items()},
                mape=float(mape),
                rmse=float(rmse),
                mae=float(mae),
                y_pred=[float(x) for x in list(y_forecast.values)],
                y_forecast=[float(x) for x in list(y_forecast.values)],
                horizon=len(test),
                n_obs=len(df_hist),
                y_true=test[y_col].astype(float).tolist(),
                window={"start": ws.strftime("%Y-%m-%d"), "end": we.strftime("%Y-%m-%d")},
                df=df.to_dict(orient="records") if req.return_df else None,
            )

        horizon = int(req.horizon)
        if horizon < 1:
            _raise_422("horizon debe ser >= 1")

        monthly_hint = mes_col is not None and dia_col is None
        future_index = _infer_future_index(df_hist, horizon, monthly_hint)
        df_future = pd.DataFrame({"__dt": future_index})
        existing_future = df.loc[~mask_hist].copy()
        if not existing_future.empty:
            existing_future = existing_future.drop_duplicates(subset=["__dt"], keep="last")
            df_future = df_future.merge(existing_future[["__dt", *[c for c in predictors if c in existing_future.columns]]], on="__dt", how="left")

        for fv in req.scenario_future_values:
            var = _safe_alias(fv.var)
            if var in predictors:
                df_future.loc[pd.to_datetime(df_future["__dt"]) == pd.to_datetime(fv.date), var] = float(fv.value)
        if predictors:
            df_future = _apply_overrides(df_future, req.scenario_overrides if req.scenario_mode == "future" else [], predictors)
            _validate_missing_future(df_future, predictors)
            df_hist.loc[:, predictors] = df_hist[predictors].fillna(0.0)

        df_hist_sup, lag_cols = (df_hist.copy(), [])
        if req.use_target_lags and req.max_lag > 0:
            df_hist_sup, lag_cols = _add_target_lags_hist(df_hist, y_col=y_col, max_lag=req.max_lag)
        feature_cols = predictors + lag_cols
        if not feature_cols:
            _raise_422("No hay features. Activa use_target_lags o añade predictors.")

        n = len(df_hist_sup)
        n_train = int(n * req.train_ratio)
        n_test = n - n_train
        if n_train <= 0 or n_test <= 0:
            _raise_422(f"Split inválido (histórico): n={n}, n_train={n_train}, n_test={n_test}. Ajusta train_ratio.")

        train = df_hist_sup.iloc[:n_train].copy()
        test = df_hist_sup.iloc[n_train:n_train + n_test].copy()

        if req.auto_params:
            best_params = best_xgboost_params(df=df_hist_sup, exog_cols=feature_cols, column_y=y_col, periodos_a_predecir=n_test, random_state=int(XGBOOST_CFG.get("random_state", 42)))
            xgb_params = {"objective": XGB_ENGINE_CFG.get("objective", "reg:squarederror"), "tree_method": XGB_ENGINE_CFG.get("tree_method", "hist"), "n_jobs": int(XGB_ENGINE_CFG.get("n_jobs", 1)), "random_state": int(XGBOOST_CFG.get("random_state", 42)), **best_params}
        else:
            xgb_params = req.xgb_params or {**XGB_DEFAULT_PARAMS_CFG, "objective": XGB_ENGINE_CFG.get("objective", "reg:squarederror"), "tree_method": XGB_ENGINE_CFG.get("tree_method", "hist"), "n_jobs": int(XGB_ENGINE_CFG.get("n_jobs", 1)), "random_state": int(XGBOOST_CFG.get("random_state", 42))}

        model_fit = create_xgboost_model(train=train, exog_cols=feature_cols, column_y=y_col, xgb_params=xgb_params)
        if req.recursive_forecast and req.use_target_lags and req.max_lag > 0:
            pred_test = _recursive_predict(model_fit, train=train, test=test, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
        else:
            pred_test = pd.Series(model_fit.predict(test[feature_cols]), index=test.index, name=y_col)
        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        model_fit_full = create_xgboost_model(train=df_hist_sup, exog_cols=feature_cols, column_y=y_col, xgb_params=xgb_params)
        if req.use_target_lags and req.max_lag > 0:
            y_forecast = _recursive_predict(model_fit_full, train=df_hist, test=df_future, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
            y_forecast_list = [float(x) for x in y_forecast.values]
        else:
            y_forecast_list = [float(x) for x in model_fit_full.predict(df_future[feature_cols])]

        return XGBoostRunResponse(
            y_col=y_col,
            predictors=predictors,
            feature_cols=feature_cols,
            n=len(df),
            n_train=n_train,
            n_test=n_test,
            xgb_params={k: (float(v) if isinstance(v, np.floating) else v) for k, v in xgb_params.items()},
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
