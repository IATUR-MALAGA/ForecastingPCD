from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from back.api.routes.SARIMAX_model import (
    COMMON_CFG,
    FilterSelection,
    _detect_calendar_cols as sarimax_detect_calendar_cols,
    _force_numeric as sarimax_force_numeric,
    _get_dataframe,
    _get_horizon,
    _maybe_add_fourier,
    _prepare_exog_cols,
    _prepare_exog_matrices,
    _select_params,
    _sort_temporally as sarimax_sort_temporally,
    _split_hist_future as sarimax_split_hist_future,
    _train_test_split_hist,
)
from back.api.routes.XGBoost_model import (
    XGB_DEFAULT_PARAMS_CFG,
    XGB_ENGINE_CFG,
    XGBOOST_CFG,
    _add_target_lags_hist,
    _detect_calendar_cols as xgb_detect_calendar_cols,
    _force_numeric as xgb_force_numeric,
    _recursive_predict,
    _sort_temporally as xgb_sort_temporally,
    _split_hist_future as xgb_split_hist_future,
)
from back.config import settings
from back.models.SARIMAX.sarimax_model import create_sarimax_model
from back.models.SARIMAX.sarimax_statistics import compute_metrics as sarimax_metrics
from back.models.XGBoost.xgboost_model import best_xgboost_params, create_xgboost_model
from back.models.XGBoost.xgboost_statistics import compute_metrics as xgb_metrics
from front.utils.utils import _safe_alias, create_dataframe_based_on_selection


router = APIRouter(
    prefix=settings.get("server.routes.scenarios_prefix", "/models/scenarios"),
    tags=["Scenarios"],
)


class ScenarioOverride(BaseModel):
    var: str
    op: Literal["set", "add", "mul", "pct"]
    value: float
    start: Optional[str] = None
    end: Optional[str] = None


class ScenariosRunRequest(BaseModel):
    model: Literal["sarimax", "xgboost"]
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None
    scenario_overrides: list[ScenarioOverride] = Field(default_factory=list)

    train_ratio: float = Field(float(COMMON_CFG.get("train_ratio", 0.70)), gt=0.0, lt=1.0)
    horizon: int = int(COMMON_CFG.get("horizon", 1))
    return_df: bool = bool(COMMON_CFG.get("return_df", True))

    # SARIMAX
    auto_params: bool = True
    s: int = 12
    order: Optional[tuple[int, int, int]] = None
    seasonal_order: Optional[tuple[int, int, int, int]] = None

    # XGBoost
    xgb_params: Optional[dict[str, Any]] = None
    use_target_lags: bool = bool(XGBOOST_CFG.get("use_target_lags", True))
    max_lag: int = Field(int(XGBOOST_CFG.get("max_lag", 12)), ge=0)
    recursive_forecast: bool = bool(XGBOOST_CFG.get("recursive_forecast", True))



def _raise_422(msg: str):
    raise HTTPException(status_code=422, detail=msg)


def _future_dates(df_future: pd.DataFrame) -> pd.Series:
    if {"anio", "mes", "dia"}.issubset(df_future.columns):
        return pd.to_datetime(dict(year=df_future["anio"], month=df_future["mes"], day=df_future["dia"]), errors="coerce")
    if {"anio", "mes"}.issubset(df_future.columns):
        return pd.to_datetime(dict(year=df_future["anio"], month=df_future["mes"], day=1), errors="coerce")
    return pd.Series([pd.NaT] * len(df_future), index=df_future.index)


def _parse_dt(v: Optional[str]) -> Optional[pd.Timestamp]:
    if v in (None, ""):
        return None
    try:
        return pd.Timestamp(datetime.fromisoformat(v[:10]))
    except Exception:
        _raise_422(f"Fecha inválida en override: '{v}'. Formato esperado YYYY-MM-DD")


def _apply_overrides(df_future: pd.DataFrame, overrides: list[ScenarioOverride]) -> pd.DataFrame:
    out = df_future.copy()
    if not overrides:
        return out

    future_dates = _future_dates(out)

    for ov in overrides:
        col = _safe_alias(ov.var)
        if col not in out.columns:
            _raise_422(f"Override referencia variable no disponible: {ov.var}")

        start = _parse_dt(ov.start)
        end = _parse_dt(ov.end)

        mask = pd.Series(True, index=out.index)
        if start is not None:
            mask &= future_dates >= start
        if end is not None:
            mask &= future_dates <= end

        base = pd.to_numeric(out.loc[mask, col], errors="coerce")
        if ov.op == "set":
            out.loc[mask, col] = float(ov.value)
        elif ov.op == "add":
            out.loc[mask, col] = base + float(ov.value)
        elif ov.op == "mul":
            out.loc[mask, col] = base * float(ov.value)
        elif ov.op == "pct":
            out.loc[mask, col] = base * (1.0 + float(ov.value) / 100.0)

    return out


def _run_sarimax(req: ScenariosRunRequest) -> dict:
    df = _get_dataframe(req)
    dia_col, mes_col, ano_col = sarimax_detect_calendar_cols(df)

    y_col = _safe_alias(req.target_var)
    exog_cols = _prepare_exog_cols(req)

    df, exog_cols, use_fourier = _maybe_add_fourier(df, req, dia_col, exog_cols)
    df = sarimax_sort_temporally(df, ano_col, mes_col, dia_col)
    df = sarimax_force_numeric(df, y_col, exog_cols)

    df_hist, df_future_all = sarimax_split_hist_future(df, y_col)
    horizon = _get_horizon(req, df_future_all)
    df_future = df_future_all.iloc[:horizon].copy()
    df_future_scn = _apply_overrides(df_future, req.scenario_overrides)

    df_hist, exog_future_base = _prepare_exog_matrices(df, df_hist, df_future, exog_cols)
    _, exog_future_scn = _prepare_exog_matrices(df, df_hist.copy(), df_future_scn, exog_cols)

    train, test, n_train, n_test = _train_test_split_hist(df_hist, req.train_ratio)
    exog_test = test[exog_cols].astype(float) if exog_cols else None

    order, seas = _select_params(req, df_hist, exog_cols, y_col, n_test, use_fourier)

    model_fit = create_sarimax_model(train=train, exog_cols=exog_cols, column_y=y_col, order=order, seasonal_order=seas)
    pred_test = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test)
    mape, rmse, mae = sarimax_metrics(pred=pred_test, df_test=test, indicador=y_col)

    model_fit_full = create_sarimax_model(train=df_hist, exog_cols=exog_cols, column_y=y_col, order=order, seasonal_order=seas)
    y_forecast_base = model_fit_full.predict(start=len(df_hist), end=len(df_hist) + horizon - 1, exog=exog_future_base)
    y_forecast_scn = model_fit_full.predict(start=len(df_hist), end=len(df_hist) + horizon - 1, exog=exog_future_scn)

    return {
        "model": "sarimax",
        "target_var": req.target_var,
        "predictors_used": exog_cols,
        "horizon": horizon,
        "y_col": y_col,
        "n_obs": len(df_hist),
        "mape": float(mape),
        "rmse": float(rmse),
        "mae": float(mae),
        "df": df.to_dict(orient="records") if req.return_df else None,
        "baseline": {"y_forecast": [float(x) for x in list(y_forecast_base)]},
        "scenario": {"y_forecast": [float(x) for x in list(y_forecast_scn)]},
        "n_train": n_train,
        "n_test": n_test,
    }


def _run_xgboost(req: ScenariosRunRequest) -> dict:
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

    dia_col, mes_col, ano_col = xgb_detect_calendar_cols(df)
    df = xgb_sort_temporally(df, ano_col, mes_col, dia_col)
    df = xgb_force_numeric(df, [y_col] + predictors)

    df_hist, df_future_all = xgb_split_hist_future(df, y_col)
    horizon = _get_horizon(req, df_future_all)
    df_future = df_future_all.iloc[:horizon].copy()
    df_future_scn = _apply_overrides(df_future, req.scenario_overrides)

    if predictors:
        df_hist.loc[:, predictors] = df_hist[predictors].fillna(0.0)
        if df_future[predictors].isna().any().any() or df_future_scn[predictors].isna().any().any():
            _raise_422("Hay NaNs en predictores futuros: no se puede predecir")

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
        _raise_422(f"Split inválido (histórico): n={n}, n_train={n_train}, n_test={n_test}.")

    train = df_hist_sup.iloc[:n_train].copy()
    test = df_hist_sup.iloc[n_train:n_train + n_test].copy()

    if req.auto_params:
        best_params = best_xgboost_params(
            df=df_hist_sup,
            exog_cols=feature_cols,
            column_y=y_col,
            periodos_a_predecir=n_test,
            random_state=int(XGBOOST_CFG.get("random_state", 42)),
        )
        xgb_params = {
            "objective": XGB_ENGINE_CFG.get("objective", "reg:squarederror"),
            "tree_method": XGB_ENGINE_CFG.get("tree_method", "hist"),
            "n_jobs": int(XGB_ENGINE_CFG.get("n_jobs", 1)),
            "random_state": int(XGBOOST_CFG.get("random_state", 42)),
            **best_params,
        }
    else:
        xgb_params = req.xgb_params or {
            **XGB_DEFAULT_PARAMS_CFG,
            "objective": XGB_ENGINE_CFG.get("objective", "reg:squarederror"),
            "tree_method": XGB_ENGINE_CFG.get("tree_method", "hist"),
            "n_jobs": int(XGB_ENGINE_CFG.get("n_jobs", 1)),
            "random_state": int(XGBOOST_CFG.get("random_state", 42)),
        }

    model_fit = create_xgboost_model(train=train, exog_cols=feature_cols, column_y=y_col, xgb_params=xgb_params)

    if req.recursive_forecast and req.use_target_lags and req.max_lag > 0:
        pred_test = _recursive_predict(model_fit, train=train, test=test, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
    else:
        pred_test = pd.Series(model_fit.predict(test[feature_cols]), index=test.index, name=y_col)

    mape, rmse, mae = xgb_metrics(pred=pred_test, df_test=test, indicador=y_col)

    model_fit_full = create_xgboost_model(train=df_hist_sup, exog_cols=feature_cols, column_y=y_col, xgb_params=xgb_params)

    if req.use_target_lags and req.max_lag > 0:
        y_forecast_base = _recursive_predict(model_fit_full, train=df_hist, test=df_future, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
        y_forecast_scn = _recursive_predict(model_fit_full, train=df_hist, test=df_future_scn, y_col=y_col, feature_cols=feature_cols, max_lag=req.max_lag)
        base_vals = [float(x) for x in y_forecast_base.values]
        scn_vals = [float(x) for x in y_forecast_scn.values]
    else:
        base_vals = [float(x) for x in model_fit_full.predict(df_future[feature_cols])]
        scn_vals = [float(x) for x in model_fit_full.predict(df_future_scn[feature_cols])]

    return {
        "model": "xgboost",
        "target_var": req.target_var,
        "predictors_used": predictors,
        "horizon": horizon,
        "y_col": y_col,
        "n_obs": len(df_hist),
        "mape": float(mape),
        "rmse": float(rmse),
        "mae": float(mae),
        "df": df.to_dict(orient="records") if req.return_df else None,
        "baseline": {"y_forecast": base_vals},
        "scenario": {"y_forecast": scn_vals},
        "n_train": n_train,
        "n_test": n_test,
    }


@router.post("/run")
def scenarios_run(req: ScenariosRunRequest):
    try:
        if req.model == "sarimax":
            return _run_sarimax(req)
        return _run_xgboost(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error escenarios run: {e}")
