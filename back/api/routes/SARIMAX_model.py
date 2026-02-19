from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from back.config import settings
from back.models.SARIMAX.sarimax_model import best_sarimax_params, create_sarimax_model
from back.models.SARIMAX.sarimax_statistics import compute_metrics
from front.utils.utils import _find_col, _safe_alias, add_fourier_annual_terms, create_dataframe_based_on_selection


SARIMAX_CFG = settings.get("models.sarimax", {})
COMMON_CFG = settings.get("models.common", {})
FOURIER_CFG = SARIMAX_CFG.get("fourier", {}) if isinstance(SARIMAX_CFG, dict) else {}


router = APIRouter(
    prefix=settings.get("server.routes.sarimax_prefix", "/models/sarimax"),
    tags=["SARIMAX_model"],
)


class FilterSelection(BaseModel):
    table: str
    col: str
    values: list[str]


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


class SarimaxRunRequest(BaseModel):
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None

    train_ratio: float = Field(float(COMMON_CFG.get("train_ratio", 0.70)), gt=0.0, lt=1.0)

    auto_params: bool = bool(SARIMAX_CFG.get("auto_params", True))
    s: int = int(SARIMAX_CFG.get("seasonal_period_s", 12))

    order: Optional[tuple[int, int, int]] = None
    seasonal_order: Optional[tuple[int, int, int, int]] = None
    horizon: int = int(COMMON_CFG.get("horizon", 1))
    return_df: bool = bool(COMMON_CFG.get("return_df", True))

    scenario_mode: Optional[Literal["future", "past"]] = None
    scenario_overrides: list[ScenarioOverride] = Field(default_factory=list)
    scenario_future_values: list[ScenarioFutureValue] = Field(default_factory=list)
    scenario_window: Optional[ScenarioWindow] = None


class SarimaxRunResponse(BaseModel):
    y_col: str
    exog_cols: list[str]
    n: int
    n_train: int
    n_test: int
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
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


def _raise_422(detail: str) -> None:
    raise HTTPException(status_code=422, detail=detail)


def _build_time_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str]]:
    dia_col = _find_col(df, "dia", _safe_alias("dia"))
    mes_col = _find_col(df, "mes", _safe_alias("mes"))
    ano_col = _find_col(df, "anio", "año", "ano", _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano"))

    out = df.copy()
    if ano_col and mes_col and ano_col in out.columns and mes_col in out.columns:
        day = out[dia_col] if dia_col and dia_col in out.columns else 1
        out["__dt"] = pd.to_datetime(
            dict(year=pd.to_numeric(out[ano_col], errors="coerce"), month=pd.to_numeric(out[mes_col], errors="coerce"), day=pd.to_numeric(day, errors="coerce")),
            errors="coerce",
        )
    elif isinstance(out.index, pd.DatetimeIndex):
        out["__dt"] = pd.to_datetime(out.index)
    elif "fecha" in out.columns:
        out["__dt"] = pd.to_datetime(out["fecha"], errors="coerce")
    else:
        _raise_422("No fue posible construir un time_index robusto")

    out = out.dropna(subset=["__dt"]).sort_values("__dt", kind="mergesort").reset_index(drop=True)
    return out, dia_col, mes_col, ano_col


def _get_dataframe(req) -> pd.DataFrame:
    df = create_dataframe_based_on_selection(
        target_var=req.target_var,
        predictors=req.predictors,
        filters_by_var={k: [f.model_dump() for f in v] for k, v in (req.filters_by_var or {}).items()},
    )
    if df is None or df.empty:
        _raise_422("El dataframe resultante está vacío")
    return df


def _prepare_exog_cols(req) -> List[str]:
    exog_cols = [_safe_alias(c) for c in (req.predictors or [])]
    calendar_like = {_safe_alias("dia"), _safe_alias("mes"), _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano")}
    return [c for c in exog_cols if c not in calendar_like]


def _maybe_add_fourier(df: pd.DataFrame, req, dia_col: Optional[str], exog_cols: List[str]) -> Tuple[pd.DataFrame, List[str], bool]:
    use_fourier = bool(FOURIER_CFG.get("enabled_when_daily", True)) and dia_col is not None
    if not use_fourier:
        return df, exog_cols, False

    k_value = int(getattr(req, "fourier_k", FOURIER_CFG.get("k", 6)) or FOURIER_CFG.get("k", 6))
    m_value = int(FOURIER_CFG.get("m", 365))
    df, fourier_cols = add_fourier_annual_terms(df, dia_col=dia_col, K=k_value, m=m_value)
    fourier_cols_safe = [_safe_alias(c) for c in fourier_cols]
    df = df.rename(columns=dict(zip(fourier_cols, fourier_cols_safe)))
    return df, (exog_cols + fourier_cols_safe), True


def _force_numeric(df: pd.DataFrame, y_col: str, exog_cols: List[str]) -> pd.DataFrame:
    if y_col not in df.columns:
        _raise_422(f"No existe la columna objetivo '{y_col}' en el dataframe")
    for c in [y_col, *exog_cols]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def _infer_future_index(df_hist: pd.DataFrame, horizon: int, monthly_hint: bool) -> pd.DatetimeIndex:
    last_dt = pd.to_datetime(df_hist["__dt"]).max()
    if pd.isna(last_dt):
        _raise_422("No se pudo inferir fecha final histórica")
    if monthly_hint:
        start = (last_dt + pd.offsets.MonthBegin(1)).normalize()
        return pd.date_range(start=start, periods=horizon, freq="MS")
    return pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=horizon, freq="D")


def _apply_overrides(df: pd.DataFrame, overrides: list[ScenarioOverride], dt_col: str, exog_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for ov in overrides:
        var = _safe_alias(ov.var)
        if var not in exog_cols or var not in out.columns:
            continue
        mask = pd.Series(True, index=out.index)
        if ov.start:
            mask &= pd.to_datetime(out[dt_col]) >= pd.to_datetime(ov.start)
        if ov.end:
            mask &= pd.to_datetime(out[dt_col]) <= pd.to_datetime(ov.end)
        if ov.op == "set":
            out.loc[mask, var] = float(ov.value)
        elif ov.op == "add":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") + float(ov.value)
        elif ov.op == "mul":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") * float(ov.value)
        elif ov.op == "pct":
            out.loc[mask, var] = pd.to_numeric(out.loc[mask, var], errors="coerce") * (1.0 + (float(ov.value) / 100.0))
    return out


def _validate_missing_future(df_future: pd.DataFrame, exog_cols: list[str]) -> None:
    missing = []
    for var in exog_cols:
        if var not in df_future.columns:
            continue
        miss_rows = df_future[df_future[var].isna()]["__dt"]
        for dt in miss_rows:
            missing.append({"var": var, "date": pd.to_datetime(dt).strftime("%Y-%m-%d")})
    if missing:
        raise HTTPException(status_code=400, detail={"detail": "Missing future exogenous values", "missing": missing})


def _select_params(req, df_hist: pd.DataFrame, exog_cols: List[str], y_col: str, n_test: int, use_fourier: bool):
    if req.auto_params:
        if use_fourier:
            order, seas = best_sarimax_params(df=df_hist, exog_cols=exog_cols, column_y=y_col, s=1, periodos_a_predecir=n_test, seasonal=False)
            seas = (0, 0, 0, 0)
        else:
            order, seas = best_sarimax_params(df=df_hist, exog_cols=exog_cols, column_y=y_col, s=req.s, periodos_a_predecir=n_test, seasonal=True)
        return order, seas

    default_order = tuple(SARIMAX_CFG.get("default_order", [0, 1, 0]))
    default_seasonal = tuple(SARIMAX_CFG.get("default_seasonal_order", [0, 0, 0, 12]))
    order = req.order or default_order
    seas = (0, 0, 0, 0) if use_fourier else (req.seasonal_order or default_seasonal)
    return order, seas


@router.post("/run", response_model=SarimaxRunResponse)
def sarimax_run(req: SarimaxRunRequest):
    try:
        df = _get_dataframe(req)
        y_col = _safe_alias(req.target_var)
        exog_cols = _prepare_exog_cols(req)

        df, dia_col, mes_col, _ = _build_time_index(df)
        df, exog_cols, use_fourier = _maybe_add_fourier(df, req, dia_col, exog_cols)
        df = _force_numeric(df, y_col, exog_cols)

        mask_hist = df[y_col].notna()
        df_hist = df.loc[mask_hist].copy()
        if len(df_hist) < int(COMMON_CFG.get("min_historical_rows", 3)):
            _raise_422("Histórico insuficiente para entrenar SARIMAX")

        if req.scenario_mode == "past":
            if req.scenario_window is None:
                _raise_422("scenario_window es obligatorio cuando scenario_mode='past'")
            ws = pd.to_datetime(req.scenario_window.start)
            we = pd.to_datetime(req.scenario_window.end)
            if ws > we:
                _raise_422("scenario_window.start debe ser <= scenario_window.end")
            hist_min, hist_max = pd.to_datetime(df_hist["__dt"]).min(), pd.to_datetime(df_hist["__dt"]).max()
            if ws < hist_min or we > hist_max:
                _raise_422("scenario_window debe estar dentro del rango histórico observado")

            train = df_hist[pd.to_datetime(df_hist["__dt"]) < ws].copy()
            test = df_hist[(pd.to_datetime(df_hist["__dt"]) >= ws) & (pd.to_datetime(df_hist["__dt"]) <= we)].copy()
            if train.empty or test.empty:
                _raise_422("No hay datos suficientes para entrenar/probar en la ventana indicada")
            if exog_cols:
                train.loc[:, exog_cols] = train[exog_cols].fillna(0.0)
                test = _apply_overrides(test, req.scenario_overrides, "__dt", exog_cols)
                if test[exog_cols].isna().any().any():
                    _raise_422("Hay NaNs en exógenas de la ventana de escenario")

            order, seas = _select_params(req, train, exog_cols, y_col, len(test), use_fourier)
            model_fit = create_sarimax_model(train=train, exog_cols=exog_cols, column_y=y_col, order=order, seasonal_order=seas)
            exog_test = test[exog_cols].astype(float) if exog_cols else None
            y_forecast = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test)
            y_true = test[y_col].astype(float).tolist()
            mape, rmse, mae = compute_metrics(pred=y_forecast, df_test=test, indicador=y_col)
            return SarimaxRunResponse(
                y_col=y_col,
                exog_cols=exog_cols,
                n=len(df),
                n_train=len(train),
                n_test=len(test),
                order=order,
                seasonal_order=seas,
                mape=float(mape),
                rmse=float(rmse),
                mae=float(mae),
                y_pred=[float(x) for x in list(y_forecast)],
                y_forecast=[float(x) for x in list(y_forecast)],
                horizon=len(test),
                n_obs=len(df_hist),
                y_true=y_true,
                window={"start": ws.strftime("%Y-%m-%d"), "end": we.strftime("%Y-%m-%d")},
                df=df.to_dict(orient="records") if req.return_df else None,
            )

        horizon = int(getattr(req, "horizon", COMMON_CFG.get("horizon", 1)) or COMMON_CFG.get("horizon", 1))
        if horizon < 1:
            _raise_422("horizon debe ser >= 1")

        monthly_hint = mes_col is not None and dia_col is None
        future_index = _infer_future_index(df_hist, horizon, monthly_hint)
        df_future = pd.DataFrame({"__dt": future_index})

        existing_future = df.loc[~mask_hist].copy()
        if not existing_future.empty:
            existing_future = existing_future.drop_duplicates(subset=["__dt"], keep="last")
            df_future = df_future.merge(existing_future[["__dt", *[c for c in exog_cols if c in existing_future.columns]]], on="__dt", how="left")

        for fv in req.scenario_future_values:
            var = _safe_alias(fv.var)
            if var in exog_cols:
                date = pd.to_datetime(fv.date)
                df_future.loc[pd.to_datetime(df_future["__dt"]) == date, var] = float(fv.value)

        if exog_cols:
            df_future = _apply_overrides(df_future, req.scenario_overrides if req.scenario_mode == "future" else [], "__dt", exog_cols)
            _validate_missing_future(df_future, exog_cols)
            exog_future = df_future[exog_cols].astype(float)
            df_hist.loc[:, exog_cols] = df_hist[exog_cols].fillna(0.0)
        else:
            exog_future = None

        n_train = int(len(df_hist) * req.train_ratio)
        n_test = len(df_hist) - n_train
        if n_train <= 0 or n_test <= 0:
            _raise_422(f"Split inválido (histórico): n={len(df_hist)}, n_train={n_train}, n_test={n_test}. Ajusta train_ratio.")
        train = df_hist.iloc[:n_train]
        test = df_hist.iloc[n_train:n_train + n_test]
        exog_test = test[exog_cols].astype(float) if exog_cols else None

        order, seas = _select_params(req, df_hist, exog_cols, y_col, n_test, use_fourier)
        model_fit = create_sarimax_model(train=train, exog_cols=exog_cols, column_y=y_col, order=order, seasonal_order=seas)
        pred_test = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test)
        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        model_fit_full = create_sarimax_model(train=df_hist, exog_cols=exog_cols, column_y=y_col, order=order, seasonal_order=seas)
        y_forecast = model_fit_full.predict(start=len(df_hist), end=len(df_hist) + horizon - 1, exog=exog_future)

        return SarimaxRunResponse(
            y_col=y_col,
            exog_cols=exog_cols,
            n=len(df),
            n_train=n_train,
            n_test=n_test,
            order=order,
            seasonal_order=seas,
            mape=float(mape),
            rmse=float(rmse),
            mae=float(mae),
            y_pred=[float(x) for x in list(pred_test)],
            y_forecast=[float(x) for x in list(y_forecast)],
            horizon=horizon,
            n_obs=len(df_hist),
            df=df.to_dict(orient="records") if req.return_df else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error SARIMAX run: {e}")
