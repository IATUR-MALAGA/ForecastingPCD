from __future__ import annotations

from typing import Any, List, Optional, Tuple

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
    df: Optional[list[dict[str, Any]]] = None


def _raise_422(detail: str) -> None:
    raise HTTPException(status_code=422, detail=detail)


def _get_dataframe(req) -> pd.DataFrame:
    df = create_dataframe_based_on_selection(
        target_var=req.target_var,
        predictors=req.predictors,
        filters_by_var={k: [f.model_dump() for f in v] for k, v in (req.filters_by_var or {}).items()},
    )
    if df is None or df.empty:
        _raise_422("El dataframe resultante está vacío")
    return df


def _detect_calendar_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    dia_col = _find_col(df, "dia", _safe_alias("dia"))
    mes_col = _find_col(df, "mes", _safe_alias("mes"))
    ano_col = _find_col(df, "anio", "año", "ano", _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano"))
    return dia_col, mes_col, ano_col


def _prepare_exog_cols(req) -> List[str]:
    exog_cols = [_safe_alias(c) for c in (req.predictors or [])]
    calendar_like = {
        _safe_alias("dia"), _safe_alias("mes"),
        _safe_alias("anio"), _safe_alias("año"), _safe_alias("ano"),
    }
    return [c for c in exog_cols if c not in calendar_like]


def _maybe_add_fourier(
    df: pd.DataFrame,
    req,
    dia_col: Optional[str],
    exog_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], bool]:
    use_fourier = bool(FOURIER_CFG.get("enabled_when_daily", True)) and dia_col is not None
    if not use_fourier:
        return df, exog_cols, False

    k_value = int(getattr(req, "fourier_k", FOURIER_CFG.get("k", 6)) or FOURIER_CFG.get("k", 6))
    m_value = int(FOURIER_CFG.get("m", 365))
    df, fourier_cols = add_fourier_annual_terms(df, dia_col=dia_col, K=k_value, m=m_value)

    fourier_cols_safe = [_safe_alias(c) for c in fourier_cols]
    df = df.rename(columns=dict(zip(fourier_cols, fourier_cols_safe)))
    return df, (exog_cols + fourier_cols_safe), True


def _sort_temporally(df: pd.DataFrame, ano_col: Optional[str], mes_col: Optional[str], dia_col: Optional[str]) -> pd.DataFrame:
    sort_cols = [c for c in [ano_col, mes_col, dia_col] if c and c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _force_numeric(df: pd.DataFrame, y_col: str, exog_cols: List[str]) -> pd.DataFrame:
    if y_col not in df.columns:
        _raise_422(f"No existe la columna objetivo '{y_col}' en el dataframe")

    for c in [y_col, *exog_cols]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.replace([np.inf, -np.inf], np.nan)


def _split_hist_future(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_hist = df[y_col].notna()
    df_hist = df.loc[mask_hist].copy()
    df_future_all = df.loc[~mask_hist].copy()

    if len(df_hist) < int(COMMON_CFG.get("min_historical_rows", 3)):
        _raise_422("Histórico insuficiente para entrenar SARIMAX")

    return df_hist, df_future_all


def _get_horizon(req, df_future_all: pd.DataFrame) -> int:
    horizon = int(getattr(req, "horizon", COMMON_CFG.get("horizon", 1)) or COMMON_CFG.get("horizon", 1))
    if horizon < 1:
        _raise_422("horizon debe ser >= 1")

    if len(df_future_all) < horizon:
        _raise_422(
            f"No hay suficientes filas futuras para horizon={horizon}. "
            f"Filas futuras disponibles: {len(df_future_all)}."
        )
    return horizon


def _prepare_exog_matrices(
    df: pd.DataFrame,
    df_hist: pd.DataFrame,
    df_future: pd.DataFrame,
    exog_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not exog_cols:
        return None, None

    missing = [c for c in exog_cols if c not in df.columns]
    if missing:
        _raise_422(f"Faltan columnas exógenas en df: {missing}")

    df_hist.loc[:, exog_cols] = df_hist[exog_cols].fillna(0.0)
    if df_future[exog_cols].isna().any().any():
        _raise_422("Hay NaNs en exógenas futuras: no se puede predecir")

    exog_future = df_future[exog_cols].astype(float)
    return df_hist, exog_future


def _train_test_split_hist(df_hist: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    n = len(df_hist)
    n_train = int(n * train_ratio)
    n_test = n - n_train

    if n_train <= 0 or n_test <= 0:
        _raise_422(f"Split inválido (histórico): n={n}, n_train={n_train}, n_test={n_test}. Ajusta train_ratio.")

    train = df_hist.iloc[:n_train]
    test = df_hist.iloc[n_train:n_train + n_test]
    return train, test, n_train, n_test


def _select_params(req, df_hist: pd.DataFrame, exog_cols: List[str], y_col: str, n_test: int, use_fourier: bool):
    if req.auto_params:
        if use_fourier:
            order, seas = best_sarimax_params(
                df=df_hist,
                exog_cols=exog_cols,
                column_y=y_col,
                s=1,
                periodos_a_predecir=n_test,
                seasonal=False,
            )
            seas = (0, 0, 0, 0)
        else:
            order, seas = best_sarimax_params(
                df=df_hist,
                exog_cols=exog_cols,
                column_y=y_col,
                s=req.s,
                periodos_a_predecir=n_test,
                seasonal=True,
            )
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
        dia_col, mes_col, ano_col = _detect_calendar_cols(df)

        y_col = _safe_alias(req.target_var)
        exog_cols = _prepare_exog_cols(req)

        df, exog_cols, use_fourier = _maybe_add_fourier(df, req, dia_col, exog_cols)
        df = _sort_temporally(df, ano_col, mes_col, dia_col)
        df = _force_numeric(df, y_col, exog_cols)

        df_hist, df_future_all = _split_hist_future(df, y_col)
        horizon = _get_horizon(req, df_future_all)
        df_future = df_future_all.iloc[:horizon].copy()

        df_hist, exog_future = _prepare_exog_matrices(df, df_hist, df_future, exog_cols)
        train, test, n_train, n_test = _train_test_split_hist(df_hist, req.train_ratio)
        exog_test = test[exog_cols].astype(float) if exog_cols else None

        order, seas = _select_params(req, df_hist, exog_cols, y_col, n_test, use_fourier)

        model_fit = create_sarimax_model(
            train=train,
            exog_cols=exog_cols,
            column_y=y_col,
            order=order,
            seasonal_order=seas,
        )
        pred_test = model_fit.predict(
            start=len(train),
            end=len(train) + len(test) - 1,
            exog=exog_test,
        )
        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        model_fit_full = create_sarimax_model(
            train=df_hist,
            exog_cols=exog_cols,
            column_y=y_col,
            order=order,
            seasonal_order=seas,
        )
        y_forecast = model_fit_full.predict(
            start=len(df_hist),
            end=len(df_hist) + horizon - 1,
            exog=exog_future,
        )

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
