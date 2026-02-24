from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from front.utils.utils import _find_col, add_fourier_annual_terms, create_dataframe_based_on_selection
from front.utils.utils import _safe_alias
from back.models.SARIMAX.sarimax_model import best_sarimax_params, create_sarimax_model
from back.models.SARIMAX.sarimax_statistics import compute_metrics

router = APIRouter(prefix="/models/sarimax", tags=["SARIMAX_model"])


# ------------ Schemas ------------

class FilterSelection(BaseModel):
    table: str
    col: str
    values: list[str]

class SarimaxRunRequest(BaseModel):
    target_var: str
    predictors: list[str] = Field(default_factory=list)
    filters_by_var: Optional[dict[str, list[FilterSelection]]] = None

    train_ratio: float = Field(0.70, gt=0.0, lt=1.0)

    auto_params: bool = True
    s: int = 12

    order: Optional[tuple[int, int, int]] = None
    seasonal_order: Optional[tuple[int, int, int, int]] = None

    return_df: bool = True


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

    df: Optional[list[dict[str, Any]]] = None


# ------------ Endpoint ------------

@router.post("/run", response_model=SarimaxRunResponse)
def sarimax_run(req: SarimaxRunRequest):
    try:
        df = create_dataframe_based_on_selection(
            target_var=req.target_var,
            predictors=req.predictors,
            filters_by_var={k: [f.model_dump() for f in v] for k, v in (req.filters_by_var or {}).items()}
        )

        if df is None or len(df) == 0:
            raise HTTPException(status_code=422, detail="El dataframe resultante está vacío")

        # Columnas calendario (robusto a alias)
        dia_col = _find_col(df, "dia", _safe_alias("dia"))
        mes_col = _find_col(df, "mes", _safe_alias("mes"))
        ano_col = _find_col(df, "año", "ano", _safe_alias("año"), _safe_alias("ano"))

        use_fourier = dia_col is not None  # prioridad: si hay 'dia', usas Fourier

        exog_cols = [_safe_alias(c) for c in (req.predictors or [])]
        y_col = _safe_alias(req.target_var)

        # (Opcional) evita meter columnas de calendario como exog “tal cual”
        calendar_like = {_safe_alias("dia"), _safe_alias("mes"), _safe_alias("año"), _safe_alias("ano")}
        exog_cols = [c for c in exog_cols if c not in calendar_like]

        # Si diario: añade Fourier anual
        if use_fourier:
            K = getattr(req, "fourier_k", 6)  # si no existe en tu request, usa 6
            df, fourier_cols = add_fourier_annual_terms(df, dia_col=dia_col, K=K, m=365)

            # Si tu _safe_alias cambia nombres, renómbralos a versión safe
            fourier_cols_safe = [_safe_alias(c) for c in fourier_cols]
            rename_map = dict(zip(fourier_cols, fourier_cols_safe))
            df = df.rename(columns=rename_map)

            exog_cols = exog_cols + fourier_cols_safe

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
        exog_test = test[exog_cols] if exog_cols else None
        if req.auto_params:
            if use_fourier:
                # NO estacional: anualidad ya está en Fourier (exog)
                order, seas = best_sarimax_params(
                    df=df,
                    exog_cols=exog_cols,
                    column_y=y_col,
                    s=1,  # se ignora al no ser estacional
                    periodos_a_predecir=n_test,
                    seasonal=False
                )
                seas = (0, 0, 0, 0)  # asegúralo explícito
            else:
                # Mensual (mes+año) o lo que sea: tu comportamiento actual
                order, seas = best_sarimax_params(
                    df=df,
                    exog_cols=exog_cols,
                    column_y=y_col,
                    s=req.s,
                    periodos_a_predecir=n_test,
                    seasonal=True
                )
        else:
            order = req.order or (0, 1, 0)
            if use_fourier:
                seas = (0, 0, 0, 0)
            else:
                seas = req.seasonal_order or (0, 0, 0, 12)

        model_fit = create_sarimax_model(
            train=train,
            exog_cols=exog_cols,
            column_y=y_col,
            order=order,
            seasonal_order=seas
        )

        pred_test = model_fit.predict(
            start=len(train),
            end=len(train) + len(test) - 1,
            exog=exog_test
        )

        mape, rmse, mae = compute_metrics(pred=pred_test, df_test=test, indicador=y_col)

        out = SarimaxRunResponse(
            y_col=y_col,
            exog_cols=exog_cols,
            n=n,
            n_train=n_train,
            n_test=n_test,
            order=order,
            seasonal_order=seas,
            mape=float(mape),
            rmse=float(rmse),
            mae=float(mae),
            y_pred=[float(x) for x in list(pred_test)],
            df=df.to_dict(orient="records") if req.return_df else None
        )
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error SARIMAX run: {e}")
