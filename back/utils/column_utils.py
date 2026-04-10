import re
from typing import Any, Optional, Tuple

import pandas as pd
from fastapi import HTTPException


def _safe_alias(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\W+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def _find_col(df: Any, *candidates: str):
    columns = getattr(df, "columns", [])
    for c in candidates:
        if c and c in columns:
            return c
    return None


def build_time_index(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str]]:
    """
    Build a '__dt' datetime column from anio/mes/dia columns, a DatetimeIndex,
    or a 'fecha' column.  Returns (df_with___dt, dia_col, mes_col, ano_col).
    """
    dia_col = _find_col(df, "dia", _safe_alias("dia"))
    mes_col = _find_col(df, "mes", _safe_alias("mes"))
    ano_col = _find_col(
        df,
        "anio",
        "año",
        "ano",
        _safe_alias("anio"),
        _safe_alias("año"),
        _safe_alias("ano"),
    )

    out = df.copy()
    if ano_col and mes_col and ano_col in out.columns and mes_col in out.columns:
        day = out[dia_col] if dia_col and dia_col in out.columns else 1
        out["__dt"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(out[ano_col], errors="coerce"),
                month=pd.to_numeric(out[mes_col], errors="coerce"),
                day=pd.to_numeric(day, errors="coerce"),
            ),
            errors="coerce",
        )
    elif isinstance(out.index, pd.DatetimeIndex):
        out["__dt"] = pd.to_datetime(out.index)
    elif "fecha" in out.columns:
        out["__dt"] = pd.to_datetime(out["fecha"], errors="coerce")
    else:
        raise HTTPException(
            status_code=422,
            detail="No fue posible construir un time_index robusto",
        )

    out = (
        out.dropna(subset=["__dt"])
        .sort_values("__dt", kind="mergesort")
        .reset_index(drop=True)
    )
    return out, dia_col, mes_col, ano_col
