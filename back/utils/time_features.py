import numpy as np
import pandas as pd


def add_fourier_annual_terms(
    df: pd.DataFrame,
    dia_col: str = "dia",
    K: int = 6,
    m: int = 365,
    anio_col: str = "anio",
    mes_col: str = "mes",
    fecha_col: str = "fecha",
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    df[dia_col] = pd.to_numeric(df[dia_col], errors="raise").astype(int)
    if not df[dia_col].between(1, 31).all():
        bad = df.loc[~df[dia_col].between(1, 31), dia_col].head(5).tolist()
        raise ValueError(
            f"Valores inválidos en '{dia_col}' (deben ser 1..31). Ejemplos: {bad}"
        )

    if anio_col not in df.columns or mes_col not in df.columns:
        raise ValueError(
            f"Faltan columnas '{anio_col}' y/o '{mes_col}' para construir '{fecha_col}'."
        )

    df[fecha_col] = pd.to_datetime(
        dict(
            year=df[anio_col].astype(int),
            month=df[mes_col].astype(int),
            day=df[dia_col],
        ),
        errors="raise",
    )

    df = df.sort_values(fecha_col)
    t = (df[fecha_col] - df[fecha_col].min()).dt.days.to_numpy()

    cols: list[str] = []
    for k in range(1, K + 1):
        ccol = f"fourier_cos{k}_{m}"
        scol = f"fourier_sin{k}_{m}"
        df[ccol] = np.cos(2 * np.pi * k * t / m)
        df[scol] = np.sin(2 * np.pi * k * t / m)
        cols.extend([ccol, scol])

    return df, cols
