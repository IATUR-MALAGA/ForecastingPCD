import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

def plot_predictions(
    df,
    pred,
    title,
    ylabel,
    xlabel,
    column_y,
    periodos_a_predecir=2,
    holidays_col=None
):
    df = df.copy()

    # Índice temporal
    if not isinstance(df.index, pd.DatetimeIndex):
        if {"anio", "mes"}.issubset(df.columns):
            if "dia" in df.columns:
                idx = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=df["dia"]))
            else:
                idx = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
            df = df.set_index(idx).sort_index()
        else:
            raise ValueError("df debe tener DatetimeIndex o columnas anio/mes(/dia).")

    periodos_a_predecir = int(periodos_a_predecir or 1)
    periodos_a_predecir = max(1, min(periodos_a_predecir, len(df) - 1))
    train = df.iloc[:-periodos_a_predecir]

    # Alinear índice de pred al final del df (ideal si df ya incluye futuro)
    if not isinstance(pred.index, pd.DatetimeIndex):
        pred = pd.Series(pred.values, index=df.index[-len(pred):], name=getattr(pred, "name", None))

    fig, ax = plt.subplots(figsize=(12, 4))

    # Train
    train[column_y].plot(ax=ax, label="Train")

    # Prediction (si 1 punto, scatter grande)
    if len(pred) == 1:
        ax.scatter(pred.index, pred.values, label="Prediction", zorder=5, s=30, color='red')
    else:
        pred.plot(ax=ax, label="Prediction")

    # ✅ CLAVE: reescalar para que entre la predicción en el eje
    ax.relim()
    ax.autoscale_view()

    # Padding temporal para que el último punto no quede pegado al borde
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        dt_min, dt_max = df.index.min(), df.index.max()
        # tamaño de paso típico (median) para monthly/daily/etc.
        step = (df.index.to_series().diff().median())
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=30)  # fallback razonable
        pad = step * 2  # 2 pasos de margen
        ax.set_xlim(dt_min - pad, dt_max + pad)
    else:
        ax.set_xlim(df.index.min(), df.index.max())

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    # Holidays
    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]
        for x in holidays_pred.index:
            ax.axvline(x=x, color='k', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    fig.tight_layout()
    return fig
