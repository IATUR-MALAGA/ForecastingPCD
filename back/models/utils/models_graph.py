from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from back.config import settings


def plot_predictions(
    df,
    pred,
    title,
    ylabel,
    xlabel,
    column_y,
    periodos_a_predecir=2,
    holidays_col=None,
):
    df = df.copy()

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

    if not isinstance(pred.index, pd.DatetimeIndex):
        pred = pd.Series(pred.values, index=df.index[-len(pred):], name=getattr(pred, "name", None))

    figsize = tuple(settings.get("plots.predictions.figsize", [12, 4]))
    scatter_size = int(settings.get("plots.predictions.scatter_size_single_point", 30))
    scatter_color = settings.get("plots.predictions.prediction_scatter_color", "red")

    fig, ax = plt.subplots(figsize=figsize)
    train[column_y].plot(ax=ax, label="Train")

    if len(pred) == 1:
        ax.scatter(pred.index, pred.values, label="Prediction", zorder=5, s=scatter_size, color=scatter_color)
    else:
        pred.plot(ax=ax, label="Prediction")

    ax.relim()
    ax.autoscale_view()

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        dt_min, dt_max = df.index.min(), df.index.max()
        step = df.index.to_series().diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            fallback_days = int(settings.get("plots.predictions.fallback_step_days", 30))
            step = pd.Timedelta(days=fallback_days)
        padding_steps = int(settings.get("plots.predictions.x_axis_padding_steps", 2))
        pad = step * padding_steps
        ax.set_xlim(dt_min - pad, dt_max + pad)
    else:
        ax.set_xlim(df.index.min(), df.index.max())

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]
        for x in holidays_pred.index:
            ax.axvline(x=x, color="k", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    fig.tight_layout()
    return fig
