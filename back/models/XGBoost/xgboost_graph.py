import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

def plot_predictions(
    df: pd.DataFrame,
    pred,
    title: str = "XGBoost Prediction",
    ylabel: str = "Valor",
    xlabel: str = "Fecha",
    column_y: str = "y_turistas",
    periodos_a_predecir: int = 2,
    holidays_col: str | None = None,
):
    train = df.iloc[:-periodos_a_predecir]

    if not isinstance(pred.index, pd.DatetimeIndex):
        pred = pd.Series(pred.values, index=df.index[-len(pred):], name=getattr(pred, "name", None))

    fig, ax = plt.subplots(figsize=(12, 4))

    train[column_y].plot(ax=ax, label="Train")
    pred.plot(ax=ax, label="XGBoost Prediction")

    ax.autoscale(axis="x", tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_pred[holidays_col] == 1].index]
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
