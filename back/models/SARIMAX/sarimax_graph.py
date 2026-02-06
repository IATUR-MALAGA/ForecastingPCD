import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

def plot_predictions(
    df,
    pred,
    title='Tourist Visitors Prediction',
    ylabel='Number of Tourists',
    xlabel='Date',
    column_y='y_turistas',
    periodos_a_predecir=2,
    holidays_col=None
):
    train = df.iloc[:-periodos_a_predecir]

    # Asegura que pred tenga índice de fechas (MUY recomendable)
    # Si pred viene con índice range(...) o no-datetime, lo alineamos al final del df
    if not isinstance(pred.index, pd.DatetimeIndex):
        pred = pd.Series(pred.values, index=df.index[-len(pred):], name=getattr(pred, "name", None))

    # Figura / eje nuevos SIEMPRE (evita acumulación entre renders)
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot sin legend=True aquí; la ponemos una sola vez al final
    train[column_y].plot(ax=ax, label="Train")
    pred.plot(ax=ax, label="Prediction")

    ax.autoscale(axis='x', tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    # Holidays
    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]
        for x in holidays_pred.index:
            ax.axvline(x=x, color='k', alpha=0.3)

    # (Opcional) Deduplicar por si algo añade líneas extra
    handles, labels = ax.get_legend_handles_labels()
    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    fig.tight_layout()
    return fig   # <-- devuelve figura (más cómodo para Shiny)
