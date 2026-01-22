# back/models/XGBoost/xgboost_graph.py

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(
    df: pd.DataFrame,
    pred,               # Series/DataFrame with predictions, indexed by date
    title: str = "XGBoost Prediction",
    ylabel: str = "Valor",
    xlabel: str = "Fecha",
    column_y: str = "y_turistas",
    periodos_a_predecir: int = 2,
    holidays_col: str | None = None,   # opcional
):
    """
    Gráfico para XGBoost (muy similar al de SARIMAX).

    - df: DataFrame con la serie original, índice de fechas.
    - pred: Serie/DataFrame con las predicciones, indexada por fecha.
    - column_y: nombre de la columna objetivo en df.
    - periodos_a_predecir: número de pasos futuros (para separar train).
    - holidays_col: nombre de columna de festivos (0/1) opcional.

    Devuelve el objeto Axes de Matplotlib.
    """

    # 1) Train = datos antes del tramo de predicción
    train = df.iloc[:-periodos_a_predecir]

    # 2) Serie real de entrenamiento
    ax = train[column_y].plot(
        legend=True,
        label="Train",
        title=title,
    )

    # 3) Predicciones
    pred.plot(
        legend=True,
        label="XGBoost Prediction",
        ax=ax,
    )

    ax.autoscale(axis="x", tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # 4) Festivos durante el periodo de predicción (si aplica)
    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]

        for x in holidays_pred.index:
            ax.axvline(x=x, color="k", alpha=0.3)

    return ax
