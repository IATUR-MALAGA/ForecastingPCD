# back/models/XGBoost/xgboost_model.py

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


# -----------------------------
# Helper para construir la X
# -----------------------------
def _build_X(df: pd.DataFrame, exog_cols: Optional[List[str]], column_y: str) -> pd.DataFrame:
    """
    Construye la matriz de features X a partir de un DataFrame que contiene la columna objetivo.

    - Si exog_cols tiene elementos: usamos TODAS las columnas menos la y
      (incluye exógenas originales + lags generados).
    - Si exog_cols está vacío o es None: usamos SOLO la columna y como feature.
    """
    if exog_cols is not None and len(exog_cols) > 0:
        # Hay exógenas -> queremos usar todo (lags incluidos) menos la y
        return df.drop(columns=[column_y])
    else:
        # Sin exógenas -> solo y como feature
        return df[[column_y]]


# -----------------------------
# Entrenar modelo XGBoost
# -----------------------------
def create_xgboost_model(
    train: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = "turistas",
    xgb_params: Optional[Dict] = None
) -> XGBRegressor:
    """
    Entrena un XGBRegressor sobre 'train', usando exog_cols (si hay) o solo y (si no).
    Devuelve el modelo entrenado.
    """

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,   # ¡OJO! debe ser escalar, no lista
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        }

    X_train = _build_X(train, exog_cols, column_y)
    y_train = train[column_y]

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    return model


# -----------------------------
# Predecir con XGBoost
# -----------------------------
def predict_xgboost(
    model_fit: XGBRegressor,
    df_future: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = "turistas",
) -> pd.Series:
    """
    Genera predicciones para df_future (mismas columnas que el train).

    - Usa la misma lógica de features que create_xgboost_model.
    - Devuelve una Serie con el mismo índice que df_future.
    """
    X_future = _build_X(df_future, exog_cols, column_y)
    preds = model_fit.predict(X_future)
    return pd.Series(preds, index=df_future.index, name=column_y)


# -----------------------------
# Búsqueda de mejores parámetros
# -----------------------------
def best_xgboost_params(
    df: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = "turistas",
    periodos_a_predecir: int = 2,
    random_state: int = 42
) -> Dict:
    """
    Devuelve los mejores hiperparámetros para XGBoost en una serie temporal.

    - df: contiene y + exógenas + lags.
    - exog_cols: lista de columnas exógenas originales (puede estar vacía).
    - Si NO hay exógenas: usa SOLO y como feature.
    - periodos_a_predecir: se reservan los últimos N periodos (no se usan en la búsqueda).
    """

    param_grid = {
        # Capacidad del modelo
        "n_estimators":     [100, 200, 400],      # pocos árboles: suficiente para 67 puntos
        "max_depth":        [2, 3, 4, 5],         # árboles poco profundos

        # Regularización de nodos/hojas
        "min_child_weight": [1, 5, 10],          # más alto => más suave, menos overfitting

        # Aprendizaje y muestreo
        "learning_rate":    [0.03, 0.05, 0.1],   # no muy pequeño (no tienes 10k puntos)
        "subsample":        [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],

        # Regularización L2 y L1
        "reg_lambda":       [0.0, 1.0, 5.0],
        "reg_alpha":        [0.0, 0.1, 1.0],
    }
    df = df.copy()
    if periodos_a_predecir >= len(df):
        raise ValueError(
            f"periodos_a_predecir={periodos_a_predecir} es mayor o igual que el número de filas ({len(df)})."
        )

    # 1) Quitamos los últimos N periodos (futuro)
    df_train = df.iloc[:-periodos_a_predecir]

    # 2) Construimos X e y con la misma lógica que el entrenamiento
    y_train = df_train[column_y]
    X_train = _build_X(df_train, exog_cols, column_y)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,              # para no pelearte con debugpy
        random_state=42,
    )

    cv = TimeSeriesSplit(n_splits=3)  # con 67 puntos, 3 splits es más razonable que 5

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,   # con este grid, 15–30 iteraciones van bien
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )


    print(f"[best_xgboost_params] X_train shape={X_train.shape}, y_train shape={y_train.shape}")

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_params
