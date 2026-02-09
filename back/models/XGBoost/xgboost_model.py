# back/models/XGBoost/xgboost_model.py

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# -----------------------------
# Helper para construir la X
# -----------------------------
def _build_X(df: pd.DataFrame, feature_cols: Optional[List[str]], column_y: str) -> pd.DataFrame:
    """
    Construye la matriz de features X.

    Recomendación para series temporales:
    - X = (exógenas + lags + calendario) => columnas numéricas y categóricas
    - NO usar la y contemporánea como feature.

    Si feature_cols viene con elementos -> usa esas columnas.
    Si feature_cols es None o [] -> usa TODO menos column_y (si existe).
    """
    if feature_cols is not None and len(feature_cols) > 0:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en X: {missing[:20]}")
        return df[feature_cols]

    # fallback: todo menos y
    if column_y not in df.columns:
        raise ValueError(f"column_y '{column_y}' no está en df.columns")
    X = df.drop(columns=[column_y])
    if X.shape[1] == 0:
        raise ValueError("No hay features: df solo contiene la variable objetivo.")
    return X


def _make_preprocess_and_cols(X: pd.DataFrame):
    """Crea ColumnTransformer + listas de columnas numéricas/categóricas."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return preprocess, cat_cols, num_cols


# -----------------------------
# Entrenar modelo XGBoost (Pipeline)
# -----------------------------
def create_xgboost_model(
    train: pd.DataFrame,
    exog_cols: Optional[List[str]],     # aquí exog_cols = feature_cols reales (exógenas+lags+etc.)
    column_y: str = "turistas",
    xgb_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Entrena un Pipeline(preprocess + XGBRegressor) y lo devuelve entrenado.
    Esto evita errores por columnas categóricas (ej. 'mun_dest').
    """
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
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

    preprocess, cat_cols, num_cols = _make_preprocess_and_cols(X_train)

    model = XGBRegressor(**xgb_params)

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    return pipe


# -----------------------------
# Predecir con XGBoost (Pipeline)
# -----------------------------
def predict_xgboost(
    model_fit: Pipeline,
    df_future: pd.DataFrame,
    exog_cols: Optional[List[str]],   # feature_cols reales
    column_y: str = "turistas",
) -> pd.Series:
    """
    Predice con el Pipeline entrenado.
    """
    X_future = _build_X(df_future, exog_cols, column_y)
    preds = model_fit.predict(X_future)
    return pd.Series(preds, index=df_future.index, name=column_y)


# -----------------------------
# Búsqueda de mejores parámetros (Pipeline + OHE)
# -----------------------------
def best_xgboost_params(
    df: pd.DataFrame,
    exog_cols: Optional[List[str]],     # feature_cols reales
    column_y: str = "turistas",
    periodos_a_predecir: int = 2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    RandomizedSearchCV sobre Pipeline(prep + XGB) con TimeSeriesSplit.
    Devuelve parámetros del modelo (sin prefijo 'model__').
    """

    print("\n[best_xgboost_params] START")
    print(f"  df.shape={df.shape}")
    print(f"  column_y={column_y}")
    print(f"  periodos_a_predecir={periodos_a_predecir}")
    print(f"  exog_cols type={type(exog_cols)} len={0 if exog_cols is None else len(exog_cols)}")

    if column_y not in df.columns:
        raise ValueError(f"[best_xgboost_params] column_y '{column_y}' no está en df.columns")

    if periodos_a_predecir >= len(df):
        raise ValueError(
            f"[best_xgboost_params] periodos_a_predecir={periodos_a_predecir} >= len(df)={len(df)}"
        )

    # 1) Quitamos los últimos N periodos (futuro)
    df_train = df.iloc[:-periodos_a_predecir]
    print(f"  df_train.shape={df_train.shape}")

    # 2) y
    y_train = df_train[column_y]
    print(f"  y_train.shape={y_train.shape} y_train.nan={y_train.isna().sum()}")

    # 3) X
    X_train = _build_X(df_train, exog_cols, column_y)
    print(f"  X_train.shape={X_train.shape}")
    print("  X_train dtypes (top):")
    print(X_train.dtypes.value_counts())

    preprocess, cat_cols, num_cols = _make_preprocess_and_cols(X_train)
    if cat_cols:
        print(f"  cat_cols={cat_cols[:20]}")
    print(f"  num_cols(n)={len(num_cols)}")

    # grid (ojo con prefijo model__)
    param_grid = {
        "model__n_estimators":     [100, 200, 400],
        "model__max_depth":        [2, 3, 4, 5],
        "model__min_child_weight": [1, 5, 10],
        "model__learning_rate":    [0.03, 0.05, 0.1],
        "model__subsample":        [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__reg_lambda":       [0.0, 1.0, 5.0],
        "model__reg_alpha":        [0.0, 0.1, 1.0],
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", xgb),
    ])

    cv = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=2,
        error_score="raise",
    )

    print("  >>> llamando a random_search.fit(...) con Pipeline + OneHotEncoder")
    try:
        random_search.fit(X_train, y_train)
    except Exception as e:
        print("\n[best_xgboost_params] ERROR en fit():")
        print("  Tipo:", type(e).__name__)
        print("  Msg :", str(e))
        print("  X_train.head():")
        print(X_train.head(3))
        raise

    best_params = random_search.best_params_
    print(f"\n[best_xgboost_params] OK best_params(raw)={best_params}")

    # Devuelve params “limpios” para XGBRegressor (sin prefijo model__)
    clean = {k.replace("model__", ""): v for k, v in best_params.items()}
    print(f"[best_xgboost_params] OK best_params(clean)={clean}")
    return clean
