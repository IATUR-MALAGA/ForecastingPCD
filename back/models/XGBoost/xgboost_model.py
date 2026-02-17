# back/models/XGBoost/xgboost_model.py

from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from back.config import settings


def _build_X(df: pd.DataFrame, feature_cols: Optional[List[str]], column_y: str) -> pd.DataFrame:
    if feature_cols is not None and len(feature_cols) > 0:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en X: {missing[:20]}")
        return df[feature_cols]

    if column_y not in df.columns:
        raise ValueError(f"column_y '{column_y}' no está en df.columns")
    X = df.drop(columns=[column_y])
    if X.shape[1] == 0:
        raise ValueError("No hay features: df solo contiene la variable objetivo.")
    return X


def _make_preprocess_and_cols(X: pd.DataFrame):
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


def _default_xgb_params() -> Dict[str, Any]:
    params = dict(settings.get("models.xgboost.default_params", {}) or {})
    params.setdefault("objective", settings.get("models.xgboost.engine.objective", "reg:squarederror"))
    params.setdefault("tree_method", settings.get("models.xgboost.engine.tree_method", "hist"))
    params.setdefault("n_jobs", int(settings.get("models.xgboost.engine.n_jobs", 1)))
    params.setdefault("random_state", int(settings.get("models.xgboost.random_state", 42)))
    return params


def create_xgboost_model(
    train: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = settings.get("models.xgboost.target_column", "turistas"),
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    if xgb_params is None:
        xgb_params = _default_xgb_params()

    X_train = _build_X(train, exog_cols, column_y)
    y_train = train[column_y]

    preprocess, cat_cols, num_cols = _make_preprocess_and_cols(X_train)

    model = XGBRegressor(**xgb_params)

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def predict_xgboost(
    model_fit: Pipeline,
    df_future: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = settings.get("models.xgboost.target_column", "turistas"),
) -> pd.Series:
    X_future = _build_X(df_future, exog_cols, column_y)
    preds = model_fit.predict(X_future)
    return pd.Series(preds, index=df_future.index, name=column_y)


def best_xgboost_params(
    df: pd.DataFrame,
    exog_cols: Optional[List[str]],
    column_y: str = settings.get("models.xgboost.target_column", "turistas"),
    periodos_a_predecir: int = int(settings.get("models.xgboost.auto_search.periodos_a_predecir", 2)),
    random_state: int = int(settings.get("models.xgboost.random_state", 42)),
) -> Dict[str, Any]:
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

    df_train = df.iloc[:-periodos_a_predecir]
    print(f"  df_train.shape={df_train.shape}")

    y_train = df_train[column_y]
    print(f"  y_train.shape={y_train.shape} y_train.nan={y_train.isna().sum()}")

    X_train = _build_X(df_train, exog_cols, column_y)
    print(f"  X_train.shape={X_train.shape}")
    print("  X_train dtypes (top):")
    print(X_train.dtypes.value_counts())

    preprocess, cat_cols, num_cols = _make_preprocess_and_cols(X_train)
    if cat_cols:
        print(f"  cat_cols={cat_cols[:20]}")
    print(f"  num_cols(n)={len(num_cols)}")

    raw_grid = settings.get("models.xgboost.auto_search.param_grid", {}) or {}
    param_grid = {f"model__{k}": v for k, v in raw_grid.items()}

    xgb = XGBRegressor(
        objective=settings.get("models.xgboost.engine.objective", "reg:squarederror"),
        tree_method=settings.get("models.xgboost.engine.tree_method", "hist"),
        n_jobs=int(settings.get("models.xgboost.engine.n_jobs", 1)),
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", xgb)])

    cv = TimeSeriesSplit(n_splits=int(settings.get("models.xgboost.auto_search.cv_splits", 3)))

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=int(settings.get("models.xgboost.auto_search.n_iter", 20)),
        scoring=settings.get("models.xgboost.auto_search.scoring", "neg_root_mean_squared_error"),
        cv=cv,
        random_state=random_state,
        n_jobs=int(settings.get("models.xgboost.auto_search.n_jobs", 1)),
        verbose=int(settings.get("models.xgboost.auto_search.verbose", 2)),
        error_score=settings.get("models.xgboost.auto_search.error_score", "raise"),
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

    clean = {k.replace("model__", ""): v for k, v in best_params.items()}
    print(f"[best_xgboost_params] OK best_params(clean)={clean}")
    return clean
