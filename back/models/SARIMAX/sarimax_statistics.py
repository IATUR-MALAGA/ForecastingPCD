import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from back.config import settings


def _get_pred_values(pred):
    """
    Robustly get predicted values (handles pandas Series, or (index, array)-like).
    Returns a numpy array of predictions.
    """
    if hasattr(pred, "values"):
        return np.asarray(pred.values)
    elif isinstance(pred, tuple) and len(pred) > 1:
        return np.asarray(pred[1])
    else:
        return np.asarray(pred)


def _mape(y_true, y_pred):
    """Compute standard MAPE (%) excluding zero targets to avoid division by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0

    ape = np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    return float(np.mean(ape) * 100.0)


def compute_metrics(pred, df_test, indicador=settings.get("models.metrics.target_column", "turistas")):
    y_true = np.asarray(df_test[indicador], dtype=float)
    y_pred = _get_pred_values(pred)

    # Alineación defensiva por si longitudes no coinciden por edge-cases de slicing
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = np.asarray(y_pred, dtype=float)[:n]

    mape = _mape(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return mape, rmse, mae
