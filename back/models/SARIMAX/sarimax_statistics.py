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


def compute_metrics(pred, df_test, indicador=settings.get("models.metrics.target_column", "turistas") ):
    
    mape = mean_absolute_error(df_test[indicador], _get_pred_values(pred)) / df_test[indicador].mean() * 100
    rmse = root_mean_squared_error(df_test[indicador], _get_pred_values(pred))
    mae = mean_absolute_error(df_test[indicador], _get_pred_values(pred))
    
    return mape, rmse, mae
