import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def _get_pred_values(pred):
    if hasattr(pred, "values"):
        return np.asarray(pred.values)
    elif isinstance(pred, tuple) and len(pred) > 1:
        return np.asarray(pred[1])
    else:
        return np.asarray(pred)

def compute_metrics(pred, df_test, indicador='turistas'):
    mape = mean_absolute_error(df_test[indicador], _get_pred_values(pred)) / df_test[indicador].mean() * 100
    rmse = root_mean_squared_error(df_test[indicador], _get_pred_values(pred))
    mae = mean_absolute_error(df_test[indicador], _get_pred_values(pred))
    return mape, rmse, mae
