import numpy as np
from back.data_management import split_train_test
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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


def compute_metrics(df_train, df_test, exog_cols, model, indicador='turistas' ):
    pred = model.predict(start=len(df_train), end=len(df_train)+len(df_test)-1, exog=exog_cols)
    
    mape = mean_absolute_error(df_test[indicador], _get_pred_values(pred)) / df_test[indicador].mean() * 100
    rmse = root_mean_squared_error(df_test[indicador], _get_pred_values(pred))
    mae = mean_absolute_error(df_test[indicador], _get_pred_values(pred))
    
    return mape, rmse, mae

def compute_elasticity(df, exog_col, results, column_y='turistas', train_size=0.7):
    """
    Compute the elasticity of a given exogenous variable
    on the test set using an already fitted model.
    """
    _ , test = split_train_test(df, train_size=train_size)

    start = test.index[0]
    end = test.index[-1]

    pred = results.predict(
        start=start,
        end=end,
        exog=test[[exog_col]]
    )

    elasticity = (np.cov(test[exog_col], pred) /
                    (np.var(test[exog_col]) * np.mean(pred))) * (np.mean(test[exog_col]))
    return elasticity