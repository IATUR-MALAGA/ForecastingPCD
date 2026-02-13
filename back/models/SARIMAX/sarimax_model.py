import traceback
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def create_sarimax_model(train, exog_cols, column_y='turistas',
                         order=(0, 1, 0),
                         seasonal_order=(0, 0, 0, 12)
                         ):
    """
    Fit a SARIMAX model with exogenous variables and
    return the fitted results object.
    """

 # los datos menos los últimos N periodos
    if exog_cols is None or len(exog_cols) == 0:
        exog = None
    else:
        exog = train[exog_cols]
    
    model = SARIMAX(
        train[column_y],
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_invertibility=False,
        enforce_stationarity=False
    )
    results = model.fit()
    return results #Devuelve el modelo entrenado con train


def predict_sarimax(model_fit, start, end, exog_cols):
    """
    Generate predictions using the fitted SARIMAX model
    on the test set.
    """
    

    pred = model_fit.predict(start = start, end = end, exog=exog_cols)
    return pred


def best_sarimax_params(
    df,
    exog_cols,
    column_y='turistas',
    s=12,
    periodos_a_predecir=2,
    seasonal=True
):
    # RANGOS (como ya tienes)
    p_values=[0, 2]
    d_values=[0, 2]
    q_values=[0, 2]
    P_values=[0, 2]
    D_values=[0, 2]
    Q_values=[0, 2]

    # 1) mismos rangos temporales para y y X
    df = df.iloc[:-periodos_a_predecir]
    y_train = df[column_y]
    x_train = df[exog_cols] if exog_cols else None
    if x_train is not None and x_train.empty:
        x_train = None

    # 2) min/max
    p_min, p_max = min(p_values), max(p_values)
    q_min, q_max = min(q_values), max(q_values)
    max_d = max(d_values)

    # Si NO es estacional: forzamos P,D,Q=0 y m=1 (pmdarima requiere m>=1)
    if not seasonal:
        P_min = P_max = 0
        Q_min = Q_max = 0
        max_D = 0
        m = 1
    else:
        P_min, P_max = min(P_values), max(P_values)
        Q_min, Q_max = min(Q_values), max(Q_values)
        max_D = max(D_values)
        m = s

    try:
        model = auto_arima(
            y=y_train,
            X=x_train,
            start_p=p_min, max_p=p_max,
            start_q=q_min, max_q=q_max,
            max_d=max_d,

            seasonal=seasonal, m=m,
            start_P=P_min, max_P=P_max,
            start_Q=Q_min, max_Q=Q_max,
            max_D=max_D,

            test='adf',
            stepwise=True,
            trace=True,
            error_action='trace',
            suppress_warnings=False
        )
    except Exception:
        traceback.print_exc()
        raise

    return model.order, model.seasonal_order