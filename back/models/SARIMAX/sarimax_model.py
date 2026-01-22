from statsmodels.tsa.statespace.sarimax import SARIMAX
from back.data_management import laggify_df
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
    max_lag=0
):
    p_values=[0, 2]
    d_values=[0, 2]
    q_values=[0, 2]
    P_values=[0, 2]
    D_values=[0, 2]
    Q_values=[0, 2]
    # 1) Mismo rango temporal para y y X
    df = df.iloc[:-periodos_a_predecir]  # los datos menos los últimos N periodos
    y_train = df[column_y]
    x_train = df[exog_cols]

    # 2) Mínimos y máximos a partir de las listas
    p_min, p_max = min(p_values), max(p_values)
    q_min, q_max = min(q_values), max(q_values)
    P_min, P_max = min(P_values), max(P_values)
    Q_min, Q_max = min(Q_values), max(Q_values)

    max_d = max(d_values)
    max_D = max(D_values)

    model = auto_arima(
        y=y_train,
        X=(x_train if not x_train.empty else None),
        # No estacional
        start_p=p_min,
        max_p=p_max,
        start_q=q_min,
        max_q=q_max,
        max_d=max_d,      # d será 0..max_d (si quieres d fijo, pon d=valor y quita max_d)

        # Estacional
        seasonal=True,
        m=s,
        start_P=P_min,
        max_P=P_max,
        start_Q=Q_min,
        max_Q=Q_max,
        max_D=max_D,      # igual que d: 0..max_D (o D=fijo)

        test='adf',
        stepwise=True,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )

    order = model.order
    seas = model.seasonal_order
    return order, seas