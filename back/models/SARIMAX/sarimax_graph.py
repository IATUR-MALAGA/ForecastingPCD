import matplotlib.pyplot as plt

def plot_predictions(
    df,
    pred,               # Series/DataFrame with predictions, indexed by date
    title='Tourist Visitors Prediction',
    ylabel='Number of Tourists',
    xlabel='Date',
    column_y='y_turistas',
    periodos_a_predecir=2,
    holidays_col=None   # optional: no holidays by default
):
    """
    Plot train (actual values before `start_test_date`) and predictions.

    - `df` must have a DatetimeIndex.
    - `pred` should be indexed by date (same freq as df).
    - If `holidays_col` is provided and exists in `df`, holidays falling
      within the prediction period are marked with vertical lines.

    Returns the Matplotlib Axes object.
    """
    # 1) Train = data before the chosen test/forecast start date
    train = df.iloc[:-periodos_a_predecir]

    # 2) Plot train actual values
    ax = train[column_y].plot(
        legend=True,
        label='Train',
        title=title
    )

    # 3) Plot predictions
    pred.plot(
        legend=True,
        label='Prediction',
        ax=ax
    )

    ax.autoscale(axis='x', tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # 4) Vertical lines on holidays during the prediction period
    if holidays_col is not None and holidays_col in df.columns:
        # Use only holidays in the date range covered by the predictions
        # (intersection of df index with pred index)
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]

        for x in holidays_pred.index:
            ax.axvline(x=x, color='k', alpha=0.3)

    return ax
