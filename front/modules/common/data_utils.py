"""Helper utilities shared across front modules.

These utilities centralize common data loading and exogenous-variable helpers
that were duplicated across modules.
"""
from typing import Callable, Iterable, Dict, List
import pandas as pd
from pandas.api.types import is_integer_dtype


BuscarFunc = Callable[..., str]


def cargar_df(var_predict: str, buscar_func: BuscarFunc, *, date_column: str = "Mes") -> pd.DataFrame:
    """Load and normalize a dataframe for the given variable.

    Args:
        var_predict: Front-end variable name selected by the user.
        buscar_func: Function used to translate names (e.g., ``buscar_nombre_equivalente``).
        date_column: Column containing month information to convert into a datetime index.

    Returns:
        DataFrame indexed by a ``Fecha`` datetime column with integer columns promoted
        to float for downstream compatibility.
    """
    back_var = buscar_func(tipo="variable", nombre=var_predict)
    back_route = buscar_func(tipo="fichero", nombre=back_var)
    df = pd.read_csv(back_route)

    for col in df.columns:
        try:
            if is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype(float)
        except Exception:
            # If dtype inspection fails for any reason, skip silently.
            pass

    if date_column in df.columns:
        df["Fecha"] = pd.to_datetime(df[date_column] + "-01")
        df.set_index("Fecha", inplace=True)
        df.drop(date_column, axis=1, inplace=True)

    return df


def translate_var(front_var: str, buscar_func: BuscarFunc) -> str:
    """Translate a front variable name into its backend equivalent."""
    return buscar_func(tipo="variable", nombre=front_var)


def get_selected_exog(input_obj) -> List[str]:
    """Safely get selected exogenous variables from Shiny ``input``."""
    try:
        return input_obj.variables_exogenas() or []
    except AttributeError:
        return []


def translate_exog_vars(input_obj, buscar_func: BuscarFunc) -> List[str]:
    """Return backend variable names for the selected exogenous vars."""
    return [buscar_func(tipo="variable", nombre=var) for var in get_selected_exog(input_obj)]


def selected_lags(input_obj, buscar_func: BuscarFunc) -> Dict[str, List[int]]:
    """Return selected lags per exogenous variable using backend names."""
    lags: Dict[str, List[int]] = {}
    for var_exog in get_selected_exog(input_obj):
        back_var = buscar_func(tipo="variable", nombre=var_exog)
        try:
            lags_list_raw: Iterable[str] = getattr(input_obj, f"lags_{var_exog}")()
            lags_int = [int(l.replace("lag", "")) for l in lags_list_raw]
        except AttributeError:
            lags_int = []
        lags[back_var] = lags_int
    return lags


def max_selected_lag(input_obj) -> int:
    """Return the maximum lag number across all selected exogenous vars."""
    max_lag = 0
    for var_exog in get_selected_exog(input_obj):
        try:
            lags_list_raw = getattr(input_obj, f"lags_{var_exog}")()
            for lag in lags_list_raw:
                lag_num = int(lag.replace("lag", ""))
                max_lag = max(max_lag, lag_num)
        except AttributeError:
            continue
    return max_lag
