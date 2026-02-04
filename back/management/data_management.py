from datetime import date, datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from psycopg import sql

from back.database.core import fetch_data
from back.database.repository import get_bool_group_filters, get_metadata_for_variable

DateLike = Union[date, datetime, str]


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.fromisoformat(d[:10]).date()
    raise TypeError(f"Tipo de fecha no soportado: {type(d)}")


def check_date_and_temporality(
    start_date_1: DateLike,
    start_date_2: DateLike,
    end_date_1: DateLike,
    end_date_2: DateLike,
    temporality_1: str,
    temporality_2: str,
) -> bool:
    if temporality_1 is None or temporality_2 is None:
        return False
    if temporality_1.strip().lower() != temporality_2.strip().lower():
        return False

    s1, e1 = _to_date(start_date_1), _to_date(end_date_1)
    s2, e2 = _to_date(start_date_2), _to_date(end_date_2)

    if s1 > e1 or s2 > e2:
        return False

    return (s2 >= s1) and (e2 <= e1)


def get_col_ref_and_table(nombre: str) -> Tuple[str, str, str]:
    rows = get_metadata_for_variable(nombre)
    if not rows:
        raise ValueError(f"No existe metadata para la variable '{nombre}' en tbl_catalogo_variables")

    row = rows[0]
    col_ref = row.get("nombre_colum_ref")
    table = row.get("nombre_tabla")
    name = row.get("nombre") or nombre

    if not col_ref or not table:
        raise ValueError(
            f"Metadata incompleta para '{nombre}': nombre_colum_ref={col_ref}, nombre_tabla={table}"
        )

    return col_ref, table, name


def _rows_to_df(rows, columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)

    if isinstance(rows[0], dict):
        return pd.DataFrame(rows).reindex(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def _safe_alias(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\W+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def create_where_clauses(
    filters_by_var: Optional[dict[str, list[dict]]],
    var_name: str,
    target_table: Optional[str] = None,
) -> Tuple[list[sql.Composable], list, list[str]]:
    """
    Devuelve:
      - clauses: lista de trozos SQL para el WHERE (se unirán con AND)
      - params: parámetros para placeholders
      - group_cols: columnas por las que hay que agrupar (según union_grupo)
    """
    clauses: list[sql.Composable] = []
    params: list = []
    group_cols: list[str] = []

    if not filters_by_var:
        return clauses, params, group_cols

    filters_for_var = filters_by_var.get(var_name, []) or []
    for f in filters_for_var:
        f_table = f.get("table")
        column = f.get("col")
        values = f.get("values", []) or []

        # si target_table está definido, solo aplico filtros de esa tabla
        if target_table and f_table and f_table != target_table:
            continue

        if column and values:
            placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(values))
            clause = sql.SQL("{col}::text IN ({vals})").format(
                col=sql.Identifier(column),
                vals=placeholders
            )
            clauses.append(clause)
            params.extend(values)

            # union_grupo -> se agrupa por esa columna (asunción)
            try:
                gb = get_bool_group_filters(column) or []
                if gb and gb[0].get("union_grupo") == 1:
                    group_cols.append(column)
            except Exception:
                # si repo falla o no hay config, no agrupamos
                pass

    # unique preservando orden
    seen = set()
    group_cols = [c for c in group_cols if not (c in seen or seen.add(c))]

    return clauses, params, group_cols


def create_dataframe_based_on_selection(
    target_var: str,
    predictors: List[str],
    filters_by_var: dict[str, list[dict]] | None = None
) -> pd.DataFrame:
    # --- Target ---
    target_col, target_table, target_name = get_col_ref_and_table(target_var)
    target_alias = _safe_alias(target_name or target_col)

    where_clauses_target, target_params, group_cols_target = create_where_clauses(
        filters_by_var, target_name, target_table=target_table
    )
    where_sql_target = sql.SQL(" AND ").join(where_clauses_target) if where_clauses_target else sql.SQL("TRUE")

    group_select = sql.SQL("")
    if group_cols_target:
        group_select = sql.SQL(", ").join(sql.Identifier(c) for c in group_cols_target) + sql.SQL(", ")

    group_by = sql.SQL(", ").join(
        [*(sql.Identifier(c) for c in group_cols_target), sql.Identifier("anio"), sql.Identifier("mes")]
    )

    q_target = sql.SQL("""
        SELECT {group_select} SUM({col}) AS {alias}, anio, mes
        FROM "IA".{table}
        WHERE {where}
        GROUP BY {group_by}
    """).format(
        group_select=group_select,
        col=sql.Identifier(target_col),
        alias=sql.Identifier(target_alias),
        table=sql.Identifier(target_table),
        where=where_sql_target,
        group_by=group_by
    )

    target_cols = [*group_cols_target, target_alias, "anio", "mes"]
    target_rows = fetch_data(q_target, target_params)
    df = _rows_to_df(target_rows, target_cols)

    # claves base del DF target
    base_keys = [*group_cols_target, "anio", "mes"]

    # --- Predictors ---
    for i, p in enumerate(predictors, start=1):
        p_col, p_table, p_name = get_col_ref_and_table(p)
        p_alias = _safe_alias(p_name or f"pred_{i}_{p_col}")

        where_clauses_p, p_params, _group_cols_p = create_where_clauses(
            filters_by_var, p_name, target_table=p_table
        )
        where_sql_p = sql.SQL(" AND ").join(where_clauses_p) if where_clauses_p else sql.SQL("TRUE")

        # Si predictor está en la MISMA tabla que el target, puedo desglosar por los mismos grupos
        group_cols_pred = group_cols_target if (p_table == target_table) else []
        pred_group_select = sql.SQL("")
        if group_cols_pred:
            pred_group_select = sql.SQL(", ").join(sql.Identifier(c) for c in group_cols_pred) + sql.SQL(", ")

        pred_group_by = sql.SQL(", ").join(
            [*(sql.Identifier(c) for c in group_cols_pred), sql.Identifier("anio"), sql.Identifier("mes")]
        )

        # Agregamos predictor por mes (y por grupo si aplica) para evitar duplicados al merge
        q_pred = sql.SQL("""
            SELECT {group_select} SUM({col}) AS {alias}, anio, mes
            FROM "IA".{table}
            WHERE {where}
            GROUP BY {group_by}
        """).format(
            group_select=pred_group_select,
            col=sql.Identifier(p_col),
            alias=sql.Identifier(p_alias),
            table=sql.Identifier(p_table),
            where=where_sql_p,
            group_by=pred_group_by
        )

        pred_cols = [*group_cols_pred, p_alias, "anio", "mes"]
        pred_rows = fetch_data(q_pred, p_params)
        df_pred = _rows_to_df(pred_rows, pred_cols)

        # Merge: si df_pred no tiene columnas de grupo (porque tabla distinta), se hace por tiempo
        join_keys = ["anio", "mes"]
        if group_cols_target and group_cols_pred:
            join_keys = base_keys

        df = df.merge(df_pred, on=join_keys, how="left")

    if not df.empty:
        df = df.sort_values(base_keys).reset_index(drop=True)

    return df
