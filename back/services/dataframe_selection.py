from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple
import unicodedata

import numpy as np
import pandas as pd
from fastapi import HTTPException
from psycopg import sql

from back.config import settings
from back.database.repository import (
    get_aggregated_series,
    get_bool_group_filters,
    get_variable_definition,
    table_has_column,
)
from back.utils.column_utils import _safe_alias


DEFAULT_SCHEMA = settings.get("db.default_schema", "IA")


def _normalize_catalog_operation(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def resolve_catalog_aggregation(operation: str | None, default: str = "SUM") -> str:
    normalized = _normalize_catalog_operation(operation)
    if normalized in ("", "sum", "suma", "total"):
        return "SUM"
    if normalized in ("avg", "average", "mean", "media", "promedio"):
        return "AVG"
    if normalized in ("min", "minimum", "minimo"):
        return "MIN"
    if normalized in ("max", "maximum", "maximo"):
        return "MAX"
    return default


def get_target_aggregation(nombre: str, cache: Any = None) -> str:
    if cache and hasattr(cache, "get_meta"):
        row = cache.get_meta(nombre) or {}
    else:
        row = get_variable_definition(nombre) or {}
    return resolve_catalog_aggregation(row.get("operacion_obj"), default="SUM")


def get_col_ref_and_table(nombre: str, cache: Any = None) -> Tuple[str, str, str]:
    if cache and hasattr(cache, "get_meta"):
        row = cache.get_meta(nombre) or {}
    else:
        row = get_variable_definition(nombre) or {}

    if not row:
        raise ValueError(
            f"No existe metadata para la variable '{nombre}' en tbl_catalogo_variables"
        )

    col_ref = row.get("nombre_colum_ref")
    table = row.get("nombre_tabla")
    name = row.get("nombre") or nombre

    if not col_ref or not table:
        raise ValueError(
            f"Metadata incompleta para '{nombre}': nombre_colum_ref={col_ref}, nombre_tabla={table}"
        )

    return col_ref, table, name


def _rows_to_df(rows, columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)

    if isinstance(rows[0], dict):
        return pd.DataFrame(rows).reindex(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def create_where_clauses(
    filters_by_var: Optional[dict[str, list[dict]]],
    var_name: str,
    target_table: Optional[str] = None,
) -> Tuple[list[sql.Composable], list, list[str]]:
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

        if target_table and f_table and f_table != target_table:
            continue

        if f.get("kind") == "date_range" or column == "__date_range__":
            start = f.get("start")
            end = f.get("end")
            year_col = f.get("year_col")
            month_col = f.get("month_col")
            day_col = f.get("day_col")

            if not (start and end and year_col):
                raise ValueError(
                    f"Filtro 'date_range' incompleto para '{var_name}': "
                    f"se requieren start, end y year_col. Recibido: {f!r}"
                )

            if day_col and month_col:
                date_expr = sql.SQL("make_date({y}::int, {m}::int, {d}::int)").format(
                    y=sql.Identifier(year_col),
                    m=sql.Identifier(month_col),
                    d=sql.Identifier(day_col),
                )
            elif month_col:
                date_expr = sql.SQL("make_date({y}::int, {m}::int, 1)").format(
                    y=sql.Identifier(year_col),
                    m=sql.Identifier(month_col),
                )
            else:
                date_expr = sql.SQL("make_date({y}::int, 1, 1)").format(
                    y=sql.Identifier(year_col),
                )

            clauses.append(
                sql.SQL("{expr} BETWEEN %s::date AND %s::date").format(expr=date_expr)
            )
            params.extend([start, end])
            continue

        if column and values:
            placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(values))
            clause = sql.SQL("{col}::text IN ({vals})").format(
                col=sql.Identifier(column), vals=placeholders
            )
            clauses.append(clause)
            params.extend(values)

            try:
                gb = get_bool_group_filters(column) or []
                if gb and gb[0].get("union_grupo") == 1:
                    group_cols.append(column)
            except Exception:
                pass

    seen = set()
    group_cols = [c for c in group_cols if not (c in seen or seen.add(c))]

    return clauses, params, group_cols


@lru_cache(maxsize=256)
def _table_has_column_cached(schema: str, table: str, column: str) -> bool:
    return table_has_column(table, column, schema=schema)


def _detect_time_cols(table: str, schema: str = DEFAULT_SCHEMA) -> list[str]:
    cols = []
    for c in ("anio", "mes", "dia"):
        if _table_has_column_cached(schema, table, c):
            cols.append(c)
    if not cols:
        raise HTTPException(
            status_code=422,
            detail=f'La tabla "{schema}".{table} no tiene columnas temporales (anio/mes/dia).',
        )
    return cols


def _dt_series_from_time_cols(df_: pd.DataFrame, time_cols_: list[str]) -> pd.Series:
    year = df_["anio"].astype(int)
    month = df_["mes"].astype(int) if "mes" in time_cols_ else 1
    day = df_["dia"].astype(int) if "dia" in time_cols_ else 1
    return pd.to_datetime({"year": year, "month": month, "day": day}, errors="coerce")


def _future_dates(
    target_end_dt: pd.Timestamp, pred_end_dt: pd.Timestamp, time_cols_: list[str]
) -> pd.DatetimeIndex:
    if pred_end_dt <= target_end_dt:
        return pd.DatetimeIndex([])

    if "dia" in time_cols_:
        start = target_end_dt + pd.Timedelta(days=1)
        return pd.date_range(start=start, end=pred_end_dt, freq="D")
    if "mes" in time_cols_:
        start = target_end_dt + pd.offsets.MonthBegin(1)
        return pd.date_range(start=start, end=pred_end_dt, freq="MS")
    start = target_end_dt + pd.offsets.YearBegin(1)
    return pd.date_range(start=start, end=pred_end_dt, freq="YS")


def _time_cols_df_from_dates(
    dates: pd.DatetimeIndex, time_cols_: list[str]
) -> pd.DataFrame:
    out = {}
    if "anio" in time_cols_:
        out["anio"] = dates.year.astype(int)
    if "mes" in time_cols_:
        out["mes"] = dates.month.astype(int)
    if "dia" in time_cols_:
        out["dia"] = dates.day.astype(int)
    return pd.DataFrame(out)


def create_dataframe_based_on_selection(
    target_var: str,
    predictors: list[str],
    filters_by_var: dict[str, list[dict]] | None = None,
    cache: Any = None,
) -> pd.DataFrame:
    target_col, target_table, target_name = get_col_ref_and_table(
        target_var, cache=cache
    )
    target_alias = _safe_alias(target_name or target_col)
    target_agg = get_target_aggregation(target_name, cache=cache)

    time_cols = _detect_time_cols(target_table)

    where_clauses_target, target_params, group_cols_target = create_where_clauses(
        filters_by_var, target_name, target_table=target_table
    )
    target_rows = get_aggregated_series(
        schema=DEFAULT_SCHEMA,
        table=target_table,
        value_col=target_col,
        alias=target_alias,
        time_cols=time_cols,
        where_clauses=where_clauses_target,
        params=target_params,
        group_cols=group_cols_target,
        agg=target_agg,
    )

    target_cols = [*group_cols_target, target_alias, *time_cols]
    df_target = _rows_to_df(target_rows, target_cols)

    base_keys = [*group_cols_target, *time_cols]
    if df_target.empty:
        return df_target

    pred_dfs: list[tuple[pd.DataFrame, list[str]]] = []
    min_pred_end_dt: pd.Timestamp | None = None

    for i, p in enumerate(predictors or [], start=1):
        p_col, p_table, p_name = get_col_ref_and_table(p, cache=cache)
        p_alias = _safe_alias(p_name or f"pred_{i}_{p_col}")

        pred_time_cols = _detect_time_cols(p_table)
        if pred_time_cols != time_cols:
            raise ValueError(
                f"Granularidad temporal distinta.\n"
                f'- Target "{DEFAULT_SCHEMA}".{target_table}: {time_cols}\n'
                f'- Pred   "{DEFAULT_SCHEMA}".{p_table}: {pred_time_cols}'
            )

        where_clauses_p, p_params, _group_cols_p = create_where_clauses(
            filters_by_var, p_name, target_table=p_table
        )

        group_cols_pred = group_cols_target if p_table == target_table else []
        pred_rows = get_aggregated_series(
            schema=DEFAULT_SCHEMA,
            table=p_table,
            value_col=p_col,
            alias=p_alias,
            time_cols=time_cols,
            where_clauses=where_clauses_p,
            params=p_params,
            group_cols=group_cols_pred,
        )

        pred_cols = [*group_cols_pred, p_alias, *time_cols]
        df_pred = _rows_to_df(pred_rows, pred_cols)

        join_keys = [*time_cols]
        if group_cols_target and group_cols_pred:
            join_keys = base_keys

        pred_dfs.append((df_pred, join_keys))

        if not df_pred.empty:
            dt_max = _dt_series_from_time_cols(df_pred, time_cols).max()
            if pd.notna(dt_max):
                if min_pred_end_dt is None or dt_max < min_pred_end_dt:
                    min_pred_end_dt = dt_max

    df_base = df_target.copy()
    target_end_dt = _dt_series_from_time_cols(df_target, time_cols).max()

    if (
        min_pred_end_dt is not None
        and pd.notna(target_end_dt)
        and min_pred_end_dt > target_end_dt
    ):
        fut_dates = _future_dates(target_end_dt, min_pred_end_dt, time_cols)
        if len(fut_dates) > 0:
            df_time_future = _time_cols_df_from_dates(fut_dates, time_cols)

            if group_cols_target:
                df_groups = df_target[group_cols_target].drop_duplicates().copy()
                df_groups["_k"] = 1
                df_time_future["_k"] = 1
                df_future = df_groups.merge(df_time_future, on="_k", how="outer").drop(
                    columns=["_k"]
                )
            else:
                df_future = df_time_future

            df_future[target_alias] = np.nan
            df_base = pd.concat([df_target, df_future], ignore_index=True, sort=False)

    for df_pred, join_keys in pred_dfs:
        if df_pred is None or df_pred.empty:
            continue
        df_base = df_base.merge(df_pred, on=join_keys, how="left")

    if not df_base.empty:
        df_base = df_base.sort_values(base_keys).reset_index(drop=True)

    return df_base
