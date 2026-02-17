from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from psycopg import sql

from back.config import settings

from .core import fetch_data
from .queries import (
    GET_TABLES_IN_SCHEMA,
    GET_COLUMNS_IN_TABLE,
    GET_CATALOG_NAMES,
    GET_CATEGORIES,
    GET_METADATA_FOR_VARIABLE,
    GET_DATE_RANGE_FOR_VARIABLE,
    GET_FILTERS_FOR_VARIABLE,
    GET_TABLE_NAME_FOR_VARIABLE,
    GET_BOOL_GROUP_FILTERS,
)


# -------------------------
# Helpers (internos)
# -------------------------

def _as_str_list(rows: Sequence[Any], key: str) -> list[str]:
    """
    Convierte rows a list[str].
    Soporta filas como dict_row ({"table_name": ...}) o tuplas (("tbl",),).
    """
    out: list[str] = []
    for r in rows or []:
        if isinstance(r, dict):
            v = r.get(key)
        elif isinstance(r, (list, tuple)) and r:
            v = r[0]
        else:
            v = r

        if v is not None:
            out.append(str(v))
    return out


# -------------------------
# Catalog / metadata
# -------------------------

def get_all_tables_in_schema(schema: str = settings.get("db.default_schema", "IA")) -> list[str]:
    rows = fetch_data(GET_TABLES_IN_SCHEMA, (schema,)) or []
    return _as_str_list(rows, key="table_name")

def get_table_columns(schema: str, table: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_COLUMNS_IN_TABLE, (schema, table)) or []

def get_names_in_table_catalog() -> List[Dict[str, Any]]:
    return fetch_data(GET_CATALOG_NAMES) or []

def get_categories_in_catalog() -> List[Dict[str, Any]]:
    return fetch_data(GET_CATEGORIES) or []

def get_metadata_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_METADATA_FOR_VARIABLE, (nombre,)) or []

def get_filters_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_FILTERS_FOR_VARIABLE, (nombre, nombre)) or []

def get_tableName_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_TABLE_NAME_FOR_VARIABLE, (nombre,)) or []

def get_bool_group_filters(filtro: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_BOOL_GROUP_FILTERS, (filtro,)) or []


# -------------------------
# Safe dynamic-table queries
# -------------------------

def get_date_range_for_variable(nombre_tabla: str, schema: str = settings.get("db.default_schema", "IA")) -> List[Dict[str, Any]]:
    try:
        # GET_DATE_RANGE_FOR_VARIABLE ya es sql.SQL(...), así que .format con Identifiers es correcto y seguro.
        q = GET_DATE_RANGE_FOR_VARIABLE.format(
            schema=sql.Identifier(schema),
            nombre_tabla=sql.Identifier(nombre_tabla),
        )
        return fetch_data(q) or []
    except Exception as e:
        print(f"Error in get_date_range_for_variable: {e}")
        return []

def get_distinct_values_for_column(schema: str, table: str, column: str) -> list[str]:
    q = sql.SQL("""
        SELECT DISTINCT {col}::text AS value
        FROM {schema}.{table}
        WHERE {col} IS NOT NULL
        ORDER BY 1;
    """).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        col=sql.Identifier(column),
    )

    rows = fetch_data(q) or []
    # rows suele venir como [{"value": "..."}]
    return _as_str_list(rows, key="value")

def get_all_data(
    schema: str,
    table: str,
    limit: int = int(settings.get("db.queries.default_limit", 100)),
    offset: int = int(settings.get("db.queries.default_offset", 0)),
) -> List[Dict[str, Any]]:
    """
    ⚠️ Ojo: exponer esto como endpoint sin límites es peligroso.
    Por eso obligamos a limit/offset y usamos Identifiers (safe).
    """
    q = sql.SQL("SELECT * FROM {}.{} LIMIT %s OFFSET %s").format(
        sql.Identifier(schema),
        sql.Identifier(table),
    )
    return fetch_data(q, (limit, offset)) or []


# -------------------------
# Series
# -------------------------

def get_monthly_series_with_filters(
    schema: str,
    table: str,
    value_col: str,
    filters: Optional[dict[str, list[str]]] = None,
    agg: str = "SUM",
) -> List[Dict[str, Any]]:
    """
    Devuelve filas: [{"fecha": date, "value": number}, ...]
    - Agrega por mes (fecha = primer día del mes)
    - Aplica filtros usando col::text IN (...), consistente con DISTINCT::text
    """
    agg_u = (agg or "SUM").strip().upper()
    allowed_aggs = [str(x).upper() for x in settings.get("db.queries.allowed_aggregations", ["SUM", "AVG", "MIN", "MAX"]) or []]
    if agg_u not in set(allowed_aggs):
        raise ValueError(f"Agregación no permitida: {agg}")

    filters = filters or {}
    params: list[Any] = []
    clauses: list[sql.Composable] = []

    for col, vals in filters.items():
        if not vals:
            continue
        ph = sql.SQL(", ").join([sql.Placeholder() for _ in vals])
        clauses.append(
            sql.SQL("{col}::text IN ({ph})").format(
                col=sql.Identifier(col),
                ph=ph,
            )
        )
        params.extend(vals)

    where_extra = sql.SQL("")
    if clauses:
        where_extra = sql.SQL(" AND ") + sql.SQL(" AND ").join(clauses)

    month_case = sql.SQL("""
        CASE
            WHEN mes_txt ~ '^[0-9][0-9]?$'
                AND mes_txt::int BETWEEN 1 AND 12
                THEN mes_txt::int
            WHEN mes_txt = 'enero' THEN 1
            WHEN mes_txt = 'febrero' THEN 2
            WHEN mes_txt = 'marzo' THEN 3
            WHEN mes_txt = 'abril' THEN 4
            WHEN mes_txt = 'mayo' THEN 5
            WHEN mes_txt = 'junio' THEN 6
            WHEN mes_txt = 'julio' THEN 7
            WHEN mes_txt = 'agosto' THEN 8
            WHEN mes_txt IN ('septiembre','setiembre') THEN 9
            WHEN mes_txt = 'octubre' THEN 10
            WHEN mes_txt = 'noviembre' THEN 11
            WHEN mes_txt = 'diciembre' THEN 12
            ELSE NULL
        END
    """)

    q = sql.SQL("""
        WITH src AS (
            SELECT
                anio::int AS anio,
                lower(trim(mes::text)) AS mes_txt,
                {val_col}::numeric AS val
            FROM {schema}.{table}
            WHERE anio IS NOT NULL
              AND mes  IS NOT NULL
              AND {val_col} IS NOT NULL
              {where_extra}
        ),
        t AS (
            SELECT
                make_date(anio, {month_case}, 1) AS fecha,
                val
            FROM src
        )
        SELECT
            fecha,
            {agg}(val) AS value
        FROM t
        WHERE fecha IS NOT NULL
        GROUP BY fecha
        ORDER BY fecha;
    """).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        val_col=sql.Identifier(value_col),
        where_extra=where_extra,
        month_case=month_case,
        agg=sql.SQL(agg_u),
    )

    return fetch_data(q, tuple(params)) or []
