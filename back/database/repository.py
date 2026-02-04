from typing import Any, Dict, List
from .core import fetch_data
from .queries import *
from psycopg import sql


def get_all_tables_in_schema(schema: str = "IA") -> List[str]:
    return fetch_data(GET_TABLES_IN_SCHEMA, (schema,))

def get_table_columns(schema: str, table: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_COLUMNS_IN_TABLE, (schema, table))

def get_names_in_table_catalog() -> List[Dict[str, Any]]:
    return fetch_data(GET_CATALOG_NAMES)

def get_all_data(schema: str, table: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_ALL_DATA_IN_TABLE)

def get_categories_in_catalog() -> List[Dict[str, Any]]:
    return fetch_data(GET_CATEGORIES)

def get_metadata_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_METADATA_FOR_VARIABLE, (nombre,))

def get_date_range_for_variable(nombre_tabla: str, schema: str = "IA") -> List[Dict[str, Any]]:
    q = GET_DATE_RANGE_FOR_VARIABLE.format(
        schema=sql.Identifier(schema),
        nombre_tabla=sql.Identifier(nombre_tabla)
    )
    return fetch_data(q)

def get_filters_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_FILTERS_FOR_VARIABLE, (nombre, nombre))


def get_distinct_values_for_column(schema: str, table: str, column: str) -> List[str]:
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
    return [r.get("value") for r in rows if r.get("value") is not None]

def get_tableName_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_TABLE_NAME_FOR_VARIABLE, (nombre,))   


from typing import Optional, Tuple
from psycopg import sql

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
    if agg_u not in {"SUM", "AVG", "MIN", "MAX"}:
        raise ValueError(f"Agregación no permitida: {agg}")

    filters = filters or {}
    params: list[Any] = []
    clauses: list[sql.Composable] = []

    # filtros: comparamos contra col::text, porque el selectize usa DISTINCT col::text
    for col, vals in filters.items():
        if not vals:
            continue
        ph = sql.SQL(", ").join([sql.Placeholder() for _ in vals])
        clauses.append(
            sql.SQL("{col}::text IN ({ph})").format(
                col=sql.Identifier(col),
                ph=ph
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

    return fetch_data(q, tuple(params))

def get_bool_group_filters(filtro: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_BOOL_GROUP_FILTERS, (filtro,))