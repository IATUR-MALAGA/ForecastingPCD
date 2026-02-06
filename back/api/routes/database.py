from typing import Any
from fastapi import APIRouter, Query

from back.api.deps import ensure_pg_identifier

from back.database.repository import (
    get_all_tables_in_schema,
    get_table_columns,
    get_names_in_table_catalog,
    get_categories_in_catalog,
    get_distinct_values_for_column,
    get_metadata_for_variable,
    get_date_range_for_variable,
    get_filters_for_variable,
    get_tableName_for_variable,
    get_bool_group_filters
)

# Si NO renombraste y sigue siendo router.py, usa esto:
# from back.database.router import (...)

router = APIRouter(prefix="/database", tags=["database"])

@router.get("/schemas/{schema}/tables", response_model=list[str])
def list_tables(schema: str):
    schema = ensure_pg_identifier(schema, "schema")
    return get_all_tables_in_schema(schema)

@router.get("/schemas/{schema}/tables/{table}/columns", response_model=list[dict[str, Any]])
def list_columns(schema: str, table: str):
    schema = ensure_pg_identifier(schema, "schema")
    table = ensure_pg_identifier(table, "table")
    return get_table_columns(schema, table)

@router.get("/names", response_model=list[dict[str, Any]])
def catalog_names():
    return get_names_in_table_catalog()

@router.get("/categories", response_model=list[dict[str, Any]])
def catalog_categories():
    return get_categories_in_catalog()

@router.get("/distinct/{schema}/{table}/{column}", response_model=list[str])
def distinct_values(schema: str, table: str, column: str):
    schema = ensure_pg_identifier(schema, "schema")
    table = ensure_pg_identifier(table, "table")
    column = ensure_pg_identifier(column, "column")
    return get_distinct_values_for_column(schema, table, column)

@router.get("/{nombre}/metadata", response_model=list[dict[str, Any]])
def variable_metadata(nombre: str):
    return get_metadata_for_variable(nombre)

@router.get("/{nombre}/filters", response_model=list[dict[str, Any]])
def variable_filters(nombre: str):
    return get_filters_for_variable(nombre)

@router.get("/{nombre}/table", response_model=list[dict[str, Any]])
def variable_table(nombre: str):
    return get_tableName_for_variable(nombre)

@router.get("/{nombre_tabla}/date-range", response_model=list[dict[str, Any]])
def variable_date_range(nombre_tabla: str, schema: str = Query("IA")):
    schema = ensure_pg_identifier(schema, "schema")
    nombre_tabla = ensure_pg_identifier(nombre_tabla, "nombre_tabla")
    return get_date_range_for_variable(nombre_tabla, schema=schema)

@router.get("/bool-group/{filtro}", response_model=list[dict[str, Any]])
def bool_group_filters(filtro: str):
    return get_bool_group_filters(filtro)