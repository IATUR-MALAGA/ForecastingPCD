from typing import Any, Dict, List
from .core import fetch_data
from .queries import *

def get_all_tables_in_schema(schema: str = "IA") -> List[str]:
    return fetch_data(GET_TABLES_IN_SCHEMA, (schema,))

def get_table_columns(schema: str, table: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_COLUMNS_IN_TABLE, (schema, table))

def get_names_in_table_catalog() -> List[Dict[str, Any]]:
    return fetch_data(GET_CATALOG_NAMES)

def get_all_data(schema: str, table: str) -> List[Dict[str, Any]]:
    return fetch_data(GET_DATA_IN_TABLE)