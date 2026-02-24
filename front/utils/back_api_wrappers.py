import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

# Configurable por entorno
BASE_URL = os.getenv("BACK_API_URL", "http://127.0.0.1:8000")
TIMEOUT = float("600")

_client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)

def _p(seg: str) -> str:
    # para database con espacios, acentos, etc.
    return quote(str(seg), safe="")

def _get(path: str, params: Optional[dict] = None) -> Any:
    r = _client.get(path, params=params)
    r.raise_for_status()
    return r.json()

def _post(path: str, payload: dict) -> Any:
    r = _client.post(path, json=payload)
    r.raise_for_status()
    return r.json()
################################################################################################
# --- Wrappers DATABASE ---
################################################################################################

def get_names_in_table_catalog() -> List[Dict[str, Any]]:
    return _get("/api/database/names")

def get_metadata_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return _get(f"/api/database/{_p(nombre)}/metadata")

def get_date_range_for_variable(nombre_tabla: str, schema: str = "IA") -> List[Dict[str, Any]]:
    return _get(f"/api/database/{_p(nombre_tabla)}/date-range", params={"schema": schema})

def get_filters_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return _get(f"/api/database/{_p(nombre)}/filters")

def get_distinct_values_for_column(schema: str, table: str, column: str) -> List[str]:
    return _get(f"/api/database/distinct/{_p(schema)}/{_p(table)}/{_p(column)}")

def get_table_columns(schema: str, table: str) -> List[Dict[str, Any]]:
    return _get(f"/api/database/schemas/{_p(schema)}/tables/{_p(table)}/columns")

def get_tableName_for_variable(nombre: str) -> List[Dict[str, Any]]:
    return _get(f"/api/database/{_p(nombre)}/table")

################################################################################################
# --- Wrappers MODELS ---
################################################################################################


def sarimax_run(payload: dict) -> dict:
    return _post("/api/models/sarimax/run", payload)

def xgboost_run(payload: dict) -> dict:
    return _post("/api/models/xgboost/run", payload)
 