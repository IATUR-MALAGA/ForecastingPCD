import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ForecastingPCD
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Dict, List
import json

from back.database.repository import (
    get_all_tables_in_schema,
    get_table_columns,
    get_names_in_table_catalog,
)
def _pretty(obj: Any) -> str:
    """Representación legible (JSON cuando se puede)."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def main() -> int:
    schema = "IA"
    table = "tbl_aena_pasajeros_mensual"

    print("=== PRUEBA DE FUNCIONES BBDD ===")
    print(f"Schema: {schema}")
    print(f"Table : {table}\n")

    try:
        # 1) Tablas del schema
        print("1) get_all_tables_in_schema()")
        tables: List[str] = get_all_tables_in_schema(schema)
        if tables:
            # Muestra algunas para ver que el formato es correcto
            preview = tables[:20]
            print(f"   -> Preview (hasta 20): {preview}")
            # Confirmación rápida de si la tabla objetivo existe en el listado
            exists = any(row.get("table_name") == table for row in tables)
            print(f"   -> ¿Existe '{table}' en el schema? {exists}")
        print()

        # 2) Columnas de la tabla
        print("2) get_table_columns()")
        cols: List[Dict[str, Any]] = get_table_columns(schema, table)
        print(f"   -> Total columnas: {len(cols)}")
        if cols:
            print("   -> Columnas (detalle):")
            print(_pretty(cols))
        print()

        # 3) Nombres del catálogo
        print("3) get_names_in_table_catalog()")
        catalog_names: List[Dict[str, Any]] = get_names_in_table_catalog()
        print(f"   -> Total entradas: {len(catalog_names)}")
        if catalog_names:
            print("   -> Preview (hasta 10):")
            print(_pretty(catalog_names[:10]))
        print()

        print("=== FIN PRUEBAS ===")
        return 0

    except Exception as e:
        print("ERROR durante las pruebas:")
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
