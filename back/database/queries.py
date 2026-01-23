GET_TABLES_IN_SCHEMA = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = %s
      AND table_type = 'BASE TABLE'
    ORDER BY table_name;
"""

GET_COLUMNS_IN_TABLE = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
"""

GET_CATALOG_NAMES = """
    SELECT nombre_tabla, nombre
    FROM "IA".tbl_catalogo_variables
    ORDER BY nombre_tabla, nombre;
"""
GET_DATA_IN_TABLE = """
    SELECT *
    FROM "{schema}"."{table}"; 
"""