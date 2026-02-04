from psycopg import sql

GET_DATE_RANGE_FOR_VARIABLE = sql.SQL("""
    WITH src AS (
        SELECT
            anio::int AS anio,
            lower(trim(mes::text)) AS mes_txt
        FROM {schema}.{nombre_tabla}
        WHERE anio IS NOT NULL
        AND mes  IS NOT NULL
    ),
    t AS (
        SELECT
            make_date(
                anio,
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
                END,
                1
            ) AS fecha_mes
        FROM src
    )
    SELECT
        MIN(fecha_mes) AS fecha_inicio,
        (MAX(fecha_mes) + INTERVAL '1 month' - INTERVAL '1 day')::date AS fecha_fin
    FROM t
    WHERE fecha_mes IS NOT NULL;
""")



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
    SELECT nombre_tabla, nombre, categoria
    FROM "IA".tbl_catalogo_variables
    ORDER BY categoria, nombre;
"""

GET_ALL_DATA_IN_TABLE = """
    SELECT *
    FROM "{schema}"."{table}"; 
"""

GET_CATEGORIES = """
    SELECT DISTINCT categoria
    FROM "IA".tbl_catalogo_variables
    WHERE categoria IS NOT NULL
    ORDER BY categoria;
"""

GET_METADATA_FOR_VARIABLE = """
    SELECT temporalidad, descripcion, fuente, granularidad, unidad_medida, nombre_colum_ref, nombre_tabla, nombre
    FROM "IA".tbl_catalogo_variables
    WHERE nombre = %s;
"""

GET_FILTERS_FOR_VARIABLE = """
    SELECT
        nombre_tabla,
        variable,
        filtro,
        id_catalogo,
        id_filtro,
        nombre_filtro
    FROM "IA".tbl_admin_filtro
    WHERE variable = %s OR nombre_tabla = %s
    ORDER BY id_catalogo NULLS LAST, id_filtro NULLS LAST, filtro;
"""


GET_TABLE_NAME_FOR_VARIABLE = """
    SELECT nombre, nombre_tabla
    FROM "IA".tbl_catalogo_variables
    WHERE nombre = %s;
"""

GET_BOOL_GROUP_FILTERS = """
    SELECT union_grupo
    FROM "IA".tbl_admin_filtro
    WHERE filtro = %s
    """