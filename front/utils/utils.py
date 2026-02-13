# predicciones/utils.py
from datetime import date, datetime
import re
import hashlib
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from shiny import ui
from click import Tuple
import pandas as pd
from psycopg import sql

from back.database.core import fetch_data
from back.database.repository import get_bool_group_filters, get_metadata_for_variable
from front.utils.back_api_wrappers import (
    get_date_range_for_variable,
    get_filters_for_variable,
    get_distinct_values_for_column,
    get_table_columns,
    get_metadata_for_variable as get_metadata_for_variable_api,
)

DateLike = Union[date, datetime, str]

def slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"

def stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{h}"

def group_by_category(catalog_entries, exclude_name: str | None = None) -> OrderedDict:
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for entry in (catalog_entries or []):
        if not entry:
            continue
        name = entry.get("nombre")
        if not name or (exclude_name and name == exclude_name):
            continue
        cat = entry.get("categoria") or "Sin categoría"
        grouped.setdefault(cat, []).append(name)

    grouped_sorted = OrderedDict()
    for cat in sorted(grouped.keys()):
        grouped_sorted[cat] = sorted(grouped[cat])
    return grouped_sorted

def fmt(v) -> str:
    if v is None:
        return "—"
    s = str(v).strip()
    return s if s else "—"


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


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.fromisoformat(d[:10]).date()
    raise TypeError(f"Tipo de fecha no soportado: {type(d)}")


def build_name_to_table(catalog_entries) -> dict[str, str]:
    name_to_table: dict[str, str] = {}
    for entry in (catalog_entries or []):
        if not entry:
            continue
        name = entry.get("nombre")
        if not name:
            continue
        name_to_table[name] = entry.get("nombre_tabla") or name
    return name_to_table


class PrediccionesCache:
    def __init__(self, name_to_table: dict[str, str] | None = None) -> None:
        self.name_to_table = name_to_table or {}
        self._date_range_cache: dict[str, tuple[object | None, object | None]] = {}
        self._metadata_cache: dict[str, dict] = {}
        self._filters_cache: dict[str, list[dict]] = {}
        self._distinct_cache: dict[tuple[str, str, str], list[str]] = {}
        self._table_cols_cache: dict[str, set[str]] = {}

    def build_name_to_table(self, catalog_entries) -> dict[str, str]:
        self.name_to_table = build_name_to_table(catalog_entries)
        return self.name_to_table

    def get_date_range(self, nombre_var: str):
        table = self.name_to_table.get(nombre_var, nombre_var)
        if table in self._date_range_cache:
            return self._date_range_cache[table]

        try:
            rows = get_date_range_for_variable(table) or []
            row = rows[0] if rows else {}
            start = row.get("fecha_inicio")
            end = row.get("fecha_fin")
        except Exception:
            start, end = None, None

        self._date_range_cache[table] = (start, end)
        return start, end

    def get_meta(self, nombre: str) -> dict:
        if nombre in self._metadata_cache:
            return self._metadata_cache[nombre]
        rows = get_metadata_for_variable_api(nombre) or []
        meta = rows[0] if rows else {}
        self._metadata_cache[nombre] = meta
        return meta

    def get_filters(self, nombre_var_o_tabla: str) -> list[dict]:
        if nombre_var_o_tabla in self._filters_cache:
            return self._filters_cache[nombre_var_o_tabla]

        rows = get_filters_for_variable(nombre_var_o_tabla) or []
        out: list[dict] = []

        default_table = self.name_to_table.get(nombre_var_o_tabla, nombre_var_o_tabla)

        for r in rows:
            table = r.get("nombre_tabla") or default_table
            col = r.get("filtro")
            if not col:
                continue

            label = r.get("nombre_filtro") or col
            out.append({"table": table, "col": col, "label": label})

        seen = set()
        uniq = []
        for item in out:
            k = (item["table"], item["col"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(item)

        self._filters_cache[nombre_var_o_tabla] = uniq
        return uniq

    def get_distinct(self, schema: str, table: str, col: str) -> list[str]:
        key = (schema, table, col)
        if key in self._distinct_cache:
            return self._distinct_cache[key]
        vals = get_distinct_values_for_column(schema, table, col) or []
        self._distinct_cache[key] = vals
        return vals

    def get_table_cols(self, schema: str, table: str) -> set[str]:
        key = f"{schema}.{table}"
        if key in self._table_cols_cache:
            return self._table_cols_cache[key]
        cols = get_table_columns(schema, table) or []
        s = {c.get("column_name") for c in cols if c.get("column_name")}
        self._table_cols_cache[key] = s
        return s


def compatibilidad_con_objetivo(
    predictor_name: str,
    predictor_meta: dict,
    target_name: str,
    target_meta: dict,
    target_start,
    target_end,
    cache: PrediccionesCache,
) -> tuple[bool, str]:
    """
    Compatibilidad predictor vs objetivo usando:
    - misma temporalidad
    - y que el predictor CUBRA el rango del objetivo
      (esto se logra pasando predictor como "1" y objetivo como "2"
       porque check_date_and_temporality comprueba que 2 está contenido en 1).
    """
    if not target_name:
        return False, "Sin objetivo seleccionado"

    pred_temp = predictor_meta.get("temporalidad")
    tgt_temp = target_meta.get("temporalidad")

    pred_start, pred_end = cache.get_date_range(predictor_name)

    if pred_temp is None or tgt_temp is None:
        return False, "Temporalidad no definida"

    if pred_start is None or pred_end is None or target_start is None or target_end is None:
        return False, "Sin rango de fechas"

    ok = check_date_and_temporality(
        pred_start, target_start,
        pred_end, target_end,
        pred_temp, tgt_temp
    )

    if ok:
        return True, ""
    if pred_temp.strip().lower() != tgt_temp.strip().lower():
        return False, "Temporalidad distinta"
    return False, "El predictor no cubre el rango del objetivo"


def panel_styles() -> ui.tags.style:
    return ui.tags.style(
        """
        .var-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 6px 0;
        }

        /* Botón base */
        .var-pick {
            text-align: left;
            width: 100%;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 8px 10px;
            background: #ffffff;
            cursor: pointer;
        }

        .var-pick:hover { background: #f6f8fa; }

        .var-pick.is-selected {
            font-weight: 700;
            background: #d1e7dd;
            border-color: #198754;
            color: #0f5132;
        }

        .selection-pill {
            padding: 6px 10px;
            border-radius: 999px;
            display: inline-block;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            margin-bottom: 8px;
        }

        /* Panel 2: tarjetas por variable */
        .var-item {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px 12px;
            background: #fff;
            margin-bottom: 8px;
        }

        .var-item .form-check { margin: 0; }
        .var-item .form-check-label { font-weight: 600; }

        .var-meta {
            margin-top: 6px;
            padding-left: 24px; /* alinear con el checkbox */
            font-size: 0.92rem;
            color: #24292f;
        }

        .var-meta-grid {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 4px 10px;
            margin-top: 4px;
        }

        .var-meta-key { color: #57606a; }
        .var-desc {
            margin-top: 6px;
            color: #24292f;
        }
                .compat-badge {
            padding: 2px 10px;
            border-radius: 999px;
            display: inline-block;
            font-size: 0.85rem;
            border: 1px solid;
            font-weight: 600;
        }
        .compat-yes {
            background: #d1e7dd;
            border-color: #198754;
            color: #0f5132;
        }
        .compat-no {
            background: #f8d7da;
            border-color: #dc3545;
            color: #842029;
        }
        .compat-reason {
            margin-top: 4px;
            font-size: 0.85rem;
            color: #57606a;
        }
        .compat-reason-box {
            margin-top: 4px;
            padding: 6px 8px;
            background: #fef3f2;
            border-left: 3px solid #f04438;
            font-size: 0.9em;
        }
        .compat-reason-box .reason-text {
            color: #b42318;
        }

        """
    )


def get_col_ref_and_table(nombre: str, cache: PrediccionesCache | None = None) -> Tuple[str, str, str]:
    if cache:
        rows = [cache.get_meta(nombre)]
    else:
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

#Server
def create_dataframe_based_on_selection(
    target_var: str,
    predictors: List[str],
    filters_by_var: dict[str, list[dict]] | None = None,
    cache: PrediccionesCache | None = None,
) -> pd.DataFrame:
    # --- helpers (mínimos) ---
    _has_col_cache: dict[tuple[str, str], bool] = {}

    def _table_has_col(table: str, col: str) -> bool:
        key = (table, col)
        if key in _has_col_cache:
            return _has_col_cache[key]

        q = sql.SQL("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
        """)
        rows = fetch_data(q, ["IA", table, col])
        exists = bool(rows)
        _has_col_cache[key] = exists
        return exists

    def _detect_time_cols(table: str) -> list[str]:
        # Detecta qué columnas temporales existen (en orden)
        cols = []
        for c in ("anio", "mes", "dia"):
            if _table_has_col(table, c):
                cols.append(c)
        if not cols:
            raise ValueError(f'La tabla "IA".{table} no tiene columnas temporales (anio/mes/dia).')
        return cols

    # --- Target ---
    target_col, target_table, target_name = get_col_ref_and_table(target_var, cache=cache)
    target_alias = _safe_alias(target_name or target_col)

    # detectamos granularidad temporal del target
    time_cols = _detect_time_cols(target_table)  # p.ej. ["anio","mes"] o ["anio","mes","dia"] o ["anio"]

    where_clauses_target, target_params, group_cols_target = create_where_clauses(
        filters_by_var, target_name, target_table=target_table
    )
    where_sql_target = sql.SQL(" AND ").join(where_clauses_target) if where_clauses_target else sql.SQL("TRUE")

    group_select = sql.SQL("")
    if group_cols_target:
        group_select = sql.SQL(", ").join(sql.Identifier(c) for c in group_cols_target) + sql.SQL(", ")

    time_select = sql.SQL(", ").join(sql.Identifier(c) for c in time_cols)

    group_by = sql.SQL(", ").join(
        [*(sql.Identifier(c) for c in group_cols_target), *(sql.Identifier(c) for c in time_cols)]
    )

    q_target = sql.SQL("""
        SELECT {group_select} SUM({col}) AS {alias}, {time_select}
        FROM "IA".{table}
        WHERE {where}
        GROUP BY {group_by}
    """).format(
        group_select=group_select,
        col=sql.Identifier(target_col),
        alias=sql.Identifier(target_alias),
        time_select=time_select,
        table=sql.Identifier(target_table),
        where=where_sql_target,
        group_by=group_by
    )

    target_cols = [*group_cols_target, target_alias, *time_cols]
    target_rows = fetch_data(q_target, target_params)
    df = _rows_to_df(target_rows, target_cols)
    
    # claves base del DF target
    base_keys = [*group_cols_target, *time_cols]

    # --- Predictors ---
    for i, p in enumerate(predictors, start=1):
        p_col, p_table, p_name = get_col_ref_and_table(p, cache=cache)
        p_alias = _safe_alias(p_name or f"pred_{i}_{p_col}")

        # comprobamos misma granularidad temporal que el target
        pred_time_cols = _detect_time_cols(p_table)
        if pred_time_cols != time_cols:
            raise ValueError(
                f"Granularidad temporal distinta.\n"
                f'- Target "IA".{target_table}: {time_cols}\n'
                f'- Pred   "IA".{p_table}: {pred_time_cols}'
            )

        where_clauses_p, p_params, _group_cols_p = create_where_clauses(
            filters_by_var, p_name, target_table=p_table
        )
        where_sql_p = sql.SQL(" AND ").join(where_clauses_p) if where_clauses_p else sql.SQL("TRUE")

        # Si predictor está en la MISMA tabla que el target, puedo desglosar por los mismos grupos
        group_cols_pred = group_cols_target if (p_table == target_table) else []
        pred_group_select = sql.SQL("")
        if group_cols_pred:
            pred_group_select = sql.SQL(", ").join(sql.Identifier(c) for c in group_cols_pred) + sql.SQL(", ")

        pred_time_select = sql.SQL(", ").join(sql.Identifier(c) for c in time_cols)

        pred_group_by = sql.SQL(", ").join(
            [*(sql.Identifier(c) for c in group_cols_pred), *(sql.Identifier(c) for c in time_cols)]
        )

        # Agregamos predictor por la misma granularidad temporal (y por grupo si aplica)
        q_pred = sql.SQL("""
            SELECT {group_select} SUM({col}) AS {alias}, {time_select}
            FROM "IA".{table}
            WHERE {where}
            GROUP BY {group_by}
        """).format(
            group_select=pred_group_select,
            col=sql.Identifier(p_col),
            alias=sql.Identifier(p_alias),
            time_select=pred_time_select,
            table=sql.Identifier(p_table),
            where=where_sql_p,
            group_by=pred_group_by
        )

        pred_cols = [*group_cols_pred, p_alias, *time_cols]
        pred_rows = fetch_data(q_pred, p_params)
        df_pred = _rows_to_df(pred_rows, pred_cols)

        # Merge: si df_pred no tiene columnas de grupo (porque tabla distinta), se hace por tiempo
        join_keys = [*time_cols]
        if group_cols_target and group_cols_pred:
            join_keys = base_keys

        df = df.merge(df_pred, on=join_keys, how="left")

    if not df.empty:
        df = df.sort_values(base_keys).reset_index(drop=True)

    return df


def _find_col(df: pd.DataFrame, *candidates: str):
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def add_fourier_annual_terms(
    df: pd.DataFrame,
    dia_col: str = "dia",      # día del mes (1..31)
    K: int = 6,
    m: int = 365,
    anio_col: str = "anio",
    mes_col: str = "mes",
    fecha_col: str = "fecha",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Añade términos de Fourier (sen/cos) para anualidad, asumiendo que `dia_col`
    es el día del mes (1..31) y que existen `anio_col` y `mes_col`.

    Devuelve:
      - df_modificado (ordenado por fecha)
      - lista con los nombres de las columnas Fourier creadas
    """
    df = df.copy()

    # 1) Normaliza día del mes
    df[dia_col] = pd.to_numeric(df[dia_col], errors="raise").astype(int)
    if not df[dia_col].between(1, 31).all():
        bad = df.loc[~df[dia_col].between(1, 31), dia_col].head(5).tolist()
        raise ValueError(f"Valores inválidos en '{dia_col}' (deben ser 1..31). Ejemplos: {bad}")

    # 2) Construye fecha diaria real
    if anio_col not in df.columns or mes_col not in df.columns:
        raise ValueError(f"Faltan columnas '{anio_col}' y/o '{mes_col}' para construir '{fecha_col}'.")

    df[fecha_col] = pd.to_datetime(
        dict(
            year=df[anio_col].astype(int),
            month=df[mes_col].astype(int),
            day=df[dia_col],
        ),
        errors="raise",
    )

    # 3) Orden temporal
    df = df.sort_values(fecha_col)

    # 4) Índice t (días desde el inicio)
    t = (df[fecha_col] - df[fecha_col].min()).dt.days.to_numpy()

    # 5) Genera Fourier anual
    cols: list[str] = []
    for k in range(1, K + 1):
        ccol = f"fourier_cos{k}_{m}"
        scol = f"fourier_sin{k}_{m}"
        df[ccol] = np.cos(2 * np.pi * k * t / m)
        df[scol] = np.sin(2 * np.pi * k * t / m)
        cols.extend([ccol, scol])

    return df, cols



ICON_SVG_INFO = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16">
  <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
  <path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/>
</svg>"""
