# predicciones/utils.py
from datetime import date, datetime
import re
import hashlib
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
from shiny import ui
import pandas as pd

from back.utils.column_utils import _safe_alias
from back.utils.time_features import add_fourier_annual_terms
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
    for entry in catalog_entries or []:
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


def fmt_num(value, decimals: int = 2, suffix: str = "") -> str:
    """Formatea un número al estilo español: punto como separador de miles, coma decimal."""
    try:
        num = float(value)
    except (ValueError, TypeError):
        return str(value)
    formatted = f"{num:,.{decimals}f}"  # e.g. "1,234.56"
    # Swap: comma→placeholder, dot→comma, placeholder→dot
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted}{suffix}"


def fmt_date_by_temporality(date_value, temporality: str = None) -> str:
    """
    Formatea una fecha según la temporalidad.
    - Mensual: MM/YYYY
    - Diaria: DD/MM/YYYY
    - Anual: YYYY
    - Otro/None: formato completo
    """
    if not date_value:
        return "—"

    if isinstance(date_value, str):
        try:
            date_value = datetime.fromisoformat(date_value[:10]).date()
        except (ValueError, TypeError):
            return str(date_value)

    if not temporality:
        return date_value.strftime("%d/%m/%Y")

    temp_lower = temporality.strip().lower()

    if temp_lower in ("mensual", "mes", "monthly", "month"):
        return date_value.strftime("%m/%Y")
    elif temp_lower in ("diaria", "dia", "día", "daily", "day"):
        return date_value.strftime("%d/%m/%Y")
    elif temp_lower in ("anual", "año", "ano", "year", "yearly", "annual"):
        return date_value.strftime("%Y")
    else:
        return date_value.strftime("%d/%m/%Y")


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


def diff_en_temporalidad(
    from_date: DateLike,
    to_date: DateLike,
    temporality: str,
) -> int | None:
    """
    Devuelve cuántos pasos (según temporality) hay entre from_date y to_date:
      - mensual -> meses
      - trimestral -> trimestres
      - anual -> años
      - semanal -> semanas
      - diaria -> días

    Si to_date < from_date, devuelve un número negativo.
    Si no reconoce la temporalidad, devuelve None.
    """
    if temporality is None:
        return None

    t = temporality.strip().lower()
    d1 = _to_date(from_date)
    d2 = _to_date(to_date)

    # Mensual
    if "mens" in t or "mes" in t or "month" in t:
        return (d2.year - d1.year) * 12 + (d2.month - d1.month)

    # Trimestral / Quarterly
    if "trim" in t or "trimes" in t or "quart" in t:
        q1 = (d1.month - 1) // 3
        q2 = (d2.month - 1) // 3
        return (d2.year - d1.year) * 4 + (q2 - q1)

    # Anual
    if "anual" in t or "año" in t or "ano" in t or "year" in t:
        return d2.year - d1.year

    # Semanal
    if "seman" in t or "week" in t:
        return (d2 - d1).days // 7

    # Diaria
    if "diar" in t or "día" in t or "dia" in t or "daily" in t or "day" in t:
        return (d2 - d1).days

    return None


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
    for entry in catalog_entries or []:
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
        self._distinct_complete_cache: dict[tuple[str, str, str], list[str]] = {}
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

    def get_distinct_complete(self, schema: str, table: str, col: str) -> list[str]:
        """Como get_distinct pero solo devuelve valores con cobertura temporal completa (sin huecos). Hace fallback al DISTINCT simple si es necesario."""
        key = (schema, table, col)
        if key in self._distinct_complete_cache:
            return self._distinct_complete_cache[key]
        vals = get_distinct_values_for_column(schema, table, col, complete=True) or []
        self._distinct_complete_cache[key] = vals
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

    if (
        pred_start is None
        or pred_end is None
        or target_start is None
        or target_end is None
    ):
        return False, "Sin rango de fechas"

    ok = check_date_and_temporality(
        pred_start, target_start, pred_end, target_end, pred_temp, tgt_temp
    )

    if ok:
        return True, ""
    if pred_temp.strip().lower() != tgt_temp.strip().lower():
        return False, "Temporalidad distinta"
    return False, "El predictor no cubre el rango del objetivo"


def detect_temporal_filters(filtros: list[dict]) -> dict:
    anio_filter = next(
        (f for f in filtros if f["col"].lower().strip() in ("anio", "año", "ano")), None
    )
    mes_filter = next((f for f in filtros if f["col"].lower().strip() == "mes"), None)
    dia_filter = next(
        (f for f in filtros if f["col"].lower().strip() in ("dia", "día")), None
    )

    table = None
    if anio_filter or mes_filter or dia_filter:
        table = (anio_filter or mes_filter or dia_filter)["table"]

    return {
        "anio": anio_filter,
        "mes": mes_filter,
        "dia": dia_filter,
        "table": table,
        "has_any": bool(anio_filter or mes_filter or dia_filter),
    }


def create_calendar_filter(
    filtros: list[dict],
    cache,
    stable_id_func,
    start_date=None,
    end_date=None,
    current_input=None,
):
    temp = detect_temporal_filters(filtros)

    if not temp["has_any"]:
        return None

    anio_filter = temp["anio"]
    mes_filter = temp["mes"]
    dia_filter = temp["dia"]
    table = temp["table"]

    if start_date and isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date[:10]).date()
    if end_date and isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date[:10]).date()

    if anio_filter and not mes_filter and not dia_filter:
        input_id = stable_id_func("flt", f"{table}__anio")
        anios = cache.get_distinct("IA", table, anio_filter["col"])
        anios_sorted = sorted([str(a) for a in anios if a], reverse=True)

        return ui.tags.div(
            ui.tags.label("Filtros de Temporalidad", class_="calendar-filter-title"),
            ui.input_selectize(
                input_id,
                "Año",
                choices=anios_sorted,
                multiple=True,
                options={
                    "placeholder": "Selecciona año(s)",
                    "plugins": ["remove_button"],
                },
            ),
            class_="calendar-filter-container",
        )

    #
    if anio_filter and mes_filter and not dia_filter:
        from datetime import date

        date_input_id = stable_id_func("flt", f"{table}__date_range")
        container_id = f"container_{date_input_id}"

        min_date = start_date if start_date else date(2000, 1, 1)
        max_date = end_date if end_date else date.today()

        if current_input and date_input_id in current_input:
            current_val = current_input[date_input_id]()
            if current_val and len(current_val) == 2:
                start_val = current_val[0]
                end_val = current_val[1]
            else:
                start_val = min_date
                end_val = max_date
        else:
            start_val = min_date
            end_val = max_date

        return ui.tags.div(
            ui.tags.label(
                "Filtros de Temporalidad (Mes/Año)", class_="calendar-filter-title"
            ),
            ui.input_date_range(
                date_input_id,
                "Seleccionar Período",
                start=start_val,
                end=end_val,
                min=min_date,
                max=max_date,
                format="mm/yyyy",
                language="es",
                separator="a",
                startview="year",
            ),
            ui.tags.script(f"""
                (function() {{
                    var attempts = 0;
                    var maxAttempts = 50;
                    
                    var interval = setInterval(function() {{
                        attempts++;
                        
                        var $inputs = $('#{date_input_id} input');
                        if ($inputs.length > 0) {{
                            var allConfigured = true;
                            
                            $inputs.each(function() {{
                                var $input = $(this);
                                
                                // Esperar a que exista el datepicker
                                if ($input.data('datepicker')) {{
                                    // Destruir la inicialización de Shiny
                                    $input.datepicker('destroy');
                                    
                                    // Reinicializar con configuración simple
                                    $input.datepicker({{
                                        format: "mm-yyyy",
                                        startView: "months", 
                                        minViewMode: "months",
                                        language: "es",
                                        autoclose: true,
                                        startDate: new Date({min_date.year}, {min_date.month - 1}, 1),
                                        endDate: new Date({max_date.year}, {max_date.month - 1}, 1)
                                    }});
                                    
                                    console.log('Datepicker configurado en modo SOLO MESES');
                                }} else {{
                                    allConfigured = false;
                                }}
                            }});
                            
                            if (allConfigured) {{
                                clearInterval(interval);
                            }}
                        }}
                        
                        if (attempts >= maxAttempts) {{
                            clearInterval(interval);
                        }}
                    }}, 200);
                }})();
            """),
            ui.tags.small(
                f"Haz clic en el mes deseado. El día es orientativo, se usará el mes para el entrenamiento. Rango disponible: {min_date.strftime('%m/%Y')} a {max_date.strftime('%m/%Y')}.",
                style="display: block; color: #57606a; font-size: 0.85em; margin-top: 4px; font-style: italic;",
            ),
            id=container_id,
            class_="calendar-filter-container",
        )

    if dia_filter:
        from datetime import date

        date_input_id = stable_id_func("flt", f"{table}__date_range")

        min_date = start_date if start_date else date(2000, 1, 1)
        max_date = end_date if end_date else date.today()

        if current_input and date_input_id in current_input:
            current_val = current_input[date_input_id]()
            if current_val and len(current_val) == 2:
                start_val = current_val[0]
                end_val = current_val[1]
            else:
                start_val = min_date
                end_val = max_date
        else:
            start_val = min_date
            end_val = max_date

        return ui.tags.div(
            ui.tags.label("Filtros de Temporalidad", class_="calendar-filter-title"),
            ui.input_date_range(
                date_input_id,
                "Seleccionar Período",
                start=start_val,
                end=end_val,
                min=min_date,
                max=max_date,
                format="dd/mm/yyyy",
                language="es",
                separator="a",
            ),
            class_="calendar-filter-container",
        )

    return None


def process_date_range_filters(date_range, filtros, table):
    if not date_range or len(date_range) != 2:
        return []

    start_date, end_date = date_range
    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return []

    temp = detect_temporal_filters(filtros)

    if not temp["anio"]:
        return []

    if temp["mes"] and not temp["dia"]:
        start_dt = start_dt.to_period("M").to_timestamp(how="start")
        end_dt = end_dt.to_period("M").to_timestamp(how="end")
    elif not temp["mes"] and not temp["dia"]:
        start_dt = pd.Timestamp(year=start_dt.year, month=1, day=1)
        end_dt = pd.Timestamp(year=end_dt.year, month=12, day=31)

    return [
        {
            "table": table,
            "kind": "date_range",
            "col": "__date_range__",
            "values": [],
            "start": start_dt.date().isoformat(),
            "end": end_dt.date().isoformat(),
            "year_col": temp["anio"]["col"],
            "month_col": temp["mes"]["col"] if temp["mes"] else None,
            "day_col": temp["dia"]["col"] if temp["dia"] else None,
        }
    ]


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
            padding: 10px 16px;
            border-radius: 12px;
            display: block;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            margin-bottom: 8px;
            width: fit-content;
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

        /* Filtros de calendario PANEL3 */
        .calendar-filter-container {
            padding: 12px;
            background: #f6f8fa;
            border-radius: 6px;
            margin-bottom: 12px;
        }

        .calendar-filter-title {
            font-weight: 600;
            margin-bottom: 8px;
            display: block;
            color: #0969da;
        }

        .calendar-controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }

        /* Separadores Panel 3 */
        .panel3-section {
            margin-bottom: 16px;
        }

        .panel3-section-header {
            display: flex;
            align-items: center;
            font-size: 1rem;
            font-weight: 700;
            padding: 8px 14px;
            border-radius: 6px;
            margin-bottom: 10px;
            letter-spacing: 0.02em;
        }

        .panel3-section-target {
            background: #dff0ff;
            border-left: 4px solid #0969da;
            color: #0550ae;
        }

        .panel3-section-exog {
            background: #f0f0f0;
            border-left: 4px solid #6e7781;
            color: #424a53;
        }

        .panel3-divider {
            border: none;
            border-top: 2px dashed #d0d7de;
            margin: 18px 0;
        }

       

        """
    )


ICON_SVG_INFO = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16">
  <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
  <path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/>
</svg>"""


def humanize_error(error_msg: str) -> str:
    """
    Traduce errores técnicos del backend a mensajes amigables para el usuario.
    """
    if not error_msg:
        return "Error desconocido."

    e = str(error_msg)

    if "Invalid integer data type 'O'" in e or "dtype 'O'" in e:
        return "Uno de los valores numéricos introducidos es demasiado grande o contiene caracteres no válidos."
    return e
