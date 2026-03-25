import asyncio
import pandas as pd
import plotly.graph_objects as go
from shiny import ui, reactive, render, module
from front.utils.back_api_wrappers import sarimax_run
from front.utils.back_api_wrappers import xgboost_run
from front.utils.back_api_wrappers import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
)
from back.models.utils.models_graph import (
    build_interactive_plot_html,
    compute_time_axis_bounds,
    ensure_datetime_index,
)

from front.utils.utils import (
    ICON_SVG_INFO,
    _to_date,
    detect_temporal_filters,
    diff_en_temporalidad,
    slug as _slug,
    stable_id as _stable_id,
    group_by_category as _group_by_category,
    fmt as _fmt,
    fmt_num,
    fmt_date_by_temporality as _fmt_date_temp,
    metadata_decimals,
    normalize_temporality,
    build_name_to_table,
    PrediccionesCache,
    shift_date_by_temporality,
    humanize_error,
    panel_styles,
    create_calendar_filter,
    process_date_range_filters,
    detect_temporal_filters,
)


# -----------------------------
# Module
# -----------------------------
@module.server
def predicciones_server(input, output, session):
    current_step = reactive.Value(1)

    target_var_rv = reactive.Value(None)
    predictors_rv = reactive.Value([])
    pred_results_rv = reactive.Value(None)
    pred_results_err_rv = reactive.Value(None)
    _SPINNER_ID = "_pred_spinner"
    saved_filter_values_rv = reactive.Value({})

    catalog_entries = get_names_in_table_catalog() or []
    name_to_table = build_name_to_table(catalog_entries)
    cache = PrediccionesCache(name_to_table)
    PANEL_STYLES = panel_styles()

    ##########################################################################################
    # Panel 1: SELECCION DE VARIABLE OBJETIVO
    ##########################################################################################
    _registered_pick_handlers: set[str] = set()

    @output
    @render.ui
    def step_indicator():
        step = current_step.get()
        labels = ["Objetivo", "Predictoras", "Filtros", "Predicciones"]
        nodes = []

        for i, lbl in enumerate(labels, start=1):
            classes = "step-item"
            if i < step:
                classes += " completed"
            elif i == step:
                classes += " active"

            nodes.append(
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.span(str(i), class_="step-number"),
                        class_="step-circle",
                    ),
                    ui.tags.span(lbl, class_="step-label"),
                    class_=classes,
                )
            )

            if i < len(labels):
                nodes.append(ui.tags.div(class_="step-connector"))

        return ui.div(
            PANEL_STYLES,
            ui.tags.div(*nodes, class_="step-indicator"),
            style="margin:8px 0;",
        )

    @reactive.Effect
    def _init_target_var():
        if target_var_rv.get() is None:
            grouped = _group_by_category(catalog_entries)
            all_names = [n for names in grouped.values() for n in names]
            if all_names:
                target_var_rv.set(all_names[0])

    @output
    @render.ui
    def step_panel_1():
        if current_step.get() != 1:
            return ui.div()

        grouped = _group_by_category(catalog_entries)
        all_names = [n for names in grouped.values() for n in names]

        selected = target_var_rv.get()

        panels = []
        for cat, names in grouped.items():
            blocks = []
            for name in names:
                btn_id = _stable_id("pick_target", name)

                if btn_id not in _registered_pick_handlers:
                    _registered_pick_handlers.add(btn_id)

                    @reactive.Effect
                    @reactive.event(input[btn_id])
                    def _on_pick_target(_name=name):
                        target_var_rv.set(_name)

                meta = cache.get_meta(name)
                temporalidad = _fmt(meta.get("temporalidad"))
                granularidad = _fmt(meta.get("granularidad"))
                unidad_medida = _fmt(meta.get("unidad_medida"))
                fuente = _fmt(meta.get("fuente"))
                descripcion = _fmt(meta.get("descripcion"))

                blocks.append(
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_action_button(
                                btn_id,
                                name,
                                class_=(
                                    "var-pick is-selected"
                                    if name == selected
                                    else "var-pick"
                                ),
                            ),
                            style="display: flex; align-items: baseline; gap: 6px;",
                        ),
                        ui.tags.details(
                            ui.tags.summary(
                                "Ver más",
                                style="cursor: pointer; margin-top: 6px; font-size: 0.9em; color: #666;",
                            ),
                            ui.tags.div(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.strong("Temporalidad: "),
                                        temporalidad,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Granularidad: "),
                                        granularidad,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Unidad medida: "),
                                        unidad_medida,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Fuente: "),
                                        fuente,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Descripción: "),
                                        descripcion,
                                        style="margin-bottom: 8px;",
                                    ),
                                ),
                                style="margin-top: 8px; padding: 8px 0;",
                            ),
                        ),
                        class_="var-item",
                    )
                )

            panels.append(
                ui.accordion_panel(
                    cat,
                    ui.div(*blocks),
                    value=_slug(cat),
                )
            )

        return ui.div(
            PANEL_STYLES,
            ui.div(
                ui.tags.div(
                    "\U0001f3af", style="font-size:2.5rem; margin-bottom:0.5rem;"
                ),
                ui.h3(
                    "Seleccionar variable objetivo",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Elige la variable que deseas predecir. Esta será la variable dependiente "
                    "del modelo, es decir, el valor que los algoritmos intentarán estimar a partir "
                    "de las variables predictoras que selecciones en el siguiente paso.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
            ui.div(
                ui.tags.span("\u2705 Seleccionada: ", style="font-weight:600;"),
                ui.tags.span(selected or "—"),
                class_="selection-pill",
            ),
            ui.accordion(*panels, id="acc_target", open=True, multiple=True),
            ui.div(
                ui.input_action_button("btn_next_1", "Siguiente →"),
                style="margin-top: 12px;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.btn_next_1)
    def _go_step_2():
        current_step.set(2)

    ##########################################################################################
    # Panel 2: SELECCION DE VARIABLES EXÓGENAS + METADATA
    ##########################################################################################
    @reactive.Calc
    def predictor_pairs():
        """Lista estable de (input_id, nombre) para todas las predictoras visibles."""
        target_var = target_var_rv.get()
        grouped = _group_by_category(catalog_entries, exclude_name=target_var)
        pairs: list[tuple[str, str]] = []
        for _, names in grouped.items():
            for name in names:
                var_id = _stable_id("pred", name)
                pairs.append((var_id, name))
        return pairs

    def _is_predictor_selectable(
        predictor_name: str,
        predictor_meta: dict,
        target_name: str,
        target_meta: dict,
        target_start,
        target_end,
    ) -> tuple[bool, str]:
        if not target_name:
            return False, "Sin objetivo seleccionado"

        pred_temp = normalize_temporality(predictor_meta.get("temporalidad"))
        tgt_temp = normalize_temporality(target_meta.get("temporalidad"))
        pred_start, pred_end = cache.get_date_range(predictor_name)

        if not pred_temp or not tgt_temp:
            return False, "Temporalidad no definida"
        if pred_temp != tgt_temp:
            return False, "Temporalidad distinta"
        if (
            pred_start is None
            or pred_end is None
            or target_start is None
            or target_end is None
        ):
            return False, "Sin rango de fechas"
        if _to_date(pred_end) < _to_date(target_start):
            return False, "El predictor termina antes de que empiece el objetivo"
        if _to_date(pred_start) > _to_date(target_end):
            return False, "El predictor empieza después de que termine el objetivo"
        return True, ""

    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()

        target_var = target_var_rv.get()
        grouped = _group_by_category(catalog_entries, exclude_name=target_var)
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_start_raw, target_end_raw = (
            effective_target_range() if target_var else (None, None)
        )
        target_temp = _fmt(target_meta.get("temporalidad"))
        target_start = _fmt_date_temp(target_start_raw, target_meta.get("temporalidad"))
        target_end = _fmt_date_temp(target_end_raw, target_meta.get("temporalidad"))
        panels = []
        for cat, names in grouped.items():
            var_blocks = []

            for name in names:
                var_id = _stable_id("pred", name)
                meta = cache.get_meta(name)
                compat, reason = _is_predictor_selectable(
                    predictor_name=name,
                    predictor_meta=meta,
                    target_name=target_var,
                    target_meta=target_meta,
                    target_start=target_start_raw,
                    target_end=target_end_raw,
                )

                badge = ui.tags.span(
                    "Compatible" if compat else "No compatible",
                    class_=(
                        "compat-badge compat-yes"
                        if compat
                        else "compat-badge compat-no"
                    ),
                )

                info_icon = None
                if not compat and reason:
                    info_icon = ui.tooltip(
                        ui.tags.span(
                            ui.HTML(ICON_SVG_INFO),
                        ),
                        reason,
                    )
                temporalidad = _fmt(meta.get("temporalidad"))
                granularidad = _fmt(meta.get("granularidad"))
                unidad_medida = _fmt(meta.get("unidad_medida"))
                fuente = _fmt(meta.get("fuente"))
                descripcion = _fmt(meta.get("descripcion"))
                current_checked = bool(input[var_id]()) if var_id in input else False

                selector = (
                    ui.input_checkbox(var_id, name, value=current_checked)
                    if compat
                    else ui.tags.span(name, style="font-weight:600; color:#6e7781;")
                )

                var_blocks.append(
                    ui.tags.div(
                        ui.tags.div(
                            selector,
                            badge,
                            info_icon,
                            style="display: flex; align-items: baseline; gap: 6px;",
                        ),
                        ui.tags.details(
                            ui.tags.summary(
                                "Ver más",
                                style="cursor: pointer; margin-top: 6px; font-size: 0.9em; color: #666;",
                            ),
                            ui.tags.div(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.strong("Temporalidad: "),
                                        temporalidad,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Granularidad: "),
                                        granularidad,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Unidad medida: "),
                                        unidad_medida,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Fuente: "),
                                        fuente,
                                        style="margin-bottom: 8px;",
                                    ),
                                    ui.tags.div(
                                        ui.tags.strong("Descripción: "),
                                        descripcion,
                                        style="margin-bottom: 8px;",
                                    ),
                                ),
                                style="margin-top: 8px; padding: 8px 0;",
                            ),
                        ),
                        class_="var-item",
                    )
                )

            panels.append(
                ui.accordion_panel(
                    cat,
                    ui.div(*var_blocks),
                    value=_slug(cat),
                )
            )

        return ui.div(
            PANEL_STYLES,
            ui.div(
                ui.tags.div(
                    "\U0001f4ca", style="font-size:2.5rem; margin-bottom:0.5rem;"
                ),
                ui.h3(
                    "Seleccionar variables predictoras",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Elige las variables exógenas que alimentarán el modelo. Solo se pueden "
                    "seleccionar aquellas que sean compatibles en temporalidad y rango con la "
                    "variable objetivo elegida en el paso anterior.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
            ui.div(
                ui.tags.div(
                    ui.tags.span("Objetivo actual: ", style="font-weight:600;"),
                    ui.tags.span(target_var or "—"),
                ),
                ui.tags.div(
                    ui.tags.span("Temporalidad: ", style="font-weight:600;"),
                    ui.tags.span(target_temp),
                    ui.tags.span(
                        "  ·  Rango: ", style="font-weight:600; margin-left:10px;"
                    ),
                    ui.tags.span(f"{_fmt(target_start)} → {_fmt(target_end)}"),
                    style="margin-top:4px;",
                ),
                ui.output_ui("max_preds_line"),  # <-- NUEVO: se renderiza aparte
                class_="selection-pill",
            ),
            ui.accordion(*panels, id="acc_predictors", open=True, multiple=True),
            ui.div(
                ui.input_action_button("btn_prev_2", "← Anterior"),
                ui.input_action_button("btn_next_2", "Siguiente →"),
                style="margin-top: 12px; display: flex; gap: 8px;",
            ),
        )

    @reactive.Calc
    def selected_predictors():
        target_var = target_var_rv.get()
        if not target_var:
            return []

        target_meta = cache.get_meta(target_var) or {}
        target_start, target_end = effective_target_range()

        selected = []
        for var_id, name in predictor_pairs():
            if not (var_id in input and input[var_id]()):
                continue

            p_meta = cache.get_meta(name) or {}
            compat, _ = _is_predictor_selectable(
                predictor_name=name,
                predictor_meta=p_meta,
                target_name=target_var,
                target_meta=target_meta,
                target_start=target_start,
                target_end=target_end,
            )
            if compat:
                selected.append(name)
        return sorted(set(selected))

    def _checked_predictors(
        target_name: str | None = None,
        target_meta: dict | None = None,
        target_start=None,
        target_end=None,
    ) -> list[str]:
        selected = []
        for var_id, name in predictor_pairs():
            if not (var_id in input and input[var_id]()):
                continue

            if (
                target_name
                and target_meta
                and target_start is not None
                and target_end is not None
            ):
                p_meta = cache.get_meta(name) or {}
                selectable, _ = _is_predictor_selectable(
                    predictor_name=name,
                    predictor_meta=p_meta,
                    target_name=target_name,
                    target_meta=target_meta,
                    target_start=target_start,
                    target_end=target_end,
                )
                if not selectable:
                    continue

            selected.append(name)
        return sorted(set(selected))

    @output
    @render.ui
    def max_preds_line():
        # Si no estás en el panel 2, no muestres nada (evitas invalidaciones innecesarias)
        if current_step.get() != 2:
            return ui.div()

        m = max_num_predictions()
        txt = "—" if m is None else str(m)

        return ui.tags.div(
            ui.tags.span("Número máximo de predicciones: ", style="font-weight:600;"),
            ui.tags.span(txt),
            style="margin-top:4px;",
        )

    @reactive.Calc
    def max_num_predictions():
        target_var = target_var_rv.get()
        if not target_var:
            return None

        target_meta = cache.get_meta(target_var) or {}
        tgt_temp = target_meta.get("temporalidad")

        target_start, target_end = effective_target_range()
        if target_start is None or target_end is None:
            target_start, target_end = cache.get_date_range(target_var)
        if target_end is None or tgt_temp is None:
            return None

        preds = selected_predictors()
        if not preds:
            return None  # si no hay predictoras seleccionadas

        # Para varias predictoras: nos quedamos con el END más pequeño
        min_pred_end = None

        for p in preds:
            # Si alguna seleccionada no es compatible, no podemos garantizar el horizonte común
            _p_start, p_end = cache.get_date_range(p)
            if p_end is None:
                return 0

            if min_pred_end is None or _to_date(p_end) < _to_date(min_pred_end):
                min_pred_end = p_end

        if min_pred_end is None:
            return 0

        n = diff_en_temporalidad(target_end, min_pred_end, tgt_temp)
        if n is None:
            return None

        return max(0, n)

    @reactive.Effect
    def _sync_predictors_rv():
        predictors_rv.set(selected_predictors())

    @reactive.Effect
    @reactive.event(input.btn_prev_2)
    def _go_step_1():
        current_step.set(1)

    @reactive.Effect
    @reactive.event(input.btn_next_2)
    def _go_step_3():
        current_step.set(3)

    ##########################################################################################
    # Panel 3: Config Variables
    ##########################################################################################
    @reactive.Calc
    def target_selected_range() -> tuple:
        """Devuelve el rango completo por defecto de la variable objetivo (sin selección manual)."""
        target = target_var_rv.get()
        if not target:
            return (None, None)
        return cache.get_date_range(target)

    @reactive.Calc
    def effective_target_range() -> tuple:
        target_var = target_var_rv.get()
        if not target_var:
            return (None, None)

        target_start, target_end = target_selected_range()
        if target_start is None or target_end is None:
            target_start, target_end = cache.get_date_range(target_var)

        target_meta = cache.get_meta(target_var) or {}
        tgt_temp = target_meta.get("temporalidad")
        preds = _checked_predictors(
            target_name=target_var,
            target_meta=target_meta,
            target_start=target_start,
            target_end=target_end,
        )

        if not preds or target_start is None or target_end is None or tgt_temp is None:
            return (target_start, target_end)

        min_pred_end = None
        for predictor in preds:
            _p_start, p_end = cache.get_date_range(predictor)
            if p_end is None:
                return (target_start, target_end)
            if min_pred_end is None or _to_date(p_end) < _to_date(min_pred_end):
                min_pred_end = p_end

        if min_pred_end is None:
            return (target_start, target_end)

        n = diff_en_temporalidad(target_end, min_pred_end, tgt_temp)
        if n is None or n > 0:
            return (target_start, target_end)

        adjusted_end = shift_date_by_temporality(min_pred_end, tgt_temp, -2)
        if pd.isna(adjusted_end):
            return (target_start, target_end)
        if _to_date(adjusted_end) < _to_date(target_start):
            return (target_start, target_end)

        return (target_start, adjusted_end.date().isoformat())

    @reactive.Calc
    def vars_to_config() -> list[dict]:
        """
        Devuelve una lista de dicts con:
        - pretty: nombre bonito (lo que muestras)
        - table:  nombre real de la tabla (lo que consultas)
        - is_target: True si es la variable objetivo
        """
        target = target_var_rv.get()
        preds = predictors_rv.get() or []

        ordered_pretty: list[str] = []
        for v in [target, *preds]:
            if v and v not in ordered_pretty:
                ordered_pretty.append(v)

        out: list[dict] = []
        for pretty in ordered_pretty:
            table = name_to_table.get(pretty)

            if not table:
                rows = get_tableName_for_variable(pretty) or []
                table = rows[0].get("nombre_tabla") if rows else pretty

            out.append(
                {"pretty": pretty, "table": table, "is_target": (pretty == target)}
            )

        return out

    @reactive.Calc
    def selected_filters_by_var() -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        target_var = target_var_rv.get()

        extend_steps = 0
        if "pred_horizon" in input:
            try:
                extend_steps = max(0, int(input.pred_horizon() or 0))
            except Exception:
                extend_steps = 0

        target_temporal_filters = []

        for item in vars_to_config():
            pretty = item["pretty"]
            table = item["table"]
            is_target = item.get("is_target", False)

            filtros = cache.get_filters(table)
            selected_list: list[dict] = []

            temp = detect_temporal_filters(filtros)

            if is_target and (temp["mes"] or temp["dia"]):
                start_def, end_def = effective_target_range()
                if start_def and end_def:
                    date_range = (start_def, end_def)
                    temporal_filters = process_date_range_filters(
                        date_range, filtros, table
                    )
                    selected_list.extend(temporal_filters)
                    # Guardar para aplicar a las exógenas
                    target_temporal_filters = temporal_filters

            elif is_target and temp["anio"]:
                start_def, end_def = effective_target_range()
                if start_def and end_def:
                    start_year = pd.to_datetime(start_def, errors="coerce").year
                    end_year = pd.to_datetime(end_def, errors="coerce").year
                    if start_year and end_year:
                        years = list(range(start_year, end_year + 1))
                        temporal_filter = {
                            "table": table,
                            "col": temp["anio"]["col"],
                            "values": [str(y) for y in years],
                        }
                        selected_list.append(temporal_filter)
                        target_temporal_filters = [temporal_filter]

            if not is_target and target_temporal_filters:
                for tf in target_temporal_filters:
                    if (
                        tf.get("kind") == "date_range"
                        or tf.get("col") == "__date_range__"
                    ):
                        tf_copy = dict(tf)
                        tf_copy["table"] = table
                        if extend_steps > 0 and tf_copy.get("end"):
                            end_dt = pd.to_datetime(tf_copy.get("end"), errors="coerce")
                            if pd.notna(end_dt):
                                if tf_copy.get("day_col"):
                                    end_dt = end_dt + pd.Timedelta(days=extend_steps)
                                elif tf_copy.get("month_col"):
                                    end_dt = end_dt + pd.DateOffset(months=extend_steps)
                                else:
                                    end_dt = end_dt + pd.DateOffset(years=extend_steps)
                                tf_copy["end"] = end_dt.date().isoformat()
                        selected_list.append(tf_copy)
                    else:
                        selected_list.append(
                            {
                                "table": table,
                                "col": tf["col"],
                                "values": tf["values"],
                            }
                        )

            for f in filtros:
                col_lower = f["col"].lower().strip()
                if col_lower in ("anio", "año", "ano", "mes", "dia", "día"):
                    continue

                input_id = _stable_id("flt", f"{f['table']}__{f['col']}")
                if input_id in input:
                    vals = input[input_id]()
                    if vals:
                        selected_list.append(
                            {
                                "table": f["table"],
                                "col": f["col"],
                                "values": list(vals)
                                if isinstance(vals, (list, tuple))
                                else [str(vals)],
                            }
                        )

            out[pretty] = selected_list

        return out

    @output
    @render.ui
    def step_panel_3():
        if current_step.get() != 3:
            return ui.div()

        vars_sel = vars_to_config()
        if not vars_sel:
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 3: configurar filtros"),
                ui.p("No hay variables seleccionadas para configurar."),
            )

        target_var = target_var_rv.get()
        target_start, target_end = target_selected_range()
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_temporality = target_meta.get("temporalidad")

        # --- Build body for each variable ---
        def _var_body(item):
            pretty = item["pretty"]
            table = item["table"]
            is_target = item.get("is_target", False)
            filtros = cache.get_filters(table)

            if not filtros:
                if is_target:
                    return ui.p(
                        "Sin filtros configurados en tbl_admin_filtros para esta variable/tabla."
                    )
                else:
                    return ui.div()
            else:
                controls = []

                for f in filtros:
                    col_lower = f["col"].lower().strip()
                    if col_lower in ("anio", "año", "ano", "mes", "dia", "día"):
                        continue

                    t = f["table"]
                    col = f["col"]
                    label = f.get("label") or col

                    cols_set = cache.get_table_cols("IA", t)
                    if col not in cols_set:
                        controls.append(
                            ui.tags.div(
                                ui.tags.b(f"{col}"),
                                ui.tags.span(
                                    f"  (No existe en IA.{t})",
                                    style="color:#b42318; margin-left:6px;",
                                ),
                                style="margin-bottom:10px;",
                            )
                        )
                        continue

                    input_id = _stable_id("flt", f"{t}__{col}")
                    choices = cache.get_distinct_complete("IA", t, col)

                    _saved_val = saved_filter_values_rv.get().get(input_id, [])
                    controls.append(
                        ui.tags.div(
                            ui.input_selectize(
                                input_id,
                                label=label,
                                choices=choices,
                                selected=_saved_val if _saved_val else [],
                                multiple=True,
                                options={
                                    "placeholder": "Selecciona uno o varios valores (vacío = sin filtro)",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            style="margin-bottom: 12px;",
                        )
                    )

                return (
                    ui.div(*controls) if controls else ui.p("Sin filtros disponibles.")
                )

        # --- Separate target vs predictors ---
        target_item = None
        predictor_items = []
        for item in vars_sel:
            if item.get("is_target"):
                target_item = item
            else:
                predictor_items.append(item)

        # Target box
        target_box_content = (
            ui.div(
                ui.tags.div(
                    ui.tags.span(
                        target_item["pretty"], style="font-weight:700; font-size:1rem;"
                    ),
                    style="margin-bottom:8px;",
                ),
                _var_body(target_item),
            )
            if target_item
            else ui.p("No se ha seleccionado variable objetivo.")
        )

        target_box = ui.tags.div(
            ui.tags.div(
                "🎯 Variable objetivo",
                style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b;",
            ),
            target_box_content,
            style=(
                "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
            ),
        )

        # Predictors box
        if predictor_items:
            pred_panels = []
            for item in predictor_items:
                pred_panels.append(
                    ui.accordion_panel(
                        ui.tags.strong(item["pretty"]),
                        _var_body(item),
                        value=_slug(item["table"]),
                    )
                )

            predictors_box = ui.tags.div(
                ui.tags.div(
                    "📊 Variables predictoras",
                    ui.tooltip(
                        ui.tags.span(
                            ui.HTML(ICON_SVG_INFO),
                        ),
                        "Cada variable predictora se ajustará automáticamente al rango temporal de la variable objetivo.",
                    ),
                    style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b; display:flex; align-items:center; gap:4px;",
                ),
                ui.accordion(
                    *pred_panels, id="acc_filters_preds", open=True, multiple=True
                ),
                style=(
                    "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                    "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
                ),
            )
        else:
            predictors_box = ui.tags.div(
                ui.tags.div(
                    "📊 Variables predictoras",
                    style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b;",
                ),
                ui.p(
                    "No se han seleccionado variables predictoras.",
                    style="color:#6b7280;",
                ),
                style=(
                    "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                    "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
                ),
            )

        return ui.div(
            PANEL_STYLES,
            ui.div(
                ui.tags.div(
                    "\U0001f527", style="font-size:2.5rem; margin-bottom:0.5rem;"
                ),
                ui.h3(
                    "Configurar filtros",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Para cada variable se muestran los filtros correspondientes. "
                    "El rango temporal de las exógenas se ajusta automáticamente al de la "
                    "variable objetivo.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
            ui.tags.div(
                target_box,
                predictors_box,
                style="display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;",
            ),
            ui.div(
                ui.input_action_button("btn_prev_3", "← Anterior"),
                ui.input_action_button("btn_next_3", "Siguiente →"),
                style="margin-top: 12px; display: flex; gap: 8px;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.btn_prev_3)
    def _go_step_2():
        current_step.set(2)

    @reactive.Effect
    @reactive.event(input.btn_next_3)
    def _go_step_4():
        saved = {}
        for item in vars_to_config():
            t = item["table"]
            filtros = cache.get_filters(t)
            for f in filtros:
                col_lower = f["col"].lower().strip()
                if col_lower in ("anio", "año", "ano", "mes", "dia", "día"):
                    continue
                input_id = _stable_id("flt", f"{t}__{f['col']}")
                if input_id in input:
                    vals = input[input_id]()
                    if vals:
                        saved[input_id] = (
                            list(vals)
                            if isinstance(vals, (list, tuple))
                            else [str(vals)]
                        )
        saved_filter_values_rv.set(saved)
        current_step.set(4)

    ##########################################################################################
    # Panel 4: Play with Model and Variables (CON BOTÓN: NO calcula hasta pulsar)
    ##########################################################################################

    # ------------------------
    # Helpers (puros) Panel 4
    # ------------------------
    MODEL_RUNNERS = {
        "sarimax": sarimax_run,
        "xgboost": xgboost_run,
    }

    def _build_payload(
        model: str, target: str, predictors_used: list[str], filters: dict, horizon: int
    ) -> dict:
        base = {
            "target_var": target,
            "predictors": list(predictors_used or []),
            "filters_by_var": filters,
            "train_ratio": 0.70,
            "auto_params": True,
            "return_df": True,
            "horizon": int(horizon),
        }

        if model == "sarimax":
            base.update({"s": 12})
        elif model == "xgboost":
            base.update(
                {
                    "use_target_lags": True,
                    "max_lag": 12,
                    "recursive_forecast": True,
                }
            )
        return base

    def _pred_table_decimals() -> int:
        target = target_var_rv.get()
        meta = cache.get_meta(target) if target else {}
        return metadata_decimals(meta)

    def _parse_forecast_response(resp: dict):
        df = pd.DataFrame(resp.get("df") or [])
        if df.empty:
            return None

        y_col = resp["y_col"]
        n_obs = int(resp["n_obs"])
        h = int(resp["horizon"])

        future = df.iloc[n_obs : n_obs + h].copy()
        pred_vals = list(resp.get("y_forecast") or [])
        future_dates = None
        if {"anio", "mes", "dia"}.issubset(future.columns):
            future_dates = pd.to_datetime(
                dict(year=future["anio"], month=future["mes"], day=future["dia"]),
                errors="coerce",
            )
        elif {"anio", "mes"}.issubset(future.columns):
            future_dates = pd.to_datetime(
                dict(year=future["anio"], month=future["mes"], day=1), errors="coerce"
            )
        elif "__dt" in future.columns:
            future_dates = pd.to_datetime(future["__dt"], errors="coerce")

        if len(pred_vals) != len(future):
            m = min(len(pred_vals), len(future))
            pred_vals = pred_vals[:m]
            future = future.iloc[:m].copy()
            if future_dates is not None:
                future_dates = pd.to_datetime(future_dates, errors="coerce")[:m]

        pred_index = future_dates if future_dates is not None else future.index
        pred_series = pd.Series(pred_vals, index=pred_index, name="Prediction")
        return df, y_col, future, h, pred_vals, pred_series

    def _build_pred_df(
        future: pd.DataFrame, pred_vals, date_fmt: str = "%d-%m-%Y"
    ) -> pd.DataFrame:
        if {"anio", "mes"}.issubset(future.columns):
            if "dia" in future.columns:
                fechas = pd.to_datetime(
                    dict(year=future["anio"], month=future["mes"], day=future["dia"]),
                    errors="coerce",
                )
            else:
                fechas = pd.to_datetime(
                    dict(year=future["anio"], month=future["mes"], day=1),
                    errors="coerce",
                )
        else:
            fechas = future.index  # fallback

        pred_df = pd.DataFrame({"Fecha": fechas, "Predicción": pred_vals})
        pred_df["Fecha"] = pd.to_datetime(
            pred_df["Fecha"], errors="coerce"
        ).dt.strftime(date_fmt)
        return pred_df

    def _pack_result(
        model: str,
        resp: dict,
        fig,
        pred_df: pd.DataFrame,
        predictors_used: list[str],
        h: int,
    ) -> dict:
        base = {
            "model": model,
            "mape": resp["mape"],
            "rmse": resp["rmse"],
            "mae": resp["mae"],
            "fig": fig,
            "predictors_used": predictors_used,
            "horizon": h,
            "pred_df": pred_df,
        }

        # Extra por modelo (para mostrar en UI)
        if model == "sarimax":
            base.update(
                {
                    "order": resp.get("order"),
                    "seasonal_order": resp.get("seasonal_order"),
                }
            )
        elif model == "xgboost":
            base.update(
                {
                    "xgb_params": resp.get("xgb_params"),
                    "feature_cols": resp.get("feature_cols"),
                }
            )
        return base

    def _kpi_card(label: str, value: str):
        return ui.tags.div(
            ui.tags.div(
                _metric_label_with_info(label),
                style="font-size:12px; color:#6b7280; margin-bottom:4px;",
            ),
            ui.tags.div(value, style="font-size:20px; font-weight:700;"),
            style=(
                "flex: 1 1 160px; padding: 10px 12px; border: 1px solid #e5e7eb; "
                "border-radius: 12px; background: #ffffff;"
            ),
        )

    def _pill(text: str):
        return ui.tags.span(
            text,
            style=(
                "display:inline-block; padding: 3px 10px; border-radius: 999px; "
                "background:#f3f4f6; border:1px solid #e5e7eb; font-size:12px;"
            ),
        )

    METRIC_DESCRIPTIONS = {
        "MAPE": "Error porcentual absoluto medio (en %).",
        "RMSE": "Raíz del error cuadrático medio (penaliza más los errores grandes).",
        "MAE": "Error absoluto medio (promedio del error en la escala original).",
    }

    def _metric_info_tooltip(description: str):
        return ui.tooltip(
            ui.tags.span(
                ui.HTML(ICON_SVG_INFO), style="display:inline-flex; cursor:help;"
            ),
            description,
        )

    def _metric_label_with_info(label: str):
        return ui.tags.div(
            ui.tags.span(label),
            _metric_info_tooltip(
                METRIC_DESCRIPTIONS.get(label, "Métrica de error del modelo.")
            ),
            style="display:flex; align-items:center; gap:6px;",
        )

    def _metrics_info_tooltip():
        return _metric_info_tooltip(
            "MAPE, RMSE y MAE evalúan el error del modelo desde distintas perspectivas."
        )

    def _build_table_html(df: pd.DataFrame) -> ui.Tag:
        header_cells = [
            ui.tags.th(
                c,
                style="padding:6px 10px; border-bottom:2px solid #e5e7eb; text-align:left;",
            )
            for c in df.columns
        ]
        rows = []
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                cells.append(
                    ui.tags.td(
                        "" if pd.isna(row[col]) else str(row[col]),
                        style="padding:6px 10px; border-bottom:1px solid #f0f0f0;",
                    )
                )
            rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*header_cells)),
                ui.tags.tbody(*rows),
                style="border-collapse:collapse; width:100%; font-size:0.9rem;",
            ),
            style="overflow:auto; max-height:420px;",
        )

    def _build_prediction_figure(df, pred, title, ylabel, xlabel, column_y, decimals):
        df_plot = ensure_datetime_index(df)
        pred_index = pd.to_datetime(pred.index, errors="coerce")
        pred_series = pd.Series(pred.values, index=pred_index, name="Predicción")
        pred_series = pred_series[~pd.isna(pred_series.index)]

        all_index = pd.DatetimeIndex(
            df_plot.index.tolist() + pred_series.index.tolist()
        )
        x_min, x_max = compute_time_axis_bounds(all_index)

        customdata = [
            [
                ts.strftime("%d-%m-%Y"),
                None,
                fmt_num(val, decimals),
                None,
                None,
                "prediccion",
            ]
            for ts, val in zip(pred_series.index, pred_series.values)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[column_y],
                mode="lines",
                name="Real",
                line={"color": "#1f77b4", "width": 2},
                hovertemplate=f"Fecha: %{{x|%d-%m-%Y}}<br>Valor real: %{{y:,.{decimals}f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pred_series.index,
                y=pred_series.values,
                mode="lines+markers",
                name="Predicción",
                line={"color": "#e11d48", "width": 3},
                marker={"color": "#e11d48", "size": 9},
                customdata=customdata,
                hovertemplate=f"Fecha: %{{x|%d-%m-%Y}}<br>Predicción: %{{y:,.{decimals}f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title,
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
            hovermode="closest",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend={"orientation": "h", "y": 1.08, "x": 0.01},
            xaxis={
                "title": xlabel,
                "range": [x_min, x_max],
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
            yaxis={
                "title": ylabel,
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
        )
        return fig

    # ------------------------
    # Inputs auxiliares
    # ------------------------
    @reactive.calc
    def exog_choices():
        return list(predictors_rv.get() or [])

    @reactive.calc
    def selected_model():
        if "model_choice" not in input:
            return "sarimax"
        return input.model_choice() or "sarimax"

    @reactive.calc
    def exog_selected():
        choices = exog_choices()
        if "model_exogs" not in input:
            return choices
        sel = list(input.model_exogs() or [])
        return [s for s in sel if s in choices]

    # ------------------------
    # Almacén de resultados (solo se llena al pulsar botón)
    # ------------------------
    pred_results_rv = reactive.Value(None)
    last_sig_rv = reactive.Value(None)

    @reactive.calc
    def max_preds_available():
        m = max_num_predictions()
        if m is None:
            return 0
        try:
            return max(0, int(m))
        except Exception:
            return 0

    @reactive.calc
    def pred_horizon():
        m = max_preds_available()
        if m < 1:
            return 0
        if "pred_horizon" not in input:
            return 1
        try:
            v = int(input.pred_horizon())
        except Exception:
            v = 1
        return min(max(v, 1), m)

    @reactive.calc
    def pred_signature():
        if current_step.get() != 4:
            return None
        model = selected_model()
        exogs = tuple(exog_selected() or [])
        target = target_var_rv.get()
        filters = selected_filters_by_var()
        horizon = pred_horizon()
        return (model, target, exogs, repr(filters), horizon)

    @reactive.effect
    def _invalidate_prediction_when_inputs_change():
        sig = pred_signature()
        if sig is None:
            return
        last = last_sig_rv.get()
        if last is not None and sig != last:
            pred_results_rv.set(None)
            pred_results_err_rv.set(None)

    # ------------------------
    # Cálculo bajo demanda: SOLO al pulsar "Calcula predicción"
    # ------------------------
    @reactive.effect
    @reactive.event(input.calc_pred)
    async def _compute_prediction_on_click():
        if current_step.get() != 4:
            return
        if input.calc_pred() == 0:
            return
        if "model_exogs" not in input:
            pred_results_rv.set(None)
            return

        # Leer valores reactivos en el hilo principal
        horizon = pred_horizon()
        if horizon < 1:
            pred_results_rv.set(None)
            last_sig_rv.set(pred_signature())
            return

        model = selected_model()
        predictors_used = exog_selected()
        target = target_var_rv.get()
        filters = selected_filters_by_var()
        sig = pred_signature()
        decimals = _pred_table_decimals()

        runner = MODEL_RUNNERS.get(model)
        if runner is None:
            pred_results_rv.set(None)
            last_sig_rv.set(sig)
            return

        payload = _build_payload(model, target, predictors_used, filters, horizon)

        # Insertar spinner (bypass flush, se envía inmediatamente al browser)
        _spinner_id = _SPINNER_ID
        ui.insert_ui(
            ui.tags.div(
                ui.tags.div(class_="graph-spinner"),
                ui.tags.div("Calculando predicción...", class_="graph-loading-text"),
                class_="graph-loading-container",
                id=_spinner_id,
            ),
            selector="#pred_result_area",
            where="afterBegin",
            immediate=True,
        )

        try:

            def _run_prediction():
                resp = runner(payload)
                parsed = _parse_forecast_response(resp)
                if parsed is None:
                    return None
                df, y_col, future, h, pred_vals, pred_series = parsed
                pred_df = _build_pred_df(future, pred_vals, date_fmt="%d-%m-%Y")
                fig = _build_prediction_figure(
                    df=df,
                    pred=pred_series,
                    title=(
                        "Predicciones SARIMAX"
                        if model == "sarimax"
                        else "Predicciones XGBoost"
                    ),
                    ylabel="Valores",
                    xlabel="Fecha",
                    column_y=y_col,
                    decimals=decimals,
                )
                return _pack_result(model, resp, fig, pred_df, predictors_used, h)

            result = await asyncio.to_thread(_run_prediction)
            pred_results_rv.set(result)
            pred_results_err_rv.set(None)
            last_sig_rv.set(sig)
        except Exception as e:
            err_msg = str(e)
            if hasattr(e, "response"):
                try:
                    body = e.response.json()
                    if "detail" in body:
                        err_msg = str(body["detail"])
                except Exception:
                    pass
            pred_results_err_rv.set(humanize_error(err_msg))
            pred_results_rv.set(None)
            last_sig_rv.set(sig)
        finally:
            ui.remove_ui(selector=f"#{_spinner_id}", immediate=True)

    # ------------------------
    # Outputs (usa el almacén, no calcula)
    # ------------------------
    @output
    @render.ui
    def model_plot():
        res = pred_results_rv.get()
        if not res:
            return ui.div()

        return ui.HTML(
            build_interactive_plot_html(
                res["fig"],
                session.ns("pred_plot_widget"),
                session.ns("pred_plot_click"),
            )
        )

    @output
    @render.ui
    def pred_table():
        res = pred_results_rv.get()
        if not res or res.get("pred_df") is None:
            return ui.div()

        decimals = _pred_table_decimals()

        click = input.pred_plot_click() if "pred_plot_click" in input else None
        if click and click.get("scenario") is not None:
            click_y = pd.to_numeric(click.get("y"), errors="coerce")
            detail_df = pd.DataFrame(
                [
                    {
                        "Fecha": click.get("date_label") or "",
                        "Predicción": (
                            click.get("scenario")
                            or (fmt_num(click_y, decimals) if pd.notna(click_y) else "")
                        ),
                    }
                ]
            )
            return ui.tags.div(
                ui.tags.div(
                    "Punto seleccionado",
                    style="font-weight:600; margin-bottom:8px;",
                ),
                _build_table_html(detail_df),
            )

        df = res["pred_df"].copy()
        if "Predicción" in df.columns:
            df["Predicción"] = pd.to_numeric(df["Predicción"], errors="coerce").apply(
                lambda v: fmt_num(v, decimals) if pd.notna(v) else v
            )
        return ui.tags.div(
            ui.tags.div(
                "Haz clic en un punto para ver su detalle.",
                style="color:#6b7280; font-size:0.9rem; margin-bottom:8px;",
            ),
            _build_table_html(df),
        )

    # ------------------------
    # UI Panel 4
    # ------------------------
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        choices = exog_choices()
        selected = exog_selected()
        model = selected_model()

        m = max_preds_available()
        h = pred_horizon()
        res = pred_results_rv.get()

        # Header (inputs)
        header = ui.card(
            ui.tags.div(
                ui.h3(
                    "Panel 4: Modelo y exógenas", style="margin:0; text-align:center;"
                ),
                ui.tags.div(
                    "Elige el modelo y las exógenas. La predicción SOLO se ejecuta al pulsar el botón.",
                    style="color:#6b7280; margin-top:4px; text-align:center;",
                ),
                style="width:100%;",
            ),
            ui.tags.hr(style="margin:12px 0;"),
            ui.tags.div(
                ui.tags.div(
                    ui.input_radio_buttons(
                        "model_choice",
                        "Modelo",
                        choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                        selected=model,
                        inline=True,
                    ),
                    style="flex: 1 1 260px;",
                ),
                ui.tags.div(
                    ui.input_checkbox_group(
                        "model_exogs",
                        "Variables exógenas (activar/desactivar)",
                        choices=choices,
                        selected=selected,
                    ),
                    style="flex: 2 1 420px;",
                ),
                style="display:flex; gap:14px; flex-wrap:wrap; justify-content:center;",
            ),
            ui.tags.div(
                (
                    ui.input_slider(
                        "pred_horizon",
                        "Valores a predecir",
                        min=1,
                        max=m,
                        value=(h if h >= 1 else 1),
                        step=1,
                    )
                    if m >= 1
                    else ui.tags.div(
                        ui.tags.b("Valores a predecir: "),
                        ui.tags.span(
                            "— (selecciona exógenas compatibles para habilitar el horizonte)"
                        ),
                        style="margin-top: 8px; color:#6b7280; text-align:center;",
                    )
                ),
                style="margin-top:10px;",
            ),
            # BOTÓN debajo del slider, centrado
            ui.tags.div(
                ui.input_action_button(
                    "calc_pred", "Calcula predicción", class_="btn-primary"
                ),
                style="margin-top: 10px; display:flex; justify-content:center;",
            ),
            style="padding: 14px; border-radius: 14px;",
        )

        footer = ui.tags.div(
            ui.input_action_button("btn_prev_4", "← Anterior"),
            style="margin-top: 12px;",
        )

        # Estado: calculando (spinner se inyecta via insert_ui)

        # Estado sin resultados ni errores
        err = pred_results_err_rv.get()

        if res is None:
            status_content = []
            if err:
                status_content = [
                    ui.tags.div(
                        ui.tags.b("Error: "),
                        ui.tags.span(err),
                        style=(
                            "margin-top: 10px; padding: 10px 12px; border: 1px solid #fecaca; "
                            "border-radius: 12px; background:#fef2f2; color:#991b1b;"
                        ),
                    )
                ]
            else:
                status_content = [
                    ui.tags.div(
                        ui.tags.span("Estado: ", style="font-weight:600;"),
                        ui.tags.span(
                            "listo para calcular. Pulsa «Calcula predicción».",
                            style="color:#6b7280;",
                        ),
                        style=(
                            "margin-top: 10px; padding: 10px 12px; border: 1px dashed #d1d5db; "
                            "border-radius: 12px; background:#fafafa;"
                        ),
                    )
                ]

            return ui.div(
                PANEL_STYLES,
                header,
                ui.tags.div(
                    *status_content,
                    id="pred_result_area",
                ),
                footer,
            )

        # Resultados
        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]

        kpis = ui.tags.div(
            _kpi_card("MAPE", fmt_num(mape, 2, "%")),
            _kpi_card("RMSE", fmt_num(rmse, 2)),
            _kpi_card("MAE", fmt_num(mae, 2)),
            style="display:flex; gap:12px; flex-wrap:wrap; margin-top: 10px;",
        )

        kpis_title = ui.tags.div(
            ui.tags.span("Métricas del modelo", style="font-weight:600;"),
            _metrics_info_tooltip(),
            style="display:flex; align-items:center; gap:6px; margin-top:10px;",
        )

        exogs_line = ui.tags.div(
            ui.tags.span(
                "Exógenas activas: ", style="font-weight:600; margin-right:6px;"
            ),
            _pill(
                ", ".join(res["predictors_used"])
                if res["predictors_used"]
                else "Ninguna"
            ),
            style="margin-top: 10px;",
        )

        # Layout plot + tabla (responsive)
        body = ui.tags.div(
            ui.card(
                ui.h5("Evolución temporal", style="margin:0 0 8px 0;"),
                ui.output_ui("model_plot"),
                style=(
                    "padding: 12px; border-radius: 14px;"
                    "flex: 2 1 640px; min-width: 520px;"
                ),
            ),
            ui.card(
                ui.h5("Detalle / valores predichos", style="margin:0 0 8px 0;"),
                ui.tags.div(  # wrapper para controlar altura/scroll si crece
                    ui.output_ui("pred_table"),
                    style="max-height: 420px; overflow:auto;",
                ),
                style=(
                    "padding: 12px; border-radius: 14px;"
                    "flex: 1 1 420px; min-width: 340px;"
                ),
            ),
            style=(
                "display:flex; gap:12px; flex-wrap:wrap;"
                "align-items:flex-start;"  # <- evita que la tabla se estire en alto
                "margin-top: 12px;"
            ),
        )

        return ui.div(
            PANEL_STYLES,
            header,
            ui.tags.div(
                exogs_line,
                kpis_title,
                kpis,
                body,
                id="pred_result_area",
            ),
            footer,
        )

    @reactive.Effect
    @reactive.event(input.btn_prev_4)
    def _go_step_3_from_4():
        # Borrar resultados de predicción al volver al panel 3
        pred_results_rv.set(None)
        current_step.set(3)
