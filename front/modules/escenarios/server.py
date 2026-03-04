import pandas as pd
from shiny import module, reactive, render, ui
import httpx
from back.models.utils.models_graph import plot_predictions
from front.utils.back_api_wrappers import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
    sarimax_run,
    xgboost_run,
)
from front.utils.utils import (
    ICON_SVG_INFO,
    PrediccionesCache,
    _safe_alias,
    build_name_to_table,
    compatibilidad_con_objetivo,
    create_calendar_filter,
    detect_temporal_filters,
    fmt as _fmt,
    fmt_date_by_temporality as _fmt_date_temp,
    group_by_category,
    panel_styles,
    process_date_range_filters,
    slug as _slug,
    stable_id,
)


# -----------------------------
# Module
# -----------------------------
@module.server
def escenarios_server(input, output, session):
    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    current_step = reactive.Value(0)

    scenario_type_rv = reactive.Value(None)  # "futuro" o "pasado"
    target_var_rv = reactive.Value(None)  # objetivo seleccionado
    predictors_rv = reactive.Value([])  # exógenas seleccionadas (panel 2)

    base_info_rv = reactive.Value(None)  # base histórica cargada (modo pasado)
    scenario_res_rv = reactive.Value(None)  # resultado de escenario (pasado/futuro)
    last_sig_rv = reactive.Value(None)  # firma usada para invalidación de resultados
    saved_filter_values_rv = reactive.Value({})  # filtros guardados del panel 3
    saved_horizon_rv = reactive.Value(2)           # horizonte guardado del panel 4
    saved_fut_cell_values_rv = reactive.Value({})  # valores de celdas exógenas guardados

    # ---------------------------------------------------------------------
    # Static data / cache
    # ---------------------------------------------------------------------
    catalog_entries = get_names_in_table_catalog() or []
    name_to_table = build_name_to_table(catalog_entries)
    cache = PrediccionesCache(name_to_table)
    PANEL_STYLES = panel_styles()

    # Evita registrar múltiples handlers para IDs dinámicos
    _registered_pick_handlers: set[str] = set()

    MODEL_RUNNERS = {
        "sarimax": sarimax_run,
        "xgboost": xgboost_run,
    }

    # ---------------------------------------------------------------------
    # UI helpers (puros)
    # ---------------------------------------------------------------------
    def _extract_dates(df: pd.DataFrame) -> pd.Series:
        """Extrae fechas de un df, intentando columnas estándar, si no, usa índice."""
        if "__dt" in df.columns:
            return pd.to_datetime(df["__dt"], errors="coerce")
        if {"anio", "mes", "dia"}.issubset(df.columns):
            return pd.to_datetime(
                dict(year=df["anio"], month=df["mes"], day=df["dia"]),
                errors="coerce",
            )
        if {"anio", "mes"}.issubset(df.columns):
            return pd.to_datetime(
                dict(year=df["anio"], month=df["mes"], day=1),
                errors="coerce",
            )
        return pd.to_datetime(df.index, errors="coerce")

    # ---------------------------------------------------------------------
    # Temporalidad / normalización (mensual vs diario)
    # ---------------------------------------------------------------------
    @reactive.Calc
    def target_temporalidad() -> str:
        meta = cache.get_meta(target_var_rv.get()) or {}
        return str(meta.get("temporalidad", "")).lower()

    def is_monthly(temp: str) -> bool:
        t = (temp or "").lower()
        return ("mes" in t) or ("mens" in t) or ("monthly" in t)

    def norm_dt(dt: pd.Timestamp, temp: str) -> pd.Timestamp:
        """Normaliza timestamp según temporalidad (inicio de mes o inicio de día)."""
        if pd.isna(dt):
            return pd.NaT
        return (
            dt.to_period("M").to_timestamp(how="start")
            if is_monthly(temp)
            else dt.normalize()
        )

    def granularity(temp: str) -> str:
        """Frecuencia para date_range."""
        return "MS" if is_monthly(temp) else "D"

    def parse_user_dt(txt: str, temp: str) -> pd.Timestamp:
        """
        Parsea texto de fecha. Si mensual, admite YYYY-MM (se completa con -01).
        """
        s = (txt or "").strip()
        if not s:
            return pd.NaT

        if is_monthly(temp) and len(s) == 7 and s[4] == "-":  # YYYY-MM
            s = f"{s}-01"

        dt = pd.to_datetime(s, errors="coerce")
        return norm_dt(dt, temp)

    def normalize_dt_series(x, temp: str) -> pd.Series:
        s = pd.to_datetime(x, errors="coerce")
        if is_monthly(temp):
            return s.dt.to_period("M").dt.to_timestamp(how="start")
        return s.dt.normalize()

    def dt_str(dt: pd.Timestamp, temp: str, *, kind: str = "key") -> str:
        """
        kind="key"  -> YYYY-MM-DD (estable para ids/claves)
        kind="label"-> YYYY-MM si mensual, YYYY-MM-DD si diario
        """
        d = norm_dt(dt, temp)
        if pd.isna(d):
            return ""
        if kind == "label" and is_monthly(temp):
            return d.strftime("%Y-%m")
        return d.strftime("%Y-%m-%d")

    def _is_monthly(temp: str) -> bool:
        t = (temp or "").lower()
        return ("mes" in t) or ("mens" in t) or ("monthly" in t)

    def _dt_label(dt: pd.Timestamp, temp: str) -> str:
        if pd.isna(dt):
            return ""
        d = (
            dt.to_period("M").to_timestamp(how="start")
            if _is_monthly(temp)
            else dt.normalize()
        )
        return d.strftime("%Y-%m") if _is_monthly(temp) else d.strftime("%Y-%m-%d")

    # ---------------------------------------------------------------------
    # Step indicator (opcional)
    # ---------------------------------------------------------------------
    @output
    @render.ui
    def step_indicator():
        step = current_step.get()
        stype = scenario_type_rv.get()
        # En paso 0 o modo pasado no mostramos indicador de pasos
        if step == 0 or stype != "futuro":
            return ui.div()

        labels = ["Objetivo", "Predictoras", "Filtros", "Escenarios"]
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

        stype = scenario_type_rv.get() or ""
        type_label = (
            "Futuros" if stype == "futuro" else "Pasados" if stype == "pasado" else ""
        )
        badge = (
            ui.tags.span(
                f"Modo: Escenarios {type_label}",
                style=(
                    "margin-left:16px; font-size:0.85rem; color:#4f46e5; "
                    "background:#eef2ff; padding:4px 12px; border-radius:999px; font-weight:600;"
                ),
            )
            if type_label
            else ui.span()
        )
        return ui.div(
            PANEL_STYLES,
            ui.tags.div(*nodes, class_="step-indicator"),
            ui.tags.div(
                badge, style="display:flex; justify-content:flex-end; margin-top:6px;"
            ),
            style="margin:8px 0;",
        )

    # =====================================================================
    # Panel 0: Tipo de escenario (Pasado / Futuro)
    # =====================================================================
    @output
    @render.ui
    def step_panel_0():
        if current_step.get() != 0:
            return ui.div()

        return ui.div(
            PANEL_STYLES,
            ui.tags.div(
                ui.h3(
                    "¿Qué tipo de escenario quieres explorar?",
                    style="text-align:center; margin-bottom:8px;",
                ),
                ui.tags.p(
                    "Elige entre analizar escenarios sobre datos pasados o generar predicciones hacia el futuro.",
                    style="text-align:center; color:#475569; margin-bottom:24px;",
                ),
                ui.tags.div(
                    ui.tags.div(
                        ui.input_action_button(
                            "esc_choose_pasado",
                            ui.tags.div(
                                ui.tags.div(
                                    "\U0001f4c5",
                                    style="font-size:2.5rem; margin-bottom:8px;",
                                ),
                                ui.tags.div(
                                    "Escenarios Pasados",
                                    style="font-size:1.25rem; font-weight:700; margin-bottom:6px;",
                                ),
                                ui.tags.div(
                                    "Modifica valores históricos de las exógenas y observa cómo habría cambiado la predicción.",
                                    style="font-size:0.85rem; color:#64748b; font-weight:400;",
                                ),
                            ),
                            class_="esc-type-card",
                        ),
                        style="width:300px; height:220px; display:flex; flex-shrink:0;",
                    ),
                    ui.tags.div(
                        ui.input_action_button(
                            "esc_choose_futuro",
                            ui.tags.div(
                                ui.tags.div(
                                    "\U0001f680",
                                    style="font-size:2.5rem; margin-bottom:8px;",
                                ),
                                ui.tags.div(
                                    "Escenarios Futuros",
                                    style="font-size:1.25rem; font-weight:700; margin-bottom:6px;",
                                ),
                                ui.tags.div(
                                    "Define valores futuros para las exógenas y genera predicciones hacia adelante.",
                                    style="font-size:0.85rem; color:#64748b; font-weight:400;",
                                ),
                            ),
                            class_="esc-type-card",
                        ),
                        style="width:300px; height:220px; display:flex; flex-shrink:0;",
                    ),
                    style="display:flex; flex-direction:row; justify-content:center; gap:24px;",
                ),
                style="margin:0 auto;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.esc_choose_pasado)
    def _choose_pasado():
        scenario_type_rv.set("pasado")
        current_step.set(1)

    @reactive.Effect
    @reactive.event(input.esc_choose_futuro)
    def _choose_futuro():
        scenario_type_rv.set("futuro")
        current_step.set(1)

    # =====================================================================
    # Panel 1: Objetivo
    # =====================================================================
    @output
    @render.ui
    def step_panel_1():
        if current_step.get() != 1 or scenario_type_rv.get() != "futuro":
            return ui.div()

        grouped = group_by_category(catalog_entries)
        all_names = [n for names in grouped.values() for n in names]

        if target_var_rv.get() is None and all_names:
            target_var_rv.set(all_names[0])

        selected = target_var_rv.get()

        panels = []
        for cat, names in grouped.items():
            btns = []
            for name in names:
                btn_id = stable_id("esc_target", name)

                if btn_id not in _registered_pick_handlers:
                    _registered_pick_handlers.add(btn_id)

                    @reactive.Effect
                    @reactive.event(input[btn_id])
                    def _on_pick_target(_name=name):
                        target_var_rv.set(_name)
                        # Cambiar objetivo invalida lo calculado
                        scenario_res_rv.set(None)
                        base_info_rv.set(None)

                btns.append(
                    ui.input_action_button(
                        btn_id,
                        name,
                        class_=(
                            "var-pick is-selected" if selected == name else "var-pick"
                        ),
                    )
                )

            panels.append(
                ui.accordion_panel(
                    cat,
                    ui.div(*btns, class_="var-list"),
                    value=_slug(cat),
                )
            )

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 1: Seleccionar variable objetivo"),
            ui.accordion(*panels, id="esc_acc_target", open=True, multiple=True),
            ui.div(
                ui.input_action_button("esc_prev_1", "← Anterior"),
                ui.input_action_button("esc_next_1", "Siguiente →"),
                style="display:flex;gap:8px;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_1)
    def _go_step_0_from_1():
        current_step.set(0)

    @reactive.Effect
    @reactive.event(input.esc_next_1)
    def _go_step_2():
        current_step.set(2)

    # =====================================================================
    # Panel 2: Predictoras + compatibilidad
    # =====================================================================
    @reactive.Calc
    def predictor_pairs():
        grouped = group_by_category(catalog_entries, exclude_name=target_var_rv.get())
        pairs = []
        for _, names in grouped.items():
            for name in names:
                pairs.append((stable_id("esc_pred", name), name))
        return pairs

    @reactive.Calc
    def selected_predictors():
        target = target_var_rv.get()
        if not target:
            return []

        target_meta = cache.get_meta(target) or {}
        target_start, target_end = cache.get_date_range(target)

        selected = []
        for var_id, name in predictor_pairs():
            if not (var_id in input and input[var_id]()):
                continue

            ok, _ = compatibilidad_con_objetivo(
                predictor_name=name,
                predictor_meta=cache.get_meta(name),
                target_name=target,
                target_meta=target_meta,
                target_start=target_start,
                target_end=target_end,
                cache=cache,
            )
            if ok:
                selected.append(name)
        return sorted(set(selected))

    @reactive.Effect
    def _sync_predictors_rv():
        predictors_rv.set(selected_predictors())

    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2 or scenario_type_rv.get() != "futuro":
            return ui.div()

        target = target_var_rv.get()
        grouped = group_by_category(catalog_entries, exclude_name=target)

        target_meta = cache.get_meta(target) if target else {}
        ts, te = cache.get_date_range(target) if target else (None, None)

        selected_set = set(selected_predictors())
        target_temp = _fmt(target_meta.get("temporalidad"))

        panels = []
        for cat, names in grouped.items():
            blocks = []
            for name in names:
                var_id = stable_id("esc_pred", name)
                meta = cache.get_meta(name)

                ok, reason = compatibilidad_con_objetivo(
                    predictor_name=name,
                    predictor_meta=meta,
                    target_name=target,
                    target_meta=target_meta,
                    target_start=ts,
                    target_end=te,
                    cache=cache,
                )

                badge = ui.tags.span(
                    "Compatible" if ok else "No compatible",
                    class_=(
                        "compat-badge compat-yes"
                        if ok
                        else "compat-badge compat-no"
                    ),
                )

                info_icon = None
                if not ok and reason:
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


                selector = (
                    ui.input_checkbox(var_id, name, value=(name in selected_set))
                    if ok
                    else ui.tags.span(name, style="font-weight:600; color:#6e7781;")
                )

                blocks.append(
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


            panels.append(ui.accordion_panel(cat, ui.div(*blocks), value=_slug(cat)))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 2: Seleccionar exógenas"),
            ui.accordion(*panels, id="esc_acc_preds", open=True, multiple=True),
            ui.div(
                ui.input_action_button("esc_prev_2", "← Anterior"),
                ui.input_action_button("esc_next_2", "Siguiente →"),
                style="display:flex;gap:8px;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_2)
    def _go_step_1_from_2():
        current_step.set(1)

    @reactive.Effect
    @reactive.event(input.esc_next_2)
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
        if "esc_fut_horizon" in input:
            try:
                extend_steps = max(0, int(input.esc_fut_horizon() or 0))
            except Exception:
                extend_steps = 0

        
        target_temporal_filters = []

        for item in vars_to_config():
            pretty = item["pretty"]
            table = item["table"]
            is_target = item["is_target"]

            filtros = cache.get_filters(table)
            selected_list: list[dict] = []

            temp = detect_temporal_filters(filtros)

            
            if is_target and (temp["mes"] or temp["dia"]):
                start_def, end_def = target_selected_range()
                if start_def and end_def:
                    date_range = (start_def, end_def)
                    temporal_filters = process_date_range_filters(
                        date_range, filtros, table
                    )
                    selected_list.extend(temporal_filters)
                    
                    target_temporal_filters = temporal_filters

            elif is_target and temp["anio"]:
                start_def, end_def = target_selected_range()
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

                input_id = stable_id("flt", f"{f['table']}__{f['col']}")
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
        if current_step.get() != 3 or scenario_type_rv.get() != "futuro":
            return ui.div()

        vars_sel = vars_to_config()
        if not vars_sel:
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 3: configurar filtros"),
                ui.p("No hay variables seleccionadas para configurar."),
            )

        panels = []

        target_var = target_var_rv.get()
        target_start, target_end = target_selected_range()
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_temporality = target_meta.get("temporalidad")

       
        display_start = (
            target_start
            if target_start
            else (cache.get_date_range(target_var)[0] if target_var else None)
        )
        display_end = (
            target_end
            if target_end
            else (cache.get_date_range(target_var)[1] if target_var else None)
        )

        
        target_start_fmt = (
            _fmt_date_temp(display_start, target_temporality) if display_start else "—"
        )
        target_end_fmt = (
            _fmt_date_temp(display_end, target_temporality) if display_end else "—"
        )

        
        range_status = (
            "disponible"
            if target_start
            else "disponible"
        )
        for item in vars_sel:
            pretty = item["pretty"]
            table = item["table"]
            is_target = item["is_target"]

            filtros = cache.get_filters(table)

            start_date, end_date = cache.get_date_range(pretty)

            if not filtros:
                if is_target:
                    body = ui.p(
                        "Sin filtros configurados en tbl_admin_filtros para esta variable/tabla."
                    )
                else:
                    
                    body = ui.div(
                        ui.tags.div(
                            ui.tags.span(
                                "📌 Variable Exógena",
                                style="font-weight:600; color:#6e7781;",
                            ),
                            style="margin-bottom:12px;",
                        ),
                        ui.tags.div(
                            "✓ Esta variable se ajustará automáticamente al rango temporal de la variable objetivo.",
                            style="padding:8px; background-color:#dff6dd; border-left:3px solid #1a7f37; color:#1a7f37; border-radius:4px;",
                        ),
                        ui.tags.div(
                            ui.tags.span(
                                f"Rango {range_status}: ",
                                style="font-weight:500; margin-top:8px; display:inline-block;",
                            ),
                            ui.tags.span(f"{target_start_fmt} → {target_end_fmt}"),
                            style="margin-top:8px;",
                        ),
                    )
            else:
                controls = []

                if not is_target:
                    # Para las exógenas, mostrar mensaje informativo
                    controls.append(
                        ui.tags.div(
                            ui.tags.div(
                                ui.tags.span(
                                    "📌 Variable Exógena",
                                    style="font-weight:600; color:#6e7781;",
                                ),
                                style="margin-bottom:12px;",
                            ),
                            ui.tags.div(
                                "✓ Esta variable se ajustará automáticamente al rango temporal de la variable objetivo.",
                                style="padding:8px; background-color:#dff6dd; border-left:3px solid #1a7f37; color:#1a7f37; border-radius:4px; margin-bottom:12px;",
                            ),
                            ui.tags.div(
                                ui.tags.span(
                                    f"Rango {range_status}: ", style="font-weight:500;"
                                ),
                                ui.tags.span(f"{target_start_fmt} → {target_end_fmt}"),
                                style="margin-bottom:16px;",
                            ),
                            style="margin-bottom:16px;",
                        )
                    )
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

                    input_id = stable_id("flt", f"{t}__{col}")
                    choices = cache.get_distinct("IA", t, col)
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

                body = (
                    ui.div(*controls) if controls else ui.p("Sin filtros disponibles.")
                )

            panels.append(
                ui.accordion_panel(
                    pretty,
                    body,
                    value=_slug(table),
                )
            )

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 3: configurar filtros"),
            ui.tags.ul(
                ui.tags.li("Para cada variable se muestran sus filtros configurados."),
                ui.tags.li(
                    "En exógenas, el rango temporal se ajusta automáticamente al rango elegido en la variable objetivo."
                ),
            ),
            ui.accordion(*panels, id="acc_filters", open=True, multiple=True),
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
                input_id = stable_id("flt", f"{t}__{f['col']}")
                if input_id in input:
                    vals = input[input_id]()
                    if vals:
                        saved[input_id] = list(vals) if isinstance(vals, (list, tuple)) else [str(vals)]
        saved_filter_values_rv.set(saved)
        current_step.set(4)

    # =====================================================================
    # Panel 4: Escenarios FUTUROS (exógenas inventadas por el usuario)
    # =====================================================================

    # --- Resultado / error (reusa scenario_res_rv si ya lo tienes) ---
    scenario_err_rv = reactive.Value(None)  # error legible

    # ------------------------
    # Helpers Panel 4 (puros)
    # ------------------------
    def _cell_id(exog_name: str, k: int) -> str:
        # id estable por exógena + periodo
        return stable_id("esc_fut_val", f"{exog_name}__P{k}")

    def _infer_future_index_from_target_end(horizon: int) -> pd.DatetimeIndex:
        """
        Backend infiere future_index a partir del último __dt histórico.
        Aquí lo aproximamos con el end del target (debería coincidir).
        """
        target = target_var_rv.get()
        if not target:
            return pd.DatetimeIndex([])

        _s, end = target_selected_range()
        if end is None:
            _s, end = cache.get_date_range(target)
        if end is None:
            return pd.DatetimeIndex([])

        temp = target_temporalidad()
        end_dt = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_dt):
            return pd.DatetimeIndex([])

        if _is_monthly(temp):
            start = (end_dt + pd.offsets.MonthBegin(1)).normalize()
            return pd.date_range(start=start, periods=horizon, freq="MS")
        start = (end_dt + pd.Timedelta(days=1)).normalize()
        return pd.date_range(start=start, periods=horizon, freq="D")

    def _parse_forecast_response(
        resp: dict, fallback_index: pd.DatetimeIndex | None = None
    ):
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

        if len(pred_vals) != len(future):
            if fallback_index is not None and len(fallback_index) == len(pred_vals):
                future = pd.DataFrame({"__dt": fallback_index})
                future_dates = pd.to_datetime(fallback_index, errors="coerce")
            else:
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
            fechas = future.index

        pred_df = pd.DataFrame({"Fecha": fechas, "Predicción": pred_vals})
        pred_df["Fecha"] = pd.to_datetime(
            pred_df["Fecha"], errors="coerce"
        ).dt.strftime(date_fmt)
        return pred_df

    # ------------------------
    # Inputs reactivos
    # ------------------------
    @reactive.Calc
    def fut_horizon() -> int:
        # (1) periodos a predecir
        if "esc_fut_horizon" not in input:
            return 2
        try:
            h = int(input.esc_fut_horizon() or 2)
        except Exception:
            h = 2
        return max(1, min(h, 60))  # cap seguridad

    @reactive.Calc
    def fut_exogs() -> list[str]:
        # exógenas seleccionadas (Panel 2)
        return list(predictors_rv.get() or [])

    @reactive.Calc
    def fut_active_exogs() -> list[str]:
        exogs = fut_exogs()
        if "esc_fut_active_exogs" not in input:
            return exogs

        selected = input.esc_fut_active_exogs() or []
        if isinstance(selected, str):
            selected = [selected]

        selected_set = set(selected)
        return [ex for ex in exogs if ex in selected_set]

    @reactive.Calc
    def fut_model() -> str:
        if "esc_fut_model" not in input:
            return "sarimax"
        return input.esc_fut_model() or "sarimax"

    @reactive.Calc
    def fut_future_index() -> pd.DatetimeIndex:
        return _infer_future_index_from_target_end(fut_horizon())

    @reactive.Calc
    def fut_matrix_values() -> dict[str, list[float | None]]:
        """
        dict: exog -> [P1..Ph] (None si no relleno)
        """
        exogs = fut_active_exogs()
        h = fut_horizon()
        out: dict[str, list[float | None]] = {}
        for ex in exogs:
            row = []
            for k in range(1, h + 1):
                cid = _cell_id(ex, k)
                v = input[cid]() if cid in input else None
                row.append(v)
            out[ex] = row
        return out

    @reactive.Calc
    def fut_signature():
        if current_step.get() != 4:
            return None
        return (
            fut_model(),
            target_var_rv.get(),
            tuple(fut_exogs()),
            tuple(fut_active_exogs()),
            fut_horizon(),
            repr(fut_matrix_values()),
            repr(selected_filters_by_var()),
        )

    
    @reactive.Effect
    def _save_horizon():
        if current_step.get() != 4:
            return
        if "esc_fut_horizon" not in input:
            return
        try:
            saved_horizon_rv.set(max(1, int(input.esc_fut_horizon() or 2)))
        except Exception:
            pass

  
    @reactive.Effect
    def _save_cells_before_exog_toggle():
        if current_step.get() != 4:
            return
      
        if "esc_fut_active_exogs" not in input:
            return
        _ = input.esc_fut_active_exogs()
        
        with reactive.isolate():
            exogs = list(fut_exogs())
            h = int(fut_horizon())
            saved = dict(saved_fut_cell_values_rv.get())
        for ex in exogs:
            for k in range(1, h + 1):
                cid = _cell_id(ex, k)
                if cid in input:
                    v = input[cid]()
                    if v is not None:
                        saved[cid] = v
        saved_fut_cell_values_rv.set(saved)

    # ------------------------
    # UI: tabla editable (2)
    # ------------------------
    @output
    @render.ui
    def esc_future_exog_selector():
        if current_step.get() != 4:
            return ui.div()

        exogs = fut_exogs()
        if not exogs:
            return ui.div()

        return ui.tags.div(
            ui.tags.b("2) Exógenas activas"),
            ui.input_checkbox_group(
                "esc_fut_active_exogs",
                label="",
                choices={ex: ex for ex in exogs},
                selected=exogs,
                inline=False,
            ),
            ui.tags.span(
                "Desmarca una exógena para excluirla del cálculo y ocultar sus valores futuros.",
                style="font-size:12px; color:#6b7280;",
            ),
            style=(
                "margin-top:10px; padding:10px 12px; border:1px solid #e5e7eb; "
                "border-radius:12px; background:#fff;"
            ),
        )

    @output
    @render.ui
    def esc_future_exog_table():
        if current_step.get() != 4:
            return ui.div()

        exogs = fut_active_exogs()
        h = fut_horizon()
        idx = fut_future_index()
        temp = target_temporalidad()

        if not exogs:
            return ui.tags.div(
                ui.tags.b("No hay exógenas activas."),
                ui.tags.span(" Marca al menos una exógena en el selector para continuar."),
                style="color:#6b7280;",
            )

        if idx.empty:
            return ui.tags.div(
                ui.tags.b("No se pudo inferir el calendario futuro."),
                ui.tags.span(" Revisa el rango de fechas del objetivo."),
                style="color:#6b7280;",
            )

        header_cells = [
            ui.tags.th("Exógena", style="position:sticky; left:0; background:#fff;")
        ]
        for k in range(1, h + 1):
            header_cells.append(
                ui.tags.th(
                    ui.tags.div(f"P{k}", style="font-weight:700;"),
                    ui.tags.div(
                        _dt_label(idx[k - 1], temp),
                        style="font-size:12px; color:#6b7280; margin-top:2px;",
                    ),
                )
            )

        _cell_saved = saved_fut_cell_values_rv.get()

        body_rows = []
        for ex in exogs:
            cells = [
                ui.tags.td(
                    ui.tags.span(ex),
                    style="position:sticky; left:0; background:#fff; font-weight:600; white-space:nowrap;",
                )
            ]
            for k in range(1, h + 1):
                cid = _cell_id(ex, k)
                _init_val = _cell_saved.get(cid, None)
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(
                            cid, label="", value=_init_val, step=0.01
                        ),  # ✅ sin ns()
                        style="min-width:120px;",
                    )
                )
            body_rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.div(
                ui.tags.b("3) Valores futuros de exógenas activas"),
                ui.tags.span(" (rellena todas las celdas)"),
                style="margin-bottom:8px;",
            ),
            ui.tags.div(
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(*header_cells)),
                    ui.tags.tbody(*body_rows),
                    style="border-collapse:collapse; width:max-content;",
                ),
                style="overflow:auto; max-width:100%; border:1px solid #e5e7eb; border-radius:12px; padding:8px;",
            ),
        )

    # ------------------------
    # Cálculo bajo demanda (3)
    # ------------------------
    @reactive.Effect
    @reactive.event(input.esc_fut_calc)
    def _compute_future_scenario_on_click():
        if current_step.get() != 4:
            return
        if int(input.esc_fut_calc() or 0) == 0:
            return

        
        _exogs_snap = fut_exogs()
        _h_snap = fut_horizon()
        _saved_snap = dict(saved_fut_cell_values_rv.get())
        for _ex in _exogs_snap:
            for _k in range(1, _h_snap + 1):
                _cid = _cell_id(_ex, _k)
                if _cid in input:
                    _v = input[_cid]()
                    if _v is not None:
                        _saved_snap[_cid] = _v
        saved_fut_cell_values_rv.set(_saved_snap)
        # ─────────────────────────────────────────────────────────────────────

       
        try:
            n_clicks = int(input.esc_fut_calc() or 0)
        except Exception:
            n_clicks = -1

        scenario_err_rv.set(f"Calculando… (click={n_clicks})")
        scenario_res_rv.set(None)

        try:
            target = target_var_rv.get()
            exogs = fut_active_exogs()
            h = fut_horizon()
            model = fut_model()
            filters = selected_filters_by_var()

            if not target:
                scenario_err_rv.set("No hay variable objetivo seleccionada.")
                last_sig_rv.set(fut_signature())
                return

            if not exogs:
                scenario_err_rv.set("No hay exógenas activas en el Panel 4.")
                last_sig_rv.set(fut_signature())
                return

            idx = fut_future_index()
            if idx.empty or len(idx) != h:
                scenario_err_rv.set("No pude construir el índice temporal futuro.")
                last_sig_rv.set(fut_signature())
                return

            mat = fut_matrix_values()

            for ex in exogs:
                for k, v in enumerate(mat.get(ex, []), start=1):
                    if v is None:
                        scenario_err_rv.set(f"Falta valor para '{ex}' en P{k}.")
                        last_sig_rv.set(fut_signature())
                        return

            future_values = []
            for j, dt in enumerate(idx):
                date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
                for ex in exogs:
                    future_values.append(
                        {"var": ex, "date": date_str, "value": float(mat[ex][j])}
                    )

            runner = MODEL_RUNNERS.get(model)
            if runner is None:
                scenario_err_rv.set(f"Modelo no soportado: {model}")
                last_sig_rv.set(fut_signature())
                return

            payload = {
                "target_var": target,
                "predictors": list(exogs),
                "filters_by_var": filters,
                "train_ratio": 0.70,
                "auto_params": True,
                "return_df": True,
                "horizon": int(h),
                "scenario_mode": "future",
                "scenario_future_values": future_values,
                "scenario_overrides": [],
            }

            if model == "sarimax":
                payload.update({"s": 12})
            elif model == "xgboost":
                payload.update(
                    {"use_target_lags": True, "max_lag": 12, "recursive_forecast": True}
                )

            resp = runner(payload)

            parsed = _parse_forecast_response(resp, fallback_index=idx)
            if parsed is None:
                scenario_err_rv.set("El backend devolvió df vacío (resp['df']).")
                last_sig_rv.set(fut_signature())
                return

            df, y_col, future, h2, pred_vals, pred_series = parsed

            fig = plot_predictions(
                df=df,
                pred=pred_series,
                title=(
                    "Escenario futuro (SARIMAX)"
                    if model == "sarimax"
                    else "Escenario futuro (XGBoost)"
                ),
                ylabel="Valores",
                xlabel="Fecha",
                column_y=y_col,
                periodos_a_predecir=h2,
                holidays_col=None,
            )

            pred_df = _build_pred_df(future, pred_vals, date_fmt="%d-%m-%Y")

            scenario_res_rv.set(
                {
                    "model": model,
                    "fig": fig,
                    "pred_df": pred_df,
                }
            )
            scenario_err_rv.set(None)
            last_sig_rv.set(fut_signature())

        except Exception as e:
            scenario_err_rv.set(f"Fallo al calcular: {type(e).__name__}: {e}")
            last_sig_rv.set(fut_signature())
            return

    # ------------------------
    # Outputs
    # ------------------------
    @output
    @render.plot
    def esc_fut_plot():
        res = scenario_res_rv.get()
        return None if not res else res["fig"]

    @output
    @render.data_frame
    def esc_fut_table():
        res = scenario_res_rv.get()
        if not res or res.get("pred_df") is None:
            return render.DataGrid(pd.DataFrame())
        df = res["pred_df"].copy()
        if "Predicción" in df.columns:
            df["Predicción"] = pd.to_numeric(df["Predicción"], errors="coerce").round(4)
        return render.DataGrid(df)

    # ------------------------
    # UI Panel 4
    # ------------------------
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4 or scenario_type_rv.get() != "futuro":
            return ui.div()

        res = scenario_res_rv.get()
        err = scenario_err_rv.get()

        with reactive.isolate():
            _init_horizon = saved_horizon_rv.get()

        header = ui.card(
            ui.h3("Panel 4: Escenarios futuros", style="margin:0; text-align:center;"),
            ui.tags.div(
                "1) Periodos · 2) Valores exógenas · 3) Modelo · 4) Calcular",
                style="color:#6b7280; margin-top:4px; text-align:center;",
            ),
            ui.tags.hr(style="margin:12px 0;"),
            ui.tags.div(
                ui.input_numeric(
                    "esc_fut_horizon",
                    "1) Periodos a predecir",
                    value=_init_horizon,
                    min=1,
                    max=60,
                    step=1,
                ),
                style="max-width:320px; margin:0 auto;",
            ),
            style="padding:14px; border-radius:14px;",
        )

        model_box = ui.card(
            ui.tags.div(
                ui.input_radio_buttons(
                    "esc_fut_model",
                    "3) Modelo",
                    choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                    selected=fut_model(),
                    inline=True,
                ),
                style="display:flex; justify-content:center;",
            ),
            ui.tags.div(
                ui.input_action_button(
                    "esc_fut_calc", "Calcular", class_="btn-primary"
                ),
                style="margin-top:10px; display:flex; justify-content:center;",
            ),
            style="padding:14px; border-radius:14px; margin-top:12px;",
        )

        status = ui.div()
        if err:
            status = ui.tags.div(
                ui.tags.b("Error: "),
                ui.tags.span(err),
                style=(
                    "margin-top:10px; padding:10px 12px; border:1px solid #fecaca; "
                    "border-radius:12px; background:#fef2f2; color:#991b1b;"
                ),
            )
        elif res is None:
            status = ui.tags.div(
                ui.tags.b("Estado: "),
                ui.tags.span(
                    "rellena la tabla y pulsa «Calcular».", style="color:#6b7280;"
                ),
                style=(
                    "margin-top:10px; padding:10px 12px; border:1px dashed #d1d5db; "
                    "border-radius:12px; background:#fafafa;"
                ),
            )

        outputs = ui.div()
        if res is not None:
            outputs = ui.tags.div(
                ui.card(
                    ui.h5("Gráfico", style="margin:0 0 8px 0;"),
                    ui.output_plot("esc_fut_plot", width="100%", height="420px"),
                    style="padding:12px; border-radius:14px; flex:2 1 640px; min-width:520px;",
                ),
                ui.card(
                    ui.h5("Valores predichos", style="margin:0 0 8px 0;"),
                    ui.tags.div(
                        ui.output_data_frame("esc_fut_table"),
                        style="max-height:420px; overflow:auto;",
                    ),
                    style="padding:12px; border-radius:14px; flex:1 1 420px; min-width:340px;",
                ),
                style="display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; margin-top:12px;",
            )

        footer = ui.tags.div(
            ui.input_action_button("esc_prev_4", "← Anterior"),
            style="margin-top:12px;",
        )

        return ui.div(
            PANEL_STYLES,
            header,
            ui.output_ui("esc_future_exog_selector"),
            ui.output_ui("esc_future_exog_table"),
            model_box,
            status,
            outputs,
            footer,
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _go_step_3_from_4():
        
        scenario_res_rv.set(None)
        scenario_err_rv.set(None)
        saved_horizon_rv.set(2)
        saved_fut_cell_values_rv.set({})
        current_step.set(3)

    # =====================================================================
    # Escenarios PASADOS (placeholder — por desarrollar)
    # =====================================================================
    @output
    @render.ui
    def step_panel_pasado():
        if scenario_type_rv.get() != "pasado" or current_step.get() < 1:
            return ui.div()

        return ui.div(
            PANEL_STYLES,
            ui.card(
                ui.tags.div(
                    ui.tags.div(
                        "\U0001f6a7", style="font-size:3rem; margin-bottom:12px;"
                    ),
                    ui.h3("Escenarios Pasados", style="margin:0 0 8px 0;"),
                    ui.tags.p(
                        "Este módulo está en desarrollo. Aquí podrás modificar valores "
                        "históricos de las exógenas y observar cómo habría cambiado la predicción.",
                        style="color:#475569; max-width:480px; margin:0 auto;",
                    ),
                    style="text-align:center; padding:40px 20px;",
                ),
                ui.tags.div(
                    ui.input_action_button("esc_prev_pasado", "← Volver al selector"),
                    style="display:flex; justify-content:center; padding-bottom:20px;",
                ),
                style="border-radius:14px; max-width:600px; margin:24px auto;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_pasado)
    def _go_step_0_from_pasado():
        scenario_type_rv.set(None)
        current_step.set(0)
