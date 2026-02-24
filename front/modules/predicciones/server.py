import pandas as pd
from shiny import ui, reactive, render, module
from front.utils.back_api_wrappers import sarimax_run

from front.utils.back_api_wrappers import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
)


from back.models.SARIMAX.sarimax_model import best_sarimax_params, create_sarimax_model, predict_sarimax
from back.models.SARIMAX.sarimax_statistics import compute_metrics
from back.models.SARIMAX.sarimax_graph import plot_predictions
from front.utils.utils import (
    slug as _slug,  
    stable_id as _stable_id,
    group_by_category as _group_by_category,
    fmt as _fmt,
    fmt_date_by_temporality as _fmt_date_temp,
    build_name_to_table,
    PrediccionesCache,
    compatibilidad_con_objetivo,
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
    def step_panel_1():
        if current_step.get() != 1:
            return ui.div()

        grouped = _group_by_category(catalog_entries)
        all_names = [n for names in grouped.values() for n in names]

        if target_var_rv.get() is None and all_names:
            target_var_rv.set(all_names[0])

        selected = target_var_rv.get()

        panels = []
        for cat, names in grouped.items():
            btns = []
            for name in names:
                btn_id = _stable_id("pick_target", name)

                if btn_id not in _registered_pick_handlers:
                    _registered_pick_handlers.add(btn_id)

                    @reactive.Effect
                    @reactive.event(input[btn_id])
                    def _on_pick_target(_name=name):
                        target_var_rv.set(_name)

                btns.append(
                    ui.input_action_button(
                        btn_id,
                        name,
                        class_=("var-pick is-selected" if name == selected else "var-pick"),
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
            ui.h3("Panel 1: seleccionar variable objetivo"),
            ui.p("Seleccione una única variable (click)."),
            ui.div(
                ui.tags.span("Seleccionada: ", style="font-weight:600;"),
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

    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()

        target_var = target_var_rv.get()
        grouped = _group_by_category(catalog_entries, exclude_name=target_var)
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_start, target_end = cache.get_date_range(target_var) if target_var else (None, None)
        target_temp = _fmt(target_meta.get("temporalidad"))
        panels = []
        for cat, names in grouped.items():
            var_blocks = []

            for name in names:
                var_id = _stable_id("pred", name)
                meta = cache.get_meta(name)
                compat, reason = compatibilidad_con_objetivo(
                    predictor_name=name,
                    predictor_meta=meta,
                    target_name=target_var,
                    target_meta=target_meta,
                    target_start=target_start,
                    target_end=target_end,
                    cache=cache,
                )

                badge = ui.tags.span(
                    "Compatible" if compat else "No compatible",
                    class_=("compat-badge compat-yes" if compat else "compat-badge compat-no"),
                )

                reason_ui = ui.tags.div(reason, class_="compat-reason") if (not compat and reason) else ui.div()
                temporalidad = _fmt(meta.get("temporalidad"))
                granularidad = _fmt(meta.get("granularidad"))
                unidad_medida = _fmt(meta.get("unidad_medida"))
                fuente = _fmt(meta.get("fuente"))
                descripcion = _fmt(meta.get("descripcion"))

                var_blocks.append(
                    ui.tags.div(
                        ui.input_checkbox(var_id, name, value=False),
                        ui.tags.div(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.span("Compatibilidad", class_="var-meta-key"),
                                        ui.tags.div(badge, reason_ui),
                                    ),
                                    ui.tags.div(ui.tags.span("Temporalidad", class_="var-meta-key"), temporalidad),
                                    ui.tags.div(ui.tags.span("Granularidad", class_="var-meta-key"), granularidad),
                                    ui.tags.div(ui.tags.span("Unidad medida", class_="var-meta-key"), unidad_medida),
                                    ui.tags.div(ui.tags.span("Fuente", class_="var-meta-key"), fuente),
                                    class_="var-meta-grid",
                                ),

                            ui.tags.div(
                                ui.tags.span("Descripción", class_="var-meta-key"),
                                ui.tags.div(descripcion, class_="var-desc"),
                                style="margin-top:6px;",
                            ),
                            class_="var-meta",
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
            ui.h3("Panel 2: seleccionar variables predictoras"),
            ui.p("Seleccione una o varias variables (checkbox)."),
            ui.div(
                ui.tags.div(
                    ui.tags.span("Objetivo actual: ", style="font-weight:600;"),
                    ui.tags.span(target_var or "—"),
                ),
                ui.tags.div(
                    ui.tags.span("Temporalidad: ", style="font-weight:600;"),
                    ui.tags.span(target_temp),
                    ui.tags.span("  ·  Rango: ", style="font-weight:600; margin-left:10px;"),
                    ui.tags.span(f"{_fmt(target_start)} → {_fmt(target_end)}"),
                    style="margin-top:4px;",
                ),
                class_="selection-pill",
            ),

            ui.accordion(*panels, id="acc_predictors", open=True, multiple=True),
            ui.div(
                ui.input_action_button("btn_next_2", "Siguiente →"),
                style="margin-top: 12px;",
            ),
        )

    @reactive.Calc
    def selected_predictors():
        selected = []
        for var_id, name in predictor_pairs():
            if var_id in input and input[var_id]():
                selected.append(name)
        return sorted(set(selected))

    @reactive.Effect
    def _sync_predictors_rv():
        predictors_rv.set(selected_predictors())

    @reactive.Effect
    @reactive.event(input.btn_next_2)
    def _go_step_3():
        current_step.set(3)

    ##########################################################################################
    # Panel 3: Config Variables
    ##########################################################################################
    @reactive.Calc
    def target_selected_range() -> tuple:
        """Obtiene el rango temporal seleccionado por el usuario en el target"""
        target = target_var_rv.get()
        if not target:
            return (None, None)
        
        table = name_to_table.get(target)
        if not table:
            rows = get_tableName_for_variable(target) or []
            table = (rows[0].get("nombre_tabla") if rows else target)
        
        filtros = cache.get_filters(table)
        temp = detect_temporal_filters(filtros)
        
        # Si tiene mes o día, usar el date_range selector
        if temp["mes"] or temp["dia"]:
            date_input_id = _stable_id("flt", f"{table}__date_range")
            if date_input_id in input:
                date_range = input[date_input_id]()
                if date_range and len(date_range) == 2:
                    return (date_range[0], date_range[1])
        
        # Si tiene año, usar el selector de años
        elif temp["anio"]:
            anio_input_id = _stable_id("flt", f"{table}__anio")
            if anio_input_id in input:
                vals = input[anio_input_id]()
                if vals:
                    years = sorted([int(v) for v in vals])
                    return (f"{years[0]}-01-01", f"{years[-1]}-12-31")
        
        # Si no hay selección, devolver None en lugar del rango completo
        # Esto evita sobrescribir la selección del usuario
        return (None, None)
    
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
                table = (rows[0].get("nombre_tabla") if rows else pretty)

            out.append({
                "pretty": pretty, 
                "table": table,
                "is_target": (pretty == target)
            })

        return out
    
    @reactive.Calc
    def selected_filters_by_var() -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        target_var = target_var_rv.get()
        
        # Primero capturamos los filtros temporales del target
        target_temporal_filters = []

        for item in vars_to_config():
            pretty = item["pretty"]
            table = item["table"]
            is_target = item.get("is_target", False)

            filtros = cache.get_filters(table)
            selected_list: list[dict] = []
            
            temp = detect_temporal_filters(filtros)
            
            # Solo capturar filtros temporales si ES el target
            if is_target and (temp["mes"] or temp["dia"]):
                date_input_id = _stable_id("flt", f"{table}__date_range")
                if date_input_id in input:
                    date_range = input[date_input_id]()
                    if date_range:
                        temporal_filters = process_date_range_filters(date_range, filtros, table)
                        selected_list.extend(temporal_filters)
                        # Guardar para aplicar a las exógenas
                        target_temporal_filters = temporal_filters
            
            elif is_target and temp["anio"]:
                anio_input_id = _stable_id("flt", f"{table}__anio")
                if anio_input_id in input:
                    vals = input[anio_input_id]()
                    if vals:
                        temporal_filter = {
                            "table": table,
                            "col": temp["anio"]["col"],
                            "values": list(vals) if isinstance(vals, (list, tuple)) else [str(vals)]
                        }
                        selected_list.append(temporal_filter)
                        # Guardar para aplicar a las exógenas
                        target_temporal_filters = [temporal_filter]
            
            # Si es exógena, aplicar los filtros temporales del target
            if not is_target and target_temporal_filters:
                # Adaptar los filtros del target a esta tabla exógena
                for tf in target_temporal_filters:
                    selected_list.append({
                        "table": table,  # Cambiar a la tabla de la exógena
                        "col": tf["col"],
                        "values": tf["values"]
                    })
            
            # Capturar filtros no temporales para todas las variables
            for f in filtros:
                col_lower = f["col"].lower().strip()
                if col_lower in ("anio", "año", "ano", "mes", "dia", "día"):
                    continue  
                
                input_id = _stable_id("flt", f"{f['table']}__{f['col']}")
                if input_id in input:
                    vals = input[input_id]()
                    if vals:
                        selected_list.append({
                            "table": f["table"],
                            "col": f["col"],
                            "values": list(vals) if isinstance(vals, (list, tuple)) else [str(vals)]
                        })

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

        panels = []
        
        # Obtener el rango SELECCIONADO del target para mostrarlo en las exógenas
        target_var = target_var_rv.get()
        target_start, target_end = target_selected_range()
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_temporality = target_meta.get("temporalidad")
        
        # Si no hay selección, usar el rango completo disponible solo para mostrar
        display_start = target_start if target_start else (cache.get_date_range(target_var)[0] if target_var else None)
        display_end = target_end if target_end else (cache.get_date_range(target_var)[1] if target_var else None)
        
        # Formatear las fechas según la temporalidad
        target_start_fmt = _fmt_date_temp(display_start, target_temporality) if display_start else "—"
        target_end_fmt = _fmt_date_temp(display_end, target_temporality) if display_end else "—"
        
        # Mensaje sobre si está usando selección del usuario o rango completo
        range_status = "seleccionado" if target_start else "disponible (selecciona un rango en la variable objetivo)"

        for item in vars_sel:
            pretty = item["pretty"]   
            table = item["table"]
            is_target = item.get("is_target", False)

            filtros = cache.get_filters(table)
            
        
            start_date, end_date = cache.get_date_range(pretty)

            if not filtros:
                if is_target:
                    body = ui.p("Sin filtros configurados en tbl_admin_filtros para esta variable/tabla.")
                else:
                    # Exógena sin filtros: mostrar mensaje sobre el rango del target
                    body = ui.div(
                        ui.tags.div(
                            ui.tags.span("📌 Variable Exógena", style="font-weight:600; color:#6e7781;"),
                            style="margin-bottom:12px;"
                        ),
                        ui.tags.div(
                            "✓ Esta variable se ajustará automáticamente al rango temporal seleccionado en la variable objetivo.",
                            style="padding:8px; background-color:#dff6dd; border-left:3px solid #1a7f37; color:#1a7f37; border-radius:4px;"
                        ),
                        ui.tags.div(
                            ui.tags.span(f"Rango {range_status}: ", style="font-weight:500; margin-top:8px; display:inline-block;"),
                            ui.tags.span(f"{target_start_fmt} → {target_end_fmt}"),
                            style="margin-top:8px;"
                        )
                    )
            else:
                controls = []
                
                # Solo mostrar el calendar filter si ES el target
                if is_target:
                    calendar = create_calendar_filter(filtros, cache, _stable_id, start_date, end_date, input)
                    if calendar:
                        controls.append(calendar)
                else:
                    # Para las exógenas, mostrar mensaje informativo
                    controls.append(
                        ui.tags.div(
                            ui.tags.div(
                                ui.tags.span("📌 Variable Exógena", style="font-weight:600; color:#6e7781;"),
                                style="margin-bottom:12px;"
                            ),
                            ui.tags.div(
                                "✓ Esta variable se ajustará automáticamente al rango temporal seleccionado en la variable objetivo.",
                                style="padding:8px; background-color:#dff6dd; border-left:3px solid #1a7f37; color:#1a7f37; border-radius:4px; margin-bottom:12px;"
                            ),
                            ui.tags.div(
                                ui.tags.span(f"Rango {range_status}: ", style="font-weight:500;"),
                                ui.tags.span(f"{target_start_fmt} → {target_end_fmt}"),
                                style="margin-bottom:16px;"
                            ),
                            style="margin-bottom:16px;"
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

                    input_id = _stable_id("flt", f"{t}__{col}")
                    choices = cache.get_distinct("IA", t, col)

                    controls.append(
                        ui.tags.div(
                            ui.input_selectize(
                                input_id,
                                label=label,
                                choices=choices,
                                multiple=True,
                                options={
                                    "placeholder": "Selecciona uno o varios valores (vacío = sin filtro)",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            style="margin-bottom: 12px;",
                        )
                    )

                body = ui.div(*controls) if controls else ui.p("Sin filtros disponibles.")

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
            ui.p("Para cada variable, se muestran los filtros definidos en IA.tbl_admin_filtros."),
            ui.accordion(*panels, id="acc_filters", open=True, multiple=True),
            ui.div(
                ui.input_action_button("btn_next_3", "Siguiente →"),
                style="margin-top: 12px;",
        ),
    )
    @reactive.Effect
    @reactive.event(input.btn_next_3)
    def _go_step_4():
        current_step.set(4)



##########################################################################################
# Panel 4: Play with Model and Variables
##########################################################################################

    @reactive.calc
    def exog_choices():
        return list(predictors_rv.get() or [])

    @reactive.calc
    def exog_selected():
        choices = exog_choices()

        if "sarimax_exogs" not in input:
            return choices

        sel = input.sarimax_exogs() or []
        sel = list(sel)

        sel = [s for s in sel if s in choices]
        return sel


    @reactive.calc
    def sarimax_results():
        if current_step.get() != 4:
            return None

        predictors_used = exog_selected()

        payload = {
            "target_var": target_var_rv.get(),
            "predictors": predictors_used,                    
            "filters_by_var": selected_filters_by_var(),
            "train_ratio": 0.70,
            "auto_params": True,    
            "s": 12,
            "return_df": True
        }

        resp = sarimax_run(payload)

        df = pd.DataFrame(resp["df"]) if resp.get("df") else None
        if df is None or df.empty:
            return None

        y_col = resp["y_col"]
        n_train = resp["n_train"]
        n_test = resp["n_test"]

        train = df.iloc[:n_train]
        test = df.iloc[n_train:n_train + n_test]

        pred_vals = resp["y_pred"]
        pred_test = pd.Series(pred_vals, index=test.index, name="Prediction")

        fig = plot_predictions(
            df=df,
            pred=pred_test,
            title="Predicciones SARIMAX",
            ylabel="Valores",
            xlabel="Fecha",
            column_y=y_col,
            periodos_a_predecir=n_test,
            holidays_col=None
        )

        return {
            "mape": resp["mape"],
            "rmse": resp["rmse"],
            "mae": resp["mae"],
            "fig": fig,
            "order": resp["order"],
            "seasonal_order": resp["seasonal_order"],
            "predictors_used": predictors_used,
        }


    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        choices = exog_choices()
        selected = exog_selected()

        res = sarimax_results()
        if res is None:
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 4: Resultados del modelo SARIMAX"),
                ui.p("Configura las variables exógenas y se recalculará el modelo."),
                ui.input_checkbox_group(
                    "sarimax_exogs",
                    "Variables exógenas (activar/desactivar)",
                    choices=choices,
                    selected=selected,
                ),
                ui.p("Aún no hay resultados (df vacío o error)."),
            )

        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: SARIMAX — activar/desactivar exógenas"),
            ui.p("Marca qué variables exógenas quieres usar. Al cambiar, se recalcula el modelo."),

            ui.input_checkbox_group(
                "sarimax_exogs",
                "Variables exógenas (activar/desactivar)",
                choices=choices,
                selected=selected,  
            ),

            ui.tags.div(
                ui.tags.span("Exógenas activas: ", style="font-weight:600; margin-right:6px;"),
                ui.tags.span(", ".join(res["predictors_used"]) if res["predictors_used"] else "Ninguna"),
                style="margin: 10px 0;",
            ),

            ui.tags.div(
                ui.tags.div(
                    ui.tags.span("MAPE: ", style="font-weight:600; margin-right:6px;"),
                    ui.tags.span(f"{mape:.2f}%"),
                    style="margin-bottom:6px;"
                ),
                ui.tags.div(
                    ui.tags.span("RMSE: ", style="font-weight:600; margin-right:6px;"),
                    ui.tags.span(f"{rmse:.2f}"),
                    style="margin-bottom:6px;"
                ),
                ui.tags.div(
                    ui.tags.span("MAE: ", style="font-weight:600; margin-right:6px;"),
                    ui.tags.span(f"{mae:.2f}"),
                ),
                style="margin: 12px 0;"
            ),

            ui.output_plot("sarimax_plot", width="100%", height="420px"),
        )


    @output
    @render.plot
    def sarimax_plot():
        res = sarimax_results()
        if res is None:
            return None
        return res["fig"]

            



