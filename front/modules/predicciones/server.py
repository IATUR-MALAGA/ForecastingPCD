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
    build_name_to_table,
    PrediccionesCache,
    compatibilidad_con_objetivo,
    panel_styles,
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

                reason_ui = (
                    ui.tags.div(
                        ui.tags.strong("Motivo de incompatibilidad: "),
                        ui.tags.span(reason, class_="reason-text"),
                        class_="compat-reason-box",
                    )
                    if (not compat and reason)
                    else None
                )
                temporalidad = _fmt(meta.get("temporalidad"))
                granularidad = _fmt(meta.get("granularidad"))
                unidad_medida = _fmt(meta.get("unidad_medida"))
                fuente = _fmt(meta.get("fuente"))
                descripcion = _fmt(meta.get("descripcion"))

                var_blocks.append(
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_checkbox(var_id, name, value=False),
                            badge,
                            style="display: flex; align-items: baseline; gap: 6px;",
                        ),
                        ui.tags.details(
                            ui.tags.summary("Ver más", style="cursor: pointer; margin-top: 6px; font-size: 0.9em; color: #666;"),
                            ui.tags.div(
                                reason_ui if reason_ui else ui.div(),
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
                ui.input_action_button("btn_prev_2", "← Anterior"),
                ui.input_action_button("btn_next_2", "Siguiente →"),
                style="margin-top: 12px; display: flex; gap: 8px;",
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
    def vars_to_config() -> list[dict]:
        """
        Devuelve una lista de dicts con:
        - pretty: nombre bonito (lo que muestras)
        - table:  nombre real de la tabla (lo que consultas)
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

            out.append({"pretty": pretty, "table": table})

        return out
    
    @reactive.Calc
    def selected_filters_by_var() -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}

        for item in vars_to_config():
            pretty = item["pretty"]
            table = item["table"]

            filtros = cache.get_filters(table)  
            selected_list: list[dict] = []

            for f in filtros:
                t = f["table"]
                col = f["col"]

                input_id = _stable_id("flt", f"{t}__{col}")

                vals = input[input_id]() if (input_id in input) else None

                if vals:
                    selected_list.append({
                        "table": t,
                        "col": col,
                        "values": list(vals),
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

        for item in vars_sel:
            pretty = item["pretty"]   
            table = item["table"]    

            filtros = cache.get_filters(table)

            if not filtros:
                body = ui.p("Sin filtros configurados en tbl_admin_filtros para esta variable/tabla.")
            else:
                controls = []
                for f in filtros:
                    t = f["table"]
                    col = f["col"]
                    label = f.get("label") or col  


                    cols_set = cache.get_table_cols("IA", t)
                    if col not in cols_set:
                        controls.append(
                            ui.tags.div(
                                ui.tags.b(f"{col}"),
                                ui.tags.span(
                                    f"  (⚠ no existe en IA.{t})",
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

                body = ui.div(*controls)

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


    
    sarimax_results_rv = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_update_sarimax)
    def _run_sarimax_model():
        # Evitar ejecución inicial o reseteos a 0
        if input.btn_update_sarimax() == 0:
            return

        with reactive.isolate():
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

            try:
                resp = sarimax_run(payload)
                
                df = pd.DataFrame(resp["df"]) if resp.get("df") else None
                if df is None or df.empty:
                    sarimax_results_rv.set(None)
                    return

                y_col = resp["y_col"]
                n_train = resp["n_train"]
                n_test = resp["n_test"]

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

                sarimax_results_rv.set({
                    "mape": resp["mape"],
                    "rmse": resp["rmse"],
                    "mae": resp["mae"],
                    "fig": fig,
                    "order": resp["order"],
                    "seasonal_order": resp["seasonal_order"],
                    "predictors_used": predictors_used,
                })
            except Exception as e:
                print(f"Error executing SARIMAX: {e}")
                sarimax_results_rv.set(None)


    # Limpiar resultados si cambian los inputs (para obligar a recalcular)
    @reactive.Effect
    @reactive.event(input.sarimax_exogs)
    def _clear_results_on_change():
        # Solo limpiar si ya hay resultados mostrados, para evitar flicker inicial
        if sarimax_results_rv.get() is not None:
             sarimax_results_rv.set(None)


    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        with reactive.isolate():
            choices = exog_choices()
            
            # Usar exog_selected() para garantizar consistencia con choices y evitar None
            selected = exog_selected()


        res = sarimax_results_rv.get()
        
        # Botón de actualizar
        update_btn = ui.div(
            ui.input_action_button("btn_update_sarimax", "Actualizar Modelo (Recalcular)", class_="btn-success"),
            style="margin: 12px 0;"
        )

        if res is None:
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 4: Resultados del modelo SARIMAX"),
                ui.p("Configura las variables exógenas y pulsa Actualizar."),
                ui.input_checkbox_group(
                    "sarimax_exogs",
                    "Variables exógenas (activar/desactivar)",
                    choices=choices,
                    selected=selected,
                ),
                update_btn,
                ui.p("Aún no hay resultados (pulsa Actualizar)."),
                ui.div(
                    ui.input_action_button("btn_prev_4", "← Anterior"),
                    style="margin-top: 12px;",
                ),
            )

        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: SARIMAX — activar/desactivar exógenas"),
            ui.p("Marca qué variables exógenas quieres usar y pulsa Actualizar."),

            ui.input_checkbox_group(
                "sarimax_exogs",
                "Variables exógenas (activar/desactivar)",
                choices=choices,
                selected=selected,  
            ),
            
            update_btn,

            ui.tags.div(
                ui.tags.span("Exógenas activas (en modelo actual): ", style="font-weight:600; margin-right:6px;"),
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

            ui.div(
                ui.input_action_button("btn_prev_4", "← Anterior"),
                style="margin-top: 12px;",
            ),
        )


    @output
    @render.plot
    def sarimax_plot():
        res = sarimax_results_rv.get()
        if res is None:
            return None
        return res["fig"]

    @reactive.Effect
    @reactive.event(input.btn_prev_4)
    def _go_step_3_from_4():
        current_step.set(3)

            



