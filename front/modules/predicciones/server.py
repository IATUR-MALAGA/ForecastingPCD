import pandas as pd
from shiny import ui, reactive, render, module
from front.utils.back_api_wrappers import sarimax_run
from front.utils.back_api_wrappers import xgboost_run
from front.utils.back_api_wrappers import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
)
from back.models.SARIMAX.sarimax_graph import plot_predictions as plot_sarimax
from back.models.XGBoost.xgboost_graph import plot_predictions as plot_xgb

from front.utils.utils import (
    slug as _slug,  
    stable_id as _stable_id,
    group_by_category as _group_by_category,
    fmt as _fmt,
    build_name_to_table,
    PrediccionesCache,
    compatibilidad_con_objetivo,
    panel_styles,
    ICON_SVG_INFO,
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

                var_blocks.append(
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_checkbox(var_id, name, value=False),
                            badge,
                            info_icon,
                            style="display: flex; align-items: baseline; gap: 6px;",
                        ),
                        ui.tags.details(
                            ui.tags.summary("Ver más", style="cursor: pointer; margin-top: 6px; font-size: 0.9em; color: #666;"),
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
# Panel 4: Play with Model and Variables (CON BOTÓN: NO calcula hasta pulsar)
##########################################################################################

    # ------------------------
    # Inputs auxiliares
    # ------------------------
    @reactive.calc
    def exog_choices():
        return list(predictors_rv.get() or [])

    @reactive.calc
    def selected_model():
        # Evita primer ciclo cuando el input aún no existe
        if "model_choice" not in input:
            return "sarimax"  # default
        return input.model_choice() or "sarimax"

    @reactive.calc
    def exog_selected():
        choices = exog_choices()

        # Espera a que el input exista para no disparar dobles llamadas
        if "model_exogs" not in input:
            return choices

        sel = input.model_exogs() or []
        sel = list(sel)
        sel = [s for s in sel if s in choices]
        return sel

    # ------------------------
    # Almacén de resultados (solo se llena al pulsar botón)
    # ------------------------
    pred_results_rv = reactive.Value(None)
    last_sig_rv = reactive.Value(None)

    @reactive.calc
    def pred_signature():
        """
        Firma del estado de inputs que afecta a la predicción.
        Si cambia, invalidamos resultados para no mostrar predicciones antiguas.
        """
        if current_step.get() != 4:
            return None

        model = selected_model()
        exogs = tuple(exog_selected() or [])
        target = target_var_rv.get()
        filters = selected_filters_by_var()

        # filters puede ser dict/list -> lo convertimos a str estable
        return (model, target, exogs, repr(filters))

    @reactive.effect
    def _invalidate_prediction_when_inputs_change():
        sig = pred_signature()
        if sig is None:
            return

        last = last_sig_rv.get()
        if last is not None and sig != last:
            pred_results_rv.set(None)

    # ------------------------
    # Cálculo bajo demanda: SOLO al pulsar "Calcula predicción"
    # ------------------------
    @reactive.effect
    @reactive.event(input.calc_pred)
    def _compute_prediction_on_click():
        if current_step.get() != 4:
            return

        # Evita ejecutar si es la primera vez (0 clicks)
        if input.calc_pred() == 0:
            return

        # Evita primer ciclo sin inputs montados
        if "model_exogs" not in input:
            pred_results_rv.set(None)
            return

        model = selected_model()
        predictors_used = exog_selected()

        # ---------- SARIMAX ----------
        if model == "sarimax":
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
                pred_results_rv.set(None)
                last_sig_rv.set(pred_signature())
                return

            y_col = resp["y_col"]
            n_train = resp["n_train"]
            n_test = resp["n_test"]

            test = df.iloc[n_train:n_train + n_test]

            pred_vals = resp["y_pred"]
            pred_test = pd.Series(pred_vals, index=test.index, name="Prediction")

            fig = plot_sarimax(
                df=df,
                pred=pred_test,
                title="Predicciones SARIMAX",
                ylabel="Valores",
                xlabel="Fecha",
                column_y=y_col,
                periodos_a_predecir=n_test,
                holidays_col=None
            )

            pred_results_rv.set({
                "model": "sarimax",
                "mape": resp["mape"],
                "rmse": resp["rmse"],
                "mae": resp["mae"],
                "fig": fig,
                "order": resp["order"],
                "seasonal_order": resp["seasonal_order"],
                "predictors_used": predictors_used,
            })

        # ---------- XGBOOST ----------
        else:
            payload = {
                "target_var": target_var_rv.get(),
                "predictors": predictors_used,
                "filters_by_var": selected_filters_by_var(),
                "train_ratio": 0.70,

                # XGBoost
                "auto_params": True,
                "use_target_lags": True,
                "max_lag": 12,
                "recursive_forecast": True,

                "return_df": True
            }

            resp = xgboost_run(payload)

            df = pd.DataFrame(resp["df"]) if resp.get("df") else None
            if df is None or df.empty:
                pred_results_rv.set(None)
                last_sig_rv.set(pred_signature())
                return

            y_col = resp["y_col"]
            n_train = resp["n_train"]
            n_test = resp["n_test"]

            test = df.iloc[n_train:n_train + n_test]

            pred_vals = resp["y_pred"]
            pred_test = pd.Series(pred_vals, index=test.index, name="Prediction")

            fig = plot_xgb(
                df=df,
                pred=pred_test,
                title="Predicciones XGBoost",
                ylabel="Valores",
                xlabel="Fecha",
                column_y=y_col,
                periodos_a_predecir=n_test,
                holidays_col=None
            )

            pred_results_rv.set({
                "model": "xgboost",
                "mape": resp["mape"],
                "rmse": resp["rmse"],
                "mae": resp["mae"],
                "fig": fig,
                "xgb_params": resp.get("xgb_params"),
                "feature_cols": resp.get("feature_cols"),
                "predictors_used": predictors_used,
            })

        # Guarda la firma del estado con el que se calculó (para invalidar si cambian inputs)
        last_sig_rv.set(pred_signature())

    # ------------------------
    # Plot unificado (usa el almacén, no calcula)
    # ------------------------
    @output
    @render.plot
    def model_plot():
        res = pred_results_rv.get()
        if not res:
            return None
        return res["fig"]

    # ------------------------
    # UI Panel 4 (incluye botón y muestra resultados si existen)
    # ------------------------
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        choices = exog_choices()
        selected = exog_selected()
        model = selected_model()

        # OJO: ahora NO usamos selected_results(); usamos el almacén
        res = pred_results_rv.get()

        title = "Panel 4: Selección de modelo + exógenas"
        subtitle = "Elige el modelo y qué exógenas usar. La predicción SOLO se ejecuta cuando pulsas el botón."

        # Cabecera + inputs siempre visibles
        header = ui.div(
            PANEL_STYLES,
            ui.h3(title),
            ui.p(subtitle),

            ui.input_radio_buttons(
                "model_choice",
                "Modelo",
                choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                selected=model,
                inline=True,
            ),

            ui.input_checkbox_group(
                "model_exogs",
                "Variables exógenas (activar/desactivar)",
                choices=choices,
                selected=selected,
            ),

            ui.input_action_button(
                "calc_pred",
                "Calcula predicción",
                class_="btn-primary",
            ),
        )
        footer = ui.div(
                ui.input_action_button("btn_prev_4", "← Anterior"),
                style="margin-top: 12px;",
            )

        if res is None:
            return ui.div(
                header,
                ui.p("Pulsa «Calcula predicción» para ejecutar el modelo con la selección actual."),
                footer
            )

        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]

        # Texto adicional según modelo
        extra = ui.div()
        if res.get("model") == "sarimax":
            extra = ui.tags.div(
                ui.tags.div(f"order: {res.get('order')}"),
                ui.tags.div(f"seasonal_order: {res.get('seasonal_order')}"),
                style="margin: 8px 0;"
            )
        elif res.get("model") == "xgboost":
            extra = ui.tags.div(
                ui.tags.div(f"params: {res.get('xgb_params')}"),
                style="margin: 8px 0;"
            )

        return ui.div(
            header,

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

            extra,

            ui.output_plot("model_plot", width="100%", height="420px"),

            footer,
        )

    @reactive.Effect
    @reactive.event(input.btn_prev_4)
    def _go_step_3_from_4():
        current_step.set(3)