import pandas as pd
from shiny import ui, reactive, render, module
from front.utils.back_api_wrappers import sarimax_run
from front.utils.back_api_wrappers import xgboost_run
from front.utils.back_api_wrappers import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
)
from back.models.utils.models_graph import plot_predictions

from front.utils.utils import (
    _to_date,
    diff_en_temporalidad,
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
            ui.h3("Panel 1: Seleccionar variable objetivo"),
            ui.p("Seleccione una única variable."),
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
        selected = []
        for var_id, name in predictor_pairs():
            if var_id in input and input[var_id]():
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

        target_start, target_end = cache.get_date_range(target_var)
        if target_end is None or tgt_temp is None:
            return None

        preds = selected_predictors()
        if not preds:
            return None  # si no hay predictoras seleccionadas

        # Para varias predictoras: nos quedamos con el END más pequeño
        min_pred_end = None

        for p in preds:
            p_meta = cache.get_meta(p) or {}
            compat, _reason = compatibilidad_con_objetivo(
                predictor_name=p,
                predictor_meta=p_meta,
                target_name=target_var,
                target_meta=target_meta,
                target_start=target_start,
                target_end=target_end,
                cache=cache,
            )
            # Si alguna seleccionada no es compatible, no podemos garantizar el horizonte común
            if not compat:
                return 0

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
    # Helpers (puros) Panel 4
    # ------------------------
    MODEL_RUNNERS = {
        "sarimax": sarimax_run,
        "xgboost": xgboost_run,
    }

    def _build_payload(model: str, target: str, predictors_used: list[str], filters: dict, horizon: int) -> dict:
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
            base.update({
                "use_target_lags": True,
                "max_lag": 12,
                "recursive_forecast": True,
            })
        return base

    def _parse_forecast_response(resp: dict):
        df = pd.DataFrame(resp.get("df") or [])
        if df.empty:
            return None

        y_col = resp["y_col"]
        n_obs = int(resp["n_obs"])
        h = int(resp["horizon"])

        future = df.iloc[n_obs:n_obs + h]
        pred_vals = resp["y_forecast"]
        pred_series = pd.Series(pred_vals, index=future.index, name="Prediction")
        return df, y_col, future, h, pred_vals, pred_series

    def _build_pred_df(future: pd.DataFrame, pred_vals, date_fmt: str = "%d-%m-%Y") -> pd.DataFrame:
        # Repite la lógica de plot_predictions: anio/mes(/dia) -> fechas reales
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
        pred_df["Fecha"] = pd.to_datetime(pred_df["Fecha"], errors="coerce").dt.strftime(date_fmt)
        return pred_df

    def _pack_result(model: str, resp: dict, fig, pred_df: pd.DataFrame, predictors_used: list[str], h: int) -> dict:
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
            base.update({
                "order": resp.get("order"),
                "seasonal_order": resp.get("seasonal_order"),
            })
        elif model == "xgboost":
            base.update({
                "xgb_params": resp.get("xgb_params"),
                "feature_cols": resp.get("feature_cols"),
            })
        return base

    def _kpi_card(label: str, value: str):
        return ui.tags.div(
            ui.tags.div(_metric_label_with_info(label), style="font-size:12px; color:#6b7280; margin-bottom:4px;"),
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

    def _metric_info_tooltip(description: str):
        return ui.tooltip(
            ui.tags.span(ui.HTML(ICON_SVG_INFO), style="display:inline-flex; cursor:help;"),
            description,
        )

    def _metric_label_with_info(label: str):
        descriptions = {
            "MAPE": "Error porcentual absoluto medio (en %).",
            "RMSE": "Raíz del error cuadrático medio (penaliza más los errores grandes).",
            "MAE": "Error absoluto medio (promedio del error en la escala original).",
        }
        return ui.tags.div(
            ui.tags.span(label),
            _metric_info_tooltip(descriptions.get(label, "Métrica de error del modelo.")),
            style="display:flex; align-items:center; gap:6px;",
        )

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

    # ------------------------
    # Cálculo bajo demanda: SOLO al pulsar "Calcula predicción"
    # ------------------------
    @reactive.effect
    @reactive.event(input.calc_pred)
    def _compute_prediction_on_click():
        if current_step.get() != 4:
            return
        if input.calc_pred() == 0:
            return
        if "model_exogs" not in input:
            pred_results_rv.set(None)
            return

        horizon = pred_horizon()
        if horizon < 1:
            pred_results_rv.set(None)
            last_sig_rv.set(pred_signature())
            return

        model = selected_model()
        predictors_used = exog_selected()
        target = target_var_rv.get()
        filters = selected_filters_by_var()

        runner = MODEL_RUNNERS.get(model)
        if runner is None:
            pred_results_rv.set(None)
            last_sig_rv.set(pred_signature())
            return

        payload = _build_payload(model, target, predictors_used, filters, horizon)
        resp = runner(payload)

        parsed = _parse_forecast_response(resp)
        if parsed is None:
            pred_results_rv.set(None)
            last_sig_rv.set(pred_signature())
            return

        df, y_col, future, h, pred_vals, pred_series = parsed
        pred_df = _build_pred_df(future, pred_vals, date_fmt="%d-%m-%Y")

        fig = plot_predictions(
            df=df,
            pred=pred_series,
            title=("Predicciones SARIMAX" if model == "sarimax" else "Predicciones XGBoost"),
            ylabel="Valores",
            xlabel="Fecha",
            column_y=y_col,
            periodos_a_predecir=h,
            holidays_col=None,
        )

        pred_results_rv.set(_pack_result(model, resp, fig, pred_df, predictors_used, h))
        last_sig_rv.set(pred_signature())

    # ------------------------
    # Outputs (usa el almacén, no calcula)
    # ------------------------
    @output
    @render.plot
    def model_plot():
        res = pred_results_rv.get()
        if not res:
            return None
        return res["fig"]

    @output
    @render.data_frame
    def pred_table():
        res = pred_results_rv.get()
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
                ui.h3("Panel 4: Modelo y exógenas", style="margin:0; text-align:center;"),
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
                        ui.tags.span("— (selecciona exógenas compatibles para habilitar el horizonte)"),
                        style="margin-top: 8px; color:#6b7280; text-align:center;",
                    )
                ),
                style="margin-top:10px;",
            ),

            # BOTÓN debajo del slider, centrado
            ui.tags.div(
                ui.input_action_button("calc_pred", "Calcula predicción", class_="btn-primary"),
                style="margin-top: 10px; display:flex; justify-content:center;",
            ),

            style="padding: 14px; border-radius: 14px;",
        )

        footer = ui.tags.div(
            ui.input_action_button("btn_prev_4", "← Anterior"),
            style="margin-top: 12px;",
        )

        # Estado sin resultados
        if res is None:
            return ui.div(
                PANEL_STYLES,
                header,
                ui.tags.div(
                    ui.tags.span("Estado: ", style="font-weight:600;"),
                    ui.tags.span("listo para calcular. Pulsa «Calcula predicción».", style="color:#6b7280;"),
                    style=(
                        "margin-top: 10px; padding: 10px 12px; border: 1px dashed #d1d5db; "
                        "border-radius: 12px; background:#fafafa;"
                    ),
                ),
                footer,
            )

        # Resultados
        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]


        kpis = ui.tags.div(
            _kpi_card("MAPE", f"{mape:.2f}%"),
            _kpi_card("RMSE", f"{rmse:.2f}"),
            _kpi_card("MAE", f"{mae:.2f}"),
            style="display:flex; gap:12px; flex-wrap:wrap; margin-top: 10px;",
        )

        kpis_title = ui.tags.div(
            ui.tags.span("Métricas del modelo", style="font-weight:600;"),
            _metrics_info_tooltip(),
            style="display:flex; align-items:center; gap:6px; margin-top:10px;",
        )

        exogs_line = ui.tags.div(
            ui.tags.span("Exógenas activas: ", style="font-weight:600; margin-right:6px;"),
            _pill(", ".join(res["predictors_used"]) if res["predictors_used"] else "Ninguna"),
            style="margin-top: 10px;",
        )

        # Layout plot + tabla (responsive)
        body = ui.tags.div(
            ui.card(
                ui.h5("Gráfico", style="margin:0 0 8px 0;"),
                ui.output_plot("model_plot", width="100%", height="420px"),
                style=(
                    "padding: 12px; border-radius: 14px;"
                    "flex: 2 1 640px; min-width: 520px;"
                ),
            ),
            ui.card(
                ui.h5("Valores predichos", style="margin:0 0 8px 0;"),
                ui.tags.div(  # wrapper para controlar altura/scroll si crece
                    ui.output_data_frame("pred_table"),
                    style="max-height: 420px; overflow:auto;",
                ),
                style=(
                    "padding: 12px; border-radius: 14px;"
                    "flex: 1 1 420px; min-width: 340px;"
                ),
            ),
            style=(
                "display:flex; gap:12px; flex-wrap:wrap;"
                "align-items:flex-start;"          # <- evita que la tabla se estire en alto
                "margin-top: 12px;"
            ),
        )


        return ui.div(
            PANEL_STYLES,
            header,
            exogs_line,
            kpis_title,
            kpis,
            body,
            footer,
        )

    @reactive.Effect
    @reactive.event(input.btn_prev_4)
    def _go_step_3_from_4():
        current_step.set(3)
