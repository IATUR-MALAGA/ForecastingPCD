import pandas as pd
from shiny import module, reactive, render, ui

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
    _to_date,
    build_name_to_table,
    compatibilidad_con_objetivo,
    diff_en_temporalidad,
    fmt,
    group_by_category,
    panel_styles,
    slug as _slug,
    stable_id,
)


@module.server
def escenarios_server(input, output, session):
    current_step = reactive.Value(1)
    target_var_rv = reactive.Value(None)
    predictors_rv = reactive.Value([])
    result_rv = reactive.Value(None)

    catalog_entries = get_names_in_table_catalog() or []
    name_to_table = build_name_to_table(catalog_entries)
    cache = PrediccionesCache(name_to_table)
    PANEL_STYLES = panel_styles()
    _registered_pick_handlers: set[str] = set()

    MODEL_RUNNERS = {"sarimax": sarimax_run, "xgboost": xgboost_run}

    @output
    @render.ui
    def step_indicator():
        labels = ["Objetivo", "Predictoras", "Filtros", "Escenarios"]
        return ui.div(PANEL_STYLES, *[ui.tags.span(f"{i}. {lbl}", class_=("selection-pill var-pick is-selected" if i == current_step.get() else "selection-pill"), style="margin-right:6px;") for i, lbl in enumerate(labels, start=1)], style="margin:8px 0;")

    @output
    @render.ui
    def step_panel_1():
        if current_step.get() != 1:
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
                    def _on_pick(_name=name):
                        target_var_rv.set(_name)
                        result_rv.set(None)

                btns.append(ui.input_action_button(btn_id, name, class_=("var-pick is-selected" if selected == name else "var-pick")))
            panels.append(ui.accordion_panel(cat, ui.div(*btns, class_="var-list"), value=_slug(cat)))

        return ui.div(PANEL_STYLES, ui.h3("Panel 1: Seleccionar variable objetivo"), ui.accordion(*panels, id="esc_acc_target", open=True, multiple=True), ui.input_action_button("esc_next_1", "Siguiente →"))

    @reactive.Effect
    @reactive.event(input.esc_next_1)
    def _next1():
        current_step.set(2)

    @reactive.Calc
    def predictor_pairs():
        grouped = group_by_category(catalog_entries, exclude_name=target_var_rv.get())
        return [(stable_id("esc_pred", name), name) for _, names in grouped.items() for name in names]

    @reactive.Calc
    def selected_predictors():
        return sorted({name for var_id, name in predictor_pairs() if var_id in input and input[var_id]()})

    @reactive.Calc
    def max_num_predictions():
        target = target_var_rv.get()
        if not target:
            return 0
        target_meta = cache.get_meta(target) or {}
        tgt_temp = target_meta.get("temporalidad")
        target_start, target_end = cache.get_date_range(target)
        if not target_end or not tgt_temp:
            return 0
        preds = selected_predictors()
        if not preds:
            return 0
        min_end = None
        for p in preds:
            ok, _ = compatibilidad_con_objetivo(p, cache.get_meta(p), target, target_meta, target_start, target_end, cache)
            if not ok:
                return 0
            _, p_end = cache.get_date_range(p)
            if not p_end:
                return 0
            if min_end is None or _to_date(p_end) < _to_date(min_end):
                min_end = p_end
        n = diff_en_temporalidad(target_end, min_end, tgt_temp)
        return max(0, n or 0)

    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()
        target = target_var_rv.get()
        grouped = group_by_category(catalog_entries, exclude_name=target)
        target_meta = cache.get_meta(target) if target else {}
        ts, te = cache.get_date_range(target) if target else (None, None)

        panels = []
        for cat, names in grouped.items():
            blocks = []
            for name in names:
                var_id = stable_id("esc_pred", name)
                meta = cache.get_meta(name)
                ok, reason = compatibilidad_con_objetivo(name, meta, target, target_meta, ts, te, cache)
                info_icon = ui.tooltip(ui.tags.span(ui.HTML(ICON_SVG_INFO)), reason) if (not ok and reason) else None
                blocks.append(ui.div(ui.input_checkbox(var_id, name, value=name in selected_predictors()), ui.tags.span("Compatible" if ok else "No compatible", class_=("compat-badge compat-yes" if ok else "compat-badge compat-no"), style="margin-left:8px;"), info_icon, class_="var-item"))
            panels.append(ui.accordion_panel(cat, ui.div(*blocks), value=_slug(cat)))

        return ui.div(PANEL_STYLES, ui.h3("Panel 2: Seleccionar exógenas"), ui.p(f"Número máximo de predicciones por datos actuales: {max_num_predictions()}"), ui.accordion(*panels, id="esc_acc_preds", open=True, multiple=True), ui.div(ui.input_action_button("esc_prev_2", "← Anterior"), ui.input_action_button("esc_next_2", "Siguiente →"), style="display:flex;gap:8px;"))

    @reactive.Effect
    def _sync_preds():
        predictors_rv.set(selected_predictors())

    @reactive.Effect
    @reactive.event(input.esc_prev_2)
    def _prev2():
        current_step.set(1)

    @reactive.Effect
    @reactive.event(input.esc_next_2)
    def _next2():
        current_step.set(3)

    @reactive.Calc
    def vars_to_config():
        target = target_var_rv.get()
        preds = predictors_rv.get() or []
        ordered = [v for v in [target, *preds] if v]
        out = []
        for pretty in ordered:
            table = name_to_table.get(pretty)
            if not table:
                rows = get_tableName_for_variable(pretty) or []
                table = (rows[0].get("nombre_tabla") if rows else pretty)
            out.append({"pretty": pretty, "table": table})
        return out

    @reactive.Calc
    def selected_filters_by_var():
        out = {}
        for item in vars_to_config():
            pretty, table = item["pretty"], item["table"]
            selected = []
            for f in cache.get_filters(table):
                input_id = stable_id("esc_flt", f"{f['table']}__{f['col']}")
                vals = input[input_id]() if input_id in input else None
                if vals:
                    selected.append({"table": f["table"], "col": f["col"], "values": list(vals)})
            out[pretty] = selected
        return out

    @output
    @render.ui
    def step_panel_3():
        if current_step.get() != 3:
            return ui.div()
        panels = []
        for item in vars_to_config():
            controls = []
            for f in cache.get_filters(item["table"]):
                input_id = stable_id("esc_flt", f"{f['table']}__{f['col']}")
                controls.append(ui.input_selectize(input_id, f.get("label") or f["col"], choices=cache.get_distinct("IA", f["table"], f["col"]), multiple=True))
            panels.append(ui.accordion_panel(item["pretty"], ui.div(*controls), value=_slug(item["table"])))
        return ui.div(PANEL_STYLES, ui.h3("Panel 3: Configurar filtros"), ui.accordion(*panels, id="esc_acc_filters", open=True, multiple=True), ui.div(ui.input_action_button("esc_prev_3", "← Anterior"), ui.input_action_button("esc_next_3", "Siguiente →"), style="display:flex;gap:8px;"))

    @reactive.Effect
    @reactive.event(input.esc_prev_3)
    def _prev3():
        current_step.set(2)

    @reactive.Effect
    @reactive.event(input.esc_next_3)
    def _next3():
        current_step.set(4)

    @reactive.Calc
    def missing_future_requirements():
        target = target_var_rv.get()
        if not target:
            return []
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"
        if mode != "future":
            return []
        horizon = int(input.esc_horizon() if "esc_horizon" in input else 1)
        target_meta = cache.get_meta(target) or {}
        temporalidad = str(target_meta.get("temporalidad", "")).lower()
        _, target_end = cache.get_date_range(target)
        if not target_end:
            return []
        freq = "MS" if "mes" in temporalidad else "D"
        start = pd.to_datetime(target_end) + (pd.offsets.MonthBegin(1) if freq == "MS" else pd.Timedelta(days=1))
        needed_dates = pd.date_range(start=start, periods=horizon, freq=freq)

        required = []
        for ex in list(input.esc_model_exogs() if "esc_model_exogs" in input else predictors_rv.get()):
            _, ex_end = cache.get_date_range(ex)
            ex_end_dt = pd.to_datetime(ex_end) if ex_end else pd.NaT
            for d in needed_dates:
                if pd.isna(ex_end_dt) or d > ex_end_dt:
                    required.append({"var": ex, "date": d.strftime("%Y-%m-%d")})
        return required

    @output
    @render.ui
    def scenario_overrides_ui():
        exogs = list(input.esc_override_exogs() if "esc_override_exogs" in input else [])
        blocks = []
        for ex in exogs:
            sid = stable_id("esc_ov", ex)
            blocks.append(ui.card(ui.h5(ex), ui.input_select(f"{sid}_op", "Operación", choices={"set": "set", "add": "add", "mul": "mul", "pct": "pct"}, selected="pct"), ui.input_numeric(f"{sid}_value", "Valor", 0.0), ui.input_text(f"{sid}_start", "Inicio (YYYY-MM-DD opcional)", ""), ui.input_text(f"{sid}_end", "Fin (YYYY-MM-DD opcional)", "")))
        return ui.div(*blocks)

    @output
    @render.ui
    def scenario_future_values_ui():
        reqs = missing_future_requirements()
        if not reqs:
            return ui.p("No se requieren rellenos adicionales para el horizonte elegido.")
        controls = [ui.p("Rellena todos los meses/días faltantes para poder calcular el escenario futuro.")]
        for i, item in enumerate(reqs):
            cid = f"esc_fv_{i}"
            controls.append(ui.input_numeric(cid, f"{item['var']} · {item['date']}", value=None))
        return ui.div(*controls)

    def _build_overrides():
        out = []
        for ex in list(input.esc_override_exogs() if "esc_override_exogs" in input else []):
            sid = stable_id("esc_ov", ex)
            out.append({"var": ex, "op": input[f"{sid}_op"]() if f"{sid}_op" in input else "pct", "value": float(input[f"{sid}_value"]() if f"{sid}_value" in input else 0.0), "start": (input[f"{sid}_start"]() if f"{sid}_start" in input else "") or None, "end": (input[f"{sid}_end"]() if f"{sid}_end" in input else "") or None})
        return out

    def _build_future_values():
        out = []
        reqs = missing_future_requirements()
        for i, item in enumerate(reqs):
            cid = f"esc_fv_{i}"
            v = input[cid]() if cid in input else None
            if v is None:
                continue
            out.append({"var": item["var"], "date": item["date"], "value": float(v)})
        return out

    def _build_payload():
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"
        payload = {
            "target_var": target_var_rv.get(),
            "predictors": list(input.esc_model_exogs() if "esc_model_exogs" in input else predictors_rv.get()),
            "filters_by_var": selected_filters_by_var(),
            "horizon": int(input.esc_horizon() if "esc_horizon" in input else 1),
            "train_ratio": 0.7,
            "return_df": True,
            "auto_params": True,
            "s": 12,
            "use_target_lags": True,
            "max_lag": 12,
            "recursive_forecast": True,
            "scenario_mode": mode,
            "scenario_overrides": _build_overrides(),
            "scenario_future_values": _build_future_values() if mode == "future" else [],
        }
        if mode == "past":
            payload["scenario_window"] = {"start": input.esc_past_start(), "end": input.esc_past_end()}
        return payload

    @reactive.Effect
    def _invalidate_on_changes():
        _ = (current_step.get(), predictors_rv.get(), target_var_rv.get())
        if "esc_scenario_mode" in input:
            input.esc_scenario_mode()
        if "esc_horizon" in input:
            input.esc_horizon()
        if "esc_model" in input:
            input.esc_model()
        if "esc_model_exogs" in input:
            input.esc_model_exogs()
        if "esc_past_start" in input:
            input.esc_past_start()
        if "esc_past_end" in input:
            input.esc_past_end()
        result_rv.set(None)

    @reactive.Effect
    @reactive.event(input.esc_calc_scenario)
    def _run_scenario():
        if (input.esc_scenario_mode() if "esc_scenario_mode" in input else "future") == "future":
            reqs = missing_future_requirements()
            if len(_build_future_values()) < len(reqs):
                ui.notification_show("Debes completar todos los valores futuros faltantes.", type="warning")
                return
        runner = MODEL_RUNNERS.get(input.esc_model() if "esc_model" in input else "xgboost")
        try:
            result_rv.set(runner(_build_payload()))
        except Exception as e:
            ui.notification_show(f"Error calculando escenario: {e}", type="error")

    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()
        preds = predictors_rv.get()
        res = result_rv.get()
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"

        result_ui = []
        if res:
            if mode == "past" and res.get("y_true"):
                result_ui.append(ui.output_plot("scenario_plot", width="100%", height="380px"))
                result_ui.append(ui.output_data_frame("scenario_table"))
            else:
                result_ui.append(ui.output_plot("scenario_plot", width="100%", height="380px"))
                result_ui.append(ui.output_data_frame("scenario_table"))
            result_ui.append(ui.output_ui("kpi_ui"))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: Escenarios"),
            ui.input_radio_buttons("esc_scenario_mode", "Tipo de escenario", choices={"future": "Escenario futuro (rellenar exógenas)", "past": "Escenario pasado (contrafactual)"}, selected=mode),
            ui.input_radio_buttons("esc_model", "Modelo", choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"}, selected="xgboost", inline=True),
            ui.input_checkbox_group("esc_model_exogs", "Exógenas activas", choices=preds, selected=preds),
            (ui.div(ui.input_numeric("esc_horizon", "Horizonte futuro", value=1, min=1, max=120), ui.p("Si el horizonte supera los datos disponibles de exógenas, debes rellenar los valores faltantes."), ui.h5("Rellenar valores futuros"), ui.output_ui("scenario_future_values_ui")) if mode == "future" else ui.div(ui.input_text("esc_past_start", "Inicio ventana pasada (YYYY-MM-DD)", ""), ui.input_text("esc_past_end", "Fin ventana pasada (YYYY-MM-DD)", ""))),
            ui.input_checkbox_group("esc_override_exogs", "Exógenas a transformar", choices=list(input.esc_model_exogs() if "esc_model_exogs" in input else preds), selected=[]),
            ui.output_ui("scenario_overrides_ui"),
            ui.input_action_button("esc_calc_scenario", ("Calcular escenario" if mode == "future" else "Calcular escenario pasado"), class_="btn-primary"),
            *result_ui,
            ui.input_action_button("esc_prev_4", "← Anterior"),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _prev4():
        current_step.set(3)

    @output
    @render.plot
    def scenario_plot():
        res = result_rv.get()
        if not res:
            return None
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        if res.get("y_true"):
            ax.plot(pd.Series(res.get("y_true"), dtype="float").values, label="Real")
            ax.plot(pd.Series(res.get("y_forecast"), dtype="float").values, label="Escenario")
            ax.set_title("Real vs Escenario")
        else:
            s = pd.Series(res.get("y_forecast", []), dtype="float")
            ax.plot(s.values, label="Escenario")
            ax.set_title("Predicción de escenario")
        ax.legend()
        return fig

    @output
    @render.data_frame
    def scenario_table():
        res = result_rv.get()
        if not res:
            return render.DataGrid(pd.DataFrame())
        if res.get("y_true"):
            df = pd.DataFrame({"Real": pd.to_numeric(res["y_true"], errors="coerce"), "Escenario": pd.to_numeric(res["y_forecast"], errors="coerce")})
            df["Delta"] = df["Escenario"] - df["Real"]
            df["Delta_%"] = (df["Delta"] / df["Real"].replace(0, pd.NA)) * 100.0
            df.insert(0, "Fecha", range(1, len(df) + 1))
            return render.DataGrid(df)
        return render.DataGrid(pd.DataFrame({"Fecha": range(1, len(res.get("y_forecast", [])) + 1), "Predicción": pd.to_numeric(res.get("y_forecast", []), errors="coerce")}))

    @output
    @render.ui
    def kpi_ui():
        res = result_rv.get()
        if not res:
            return ui.div()
        return ui.div(ui.tags.span(f"MAPE: {res.get('mape', 0):.3f}", class_="selection-pill"), ui.tags.span(f"RMSE: {res.get('rmse', 0):.3f}", class_="selection-pill"), ui.tags.span(f"MAE: {res.get('mae', 0):.3f}", class_="selection-pill"))
