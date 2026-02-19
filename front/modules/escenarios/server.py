import pandas as pd
from shiny import module, reactive, render, ui

from front.utils.back_api_wrappers import (
    escenarios_run,
    get_names_in_table_catalog,
    get_tableName_for_variable,
)
from front.utils.utils import (
    ICON_SVG_INFO,
    PrediccionesCache,
    _fmt,
    _group_by_category,
    _stable_id,
    _to_date,
    build_name_to_table,
    compatibilidad_con_objetivo,
    diff_en_temporalidad,
    panel_styles,
    slug as _slug,
)


@module.server
def escenarios_server(input, output, session):
    current_step = reactive.Value(1)
    target_var_rv = reactive.Value(None)
    predictors_rv = reactive.Value([])
    baseline_rv = reactive.Value(None)
    scenario_rv = reactive.Value(None)

    catalog_entries = get_names_in_table_catalog() or []
    name_to_table = build_name_to_table(catalog_entries)
    cache = PrediccionesCache(name_to_table)
    PANEL_STYLES = panel_styles()
    _registered_pick_handlers: set[str] = set()

    @output
    @render.ui
    def step_indicator():
        labels = ["Objetivo", "Predictoras", "Filtros", "Baseline", "Escenario"]
        chips = []
        for idx, lbl in enumerate(labels, start=1):
            cls = "selection-pill"
            if idx == current_step.get():
                cls += " var-pick is-selected"
            chips.append(ui.tags.span(f"{idx}. {lbl}", class_=cls, style="margin-right:6px;"))
        return ui.div(PANEL_STYLES, *chips, style="margin:8px 0;")

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
                btn_id = _stable_id("esc_target", name)
                if btn_id not in _registered_pick_handlers:
                    _registered_pick_handlers.add(btn_id)

                    @reactive.Effect
                    @reactive.event(input[btn_id])
                    def _on_pick(_name=name):
                        target_var_rv.set(_name)
                        baseline_rv.set(None)
                        scenario_rv.set(None)

                btns.append(ui.input_action_button(btn_id, name, class_=("var-pick is-selected" if selected == name else "var-pick")))
            panels.append(ui.accordion_panel(cat, ui.div(*btns, class_="var-list"), value=_slug(cat)))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 1: Seleccionar variable objetivo"),
            ui.accordion(*panels, id="esc_acc_target", open=True, multiple=True),
            ui.input_action_button("esc_next_1", "Siguiente →"),
        )

    @reactive.Effect
    @reactive.event(input.esc_next_1)
    def _next1():
        current_step.set(2)

    @reactive.Calc
    def predictor_pairs():
        grouped = _group_by_category(catalog_entries, exclude_name=target_var_rv.get())
        pairs = []
        for _, names in grouped.items():
            for name in names:
                pairs.append((_stable_id("esc_pred", name), name))
        return pairs

    @reactive.Calc
    def selected_predictors():
        vals = []
        for var_id, name in predictor_pairs():
            if var_id in input and input[var_id]():
                vals.append(name)
        return sorted(set(vals))

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
        grouped = _group_by_category(catalog_entries, exclude_name=target)
        target_meta = cache.get_meta(target) if target else {}
        ts, te = cache.get_date_range(target) if target else (None, None)

        panels = []
        for cat, names in grouped.items():
            blocks = []
            for name in names:
                var_id = _stable_id("esc_pred", name)
                meta = cache.get_meta(name)
                ok, reason = compatibilidad_con_objetivo(name, meta, target, target_meta, ts, te, cache)
                info_icon = ui.tooltip(ui.tags.span(ui.HTML(ICON_SVG_INFO)), reason) if (not ok and reason) else None
                blocks.append(ui.div(
                    ui.input_checkbox(var_id, name, value=name in selected_predictors()),
                    ui.tags.span("Compatible" if ok else "No compatible", class_=("compat-badge compat-yes" if ok else "compat-badge compat-no"), style="margin-left:8px;"),
                    info_icon,
                    class_="var-item",
                ))
            panels.append(ui.accordion_panel(cat, ui.div(*blocks), value=_slug(cat)))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 2: Seleccionar exógenas"),
            ui.tags.div(f"Máximo horizonte disponible: {max_num_predictions()}", class_="selection-pill"),
            ui.accordion(*panels, id="esc_acc_predictors", open=True, multiple=True),
            ui.div(ui.input_action_button("esc_prev_2", "← Anterior"), ui.input_action_button("esc_next_2", "Siguiente →"), style="display:flex;gap:8px;"),
        )

    @reactive.Effect
    def _sync_pred():
        predictors_rv.set(selected_predictors())

    @reactive.Effect
    @reactive.event(selected_predictors)
    def _invalidate_pred_selection():
        baseline_rv.set(None)
        scenario_rv.set(None)

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
                input_id = _stable_id("esc_flt", f"{f['table']}__{f['col']}")
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
                input_id = _stable_id("esc_flt", f"{f['table']}__{f['col']}")
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

    def _base_payload(overrides: list[dict]):
        return {
            "model": input.esc_model() if "esc_model" in input else "xgboost",
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
            "scenario_overrides": overrides,
        }

    @reactive.Effect
    @reactive.event(input.esc_calc_baseline)
    def _run_baseline():
        if max_num_predictions() <= 0:
            ui.notification_show("No hay horizonte disponible. Selecciona exógenas compatibles.", type="warning")
            return
        try:
            resp = escenarios_run(_base_payload([]))
            baseline_rv.set(resp)
            scenario_rv.set(None)
        except Exception as e:
            ui.notification_show(f"Error calculando baseline: {e}", type="error")

    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()
        preds = predictors_rv.get()
        m = max_num_predictions()
        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: Calcular baseline"),
            ui.input_radio_buttons("esc_model", "Modelo", choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"}, selected="xgboost", inline=True),
            ui.input_checkbox_group("esc_model_exogs", "Exógenas activas", choices=preds, selected=preds),
            (ui.input_slider("esc_horizon", "Horizonte", min=1, max=m, value=1 if m >= 1 else 1) if m >= 1 else ui.p("Sin horizonte disponible", style="color:#b42318;")),
            ui.input_action_button("esc_calc_baseline", "Calcular baseline", class_="btn-primary"),
            ui.div(ui.input_action_button("esc_prev_4", "← Anterior"), ui.input_action_button("esc_next_4", "Siguiente →"), style="display:flex;gap:8px;margin-top:10px;"),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _prev4():
        current_step.set(3)

    @reactive.Effect
    @reactive.event(input.esc_next_4)
    def _next4():
        current_step.set(5)

    @output
    @render.ui
    def overrides_ui():
        exogs = list(input.esc_scn_exogs() if "esc_scn_exogs" in input else [])
        blocks = []
        for ex in exogs:
            sid = _stable_id("esc_ov", ex)
            blocks.append(ui.card(
                ui.h5(ex),
                ui.input_select(f"{sid}_op", "Operación", choices={"set": "set", "add": "add", "mul": "mul", "pct": "pct"}, selected="pct"),
                ui.input_numeric(f"{sid}_value", "Valor", 0.0),
                ui.input_text(f"{sid}_start", "Inicio (YYYY-MM-DD, opcional)", ""),
                ui.input_text(f"{sid}_end", "Fin (YYYY-MM-DD, opcional)", ""),
            ))
        return ui.div(*blocks)

    def _build_overrides():
        out = []
        for ex in list(input.esc_scn_exogs() if "esc_scn_exogs" in input else []):
            sid = _stable_id("esc_ov", ex)
            op = input[f"{sid}_op"]() if f"{sid}_op" in input else "pct"
            value = input[f"{sid}_value"]() if f"{sid}_value" in input else 0.0
            start = input[f"{sid}_start"]() if f"{sid}_start" in input else ""
            end = input[f"{sid}_end"]() if f"{sid}_end" in input else ""
            out.append({"var": ex, "op": op, "value": float(value or 0.0), "start": (start or None), "end": (end or None)})
        return out

    @reactive.Effect
    @reactive.event(input.esc_calc_scenario)
    def _run_scenario():
        if baseline_rv.get() is None:
            ui.notification_show("Primero calcula baseline.", type="warning")
            return
        try:
            resp = escenarios_run(_base_payload(_build_overrides()))
            scenario_rv.set(resp)
        except Exception as e:
            ui.notification_show(f"Error calculando escenario: {e}", type="error")

    @output
    @render.plot
    def scenario_plot():
        res = scenario_rv.get()
        if not res:
            return None
        base = pd.Series(res["baseline"]["y_forecast"])
        scn = pd.Series(res["scenario"]["y_forecast"])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(base.values, label="Baseline")
        ax.plot(scn.values, label="Escenario")
        ax.legend()
        ax.set_title("Comparación Baseline vs Escenario")
        return fig

    @output
    @render.data_frame
    def scenario_table():
        res = scenario_rv.get()
        if not res or not res.get("df"):
            return render.DataGrid(pd.DataFrame())
        df = pd.DataFrame(res["df"])
        n_obs = int(res["n_obs"])
        h = int(res["horizon"])
        future = df.iloc[n_obs:n_obs + h].copy()
        if {"anio", "mes"}.issubset(future.columns):
            future["Fecha"] = pd.to_datetime(dict(year=future["anio"], month=future["mes"], day=(future["dia"] if "dia" in future.columns else 1)), errors="coerce")
        else:
            future["Fecha"] = future.index
        future["Baseline"] = pd.to_numeric(res["baseline"]["y_forecast"], errors="coerce")
        future["Escenario"] = pd.to_numeric(res["scenario"]["y_forecast"], errors="coerce")
        future["Delta"] = future["Escenario"] - future["Baseline"]
        future["Delta_%"] = (future["Delta"] / future["Baseline"].replace(0, pd.NA)) * 100.0
        out = future[["Fecha", "Baseline", "Escenario", "Delta", "Delta_%"]].copy()
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.strftime("%d-%m-%Y")
        return render.DataGrid(out)

    @output
    @render.ui
    def kpi_ui():
        res = scenario_rv.get()
        if not res:
            return ui.div()
        base = pd.Series(res["baseline"]["y_forecast"], dtype="float")
        scn = pd.Series(res["scenario"]["y_forecast"], dtype="float")
        d = scn - base
        return ui.div(
            ui.tags.span(f"Suma delta: {d.sum():.3f}", class_="selection-pill"),
            ui.tags.span(f"Promedio delta: {d.mean():.3f}", class_="selection-pill"),
            ui.tags.span(f"Último delta: {d.iloc[-1]:.3f}", class_="selection-pill"),
        )

    @output
    @render.ui
    def step_panel_5():
        if current_step.get() != 5:
            return ui.div()
        exogs = list(input.esc_model_exogs() if "esc_model_exogs" in input else predictors_rv.get())
        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 5: Definir escenario"),
            ui.input_checkbox_group("esc_scn_exogs", "Exógenas a modificar", choices=exogs, selected=[]),
            ui.output_ui("overrides_ui"),
            ui.input_action_button("esc_calc_scenario", "Calcular escenario", class_="btn-primary"),
            ui.output_ui("kpi_ui"),
            ui.output_plot("scenario_plot", width="100%", height="380px"),
            ui.output_data_frame("scenario_table"),
            ui.input_action_button("esc_prev_5", "← Anterior"),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_5)
    def _prev5():
        current_step.set(4)
