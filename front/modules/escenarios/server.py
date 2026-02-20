from tracemalloc import start
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

    base_info_rv = reactive.Value(None)
    scenario_res_rv = reactive.Value(None)
    last_sig_rv = reactive.Value(None)

    def _extract_dates(df: pd.DataFrame) -> pd.Series:
        if "__dt" in df.columns:
            return pd.to_datetime(df["__dt"], errors="coerce")
        if {"anio", "mes", "dia"}.issubset(df.columns):
            return pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=df["dia"]), errors="coerce")
        if {"anio", "mes"}.issubset(df.columns):
            return pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1), errors="coerce")
        return pd.to_datetime(df.index, errors="coerce")

    @reactive.Calc
    def esc_selected_model():
        return input.esc_model() if "esc_model" in input else "xgboost"

    @reactive.Calc
    def esc_active_exogs():
        allowed = set(predictors_rv.get() or [])
        selected = list(input.esc_model_exogs() if "esc_model_exogs" in input else (predictors_rv.get() or []))
        return [x for x in selected if x in allowed]

    @reactive.Calc
    def target_temporalidad():
        target_meta = cache.get_meta(target_var_rv.get()) or {}
        return str(target_meta.get("temporalidad", "")).lower()

    def _is_monthly(temp: str) -> bool:
        t = (temp or "").lower()
        # ajusta si en tu catálogo se usa "mensual" / "mes" / etc.
        return ("mes" in t) or ("mens" in t) or ("monthly" in t)

    def _granularity(temp: str) -> str:
        # Esto SÍ está bien para date_range
        return "MS" if _is_monthly(temp) else "D"

    def _parse_user_dt(txt: str, temp: str) -> pd.Timestamp:
        s = (txt or "").strip()
        if not s:
            return pd.NaT
        if _is_monthly(temp) and len(s) == 7 and s[4] == "-":  # YYYY-MM
            s = f"{s}-01"
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        # 👇 aquí el fix
        return dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()

    def _normalize_dt_series(x, temp: str) -> pd.Series:
        s = pd.to_datetime(x, errors="coerce")
        if _is_monthly(temp):
            # 👇 aquí el fix
            return s.dt.to_period("M").dt.to_timestamp(how="start")
        return s.dt.normalize()

    def _dt_key(dt: pd.Timestamp, temp: str) -> str:
        if pd.isna(dt):
            return ""
        # 👇 aquí el fix
        d = dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()
        return d.strftime("%Y-%m-%d")

    def _dt_label(dt: pd.Timestamp, temp: str) -> str:
        if pd.isna(dt):
            return ""
        # 👇 aquí el fix
        d = dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()
        return d.strftime("%Y-%m") if _is_monthly(temp) else d.strftime("%Y-%m-%d")

    @reactive.Calc
    def target_end_date():
        _, end = cache.get_date_range(target_var_rv.get())
        return pd.to_datetime(end, errors="coerce") if end else pd.NaT

    @reactive.Calc
    def future_requirements():
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"
        if mode != "future":
            return {"rows": [], "horizon": 0, "max_exog_date": pd.NaT, "temp": target_temporalidad()}

        temp = target_temporalidad()

        until_txt = (input.esc_future_until() if "esc_future_until" in input else "") or ""
        until_dt = _parse_user_dt(until_txt, temp)

        t_end = target_end_date()
        t_end = _parse_user_dt(t_end.strftime("%Y-%m-%d") if not pd.isna(t_end) else "", temp)

        if pd.isna(until_dt) or pd.isna(t_end):
            return {"rows": [], "horizon": 0, "max_exog_date": pd.NaT, "temp": temp}

        freq = _granularity(temp)
        first = t_end + (pd.offsets.MonthBegin(1) if freq == "MS" else pd.Timedelta(days=1))
        first = _parse_user_dt(first.strftime("%Y-%m-%d"), temp)

        needed = pd.date_range(start=first, end=until_dt, freq=freq)
        if len(needed) < 1:
            return {"rows": [], "horizon": 0, "max_exog_date": pd.NaT, "temp": temp}

        rows, max_exog = [], pd.NaT
        for ex in esc_active_exogs():
            _, ex_end = cache.get_date_range(ex)
            ex_end_dt = _parse_user_dt(ex_end or "", temp)

            if pd.isna(max_exog) or (not pd.isna(ex_end_dt) and ex_end_dt > max_exog):
                max_exog = ex_end_dt

            for d in needed:
                d0 = _parse_user_dt(pd.to_datetime(d).strftime("%Y-%m-%d"), temp)
                if pd.isna(ex_end_dt) or d0 > ex_end_dt:
                    rows.append({"var": ex, "date": _dt_key(d0, temp), "label": _dt_label(d0, temp)})

        return {"rows": rows, "horizon": len(needed), "max_exog_date": max_exog, "temp": temp}


    def _build_past_overrides_per_date():
        info = base_info_rv.get()
        if not info:
            return []

        ws = info.get("window", {}).get("start")
        we = info.get("window", {}).get("end")
        if not ws or not we:
            return []

        dfb = info.get("df_slice", pd.DataFrame()).copy()
        if dfb.empty or "Fecha" not in dfb.columns:
            return []

        dfb["Fecha"] = pd.to_datetime(dfb["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        dfb = dfb.dropna(subset=["Fecha"])

        dates_all = sorted(set(dfb["Fecha"].tolist()))
        selected_dates = list(input.esc_past_edit_dates() if "esc_past_edit_dates" in input else dates_all)
        selected_dates = [d for d in selected_dates if d in dates_all]
        if not selected_dates:
            return []

        dfb = dfb[dfb["Fecha"].isin(selected_dates)].copy()

        overrides = []
        for ex in list(input.esc_past_edit_exogs() if "esc_past_edit_exogs" in input else []):
            if ex not in dfb.columns:
                continue

            for _, row in dfb[["Fecha", ex]].iterrows():
                date = row["Fecha"]
                cid = stable_id("esc_past_set", f"{ws}__{we}__{ex}__{date}")
                new_val = input[cid]() if cid in input else None
                if new_val is None:
                    continue

                overrides.append({
                    "var": ex,
                    "op": "set",
                    "value": float(new_val),
                    "start": date,
                    "end": date,
                })

        return overrides


    def _build_future_values():
        out, missing = [], False
        for item in future_requirements()["rows"]:
            cid = stable_id("esc_fut_val", f"{item['var']}__{item['date']}")
            v = input[cid]() if cid in input else None
            if v is None:
                missing = True
                continue
            out.append({"var": item["var"], "date": item["date"], "value": float(v)})
        return out, missing

    @reactive.Calc
    def scenario_signature():
        if current_step.get() != 4:
            return None
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"
        return (
            mode,
            esc_selected_model(),
            tuple(esc_active_exogs()),
            (input.esc_past_start() if "esc_past_start" in input else ""),
            (input.esc_past_end() if "esc_past_end" in input else ""),
            tuple(input.esc_past_edit_exogs() if "esc_past_edit_exogs" in input else []),   # <-- NUEVO
            tuple(input.esc_past_edit_dates() if "esc_past_edit_dates" in input else []), # <-- NUEVO
            (input.esc_future_until() if "esc_future_until" in input else ""),
            repr(selected_filters_by_var()),
            target_var_rv.get(),
        )


    @reactive.Effect
    def _invalidate_scenario_on_change():
        sig = scenario_signature()
        if sig is None:
            return
        last = last_sig_rv.get()
        if last is not None and sig != last:
            scenario_res_rv.set(None)
            base_info_rv.set(None)

    @reactive.Effect
    @reactive.event(input.esc_load_base_past)
    def _load_base_past():
        start = (input.esc_past_start() if "esc_past_start" in input else "") or ""
        end = (input.esc_past_end() if "esc_past_end" in input else "") or ""
        temp = target_temporalidad()
        start_dt = _parse_user_dt(start, temp)
        end_dt = _parse_user_dt(end, temp)

        if pd.isna(start_dt) or pd.isna(end_dt):
            ui.notification_show("Debes indicar inicio y fin válidos (YYYY-MM-DD).", type="warning")
            return
        if start_dt > end_dt:
            ui.notification_show("La fecha de inicio debe ser menor o igual que la fecha fin.", type="warning")
            return
        runner = MODEL_RUNNERS.get(esc_selected_model())
        payload = {
            "target_var": target_var_rv.get(),
            "predictors": esc_active_exogs(),
            "filters_by_var": selected_filters_by_var(),
            "horizon": 1,
            "train_ratio": 0.7,
            "return_df": True,
            "auto_params": True,
            "s": 12,
            "use_target_lags": True,
            "max_lag": 12,
            "recursive_forecast": True,
            "scenario_mode": "past",
            "scenario_window": {"start": start, "end": end},
            "scenario_overrides": [],
            "scenario_future_values": [],
        }
        try:
            res = runner(payload)
            df = pd.DataFrame(res.get("df") or [])
            if df.empty:
                ui.notification_show("No se pudo cargar la base histórica para ese rango.", type="warning")
                return
            dt = _extract_dates(df)
            dt = _normalize_dt_series(dt, temp)
            y_col = res.get("y_col")
            keep_cols = [c for c in [*esc_active_exogs(), y_col] if c and c in df.columns]
            sl = df.copy()
            sl["Fecha"] = dt
            sl = sl[(sl["Fecha"] >= start_dt) & (sl["Fecha"] <= end_dt)]
            base_info_rv.set({"window": {"start": start, "end": end}, "df_slice": sl[["Fecha", *keep_cols]] if keep_cols else sl[["Fecha"]], "target_col": y_col})
            last_sig_rv.set(scenario_signature())
        except Exception as e:
            ui.notification_show(f"Error cargando base histórica: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.esc_calc_past)
    def _run_past_scenario():
        start = (input.esc_past_start() if "esc_past_start" in input else "") or ""
        end = (input.esc_past_end() if "esc_past_end" in input else "") or ""
        start_dt, end_dt = pd.to_datetime(start, errors="coerce"), pd.to_datetime(end, errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
            ui.notification_show("Rango pasado inválido: revisa inicio y fin.", type="warning")
            return
        if base_info_rv.get() is None:
            ui.notification_show("Primero carga valores base (pasado).", type="warning")
            return
        overrides = _build_past_overrides_per_date()
        runner = MODEL_RUNNERS.get(esc_selected_model())
        payload = {
            "target_var": target_var_rv.get(),
            "predictors": esc_active_exogs(),
            "filters_by_var": selected_filters_by_var(),
            "horizon": 1,
            "train_ratio": 0.7,
            "return_df": True,
            "auto_params": True,
            "s": 12,
            "use_target_lags": True,
            "max_lag": 12,
            "recursive_forecast": True,
            "scenario_mode": "past",
            "scenario_window": {"start": start, "end": end},
            "scenario_overrides": overrides,
            "scenario_future_values": [],
        }
        try:
            res = runner(payload)
            scenario_res_rv.set({"mode": "past", "raw": res, "window": {"start": start, "end": end}, "overrides": overrides})
            last_sig_rv.set(scenario_signature())
        except Exception as e:
            ui.notification_show(f"Error calculando escenario pasado: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.esc_calc_future)
    def _run_future_scenario():
        req = future_requirements()
        temp = req.get("temp", target_temporalidad())

        until_txt = (input.esc_future_until() if "esc_future_until" in input else "") or ""
        until_dt = _parse_user_dt(until_txt, temp)
        if pd.isna(until_dt):
            ui.notification_show("Debes indicar una fecha objetivo futura válida.", type="warning")
            return

        max_exog = req["max_exog_date"]
        if not pd.isna(max_exog) and until_dt <= _parse_user_dt(max_exog.strftime("%Y-%m-%d"), temp):
            ui.notification_show("La fecha objetivo debe ser posterior al máximo histórico de exógenas activas.", type="warning")
            return
        future_vals, missing = _build_future_values()
        if missing:
            ui.notification_show("Debes completar todos los valores futuros requeridos.", type="warning")
            return
        runner = MODEL_RUNNERS.get(esc_selected_model())
        payload = {
            "target_var": target_var_rv.get(),
            "predictors": esc_active_exogs(),
            "filters_by_var": selected_filters_by_var(),
            "horizon": int(req["horizon"]),
            "train_ratio": 0.7,
            "return_df": True,
            "auto_params": True,
            "s": 12,
            "use_target_lags": True,
            "max_lag": 12,
            "recursive_forecast": True,
            "scenario_mode": "future",
            "scenario_overrides": [],
            "scenario_future_values": future_vals,
        }
        try:
            res = runner(payload)
            scenario_res_rv.set({"mode": "future", "raw": res, "future_until": until_txt, "horizon": req["horizon"]})
            last_sig_rv.set(scenario_signature())
        except Exception as e:
            ui.notification_show(f"Error calculando escenario futuro: {e}", type="error")

    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()
        preds = predictors_rv.get()
        res = scenario_res_rv.get()
        mode = input.esc_scenario_mode() if "esc_scenario_mode" in input else "future"

        mode_ui = ui.div(
            ui.input_text("esc_future_until", "Fecha objetivo futura (YYYY-MM-DD)", ""),
            ui.output_ui("esc_future_horizon_ui"),
            ui.h5("Valores futuros por exógena y fecha"),
            ui.output_ui("scenario_future_values_ui"),
            ui.input_action_button("esc_calc_future", "Calcular escenario", class_="btn-primary"),
        ) if mode == "future" else ui.div(
            ui.input_text("esc_past_start", "Inicio ventana pasada (YYYY-MM-DD)", ""),
            ui.input_text("esc_past_end", "Fin ventana pasada (YYYY-MM-DD)", ""),
            ui.input_checkbox_group("esc_past_edit_exogs", "Exógenas a modificar", choices=esc_active_exogs(), selected=[]),
            ui.input_action_button("esc_load_base_past", "Cargar valores base (pasado)"),
            ui.output_ui("esc_base_info_ui"),
            ui.output_ui("esc_past_dates_ui"),          # <-- NUEVO
            ui.output_ui("esc_past_values_editor_ui"),  # <-- Editor usa SOLO esas fechas
            ui.input_action_button("esc_calc_past", "Calcular escenario", class_="btn-primary"),
        )


        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: Escenarios"),
            ui.input_radio_buttons("esc_scenario_mode", "Tipo de escenario", choices={"past": "Escenario pasado", "future": "Escenario futuro"}, selected=mode),
            ui.input_radio_buttons("esc_model", "Modelo", choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"}, selected="xgboost", inline=True),
            ui.input_checkbox_group("esc_model_exogs", "Exógenas activas", choices=preds, selected=preds),
            mode_ui,
            ui.output_plot("scenario_plot", width="100%", height="380px"),
            ui.output_data_frame("scenario_table"),
            ui.output_ui("kpi_ui"),
            ui.input_action_button("esc_prev_4", "← Anterior"),
        )

    @output
    @render.ui
    def esc_past_values_editor_ui():
        info = base_info_rv.get()
        if not info:
            return ui.p("Carga valores base (pasado) para editar por fecha.")

        dfb = info.get("df_slice", pd.DataFrame()).copy()
        if dfb.empty or "Fecha" not in dfb.columns:
            return ui.p("No hay datos base en la ventana seleccionada.")

        selected_exogs = list(input.esc_past_edit_exogs() if "esc_past_edit_exogs" in input else [])
        if not selected_exogs:
            return ui.p("Selecciona exógenas a modificar.")

        # Fechas disponibles
        temp = target_temporalidad()
        dfb["Fecha_dt"] = _normalize_dt_series(dfb["Fecha"], temp)
        dfb = dfb.dropna(subset=["Fecha_dt"])
        dfb["Fecha"] = dfb["Fecha_dt"].dt.strftime("%Y-%m-%d")  # canónica


        # Fechas seleccionadas por el usuario (si no hay, usa todas)
        dates_all = sorted(set(dfb["Fecha"].tolist()))
        selected_dates = list(input.esc_past_edit_dates() if "esc_past_edit_dates" in input else dates_all)
        selected_dates = [d for d in selected_dates if d in dates_all]
        if not selected_dates:
            return ui.p("Selecciona al menos una fecha a modificar.")

        dfb = dfb[dfb["Fecha"].isin(selected_dates)].copy()
        dfb = dfb.sort_values("Fecha")

        ws = info.get("window", {}).get("start")
        we = info.get("window", {}).get("end")

        blocks = []
        for ex in selected_exogs:
            if ex not in dfb.columns:
                continue

            inputs = []
            for _, row in dfb[["Fecha", ex]].iterrows():
                date = row["Fecha"]
                base_val = pd.to_numeric(row[ex], errors="coerce")
                cid = stable_id("esc_past_set", f"{ws}__{we}__{ex}__{date}")

                label = f"{_dt_label(pd.to_datetime(date), temp)} · base={fmt(base_val)}"

                default_val = None if pd.isna(base_val) else float(base_val)

                inputs.append(
                    ui.input_numeric(
                        cid,
                        label,
                        value=default_val,
                    )
                )

            blocks.append(
                ui.card(
                    ui.h5(ex),
                    ui.div(*inputs),
                )
            )

        if not blocks:
            return ui.p("No hay exógenas seleccionadas con datos base en las fechas elegidas.")
        return ui.div(*blocks)


    @output
    @render.ui
    def scenario_future_values_ui():
        req = future_requirements()
        reqs = req["rows"]
        if not reqs:
            return ui.p("No hay valores futuros requeridos para la fecha seleccionada.")

        blocks = [ui.p("Completa todos los valores para evitar huecos intermedios.")]
        grouped = {}
        for item in reqs:
            grouped.setdefault(item["var"], []).append(item)

        for var, items in grouped.items():
            rows = []
            for it in items:
                cid = stable_id("esc_fut_val", f"{var}__{it['date']}")
                rows.append(ui.input_numeric(cid, it.get("label", it["date"]), value=None))
            blocks.append(ui.card(ui.h5(var), ui.div(*rows)))

        return ui.div(*blocks)


    @output
    @render.ui
    def esc_future_horizon_ui():
        req = future_requirements()
        return ui.p(f"Horizonte calculado: {req['horizon']} periodos")

    @output
    @render.ui
    def esc_base_info_ui():
        info = base_info_rv.get()
        if not info:
            return ui.p("Carga los valores base para ver la ventana histórica y exógenas.")
        df = info.get("df_slice", pd.DataFrame()).copy()
        if df.empty:
            return ui.p("No hay datos base en la ventana seleccionada.")
        exogs = [c for c in df.columns if c not in ["Fecha", info.get("target_col")]]
        rows = []
        for ex in exogs:
            rows.append(ui.tags.li(f"{ex}: media base = {pd.to_numeric(df[ex], errors='coerce').mean():.3f}"))
        return ui.div(ui.p(f"Ventana base cargada: {info['window']['start']} → {info['window']['end']}"), ui.tags.ul(*rows), ui.output_data_frame("esc_base_table"))

    @output
    @render.ui
    def esc_past_dates_ui():
        info = base_info_rv.get()
        if not info:
            return ui.p("Carga valores base (pasado) para seleccionar las fechas a modificar.")

        temp = target_temporalidad()
        dfb = info.get("df_slice", pd.DataFrame()).copy()
        if dfb.empty or "Fecha" not in dfb.columns:
            return ui.p("No hay fechas base disponibles en la ventana seleccionada.")

        dts = _normalize_dt_series(dfb["Fecha"], temp)
        dts = [d for d in dts.tolist() if not pd.isna(d)]
        dts = sorted(set(dts))
        if not dts:
            return ui.p("No hay fechas válidas en la ventana base.")

        choices = {_dt_key(d, temp): _dt_label(d, temp) for d in dts}

        selected = list(input.esc_past_edit_dates() if "esc_past_edit_dates" in input else list(choices.keys()))
        selected = [d for d in selected if d in choices]
        if not selected:
            selected = list(choices.keys())

        return ui.input_checkbox_group(
            "esc_past_edit_dates",
            "Fechas a modificar (meses/días)",
            choices=choices,
            selected=selected,
        )


    @output
    @render.data_frame
    def esc_base_table():
        info = base_info_rv.get()
        if not info:
            return render.DataGrid(pd.DataFrame())
        df = info.get("df_slice", pd.DataFrame()).copy()
        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        return render.DataGrid(df)

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _prev4():
        current_step.set(3)

    @output
    @render.plot
    def scenario_plot():
        packed = scenario_res_rv.get()
        if not packed:
            return None
        res = packed.get("raw") or {}
        if not res:
            return None
        df = pd.DataFrame(res.get("df") or [])
        if df.empty:
            return None
        n_obs = int(res.get("n_obs", 0))
        h = len(res.get("y_forecast", []) or [])
        future = df.iloc[n_obs:n_obs + h].copy()
        pred_series = pd.Series(pd.to_numeric(res.get("y_forecast", []), errors="coerce"), index=future.index)
        return plot_predictions(df=df, pred=pred_series, title=("Base vs Escenario" if packed.get("mode") == "past" else "Predicción de escenario"), ylabel="Valores", xlabel="Fecha", column_y=res.get("y_col"), periodos_a_predecir=h, holidays_col=None)

    @output
    @render.data_frame
    def scenario_table():
        packed = scenario_res_rv.get()
        if not packed:
            return render.DataGrid(pd.DataFrame())
        res = packed.get("raw") or {}
        df_raw = pd.DataFrame(res.get("df") or [])
        if packed.get("mode") == "past" and res.get("y_true"):
            start_dt = pd.to_datetime(packed.get("window", {}).get("start"), errors="coerce")
            end_dt = pd.to_datetime(packed.get("window", {}).get("end"), errors="coerce")
            dt = _extract_dates(df_raw)
            dates = dt[(dt >= start_dt) & (dt <= end_dt)]
            y_true = pd.Series(pd.to_numeric(res["y_true"], errors="coerce"))
            y_pred = pd.Series(pd.to_numeric(res["y_forecast"], errors="coerce"))
            n = min(len(dates), len(y_true), len(y_pred))
            df = pd.DataFrame({"Fecha": dates.iloc[:n].dt.strftime("%Y-%m-%d").values, "Base_target": y_true.iloc[:n].values, "Escenario_target": y_pred.iloc[:n].values})
            df["Delta"] = df["Escenario_target"] - df["Base_target"]
            df["Delta_%"] = (df["Delta"] / df["Base_target"].replace(0, pd.NA)) * 100.0
            return render.DataGrid(df)
        n_obs = int(res.get("n_obs", 0))
        h = len(res.get("y_forecast", []) or [])
        future = df_raw.iloc[n_obs:n_obs + h].copy()
        dates = _extract_dates(future).dt.strftime("%Y-%m-%d")
        return render.DataGrid(pd.DataFrame({"Fecha": dates, "Predicción": pd.to_numeric(res.get("y_forecast", []), errors="coerce")}))

    @output
    @render.ui
    def kpi_ui():
        packed = scenario_res_rv.get()
        if not packed:
            return ui.div()
        res = packed.get("raw") or {}
        extras = []
        if packed.get("mode") == "past" and base_info_rv.get() is not None:
            df_base = base_info_rv.get().get("df_slice", pd.DataFrame()).copy()
            for ov in packed.get("overrides", []):
                var = ov.get("var")
                if var not in df_base.columns:
                    continue
                base_mean = pd.to_numeric(df_base[var], errors="coerce").mean()
                op, val = ov.get("op"), float(ov.get("value", 0.0))
                mod_mean = base_mean
                if op == "set":
                    mod_mean = val
                elif op == "add":
                    mod_mean = base_mean + val
                elif op == "mul":
                    mod_mean = base_mean * val
                elif op == "pct":
                    mod_mean = base_mean * (1.0 + (val / 100.0))
                extras.append(ui.tags.span(f"{var}: base media={base_mean:.3f} | modificada media={mod_mean:.3f}", class_="selection-pill"))
        return ui.div(ui.tags.span(f"MAPE: {res.get('mape', 0):.3f}", class_="selection-pill"), ui.tags.span(f"RMSE: {res.get('rmse', 0):.3f}", class_="selection-pill"), ui.tags.span(f"MAE: {res.get('mae', 0):.3f}", class_="selection-pill"), *extras)
