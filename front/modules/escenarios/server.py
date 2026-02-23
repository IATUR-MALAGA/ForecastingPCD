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


# -----------------------------
# Module
# -----------------------------
@module.server
def escenarios_server(input, output, session):
    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    current_step = reactive.Value(1)

    target_var_rv = reactive.Value(None)      # objetivo seleccionado
    predictors_rv = reactive.Value([])        # exógenas seleccionadas (panel 2)

    base_info_rv = reactive.Value(None)       # base histórica cargada (modo pasado)
    scenario_res_rv = reactive.Value(None)    # resultado de escenario (pasado/futuro)
    last_sig_rv = reactive.Value(None)        # firma usada para invalidación de resultados

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

    METRIC_DESCRIPTIONS = {
        "MAPE": "Error porcentual absoluto medio (en %).",
        "RMSE": "Raíz del error cuadrático medio (penaliza más los errores grandes).",
        "MAE": "Error absoluto medio (promedio del error en la escala original).",
    }

    # ---------------------------------------------------------------------
    # UI helpers (puros)
    # ---------------------------------------------------------------------
    def _metric_info_tooltip(description: str):
        return ui.tooltip(
            ui.tags.span(ui.HTML(ICON_SVG_INFO), style="display:inline-flex; cursor:help;"),
            description,
        )

    def _metric_pill(label: str, value: float):
        return ui.tags.span(
            ui.tags.span(f"{label}: {value:.3f}"),
            _metric_info_tooltip(METRIC_DESCRIPTIONS.get(label, "Métrica de error del modelo.")),
            class_="selection-pill",
            style="display:inline-flex; align-items:center; gap:6px;",
        )

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

    def _is_monthly(temp: str) -> bool:
        t = (temp or "").lower()
        return ("mes" in t) or ("mens" in t) or ("monthly" in t)

    def _granularity(temp: str) -> str:
        """Frecuencia para date_range."""
        return "MS" if _is_monthly(temp) else "D"

    def _parse_user_dt(txt: str, temp: str) -> pd.Timestamp:
        """
        Parsea texto de fecha. Si mensual, admite YYYY-MM y normaliza al inicio de mes.
        Si diario, normaliza al inicio de día.
        """
        s = (txt or "").strip()
        if not s:
            return pd.NaT

        if _is_monthly(temp) and len(s) == 7 and s[4] == "-":  # YYYY-MM
            s = f"{s}-01"

        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return pd.NaT

        return dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()

    def _normalize_dt_series(x, temp: str) -> pd.Series:
        s = pd.to_datetime(x, errors="coerce")
        if _is_monthly(temp):
            return s.dt.to_period("M").dt.to_timestamp(how="start")
        return s.dt.normalize()

    def _dt_key(dt: pd.Timestamp, temp: str) -> str:
        if pd.isna(dt):
            return ""
        d = dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()
        return d.strftime("%Y-%m-%d")

    def _dt_label(dt: pd.Timestamp, temp: str) -> str:
        if pd.isna(dt):
            return ""
        d = dt.to_period("M").to_timestamp(how="start") if _is_monthly(temp) else dt.normalize()
        return d.strftime("%Y-%m") if _is_monthly(temp) else d.strftime("%Y-%m-%d")

    # ---------------------------------------------------------------------
    # Step indicator (opcional)
    # ---------------------------------------------------------------------
    @output
    @render.ui
    def step_indicator():
        labels = ["Objetivo", "Predictoras", "Filtros", "Escenarios"]
        pills = []
        for i, lbl in enumerate(labels, start=1):
            pills.append(
                ui.tags.span(
                    f"{i}. {lbl}",
                    class_=(
                        "selection-pill var-pick is-selected"
                        if i == current_step.get()
                        else "selection-pill"
                    ),
                    style="margin-right:6px;",
                )
            )
        return ui.div(PANEL_STYLES, *pills, style="margin:8px 0;")

    # =====================================================================
    # Panel 1: Objetivo
    # =====================================================================
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
                    def _on_pick_target(_name=name):
                        target_var_rv.set(_name)
                        # Cambiar objetivo invalida lo calculado
                        scenario_res_rv.set(None)
                        base_info_rv.set(None)

                btns.append(
                    ui.input_action_button(
                        btn_id,
                        name,
                        class_=("var-pick is-selected" if selected == name else "var-pick"),
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
            ui.input_action_button("esc_next_1", "Siguiente →"),
        )

    @reactive.Effect
    @reactive.event(input.esc_next_1)
    def _go_step_2():
        current_step.set(2)

    # =====================================================================
    # Panel 2: Predictoras + compatibilidad + horizonte máx.
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
        selected = []
        for var_id, name in predictor_pairs():
            if var_id in input and input[var_id]():
                selected.append(name)
        return sorted(set(selected))

    @reactive.Calc
    def max_num_predictions():
        """
        Horizonte máximo común: desde end del target hasta el mínimo end de predictoras seleccionadas,
        en unidades de la temporalidad del target.
        """
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
            ok, _ = compatibilidad_con_objetivo(
                predictor_name=p,
                predictor_meta=cache.get_meta(p),
                target_name=target,
                target_meta=target_meta,
                target_start=target_start,
                target_end=target_end,
                cache=cache,
            )
            if not ok:
                return 0

            _, p_end = cache.get_date_range(p)
            if not p_end:
                return 0

            if min_end is None or _to_date(p_end) < _to_date(min_end):
                min_end = p_end

        n = diff_en_temporalidad(target_end, min_end, tgt_temp)
        return max(0, int(n or 0))

    @reactive.Effect
    def _sync_predictors_rv():
        predictors_rv.set(selected_predictors())

    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()

        target = target_var_rv.get()
        grouped = group_by_category(catalog_entries, exclude_name=target)

        target_meta = cache.get_meta(target) if target else {}
        ts, te = cache.get_date_range(target) if target else (None, None)

        selected_set = set(selected_predictors())

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

                info_icon = (
                    ui.tooltip(ui.tags.span(ui.HTML(ICON_SVG_INFO)), reason)
                    if (not ok and reason)
                    else None
                )

                blocks.append(
                    ui.div(
                        ui.input_checkbox(var_id, name, value=(name in selected_set)),
                        ui.tags.span(
                            "Compatible" if ok else "No compatible",
                            class_=("compat-badge compat-yes" if ok else "compat-badge compat-no"),
                            style="margin-left:8px;",
                        ),
                        info_icon,
                        class_="var-item",
                    )
                )

            panels.append(ui.accordion_panel(cat, ui.div(*blocks), value=_slug(cat)))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 2: Seleccionar exógenas"),
            ui.p(f"Número máximo de predicciones por datos actuales: {max_num_predictions()}"),
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

    # =====================================================================
    # Panel 3: Filtros
    # =====================================================================
    @reactive.Calc
    def vars_to_config():
        """
        Lista estable de variables seleccionadas (objetivo + predictoras), con su tabla real.
        """
        target = target_var_rv.get()
        preds = predictors_rv.get() or []

        ordered = []
        for v in [target, *preds]:
            if v and v not in ordered:
                ordered.append(v)

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
                controls.append(
                    ui.input_selectize(
                        input_id,
                        f.get("label") or f["col"],
                        choices=cache.get_distinct("IA", f["table"], f["col"]),
                        multiple=True,
                        options={
                            "placeholder": "Selecciona uno o varios valores (vacío = sin filtro)",
                            "plugins": ["remove_button"],
                        },
                    )
                )
            panels.append(ui.accordion_panel(item["pretty"], ui.div(*controls), value=_slug(item["table"])))

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 3: Configurar filtros"),
            ui.accordion(*panels, id="esc_acc_filters", open=True, multiple=True),
            ui.div(
                ui.input_action_button("esc_prev_3", "← Anterior"),
                ui.input_action_button("esc_next_3", "Siguiente →"),
                style="display:flex;gap:8px;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_3)
    def _go_step_2_from_3():
        current_step.set(2)

    @reactive.Effect
    @reactive.event(input.esc_next_3)
    def _go_step_4():
        current_step.set(4)

    # =====================================================================
    # Panel 4: Escenarios PASADOS (rediseñado)
    # =====================================================================

    # ------------------------
    # Modelo + exógenas activas (en el modelo)
    # ------------------------
    @reactive.Calc
    def esc_selected_model():
        return input.esc_model() if "esc_model" in input else "xgboost"

    @reactive.Calc
    def esc_active_exogs():
        allowed = set(predictors_rv.get() or [])
        selected = list(
            input.esc_model_exogs()
            if "esc_model_exogs" in input
            else (predictors_rv.get() or [])
        )
        return [x for x in selected if x in allowed]

    # ------------------------
    # Base cargada (ventana histórica)
    # ------------------------
    base_info_rv = reactive.Value(None)       # {"window":{start,end}, "df_base": df, "target_col": str, "temp": str}
    scenario_res_rv = reactive.Value(None)    # {"table": df_result, "plot": {...}, "raw": res, ...}
    last_sig_rv = reactive.Value(None)

    @reactive.Calc
    def base_window():
        """Inicio/fin como texto y como timestamp normalizado según temporalidad."""
        start_txt = (input.esc_past_start() if "esc_past_start" in input else "") or ""
        end_txt = (input.esc_past_end() if "esc_past_end" in input else "") or ""
        temp = target_temporalidad()
        start_dt = _parse_user_dt(start_txt, temp)
        end_dt = _parse_user_dt(end_txt, temp)
        return {"start_txt": start_txt, "end_txt": end_txt, "start_dt": start_dt, "end_dt": end_dt, "temp": temp}

    @reactive.Calc
    def available_dates_from_base():
        """Fechas disponibles en base cargada (claves canónicas YYYY-MM-DD)."""
        info = base_info_rv.get()
        if not info:
            return []
        dfb = info.get("df_base", pd.DataFrame()).copy()
        if dfb.empty or "Fecha" not in dfb.columns:
            return []
        # Fecha ya viene normalizada; garantizamos formato canónico:
        dt = _normalize_dt_series(dfb["Fecha"], info["temp"])
        dt = [d for d in dt.tolist() if not pd.isna(d)]
        dt = sorted(set(dt))
        return [ _dt_key(d, info["temp"]) for d in dt if _dt_key(d, info["temp"]) ]

    @reactive.Calc
    def edit_exogs():
        """Exógenas que el usuario quiere modificar (subconjunto de activas)."""
        active = set(esc_active_exogs() or [])
        sel = list(input.esc_edit_exogs() if "esc_edit_exogs" in input else [])
        return [x for x in sel if x in active]

    @reactive.Calc
    def edit_dates():
        """Fechas que el usuario quiere modificar (claves canónicas)."""
        choices = set(available_dates_from_base())
        sel = list(input.esc_edit_dates() if "esc_edit_dates" in input else [])
        sel = [d for d in sel if d in choices]
        return sel

    def _base_value_for(exog: str, date_key: str):
        """Devuelve el valor base (numérico) de la exógena en esa fecha (o None)."""
        info = base_info_rv.get()
        if not info:
            return None
        dfb = info.get("df_base", pd.DataFrame()).copy()
        if dfb.empty or "Fecha" not in dfb.columns:
            return None
        exog_col = exog if exog in dfb.columns else _safe_alias(exog)
        if exog_col not in dfb.columns:
            return None
        temp = info["temp"]
        # Creamos key canónica por fila:
        dt = _normalize_dt_series(dfb["Fecha"], temp)
        keys = dt.dt.strftime("%Y-%m-%d")
        dfb = dfb.assign(__key=keys)
        row = dfb[dfb["__key"] == date_key]
        if row.empty:
            return None
        v = pd.to_numeric(row.iloc[0][exog_col], errors="coerce")
        return None if pd.isna(v) else float(v)

    def _build_overrides_and_missing():
        """
        Construye overrides por (exog, fecha) seleccionados.
        Requiere que cada celda seleccionada tenga un nuevo valor.
        """
        info = base_info_rv.get()
        if not info:
            return [], True

        ws = info["window"]["start"]
        we = info["window"]["end"]

        overrides = []
        missing = False

        for ex in edit_exogs():
            for d in edit_dates():
                cid = stable_id("esc_past_set", f"{ws}__{we}__{ex}__{d}")
                new_val = input[cid]() if cid in input else None
                if new_val is None:
                    missing = True
                    continue
                overrides.append({
                    "var": ex,
                    "op": "set",
                    "value": float(new_val),
                    "start": d,
                    "end": d,
                })

        return overrides, missing

    @reactive.Calc
    def overrides_signature():
        """
        Firma de valores introducidos (para invalidar resultados si cambian inputs numéricos).
        """
        info = base_info_rv.get()
        if not info:
            return None
        ws = info["window"]["start"]
        we = info["window"]["end"]
        tup = []
        for ex in edit_exogs():
            for d in edit_dates():
                cid = stable_id("esc_past_set", f"{ws}__{we}__{ex}__{d}")
                v = input[cid]() if cid in input else None
                tup.append((ex, d, v))
        return repr(tup)

    @reactive.Calc
    def scenario_signature():
        """
        Si cambia cualquier cosa relevante, invalidamos resultados.
        """
        if current_step.get() != 4:
            return None
        bw = base_window()
        return (
            esc_selected_model(),
            tuple(esc_active_exogs()),
            bw["start_txt"],
            bw["end_txt"],
            tuple(edit_exogs()),
            tuple(edit_dates()),
            overrides_signature(),
            repr(selected_filters_by_var()),
            target_var_rv.get(),
        )

    @reactive.Effect
    def _invalidate_on_change():
        sig = scenario_signature()
        if sig is None:
            return
        last = last_sig_rv.get()
        if last is not None and sig != last:
            scenario_res_rv.set(None)

    # ------------------------
    # Acción: cargar base (ventana histórica)
    # ------------------------
    @reactive.Effect
    @reactive.event(input.esc_load_base_past)
    def _load_base_past():
        bw = base_window()
        start_txt, end_txt = bw["start_txt"], bw["end_txt"]
        start_dt, end_dt = bw["start_dt"], bw["end_dt"]
        temp = bw["temp"]

        if pd.isna(start_dt) or pd.isna(end_dt):
            ui.notification_show("Debes indicar inicio y fin válidos (YYYY-MM-DD o YYYY-MM si es mensual).", type="warning")
            return
        if start_dt > end_dt:
            ui.notification_show("La fecha de inicio debe ser menor o igual que la fecha fin.", type="warning")
            return

        runner = MODEL_RUNNERS.get(esc_selected_model())
        if runner is None:
            ui.notification_show("Modelo no soportado.", type="error")
            return

        payload = {
            "target_var": target_var_rv.get(),
            "predictors": esc_active_exogs(),
            "filters_by_var": selected_filters_by_var(),
            "horizon": 1,             # no importa aquí; queremos df completo para la ventana
            "train_ratio": 0.7,
            "return_df": True,
            "auto_params": True,
            "s": 12,
            "use_target_lags": True,
            "max_lag": 12,
            "recursive_forecast": True,
            "scenario_mode": "past",
            "scenario_window": {"start": start_txt, "end": end_txt},
            "scenario_overrides": [],
            "scenario_future_values": [],
        }

        try:
            res = runner(payload)
            df = pd.DataFrame(res.get("df") or [])
            if df.empty:
                ui.notification_show("No se pudo cargar la base histórica para ese rango.", type="warning")
                return

            y_col = res.get("y_col")

            # Construimos df_base con Fecha + exógenas activas + target real (si está en df)
            dt = _normalize_dt_series(_extract_dates(df), temp)
            sl = df.copy()
            sl["Fecha"] = dt

            sl = sl[(sl["Fecha"] >= start_dt) & (sl["Fecha"] <= end_dt)].copy()
            if sl.empty:
                ui.notification_show("La ventana no tiene datos tras aplicar el filtro de fechas.", type="warning")
                return

            keep_cols = []
            for c in [*esc_active_exogs(), y_col]:
                if not c:
                    continue
                resolved = c if c in sl.columns else _safe_alias(c)
                if resolved in sl.columns:
                    keep_cols.append(resolved)
            if "Fecha" not in keep_cols:
                keep_cols = ["Fecha"] + [c for c in keep_cols if c != "Fecha"]
            else:
                keep_cols = ["Fecha"] + [c for c in keep_cols if c != "Fecha"]

            df_base = sl[keep_cols].copy()

            base_info_rv.set({
                "window": {"start": start_txt, "end": end_txt},
                "df_base": df_base,
                "target_col": y_col,
                "temp": temp,
            })

            scenario_res_rv.set(None)
            last_sig_rv.set(scenario_signature())
            ui.notification_show("Base histórica cargada.", type="message")

        except Exception as e:
            ui.notification_show(f"Error cargando base histórica: {e}", type="error")

    # ------------------------
    # Acción: calcular escenario pasado
    # ------------------------
    def _build_past_result_table(res: dict, selected_date_keys: list[str], temp: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
        df = pd.DataFrame(res.get("df") or [])
        if df.empty:
            return pd.DataFrame(), None

        y_col = res.get("y_col")
        dt = _normalize_dt_series(_extract_dates(df), temp)

        mask = (dt >= start_dt) & (dt <= end_dt)
        dates = dt[mask].reset_index(drop=True)

        # y_true: preferimos res["y_true"] si viene; si no, intentamos del df/y_col
        y_true_raw = res.get("y_true")
        if y_true_raw:
            y_true = pd.Series(pd.to_numeric(y_true_raw, errors="coerce")).reset_index(drop=True)
        elif y_col and y_col in df.columns:
            y_true = pd.Series(pd.to_numeric(df.loc[mask, y_col], errors="coerce")).reset_index(drop=True)
        else:
            y_true = pd.Series([], dtype="float64")

        y_pred = pd.Series(pd.to_numeric(res.get("y_forecast", []) or [], errors="coerce")).reset_index(drop=True)

        n = min(len(dates), len(y_true), len(y_pred))
        if n < 1:
            return pd.DataFrame(), None

        out = pd.DataFrame({
            "Fecha_dt": dates.iloc[:n].values,
            "Real": y_true.iloc[:n].values,
            "Escenario": y_pred.iloc[:n].values,
        })
        out["Fecha"] = pd.to_datetime(out["Fecha_dt"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Filtrar SOLO fechas editadas
        sel = set(selected_date_keys or [])
        out = out[out["Fecha"].isin(sel)].copy()
        out = out.sort_values("Fecha_dt")

        if out.empty:
            return pd.DataFrame(), None

        out["Delta"] = out["Escenario"] - out["Real"]
        out["Delta_%"] = (out["Delta"] / pd.to_numeric(out["Real"], errors="coerce").replace(0, pd.NA)) * 100.0

        plot_pack = {
            "dates": pd.to_datetime(out["Fecha"], errors="coerce"),
            "y_true": pd.to_numeric(out["Real"], errors="coerce"),
            "y_pred": pd.to_numeric(out["Escenario"], errors="coerce"),
        }
        out = out.drop(columns=["Fecha_dt"])
        return out, plot_pack

    @reactive.Effect
    @reactive.event(input.esc_calc_past)
    def _run_past_scenario():
        info = base_info_rv.get()
        if not info:
            ui.notification_show("Primero carga la base histórica (ventana).", type="warning")
            return

        if not edit_exogs():
            ui.notification_show("Selecciona al menos una exógena a modificar.", type="warning")
            return
        if not edit_dates():
            ui.notification_show("Selecciona al menos una fecha a modificar.", type="warning")
            return

        overrides, missing = _build_overrides_and_missing()
        if missing:
            ui.notification_show("Faltan nuevos valores: completa todas las celdas seleccionadas.", type="warning")
            return

        info = base_info_rv.get()
        win_start = info["window"]["start"]
        win_end = info["window"]["end"]
        start_dt, end_dt = bw["start_dt"], bw["end_dt"]
        temp = info["temp"]

        runner = MODEL_RUNNERS.get(esc_selected_model())
        if runner is None:
            ui.notification_show("Modelo no soportado.", type="error")
            return

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
            "scenario_window": {"start": win_start, "end": win_end},
            "scenario_overrides": overrides,
            "scenario_future_values": [],
        }

        try:
            res = runner(payload)

            table_df, plot_pack = _build_past_result_table(
                res=res,
                selected_date_keys=edit_dates(),
                temp=temp,
                start_dt=start_dt,
                end_dt=end_dt,
            )
            if table_df.empty:
                ui.notification_show("No se pudieron construir resultados para las fechas editadas.", type="warning")
                return

            scenario_res_rv.set({
                "raw": res,
                "table": table_df,
                "plot": plot_pack,
                "window": {"start": start_txt, "end": end_txt},
                "overrides": overrides,
                "edited_dates": edit_dates(),
                "edited_exogs": edit_exogs(),
            })
            last_sig_rv.set(scenario_signature())
            ui.notification_show("Escenario calculado.", type="message")

        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            ui.notification_show(f"Error {e.response.status_code}: {detail}", type="error")

    # ------------------------
    # UI Panel 4
    # ------------------------
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        preds = predictors_rv.get() or []
        info = base_info_rv.get()

        base_controls = ui.card(
            ui.h4("Ventana histórica"),
            ui.input_text("esc_past_start", "Inicio (YYYY-MM-DD o YYYY-MM si es mensual)", ""),
            ui.input_text("esc_past_end", "Fin (YYYY-MM-DD o YYYY-MM si es mensual)", ""),
            ui.input_action_button("esc_load_base_past", "Cargar valores base", class_="btn-secondary"),
            style="padding: 12px; border-radius: 14px;",
        )

        if not info:
            # Solo mostramos selección modelo + exógenas activas + ventana
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 4: Escenario pasado"),
                ui.input_radio_buttons(
                    "esc_model",
                    "Modelo",
                    choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                    selected="xgboost",
                    inline=True,
                ),
                ui.input_checkbox_group(
                    "esc_model_exogs",
                    "Exógenas activas (en el modelo)",
                    choices=preds,
                    selected=preds,
                ),
                base_controls,
                ui.p("Carga la base para poder seleccionar fechas y editar valores."),
                ui.input_action_button("esc_prev_4", "← Anterior"),
            )

        # Con base cargada: selector de exógenas a modificar + fechas + editor
        temp = info["temp"]

        # Fechas (choices: key -> label)
        keys = available_dates_from_base()
        # Mostrar friendly label según temporalidad
        choices_dates = {}
        for k in keys:
            dt = pd.to_datetime(k, errors="coerce")
            choices_dates[k] = _dt_label(dt, temp) if not pd.isna(dt) else k

        selected_dates = edit_dates() or list(choices_dates.keys())
        selected_dates = [d for d in selected_dates if d in choices_dates]

        editor_block = ui.output_ui("esc_past_editor_ui")

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: Escenario pasado"),
            ui.input_radio_buttons(
                "esc_model",
                "Modelo",
                choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                selected="xgboost",
                inline=True,
            ),
            ui.input_checkbox_group(
                "esc_model_exogs",
                "Exógenas activas (en el modelo)",
                choices=preds,
                selected=list(set(preds).intersection(set(esc_active_exogs() or preds))),
            ),
            base_controls,
            ui.tags.hr(style="margin: 12px 0;"),
            ui.h4("Edición de escenario"),
            ui.input_checkbox_group(
                "esc_edit_exogs",
                "Exógenas a modificar",
                choices=esc_active_exogs(),
                selected=edit_exogs(),
            ),
            ui.input_checkbox_group(
                "esc_edit_dates",
                "Fechas a modificar",
                choices=choices_dates,
                selected=selected_dates,
            ),
            editor_block,
            ui.tags.div(
                ui.input_action_button("esc_calc_past", "Calcular escenario", class_="btn-primary"),
                style="margin-top: 12px;",
            ),
            ui.tags.hr(style="margin: 12px 0;"),
            ui.h4("Resultados (solo fechas editadas)"),
            ui.output_plot("scenario_plot", width="100%", height="360px"),
            ui.output_data_frame("scenario_table"),
            ui.output_ui("kpi_ui"),
            ui.input_action_button("esc_prev_4", "← Anterior"),
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _go_step_3_from_4():
        current_step.set(3)

    # ------------------------
    # Editor UI: Fecha | base | nuevo
    # ------------------------
    @output
    @render.ui
    def esc_past_editor_ui():
        info = base_info_rv.get()
        if not info:
            return ui.div()

        if not edit_exogs():
            return ui.p("Selecciona exógenas a modificar para mostrar el editor.")
        if not edit_dates():
            return ui.p("Selecciona fechas a modificar para mostrar el editor.")

        ws = info["window"]["start"]
        we = info["window"]["end"]

        # Tabla por exógena
        blocks = []
        for ex in edit_exogs():
            rows = []
            for d in edit_dates():
                base_val = _base_value_for(ex, d)
                cid = stable_id("esc_past_set", f"{ws}__{we}__{ex}__{d}")

                dt = pd.to_datetime(d, errors="coerce")
                date_lbl = _dt_label(dt, info["temp"]) if not pd.isna(dt) else d

                rows.append(
                    ui.tags.tr(
                        ui.tags.td(date_lbl, style="padding:6px 8px; white-space:nowrap;"),
                        ui.tags.td(fmt(base_val), style="padding:6px 8px;"),
                        ui.tags.td(
                            ui.input_numeric(cid, "", value=None),
                            style="padding:6px 8px; min-width:180px;",
                        ),
                    )
                )

            table = ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Fecha", style="text-align:left; padding:6px 8px;"),
                        ui.tags.th("Valor base", style="text-align:left; padding:6px 8px;"),
                        ui.tags.th("Nuevo valor", style="text-align:left; padding:6px 8px;"),
                    )
                ),
                ui.tags.tbody(*rows),
                style="width:100%; border-collapse:collapse;",
            )

            blocks.append(
                ui.card(
                    ui.h5(ex, style="margin:0 0 8px 0;"),
                    table,
                    style="padding: 12px; border-radius: 14px;",
                )
            )

        return ui.div(*blocks)

    # ------------------------
    # Outputs: plot + table + KPIs
    # ------------------------
    @output
    @render.data_frame
    def scenario_table():
        packed = scenario_res_rv.get()
        if not packed:
            return render.DataGrid(pd.DataFrame())
        df = packed.get("table", pd.DataFrame()).copy()
        if df.empty:
            return render.DataGrid(pd.DataFrame())

        # redondeo amable
        for c in ["Real", "Escenario", "Delta", "Delta_%"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Delta_%" in df.columns:
            df["Delta_%"] = df["Delta_%"].round(3)
        return render.DataGrid(df)

    @output
    @render.plot
    def scenario_plot():
        packed = scenario_res_rv.get()
        if not packed:
            return None
        p = packed.get("plot")
        if not p:
            return None

        import matplotlib.pyplot as plt

        dates = p["dates"]
        y_true = p["y_true"]
        y_pred = p["y_pred"]

        fig, ax = plt.subplots()
        ax.plot(dates, y_true, marker="o", label="Real (target)")
        ax.plot(dates, y_pred, marker="o", label="Escenario (pred)")
        ax.set_title("Escenario vs Real (fechas editadas)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.legend()
        fig.autofmt_xdate()
        return fig

    @output
    @render.ui
    def kpi_ui():
        packed = scenario_res_rv.get()
        if not packed:
            return ui.div()
        res = packed.get("raw") or {}
        return ui.div(
            _metric_pill("MAPE", float(res.get("mape", 0) or 0)),
            _metric_pill("RMSE", float(res.get("rmse", 0) or 0)),
            _metric_pill("MAE", float(res.get("mae", 0) or 0)),
            ui.tags.span(f"Puntos editados: {len(packed.get('edited_dates', []) or [])}", class_="selection-pill"),
        )
