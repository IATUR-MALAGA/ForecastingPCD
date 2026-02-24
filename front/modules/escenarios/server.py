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

    def is_monthly(temp: str) -> bool:
        t = (temp or "").lower()
        return ("mes" in t) or ("mens" in t) or ("monthly" in t)

    def norm_dt(dt: pd.Timestamp, temp: str) -> pd.Timestamp:
        """Normaliza timestamp según temporalidad (inicio de mes o inicio de día)."""
        if pd.isna(dt):
            return pd.NaT
        return dt.to_period("M").to_timestamp(how="start") if is_monthly(temp) else dt.normalize()

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

    def _parse_forecast_response(resp: dict):
        df = pd.DataFrame(resp.get("df") or [])
        if df.empty:
            return None

        y_col = resp["y_col"]
        n_obs = int(resp["n_obs"])
        h = int(resp["horizon"])

        future = df.iloc[n_obs : n_obs + h]
        pred_vals = resp["y_forecast"]
        pred_series = pd.Series(pred_vals, index=future.index, name="Prediction")
        return df, y_col, future, h, pred_vals, pred_series

    def _build_pred_df(future: pd.DataFrame, pred_vals, date_fmt: str = "%d-%m-%Y") -> pd.DataFrame:
        # igual que tu módulo de predicciones
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
        pred_df["Fecha"] = pd.to_datetime(pred_df["Fecha"], errors="coerce").dt.strftime(date_fmt)
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
        exogs = fut_exogs()
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
            fut_horizon(),
            repr(fut_matrix_values()),
            repr(selected_filters_by_var()),
        )

    @reactive.Effect
    def _invalidate_future_results_on_change():
        sig = fut_signature()
        if sig is None:
            return
        last = last_sig_rv.get()
        if last is not None and sig != last:
            scenario_res_rv.set(None)
            scenario_err_rv.set(None)

    # ------------------------
    # UI: tabla editable (2)
    # ------------------------
    @output
    @render.ui
    def esc_future_exog_table():
        if current_step.get() != 4:
            return ui.div()

        exogs = fut_exogs()
        h = fut_horizon()
        idx = fut_future_index()
        temp = target_temporalidad()

        if not exogs:
            return ui.tags.div(
                ui.tags.b("No hay exógenas seleccionadas."),
                ui.tags.span(" Vuelve al Panel 2 para seleccionar al menos una."),
                style="color:#6b7280;",
            )

        if idx.empty:
            return ui.tags.div(
                ui.tags.b("No se pudo inferir el calendario futuro."),
                ui.tags.span(" Revisa el rango de fechas del objetivo."),
                style="color:#6b7280;",
            )

        header_cells = [ui.tags.th("Exógena", style="position:sticky; left:0; background:#fff;")]
        for k in range(1, h + 1):
            header_cells.append(
                ui.tags.th(
                    ui.tags.div(f"P{k}", style="font-weight:700;"),
                    ui.tags.div(_dt_label(idx[k-1], temp), style="font-size:12px; color:#6b7280; margin-top:2px;"),
                )
            )

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
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(cid, label="", value=None, step=0.01),  # ✅ sin ns()
                        style="min-width:120px;",
                    )
                )
            body_rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.div(
                ui.tags.b("2) Valores futuros de exógenas"),
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

        # 🔎 Debug inmediato: si esto no aparece, el evento no está entrando
        try:
            n_clicks = int(input.esc_fut_calc() or 0)
        except Exception:
            n_clicks = -1

        scenario_err_rv.set(f"Calculando… (click={n_clicks})")
        scenario_res_rv.set(None)

        try:
            target = target_var_rv.get()
            exogs = fut_exogs()
            h = fut_horizon()
            model = fut_model()
            filters = selected_filters_by_var()

            if not target:
                scenario_err_rv.set("No hay variable objetivo seleccionada.")
                last_sig_rv.set(fut_signature())
                return

            if not exogs:
                scenario_err_rv.set("No hay exógenas seleccionadas (Panel 2).")
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
                    future_values.append({"var": ex, "date": date_str, "value": float(mat[ex][j])})

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
                payload.update({"use_target_lags": True, "max_lag": 12, "recursive_forecast": True})

            resp = runner(payload)

            parsed = _parse_forecast_response(resp)
            if parsed is None:
                scenario_err_rv.set("El backend devolvió df vacío (resp['df']).")
                last_sig_rv.set(fut_signature())
                return

            df, y_col, future, h2, pred_vals, pred_series = parsed

            fig = plot_predictions(
                df=df,
                pred=pred_series,
                title=("Escenario futuro (SARIMAX)" if model == "sarimax" else "Escenario futuro (XGBoost)"),
                ylabel="Valores",
                xlabel="Fecha",
                column_y=y_col,
                periodos_a_predecir=h2,
                holidays_col=None,
            )

            pred_df = _build_pred_df(future, pred_vals, date_fmt="%d-%m-%Y")

            scenario_res_rv.set({
                "model": model,
                "fig": fig,
                "pred_df": pred_df,
                "mape": resp.get("mape"),
                "rmse": resp.get("rmse"),
                "mae": resp.get("mae"),
            })
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
        if current_step.get() != 4:
            return ui.div()

        res = scenario_res_rv.get()
        err = scenario_err_rv.get()

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
                    value=2,
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
                ui.input_action_button("esc_fut_calc", "Calcular", class_="btn-primary"),
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
                ui.tags.span("rellena la tabla y pulsa «Calcular».", style="color:#6b7280;"),
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
            ui.output_ui("esc_future_exog_table"),
            model_box,
            status,
            outputs,
            footer,
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _go_step_3_from_4():
        current_step.set(3)        