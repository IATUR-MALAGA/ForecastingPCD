import asyncio
import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
import httpx
from back.models.utils.models_graph import (
    build_interactive_plot_html,
    compute_time_axis_bounds,
    ensure_datetime_index,
)
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
    fmt_num,
    fmt_date_by_temporality as _fmt_date_temp,
    group_by_category,
    humanize_error,
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
    saved_horizon_rv = reactive.Value(2)  # horizonte guardado del panel 4
    saved_fut_cell_values_rv = reactive.Value(
        {}
    )  # valores de celdas exógenas guardados
    fut_gen_rv = reactive.Value(
        0
    )  # generación de inputs de celdas (cambia al entrar al panel 4)
    _SPINNER_ID = "_esc_spinner"

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

    def _build_table_html(df: pd.DataFrame, signed_cols: tuple[str, ...] = ()) -> ui.Tag:
        def _parse_signed_value(val):
            if val is None or pd.isna(val):
                return None

            s = str(val).strip()
            if not s:
                return None

            # Limpieza básica
            s = (
                s.replace("%", "")
                .replace("\xa0", "")
                .replace(" ", "")
                .replace("−", "-")   # minus unicode
            )

            # Normalizar separadores decimales/miles
            if "," in s and "." in s:
                # Si la coma va al final, asumimos formato europeo: 1.234,56
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    # formato tipo 1,234.56
                    s = s.replace(",", "")
            elif "," in s:
                # formato tipo 12,34
                s = s.replace(",", ".")

            try:
                return float(s)
            except (TypeError, ValueError):
                return None

        header_cells = [
            ui.tags.th(
                c,
                style="padding:6px 10px; border-bottom:2px solid #e5e7eb; text-align:left;",
            )
            for c in df.columns
        ]

        rows = []
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                style = "padding:6px 10px; border-bottom:1px solid #f0f0f0;"

                if col in signed_cols:
                    num = _parse_signed_value(val)
                    if num is not None:
                        color = "#dc2626" if num < 0 else "#16a34a" if num > 0 else "#475569"
                        style += f" color:{color}; font-weight:600;"

                cells.append(ui.tags.td("" if pd.isna(val) else str(val), style=style))
            rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*header_cells)),
                ui.tags.tbody(*rows),
                style="border-collapse:collapse; width:100%; font-size:0.9rem;",
            ),
            style="overflow:auto; max-height:420px;",
        )

    def _scenario_table_decimals() -> int:
        """
        Lee metadata.decimales (binario):
          - 1 => mostrar 2 decimales
          - 0 => mostrar 0 decimales
        Incluye debug por consola para validar lectura.
        """
        target = target_var_rv.get()
        meta = {}
        raw = None
        flag = None

        if target:
            try:
                meta = cache.get_meta(target) or {}
                raw = meta.get("decimales", None)
            except Exception as e:
                print(
                    f"[DEBUG decimales][escenarios] error leyendo metadata para {target!r}: {type(e).__name__}: {e}"
                )

        if isinstance(raw, bool):
            flag = 1 if raw else 0
        elif raw is not None:
            txt = str(raw).strip().lower()
            if txt in ("1", "true", "t", "si", "sí", "y", "yes"):
                flag = 1
            elif txt in ("0", "false", "f", "no", "n"):
                flag = 0
            else:
                try:
                    flag = 1 if int(float(txt)) == 1 else 0
                except Exception:
                    flag = None

        decimals = 2 if flag == 1 else 0 if flag == 0 else 2

        print(
            f"[DEBUG decimales][escenarios] target={target!r} raw={raw!r} flag={flag!r} applied_decimals={decimals} meta_keys={list(meta.keys()) if isinstance(meta, dict) else None}"
        )
        return decimals

    def _signed_fmt(value, digits: int = 4, suffix: str = "") -> str:
        if value is None or pd.isna(value):
            return ""
        prefix = "+" if float(value) > 0 else ""
        return f"{prefix}{fmt_num(float(value), digits, suffix)}"

    def _build_future_plot(df, pred, title, ylabel, xlabel, column_y, trace_name):
        df_plot = ensure_datetime_index(df)
        pred_index = pd.to_datetime(pred.index, errors="coerce")
        pred_series = pd.Series(pred.values, index=pred_index, name=trace_name)
        pred_series = pred_series[~pd.isna(pred_series.index)]

        all_index = pd.DatetimeIndex(df_plot.index.tolist() + pred_series.index.tolist())
        x_min, x_max = compute_time_axis_bounds(all_index)
        customdata = [
            [ts.strftime("%d-%m-%Y"), None, fmt_num(val, 4), None, None, trace_name.lower()]
            for ts, val in zip(pred_series.index, pred_series.values)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[column_y],
                mode="lines",
                name="Real",
                line={"color": "#2563eb", "width": 2},
                hovertemplate="Fecha: %{x|%d-%m-%Y}<br>Valor real: %{y:,.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pred_series.index,
                y=pred_series.values,
                mode="lines+markers",
                name=trace_name,
                line={"color": "#e11d48", "width": 3},
                marker={"color": "#e11d48", "size": 9},
                customdata=customdata,
                hovertemplate=f"Fecha: %{{x|%d-%m-%Y}}<br>{trace_name}: %{{y:,.4f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title,
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
            hovermode="closest",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend={"orientation": "h", "y": 1.08, "x": 0.01},
            xaxis={
                "title": xlabel,
                "range": [x_min, x_max],
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
            yaxis={
                "title": ylabel,
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
        )
        return fig

    def _build_past_plot(
        df,
        pred,
        title,
        ylabel,
        xlabel,
        column_y,
        window_end,
        actual_values=None,
    ):
        df_plot = ensure_datetime_index(df)
        pred_index = pd.to_datetime(pred.index, errors="coerce")
        pred_series = pd.Series(pred.values, index=pred_index, name="Escenario")
        pred_series = pred_series[~pd.isna(pred_series.index)]
        if actual_values is not None:
            actual_on_pred = pd.Series(
                pd.to_numeric(list(actual_values)[: len(pred_series)], errors="coerce"),
                index=pred_series.index,
                name="Valor real",
            )
        else:
            real_series = pd.to_numeric(df_plot[column_y], errors="coerce")
            if real_series.index.has_duplicates:
                real_series = real_series.groupby(level=0).mean()
            actual_on_pred = real_series.reindex(pred_series.index)

        modified_mask = pred_series.index <= pd.to_datetime(window_end, errors="coerce")
        impact_mask = pred_series.index > pd.to_datetime(window_end, errors="coerce")

        def _customdata(segment_series, actual, segment_name):
            diff = segment_series - actual
            diff_pct = ((segment_series - actual) / actual.replace(0, float("nan"))) * 100
            return [
                [
                    ts.strftime("%d-%m-%Y"),
                    fmt_num(real, 4) if pd.notna(real) else "",
                    fmt_num(pred_val, 4),
                    _signed_fmt(diff_val, 4),
                    _signed_fmt(diff_pct_val, 2, "%"),
                    segment_name,
                ]
                for ts, real, pred_val, diff_val, diff_pct_val in zip(
                    segment_series.index,
                    actual.values,
                    segment_series.values,
                    diff.values,
                    diff_pct.values,
                )
            ]

        all_index = pd.DatetimeIndex(df_plot.index.tolist() + pred_series.index.tolist())
        x_min, x_max = compute_time_axis_bounds(all_index)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[column_y],
                mode="lines",
                name="Valor real",
                line={"color": "#2563eb", "width": 2},
                hovertemplate="Fecha: %{x|%d-%m-%Y}<br>Valor real: %{y:,.4f}<extra></extra>",
            )
        )

        modified_series = pred_series[modified_mask]
        modified_actual = actual_on_pred[modified_mask]
        if not modified_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=modified_series.index,
                    y=modified_series.values,
                    mode="lines+markers",
                    name="Escenario modificado",
                    line={"color": "#f97316", "width": 3},
                    marker={"color": "#f97316", "size": 9},
                    customdata=_customdata(modified_series, modified_actual, "modificado"),
                    hovertemplate=(
                        "Fecha: %{x|%d-%m-%Y}<br>"
                        "Escenario: %{customdata[2]}<br>"
                        "Valor real: %{customdata[1]}<br>"
                        "% Diferencia: %{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )

        impact_series = pred_series[impact_mask]
        impact_actual = actual_on_pred[impact_mask]
        if not modified_series.empty and not impact_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=[modified_series.index[-1], impact_series.index[0]],
                    y=[modified_series.iloc[-1], impact_series.iloc[0]],
                    mode="lines",
                    name="Conexión impacto",
                    line={"color": "#7c3aed", "width": 3},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if not impact_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=impact_series.index,
                    y=impact_series.values,
                    mode="lines+markers",
                    name="Impacto posterior",
                    line={"color": "#7c3aed", "width": 3},
                    marker={"color": "#7c3aed", "size": 8},
                    customdata=_customdata(impact_series, impact_actual, "posterior"),
                    hovertemplate=(
                        "Fecha: %{x|%d-%m-%Y}<br>"
                        "Escenario: %{customdata[2]}<br>"
                        "Valor real: %{customdata[1]}<br>"
                        "% Diferencia: %{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
            hovermode="closest",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend={"orientation": "h", "y": 1.08, "x": 0.01},
            xaxis={
                "title": xlabel,
                "range": [x_min, x_max],
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
            yaxis={
                "title": ylabel,
                "showgrid": True,
                "gridcolor": "#e5e7eb",
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": "#94a3b8",
                "spikedash": "dot",
            },
        )
        return fig

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
        # En paso 0 no mostramos indicador de pasos
        if step == 0 or stype not in ("futuro", "pasado"):
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
    @reactive.Effect
    def _init_target_var():
        if target_var_rv.get() is None:
            grouped = group_by_category(catalog_entries)
            all_names = [n for names in grouped.values() for n in names]
            if all_names:
                target_var_rv.set(all_names[0])

    @output
    @render.ui
    def step_panel_1():
        if current_step.get() != 1 or scenario_type_rv.get() not in (
            "futuro",
            "pasado",
        ):
            return ui.div()

        grouped = group_by_category(catalog_entries)
        all_names = [n for names in grouped.values() for n in names]

        selected = target_var_rv.get()

        panels = []
        for cat, names in grouped.items():
            blocks = []
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

                meta = cache.get_meta(name)
                temporalidad = _fmt(meta.get("temporalidad"))
                granularidad = _fmt(meta.get("granularidad"))
                unidad_medida = _fmt(meta.get("unidad_medida"))
                fuente = _fmt(meta.get("fuente"))
                descripcion = _fmt(meta.get("descripcion"))

                blocks.append(
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_action_button(
                                btn_id,
                                name,
                                class_=(
                                    "var-pick is-selected" if selected == name else "var-pick"
                                ),
                            ),
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

            panels.append(
                ui.accordion_panel(
                    cat,
                    ui.div(*blocks),
                    value=_slug(cat),
                )
            )

        return ui.div(
            PANEL_STYLES,
            ui.div(
                ui.tags.div("\U0001f3af", style="font-size:2.5rem; margin-bottom:0.5rem;"),
                ui.h3(
                    "Seleccionar variable objetivo",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Elige la variable que deseas predecir. Esta será la variable dependiente "
                    "del modelo, es decir, el valor que los algoritmos intentarán estimar a partir "
                    "de las variables predictoras que selecciones en el siguiente paso.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
            ui.div(
                ui.tags.span("\u2705 Seleccionada: ", style="font-weight:600;"),
                ui.tags.span(selected or "—"),
                class_="selection-pill",
            ),
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
        if current_step.get() != 2 or scenario_type_rv.get() not in (
            "futuro",
            "pasado",
        ):
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
                        "compat-badge compat-yes" if ok else "compat-badge compat-no"
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
            ui.div(
                ui.tags.div("\U0001f4ca", style="font-size:2.5rem; margin-bottom:0.5rem;"),
                ui.h3(
                    "Seleccionar variables predictoras",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Elige las variables exógenas que alimentarán el modelo. Solo se pueden "
                    "seleccionar aquellas que sean compatibles en temporalidad y rango con la "
                    "variable objetivo elegida en el paso anterior.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
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
        if current_step.get() != 3 or scenario_type_rv.get() not in (
            "futuro",
            "pasado",
        ):
            return ui.div()

        vars_sel = vars_to_config()
        if not vars_sel:
            return ui.div(
                PANEL_STYLES,
                ui.h3("Panel 3: configurar filtros"),
                ui.p("No hay variables seleccionadas para configurar."),
            )

        target_var = target_var_rv.get()
        target_start, target_end = target_selected_range()
        target_meta = cache.get_meta(target_var) if target_var else {}
        target_temporality = target_meta.get("temporalidad")

        # --- Build body for each variable ---
        def _var_body(item):
            pretty = item["pretty"]
            table = item["table"]
            is_target = item.get("is_target", False)
            filtros = cache.get_filters(table)

            if not filtros:
                if is_target:
                    return ui.p(
                        "Sin filtros configurados en tbl_admin_filtros para esta variable/tabla."
                    )
                else:
                    return ui.div()
            else:
                controls = []

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
                    choices = cache.get_distinct_complete("IA", t, col)
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

                return (
                    ui.div(*controls) if controls else ui.p("Sin filtros disponibles.")
                )

        # --- Separate target vs predictors ---
        target_item = None
        predictor_items = []
        for item in vars_sel:
            if item.get("is_target"):
                target_item = item
            else:
                predictor_items.append(item)

        # Target box
        target_box_content = (
            ui.div(
                ui.tags.div(
                    ui.tags.span(target_item["pretty"], style="font-weight:700; font-size:1rem;"),
                    style="margin-bottom:8px;",
                ),
                _var_body(target_item),
            )
            if target_item
            else ui.p("No se ha seleccionado variable objetivo.")
        )

        target_box = ui.tags.div(
            ui.tags.div(
                "🎯 Variable objetivo",
                style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b;",
            ),
            target_box_content,
            style=(
                "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
            ),
        )

        # Predictors box
        if predictor_items:
            pred_panels = []
            for item in predictor_items:
                pred_panels.append(
                    ui.accordion_panel(
                        ui.tags.strong(item["pretty"]),
                        _var_body(item),
                        value=_slug(item["table"]),
                    )
                )

            predictors_box = ui.tags.div(
                ui.tags.div(
                    "📊 Variables predictoras",
                    ui.tooltip(
                        ui.tags.span(
                            ui.HTML(ICON_SVG_INFO),
                        ),
                        "Cada variable predictora se ajustará automáticamente al rango temporal de la variable objetivo.",
                    ),
                    style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b; display:flex; align-items:center; gap:4px;",
                ),
                ui.accordion(*pred_panels, id="esc_acc_filters_preds", open=True, multiple=True),
                style=(
                    "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                    "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
                ),
            )
        else:
            predictors_box = ui.tags.div(
                ui.tags.div(
                    "📊 Variables predictoras",
                    style="font-size:1.1rem; font-weight:700; margin-bottom:12px; color:#1e293b;",
                ),
                ui.p("No se han seleccionado variables predictoras.", style="color:#6b7280;"),
                style=(
                    "padding:16px; border:1px solid #d0d7de; border-radius:12px; "
                    "background:#ffffff; flex:1 1 0; min-width:0; align-self:flex-start;"
                ),
            )

        return ui.div(
            PANEL_STYLES,
            ui.div(
                ui.tags.div("\U0001f527", style="font-size:2.5rem; margin-bottom:0.5rem;"),
                ui.h3(
                    "Configurar filtros",
                    style="text-align:center; font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;",
                ),
                ui.tags.p(
                    "Para cada variable se muestran sus filtros configurados. "
                    "En exógenas, el rango temporal se ajusta automáticamente al rango "
                    "elegido en la variable objetivo.",
                    style="text-align:center; color:#475569; max-width:600px; margin:0 auto 1.5rem; line-height:1.6;",
                ),
                style="text-align:center; margin-bottom:1rem;",
            ),
            ui.tags.div(
                target_box,
                predictors_box,
                style="display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;",
            ),
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
                        saved[input_id] = (
                            list(vals)
                            if isinstance(vals, (list, tuple))
                            else [str(vals)]
                        )
        saved_filter_values_rv.set(saved)

        saved_horizon_rv.set(2)
        saved_fut_cell_values_rv.set({})
        fut_gen_rv.set(fut_gen_rv.get() + 1)
        scenario_res_rv.set(None)
        scenario_err_rv.set(None)
        current_step.set(4)

    # =====================================================================
    # Panel 4: Escenarios FUTUROS (exógenas inventadas por el usuario)
    # =====================================================================

    # --- Resultado / error (reusa scenario_res_rv si ya lo tienes) ---
    scenario_err_rv = reactive.Value(None)  # error legible

    # ------------------------
    # Helpers Panel 4 (puros)
    # ------------------------
    def _cell_id(exog_name: str, k: int, gen: int = 0) -> str:
        # id estable por exógena + periodo + generación (cambia al resetear el panel)
        return stable_id("esc_fut_val", f"{exog_name}__P{k}__G{gen}")

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
        gen = fut_gen_rv.get()
        out: dict[str, list[float | None]] = {}
        for ex in exogs:
            row = []
            for k in range(1, h + 1):
                cid = _cell_id(ex, k, gen)
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
    @reactive.event(input.esc_fut_active_exogs)
    def _save_cells_before_exog_toggle():
        if current_step.get() != 4:
            return

        with reactive.isolate():
            exogs = list(fut_exogs())
            h = int(fut_horizon())
            gen = fut_gen_rv.get()
            saved = dict(saved_fut_cell_values_rv.get())
            for ex in exogs:
                for k in range(1, h + 1):
                    cid = _cell_id(ex, k, gen)
                    if cid in input:
                        v = input[cid]()
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
            ui.tags.span(
                "Selecciona las exógenas que quieres incluir en el escenario futuro.",
                style="font-size:12px; color:#6b7280; margin-bottom:8px; display:block;",
            ),
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
                ui.tags.span(
                    " Marca al menos una exógena en el selector para continuar."
                ),
                style="color:#6b7280;",
            )

        if idx.empty:
            return ui.tags.div(
                ui.tags.b("No se pudo inferir el calendario futuro."),
                ui.tags.span(" Revisa el rango de fechas del objetivo."),
                style="color:#6b7280;",
            )

        header_cells = [
            ui.tags.th("Fecha", style="position:sticky; top:0; background:#f8fafc; z-index:2; border-bottom:2px solid #e2e8f0; padding:8px 12px;")
        ]
        for ex in exogs:
            header_cells.append(
                ui.tags.th(
                    ex,
                    style="position:sticky; top:0; background:#f8fafc; z-index:2; border-bottom:2px solid #e2e8f0; text-align:center; padding:8px 12px; min-width:110px;"
                )
            )

        _cell_saved = saved_fut_cell_values_rv.get()
        _gen = fut_gen_rv.get()

        body_rows = []
        for k in range(1, h + 1):
            cells = [
                ui.tags.td(
                    _dt_label(idx[k - 1], temp),
                    style="position:sticky; left:0; background:#fff; font-weight:600; white-space:nowrap; z-index:1; border-bottom:1px solid #e5e7eb; padding:8px 12px;",
                )
            ]
            for ex in exogs:
                cid = _cell_id(ex, k, _gen)
                _init_val = _cell_saved.get(cid, None)
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(
                            cid, label="", value=_init_val, step=0.01, width="100%"
                        ),
                        style="text-align:center; border-bottom:1px solid #e5e7eb; padding:4px 8px;",
                    )
                )
            body_rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.style(
                """
                .esc-fut-table td .form-group { margin-bottom: 0 !important; }
                .esc-fut-table td .shiny-input-container { margin-bottom: 0 !important; }
                """
            ),
            ui.tags.div(
                ui.tags.b("3) Valores futuros de exógenas activas"),
                ui.tags.span(" (rellena todas las celdas)"),
                style="margin-bottom:8px;",
            ),
            ui.tags.div(
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(*header_cells)),
                    ui.tags.tbody(*body_rows),
                    class_="esc-fut-table",
                    style="border-collapse:collapse; min-width:100%; background:#fff;",
                ),
                style="overflow:auto; width:100%; max-height:400px; border:1px solid #e5e7eb; border-radius:12px; padding:0; background:#fff;",
            ),
        )

    # ------------------------
    # Cálculo bajo demanda (3)
    # ------------------------
    @reactive.Effect
    @reactive.event(input.esc_fut_calc)
    async def _compute_future_scenario_on_click():
        if current_step.get() != 4:
            return
        if int(input.esc_fut_calc() or 0) == 0:
            return

        with reactive.isolate():
            _exogs_snap = fut_exogs()
            _h_snap = fut_horizon()
            _gen_snap = fut_gen_rv.get()
            _saved_snap = dict(saved_fut_cell_values_rv.get())
            for _ex in _exogs_snap:
                for _k in range(1, _h_snap + 1):
                    _cid = _cell_id(_ex, _k, _gen_snap)
                    if _cid in input:
                        _v = input[_cid]()
                        _saved_snap[_cid] = _v
            saved_fut_cell_values_rv.set(_saved_snap)
        # ─────────────────────────────────────────────────────────────────────

        scenario_err_rv.set(None)
        scenario_res_rv.set(None)

        # Leer valores reactivos en el hilo principal
        target = target_var_rv.get()
        exogs = fut_active_exogs()
        h = fut_horizon()
        model = fut_model()
        filters = selected_filters_by_var()
        sig = fut_signature()

        if not target:
            scenario_err_rv.set("No hay variable objetivo seleccionada.")
            last_sig_rv.set(sig)
            return

        if not exogs:
            scenario_err_rv.set("No hay exógenas activas en el Panel 4.")
            last_sig_rv.set(sig)
            return

        idx = fut_future_index()
        if idx.empty or len(idx) != h:
            scenario_err_rv.set("No pude construir el índice temporal futuro.")
            last_sig_rv.set(sig)
            return

        mat = fut_matrix_values()

        for ex in exogs:
            for k, v in enumerate(mat.get(ex, []), start=1):
                if v is None:
                    scenario_err_rv.set(f"Falta valor para '{ex}' en P{k}.")
                    last_sig_rv.set(sig)
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
            last_sig_rv.set(sig)
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

        # Insertar spinner (bypass flush, se envía inmediatamente al browser)
        _spinner_id = _SPINNER_ID
        ui.insert_ui(
            ui.tags.div(
                ui.tags.div(class_="graph-spinner"),
                ui.tags.div("Calculando escenario...", class_="graph-loading-text"),
                class_="graph-loading-container",
                id=_spinner_id,
            ),
            selector="#esc_result_area",
            where="afterBegin",
            immediate=True,
        )

        try:

            def _run_scenario():
                resp = runner(payload)
                parsed = _parse_forecast_response(resp, fallback_index=idx)
                if parsed is None:
                    return None
                df, y_col, future, h2, pred_vals, pred_series = parsed
                fig = _build_future_plot(
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
                    trace_name="Escenario",
                )
                pred_df = _build_pred_df(future, pred_vals, date_fmt="%d-%m-%Y")
                return {"model": model, "fig": fig, "pred_df": pred_df}

            result = await asyncio.to_thread(_run_scenario)

            if result is None:
                scenario_err_rv.set("El backend devolvió df vacío (resp['df']).")
            else:
                result["mode"] = "future"
                scenario_res_rv.set(result)
                scenario_err_rv.set(None)
            last_sig_rv.set(sig)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            if hasattr(e, "response"):
                try:
                    body = e.response.json()
                    if "detail" in body:
                        err_msg = str(body["detail"])
                except Exception:
                    pass
            scenario_err_rv.set(f"Fallo al calcular: {humanize_error(err_msg)}")
            last_sig_rv.set(sig)
            return
        finally:
            ui.remove_ui(selector=f"#{_spinner_id}", immediate=True)

    # ------------------------
    # Outputs
    # ------------------------
    @output
    @render.ui
    def esc_fut_plot():
        res = scenario_res_rv.get()
        if not res or res.get("mode") == "past":
            return ui.div()

        return ui.HTML(
            build_interactive_plot_html(
                res["fig"],
                session.ns("esc_fut_plot_widget"),
                session.ns("esc_fut_plot_click"),
            )
        )

    @output
    @render.ui
    def esc_fut_table():
        res = scenario_res_rv.get()
        if not res or res.get("mode") == "past" or res.get("pred_df") is None:
            return ui.div()

        decimals = _scenario_table_decimals()

        click = input.esc_fut_plot_click() if "esc_fut_plot_click" in input else None
        if click and click.get("scenario") is not None:
            click_y = pd.to_numeric(click.get("y"), errors="coerce")
            detail_df = pd.DataFrame(
                [
                    {
                        "Fecha": click.get("date_label") or "",
                        "Escenario": (
                            fmt_num(click_y, decimals)
                            if pd.notna(click_y)
                            else (click.get("scenario") or "")
                        ),
                    }
                ]
            )
            return ui.tags.div(
                ui.tags.div(
                    "Punto seleccionado",
                    style="font-weight:600; margin-bottom:8px;",
                ),
                _build_table_html(detail_df),
            )

        df = res["pred_df"].copy()
        if "Predicción" in df.columns:
            df["Predicción"] = pd.to_numeric(df["Predicción"], errors="coerce").apply(
                lambda v: fmt_num(v, decimals) if pd.notna(v) else v
            )
        return ui.tags.div(
            ui.tags.div(
                "Haz clic en un punto para ver su detalle.",
                style="color:#6b7280; font-size:0.9rem; margin-bottom:8px;",
            ),
            _build_table_html(df),
        )

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
                    ui.h5("Evolución temporal", style="margin:0 0 8px 0;"),
                    ui.output_ui("esc_fut_plot"),
                    style="padding:12px; border-radius:14px; flex:2 1 640px; min-width:520px;",
                ),
                ui.card(
                    ui.h5("Detalle / valores predichos", style="margin:0 0 8px 0;"),
                    ui.tags.div(
                        ui.output_ui("esc_fut_table"),
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
            ui.tags.div(
                status,
                outputs,
                id="esc_result_area",
            ),
            footer,
        )

    @reactive.Effect
    @reactive.event(input.esc_prev_4)
    def _go_step_3_from_4():
        scenario_res_rv.set(None)
        scenario_err_rv.set(None)
        current_step.set(3)

    # =====================================================================
    # Escenarios PASADOS — Panel 4
    # =====================================================================

    past_scenario_err_rv = reactive.Value(None)
    saved_past_cell_values_rv = reactive.Value({})  # valores de celdas exógenas guardados (pasado)

    base_info_rv = reactive.Value({})  # lookup de valores históricos por (exog, fecha)


    def _build_past_base_lookup(
        df: pd.DataFrame, exogs: list[str], temp: str
    ) -> dict[tuple[str, str], float]:
        if df.empty:
            return {}

        work = df.copy()
        work["__dt"] = normalize_dt_series(_extract_dates(work), temp)

        lookup: dict[tuple[str, str], float] = {}
        for ex in exogs:
            col_name = next(
                (c for c in (ex, _safe_alias(ex)) if c in work.columns),
                None,
            )
            if col_name is None:
                continue

            values = pd.to_numeric(work[col_name], errors="coerce")
            for dt, val in zip(work["__dt"], values):
                if pd.notna(dt) and pd.notna(val):
                    lookup[(ex, dt_str(dt, temp, kind="key"))] = float(val)

        return lookup


    @reactive.Effect
    async def _load_past_base_values():
        if current_step.get() != 4 or scenario_type_rv.get() != "pasado":
            base_info_rv.set({})
            return

        target = target_var_rv.get()
        exogs = past_active_exogs()
        filters = selected_filters_by_var()
        dates = past_window_dates()
        model = past_model()

        if not target or not exogs or dates.empty:
            base_info_rv.set({})
            return

        temp = target_temporalidad()
        ws_dt = pd.to_datetime(dates.min(), errors="coerce")
        we_dt = pd.to_datetime(dates.max(), errors="coerce")

        if pd.isna(ws_dt) or pd.isna(we_dt):
            base_info_rv.set({})
            return

        payload = {
            "target_var": target,
            "predictors": list(exogs),
            "filters_by_var": filters,
            "train_ratio": 0.70,
            "auto_params": False,
            "return_df": True,
            "horizon": len(dates),
            "scenario_mode": "past",
            "scenario_window": {
                "start": ws_dt.strftime("%Y-%m-%d"),
                "end": we_dt.strftime("%Y-%m-%d"),
            },
            "scenario_overrides": [],
            "scenario_future_values": [],
        }

        if model == "sarimax":
            payload.update({"s": 12})
        elif model == "xgboost":
            payload.update(
                {
                    "use_target_lags": True,
                    "max_lag": 12,
                    "recursive_forecast": True,
                }
            )

        runner = MODEL_RUNNERS.get(model) or sarimax_run

        try:
            def _load_base():
                resp = runner(payload)
                df_resp = pd.DataFrame(resp.get("df") or [])
                return _build_past_base_lookup(df_resp, exogs, temp)

            base_info_rv.set(await asyncio.to_thread(_load_base))
        except Exception:
            base_info_rv.set({})

    def _past_cell_id(exog_name: str, k: int) -> str:
        return stable_id("esc_past_val", f"{exog_name}__P{k}")

    # --- Reactive calcs (past) ---

    @reactive.Calc
    def _past_target_table() -> str:
        """Nombre real de la tabla de la variable objetivo."""
        target = target_var_rv.get()
        if not target:
            return ""
        return name_to_table.get(target, target)

    @reactive.Calc
    def _past_calendar_input_id() -> str | None:
        """ID del input generado por create_calendar_filter para el target."""
        table = _past_target_table()
        if not table:
            return None
        filtros = cache.get_filters(table)
        temp = detect_temporal_filters(filtros)
        if not temp["has_any"]:
            return None
        # create_calendar_filter genera un date_range si hay mes o dia,
        # o un selectize de anio si solo hay anio.
        if temp["mes"] or temp["dia"]:
            return stable_id("flt", f"{temp['table']}__date_range")
        if temp["anio"]:
            return stable_id("flt", f"{temp['table']}__anio")
        return None

    @reactive.Calc
    def past_exogs() -> list[str]:
        return list(predictors_rv.get() or [])

    @reactive.Calc
    def past_active_exogs() -> list[str]:
        exogs = past_exogs()
        if "esc_past_active_exogs" not in input:
            return exogs
        selected = input.esc_past_active_exogs() or []
        if isinstance(selected, str):
            selected = [selected]
        return [ex for ex in exogs if ex in set(selected)]

    @reactive.Calc
    def past_model() -> str:
        if "esc_past_model" not in input:
            return "sarimax"
        return input.esc_past_model() or "sarimax"

    @reactive.Calc
    def past_window_range():
        cal_id = _past_calendar_input_id()
        if cal_id is None or cal_id not in input:
            return (None, None)
        val = input[cal_id]()
        if val and len(val) >= 2:
            return (val[0], val[1])
        return (None, None)

    @reactive.Calc
    def past_window_dates() -> pd.DatetimeIndex:
        ws, we = past_window_range()
        if ws is None or we is None:
            return pd.DatetimeIndex([])
        temp = target_temporalidad()
        start_dt = pd.to_datetime(ws, errors="coerce")
        end_dt = pd.to_datetime(we, errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            return pd.DatetimeIndex([])
        if _is_monthly(temp):
            start_dt = start_dt.to_period("M").to_timestamp(how="start")
            end_dt = end_dt.to_period("M").to_timestamp(how="start")
            freq = "MS"
        else:
            start_dt = start_dt.normalize()
            end_dt = end_dt.normalize()
            freq = "D"
        return pd.date_range(start=start_dt, end=end_dt, freq=freq)

    @reactive.Calc
    def past_matrix_values() -> dict[str, list[float | None]]:
        exogs = past_active_exogs()
        dates = past_window_dates()
        out: dict[str, list[float | None]] = {}
        for ex in exogs:
            row = []
            for k in range(1, len(dates) + 1):
                cid = _past_cell_id(ex, k)
                v = input[cid]() if cid in input else None
                row.append(v)
            out[ex] = row
        return out

    def _save_past_cells():
        """Guarda valores actuales de celdas pasadas en saved_past_cell_values_rv."""
        with reactive.isolate():
            exogs = list(past_exogs())
            dates = past_window_dates()
            saved = dict(saved_past_cell_values_rv.get())
            for ex in exogs:
                for k in range(1, len(dates) + 1):
                    cid = _past_cell_id(ex, k)
                    if cid in input:
                        v = input[cid]()
                        saved[cid] = v
            saved_past_cell_values_rv.set(saved)

    @reactive.Effect
    @reactive.event(input.esc_past_active_exogs)
    def _save_past_cells_before_exog_toggle():
        if current_step.get() != 4 or scenario_type_rv.get() != "pasado":
            return
        _save_past_cells()

    # --- Exog selector (past) ---

    @output
    @render.ui
    def esc_past_exog_selector():
        if current_step.get() != 4 or scenario_type_rv.get() != "pasado":
            return ui.div()
        exogs = past_exogs()
        if not exogs:
            return ui.div()
        return ui.tags.div(
            ui.tags.b("2) Exógenas activas"),
            ui.tags.span(
                "Selecciona las exógenas que quieres incluir en el escenario futuro.",
                style="font-size:12px; color:#6b7280; margin-bottom:8px; display:block;",
            ),
            ui.input_checkbox_group(
                "esc_past_active_exogs",
                label="",
                choices={ex: ex for ex in exogs},
                selected=exogs,
                inline=False,
            ),
            ui.tags.span(
                "Desmarca una exógena para excluirla del escenario.",
                style="font-size:12px; color:#6b7280;",
            ),
            style=(
                "margin-top:10px; padding:10px 12px; border:1px solid #e5e7eb; "
                "border-radius:12px; background:#fff;"
            ),
        )

    # --- Editable table (past) ---

    @output
    @render.ui
    def esc_past_exog_table():
        if current_step.get() != 4 or scenario_type_rv.get() != "pasado":
            return ui.div()

        exogs = past_active_exogs()
        dates = past_window_dates()
        temp = target_temporalidad()
        _base_lookup = base_info_rv.get() or {}

        if not exogs:
            return ui.tags.div(
                ui.tags.b("No hay exógenas activas."),
                ui.tags.span(
                    " Marca al menos una exógena en el selector para poder editar valores."
                ),
                style="color:#6b7280; margin-top:8px;",
            )

        if dates.empty:
            return ui.tags.div(
                ui.tags.b("Selecciona un rango de fechas válido para ver la tabla."),
                style="color:#6b7280; margin-top:8px;",
            )

        h = len(dates)
        if h > 60:
            return ui.tags.div(
                ui.tags.b(f"Rango demasiado amplio ({h} periodos)."),
                ui.tags.span(
                    " Reduce el rango a un máximo de 60 periodos para editar valores."
                ),
                style="color:#991b1b; margin-top:8px;",
            )

        header_cells = [
            ui.tags.th("Fecha", style="position:sticky; top:0; background:#f8fafc; z-index:2; border-bottom:2px solid #e2e8f0; padding:8px 12px;")
        ]
        for ex in exogs:
            header_cells.append(
                ui.tags.th(
                    ex,
                    style="position:sticky; top:0; background:#f8fafc; z-index:2; border-bottom:2px solid #e2e8f0; text-align:center; padding:8px 12px; min-width:110px;"
                )
            )

        _cell_saved = saved_past_cell_values_rv.get()

        body_rows = []
        for k in range(1, h + 1):
            cells = [
                ui.tags.td(
                    _dt_label(dates[k - 1], temp),
                    style="position:sticky; left:0; background:#fff; font-weight:600; white-space:nowrap; z-index:1; border-bottom:1px solid #e5e7eb; padding:8px 12px;",
                )
            ]

            for ex in exogs:
                cid = _past_cell_id(ex, k)
                _init_val = _cell_saved.get(cid, None)

                prev_key = (ex, dt_str(dates[k - 1], temp, kind="key"))
                prev_val = _base_lookup.get(prev_key)

                cells.append(
                    ui.tags.td(
                        ui.tags.div(
                            {
                                "class": "esc-past-num-wrap" + (
                                    " has-value" if _init_val is not None else ""
                                )
                            },
                            ui.input_numeric(
                                cid,
                                label="",
                                value=_init_val,
                                step=0.01,
                                width="100%"
                            ),
                            ui.tags.span(
                                "" if prev_val is None else fmt_num(prev_val, 4),
                                class_="esc-past-num-ghost",
                            ),
                        ),
                        style="text-align:center; border-bottom:1px solid #e5e7eb; padding:4px 8px;",
                    )
                )

            body_rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            ui.tags.style(
                """
                .esc-past-num-wrap {
                    position: relative;
                    min-width: 100px;
                    margin: 0 auto;
                }

                .esc-past-num-wrap .form-group,
                .esc-past-num-wrap .shiny-input-container {
                    margin-bottom: 0 !important;
                }

                .esc-past-num-wrap label {
                    display: none !important;
                    margin: 0 !important;
                }

                .esc-past-num-wrap input[type="number"] {
                    position: relative;
                    z-index: 2;
                    background: transparent !important;
                    text-align: right;
                }

                .esc-past-num-ghost {
                    position: absolute;
                    left: auto;
                    right: 28px;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #94a3b8;
                    pointer-events: none;
                    z-index: 1;
                    white-space: nowrap;
                }

                .esc-past-num-wrap.has-value .esc-past-num-ghost {
                    display: none;
                }
                """
            ),
            ui.tags.script(
                """
                if (!window.__escPastGhostInit) {
                    window.__escPastGhostInit = true;

                    document.addEventListener("input", function(e) {
                        const el = e.target;
                        if (!el || el.type !== "number") return;

                        const wrap = el.closest(".esc-past-num-wrap");
                        if (!wrap) return;

                        if (el.value === "" || el.value === null) {
                            wrap.classList.remove("has-value");
                        } else {
                            wrap.classList.add("has-value");
                        }
                    });
                }

                setTimeout(function() {
                    document.querySelectorAll(".esc-past-num-wrap input[type='number']").forEach(function(el) {
                        const wrap = el.closest(".esc-past-num-wrap");
                        if (!wrap) return;

                        if (el.value === "" || el.value === null) {
                            wrap.classList.remove("has-value");
                        } else {
                            wrap.classList.add("has-value");
                        }
                    });
                }, 0);
                """
            ),
            ui.tags.div(
                ui.tags.b("3) Valores modificados de exógenas"),
                ui.tags.span(
                    " (el valor histórico aparece en gris dentro del input; "
                    "si escribes un nuevo valor, sustituye al histórico)"
                ),
                style="margin-bottom:8px;",
            ),
            ui.tags.div(
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(*header_cells)),
                    ui.tags.tbody(*body_rows),
                    style="border-collapse:collapse; min-width:100%; background:#fff;",
                ),
                style=(
                    "overflow:auto; width:100%; max-height:400px; border:1px solid #e5e7eb; "
                    "border-radius:12px; padding:0; background:#fff;"
                ),
            ),
        )

    # --- Compute past scenario ---

    @reactive.Effect
    @reactive.event(input.esc_past_calc)
    async def _compute_past_scenario():
        if current_step.get() != 4 or scenario_type_rv.get() != "pasado":
            return
        if int(input.esc_past_calc() or 0) == 0:
            return

        past_scenario_err_rv.set(None)
        scenario_res_rv.set(None)

        # Guardar valores de celdas antes de calcular
        _save_past_cells()

        target = target_var_rv.get()
        exogs = past_active_exogs()
        model = past_model()
        filters = selected_filters_by_var()
        ws, we = past_window_range()
        dates = past_window_dates()

        if not target:
            past_scenario_err_rv.set("No hay variable objetivo seleccionada.")
            return
        if not ws or not we:
            past_scenario_err_rv.set("Selecciona un rango de fechas.")
            return
        if dates.empty:
            past_scenario_err_rv.set(
                "El rango de fechas seleccionado no genera periodos."
            )
            return

        # Build overrides: only for cells with a value entered
        mat = past_matrix_values()
        overrides = []
        for ex in exogs:
            vals = mat.get(ex, [])
            for k, v in enumerate(vals):
                if v is not None:
                    date_str = pd.to_datetime(dates[k]).strftime("%Y-%m-%d")
                    overrides.append(
                        {
                            "var": ex,
                            "op": "set",
                            "value": float(v),
                            "start": date_str,
                            "end": date_str,
                        }
                    )

        runner = MODEL_RUNNERS.get(model)
        if runner is None:
            past_scenario_err_rv.set(f"Modelo no soportado: {model}")
            return

        # Normalize window dates to match past_window_dates() logic
        temp = target_temporalidad()
        ws_dt = pd.to_datetime(ws, errors="coerce")
        we_dt = pd.to_datetime(we, errors="coerce")
        if _is_monthly(temp):
            ws_dt = ws_dt.to_period("M").to_timestamp(how="start")
            we_dt = we_dt.to_period("M").to_timestamp(how="end")
        else:
            ws_dt = ws_dt.normalize()
            we_dt = we_dt.normalize()
        ws_str = ws_dt.strftime("%Y-%m-%d")
        we_str = we_dt.strftime("%Y-%m-%d")

        payload = {
            "target_var": target,
            "predictors": list(exogs),
            "filters_by_var": filters,
            "train_ratio": 0.70,
            "auto_params": True,
            "return_df": True,
            "horizon": len(dates),
            "scenario_mode": "past",
            "scenario_window": {"start": ws_str, "end": we_str},
            "scenario_overrides": overrides,
            "scenario_future_values": [],
        }

        if model == "sarimax":
            payload.update({"s": 12})
        elif model == "xgboost":
            payload.update(
                {
                    "use_target_lags": True,
                    "max_lag": 12,
                    "recursive_forecast": True,
                }
            )

        # Insertar spinner
        _spinner_id = _SPINNER_ID
        ui.insert_ui(
            ui.tags.div(
                ui.tags.div(class_="graph-spinner"),
                ui.tags.div("Calculando escenario...", class_="graph-loading-text"),
                class_="graph-loading-container",
                id=_spinner_id,
            ),
            selector="#esc_past_result_area",
            where="afterBegin",
            immediate=True,
        )

        try:

            def _run_past_scenario():
                resp = runner(payload)

                df_resp = pd.DataFrame(resp.get("df") or [])
                if df_resp.empty:
                    return None, "El backend devolvió un dataframe vacío."

                y_col = resp["y_col"]
                y_forecast = list(resp.get("y_forecast") or [])
                y_true = list(resp.get("y_true") or [])
                horizon_resp = int(resp.get("horizon", len(dates)))

                scenario_dates = pd.to_datetime(df_resp.get("__dt"), errors="coerce")
                scenario_dates = scenario_dates[scenario_dates >= ws_dt]
                pred_index = pd.DatetimeIndex(scenario_dates)
                m = min(len(y_forecast), len(pred_index))
                y_forecast = y_forecast[:m]
                pred_index = pred_index[:m]
                if y_true:
                    y_true = y_true[:m]

                pred_series = pd.Series(y_forecast, index=pred_index, name="Prediction")

                fig = _build_past_plot(
                    df=df_resp,
                    pred=pred_series,
                    title=f"Escenario pasado ({model.upper()})",
                    ylabel="Valores",
                    xlabel="Fecha",
                    column_y=y_col,
                    window_end=we_dt,
                    actual_values=y_true,
                )

                temp_inner = target_temporalidad()
                date_fmt = "%m-%Y" if _is_monthly(temp_inner) else "%d-%m-%Y"
                results_data: dict = {
                    "Fecha": [d.strftime(date_fmt) for d in pred_index],
                    "Escenario": y_forecast,
                }
                if y_true:
                    results_data["Valor real"] = y_true

                pred_df = pd.DataFrame(results_data)

                if "Valor real" in pred_df.columns and "Escenario" in pred_df.columns:
                    real = pd.to_numeric(pred_df["Valor real"], errors="coerce")
                    pred = pd.to_numeric(pred_df["Escenario"], errors="coerce")
                    pred_df["Diferencia"] = pred - real
                    pred_df["% Diferencia"] = ((pred - real) / real.replace(0, float("nan"))) * 100

                return {
                    "model": model,
                    "fig": fig,
                    "pred_df": pred_df,
                    "mode": "past",
                }, None

            result, err_msg = await asyncio.to_thread(_run_past_scenario)

            if result is None:
                past_scenario_err_rv.set(err_msg or "Error desconocido.")
            else:
                scenario_res_rv.set(result)
                past_scenario_err_rv.set(None)

        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            past_scenario_err_rv.set(f"Fallo al calcular: {humanize_error(detail)}")
        except Exception as e:
            past_scenario_err_rv.set(f"Fallo al calcular: {humanize_error(str(e))}")
        finally:
            ui.remove_ui(selector=f"#{_spinner_id}", immediate=True)

    # --- Outputs (past) ---

    @output
    @render.ui
    def esc_past_plot():
        res = scenario_res_rv.get()
        if not res or res.get("mode") != "past":
            return ui.div()
        return ui.HTML(
            build_interactive_plot_html(
                res["fig"],
                session.ns("esc_past_plot_widget"),
                session.ns("esc_past_plot_click"),
            )
        )

    @output
    @render.ui
    def esc_past_table():
        res = scenario_res_rv.get()
        if not res or res.get("mode") != "past" or res.get("pred_df") is None:
            return ui.div()

        decimals = _scenario_table_decimals()

        click = input.esc_past_plot_click() if "esc_past_plot_click" in input else None
        if click and click.get("scenario") is not None:
            click_y = pd.to_numeric(click.get("y"), errors="coerce")
            detail_df = pd.DataFrame(
                [
                    {
                        "Fecha": click.get("date_label") or "",
                        "Valor real": click.get("real") or "",
                        "Escenario": (
                            fmt_num(click_y, decimals)
                            if pd.notna(click_y)
                            else (click.get("scenario") or "")
                        ),
                        "Diferencia": click.get("diff") or "",
                        "% Diferencia": click.get("diff_pct") or "",
                    }
                ]
            )
            return ui.tags.div(
                ui.tags.div(
                    "Punto seleccionado",
                    style="font-weight:600; margin-bottom:8px;",
                ),
                _build_table_html(detail_df, signed_cols=("Diferencia", "% Diferencia")),
            )

        df = res["pred_df"].copy()
        for col in ["Escenario", "Valor real"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").apply(
                    lambda v: fmt_num(v, decimals) if pd.notna(v) else v
                )
        if "Diferencia" in df.columns:
            df["Diferencia"] = pd.to_numeric(df["Diferencia"], errors="coerce").apply(
                lambda v: _signed_fmt(v, decimals) if pd.notna(v) else v
            )
        if "% Diferencia" in df.columns:
            df["% Diferencia"] = pd.to_numeric(df["% Diferencia"], errors="coerce").apply(
                lambda v: _signed_fmt(v, 2, "%") if pd.notna(v) else v
            )

        return ui.tags.div(
            ui.tags.div(
                "Haz clic en un punto del escenario para ver su detalle.",
                style="color:#6b7280; font-size:0.9rem; margin-bottom:8px;",
            ),
            _build_table_html(df, signed_cols=("Diferencia", "% Diferencia")),
        )

    # --- Panel UI (past) ---

    @output
    @render.ui
    def step_panel_pasado():
        if scenario_type_rv.get() != "pasado" or current_step.get() != 4:
            return ui.div()

        target = target_var_rv.get()
        target_start_raw, target_end_raw = cache.get_date_range(target or "")
        temp = target_temporalidad()

        min_dt = (
            pd.to_datetime(target_start_raw)
            if target_start_raw
            else pd.Timestamp("2000-01-01")
        )
        max_dt = (
            pd.to_datetime(target_end_raw) if target_end_raw else pd.Timestamp.now()
        )
        min_date = min_dt.date()
        max_date = max_dt.date()

        # Sensible default: last few periods of the historical range
        if _is_monthly(temp):
            default_start = (max_dt - pd.DateOffset(months=6)).date()
        else:
            default_start = (max_dt - pd.Timedelta(days=30)).date()
        default_start = max(min_date, default_start)

        # --- 1) Date range selector (using create_calendar_filter) ---
        table = _past_target_table()
        filtros = cache.get_filters(table) if table else []

        calendar_widget = create_calendar_filter(
            filtros=filtros,
            cache=cache,
            stable_id_func=stable_id,
            start_date=min_date,
            end_date=max_date,
            current_input=input,
        )

        if calendar_widget is None:
            # Fallback: no temporal filters configured
            calendar_widget = ui.tags.div(
                ui.tags.span(
                    "No se encontraron filtros temporales para esta variable.",
                    style="color:#991b1b;",
                ),
                style="padding:8px;",
            )

        date_range_card = ui.card(
            ui.h3(
                "Panel 4: Escenarios pasados",
                style="margin:0; text-align:center;",
            ),
            ui.tags.div(
                "Modifica valores históricos de las exógenas y observa "
                "cómo habría cambiado la predicción.",
                style="color:#6b7280; margin-top:4px; text-align:center;",
            ),
            ui.tags.hr(style="margin:12px 0;"),
            ui.tags.div(
                ui.tags.b("1) Rango de fechas a predecir"),
                ui.tags.span(
                    " (selecciona la ventana temporal del escenario)",
                    style="font-size:0.85rem; color:#6b7280;",
                ),
                calendar_widget,
                ui.tags.small(
                    "El modelo se entrena con datos anteriores al inicio "
                    "de la ventana seleccionada.",
                    style=(
                        "display:block; color:#57606a; font-size:0.85em; "
                        "margin-top:4px; font-style:italic;"
                    ),
                ),
                style="max-width:480px; margin:0 auto;",
            ),
            style="padding:14px; border-radius:14px;",
        )

        # --- 4) Model + compute ---
        model_box = ui.card(
            ui.tags.div(
                ui.input_radio_buttons(
                    "esc_past_model",
                    "4) Modelo",
                    choices={"xgboost": "XGBoost", "sarimax": "SARIMAX"},
                    selected=past_model(),
                    inline=True,
                ),
                style="display:flex; justify-content:center;",
            ),
            ui.tags.div(
                ui.input_action_button(
                    "esc_past_calc",
                    "Calcular escenario",
                    class_="btn-primary",
                ),
                style="margin-top:10px; display:flex; justify-content:center;",
            ),
            style="padding:14px; border-radius:14px; margin-top:12px;",
        )

        footer = ui.tags.div(
            ui.input_action_button("esc_prev_pasado", "← Anterior"),
            style="margin-top:12px;",
        )

        return ui.div(
            PANEL_STYLES,
            date_range_card,
            ui.output_ui("esc_past_exog_selector"),
            ui.output_ui("esc_past_exog_table"),
            model_box,
            ui.tags.div(
                ui.output_ui("esc_past_status_results"),
                id="esc_past_result_area",
            ),
            footer,
        )

    @output
    @render.ui
    def esc_past_status_results():
        res = scenario_res_rv.get()
        err = past_scenario_err_rv.get()

        status = ui.div()
        if err:
            status = ui.tags.div(
                ui.tags.b("Estado: "),
                ui.tags.span(err),
                style=(
                    "margin-top:10px; padding:10px 12px; border:1px solid #fecaca; "
                    "border-radius:12px; background:#fef2f2; color:#991b1b;"
                ),
            )
        elif res is None or res.get("mode") != "past":
            status = ui.tags.div(
                ui.tags.b("Estado: "),
                ui.tags.span(
                    "selecciona fechas, modifica exógenas y pulsa "
                    "«Calcular escenario».",
                    style="color:#6b7280;",
                ),
                style=(
                    "margin-top:10px; padding:10px 12px; "
                    "border:1px dashed #d1d5db; border-radius:12px; "
                    "background:#fafafa;"
                ),
            )

        outputs = ui.div()
        if res is not None and res.get("mode") == "past":
            outputs = ui.tags.div(
                ui.card(
                    ui.h5("Evolución temporal", style="margin:0 0 8px 0;"),
                    ui.output_ui("esc_past_plot"),
                    style=(
                        "padding:12px; border-radius:14px; "
                        "flex:2 1 640px; min-width:520px;"
                    ),
                ),
                ui.card(
                    ui.h5("Predicción vs valor real", style="margin:0 0 8px 0;"),
                    ui.tags.div(
                        ui.output_ui("esc_past_table"),
                        style="max-height:420px; overflow:auto;",
                    ),
                    style=(
                        "padding:12px; border-radius:14px; "
                        "flex:1 1 420px; min-width:340px;"
                    ),
                ),
                style=(
                    "display:flex; gap:12px; flex-wrap:wrap; "
                    "align-items:flex-start; margin-top:12px;"
                ),
            )

        return ui.div(status, outputs)

    @reactive.Effect
    @reactive.event(input.esc_prev_pasado)
    def _go_step_3_from_pasado():
        current_step.set(3)
