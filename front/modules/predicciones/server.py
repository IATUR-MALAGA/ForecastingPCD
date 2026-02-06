import re
import hashlib
from collections import OrderedDict
import pandas as pd
from shiny import ui, reactive, render, module
from front.utils.back_api import sarimax_run

from front.utils.back_api import (
    get_names_in_table_catalog,
    get_metadata_for_variable,
    get_date_range_for_variable, 
    get_filters_for_variable,
    get_distinct_values_for_column,
    get_table_columns,
    get_tableName_for_variable
)


from back.models.SARIMAX.sarimax_model import best_sarimax_params, create_sarimax_model, predict_sarimax
from back.models.SARIMAX.sarimax_statistics import compute_metrics
from back.models.SARIMAX.sarimax_graph import plot_predictions
from front.modules.predicciones.utils import (
    _safe_alias,
    check_date_and_temporality,
    create_dataframe_based_on_selection,
    slug as _slug,  
    stable_id as _stable_id,
    group_by_category as _group_by_category,
    fmt as _fmt
)


# -----------------------------
# Module
# -----------------------------
@module.server
def predicciones_server(input, output, session):

    current_step = reactive.Value(1)

    target_var_rv = reactive.Value(None)      # Panel 1 (selección única)
    predictors_rv = reactive.Value([])        # Panel 2 (selección múltiple)

    catalog_entries = get_names_in_table_catalog() or []
    # Mapa nombre(variable)->tabla (por si en catálogo viene nombre_tabla)
    _name_to_table: dict[str, str] = {}
    for e in (catalog_entries or []):
        n = e.get("nombre")
        if not n:
            continue
        _name_to_table[n] = e.get("nombre_tabla") or n

    # Cache de rango de fechas por tabla (por sesión)
    _date_range_cache: dict[str, tuple[object | None, object | None]] = {}

    def _get_date_range(nombre_var: str):
        table = _name_to_table.get(nombre_var, nombre_var)
        if table in _date_range_cache:
            return _date_range_cache[table]

        try:
            rows = get_date_range_for_variable(table) or []
            row = rows[0] if rows else {}
            start = row.get("fecha_inicio")
            end = row.get("fecha_fin")
        except Exception:
            start, end = None, None

        _date_range_cache[table] = (start, end)
        return start, end


    def _compatibilidad_con_objetivo(
        predictor_name: str,
        predictor_meta: dict,
        target_name: str,
        target_meta: dict,
        target_start,
        target_end,
    ) -> tuple[bool, str]:
        """
        Compatibilidad predictor vs objetivo usando:
        - misma temporalidad
        - y que el predictor CUBRA el rango del objetivo
          (esto se logra pasando predictor como "1" y objetivo como "2"
           porque check_date_and_temporality comprueba que 2 está contenido en 1).
        """
        if not target_name:
            return False, "Sin objetivo seleccionado"

        pred_temp = predictor_meta.get("temporalidad")
        tgt_temp = target_meta.get("temporalidad")

        pred_start, pred_end = _get_date_range(predictor_name)

        if pred_temp is None or tgt_temp is None:
            return False, "Temporalidad no definida"

        if pred_start is None or pred_end is None or target_start is None or target_end is None:
            return False, "Sin rango de fechas"

        ok = check_date_and_temporality(
            pred_start, target_start,
            pred_end,   target_end,
            pred_temp,  tgt_temp
        )

        if ok:
            return True, ""
        # Si falla, puede ser temporalidad distinta o rango insuficiente;
        # temporalidad distinta ya la capturó check_date_and_temporality, pero damos motivo más claro:
        if pred_temp.strip().lower() != tgt_temp.strip().lower():
            return False, "Temporalidad distinta"
        return False, "El predictor no cubre el rango del objetivo"

    # Cache de metadata por variable (por sesión)
    _metadata_cache: dict[str, dict] = {}

    def _get_meta(nombre: str) -> dict:
        if nombre in _metadata_cache:
            return _metadata_cache[nombre]
        rows = get_metadata_for_variable(nombre) or []
        meta = rows[0] if rows else {}
        _metadata_cache[nombre] = meta
        return meta
    
        # Cache filtros por variable (por sesión)
    _filters_cache: dict[str, list[dict]] = {}

    # Cache DISTINCT values por (tabla, columna)
    _distinct_cache: dict[tuple[str, str], list[str]] = {}

    # Cache columnas por tabla (para validar que existan)
    _table_cols_cache: dict[str, set[str]] = {}

    def _get_table_cols(schema: str, table: str) -> set[str]:
        key = f"{schema}.{table}"
        if key in _table_cols_cache:
            return _table_cols_cache[key]
        cols = get_table_columns(schema, table) or []
        s = {c.get("column_name") for c in cols if c.get("column_name")}
        _table_cols_cache[key] = s
        return s

    def _get_filters(nombre_var: str) -> list[dict]:
        if nombre_var in _filters_cache:
            return _filters_cache[nombre_var]

        rows = get_filters_for_variable(nombre_var) or []
        out: list[dict] = []

        default_table = _name_to_table.get(nombre_var, nombre_var)

        for r in rows:
            table = r.get("nombre_tabla") or default_table
            col = r.get("filtro")  # nombre real columna
            if not col:
                continue

            label = r.get("nombre_filtro") or col  # nombre bonito (fallback a col)
            out.append({"table": table, "col": col, "label": label})

        # dedup conservando orden (por tabla+col)
        seen = set()
        uniq = []
        for item in out:
            k = (item["table"], item["col"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(item)

        _filters_cache[nombre_var] = uniq
        return uniq



    def _get_distinct(schema: str, table: str, col: str) -> list[str]:
        key = (table, col)
        if key in _distinct_cache:
            return _distinct_cache[key]
        vals = get_distinct_values_for_column(schema, table, col) or []
        _distinct_cache[key] = vals
        return vals


    PANEL_STYLES = ui.tags.style(
        """
        .var-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 6px 0;
        }

        /* Botón base */
        .var-pick {
            text-align: left;
            width: 100%;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 8px 10px;
            background: #ffffff;
            cursor: pointer;
        }

        .var-pick:hover { background: #f6f8fa; }

        .var-pick.is-selected {
            font-weight: 700;
            background: #d1e7dd;
            border-color: #198754;
            color: #0f5132;
        }

        .selection-pill {
            padding: 6px 10px;
            border-radius: 999px;
            display: inline-block;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            margin-bottom: 8px;
        }

        /* Panel 2: tarjetas por variable */
        .var-item {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px 12px;
            background: #fff;
            margin-bottom: 8px;
        }

        .var-item .form-check { margin: 0; }
        .var-item .form-check-label { font-weight: 600; }

        .var-meta {
            margin-top: 6px;
            padding-left: 24px; /* alinear con el checkbox */
            font-size: 0.92rem;
            color: #24292f;
        }

        .var-meta-grid {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 4px 10px;
            margin-top: 4px;
        }

        .var-meta-key { color: #57606a; }
        .var-desc {
            margin-top: 6px;
            color: #24292f;
        }
                .compat-badge {
            padding: 2px 10px;
            border-radius: 999px;
            display: inline-block;
            font-size: 0.85rem;
            border: 1px solid;
            font-weight: 600;
        }
        .compat-yes {
            background: #d1e7dd;
            border-color: #198754;
            color: #0f5132;
        }
        .compat-no {
            background: #f8d7da;
            border-color: #dc3545;
            color: #842029;
        }
        .compat-reason {
            margin-top: 4px;
            font-size: 0.85rem;
            color: #57606a;
        }

        """
    )

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
        target_meta = _get_meta(target_var) if target_var else {}
        target_start, target_end = _get_date_range(target_var) if target_var else (None, None)
        target_temp = _fmt(target_meta.get("temporalidad"))
        panels = []
        for cat, names in grouped.items():
            var_blocks = []

            for name in names:
                var_id = _stable_id("pred", name)
                meta = _get_meta(name)
                compat, reason = _compatibilidad_con_objetivo(
                    predictor_name=name,
                    predictor_meta=meta,
                    target_name=target_var,
                    target_meta=target_meta,
                    target_start=target_start,
                    target_end=target_end,
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
            table = _name_to_table.get(pretty)

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

            filtros = _get_filters(table)  # mismos filtros que renderizas en Panel 3
            selected_list: list[dict] = []

            for f in filtros:
                t = f["table"]
                col = f["col"]

                # OJO: tiene que ser EXACTAMENTE el mismo input_id que usas en Panel 3
                input_id = _stable_id("flt", f"{t}__{col}")

                vals = input[input_id]() if (input_id in input) else None

                # selectize multiple suele devolver lista/tupla; vacío => []/None
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
            pretty = item["pretty"]   # <- nombre bonito (UI)
            table = item["table"]     # <- nombre real (queries)

            # IMPORTANTE: pides filtros usando el nombre REAL de tabla
            filtros = _get_filters(table)

            if not filtros:
                body = ui.p("Sin filtros configurados en tbl_admin_filtros para esta variable/tabla.")
            else:
                controls = []
                for f in filtros:
                    t = f["table"]
                    col = f["col"]
                    label = f.get("label") or col  # <-- NUEVO


                    cols_set = _get_table_cols("IA", t)
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

                    # ID estable: usa la TABLA REAL + columna (evitas colisiones y es consistente)
                    input_id = _stable_id("flt", f"{t}__{col}")

                    choices = _get_distinct("IA", t, col)

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

            # Acordeón: muestra NOMBRE BONITO
            # value: usa algo estable y único (tabla real suele ser mejor que pretty)
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
    def sarimax_results():
        if current_step.get() != 4:
            return None

        payload = {
            "target_var": target_var_rv.get(),
            "predictors": predictors_rv.get(),
            "filters_by_var": selected_filters_by_var(),  # ya tiene table/col/values
            "train_ratio": 0.70,
            "auto_params": True,   # OJO: puede ser lento
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

        # reconstruye pred_test como Series para reutilizar tu plot_predictions
        pred_vals = resp["y_pred"]
        pred_test = pd.Series(pred_vals, index=range(len(train), len(train) + len(test)))

        fig_or_ax = plot_predictions(
            df=df,
            pred=pred_test,
            title="Predicciones SARIMAX",
            ylabel="Valores",
            xlabel="Fecha",
            column_y=y_col,
            periodos_a_predecir=n_test,
            holidays_col=None
        )
        fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax

        return {
            "mape": resp["mape"],
            "rmse": resp["rmse"],
            "mae": resp["mae"],
            "fig": fig,
            "order": resp["order"],
            "seasonal_order": resp["seasonal_order"],
        }


    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()

        res = sarimax_results()
        if res is None:
            return ui.div()

        mape, rmse, mae = res["mape"], res["rmse"], res["mae"]

        return ui.div(
            PANEL_STYLES,
            ui.h3("Panel 4: Resultados del modelo SARIMAX"),
            ui.p("Aquí se muestran los resultados del modelo SARIMAX ajustado con las variables seleccionadas."),

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
        





