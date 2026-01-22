from shiny import reactive, render, ui, module, req
import pandas as pd
import numpy as np
import logging
from datetime import datetime, date

# Configure logger
logger = logging.getLogger(__name__)
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pandas.api.types import is_integer_dtype
from typing import Any
from xgboost import XGBRegressor

# Importar funciones del backend
from back.models.XGBoost.xgboost_model import create_xgboost_model, predict_xgboost, best_xgboost_params
from back.models.SARIMAX.sarimax_model import create_sarimax_model, predict_sarimax, best_sarimax_params
from back.data_management import buscar_nombre_equivalente, add_exog_df, laggify_df, cargar_exog
from front.utils.formatters import format_variable_name
from back.models.XGBoost.xgboost_model import (
    best_xgboost_params,
    create_xgboost_model,
    predict_xgboost,
    _build_X,
)

# Variables disponibles (same as in ui.py)
try:
    choices_obj = list(buscar_nombre_equivalente("Variables_Objetivo", "Disponibles"))
    selected_obj = buscar_nombre_equivalente("Variables_Objetivo", "Predeterminado")
except Exception:
    choices_obj = ["Turistas"]
    selected_obj = "Turistas"

@module.server
def escenarios_server(input, output, session):
    fecha_hoy = "01/11/2024" # Valor predeterminado en esta prueba de concepto
    fecha_hoy = datetime.strptime(fecha_hoy, "%d/%m/%Y").date() # date.today()
    
    # Wizard step state
    current_step = reactive.Value(1)
    TOTAL_STEPS = 5
    
    # --- Estado Reactivo ---
    # Almacena los datos del escenario (fechas y valores exógenos)
    scenario_state = reactive.Value[Any](None) 
    # Almacena los resultados de la predicción
    resultados_escenario = reactive.Value[Any](None)

    # Step navigation handlers - each step has unique button IDs
    @reactive.Effect
    @reactive.event(input.btn_next_1)
    def go_next_1():
        current_step.set(2)
    
    @reactive.Effect
    @reactive.event(input.btn_next_2)
    def go_next_2():
        current_step.set(3)
    
    @reactive.Effect
    @reactive.event(input.btn_next_3)
    def go_next_3():
        current_step.set(4)
    
    @reactive.Effect
    @reactive.event(input.btn_next_4)
    def go_next_4():
        current_step.set(5)
    
    @reactive.Effect
    @reactive.event(input.btn_prev_2)
    def go_prev_2():
        current_step.set(1)
    
    @reactive.Effect
    @reactive.event(input.btn_prev_3)
    def go_prev_3():
        current_step.set(2)
    
    @reactive.Effect
    @reactive.event(input.btn_prev_4)
    def go_prev_4():
        current_step.set(3)
    
    @reactive.Effect
    @reactive.event(input.btn_prev_5)
    def go_prev_5():
        current_step.set(4)
    
    # Jump to results after executing scenario
    @reactive.Effect
    @reactive.event(resultados_escenario)
    def jump_to_results():
        if resultados_escenario.get() is not None:
            current_step.set(5)
    
    # Step Indicator Renderer
    @output
    @render.ui
    def step_indicator():
        step = current_step.get()
        steps = [
            ("1", "Datos"),
            ("2", "Variables"),
            ("3", "Escenario"),
            ("4", "Modelo"),
            ("5", "Resultados")
        ]
        
        items = []
        for i, (num, label) in enumerate(steps):
            step_num = i + 1
            classes = "step-item"
            if step_num < step:
                classes += " completed"
            elif step_num == step:
                classes += " active"
            
            items.append(
                ui.div(
                    ui.div(
                        ui.span(num, class_="step-number"),
                        class_="step-circle"
                    ),
                    ui.span(label, class_="step-label"),
                    ui.div(class_="step-connector") if i < len(steps) - 1 else None,
                    class_=classes
                )
            )
        
        return ui.div(*items, class_="step-indicator")
    
    # Step Panel 1: Datos
    @output
    @render.ui
    def step_panel_1():
        if current_step.get() != 1:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("📊 Paso 1: Seleccionar Datos", class_="step-panel-title"),
                ui.p("Seleccione la variable objetivo para el escenario.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.div(
                    ui.h5("Modelo de Predicción"),
                    ui.input_select(
                        "modelo",
                        "Modelo:",
                        choices=["SARIMAX", "XGBoost"],
                        selected="SARIMAX"
                    ),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h5("Variable Objetivo"),
                    ui.input_select(
                        "variable_objetivo",
                        "Variable a Predecir:",
                        choices={var: format_variable_name(var) for var in choices_obj},
                        selected=selected_obj
                    ),
                    class_="wizard-form-section"
                ),
                class_="wizard-form-grid"
            ),
            ui.div(
                ui.div(),
                ui.input_action_button("btn_next_1", "Siguiente →", class_="btn-step-next"),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 2: Variables
    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("📈 Paso 2: Variables Exógenas", class_="step-panel-title"),
                ui.p("Seleccione las variables externas que influyen en su predicción.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.div(
                    ui.h5("Categorías"),
                    ui.output_ui("categoria_exogenas_select"),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h5("Variables"),
                    ui.output_ui("variables_exogenas_select"),
                    class_="wizard-form-section"
                ),
                class_="wizard-form-grid"
            ),
            ui.div(
                ui.input_action_button("btn_prev_2", "← Anterior", class_="btn-step-prev"),
                ui.input_action_button("btn_next_2", "Siguiente →", class_="btn-step-next"),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 3: Escenario
    @output
    @render.ui
    def step_panel_3():
        if current_step.get() != 3:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("🎯 Paso 3: Definir Escenario", class_="step-panel-title"),
                ui.p("Elija el tipo de escenario y el período a analizar.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.div(
                    ui.h5("Tipo de Escenario"),
                    ui.input_radio_buttons(
                        "tipo_escenario",
                        None,
                        choices={"historico": "🕐 Qué hubiera pasado (Histórico)", "futuro": "🔮 Qué pasará (Futuro)"},
                        selected="historico"
                    ),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h5("Período"),
                    ui.output_ui("ui_fechas_escenario"),
                    class_="wizard-form-section"
                ),
                class_="wizard-form-grid"
            ),
            ui.div(
                ui.input_action_button("btn_prev_3", "← Anterior", class_="btn-step-prev"),
                ui.input_action_button("btn_next_3", "Siguiente →", class_="btn-step-next"),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 4: Modelo
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("⚙️ Paso 4: Configurar Modelo y Variables", class_="step-panel-title"),
                ui.p("Ajuste los parámetros SARIMAX y configure los valores de las variables.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.panel_conditional(
                "input.modelo === 'SARIMAX'",
                ui.div(
                    ui.div(
                        ui.h5("Auto-ajuste"),
                        ui.input_action_button(
                            "auto_ajustar_sarimax",
                            "🔄 Auto-ajustar parámetros",
                            class_="btn-autoadjust"
                        ),
                        ui.p("Calcula automáticamente los mejores parámetros.", class_="sidebar-hint"),
                        class_="wizard-form-section"
                    ),
                    ui.div(
                        ui.h5("Orden No Estacional (p,d,q)"),
                        ui.row(
                            ui.column(4, ui.input_numeric("p", "p:", value=1, min=0)),
                            ui.column(4, ui.input_numeric("d", "d:", value=1, min=0)),
                            ui.column(4, ui.input_numeric("q", "q:", value=1, min=0))
                        ),
                        class_="wizard-form-section"
                    ),
                    ui.div(
                        ui.h5("Orden Estacional (P,D,Q,s)"),
                        ui.row(
                            ui.column(3, ui.input_numeric("P", "P:", value=1, min=0)),
                            ui.column(3, ui.input_numeric("D", "D:", value=1, min=0)),
                            ui.column(3, ui.input_numeric("Q", "Q:", value=1, min=0)),
                            ui.column(3, ui.input_numeric("s", "s:", value=12, min=1))
                        ),
                        class_="wizard-form-section"
                    ),
                    class_="wizard-form-grid"
                )
            ),
            ui.panel_conditional(
                "input.modelo === 'XGBoost'",
                ui.div(
                    ui.div(
                        ui.h5("Auto-ajuste XGBoost"),
                        ui.input_action_button(
                            "auto_ajustar_xgb",
                            "🔄 Auto-ajustar parámetros",
                            class_="btn-autoadjust"
                        ),
                        ui.p("Calcula automáticamente los mejores parámetros.", class_="sidebar-hint"),
                        class_="wizard-form-section"
                    ),
                    ui.div(
                        ui.h5("Parámetros del Modelo"),
                        ui.row(
                            ui.column(4, ui.input_numeric("xgb_n_estimators", "N Estimators:", value=100)),
                            ui.column(4, ui.input_numeric("xgb_max_depth", "Max Depth:", value=3)),
                            ui.column(4, ui.input_numeric("xgb_learning_rate", "Learning Rate:", value=0.1, step=0.01)),
                        ),
                        ui.row(
                            ui.column(4, ui.input_numeric("xgb_min_child_weight", "Min Child Weight:", value=1)),
                            ui.column(4, ui.input_numeric("xgb_subsample", "Subsample:", value=1.0, step=0.1, max=1.0)),
                            ui.column(4, ui.input_numeric("xgb_colsample_bytree", "Colsample By Tree:", value=1.0, step=0.1, max=1.0)),
                        ),
                        ui.row(
                            ui.column(4, ui.input_numeric("xgb_reg_lambda", "Reg Lambda:", value=1.0, step=0.1)),
                            ui.column(4, ui.input_numeric("xgb_reg_alpha", "Reg Alpha:", value=0.0, step=0.1)),
                        ),
                        class_="wizard-form-section"
                    ),
                    class_="wizard-form-grid"
                )
            ),
            ui.div(
                ui.h5("Editor de Variables Exógenas"),
                ui.p("Ajuste los valores de las variables para el período seleccionado:", class_="sidebar-hint"),
                ui.div(
                    ui.output_ui("ui_editor_exogenas"),
                    style="overflow-x: auto; max-height: 350px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: var(--bg-hover);"
                ),
                class_="wizard-form-section",
                style="margin-top: 1.5rem;"
            ),
            ui.div(
                ui.input_action_button("btn_prev_4", "← Anterior", class_="btn-step-prev"),
                ui.input_action_button(
                    "ejecutar_escenario",
                    "🚀 Ejecutar Escenario",
                    class_="btn-train"
                ),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 5: Resultados
    @output
    @render.ui
    def step_panel_5():
        if current_step.get() != 5:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("📊 Paso 5: Resultados del Escenario", class_="step-panel-title"),
                ui.p("Visualice y analice los resultados del escenario simulado.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.h4("Gráfica del Escenario"),
                ui.output_plot("plot_escenario", height="400px"),
                class_="results-grid-full"
            ),
            ui.div(
                ui.h4("Tabla de Resultados"),
                ui.output_table("tabla_resultados"),
                style="margin-top: 1.5rem;"
            ),
            ui.div(
                ui.input_action_button("btn_prev_5", "← Volver a configurar", class_="btn-step-prev"),
                ui.div(),
                class_="step-navigation"
            ),
            class_="step-panel"
        )

    # --- Helpers (Reutilizados/Adaptados) ---
    def cargar_df(var_predict):
        back_var = buscar_nombre_equivalente(tipo="variable", nombre=var_predict)
        back_route = buscar_nombre_equivalente(tipo="fichero", nombre=back_var)
        df = pd.read_csv(back_route)
        # promote integer columns to float so downstream code in front/ works
        # with float dtypes even when values are whole numbers
        for col in df.columns:
            try:
                if is_integer_dtype(df[col].dtype):
                    df[col] = df[col].astype(float)
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to float: {e}")
                # Optional: Show a visual notification to the user
                # ui.notification_show(f"Warning: Column {col} issue", type="warning")

        # Construir columna e índice de fechas
        fecha_col = None
        if 'Mes' in df.columns:
            fecha_col = 'Mes'
        elif 'mes' in df.columns:
            fecha_col = 'mes'
        
        if fecha_col:
            df['Fecha'] = pd.to_datetime(df[fecha_col] + '-01')
            df.set_index('Fecha', inplace=True)
            df.drop(fecha_col, axis=1, inplace=True)
        else:
            raise KeyError("La columna 'Mes' o 'mes' no existe en el DataFrame.")
        
        # IMPORTANTE: Ordenar por fecha ascendente (de más antiguo a más reciente)
        df.sort_index(inplace=True)
        
        return df

    def translate_var(front_var):
        return buscar_nombre_equivalente(tipo="variable", nombre=front_var)

    def get_selected_exog():
        """Safely get selected exogenous variables, handling case where input doesn't exist yet"""
        try:
            return input.variables_exogenas() or []
        except AttributeError:
            return []

    def translate_exog_vars():
        exog_cols = []
        for var_exog in get_selected_exog():
            exog_cols.append(buscar_nombre_equivalente(tipo="variable", nombre=var_exog))
        return exog_cols

    # --- UI Dinámica: Variables Exógenas ---
    @output
    @render.ui
    def categoria_exogenas_select():
        """Muestra las categorías de variables exógenas disponibles para la variable objetivo seleccionada"""
        try:
            var_obj = input.variable_objetivo()
            # Obtener las variables exógenas permitidas para esta variable objetivo
            variables_exogenas_dict = buscar_nombre_equivalente("Variables_Exogenas", "Por_Variable_Objetivo") or {}
            exog_vars_permitidas = variables_exogenas_dict.get(var_obj, [])
            
            if not exog_vars_permitidas:
                return ui.div("No hay variables exógenas disponibles para esta variable objetivo.")
            
            # Obtener todas las categorías
            categorias_dict = buscar_nombre_equivalente("Variables_Exogenas", "Categorias") or {}
            
            # Filtrar solo las categorías que contienen al menos una variable permitida
            categorias_disponibles = []
            for categoria, variables in categorias_dict.items():
                if any(var in exog_vars_permitidas for var in variables):
                    categorias_disponibles.append(categoria)
            
            if not categorias_disponibles:
                return ui.div("No hay categorías disponibles para esta variable objetivo.")
            
            return ui.input_checkbox_group(
                "categorias_exogenas",
                "Categorías de Variables Exógenas:",
                choices=categorias_disponibles,
                selected=[]
            )
        except:
            return ui.div("Seleccione una variable objetivo primero.")

    @output
    @render.ui
    def variables_exogenas_select():
        """Selector múltiple compacto de variables exógenas"""
        try:
            # 1. Obtener la variable objetivo seleccionada
            var_obj = input.variable_objetivo()
            
            # 2. Obtener las variables exógenas permitidas para esta variable objetivo
            variables_exogenas_dict = buscar_nombre_equivalente("Variables_Exogenas", "Por_Variable_Objetivo") or {}
            exog_vars_permitidas = variables_exogenas_dict.get(var_obj, [])
            
            if not exog_vars_permitidas:
                return ui.div("No hay variables exógenas disponibles para esta variable objetivo.", 
                             style="color: #666; font-style: italic;")
            
            # 3. Filtrar por categorías seleccionadas
            categorias_seleccionadas = input.categorias_exogenas() or []
            
            if not categorias_seleccionadas:
                return ui.div("Seleccione al menos una categoría arriba.", 
                             style="color: #666; font-style: italic;")
            
            categorias_dict = buscar_nombre_equivalente("Variables_Exogenas", "Categorias") or {}
            
            # 4. Obtener solo las variables que están en las categorías seleccionadas Y son permitidas
            exog_vars = []
            for categoria in categorias_seleccionadas:
                vars_en_categoria = categorias_dict.get(categoria, [])
                # Solo agregar las que están permitidas para esta variable objetivo
                vars_filtradas = [v for v in vars_en_categoria if v in exog_vars_permitidas]
                exog_vars.extend(vars_filtradas)
            
            if not exog_vars:
                return ui.div("No hay variables disponibles en las categorías seleccionadas.",
                             style="color: #666; font-style: italic;")

            # Selector múltiple compacto (con tamaño fijo para evitar scroll excesivo)
            return ui.input_selectize(
                "variables_exogenas",
                "Seleccione Variables Exógenas:",
                choices=exog_vars,
                selected=[],
                multiple=True,
                options={'placeholder': 'Buscar y seleccionar variables...'}
            )
        except Exception as e:
            return ui.div(f"Seleccione una variable objetivo primero.",
                         style="color: #666; font-style: italic;")

    # --- UI Dinámica: Fechas ---
    @output
    @render.ui
    def ui_fechas_escenario():
        tipo = input.tipo_escenario()
        req(tipo)
        if tipo == "historico":
            # Rango de fechas para histórico
            # Idealmente deberíamos saber el rango disponible de los datos, 
            # pero por ahora pondremos un rango genérico o inputs de fecha.
            return ui.div(
                ui.input_date("fecha_inicio", "Fecha Inicio Predicción:", value="2024-01-01"),
                ui.input_date("fecha_fin", "Fecha Fin Predicción:", value="2024-12-01")
            )
        else:
            # Futuro: Fecha inicio (usualmente mes siguiente al último dato) y duración
            return ui.div(
                ui.input_date("fecha_inicio_futuro", "Fecha Inicio Predicción:", value=fecha_hoy.replace(day=1) + relativedelta(months=1)),
                ui.input_numeric("meses_futuros", "Meses a predecir:", value=6, min=1, max=24)
            )

    # --- Generar Editor de Variables (auto-render when on step 4) ---
    @output
    @render.ui
    def ui_editor_exogenas():
        # Only render when on step 4
        if current_step.get() != 4:
            return ui.div()
            
        vars_exog = get_selected_exog()
        if not vars_exog:
            return ui.tags.div("Seleccione al menos una variable exógena.", style="color: red;")
        
        tipo = input.tipo_escenario()
        
        # Calcular rango de fechas
        dates = []
        if tipo == "historico":
            start = pd.to_datetime(input.fecha_inicio())
            end = pd.to_datetime(input.fecha_fin())
            if start > end:
                return ui.tags.div("La fecha de inicio debe ser anterior a la fecha fin.", style="color: red;")
            
            current = start
            while current <= end:
                dates.append(current)
                current += relativedelta(months=1)
        else:
            start = pd.to_datetime(input.fecha_inicio_futuro())
            months = input.meses_futuros()
            for i in range(months):
                dates.append(start + relativedelta(months=i))
        
        # Obtener datos base (si es histórico)
        base_data = {}
        if tipo == "historico":
            # Cargar datos reales para pre-llenar
            # Asumimos que todas las exógenas están en el mismo DF que la variable objetivo 
            # O cargamos un DF maestro. Por simplicidad, cargamos el DF de la variable objetivo
            # y esperamos que tenga las exógenas (si se hizo merge antes).
            # PERO: data_loader carga archivos individuales.
            # Mejor estrategia: Cargar cada variable exógena individualmente.
            
            for var_front in vars_exog:
                var_back = translate_var(var_front)
                try:
                    # Usar cargar_exog para cargar correctamente la variable y renombrar columna
                    filepath = buscar_nombre_equivalente(tipo="fichero", nombre=var_back)
                    df_exog = cargar_exog(filepath, var_back)
                    
                    # Filtrar por fechas
                    mask = (df_exog.index >= dates[0]) & (df_exog.index <= dates[-1])
                    subset = df_exog.loc[mask]
                    # Reindexar para asegurar todas las fechas
                    subset = subset.reindex(dates)
                    base_data[var_front] = subset[var_back].fillna(0).values
                except Exception as e:
                    # Si falla, llenar con ceros
                    logger.warning(f"Error cargando {var_front}: {e}")
                    base_data[var_front] = [0] * len(dates)
        else:
            # Futuro: Llenar con 0 o último valor (aquí 0 por simplicidad)
            for var_front in vars_exog:
                base_data[var_front] = [0] * len(dates)

        # Guardar estado del escenario (fechas y vars) para usar al ejecutar
        scenario_state.set({
            "dates": dates,
            "vars": vars_exog,
            "tipo": tipo
        })

        # Construir UI Grid
        # Filas: Fechas, Columnas: Variables
        
        # Improved header: use fixed "Fecha" column and variable columns with
        # a minimum width so labels are readable and horizontal scrolling works.
        header_cells = [ui.column(2, ui.tags.strong("Fecha"))]
        for v in vars_exog:
            header_cells.append(
                ui.column(
                    2,
                    ui.tags.div(
                        v,
                        title=v,
                        style=(
                            "min-width:140px; max-width:220px; "
                            "font-weight:600; text-align:center; word-wrap:break-word;"
                        ),
                    )
                )
            )

        header = ui.row(*header_cells, style="display:flex; align-items:center; gap:8px;")
        
        rows = []
        for i, d in enumerate(dates):
            date_str = d.strftime("%Y_%m_%d")
            cols = [ui.column(2, ui.tags.span(d.strftime("%Y-%m")))]
            
            for var in vars_exog:
                val = base_data[var][i]
                input_id = f"val_{var}_{date_str}"
                # Wrap input in a div with min-width so columns align with header
                cols.append(
                    ui.column(
                        2,
                        ui.tags.div(
                            ui.input_numeric(input_id, None, value=str(val), width="100%"),
                            style="min-width:140px; max-width:220px;"
                        )
                    )
                )
            rows.append(ui.row(*cols, style="margin-bottom: 5px;"))
            
        return ui.div(header, *rows)

    # --- Auto-ajustar SARIMAX ---
    @reactive.Effect
    @reactive.event(input.auto_ajustar_sarimax)
    def detect_best_parameters():
        if input.modelo() != "SARIMAX":
            return
        
        # Validate that exogenous variables are selected
        exog_vars = get_selected_exog()
        if not exog_vars:
            ui.notification_show(
                "Debe seleccionar al menos una categoría y variable exógena antes de auto-ajustar.",
                type="warning",
                duration=5
            )
            return
        
        try:
            ui.notification_show("Calculando mejores parámetros... esto puede tardar unos segundos.", type="message")
            
            var_predict = input.variable_objetivo()
            exog_cols = translate_exog_vars()
            df = cargar_df(var_predict=var_predict)
            df = add_exog_df(df=df, exog_cols=exog_cols)
            
            tipo = input.tipo_escenario()
            if tipo == "historico":
                start_date = pd.to_datetime(input.fecha_inicio())
                # Usar datos hasta antes del inicio del escenario
                df_subset = df[df.index < start_date].copy()
                
                len_before = len(df_subset)
                df_subset.dropna(inplace=True) # FIX: Drop NaNs introduced by missing exogenous data
                len_after = len(df_subset)
                
                if df_subset.empty:
                    ui.notification_show("No hay datos históricos suficientes antes de la fecha de inicio.", type="error")
                    return
                
                if len_after < len_before:
                     start_avail = df_subset.index.min().strftime('%Y-%m-%d')
                     end_avail = df_subset.index.max().strftime('%Y-%m-%d')
                     ui.notification_show(f"Aviso: Datos limitados por variables exógenas. Usando rango: {start_avail} a {end_avail}", type="warning", duration=10)

                periodos = 1 # Mínimo recorte
            else:
                df_subset = df.copy()
                
                len_before = len(df_subset)
                df_subset.dropna(inplace=True) # FIX: Drop NaNs introduced by missing exogenous data
                len_after = len(df_subset)
                
                if len_after < len_before:
                     start_avail = df_subset.index.min().strftime('%Y-%m-%d')
                     end_avail = df_subset.index.max().strftime('%Y-%m-%d')
                     ui.notification_show(f"Aviso: Datos limitados por variables exógenas. Usando rango: {start_avail} a {end_avail}", type="warning", duration=10)

                periodos = input.meses_futuros()
                
            order, seasonal = best_sarimax_params(
                df_subset, 
                exog_cols,
                column_y=translate_var(var_predict),
                periodos_a_predecir=periodos
            )
            
            ui.update_numeric("p", value=order[0])
            ui.update_numeric("d", value=order[1])
            ui.update_numeric("q", value=order[2])
            ui.update_numeric("P", value=seasonal[0])
            ui.update_numeric("D", value=seasonal[1])
            ui.update_numeric("Q", value=seasonal[2])
            ui.update_numeric("s", value=seasonal[3])
            
            ui.notification_show("Parámetros SARIMAX actualizados.", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error al auto-ajustar: {str(e)}", type="error")
            logger.error(f"Error al auto-ajustar: {e}")

    # --- Auto-ajustar XGBoost ---
    @reactive.Effect
    @reactive.event(input.auto_ajustar_xgb)
    def detect_best_parameters_xgb():
        if input.modelo() != "XGBoost":
            return

        # Validate that exogenous variables are selected
        exog_vars = get_selected_exog()
        if not exog_vars:
            ui.notification_show(
                "Debe seleccionar al menos una categoría y variable exógena antes de auto-ajustar.",
                type="warning",
                duration=5
            )
            return

        try:
            ui.notification_show("Calculando mejores parámetros XGBoost... esto puede tardar unos segundos.", type="message")

            var_predict = input.variable_objetivo()
            # En tu código original se usan translate_exog_vars() que retorna lista de nombres backend
            exog_cols = translate_exog_vars()
            
            # Cargar y preparar datos (reutilizamos funciones existentes en server.py)
            df = cargar_df(var_predict=var_predict)
            df = add_exog_df(df=df, exog_cols=exog_cols)
            
            # Nota: No usamos lags complejos aquí porque la UI original no tiene selector de lags en este archivo
            exog_cols_full = exog_cols

            tipo = input.tipo_escenario()
            if tipo == "historico":
                start_date = pd.to_datetime(input.fecha_inicio())
                df_subset = df[df.index < start_date].copy()
                df_subset.dropna(inplace=True)
                
                if df_subset.empty:
                    ui.notification_show("No hay datos históricos suficientes antes de la fecha de inicio.", type="error")
                    return
                periodos = 1
            else:
                df_subset = df.copy()
                df_subset.dropna(inplace=True)
                periodos = input.meses_futuros()

            best_params = best_xgboost_params(
                df_subset,
                exog_cols_full,
                column_y=translate_var(var_predict),
                periodos_a_predecir=periodos,
            )

            ui.update_numeric("xgb_n_estimators", value=best_params.get("n_estimators"))
            ui.update_numeric("xgb_learning_rate", value=best_params.get("learning_rate"))
            ui.update_numeric("xgb_max_depth", value=best_params.get("max_depth"))
            ui.update_numeric("xgb_min_child_weight", value=best_params.get("min_child_weight"))
            ui.update_numeric("xgb_subsample", value=best_params.get("subsample"))
            ui.update_numeric("xgb_colsample_bytree", value=best_params.get("colsample_bytree"))
            ui.update_numeric("xgb_reg_lambda", value=best_params.get("reg_lambda"))
            ui.update_numeric("xgb_reg_alpha", value=best_params.get("reg_alpha"))

            ui.notification_show("Parámetros XGBoost actualizados.", type="message")

        except Exception as e:
            ui.notification_show(f"Error al auto-ajustar XGBoost: {str(e)}", type="error")
            logger.error(f"Error al auto-ajustar XGBoost: {e}")

    # --- Ejecutar Escenario ---
    @reactive.Effect
    @reactive.event(input.ejecutar_escenario)
    def run_scenario():
        state = scenario_state.get()
        if not state:
            ui.notification_show("Primero configure y genere el editor de variables.", type="warning")
            return

        dates = state["dates"]
        vars_front = state["vars"]
        tipo = state["tipo"]
        
        # 1. Recolectar datos del editor
        data_dict = {"Fecha": dates}
        vars_back = []
        
        for var_front in vars_front:
            var_back = translate_var(var_front)
            vars_back.append(var_back)
            values = []
            for d in dates:
                date_str = d.strftime("%Y_%m_%d")
                input_id = f"val_{var_front}_{date_str}"
                # Leer valor del input. Nota: input[id]()
                # Necesitamos acceder dinámicamente a input
                # En Shiny for Python, input es un objeto, podemos usar getattr o input[id] si fuera dict
                # Pero input es un objeto Input.
                # No podemos iterar input fácilmente.
                # Pero sabemos los IDs.
                try:
                    val = getattr(input, input_id)()
                    values.append(val)
                except AttributeError:
                    values.append(0) # Fallback
            
            data_dict[var_back] = values
            
        df_scenario_exog = pd.DataFrame(data_dict)
        df_scenario_exog.set_index("Fecha", inplace=True)
        df_scenario_exog.fillna(0, inplace=True) # FIX: Ensure no NaNs in user input
        
        # 2. Cargar datos históricos para entrenamiento
        target_front = input.variable_objetivo()
        target_back = translate_var(target_front)
        
        df_full = cargar_df(target_front)
        
        # Añadir exógenas históricas al df_full para entrenar
        # Esto es complejo porque necesitamos las exógenas históricas alineadas.
        # Reutilizamos add_exog_df
        df_full = add_exog_df(df=df_full, exog_cols=vars_back)
        
        # Definir punto de corte para entrenamiento
        start_date_scenario = dates[0]
        
        # Entrenar con datos ANTES del inicio del escenario
        df_train = df_full[df_full.index < start_date_scenario].copy()
        df_train.dropna(inplace=True) # FIX: Drop NaNs introduced by missing exogenous data
        
        if df_train.empty:
            ui.notification_show("No hay datos históricos suficientes antes de la fecha de inicio seleccionada.", type="error")
            return

        # Configurar modelo según selección
        modelo = input.modelo()
        
        try:
            predictions = None
            
            if modelo == "SARIMAX":
                order = (input.p(), input.d(), input.q())
                seasonal_order = (input.P(), input.D(), input.Q(), input.s())
                
                model = create_sarimax_model(
                    train=df_train,
                    exog_cols=vars_back,
                    column_y=target_back,
                    order=order,
                    seasonal_order=seasonal_order
                )
                
                # Predecir
                predictions = predict_sarimax(
                    model_fit=model,
                    start=dates[0],
                    end=dates[-1],
                    exog_cols=df_scenario_exog[vars_back]
                )
                
            elif modelo == "XGBoost":
                xgb_params = {
                    "n_estimators": input.xgb_n_estimators(),
                    "max_depth": input.xgb_max_depth(),
                    "min_child_weight": input.xgb_min_child_weight(),
                    "learning_rate": input.xgb_learning_rate(),
                    "subsample": input.xgb_subsample(),
                    "colsample_bytree": input.xgb_colsample_bytree(),
                    "reg_lambda": input.xgb_reg_lambda(),
                    "reg_alpha": input.xgb_reg_alpha(),
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "random_state": 42,
                }
                
                # Crear modelo XGBoost
                model = create_xgboost_model(
                    train=df_train,
                    exog_cols=vars_back, # Usamos variables seleccionadas sin lags especiales por ahora
                    column_y=target_back,
                    xgb_params=xgb_params,
                )
                
                # Preparar df_future para predicción
                df_future = df_scenario_exog.copy()
                # predict_xgboost espera que la columna objetivo exista en df_future (aunque sea NaN o ignorada para X)
                df_future[target_back] = np.nan 
                
                predictions = predict_xgboost(
                    model_fit=model,
                    df_future=df_future,
                    exog_cols=vars_back,
                    column_y=target_back,
                )
            
            else:
                 ui.notification_show(f"Modelo {modelo} no soportado o implementado.", type="error")
                 return

            
            resultados_escenario.set({
                "predictions": predictions,
                "dates": dates,
                "tipo": tipo,
                "target_back": target_back,
                "df_full": df_full # Para comparar con real si es histórico
            })
            
            ui.notification_show("Escenario ejecutado correctamente.", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error al ejecutar modelo: {str(e)}", type="error")
            logger.error(f"Error al ejecutar modelo: {e}")

    # --- Gráfica ---
    @output
    @render.plot
    def plot_escenario():
        res = resultados_escenario.get()
        
        # Always try to load historical data for context
        target_front = input.variable_objetivo()
        try:
            df_full = cargar_df(target_front)
            target_back = translate_var(target_front)
        except Exception as e:
            logger.warning(f"Could not load historical data for plot: {e}")
            df_full = None
            target_back = None

        if df_full is None and res is None:
            return # Nothing to show

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Historical
        if df_full is not None and target_back in df_full.columns:
            # Show full history up to prediction start
            if res:
                # start_plot = res["predictions"].index[0] - relativedelta(months=24)
                # mask_hist = (df_full.index >= start_plot) & (df_full.index < res["predictions"].index[0])
                mask_hist = (df_full.index < res["predictions"].index[0])
                ax.plot(df_full.loc[mask_hist].index, df_full.loc[mask_hist, target_back], label="Histórico", color="black")
            else:
                # Show full available data
                ax.plot(df_full.index, df_full[target_back], label="Histórico", color="black")

        # Plot Scenario
        if res:
            preds = res["predictions"]
            tipo = res["tipo"]
            ax.plot(preds.index, preds, label="Escenario (Predicción)", color="blue", linestyle="--", marker="o")
            
            if tipo == "historico" and df_full is not None:
                common_dates = df_full.index.intersection(preds.index)
                if not common_dates.empty:
                    ax.plot(common_dates, df_full.loc[common_dates, target_back], label="Realidad", color="green", alpha=0.6)
        
        ax.set_title(f"Escenario: {input.variable_objetivo()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

    # --- Tabla ---
    @output
    @render.table
    def tabla_resultados():
        res = resultados_escenario.get()
        if not res:
            return pd.DataFrame()
        
        preds = res["predictions"]
        df = pd.DataFrame(preds).reset_index()
        df.columns = ["Fecha", "Predicción Escenario"]
        # Convertir Predicción Escenario a entero (sin decimales)
        try:
            df["Predicción Escenario"] = df["Predicción Escenario"].round().astype("Int64")
        except Exception:
            df["Predicción Escenario"] = pd.to_numeric(df["Predicción Escenario"], errors="coerce").round().astype("Int64")
        
        if res["tipo"] == "historico":
            df_full = res["df_full"]
            target = res["target_back"]
            # Merge con real
            real = df_full.loc[df_full.index.isin(preds.index), [target]].reset_index()
            real.columns = ["Fecha", "Valor Real"]
            df = pd.merge(df, real, on="Fecha", how="left")
            df["Diferencia"] = df["Predicción Escenario"] - df["Valor Real"]
            df["% Diferencia"] = (df["Diferencia"] / df["Valor Real"].replace(0, np.nan)) * 100
            df["Diferencia valor absoluto"] = df["Diferencia"].abs()
            
        return df