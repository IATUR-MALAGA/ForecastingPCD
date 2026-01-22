import traceback
import logging
from shiny import reactive, render
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
import numpy as np
from pandas.api.types import is_integer_dtype
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import json
from pathlib import Path
from shiny import ui, render, module
from front.modules.common.data_utils import (
    cargar_df,
    get_selected_exog,
    selected_lags,
    translate_exog_vars,
    translate_var,
)


# Importar funciones del backend SARIMAX y xgboost
from back.models.XGBoost.xgboost_graph import plot_predictions as plot_xgb_predictions
from back.models.SARIMAX.sarimax_graph import plot_predictions
from back.models.SARIMAX.sarimax_model import create_sarimax_model, predict_sarimax, best_sarimax_params
from back.models.SARIMAX.sarimax_statistics import compute_metrics
from back.data_management import buscar_nombre_equivalente, add_exog_df, laggify_df
from front.utils.formatters import format_variable_name
from back.models.XGBoost.xgboost_model import (
    best_xgboost_params,
    create_xgboost_model,
    predict_xgboost,
    _build_X,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


@module.server
def predicciones_server(input, output, session):
    
    # Wizard step state
    current_step = reactive.Value(1)
    TOTAL_STEPS = 4
    
    # predicciones = reactive.Value[object](None)
    predicciones = reactive.Value[object](None)
    metricas = reactive.Value[object](None)
    
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
    
    # Jump to results after training
    @reactive.Effect
    @reactive.event(predicciones)
    def jump_to_results():
        if predicciones.get() is not None:
            current_step.set(4)
    
    # Step Indicator Renderer
    @output
    @render.ui
    def step_indicator():
        step = current_step.get()
        steps = [
            ("1", "Datos"),
            ("2", "Variables"),
            ("3", "Modelo"),
            ("4", "Resultados")
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
                ui.p("Seleccione la variable que desea predecir.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.div(
                    ui.h5("Modelo de Predicción"),
                    ui.output_ui("modelo_select"),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h5("Variable Objetivo"),
                    ui.output_ui("categoria_objetivo_select"),
                    ui.output_ui("variable_objetivo_select"),
                    class_="wizard-form-section"
                ),
                class_="wizard-form-grid"
            ),
            ui.div(
                ui.div(),  # Empty left side
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
                    ui.h5("Categorías y Variables"),
                    ui.output_ui("categoria_exogenas_select"),
                    ui.output_ui("variables_exogenas_select"),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h5("Configuración de Lags"),
                    ui.output_ui("lags_selector"),
                    class_="wizard-form-section"
                ),
                class_="wizard-form-grid"
            ),
            ui.div(
                ui.h5("Horizonte de Predicción"),
                ui.input_numeric(
                    "periodos_futuros",
                    "Períodos Futuros a Predecir:",
                    value=2,
                    min=1,
                    max=12
                ),
                class_="wizard-form-section",
                style="max-width: 300px; margin-top: 1rem;"
            ),
            ui.div(
                ui.input_action_button("btn_prev_2", "← Anterior", class_="btn-step-prev"),
                ui.input_action_button("btn_next_2", "Siguiente →", class_="btn-step-next"),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 3: Modelo
    @output
    @render.ui
    def step_panel_3():
        if current_step.get() != 3:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("⚙️ Paso 3: Configurar y Entrenar Modelo", class_="step-panel-title"),
                ui.p("Ajuste los parámetros SARIMAX y entrene el modelo.", class_="step-panel-description"),
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
                            ui.column(4, ui.input_numeric("p", "p:", value=1, min=0, max=5)),
                            ui.column(4, ui.input_numeric("d", "d:", value=1, min=0, max=2)),
                            ui.column(4, ui.input_numeric("q", "q:", value=1, min=0, max=5))
                        ),
                        class_="wizard-form-section"
                    ),
                    ui.div(
                        ui.h5("Orden Estacional (P,D,Q,s)"),
                        ui.row(
                            ui.column(3, ui.input_numeric("P", "P:", value=1, min=0, max=3)),
                            ui.column(3, ui.input_numeric("D", "D:", value=1, min=0, max=2)),
                            ui.column(3, ui.input_numeric("Q", "Q:", value=1, min=0, max=3)),
                            ui.column(3, ui.input_numeric("s", "s:", value=12, min=1, max=12))
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
                            ui.h5("Auto-ajuste"),
                            ui.input_action_button(
                                "auto_ajustar_xgboost",
                                "🔄 Auto-ajustar parámetros",
                                class_="btn-autoadjust"
                            ),
                            ui.p("Calcula automáticamente los mejores parámetros.", class_="sidebar-hint"),
                            class_="wizard-form-section"
                    ),
                    ui.div(
                        ui.h5("Parámetros XGBoost"),
                        ui.row(
                            ui.column(6, ui.input_numeric("n_estimators", "N Estimators:", value=100, min=10, max=1000)),
                            ui.column(6, ui.input_numeric("max_depth", "Max Depth:", value=6, min=1, max=20)),
                        ),
                        ui.row(
                            ui.column(6, ui.input_numeric("learning_rate", "Learning Rate:", value=0.1, min=0.001, max=1.0, step=0.01)),
                            ui.column(6, ui.input_numeric("subsample", "Subsample:", value=0.8, min=0.1, max=1.0, step=0.1))
                        ),
                        ui.row(
                            ui.column(6, ui.input_numeric("min_child_weight", "Min Child Weight:", value=1, min=0, max=20)),
                            ui.column(6, ui.input_numeric("colsample_bytree", "Colsample ByTree:", value=0.8, min=0.1, max=1.0, step=0.1))
                        ),
                        ui.row(
                            ui.column(6, ui.input_numeric("reg_lambda", "Reg Lambda (L2):", value=1.0, min=0.0, max=10.0, step=0.1)),
                            ui.column(6, ui.input_numeric("reg_alpha", "Reg Alpha (L1):", value=0.0, min=0.0, max=10.0, step=0.1))
                        ),
                        class_="wizard-form-section"
                    ),
                    class_="wizard-form-grid"
                )
            ),
            ui.div(
                ui.output_ui("resumen_parametros"),
                style="margin-top: 1.5rem;"
            ),
            ui.div(
                ui.input_action_button("btn_prev_3", "← Anterior", class_="btn-step-prev"),
                ui.input_action_button(
                    "entrenar_modelo",
                    "🚀 Entrenar Modelo",
                    class_="btn-train"
                ),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    # Step Panel 4: Resultados
    @output
    @render.ui
    def step_panel_4():
        if current_step.get() != 4:
            return ui.div()
        
        return ui.div(
            ui.div(
                ui.h2("📊 Paso 4: Resultados", class_="step-panel-title"),
                ui.p("Visualice las predicciones y métricas del modelo entrenado.", class_="step-panel-description"),
                class_="step-panel-header"
            ),
            ui.div(
                ui.div(
                    ui.h4("Métricas del Modelo"),
                    ui.output_ui("mape_output"),
                    class_="wizard-form-section"
                ),
                ui.div(
                    ui.h4("Configuración Utilizada"),
                    ui.output_ui("resumen_parametros_results"),
                    class_="wizard-form-section"
                ),
                class_="results-grid"
            ),
            ui.div(
                ui.h4("Gráfica de Predicción"),
                ui.output_plot("grafica_prediccion", height="400px"),
                class_="results-grid-full",
                style="margin-top: 1.5rem;"
            ),
            ui.div(
                ui.h4("Tabla de Predicciones"),
                ui.output_table("tabla_predicciones"),
                style="margin-top: 1.5rem;"
            ),
            ui.div(
                ui.input_action_button("btn_prev_4", "← Volver a configurar", class_="btn-step-prev"),
                ui.div(),  # Empty right side
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
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
    

    def get_file_path(front_var):
        back_var = buscar_nombre_equivalente(tipo="variable", nombre=front_var)
        back_route = buscar_nombre_equivalente(tipo="fichero", nombre=back_var)
        return back_route
    

    def get_title(front_var):
        back_var = buscar_nombre_equivalente(tipo="variable", nombre=front_var)
        title = buscar_nombre_equivalente(tipo="titulo_grafica", nombre=back_var)
        return title
    
    

    def get_selected_exog():
        """Safely get selected exogenous variables, handling case where input doesn't exist yet"""
        try:
            return input.variables_exogenas() or []
        except AttributeError:
            return []

    def translate_exog_vars():
        """Returns list of backend variable names for selected exogenous vars"""
        exog_cols = []
        for var_exog in get_selected_exog():
            exog_cols.append(buscar_nombre_equivalente(tipo="variable", nombre=var_exog))
        return exog_cols
    
    def selected_lags():
        """Returns dict {backend_var: [int_lags...]} for selected exogenous vars"""
        lags = {}
        try:
            # 1. Obtener lags comunes (si se seleccionaron)
            lags_comunes_raw = input.lags_comunes() or []
            lags_comunes_int = [int(l.replace("lag", "")) for l in lags_comunes_raw]
            
            # 2. Para cada variable exógena seleccionada
            for var_exog in get_selected_exog():
                back_var = buscar_nombre_equivalente(tipo="variable", nombre=var_exog)
                
                # Intentar obtener lags individuales de esta variable
                try:
                    lags_individuales_raw = getattr(input, f"lags_{var_exog}")() or []
                    lags_individuales_int = [int(l.replace("lag", "")) for l in lags_individuales_raw]
                except AttributeError:
                    lags_individuales_int = []
                
                # Combinar lags comunes + individuales (sin duplicados)
                lags_combinados = list(set(lags_comunes_int + lags_individuales_int))
                lags_combinados.sort()
                
                lags[back_var] = lags_combinados
        except:
            pass
        return lags
    
    def max_selected_lag():
        """Returns the maximum lag number across all selected lags"""
        max_lag = 0
        try:
            # Revisar lags comunes
            lags_comunes_raw = input.lags_comunes() or []
            if lags_comunes_raw:
                lags_int = [int(l.replace("lag", "")) for l in lags_comunes_raw]
                max_lag = max(max_lag, max(lags_int))
            
            # Revisar lags individuales de cada variable
            for var_exog in get_selected_exog():
                try:
                    lags_individuales_raw = getattr(input, f"lags_{var_exog}")() or []
                    if lags_individuales_raw:
                        lags_int = [int(l.replace("lag", "")) for l in lags_individuales_raw]
                        max_lag = max(max_lag, max(lags_int))
                except AttributeError:
                    logger.debug(f"AttributeError in max_selected_lag for {var_exog}")
        except:
            pass
        return max_lag
    
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

            # Crear diccionario con nombres formateados para mostrar
            choices_dict = {var: format_variable_name(var) for var in exog_vars}
            
            # Selector múltiple compacto (con tamaño fijo para evitar scroll excesivo)
            return ui.input_selectize(
                "variables_exogenas",
                "Seleccione Variables Exógenas:",
                choices=choices_dict,
                selected=[],
                multiple=True,
                options={'placeholder': 'Buscar y seleccionar variables...'}
            )
        except Exception as e:
            return ui.div(f"Seleccione una variable objetivo primero.",
                         style="color: #666; font-style: italic;")
    
    @output
    @render.ui
    def lags_selector():
        """Selector de lags individualizado por variable pero en formato compacto"""
        try:
            vars_seleccionadas = input.variables_exogenas() or []
            
            if not vars_seleccionadas:
                return ui.div("Seleccione variables exógenas arriba para configurar lags.",
                             style="color: #666; font-style: italic; margin-top: 10px;")
            
            num_vars = len(vars_seleccionadas)
            
            # Panel de lags comunes (opcional)
            lags_comunes_panel = ui.div(
                ui.tags.details(
                    ui.tags.summary(
                        ui.tags.strong("Aplicar lags comunes a todas las variables"),
                        style="cursor: pointer; color: #0066cc; padding: 8px;"
                    ),
                    ui.div(
                        ui.input_checkbox_group(
                            "lags_comunes",
                            None,
                            choices={f"lag{i}": f"Lag {i}" for i in range(1, 13)},
                            selected=[],
                            inline=True
                        ),
                        ui.tags.p(
                            f"Los lags seleccionados se aplicarán a las {num_vars} variables.",
                            style="font-size: 0.85em; color: #666; margin-top: 8px;"
                        ),
                        style="padding: 10px; background-color: #e8f4f8; border-radius: 4px; margin-top: 5px;"
                    ),
                    style="margin-bottom: 15px;"
                )
            )
            
            # Lags individuales por variable (colapsables)
            lags_individuales = ui.div(
                ui.tags.p(
                    ui.tags.strong("Configuración individual de lags por variable:"),
                    style="margin-bottom: 8px;"
                )
            )
            
            var_panels = []
            for var in vars_seleccionadas:
                panel = ui.tags.details(
                    ui.tags.summary(
                        f"{var}",
                        style="cursor: pointer; padding: 6px; margin-bottom: 3px;"
                    ),
                    ui.div(
                        ui.input_checkbox_group(
                            f"lags_{var}",
                            None,
                            choices={f"lag{i}": f"Lag {i}" for i in range(1, 13)},
                            selected=[],
                            inline=True
                        ),
                        style="padding: 8px; background-color: #f8f9fa; border-left: 3px solid #0066cc; margin: 5px 0;"
                    )
                )
                var_panels.append(panel)
            
            return ui.div(
                ui.tags.p(
                    f"{num_vars} variable(s) seleccionada(s)",
                    style="font-weight: 500; color: #333; margin-bottom: 12px;"
                ),
                lags_comunes_panel,
                lags_individuales,
                *var_panels,
                style="background-color: #ffffff; padding: 12px; border-radius: 5px; margin-top: 10px; border: 1px solid #ddd;"
            )
        except:
            return ui.div("")

    @reactive.Calc
    def variables_exogenas_con_lags():
        """
        Returns dict: {backend_var: [int_lags...], ...}
        Only includes selected variables with their integer lags.
        Combines common lags + individual lags for each variable.
        """
        resultado = {}
        try:
            # Obtener lags comunes
            lags_comunes_raw = input.lags_comunes() or []
            lags_comunes_int = [int(l.replace("lag", "")) for l in lags_comunes_raw]
            
            # Para cada variable seleccionada
            seleccion = get_selected_exog()
            for var in seleccion:
                back_var = buscar_nombre_equivalente(tipo="variable", nombre=var)
                
                # Obtener lags individuales
                try:
                    lags_individuales_raw = getattr(input, f"lags_{var}")() or []
                    lags_individuales_int = [int(l.replace("lag", "")) for l in lags_individuales_raw]
                except AttributeError:
                    lags_individuales_int = []
                
                # Combinar y eliminar duplicados
                lags_combinados = list(set(lags_comunes_int + lags_individuales_int))
                lags_combinados.sort()
                
                resultado[back_var] = lags_combinados
        except:
            pass
        return resultado
    

    @output
    @render.ui
    def categoria_objetivo_select():
        categorias_dict = buscar_nombre_equivalente("Variables_Objetivo", "Categorias") or {}
        predeterminada = buscar_nombre_equivalente("Variables_Objetivo", "Predeterminado_Categoria") or list(categorias_dict.keys())[0]
        return ui.input_select(
            "categoria_objetivo",
            "Categoría:",
            choices=list(categorias_dict.keys()),
            selected=predeterminada
        )
    
    @output
    @render.ui
    def variable_objetivo_select():
        try:
            categoria = input.categoria_objetivo()
            categorias_dict = buscar_nombre_equivalente("Variables_Objetivo", "Categorias") or {}
            variables = categorias_dict.get(categoria, [])
            predeterminada = buscar_nombre_equivalente("Variables_Objetivo", "Predeterminado")
            
            # Crear diccionario con nombres formateados
            choices_dict = {var: format_variable_name(var) for var in variables}
            
            return ui.input_select(
                "variable_objetivo",
                "Variable a Predecir:",
                choices=choices_dict,
                selected=predeterminada if predeterminada in variables else (variables[0] if variables else None)
            )
        except:
            return ui.div("Seleccione una categoría")
    
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
    def modelo_select():
        disponibles = buscar_nombre_equivalente("Modelos", "Disponibles") or []
        predeterminado = buscar_nombre_equivalente("Modelos", "Predeterminado")
        return ui.input_select("modelo", "Modelo de Predicción:",
                               choices=list(disponibles),
                               selected=predeterminado)
    

    # example: call SARIMAX auto-adjust when its button is clicked
    @reactive.Effect
    @reactive.event(input.auto_ajustar_sarimax)
    def detect_best_parameters():
        if input.modelo() != "SARIMAX":
            return  # only for SARIMAX model
        
        # Validate that exogenous variables are selected
        exog_vars = get_selected_exog()
        if not exog_vars:
            ui.notification_show(
                "Debe seleccionar al menos una categoría y variable exógena antes de auto-ajustar.",
                type="warning",
                duration=5
            )
            return
        
        var_predict = input.variable_objetivo()
        selected_lags_dict = selected_lags()
        exog_cols = translate_exog_vars()
        df = cargar_df(var_predict=var_predict)
        df = add_exog_df(df=df, exog_cols=exog_cols)
        df = laggify_df(df, exog_cols, selected_lags_dict)


        order, seasonal = best_sarimax_params(df, exog_cols,
                                                column_y=translate_var(input.variable_objetivo()),
                                                periodos_a_predecir=input.periodos_futuros()
                                            )
        ui.update_numeric("p", value=order[0])
        ui.update_numeric("d", value=order[1])
        ui.update_numeric("q", value=order[2])
        ui.update_numeric("P", value=seasonal[0])
        ui.update_numeric("D", value=seasonal[1])
        ui.update_numeric("Q", value=seasonal[2])
        ui.update_numeric("s", value=seasonal[3])

    @reactive.Effect
    @reactive.event(input.auto_ajustar_xgboost)
    def detect_best_parameters_xgboost():
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

        var_predict = input.variable_objetivo()
        back_var_predict = translate_var(var_predict)
        exog_cols = translate_exog_vars()
        selected_lags_dict = selected_lags()
        
        df = cargar_df(var_predict=var_predict)
        df = add_exog_df(df=df, exog_cols=exog_cols)
        df = laggify_df(df, exog_cols, selected_lags_dict)

        best_params = best_xgboost_params(
            df=df,
            exog_cols=exog_cols,
            column_y=back_var_predict,
            periodos_a_predecir=input.periodos_futuros()
        )
        
        # Actualizar los inputs de la UI con los mejores parámetros encontrados
        if best_params:
            ui.update_numeric("n_estimators", value=best_params.get("n_estimators"))
            ui.update_numeric("max_depth", value=best_params.get("max_depth"))
            ui.update_numeric("learning_rate", value=best_params.get("learning_rate"))
            ui.update_numeric("subsample", value=best_params.get("subsample"))
            ui.update_numeric("min_child_weight", value=best_params.get("min_child_weight"))
            ui.update_numeric("colsample_bytree", value=best_params.get("colsample_bytree"))
            ui.update_numeric("reg_lambda", value=best_params.get("reg_lambda"))
            ui.update_numeric("reg_alpha", value=best_params.get("reg_alpha"))
            
            ui.notification_show(
                "Parámetros óptimos encontrados para XGBoost.",
                type="message",
                duration=5
            )
    
    # Resumen de parámetros del modelo
    @output
    @render.ui
    def resumen_parametros():
        """Muestra un resumen de los parámetros del modelo seleccionados"""
        
        
        modelo = input.modelo()
        
        if modelo == "SARIMAX":
            # Obtener parámetros
            p, d, q = input.p(), input.d(), input.q()
            P, D, Q, s = input.P(), input.D(), input.Q(), input.s()
            
            return ui.div(
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.strong("📊 Parámetros Seleccionados"),
                        style="font-size: 1.1em; margin-bottom: 10px;"
                    ),
                    ui.tags.div(
                        ui.tags.span("Modelo: ", style="font-weight: bold;"),
                        ui.tags.span(f"{modelo}", style="color: #0066cc;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Orden no estacional (p,d,q): ", style="font-weight: bold;"),
                        ui.tags.span(f"({p},{d},{q})", style="color: #28a745;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Orden estacional (P,D,Q,s): ", style="font-weight: bold;"),
                        ui.tags.span(f"({P},{D},{Q},{s})", style="color: #28a745;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Variables exógenas: ", style="font-weight: bold;"),
                        ui.tags.span(
                            ", ".join(get_selected_exog()) if get_selected_exog() else "Ninguna",
                            style="color: #6c757d;"
                        )
                    ),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0066cc;"
                )
            )
        elif modelo == "XGBoost":
            n_estimators = input.n_estimators()
            max_depth = input.max_depth()
            learning_rate = input.learning_rate()
            subsample = input.subsample()
            min_child_weight = input.min_child_weight()
            colsample_bytree = input.colsample_bytree()
            reg_lambda = input.reg_lambda()
            reg_alpha = input.reg_alpha()

            return ui.div(
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.strong("📊 Parámetros Seleccionados"),
                        style="font-size: 1.1em; margin-bottom: 10px;"
                    ),
                    ui.tags.div(
                        ui.tags.span("Modelo: ", style="font-weight: bold;"),
                        ui.tags.span(f"{modelo}", style="color: #0066cc;")
                    ),
                    ui.tags.div(
                        ui.tags.strong("Hiperparámetros:"),
                        ui.tags.ul(
                            ui.tags.li(f"N Estimators: {n_estimators}"),
                            ui.tags.li(f"Max Depth: {max_depth}"),
                            ui.tags.li(f"Learning Rate: {learning_rate}"),
                            ui.tags.li(f"Subsample: {subsample}"),
                            ui.tags.li(f"Min Child Weight: {min_child_weight}"),
                            ui.tags.li(f"Colsample ByTree: {colsample_bytree}"),
                            ui.tags.li(f"Reg Lambda: {reg_lambda}"),
                            ui.tags.li(f"Reg Alpha: {reg_alpha}"),
                            style="list-style-type: none; padding-left: 10px; margin-top: 5px;"
                        )
                    ),
                    ui.tags.div(
                        ui.tags.span("Variables exógenas: ", style="font-weight: bold;"),
                        ui.tags.span(
                            ", ".join(get_selected_exog()) if get_selected_exog() else "Ninguna",
                            style="color: #6c757d;"
                        )
                    ),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0066cc;"
                )
            )
        else:
            return ui.div(
                ui.tags.p(f"Modelo seleccionado: {modelo}"),
                style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;"
            )
    
    # Duplicate for results panel (unique ID to avoid conflicts)
    @output
    @render.ui
    def resumen_parametros_results():
        """Copy of resumen_parametros for results step"""
        modelo = input.modelo()
        
        if modelo == "SARIMAX":
            p, d, q = input.p(), input.d(), input.q()
            P, D, Q, s = input.P(), input.D(), input.Q(), input.s()
            
            return ui.div(
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.span("Modelo: ", style="font-weight: bold;"),
                        ui.tags.span(f"{modelo}", style="color: #0066cc;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Orden (p,d,q): ", style="font-weight: bold;"),
                        ui.tags.span(f"({p},{d},{q})", style="color: #28a745;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Estacional (P,D,Q,s): ", style="font-weight: bold;"),
                        ui.tags.span(f"({P},{D},{Q},{s})", style="color: #28a745;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Exógenas: ", style="font-weight: bold;"),
                        ui.tags.span(
                            ", ".join(get_selected_exog()) if get_selected_exog() else "Ninguna",
                            style="color: #6c757d;"
                        )
                    ),
                    style="font-size: 0.9em;"
                )
            )
        elif modelo == "XGBoost":
            return ui.div(
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.span("Modelo: ", style="font-weight: bold;"),
                        ui.tags.span(f"{modelo}", style="color: #0066cc;")
                    ),
                    ui.tags.div(
                        ui.tags.span("Exógenas: ", style="font-weight: bold;"),
                        ui.tags.span(
                            ", ".join(get_selected_exog()) if get_selected_exog() else "Ninguna",
                            style="color: #6c757d;"
                        )
                    ),
                    style="font-size: 0.9em;"
                )
            )
        return ui.div(f"Modelo: {modelo}")
    
    # Salida del MAPE
    @output
    @render.ui
    def mape_output():
        """Muestra las métricas del modelo"""
        
        
        metrics = metricas.get()
        if isinstance(metrics, dict):
            mape = metrics.get('MAPE', 'N/A')
            rmse = metrics.get('RMSE', 'N/A')
            mae = metrics.get('MAE', 'N/A')
        else:
            mape = "No calculado"
            rmse = "No calculado"
            mae = "No calculado"
        
        return ui.div(
            ui.tags.div(
                ui.tags.strong("MAPE del Test: "),
                ui.tags.span(str(mape), style="color: green; font-size: 1.2em;"),
                style="margin-bottom: 10px;"
            ),
            ui.tags.div(
                ui.tags.strong("RMSE: "),
                ui.tags.span(str(rmse)),
                style="margin-bottom: 5px;"
            ),
            ui.tags.div(
                ui.tags.strong("MAE: "),
                ui.tags.span(str(mae))
            )
        )
    
    # Gráfica de predicción
    @output
    @render.plot
    def grafica_prediccion():
        front_var = input.variable_objetivo()

        back_var = translate_var(front_var)
        df = cargar_df(front_var) # Aquí no hace falta añadir exógenas ni lags, solo es para el plot

        periodos_a_predecir = input.periodos_futuros()

        title = get_title(front_var)
        ylabel = buscar_nombre_equivalente(tipo="nombre_descriptivo", nombre=back_var)

        # ====== NUEVO: comprobar si hay predicciones ======
        pred = predicciones.get()
        if pred is None:
            fig, ax = plt.subplots()
            ax.text(
                0.5, 0.5,
                "Primero entrene el modelo\n(haga clic en 'Entrenar')",
                ha="center", va="center", fontsize=12
            )
            ax.axis("off")
            return fig
        # ==================================================

        if input.modelo() == "XGBoost":
            ax = plot_xgb_predictions(
                df=df,
                pred=pred,
                title=title,
                ylabel=ylabel,
                xlabel='Fecha',
                column_y=back_var,
                periodos_a_predecir=periodos_a_predecir
            )
        else:
            ax = plot_predictions(
                df=df,
                pred=pred,
                title=title,
                ylabel=ylabel,
                xlabel='Fecha',
                column_y=back_var,
                periodos_a_predecir=periodos_a_predecir
            )
        fig = ax.get_figure()

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    @output
    @render.table
    def tabla_predicciones():
        if predicciones.get() is None:
            return pd.DataFrame()  # empty DataFrame if no predictions yet
        else:
            # assure that the output is a DataFrame
            pred = predicciones.get()
            variable_objetivo = input.variable_objetivo()
            if isinstance(pred, pd.Series):
                s = pred
                result = s.reset_index()
                result.columns = ['Fecha',  variable_objetivo]
                result['Fecha'] = pd.to_datetime(result['Fecha'])
                return result
            elif isinstance(pred, pd.DataFrame):
                return pred
            else:
                raise ValueError("Predicciones no es una Serie ni un DataFrame")
    
    # Acción al entrenar el modelo
    @reactive.Effect
    @reactive.event(input.entrenar_modelo)
    def entrenar_modelo():
        # Validate that exogenous variables are selected
        exog_vars = get_selected_exog()
        if not exog_vars:
            ui.notification_show(
                "Debe seleccionar al menos una categoría y variable exógena antes de entrenar el modelo.",
                type="warning",
                duration=5
            )
            return
        
        modelo = input.modelo()
        # get the parameters
        if modelo == "SARIMAX":
            entrenar_modelo_sarimax()
        elif modelo == "XGBoost":
            entrenar_modelo_xgboost()
        else:
            raise Exception(f"Modelo '{modelo}' no implementado aún")
        
    def entrenar_modelo_sarimax():
        order = (input.p(), input.d(), input.q())
        seasonal_order = (input.P(), input.D(), input.Q(), input.s())
        front_var_predict = input.variable_objetivo()
        
        back_var_predict = translate_var(front_var_predict)
    
        exog_cols = translate_exog_vars()

        periodos_a_predecir = input.periodos_futuros()

        df = cargar_df(front_var_predict)
        df = add_exog_df(df=df, exog_cols=exog_cols)
        df = laggify_df(df, exog_cols, selected_lags())


        model_train_metrics(
            df=df,
            periodos_a_predecir=periodos_a_predecir,
            exog_cols=exog_cols,
            order=order,
            seasonal_order=seasonal_order,
            back_var_predict=back_var_predict
        )

        model_predictions(
            df=df,
            periodos_a_predecir=periodos_a_predecir,
            exog_cols=exog_cols,
            order=order,
            seasonal_order=seasonal_order,
            back_var_predict=back_var_predict
        )
        

        
        

    def model_train_metrics(df, periodos_a_predecir, exog_cols, order, seasonal_order, back_var_predict):
        df_model = df.iloc[:-periodos_a_predecir]   # todos menos los últimos N

        n_total = len(df_model)
        n_train = int(n_total * 0.7)

        df_model_train = df_model.iloc[:n_train].copy()
        df_model_test  = df_model.iloc[n_train:].copy()

        sarimax_model = create_sarimax_model(
            train=df_model_train,
            exog_cols=exog_cols,
            column_y=back_var_predict,
            order=order,
            seasonal_order=seasonal_order,
        )
        mape, rmse, mae = compute_metrics(
            df_train=df_model_train,
            df_test=df_model_test,
            indicador=back_var_predict,
            model = sarimax_model,
            exog_cols=df_model_test[exog_cols]
        )
        metricas.set({
            'MAPE': round(mape, 2),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        })

    def model_predictions(df, periodos_a_predecir, exog_cols, order, seasonal_order, back_var_predict):
        df_model = df.iloc[:-periodos_a_predecir]   # todos menos los últimos N

        sarimax_model = create_sarimax_model(
            train=df_model,
            exog_cols=exog_cols,
            column_y=back_var_predict,
            order=order,
            seasonal_order=seasonal_order,
        )
        start_pred = df.index[-periodos_a_predecir]
        end_pred = df.index[-1]
        df_pred = df[-periodos_a_predecir:]
        predictions = predict_sarimax(
            model_fit=sarimax_model,
            start=start_pred,
            end=end_pred,
            exog_cols=df_pred[exog_cols] if exog_cols else None
        )
        predicciones.set(predictions)

    def entrenar_modelo_xgboost():
        front_var_predict = input.variable_objetivo()
        back_var_predict = translate_var(front_var_predict)
        exog_cols = translate_exog_vars()
        periodos_a_predecir = input.periodos_futuros()

        df = cargar_df(front_var_predict)
        df = add_exog_df(df=df, exog_cols=exog_cols)
        df = laggify_df(df, exog_cols, selected_lags())

        model_train_metrics_xgboost(
            df=df,
            periodos_a_predecir=periodos_a_predecir,
            exog_cols=exog_cols,
            back_var_predict=back_var_predict
        )
        
        model_predictions_xgboost(
            df=df,
            periodos_a_predecir=periodos_a_predecir,
            exog_cols=exog_cols,
            back_var_predict=back_var_predict
        )

    def model_train_metrics_xgboost(df, periodos_a_predecir, exog_cols, back_var_predict):
        df_model = df.iloc[:-periodos_a_predecir]

        n_total = len(df_model)
        n_train = int(n_total * 0.7)

        df_model_train = df_model.iloc[:n_train].copy()
        df_model_test  = df_model.iloc[n_train:].copy()

        xgb_params = {
            "n_estimators": input.n_estimators(),
            "max_depth": input.max_depth(),
            "learning_rate": input.learning_rate(),
            "min_child_weight": input.min_child_weight(),
            "subsample": input.subsample(),
            "colsample_bytree": input.colsample_bytree(),
            "reg_lambda": input.reg_lambda(),
            "reg_alpha": input.reg_alpha(),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        }

        xgb_model = create_xgboost_model(
            train=df_model_train,
            exog_cols=exog_cols,
            column_y=back_var_predict,
            xgb_params=xgb_params
        )
        
        X_test = _build_X(df_model_test, exog_cols, back_var_predict)
        y_test = df_model_test[back_var_predict]
        
        preds = xgb_model.predict(X_test)
        
        mape = mean_absolute_error(y_test, preds) / y_test.mean() * 100
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        metricas.set({
            'MAPE': round(mape, 2),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        })

    def model_predictions_xgboost(df, periodos_a_predecir, exog_cols, back_var_predict):
        df_model = df.iloc[:-periodos_a_predecir]

        xgb_params = {
            "n_estimators": input.n_estimators(),
            "max_depth": input.max_depth(),
            "learning_rate": input.learning_rate(),
            "min_child_weight": input.min_child_weight(),
            "subsample": input.subsample(),
            "colsample_bytree": input.colsample_bytree(),
            "reg_lambda": input.reg_lambda(),
            "reg_alpha": input.reg_alpha(),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        }

        xgb_model = create_xgboost_model(
            train=df_model,
            exog_cols=exog_cols,
            column_y=back_var_predict,
            xgb_params=xgb_params
        )
        
        df_future = df.iloc[-periodos_a_predecir:].copy()
        X_future = _build_X(df_future, exog_cols, back_var_predict)
        
        preds = xgb_model.predict(X_future)
        
        predictions_series = pd.Series(data=preds, index=df_future.index)
        predicciones.set(predictions_series)