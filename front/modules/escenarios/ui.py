from shiny import ui, module
from back.data_management import buscar_nombre_equivalente
from front.utils.formatters import format_variable_name

# Variables disponibles (hardcoded or fetched)
try:
    choices_obj = list(buscar_nombre_equivalente("Variables_Objetivo", "Disponibles"))
    selected_obj = buscar_nombre_equivalente("Variables_Objetivo", "Predeterminado")
except Exception:
    choices_obj = ["Turistas"]
    selected_obj = "Turistas"

@module.ui
def escenarios_ui():
    return ui.div(
        # Step Indicator (rendered dynamically from server)
        ui.output_ui("step_indicator"),
        
        # Wizard Container
        ui.div(
            # Step 1: Datos
            ui.output_ui("step_panel_1"),
            
            # Step 2: Variables
            ui.output_ui("step_panel_2"),
            
            # Step 3: Escenario
            ui.output_ui("step_panel_3"),
            
            # Step 4: Modelo
            ui.output_ui("step_panel_4"),
            
            # Step 5: Resultados
            ui.output_ui("step_panel_5"),
            
            class_="wizard-container"
        )
    )
