from shiny import ui, module

from back.data_management import buscar_nombre_equivalente

@module.ui
def predicciones_ui():
    return ui.div(
        # Step Indicator (rendered dynamically from server)
        ui.output_ui("step_indicator"),
        
        # Wizard Container
        ui.div(
            # Step 1: Datos
            ui.output_ui("step_panel_1"),
            
            # Step 2: Variables
            ui.output_ui("step_panel_2"),
            
            # Step 3: Modelo
            ui.output_ui("step_panel_3"),
            
            # Step 4: Resultados
            ui.output_ui("step_panel_4"),
            
            class_="wizard-container"
        )
    )
