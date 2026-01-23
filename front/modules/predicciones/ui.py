from shiny import ui, module


@module.ui
def predicciones_ui():
    return ui.div(
        # Step Indicator (rendered dynamically from server)
        ui.output_ui("step_indicator"),
        
        # Wizard Container
        ui.div(
            # Step 1: Select Indicator
            ui.output_ui("step_panel_1"),
            
            # Step 2: Select Variables
            ui.output_ui("step_panel_2"),
            
            # Step 3: Config Variables
            ui.output_ui("step_panel_3"),
            
            # Step 4: Play with Model and Variables
            ui.output_ui("step_panel_4"),
            
            class_="wizard-container"
        )
    )
