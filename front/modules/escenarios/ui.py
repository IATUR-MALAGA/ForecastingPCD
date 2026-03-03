from shiny import module, ui

@module.ui
def escenarios_ui():
    return ui.div(
        ui.div(
            ui.output_ui("step_indicator"),
            ui.output_ui("step_panel_0"),
            # --- Escenarios futuros ---
            ui.output_ui("step_panel_1"),
            ui.output_ui("step_panel_2"),
            ui.output_ui("step_panel_3"),
            ui.output_ui("step_panel_4"),
            # --- Escenarios pasados (placeholder) ---
            ui.output_ui("step_panel_pasado"),
            class_="wizard-container",
        )
    )