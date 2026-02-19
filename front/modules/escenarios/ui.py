from shiny import module, ui


@module.ui
def escenarios_ui():
    return ui.div(
        ui.output_ui("step_indicator"),
        ui.div(
            ui.output_ui("step_panel_1"),
            ui.output_ui("step_panel_2"),
            ui.output_ui("step_panel_3"),
            ui.output_ui("step_panel_4"),
            class_="wizard-container",
        ),
    )
