from shiny import ui, render, module, reactive

@module.server
def predicciones_server(input, output, session):

    current_step = reactive.Value(1)
    TOTAL_STEPS = 4

    ##########################################################################################
    #Panel 1: Select Indicator
    ##########################################################################################
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
                ui.div(),  # Empty left side
                ui.input_action_button("btn_next_1", "Siguiente →", class_="btn-step-next"),
                class_="step-navigation"
            ),
            class_="step-panel"
        )
    
    ##########################################################################################
    #Panel 2: Select Variables
    ##########################################################################################
    @output
    @render.ui
    def step_panel_2():
        if current_step.get() != 2:
            return ui.div()
        
    ##########################################################################################
    #Panel 3: Config Variables
    ##########################################################################################
    @output
    @render.ui
    def step_panel_3():
        if current_step.get() != 3:
            return ui.div()
        
    ##########################################################################################
    #Panel 4: Play with Model and Variables
    ##########################################################################################
    @output
    @render.ui
    def step_panel_4():
            if current_step.get() != 4:
            return ui.div()