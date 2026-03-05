from pathlib import Path
from shiny import App, ui, reactive, render
from front.modules.escenarios.ui import escenarios_ui
from front.modules.escenarios.server import escenarios_server
from front.modules.predicciones.ui import predicciones_ui
from front.modules.predicciones.server import predicciones_server
#from front.modules.carga_generacion.ui import carga_generacion_ui
#from front.modules.carga_generacion.server import carga_generacion_server
from project_config import get_config
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="images/styles.css"),
    ),
    # Styled App Header
    ui.div(
        ui.div(
            ui.img(src="images/logo_sicuma.png", height="60px", class_="logo-img"),
            ui.h1(get_config("frontend.shiny.title")),
            class_="app-header-content"
        ),
        class_="app-header"
    ),
    ui.output_ui("home_blocks"),
    ui.output_ui("selected_module"),
)


def server(input, output, session):
    # Llamar al servidor del módulo de predicciones
    predicciones_server("predicciones")
    escenarios_server("escenarios")
    #carga_generacion_server("carga_generacion")

    selected_module_rv = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.open_predicciones)
    def _open_predicciones():
        selected_module_rv.set("predicciones")

    @reactive.Effect
    @reactive.event(input.open_escenarios)
    def _open_escenarios():
        selected_module_rv.set("escenarios")

    @reactive.Effect
    @reactive.event(input.back_to_home)
    def _back_to_home():
        selected_module_rv.set(None)

    @output
    @render.ui
    def home_blocks():
        if selected_module_rv.get() is not None:
            return ui.div()
        return ui.div(
            ui.h3("¿Qué quieres hacer?", style="text-align:center; margin-bottom:8px;"),
            ui.tags.p(
                "Selecciona el módulo con el que quieres trabajar",
                style="text-align:center; color:#475569; margin-bottom:24px;",
            ),
            ui.tags.div(
                ui.tags.div(
                    ui.input_action_button(
                        "open_predicciones",
                        ui.tags.div(
                            ui.tags.div("\U0001f52e", style="font-size:2.5rem; margin-bottom:8px;"),
                            ui.tags.div("Predicciones", style="font-size:1.25rem; font-weight:700; margin-bottom:6px;"),
                            ui.tags.div(
                                "Genera pronósticos con SARIMAX/XGBoost y compara métricas del modelo.",
                                style="font-size:0.85rem; color:#64748b; font-weight:400;",
                            ),
                        ),
                        class_="esc-type-card",
                    ),
                    style="width:300px; height:220px; display:flex; flex-shrink:0;",
                ),
                ui.tags.div(
                    ui.input_action_button(
                        "open_escenarios",
                        ui.tags.div(
                            ui.tags.div("📈", style="font-size:2.5rem; margin-bottom:8px;"),
                            ui.tags.div("Escenarios", style="font-size:1.25rem; font-weight:700; margin-bottom:6px;"),
                            ui.tags.div(
                                "Simula escenarios futuros modificando exógenas y analiza el impacto en la predicción.",
                                style="font-size:0.85rem; color:#64748b; font-weight:400;",
                            ),
                        ),
                        class_="esc-type-card",
                    ),
                    style="width:300px; height:220px; display:flex; flex-shrink:0;",
                ),
                style="display:flex; flex-direction:row; justify-content:center; gap:24px;",
            ),
            style="margin:0 auto; padding-top:40px;",
        )

    @output
    @render.ui
    def selected_module():
        sel = selected_module_rv.get()
        if sel is None:
            return ui.div()

        if sel == "predicciones":
            content = predicciones_ui("predicciones")
        else:
            content = escenarios_ui("escenarios")

        return ui.div(
            ui.div(
                ui.input_action_button("back_to_home", "← Volver"),
                class_="module-back-row",
            ),
            content,
            class_="module-wrapper",
        )



# Directorio de la aplicación
app_dir = Path(__file__).parent

# Crear la aplicación Shiny
app = App(app_ui, server, static_assets=app_dir / "front" / "www")

# Punto de entrada para ejecución
if __name__ == "__main__":
    from shiny import run_app
    run_app(app, port=8001)