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
            ui.div(
                ui.img(src="images/logo_sicuma.png", height="60px", class_="logo-img logo-sicuma"),
                ui.img(src="images/logo_iatur.png", height="60px", class_="logo-img logo-iatur"),
                class_="header-logos"
            ),
            ui.h1(get_config("frontend.shiny.title")),
            class_="app-header-content"
        ),
        class_="app-header"
    ),
    ui.output_ui("landing_page"),
    ui.output_ui("home_blocks"),
    ui.output_ui("selected_module"),
)


def server(input, output, session):
    # Llamar al servidor del módulo de predicciones
    predicciones_server("predicciones")
    escenarios_server("escenarios")
    #carga_generacion_server("carga_generacion")

    # "landing" = landing page, "home" = module selection, "predicciones"/"escenarios" = module
    current_view = reactive.Value("landing")

    @reactive.Effect
    @reactive.event(input.btn_comenzar)
    def _comenzar():
        current_view.set("home")

    @reactive.Effect
    @reactive.event(input.open_predicciones)
    def _open_predicciones():
        current_view.set("predicciones")

    @reactive.Effect
    @reactive.event(input.open_escenarios)
    def _open_escenarios():
        current_view.set("escenarios")

    @reactive.Effect
    @reactive.event(input.back_to_home)
    def _back_to_home():
        current_view.set("home")

    @reactive.Effect
    @reactive.event(input.back_to_landing)
    def _back_to_landing():
        current_view.set("landing")

    @output
    @render.ui
    def landing_page():
        if current_view.get() != "landing":
            return ui.div()
        return ui.div(
            ui.div(
                ui.div(
                    ui.tags.div("\U0001f30d", style="font-size:4rem; margin-bottom:1rem;"),
                    ui.h2(
                        "Plataforma de Predicción y Análisis Turístico",
                        class_="landing-title",
                    ),
                    ui.tags.p(
                        "Bienvenido al sistema inteligente de predicción de demanda turística. "
                        "Esta plataforma te permite generar pronósticos precisos utilizando modelos "
                        "avanzados de machine learning, así como simular escenarios pasados y futuros "
                        "modificando variables exógenas para analizar su impacto en la predicción.",
                        class_="landing-description",
                    ),
                    
                    ui.input_action_button(
                        "btn_comenzar",
                        ui.tags.span("Comenzar", style="margin-right:0.5rem;"),
                        class_="btn-comenzar",
                    ),
                    class_="landing-card",
                ),
                class_="landing-container",
            ),
        )

    @output
    @render.ui
    def home_blocks():
        if current_view.get() != "home":
            return ui.div()
        return ui.div(
            ui.div(
                ui.input_action_button("back_to_landing", "← Volver"),
                class_="module-back-row",
            ),
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
                                "Genera pronósticos con modelos de Machine Learning y compara métricas del modelo.",
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
                                "Simula escenarios pasados y futuros modificando variables predictoras y analiza el impacto en la predicción.",
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
        sel = current_view.get()
        if sel not in ("predicciones", "escenarios"):
            return ui.div()

        if sel == "predicciones":
            content = predicciones_ui("predicciones")
        else:
            content = escenarios_ui("escenarios")

        return ui.div(
            ui.div(
                ui.input_action_button("back_to_home", " Inicio"),
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