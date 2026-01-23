from pathlib import Path
from shiny import App, ui, module
#from front.modules.escenarios.ui import escenarios_ui
#from front.modules.escenarios.server import escenarios_server
from front.modules.predicciones.ui import predicciones_ui
from front.modules.predicciones.server import predicciones_server
#from front.modules.carga_generacion.ui import carga_generacion_ui
#from front.modules.carga_generacion.server import carga_generacion_server
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="images/styles.css"),
    ),
    # Styled App Header
    ui.div(
        ui.div(
            ui.img(src="images/logo_sicuma.png", height="60px", class_="logo-img"),
            ui.h1("Sistema de Predicción de Turismo"),
            ui.p("Nuestro enfoque imagina que estamos en Noviembre de 2024"),
            class_="app-header-content"
        ),
        class_="app-header"
    ),
    ui.navset_tab(
        # panel de predicción
        ui.nav_panel(
            "Predicciones",
            predicciones_ui("predicciones"),
        ),
        #ui.nav_panel(
        #    "Escenarios",
        #    escenarios_ui("escenarios"),
        #),
        # ui.nav_panel(
        #     "Carga y Generación de Datos",
        #     carga_generacion_ui("carga_generacion"),
        # ),
    ),
)


def server(input, output, session):
    # Llamar al servidor del módulo de predicciones
    predicciones_server("predicciones"),
    #escenarios_server("escenarios"),
    #carga_generacion_server("carga_generacion")



# Directorio de la aplicación
app_dir = Path(__file__).parent

# Crear la aplicación Shiny
app = App(app_ui, server, static_assets=app_dir / "front" / "www")

# Punto de entrada para ejecución
if __name__ == "__main__":
    from shiny import run_app
    run_app(app, port=8000)