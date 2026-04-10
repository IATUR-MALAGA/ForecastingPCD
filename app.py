from pathlib import Path
from shiny import App, ui, reactive, render
from front.modules.escenarios.ui import escenarios_ui
from front.modules.escenarios.server import escenarios_server
from front.modules.predicciones.ui import predicciones_ui
from front.modules.predicciones.server import predicciones_server

# from front.modules.carga_generacion.ui import carga_generacion_ui
# from front.modules.carga_generacion.server import carga_generacion_server
from project_config import get_config


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="images/styles.css"),
    ),
    # Styled App Header
    ui.div(
        ui.div(
            ui.div(
                ui.img(
                    src="images/logo_sicuma.png",
                    height="60px",
                    class_="logo-img logo-sicuma",
                ),
                ui.img(
                    src="images/logo_iatur.png",
                    height="60px",
                    class_="logo-img logo-iatur",
                ),
                class_="header-logos",
            ),
            ui.h1(get_config("frontend.shiny.title")),
            class_="app-header-content",
        ),
        class_="app-header",
    ),
    ui.div(
        ui.input_action_button(
            "open_help_docs",
            ui.HTML(
                "<span style='display:inline-flex; align-items:center; gap:6px;'>"
                "<svg width='25' height='25' viewBox='0 0 24 24' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' style='display:block;'>"
                "<circle cx='12' cy='12' r='9' stroke='white' stroke-width='2'/>"
                "<circle cx='12' cy='12' r='3.5' stroke='white' stroke-width='2'/>"
                "<line x1='12' y1='3' x2='12' y2='8' stroke='white' stroke-width='2'/>"
                "<line x1='21' y1='12' x2='16' y2='12' stroke='white' stroke-width='2'/>"
                "<line x1='12' y1='21' x2='12' y2='16' stroke='white' stroke-width='2'/>"
                "<line x1='3' y1='12' x2='8' y2='12' stroke='white' stroke-width='2'/>"
                "</svg>"
                "<span>Ayuda</span>"
                "</span>"
            ),
            class_="btn-help-docs help-floating-btn",
        ),
        class_="help-floating-btn-wrap",
    ),
    ui.output_ui("landing_page"),
    ui.output_ui("home_blocks"),
    ui.output_ui("help_docs"),
    ui.output_ui("selected_module"),
)


def server(input, output, session):
    # Llamar al servidor del modulo de predicciones
    predicciones_server("predicciones")
    escenarios_server("escenarios")
    # carga_generacion_server("carga_generacion")

    # "landing" = landing page, "home" = module selection, "predicciones"/"escenarios" = module
    current_view = reactive.Value("landing")
    previous_view = reactive.Value("home")

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
    @reactive.event(input.open_help_docs)
    def _open_help_docs():
        current = current_view.get()
        if current != "help_docs":
            previous_view.set(current)
        current_view.set("help_docs")

    @reactive.Effect
    @reactive.event(input.back_to_home)
    def _back_to_home():
        current_view.set("home")

    @reactive.Effect
    @reactive.event(input.back_from_help)
    def _back_from_help():
        current_view.set(previous_view.get() or "home")

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
                    ui.tags.div(
                        "\U0001f30d", style="font-size:4rem; margin-bottom:1rem;"
                    ),
                    ui.h2(
                        "Plataforma de Prediccion y Analisis Turistico",
                        class_="landing-title",
                    ),
                    ui.tags.p(
                        "Bienvenido al sistema inteligente de prediccion de demanda turistica. "
                        "Esta plataforma te permite generar pronosticos precisos utilizando modelos "
                        "avanzados de machine learning, asi como simular escenarios pasados y futuros "
                        "modificando variables exogenas para analizar su impacto en la prediccion.",
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
                ui.input_action_button("back_to_landing", "<- Volver"),
                class_="module-back-row",
            ),
            ui.h3("Que quieres hacer?", style="text-align:center; margin-bottom:8px;"),
            ui.tags.p(
                "Selecciona el modulo con el que quieres trabajar",
                style="text-align:center; color:#475569; margin-bottom:24px;",
            ),
            ui.tags.div(
                ui.tags.div(
                    ui.input_action_button(
                        "open_predicciones",
                        ui.tags.div(
                            ui.tags.div(
                                "\U0001f52e",
                                style="font-size:2.5rem; margin-bottom:8px;",
                            ),
                            ui.tags.div(
                                "Predicciones",
                                style="font-size:1.25rem; font-weight:700; margin-bottom:6px;",
                            ),
                            ui.tags.div(
                                "Genera pronosticos con modelos de Machine Learning y compara metricas del modelo.",
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
                            ui.tags.div(
                                "\U0001f4c8",
                                style="font-size:2.5rem; margin-bottom:8px;",
                            ),
                            ui.tags.div(
                                "Escenarios",
                                style="font-size:1.25rem; font-weight:700; margin-bottom:6px;",
                            ),
                            ui.tags.div(
                                "Simula escenarios pasados y futuros modificando variables predictoras y analiza el impacto en la prediccion.",
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
    def help_docs():
        if current_view.get() != "help_docs":
            return ui.div()

        return ui.div(
            ui.div(
                ui.div(
                    ui.input_action_button("back_from_help", "<- Volver al inicio"),
                    class_="module-back-row",
                ),
                ui.div(
                    ui.h3("Ayuda y documentacion", class_="help-docs-title"),
                    ui.tags.p(
                        "La plataforma implementa un flujo analitico guiado para el modelado de la demanda "
                        "turistica y el analisis de escenarios de simulacion. Su objetivo es facilitar la "
                        "construccion de predicciones consistentes, comparables y trazables a partir de una "
                        "variable objetivo y un conjunto de variables explicativas o exogenas. "
                        "La aplicacion ha sido disenada para combinar rigor metodologico con una experiencia "
                        "de uso estructurada, de modo que el usuario pueda configurar modelos, validar "
                        "supuestos, ejecutar calculos y analizar resultados sin perder continuidad entre pasos. "
                        "Toda la logica de la plataforma esta orientada a reducir errores de configuracion, "
                        "mejorar la interpretacion analitica y favorecer la reproducibilidad del proceso.",
                        class_="help-docs-intro",
                    ),
                    ui.div(
                        ui.tags.div(
                            "Secciones de ayuda", class_="help-docs-sections-title"
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "1. Vision general, estructura funcional y navegacion"
                                ),
                                ui.tags.p(
                                    "La plataforma se organiza en varios modulos conectados mediante un sistema "
                                    "de navegacion reactiva que permite moverse entre la pantalla de inicio, "
                                    "el modulo de predicciones, el modulo de escenarios y esta seccion de ayuda "
                                    "sin perder el contexto de trabajo. Esta arquitectura facilita una experiencia "
                                    "fluida y continua, especialmente en procesos analiticos donde el usuario "
                                    "necesita consultar documentacion mientras configura un modelo o revisa resultados."
                                ),
                                ui.tags.p(
                                    "El flujo de uso sigue una logica guiada por pasos. Antes de ejecutar un "
                                    "modelo, el usuario debe definir la variable objetivo, seleccionar las "
                                    "predictoras compatibles, aplicar los filtros necesarios y, finalmente, "
                                    "escoger el algoritmo y sus parametros principales. Esta secuencia no solo "
                                    "ordena la experiencia de uso, sino que tambien ayuda a mantener coherencia "
                                    "entre las decisiones analiticas tomadas en cada fase."
                                ),
                                ui.tags.p(
                                    "El boton de ayuda permanece disponible como acceso global, de manera que "
                                    "puede consultarse desde cualquier vista sin interrumpir completamente el "
                                    "trabajo en curso. Al volver atras, el usuario regresa a la pantalla previa, "
                                    "lo que favorece la consulta contextual de la documentacion y mejora la "
                                    "usabilidad general de la plataforma."
                                ),
                                ui.tags.p(
                                    "Desde el punto de vista funcional, la aplicacion esta pensada para cubrir "
                                    "dos necesidades principales: por un lado, la generacion de predicciones "
                                    "sobre variables de interes turistico; por otro, la construccion de escenarios "
                                    "alternativos que permitan analizar como cambiarian los resultados si se "
                                    "modificaran determinadas condiciones o factores exogenos."
                                ),
                                value="vision_navegacion",
                            ),
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "2. Modulo de Predicciones: configuracion, ejecucion y uso analitico"
                                ),
                                ui.tags.p(
                                    "El modulo de Predicciones permite estimar el comportamiento futuro de una "
                                    "variable objetivo a partir de su historico y de un conjunto de variables "
                                    "explicativas seleccionadas por el usuario. Este modulo constituye la via "
                                    "principal para generar pronosticos operativos y compararlos bajo distintos "
                                    "enfoques de modelado."
                                ),
                                ui.tags.ol(
                                    ui.tags.li(
                                        "Definicion del objetivo: en el primer panel se selecciona la variable "
                                        "que se desea predecir. La plataforma muestra su informacion asociada, "
                                        "incluyendo temporalidad, granularidad, unidad de medida, fuente y "
                                        "descripcion, lo que ayuda a contextualizar el analisis antes de comenzar."
                                    ),
                                    ui.tags.li(
                                        "Seleccion de predictoras: en el segundo panel se muestran las variables "
                                        "compatibles con el objetivo. Esta compatibilidad se basa en criterios "
                                        "como la cobertura temporal, la estructura de datos y la coherencia con "
                                        "la serie objetivo, evitando combinaciones poco solidas desde el punto "
                                        "de vista metodologico."
                                    ),
                                    ui.tags.li(
                                        "Aplicacion de filtros: en el tercer panel se configuran filtros "
                                        "especificos sobre las variables disponibles. Estos filtros permiten "
                                        "acotar el analisis por segmentos, categorias o dimensiones relevantes, "
                                        "manteniendo la sincronizacion con la variable objetivo y garantizando "
                                        "consistencia en la muestra utilizada."
                                    ),
                                    ui.tags.li(
                                        "Configuracion del modelo: en el cuarto panel se escoge el algoritmo "
                                        "de prediccion, se activan o desactivan exogenas, se define el horizonte "
                                        "de pronostico y se lanza la ejecucion de forma explicita mediante el "
                                        "boton de calculo."
                                    ),
                                ),
                                ui.tags.p(
                                    "Una vez ejecutado el modelo, la plataforma devuelve una salida homologada "
                                    "que facilita el analisis comparativo. El usuario dispone de un grafico "
                                    "interactivo para observar la evolucion temporal de la prediccion, una tabla "
                                    "detallada con los valores estimados y un conjunto de metricas de error que "
                                    "permiten evaluar la calidad del ajuste. Esta combinacion de elementos visuales "
                                    "y tabulares facilita tanto la interpretacion rapida como la revision tecnica "
                                    "del resultado."
                                ),
                                ui.tags.p(
                                    "Este modulo es especialmente util cuando se busca generar previsiones de "
                                    "demanda para apoyo a la planificacion, anticipacion de necesidades operativas "
                                    "o evaluacion preliminar del comportamiento esperado bajo las condiciones "
                                    "historicamente observadas en los datos."
                                ),
                                value="modulo_predicciones",
                            ),
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "3. Modulo de Escenarios: simulacion de contextos pasados y futuros"
                                ),
                                ui.tags.p(
                                    "El modulo de Escenarios amplía la capacidad analitica de la plataforma al "
                                    "permitir estudiar como variaria la prediccion si se modificaran determinadas "
                                    "variables exogenas. A diferencia del modulo de Predicciones, cuyo objetivo "
                                    "principal es estimar valores futuros bajo la estructura observada en los datos, "
                                    "el modulo de Escenarios esta orientado a la simulacion, la evaluacion de "
                                    "sensibilidad y el analisis de impacto."
                                ),
                                ui.tags.p(
                                    "El usuario comienza seleccionando el modo de trabajo: escenario pasado o "
                                    "escenario futuro. A partir de ese punto, la estructura general del flujo "
                                    "mantiene la misma logica que en el modulo de Predicciones: definicion del "
                                    "objetivo, seleccion de predictoras compatibles, aplicacion de filtros y "
                                    "configuracion final del modelo. Esta homogeneidad metodologica permite que "
                                    "los resultados obtenidos en ambos modulos sean comparables y consistentes."
                                ),
                                ui.tags.p(
                                    "En los escenarios futuros, el usuario define un horizonte de prediccion y "
                                    "cumplimenta una matriz editable con los valores que desea asignar a cada "
                                    "variable exogena en cada periodo futuro. La plataforma genera automaticamente "
                                    "el calendario correspondiente segun la temporalidad del objetivo, valida que "
                                    "las variables activas dispongan de informacion completa y construye una "
                                    "estructura de entrada preparada para el calculo del escenario."
                                ),
                                ui.tags.p(
                                    "En los escenarios pasados, el usuario selecciona una ventana temporal "
                                    "historica ya observada y plantea valores alternativos para una o varias "
                                    "variables exogenas. De este modo puede simular que habria ocurrido si las "
                                    "condiciones del pasado hubieran sido diferentes. Esta funcionalidad resulta "
                                    "especialmente util para analisis contrafactuales, revision de decisiones "
                                    "anteriores, estimacion de impacto y evaluacion de hipotesis."
                                ),
                                ui.tags.ul(
                                    ui.tags.li(
                                        "Validacion de entradas: antes de ejecutar, el sistema comprueba que "
                                        "existan objetivo, predictoras activas y valores completos en todas "
                                        "las celdas necesarias para el escenario."
                                    ),
                                    ui.tags.li(
                                        "Construccion del escenario: en modo futuro se organiza la informacion "
                                        "por variable, fecha y valor proyectado; en modo pasado se define una "
                                        "ventana historica y se aplican overrides sobre los periodos seleccionados."
                                    ),
                                    ui.tags.li(
                                        "Lectura del impacto: los resultados permiten comparar la trayectoria "
                                        "estimada del escenario frente a una referencia base, identificando "
                                        "diferencias tanto dentro del tramo modificado como en su posible "
                                        "efecto posterior."
                                    ),
                                ),
                                ui.tags.p(
                                    "Este modulo es clave para la planificacion estrategica, la valoracion de "
                                    "politicas publicas, el analisis de capacidad y la exploracion de supuestos "
                                    "de negocio dentro del ambito turistico."
                                ),
                                value="modulo_escenarios",
                            ),
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "4. Modelos disponibles, parametros principales y criterios de uso"
                                ),
                                ui.tags.p(
                                    "La plataforma permite trabajar con dos familias principales de modelos: "
                                    "SARIMAX y XGBoost. Ambos enfoques comparten una salida estandarizada para "
                                    "facilitar la comparacion de resultados, pero responden a logicas de modelado "
                                    "distintas y pueden ser mas adecuados segun el tipo de serie, la relacion "
                                    "entre variables y el objetivo del analisis."
                                ),
                                ui.tags.ul(
                                    ui.tags.li(
                                        ui.tags.strong("SARIMAX"),
                                        ": es un modelo de series temporales que combina componentes "
                                        "autorregresivos, de medias moviles, diferenciacion y, cuando procede, "
                                        "estructura estacional. Resulta especialmente util cuando la dinamica "
                                        "temporal de la serie es interpretable y existe interes en capturar "
                                        "patrones regulares como tendencia o estacionalidad. En configuraciones "
                                        "automaticas, el sistema puede explorar parametros optimos y, en algunos "
                                        "casos de series diarias, incorporar terminos Fourier para representar "
                                        "estacionalidad anual.",
                                    ),
                                    ui.tags.li(
                                        ui.tags.strong("XGBoost"),
                                        ": es un modelo de aprendizaje automatico basado en arboles de decision "
                                        "potenciados, adecuado para capturar relaciones no lineales, interacciones "
                                        "complejas entre variables y efectos menos evidentes en la estructura de "
                                        "los datos. Puede trabajar con variables exogenas y con lags de la serie "
                                        "objetivo, utilizando opciones como use_target_lags, max_lag y "
                                        "recursive_forecast para incorporar memoria temporal al proceso predictivo.",
                                    ),
                                ),
                                ui.tags.p(
                                    "En ambos casos, la ejecucion mantiene una logica comun de entrenamiento y "
                                    "evaluacion. La division entre entrenamiento y prueba se realiza con un "
                                    "train_ratio de 0.70, el horizonte de prediccion queda definido de forma "
                                    "explicita y la salida se devuelve en un formato estructurado que permite "
                                    "trazabilidad, auditoria y comparacion entre modelos."
                                ),
                                ui.tags.p(
                                    "Como criterio general, SARIMAX suele ser apropiado cuando la serie presenta "
                                    "una estructura temporal clara y se busca interpretabilidad estadistica, "
                                    "mientras que XGBoost puede aportar ventajas cuando existen patrones no "
                                    "lineales, mayor complejidad en las relaciones entre variables o necesidad "
                                    "de capturar señales combinadas de distinta naturaleza."
                                ),
                                value="modelos_parametros",
                            ),
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "5. Resultados, metricas y lectura de salidas"
                                ),
                                ui.tags.p(
                                    "Las salidas de la plataforma estan disenadas para facilitar tanto el analisis "
                                    "tecnico como la toma de decisiones operativas. Por ello, los resultados se "
                                    "presentan de forma complementaria en varios formatos: visual, tabular y "
                                    "resumido mediante indicadores cuantitativos de rendimiento."
                                ),
                                ui.tags.ul(
                                    ui.tags.li(
                                        "Graficos interactivos: permiten explorar la evolucion temporal de la "
                                        "serie observada, la prediccion generada y, en su caso, las diferencias "
                                        "entre el escenario base y el escenario simulado. El detalle por punto "
                                        "facilita la inspeccion fina por fecha y valor."
                                    ),
                                    ui.tags.li(
                                        "Tablas de resultados: muestran los valores predichos o simulados con "
                                        "un nivel de detalle suficiente para revision, auditoria y posible "
                                        "exportacion posterior a otros flujos de trabajo."
                                    ),
                                    ui.tags.li(
                                        "Metricas de error: se incluyen indicadores como MAPE, RMSE y MAE, que "
                                        "permiten evaluar el comportamiento del modelo desde perspectivas "
                                        "complementarias. El MAPE ofrece una lectura relativa del error, el RMSE "
                                        "penaliza con mayor intensidad los errores grandes y el MAE resume el "
                                        "desvio absoluto medio de forma mas interpretable."
                                    ),
                                ),
                                ui.tags.p(
                                    "La lectura de resultados no debe limitarse a identificar el valor de una "
                                    "sola metrica. Es recomendable observar conjuntamente la estabilidad del "
                                    "grafico, la coherencia temporal de los cambios, la sensibilidad del modelo "
                                    "ante distintas exogenas y la consistencia general con el conocimiento del "
                                    "fenomeno turistico que se esta analizando."
                                ),
                                value="salidas_metricas",
                            ),
                            ui.accordion_panel(
                                ui.tags.strong(
                                    "6. Buenas practicas de uso, interpretacion y trabajo profesional"
                                ),
                                ui.tags.p(
                                    "Para obtener resultados utiles y metodologicamente solidos, conviene utilizar "
                                    "la plataforma siguiendo una serie de criterios de trabajo analitico. La "
                                    "herramienta ayuda a estructurar el proceso, pero la calidad final del analisis "
                                    "depende tambien de la coherencia con la que se seleccionen variables, filtros "
                                    "y supuestos."
                                ),
                                ui.tags.ol(
                                    ui.tags.li(
                                        "Verifica siempre la compatibilidad entre la variable objetivo y las "
                                        "predictoras seleccionadas, especialmente en terminos de temporalidad, "
                                        "cobertura historica y sentido de negocio."
                                    ),
                                    ui.tags.li(
                                        "Aplica filtros con criterio analitico y evita construir muestras "
                                        "demasiado restringidas o poco comparables, ya que esto puede introducir "
                                        "sesgos o debilitar la robustez del modelo."
                                    ),
                                    ui.tags.li(
                                        "Compara el desempeno de SARIMAX y XGBoost siempre que sea posible. "
                                        "Elegir un modelo solo por una metrica aislada puede llevar a conclusiones "
                                        "parciales si no se considera tambien la interpretabilidad y la estabilidad."
                                    ),
                                    ui.tags.li(
                                        "Cuando trabajes con escenarios, documenta de forma explicita los valores "
                                        "asignados a las exogenas y el razonamiento que justifica cada supuesto. "
                                        "Esto mejora la trazabilidad y permite reproducir el analisis con claridad."
                                    ),
                                    ui.tags.li(
                                        "Interpreta los resultados dentro de su contexto sectorial. En turismo, "
                                        "las variaciones observadas pueden estar asociadas a factores estacionales, "
                                        "institucionales, economicos o territoriales que deben considerarse antes "
                                        "de convertir una salida estadistica en una decision operativa."
                                    ),
                                    ui.tags.li(
                                        "Utiliza la plataforma como herramienta de apoyo a la decision, no como "
                                        "sustituto del juicio experto. La combinacion entre evidencia cuantitativa "
                                        "y conocimiento del dominio es la que produce conclusiones mas utiles."
                                    ),
                                ),
                                value="buenas_practicas",
                            ),
                            id="help_docs_acc",
                            open=False,
                            multiple=True,
                            class_="help-docs-accordion",
                        ),
                        class_="help-docs-sections-box",
                    ),
                    class_="help-docs-card",
                ),
                class_="help-docs-page-frame",
            ),
            class_="help-docs-container",
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


# Directorio de la aplicacion
app_dir = Path(__file__).parent

# Crear la aplicacion Shiny
app = App(app_ui, server, static_assets=app_dir / "front" / "www")

# Punto de entrada para ejecucion
if __name__ == "__main__":
    from shiny import run_app

    run_app(app, port=8001)
