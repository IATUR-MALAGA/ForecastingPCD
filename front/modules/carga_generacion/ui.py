from shiny import ui, module
import json

# Cargar los indicadores disponibles desde el JSON
with open("equivalent_dictionary.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    indicadores_disponibles = data["Variables_Objetivo"]["Disponibles"]

@module.ui
def carga_generacion_ui():
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.link(rel="stylesheet", href="styles.css"),
            ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
            # JavaScript para manejar el input de archivos y drag & drop
            ui.tags.script("""
                function triggerFileInput() {
                    setTimeout(function() {
                        const input = document.querySelector('#csv_file_wrapper input[type=file]');
                        console.log('Input found:', input);
                        if (input) {
                            input.click();
                        } else {
                            console.error('File input not found in #csv_file_wrapper');
                        }
                    }, 100);
                }
                
                // Drag and drop functionality
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('Setting up drag and drop...');
                    
                    // Function to setup drag and drop on upload zone
                    function setupDragDrop() {
                        const uploadZone = document.querySelector('.upload-zone');
                        const fileInput = document.querySelector('#csv_file_wrapper input[type=file]');
                        
                        if (!uploadZone || !fileInput) {
                            console.log('Upload zone or file input not found, retrying...');
                            setTimeout(setupDragDrop, 500);
                            return;
                        }
                        
                        console.log('Upload zone found, setting up handlers');
                        
                        // Prevent default drag behaviors
                        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                            uploadZone.addEventListener(eventName, preventDefaults, false);
                            document.body.addEventListener(eventName, preventDefaults, false);
                        });
                        
                        function preventDefaults(e) {
                            e.preventDefault();
                            e.stopPropagation();
                        }
                        
                        // Highlight drop area when item is dragged over it
                        ['dragenter', 'dragover'].forEach(eventName => {
                            uploadZone.addEventListener(eventName, highlight, false);
                        });
                        
                        ['dragleave', 'drop'].forEach(eventName => {
                            uploadZone.addEventListener(eventName, unhighlight, false);
                        });
                        
                        function highlight(e) {
                            uploadZone.classList.add('drag-over');
                        }
                        
                        function unhighlight(e) {
                            uploadZone.classList.remove('drag-over');
                        }
                        
                        // Handle dropped files
                        uploadZone.addEventListener('drop', handleDrop, false);
                        
                        function handleDrop(e) {
                            const dt = e.dataTransfer;
                            const files = dt.files;
                            
                            console.log('Files dropped:', files.length);
                            
                            if (files.length > 0) {
                                // Check if file is CSV
                                const file = files[0];
                                if (file.name.endsWith('.csv')) {
                                    console.log('CSV file detected:', file.name);
                                    // Transfer the file to the Shiny input
                                    const dataTransfer = new DataTransfer();
                                    dataTransfer.items.add(file);
                                    fileInput.files = dataTransfer.files;
                                    
                                    // Trigger change event
                                    const event = new Event('change', { bubbles: true });
                                    fileInput.dispatchEvent(event);
                                } else {
                                    alert('Por favor, selecciona un archivo CSV');
                                }
                            }
                        }
                    }
                    
                    // Initial setup
                    setupDragDrop();
                    
                    // Re-setup when Shiny updates the DOM
                    if (window.Shiny) {
                        Shiny.addCustomMessageHandler('setupDragDrop', setupDragDrop);
                    }
                });
            """),
        ),

        # Hidden file input (global, persists across layout changes)
        # Using position: absolute with negative coordinates instead of display:none
        ui.tags.div(
            ui.input_file("csv_file", "", multiple=False, accept=[".csv"]), 
            style="position: absolute; left: -9999px; width: 1px; height: 1px;", 
            id="csv_file_wrapper"
        ),

        # Dashboard container
        ui.tags.div(
            ui.tags.h2("Cargar y visualizar CSV", class_="dashboard-title"),
            ui.tags.p("Selecciona un archivo CSV para cargar y previsualizar sus datos", class_="dashboard-subtitle"),
            class_="dashboard-header"
        ),

        # Dynamic layout (centered when no file, two-column when file loaded)
        ui.output_ui("main_layout"),
    )