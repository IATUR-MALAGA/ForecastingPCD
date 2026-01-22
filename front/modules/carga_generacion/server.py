from shiny import reactive, render, ui, module
import pandas as pd
import json

from back import data_loader, data_saver, generate_monthly_weights, data_generation
from front.utils.formatters import format_variable_name

# Ruta al archivo JSON
equivalent_dict_path = "equivalent_dictionary.json"

# Suppress debug logs by setting the logging level to WARNING
import logging
logging.basicConfig(level=logging.WARNING)

@module.server
def carga_generacion_server(input, output, session):
    

    # Reactive calculation that returns a pandas DataFrame or None
    @reactive.Calc
    def uploaded_df():
        f = input.csv_file()
        if not f:
            return None
        try:
            df = data_loader.load_csv_from_shiny_fileinfo(f)
            return df
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    # Reactive value to track if CSV has been saved
    csv_saved = reactive.Value(False)
    
    # Reactive values for monthly conversion
    monthly_df = reactive.Value(None)  # Stores generated monthly data
    monthly_generated = reactive.Value(False)  # Track if monthly data has been generated

    @output
    @render.text
    def csv_info():
        # Show info for monthly data if generated, otherwise show original
        if monthly_generated.get() and monthly_df.get() is not None:
            df = monthly_df.get()
            data_type_label = "📊 Datos Mensuales Generados"
        else:
            df = uploaded_df()
            data_type_label = "📊 Archivo Cargado"
            
        if df is None:
            return ""  # Don't show anything when no file is loaded
        # If the loader returned an error DataFrame, show the error message
        if isinstance(df, pd.DataFrame) and "error" in df.columns:
            return f"❌ Error al procesar el archivo:\n\n{str(df['error'].iloc[0])}"
        if df.empty:
            return "⚠️  El archivo se cargó correctamente pero no contiene datos."
        
        cols = ", ".join(map(str, df.columns[:10]))
        more_cols = f" (+{len(df.columns) - 10} más)" if len(df.columns) > 10 else ""
        
        # Add note if showing sample for large datasets
        sample_note = ""
        if len(df) > 100000:
            sample_note = f"\n⚠️  Mostrando muestra de 50,000 filas (dataset completo disponible para procesamiento)"
        
        status_icon = "✓" if not monthly_generated.get() else "🎉"
        status_text = "cargado exitosamente" if not monthly_generated.get() else "generados exitosamente"
        
        return f"{status_icon} {data_type_label} {status_text}\n\n📊 Filas: {df.shape[0]:,}\n📋 Columnas: {df.shape[1]}\n🏷️  Primeras columnas: {cols}{more_cols}{sample_note}"

    @output
    @render.data_frame
    def csv_table():
        # Show monthly data if generated, otherwise show original
        if monthly_generated.get() and monthly_df.get() is not None:
            df = monthly_df.get()
        else:
            df = uploaded_df()
            
        if df is None or df.empty:
            return None
        if isinstance(df, pd.DataFrame) and "error" in df.columns:
            return None
        
        # If monthly data was generated, transform it to the simplified view
        display_df = df
        if monthly_generated.get() and isinstance(df, pd.DataFrame):
            # Get the target column name from DataFrame attrs if available
            target_col_name = None
            try:
                target_col_name = df.attrs.get('synthetic_target')
            except Exception:
                pass
            
            # Case 1: Data from annual->monthly conversion (has year, month columns)
            if {"year", "month"}.issubset(set(df.columns)):
                # Map month names to month numbers
                month_map = {
                    "January": "01", "February": "02", "March": "03", "April": "04",
                    "May": "05", "June": "06", "July": "07", "August": "08",
                    "September": "09", "October": "10", "November": "11", "December": "12"
                }

                # Build simplified DataFrame
                tmp = df.copy()
                tmp["year"] = tmp["year"].astype(int)
                tmp["MES"] = tmp.apply(lambda r: f"{int(r['year'])}-{month_map.get(r['month'], '00')}", axis=1)
                
                # Find the value column (the target column, not temporal columns)
                value_col = None
                if target_col_name and target_col_name in df.columns:
                    value_col = target_col_name
                else:
                    # Find first non-temporal numeric column
                    for col in df.columns:
                        if col not in ['year', 'month', 'Mes', 'mes', 'weight', 'ponderacion'] and pd.api.types.is_numeric_dtype(df[col]):
                            value_col = col
                            break
                
                if value_col:
                    tmp[value_col.upper()] = tmp[value_col]
                    cols = ["MES", value_col.upper()]
                    
                    # Include ponderacion if available
                    if "weight" in tmp.columns:
                        tmp["PONDERACION"] = tmp["weight"]
                        cols.append("PONDERACION")
                    
                    display_df = tmp[cols]
                else:
                    display_df = df
            
            # Case 2: Synthetic data generated (has Mes column and target column directly)
            elif "Mes" in df.columns:
                if target_col_name and target_col_name in df.columns:
                    # Show Mes and the target column
                    tmp = df.copy()
                    tmp["MES"] = tmp["Mes"]
                    tmp[target_col_name.upper()] = tmp[target_col_name]
                    
                    # Include ponderacion if available
                    cols = ["MES", target_col_name.upper()]
                    if "weight" in tmp.columns:
                        tmp["PONDERACION"] = tmp["weight"]
                        cols.append("PONDERACION")
                    elif "ponderacion" in tmp.columns:
                        tmp["PONDERACION"] = tmp["ponderacion"]
                        cols.append("PONDERACION")
                    
                    display_df = tmp[cols]
                else:
                    # Fallback: show as-is
                    display_df = df
            else:
                # If structure is different, fall back to showing the DataFrame as-is
                display_df = df

        # For very large datasets (>100k rows), show only a sample for performance
        # The full dataset is still available in uploaded_df() for processing
        if len(df) > 100000:
            print(f"Dataset grande ({len(df):,} filas). Mostrando muestra de 50,000 filas.")
            # Show first 25k and last 25k rows to give a representative sample
            display_df = pd.concat([df.head(25000), df.tail(25000)], ignore_index=True)
        
        # Use render.DataGrid with virtual scrolling and pagination
        # This loads data incrementally as user scrolls
        return render.DataGrid(
            display_df, 
            height="400px",  # Reduced height for better space usage
            width="100%",
            selection_mode="none",  # Disable selection for better performance
        )

    # Dynamic UI for compact CSV info banner (inside preview section)
    @output
    @render.ui
    def csv_info_banner():
        # Show info for monthly data if generated, otherwise show original
        if monthly_generated.get() and monthly_df.get() is not None:
            df = monthly_df.get()
        else:
            df = uploaded_df()
        
        if df is None or df.empty:
            return None
        
        # Get banner info from data_saver
        banner_type, icon, message = data_saver.get_csv_info_banner(df)
        
        if not banner_type:
            return None
        
        # Render banner based on type
        icon_style = None
        if banner_type == "success":
            icon_style = "color: #10b981;"
        elif banner_type == "warning":
            icon_style = "color: #ffc107;"
        
        return ui.tags.div(
            ui.tags.i(class_=icon, style=icon_style),
            f" {message}",
            class_=f"info-banner info-banner-{banner_type}"
        )

    # Dynamic UI for preview section (only shown when CSV is loaded)
    @output
    @render.ui
    def preview_section():
        df = uploaded_df()
        
        # Don't show anything if no CSV is loaded
        if df is None:
            return None
        
        # Show preview card when CSV is loaded
        return ui.tags.div(
            ui.tags.h4(ui.tags.i(class_="fas fa-table"), " Vista previa de datos", class_="card-title"),
            
            # Data table preview with virtual scrolling
            ui.tags.div(
                ui.output_data_frame("csv_table"),
                class_="table-container"
            ),
            
            class_="preview-card"
        )

    @output
    @render.text
    def csv_filename():
        # Show the currently selected filename (updates when selecting file)
        f = input.csv_file()
        if not f:
            return "Ningún archivo seleccionado"
        # f is typically a list of dicts with key 'name'
        fileinfo = f[0] if isinstance(f, list) else f
        name = fileinfo.get("name") or fileinfo.get("filename") or fileinfo.get("path")
        if not name:
            return "Archivo seleccionado (nombre no disponible)"
        return name

    # Dynamic main layout (centered when no file, dashboard when file loaded)
    @output
    @render.ui
    def main_layout():
        df = uploaded_df()
        
        # No file loaded: centered elegant layout
        if df is None:
            return ui.tags.div(
                ui.tags.div(
                    # Welcome header
                    ui.tags.div(
                        ui.tags.h2(
                            ui.tags.i(class_="fas fa-chart-line", style="color: var(--primary-color);"),
                            " Análisis de Datos CSV",
                            class_="welcome-title"
                        ),
                        ui.tags.p(
                            "Carga, visualiza y procesa tus archivos CSV de forma intuitiva",
                            class_="welcome-subtitle"
                        ),
                        class_="welcome-header"
                    ),
                    
                    # Upload section
                    ui.tags.div(
                        ui.tags.button(
                            ui.tags.i(class_="fas fa-cloud-upload-alt", style="font-size: 2rem; margin-bottom: 1rem;"),
                            ui.tags.div("Selecciona tu archivo CSV", style="font-size: 1.1rem; font-weight: 600;"),
                            ui.tags.div("o arrástralo aquí", style="font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem;"),
                            id="csv_select_btn",
                            class_="upload-zone",
                            onclick="triggerFileInput();"
                        ),
                        
                        # File info display
                        ui.tags.div(
                            ui.output_text("csv_filename"),
                            class_="file-name-display"
                        ),
                        
                        class_="upload-container"
                    ),
                    
                    class_="welcome-card"
                ),
                class_="centered-container"
            )
        
        # File loaded: Professional dashboard layout
        return ui.tags.div(
            # Top bar: File name + Quick info + action button
            ui.tags.div(
                ui.row(
                    ui.column(4,
                        # File name display
                        ui.tags.div(
                            ui.tags.i(class_="fas fa-file-csv", style="color: var(--primary); font-size: 1.25rem;"),
                            ui.tags.span(
                                ui.output_text("csv_filename"),
                                style="margin-left: 0.75rem; font-weight: 600; color: var(--text-primary); font-size: 1rem;"
                            ),
                            style="display: flex; align-items: center;"
                        ),
                    ),
                    ui.column(4,
                        ui.output_ui("csv_info_banner"),
                    ),
                    ui.column(4,
                        ui.tags.div(
                            ui.tags.button(
                                ui.tags.i(class_="fas fa-sync-alt"),
                                " Cambiar archivo",
                                id="csv_select_btn2",
                                class_="btn-change-file",
                                onclick="triggerFileInput();"
                            ),
                            style="text-align: right;"
                        ),
                    ),
                ),
                class_="top-bar"
            ),
            
            # Main content: Data preview (full width)
            ui.tags.div(
                ui.output_ui("preview_section"),
                class_="main-section"
            ),
            
            # Dynamic layout based on generation status
            ui.output_ui("dynamic_actions_layout"),
        )

    # Dynamic actions layout - shows monthly conversion or save section
    @output
    @render.ui
    def dynamic_actions_layout():
        df = uploaded_df()
        
        if df is None or df.empty:
            return None
        
        # Check if data is annual
        is_annual, year_col = data_loader.is_annual_dataset(df)
        
        # Detect short monthly series (has 'mes' column and <12 unique months)
        has_mes_col = any(c.lower() == 'mes' or c.lower() == 'mes' or c.lower() == 'mes' for c in df.columns)
        short_monthly = False
        if not is_annual and has_mes_col:
            try:
                # normalize column name for mes
                mes_col = next((c for c in df.columns if c.lower() == 'mes' or c.lower() == 'mes'), None)
                if mes_col is not None:
                    uniq_months = pd.to_datetime(df[mes_col], errors='coerce').dropna().nunique()
                    if uniq_months < 12:
                        short_monthly = True
            except Exception:
                short_monthly = False

        if is_annual and not monthly_generated.get():
            # Show monthly conversion section
            return ui.tags.div(
                ui.output_ui("monthly_conversion_section"),
                class_="actions-section"
            )
        elif short_monthly and not monthly_generated.get():
            # Show short monthly generation section (interpolate or TVAE-GAN)
            return ui.tags.div(
                ui.output_ui("short_monthly_generation_section"),
                class_="actions-section"
            )
        else:
            # Show save section
            return ui.tags.div(
                ui.output_ui("save_data_section"),
                class_="actions-section"
            )
    
    # ==================== MONTHLY CONVERSION SECTION ====================
    @output
    @render.ui
    def monthly_conversion_section():
        df = uploaded_df()
        
        if df is None or df.empty:
            return None
        
        # Check if data is annual
        is_annual, year_col = data_loader.is_annual_dataset(df)
        
        if not is_annual:
            return None  # Don't show this section for non-annual data
        
        # Get numeric columns (excluding year column)
        numeric_cols = data_loader.get_numeric_columns(df, exclude_columns=[year_col])

        # If pandas didn't detect numeric columns (e.g. values like "1.234,56"),
        # try a permissive coercion to find numeric-like columns so the UI still
        # offers the generation option even for small files.
        if not numeric_cols:
            permissive = []
            for col in df.columns:
                if col == year_col:
                    continue
                try:
                    # Replace common thousand/decimal separators and try to coerce
                    s = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                    coerced = pd.to_numeric(s, errors='coerce')
                    if coerced.notna().any():
                        permissive.append(col)
                except Exception:
                    continue

            if permissive:
                numeric_cols = {c: c for c in permissive}
            else:
                return ui.tags.div(
                    ui.tags.div(
                        ui.tags.i(class_="fas fa-exclamation-triangle"),
                        " No se encontraron columnas numéricas para convertir.",
                        class_="alert-warning-compact"
                    )
                )
        
        # Create choices dict for column selection
        column_choices = {col: col for col in numeric_cols}
        
        return ui.tags.div(
            ui.tags.h4(
                ui.tags.i(class_="fas fa-calendar-alt"), 
                " Conversión Anual → Mensual", 
                class_="card-title"
            ),
            
            ui.tags.p(
                "Se han detectado datos anuales. Selecciona cómo quieres convertirlos a datos mensuales:",
                class_="section-description-compact"
            ),
            
            # Column selection
            ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-table"),
                    " Columna a convertir:",
                    class_="input-label"
                ),
                ui.input_select(
                    "column_to_convert",
                    "",
                    choices=column_choices,
                    width="100%"
                ),
                class_="input-group"
            ),
            
            # Method selection
            ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-cogs"),
                    " Método de ponderación:",
                    class_="input-label"
                ),
                ui.input_radio_buttons(
                    "conversion_method",
                    "",
                    choices={
                        "manual": "Ponderación Manual (introduce % por mes)",
                        "automatic": "Ponderación Automática (TVAE-GAN)"
                    },
                    selected="manual"
                ),
                class_="input-group"
            ),
            
            # Manual weights section (conditional)
            ui.output_ui("manual_weights_section"),
            
            # Automatic method info (conditional)
            ui.output_ui("automatic_method_info"),
            
            # Generate button
            ui.tags.div(
                ui.input_action_button(
                    "generate_monthly_btn",
                    ui.tags.span(
                        ui.tags.i(class_="fas fa-magic"),
                        " Generar Datos Mensuales"
                    ),
                    class_="btn-generate btn-large"
                ),
                style="margin-top: 1.5rem;"
            ),
            
            # Status message
            ui.output_ui("monthly_conversion_status"),
            
            class_="upload-card"
        )


    # ==================== SHORT MONTHLY GENERATION (for <12 months) ====================
    @output
    @render.ui
    def short_monthly_generation_section():
        df = uploaded_df()
        if df is None or df.empty:
            return None

        # Find mes column
        mes_col = next((c for c in df.columns if c.lower() == 'mes'), None)
        if mes_col is None:
            return None

        # Prepare numeric column choices (permissive)
        numeric_cols = data_loader.get_numeric_columns(df)
        if not numeric_cols:
            permissive = []
            for col in df.columns:
                if col == mes_col:
                    continue
                try:
                    s = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    coerced = pd.to_numeric(s, errors='coerce')
                    if coerced.notna().any():
                        permissive.append(col)
                except Exception:
                    continue
            if permissive:
                numeric_cols = {c: c for c in permissive}
            else:
                return ui.tags.div(
                    ui.tags.div(
                        ui.tags.i(class_="fas fa-exclamation-triangle"),
                        " No se encontraron columnas numéricas para generar.",
                        class_="alert-warning-compact"
                    )
                )

        column_choices = {col: col for col in (numeric_cols if isinstance(numeric_cols, list) else numeric_cols)}

        return ui.tags.div(
            ui.tags.h4(
                ui.tags.i(class_="fas fa-magic"),
                " Completar Serie Mensual (corta)",
                class_="card-title"
            ),

            ui.tags.p(
                "Se han detectado menos de 12 meses. Puedes completar los meses faltantes mediante interpolación o usando TVAE-GAN.",
                class_="section-description-compact"
            ),

            ui.tags.div(
                ui.tags.label(ui.tags.i(class_="fas fa-table"), " Columna objetivo:"),
                ui.input_select("short_target_column", "", choices=column_choices, width="100%"),
                class_="input-group"
            ),

            ui.tags.div(
                ui.tags.label(ui.tags.i(class_="fas fa-cogs"), " Método:"),
                ui.input_radio_buttons(
                    "short_method",
                    "",
                    choices={
                        "manual": "Ponderación Manual (introduce % para meses faltantes)",
                        "tvae": "Generar con TVAE-GAN (más realista)"
                    },
                    selected="manual"
                ),
                class_="input-group"
            ),

            # Manual weights section for missing months
            ui.output_ui("short_manual_weights_section"),

            ui.tags.div(
                ui.input_action_button(
                    "generate_short_btn",
                    ui.tags.span(ui.tags.i(class_="fas fa-play"), " Completar meses"),
                    class_="btn-generate btn-large"
                ),
                style="margin-top: 1rem;"
            ),

            ui.output_ui("monthly_conversion_status"),

            class_="upload-card"
        )


    @output
    @render.ui
    def short_manual_weights_section():
        """Render numeric inputs for the missing months when user selects manual method."""
        df = uploaded_df()
        if df is None or df.empty:
            return None

        if input.short_method() != 'manual':
            return None

        mes_col = next((c for c in df.columns if c.lower() == 'mes'), None)
        if mes_col is None:
            return None

        # parse mes column and group by year
        tmp = df.copy()
        try:
            tmp[mes_col] = pd.to_datetime(tmp[mes_col].astype(str), errors='coerce')
        except Exception:
            return None

        years = tmp[mes_col].dt.year.dropna().unique().tolist()
        if not years:
            return None

        ui_blocks = []
        month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]

        for year in sorted(years):
            year_rows = tmp[tmp[mes_col].dt.year == year]
            present_months = set(year_rows[mes_col].dt.month.dropna().astype(int).tolist())
            missing = [m for m in range(1,13) if m not in present_months]
            if not missing:
                continue

            # Default equal share among missing months
            default_pct = round(100 / len(missing), 2)

            ui_blocks.append(ui.tags.h5(f"Año {int(year)} - Meses faltantes: {', '.join(str(m) for m in missing)}"))

            for m in missing:
                input_id = f"short_weight_{year}_{m:02d}"
                ui_blocks.append(
                    ui.tags.div(
                        ui.tags.label(f"{month_names[m-1]}:"),
                        ui.input_numeric(input_id, "", value=default_pct, min=0, max=100, step=0.1, width="100%"),
                        class_="input-group-compact"
                    )
                )

        if not ui_blocks:
            return None

        return ui.tags.div(*ui_blocks, style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;")
    
    # Manual weights input section
    @output
    @render.ui
    def manual_weights_section():
        if input.conversion_method() != "manual":
            return None
        
        months = [
            ("January", "Enero"), ("February", "Febrero"), ("March", "Marzo"),
            ("April", "Abril"), ("May", "Mayo"), ("June", "Junio"),
            ("July", "Julio"), ("August", "Agosto"), ("September", "Septiembre"),
            ("October", "Octubre"), ("November", "Noviembre"), ("December", "Diciembre")
        ]
        
        # Default equal weights (8.33% each month)
        default_weight = round(100 / 12, 2)
        
        return ui.tags.div(
            ui.tags.h5(
                ui.tags.i(class_="fas fa-sliders-h"),
                " Introduce los porcentajes por mes:",
                style="margin-top: 1rem; margin-bottom: 0.75rem; color: var(--text-primary);"
            ),
            ui.tags.small(
                "Los valores deben sumar 100%. Por defecto, se usa distribución uniforme (8.33% cada mes).",
                class_="help-text",
                style="display: block; margin-bottom: 1rem;"
            ),
            
            # Create 3 columns of 4 months each
            ui.row(
                ui.column(4,
                    *[ui.tags.div(
                        ui.tags.label(
                            f"{spanish_name}:",
                            style="font-size: 0.9rem; font-weight: 500;"
                        ),
                        ui.input_numeric(
                            f"weight_{eng_name}",
                            "",
                            value=default_weight,
                            min=0,
                            max=100,
                            step=0.1,
                            width="100%"
                        ),
                        class_="input-group-compact"
                    ) for eng_name, spanish_name in months[0:4]]
                ),
                ui.column(4,
                    *[ui.tags.div(
                        ui.tags.label(
                            f"{spanish_name}:",
                            style="font-size: 0.9rem; font-weight: 500;"
                        ),
                        ui.input_numeric(
                            f"weight_{eng_name}",
                            "",
                            value=default_weight,
                            min=0,
                            max=100,
                            step=0.1,
                            width="100%"
                        ),
                        class_="input-group-compact"
                    ) for eng_name, spanish_name in months[4:8]]
                ),
                ui.column(4,
                    *[ui.tags.div(
                        ui.tags.label(
                            f"{spanish_name}:",
                            style="font-size: 0.9rem; font-weight: 500;"
                        ),
                        ui.input_numeric(
                            f"weight_{eng_name}",
                            "",
                            value=default_weight,
                            min=0,
                            max=100,
                            step=0.1,
                            width="100%"
                        ),
                        class_="input-group-compact"
                    ) for eng_name, spanish_name in months[8:12]]
                )
            ),
            
            # Sum validation display
            ui.output_ui("weights_sum_display"),
            
            style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;"
        )
    
    # Display sum of weights
    @output
    @render.ui
    def weights_sum_display():
        if input.conversion_method() != "manual":
            return None
        
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        
        try:
            total = sum([input[f"weight_{month}"]() for month in months])
            
            if abs(total - 100) < 0.1:
                icon = "fa-check-circle"
                color = "#10b981"
                message = f"✓ Suma correcta: {total:.2f}%"
            else:
                icon = "fa-exclamation-triangle"
                color = "#f59e0b"
                message = f"⚠ Suma actual: {total:.2f}% (debe ser 100%)"
            
            return ui.tags.div(
                ui.tags.i(class_=f"fas {icon}", style=f"color: {color};"),
                f" {message}",
                style=f"margin-top: 1rem; font-weight: 600; color: {color};"
            )
        except:
            return None
    
    # Automatic method info section
    @output
    @render.ui
    def automatic_method_info():
        if input.conversion_method() != "automatic":
            return None
        
        return ui.tags.div(
            ui.tags.div(
                ui.tags.i(class_="fas fa-robot", style="font-size: 1.5rem; color: var(--primary-color);"),
                ui.tags.h5(
                    "Generación Automática con TVAE-GAN",
                    style="margin-top: 0.5rem; margin-bottom: 0.5rem;"
                ),
                ui.tags.p(
                    "El sistema utilizará inteligencia artificial (TVAE-GAN) para generar automáticamente "
                    "la distribución mensual basándose en patrones temporales y el tipo de indicador.",
                    style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;"
                ),
                ui.tags.ul(
                    ui.tags.li("Detecta patrones estacionales automáticamente"),
                    ui.tags.li("Considera eventos especiales (COVID, festividades, etc.)"),
                    ui.tags.li("Genera distribuciones realistas basadas en datos históricos"),
                    style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 1rem;"
                ),
                style="background: #f0f9ff; padding: 1.25rem; border-radius: 8px; border-left: 4px solid var(--primary-color);"
            ),
            style="margin-top: 1rem;"
        )
    
    # Monthly conversion status messages
    monthly_conversion_message = reactive.Value("")
    monthly_conversion_success = reactive.Value(True)
    
    @output
    @render.ui
    def monthly_conversion_status():
        msg = monthly_conversion_message.get()
        if not msg:
            return None
        
        icon = "fa-check-circle" if monthly_conversion_success.get() else "fa-exclamation-triangle"
        alert_class = "alert-success-compact" if monthly_conversion_success.get() else "alert-warning-compact"
        
        return ui.tags.div(
            ui.tags.div(
                ui.tags.i(class_=f"fas {icon}"),
                f" {msg}",
                class_=alert_class
            ),
            class_="mt-3"
        )
    
    # Observer for generate monthly button
    @reactive.Effect
    @reactive.event(input.generate_monthly_btn)
    def _():
        """
        Handle monthly data generation (manual or automatic).
        """
        df = uploaded_df()
        if df is None or df.empty:
            monthly_conversion_message.set("No hay datos para convertir.")
            monthly_conversion_success.set(False)
            return
        
        # Get parameters
        method = input.conversion_method()
        column = input.column_to_convert()
        is_annual, year_col = data_loader.is_annual_dataset(df)
        
        if not is_annual:
            monthly_conversion_message.set("Los datos no son anuales.")
            monthly_conversion_success.set(False)
            return
        
        try:
            if method == "manual":
                # Get manual weights from inputs
                months = ["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"]
                
                weights_percent = {month: input[f"weight_{month}"]() for month in months}
                
                # Validate sum
                total = sum(weights_percent.values())
                if abs(total - 100) > 1:  # Allow 1% tolerance
                    monthly_conversion_message.set(f"Los porcentajes deben sumar 100% (actual: {total:.2f}%)")
                    monthly_conversion_success.set(False)
                    return
                
                # Convert percentages to weights (0-1 range)
                weights = {month: pct / 100 for month, pct in weights_percent.items()}
                
                # Generate monthly data
                monthly_data = generate_monthly_weights.apply_manual_monthly_weights(
                    df=df,
                    column=column,
                    manual_weights=weights,
                    year_column=year_col
                )
                
                monthly_conversion_message.set(f"✓ Datos mensuales generados correctamente ({len(monthly_data)} filas)")
                monthly_conversion_success.set(True)
                
            else:  # automatic
                # Use TVAE-GAN for automatic generation (also for short series)
                monthly_conversion_message.set("🤖 Generando datos con TVAE-GAN... (esto puede tardar varios minutos)")
                monthly_conversion_success.set(True)

                try:
                    # Infer sensible start/end dates. If the uploaded DF has few rows (<12),
                    # expand the range to give the generator context (one year before/after).
                    if year_col and year_col in df.columns:
                        try:
                            min_year = int(df[year_col].min())
                            max_year = int(df[year_col].max())
                            if len(df) < 12:
                                start_date = f"{max(1900, min_year-1)}-01-01"
                                end_date = f"{max_year+1}-12-31"
                            else:
                                start_date = f"{min_year}-01-01"
                                end_date = f"{max_year}-12-31"
                        except Exception:
                            start_date = "2000-01-01"
                            end_date = "2030-12-31"
                    else:
                        # Try temporal columns
                        temporal_cols = data_generation.get_temporal_columns(df)
                        if temporal_cols:
                            tc = temporal_cols[0]
                            try:
                                start = pd.to_datetime(df[tc], errors='coerce').min()
                                end = pd.to_datetime(df[tc], errors='coerce').max()
                                # If few rows, expand one year each side
                                if len(df) < 12 and pd.notna(start) and pd.notna(end):
                                    start_date = (start - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
                                    end_date = (end + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
                                else:
                                    start_date = start.strftime("%Y-%m-%d") if pd.notna(start) else "2000-01-01"
                                    end_date = end.strftime("%Y-%m-%d") if pd.notna(end) else "2030-12-31"
                            except Exception:
                                start_date = "2000-01-01"
                                end_date = "2030-12-31"
                        else:
                            start_date = "2000-01-01"
                            end_date = "2030-12-31"

                    # Call the backend TVAE-GAN pipeline. This is long-running and runs
                    # preprocessing/training scripts under back/tvae_gan_generator via subprocess.
                    monthly_data, meta = data_generation.generate_synthetic_data(
                        df=df,
                        target_column=column,
                        start_date=start_date,
                        end_date=end_date,
                        granularity="MS",
                    )

                    monthly_conversion_message.set(f"✓ Datos mensuales generados automáticamente ({len(monthly_data)} filas)")
                    monthly_conversion_success.set(True)
                except Exception as e:
                    monthly_conversion_message.set(f"Error generando con TVAE-GAN: {str(e)}")
                    monthly_conversion_success.set(False)
                    import traceback
                    traceback.print_exc()
            
            # Rename 'monthly_value' column to the actual target column name
            # so the DataFrame has the correct column name for saving and display
            if 'monthly_value' in monthly_data.columns and column:
                monthly_data = monthly_data.rename(columns={'monthly_value': column})
                print(f"✅ Renamed 'monthly_value' to '{column}'")
                # Mark the DataFrame with the target column name for downstream use
                try:
                    monthly_data.attrs['synthetic_target'] = column
                except Exception:
                    pass
            
            # Store generated monthly data
            monthly_df.set(monthly_data)
            monthly_generated.set(True)
            
        except Exception as e:
            monthly_conversion_message.set(f"Error al generar datos mensuales: {str(e)}")
            monthly_conversion_success.set(False)
            import traceback
            traceback.print_exc()

    # Observer for short monthly generation button
    @reactive.Effect
    @reactive.event(input.generate_short_btn)
    def _short_generate():
        df = uploaded_df()
        if df is None or df.empty:
            monthly_conversion_message.set("No hay datos para convertir.")
            monthly_conversion_success.set(False)
            return

        mes_col = next((c for c in df.columns if c.lower() == 'mes'), None)
        if mes_col is None:
            monthly_conversion_message.set("No se encontró la columna 'mes'.")
            monthly_conversion_success.set(False)
            return

        column = input.short_target_column()
        method = input.short_method()

        try:
            # Parse dates
            tmp = df.copy()
            tmp[mes_col] = pd.to_datetime(tmp[mes_col].astype(str), errors='coerce')
            tmp = tmp.dropna(subset=[mes_col])
            if tmp.empty:
                monthly_conversion_message.set("No se pudieron parsear las fechas en la columna 'mes'.")
                monthly_conversion_success.set(False)
                return

            # Group by year and process each year separately
            years = tmp[mes_col].dt.year.dropna().unique().tolist()

            all_parts = []

            for year in sorted(years):
                year_rows = tmp[tmp[mes_col].dt.year == year].copy()
                present_months = set(year_rows[mes_col].dt.month.dropna().astype(int).tolist())
                missing = [m for m in range(1,13) if m not in present_months]

                # If manual method: collect user percentages for missing months
                if method == 'manual':
                    # Build weights dict for this year
                    weights = {}
                    for m in missing:
                        input_id = f"short_weight_{year}_{m:02d}"
                        try:
                            pct = float(input[input_id]())
                        except Exception:
                            pct = 0.0
                        weights[m] = pct

                    total_pct = sum(weights.values())
                    # Allow totals <= 100; reject only if >100
                    if total_pct > 100.0 + 1e-6:
                        monthly_conversion_message.set(f"Los porcentajes para {year} no pueden sumar más de 100% (actual: {total_pct:.2f}%).")
                        monthly_conversion_success.set(False)
                        return

                    # Convert to fractions
                    fractions = {m: w/100.0 for m,w in weights.items()}
                    p_missing = sum(fractions.values())

                    # Sum known months values
                    try:
                        # ensure numeric coercion for target column
                        year_rows['__val__'] = pd.to_numeric(year_rows[column].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
                    except Exception:
                        year_rows['__val__'] = pd.to_numeric(year_rows[column], errors='coerce')

                    S_known = year_rows['__val__'].sum(skipna=True)

                    if p_missing >= 1.0:
                        monthly_conversion_message.set(f"No es posible calcular: la suma de porcentajes faltantes es 100% o mayor para {year}.")
                        monthly_conversion_success.set(False)
                        return

                    # Annual total T
                    T = S_known / (1.0 - p_missing)

                    # Build rows for known months (keep original) and missing months using fractions
                    # known
                    known_df = year_rows[[mes_col, column]].copy()
                    known_df = known_df.rename(columns={mes_col: 'Mes'})
                    # missing
                    missing_rows = []
                    for m, frac in fractions.items():
                        month_dt = pd.Timestamp(year=year, month=m, day=1)
                        val = round(T * frac, 3)
                        missing_rows.append({'Mes': month_dt.strftime('%Y-%m'), column: val})

                    combined = pd.concat([known_df, pd.DataFrame(missing_rows)], ignore_index=True)
                    all_parts.append(combined)

                else:
                    # TVAE-GAN generation for this year: request generation for full year
                    start_date = pd.Timestamp(year=year, month=1, day=1).strftime('%Y-%m-%d')
                    end_date = pd.Timestamp(year=year, month=12, day=31).strftime('%Y-%m-%d')
                    monthly_part, meta = data_generation.generate_synthetic_data(
                        df=year_rows.reset_index(drop=True),
                        target_column=column,
                        start_date=start_date,
                        end_date=end_date,
                        granularity='MS'
                    )
                    all_parts.append(monthly_part)

            # Combine all parts (for manual we created DataFrames per year)
            if method == 'manual':
                if not all_parts:
                    monthly_conversion_message.set("No se generaron partes para la serie.")
                    monthly_conversion_success.set(False)
                    return
                monthly_data = pd.concat(all_parts, ignore_index=True)
            else:
                # monthly_data already set when using TVAE per year (all_parts contains DataFrames)
                if isinstance(all_parts, list) and len(all_parts) == 1:
                    monthly_data = all_parts[0]
                else:
                    monthly_data = pd.concat(all_parts, ignore_index=True)

            # Ensure correct column naming
            if 'monthly_value' in monthly_data.columns and column:
                monthly_data = monthly_data.rename(columns={'monthly_value': column})
            try:
                monthly_data.attrs['synthetic_target'] = column
            except Exception:
                pass

            monthly_df.set(monthly_data)
            monthly_generated.set(True)
            monthly_conversion_message.set(f"✓ Serie completada ({len(monthly_data)} filas)")
            monthly_conversion_success.set(True)

        except Exception as e:
            monthly_conversion_message.set(f"Error completando la serie: {str(e)}")
            monthly_conversion_success.set(False)
            import traceback
            traceback.print_exc()
    
    # ==================== END MONTHLY CONVERSION SECTION ====================
    
    # Dynamic UI for save data section
    @output
    @render.ui
    def save_data_section():
        df = uploaded_df()
        
        # Only show if CSV is loaded and valid
        if df is None or df.empty:
            return None
        if isinstance(df, pd.DataFrame) and "error" in df.columns:
            return None
        
        # Create the save interface
        return ui.tags.div(
            ui.tags.h4(
                ui.tags.i(class_="fas fa-save"), 
                " Guardar Datos", 
                class_="card-title"
            ),
            
            ui.tags.p(
                "Guarda el CSV cargado en el indicador seleccionado.",
                class_="section-description-compact"
            ),
            
            # Save button
            ui.input_action_button(
                "save_btn",
                ui.tags.span(
                    ui.tags.i(class_="fas fa-save"),
                    " Guardar CSV"
                ),
                class_="btn-save btn-large"
            ),
            
            class_="upload-card"
        )
    
    # Observer for save button - SAVE CSV DATA
    # Dynamic UI for save data section
    @output
    @render.ui
    def save_data_section():
        # Use monthly data if available, otherwise use original uploaded data
        if monthly_generated.get() and monthly_df.get() is not None:
            df = monthly_df.get()
            data_type_label = "datos mensuales generados"
        else:
            df = uploaded_df()
            data_type_label = "CSV cargado"
        
        # Only show if CSV is loaded and valid
        if df is None or df.empty:
            return None
        if isinstance(df, pd.DataFrame) and "error" in df.columns:
            return None
        
        # Get existing folders
        folders = data_saver.list_existing_folders()
        
        # Check if CSV has already been saved
        already_saved = csv_saved.get()
        
        return ui.tags.div(
            ui.tags.h4(
                ui.tags.i(class_="fas fa-save"), 
                " Guardar CSV", 
                class_="card-title"
            ),
            
            ui.tags.p(
                f"Guarda los {data_type_label} en data/input." if not already_saved else "CSV ya guardado.",
                class_="section-description-compact"
            ),
            
            # Show success message if already saved
            ui.tags.div(
                ui.tags.div(
                    ui.tags.i(class_="fas fa-check-circle"),
                    " El CSV ya ha sido guardado. Carga un nuevo archivo para guardar otro.",
                    class_="alert-success-compact"
                ),
                class_="mb-3"
            ) if already_saved else None,
            
            # Type selection: Indicator or Exogenous Variable (disabled if already saved)
            ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-tags"),
                    " Tipo de dato ",
                    class_="input-label",
                    style="margin-right: 1rem;"
                ),
                ui.input_radio_buttons(
                    "data_type",
                    "",
                    choices={"indicador": "Fuente principal", "exogena": "Variable Exógena"},
                    selected="indicador"
                ),
                class_="input-group",
                style="opacity: 0.5; pointer-events: none;" if already_saved else None
            ),
            
            # Variable name input (for both indicator and exogenous variable)
            ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-signature"),
                    " Nombre de la variable:",
                    class_="input-label"
                ),
                ui.input_text(
                    "variable_name",
                    "",
                    placeholder="Ej: llegadas_avion, turistas_malaga",
                    width="100%"
                ),
                ui.tags.small(
                    "Este nombre se usará para identificar la variable en el diccionario de equivalencias.",
                    class_="help-text"
                ),
                class_="input-group",
                style="opacity: 0.5; pointer-events: none;" if already_saved else None
            ),
            
            # Conditional sections based on data type
            # If Fuente principal (indicador) - show folder selection
            ui.panel_conditional(
                "input.data_type == 'indicador'",
                ui.tags.div(
                    # Folder selection
                    ui.tags.div(
                        ui.tags.label(
                            ui.tags.i(class_="fas fa-folder"),
                            " Selecciona carpeta:",
                            class_="input-label"
                        ),
                        ui.input_select(
                            "save_folder",
                            "",
                            choices=data_saver.get_folder_choices(),
                            width="100%"
                        ),
                        class_="input-group",
                        style="opacity: 0.5; pointer-events: none;" if already_saved else None
                    ),
                    # Option to create new folder
                    ui.tags.div(
                        ui.input_checkbox(
                            "create_new_folder",
                            "Crear nueva carpeta",
                            value=False
                        ),
                        class_="input-group",
                        style="opacity: 0.5; pointer-events: none;" if already_saved else None
                    ),
                    # New folder name input (conditional)
                    ui.output_ui("new_folder_input"),
                )
            ),
            
            # If Variable Exógena - show folder selector and indicator target
            ui.panel_conditional(
                "input.data_type == 'exogena'",
                ui.tags.div(
                    # Folder selection for exogenous variables (from data/input)
                    ui.tags.div(
                        ui.tags.label(
                            ui.tags.i(class_="fas fa-folder"),
                            " Selecciona carpeta:",
                            class_="input-label"
                        ),
                        ui.output_ui("exogenous_folder_selector"),
                        class_="input-group",
                        style="opacity: 0.5; pointer-events: none;" if already_saved else None
                    ),
                    # Option to create new folder
                    ui.tags.div(
                        ui.input_checkbox(
                            "create_new_folder_exogena",
                            "Crear nueva carpeta",
                            value=False
                        ),
                        class_="input-group",
                        style="opacity: 0.5; pointer-events: none;" if already_saved else None
                    ),
                    # New folder name input (conditional)
                    ui.output_ui("new_folder_input_exogena"),
                    # Step 1: Select indicator category/folder
                    ui.tags.div(
                        ui.tags.label(
                            ui.tags.i(class_="fas fa-layer-group"),
                            " Categoría del indicador:",
                            class_="input-label"
                        ),
                        ui.output_ui("indicator_category_selector"),
                        ui.tags.small(
                            "Selecciona la categoría del indicador objetivo.",
                            class_="help-text"
                        ),
                        class_="input-group",
                        style="opacity: 0.5; pointer-events: none;" if already_saved else None
                    ),
                    # Step 2: Select specific indicator from category
                    ui.output_ui("indicador_target_selector_wrapper"),
                )
            ),
            
            # Save button (disabled if already saved)
            ui.tags.div(
                ui.input_action_button(
                    "save_btn",
                    ui.tags.span(
                        ui.tags.i(class_="fas fa-download" if not already_saved else "fas fa-check"),
                        " Guardar CSV" if not already_saved else " Guardado"
                    ),
                    class_="btn-generate btn-large",
                ),
                style="opacity: 0.5; pointer-events: none;" if already_saved else None
            ),
            
            # Status message
            ui.output_ui("save_status"),
            
            class_="upload-card"
        )
    
    # Conditional input for new folder name (for indicadores)
    @output
    @render.ui
    def new_folder_input():
        if input.create_new_folder():
            return ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-folder-plus"),
                    " Nombre de la nueva carpeta:",
                    class_="input-label"
                ),
                ui.input_text(
                    "new_folder_name",
                    "",
                    placeholder="Ej: Turismo inteligente",
                    width="100%"
                ),
                class_="input-group"
            )
        return None
    
    # Dynamic exogenous folder selector
    @output
    @render.ui
    def exogenous_folder_selector():
        folders = data_saver.get_folder_choices("exogena")
        return ui.input_select(
            "save_folder_exogena",
            "",
            choices=folders,
            width="100%"
        )
    
    # Conditional input for new folder name (for exogenous variables)
    @output
    @render.ui
    def new_folder_input_exogena():
        if input.create_new_folder_exogena():
            return ui.tags.div(
                ui.tags.label(
                    ui.tags.i(class_="fas fa-folder-plus"),
                    " Nombre de la nueva carpeta:",
                    class_="input-label"
                ),
                ui.input_text(
                    "new_folder_name_exogena",
                    "",
                    placeholder="Ej: clima, turistas, etc.",
                    width="100%"
                ),
                class_="input-group"
            )
        return None
    
    # Indicator category selector (folders from indicadores/)
    @output
    @render.ui
    def indicator_category_selector():
        categories = data_saver.get_available_indicators()  # Gets indicator folders
        choices_dict = {cat: format_variable_name(cat) for cat in categories}
        
        return ui.input_select(
            "indicator_category",
            "",
            choices=choices_dict,
            width="100%"
        )
    
    # Dynamic indicator target selector wrapper (shows only when category is selected)
    @output
    @render.ui
    def indicador_target_selector_wrapper():
        # Get selected category
        try:
            selected_category = input.indicator_category()
        except:
            return None
        
        if not selected_category:
            return None
        
        # Get indicators from the selected category
        indicators = data_saver.get_indicators_by_folder(selected_category)
        
        if not indicators:
            return ui.tags.div(
                ui.tags.div(
                    ui.tags.i(class_="fas fa-info-circle"),
                    " No hay indicadores en esta categoría.",
                    class_="alert-info-compact"
                )
            )
        
        choices_dict = {ind: format_variable_name(ind) for ind in indicators}
        
        return ui.tags.div(
            ui.tags.label(
                ui.tags.i(class_="fas fa-bullseye"),
                " Indicador objetivo:",
                class_="input-label"
            ),
            ui.input_select(
                "indicador_target",
                "",
                choices=choices_dict,
                width="100%"
            ),
            ui.tags.small(
                f"Indicadores disponibles en {format_variable_name(selected_category)}.",
                class_="help-text"
            ),
            class_="input-group"
        )
    
    # Save status message
    save_message = reactive.Value("")
    save_success = reactive.Value(True)
    
    @output
    @render.ui
    def save_status():
        msg = save_message.get()
        if not msg:
            return None
        
        icon = "fa-check-circle" if save_success.get() else "fa-exclamation-triangle"
        alert_class = "alert-success-compact" if save_success.get() else "alert-warning-compact"
        
        return ui.tags.div(
            ui.tags.div(
                ui.tags.i(class_=f"fas {icon}"),
                f" {msg}",
                class_=alert_class
            ),
            class_="mt-3"
        )
    
    # Observer for save button
    @reactive.Effect
    @reactive.event(input.save_btn)
    def _():
        """
        Handle CSV saving to data/input folder.
        """
        # Get the data to save (monthly if generated, otherwise original)
        if monthly_generated.get() and monthly_df.get() is not None:
            df_to_save = monthly_df.get()
        else:
            df_to_save = uploaded_df()
            
        if df_to_save is None or df_to_save.empty:
            save_message.set("No hay datos para guardar.")
            save_success.set(False)
            return
        
        # Get data type from radio buttons
        data_type = input.data_type()
        
        # Get file name (nombre del CSV)
        file_name = input.variable_name()
        
        if not file_name or file_name.strip() == "":
            save_message.set("Debes especificar un nombre para el archivo")
            save_success.set(False)
            return
        
        # Get parameters based on data type
        if data_type == "exogena":
            # For exogenous variables
            create_new = input.create_new_folder_exogena()
            folder_name = input.save_folder_exogena() if not create_new else ""
            new_folder_name = input.new_folder_name_exogena() if create_new else ""
            indicador_target = input.indicador_target()
        else:
            # For indicators
            create_new = input.create_new_folder()
            folder_name = input.save_folder()
            new_folder_name = input.new_folder_name() if create_new else ""
            indicador_target = None
        
        # Use data_saver to process the complete save request
        success, msg = data_saver.process_save_request(
            df=df_to_save,
            create_new=create_new,
            folder_name=folder_name,
            new_folder_name=new_folder_name,
            file_name=file_name,
            data_type=data_type,
            indicador_target=indicador_target
        )
        
        save_message.set(msg)
        save_success.set(success)
        
        # Mark as saved if successful
        if success:
            csv_saved.set(True)
    
    # Reset csv_saved when a new file is uploaded
    @reactive.Effect
    def _():
        # Watch for file changes
        input.csv_file()
        # Reset saved status when new file is selected
        csv_saved.set(False)
        save_message.set("")
        # Reset monthly generation status
        monthly_generated.set(False)
        monthly_df.set(None)
        monthly_conversion_message.set("")






