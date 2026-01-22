"""
Utilidades para formatear nombres de variables y hacer que se muestren de forma legible
"""

def format_variable_name(name: str) -> str:
    """
    Convierte nombres técnicos con guiones bajos a nombres legibles.
    Reemplaza _ por espacios y capitaliza la primera letra de cada palabra.
    
    Ejemplos:
        - "precio_m2_vivienda_alquiler_carretera_cadiz" -> "Precio M2 Vivienda Alquiler Carretera Cadiz"
        - "Número_de_transacciones_inmobiliarias_en_vivienda_de_segunda_mano" -> "Número De Transacciones Inmobiliarias En Vivienda De Segunda Mano"
    """
    if not name:
        return name
    
    # Reemplazar guiones bajos por espacios
    formatted = name.replace('_', ' ')
    
    # Capitalizar la primera letra de cada palabra
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted
