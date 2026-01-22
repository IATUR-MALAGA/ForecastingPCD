import json
from typing import Any, Dict, Union, Tuple, List

def read_json(source: Union[str, bytes], is_file: bool = True) -> Dict[str, Any]:
    """
    Reads JSON from a file path (is_file=True) or from a JSON string/bytes (is_file=False)
    and returns it as a Python dict.

    Args:
        source: File path (if is_file=True) or JSON content as string/bytes (if is_file=False).
        is_file: True if source is a file path; False if source is JSON content.

    Returns:
        A dictionary with the parsed JSON content.

    Raises:
        FileNotFoundError: If the file path does not exist (file mode).
        json.JSONDecodeError: If the content is not valid JSON.
        ValueError: If the JSON root is not an object/dict.
    """
    if is_file:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        data = json.loads(source)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict).")

    return data



def get_all_variables(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list (array) with all variables.

    Expected JSON shape:
      {"variables": [ {...}, {...} ]}
    """
    variables = data.get("variables", [])
    if not isinstance(variables, list):
        raise ValueError('Expected "variables" to be a list.')
    return variables


def get_all_variables_table_and_name(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns a list of (NombreTabla, Nombre) tuples for all variables.

    Expected JSON shape:
      {"variables": [ { ... "NombreTabla": "...", "Nombre": "..." ... }, ... ]}
    """
    variables = data.get("variables", [])
    if not isinstance(variables, list):
        raise ValueError('Expected "variables" to be a list.')

    results: List[Tuple[str, str]] = []
    for v in variables:
        if not isinstance(v, dict):
            continue

        table = str(v.get("NombreTabla", "")).strip()
        name = str(v.get("Nombre", "")).strip()

        if table and name:
            results.append((table, name))

    return results




