import re
from typing import Any


def _safe_alias(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\W+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def _find_col(df: Any, *candidates: str):
    columns = getattr(df, "columns", [])
    for c in candidates:
        if c and c in columns:
            return c
    return None
