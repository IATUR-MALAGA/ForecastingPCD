import re
from fastapi import HTTPException

_PG_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def ensure_pg_identifier(value: str, field: str) -> str:
    if not _PG_IDENT.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"{field} inválido: {value!r}. Solo letras/números/_ y no empieces por número."
        )
    return value
