from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Agg = Literal["SUM", "AVG", "MIN", "MAX"]

class MonthlySeriesRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(..., alias="schema", examples=["IA"])
    table: str = Field(..., examples=["mi_tabla"])
    value_col: str = Field(..., examples=["valor"])
    agg: Agg = Field("SUM")
    filters: dict[str, list[str]] | None = None

class MonthlyPoint(BaseModel):
    fecha: date
    value: float
