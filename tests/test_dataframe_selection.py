from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from back.services import dataframe_selection as ds


def test_resolve_catalog_aggregation_maps_catalog_values() -> None:
    assert ds.resolve_catalog_aggregation("suma") == "SUM"
    assert ds.resolve_catalog_aggregation("media") == "AVG"
    assert ds.resolve_catalog_aggregation("MAX") == "MAX"
    assert ds.resolve_catalog_aggregation(None) == "SUM"


def test_create_dataframe_uses_catalog_aggregation_only_for_target(monkeypatch) -> None:
    definitions = {
        "Target": {
            "nombre_colum_ref": "target_col",
            "nombre_tabla": "target_table",
            "nombre": "Target",
            "operacion_obj": "media",
        },
        "Predictor": {
            "nombre_colum_ref": "pred_col",
            "nombre_tabla": "pred_table",
            "nombre": "Predictor",
            "operacion_obj": "suma",
        },
    }
    calls: list[dict] = []

    monkeypatch.setattr(ds, "get_variable_definition", lambda name: definitions[name])
    monkeypatch.setattr(ds, "_detect_time_cols", lambda _table: ["anio", "mes"])
    monkeypatch.setattr(
        ds,
        "create_where_clauses",
        lambda filters_by_var, var_name, target_table=None: ([], [], []),
    )

    def fake_get_aggregated_series(
        schema,
        table,
        value_col,
        alias,
        time_cols,
        where_clauses=None,
        params=None,
        group_cols=None,
        agg=None,
    ):
        calls.append({"table": table, "alias": alias, "agg": agg})
        return [{alias: 1.0, "anio": 2024, "mes": 1}]

    monkeypatch.setattr(ds, "get_aggregated_series", fake_get_aggregated_series)

    df = ds.create_dataframe_based_on_selection("Target", ["Predictor"])

    assert list(df.columns) == ["Target", "anio", "mes", "Predictor"]
    assert calls[0]["table"] == "target_table"
    assert calls[0]["agg"] == "AVG"
    assert calls[1]["table"] == "pred_table"
    assert calls[1]["agg"] is None
