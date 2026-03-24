from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from back.api.main import app


client = TestClient(app)


def test_metadata_route_accepts_catalog_names_with_spaces() -> None:
    response = client.get("/api/database/Número de turistas por municipio/metadata")

    assert response.status_code == 200
    data = response.json()
    assert data
    assert data[0]["nombre"] == "Número de turistas por municipio"


def test_sarimax_run_smoke_without_predictors() -> None:
    payload = {
        "target_var": "Número de turistas por municipio",
        "predictors": [],
        "filters_by_var": {},
        "train_ratio": 0.7,
        "auto_params": False,
        "order": [0, 1, 0],
        "seasonal_order": [0, 0, 0, 0],
        "horizon": 3,
        "return_df": False,
    }

    response = client.post("/api/models/sarimax/run", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["n_obs"] > 0
    assert data["horizon"] == 3
    assert len(data["y_forecast"]) == 3


def test_xgboost_run_smoke_without_predictors() -> None:
    payload = {
        "target_var": "Número de turistas por municipio",
        "predictors": [],
        "filters_by_var": {},
        "train_ratio": 0.7,
        "auto_params": False,
        "xgb_params": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "use_target_lags": True,
        "max_lag": 6,
        "recursive_forecast": True,
        "horizon": 3,
        "return_df": False,
    }

    response = client.post("/api/models/xgboost/run", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["n_obs"] > 0
    assert data["horizon"] == 3
    assert len(data["y_forecast"]) == 3


def test_model_routes_accept_future_exogenous_values_for_predictor_merge() -> None:
    base_payload = {
        "target_var": "Número de turistas por municipio",
        "predictors": ["Número de pasajeros por aeropuerto"],
        "filters_by_var": {},
        "train_ratio": 0.7,
        "auto_params": False,
        "horizon": 3,
        "return_df": False,
        "scenario_future_values": [
            {
                "var": "Número de pasajeros por aeropuerto",
                "date": "2026-01-01",
                "value": 1000000,
            }
        ],
    }

    sarimax_payload = {
        **base_payload,
        "order": [0, 1, 0],
        "seasonal_order": [0, 0, 0, 0],
    }
    xgboost_payload = {
        **base_payload,
        "xgb_params": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "use_target_lags": True,
        "max_lag": 6,
        "recursive_forecast": True,
    }

    sarimax_response = client.post("/api/models/sarimax/run", json=sarimax_payload)
    xgboost_response = client.post("/api/models/xgboost/run", json=xgboost_payload)

    assert sarimax_response.status_code == 200
    assert xgboost_response.status_code == 200
