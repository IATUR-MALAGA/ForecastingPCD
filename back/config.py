from __future__ import annotations

import os
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yml"

ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "FORECASTINGPCD_SERVER_HOST": ("server.host", str),
    "FORECASTINGPCD_SERVER_PORT": ("server.port", int),
    "FORECASTINGPCD_DB_HOST": ("db.connection.host", str),
    "FORECASTINGPCD_DB_PORT": ("db.connection.port", int),
    "FORECASTINGPCD_DB_NAME": ("db.connection.dbname", str),
    "FORECASTINGPCD_DB_USER": ("db.connection.user", str),
}

REQUIRED_PATHS: dict[str, type] = {
    "server.host": str,
    "server.port": int,
    "server.api_prefix": str,
    "server.title": str,
    "server.version": str,
    "db.default_schema": str,
    "db.identifier_regex": str,
    "db.connection.host": str,
    "db.connection.port": int,
    "db.connection.dbname": str,
    "models.common.train_ratio": (int, float),
    "models.common.horizon": int,
    "models.common.min_historical_rows": int,
    "models.sarimax.seasonal_period_s": int,
    "models.xgboost.max_lag": int,
}


def _set_dotted(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _get_dotted(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Falta la clave obligatoria '{dotted_path}' en config.yml")
        value = value[part]
    return value


def _convert(value: str, cast_type: Any) -> Any:
    if cast_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return cast_type(value)


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    for env_name, (path, cast_type) in ENV_OVERRIDES.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        _set_dotted(merged, path, _convert(raw_value, cast_type))
    return merged


def _validate_required(config: dict[str, Any]) -> None:
    for path, expected_type in REQUIRED_PATHS.items():
        value = _get_dotted(config, path)
        if not isinstance(value, expected_type):
            raise ValueError(
                f"La clave '{path}' debe ser de tipo {expected_type}, recibido {type(value)}"
            )


def _validate_constraints(config: dict[str, Any]) -> None:
    server_port = int(_get_dotted(config, "server.port"))
    db_port = int(_get_dotted(config, "db.connection.port"))
    train_ratio = float(_get_dotted(config, "models.common.train_ratio"))
    horizon = int(_get_dotted(config, "models.common.horizon"))
    min_hist = int(_get_dotted(config, "models.common.min_historical_rows"))
    sarimax_s = int(_get_dotted(config, "models.sarimax.seasonal_period_s"))
    xgb_max_lag = int(_get_dotted(config, "models.xgboost.max_lag"))

    if server_port <= 0:
        raise ValueError("server.port debe ser > 0")
    if db_port <= 0:
        raise ValueError("db.connection.port debe ser > 0")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("models.common.train_ratio debe estar entre 0 y 1")
    if horizon < 1:
        raise ValueError("models.common.horizon debe ser >= 1")
    if min_hist < 1:
        raise ValueError("models.common.min_historical_rows debe ser >= 1")
    if sarimax_s < 1:
        raise ValueError("models.sarimax.seasonal_period_s debe ser >= 1")
    if xgb_max_lag < 0:
        raise ValueError("models.xgboost.max_lag debe ser >= 0")


def _normalize_paths(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    paths = normalized.setdefault("paths", {})

    root = Path(paths.get("root", ".")).expanduser()
    if not root.is_absolute():
        root = (ROOT_DIR / root).resolve()

    paths["root"] = root

    for key, value in list(paths.items()):
        if key == "root" or not isinstance(value, str):
            continue
        path_value = Path(value).expanduser()
        paths[key] = (root / path_value).resolve() if not path_value.is_absolute() else path_value.resolve()

    return normalized


@dataclass(frozen=True)
class Settings:
    data: dict[str, Any]
    config_path: Path

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self.data
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


def load_config() -> Settings:
    config_path = CONFIG_PATH.resolve()
    logger.info("Loading configuration from %s", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe archivo de configuración: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}

    if not isinstance(parsed, dict):
        raise ValueError("config.yml debe contener un objeto YAML en la raíz")

    parsed = _apply_env_overrides(parsed)
    parsed = _normalize_paths(parsed)
    _validate_required(parsed)
    _validate_constraints(parsed)
    return Settings(data=parsed, config_path=config_path)


settings = load_config()
