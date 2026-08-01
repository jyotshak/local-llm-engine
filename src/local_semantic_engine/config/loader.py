"""Load TOML configuration and environment overrides deterministically."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.core.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.toml"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _parse_environment_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _environment_overrides(environment: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    prefix = "LSE_"
    for key, value in environment.items():
        if not key.startswith(prefix) or key == "LSE_CONFIG_PATH":
            continue
        path = key.removeprefix(prefix).lower().split("__")
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _parse_environment_value(value)
    return result


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Could not parse configuration file: {path}.") from exc


def load_settings(
    config_path: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load defaults, optional TOML, then ``LSE_`` environment overrides."""

    current_environment = os.environ if environment is None else environment
    path_from_environment = current_environment.get("LSE_CONFIG_PATH")
    selected_path = config_path or (Path(path_from_environment) if path_from_environment else None)

    data: dict[str, Any] = _read_toml(DEFAULT_CONFIG_PATH)
    if selected_path is not None:
        data = _deep_merge(data, _read_toml(selected_path))
    data = _deep_merge(data, _environment_overrides(current_environment))

    try:
        return AppSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError("Configuration values are invalid.", details=[str(exc)]) from exc
