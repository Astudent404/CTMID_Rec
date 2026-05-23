from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_merged_config(paths: list[str | Path]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        config = deep_update(config, load_config(path))
    return config


def resolve_project_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root)
    data_cfg = config.get("data", {})
    for key in ("raw_inter_path", "raw_meta_path", "processed_dir", "text_feature_path", "image_feature_path"):
        value = data_cfg.get(key)
        if value and not Path(value).is_absolute():
            data_cfg[key] = str(project_root / value)
    train_cfg = config.get("train", {})
    checkpoint_dir = train_cfg.get("checkpoint_dir")
    if checkpoint_dir and not Path(checkpoint_dir).is_absolute():
        train_cfg["checkpoint_dir"] = str(project_root / checkpoint_dir)
    return config
