from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

from tests.test_utils import ensure_project_on_path

ensure_project_on_path()

from src.utils.config import ConfigManager, ConfigValidationError, merge_dict

MTIME_RESOLUTION_SECONDS = 1.1  # 文件系统 mtime 精度缓冲


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_basic_loading_and_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, {"logging": {"level": "DEBUG"}})

    defaults = {"project": {"name": "老王干活", "version": "0.1.0"}}
    manager = ConfigManager(config_path, defaults=defaults, auto_reload=True)

    assert manager.get("project.name") == "老王干活"
    assert manager.get("project.version") == "0.1.0"
    assert manager.get("logging.level") == "DEBUG"
    assert manager.get("missing.key", default="fallback") == "fallback"

    merged = merge_dict({"a": {"b": 1}}, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}

    target_path = tmp_path / "outputs" / "result.log"
    manager.set("paths.output", str(target_path))
    ensured = manager.ensure_paths(["paths.output"])
    assert ensured["paths.output"] == target_path
    assert target_path.parent.exists()


def test_validate_reports_missing_keys(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, {"logging": {"level": "INFO"}})

    manager = ConfigManager(config_path, auto_reload=False)

    with pytest.raises(ConfigValidationError) as exc_info:
        manager.validate(["logging.level", "logging.path"])
    assert "logging.path" in exc_info.value.missing_fields


def test_auto_reload_detects_changes(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, {"logging": {"level": "DEBUG"}})

    manager = ConfigManager(
        config_path,
        defaults={"project": {"name": "reload-test"}},
        auto_reload=True,
    )
    assert manager.get("project.name") == "reload-test"
    assert manager.get("logging.level") == "DEBUG"

    time.sleep(MTIME_RESOLUTION_SECONDS)
    updated = {"logging": {"level": "INFO"}, "camera": {"fps": 60}}
    _write_yaml(config_path, updated)
    os.utime(config_path, None)

    assert manager.reload_if_changed()
    assert manager.get("logging.level") == "INFO"
    assert manager.get("camera.fps") == 60
