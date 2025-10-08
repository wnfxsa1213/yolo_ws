"""
配置管理模块。

提供 YAML 加载、字段校验、热加载等能力。
"""
from __future__ import annotations

import os
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Union

import yaml

from . import ensure_dir

__all__ = ["ConfigError", "ConfigValidationError", "ConfigManager", "merge_dict"]

_MISSING = object()


class ConfigError(RuntimeError):
    """通用配置异常。"""


class ConfigValidationError(ConfigError):
    """配置校验异常。"""

    def __init__(self, missing_fields: Sequence[str]):
        message = "缺失必要配置项: " + ", ".join(missing_fields)
        super().__init__(message)
        self.missing_fields = list(missing_fields)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"解析 YAML 失败: {path}") from exc

    if not isinstance(content, MutableMapping):
        raise ConfigError(f"配置文件根节点必须是键值对: {path}")
    return dict(content)


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    递归合并字典，override 优先。
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, Mapping):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _split_key_path(path: str) -> Iterator[str]:
    for part in path.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"非法配置路径: '{path}'")
        yield part


@dataclass
class ConfigSource:
    path: Path
    defaults: Dict[str, Any]


class ConfigManager:
    """
    YAML 配置管理器。

    支持：
    - 默认配置合并
    - 必要字段校验
    - 自动重载（可选）
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        auto_reload: bool = False,
    ) -> None:
        self._root = Path(__file__).resolve().parents[2]
        self._source = ConfigSource(
            path=self._resolve_path(path),
            defaults=dict(defaults) if defaults else {},
        )
        self._auto_reload = auto_reload
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._mtime: float = 0.0

        self.reload()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (self._root / p).resolve()
        return p

    def reload(self) -> None:
        """强制重新加载配置。"""
        with self._lock:
            content = _load_yaml(self._source.path)
            merged = merge_dict(self._source.defaults, content)
            resolved = _resolve_env(merged)
            self._data = resolved
            try:
                self._mtime = self._source.path.stat().st_mtime
            except OSError:
                self._mtime = 0.0

    def reload_if_changed(self) -> bool:
        """检测文件更新时间变化后重新加载。"""
        if not self._auto_reload:
            return False
        try:
            mtime = self._source.path.stat().st_mtime
        except OSError:
            return False
        if mtime != self._mtime:
            self.reload()
            return True
        return False

    def as_dict(self, deep: bool = False) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data) if deep else dict(self._data)

    def get(
        self,
        key_path: str,
        default: Any = _MISSING,
        *,
        expected_type: Optional[type] = None,
        fallback_type: Optional[type] = None,
    ) -> Any:
        """
        获取配置项。

        Args:
            key_path: 点分路径，如 "model.engine_path"。
            default: 默认返回值。
            expected_type: 期望类型，不符合则抛异常。
            fallback_type: 允许自动转换的类型。
        """
        with self._lock:
            current: Any = self._data
            for key in _split_key_path(key_path):
                if not isinstance(current, Mapping) or key not in current:
                    if default is _MISSING:
                        raise KeyError(f"未找到配置项: {key_path}")
                    return default
                current = current[key]

            if expected_type and current is not None and not isinstance(
                current, expected_type
            ):
                if fallback_type and isinstance(current, fallback_type):
                    return expected_type(current)  # type: ignore[arg-type]
                raise TypeError(
                    f"配置项 {key_path} 类型错误，期望 {expected_type}, 实际 {type(current)}"
                )
            return current

    def set(self, key_path: str, value: Any) -> None:
        with self._lock:
            target = self._data
            parts = list(_split_key_path(key_path))
            for key in parts[:-1]:
                node = target.setdefault(key, {})
                if not isinstance(node, MutableMapping):
                    raise ConfigError(f"无法写入配置: {key_path}")
                target = node
            target[parts[-1]] = value

    def validate(self, required_fields: Iterable[str]) -> None:
        missing = []
        for field in required_fields:
            try:
                value = self.get(field)
            except KeyError:
                missing.append(field)
                continue
            if value in (None, ""):
                missing.append(field)
        if missing:
            raise ConfigValidationError(missing)

    def ensure_paths(self, keys: Iterable[str]) -> Dict[str, Path]:
        """
        确保指定路径配置存在并创建父目录，返回 Path 映射。
        """
        resolved: Dict[str, Path] = {}
        for key in keys:
            path_value = self.get(key)
            if not isinstance(path_value, str):
                raise TypeError(f"配置项 {key} 必须是路径字符串")
            path = self._resolve_path(path_value)
            if path.suffix:
                ensure_dir(path.parent)
            else:
                ensure_dir(path)
            resolved[key] = path
        return resolved

    def project_root(self) -> Path:
        return self._root

    def __getitem__(self, item: str) -> Any:
        with self._lock:
            return self._data[item]

    def __contains__(self, item: object) -> bool:
        with self._lock:
            return item in self._data

    def __repr__(self) -> str:
        return f"ConfigManager(path={self._source.path!s})"
