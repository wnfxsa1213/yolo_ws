"""
通用工具函数。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Union

__all__ = ["get_timestamp", "ensure_dir", "parse_version"]


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    返回格式化时间戳。

    Args:
        fmt: strftime 格式字符串。
    """
    return datetime.now().strftime(fmt)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目标目录存在。

    Args:
        path: 目录路径。

    Returns:
        创建或存在的目录 Path。
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def parse_version(version: Union[str, Iterable[int]]) -> Tuple[int, ...]:
    """
    将版本号解析为整数元组，便于比较。

    Args:
        version: 版本字符串或数字迭代。
    """
    if isinstance(version, str):
        parts = version.strip().split(".")
    else:
        parts = list(version)
    parsed = []
    for part in parts:
        try:
            parsed.append(int(str(part).strip()))
        except ValueError as exc:
            raise ValueError(f"非法版本段: {part}") from exc
    return tuple(parsed)
