"""
Python接口，封装C++检测与坐标转换模块。
"""
from __future__ import annotations

from importlib import import_module

try:
    detection_core = import_module("detection_core")
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "未找到 detection_core 模块，请先在 src/algorithms 下运行 CMake 构建"
    ) from exc

YOLODetector = detection_core.YOLODetector
Detection = detection_core.Detection
CoordinateTransformer = detection_core.CoordinateTransformer
Intrinsics = detection_core.Intrinsics
GimbalOffsets = detection_core.GimbalOffsets

__all__ = [
    "YOLODetector",
    "Detection",
    "CoordinateTransformer",
    "Intrinsics",
    "GimbalOffsets",
]
