#!/usr/bin/env python3
"""
导出YOLOv8 PyTorch权重为ONNX模型。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def export(weights: Path, imgsz: int, output: Path, opset: int, dynamic: bool) -> None:
    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        optimize=True,
        dynamic=dynamic,
        simplify=not dynamic,
        half=False,
        project=output.parent,
        name=output.stem,
    )
    produced = output if output.suffix == ".onnx" else output.with_suffix(".onnx")
    print(f"✅ ONNX模型导出成功: {produced.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出YOLOv8模型到ONNX")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n.pt"), help="PyTorch权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（正方形）")
    parser.add_argument("--output", type=Path, default=Path("models/yolov8n.onnx"), help="输出ONNX文件路径")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset版本")
    parser.add_argument("--dynamic", action="store_true", help="导出动态维度模型")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.weights, args.imgsz, args.output, args.opset, args.dynamic)
