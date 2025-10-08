#!/usr/bin/env python3
"""
将ONNX模型转换为TensorRT引擎。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    max_workspace_gb: int,
    batch: int,
    height: int,
    width: int,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("ONNX解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("⚙️ 启用FP16精度")

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape = (1, 3, height, width)
    opt_shape = (batch, 3, height, width)
    max_shape = (batch, 3, height, width)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print("🚀 开始构建TensorRT引擎...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("构建TensorRT引擎失败")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    print(f"✅ 引擎已保存: {engine_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX -> TensorRT")
    parser.add_argument("--onnx", type=Path, default=Path("models/yolov8n.onnx"), help="ONNX模型路径")
    parser.add_argument("--engine", type=Path, default=Path("models/yolov8n_fp16.engine"), help="输出引擎路径")
    parser.add_argument("--fp16", action="store_true", help="启用FP16精度")
    parser.add_argument("--workspace", type=int, default=2, help="工作空间大小(GB)")
    parser.add_argument("--batch", type=int, default=1, help="优化profile批大小")
    parser.add_argument("--height", type=int, default=640, help="输入高度")
    parser.add_argument("--width", type=int, default=640, help="输入宽度")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_engine(args.onnx, args.engine, args.fp16, args.workspace, args.batch, args.height, args.width)
