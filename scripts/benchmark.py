#!/usr/bin/env python3
"""
YOLO TensorRT推理性能基准测试。
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from detection_core import YOLODetector


def load_image(image_path: Path, input_size: tuple[int, int]) -> np.ndarray:
    if image_path and image_path.suffix and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"无法读取图像: {image_path}")
        return img

    w, h = input_size
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def benchmark(detector: YOLODetector, image: np.ndarray, iterations: int) -> None:
    # 预热
    for _ in range(5):
        detector.detect(image)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detections = detector.detect(image)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    avg = statistics.mean(times)
    std = statistics.pstdev(times)
    print("========== Benchmark ==========")
    print(f"Iterations       : {iterations}")
    print(f"Average time (ms): {avg:.2f}")
    print(f"Stddev  time (ms): {std:.2f}")
    print(f"FPS              : {1000.0 / avg:.1f}")
    print(f"Detections/sample: {len(detections)}")
    print("================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO TensorRT Benchmark")
    parser.add_argument("--engine", type=Path, default=Path("models/yolov8n_fp16.engine"), help="TensorRT引擎路径")
    parser.add_argument("--image", type=Path, default=Path(), help="测试图像路径（留空则随机生成）")
    parser.add_argument("--iterations", type=int, default=100, help="测试次数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = YOLODetector(str(args.engine))
    img = load_image(args.image, (detector.input_width, detector.input_height))
    benchmark(detector, img, args.iterations)
