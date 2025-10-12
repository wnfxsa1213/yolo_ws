#!/usr/bin/env python3
"""
Hikvision 混合架构端到端基准测试。

使用宿主机（或容器内）代理连接 camera_server，对连续帧进行采集，
统计 FPS、往返延迟以及 CPU 使用情况。
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List, Tuple

from utils.config import ConfigManager
from vision.camera import AravisCamera
from vision.hikvision_proxy import HikCameraProxy, HikCameraProxyConfig


def _percentile(samples: List[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    pos = (len(ordered) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark(
    camera, duration: float, timeout: float
) -> Tuple[float, float, float, float, float, float, float, float]:
    start = time.time()
    end = start + duration
    frames = 0
    attempts = 0
    latency_samples: List[float] = []
    cpu_start = time.process_time()

    while time.time() < end:
        attempts += 1
        iter_start = time.time()
        frame, ts = camera.capture(timeout=timeout)
        iter_end = time.time()
        if frame is None:
            continue
        frames += 1
        latency_samples.append((iter_end - iter_start) * 1000.0)  # ms

    elapsed = time.time() - start
    cpu_elapsed = time.process_time() - cpu_start
    fps = frames / elapsed if elapsed > 0 else 0.0
    avg_latency = statistics.mean(latency_samples) if latency_samples else 0.0
    min_latency = min(latency_samples) if latency_samples else 0.0
    max_latency = max(latency_samples) if latency_samples else 0.0
    p95_latency = _percentile(latency_samples, 95.0)
    p99_latency = _percentile(latency_samples, 99.0)
    cpu_percent = (cpu_elapsed / elapsed) * 100.0 if elapsed > 0 else 0.0
    drop_rate = (
        (attempts - frames) / attempts * 100.0 if attempts > 0 and attempts >= frames else 0.0
    )
    return (
        fps,
        avg_latency,
        min_latency,
        max_latency,
        p95_latency,
        p99_latency,
        cpu_percent,
        drop_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hikvision/Aravis 基准测试")
    parser.add_argument("--socket", default="/tmp/hikvision.sock", help="Unix socket path")
    parser.add_argument(
        "--backend", choices=["proxy", "aravis"], default="proxy", help="测试后端类型"
    )
    parser.add_argument(
        "--camera-config",
        default="config/camera_config.yaml",
        help="Aravis 配置文件路径（仅 backend=aravis 生效）",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="benchmark duration (seconds)")
    parser.add_argument("--timeout", type=float, default=0.5, help="per frame timeout (seconds)")
    parser.add_argument("--warmup", type=int, default=3, help="warmup frames")
    args = parser.parse_args()

    if args.backend == "proxy":
        cfg = HikCameraProxyConfig(socket_path=Path(args.socket))
        camera = HikCameraProxy(cfg)
        camera.open()
    else:
        cfg_manager = ConfigManager(args.camera_config)
        aravis_cfg = cfg_manager.get("aravis", expected_type=dict)
        camera = AravisCamera(aravis_cfg)
        if not camera.open():
            raise RuntimeError("Aravis 相机打开失败")
    try:
        for _ in range(args.warmup):
            camera.capture(timeout=args.timeout)

        (
            fps,
            latency,
            min_lat,
            max_lat,
            p95,
            p99,
            cpu,
            drop_rate,
        ) = benchmark(camera, args.duration, args.timeout)

        print("=" * 50)
        print("  海康混合架构端到端基准结果")
        print("=" * 50)
        print(f"  测试时长:     {args.duration:.1f}s")
        print(f"  帧率(FPS):    {fps:.2f}")
        print(f"  平均延迟:     {latency:.2f} ms")
        print(f"  最小延迟:     {min_lat:.2f} ms")
        print(f"  最大延迟:     {max_lat:.2f} ms")
        print(f"  P95 延迟:     {p95:.2f} ms")
        print(f"  P99 延迟:     {p99:.2f} ms")
        print(f"  进程 CPU:     {cpu:.1f}%")
        print(f"  丢帧率:       {drop_rate:.2f}%")
        print("=" * 50)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
