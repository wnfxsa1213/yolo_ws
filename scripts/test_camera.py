#!/usr/bin/env python3
"""
Aravis 相机测试脚本。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import yaml

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import cv2
except ImportError:  # pragma: no cover - 测试脚本允许缺少 OpenCV
    cv2 = None  # type: ignore

from utils.logger import setup_logger  # noqa: E402
from vision.camera import AravisCamera, CameraError  # noqa: E402


FrameTuple = Tuple[int, np.ndarray, float]


def load_camera_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "aravis" not in data:
        raise SystemExit(f"配置 {path} 缺少 aravis 节点，赶紧修")
    return data["aravis"]


def ensure_cv2(reason: str) -> "cv2":
    if cv2 is None:
        raise SystemExit(f"{reason} 需要 OpenCV，赶紧 `sudo apt install python3-opencv`")
    return cv2


def capture_frames(
    camera: AravisCamera,
    frames: int,
    timeout: float,
    logger,
) -> Iterator[FrameTuple]:
    for idx in range(max(frames, 0)):
        image, ts = camera.capture(timeout=timeout)
        if image is None:
            logger.warning("第 %d 帧超时", idx)
            continue
        logger.info("第 %d 帧 OK, timestamp=%.2fms shape=%s", idx, ts, image.shape)
        yield idx, image, ts


def save_frame(image: np.ndarray, index: int, directory: Path) -> None:
    cv_alias = ensure_cv2("保存图片")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"frame_{index:04d}.png"
    if not cv_alias.imwrite(str(target), image):  # pragma: no cover - OpenCV 返回 False 很罕见
        raise SystemExit(f"保存帧 {target} 失败，检查磁盘权限")


def preview_frame(image: np.ndarray) -> bool:
    cv_alias = ensure_cv2("显示预览")
    cv_alias.imshow("Aravis Preview", image)
    return (cv_alias.waitKey(1) & 0xFF) != ord("q")


def run(args: argparse.Namespace) -> None:
    config = load_camera_config(Path(args.config))
    logger = setup_logger("camera_test", level="INFO", console=True, file=False, force=True)
    try:
        cam = AravisCamera(config)
    except CameraError as exc:
        raise SystemExit(f"初始化相机翻车: {exc}") from exc

    logger.info("打开相机: %s", config.get("device_id") or "<默认>")
    if not cam.open():
        raise SystemExit("相机没打开，别搁这浪费时间")

    save_dir_path = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir_path is not None:
        ensure_cv2("保存图片")
    if args.show:
        ensure_cv2("显示预览")

    start = time.time()
    captured = 0
    try:
        for idx, image, _ts in capture_frames(cam, args.frames, args.timeout, logger):
            captured += 1
            if save_dir_path is not None:
                save_frame(image, idx, save_dir_path)
            if args.show and not preview_frame(image):
                break
    finally:
        stats = None
        try:
            stats = cam.get_stream_statistics()
        except Exception as exc:  # pragma: no cover - 统计获取失败不致命
            logger.warning("获取流统计失败: %s", exc)
        cam.close()
        if args.show and cv2 is not None:
            cv2.destroyAllWindows()
    duration = time.time() - start
    if captured:
        logger.info("平均FPS: %.2f", captured / duration)
    if stats:
        logger.info("流统计: %s", {key: int(value) for key, value in stats.items()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aravis Camera Smoke Test")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/camera_config.yaml"), help="相机配置文件")
    parser.add_argument("--frames", type=int, default=30, help="测试帧数")
    parser.add_argument("--timeout", type=float, default=0.5, help="采集超时 (秒)")
    parser.add_argument("--save-dir", help="保存图片目录，可选")
    parser.add_argument("--show", action="store_true", help="OpenCV 窗口预览")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
