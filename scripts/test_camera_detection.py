#!/usr/bin/env python3
"""
实时摄像头 + YOLO 检测联合测试脚本。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from algorithms import CoordinateTransformer, Intrinsics, YOLODetector  # type: ignore
from utils.config import ConfigManager
from utils.logger import setup_logger
from vision.camera import AravisCamera, CameraInterface  # type: ignore


def load_camera(cfg: ConfigManager) -> tuple[CameraInterface, CoordinateTransformer]:
    camera_cfg_path = cfg.get("camera.config_path")
    camera_cfg = ConfigManager(camera_cfg_path).get("aravis")
    camera = AravisCamera(camera_cfg)

    intr_cfg = cfg.get(
        "camera.intrinsics",
        default={
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": camera_cfg.get("width", 1280) / 2,
            "cy": camera_cfg.get("height", 1024) / 2,
        },
    )
    intrinsics = Intrinsics(intr_cfg["fx"], intr_cfg["fy"], intr_cfg["cx"], intr_cfg["cy"])
    transformer = CoordinateTransformer(intrinsics)
    return camera, transformer


def preprocess_frame(frame: np.ndarray, size: tuple[int, int], rotate: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    pad_w = size[0] - new_w
    pad_h = size[1] - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    cropped = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE) if rotate else cropped


def draw_detections(frame: np.ndarray, detections: Iterable, pitch_yaw: tuple[float, float] | None) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = f"ID:{det.class_id} conf:{det.confidence:.2f}"
        cv2.putText(vis, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if pitch_yaw:
        cv2.putText(
            vis,
            f"pitch:{pitch_yaw[0]:.2f} yaw:{pitch_yaw[1]:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return vis


def run(args: argparse.Namespace) -> None:
    logger = setup_logger("camera_detection", level="INFO", console=True, file=False, force=True)
    cfg = ConfigManager("config/system_config.yaml")
    engine_path = cfg.get("model.engine_path")
    detector = YOLODetector(engine_path, confidence_threshold=args.confidence)

    camera, transformer = load_camera(cfg)
    logger.info("打开相机...")
    if not camera.open():
        raise SystemExit("相机打开失败")

    frame_total = 0
    detected_frames = 0
    start_time = time.perf_counter()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while args.frames <= 0 or frame_total < args.frames:
            frame, timestamp = camera.capture()
            if frame is None:
                logger.warning("采集失败，丢帧")
                continue

            frame_total += 1
            processed = preprocess_frame(frame, (args.size, args.size), args.rotate)
            detections = detector.detect(processed)
            pitch_yaw = None
            if detections:
                detected_frames += 1
                det = detections[0]
                cx = (det.x1 + det.x2) / 2
                cy = (det.y1 + det.y2) / 2
                pitch_yaw = transformer.pixel_to_angle(cx, cy, processed.shape[1], processed.shape[0])

            if frame_total % args.log_interval == 0:
                if detections:
                    logger.info(
                        "frame %d | det=%d | pitch=%.2f yaw=%.2f | infer=%.2fms",
                        frame_total,
                        len(detections),
                        pitch_yaw[0],
                        pitch_yaw[1],
                        detector.inference_time_ms,
                    )
                else:
                    logger.info("frame %d | det=0 | infer=%.2fms", frame_total, detector.inference_time_ms)

            if args.show or save_dir:
                vis = draw_detections(processed, detections, pitch_yaw)
                if args.show:
                    cv2.imshow("Camera Detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if save_dir:
                    cv2.imwrite(str(save_dir / f"frame_{frame_total:05d}.png"), vis)

    finally:
        duration = time.perf_counter() - start_time
        camera.close()
        cv2.destroyAllWindows()
        fps = frame_total / duration if duration > 0 else 0
        logger.info(
            "===== 统计 =====\n总帧数=%d | 命中帧=%d | FPS=%.2f | 时长=%.2fs",
            frame_total,
            detected_frames,
            fps,
            duration,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="摄像头 + YOLO 联合测试")
    parser.add_argument("--frames", type=int, default=500, help="采集帧数，<=0 表示持续运行")
    parser.add_argument("--confidence", type=float, default=0.4, help="检测阈值")
    parser.add_argument("--size", type=int, default=640, help="输出分辨率 (正方形)")
    parser.add_argument("--rotate", action="store_true", help="顺时针旋转90°")
    parser.add_argument("--show", action="store_true", help="是否显示OpenCV窗口")
    parser.add_argument("--save-dir", help="保存检测结果的目录")
    parser.add_argument("--log-interval", type=int, default=30, help="日志输出间隔帧数")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
