from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from algorithms import CoordinateTransformer, GimbalOffsets, Intrinsics, YOLODetector
from serial_comm import SerialCommunicator
from utils.config import ConfigError, ConfigManager
from utils.logger import setup_logger

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


# --------------------------------------------------------------------------- #
# 辅助数据结构
# --------------------------------------------------------------------------- #


@dataclass
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5


@dataclass
class AxisFilterConfig:
    max_velocity: float
    limits: Tuple[float, float]


@dataclass
class SmoothingConfig:
    alpha: float
    deadband: float


class CommandSmoother:
    """简单的指数平滑 + 限速封装。"""

    @dataclass
    class _AxisState:
        config: AxisFilterConfig
        smoothing: SmoothingConfig
        value: Optional[float] = None
        time: Optional[float] = None

    def __init__(
        self,
        pitch_cfg: AxisFilterConfig,
        yaw_cfg: AxisFilterConfig,
        smoothing: SmoothingConfig,
    ) -> None:
        self._states = {
            "pitch": self._AxisState(config=pitch_cfg, smoothing=smoothing),
            "yaw": self._AxisState(config=yaw_cfg, smoothing=smoothing),
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _step(self, axis: str, target: float, now: float) -> Tuple[float, bool]:
        state = self._states[axis]
        cfg = state.config
        smoothing = state.smoothing

        target = self._clamp(target, cfg.limits[0], cfg.limits[1])
        prev_value = state.value
        prev_time = state.time

        changed = False

        if prev_value is None or prev_time is None:
            filtered = target
            changed = True
        else:
            dt = max(now - prev_time, 1e-3)
            if cfg.max_velocity > 0:
                max_step = cfg.max_velocity * dt
                target = self._clamp(target, prev_value - max_step, prev_value + max_step)
            alpha = self._clamp(smoothing.alpha, 0.0, 1.0)
            candidate = alpha * target + (1.0 - alpha) * prev_value
            if abs(candidate - prev_value) < smoothing.deadband:
                filtered = prev_value
            else:
                filtered = candidate
                changed = True
        state.value = filtered
        state.time = now
        return filtered, changed

    def update(
        self, pitch_target: float, yaw_target: float, now: float
    ) -> Tuple[float, float, bool]:

        filtered_pitch, pitch_changed = self._step("pitch", pitch_target, now)
        filtered_yaw, yaw_changed = self._step("yaw", yaw_target, now)
        changed = pitch_changed or yaw_changed
        return filtered_pitch, filtered_yaw, changed

    def reset(self) -> None:
        for state in self._states.values():
            state.value = None
            state.time = None


# --------------------------------------------------------------------------- #
# 配置与初始化
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson 云台主程序")
    parser.add_argument("--config", default="config/system_config.yaml", help="系统配置文件路径")
    parser.add_argument("--max-frames", type=int, default=0, help="处理的最大帧数，0 表示无限")
    parser.add_argument("--no-serial", action="store_true", help="禁用串口发送，调试用")
    parser.add_argument("--no-camera", action="store_true", help="禁用相机采集，仅做离线测试")
    parser.add_argument("--dry-run", action="store_true", help="不做推理，直接发送默认指令")
    return parser.parse_args(list(argv) if argv is not None else None)


def setup_logging(config: ConfigManager) -> logging.Logger:
    log_cfg = config.get("logging", default={}, expected_type=dict)
    project_cfg = config.get("project", default={}, expected_type=dict)
    logger = setup_logger(
        "target_tracker",
        level=project_cfg.get("log_level", "INFO"),
        log_dir=log_cfg.get("log_dir", "logs"),
        filename=log_cfg.get("file_name"),
        rotation=log_cfg.get("rotation", "time"),
        when=log_cfg.get("when", "midnight"),
        interval=int(log_cfg.get("interval", 1)),
        backup_count=int(log_cfg.get("backup_count", 7)),
        max_bytes=int(log_cfg.get("max_bytes", 10 * 1024 * 1024)),
        force=True,
    )
    return logger


def load_intrinsics(path: Optional[str]) -> Intrinsics:
    if not path:
        return Intrinsics()
    intr_path = Path(path)
    if not intr_path.exists():
        return Intrinsics()
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取相机内参。")
    with intr_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"相机内参格式非法: {path}")
    return Intrinsics(
        fx=float(data.get("fx", 1000.0)),
        fy=float(data.get("fy", 1000.0)),
        cx=float(data.get("cx", 0.0)),
        cy=float(data.get("cy", 0.0)),
    )


def build_coordinate_transformer(cfg: ConfigManager) -> CoordinateTransformer:
    intrinsics_path = cfg.get("camera.intrinsics_path", default=None)
    intrinsics = load_intrinsics(intrinsics_path)
    offsets_cfg = cfg.get("control.offsets", default=None)
    if isinstance(offsets_cfg, dict):
        offsets = GimbalOffsets(
            pitch_offset_deg=float(offsets_cfg.get("pitch", 0.0)),
            yaw_offset_deg=float(offsets_cfg.get("yaw", 0.0)),
        )
    else:
        offsets = GimbalOffsets()
    return CoordinateTransformer(intrinsics, offsets)


def make_command_smoother(cfg: ConfigManager) -> CommandSmoother:
    smoothing_cfg = cfg.get("control.smoothing", default={}, expected_type=dict)
    smoothing = SmoothingConfig(
        alpha=float(smoothing_cfg.get("alpha", 0.7)),
        deadband=float(smoothing_cfg.get("deadband_deg", 0.1)),
    )
    pitch_cfg = cfg.get("control.pitch", default={}, expected_type=dict)
    yaw_cfg = cfg.get("control.yaw", default={}, expected_type=dict)

    def axis_config(raw: Dict[str, float], default_limits: Tuple[float, float]) -> AxisFilterConfig:
        limits = raw.get("limits", default_limits)
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            low, high = float(limits[0]), float(limits[1])
        else:
            low, high = default_limits
        return AxisFilterConfig(
            max_velocity=float(raw.get("max_velocity", 0.0)),
            limits=(low, high),
        )

    pitch_axis = axis_config(pitch_cfg, (-90.0, 90.0))
    yaw_axis = axis_config(yaw_cfg, (-180.0, 180.0))
    return CommandSmoother(pitch_axis, yaw_axis, smoothing)


# --------------------------------------------------------------------------- #
# 主循环逻辑
# --------------------------------------------------------------------------- #


def select_target(
    detections: Iterable[DetectionBox],
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    priority: str = "center_distance",
) -> Optional[DetectionBox]:
    candidates: List[DetectionBox] = [
        det for det in detections if det.confidence >= min_confidence
    ]
    if not candidates:
        return None

    cx_ref = frame_width / 2.0
    cy_ref = frame_height / 2.0

    def score(det: DetectionBox) -> float:
        if priority == "bbox_area":
            return -(det.width * det.height)
        if priority == "confidence":
            return -det.confidence
        # 默认优先中心距离
        cx, cy = det.center
        dx = cx - cx_ref
        dy = cy - cy_ref
        return dx * dx + dy * dy

    return min(candidates, key=score)


def detection_to_boxes(raw_detections: Iterable) -> List[DetectionBox]:
    boxes: List[DetectionBox] = []
    for det in raw_detections:
        boxes.append(
            DetectionBox(
                x1=float(det.x1),
                y1=float(det.y1),
                x2=float(det.x2),
                y2=float(det.y2),
                confidence=float(det.confidence),
                class_id=int(det.class_id),
            )
        )
    return boxes


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = ConfigManager(args.config)
    logger = setup_logging(cfg)
    logger.info("=== Target Tracker 启动 ===")

    if args.no_serial:
        logger.warning("串口功能已禁用 (--no-serial)")
        serial = None
    else:
        serial_cfg = cfg.get("serial", expected_type=dict)
        serial = SerialCommunicator(
            port=serial_cfg.get("port", "/dev/ttyTHS1"),
            baudrate=int(serial_cfg.get("baudrate", 460800)),
            read_timeout=float(serial_cfg.get("timeout", 0.05)),
            reconnect_interval=float(serial_cfg.get("reconnect_interval", 1.0)),
            heartbeat_interval=float(serial_cfg.get("heartbeat_interval", 0.05)),
        )
        serial.start()
        serial.request_status()

    if args.no_camera:
        logger.warning("未启用相机 (--no-camera)，之后的推理将无法进行")
        camera = None
        frame_size = (cfg.get("camera.resolution")[0], cfg.get("camera.resolution")[1])
    else:
        camera_cfg_path = cfg.get("camera.config_path", default="config/camera_config.yaml")
        camera_cfg = ConfigManager(camera_cfg_path).get("aravis", expected_type=dict)
        from vision.camera import AravisCamera  # 延迟导入避免无此依赖时直接崩溃

        camera = AravisCamera(camera_cfg)
        if not camera.open():
            raise RuntimeError("相机打开失败")
        frame_size = (camera._width or cfg.get("camera.resolution")[0], camera._height or cfg.get("camera.resolution")[1])  # type: ignore[attr-defined]

    model_cfg = cfg.get("model", expected_type=dict)
    detector = YOLODetector(
        model_cfg.get("engine_path", "models/yolov8n_fp16.engine"),
        confidence_threshold=float(model_cfg.get("conf_threshold", 0.5)),
        nms_threshold=float(model_cfg.get("nms_threshold", 0.45)),
    )
    transformer = build_coordinate_transformer(cfg)
    smoother = make_command_smoother(cfg)

    tracking_cfg = cfg.get("tracking", default={}, expected_type=dict)
    min_conf = float(tracking_cfg.get("min_confidence", 0.5))
    priority = tracking_cfg.get("priority", "center_distance")
    max_lost_frames = int(tracking_cfg.get("max_lost_frames", 5))
    return_damping = float(tracking_cfg.get("return_damping", 0.9))
    return_deadband = float(tracking_cfg.get("return_deadband_deg", 0.1))

    if not (0.0 < return_damping < 1.0):
        raise ConfigError(f"return_damping 必须在 (0, 1) 范围内: {return_damping}")
    if return_deadband < 0:
        raise ConfigError(f"return_deadband_deg 必须 ≥ 0: {return_deadband}")
    if max_lost_frames < 0:
        raise ConfigError(f"max_lost_frames 必须 ≥ 0: {max_lost_frames}")

    debug_cfg = cfg.get("debug", default={}, expected_type=dict)
    print_fps = bool(debug_cfg.get("print_fps", False))
    show_image = bool(debug_cfg.get("show_image", False))
    display_interval = int(debug_cfg.get("display_interval", 2))  # 每N帧显示一次
    if display_interval < 1:
        logger.warning("debug.display_interval=%s 非法，自动回退到 1 (每帧显示)", display_interval)
        display_interval = 1
    window_name = "Target Tracker"
    window_created = False
    if show_image:
        if cv2 is None:  # type: ignore[truthy-function]
            raise ConfigError("show_image 需要安装 OpenCV (python3-opencv)")
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # type: ignore[attr-defined]
            window_created = True
        except cv2.error as exc:  # type: ignore[attr-defined]
            logger.warning("CV窗口创建失败(%s)，自动禁用 show_image。", exc)
            show_image = False

    running = True

    def handle_signal(signum, _frame):  # pragma: no cover - signal handler
        nonlocal running
        logger.info("收到结束信号(%s)，准备退出...", signum)
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    frame_counter = 0
    t_start = time.perf_counter()
    last_status_log = 0.0
    latest_pitch = 0.0
    latest_yaw = 0.0
    target_lost_frames = max_lost_frames + 1  # 启动视为“无目标”状态
    last_valid_pitch = 0.0
    last_valid_yaw = 0.0
    first_target_acquired = False
    cmd_sent_count = 0
    cmd_skipped_count = 0

    try:
        while running:
            if args.max_frames and frame_counter >= args.max_frames:
                logger.info("达到最大帧数 %d，退出主循环", args.max_frames)
                break

            if args.no_camera:
                logger.debug("未启用相机，跳过检测循环")
                time.sleep(0.1)
                continue

            frame, timestamp_ms = camera.capture(timeout=0.5)  # type: ignore[union-attr]
            if frame is None:
                logger.warning("未获取到图像帧")
                continue

            if args.dry_run:
                detections: List[DetectionBox] = []
            else:
                raw_dets = detector.detect(frame)
                detections = detection_to_boxes(raw_dets)

            target = select_target(
                detections,
                frame.shape[1],
                frame.shape[0],
                min_confidence=min_conf,
                priority=priority,
            )

            now = time.perf_counter()
            if target:
                cx, cy = target.center
                pitch_raw, yaw_raw = transformer.pixel_to_angle(
                    cx, cy, frame.shape[1], frame.shape[0]
                )
                last_valid_pitch, last_valid_yaw = pitch_raw, yaw_raw
                target_lost_frames = 0
                first_target_acquired = True
                logger.debug(
                    "目标跟踪: pitch=%.2f° yaw=%.2f°",
                    pitch_raw,
                    yaw_raw,
                )
                has_target = True
            else:
                has_target = False
                if not first_target_acquired:
                    pitch_raw, yaw_raw = 0.0, 0.0
                    logger.debug("等待首次目标获取")
                else:
                    target_lost_frames += 1
                    if target_lost_frames <= max_lost_frames:
                        pitch_raw, yaw_raw = last_valid_pitch, last_valid_yaw
                        logger.debug(
                            "目标丢失(保持): %d/%d",
                            target_lost_frames,
                            max_lost_frames,
                        )
                    else:
                        decay_steps = target_lost_frames - max_lost_frames
                        decay_factor = return_damping ** decay_steps
                        pitch_raw = last_valid_pitch * decay_factor
                        yaw_raw = last_valid_yaw * decay_factor
                        if abs(pitch_raw) < return_deadband:
                            pitch_raw = 0.0
                        if abs(yaw_raw) < return_deadband:
                            yaw_raw = 0.0
                        logger.debug(
                            "目标丢失(衰减): step=%d pitch=%.2f° yaw=%.2f°",
                            decay_steps,
                            pitch_raw,
                            yaw_raw,
                        )

            filtered_pitch, filtered_yaw, should_send = smoother.update(
                pitch_raw, yaw_raw, now
            )
            latest_pitch, latest_yaw = filtered_pitch, filtered_yaw

            if serial:
                if should_send:
                    serial.send_command(
                        filtered_pitch, filtered_yaw, laser_on=False, heartbeat=True
                    )
                    cmd_sent_count += 1
                    logger.debug(
                        "发送命令: pitch=%.2f° yaw=%.2f°",
                        filtered_pitch,
                        filtered_yaw,
                    )
                else:
                    cmd_skipped_count += 1
                    logger.debug("命令未变化，跳过发送")

                status = serial.get_latest_status()
                if status and now - last_status_log > 1.0:
                    metrics = serial.get_metrics()
                    logger.info(
                        "串口状态 mode=%d flags=0x%02X pitch=%.2f yaw=%.2f metrics=%s",
                        status.mode,
                        status.flags,
                        status.commanded_pitch_deg,
                        status.commanded_yaw_deg,
                        metrics,
                    )
                    last_status_log = now
            else:
                if frame_counter % 50 == 0:
                    logger.debug("DRY-RUN 指令: pitch=%.2f, yaw=%.2f", filtered_pitch, filtered_yaw)

            frame_counter += 1

            # 跳帧显示以优化性能
            should_display = (frame_counter % display_interval == 0)
            if show_image and window_created and cv2 is not None and should_display:
                try:
                    # 仅在需要显示时才拷贝图像
                    vis = frame.copy()

                    # 批量绘制检测框（减少函数调用开销）
                    for det in detections:
                        top_left = (int(det.x1), int(det.y1))
                        bottom_right = (int(det.x2), int(det.y2))
                        is_target = target is not None and det is target
                        color = (0, 0, 255) if is_target else (0, 255, 0)
                        thickness = 2 if is_target else 1
                        cv2.rectangle(vis, top_left, bottom_right, color, thickness)  # type: ignore[attr-defined]

                        # 简化标签绘制（减少字符串格式化）
                        label = f"id:{det.class_id} {det.confidence:.2f}"
                        cv2.putText(
                            vis,
                            label,
                            (top_left[0], max(12, top_left[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                    # 绘制目标中心标记
                    if target:
                        cx, cy = target.center
                        cv2.drawMarker(  # type: ignore[attr-defined]
                            vis,
                            (int(cx), int(cy)),
                            (255, 0, 0),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=18,
                            thickness=2,
                            line_type=cv2.LINE_AA,
                        )

                    # 合并状态信息绘制
                    elapsed_total = time.perf_counter() - t_start
                    fps_vis = frame_counter / elapsed_total if elapsed_total > 0 else 0.0
                    status_lines = [
                        f"pitch:{filtered_pitch:.2f} yaw:{filtered_yaw:.2f}",
                        f"FPS:{fps_vis:.1f} target:{'Y' if target else 'N'}",
                    ]
                    for i, line in enumerate(status_lines):
                        cv2.putText(
                            vis,
                            line,
                            (12, 24 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2 if i == 0 else 1,
                            cv2.LINE_AA,
                        )

                    cv2.imshow(window_name, vis)  # type: ignore[attr-defined]
                except cv2.error as exc:  # type: ignore[attr-defined]
                    logger.warning("show_image 渲染失败: %s，关闭调试窗口。", exc)
                    show_image = False
                    if window_created:
                        try:
                            cv2.destroyWindow(window_name)  # type: ignore[attr-defined]
                        except cv2.error:  # type: ignore[attr-defined]
                            pass
                        window_created = False

            # 始终处理键盘事件，即使不显示图像
            if show_image and window_created and cv2 is not None:
                key = cv2.waitKey(1) & 0xFF  # type: ignore[attr-defined]
                if key in (27, ord("q")):
                    logger.info("检测到退出按键(Q/ESC)，终止主循环。")
                    running = False
                    break

            if print_fps and frame_counter % 50 == 0:
                elapsed = now - t_start
                fps = frame_counter / elapsed if elapsed > 0 else 0.0
                logger.info("处理帧数=%d, FPS=%.2f, target=%s", frame_counter, fps, has_target)

    finally:
        logger.info(
            "执行完毕，累计处理帧数=%d, 最终指令 pitch=%.2f yaw=%.2f",
            frame_counter,
            latest_pitch,
            latest_yaw,
        )
        if serial:
            total_cmds = cmd_sent_count + cmd_skipped_count
            skip_rate = cmd_skipped_count / total_cmds if total_cmds > 0 else 0.0
            logger.info(
                "命令统计: 发送=%d, 跳过=%d, 跳过率=%.1f%%",
                cmd_sent_count,
                cmd_skipped_count,
                skip_rate * 100.0,
            )
        if serial:
            serial.stop()
        if not args.no_camera and camera:
            camera.close()  # type: ignore[union-attr]
        if window_created and cv2 is not None:
            cv2.destroyWindow(window_name)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 程序入口
# --------------------------------------------------------------------------- #


def main(argv: Optional[Iterable[str]] = None) -> None:
    try:
        args = parse_args(argv)
        run_pipeline(args)
    except ConfigError as exc:
        print(f"[ERROR] 配置错误: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("用户中断，退出。", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - 兜底
        print(f"[ERROR] 程序异常: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
