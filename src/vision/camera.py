"""
Aravis 相机模块。

这个SB SDK不行就换开源的，保持接口简单方便测试。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis, GLib
except Exception:  # pragma: no cover
    Aravis = None  # type: ignore
    GLib = None  # type: ignore


_ARAVIS_COMMON_ERRORS: Tuple[type, ...] = (
    RuntimeError,
    AttributeError,
    ValueError,
    TypeError,
)
if "GLib" in globals() and GLib is not None:
    _ARAVIS_COMMON_ERRORS = (GLib.Error,) + _ARAVIS_COMMON_ERRORS
else:  # pragma: no cover - GLib not available
    GLib = None  # type: ignore


logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """相机异常。"""


class CameraInterface(ABC):
    """相机抽象接口。"""

    name: str

    def __init__(self, name: str = "camera") -> None:
        self.name = name

    @abstractmethod
    def open(self) -> bool:
        """打开相机设备。"""

    @abstractmethod
    def close(self) -> None:
        """关闭相机，资源必须释放干净。"""

    @abstractmethod
    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """
        采集一帧图像。

        Returns:
            (image, timestamp_ms)
        """

    @abstractmethod
    def get_intrinsics(self) -> Dict[str, float]:
        """返回内参，不标定就别指望特别准。"""

    @abstractmethod
    def set_exposure(self, exposure_us: float) -> bool:
        """设置曝光时间（微秒）。"""

    @abstractmethod
    def set_gain(self, gain_db: float) -> bool:
        """设置增益（dB）。"""


@dataclass
class AravisCameraConfig:
    """Aravis 相机配置。"""

    device_id: Optional[str] = None
    pixel_format: Optional[str] = None
    packet_size: Optional[int] = None
    packet_delay: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    auto_exposure: bool = False
    auto_gain: bool = False
    stream_buffer_count: int = 8

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "AravisCameraConfig":
        return cls(
            device_id=raw.get("device_id"),
            pixel_format=raw.get("pixel_format"),
            packet_size=raw.get("packet_size"),
            packet_delay=raw.get("packet_delay"),
            width=raw.get("width"),
            height=raw.get("height"),
            exposure_us=raw.get("exposure_us"),
            gain_db=raw.get("gain_db"),
            auto_exposure=bool(raw.get("auto_exposure", False)),
            auto_gain=bool(raw.get("auto_gain", False)),
            stream_buffer_count=int(raw.get("stream_buffer_count", 8)),
        )


class AravisCamera(CameraInterface):
    """基于 Aravis 的 GigE 相机实现。"""

    def __init__(self, config: Dict[str, object], name: str = "aravis") -> None:
        super().__init__(name=name)
        if Aravis is None:
            raise CameraError("Aravis 没装好，赶紧 `sudo apt install gir1.2-aravis-0.8`")
        self.config = AravisCameraConfig.from_dict(config)
        self._camera = None
        self._device = None
        self._stream = None
        self._is_open = False
        self._width = 0
        self._height = 0
        self._payload = 0
        self._pixel_format = ""
        self._bayer_converter: Optional[int] = None

    # --- 私有辅助 ---
    def _resolve_pixel_format(self) -> None:
        assert self._camera is not None
        desired = self.config.pixel_format
        if desired:
            try:
                if hasattr(self._camera, "set_pixel_format_from_string"):
                    self._camera.set_pixel_format_from_string(desired)
                elif hasattr(self._camera, "set_string_feature_value"):
                    self._camera.set_string_feature_value("PixelFormat", desired)
                elif hasattr(Aravis, "pixel_format_from_string") and hasattr(self._camera, "set_pixel_format"):
                    pixel_value = Aravis.pixel_format_from_string(desired)
                    self._camera.set_pixel_format(pixel_value)
                else:
                    logger.warning("当前 Aravis 版本无法设置 PixelFormat，沿用默认值")
            except _ARAVIS_COMMON_ERRORS as exc:
                device_hint = self.config.device_id or "<默认>"
                logger.warning(
                    "设置 PixelFormat=%s 失败: %s，沿用相机默认值。"
                    "跑 `arv-tool-0.8 features %s PixelFormat` 瞧瞧支持列表",
                    desired,
                    exc,
                    device_hint,
                )
        current_format = None
        try:
            if hasattr(self._camera, "get_string_feature_value"):
                current_format = self._camera.get_string_feature_value("PixelFormat")
            elif hasattr(self._camera, "get_pixel_format") and hasattr(Aravis, "pixel_format_to_string"):
                current_format = Aravis.pixel_format_to_string(self._camera.get_pixel_format())
        except _ARAVIS_COMMON_ERRORS as exc:
            logger.debug("读取 PixelFormat 失败: %s", exc)
        if not current_format:
            current_format = desired or "Unknown"
        self._pixel_format = current_format
        if "Bayer" in self._pixel_format:
            if cv2 is None:
                raise CameraError("Bayer 输出需要 OpenCV 转换，别忘了安装 python3-opencv")
            self._bayer_converter = self._guess_bayer_code(self._pixel_format)
        else:
            self._bayer_converter = None

    @staticmethod
    def _guess_bayer_code(name: str) -> Optional[int]:
        mapping = {
            "BayerRG8": getattr(cv2, "COLOR_BAYER_RG2BGR", None),
            "BayerBG8": getattr(cv2, "COLOR_BAYER_BG2BGR", None),
            "BayerGR8": getattr(cv2, "COLOR_BAYER_GR2BGR", None),
            "BayerGB8": getattr(cv2, "COLOR_BAYER_GB2BGR", None),
        }
        for key, value in mapping.items():
            if name.startswith(key) and value is not None:
                return value
        raise CameraError(f"别名不认识的 Bayer 格式: {name}")

    def _update_dimensions(self) -> None:
        assert self._camera is not None
        try:
            region = self._camera.get_region()
        except AttributeError as exc:
            raise CameraError("Aravis Camera 缺少 get_region 接口") from exc

        width: Optional[int] = None
        height: Optional[int] = None
        if isinstance(region, tuple):
            if len(region) >= 4:
                _, _, width, height = region[-4:]
        else:
            width = getattr(region, "width", None)
            height = getattr(region, "height", None)

        if not width or not height:
            raise CameraError(f"无法解析 get_region 结果: {region}")

        self._width = int(width)
        self._height = int(height)

    def _set_feature(self, feature: str, value: Optional[object]) -> None:
        if value is None:
            return
        assert self._camera is not None
        device = self._device
        try:
            if isinstance(value, bool):
                if hasattr(self._camera, "set_boolean"):
                    self._camera.set_boolean(feature, value)
                elif device and hasattr(device, "set_boolean_feature_value"):
                    device.set_boolean_feature_value(feature, value)
                else:
                    raise CameraError("不支持布尔特性设置")
            elif isinstance(value, int):
                if hasattr(self._camera, "set_integer"):
                    self._camera.set_integer(feature, int(value))
                elif device and hasattr(device, "set_integer_feature_value"):
                    device.set_integer_feature_value(feature, int(value))
                else:
                    raise CameraError("不支持整数特性设置")
            elif isinstance(value, float):
                if hasattr(self._camera, "set_float"):
                    self._camera.set_float(feature, float(value))
                elif device and hasattr(device, "set_float_feature_value"):
                    device.set_float_feature_value(feature, float(value))
                else:
                    raise CameraError("不支持浮点特性设置")
            else:
                text = str(value)
                if hasattr(self._camera, "set_string"):
                    self._camera.set_string(feature, text)
                elif device and hasattr(device, "set_string_feature_value"):
                    device.set_string_feature_value(feature, text)
                else:
                    raise CameraError("不支持字符串特性设置")
        except _ARAVIS_COMMON_ERRORS as exc:
            raise CameraError(f"设置 {feature} 失败: {exc}") from exc

    def _enable_required_interfaces(self) -> None:
        if not hasattr(Aravis, "enable_interface"):
            return

        candidates: List[str] = []
        if hasattr(Aravis, "get_available_camera_interfaces"):
            try:
                available: Optional[Iterable[object]] = Aravis.get_available_camera_interfaces()
            except _ARAVIS_COMMON_ERRORS as exc:  # pragma: no cover - 依赖底层实现
                logger.debug("查询 Aravis 接口列表失败: %s", exc)
                available = None
            if available is not None:
                for item in available:
                    name: Optional[str] = None
                    for attr in ("value_nick", "value_name", "name", "nick"):
                        value = getattr(item, attr, None)
                        if value:
                            name = str(value)
                            break
                    if not name:
                        text = str(item).strip()
                        name = text or None
                    if name:
                        candidates.append(name)

        if not candidates:
            logger.debug("Aravis 没提供接口列表，默认接口应已启用，跳过 enable_interface")
            return

        logger.debug("Aravis 可用接口: %s", ", ".join(candidates))

        seen: Set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            try:
                Aravis.enable_interface(normalized)
                logger.debug("Aravis.enable_interface 成功启用: %s", normalized)
                return
            except _ARAVIS_COMMON_ERRORS as exc:
                logger.debug("启用接口 %s 失败: %s", normalized, exc)

        logger.debug("未显式启用任何 Aravis 接口，可能已在默认状态")

    def _apply_feature(self, feature: str, value: float) -> None:
        if not self._is_open or self._camera is None:
            raise CameraError("相机没打开你调个锤子的特性")
        try:
            auto_enum = getattr(Aravis, "Auto", None)
            auto_off = getattr(auto_enum, "OFF", None) if auto_enum else None
            if feature == "ExposureTime" and hasattr(self._camera, "set_exposure_time_auto") and auto_off is not None:
                self._camera.set_exposure_time_auto(auto_off)
            if feature == "Gain" and hasattr(self._camera, "set_gain_auto") and auto_off is not None:
                self._camera.set_gain_auto(auto_off)

            if feature == "ExposureTime" and hasattr(self._camera, "set_exposure_time"):
                self._camera.set_exposure_time(float(value))
                return
            if feature == "Gain" and hasattr(self._camera, "set_gain"):
                self._camera.set_gain(float(value))
                return

            self._set_feature(feature, value)
        except CameraError:
            raise
        except _ARAVIS_COMMON_ERRORS as exc:
            raise CameraError(f"设置 {feature} 失败: {exc}") from exc

    # --- CameraInterface ---
    def open(self) -> bool:
        if self._is_open:
            return True
        try:
            if Aravis is None:
                raise CameraError("Aravis runtime 不可用")
            self._enable_required_interfaces()
            self._camera = Aravis.Camera.new(self.config.device_id)
            if self._camera is None:
                raise CameraError("找不到相机，检查 device_id 和网络配置")
            if hasattr(self._camera, "get_device"):
                self._device = self._camera.get_device()

            self._update_dimensions()
            if self.config.width or self.config.height:
                width = int(self.config.width or self._width)
                height = int(self.config.height or self._height)
                if hasattr(self._camera, "set_region"):
                    self._camera.set_region(0, 0, width, height)
                else:
                    self._set_feature("Width", width)
                    self._set_feature("Height", height)
                self._update_dimensions()

            self._resolve_pixel_format()

            if self.config.packet_size:
                if hasattr(self._camera, "gv_set_packet_size"):
                    self._camera.gv_set_packet_size(int(self.config.packet_size))
                else:
                    self._set_feature("GevSCPSPacketSize", int(self.config.packet_size))
            if self.config.packet_delay:
                if hasattr(self._camera, "gv_set_packet_delay"):
                    self._camera.gv_set_packet_delay(int(self.config.packet_delay))
                else:
                    self._set_feature("GevSCPD", int(self.config.packet_delay))

            if hasattr(self._camera, "set_exposure_time_auto"):
                auto_mode = Aravis.Auto.CONTINUOUS if self.config.auto_exposure else Aravis.Auto.OFF
                self._camera.set_exposure_time_auto(auto_mode)
            else:
                self._set_feature("ExposureAuto", "Continuous" if self.config.auto_exposure else "Off")

            if not self.config.auto_exposure and self.config.exposure_us:
                if hasattr(self._camera, "set_exposure_time"):
                    self._camera.set_exposure_time(float(self.config.exposure_us))
                else:
                    self._set_feature("ExposureTime", float(self.config.exposure_us))

            if hasattr(self._camera, "set_gain_auto"):
                auto_mode = Aravis.Auto.CONTINUOUS if self.config.auto_gain else Aravis.Auto.OFF
                self._camera.set_gain_auto(auto_mode)
            else:
                self._set_feature("GainAuto", "Continuous" if self.config.auto_gain else "Off")

            if not self.config.auto_gain and self.config.gain_db is not None:
                if hasattr(self._camera, "set_gain"):
                    self._camera.set_gain(float(self.config.gain_db))
                else:
                    self._set_feature("Gain", float(self.config.gain_db))

            self._stream = self._camera.create_stream(None, None)
            if self._stream is None:
                raise CameraError("创建流失败")

            if hasattr(self._camera, "get_payload"):
                self._payload = self._camera.get_payload()
            else:
                self._payload = self._width * self._height
            for _ in range(self.config.stream_buffer_count):
                buf = Aravis.Buffer.new_allocate(self._payload)
                self._stream.push_buffer(buf)

            self._camera.start_acquisition()
            self._is_open = True
            return True
        except CameraError:
            raise
        except _ARAVIS_COMMON_ERRORS as exc:
            raise CameraError(f"打开相机翻车: {exc}") from exc

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            if self._camera:
                self._camera.stop_acquisition()
            self._is_open = False
        finally:
            self._camera = None
            self._device = None
            self._stream = None

    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        if not self._is_open or self._stream is None:
            return None, 0.0
        timeout_us = int(max(timeout, 0.001) * 1_000_000)
        try:
            if hasattr(self._stream, "timeout_pop_buffer"):
                buffer = self._stream.timeout_pop_buffer(timeout_us)
            else:
                buffer = self._stream.pop_buffer()
        except _ARAVIS_COMMON_ERRORS as exc:  # pragma: no cover - 底层异常
            logger.warning("pop_buffer 异常: %s", exc)
            return None, 0.0
        if buffer is None:
            return None, 0.0
        try:
            data = buffer.get_data()
            if data is None:
                return None, 0.0
            frame = np.frombuffer(data, dtype=np.uint8)
            if self._bayer_converter is None:
                channels = 3 if "RGB8" in self._pixel_format or "BGR8" in self._pixel_format else 1
                if channels == 1:
                    frame = frame.reshape(self._height, self._width)
                else:
                    frame = frame.reshape(self._height, self._width, channels)
                if "RGB8" in self._pixel_format and cv2 is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = frame.reshape(self._height, self._width)
                frame = cv2.cvtColor(frame, self._bayer_converter)
            timestamp = time.time() * 1000.0
            return frame.copy(), timestamp
        finally:
            self._stream.push_buffer(buffer)

    def get_intrinsics(self) -> Dict[str, float]:
        fx = float(self._width) if self._width else 1920.0
        fy = float(self._height) if self._height else 1080.0
        return {"fx": fx, "fy": fy, "cx": fx / 2.0, "cy": fy / 2.0}

    def set_exposure(self, exposure_us: float) -> bool:
        try:
            self._apply_feature("ExposureTime", exposure_us)
            return True
        except CameraError:
            return False

    def set_gain(self, gain_db: float) -> bool:
        try:
            self._apply_feature("Gain", gain_db)
            return True
        except CameraError:
            return False


class CameraManager:
    """多相机管理，线程循环采集，队列只保留最新帧。"""

    def __init__(self) -> None:
        self._cameras: List[CameraInterface] = []
        self._queues: List[queue.Queue[Tuple[np.ndarray, float]]] = []
        self._threads: List[threading.Thread] = []
        self._running = False

    def add_camera(self, camera: CameraInterface) -> int:
        self._cameras.append(camera)
        self._queues.append(queue.Queue(maxsize=2))
        return len(self._cameras) - 1

    def start_all(self) -> bool:
        if self._running:
            return True
        opened: List[CameraInterface] = []
        for cam in self._cameras:
            if not cam.open():
                for opened_cam in opened:
                    opened_cam.close()
                self._running = False
                return False
            opened.append(cam)
        self._running = True
        for idx, cam in enumerate(self._cameras):
            thread = threading.Thread(
                target=self._capture_loop, args=(idx, cam), daemon=True
            )
            thread.start()
            self._threads.append(thread)
        return True

    def _capture_loop(self, index: int, camera: CameraInterface) -> None:
        queue_obj = self._queues[index]
        while self._running:
            try:
                frame, ts = camera.capture(timeout=1.0)
            except CameraError as exc:
                logger.warning("camera[%s] 采集异常: %s", camera.name, exc)
                time.sleep(0.1)
                continue
            if frame is None:
                continue
            try:
                queue_obj.put_nowait((frame, ts))
            except queue.Full:
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    pass
                try:
                    queue_obj.put_nowait((frame, ts))
                except queue.Full:
                    pass

    def get_frame(self, camera_index: int = 0, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], float]:
        try:
            return self._queues[camera_index].get(timeout=timeout)
        except (queue.Empty, IndexError):
            return None, 0.0

    def stop_all(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2.0)
        for camera in self._cameras:
            camera.close()
        self._threads.clear()

    def __len__(self) -> int:
        return len(self._cameras)
