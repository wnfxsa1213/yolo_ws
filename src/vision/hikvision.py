"""
海康威视 MVS SDK 相机实现。
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .camera import CameraError, CameraInterface

try:  # pragma: no cover - OpenCV 非硬依赖
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - 宿主环境若无 MVS SDK，会在运行时报错
    import MvCameraControl_class as mv  # type: ignore
except ImportError as _sdk_exc:  # pragma: no cover
    mv = None  # type: ignore
    _SDK_IMPORT_ERROR = _sdk_exc
else:
    _SDK_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

if mv is None:
    _SDK_OK = 0
else:
    if hasattr(mv, "MV_OK"):
        _SDK_OK = getattr(mv, "MV_OK")
    else:
        _SDK_OK = 0
        logger.warning("MVS SDK 未定义 MV_OK 常量，默认使用 0 作为成功返回码。")

_SDK_LOCK = threading.Lock()
_SDK_REFCOUNT = 0
_BAYER_TO_BGR: Dict[int, int] = {}
if mv is not None and cv2 is not None:  # pragma: no cover - 依赖实际环境
    _candidate_bayer = {
        getattr(mv, "PixelType_Gvsp_BayerRG8", None): cv2.COLOR_BayerRG2BGR,
        getattr(mv, "PixelType_Gvsp_BayerBG8", None): cv2.COLOR_BayerBG2BGR,
        getattr(mv, "PixelType_Gvsp_BayerGB8", None): cv2.COLOR_BayerGB2BGR,
        getattr(mv, "PixelType_Gvsp_BayerGR8", None): cv2.COLOR_BayerGR2BGR,
    }
    _BAYER_TO_BGR = {k: v for k, v in _candidate_bayer.items() if k is not None and v is not None}


def _ensure_sdk_loaded() -> Any:
    if mv is None:
        raise CameraError(
            "未检测到海康 MVS SDK Python 绑定，请确认容器已正确配置 PYTHONPATH。"
        ) from _SDK_IMPORT_ERROR
    return mv


def _sdk_status_ok(ret: int) -> bool:
    return ret == 0 or ret == _SDK_OK


def _sdk_initialize() -> None:
    sdk = _ensure_sdk_loaded()
    global _SDK_REFCOUNT
    with _SDK_LOCK:
        if _SDK_REFCOUNT == 0:
            ret = sdk.MvCamera.MV_CC_Initialize()
            if not _sdk_status_ok(ret):
                raise CameraError(f"MVS SDK 初始化失败，错误码 0x{ret:08X}")
        _SDK_REFCOUNT += 1


def _sdk_finalize() -> None:
    if mv is None:
        return
    global _SDK_REFCOUNT
    with _SDK_LOCK:
        if _SDK_REFCOUNT == 0:
            return
        _SDK_REFCOUNT -= 1
        if _SDK_REFCOUNT == 0:
            ret = mv.MvCamera.MV_CC_Finalize()
            if not _sdk_status_ok(ret):  # pragma: no cover - 最后一轮清理失败仅记录日志
                logger.debug("MVS SDK Finalize 返回非零错误码: 0x%08X", ret)


def _uint32_to_ipv4(value: int) -> str:
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _hex_error(ret: int) -> str:
    return f"0x{ret:08X}"


@dataclass
class HikCameraConfig:
    """HikCamera 初始化配置。"""

    device_ip: str
    local_ip: Optional[str] = None
    netmask: str = "255.255.255.0"
    gateway: str = "0.0.0.0"
    adapter_name: Optional[str] = None
    stream_timeout_ms: int = 1000
    stream_buffer_count: int = 4
    width: Optional[int] = None
    height: Optional[int] = None
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    intrinsics: Optional[Dict[str, float]] = None
    sdk_log_path: Optional[str] = None

    def __post_init__(self) -> None:
        """配置合法性校验，避免运行期出现基础错误。"""
        import ipaddress

        try:
            ipaddress.ip_address(self.device_ip)
        except ValueError as exc:  # pragma: no cover - 防御式检查
            raise ValueError(f"无效的设备 IP 地址: {self.device_ip}") from exc

        if self.stream_timeout_ms <= 0:
            raise ValueError(f"stream_timeout_ms 必须 > 0，当前值: {self.stream_timeout_ms}")

        if self.stream_buffer_count <= 0:
            raise ValueError(f"stream_buffer_count 必须 > 0，当前值: {self.stream_buffer_count}")

        if self.width is not None and self.width <= 0:
            raise ValueError(f"width 必须 > 0，当前值: {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"height 必须 > 0，当前值: {self.height}")


class HikCamera(CameraInterface):
    """海康威视 MVS SDK 相机后端。"""

    def __init__(self, config: HikCameraConfig, name: str = "hikvision") -> None:
        super().__init__(name=name)
        self._config = config
        self._is_open: bool = False
        self._mv_cam: Optional[Any] = None
        self._device_info: Optional[Any] = None
        self._payload_size: int = 0
        self._width: int = 0
        self._height: int = 0
        self._timeout_ms: int = max(int(self._config.stream_timeout_ms), 1)
        self._sdk_attached: bool = False

    @property
    def config(self) -> HikCameraConfig:
        """返回配置副本，避免外部直接修改内部状态。"""
        clone = replace(self._config)
        if clone.intrinsics is not None:
            clone.intrinsics = dict(clone.intrinsics)
        return clone

    def open(self) -> bool:
        """打开海康相机设备。"""
        if self._is_open:
            logger.warning("HikCamera[%s] 已处于打开状态，跳过重复打开。", self.name)
            return True

        sdk = _ensure_sdk_loaded()
        sdk_acquired = False
        try:
            _sdk_initialize()
            sdk_acquired = True
        except CameraError:
            raise
        except Exception as exc:  # pragma: no cover - 防御式兜底
            raise CameraError(f"MVS SDK 初始化失败: {exc}") from exc

        cam: Optional[Any] = None
        try:
            device_list = sdk.MV_CC_DEVICE_INFO_LIST()
            ret = sdk.MvCamera.MV_CC_EnumDevices(getattr(sdk, "MV_GIGE_DEVICE", 1), device_list)
            if not _sdk_status_ok(ret):
                raise CameraError(f"枚举海康相机失败，错误码 {_hex_error(ret)}")

            if device_list.nDeviceNum == 0:
                raise CameraError("未检测到任何海康相机，请检查连接与网段配置。")

            selected = self._select_device(device_list)
            if selected is None:
                raise CameraError(f"未找到 IP 为 {self._config.device_ip} 的海康相机。")

            cam = sdk.MvCamera()
            ret = cam.MV_CC_CreateHandle(selected)
            if not _sdk_status_ok(ret):
                raise CameraError(f"创建设备句柄失败，错误码 {_hex_error(ret)}")

            ret = cam.MV_CC_OpenDevice(getattr(sdk, "MV_ACCESS_Exclusive", 1), 0)
            if not _sdk_status_ok(ret):
                raise CameraError(f"打开海康相机失败，错误码 {_hex_error(ret)}")

            self._mv_cam = cam
            self._device_info = selected

            self._apply_network_optimisation()
            self._cache_dimensions()
            self._apply_initial_parameters()
            self._cache_dimensions()

            ret = cam.MV_CC_StartGrabbing()
            if not _sdk_status_ok(ret):
                raise CameraError(f"启动取流失败，错误码 {_hex_error(ret)}")

            self._sdk_attached = True
            self._is_open = True
            logger.info(
                "HikCamera[%s] 打开成功，分辨率 %dx%d，payload=%d 字节。",
                self.name,
                self._width,
                self._height,
                self._payload_size,
            )
            return True
        except CameraError:
            if cam is not None:
                self._safe_shutdown(cam, opened=True)
            raise
        except Exception as exc:  # pragma: no cover - 捕获底层异常
            if cam is not None:
                self._safe_shutdown(cam, opened=True)
            raise CameraError(f"打开海康相机出现异常: {exc}") from exc
        finally:
            if not self._is_open and sdk_acquired:
                _sdk_finalize()

    def close(self) -> None:
        """关闭相机并清理资源。"""
        if not self._is_open:
            return
        try:
            if self._mv_cam is not None:
                self._safe_shutdown(self._mv_cam, opened=True)
        finally:
            self._is_open = False
            self._mv_cam = None
            self._device_info = None
            self._payload_size = 0
            self._width = 0
            self._height = 0
            if self._sdk_attached:
                _sdk_finalize()
                self._sdk_attached = False
            logger.debug("HikCamera[%s] 关闭完成。", self.name)

    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """采集一帧图像。"""
        if not self._is_open or self._mv_cam is None:
            return None, 0.0

        sdk = _ensure_sdk_loaded()
        frame = sdk.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
        timeout_ms = max(int(timeout * 1000), 1)

        ret = self._mv_cam.MV_CC_GetImageBuffer(frame, timeout_ms)
        if not _sdk_status_ok(ret):
            if ret not in (getattr(sdk, "MV_E_TIMEOUT", -0x2710),):
                logger.debug("HikCamera[%s] 取流失败，错误码 %s", self.name, _hex_error(ret))
            return None, 0.0

        try:
            info = frame.stFrameInfo
            width = int(info.nWidth or info.nExtendWidth or self._width)
            height = int(info.nHeight or info.nExtendHeight or self._height)
            frame_len = int(info.nFrameLen)
            pixel_type = int(info.enPixelType)
            if width <= 0 or height <= 0 or frame_len <= 0:
                logger.debug(
                    "HikCamera[%s] 无效帧信息 width=%d height=%d len=%d",
                    self.name,
                    width,
                    height,
                    frame_len,
                )
                return None, 0.0
            if not bool(frame.pBufAddr):
                logger.debug("HikCamera[%s] 帧数据指针为空。", self.name)
                return None, 0.0

            numpy_buffer = np.ctypeslib.as_array(frame.pBufAddr, shape=(frame_len,))
            data = np.array(numpy_buffer, copy=True)
            image = self._convert_image(data, width, height, pixel_type)
            timestamp = time.time() * 1000.0
            return image, timestamp
        finally:
            self._mv_cam.MV_CC_FreeImageBuffer(frame)

    def get_intrinsics(self) -> Dict[str, float]:
        """返回相机内参。"""
        if not self._config.intrinsics:
            raise CameraError("尚未配置海康相机内参，无法返回。")
        required_keys = {"fx", "fy", "cx", "cy"}
        missing = required_keys.difference(self._config.intrinsics.keys())
        if missing:
            raise CameraError(f"相机内参不完整，缺少: {sorted(missing)}")
        return dict(self._config.intrinsics)

    def set_exposure(self, exposure_us: float) -> bool:
        """设置曝光时间（微秒）。"""
        self._config.exposure_us = exposure_us
        if not self._is_open or self._mv_cam is None:
            return True
        ret = self._mv_cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        if not _sdk_status_ok(ret):
            logger.error("HikCamera[%s] 设置曝光失败，错误码 %s", self.name, _hex_error(ret))
            return False
        return True

    def set_gain(self, gain_db: float) -> bool:
        """设置增益（dB）。"""
        self._config.gain_db = gain_db
        if not self._is_open or self._mv_cam is None:
            return True
        ret = self._mv_cam.MV_CC_SetFloatValue("Gain", float(gain_db))
        if not _sdk_status_ok(ret):
            logger.error("HikCamera[%s] 设置增益失败，错误码 %s", self.name, _hex_error(ret))
            return False
        return True

    # --- 内部辅助方法 --------------------------------------------------

    def _select_device(self, device_list: Any) -> Optional[Any]:
        sdk = _ensure_sdk_loaded()
        desired_ip = self._config.device_ip
        for idx in range(device_list.nDeviceNum):
            if not bool(device_list.pDeviceInfo[idx]):
                continue
            dev_ptr = ctypes.cast(device_list.pDeviceInfo[idx], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO))
            dev_info = dev_ptr.contents
            if dev_info.nTLayerType != getattr(sdk, "MV_GIGE_DEVICE", 1):
                continue
            current_ip = _uint32_to_ipv4(dev_info.SpecialInfo.stGigEInfo.nCurrentIp)
            if current_ip == desired_ip:
                logger.debug("HikCamera[%s] 命中目标设备: %s", self.name, current_ip)
                return dev_info
        return None

    def _apply_network_optimisation(self) -> None:
        if self._mv_cam is None or self._device_info is None or mv is None:
            return
        if self._device_info.nTLayerType != getattr(mv, "MV_GIGE_DEVICE", 1):
            return

        packet_size = self._mv_cam.MV_CC_GetOptimalPacketSize()
        if isinstance(packet_size, int) and packet_size > 0:
            ret = self._mv_cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            if not _sdk_status_ok(ret):
                logger.debug("设置 GevSCPSPacketSize 失败，错误码 %s", _hex_error(ret))
        else:
            logger.debug("获取最优包大小失败，返回值: %s", packet_size)

        ret = self._mv_cam.MV_CC_SetEnumValue("TriggerMode", getattr(mv, "MV_TRIGGER_MODE_OFF", 0))
        if not _sdk_status_ok(ret):
            logger.debug("关闭触发模式失败，错误码 %s", _hex_error(ret))

        if self._config.sdk_log_path:
            ret = self._mv_cam.MV_CC_SetSDKLogPath(self._config.sdk_log_path)
            if not _sdk_status_ok(ret):
                logger.debug("设置 SDK 日志路径失败，错误码 %s", _hex_error(ret))

    def _cache_dimensions(self) -> None:
        if self._mv_cam is None or mv is None:
            return
        self._payload_size = self._get_int_feature("PayloadSize") or self._payload_size
        self._width = self._get_int_feature("Width") or self._width or (self._config.width or 0)
        self._height = self._get_int_feature("Height") or self._height or (self._config.height or 0)
        if self._payload_size == 0 and self._width and self._height:
            self._payload_size = self._width * self._height

    def _apply_initial_parameters(self) -> None:
        if self._mv_cam is None or mv is None:
            return
        if self._config.exposure_us is not None:
            ret = self._mv_cam.MV_CC_SetFloatValue("ExposureTime", float(self._config.exposure_us))
            if not _sdk_status_ok(ret):
                logger.debug("初始曝光设置失败，错误码 %s", _hex_error(ret))
        if self._config.gain_db is not None:
            ret = self._mv_cam.MV_CC_SetFloatValue("Gain", float(self._config.gain_db))
            if not _sdk_status_ok(ret):
                logger.debug("初始增益设置失败，错误码 %s", _hex_error(ret))
        if self._config.width is not None:
            ret = self._mv_cam.MV_CC_SetIntValue("Width", int(self._config.width))
            if not _sdk_status_ok(ret):
                logger.debug("初始 Width 设置失败，错误码 %s", _hex_error(ret))
        if self._config.height is not None:
            ret = self._mv_cam.MV_CC_SetIntValue("Height", int(self._config.height))
            if not _sdk_status_ok(ret):
                logger.debug("初始 Height 设置失败，错误码 %s", _hex_error(ret))

    def _get_int_feature(self, key: str) -> Optional[int]:
        if self._mv_cam is None or mv is None:
            return None
        value = mv.MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        ret = self._mv_cam.MV_CC_GetIntValue(key, value)
        if not _sdk_status_ok(ret):
            logger.debug("读取整型特性 %s 失败，错误码 %s", key, _hex_error(ret))
            return None
        return int(value.nCurValue)

    def _safe_shutdown(self, cam: Any, opened: bool) -> None:
        try:
            if opened:
                ret = cam.MV_CC_StopGrabbing()
                if not _sdk_status_ok(ret):
                    logger.debug("停止取流返回错误码 %s", _hex_error(ret))
        except Exception:  # pragma: no cover - best effort
            logger.debug("停止取流时出现异常", exc_info=True)
        try:
            ret = cam.MV_CC_CloseDevice()
            if not _sdk_status_ok(ret):
                logger.debug("关闭设备返回错误码 %s", _hex_error(ret))
        except Exception:  # pragma: no cover
            logger.debug("关闭设备时出现异常", exc_info=True)
        try:
            ret = cam.MV_CC_DestroyHandle()
            if not _sdk_status_ok(ret):
                logger.debug("销毁句柄返回错误码 %s", _hex_error(ret))
        except Exception:  # pragma: no cover
            logger.debug("销毁句柄时出现异常", exc_info=True)

    def _convert_image(self, data: np.ndarray, width: int, height: int, pixel_type: int) -> np.ndarray:
        """根据像素格式转换成 BGR 图像。"""
        if width <= 0 or height <= 0:
            raise CameraError("帧尺寸非法，无法转换。")

        if pixel_type in _BAYER_TO_BGR and cv2 is not None:
            raw = data.reshape(height, width)
            return cv2.cvtColor(raw, _BAYER_TO_BGR[pixel_type])

        if mv is not None and pixel_type == getattr(mv, "PixelType_Gvsp_BGR8_Packed", -1):
            return data.reshape(height, width, 3)

        if mv is not None and pixel_type == getattr(mv, "PixelType_Gvsp_RGB8_Packed", -1):
            rgb = data.reshape(height, width, 3)
            return rgb[..., ::-1]

        if mv is not None and pixel_type == getattr(mv, "PixelType_Gvsp_Mono8", -1):
            return data.reshape(height, width)

        expected = width * height
        if data.size == expected:
            return data.reshape(height, width)
        if data.size == expected * 3:
            return data.reshape(height, width, 3)

        logger.debug(
            "未知像素格式 0x%X，按一维数组返回，长度=%d，分辨率=%dx%d。",
            pixel_type,
            data.size,
            width,
            height,
        )
        return data.copy()
