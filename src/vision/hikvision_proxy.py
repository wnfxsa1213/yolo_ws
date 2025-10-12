"""宿主机侧 Hikvision 相机代理（阶段 5 草案）。"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .camera import CameraError, CameraInterface

_HEADER = struct.Struct("!4sI")
_FRAME_META = struct.Struct("!I I I d B")


@dataclass
class HikCameraProxyConfig:
    """HikCameraProxy 配置。"""

    socket_path: Path
    capture_timeout: float = 0.5
    connect_timeout: float = 2.0
    intrinsics: Optional[Dict[str, float]] = None


class HikCameraProxy(CameraInterface):
    """通过 IPC 与容器 camera_server 交互的相机实现。"""

    def __init__(self, config: HikCameraProxyConfig, name: str = "hikvision_proxy") -> None:
        super().__init__(name=name)
        self._cfg = config
        self._sock: Optional[socket.socket] = None

    def open(self) -> bool:
        if self._sock is not None:
            return True
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._cfg.connect_timeout)
        try:
            sock.connect(str(self._cfg.socket_path))
        except OSError as exc:
            raise CameraError(f"连接 HikCamera server 失败: {exc}") from exc
        self._sock = sock
        try:
            self._send_command(b"PING", b"")
            code, _ = self._recv_response()
            if code != b"PONG":
                raise CameraError(f"心跳握手失败，返回码 {code.decode(errors='ignore')}")
        except Exception:
            self.close()
            raise
        return True

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            try:
                self._send_command(b"STOP", b"")
            except Exception:
                pass
            self._sock.close()
        finally:
            self._sock = None

    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        if self._sock is None:
            raise CameraError("相机尚未打开")
        self._sock.settimeout(max(timeout, self._cfg.capture_timeout))
        self._send_command(b"CAPT", b"")
        code, payload = self._recv_response()
        if code == b"FRAM":
            if len(payload) < _FRAME_META.size:
                raise CameraError("帧数据 payload 长度不足")
            frame_id, width, height, timestamp_ms, channels = _FRAME_META.unpack(payload[: _FRAME_META.size])
            data = payload[_FRAME_META.size :]
            expected = width * height * (channels if channels > 0 else 1)
            if len(data) != expected:
                raise CameraError(f"帧数据长度不匹配，期望 {expected} 实际 {len(data)}")
            arr = np.frombuffer(data, dtype=np.uint8)
            if channels == 1:
                frame = arr.reshape(height, width)
            else:
                frame = arr.reshape(height, width, channels)
            return frame.copy(), timestamp_ms
        if code in (b"FAIL", b"ERRO"):
            return None, 0.0
        raise CameraError(f"未知响应码: {code.decode(errors='ignore')}")

    def get_intrinsics(self) -> Dict[str, float]:
        if not self._cfg.intrinsics:
            raise CameraError("未配置相机内参")
        return dict(self._cfg.intrinsics)

    def set_exposure(self, exposure_us: float) -> bool:
        if self._sock is None:
            return False
        payload = struct.pack("!d", float(exposure_us))
        self._send_command(b"SEXP", payload)
        code, _ = self._recv_response()
        return code == b"OKAY"

    def set_gain(self, gain_db: float) -> bool:
        if self._sock is None:
            return False
        payload = struct.pack("!d", float(gain_db))
        self._send_command(b"SGAI", payload)
        code, _ = self._recv_response()
        return code == b"OKAY"

    # internal ---------------------------------------------------
    def _send_command(self, command: bytes, payload: bytes) -> None:
        if len(command) != 4:
            raise ValueError("命令码必须是 4 字节")
        if self._sock is None:
            raise CameraError("socket 已关闭")
        header = _HEADER.pack(command, len(payload))
        self._sock.sendall(header + payload)

    def _recv_response(self) -> Tuple[bytes, bytes]:
        if self._sock is None:
            raise CameraError("socket 已关闭")
        header = self._recv_exact(_HEADER.size)
        code, length = _HEADER.unpack(header)
        payload = self._recv_exact(length) if length else b""
        return code, payload

    def _recv_exact(self, size: int) -> bytes:
        if self._sock is None:
            raise CameraError("socket 已关闭")
        chunks = bytearray()
        remaining = size
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise CameraError("socket 意外关闭")
            chunks.extend(chunk)
            remaining -= len(chunk)
        return bytes(chunks)
