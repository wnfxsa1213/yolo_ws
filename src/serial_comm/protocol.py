from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Deque, List, Optional

from collections import deque

FRAME_HEADER = b"\xAA\x55"
FRAME_TYPE_COMMAND = 0x01
FRAME_TYPE_HEARTBEAT = 0x02
FRAME_TYPE_STATUS_REQUEST = 0x10
FRAME_TYPE_STATUS = 0x81

_PITCH_MIN_CDEG = -9000
_PITCH_MAX_CDEG = 9000
_YAW_MIN_CDEG = -18000
_YAW_MAX_CDEG = 18000


def _xor_crc(frame_type: int, payload: bytes) -> int:
    """老王用最土的按字节 XOR 校验，够快够粗暴。"""
    crc = frame_type ^ (len(payload) & 0xFF)
    for byte in payload:
        crc ^= byte
    return crc & 0xFF


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _pack_frame(frame_type: int, payload: bytes) -> bytes:
    crc = _xor_crc(frame_type, payload)
    header = bytes((frame_type & 0xFF, len(payload) & 0xFF))
    return FRAME_HEADER + header + payload + bytes((crc,))


@dataclass
class StatusPacket:
    commanded_pitch_deg: float
    commanded_yaw_deg: float
    target_pitch_deg: float
    target_yaw_deg: float
    mode: int
    flags: int


@dataclass
class DecodedFrame:
    frame_type: int
    payload: bytes
    parsed: Optional[StatusPacket] = None


class ProtocolEncoder:
    """把高层命令塞进 MCU 协议帧的工具。"""

    def encode_command(
        self,
        pitch_deg: float,
        yaw_deg: float,
        laser_on: bool,
        heartbeat: bool = True,
    ) -> bytes:
        pitch_cdeg = _clamp(int(round(pitch_deg * 100.0)), _PITCH_MIN_CDEG, _PITCH_MAX_CDEG)
        yaw_cdeg = _clamp(int(round(yaw_deg * 100.0)), _YAW_MIN_CDEG, _YAW_MAX_CDEG)
        flags = 0x01 if heartbeat else 0x00
        payload = struct.pack(
            "<hhBB",
            pitch_cdeg,
            yaw_cdeg,
            1 if laser_on else 0,
            flags,
        )
        return _pack_frame(FRAME_TYPE_COMMAND, payload)

    def encode_heartbeat(self) -> bytes:
        return _pack_frame(FRAME_TYPE_HEARTBEAT, b"")

    def encode_status_request(self) -> bytes:
        return _pack_frame(FRAME_TYPE_STATUS_REQUEST, b"")


class ProtocolDecoder:
    """把串口里那点破数据按状态机扒回来。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._frames: Deque[DecodedFrame] = deque()
        self._crc_errors = 0

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._buffer.extend(data)
        while True:
            start = self._buffer.find(FRAME_HEADER)
            if start < 0:
                self._buffer.clear()
                return
            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < 4:
                return
            frame_type = self._buffer[2]
            payload_len = self._buffer[3]
            total_len = 4 + payload_len + 1
            if len(self._buffer) < total_len:
                return
            payload = bytes(self._buffer[4 : 4 + payload_len])
            crc = self._buffer[4 + payload_len]
            if crc != _xor_crc(frame_type, payload):
                # 再遇到烂帧直接扔掉头部，重新同步。
                self._crc_errors += 1
                del self._buffer[:2]
                continue
            del self._buffer[:total_len]
            parsed = self._parse_frame(frame_type, payload)
            self._frames.append(DecodedFrame(frame_type, payload, parsed))

    def _parse_frame(self, frame_type: int, payload: bytes) -> Optional[StatusPacket]:
        if frame_type != FRAME_TYPE_STATUS or len(payload) != 10:
            return None
        commanded_pitch, commanded_yaw, target_pitch, target_yaw, mode, flags = struct.unpack(
            "<hhhhBB", payload
        )
        return StatusPacket(
            commanded_pitch_deg=commanded_pitch / 100.0,
            commanded_yaw_deg=commanded_yaw / 100.0,
            target_pitch_deg=target_pitch / 100.0,
            target_yaw_deg=target_yaw / 100.0,
            mode=mode,
            flags=flags,
        )

    def pop_frame(self) -> Optional[DecodedFrame]:
        if not self._frames:
            return None
        return self._frames.popleft()

    def get_all_frames(self) -> List[DecodedFrame]:
        frames = list(self._frames)
        self._frames.clear()
        return frames

    @property
    def crc_errors(self) -> int:
        return self._crc_errors


__all__ = [
    "DecodedFrame",
    "FRAME_TYPE_COMMAND",
    "FRAME_TYPE_HEARTBEAT",
    "FRAME_TYPE_STATUS",
    "FRAME_TYPE_STATUS_REQUEST",
    "ProtocolDecoder",
    "ProtocolEncoder",
    "StatusPacket",
]
