from __future__ import annotations

import struct

from serial_comm.protocol import (
    FRAME_TYPE_STATUS,
    ProtocolDecoder,
    ProtocolEncoder,
)


def test_encode_command_roundtrip() -> None:
    encoder = ProtocolEncoder()
    frame = encoder.encode_command(12.34, -56.78, laser_on=True, heartbeat=False)
    # 帧结构：AA55 | type | len | payload | crc
    assert frame.startswith(b"\xAA\x55\x01\x06")

    decoder = ProtocolDecoder()
    decoder.feed(frame)
    decoded = decoder.pop_frame()
    assert decoded is not None
    assert decoded.frame_type == 0x01
    assert decoded.payload == struct.pack("<hhBB", 1234, -5678, 1, 0)


def test_status_decode() -> None:
    payload = struct.pack("<hhhhBB", 100, -200, 300, -400, 2, 0b1010)
    crc = 0
    frame_type = FRAME_TYPE_STATUS
    payload_len = len(payload)
    crc = frame_type ^ payload_len
    for b in payload:
        crc ^= b
    frame = b"\xAA\x55" + bytes((frame_type, payload_len)) + payload + bytes((crc,))

    decoder = ProtocolDecoder()
    decoder.feed(frame)
    decoded = decoder.pop_frame()
    assert decoded is not None
    assert decoded.frame_type == FRAME_TYPE_STATUS
    assert decoded.parsed is not None
    status = decoded.parsed
    assert status.commanded_pitch_deg == 1.0
    assert status.commanded_yaw_deg == -2.0
    assert status.target_pitch_deg == 3.0
    assert status.target_yaw_deg == -4.0
    assert status.mode == 2
    assert status.flags == 0b1010


def test_crc_error_resync() -> None:
    encoder = ProtocolEncoder()
    frame = bytearray(encoder.encode_heartbeat())
    frame[-1] ^= 0xFF  # 搞坏 CRC
    decoder = ProtocolDecoder()
    decoder.feed(bytes(frame))
    assert decoder.pop_frame() is None

