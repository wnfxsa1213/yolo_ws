#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys
import time
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from serial_comm import SerialCommunicator, StatusPacket
from serial_comm.protocol import FRAME_HEADER, ProtocolEncoder, _xor_crc


def format_status(status: Optional[StatusPacket]) -> str:
    if status is None:
        return "status=None"
    return (
        f"mode={status.mode} flags=0x{status.flags:02X} "
        f"cmd_pitch={status.commanded_pitch_deg:.2f} "
        f"cmd_yaw={status.commanded_yaw_deg:.2f} "
        f"target_pitch={status.target_pitch_deg:.2f} "
        f"target_yaw={status.target_yaw_deg:.2f}"
    )


def build_frame_bytes(frame_type: int, payload: bytes) -> bytes:
    return FRAME_HEADER + bytes((frame_type & 0xFF, len(payload) & 0xFF)) + payload + bytes(
        (_xor_crc(frame_type, payload),)
    )


def run(
    port: str,
    duration: Optional[float],
    hz: float,
    amplitude_pitch: float,
    amplitude_yaw: float,
    dump_commands: bool,
    dump_status: bool,
    dump_file: Optional[Path],
) -> None:
    interval = 1.0 / hz
    encoder = ProtocolEncoder()
    dump_handle = open(dump_file, "a", encoding="utf-8") if dump_file else None

    def close_dump() -> None:
        if dump_handle:
            dump_handle.close()

    status_frames: list[str] = []

    def frame_logger(frame) -> None:
        raw = build_frame_bytes(frame.frame_type, frame.payload)
        hex_repr = raw.hex()
        if dump_status:
            print(f"[RX] {hex_repr}")
        if dump_handle:
            dump_handle.write(f"RX {hex_repr}\n")
        status_frames.append(hex_repr)

    with SerialCommunicator(port) as comm:
        if dump_status or dump_handle:
            comm.set_frame_callback(frame_logger)

        comm.request_status()
        time.sleep(0.2)
        print("initial:", format_status(comm.get_latest_status()))

        start = time.monotonic()
        steps = 0

        while duration is None or time.monotonic() - start < duration:
            t = time.monotonic() - start
            pitch = amplitude_pitch * math.sin(2 * math.pi * 0.1 * t)
            yaw = amplitude_yaw * math.sin(2 * math.pi * 0.05 * t)
            heartbeat = True
            laser_on = steps % int(max(1, hz)) == 0

            if dump_commands or dump_handle:
                cmd_frame = encoder.encode_command(pitch, yaw, laser_on, heartbeat=heartbeat)
                hex_cmd = cmd_frame.hex()
                if dump_commands:
                    print(f"[TX] {hex_cmd}")
                if dump_handle:
                    dump_handle.write(f"TX {hex_cmd}\n")

            comm.send_command(pitch, yaw, laser_on=laser_on, heartbeat=heartbeat)

            status = comm.get_latest_status()
            print(f"[{t:6.2f}s] {format_status(status)}")

            age = comm.last_status_age()
            if age and age > 0.2:
                print(f"⚠️  status age {age:.3f}s > 0.2s, requesting refresh")
            if age and age > 0.2:
                comm.request_status()

            steps += 1
            time.sleep(max(0.0, start + steps * interval - time.monotonic()))

        metrics = comm.get_metrics()
        print("metrics:", metrics)
        if dump_handle:
            dump_handle.write(f"METRICS {metrics}\n")
            dump_handle.write(f"STATUS_FRAMES {status_frames}\n")

        comm.set_frame_callback(None)

    close_dump()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson↔STM32 串口功能测试")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="串口设备路径")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="测试时长 (秒)，留空表示持续运行直到 Ctrl-C",
    )
    parser.add_argument("--hz", type=float, default=50.0, help="命令帧频率")
    parser.add_argument("--amp-pitch", type=float, default=5.0, help="pitch 振幅 (度)")
    parser.add_argument("--amp-yaw", type=float, default=10.0, help="yaw 振幅 (度)")
    parser.add_argument("--dump-commands", action="store_true", help="输出发送的命令帧十六进制")
    parser.add_argument("--dump-status", action="store_true", help="输出接收的状态帧十六进制")
    parser.add_argument(
        "--dump-file",
        type=Path,
        help="将命令帧/状态帧/指标追加写入指定文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(
            port=args.port,
            duration=args.duration,
            hz=args.hz,
            amplitude_pitch=args.amp_pitch,
            amplitude_yaw=args.amp_yaw,
            dump_commands=args.dump_commands,
            dump_status=args.dump_status,
            dump_file=args.dump_file,
        )
    except KeyboardInterrupt:
        print("中断，退出测试")


if __name__ == "__main__":
    main()
