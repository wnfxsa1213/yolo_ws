#!/usr/bin/env python3
"""Hikvision 相机取流服务（容器内运行）。

该脚本作为阶段 4 的参考实现，核心目标：
- 在容器内封装 HikCamera，以 Unix Domain Socket 提供帧数据与控制能力。
- 定义轻量通信协议，便于宿主机代理（阶段 5）集成。
- 保留足够的日志与错误处理钩子，方便后续扩展 supervisor/systemd 守护。
"""
from __future__ import annotations

import argparse
import logging
import os
import selectors
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from vision.hikvision import HikCamera, HikCameraConfig

logger = logging.getLogger("camera_server")

# 命令/响应报文头：4 字节命令 + 4 字节 payload 长度
_HEADER = struct.Struct("!4sI")
# 帧元数据：frame_id:uint32 width:uint32 height:uint32 timestamp_ms:float64 channels:uint8
_FRAME_META = struct.Struct("!I I I d B")


class CameraServerError(RuntimeError):
    """服务器级错误。"""


@dataclass
class ServerConfig:
    socket_path: Path
    camera: HikCameraConfig
    backlog: int = 1
    capture_timeout: float = 0.5
    heartbeat_interval: float = 10.0


class CameraServer:
    """Unix Socket + HikCamera 服务实现。"""

    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._camera = HikCamera(cfg.camera)
        self._selector = selectors.DefaultSelector()
        self._sock: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._running = False
        self._frame_id = 0
        self._lock = threading.Lock()
        self._last_heartbeat = time.time()

    # life cycle -------------------------------------------------
    def start(self) -> None:
        self._prepare_socket()
        self._camera.open()
        self._running = True
        logger.info("camera_server 已启动，监听 %s", self._cfg.socket_path)
        try:
            while self._running:
                events = self._selector.select(timeout=1.0)
                now = time.time()
                if (
                    self._client
                    and self._cfg.heartbeat_interval > 0
                    and now - self._last_heartbeat > self._cfg.heartbeat_interval * 2
                ):
                    logger.warning("心跳超时，断开客户端")
                    self._disconnect_client()
                if not events:
                    continue
                for key, mask in events:
                    callback = key.data
                    try:
                        callback(key.fileobj, mask)
                    except Exception:  # pragma: no cover
                        logger.exception("处理事件异常，断开连接")
                        if key.fileobj is self._client:
                            self._disconnect_client()
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._disconnect_client()
        if self._sock:
            try:
                self._selector.unregister(self._sock)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            if self._cfg.socket_path.exists():
                self._cfg.socket_path.unlink()
            self._sock = None
        self._camera.close()
        logger.info("camera_server 已停止")

    # socket ops -------------------------------------------------
    def _prepare_socket(self) -> None:
        path = self._cfg.socket_path
        if path.exists():
            logger.warning("socket %s 已存在，先删除旧文件", path)
            path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        sock.listen(self._cfg.backlog)
        sock.setblocking(False)
        self._sock = sock
        self._selector.register(sock, selectors.EVENT_READ, self._accept)

    def _accept(self, sock_obj: socket.socket, _mask: int) -> None:
        conn, _ = sock_obj.accept()
        conn.setblocking(True)
        if self._client is not None:
            logger.warning("已有客户端连接，拒绝新的连接")
            conn.close()
            return
        self._client = conn
        self._selector.register(conn, selectors.EVENT_READ, self._on_message)
        self._last_heartbeat = time.time()
        logger.info("客户端已连接")

    def _disconnect_client(self) -> None:
        if self._client is None:
            return
        try:
            self._selector.unregister(self._client)
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None
        logger.info("客户端已断开")

    # message protocol -------------------------------------------
    def _on_message(self, conn: socket.socket, _mask: int) -> None:
        try:
            header = conn.recv(_HEADER.size)
            if not header:
                self._disconnect_client()
                return
            if len(header) < _HEADER.size:
                # 数据尚未完整，等待下个事件
                return
            cmd_raw, length = _HEADER.unpack(header)
            payload = bytearray()
            while len(payload) < length:
                chunk = conn.recv(length - len(payload))
                if not chunk:
                    logger.warning("payload 接收中断，连接可能已关闭")
                    self._disconnect_client()
                    return
                payload.extend(chunk)
            if len(payload) != length:
                logger.warning("payload 长度不符，期望 %d 实际 %d", length, len(payload))
                return
            self._dispatch(conn, cmd_raw, bytes(payload))
        except BlockingIOError:
            return
        except OSError as exc:
            logger.warning("socket 异常: %s", exc)
            self._disconnect_client()

    def _dispatch(self, conn: socket.socket, cmd: bytes, payload: bytes) -> None:
        command = cmd.decode("ascii", errors="ignore")
        if command == "PING":
            self._last_heartbeat = time.time()
            self._send_response(conn, b"PONG", b"")
        elif command == "CAPT":
            self._handle_capture(conn)
        elif command == "SEXP":
            self._handle_set_exposure(conn, payload)
        elif command == "SGAI":
            self._handle_set_gain(conn, payload)
        elif command == "STOP":
            self._send_response(conn, b"STOP", b"")
            self._disconnect_client()
        else:
            logger.debug("未知命令: %s", command)
            self._send_response(conn, b"ERRO", b"unknown command")

    def _handle_capture(self, conn: socket.socket) -> None:
        frame, timestamp_ms = self._camera.capture(timeout=self._cfg.capture_timeout)
        if frame is None:
            self._send_response(conn, b"FAIL", b"timeout")
            return
        with self._lock:
            self._frame_id += 1
            frame_id = self._frame_id
        if frame.ndim == 2:
            height, width = frame.shape
            channels = 1
        else:
            height, width, channels = frame.shape
        payload = frame.tobytes()
        meta = _FRAME_META.pack(frame_id, width, height, float(timestamp_ms), channels)
        self._send_response(conn, b"FRAM", meta + payload)

    def _handle_set_exposure(self, conn: socket.socket, payload: bytes) -> None:
        if len(payload) < 8:
            self._send_response(conn, b"ERRO", b"invalid exposure payload")
            return
        exposure_us = struct.unpack("!d", payload[:8])[0]
        ok = self._camera.set_exposure(exposure_us)
        if ok:
            self._send_response(conn, b"OKAY", b"")
        else:
            self._send_response(conn, b"ERRO", b"set exposure failed")

    def _handle_set_gain(self, conn: socket.socket, payload: bytes) -> None:
        if len(payload) < 8:
            self._send_response(conn, b"ERRO", b"invalid gain payload")
            return
        gain_db = struct.unpack("!d", payload[:8])[0]
        ok = self._camera.set_gain(gain_db)
        if ok:
            self._send_response(conn, b"OKAY", b"")
        else:
            self._send_response(conn, b"ERRO", b"set gain failed")

    def _send_response(self, conn: socket.socket, code: bytes, payload: bytes) -> None:
        if len(code) != 4:
            raise ValueError("响应码必须为 4 字节")
        conn.sendall(_HEADER.pack(code, len(payload)) + payload)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hikvision camera server")
    parser.add_argument("--socket", default="/tmp/hikvision.sock", help="Unix domain socket 路径")
    parser.add_argument("--device-ip", required=True, help="相机 IP 地址")
    parser.add_argument("--width", type=int, default=None, help="期望输出宽度 (像素)")
    parser.add_argument("--height", type=int, default=None, help="期望输出高度 (像素)")
    parser.add_argument("--log", default="INFO", help="日志级别")
    parser.add_argument("--heartbeat", type=float, default=10.0, help="心跳监测周期，<=0 表示关闭")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    cfg = ServerConfig(
        socket_path=Path(args.socket),
        camera=HikCameraConfig(
            device_ip=args.device_ip,
            width=args.width if args.width else None,
            height=args.height if args.height else None,
        ),
        heartbeat_interval=args.heartbeat,
    )
    server = CameraServer(cfg)

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("收到信号 %s，准备停止", signum)
        server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("用户中断，退出")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
