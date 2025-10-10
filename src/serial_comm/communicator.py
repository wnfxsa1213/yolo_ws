from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue
import logging
from typing import Callable, Optional

import serial
from serial.serialutil import SerialException

from .protocol import (
    FRAME_TYPE_STATUS,
    ProtocolDecoder,
    ProtocolEncoder,
    StatusPacket,
)

logger = logging.getLogger(__name__)

class SerialCommunicator:
    """Jetson↔STM32 串口总管，负责发命令、扯状态、保心跳。"""

    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baudrate: int = 460800,
        heartbeat_interval: float = 0.05,
        reconnect_interval: float = 1.0,
        read_timeout: float = 0.05,
        send_queue_size: int = 64,
        serial_factory: Optional[Callable[..., serial.Serial]] = None,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_interval = reconnect_interval
        self._encoder = ProtocolEncoder()
        self._decoder = ProtocolDecoder()
        self._send_queue: Queue[bytes] = Queue(maxsize=send_queue_size)
        self._status_lock = threading.Lock()
        self._latest_status: Optional[StatusPacket] = None
        self._last_status_ts = 0.0
        self._serial: Optional[serial.Serial] = None
        self._serial_factory = serial_factory
        self._stop_event = threading.Event()
        self._tx_thread: Optional[threading.Thread] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._last_tx_ts = 0.0
        self._metrics = {
            "tx_frames": 0,
            "rx_frames": 0,
            "crc_errors": 0,
            "reconnects": 0,
        }
        self._frame_callback: Optional[Callable[[DecodedFrame], None]] = None

    # --- 生命周期 ---
    def start(self) -> None:
        if self._tx_thread and self._tx_thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_serial()
        self._tx_thread = threading.Thread(target=self._tx_loop, name="serial-tx", daemon=True)
        self._rx_thread = threading.Thread(target=self._rx_loop, name="serial-rx", daemon=True)
        self._tx_thread.start()
        self._rx_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._tx_thread:
            self._tx_thread.join(timeout=1.0)
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        self._tx_thread = None
        self._rx_thread = None
        self._close_serial()

    # --- 命令接口 ---
    def send_command(self, pitch_deg: float, yaw_deg: float, laser_on: bool, heartbeat: bool = True) -> None:
        frame = self._encoder.encode_command(pitch_deg, yaw_deg, laser_on, heartbeat)
        self._enqueue_frame(frame)

    def send_heartbeat(self) -> None:
        self._enqueue_frame(self._encoder.encode_heartbeat())

    def request_status(self) -> None:
        self._enqueue_frame(self._encoder.encode_status_request())

    def get_latest_status(self) -> Optional[StatusPacket]:
        with self._status_lock:
            return self._latest_status

    def last_status_age(self) -> Optional[float]:
        with self._status_lock:
            if self._last_status_ts <= 0.0:
                return None
            return time.monotonic() - self._last_status_ts

    def is_connected(self) -> bool:
        serial_obj = self._serial
        return bool(serial_obj and serial_obj.is_open)

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def set_frame_callback(self, callback: Optional[Callable[[DecodedFrame], None]]) -> None:
        self._frame_callback = callback

    # --- 内部实现 ---
    def _enqueue_frame(self, frame: bytes) -> None:
        while not self._stop_event.is_set():
            try:
                self._send_queue.put(frame, timeout=0.01)
                return
            except Full:
                try:
                    self._send_queue.get_nowait()
                except Empty:
                    continue

    def _tx_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = None
            try:
                frame = self._send_queue.get(timeout=0.05)
            except Empty:
                pass

            now = time.monotonic()
            if frame is None and self._heartbeat_interval > 0:
                if now - self._last_tx_ts >= self._heartbeat_interval:
                    frame = self._encoder.encode_heartbeat()

            if frame is None:
                continue

            self._ensure_serial()
            serial_obj = self._serial
            if not serial_obj or not serial_obj.is_open:
                time.sleep(self._reconnect_interval)
                continue
            try:
                serial_obj.write(frame)
                serial_obj.flush()
                self._last_tx_ts = now
                self._metrics["tx_frames"] += 1
            except SerialException as exc:
                logger.warning("Serial error in TX loop: %s", exc)
                self._handle_disconnect()

    def _rx_loop(self) -> None:
        while not self._stop_event.is_set():
            serial_obj = self._serial
            if not serial_obj or not serial_obj.is_open:
                self._ensure_serial()
                time.sleep(self._reconnect_interval)
                continue
            try:
                data = serial_obj.read(128)
            except SerialException as exc:
                logger.warning("Serial error in RX loop: %s", exc)
                self._handle_disconnect()
                continue
            if not data:
                continue
            self._decoder.feed(data)
            self._metrics["crc_errors"] = self._decoder.crc_errors
            while True:
                frame = self._decoder.pop_frame()
                if frame is None:
                    break
                if frame.frame_type == FRAME_TYPE_STATUS and isinstance(frame.parsed, StatusPacket):
                    with self._status_lock:
                        self._latest_status = frame.parsed
                        self._last_status_ts = time.monotonic()
                    self._metrics["rx_frames"] += 1
                if self._frame_callback:
                    try:
                        self._frame_callback(frame)
                    except Exception as exc:
                        logger.debug("Frame callback raised: %s", exc)

    def _ensure_serial(self) -> None:
        if self._serial and self._serial.is_open:
            return
        factory = self._serial_factory or serial.Serial
        try:
            self._serial = factory(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._read_timeout,
            )
            self._last_tx_ts = time.monotonic()
        except SerialException:
            self._handle_disconnect()

    def _close_serial(self) -> None:
        if self._serial:
            try:
                self._serial.close()
            except SerialException:
                pass
            self._serial = None

    def _handle_disconnect(self) -> None:
        self._close_serial()
        self._metrics["reconnects"] += 1
        time.sleep(self._reconnect_interval)


    def __enter__(self) -> "SerialCommunicator":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


__all__ = ["SerialCommunicator"]
