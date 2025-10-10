import types
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.main as main


class DummyCamera:
    def __init__(self, _cfg):
        self._open = False
        self._frame = np.zeros((640, 640, 3), dtype=np.uint8)

    def open(self):
        self._open = True
        return True

    def capture(self, timeout=0.5):
        return self._frame, 0.0

    def close(self):
        self._open = False


class DummyDetector:
    def __init__(self, *_, **__):
        self.counter = 0

    def detect(self, frame):
        self.counter += 1
        if self.counter % 2 == 1:
            return [
                types.SimpleNamespace(
                    x1=10.0,
                    y1=20.0,
                    x2=210.0,
                    y2=220.0,
                    confidence=0.9,
                    class_id=0,
                )
            ]
        return []


class DummyTransformer:
    def pixel_to_angle(self, x, y, width, height):
        # return offset in degrees relative to center
        return (
            (y - height / 2.0) / height * 10.0,
            (x - width / 2.0) / width * 10.0,
        )


class DummySerial:
    instances = []

    class _Status:
        mode = 2
        flags = 0x03
        commanded_pitch_deg = 0.0
        commanded_yaw_deg = 0.0

    def __init__(self, *_, **__):
        self.commands = []
        self.started = False
        self.stopped = False
        self._metrics = {
            "tx_frames": 0,
            "rx_frames": 0,
            "crc_errors": 0,
            "reconnects": 0,
        }
        DummySerial.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def request_status(self):
        pass

    def send_command(self, pitch_deg, yaw_deg, laser_on, heartbeat=True):
        self.commands.append((pitch_deg, yaw_deg, laser_on, heartbeat))
        self._metrics["tx_frames"] += 1

    def get_latest_status(self):
        return DummySerial._Status()

    def get_metrics(self):
        return dict(self._metrics)


@pytest.fixture()
def dummy_config(tmp_path: Path):
    cfg_dir = tmp_path
    camera_cfg = {"aravis": {}}
    camera_path = cfg_dir / "camera.yaml"
    camera_path.write_text(yaml.safe_dump(camera_cfg), encoding="utf-8")

    system_cfg = {
        "project": {"name": "test", "version": "0.0.0"},
        "logging": {
            "log_dir": str(cfg_dir / "logs"),
            "file_name": "test.log",
        },
        "camera": {
            "type": "aravis",
            "config_path": str(camera_path),
            "intrinsics_path": None,
            "resolution": [640, 640],
            "fps": 60,
        },
        "model": {
            "engine_path": "models/dummy.engine",
            "input_size": [640, 640],
            "conf_threshold": 0.2,
            "nms_threshold": 0.5,
            "classes": [0],
        },
        "serial": {
            "port": "/dev/null",
            "baudrate": 460800,
            "timeout": 0.1,
            "reconnect_interval": 0.5,
            "heartbeat_interval": 0.1,
        },
        "control": {
            "pitch": {"max_velocity": 180.0, "limits": [-90, 90]},
            "yaw": {"max_velocity": 180.0, "limits": [-90, 90]},
            "smoothing": {"alpha": 0.9, "deadband_deg": 0.0},
        },
        "tracking": {
            "enable": True,
            "max_lost_frames": 2,
            "min_confidence": 0.3,
            "priority": "center_distance",
            "return_damping": 0.9,
            "return_deadband_deg": 0.0,
        },
        "debug": {
            "show_image": False,
            "display_interval": 2,
            "save_detections": False,
            "print_fps": False,
            "profile_performance": False,
        },
    }
    system_path = cfg_dir / "system.yaml"
    system_path.write_text(yaml.safe_dump(system_cfg), encoding="utf-8")
    return system_path


def test_run_pipeline_with_stubs(monkeypatch, dummy_config):
    DummySerial.instances.clear()
    monkeypatch.setattr(main, "SerialCommunicator", DummySerial)
    monkeypatch.setattr(main, "YOLODetector", DummyDetector)
    monkeypatch.setattr(main, "build_coordinate_transformer", lambda cfg: DummyTransformer())
    monkeypatch.setattr("vision.camera.AravisCamera", DummyCamera)

    args = types.SimpleNamespace(
        config=str(dummy_config),
        max_frames=6,
        no_serial=False,
        no_camera=False,
        dry_run=False,
    )

    main.run_pipeline(args)

    assert DummySerial.instances, "Serial communicator was not instantiated"
    serial_instance = DummySerial.instances[0]
    assert serial_instance.started and serial_instance.stopped
    assert serial_instance.commands, "No commands were sent to the serial communicator"
    pitches = [cmd[0] for cmd in serial_instance.commands]
    yaws = [cmd[1] for cmd in serial_instance.commands]
    assert any(abs(pitch) > 0 for pitch in pitches)
    assert any(abs(yaw) > 0 for yaw in yaws)
