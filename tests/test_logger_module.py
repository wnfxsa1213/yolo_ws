from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.test_utils import ensure_project_on_path

ensure_project_on_path()

from src.utils.logger import get_logger, reset_logger, setup_logger


@pytest.fixture()
def isolated_logger(tmp_path: Path):
    name = "test_logger"
    reset_logger(name)
    yield name, tmp_path
    reset_logger(name)
    logging.shutdown()


def test_logger_writes_to_file(isolated_logger):
    name, tmp_path = isolated_logger
    log_file = tmp_path / "logger.log"

    logger = setup_logger(
        name,
        level="INFO",
        log_dir=tmp_path,
        filename=log_file.name,
        console=False,
        use_color=False,
        force=True,
    )
    logger.info("INFO message from 老王")
    logger.error("ERROR message from 老王")

    logging.shutdown()

    assert log_file.exists(), "日志文件没生成，Logger 直接罢工"
    contents = log_file.read_text(encoding="utf-8")
    assert "INFO" in contents and "ERROR" in contents, "日志内容缺失，写入失败"


def test_logger_cache_returns_same_instance():
    reset_logger("cache_logger")

    first = setup_logger(
        "cache_logger",
        level="WARNING",
        console=False,
        file=False,
        force=True,
    )
    second = get_logger("cache_logger")

    assert first is second, "logger 缓存失效，get_logger 没拿到同一个实例"
    assert second.level == logging.WARNING, "logger 级别不对劲，force 设置没生效"
