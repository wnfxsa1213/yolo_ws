"""
系统日志模块。

特性:
- 控制台彩色输出
- 文件输出带轮转
- 线程安全缓存，避免重复初始化
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union

from . import ensure_dir, get_timestamp

LOG_COLORS: Dict[int, str] = {
    logging.DEBUG: "\033[36m",  # 青色
    logging.INFO: "\033[32m",  # 绿色
    logging.WARNING: "\033[33m",  # 黄色
    logging.ERROR: "\033[31m",  # 红色
    logging.CRITICAL: "\033[41m",  # 红底
}
RESET_COLOR = "\033[0m"

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_CACHE_LOCK = threading.Lock()


class ColorFormatter(logging.Formatter):
    """终端彩色日志格式化器."""

    def __init__(self, fmt: str, datefmt: Optional[str], use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color and self._stream_supports_color()

    @staticmethod
    def _stream_supports_color() -> bool:
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        # Windows 10+ 默认支持 ANSI，其他平台直接返回 isatty 结果
        return is_tty or os.environ.get("TERM_PROGRAM") == "vscode"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self.use_color:
            color = LOG_COLORS.get(record.levelno)
            if color:
                return f"{color}{message}{RESET_COLOR}"
        return message


def _parse_log_level(level: Union[str, int, None]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized.isdigit():
            return int(normalized)
        if normalized in logging._nameToLevel:  # type: ignore[attr-defined]
            return logging._nameToLevel[normalized]  # type: ignore[attr-defined]
    raise ValueError(f"无法解析日志级别: {level}")


def setup_logger(
    name: str,
    level: Union[str, int, None] = None,
    *,
    log_dir: Union[str, Path] = "logs",
    filename: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    rotation: str = "time",
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 7,
    max_bytes: int = 10 * 1024 * 1024,
    use_color: bool = True,
    force: bool = False,
) -> logging.Logger:
    """
    初始化或获取命名 logger。

    Args:
        name: 日志器名称。
        level: 日志级别（字符串或数字）。
        log_dir: 日志目录。
        filename: 日志文件名称，默认 <name>.log。
        console: 是否输出到终端。
        file: 是否写入文件。
        rotation: 文件轮转模式: "time" 或 "size"。
        when: 时间轮转粒度 (TimedRotatingFileHandler)。
        interval: 时间轮转间隔。
        backup_count: 保留旧文件数量。
        max_bytes: 按大小轮转时的阈值。
        use_color: 控制台是否彩色输出。
        force: True 时重置已存在的 logger。
    """

    with _CACHE_LOCK:
        logger = logging.getLogger(name)
        if force or name not in _LOGGER_CACHE:
            if force:
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
            logger.setLevel(_parse_log_level(level))
            logger.propagate = False

            if console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(
                    ColorFormatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT, use_color)
                )
                logger.addHandler(console_handler)

            if file:
                log_path = ensure_dir(log_dir)
                log_filename = filename or f"{name}.log"
                file_path = log_path / log_filename

                if rotation == "size":
                    handler = RotatingFileHandler(
                        file_path,
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        encoding="utf-8",
                    )
                else:
                    handler = TimedRotatingFileHandler(
                        file_path,
                        when=when,
                        interval=interval,
                        backupCount=backup_count,
                        encoding="utf-8",
                        utc=False,
                    )
                handler.setFormatter(
                    logging.Formatter(
                        DEFAULT_LOG_FORMAT,
                        datefmt=DEFAULT_DATE_FORMAT,
                    )
                )
                logger.addHandler(handler)

            _LOGGER_CACHE[name] = logger

        else:
            if level is not None:
                logger.setLevel(_parse_log_level(level))

        return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取缓存中的 logger。

    Args:
        name: 日志器名称。
    """
    with _CACHE_LOCK:
        if name in _LOGGER_CACHE:
            return _LOGGER_CACHE[name]
        return setup_logger(name)


def build_log_file_name(base_name: str) -> str:
    """
    生成带时间戳的日志文件名。
    """
    timestamp = get_timestamp()
    return f"{base_name}_{timestamp}.log"


def reset_logger(name: str) -> None:
    """
    重置指定 logger，移除缓存和所有 handler。
    """
    with _CACHE_LOCK:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        _LOGGER_CACHE.pop(name, None)
