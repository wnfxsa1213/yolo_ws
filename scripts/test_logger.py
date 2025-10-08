#!/usr/bin/env python3
"""
日志系统自检脚本。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.logger import build_log_file_name, setup_logger  # noqa: E402


def run(args: argparse.Namespace) -> None:
    log_name = args.name
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        log_name,
        level=args.level,
        log_dir=log_dir,
        filename=args.filename or f"{log_name}.log",
        rotation=args.rotation,
        max_bytes=args.max_bytes,
        backup_count=args.backup_count,
        force=True,
    )

    logger.debug("调试信息：彩色输出检查")
    logger.info("信息级别：写入文件与终端")
    logger.warning("警告级别：留意日志格式")
    try:
        raise ValueError("示例异常")
    except ValueError:
        logger.exception("异常级别：捕获堆栈")

    target_file = log_dir / (args.filename or f"{log_name}.log")
    if not target_file.exists():
        raise SystemExit(f"日志文件未生成: {target_file}")

    with target_file.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if not lines:
        raise SystemExit("日志文件为空，检查写入逻辑")
    if not any("信息级别" in line for line in lines):
        raise SystemExit("未找到 INFO 日志，请检查级别配置")

    print(f"✅ 日志测试通过，文件位于: {target_file}")
    if args.timestamp_name:
        ts_file = log_dir / build_log_file_name(log_name)
        ts_file.write_text("timestamp test", encoding="utf-8")
        print(f"✅ 生成时间戳文件名: {ts_file.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试日志模块")
    parser.add_argument("--name", default="system", help="logger 名称")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs"), help="日志目录")
    parser.add_argument("--level", default="DEBUG", help="日志级别")
    parser.add_argument("--filename", help="日志文件名称 (默认 name.log)")
    parser.add_argument(
        "--rotation", choices=["time", "size"], default="time", help="轮转模式"
    )
    parser.add_argument("--max-bytes", type=int, default=1_048_576, help="大小轮转阈值")
    parser.add_argument("--backup-count", type=int, default=3, help="共享保留数量")
    parser.add_argument(
        "--timestamp-name",
        action="store_true",
        help="额外测试时间戳文件名生成",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
