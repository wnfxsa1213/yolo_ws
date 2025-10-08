#!/usr/bin/env python3
"""
配置管理自检脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import ConfigManager, ConfigValidationError  # noqa: E402


def run(args: argparse.Namespace) -> None:
    cfg = ConfigManager(args.config, auto_reload=args.auto_reload)

    required = [
        "project.name",
        "project.version",
        "camera.config_path",
        "model.engine_path",
        "serial.port",
    ]

    try:
        cfg.validate(required)
    except ConfigValidationError as exc:
        raise SystemExit(f"配置验证失败: {exc}")

    summary = {
        "project": cfg.get("project.name"),
        "model": cfg.get("model.engine_path"),
        "serial": cfg.get("serial.port"),
        "log_file": cfg.get("logging.file_name", default="system.log"),
    }
    print("✅ 配置加载成功:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.ensure_paths:
        paths = cfg.ensure_paths(args.ensure_paths)
        for key, path in paths.items():
            print(f"✅ {key} -> {path}")

    if args.dump_all:
        data = cfg.as_dict(deep=True)
        dump_path = Path(args.dump_all)
        dump_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✅ 完整配置已导出至 {dump_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试配置模块")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config/system_config.yaml"),
        help="主配置文件路径",
    )
    parser.add_argument(
        "--ensure-paths",
        nargs="*",
        default=["paths.recordings_dir", "paths.detections_dir"],
        help="需要确保存在的路径字段",
    )
    parser.add_argument("--auto-reload", action="store_true", help="启用自动热加载")
    parser.add_argument("--dump-all", help="导出完整配置为 JSON 文件")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
