#!/usr/bin/env python3
"""
老王拿 Logger 做烟雾测试，确认输出来不得半点幺蛾子。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_utils import ensure_project_on_path


def main() -> None:
    ensure_project_on_path()
    exit_code = pytest.main(["-q", "tests/test_logger_module.py"])
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
