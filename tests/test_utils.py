"""
测试工具：统一把项目根和 src 加进 sys.path，省得到处写重复代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def ensure_project_on_path() -> None:
    """把项目根路径塞进 sys.path，避免导入失败这个SB问题。"""
    for path in (PROJECT_ROOT, SRC_DIR):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
