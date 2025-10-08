"""
性能分析工具。
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, Optional

try:
    import psutil  # type: ignore import
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml  # type: ignore import
except ImportError:  # pragma: no cover
    pynvml = None


class PerformanceProfiler:
    """性能计时器。"""

    def __init__(self) -> None:
        self._records: Dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, elapsed_ms: float) -> None:
        self._records[name].append(elapsed_ms)

    def measure(self, name: str) -> Callable:
        """装饰器模式测量函数执行时间。"""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000.0
                self.record(name, elapsed)
                return result

            wrapper.__name__ = getattr(func, "__name__", "wrapped")
            wrapper.__doc__ = getattr(func, "__doc__", None)
            wrapper.__qualname__ = getattr(func, "__qualname__", wrapper.__name__)
            return wrapper

        return decorator

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.record(name, elapsed)

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for key, values in self._records.items():
            if not values:
                continue
            result[key] = {
                "count": float(len(values)),
                "avg_ms": statistics.fmean(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        return result

    def report(self, printer: Optional[Callable[[str], None]] = None) -> None:
        printer = printer or print
        printer("===== 性能分析报告 =====")
        for key, stats in self.summary().items():
            printer(
                f"{key:24s}: "
                f"count={int(stats['count'])} "
                f"avg={stats['avg_ms']:.2f}ms "
                f"min={stats['min_ms']:.2f}ms "
                f"max={stats['max_ms']:.2f}ms"
            )


@dataclass
class FPSCounter:
    """FPS 统计器。"""

    window: int = 60
    _timestamps: Deque[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._timestamps = deque(maxlen=self.window)

    def update(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) <= 1:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class MemoryMonitor:
    """内存监控，优雅降级处理缺失依赖的情况。"""

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._gpu_available = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                pynvml.nvmlDeviceGetCount()
                self._gpu_available = True
            except Exception:
                self._gpu_available = False

    def cpu(self) -> Dict[str, float]:
        if psutil is None:
            raise RuntimeError("缺少依赖 psutil，无法获取 CPU 内存信息")
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total / (1024**2),
            "used_mb": (vm.total - vm.available) / (1024**2),
            "percent": vm.percent,
        }

    def gpu(self) -> Dict[str, float]:
        if not self._gpu_available:
            raise RuntimeError("无法获取 GPU 内存信息，确认是否安装 pynvml")
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total_mb": info.total / (1024**2),
            "used_mb": info.used / (1024**2),
            "free_mb": info.free / (1024**2),
            "percent": (info.used / info.total * 100.0) if info.total else 0.0,
        }

    def close(self) -> None:
        if self._gpu_available and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()
