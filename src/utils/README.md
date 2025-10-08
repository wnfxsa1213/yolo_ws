# 工具模块 (Utils Module)

## 模块职责

提供通用工具类和辅助函数。

## 待实现文件

### logger.py
- **日志系统**
  - 统一日志配置
  - 控制台输出 + 文件输出
  - 日志分级（DEBUG, INFO, WARNING, ERROR）
  - 日志轮转（按大小或时间）
  - 彩色输出（终端）

### config.py
- **ConfigManager 类**
  - 加载YAML配置文件
  - 配置验证（必要字段检查）
  - 配置热加载（可选）
  - 类型转换辅助函数
  - 默认配置合并

### profiler.py
- **PerformanceProfiler 类**
  - 函数执行时间测量（装饰器）
  - 统计信息收集（平均、最大、最小）
  - 性能报告生成
  - 实时监控（FPS计算）

- **MemoryMonitor 类**
  - GPU内存监控
  - CPU内存监控
  - 内存泄漏检测

## 辅助函数
- `get_timestamp()` - 获取时间戳
- `ensure_dir(path)` - 确保目录存在
- `parse_version(version_str)` - 版本号解析

## 依赖
- pyyaml
- pynvml (GPU监控)
- psutil (CPU监控)

## 状态
⏳ 待实现
