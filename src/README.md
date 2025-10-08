# Jetson端源码目录

## 目录结构

```
src/
├── vision/                 # 视觉模块（相机接口）
│   ├── __init__.py
│   ├── camera.py          # 相机抽象接口 + 海康实现
│   └── README.md
│
├── algorithms/            # C++算法模块（高性能计算）
│   ├── detector.hpp       # YOLO检测器声明
│   ├── detector.cpp       # YOLO检测器实现
│   ├── tracker.hpp        # ByteTrack追踪器声明
│   ├── tracker.cpp        # ByteTrack追踪器实现
│   ├── coordinate.hpp     # 坐标转换声明
│   ├── coordinate.cpp     # 坐标转换实现
│   ├── bindings.cpp       # pybind11 Python绑定
│   ├── CMakeLists.txt     # CMake构建脚本
│   └── README.md
│
├── serial_comm/           # 串口通信模块
│   ├── __init__.py
│   ├── protocol.py        # 协议编解码器
│   ├── communicator.py    # 串口通信管理
│   └── README.md
│
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── logger.py          # 日志系统
│   ├── config.py          # 配置管理
│   ├── profiler.py        # 性能分析
│   └── README.md
│
├── main.py                # 主程序入口
└── README.md              # 本文件
```

## 主程序 (main.py)

### 职责
- 系统初始化（日志、配置、相机、检测器、串口）
- 主循环协调（asyncio事件循环）
- 模块间数据流管理
- 异常处理和优雅退出

### 数据流
```
相机采集 → YOLO检测 → ByteTrack追踪 → 坐标转换 → 串口发送
    ↓           ↓            ↓             ↓            ↓
  图像帧     检测框列表   追踪ID+框    pitch/yaw    H750反馈
```

### 主循环伪代码
```python
async def main():
    # 1. 初始化所有模块
    camera = HIKCamera()
    detector = YOLODetector(engine_path)
    tracker = ByteTracker()
    coord_trans = CoordinateTransformer(intrinsics)
    serial_comm = SerialCommunicator(port, baudrate)

    # 2. 主循环
    while is_running:
        # 采集图像
        frame, timestamp = camera.capture()

        # 目标检测
        detections = detector.detect(frame)

        # 目标追踪
        tracks = tracker.update(detections)

        # 选择目标（距离中心最近）
        target = select_target(tracks)

        if target:
            # 坐标转换
            pitch, yaw = coord_trans.pixel_to_angle(
                target.center_x, target.center_y
            )

            # 发送到H750
            await serial_comm.send_target(True, pitch, yaw, 0)
        else:
            # 无目标
            await serial_comm.send_target(False, 0, 0, 0)

        # 接收反馈（非阻塞）
        feedback = await serial_comm.receive_feedback()
        if feedback:
            logger.debug(f"H750状态: {feedback['mode']}")

    # 3. 清理资源
    camera.close()
    serial_comm.close()
```

## 编译与运行

### 编译C++模块
```bash
cd src/algorithms
mkdir -p build && cd build
cmake ..
make -j$(nproc)
# 输出: algorithms.cpython-310-aarch64-linux-gnu.so
```

### 运行主程序
```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行主程序
python src/main.py --config config/system_config.yaml

# 或使用uv run
uv run python src/main.py --config config/system_config.yaml
```

## 开发状态

| 模块 | 状态 | 优先级 |
|------|------|--------|
| vision/camera.py | ⏳ 待实现 | P0 |
| serial_comm/protocol.py | ⏳ 待实现 | P0 |
| serial_comm/communicator.py | ⏳ 待实现 | P0 |
| utils/logger.py | ⏳ 待实现 | P0 |
| utils/config.py | ⏳ 待实现 | P0 |
| algorithms/detector.cpp | ⏳ 待实现 | P1 |
| algorithms/coordinate.cpp | ⏳ 待实现 | P1 |
| algorithms/tracker.cpp | ⏳ 待实现 | P2 |
| main.py | ⏳ 待实现 | P1 |
| utils/profiler.py | ⏳ 待实现 | P2 |

**优先级说明：**
- P0: 基础设施，第一阶段必须实现
- P1: 核心功能，第一阶段完成主体
- P2: 增强功能，第二阶段实现
