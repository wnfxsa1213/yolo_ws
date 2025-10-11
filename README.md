# 智能云台追踪系统 - Jetson端

## 📋 项目信息

| 项目 | 内容 |
|------|------|
| **项目名称** | Gimbal Target Tracker System |
| **版本** | v1.0.0 |
| **平台** | Jetson Orin NX Super 16GB |
| **开发状态** | Phase 1 - 开发中 🚧 |
| **创建日期** | 2025-10 |

---

## 🎯 项目概述

基于Jetson边缘计算平台的智能武器站系统，实现实时目标检测、追踪和云台控制。

### 核心功能
- ✅ **实时目标检测**: YOLOv8-nano + TensorRT FP16 (7ms推理)
- ✅ **目标追踪**: ByteTrack算法（待实现）
- ✅ **云台控制**: 与STM32H750双向通信
- ✅ **遥控器接管**: ELRS接收机安全机制

### 技术亮点
- **高性能**: 142 FPS检测速度，<32ms端到端延迟
- **异构计算**: Python应用层 + C++ CUDA算法层
- **GPU优化**: 继承主环境GPU库（PyTorch 2.5.0, OpenCV 4.10.0）
- **实时通信**: 460800波特率串口 + CRSF协议

---

## 📁 项目结构

```
yolo_ws/
├── .venv/                      # 虚拟环境（--system-site-packages）
├── src/                        # 源代码目录
│   ├── vision/                # 视觉模块（相机接口）
│   ├── algorithms/            # C++算法模块（YOLO、追踪、坐标转换）
│   ├── serial_comm/           # 串口通信模块
│   ├── utils/                 # 工具模块（日志、配置、性能）
│   └── main.py                # 主程序入口
├── config/                     # 配置文件
│   ├── system_config.yaml     # 系统主配置
│   ├── camera_config.yaml     # 相机参数
│   └── camera_intrinsics.yaml # 相机内参（标定后生成）
├── scripts/                    # 工具脚本
│   ├── export_onnx.py         # 模型导出
│   ├── build_engine.py        # TensorRT构建
│   ├── test_camera.py         # 相机测试
│   ├── test_serial.py         # 串口测试
│   └── benchmark.py           # 性能测试
├── models/                     # 模型文件
│   └── yolov8n_fp16.engine    # TensorRT引擎（待生成）
├── tests/                      # 测试代码
├── logs/                       # 日志输出
├── docs/                       # 技术文档
│   ├── System_Architecture_V2.md      # 系统架构
│   ├── Jetson_Development.md          # Jetson开发指南
│   ├── ENVIRONMENT_SETUP.md           # 环境配置
│   ├── PHASE1_DEVELOPMENT_PLAN.md     # 第一阶段开发计划
│   └── ...
├── TASKLIST_PHASE1.md          # 第一阶段任务清单
├── pyproject.toml              # Python项目配置
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 环境要求

**硬件：**
- Jetson Orin NX Super 16GB
- 海康威视USB3.0工业相机（已更换为Aravis兼容的工业相机）
- STM32F407VET6开发板
- NVMe SSD 256GB+

**软件：**
- JetPack R36.4.4
- Python 3.10.12
- CUDA 12.6
- TensorRT 10.3.0
- PyTorch 2.5.0 (NVIDIA定制版)
- OpenCV 4.10.0 (CUDA版本)

### 安装步骤

#### 1. 创建虚拟环境
```bash
cd ~/yolo_ws

# ⚠️ 重要：必须使用 --system-site-packages
# 原因：继承主环境的GPU优化库
uv venv --python 3.10 --system-site-packages

# 激活环境
source .venv/bin/activate
```

#### 2. 安装项目依赖
```bash
# 使用pyproject.toml安装
uv sync

# 或手动安装（不要安装torch, opencv-python, numpy!）
uv pip install pyserial pyyaml ultralytics
```

#### 3. 验证GPU环境
```bash
python << 'EOF'
import torch
import cv2
import numpy as np

print(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"✓ OpenCV: {cv2.__version__} (CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0})")
print(f"✓ NumPy: {np.__version__}")
EOF
```

**预期输出：**
```
✓ PyTorch: 2.5.0a0+872d972e41.nv24.08 (CUDA: True)
✓ OpenCV: 4.10.0 (CUDA: True)
✓ NumPy: 1.26.4
```

#### 4. 编译C++模块（待第一阶段完成）
```bash
cd src/algorithms
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 输出：algorithms.cpython-310-aarch64-linux-gnu.so
```

#### 5. 准备模型文件（待第一阶段完成）
```bash
# 下载YOLOv8-nano预训练权重
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# GUI一键党可启动:
python scripts/model_tools_gui.py

# 导出ONNX
python scripts/export_onnx.py --weights yolov8n.pt --imgsz 640

# 构建TensorRT引擎
python scripts/build_engine.py --onnx yolov8n.onnx --fp16
```

---

## 🎮 运行

### 测试相机（第一阶段Sprint 2）
```bash
python scripts/test_camera.py --config config/camera_config.yaml
```

### 测试串口（第一阶段Sprint 3）
```bash
python scripts/test_serial.py --port /dev/ttyTHS1 --baudrate 460800
```

### 运行主程序（第一阶段Sprint 5）
```bash
python src/main.py --config config/system_config.yaml
# 需要实时预览时，确保 config/system_config.yaml 里的 debug.show_image: true 并系统已安装 python3-opencv
```

**预期输出：**
```
[INFO] 系统初始化...
[INFO] 相机连接成功: 1920x1080 @ 60fps
[INFO] YOLO引擎加载成功: yolov8n_fp16.engine
[INFO] 串口连接成功: /dev/ttyTHS1 @ 460800
[INFO] 主循环启动...
[INFO] FPS: 35.2, Detections: 2, Latency: 32ms
[INFO] Target: pitch=5.2°, yaw=-12.3°, distance=0cm
[INFO] H750反馈: mode=JETSON_CONTROL, temp=42°C
```

---

## 📊 性能指标

### Jetson Orin NX Super 16GB

| 指标 | 数值 | 说明 |
|------|------|------|
| **YOLO推理** | ~7ms | YOLOv8-nano FP16 @ 640x640 |
| **检测帧率** | 142 FPS | 仅推理，无后处理 |
| **端到端延迟** | ~32ms | 采集→检测→串口发送 |
| **CPU占用** | ~25% | 8核仅用4核 |
| **GPU占用** | ~45% | 1024 CUDA cores |
| **内存占用** | ~1.8GB / 16GB | 含模型和缓冲 |
| **功耗** | ~12W | @ 15W功耗模式 |

---

## 🧪 测试

### 运行单元测试
```bash
# 所有测试（推荐用 python -m pytest 避免 PATH 幻觉）
python -m pytest tests/ -v

# 核心单元测试
python -m pytest tests/test_logger_module.py -q
python -m pytest tests/test_config_manager.py -q

# 旧的硬件相关测试
python -m pytest tests/test_camera.py -v
python -m pytest tests/test_serial.py -v

# 脚本包装（内部会自动调用 pytest）
python scripts/test_logger.py
python scripts/test_config.py
```

### 性能基准测试
```bash
python scripts/benchmark.py --engine models/yolov8n_fp16.engine
```

---

## 📖 文档

完整的技术文档位于 `docs/` 目录：

| 文档 | 描述 |
|------|------|
| [System_Architecture_V2.md](docs/System_Architecture_V2.md) | 系统整体架构设计 |
| [Jetson_Development.md](docs/Jetson_Development.md) | Jetson开发完整指南 |
| [ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) | 环境配置详细说明 |
| [PHASE1_DEVELOPMENT_PLAN.md](docs/PHASE1_DEVELOPMENT_PLAN.md) | 第一阶段开发计划 |
| [F407_Development_V2.md](docs/F407_Development_V2.md) | STM32F407开发文档 |
| [CRSF_Protocol_Reference.md](docs/CRSF_Protocol_Reference.md) | CRSF协议参考 |
| [Quick_Start_Guide.md](docs/Quick_Start_Guide.md) | 快速开始指南 |

---

## 🛠️ 开发

### 当前阶段：Phase 1 - 基础框架

**目标：** 搭建完整基础框架，实现核心检测追踪流程

**进度：** 查看 [TASKLIST_PHASE1.md](TASKLIST_PHASE1.md)

**时间线：** 2-3周

### 开发规范

**代码风格：**
- Python: PEP 8 + Black格式化
- C++: Google C++ Style Guide

**Git提交：**
```
<type>(<scope>): <subject>

示例:
feat(camera): 实现海康相机驱动
fix(serial): 修复CRC8校验错误
```

**测试要求：**
- 单元测试覆盖率 ≥60%
- 关键函数必须有性能测试

---

## ⚠️ 常见问题

### 1. PyTorch显示无CUDA支持

**原因：** 虚拟环境未使用 `--system-site-packages`

**解决：**
```bash
rm -rf .venv
uv venv --python 3.10 --system-site-packages
source .venv/bin/activate
```

### 2. OpenCV缺少CUDA模块

**原因：** 安装了PyPI的opencv-python（CPU版本）

**解决：**
```bash
uv pip uninstall opencv-python opencv-contrib-python
# 使用系统的OpenCV 4.10.0
```

### 3. 串口无法打开

**原因：** 权限不足

**解决：**
```bash
sudo usermod -aG dialout $USER
# 重新登录
```

### 4. TensorRT引擎加载失败

**原因：** TensorRT版本不匹配

**解决：**
```bash
# 检查版本
dpkg -l | grep tensorrt
# 应为 TensorRT 10.3.0

# 重新构建引擎
python scripts/build_engine.py --onnx model.onnx --fp16
```

---

## 📞 联系与支持

**开发者：** 幽浮喵 (浮浮酱) ฅ'ω'ฅ

**问题反馈：**
- 查看 [docs/](docs/) 目录完整文档
- 查看 [TASKLIST_PHASE1.md](TASKLIST_PHASE1.md) 开发进度
- 参考 [ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) 环境配置

---

## 📜 许可证

本项目为内部开发项目。

---

## 🎉 致谢

- NVIDIA Jetson团队
- Ultralytics YOLOv8
- ExpressLRS社区
- 海康威视MVS SDK

---

**最后更新：** 2025-10-08
**项目状态：** 🚧 Phase 1 开发中
