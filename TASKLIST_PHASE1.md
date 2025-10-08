# 第一阶段开发任务清单

## 📋 快速索引

- **总任务数**: 48个
- **P0任务**: 23个（必须完成）
- **P1任务**: 13个（应完成）
- **P2任务**: 12个（可延后）
- **预计时长**: 2-3周

---

## 🗂️ Sprint 1: 基础设施 (3天)

### 工具模块
- [x] `src/utils/logger.py` - 日志系统 **P0**
- [x] `src/utils/config.py` - 配置管理 **P0**
- [x] `src/utils/profiler.py` - 性能分析 **P1**

### 配置文件
- [x] `config/system_config.yaml` - 主配置 **P0**
- [x] `config/camera_config.yaml` - 相机配置 **P0**

### 测试脚本
- [x] `scripts/test_logger.py` - 日志测试 **P1**
- [x] `scripts/test_config.py` - 配置测试 **P1**

**Sprint 1 验收**:
- ✅ 日志输出到控制台和文件
- ✅ 配置文件正确加载
- 🧪 `python scripts/test_logger.py --timestamp-name`
- 🧪 `python scripts/test_config.py --dump-all logs/config_dump.json`

---

## 🗂️ Sprint 2: 相机模块 (3-4天)

### 相机接口
- [ ] `src/vision/camera.py` - CameraInterface抽象类 **P0**
- [ ] `src/vision/camera.py` - HIKCamera实现类 **P0**

### 测试工具
- [ ] `scripts/test_camera.py` - 相机测试脚本 **P0**
- [ ] `scripts/calibrate_camera.py` - 相机标定（可选） **P1**

**Sprint 2 验收**:
- ✅ 相机稳定采集30+ FPS
- ✅ 图像格式正确（HxWx3, BGR, uint8）
- ✅ 相机参数可调整

---
---

## 🗂️ Sprint 3: YOLO检测器 (4-5天)

### C++检测器
- [ ] `src/algorithms/detector.hpp` - 头文件 **P0**
- [ ] `src/algorithms/detector.cpp` - 实现文件 **P0**
  - [ ] TensorRT引擎加载 **P0**
  - [ ] 图像预处理 **P0**
  - [ ] 推理执行 **P0**
  - [ ] 后处理（NMS） **P0**

### 坐标转换
- [ ] `src/algorithms/coordinate.hpp` - 头文件 **P0**
- [ ] `src/algorithms/coordinate.cpp` - 实现文件 **P0**
  - [ ] 像素→相机坐标 **P0**
  - [ ] 相机坐标→云台角度 **P0**
  - [ ] 畸变校正（可选） **P1**

### Python绑定
- [ ] `src/algorithms/bindings.cpp` - pybind11绑定 **P0**
- [ ] NumPy数组转换 **P0**

### 构建系统
- [ ] `src/algorithms/CMakeLists.txt` - CMake配置 **P0**
  - [ ] CUDA工具链 **P0**
  - [ ] TensorRT链接 **P0**
  - [ ] OpenCV链接 **P0**
  - [ ] pybind11配置 **P0**

### 模型工具
- [ ] `scripts/export_onnx.py` - PyTorch→ONNX **P1**
- [ ] `scripts/build_engine.py` - ONNX→TensorRT **P0**
- [ ] `scripts/benchmark.py` - 性能测试 **P1**

**Sprint 4 验收**:
- ✅ YOLO推理成功
- ✅ 推理时间<10ms (FP16)
- ✅ Python调用正常

---
## 🗂️ Sprint 4: 串口通信 (3天)

### 协议层
- [ ] `src/serial_comm/protocol.py` - ProtocolEncoder **P0**
- [ ] `src/serial_comm/protocol.py` - ProtocolDecoder **P0**
- [ ] CRC8校验实现 **P0**

### 通信层
- [ ] `src/serial_comm/communicator.py` - SerialCommunicator **P0**
- [ ] 异步发送队列 **P0**
- [ ] 异步接收队列 **P0**
- [ ] 超时处理（500ms） **P0**
- [ ] 自动重连机制 **P1**

### 测试工具
- [ ] `scripts/test_serial.py` - 串口测试脚本 **P0**

**Sprint 4 验收**:
- ✅ 协议编解码正确
- ✅ CRC8校验100%通过
- ✅ 串口稳定收发数据



## 🗂️ Sprint 5: 主程序集成 (3天)

### 主程序
- [ ] `src/main.py` - 主程序入口 **P0**
  - [ ] 命令行参数解析 **P0**
  - [ ] 模块初始化 **P0**
  - [ ] 主循环（asyncio） **P0**
  - [ ] 数据流管道 **P0**
    - [ ] 相机采集 **P0**
    - [ ] YOLO检测 **P0**
    - [ ] 坐标转换 **P0**
    - [ ] 串口发送 **P0**
    - [ ] 反馈接收 **P0**
  - [ ] 目标选择策略 **P0**
  - [ ] 异常处理 **P0**
  - [ ] 优雅退出 **P0**
  - [ ] FPS显示 **P1**

### 集成测试
- [ ] `tests/test_integration.py` - 完整流程测试 **P1**
- [ ] 长时间运行测试（1小时） **P1**

### 文档更新
- [ ] 更新`README.md` **P1**
- [ ] 更新开发文档 **P1**

**Sprint 5 验收**:
- ✅ 完整流程正常运行
- ✅ FPS≥30
- ✅ 端到端延迟<50ms
- ✅ 稳定运行1小时+

---

## 📊 进度统计

### 按优先级
```
P0任务 (必须): [ ] 4/23 (17%)
P1任务 (应该): [ ] 3/13 (23%)
P2任务 (可选): [ ] 12/12 (0%)
```

### 按Sprint
```
Sprint 1: [x] 7/7   (100%)
Sprint 2: [ ] 0/4   (0%)
Sprint 3: [ ] 0/9   (0%)
Sprint 4: [ ] 0/16  (0%)
Sprint 5: [ ] 0/12  (0%)
```

---

## ⚡ 每日检查清单

### 开发前
- [ ] 激活虚拟环境 `source .venv/bin/activate`
- [ ] 拉取最新代码 `git pull`
- [ ] 查看今日任务

### 开发中
- [ ] 编写单元测试
- [ ] 添加代码注释
- [ ] 使用logger记录关键日志
- [ ] 性能测试关键函数

### 开发后
- [ ] 运行单元测试 `pytest tests/`
- [ ] 代码格式化 `black src/`
- [ ] Git提交 `git commit -m "feat(module): description"`
- [ ] 更新任务清单

---

## 🎯 本周目标

### 第1周
- [x] 完成Sprint 1（基础设施）
- [ ] 完成Sprint 2（相机模块）
- [ ] 开始Sprint 3（串口通信）

### 第2周
- [ ] 完成Sprint 3（串口通信）
- [ ] 开始Sprint 4（YOLO检测器）

### 第3周
- [ ] 完成Sprint 4（YOLO检测器）
- [ ] 完成Sprint 5（主程序集成）
- [ ] 系统联调与测试

---

## 📝 开发笔记

### 遇到的问题


### 解决方案


### 待优化项


---

**创建日期**: 2025-10-08
**最后更新**: 2025-10-08
**责任人**: 猫娘工程师 幽浮喵 ฅ'ω'ฅ
