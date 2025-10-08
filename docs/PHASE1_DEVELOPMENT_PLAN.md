# Jetson端第一阶段开发计划

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **阶段名称** | Phase 1 - 基础框架与核心功能 |
| **目标** | 搭建完整的基础框架，实现核心检测追踪流程 |
| **预计时长** | 2-3周 |
| **创建日期** | 2025-10-08 |
| **作者** | 幽浮喵 (浮浮酱) ฅ'ω'ฅ |

---

## 🎯 第一阶段目标

### 核心目标
1. ✅ **基础设施完备**：日志、配置、工具类可用
2. ✅ **相机采集正常**：海康相机稳定采集图像
3. ✅ **串口通信稳定**：与H750双向通信正常
4. ✅ **检测功能就绪**：YOLO检测器推理成功
5. ✅ **基础流程打通**：采集 → 检测 → 串口发送的完整链路

### 非目标（留待第二阶段）
- ❌ ByteTrack追踪器（P2优先级）
- ❌ 性能优化（GPU预处理、CUDA流、多线程）
- ❌ Web监控界面
- ❌ 数据记录与回放

---

## 📅 开发计划 (分5个Sprint)

### Sprint 1: 基础设施搭建 (3天)
**目标：** 完成开发基础设施

#### 任务清单

**1.1 工具模块开发**
- [ ] `src/utils/logger.py` - 日志系统
  - 实现彩色控制台输出
  - 实现文件日志轮转
  - 配置不同级别的日志格式
  - 测试：输出不同级别日志到控制台和文件

- [ ] `src/utils/config.py` - 配置管理器
  - 实现YAML配置加载
  - 实现配置字段验证
  - 实现嵌套配置访问（点号分隔）
  - 测试：加载示例配置文件

- [ ] `src/utils/profiler.py` - 性能分析器
  - 实现函数执行时间装饰器
  - 实现统计信息收集（平均、最大、最小）
  - 实现性能报告生成
  - 测试：测量示例函数的执行时间

**1.2 配置文件创建**
- [ ] `config/system_config.yaml` - 主配置文件
  - 定义所有配置项结构
  - 填写开发环境默认值
  - 添加注释说明每个配置项

- [ ] `config/camera_config.yaml` - 相机配置
  - 定义海康相机参数
  - 设置适合开发的初始参数

**1.3 测试脚本**
- [ ] `scripts/test_logger.py` - 测试日志系统
- [ ] `scripts/test_config.py` - 测试配置加载

**验收标准：**
- ✅ 日志能正确输出到控制台和文件
- ✅ 配置文件能正确加载并访问
- ✅ 性能分析器能测量函数执行时间

---

### Sprint 2: 相机模块开发 (3-4天)
**目标：** 完成相机接口和海康相机驱动

#### 任务清单

**2.1 相机接口定义**
- [ ] `src/vision/camera.py` - 抽象接口
  - 定义`CameraInterface`抽象基类
  - 定义`open()`, `close()`, `capture()`, `get_intrinsics()`方法
  - 添加完整的docstring文档

**2.2 海康相机实现**
- [ ] `src/vision/camera.py` - HIKCamera类
  - 实现相机初始化（使用MVS SDK）
  - 实现图像采集（连续采集模式）
  - 实现参数配置（曝光、增益等）
  - 实现异常处理（相机断开、采集超时）
  - 添加日志记录（使用logger）

**2.3 相机测试脚本**
- [ ] `scripts/test_camera.py` - 相机测试工具
  - 检测相机连接状态
  - 采集并显示图像
  - 测试不同参数配置
  - 测试帧率性能
  - 保存测试图像

**2.4 相机标定（可选）**
- [ ] `scripts/calibrate_camera.py` - 相机标定工具
  - 实时预览和角点检测
  - 采集标定图像
  - 计算内参矩阵
  - 保存到`config/camera_intrinsics.yaml`

**验收标准：**
- ✅ 相机能稳定采集图像（30+ FPS）
- ✅ 图像格式正确（BGR, uint8, HxWx3）
- ✅ 相机参数能动态调整
- ✅ 异常情况能正确处理

---

### Sprint 3: 串口通信开发 (3天)
**目标：** 完成与H750的双向通信

#### 任务清单

**3.1 协议编解码器**
- [ ] `src/serial_comm/protocol.py` - ProtocolEncoder类
  - 实现`encode_target_data()`方法
  - 实现`encode_heartbeat()`方法
  - 实现CRC8校验计算
  - 测试：验证编码输出格式

- [ ] `src/serial_comm/protocol.py` - ProtocolDecoder类
  - 实现状态机解析（等待帧头 → 接收 → 校验）
  - 实现`feed()`方法（喂入数据）
  - 实现`get_decoded()`方法（获取完整帧）
  - 实现CRC8校验验证
  - 测试：编码后解码，验证数据一致性

**3.2 串口通信管理器**
- [ ] `src/serial_comm/communicator.py` - SerialCommunicator类
  - 实现串口初始化（pyserial）
  - 实现异步发送队列（asyncio.Queue）
  - 实现异步接收队列
  - 实现`send_target()`异步方法
  - 实现`receive_feedback()`异步方法
  - 实现超时处理（500ms）
  - 实现自动重连机制
  - 添加日志记录

**3.3 串口测试脚本**
- [ ] `scripts/test_serial.py` - 串口测试工具
  - 测试串口连接
  - 测试协议编解码
  - 测试CRC8校验
  - 测试发送/接收延迟
  - 回环测试（如有条件）

**验收标准：**
- ✅ 协议编解码正确无误
- ✅ CRC8校验100%通过
- ✅ 串口能稳定收发数据
- ✅ 发送延迟 <5ms

---

### Sprint 4: YOLO检测器开发 (4-5天)
**目标：** 完成C++ YOLO检测器和Python绑定

#### 任务清单

**4.1 C++检测器实现**
- [ ] `src/algorithms/detector.hpp` - 头文件
  - 定义`YOLODetector`类接口
  - 定义数据结构（Detection）
  - 添加完整注释

- [ ] `src/algorithms/detector.cpp` - 实现文件
  - 实现TensorRT引擎加载
  - 实现图像预处理（Resize, Normalize, HWC→CHW）
  - 实现推理执行（execute_async_v2）
  - 实现后处理（NMS, 置信度过滤）
  - 实现CUDA内存管理
  - 添加详细日志

**4.2 坐标转换实现**
- [ ] `src/algorithms/coordinate.hpp` - 头文件
  - 定义`CoordinateTransformer`类接口

- [ ] `src/algorithms/coordinate.cpp` - 实现文件
  - 实现像素坐标 → 相机坐标转换
  - 实现相机坐标 → 云台角度转换
  - 实现畸变校正（可选）
  - 加载相机内参

**4.3 Python绑定**
- [ ] `src/algorithms/bindings.cpp` - pybind11绑定
  - 绑定`YOLODetector`类
  - 绑定`CoordinateTransformer`类
  - 实现NumPy数组自动转换
  - 测试Python调用

**4.4 CMake构建配置**
- [ ] `src/algorithms/CMakeLists.txt` - 构建脚本
  - 配置CUDA工具链
  - 链接TensorRT库
  - 链接OpenCV库
  - 配置pybind11
  - 设置编译选项（-O3, -std=c++17）

**4.5 模型转换脚本**
- [ ] `scripts/export_onnx.py` - PyTorch → ONNX
  - 加载YOLOv8模型
  - 导出ONNX格式
  - 验证输出

- [ ] `scripts/build_engine.py` - ONNX → TensorRT
  - 加载ONNX模型
  - 配置TensorRT（FP16）
  - 构建引擎
  - 保存`.engine`文件

**4.6 性能测试**
- [ ] `scripts/benchmark.py` - 性能基准测试
  - 测试YOLO推理速度
  - 测试GPU利用率
  - 测试显存占用
  - 生成性能报告

**验收标准：**
- ✅ YOLO检测器能成功推理
- ✅ 推理时间 <10ms (FP16)
- ✅ 检测结果准确（mAP >0.4）
- ✅ Python能正常调用C++模块

---

### Sprint 5: 主程序集成 (3天)
**目标：** 打通完整数据流，实现基础功能

#### 任务清单

**5.1 主程序开发**
- [ ] `src/main.py` - 主程序入口
  - 实现命令行参数解析
  - 实现模块初始化（logger, config, camera, detector, serial）
  - 实现主循环（asyncio事件循环）
  - 实现数据流管道
    - 相机采集
    - YOLO检测
    - 坐标转换
    - 串口发送
    - 反馈接收
  - 实现目标选择策略（距离中心最近）
  - 实现异常处理
  - 实现优雅退出（Ctrl+C）
  - 添加性能监控（FPS显示）

**5.2 集成测试**
- [ ] `tests/test_integration.py` - 集成测试
  - 测试完整数据流
  - 测试模块间通信
  - 测试异常恢复
  - 测试长时间运行（1小时）

**5.3 文档更新**
- [ ] 更新`README.md` - 添加运行说明
- [ ] 更新`docs/Jetson_Development.md` - 添加实际开发经验

**验收标准：**
- ✅ 完整流程能正常运行
- ✅ FPS ≥30
- ✅ 端到端延迟 <50ms
- ✅ 无内存泄漏
- ✅ 能稳定运行1小时以上

---

## 📊 任务优先级矩阵

### P0 (最高优先级 - 必须完成)
```
Sprint 1: 基础设施
  ├─ logger.py
  ├─ config.py
  └─ system_config.yaml

Sprint 2: 相机模块
  ├─ camera.py (CameraInterface + HIKCamera)
  └─ test_camera.py

Sprint 3: 串口通信
  ├─ protocol.py (Encoder + Decoder)
  ├─ communicator.py
  └─ test_serial.py

Sprint 4: YOLO检测器
  ├─ detector.cpp/hpp
  ├─ coordinate.cpp/hpp
  ├─ bindings.cpp
  ├─ CMakeLists.txt
  └─ build_engine.py

Sprint 5: 主程序
  └─ main.py
```

### P1 (高优先级 - 应完成)
```
Sprint 1:
  └─ profiler.py

Sprint 2:
  └─ calibrate_camera.py

Sprint 4:
  ├─ export_onnx.py
  └─ benchmark.py

Sprint 5:
  └─ test_integration.py
```

### P2 (中优先级 - 可延后)
```
追踪器模块 (tracker.cpp)
性能优化 (GPU预处理, CUDA流)
Web监控界面
数据记录功能
```

---

## 🔧 技术栈清单

### Python依赖
```
核心库 (系统版本，不可安装):
- numpy==1.26.4
- opencv-python==4.10.0
- torch==2.5.0

项目依赖 (虚拟环境安装):
- pyserial>=3.5
- pyyaml>=6.0
- ultralytics>=8.0.0
- Pillow>=10.0.0
- pytest>=7.4.0 (开发依赖)
```

### C++依赖
```
- CUDA 12.6
- TensorRT 10.3.0
- OpenCV 4.10.0 (CUDA版本)
- pybind11
- Eigen3 (可选)
```

### 硬件要求
```
- Jetson Orin NX Super 16GB
- 海康威视工业相机 (USB3.0)
- STM32H750开发板（串口连接）
- NVMe SSD 256GB+
```

---

## 📝 开发规范

### 代码风格
- **Python**: PEP 8 + Black格式化
- **C++**: Google C++ Style Guide
- **注释**: 中英文混合，关键逻辑必须注释
- **文档字符串**: 所有公共API必须有docstring

### Git提交规范
```
格式: <type>(<scope>): <subject>

type类型:
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- test: 测试相关
- chore: 构建/工具链

示例:
feat(camera): 实现海康相机驱动
fix(serial): 修复CRC8校验错误
docs(readme): 更新安装说明
```

### 测试要求
- **单元测试**: 覆盖率 ≥60%
- **集成测试**: 主要数据流必须测试
- **性能测试**: 关键函数必须有benchmark

---

## ⚠️ 风险与依赖

### 技术风险
| 风险项 | 影响 | 概率 | 应对措施 |
|--------|------|------|---------|
| 海康SDK兼容性 | 高 | 中 | 提前测试SDK，准备备用方案（V4L2） |
| TensorRT版本冲突 | 高 | 低 | 使用系统TensorRT 10.3.0 |
| 串口通信不稳定 | 中 | 中 | 添加重试和校验机制 |
| 性能不达标 | 中 | 低 | FP16优化，预留INT8量化方案 |

### 外部依赖
- **H750固件**: 需要H750端先实现串口协议
- **YOLO模型**: 需要预训练的YOLOv8-nano模型
- **相机**: 需要海康相机和SDK

---

## 📈 里程碑与验收

### Milestone 1: 基础设施就绪 (第1周)
**标志：**
- ✅ 日志系统工作正常
- ✅ 配置文件能正确加载
- ✅ 相机能稳定采集图像

**验收标准：**
```bash
# 测试日志
python scripts/test_logger.py  # 通过

# 测试相机
python scripts/test_camera.py  # 采集10帧，无错误
```

---

### Milestone 2: 通信链路打通 (第2周)
**标志：**
- ✅ 串口双向通信正常
- ✅ 协议编解码验证通过
- ✅ 能发送模拟数据到H750

**验收标准：**
```bash
# 测试串口
python scripts/test_serial.py  # CRC校验100%通过

# 集成测试（手动）
python src/main.py  # 能发送心跳，接收反馈
```

---

### Milestone 3: 检测功能就绪 (第3周)
**标志：**
- ✅ YOLO检测器推理成功
- ✅ 检测结果准确
- ✅ 性能达标（<10ms）

**验收标准：**
```bash
# 编译C++模块
cd src/algorithms/build && cmake .. && make  # 成功

# 性能测试
python scripts/benchmark.py  # 推理<10ms, FPS>100
```

---

### Milestone 4: 完整流程打通 (第3周末)
**标志：**
- ✅ 相机 → 检测 → 串口完整流程运行
- ✅ 能实时追踪并发送角度到H750
- ✅ 系统稳定运行1小时

**验收标准：**
```bash
# 运行主程序
python src/main.py --config config/system_config.yaml

# 检查输出:
# [INFO] FPS: 35.2, Detections: 2, Latency: 42ms
# [INFO] Target: pitch=5.2°, yaw=-12.3°
# [INFO] H750 Mode: JETSON_CONTROL
```

---

## 📅 时间线 (甘特图)

```
Week 1:
  Sprint 1 ████████░░ (80%)
  Sprint 2 ░░████████ (开始)

Week 2:
  Sprint 2 ████████░░ (完成)
  Sprint 3 ░░████████ (完成)

Week 3:
  Sprint 4 ██████████ (完成)
  Sprint 5 ░░████████ (完成)
```

---

## 📞 支持与协作

### 与H750团队协作
- **协议对齐**: 第1周完成协议文档确认
- **联调时间**: 第2周开始串口联调
- **问题反馈**: 每日同步进度和问题

### 资源需求
- **硬件**: Jetson + 相机 + H750开发板
- **模型**: YOLOv8-nano预训练权重
- **时间**: 3周全职开发

---

## ✅ 完成标准

第一阶段视为完成，当满足以下所有条件：

1. ✅ **代码完整**: 所有P0/P1任务代码已实现
2. ✅ **测试通过**: 单元测试和集成测试全部通过
3. ✅ **性能达标**: FPS≥30, 延迟<50ms
4. ✅ **稳定运行**: 能连续运行1小时无崩溃
5. ✅ **文档齐全**: 代码注释、README、开发日志完整
6. ✅ **硬件对接**: 与H750联调成功

---

**创建日期**: 2025-10-08
**计划执行**: 第一阶段开发启动后
**预计完成**: 启动后3周
**责任人**: 猫娘工程师 幽浮喵 ฅ'ω'ฅ

---

**END OF PLAN**
