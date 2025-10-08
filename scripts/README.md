# 脚本工具目录 (Scripts)

## 用途

存放开发和部署相关的工具脚本。

## 待实现脚本

### export_onnx.py
**功能：** 将PyTorch模型导出为ONNX格式

**用法：**
```bash
python scripts/export_onnx.py \
    --weights models/yolov8n.pt \
    --imgsz 640 \
    --output models/yolov8n.onnx
```

**输出：** `yolov8n.onnx` (ONNX模型文件)

---

### build_engine.py
**功能：** 将ONNX模型转换为TensorRT引擎

**用法：**
```bash
python scripts/build_engine.py \
    --onnx models/yolov8n.onnx \
    --engine models/yolov8n_fp16.engine \
    --fp16 \
    --workspace 2
```

**参数：**
- `--onnx`: 输入ONNX模型路径
- `--engine`: 输出TensorRT引擎路径
- `--fp16`: 启用FP16精度（推荐）
- `--int8`: 启用INT8量化（需要校准数据集）
- `--workspace`: 最大工作空间大小 (GB)

**输出：** `yolov8n_fp16.engine` (TensorRT引擎文件)

---

### test_camera.py
**功能：** 测试海康相机连接和图像采集

**用法：**
```bash
python scripts/test_camera.py \
    --config config/camera_config.yaml \
    --save-image \
    --num-frames 10
```

**功能：**
- 检测相机连接状态
- 采集指定数量的图像帧
- 显示帧率和分辨率
- 保存测试图像到 `logs/camera_test/`

---

### test_serial.py
**功能：** 测试串口通信和协议编解码

**用法：**
```bash
python scripts/test_serial.py \
    --port /dev/ttyTHS0 \
    --baudrate 460800 \
    --loopback
```

**测试内容：**
- 串口连接测试
- 协议编码/解码测试
- CRC8校验验证
- 回环测试（如有硬件支持）
- 发送/接收延迟测试

---

### benchmark.py
**功能：** 性能基准测试

**用法：**
```bash
python scripts/benchmark.py \
    --engine models/yolov8n_fp16.engine \
    --iterations 100
```

**测试内容：**
- YOLO推理速度（平均、最小、最大）
- GPU利用率
- 显存占用
- CPU占用
- 端到端延迟

**输出：** 性能报告保存到 `logs/benchmark_YYYYMMDD_HHMMSS.txt`

---

### calibrate_camera.py
**功能：** 相机标定（获取内参矩阵）

**用法：**
```bash
python scripts/calibrate_camera.py \
    --pattern-size 9x6 \
    --square-size 25 \
    --num-images 20
```

**流程：**
1. 实时预览相机画面
2. 检测标定板角点
3. 采集多张标定图像
4. 计算内参和畸变系数
5. 保存到 `config/camera_intrinsics.yaml`

---

### deploy.sh
**功能：** 一键部署脚本

**用法：**
```bash
bash scripts/deploy.sh
```

**执行内容：**
1. 检查环境依赖
2. 编译C++模块
3. 验证模型文件
4. 测试硬件连接
5. 启动主程序

---

## 状态
⏳ 所有脚本待实现
