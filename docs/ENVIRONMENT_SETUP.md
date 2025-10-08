# Jetson Orin NX Super 16GB 环境配置指南

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **硬件平台** | Jetson Orin NX Super 16GB |
| **系统版本** | JetPack R36.4.4 (L4T 36.4.4) |
| **更新日期** | 2025-10-08 |
| **作者** | 幽浮喵 (浮浮酱) ฅ'ω'ฅ |

---

## 🖥️ 硬件规格

```yaml
处理器:
  GPU: 1024-core NVIDIA Ampere (8 SMs)
  CPU: 8-core ARM Cortex-A78AE
  内存: 16GB LPDDR5
  AI性能: 100 TOPS (INT8)

存储:
  推荐: NVMe SSD 256GB+
  可选: microSD卡 (性能受限)

接口:
  USB: 4x USB 3.2
  网络: 千兆以太网
  相机: MIPI CSI-2 / USB3.0

功耗模式:
  10W模式: 低功耗
  15W模式: 平衡 (推荐开发)
  25W模式: 最大性能
```

---

## 🔧 预装软件环境

### 系统信息

```bash
# 查看系统版本
cat /etc/nv_tegra_release
# 输出: R36 (release), REVISION: 4.4

# 查看JetPack版本
dpkg -l | grep nvidia-jetpack
```

### GPU加速库（主环境）

当前系统已预装以下NVIDIA优化的GPU加速库，**不可替换**：

| 库名称 | 版本 | GPU支持 | 说明 |
|--------|------|---------|------|
| **PyTorch** | 2.5.0a0+872d972e41.nv24.08 | ✅ CUDA 12.6 | NVIDIA定制版 |
| **OpenCV** | 4.10.0 | ✅ CUDA模块 | 带CUDA加速 |
| **NumPy** | 1.26.4 | ✅ GPU兼容 | 与GPU库适配 |
| **TensorRT** | 10.3.0.30 | ✅ CUDA 12.5 | 推理加速 |
| **CUDA Toolkit** | 12.6 | ✅ | 核心运行时 |
| **cuDNN** | 9.x | ✅ | 深度学习加速 |

### 验证GPU环境

```bash
# 检查CUDA
nvcc --version

# 检查GPU
nvidia-smi

# Python环境检查
python3 << 'EOF'
import torch
import cv2
import numpy as np

print("=" * 60)
print("🔍 GPU加速库检查")
print("=" * 60)
print(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"✓ OpenCV: {cv2.__version__} (CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0})")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ CUDA设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("=" * 60)
EOF
```

---

## ⚙️ uv虚拟环境配置

### 为什么需要 `--system-site-packages`？

在Jetson平台上创建虚拟环境时，**必须**使用 `--system-site-packages` 参数：

```bash
# ✅ 正确方式
uv venv --python 3.10 --system-site-packages

# ❌ 错误方式（会丢失GPU加速）
uv venv --python 3.10
```

**原因：**

1. **PyTorch GPU版本特殊**
   - NVIDIA为Jetson定制编译的PyTorch包含CUDA 12.6支持
   - PyPI上的标准PyTorch是CPU版本或x86 GPU版本
   - 如果虚拟环境重新安装，会覆盖为CPU版本，丢失GPU加速

2. **OpenCV CUDA模块**
   - 系统OpenCV 4.10.0包含CUDA加速模块（`cv2.cuda.*`）
   - PyPI上的 `opencv-python` 是CPU版本，不包含CUDA支持
   - 重新安装会失去图像处理GPU加速能力

3. **NumPy版本兼容性**
   - 系统NumPy 1.26.4与PyTorch/OpenCV版本精确匹配
   - 不同版本可能导致内存布局不兼容，引发崩溃

### 配置步骤

```bash
# 1. 进入项目目录
cd ~/yolo_ws

# 2. 创建虚拟环境（继承主环境GPU库）
uv venv --python 3.10 --system-site-packages

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 验证GPU库继承成功
python << 'EOF'
import torch
import cv2
print(f"✓ PyTorch CUDA: {torch.cuda.is_available()}")
print(f"✓ OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
EOF

# 预期输出:
# ✓ PyTorch CUDA: True
# ✓ OpenCV CUDA: True
```

---

## 📦 依赖安装策略

### 使用系统库（不可安装）

以下库**只能**使用系统版本，**禁止**在虚拟环境中安装：

```bash
# ❌ 禁止执行以下命令！
uv pip install torch
uv pip install opencv-python
uv pip install numpy
```

这些库通过 `--system-site-packages` 自动继承。

### 可安装的项目依赖

```bash
# ✅ 可以安装的库（不会影响GPU加速）
uv pip install pyserial        # 串口通信
uv pip install pyyaml           # 配置文件
uv pip install ultralytics      # YOLOv8框架
uv pip install Pillow           # 图像处理
uv pip install pytest           # 测试框架
```

### pyproject.toml配置

```toml
[project]
name = "gimbal-tracker"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    # ⚠️ 注意：不要列出 torch, opencv-python, numpy
    # 这些库使用系统版本（通过--system-site-packages继承）
    "pyserial>=3.5",
    "pyyaml>=6.0",
    "ultralytics>=8.0.0",
    "Pillow>=10.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

---

## 🔍 常见问题排查

### 1. PyTorch显示无CUDA支持

**症状：**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**原因：** 虚拟环境中安装了CPU版PyTorch覆盖了系统版本

**解决：**
```bash
# 删除虚拟环境
rm -rf .venv

# 使用正确参数重新创建
uv venv --python 3.10 --system-site-packages

# 激活并验证
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. OpenCV缺少CUDA模块

**症状：**
```python
>>> import cv2
>>> cv2.cuda.getCudaEnabledDeviceCount()
AttributeError: module 'cv2' has no attribute 'cuda'
```

**原因：** 安装了PyPI的opencv-python（CPU版本）

**解决：**
```bash
# 卸载opencv-python
uv pip uninstall opencv-python opencv-contrib-python

# 验证使用系统版本
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'cuda'))"
# 输出: 4.10.0 True
```

### 3. NumPy版本冲突

**症状：**
```
ValueError: numpy.ndarray size changed, may indicate binary incompatibility
```

**原因：** NumPy版本与PyTorch/OpenCV不匹配

**解决：**
```bash
# 确保使用系统NumPy 1.26.4
python -c "import numpy; print(numpy.__version__)"

# 如果版本不对，卸载虚拟环境中的NumPy
uv pip uninstall numpy
```

---

## 🚀 性能优化

### 解锁最大性能

```bash
# 设置最大功耗模式 (25W)
sudo nvpmodel -m 0

# 解锁CPU/GPU频率
sudo jetson_clocks

# 验证当前模式
sudo nvpmodel -q
```

### 监控系统状态

```bash
# 实时监控（Jetson专用工具）
sudo tegrastats

# 或使用nvidia-smi
watch -n 1 nvidia-smi

# 安装jtop（图形化监控）
sudo pip3 install jetson-stats
sudo jtop
```

---

## 📊 性能基准参考

### YOLOv8n推理性能 (Jetson Orin NX Super 16GB)

| 模型 | 精度 | 分辨率 | 推理时间 | FPS | 显存占用 |
|------|------|--------|---------|-----|---------|
| YOLOv8n | FP32 | 640×640 | ~14ms | 71 | 450MB |
| YOLOv8n | FP16 | 640×640 | **~7ms** | **142** | 250MB |
| YOLOv8n | INT8 | 640×640 | ~4ms | 250 | 180MB |
| YOLOv8s | FP16 | 640×640 | ~16ms | 62 | 380MB |

**测试条件：**
- TensorRT 10.3.0
- Batch Size: 1
- 25W性能模式
- CUDA 12.6

---

## 📚 参考资源

### 官方文档
- [Jetson Orin NX产品页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- [JetPack SDK文档](https://docs.nvidia.com/jetson/jetpack/)
- [TensorRT开发指南](https://docs.nvidia.com/deeplearning/tensorrt/)

### 社区资源
- [Jetson开发者论坛](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)
- [JetsonHacks](https://jetsonhacks.com/)

---

## ✅ 配置清单

完成以下检查，确保环境配置正确：

- [ ] JetPack R36.4.4 系统运行正常
- [ ] `nvidia-smi` 显示GPU信息
- [ ] PyTorch 2.5.0 CUDA可用
- [ ] OpenCV 4.10.0 带CUDA模块
- [ ] NumPy 1.26.4 版本正确
- [ ] uv虚拟环境使用 `--system-site-packages` 创建
- [ ] 虚拟环境中GPU库验证通过
- [ ] 功耗模式设置为15W或25W
- [ ] 海康相机SDK正确安装（如使用）

---

**配置完成！** o(*￣︶￣*)o

如有问题，请参考 `docs/Jetson_Development.md` 获取详细开发指南喵～ ฅ'ω'ฅ
