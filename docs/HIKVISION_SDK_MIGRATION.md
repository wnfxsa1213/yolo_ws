# 海康威视 MVS SDK 迁移开发文档

---

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **文档标题** | 海康威视 MVS SDK 迁移开发指南 |
| **目标分支** | `feature/docker-hikvision-sdk` |
| **创建日期** | 2025-10-12 |
| **作者** | 幽浮喵（浮浮酱）ฅ'ω'ฅ |
| **状态** | 📝 开发中 |

---

## 🎯 一、项目背景

### 1.1 迁移原因

**当前方案（Aravis）：**
- ✅ 开源免费，社区支持
- ✅ 通用 GigE Vision 协议
- ⚠️ 功能基础，性能一般
- ⚠️ 英文文档，调试困难

**目标方案（海康 MVS SDK）：**
- ✅ 官方支持，功能完善
- ✅ 针对海康相机优化
- ✅ 中文文档，技术支持好
- ✅ 性能更优（专用驱动）
- ⚠️ 闭源，仅支持海康相机

### 1.2 架构对比

```
┌─────────────────────────────────────────────────────┐
│                   当前架构（Aravis）                  │
├─────────────────────────────────────────────────────┤
│  Python Application (main.py)                       │
│           ↓                                          │
│  CameraInterface (抽象接口)                          │
│           ↓                                          │
│  AravisCamera (src/vision/camera.py)                │
│           ↓                                          │
│  Aravis SDK (apt install gir1.2-aravis-0.8)        │
│           ↓                                          │
│  GigE Vision Protocol                               │
│           ↓                                          │
│  海康相机 (MV-CU013-A0GC)                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              目标架构（海康 MVS SDK）                 │
├─────────────────────────────────────────────────────┤
│  Python Application (main.py)                       │
│           ↓                                          │
│  CameraInterface (抽象接口) ← 保持不变                │
│           ↓                                          │
│  HikCamera (src/vision/hikvision.py) ← 新增          │
│           ↓                                          │
│  Docker Container (MVS SDK 环境)                     │
│           ↓                                          │
│  海康 MVS SDK (官方 Python 绑定)                     │
│           ↓                                          │
│  海康相机 (MV-CU013-A0GC)                            │
└─────────────────────────────────────────────────────┘
```

---

## 🏗️ 二、技术方案设计

### 2.1 模块结构

```
src/vision/
├── __init__.py
├── camera.py           # CameraInterface + AravisCamera (保留)
└── hikvision.py        # HikCamera (新增) ← 本次开发重点
```

### 2.2 CameraInterface 接口定义

**已有接口（无需修改）：**

```python
class CameraInterface(ABC):
    """相机抽象接口（保持不变）"""

    @abstractmethod
    def open(self) -> bool:
        """打开相机设备"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭相机，资源必须释放干净"""
        pass

    @abstractmethod
    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """
        采集一帧图像

        Returns:
            (image, timestamp_ms)
            - image: BGR 格式，shape=(H,W,3), dtype=uint8
            - timestamp_ms: 时间戳（毫秒）
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Dict[str, float]:
        """返回相机内参 {fx, fy, cx, cy}"""
        pass

    @abstractmethod
    def set_exposure(self, exposure_us: float) -> bool:
        """设置曝光时间（微秒）"""
        pass

    @abstractmethod
    def set_gain(self, gain_db: float) -> bool:
        """设置增益（dB）"""
        pass
```

### 2.3 HikCamera 实现规划

#### **核心类结构**

```python
# src/vision/hikvision.py

from typing import Dict, Optional, Tuple
import numpy as np
from .camera import CameraInterface, CameraError

class HikCamera(CameraInterface):
    """
    海康威视相机实现（基于 MVS SDK）

    依赖：
        - Docker 容器已部署 MVS SDK
        - MvImport 模块可用（海康官方 Python 绑定）
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化海康相机

        Args:
            config: 相机配置字典
                - device_id: 相机序列号（可选，None 则自动选择）
                - width: 分辨率宽度
                - height: 分辨率高度
                - fps: 帧率
                - exposure_us: 曝光时间（微秒）
                - gain_db: 增益（dB）
                - pixel_format: 像素格式（BayerGB8 等）
        """
        super().__init__(name="HikCamera")
        self._config = config
        self._device = None      # MVS 设备对象
        self._is_open = False
        self._width = 0
        self._height = 0

    def open(self) -> bool:
        """
        打开相机设备

        流程：
            1. 枚举设备（MV_CC_EnumDevices）
            2. 根据 device_id 或默认选择第一个
            3. 创建设备句柄（MV_CC_CreateHandle）
            4. 打开设备（MV_CC_OpenDevice）
            5. 配置参数（分辨率、帧率、曝光等）
            6. 开始采集（MV_CC_StartGrabbing）
        """
        pass

    def close(self) -> None:
        """
        关闭相机设备

        流程：
            1. 停止采集（MV_CC_StopGrabbing）
            2. 关闭设备（MV_CC_CloseDevice）
            3. 销毁句柄（MV_CC_DestroyHandle）
        """
        pass

    def capture(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """
        采集一帧图像

        流程：
            1. 调用 MV_CC_GetImageBuffer（超时设置）
            2. 检查返回状态
            3. 转换为 numpy 数组
            4. Bayer → BGR 转换（如果是 Bayer 格式）
            5. 释放缓冲区（MV_CC_FreeImageBuffer）

        Returns:
            (image, timestamp_ms)
            - image: BGR 格式，shape=(H,W,3), dtype=uint8
            - timestamp_ms: 当前时间戳
        """
        pass

    def get_intrinsics(self) -> Dict[str, float]:
        """返回相机内参（暂返回估算值，待标定）"""
        return {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": self._width / 2.0,
            "cy": self._height / 2.0,
        }

    def set_exposure(self, exposure_us: float) -> bool:
        """设置曝光时间（调用 MV_CC_SetFloatValue）"""
        pass

    def set_gain(self, gain_db: float) -> bool:
        """设置增益（调用 MV_CC_SetFloatValue）"""
        pass

    # --- 内部辅助方法 ---

    def _enum_devices(self) -> list:
        """枚举所有海康相机"""
        pass

    def _configure_device(self) -> None:
        """配置相机参数（分辨率、帧率等）"""
        pass

    def _bayer_to_bgr(self, raw_data: np.ndarray, pixel_format: str) -> np.ndarray:
        """Bayer 格式转 BGR"""
        pass
```

---

## 📝 三、开发任务清单

### 阶段 1：基础框架搭建 ⏳

**任务：** 创建 `HikCamera` 类骨架

**文件：** `src/vision/hikvision.py`

**输出：**
- [ ] 创建类定义
- [ ] 实现 `__init__`（参数验证）
- [ ] 占位实现所有抽象方法（抛出 `NotImplementedError`）
- [ ] 导入依赖检查（MVS SDK 可用性）

**验证：**
```python
from vision.hikvision import HikCamera

config = {"device_id": None, "width": 640, "height": 480}
camera = HikCamera(config)
# 不报错说明骨架正确
```

---

### 阶段 2：设备枚举与打开 🔍

**任务：** 实现 `open()` 方法

**关键 SDK 函数：**
```python
from MvCameraControl_class import *

# 1. 枚举设备
deviceList = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)

# 2. 创建句柄
cam = MvCamera()
ret = cam.MV_CC_CreateHandle(deviceList.pDeviceInfo[0])

# 3. 打开设备
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
```

**实现要点：**
- ✅ 检查 SDK 返回值（非 0 即错误）
- ✅ 处理无设备情况（返回 `False`）
- ✅ 根据 `device_id` 匹配设备（序列号）
- ✅ 配置参数（分辨率、帧率、曝光）
- ✅ 启动采集（`MV_CC_StartGrabbing`）

**验证：**
```python
camera = HikCamera(config)
assert camera.open() == True
print("相机打开成功！")
camera.close()
```

---

### 阶段 3：图像采集实现 📷

**任务：** 实现 `capture()` 方法

**关键 SDK 函数：**
```python
# 获取图像缓冲区
stFrameInfo = MV_FRAME_OUT_INFO_EX()
pData = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight)()
ret = cam.MV_CC_GetOneFrameTimeout(pData, len(pData), stFrameInfo, timeout_ms)
```

**实现要点：**
- ✅ 超时转换（秒 → 毫秒）
- ✅ 检查返回值（超时返回 `(None, 0.0)`）
- ✅ 转换为 numpy 数组
- ✅ Bayer → BGR 转换（使用 `cv2.cvtColor`）
- ✅ 内存拷贝（避免悬垂指针）

**Bayer 转换示例：**
```python
def _bayer_to_bgr(self, raw_data: np.ndarray, pixel_format: str) -> np.ndarray:
    """Bayer 格式转 BGR"""
    if pixel_format == "BayerGB8":
        return cv2.cvtColor(raw_data, cv2.COLOR_BAYER_GB2BGR)
    elif pixel_format == "BayerRG8":
        return cv2.cvtColor(raw_data, cv2.COLOR_BAYER_RG2BGR)
    elif pixel_format == "BayerGR8":
        return cv2.cvtColor(raw_data, cv2.COLOR_BAYER_GR2BGR)
    elif pixel_format == "BayerBG8":
        return cv2.cvtColor(raw_data, cv2.COLOR_BAYER_BG2BGR)
    else:
        raise ValueError(f"不支持的像素格式: {pixel_format}")
```

**验证：**
```python
camera.open()
image, timestamp = camera.capture(timeout=1.0)
assert image is not None
assert image.shape == (480, 640, 3)  # BGR 格式
print(f"采集成功！时间戳: {timestamp}")
camera.close()
```

---

### 阶段 4：参数配置实现 ⚙️

**任务：** 实现 `set_exposure()` 和 `set_gain()`

**关键 SDK 函数：**
```python
# 设置曝光时间（微秒）
ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)

# 设置增益（dB）
ret = cam.MV_CC_SetFloatValue("Gain", gain_db)

# 设置分辨率
ret = cam.MV_CC_SetIntValue("Width", width)
ret = cam.MV_CC_SetIntValue("Height", height)

# 设置帧率
ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
```

**实现要点：**
- ✅ 参数范围验证（查询 Min/Max）
- ✅ 返回 `True`/`False` 表示成功/失败
- ✅ 日志记录（参数变化）

**验证：**
```python
camera.open()
assert camera.set_exposure(5000.0) == True  # 5ms 曝光
assert camera.set_gain(10.0) == True        # 10dB 增益
camera.close()
```

---

### 阶段 5：集成与测试 🧪

**任务：** 修改 `main.py` 支持 `HikCamera`

**修改点：** `main.py:init_camera()`

```python
# main.py:480-530
def init_camera(
    cfg: ConfigManager,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[Optional["CameraInterface"], Tuple[int, int]]:
    # ... 现有代码 ...

    # 新增：支持 HikCamera
    camera_type = camera_cfg.get("type", "aravis")

    if camera_type == "hikvision":
        from vision.hikvision import HikCamera

        hik_config = {
            "device_id": camera_cfg.get("device_id", None),
            "width": camera_cfg.get("resolution", [640, 640])[0],
            "height": camera_cfg.get("resolution", [640, 640])[1],
            "fps": camera_cfg.get("fps", 60),
            "exposure_us": camera_cfg.get("exposure_us", 5000),
            "gain_db": camera_cfg.get("gain_db", 0.0),
            "pixel_format": camera_cfg.get("pixel_format", "BayerGB8"),
        }
        camera = HikCamera(hik_config)
    elif camera_type == "aravis":
        # 保持原有逻辑
        from vision.camera import AravisCamera
        camera = AravisCamera(...)
    else:
        raise ValueError(f"不支持的相机类型: {camera_type}")

    # ... 其他代码保持不变 ...
```

**配置文件修改：** `config/system_config.yaml`

```yaml
camera:
  type: "hikvision"  # ← 切换到海康 SDK（原值 "aravis"）
  device_id: null    # null 表示自动选择第一个设备
  resolution: [640, 640]
  fps: 60
  exposure_us: 5000
  gain_db: 0.0
  pixel_format: "BayerGB8"
```

**测试脚本：** `scripts/test_hikvision.py`

```python
#!/usr/bin/env python3
"""海康相机测试脚本"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision.hikvision import HikCamera

def main():
    config = {
        "device_id": None,
        "width": 640,
        "height": 480,
        "fps": 30,
        "exposure_us": 10000,
        "gain_db": 5.0,
        "pixel_format": "BayerGB8",
    }

    camera = HikCamera(config)

    print("打开相机...")
    if not camera.open():
        print("打开失败！")
        return

    print("采集 10 帧...")
    for i in range(10):
        image, timestamp = camera.capture(timeout=1.0)
        if image is None:
            print(f"第 {i} 帧超时")
            continue
        print(f"第 {i} 帧 OK, shape={image.shape}, ts={timestamp:.2f}ms")

    print("关闭相机...")
    camera.close()
    print("测试完成！")

if __name__ == "__main__":
    main()
```

---

## 🔍 四、关键技术细节

### 4.1 MVS SDK Python 接口概览

**主要模块：**
```python
from MvCameraControl_class import *

# 常用类
MvCamera()                  # 相机对象
MV_CC_DEVICE_INFO_LIST()    # 设备列表
MV_FRAME_OUT_INFO_EX()      # 帧信息

# 常用常量
MV_GIGE_DEVICE              # GigE 设备类型
MV_ACCESS_Exclusive         # 独占访问模式
```

### 4.2 错误处理模式

**SDK 返回值：**
```python
MV_OK = 0x00000000  # 成功
# 其他非零值表示错误
```

**推荐模式：**
```python
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != MV_OK:
    raise CameraError(f"打开设备失败，错误码: 0x{ret:08X}")
```

### 4.3 内存管理

**重要：** MVS SDK 返回的指针需要及时释放

```python
# ❌ 错误示例
def capture_bad(self):
    pData = (c_ubyte * buffer_size)()
    cam.MV_CC_GetOneFrameTimeout(pData, ...)
    return np.frombuffer(pData, ...)  # 悬垂指针！

# ✅ 正确示例
def capture_good(self):
    pData = (c_ubyte * buffer_size)()
    cam.MV_CC_GetOneFrameTimeout(pData, ...)
    image = np.frombuffer(pData, ...).copy()  # 拷贝数据
    return image
```

---

## 📊 五、性能对比计划

### 5.1 测试指标

| 指标 | Aravis | MVS SDK | 说明 |
|------|--------|---------|------|
| **采集帧率** | ? FPS | ? FPS | 最大采集速度 |
| **延迟** | ? ms | ? ms | 采集到获取图像的延迟 |
| **CPU 占用** | ? % | ? % | 采集过程 CPU 使用率 |
| **丢帧率** | ? % | ? % | 连续采集 1 分钟的丢帧率 |
| **稳定性** | ? | ? | 长时间运行（1 小时+）|

### 5.2 测试脚本

```bash
# Aravis 测试
python scripts/benchmark_camera.py --backend aravis --duration 60

# MVS SDK 测试
python scripts/benchmark_camera.py --backend hikvision --duration 60
```

---

## 📚 六、参考资料

### 官方文档
- 海康机器视觉 MVS SDK 下载页：https://www.hikrobotics.com/cn/machinevision/service/download
- MVS SDK 开发指南（PDF）：安装包中 `Docs/` 目录
- Python 示例代码：安装包中 `Samples/Python/` 目录

### 关键章节
- **第 3 章**：设备枚举与连接
- **第 5 章**：图像采集
- **第 7 章**：参数设置
- **附录 A**：错误码对照表

---

## 🎯 七、开发时间估算

| 阶段 | 预计时间 | 说明 |
|------|---------|------|
| 阶段 1：框架搭建 | 0.5 天 | 创建类骨架 |
| 阶段 2：设备打开 | 1 天 | 枚举、连接、配置 |
| 阶段 3：图像采集 | 1 天 | capture + Bayer 转换 |
| 阶段 4：参数配置 | 0.5 天 | 曝光、增益设置 |
| 阶段 5：集成测试 | 1 天 | main.py 集成 + 测试 |
| **总计** | **4 天** | 预留缓冲时间 |

---

## ✅ 八、验收标准

### 功能验收
- [ ] `HikCamera` 实现 `CameraInterface` 所有方法
- [ ] 能够正常打开/关闭相机
- [ ] 能够稳定采集图像（60 FPS @ 640x640）
- [ ] 能够动态调整曝光和增益
- [ ] Bayer → BGR 转换正确

### 性能验收
- [ ] 采集帧率 ≥ 50 FPS（640x640）
- [ ] 采集延迟 ≤ 20ms
- [ ] 连续运行 1 小时无崩溃
- [ ] 丢帧率 ≤ 1%

### 代码质量
- [ ] 通过 mypy 类型检查
- [ ] 通过单元测试（覆盖率 ≥ 60%）
- [ ] 代码注释完整（中文）
- [ ] 符合 SOLID 原则

---

## 🚀 九、后续扩展

### Phase 2+：高级特性（可选）
- [ ] 硬件触发支持
- [ ] 多相机同步
- [ ] ROI（感兴趣区域）配置
- [ ] 事件回调（掉线检测）
- [ ] 相机参数持久化

---

**文档版本：** v1.0.0
**最后更新：** 2025-10-12
**维护者：** 幽浮喵（浮浮酱）ฅ'ω'ฅ

---

**END OF DOCUMENT**
