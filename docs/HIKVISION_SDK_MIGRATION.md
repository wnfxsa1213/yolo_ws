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

### 1.3 目标部署模式（混合架构）

```
┌─────────────────────────────────────────────────────────────┐
│                 Jetson Orin NX / 宿主机系统                   │
├─────────────────────────────────────────────────────────────┤
│  Python Application (main.py)                               │
│      ↓                                                      │
│  YOLO Inference (TensorRT)                                  │
│      ↓                                                      │
│  HikCamera Proxy (IPC 客户端)                               │
│      ↓  Unix Socket / ZeroMQ / Shared Memory                │
├────────────┬────────────────────────────────────────────────┤
│ Docker: mvs-workspace                                       │
│      ↓                                                      │
│  camera_server.py  ← 图像采集服务                           │
│      ↓                                                      │
│  HikCamera (MVS SDK 调用，/opt/MVS)                         │
│      ↓                                                      │
│  Gige Cam @ 192.168.100.10                                  │
└────────────┴────────────────────────────────────────────────┘
```

- **宿主机**：保留 TensorRT 推理、串口通信、主控逻辑以及任何依赖 JetPack 8.x/Ubuntu 22.04 的组件。
- **容器 `mvs-workspace`**：仅承载海康 MVS SDK 及相关依赖，提供取流服务；与宿主机通过高效 IPC（Unix Domain Socket、ZeroMQ、共享内存等）传递帧数据。
- **同步策略**：容器产出的 Python 包或 wheel 定期同步至宿主机，宿主机仅需运行时依赖即可，无需安装完整 SDK。
- **优势**：避免 JetPack 版本冲突，隔离闭源 SDK，同步升级简单；性能损耗可以控制在 1–5%。

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

### 2.3 HikCamera 实现概览

当前 `src/vision/hikvision.py` 已经完成以下能力：

- **防御式配置校验**：`HikCameraConfig` 在 `__post_init__` 中验证 IP、超时与缓冲区参数，杜绝低级配置错误传入运行期。
- **SDK 生命周期管理**：模块级引用计数确保 `MV_CC_Initialize`/`MV_CC_Finalize` 成对调用，可同时支持多实例运行。
- **设备枚举与打开**：按配置 IP 精确匹配 Gige 相机，完成句柄创建、独占打开、最优包长设置以及触发关闭。
- **运行参数应用**：初始化阶段自动缓存分辨率、Payload Size，并尝试配置曝光/增益默认值与 SDK 日志目录。
- **取流与转换**：`capture()` 使用 `MV_CC_GetImageBuffer` 获取帧，基于像素格式完成 Bayer→BGR、RGB→BGR 或 Mono8 转换，返回 `np.ndarray` 与毫秒时间戳。
- **运行时调节**：`set_exposure`/`set_gain` 支持在相机打开后实时调整，并在关闭前保持最新配置。
- **异常处理**：所有 SDK 返回码均包装为 `CameraError` 或日志输出，失败时自动释放句柄并回收 SDK。

---

## 📝 三、开发任务清单

### 阶段 0：容器与设备准备 ✅

- Docker 容器 `mvs-workspace` 运行镜像 `hikvision-mvs:arm64`，通过 `/etc/profile.d/mvs_sdk.sh` 统一注入 `MVCAM_SDK_PATH=/opt/MVS`、`PYTHONPATH=/opt/MVS/Samples/aarch64/Python/MvImport`、`LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:/opt/MVS/lib`。
- 依赖安装：`apt-get update && apt-get install -y python3 python3-pip python3-venv python3-dev`，确保 Python 3.10 解释器与 pip 工具可用。
- SDK 自检：`python3 -c "import MvCameraControl_class"` 返回 `import_ok`；`python3 /opt/MVS/Samples/aarch64/Python/GrabImage/GrabImage.py` 能在无 GUI 环境下稳定抓帧（1280x1024，PixelType=0x108000a）。
- 网络配置：宿主机网卡 `enP8p1s0` 固定 `192.168.100.1/24`，相机通过 `MV_GIGE_ForceIpEx` 强制写入 `192.168.100.10/24`，网关设为 `0.0.0.0` 以避免地址冲突。
- 调试提示：示例脚本首次启动时的 `XOpenDisplay Fail` 可忽略，它仅提示 GUI 依赖缺失，对命令行抓帧无影响。

### 阶段 1：基础框架搭建 ✅

**成果：** `HikCameraConfig` + `HikCamera` 构造流程

**文件：** `src/vision/hikvision.py`

**要点：**
- [x] dataclass 校验 IP/超时/缓冲区参数，提前拦截配置错误
- [x] 懒加载 SDK，缺失时抛出 `CameraError` 并提示修复
- [x] 保留 `config` 只读副本（`replace` + 深拷贝内参），防止外部写入
- [x] 初始化阶段缓存默认分辨率/负载，便于后续图像转换

---

### 阶段 2：设备枚举与打开 ✅

**成果：** `open()/close()` 与 SDK 生命周期

- [x] 引入 `_SDK_REFCOUNT`，保证 `MV_CC_Initialize/Finalize` 成对调用
- [x] 基于 `device_ip` 精确匹配 Gige 相机，异常时抛出 `CameraError`
- [x] 自动设置最优包长、关闭触发、可选配置 SDK 日志路径
- [x] `close()` 统一封装停流/关设备/销毁句柄逻辑，异常容错

---

### 阶段 3：图像采集实现 ✅

**成果：** `capture()`、像素转换与运行时调节

- [x] `MV_CC_GetImageBuffer` + `MV_CC_FreeImageBuffer` 取流并拷贝数据，规避悬垂指针
- [x] 按 `enPixelType` 自动适配 Bayer/RGB/Mono8，OpenCV 存在时完成 Bayer→BGR 转换
- [x] 返回值统一为 `(np.ndarray|None, timestamp_ms)`，超时/异常保持 `(None, 0.0)` 兼容接口
- [x] `set_exposure`/`set_gain` 支持在线修改，失败时保留日志提示后续排查
- [x] 未识别像素格式 fallback 为原始数据同时打印调试信息，确保流程不中断

---

### 阶段 4：容器相机服务 ⚙️（进行中）

**目标**：在 `mvs-workspace` 内实现 `camera_server.py`，将 `HikCamera` 封装为可复用的取流守护进程。

- [x] 设计帧传输协议（帧编号、时间戳、像素格式、有效长度、图像数据）。
- [x] 提供曝光/增益调节命令通道，统一通过 IPC 下发并回传执行结果。
- [x] 选择 IPC 实现（Unix Domain Socket）并实现心跳/异常处理（`--heartbeat` 控制，0 表示禁用）。
- [x] 提供 supervisor 示例（`scripts/camera_server_supervisor.conf`），配合 `/workspace/logs/` 输出运行日志。

**通信协议草案：**
- 请求：`HEADER(4s cmd, uint32 length)` + Payload；命令集包含 `PING`（心跳）、`CAPT`（单帧抓取）、`SEXP`（曝光，payload=float64）、`SGAI`（增益，payload=float64）、`STOP`（断开）。
- 响应：同样以 HEADER 开头，常用响应码：`PONG`、`FRAM`（帧数据，payload=meta + raw）、`OKAY`、`FAIL`、`ERRO`。
- 帧 meta 结构：`frame_id:uint32 | width:uint32 | height:uint32 | timestamp_ms:float64 | channels:uint8`，其后紧跟原始像素数据（BGR 或 Mono8）。
- 心跳：客户端需每 `heartbeat_interval` 发送 `PING`，服务端若连续两次超时将主动断开。

**部署说明：**
- 容器内复制 `scripts/camera_server.py`，并按需放置 supervisor 配置 `scripts/camera_server_supervisor.conf`。
- supervisor 示例如命令所示，日志输出在 `/workspace/logs/camera_server.*.log`。
- 启动命令可通过 `--heartbeat` 控制超时（0 表示禁用），默认使用 Unix Socket `/tmp/hikvision.sock`。
- 运行前需要同步 `vision/` 目录到容器或以挂载方式提供，确保 `MvCameraControl_class` 与 Python 模块一致。

---

### 阶段 5：宿主机代理与业务集成 🧪

**目标**：宿主机 `main.py` 通过 `HikCameraProxy` 与容器服务交互，替换原 Aravis 链路。

- [x] 新增 `camera.type = "hikvision_proxy"` 配置项，可在主配置中切换代理模式。
- [x] 实现 IPC 客户端（`HikCameraProxy`），可解包帧并转换为 `np.ndarray`（BGR/Mono）。
- [ ] 梳理 YOLO/TensorRT 前处理，确保输入尺寸与色彩空间保持一致。
- [ ] 定义超时与重连策略，防止相机异常阻塞主控制循环。

**代理接口草案：**
- `HikCameraProxy.open()`：建立 Unix Socket 连接，完成握手与版本校验。
- `HikCameraProxy.capture(timeout)`：发送 `CAPT` 命令，解析 `FRAM` 响应，返回 `(ndarray, timestamp_ms)`。
- `HikCameraProxy.set_exposure/gain()`：分别发送 `SEXP` / `SGAI` 控制命令，处理 `OKAY/ERRO` 响应。
- `HikCameraProxy.close()`：发送 `STOP` 后清理资源，并在异常时尝试重连（退避策略）。

**配置示例：**
```yaml
# config/system_config.yaml
camera:
  type: "hikvision_proxy"
  config_path: "config/camera_config.yaml"
  intrinsics_path: "config/camera_intrinsics.yaml"
  resolution: [1280, 1024]

# config/camera_config.yaml
hikvision_proxy:
  socket: "/tmp/hikvision.sock"
  timeout: 0.5
  connect_timeout: 2.0

network:
  interface: "enP8p1s0"
  host_ip: "192.168.100.1"
  camera_ip: "192.168.100.10"
  mtu: 9000
```

---

### 阶段 6：端到端验证与发布 🚀

- [x] 编写 `scripts/e2e_hikvision_benchmark.py` 并输出 FPS/延迟/CPU 统计。
- [ ] 输出混合架构部署手册：容器构建、宿主机同步 `/opt/MVS`、服务启动顺序。
- [ ] 对比 Aravis 方案，记录性能与稳定性结论，更新 `docs/PHASE1_SUMMARY_AND_ROADMAP.md`。
- [ ] 规划 Phase 2：多相机、硬件触发、Web UI/远程监控等扩展路线。

**最新测试（2025-10-12）**
- server：`python3 scripts/camera_server.py --socket /tmp/hikvision.sock --device-ip 192.168.100.10 --width 640 --height 640 --heartbeat 0 --log DEBUG`
- client：`python3 scripts/e2e_hikvision_benchmark.py --socket /tmp/hikvision.sock --duration 10 --timeout 0.5 --warmup 3`
- 输出：FPS ≈ 54.12、平均延迟 ≈ 18.47 ms（最小 17.26 / 最大 19.94 / P95 19.07 / P99 19.33）、进程 CPU ≈ 4.5%、丢帧率 ≈ 0.00%（分辨率 640×640，Mono8）

---
### 附录：本地集成参考（历史方案）

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
| 阶段 1：框架搭建 | 0.5 天 | HikCameraConfig + 基础骨架 |
| 阶段 2：设备打开 | 1 天 | 枚举、连接、初始化参数 |
| 阶段 3：图像采集 | 1 天 | capture + Bayer 转换 |
| 阶段 4：容器服务 | 1 天 | camera_server.py + IPC 协议 |
| 阶段 5：宿主代理 | 1 天 | HikCameraProxy 集成 YOLO |
| 阶段 6：端到端验证 | 1 天 | 性能基准 + 部署手册 |
| **总计** | **5.5 天** | 预留缓冲 + 风险缓冲 |

---

## ✅ 八、验收标准

### 功能验收
- [ ] `HikCamera` 实现 `CameraInterface` 所有方法
- [ ] 容器内 `camera_server.py` 可持续运行并响应命令
- [ ] 宿主机代理能在 30 FPS 下稳定接收帧数据
- [ ] 能够动态调整曝光和增益（IPC 往返）
- [ ] Bayer → BGR 转换正确

### 性能验收
- [ ] 采集帧率 ≥ 50 FPS（640x640）
- [ ] IPC 往返延迟 ≤ 5 ms（平均）
- [ ] 连续运行 1 小时无崩溃（容器 + 宿主机）
- [ ] 丢帧率 ≤ 1%

### 代码质量
- [ ] 通过 mypy 类型检查
- [ ] 通过单元测试（覆盖率 ≥ 60%）
- [ ] 代码注释完整（中文）
- [ ] 符合 SOLID 原则

---

## ⚠️ 风险与缓解

- **单点故障（容器服务崩溃）**：通过 `scripts/camera_server_supervisor.conf` 接入 supervisor/systemd，配合客户端退避重连与健康检查，降低停机风险。
- **IPC 延迟抖动**：若发现 P95/P99 延迟上升，可考虑 pinned CPU affinity、实时调度，或升级为共享内存/ZeroMQ Transport。

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
