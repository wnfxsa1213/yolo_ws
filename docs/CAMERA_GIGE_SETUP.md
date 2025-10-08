# 海康GigE相机配置指南

## 📋 相机信息

| 项目 | 内容 |
|------|------|
| **型号** | MV-CU013-A0GC |
| **接口** | 千兆网口（GigE Vision） |
| **分辨率** | 1300万像素 |
| **类型** | 彩色相机 |
| **镜头** | 8mm定焦镜头 |
| **带宽** | 最大1000Mbps |

---

## 🔧 硬件连接

### 连接方式
```
海康相机 (MV-CU013-A0GC)
    │
    │ 网线（超五类/六类，建议≤3米）
    │
    ▼
Jetson Orin NX (enP8p1s0 千兆网口)
```

**注意事项：**
- ✅ 使用**超五类或六类**网线（支持千兆）
- ✅ 网线长度≤10米（推荐≤3米）
- ✅ 确保网口指示灯正常（绿灯常亮，黄灯闪烁）
- ❌ 不要连接到路由器/交换机（除非配置正确）
- ❌ 不要与其他设备共享同一网段

---

## 📡 网络配置

### 1. 检查当前网络接口

```bash
# 查看网络接口
ip addr show

# 查看有线网口状态（enP8p1s0或eth0）
ip addr show enP8p1s0

# 预期输出：
# 4: enP8p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ...
#     inet 192.168.1.5/24 ...
```

### 2. 配置静态IP（为相机预留网段）

#### 方案A：使用NetworkManager（推荐）

```bash
# 查看连接名称
nmcli connection show

# 设置静态IP（为相机预留192.168.100.x段）
sudo nmcli connection modify "Wired connection 1" \
    ipv4.method manual \
    ipv4.addresses 192.168.100.1/24 \
    ipv4.gateway 192.168.100.1

# 重启连接
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"

# 验证
ip addr show enP8p1s0
# 应显示: inet 192.168.100.1/24
```

#### 方案B：直接配置网口（临时）

```bash
# 设置Jetson网口IP
sudo ip addr add 192.168.100.1/24 dev enP8p1s0
sudo ip link set enP8p1s0 up

# 验证
ip addr show enP8p1s0
```

### 3. 调整MTU支持巨型帧（重要！）

GigE相机需要9000字节的巨型帧以获得最佳性能。

```bash
# 临时设置MTU
sudo ip link set enP8p1s0 mtu 9000

# 永久设置MTU（NetworkManager）
sudo nmcli connection modify "Wired connection 1" \
    802-3-ethernet.mtu 9000

# 重启连接
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"

# 验证MTU
ip addr show enP8p1s0 | grep mtu
# 应显示: mtu 9000
```

### 4. 调整网络缓冲区（优化性能）

```bash
# 增加接收缓冲区
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400

# 永久配置（添加到/etc/sysctl.conf）
echo "net.core.rmem_max=26214400" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=26214400" | sudo tee -a /etc/sysctl.conf

# 应用配置
sudo sysctl -p
```

---

## 📦 安装 Aravis 依赖

### 1. 安装系统包

```bash
sudo apt update
sudo apt install -y     libaravis-0.8-0     libaravis-0.8-dev     gir1.2-aravis-0.8     aravis-tools     python3-gi     python3-opencv
```

> Jetson 22.04 的官方源直接提供 0.8 系列，省得自己编译。

### 2. （可选）启用 Python 虚拟环境

```bash
source .venv/bin/activate  # 项目根目录已有虚拟环境
pip install PyGObject==3.42.1  # 如需独立管理 Python 依赖
```

### 3. 验证安装

```bash
# 检查 aravis 库版本
apt info libaravis-0.8-0 | grep Version

# 确认 Python 可以导入
python -c "import gi; gi.require_version('Aravis', '0.8'); from gi.repository import Aravis; print('Aravis OK')"
```

如果想从源码编译（调试最新特性），请参考官方仓库：https://github.com/AravisProject/aravis

## 🔍 发现和配置相机

### 1. 使用 arv-tool 扫描设备

```bash
# 查找所有 GigE 设备
arv-tool-0.8 gvcp discover

# 示例输出
# 0 - 'MV-CU013-A0GC' @ 192.168.100.10 (MAC: 04:xx:xx:xx:xx:xx)
```

确认列表里能看到相机型号与IP；如果地址不在规划网段，继续下一步。

### 2. 配置相机 IP

```bash
# 临时修改 IP（断电后恢复出厂配置）
arv-tool-0.8 control --set GEVDeviceIPAddress=192.168.100.10
arv-tool-0.8 control --set GEVSubnetMask=255.255.255.0
arv-tool-0.8 control --set GEVGateway=192.168.100.1
```

> **注意**：部分机型无法直接写寄存器，可以先用 `arv-tool-0.8 gvcp discover --interface enP8p1s0` 锁定网卡，再执行配置。

### 3. 调整 Jumbo Frame / 包延迟

```bash
# 设置数据包大小与延迟（单位：字节 / Ticks）
arv-tool-0.8 control --set GevSCPSPacketSize=9000
arv-tool-0.8 control --set GevSCPD=1000
```

### 4. 快速连通性测试

```bash
ping -c 4 192.168.100.10

# 如果丢包：
#   1. 检查 Jetson 网口 MTU 是否 9000
#   2. 确认相机外接电源稳定
#   3. 更换质量好的千兆网线
```

完成上述步骤后，就可以用 `scripts/test_camera.py` 做帧率和画质验证。

## 🐍 Python SDK测试

### 创建测试脚本

```python
#!/usr/bin/env python3
import gi
import numpy as np

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis

Aravis.enable_interface("gige")
camera = Aravis.Camera.new(None)
stream = camera.create_stream(None, None)
for _ in range(8):
    stream.push_buffer(Aravis.Buffer.new_allocate(camera.get_payload()))

camera.start_acquisition()

buffer = stream.pop_buffer(1_000_000)
if buffer is None:
    raise SystemExit("❌ 超时，没拿到图像")

frame = np.frombuffer(buffer.get_data(), dtype=np.uint8)
frame = frame.reshape(camera.get_height(), camera.get_width())

print(f"✅ 采集成功: {frame.shape}, dtype={frame.dtype}")
stream.push_buffer(buffer)
```

保存为 `~/test_aravis.py` 后执行：

```bash
python ~/test_aravis.py
```

> 如果需要实时预览，可安装 `aravis-tools` 并运行 `arv-viewer-0.8`。

## 📝 相机配置参数

### 推荐配置（Aravis）

```yaml
aravis:
  device_id: null              # 默认选择第一个发现的设备
  pixel_format: "BayerRG8"     # 根据 `arv-tool-0.8 control --get PixelFormat` 调整
  width: 1920
  height: 1080
  packet_size: 9000
  packet_delay: 1000
  exposure_us: 8000
  gain_db: 6.0
  auto_exposure: false
  auto_gain: false
  stream_buffer_count: 8

network:
  interface: "enP8p1s0"
  host_ip: "192.168.100.1"
  camera_ip: "192.168.100.10"
  mtu: 9000
```

- `packet_size` / `packet_delay`：保持链路稳定，视实际网络环境微调。
- `auto_exposure` / `auto_gain`：在极端光照下可先开启自动，调好数据后锁定为手动。
- 如需更高分辨率（4096x3072），确保链路带宽和处理性能跟得上。

## ⚠️ 常见问题

### 1. `arv-tool-0.8 gvcp discover` 看不到相机

- 检查网线、电源、POE
- 确认 Jetson 网口已配置到同一网段 (192.168.100.x)
- 网口 MTU 是否为 9000：`ip addr show enP8p1s0 | grep mtu`
- 防火墙是否阻挡：`sudo ufw allow from 192.168.100.0/24`

### 2. `Aravis.Camera.new` 抛 `CameraError`

- 设备ID填写错误：使用 `arv-tool-0.8 gvcp discover --interface enP8p1s0`
- 相机被其他程序占用：结束所有抓图进程
- 需要 root 权限的寄存器：临时使用 `sudo` 运行排查

### 3. 帧率低 / 采集卡顿

- 确认 `packet_size=9000`、`packet_delay=1000`
- 调整 `stream_buffer_count` 增大缓冲池
- `sudo sysctl -w net.core.rmem_max=26214400`
- 使用 `htop` 检查 CPU/GPU 占用，避免与 TensorRT 抢资源

### 4. 图像偏色或噪点大

- 检查 `pixel_format` 是否匹配真实输出
- 正确设置 Bayer 转换：`BayerRG8` → `cv2.COLOR_BAYER_RG2BGR`
- 合理设置曝光、增益，必要时开启自动模式并记录参数

### 5. `gi.repository.Aravis` 导入失败

- `sudo apt install gir1.2-aravis-0.8 python3-gi`
- Python 虚拟环境需继承系统站点包 (`--system-site-packages`)

---

## ✅ 验证清单

安装完成后，确认以下检查项：

- [ ] Jetson 网口 IP 配置正确：192.168.100.1/24
- [ ] 网口 MTU = 9000
- [ ] 相机 IP = 192.168.100.10，可 ping 通
- [ ] Aravis 依赖安装成功 (`python -c "import gi; gi.require_version('Aravis', '0.8')"`)
- [ ] `scripts/test_camera.py --frames 60` 正常输出图像
- [ ] 帧率满足需求（≥30 FPS）
- [ ] 参数保存到 `config/camera_config.yaml`


---

**创建日期：** 2025-10-08
**相机型号：** MV-CU013-A0GC (1300万像素GigE彩色相机)
**镜头：** 8mm定焦
**作者：** 幽浮喵 (浮浮酱) ฅ'ω'ฅ
