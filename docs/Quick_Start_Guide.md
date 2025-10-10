# 智能云台追踪系统 - 快速开始指南

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | 快速开始指南 |
| **版本** | v1.0 |
| **更新日期** | 2025-10 |
| **预计时间** | 2-3小时（首次搭建）|
| **难度等级** | ⭐⭐⭐ (中等) |

---

## 🎯 开始之前

### 你将学到

- ✅ 硬件连接与配置
- ✅ 软件环境搭建
- ✅ ELRS接收机绑定
- ✅ 系统首次启动
- ✅ 基础功能测试

### 前置条件

```yaml
必备技能:
  - 基础Linux命令行操作
  - Python环境配置经验
  - 嵌入式开发基础（可选）

必备工具:
  - 螺丝刀、万用表、烙铁（如需）
  - ST-Link调试器
  - USB转TTL模块（调试用）
```

---

## 1️⃣ 硬件清单与检查

### 1.1 完整硬件清单

```
┌─ 核心计算平台 ─────────────────────────┐
│ [ ] Jetson Orin NX Super 16GB (含载板)  │
│ [ ] NVMe SSD (256GB+, 推荐) 或 SD卡     │
│ [ ] 散热风扇 + 散热片                    │
│ [ ] 电源适配器 (19V 3A 或载板供电)      │
└────────────────────────────────────────┘

┌─ 视觉系统 ─────────────────────────────┐
│ [ ] 海康威视工业相机 (USB3.0)           │
│ [ ] 相机镜头 (8mm焦距推荐)              │
│ [ ] USB3.0数据线 (0.5m-1m)              │
└────────────────────────────────────────┘

┌─ 控制系统 ─────────────────────────────┐
│ [ ] STM32H750VBT6开发板                 │
│ [ ] ST-Link V2/V3调试器                 │
│ [ ] 杜邦线 (公对公/公对母)              │
└────────────────────────────────────────┘

┌─ 遥控系统 ─────────────────────────────┐
│ [ ] ExpressLRS接收机 (EP1/EP2)          │
│ [ ] ExpressLRS遥控器 (RadioMaster等)    │
│ [ ] 遥控器天线                          │
└────────────────────────────────────────┘

┌─ 执行机构 ─────────────────────────────┐
│ [ ] 数字舵机 x2 (15kg·cm+)             │
│ [ ] 舵机延长线                          │
│ [ ] 云台机械结构                        │
│ [ ] 激光笔模块 (5mW红光)               │
└────────────────────────────────────────┘

┌─ 电源系统 ─────────────────────────────┐
│ [ ] 12V电源 (5A+)                       │
│ [ ] DC-DC降压模块 (12V→5V 5A)          │
│ [ ] 接线端子                            │
└────────────────────────────────────────┘

┌─ 调试工具 ─────────────────────────────┐
│ [ ] USB转TTL模块 (CP2102/FT232RL)      │
│ [ ] 万用表                              │
│ [ ] 逻辑分析仪 (可选)                   │
│ [ ] 示波器 (可选)                       │
└────────────────────────────────────────┘
```

### 1.2 硬件检查

```bash
# 在开始之前，逐一检查：

[硬件完整性]
[ ] 所有模块外观无损
[ ] 接口无锈蚀/污损
[ ] 线材无破损

[电源测试]
[ ] 12V电源输出正常 (万用表测量)
[ ] DC-DC模块输出5V稳定
[ ] 无短路、发热异常

[Jetson测试]
[ ] 插入SD卡能正常启动
[ ] 风扇运转正常
[ ] 可通过SSH连接
```

---

## 2️⃣ 硬件连接

### 2.1 连接拓扑图

```
                ┌──────────────────┐
                │   12V 电源       │
                └────┬─────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
    ┌───▼────┐   ┌──▼──────┐  ┌─▼──────┐
    │ Jetson │   │ DC-DC   │  │舵机电源│
    │ 5V/4A  │   │12V→5V   │  │  5V    │
    └───┬────┘   └──┬──────┘  └─┬──────┘
        │           │            │
        │           │         ┌──▼──────┐
        │           │         │ 舵机x2  │
        │           │         └─────────┘
        │           │
    ┌───▼───────────▼─────┐
    │   STM32H750          │
    │                      │
    │  PA9  ◄─► Jetson TX │ UART1 (460800)
    │  PA10 ◄─► Jetson RX │
    │                      │
    │  PB11 ◄─── ELRS RX  │ UART3 (420000)
    │                      │
    │  PA0  ───► Pitch舵机 │ TIM2_CH1 (50Hz)
    │  PA1  ───► Yaw舵机   │ TIM2_CH2 (50Hz)
    │                      │
    │  PC6  ───► 激光笔    │ TIM3_CH1 (PWM)
    └──────────────────────┘
```

### 2.2 分步连接指南

#### Step 1: STM32H750 连接

```
1. 串口1 (Jetson通信):
   STM32 PA9  (TX) ──► Jetson RX (UART1)
   STM32 PA10 (RX) ──► Jetson TX (UART1)
   STM32 GND      ──► Jetson GND

2. 串口3 (CRSF接收机):
   STM32 PB11 (RX) ──► ELRS RX
   STM32 GND       ──► ELRS GND
   STM32 5V        ──► ELRS 5V (如果接收机需要供电)

3. 舵机PWM:
   STM32 PA0 (TIM2_CH1) ──► Pitch舵机信号线 (橙色)
   STM32 PA1 (TIM2_CH2) ──► Yaw舵机信号线 (橙色)
   舵机红线 → 5V电源
   舵机棕线 → GND

4. 激光笔:
   STM32 PC6 (TIM3_CH1) ──► 激光笔控制引脚
   激光笔+ → 5V
   激光笔- → GND

5. 调试接口:
   ST-Link SWDIO ──► PA13
   ST-Link SWCLK ──► PA14
   ST-Link GND   ──► GND
   ST-Link 3.3V  ──► 3.3V (可选)

6. 调试串口 (可选):
   STM32 PA2 (UART2_TX) ──► USB-TTL RX
```

#### Step 2: Jetson连接

```
1. 相机:
   海康相机USB3.0 ──► Jetson USB3.0口

2. 串口 (与STM32):
   已在Step 1连接

3. 电源:
   12V电源 ──► Jetson DC输入 (或通过载板5V供电)

4. 网络 (首次配置):
   网线/WiFi模块 连接路由器
```

#### Step 3: ELRS接收机连接

```
1. 供电:
   5V ──► ELRS 5V
   GND ──► ELRS GND

2. CRSF输出:
   ELRS TX (CRSF) ──► STM32 PB11 (UART3_RX)
   注意: 只需连接接收机TX到STM32 RX，单向通信

3. 天线:
   连接接收机天线 (注意极性)
```

### 2.3 连接检查清单

```bash
# 通电前检查（重要！）
[ ] 所有VCC与GND无短路 (万用表蜂鸣模式)
[ ] 信号线未接错电源引脚
[ ] 舵机供电独立（不与STM32共地可能导致问题）
[ ] ST-Link连接正确

# 上电后检查
[ ] STM32电源LED亮起
[ ] Jetson启动正常（LED指示）
[ ] 舵机有轻微响应（上电自检）
[ ] ELRS接收机LED闪烁（未绑定时快闪）
```

---

## 3️⃣ 软件环境搭建

### 3.1 Jetson环境配置

#### 3.1.1 基础环境

```bash
# 1. SSH连接到Jetson
ssh username@jetson-ip

# 2. 更新系统
sudo apt update && sudo apt upgrade -y

# 3. 安装基础工具
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    libusb-1.0-0-dev

# 4. 确认JetPack版本
sudo apt-cache show nvidia-jetpack
```

#### 3.1.2 安装uv (Python包管理器)

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到PATH (如果未自动添加)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 验证安装
uv --version
```

#### 3.1.3 创建虚拟环境

```bash
# 进入项目目录
cd ~/yolo_ws

# 创建虚拟环境 (⚠️ 必须使用 --system-site-packages)
# 原因：继承主环境的GPU优化库（PyTorch 2.5.0, OpenCV 4.10.0）
uv venv --python 3.10 --system-site-packages

# 激活虚拟环境
source .venv/bin/activate

# 验证Python版本和GPU库
python --version  # 应该显示 Python 3.10.12
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}, CUDA模块: {hasattr(cv2, \"cuda\")}')"
```

#### 3.1.4 安装Python依赖

```bash
# 确保在虚拟环境中
cd ~/yolo_ws

# 如果有pyproject.toml，直接sync
uv sync

# 或手动安装项目依赖（⚠️ 不要安装 numpy, opencv-python, torch）
uv pip install \
    pyserial \
    pyyaml \
    ultralytics

# 验证GPU库可用性（这些库来自主环境）
python << 'EOF'
import numpy as np
import cv2
import torch
print(f"✓ NumPy: {np.__version__}")
print(f"✓ OpenCV: {cv2.__version__} (CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0})")
print(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
EOF
```

#### 3.1.5 海康相机SDK安装

```bash
# 1. 下载海康SDK (MVS Python版)
# 访问: https://www.hikrobotics.com/cn/machinevision/service/download

# 2. 解压并安装
tar -xvf MVS-*-Python-*.tar.gz
cd MVS-*-Python-*/

# 3. 运行安装脚本
sudo ./install.sh

# 4. 测试相机
python examples/MvImport/MvCameraControl_class.py
```

### 3.2 STM32环境配置

#### 3.2.1 STM32CubeIDE安装 (在开发电脑上)

```bash
# Linux安装
# 1. 下载 STM32CubeIDE from st.com
wget https://www.st.com/stm32cubeide-lin

# 2. 解压并安装
chmod +x *.sh
sudo ./st-stm32cubeide_*.sh

# 3. 启动IDE
stm32cubeide
```

#### 3.2.2 创建STM32工程

```
1. File → New → STM32 Project
2. 选择芯片: STM32H750VBT6
3. 项目名称: target_tracker_h750
4. 目标语言: C

CubeMX配置:
  - 时钟: 400MHz (参考H750_Development_V2.md)
  - UART1: 460800bps, Async, 8N1
  - UART3: 420000bps, Async, 8N1
  - TIM2: PWM, 50Hz, Channel 1&2
  - TIM3: PWM, 1kHz, Channel 1
  - ADC1: 12bit, Channel 18&19
  - DMA: UART1 RX, UART3 RX, ADC1

  - FreeRTOS: Enable, CMSIS_V2
    - Task1: ControlTask (1024 words, Priority 4)
    - Task2: CommTask (512 words, Priority 3)
    - Task3: MonitorTask (512 words, Priority 1)

5. 生成代码: Generate Code
```

#### 3.2.3 编译与烧录

```bash
# 1. 编译工程
Project → Build All (Ctrl+B)

# 2. 连接ST-Link

# 3. 烧录程序
Run → Debug (F11)
# 或
Run → Run (Ctrl+F11)

# 4. 查看串口输出（调试）
# 使用minicom或其他串口工具
minicom -D /dev/ttyUSB0 -b 115200
```

---

## 4️⃣ ELRS接收机绑定

### 4.1 遥控器配置

```
EdgeTX/OpenTX配置:

1. 模型设置 (MDL)
   - 外部RF: ExpressLRS
   - 更新率: 150Hz (推荐)
   - 遥测: 开启
   - 功率: Auto / 100mW

2. 输入配置 (INPUTS)
   - I1: Ail (副翼) → CH1
   - I2: Ele (升降) → CH2
   - I3: Thr (油门) → CH3
   - I4: Rud (方向) → CH4

3. 混控配置 (MIXES)
   - CH1: I1 (Ail) → Yaw轴
   - CH2: I2 (Ele) → Pitch轴
   - CH5: SA (3段开关) → 模式切换
   - CH6: SB (2段开关) → 激光开关

4. 输出配置 (OUTPUTS)
   - CH1-CH8: 保持默认范围 (-100% ~ +100%)
```

### 4.2 绑定流程

```
方法1: 自动绑定 (推荐)
1. 接收机上电
2. 遥控器: MDL → [BIND] → 确认
3. 接收机进入绑定模式（LED快闪）
4. 等待10秒，自动完成绑定
5. 绑定成功：接收机LED常亮/慢闪

方法2: 手动绑定
1. 接收机断电
2. 按住接收机BIND按钮不放
3. 接收机上电，继续按住3秒
4. 释放按钮，进入绑定模式
5. 遥控器进入绑定模式
6. 等待完成
```

### 4.3 绑定验证

```
测试步骤:
1. 遥控器打开，接收机上电
2. LED应该常亮或慢闪（信号正常）
3. 使用串口监控STM32输出：
   - 应该看到通道数据输出
   - CH1-CH4摇杆值应该响应
   - 拨动开关，CH5-CH6应该变化

测试命令 (STM32串口输出):
> CH1:992 CH2:992 CH3:172 CH4:992 CH5:172 CH6:992
> (中位值约992, 最小172, 最大1811)
```

---

## 5️⃣ 首次启动

### 5.1 启动顺序

```bash
# 1. 硬件上电顺序
┌────────────────────────────┐
│ 1. STM32上电               │ (LED应该亮起)
│    → 等待5秒 (初始化)       │
├────────────────────────────┤
│ 2. ELRS接收机上电          │ (LED闪烁/常亮)
│    → 确认遥控器信号正常     │
├────────────────────────────┤
│ 3. Jetson上电              │ (启动约30-60秒)
│    → 等待完全启动           │
├────────────────────────────┤
│ 4. 相机连接                │ (USB3.0接入)
└────────────────────────────┘

# 2. SSH连接Jetson
ssh username@jetson-ip

# 3. 激活虚拟环境
source ~/envs/ultra_uv/bin/activate

# 4. 启动主程序
cd ~/yolo_ws
python src/main.py

# 预期输出:
# [INFO] Initializing camera...
# [INFO] Camera opened successfully
# [INFO] Loading YOLOv8 model...
# [INFO] Model loaded: 30.5 FPS
# [INFO] Serial port opened: /dev/ttyTHS1
# [INFO] System ready!
```

### 5.2 系统自检

```bash
# STM32状态检查（通过串口2调试输出）
minicom -D /dev/ttyUSB0 -b 115200

# 预期输出:
# =========================
# STM32H750 Target Tracker
# Version: 1.0
# =========================
# [INIT] Servo initialized
# [INIT] CRSF initialized
# [INIT] UART1 ready
# [OK] System ready!
# Mode: INIT
# RC Signal: Valid / Invalid
# Jetson Signal: Invalid (启动中)
```

---

## 6️⃣ 基础功能测试

### 6.1 测试清单

```
[ ] Test 1: 遥控器控制测试
[ ] Test 2: 舵机响应测试
[ ] Test 3: 模式切换测试
[ ] Test 4: 相机图像测试
[ ] Test 5: YOLO检测测试
[ ] Test 6: 自动追踪测试
[ ] Test 7: 失控保护测试
```

### 6.2 Test 1: 遥控器控制

```
目标: 验证RC模式正常工作

步骤:
1. 确认遥控器开机，接收机信号正常
2. STM32应该自动进入RC_CONTROL模式
3. 拨动遥控器摇杆:
   - 左右摇杆 → Yaw轴舵机转动
   - 上下摇杆 → Pitch轴舵机转动
4. 拨动CH6开关 → 激光开关

预期结果:
✅ 舵机跟随摇杆平滑运动
✅ 无抖动、卡顿
✅ 响应延迟 <50ms
✅ LED指示快闪 (RC模式)

故障排查:
❌ 舵机不动:
   - 检查PWM引脚连接
   - 检查舵机供电
   - 用示波器测PA0/PA1波形

❌ 抖动严重:
   - 降低轨迹规划加速度参数
   - 检查电源纹波
   - 增加滤波系数
```

### 6.3 Test 2: 舵机响应

```
目标: 测试舵机角度范围和精度

步骤:
1. 遥控器摇杆推到最大
   → 舵机转到最大角度 (约+90°)
2. 摇杆回中
   → 舵机回到中位 (0°)
3. 摇杆推到最小
   → 舵机转到最小角度 (约-90°)

测量精度:
- 使用量角器/手机陀螺仪APP测量
- 误差应 <±2°

记录数据:
| 摇杆位置 | 预期角度 | 实际角度 | 误差 |
|---------|---------|---------|------|
| 最大     | +90°    | ____°   | __°  |
| 中位     | 0°      | ____°   | __°  |
| 最小     | -90°    | ____°   | __°  |
```

### 6.4 Test 3: 模式切换

```
目标: 验证三模式自动切换

测试场景1: RC → Jetson
1. 初始RC模式 (遥控器开机)
2. 关闭遥控器
3. 等待1秒，STM32应该切换到Jetson模式
4. LED指示变为慢闪

测试场景2: Jetson → RC
1. Jetson模式运行中
2. 打开遥控器
3. STM32应立即切换到RC模式 (<100ms)
4. 遥控器可立即接管控制

测试场景3: 失控保护
1. 同时关闭遥控器和Jetson程序
2. STM32应该进入Failsafe模式
3. 舵机自动归中
4. 激光关闭
5. LED SOS闪烁
```

### 6.5 Test 4: 相机图像

```bash
# 测试相机采图
cd ~/yolo_ws
python tests/test_camera.py

# 预期输出:
# [INFO] Opening camera...
# [INFO] Camera opened: 1920x1080 @ 30fps
# [INFO] Frame captured: (1080, 1920, 3)
# [INFO] Saving test image...
# [INFO] Image saved: test_frame.jpg

# 检查图像质量:
# - 曝光正常 (不过曝/欠曝)
# - 焦距清晰
# - 无明显畸变
```

### 6.6 Test 5: YOLO检测

```bash
# 测试YOLO检测
python tests/test_detection.py

# 预期输出:
# [INFO] Loading YOLOv8 model...
# [INFO] Model loaded in 2.3s
# [INFO] Running inference...
# [INFO] FPS: 32.5
# [INFO] Detected: person (0.89), car (0.76)

# 检查指标:
# - FPS ≥ 25
# - 检测准确率 >80%
# - 延迟 <50ms
```

### 6.7 Test 6: 自动追踪

```
目标: 端到端追踪功能测试

步骤:
1. 启动主程序
2. 关闭遥控器 (切换到Jetson模式)
3. 在相机视野中移动物体 (人/物体)
4. 观察云台跟随运动

预期结果:
✅ 云台能跟随目标平滑运动
✅ 目标移出视野后停止
✅ 多目标时追踪最近的
✅ 无明显延迟和抖动

性能记录:
- 端到端延迟: ____ms (目标<50ms)
- 追踪精度: 目标能保持在视野中心±____像素
- 丢失恢复: 目标重新进入视野后____秒恢复追踪
```

### 6.8 Test 7: 失控保护

```
测试失控保护机制:

场景1: RC信号丢失
1. RC模式运行中
2. 关闭遥控器
3. 1秒内应该切换到Jetson模式 (如果Jetson在线)
4. 或进入Failsafe (如果Jetson离线)

场景2: Jetson通信超时
1. Jetson模式运行中
2. 停止Jetson程序 (Ctrl+C)
3. 500ms内应该切换到RC模式 (如果RC在线)
4. 或进入Failsafe

场景3: 温度过高保护
1. 模拟高温 (加热STM32或修改代码阈值测试)
2. 温度 >85°C应该触发保护
3. 舵机停止运动
4. LED报警
```

---

## 7️⃣ 常见问题排查

### 7.1 Jetson相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 相机打不开 | USB权限/驱动问题 | `sudo usermod -aG video $USER`<br>重新登录 |
| YOLO推理慢 | 未使用TensorRT | 转换模型为.engine格式 |
| 串口无法打开 | 权限问题 | `sudo usermod -aG dialout $USER`<br>`sudo chmod 666 /dev/ttyTHS1` |
| 虚拟环境无torch | Jetson需要特殊安装 | 使用系统自带PyTorch<br>或参考NVIDIA官方指南 |

### 7.2 STM32相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 无法烧录 | ST-Link连接问题 | 检查SWDIO/SWCLK/GND连接<br>尝试按住NRST烧录 |
| 舵机不响应 | PWM配置错误 | 用示波器检查PA0/PA1波形<br>确认50Hz, 500-2500us脉宽 |
| CRSF无数据 | 波特率/接线错误 | 确认420000bps<br>检查PB11连接 |
| 串口乱码 | 波特率不匹配 | PC端设置115200/460800/420000 |

### 7.3 ELRS相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 无法绑定 | 频段不匹配 | 确认遥控器和接收机同频段<br>(2.4G/915M) |
| 信号不稳定 | 天线问题/干扰 | 检查天线连接<br>远离WiFi路由器 |
| 延迟高 | 更新率设置低 | 设置150Hz或更高 |
| 链路质量差 | 距离过远/遮挡 | 提高发射功率<br>改善天线位置 |

---

## 8️⃣ 下一步

### 8.1 性能优化

```
1. 参数调优:
   - 轨迹规划参数 (config/system_config.yaml)
   - PID参数 (如启用)
   - 检测置信度阈值

2. 模型优化:
   - 转换为TensorRT FP16
   - 考虑INT8量化 (需要标定数据)

3. 控制优化:
   - 根据实际测试调整加速度/速度限制
   - 优化滤波参数
```

### 8.2 功能扩展

```
短期:
- [ ] 添加Web监控界面
- [ ] 实现数据记录功能
- [ ] 多目标优先级管理

长期:
- [ ] 深度学习模型训练
- [ ] 云台360°旋转升级
- [ ] 多设备联动
```

### 8.3 文档参考

```
详细文档:
📘 docs/H750_Development_V2.md       - H750开发详解
📘 docs/System_Architecture_V2.md    - 系统架构设计
📘 docs/CRSF_Protocol_Reference.md   - CRSF协议详解

代码示例:
📁 src/                              - Python源代码
📁 stm32_firmware/                   - STM32固件
📁 tests/                            - 测试脚本
```

---

## 9️⃣ 获取帮助

### 9.1 社区资源

```
ExpressLRS:
  - Discord: https://discord.gg/expresslrs
  - 论坛: https://www.expresslrs.org/

Jetson:
  - 论坛: https://forums.developer.nvidia.com/
  - GitHub: NVIDIA-AI-IOT

STM32:
  - 论坛: https://community.st.com/
  - 中文论坛: https://www.stmcu.org.cn/
```

### 9.2 调试技巧

```
1. 逐步调试法:
   - 先测试单个模块
   - 再测试模块间通信
   - 最后测试整体系统

2. 日志记录:
   - Jetson: 使用Python logging模块
   - STM32: 通过UART2输出调试信息

3. 可视化调试:
   - 使用Web界面查看实时状态
   - 逻辑分析仪抓取通信波形
   - 示波器验证PWM输出
```

---

## 🎉 恭喜！

如果你完成了所有测试，系统应该已经正常运行了！

**接下来你可以：**
- 🎮 尝试不同场景的追踪测试
- ⚙️ 优化参数以获得最佳性能
- 📊 记录测试数据，分析性能指标
- 🚀 探索更多高级功能

**遇到问题？**
- 📖 查阅详细技术文档
- 🔧 使用故障排查指南
- 💬 在社区寻求帮助

---

**文档版本**: v1.0
**最后更新**: 2025-10
**作者**: 幽浮喵 (浮浮酱) ฅ'ω'ฅ

**祝你开发愉快！φ(≧ω≦*)♪**

---

**END OF DOCUMENT**
