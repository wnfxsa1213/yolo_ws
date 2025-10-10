# Jetson ↔ STM32 通信集成对接说明

面向 Jetson 侧开发同学，说明云台控制板（STM32F407）当前串口协议、握手流程与开发约束。STM32 固件已内置 `jetson_comm` 模块，按本文指引即可完成数据互联。

---

## 1. 硬件与串口参数

| 项目               | 配置                                                         |
|--------------------|--------------------------------------------------------------|
| 串口               | UART1（STM32F407 端口 `PA9/PA10`）                           |
| 速率               | 460800 bps                                                   |
| 数据位/校验/停止位 | 8N1                                                          |
| 供电与地           | 共享 5 V/3.3 V（依 IMU/通讯模块），**务必共地**              |
| 缆线建议           | 双绞线 + 屏蔽，长度 < 30 cm                                   |
| 连接提示           | 线序：Jetson TX → STM32 RX (`PA10`)，Jetson RX ← STM32 TX (`PA9`) |

> ⚠️ 若与 Jetson 主板电平不兼容（如 1.8 V），需加电平转换或串口扩展板。

---

## 2. 帧结构与 CRC

所有通信帧均为二进制，头部固定 `0xAA 0x55`：

```
Byte0  : 0xAA
Byte1  : 0x55
Byte2  : FrameType
Byte3  : PayloadLength (N)
Byte4~ : Payload[0 … N-1]
Byte4+N: CRC (XOR: FrameType ^ PayloadLength ^ Payload bytes)
```

CRC 为简单按字节异或，避免复杂表驱动，Jetson 端实现同样逻辑即可。

---

## 3. STM32 支持的帧类型

| Type  | 方向      | 名称                | Payload | 说明                                                |
|-------|-----------|---------------------|---------|-----------------------------------------------------|
| 0x01  | Jetson→MCU | `JETSON_FRAME_COMMAND` | 6 bytes | 角度命令 + 激光开关 + 标志位                         |
| 0x02  | Jetson→MCU | `JETSON_FRAME_HEARTBEAT` | 0       | 心跳（防超时）                                      |
| 0x10  | Jetson→MCU | `JETSON_FRAME_STATUS_REQUEST` | 0 | 请求 MCU 立即回传状态                               |
| 0x81  | MCU→Jetson | `JETSON_FRAME_STATUS`  | 10 bytes | 周期性状态上报（姿态、模式、信号标志等）            |

### 3.1 命令帧 `0x01`

Payload 6 字节，小端序：

| 偏移 | 类型        | 描述                               | 单位/范围                  |
|------|-------------|------------------------------------|----------------------------|
| 0    | `int16_le`  | `pitch_cdeg`                       | 0.01°，允许 -9000 ~ +9000  |
| 2    | `int16_le`  | `yaw_cdeg`                         | 0.01°，允许 -18000 ~ +18000|
| 4    | `uint8`     | `laser_on`                         | 0/1                        |
| 5    | `uint8`     | `flags`                            | Bit0=1 表示同时刷新心跳；其余为保留 |

> Pitch/Yaw 超出安全范围会被 MCU 自动限幅到 `[−30°, +30°]` / `[−90°, +90°]`。

### 3.2 状态帧 `0x81`

Payload 10 字节，小端序：

| 偏移 | 类型       | 描述                                      | 单位/范围              |
|------|------------|-------------------------------------------|------------------------|
| 0    | `int16_le` | `commanded_pitch_cdeg`（最终舵机命令）    | 0.01°                  |
| 2    | `int16_le` | `commanded_yaw_cdeg`                      | 0.01°                  |
| 4    | `int16_le` | `target_pitch_cdeg`（控制目标）           | 0.01°                  |
| 6    | `int16_le` | `target_yaw_cdeg`                         | 0.01°                  |
| 8    | `uint8`    | `mode` (`ControlMode_t`)                  | `0=INIT,1=RC,2=JETSON,3=FAILSAFE,4=ERROR` |
| 9    | `uint8`    | `flags`                                   | Bit0=RC 有效、Bit1=Jetson 有效、Bit2=激光开、Bit3=Failsafe、Bit4=Error |

状态帧由 MCU 以 20 Hz 周期发送，或在收到 `STATUS_REQUEST` 后立即发送一次。若 TX 忙，则 MCU 会丢弃该次状态帧（不阻塞控制循环）。

---

## 4. 时序与超时要求

| 项目                       | 建议值                                      |
|----------------------------|---------------------------------------------|
| 命令帧发送频率             | **≥ 50 Hz**（建议 50~100 Hz 插值后输出）    |
| 心跳帧/命令帧心跳标志      | **≤ 100 ms** 发送一次，确保 500 ms 超时不触发|
| 控制系统 Jetson 超时阈值   | 500 ms（超时会切换到 FAILSAFE 模式并关闭激光）|
| 状态帧周期                 | 20 Hz（50 ms 一次）                         |

> **务必满足心跳频率**：命令帧的 `flags Bit0` 置 1 即可充当心跳；若 500 ms 内无任何命令/心跳，STM32 将认为 Jetson 离线并回中舵机。

---

## 5. Jetson 侧开发建议

1. **驱动层**  
   - 串口选 `/dev/ttyTHS*` 或 USB 转串口，设置 460800、8N1、无流控。  
   - 推荐使用 `termios` 配合非阻塞/多线程设计，RX 线程持续读取并解析帧。

2. **协议实现**  
   - 建议封装 `encode_command(pitch_deg, yaw_deg, laser_on)`，内部完成角度→centidegree 转换及 CRC。  
   - 解析状态帧时注意小端序，可使用 `struct.unpack('<h', bytes)` 等方式。

3. **数据平滑**  
   - YOLO 输出建议通过滤波/预测后再发往 MCU（减少突变）。  
   - 实际链路中请在发送前对 `pitch/yaw` 做一次指数滑动平均（例如 `out = 0.7 * curr + 0.3 * prev`）并加上 ±0.1° deadband，避免舵机因噪声抖动。  
   - 目标角度超出安全范围 MCU 会限幅，但建议上层在发帧前自行 clamp。

4. **健康监测**  
   - 若收到状态帧 `flags` 中 RC/Jetson 失效、Failsafe/ERROR 置位，应立即记录并在 UI/日志提示。  
   - 未按期收到状态帧（>150 ms）请检查线路或请求复位。

5. **开发流程**  
   - 参考 `docs/jetson_comm_test_plan.md` 中的“串口模拟器测试”逐步验证收发。  
   - Jetson 端可先用 Python 快速打通（示例见下），再移植到 C++/Rust。

### Python 原型示例

```python
import serial
import struct
import time

PORT = "/dev/ttyUSB0"
BAUD = 460800

def make_command(pitch_deg, yaw_deg, laser_on, heartbeat=True):
    pitch_cdeg = int(max(-9000, min(9000, pitch_deg * 100.0)))
    yaw_cdeg = int(max(-18000, min(18000, yaw_deg * 100.0)))
    flags = 0x01 if heartbeat else 0x00
    payload = struct.pack('<hhBB', pitch_cdeg, yaw_cdeg, int(bool(laser_on)), flags)
    frame_type = 0x01
    length = len(payload)
    crc = frame_type ^ length
    for b in payload:
        crc ^= b
    return bytes([0xAA, 0x55, frame_type, length]) + payload + bytes([crc])

with serial.Serial(PORT, BAUD, timeout=0.05) as ser:
    while True:
        ser.write(make_command(10.0, 0.0, False))
        time.sleep(0.02)  # 50 Hz
        data = ser.read(128)
        if data:
            print("RX:", data.hex())
```

---

## 6. 故障与回退策略

| 场景                               | MCU 行为                                         | Jetson 建议操作                              |
|------------------------------------|--------------------------------------------------|----------------------------------------------|
| Jetson 超时 (>500 ms)             | 切 FAILSAFE：舵机回中、激光关闭                   | 尽快恢复通信，必要时重发状态请求             |
| 命令 CRC 错误/帧类型不支持         | 该帧丢弃                                         | 检查编码逻辑及长度                            |
| USART 出错（溢出/噪声）            | MCU 重启 RX 中断，状态帧可暂时丢失                | 监控状态帧间隔，必要时降低波特率/加屏蔽       |
| MCU 报告 ERROR 模式               | 停止输出、等待人工复位                            | 查询 `ControlSystem_GetLastError()`，重新初始化|

---

## 7. 参考资料

- `App/Src/jetson_comm.c`：STM32 端实现细节
- `docs/jetson_comm_code_review.md`：协议审查要点
- `docs/jetson_comm_test_plan.md`：步骤化测试流程
- `F407_Development_V2.md` 第 7.3 节：整体架构与进度说明

如需扩展帧类型或增加姿态反馈（IMU 数据等），请先与 MCU 侧协调保留字段与版本号，再统一升级。
