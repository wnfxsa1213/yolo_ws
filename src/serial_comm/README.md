# 串口通信模块 (Serial Communication Module)

## 模块职责

与 STM32F407 云台控制板通过 460800bps 串口互通：编码角度命令、维持心跳、解析状态帧，并提供线程化的通信总线。

## 目录结构

| 文件 | 职责 |
|------|------|
| `protocol.py` | 帧编码器/解码器，封装 XOR CRC、命令帧、心跳帧、状态帧解析 |
| `communicator.py` | 串口驱动：发送队列、接收线程、心跳保持、自动重连 |
| `__init__.py` | 模块导出 |

## 协议速览（Jetson_comm_integration v1.0）

```
AA 55 | FrameType | PayloadLen | Payload | CRC
```

CRC = 所有 `FrameType`, `PayloadLen`, `Payload` 字节按位 XOR。

| Type | 方向 | Payload | 描述 |
|------|------|---------|------|
| `0x01` | Jetson→MCU | 6B | `pitch_cdeg`, `yaw_cdeg`（int16, 0.01°），`laser_on`，`flags`(Bit0=心跳) |
| `0x02` | Jetson→MCU | 0B | 心跳帧 |
| `0x10` | Jetson→MCU | 0B | 状态立即回报请求 |
| `0x81` | MCU→Jetson | 10B | 姿态/模式反馈（四个 int16 角度 + mode + flags） |

角度 clamp：
- Pitch：`[-90°, +90°]`（协议范围 `[-9000, +9000]` cdeg）
- Yaw：`[-180°, +180°]`（协议范围 `[-18000, +18000]` cdeg）

## 使用示例

```python
from serial_comm import SerialCommunicator

comm = SerialCommunicator("/dev/ttyTHS1")
comm.start()
comm.send_command(pitch_deg=5.0, yaw_deg=-12.3, laser_on=True)

status = comm.get_latest_status()
if status:
    print(status.mode, status.flags)
```

`last_status_age()` 可用于监控 MCU 心跳，超出 0.15s 需要补发 `request_status()` 或检查线路。
