# 串口通信模块 (Serial Communication Module)

## 模块职责

与STM32H750的串口通信，包括协议编解码和数据收发。

## 待实现文件

### protocol.py
- **ProtocolEncoder 类**
  - `encode_target_data(detected, pitch, yaw, distance)` - 编码目标数据
  - `encode_heartbeat()` - 编码心跳包
  - CRC8校验计算

- **ProtocolDecoder 类**
  - `feed(data)` - 喂入接收数据
  - `get_decoded()` - 获取解码后的数据包
  - 状态机解析（等待帧头 → 接收数据 → CRC校验）
  - 返回格式：`{'mode': int, 'current_pitch': float, 'current_yaw': float, 'temperature': float}`

### communicator.py
- **SerialCommunicator 类**
  - 串口初始化 (`/dev/ttyTHS0`, 460800, 8N1)
  - `send_target(detected, pitch, yaw, distance)` - 发送目标数据（异步）
  - `receive_feedback()` - 接收反馈数据（异步）
  - 发送队列 (asyncio.Queue)
  - 接收队列 (asyncio.Queue)
  - 超时处理 (500ms)
  - 自动重连

## 协议格式

### 下行指令 (Jetson → H750)
```
帧头 | 长度 | 类型 | 数据载荷      | CRC8
0xAA | 0x0E | 0x01 | [14字节数据] | CRC
0x55 |      |      |              |

数据载荷 (14字节):
- target_detected: uint8_t (1字节)
- pitch: float (4字节)
- yaw: float (4字节)
- distance: uint16_t (2字节)
- reserved: uint8_t[3] (3字节保留)
```

### 上行反馈 (H750 → Jetson)
```
帧头 | 长度 | 类型 | 数据载荷      | CRC8
0xAA | 0x0E | 0x02 | [14字节数据] | CRC
0x55 |      |      |              |

数据载荷 (14字节):
- mode: uint8_t (0=待机, 1=RC, 2=Jetson, 3=Failsafe)
- current_pitch: float (4字节)
- current_yaw: float (4字节)
- temperature: uint8_t (1字节, 单位°C)
- reserved: uint8_t[4] (4字节保留)
```

## 依赖
- pyserial
- asyncio

## 状态
⏳ 待实现
