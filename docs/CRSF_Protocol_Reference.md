# CRSF协议参考手册

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | CRSF (Crossfire) 协议参考手册 |
| **适用范围** | ExpressLRS / TBS Crossfire |
| **版本** | v1.0 |
| **更新日期** | 2025-10 |
| **作者** | 幽浮喵 (浮浮酱) ฅ'ω'ฅ |
| **目标读者** | 嵌入式开发工程师 |

---

## 1️⃣ 协议概述

### 1.1 什么是CRSF？

**CRSF (Crossfire)** 是由Team BlackSheep (TBS)开发的高速遥控链路协议，被广泛应用于FPV无人机和RC模型领域。**ExpressLRS (ELRS)** 采用了CRSF作为接收机与飞控之间的通信协议。

### 1.2 协议特点

```yaml
优势:
  - 高更新率: 支持50Hz到500Hz
  - 低延迟: 端到端延迟<10ms
  - 高通道数: 16个遥控通道
  - 双向通信: 遥测数据回传
  - 可靠性高: CRC8校验
  - 易于扩展: 支持自定义帧类型

应用场景:
  - FPV无人机飞控通信
  - RC模型控制
  - 机器人遥控
  - 自动化设备远程控制
```

### 1.3 物理层参数

```yaml
接口: UART (串口)
波特率: 420000 bps (固定)
数据格式: 8N1 (8数据位, 无校验位, 1停止位)
电平: 3.3V TTL (兼容5V容忍)
最大帧长: 64 bytes
```

---

## 2️⃣ 帧格式详解

### 2.1 通用帧结构

```
┌──────────┬──────────┬──────────┬───────────┬──────────┐
│ Device   │ Frame    │ Frame    │  Payload  │  CRC8    │
│ Address  │ Length   │ Type     │  Data     │          │
│ (1 byte) │ (1 byte) │ (1 byte) │ (N bytes) │ (1 byte) │
└──────────┴──────────┴──────────┴───────────┴──────────┘

总长度 = Device Address(1) + Frame Length(1) + Payload Data(N) + CRC8(1)
       = 2 + N + 1 bytes
```

#### 字段说明

| 字段 | 长度 | 说明 |
|------|------|------|
| **Device Address** | 1 byte | 设备地址，标识发送方或接收方 |
| **Frame Length** | 1 byte | 帧长度，不包含地址和长度字节本身 |
| **Frame Type** | 1 byte | 帧类型，定义数据内容格式 |
| **Payload Data** | N bytes | 有效数据，长度由Frame Length决定 |
| **CRC8** | 1 byte | 校验和，对Type+Data计算 |

**注意事项：**
- Frame Length = sizeof(Type) + sizeof(Data) + sizeof(CRC8) = N + 2
- CRC8计算范围：从Frame Type到Payload Data末尾（不包括Address和Length）

### 2.2 设备地址定义

```c
/* CRSF设备地址定义 */
#define CRSF_ADDRESS_BROADCAST          0x00    // 广播地址
#define CRSF_ADDRESS_USB                0x10    // USB设备
#define CRSF_ADDRESS_TBS_CORE_PNP_PRO   0x80    // TBS Core PNP Pro
#define CRSF_ADDRESS_RESERVED1          0x8A    // 保留地址
#define CRSF_ADDRESS_CURRENT_SENSOR     0xC0    // 电流传感器
#define CRSF_ADDRESS_GPS                0xC2    // GPS模块
#define CRSF_ADDRESS_TBS_BLACKBOX       0xC4    // TBS黑匣子
#define CRSF_ADDRESS_FLIGHT_CONTROLLER  0xC8    // 飞控/STM32
#define CRSF_ADDRESS_RESERVED2          0xCA    // 保留地址
#define CRSF_ADDRESS_RACE_TAG           0xCC    // 竞赛标签
#define CRSF_ADDRESS_RADIO_TRANSMITTER  0xEA    // 遥控发射机
#define CRSF_ADDRESS_RECEIVER           0xEC    // 接收机
#define CRSF_ADDRESS_TRANSMITTER        0xEE    // 发射模块
```

**常用地址：**
- **0xC8 (Flight Controller)**: STM32H750作为飞控接收CRSF数据
- **0xEC (Receiver)**: ELRS接收机发送遥控数据

### 2.3 帧类型定义

```c
/* CRSF帧类型定义 */
#define CRSF_FRAMETYPE_GPS              0x02    // GPS位置
#define CRSF_FRAMETYPE_VARIO            0x07    // 气压计/爬升率
#define CRSF_FRAMETYPE_BATTERY_SENSOR   0x08    // 电池传感器
#define CRSF_FRAMETYPE_BARO_ALTITUDE    0x09    // 气压高度
#define CRSF_FRAMETYPE_LINK_STATISTICS  0x14    // 链路统计（重要）
#define CRSF_FRAMETYPE_RC_CHANNELS      0x16    // RC通道数据（最重要）
#define CRSF_FRAMETYPE_SUBSET_RC_CHANNELS 0x17  // 子集RC通道
#define CRSF_FRAMETYPE_LINK_RX_ID       0x1C    // 接收机ID
#define CRSF_FRAMETYPE_LINK_TX_ID       0x1D    // 发射机ID
#define CRSF_FRAMETYPE_ATTITUDE         0x1E    // 姿态（俯仰/滚转/偏航）
#define CRSF_FRAMETYPE_FLIGHT_MODE      0x21    // 飞行模式
```

**关键帧类型：**
- **0x16 (RC_CHANNELS)**: 遥控通道数据，包含所有16个通道
- **0x14 (LINK_STATISTICS)**: 链路质量、RSSI等信息

---

## 3️⃣ RC通道数据帧 (0x16)

### 3.1 帧格式

这是最重要的帧类型，包含16个遥控通道的数据。

```
帧结构:
┌─────┬─────┬─────┬──────────────────┬─────┐
│ 0xC8│ 0x18│ 0x16│  22 bytes data   │ CRC │
│ Addr│ Len │ Type│  (16 channels)   │     │
└─────┴─────┴─────┴──────────────────┴─────┘

完整帧长度: 26 bytes
  - Address: 1 byte (0xC8)
  - Length: 1 byte (0x18 = 24)
  - Type: 1 byte (0x16)
  - Data: 22 bytes (通道数据)
  - CRC8: 1 byte
```

### 3.2 通道数据编码

**关键特性：**
- 每个通道：11 bit (范围 0-2047)
- 16个通道：16 × 11 = 176 bits = 22 bytes
- 编码方式：紧密打包（bit-packed）

**通道值范围：**
```c
#define CRSF_CHANNEL_VALUE_MIN  172     // 最小值 (对应PWM 988us)
#define CRSF_CHANNEL_VALUE_MID  992     // 中位值 (对应PWM 1500us)
#define CRSF_CHANNEL_VALUE_MAX  1811    // 最大值 (对应PWM 2012us)
```

### 3.3 通道解包代码

```c
/**
 * @brief 解析CRSF RC通道数据
 * @param payload: 22字节通道数据
 * @param channels: 输出通道数组 (16个uint16_t)
 */
void crsf_parse_rc_channels(uint8_t *payload, uint16_t *channels) {
    // 使用位操作提取11bit通道值
    // 每个通道占用11bit，紧密排列

    channels[0]  = (uint16_t)((payload[0]    | payload[1]  << 8)                     & 0x07FF);
    channels[1]  = (uint16_t)((payload[1]>>3 | payload[2]  << 5)                     & 0x07FF);
    channels[2]  = (uint16_t)((payload[2]>>6 | payload[3]  << 2 | payload[4]<<10)    & 0x07FF);
    channels[3]  = (uint16_t)((payload[4]>>1 | payload[5]  << 7)                     & 0x07FF);
    channels[4]  = (uint16_t)((payload[5]>>4 | payload[6]  << 4)                     & 0x07FF);
    channels[5]  = (uint16_t)((payload[6]>>7 | payload[7]  << 1 | payload[8]<<9)     & 0x07FF);
    channels[6]  = (uint16_t)((payload[8]>>2 | payload[9]  << 6)                     & 0x07FF);
    channels[7]  = (uint16_t)((payload[9]>>5 | payload[10] << 3)                     & 0x07FF);
    channels[8]  = (uint16_t)((payload[11]   | payload[12] << 8)                     & 0x07FF);
    channels[9]  = (uint16_t)((payload[12]>>3| payload[13] << 5)                     & 0x07FF);
    channels[10] = (uint16_t)((payload[13]>>6| payload[14] << 2 | payload[15]<<10)   & 0x07FF);
    channels[11] = (uint16_t)((payload[15]>>1| payload[16] << 7)                     & 0x07FF);
    channels[12] = (uint16_t)((payload[16]>>4| payload[17] << 4)                     & 0x07FF);
    channels[13] = (uint16_t)((payload[17]>>7| payload[18] << 1 | payload[19]<<9)    & 0x07FF);
    channels[14] = (uint16_t)((payload[19]>>2| payload[20] << 6)                     & 0x07FF);
    channels[15] = (uint16_t)((payload[20]>>5| payload[21] << 3)                     & 0x07FF);
}
```

### 3.4 通道映射

```c
/* 标准通道映射（AETR） */
typedef enum {
    CRSF_CH_ROLL = 0,       // 副翼 (Aileron)
    CRSF_CH_PITCH,          // 升降 (Elevator)
    CRSF_CH_THROTTLE,       // 油门 (Throttle)
    CRSF_CH_YAW,            // 方向 (Rudder)
    CRSF_CH_AUX1,           // 辅助通道1 (AUX1)
    CRSF_CH_AUX2,           // 辅助通道2 (AUX2)
    CRSF_CH_AUX3,           // 辅助通道3 (AUX3)
    CRSF_CH_AUX4,           // 辅助通道4 (AUX4)
    CRSF_CH_AUX5,           // 辅助通道5 (AUX5)
    CRSF_CH_AUX6,           // 辅助通道6 (AUX6)
    CRSF_CH_AUX7,           // 辅助通道7 (AUX7)
    CRSF_CH_AUX8,           // 辅助通道8 (AUX8)
    // CH9-CH16: 额外辅助通道
} CRSF_Channel_t;
```

**云台应用映射建议：**
```c
CH1 (Roll)   → Yaw轴控制
CH2 (Pitch)  → Pitch轴控制
CH5 (AUX1)   → 模式切换（3段开关）
CH6 (AUX2)   → 激光开关（2段开关）
```

### 3.5 通道值转换

**转换到标准PWM (1000-2000us)：**
```c
uint16_t crsf_to_pwm(uint16_t crsf_value) {
    // CRSF: 172-1811 → PWM: 988-2012us
    // 简化映射到 1000-2000us
    return (uint16_t)(1000 + (crsf_value - 172) * 1000 / (1811 - 172));
}
```

**转换到角度 (-90° ~ +90°)：**
```c
float crsf_to_angle(uint16_t crsf_value) {
    // CRSF: 172-1811 → Angle: -90 ~ +90°
    float normalized = (float)(crsf_value - CRSF_CHANNEL_VALUE_MIN) /
                      (float)(CRSF_CHANNEL_VALUE_MAX - CRSF_CHANNEL_VALUE_MIN);
    return (normalized * 180.0f) - 90.0f;
}
```

**转换到百分比 (0-100%)：**
```c
float crsf_to_percentage(uint16_t crsf_value) {
    // CRSF: 172-1811 → Percentage: 0-100%
    return (float)(crsf_value - CRSF_CHANNEL_VALUE_MIN) * 100.0f /
           (float)(CRSF_CHANNEL_VALUE_MAX - CRSF_CHANNEL_VALUE_MIN);
}
```

---

## 4️⃣ 链路统计帧 (0x14)

### 4.1 帧格式

```
帧结构:
┌─────┬─────┬─────┬──────────────────┬─────┐
│ 0xC8│ 0x0A│ 0x14│  8 bytes data    │ CRC │
│ Addr│ Len │ Type│  (statistics)    │     │
└─────┴─────┴─────┴──────────────────┴─────┘

完整帧长度: 12 bytes
```

### 4.2 数据字段

```c
/**
 * @brief 链路统计数据结构
 */
typedef struct {
    uint8_t uplink_rssi_ant1;       // 上行RSSI天线1 (dBm, 负数表示为正)
    uint8_t uplink_rssi_ant2;       // 上行RSSI天线2
    uint8_t uplink_link_quality;    // 上行链路质量 (0-100%)
    int8_t  uplink_snr;             // 上行信噪比 (dB)
    uint8_t active_antenna;         // 当前激活天线 (0或1)
    uint8_t rf_mode;                // RF模式 (更新率)
    uint8_t uplink_tx_power;        // 上行发射功率 (mW)
    uint8_t downlink_rssi;          // 下行RSSI (dBm)
    uint8_t downlink_link_quality;  // 下行链路质量 (0-100%)
    int8_t  downlink_snr;           // 下行信噪比 (dB)
} __attribute__((packed)) CRSF_LinkStatistics_t;
```

### 4.3 解析代码

```c
/**
 * @brief 解析链路统计数据
 */
void crsf_parse_link_statistics(uint8_t *payload, CRSF_LinkStatistics_t *stats) {
    stats->uplink_rssi_ant1 = payload[0];
    stats->uplink_rssi_ant2 = payload[1];
    stats->uplink_link_quality = payload[2];
    stats->uplink_snr = (int8_t)payload[3];
    stats->active_antenna = payload[4];
    stats->rf_mode = payload[5];
    stats->uplink_tx_power = payload[6];
    stats->downlink_rssi = payload[7];
    stats->downlink_link_quality = payload[8];
    stats->downlink_snr = (int8_t)payload[9];
}

/**
 * @brief RSSI转换为dBm
 */
int16_t crsf_rssi_to_dbm(uint8_t rssi) {
    // CRSF RSSI以负数表示，需要转换
    return -(int16_t)rssi;
}
```

---

## 5️⃣ CRC8校验算法

### 5.1 CRC8-DVB-S2多项式

CRSF使用 **CRC-8/DVB-S2** 算法，多项式为 `0xD5`。

```c
/**
 * @brief CRC8-DVB-S2计算（查表法 - 快速）
 */
static const uint8_t crc8_dvb_s2_table[256] = {
    0x00, 0xD5, 0x7F, 0xAA, 0xFE, 0x2B, 0x81, 0x54,
    0x29, 0xFC, 0x56, 0x83, 0xD7, 0x02, 0xA8, 0x7D,
    0x52, 0x87, 0x2D, 0xF8, 0xAC, 0x79, 0xD3, 0x06,
    0x7B, 0xAE, 0x04, 0xD1, 0x85, 0x50, 0xFA, 0x2F,
    0xA4, 0x71, 0xDB, 0x0E, 0x5A, 0x8F, 0x25, 0xF0,
    0x8D, 0x58, 0xF2, 0x27, 0x73, 0xA6, 0x0C, 0xD9,
    0xF6, 0x23, 0x89, 0x5C, 0x08, 0xDD, 0x77, 0xA2,
    0xDF, 0x0A, 0xA0, 0x75, 0x21, 0xF4, 0x5E, 0x8B,
    0x9D, 0x48, 0xE2, 0x37, 0x63, 0xB6, 0x1C, 0xC9,
    0xB4, 0x61, 0xCB, 0x1E, 0x4A, 0x9F, 0x35, 0xE0,
    0xCF, 0x1A, 0xB0, 0x65, 0x31, 0xE4, 0x4E, 0x9B,
    0xE6, 0x33, 0x99, 0x4C, 0x18, 0xCD, 0x67, 0xB2,
    0x39, 0xEC, 0x46, 0x93, 0xC7, 0x12, 0xB8, 0x6D,
    0x10, 0xC5, 0x6F, 0xBA, 0xEE, 0x3B, 0x91, 0x44,
    0x6B, 0xBE, 0x14, 0xC1, 0x95, 0x40, 0xEA, 0x3F,
    0x42, 0x97, 0x3D, 0xE8, 0xBC, 0x69, 0xC3, 0x16,
    0xEF, 0x3A, 0x90, 0x45, 0x11, 0xC4, 0x6E, 0xBB,
    0xC6, 0x13, 0xB9, 0x6C, 0x38, 0xED, 0x47, 0x92,
    0xBD, 0x68, 0xC2, 0x17, 0x43, 0x96, 0x3C, 0xE9,
    0x94, 0x41, 0xEB, 0x3E, 0x6A, 0xBF, 0x15, 0xC0,
    0x4B, 0x9E, 0x34, 0xE1, 0xB5, 0x60, 0xCA, 0x1F,
    0x62, 0xB7, 0x1D, 0xC8, 0x9C, 0x49, 0xE3, 0x36,
    0x19, 0xCC, 0x66, 0xB3, 0xE7, 0x32, 0x98, 0x4D,
    0x30, 0xE5, 0x4F, 0x9A, 0xCE, 0x1B, 0xB1, 0x64,
    0x72, 0xA7, 0x0D, 0xD8, 0x8C, 0x59, 0xF3, 0x26,
    0x5B, 0x8E, 0x24, 0xF1, 0xA5, 0x70, 0xDA, 0x0F,
    0x20, 0xF5, 0x5F, 0x8A, 0xDE, 0x0B, 0xA1, 0x74,
    0x09, 0xDC, 0x76, 0xA3, 0xF7, 0x22, 0x88, 0x5D,
    0xD6, 0x03, 0xA9, 0x7C, 0x28, 0xFD, 0x57, 0x82,
    0xFF, 0x2A, 0x80, 0x55, 0x01, 0xD4, 0x7E, 0xAB,
    0x84, 0x51, 0xFB, 0x2E, 0x7A, 0xAF, 0x05, 0xD0,
    0xAD, 0x78, 0xD2, 0x07, 0x53, 0x86, 0x2C, 0xF9
};

uint8_t crsf_crc8_table(uint8_t *data, uint8_t len) {
    uint8_t crc = 0;
    for (uint8_t i = 0; i < len; i++) {
        crc = crc8_dvb_s2_table[crc ^ data[i]];
    }
    return crc;
}

/**
 * @brief CRC8-DVB-S2计算（循环法 - 节省Flash）
 */
uint8_t crsf_crc8_loop(uint8_t *data, uint8_t len) {
    uint8_t crc = 0;
    for (uint8_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x80) {
                crc = (crc << 1) ^ 0xD5;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}
```

---

## 6️⃣ 完整接收实现

### 6.1 状态机接收

```c
typedef enum {
    CRSF_STATE_SYNC = 0,    // 等待同步（地址字节）
    CRSF_STATE_LENGTH,      // 接收长度字节
    CRSF_STATE_DATA,        // 接收数据
} CRSF_RxState_t;

typedef struct {
    CRSF_RxState_t state;
    uint8_t buffer[CRSF_FRAME_SIZE_MAX];
    uint8_t index;
    uint8_t expected_length;
} CRSF_RxParser_t;

CRSF_RxParser_t g_crsf_parser = {0};

/**
 * @brief CRSF字节接收处理（在UART中断或DMA回调中调用）
 */
void crsf_rx_byte(uint8_t byte) {
    switch (g_crsf_parser.state) {
        case CRSF_STATE_SYNC:
            // 等待地址字节
            if (byte == CRSF_ADDRESS_FLIGHT_CONTROLLER ||
                byte == CRSF_ADDRESS_BROADCAST) {
                g_crsf_parser.buffer[0] = byte;
                g_crsf_parser.index = 1;
                g_crsf_parser.state = CRSF_STATE_LENGTH;
            }
            break;

        case CRSF_STATE_LENGTH:
            // 接收长度字节
            if (byte >= 2 && byte <= CRSF_FRAME_SIZE_MAX) {
                g_crsf_parser.buffer[1] = byte;
                g_crsf_parser.expected_length = byte + 2;  // +地址+长度
                g_crsf_parser.index = 2;
                g_crsf_parser.state = CRSF_STATE_DATA;
            } else {
                // 长度非法，重新同步
                g_crsf_parser.state = CRSF_STATE_SYNC;
            }
            break;

        case CRSF_STATE_DATA:
            // 接收数据字节
            g_crsf_parser.buffer[g_crsf_parser.index++] = byte;

            if (g_crsf_parser.index >= g_crsf_parser.expected_length) {
                // 接收完整，处理帧
                crsf_process_frame(g_crsf_parser.buffer);

                // 重新同步
                g_crsf_parser.state = CRSF_STATE_SYNC;
                g_crsf_parser.index = 0;
            }
            break;
    }
}
```

### 6.2 DMA + 空闲中断接收（推荐）

```c
uint8_t crsf_dma_buffer[CRSF_FRAME_SIZE_MAX];

/**
 * @brief UART空闲中断回调
 */
void HAL_UART_IdleCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART3) {  // CRSF接收
        // 停止DMA
        HAL_UART_DMAStop(huart);

        // 计算接收长度
        uint16_t rx_len = CRSF_FRAME_SIZE_MAX -
                         __HAL_DMA_GET_COUNTER(huart->hdmarx);

        // 逐字节处理
        for (uint16_t i = 0; i < rx_len; i++) {
            crsf_rx_byte(crsf_dma_buffer[i]);
        }

        // 重启DMA
        HAL_UART_Receive_DMA(huart, crsf_dma_buffer, CRSF_FRAME_SIZE_MAX);
    }
}
```

---

## 7️⃣ 调试与测试

### 7.1 串口调试工具

**推荐工具：**
```bash
# Linux/Mac
minicom -D /dev/ttyUSB0 -b 420000

# 或使用screen
screen /dev/ttyUSB0 420000

# 查看原始数据（十六进制）
hexdump -C /dev/ttyUSB0
```

### 7.2 逻辑分析仪

使用Saleae Logic等工具抓取UART波形：
```
配置:
  - 采样率: ≥10 MHz
  - 协议解析: Async Serial (420000 bps, 8N1)
  - 触发: 帧头 0xC8
```

### 7.3 调试输出

```c
/**
 * @brief 打印CRSF帧（调试用）
 */
void crsf_debug_print_frame(uint8_t *frame, uint8_t len) {
    printf("CRSF Frame [%d bytes]: ", len);
    for (uint8_t i = 0; i < len; i++) {
        printf("%02X ", frame[i]);
    }
    printf("\r\n");

    // 解析通道数据（如果是0x16帧）
    if (frame[2] == CRSF_FRAMETYPE_RC_CHANNELS) {
        uint16_t channels[16];
        crsf_parse_rc_channels(&frame[3], channels);

        printf("Channels: ");
        for (uint8_t i = 0; i < 8; i++) {  // 打印前8通道
            printf("CH%d:%d ", i+1, channels[i]);
        }
        printf("\r\n");
    }
}
```

### 7.4 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 无数据接收 | 波特率错误 | 确认420000bps配置 |
| CRC校验失败 | 数据损坏/干扰 | 检查线材屏蔽/飞线长度 |
| 数据不连续 | UART缓冲溢出 | 使用DMA接收 |
| 通道值异常 | 解包算法错误 | 对照参考代码检查位操作 |
| 接收机无绑定 | 未进入绑定模式 | 长按bind键重新绑定 |

---

## 8️⃣ ExpressLRS特定说明

### 8.1 ELRS更新率

```yaml
ExpressLRS支持的更新率:
  - 50Hz: 最大范围模式
  - 100Hz: 平衡模式
  - 150Hz: 低延迟模式
  - 250Hz: 竞速模式
  - 500Hz: 极限低延迟 (需要高功率)

推荐设置:
  云台应用: 100-150Hz (延迟<20ms)
  竞速无人机: 250-500Hz
```

### 8.2 ELRS配置

**通过EdgeTX/OpenTX配置：**
```
1. 模型设置 → 外部模块 → ExpressLRS
2. 更新率: 选择150Hz
3. 发射功率: 根据距离调整 (25mW-1W)
4. 开关频率: 2.4GHz (推荐) / 915MHz
```

**通过ELRS Configurator配置：**
```
1. 连接接收机到电脑
2. 选择目标固件
3. 配置选项:
   - 更新率: 150Hz
   - 遥测: 启用
   - 动态功率: 开启
4. 刷新固件
```

---

## 9️⃣ 参考资料

### 9.1 官方文档

- **ExpressLRS官方文档**: https://www.expresslrs.org/
- **TBS Crossfire文档**: https://www.team-blacksheep.com/
- **EdgeTX手册**: https://edgetx.org/

### 9.2 开源项目参考

```
GitHub参考项目:
  - ExpressLRS/ExpressLRS: 官方固件源码
  - betaflight/betaflight: 飞控CRSF实现
  - ot0tot/CRSF-for-Arduino: Arduino库
```

### 9.3 工具推荐

```
硬件:
  - ELRS接收机: EP1/EP2 (常用)
  - 遥控器: RadioMaster TX16S / Jumper T-Pro
  - USB转TTL: CP2102 / FT232RL

软件:
  - ExpressLRS Configurator
  - EdgeTX Companion
  - Betaflight Configurator (参考)
```

---

## 🔟 附录：完整示例代码

### 10.1 CRSF接收完整示例

```c
/* crsf_receiver.h */
#ifndef __CRSF_RECEIVER_H
#define __CRSF_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>

#define CRSF_CHANNEL_COUNT 16

typedef struct {
    uint16_t channels[CRSF_CHANNEL_COUNT];
    uint32_t last_update_ms;
    uint8_t link_quality;
    int8_t rssi_dbm;
    bool is_valid;
} CRSF_Data_t;

void crsf_init(void);
void crsf_update(void);
bool crsf_is_valid(void);
uint16_t crsf_get_channel(uint8_t channel);
float crsf_get_channel_angle(uint8_t channel);

#endif

/* crsf_receiver.c */
#include "crsf_receiver.h"
#include "main.h"  // HAL库头文件

CRSF_Data_t g_crsf_data = {0};
extern UART_HandleTypeDef huart3;  // CRSF UART

void crsf_init(void) {
    memset(&g_crsf_data, 0, sizeof(CRSF_Data_t));
    // 启动UART DMA接收（在main中调用）
}

void crsf_update(void) {
    // 检查超时
    if (HAL_GetTick() - g_crsf_data.last_update_ms > 1000) {
        g_crsf_data.is_valid = false;
    }
}

bool crsf_is_valid(void) {
    return g_crsf_data.is_valid;
}

uint16_t crsf_get_channel(uint8_t channel) {
    if (channel < CRSF_CHANNEL_COUNT) {
        return g_crsf_data.channels[channel];
    }
    return 992;  // 返回中位值
}

float crsf_get_channel_angle(uint8_t channel) {
    uint16_t value = crsf_get_channel(channel);
    float normalized = (float)(value - 172) / (float)(1811 - 172);
    return (normalized * 180.0f) - 90.0f;
}
```

---

**文档版本**: v1.0
**最后更新**: 2025-10
**作者**: 幽浮喵 (浮浮酱) ฅ'ω'ฅ

---

**END OF DOCUMENT**
