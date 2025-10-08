# STM32H750 嵌入式控制系统开发文档 V2.0

## 📋 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | STM32H750 云台控制系统开发规范 V2.0 |
| **版本** | v2.0 |
| **更新日期** | 2025-10 |
| **目标读者** | 嵌入式工程师 |
| **处理器** | STM32H750VBT6 @ 400MHz |
| **RTOS** | FreeRTOS |
| **开发环境** | STM32CubeIDE / Keil MDK |

---

## 📌 V2.0 主要变更

### 新增功能
- ✅ **ELRS接收机支持**：CRSF协议解析，遥控器控制
- ✅ **控制权切换机制**：RC/Jetson/Failsafe三模式自动切换
- ✅ **数字舵机驱动**：50Hz PWM输出（500-2500us）
- ✅ **平滑过渡算法**：模式切换时的斜率限制

### 架构调整
- 🔧 **前馈补偿**：预留接口，暂不实现
- 🔧 **轨迹规划**：优先实现（梯形曲线）
- 🔧 **安全优先**：RC信号强制接管机制

---

## 1️⃣ 系统概述

### 1.1 功能描述

本系统作为云台追踪系统的**实时控制核心与安全管理中枢**，负责：

- **双路控制输入**：
  - Jetson自动追踪指令（UART1, 460800bps）
  - ELRS遥控器控制（UART3, 420000bps, CRSF协议）

- **智能控制权管理**：
  - RC信号优先（安全第一）
  - 自动失控保护
  - 平滑模式切换

- **高性能运动控制**：
  - 1kHz控制环
  - 轨迹规划（平滑运动）
  - 数字舵机驱动（50Hz PWM）

- **系统监控与保护**：
  - 温度/电流监控
  - 信号超时检测
  - 故障诊断与上报

### 1.2 性能指标

```yaml
控制频率: 1000 Hz (1ms周期)
位置精度: ±0.5° (数字舵机限制)
响应时间: <50ms (端到端)
角速度范围: 0-300°/s
RC信号延迟: <20ms (ELRS 500Hz模式)
模式切换时间: <100ms (平滑过渡)
失控保护触发: <200ms
温度监控: ±0.5°C精度
CPU占用率: <60% @ 1kHz控制
```

---

## 2️⃣ 硬件接口定义

### 2.1 引脚分配表

| 功能模块 | 引脚 | 配置 | 说明 |
|---------|------|------|------|
| **UART1 (Jetson通信)** |||
| TX | PA9 | AF7, 460800bps | 发送到Jetson |
| RX | PA10 | AF7, 460800bps, DMA | 接收Jetson指令 |
| **UART3 (CRSF接收机)** |||
| RX | PB11 | AF7, 420000bps, DMA | ELRS接收机数据 |
| **TIM2 (舵机PWM - 50Hz)** |||
| Pitch | PA0 (TIM2_CH1) | AF1, 50Hz | 俯仰舵机（500-2500us）|
| Yaw | PA1 (TIM2_CH2) | AF1, 50Hz | 偏航舵机（500-2500us）|
| **TIM3 (激光PWM)** |||
| Laser | PC6 (TIM3_CH1) | AF2, 1kHz | 激光笔PWM调光 |
| **ADC1 (传感器采集)** |||
| Temp | PA4 (ADC1_IN18) | Analog, 12bit | NTC温度传感器 |
| Current | PA5 (ADC1_IN19) | Analog, 12bit | ACS712电流传感器 |
| **GPIO (状态指示)** |||
| LED_Status | PB0 | Output, PP | 系统运行状态 |
| LED_RC | PB1 | Output, PP | RC模式指示 |
| LED_Jetson | PB14 | Output, PP | Jetson模式指示 |
| **调试接口** |||
| SWDIO | PA13 | AF0 | SWD调试数据 |
| SWCLK | PA14 | AF0 | SWD调试时钟 |
| UART2_TX | PA2 | AF7, 115200bps | 调试串口输出 |

### 2.2 系统连接拓扑

```
┌──────────────┐                ┌──────────────────┐
│   Jetson     │◄──────────────►│   STM32H750      │
│   Platform   │  UART1 460800  │                  │
└──────────────┘                │  ┌────────────┐  │
                                │  │ Control    │  │
┌──────────────┐                │  │ Task 1kHz  │  │
│ ELRS 接收机   │───────────────►│  └────────────┘  │
│ (CRSF 420k)  │  UART3         │                  │
└──────────────┘                │  ┌────────────┐  │
       ▲                        │  │ CRSF Parse │  │
       │                        │  │ Task 100Hz │  │
┌──────┴───────┐                │  └────────────┘  │
│  遥控器       │                │                  │
│ (ExpressLRS) │                │  TIM2 50Hz PWM   │
└──────────────┘                └─────────┬────────┘
                                          │
                                 ┌────────▼────────┐
                                 │  数字舵机        │
                                 │  Pitch / Yaw    │
                                 └─────────────────┘
```

### 2.3 时钟配置

```c
/*============================================
 * 系统时钟配置
 *============================================*/

/* 外部时钟 */
#define HSE_VALUE       25000000    // 25MHz 外部晶振

/* PLL1 配置（主CPU时钟）*/
#define PLL1_M          5           // 分频: 25MHz / 5 = 5MHz
#define PLL1_N          160         // 倍频: 5MHz * 160 = 800MHz
#define PLL1_P          2           // CPU: 800MHz / 2 = 400MHz
#define PLL1_Q          4           // 200MHz
#define PLL1_R          4           // 200MHz

/* 总线时钟 */
#define SYSCLK_FREQ     400000000   // 400MHz (CPU)
#define AHB_FREQ        200000000   // 200MHz
#define APB1_FREQ       100000000   // 100MHz (UART1, TIM2)
#define APB2_FREQ       100000000   // 100MHz (UART3)

/* 定时器时钟 */
// TIM2时钟 = APB1 * 2 = 200MHz (when APB1 prescaler != 1)
#define TIM2_CLOCK      200000000   // 200MHz
```

---

## 3️⃣ 数字舵机PWM配置

### 3.1 舵机规格

```yaml
类型: 标准数字舵机
控制信号:
  频率: 50Hz (20ms周期)
  脉宽范围: 500us ~ 2500us
  中位: 1500us

角度映射:
  500us  → -90°  (最小角度)
  1500us →   0°  (中位)
  2500us → +90°  (最大角度)

精度: ±0.5°
死区: ±3us
响应速度: 60°/0.16s (典型)
堵转扭矩: 15-25kg·cm
```

### 3.2 TIM2配置代码

```c
/*============================================
 * TIM2 - 50Hz PWM输出配置
 * 时钟源: 200MHz
 * 目标频率: 50Hz (20ms周期)
 *============================================*/

/* 定时器参数 */
#define TIM2_CLOCK_FREQ     200000000   // 200MHz
#define SERVO_PWM_FREQ      50          // 50Hz
#define SERVO_PERIOD_MS     20          // 20ms

/* 预分频器计算: 200MHz / 200 = 1MHz (1us计数) */
#define TIM2_PRESCALER      (200 - 1)

/* 自动重载值: 1MHz / 50Hz = 20000 counts = 20ms */
#define TIM2_ARR            (20000 - 1)

/* 脉宽定义（单位：us，对应CCR寄存器值）*/
#define SERVO_PWM_MIN       500         // 500us → -90°
#define SERVO_PWM_CENTER    1500        // 1500us → 0°
#define SERVO_PWM_MAX       2500        // 2500us → +90°

/* 角度范围 */
#define SERVO_ANGLE_MIN     -90.0f      // 最小角度
#define SERVO_ANGLE_MAX     +90.0f      // 最大角度

/**
 * @brief 角度转PWM脉宽
 * @param angle: 目标角度 (°) [-90, +90]
 * @return PWM脉宽 (us)
 */
uint16_t servo_angle_to_pwm(float angle) {
    // 限幅
    if (angle < SERVO_ANGLE_MIN) angle = SERVO_ANGLE_MIN;
    if (angle > SERVO_ANGLE_MAX) angle = SERVO_ANGLE_MAX;

    // 线性映射
    float normalized = (angle - SERVO_ANGLE_MIN) /
                      (SERVO_ANGLE_MAX - SERVO_ANGLE_MIN);
    uint16_t pwm = SERVO_PWM_MIN +
                   (uint16_t)(normalized * (SERVO_PWM_MAX - SERVO_PWM_MIN));

    return pwm;
}

/**
 * @brief 设置舵机角度
 * @param channel: 通道 (TIM_CHANNEL_1=Pitch, TIM_CHANNEL_2=Yaw)
 * @param angle: 目标角度 (°)
 */
void servo_set_angle(uint32_t channel, float angle) {
    uint16_t pwm_us = servo_angle_to_pwm(angle);

    // 更新CCR寄存器（CCR值 = 微秒数）
    if (channel == TIM_CHANNEL_1) {
        __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pwm_us);
    } else if (channel == TIM_CHANNEL_2) {
        __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, pwm_us);
    }
}

/**
 * @brief TIM2 PWM初始化
 */
void servo_pwm_init(void) {
    /* TIM2基础配置 */
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = TIM2_PRESCALER;
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = TIM2_ARR;
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;

    if (HAL_TIM_PWM_Init(&htim2) != HAL_OK) {
        Error_Handler();
    }

    /* PWM通道配置 */
    TIM_OC_InitTypeDef sConfigOC = {0};
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = SERVO_PWM_CENTER;     // 初始化为中位
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    // 配置通道1 (Pitch)
    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
        Error_Handler();
    }

    // 配置通道2 (Yaw)
    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) {
        Error_Handler();
    }

    /* 启动PWM输出 */
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);

    /* 舵机归中 */
    servo_set_angle(TIM_CHANNEL_1, 0.0f);
    servo_set_angle(TIM_CHANNEL_2, 0.0f);
}
```

### 3.3 舵机标定

```c
/**
 * @brief 舵机标定程序（调试用）
 * 用于测试舵机的实际角度范围
 */
void servo_calibration(void) {
    // 中位测试
    servo_set_angle(TIM_CHANNEL_1, 0.0f);
    HAL_Delay(1000);

    // 最大角度测试
    servo_set_angle(TIM_CHANNEL_1, 90.0f);
    HAL_Delay(1000);

    // 最小角度测试
    servo_set_angle(TIM_CHANNEL_1, -90.0f);
    HAL_Delay(1000);

    // 归中
    servo_set_angle(TIM_CHANNEL_1, 0.0f);

    // 重复Yaw轴测试
    // ...
}
```

---

## 4️⃣ CRSF协议实现

### 4.1 CRSF协议概述

**CRSF (Crossfire)** 是TBS开发的高速遥控链路协议，被ExpressLRS采用。

```yaml
协议特点:
  - 高更新率: 最高500Hz
  - 低延迟: <10ms
  - 16通道支持
  - 双向通信
  - 内置CRC校验

物理层:
  - 波特率: 420000 bps
  - 格式: 8N1 (8数据位, 无校验, 1停止位)
  - 电平: 3.3V UART
```

### 4.2 CRSF帧格式

```c
/*============================================
 * CRSF帧结构
 * [设备地址] [帧长度] [类型] [数据...] [CRC8]
 *============================================*/

#define CRSF_BAUDRATE           420000
#define CRSF_FRAME_SIZE_MAX     64

/* 设备地址 */
#define CRSF_ADDRESS_BROADCAST          0x00
#define CRSF_ADDRESS_USB                0x10
#define CRSF_ADDRESS_BLUETOOTH          0x12
#define CRSF_ADDRESS_RECEIVER           0xEC
#define CRSF_ADDRESS_FLIGHT_CONTROLLER  0xC8    // STM32作为FC

/* 帧类型 */
#define CRSF_FRAMETYPE_GPS              0x02
#define CRSF_FRAMETYPE_BATTERY          0x08
#define CRSF_FRAMETYPE_LINK_STATISTICS  0x14
#define CRSF_FRAMETYPE_RC_CHANNELS      0x16    // RC通道数据（最重要）
#define CRSF_FRAMETYPE_ATTITUDE         0x1E
#define CRSF_FRAMETYPE_FLIGHT_MODE      0x21

/* RC通道参数 */
#define CRSF_CHANNEL_COUNT      16
#define CRSF_CHANNEL_VALUE_MIN  172     // 对应PWM 988us
#define CRSF_CHANNEL_VALUE_MID  992     // 对应PWM 1500us
#define CRSF_CHANNEL_VALUE_MAX  1811    // 对应PWM 2012us

/**
 * @brief CRSF数据帧结构
 */
typedef struct {
    uint8_t address;                    // 设备地址
    uint8_t length;                     // 帧长度（不含地址和长度字节）
    uint8_t type;                       // 帧类型
    uint8_t data[CRSF_FRAME_SIZE_MAX];  // 数据域
} __attribute__((packed)) CRSF_Frame_t;

/**
 * @brief RC通道数据
 */
typedef struct {
    uint16_t channels[CRSF_CHANNEL_COUNT];  // 通道值 [172-1811]
    uint32_t last_update_time;              // 最后更新时间戳
    uint8_t  link_quality;                  // 链路质量 [0-100]
    int8_t   rssi_dbm;                      // 信号强度 (dBm)
    bool     is_valid;                      // 数据有效标志
} CRSF_Channels_t;

/* 全局RC通道数据 */
CRSF_Channels_t g_crsf_channels = {0};

/* 通道功能映射（根据遥控器配置调整）*/
typedef enum {
    CRSF_CH_ROLL = 0,       // CH1: Roll (Yaw轴)
    CRSF_CH_PITCH,          // CH2: Pitch
    CRSF_CH_THROTTLE,       // CH3: Throttle (未使用)
    CRSF_CH_YAW,            // CH4: Yaw (旋转)
    CRSF_CH_AUX1,           // CH5: 模式切换（3段开关）
    CRSF_CH_AUX2,           // CH6: 激光开关
    CRSF_CH_AUX3,           // CH7: 备用
    CRSF_CH_AUX4,           // CH8: 备用
} CRSF_Channel_t;
```

### 4.3 CRC8校验算法

```c
/**
 * @brief CRC8计算（DVB-S2多项式: 0xD5）
 * @param data: 数据指针
 * @param len: 数据长度
 * @return CRC8校验值
 */
uint8_t crsf_crc8_dvb_s2(uint8_t *data, uint8_t len) {
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

### 4.4 RC通道数据解析

```c
/**
 * @brief 解析RC通道数据包
 * CRSF通道数据打包格式：11bit per channel, 22 bytes total
 *
 * @param payload: 22字节通道数据
 */
void crsf_parse_rc_channels(uint8_t *payload) {
    uint16_t *ch = g_crsf_channels.channels;

    // 使用位操作提取11bit通道值
    ch[0]  = (uint16_t)((payload[0]    | payload[1]  << 8)                     & 0x07FF);
    ch[1]  = (uint16_t)((payload[1]>>3 | payload[2]  << 5)                     & 0x07FF);
    ch[2]  = (uint16_t)((payload[2]>>6 | payload[3]  << 2 | payload[4]<<10)    & 0x07FF);
    ch[3]  = (uint16_t)((payload[4]>>1 | payload[5]  << 7)                     & 0x07FF);
    ch[4]  = (uint16_t)((payload[5]>>4 | payload[6]  << 4)                     & 0x07FF);
    ch[5]  = (uint16_t)((payload[6]>>7 | payload[7]  << 1 | payload[8]<<9)     & 0x07FF);
    ch[6]  = (uint16_t)((payload[8]>>2 | payload[9]  << 6)                     & 0x07FF);
    ch[7]  = (uint16_t)((payload[9]>>5 | payload[10] << 3)                     & 0x07FF);
    ch[8]  = (uint16_t)((payload[11]   | payload[12] << 8)                     & 0x07FF);
    ch[9]  = (uint16_t)((payload[12]>>3| payload[13] << 5)                     & 0x07FF);
    ch[10] = (uint16_t)((payload[13]>>6| payload[14] << 2 | payload[15]<<10)   & 0x07FF);
    ch[11] = (uint16_t)((payload[15]>>1| payload[16] << 7)                     & 0x07FF);
    ch[12] = (uint16_t)((payload[16]>>4| payload[17] << 4)                     & 0x07FF);
    ch[13] = (uint16_t)((payload[17]>>7| payload[18] << 1 | payload[19]<<9)    & 0x07FF);
    ch[14] = (uint16_t)((payload[19]>>2| payload[20] << 6)                     & 0x07FF);
    ch[15] = (uint16_t)((payload[20]>>5| payload[21] << 3)                     & 0x07FF);

    // 更新时间戳
    g_crsf_channels.last_update_time = HAL_GetTick();
    g_crsf_channels.is_valid = true;
}

/**
 * @brief CRSF通道值转角度
 * @param channel_value: CRSF通道值 [172-1811]
 * @return 角度 (°) [-90, +90]
 */
float crsf_channel_to_angle(uint16_t channel_value) {
    // 归一化到[0, 1]
    float normalized = (float)(channel_value - CRSF_CHANNEL_VALUE_MIN) /
                      (float)(CRSF_CHANNEL_VALUE_MAX - CRSF_CHANNEL_VALUE_MIN);

    // 映射到[-90, +90]
    float angle = (normalized * 180.0f) - 90.0f;

    // 限幅
    if (angle < -90.0f) angle = -90.0f;
    if (angle > 90.0f) angle = 90.0f;

    return angle;
}
```

### 4.5 CRSF接收处理

```c
/* UART接收缓冲区 */
uint8_t crsf_rx_buffer[CRSF_FRAME_SIZE_MAX];
uint8_t crsf_frame_buffer[CRSF_FRAME_SIZE_MAX];
uint8_t crsf_rx_index = 0;

/**
 * @brief 处理接收到的CRSF帧
 * @param frame: CRSF帧指针
 */
void crsf_process_frame(CRSF_Frame_t *frame) {
    // CRC校验
    uint8_t calc_crc = crsf_crc8_dvb_s2(&frame->type, frame->length - 1);
    uint8_t recv_crc = frame->data[frame->length - 2];

    if (calc_crc != recv_crc) {
        // CRC错误，丢弃帧
        return;
    }

    // 根据帧类型处理
    switch (frame->type) {
        case CRSF_FRAMETYPE_RC_CHANNELS:
            // 解析RC通道数据
            crsf_parse_rc_channels(frame->data);
            break;

        case CRSF_FRAMETYPE_LINK_STATISTICS:
            // 链路统计信息
            // payload[5] = LQ (Link Quality)
            // payload[6] = RSSI (dBm, signed)
            g_crsf_channels.link_quality = frame->data[5];
            g_crsf_channels.rssi_dbm = (int8_t)frame->data[6];
            break;

        default:
            // 未处理的帧类型
            break;
    }
}

/**
 * @brief UART3空闲中断回调（CRSF接收）
 */
void HAL_UART_IdleCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART3) {
        // 停止DMA
        HAL_UART_DMAStop(huart);

        // 计算接收数据长度
        uint16_t rx_len = CRSF_FRAME_SIZE_MAX -
                         __HAL_DMA_GET_COUNTER(huart->hdmarx);

        // 简单状态机解析
        for (uint16_t i = 0; i < rx_len; i++) {
            uint8_t byte = crsf_rx_buffer[i];

            if (crsf_rx_index == 0) {
                // 等待地址字节
                if (byte == CRSF_ADDRESS_FLIGHT_CONTROLLER) {
                    crsf_frame_buffer[crsf_rx_index++] = byte;
                }
            } else if (crsf_rx_index == 1) {
                // 长度字节
                if (byte >= 2 && byte <= CRSF_FRAME_SIZE_MAX) {
                    crsf_frame_buffer[crsf_rx_index++] = byte;
                } else {
                    crsf_rx_index = 0;  // 错误，重置
                }
            } else {
                // 数据字节
                crsf_frame_buffer[crsf_rx_index++] = byte;

                // 检查是否接收完整帧
                uint8_t expected_len = crsf_frame_buffer[1] + 2;
                if (crsf_rx_index >= expected_len) {
                    // 处理完整帧
                    crsf_process_frame((CRSF_Frame_t*)crsf_frame_buffer);
                    crsf_rx_index = 0;
                }
            }

            // 防止溢出
            if (crsf_rx_index >= CRSF_FRAME_SIZE_MAX) {
                crsf_rx_index = 0;
            }
        }

        // 重启DMA接收
        HAL_UART_Receive_DMA(&huart3, crsf_rx_buffer, CRSF_FRAME_SIZE_MAX);
    }
}

/**
 * @brief 检查CRSF信号有效性
 * @return true=有效, false=超时
 */
bool crsf_is_signal_valid(void) {
    const uint32_t TIMEOUT_MS = 1000;  // 1秒超时
    return (HAL_GetTick() - g_crsf_channels.last_update_time) < TIMEOUT_MS &&
           g_crsf_channels.is_valid;
}

/**
 * @brief CRSF初始化
 */
void crsf_init(void) {
    // 清零通道数据
    memset(&g_crsf_channels, 0, sizeof(CRSF_Channels_t));

    // 启动UART3 DMA接收
    HAL_UART_Receive_DMA(&huart3, crsf_rx_buffer, CRSF_FRAME_SIZE_MAX);

    // 使能UART空闲中断
    __HAL_UART_ENABLE_IT(&huart3, UART_IT_IDLE);
}
```

---

## 5️⃣ 控制权切换安全机制

### 5.1 系统状态机

```c
/**
 * @brief 控制模式定义
 */
typedef enum {
    CTRL_MODE_INIT = 0,         // 初始化（上电状态）
    CTRL_MODE_RC_CONTROL,       // RC遥控器控制（最高优先级）
    CTRL_MODE_JETSON_CONTROL,   // Jetson自动控制
    CTRL_MODE_FAILSAFE,         // 失控保护（所有信号丢失）
    CTRL_MODE_ERROR             // 错误状态
} ControlMode_t;

/**
 * @brief 控制系统状态
 */
typedef struct {
    ControlMode_t current_mode;     // 当前模式
    ControlMode_t previous_mode;    // 上一模式

    bool rc_signal_valid;           // RC信号有效标志
    bool jetson_signal_valid;       // Jetson信号有效标志

    uint32_t rc_last_update;        // RC最后更新时间
    uint32_t jetson_last_update;    // Jetson最后更新时间
    uint32_t mode_enter_time;       // 进入当前模式的时间

    float target_pitch;             // 目标俯仰角
    float target_yaw;               // 目标偏航角

    bool laser_enable;              // 激光使能

} ControlSystem_t;

/* 全局控制系统 */
ControlSystem_t g_ctrl_sys = {0};

/* 超时定义 */
#define RC_SIGNAL_TIMEOUT_MS        1000    // RC信号超时 1秒
#define JETSON_SIGNAL_TIMEOUT_MS    500     // Jetson信号超时 500ms
```

### 5.2 状态转换逻辑

```c
/**
 * @brief 更新信号有效性状态
 */
void control_update_signal_status(void) {
    uint32_t current_time = HAL_GetTick();

    // 检查RC信号
    g_ctrl_sys.rc_signal_valid = crsf_is_signal_valid();

    // 检查Jetson信号
    g_ctrl_sys.jetson_signal_valid =
        (current_time - g_ctrl_sys.jetson_last_update) < JETSON_SIGNAL_TIMEOUT_MS;
}

/**
 * @brief 控制模式状态转换
 * 优先级: RC > Jetson > Failsafe
 */
void control_state_machine(void) {
    ControlMode_t new_mode = g_ctrl_sys.current_mode;

    // 状态转换规则（按优先级）
    if (g_ctrl_sys.rc_signal_valid) {
        // RC信号有效 → 强制切换到RC模式（安全优先）
        new_mode = CTRL_MODE_RC_CONTROL;
    }
    else if (g_ctrl_sys.jetson_signal_valid) {
        // RC无效，Jetson有效 → Jetson自动模式
        new_mode = CTRL_MODE_JETSON_CONTROL;
    }
    else {
        // 两者都无效 → 失控保护
        new_mode = CTRL_MODE_FAILSAFE;
    }

    // 执行状态切换
    if (new_mode != g_ctrl_sys.current_mode) {
        control_mode_enter(new_mode);
    }
}

/**
 * @brief 进入新模式
 */
void control_mode_enter(ControlMode_t new_mode) {
    g_ctrl_sys.previous_mode = g_ctrl_sys.current_mode;
    g_ctrl_sys.current_mode = new_mode;
    g_ctrl_sys.mode_enter_time = HAL_GetTick();

    // 模式切换处理
    switch (new_mode) {
        case CTRL_MODE_RC_CONTROL:
            // 切换到RC控制
            // LED指示：快闪（200ms周期）
            led_set_pattern(LED_PATTERN_FAST_BLINK);
            break;

        case CTRL_MODE_JETSON_CONTROL:
            // 切换到Jetson控制
            // LED指示：慢闪（1000ms周期）
            led_set_pattern(LED_PATTERN_SLOW_BLINK);
            break;

        case CTRL_MODE_FAILSAFE:
            // 失控保护
            // 舵机归中
            g_ctrl_sys.target_pitch = 0.0f;
            g_ctrl_sys.target_yaw = 0.0f;
            g_ctrl_sys.laser_enable = false;
            // LED指示：SOS闪烁
            led_set_pattern(LED_PATTERN_SOS);
            break;

        default:
            break;
    }

    // 可选：通过UART2输出调试信息
    // printf("Mode: %d -> %d\r\n", g_ctrl_sys.previous_mode, new_mode);
}

/**
 * @brief 获取当前目标角度（统一接口）
 */
void control_get_target(float *pitch, float *yaw, bool *laser) {
    switch (g_ctrl_sys.current_mode) {
        case CTRL_MODE_RC_CONTROL:
            // 从CRSF通道获取
            *pitch = crsf_channel_to_angle(
                g_crsf_channels.channels[CRSF_CH_PITCH]);
            *yaw = crsf_channel_to_angle(
                g_crsf_channels.channels[CRSF_CH_ROLL]);

            // 激光控制（AUX2通道，3段开关）
            *laser = (g_crsf_channels.channels[CRSF_CH_AUX2]
                     > CRSF_CHANNEL_VALUE_MID);
            break;

        case CTRL_MODE_JETSON_CONTROL:
            // 从Jetson指令获取
            *pitch = g_ctrl_sys.target_pitch;
            *yaw = g_ctrl_sys.target_yaw;
            *laser = g_ctrl_sys.laser_enable;
            break;

        case CTRL_MODE_FAILSAFE:
            // 失控保护：归中
            *pitch = 0.0f;
            *yaw = 0.0f;
            *laser = false;
            break;

        default:
            *pitch = 0.0f;
            *yaw = 0.0f;
            *laser = false;
            break;
    }
}
```

### 5.3 平滑过渡算法

```c
/**
 * @brief 平滑过渡结构（斜率限制）
 */
typedef struct {
    float current_value;    // 当前输出值
    float target_value;     // 目标值
    float slew_rate;        // 最大变化率 (°/s)
    float dt;               // 时间步长 (s)
} SlewRateLimiter_t;

/**
 * @brief 斜率限制更新
 * @param limiter: 限制器指针
 * @param target: 目标值
 * @return 限制后的当前值
 */
float slew_rate_limit_update(SlewRateLimiter_t *limiter, float target) {
    limiter->target_value = target;

    float error = limiter->target_value - limiter->current_value;
    float max_change = limiter->slew_rate * limiter->dt;

    if (error > max_change) {
        limiter->current_value += max_change;
    } else if (error < -max_change) {
        limiter->current_value -= max_change;
    } else {
        limiter->current_value = limiter->target_value;
    }

    return limiter->current_value;
}

/* 全局斜率限制器 */
SlewRateLimiter_t g_slew_pitch = {
    .slew_rate = 150.0f,    // 150°/s
    .dt = 0.001f            // 1ms
};

SlewRateLimiter_t g_slew_yaw = {
    .slew_rate = 200.0f,    // 200°/s
    .dt = 0.001f
};
```

---

## 6️⃣ 轨迹规划算法

### 6.1 梯形速度曲线规划

```c
/**
 * @brief 梯形速度曲线规划器
 * 特点：简单高效，加减速平滑
 */
typedef struct {
    float current_pos;      // 当前位置 (°)
    float current_vel;      // 当前速度 (°/s)
    float target_pos;       // 目标位置 (°)

    float max_vel;          // 最大速度 (°/s)
    float acceleration;     // 加速度 (°/s²)
    float dt;               // 时间步长 (s)

} TrapezoidPlanner_t;

/**
 * @brief 梯形规划器更新（1kHz调用）
 * @param planner: 规划器指针
 * @param new_target: 新目标位置
 * @return 当前规划位置
 */
float trapezoid_planner_update(TrapezoidPlanner_t *p, float new_target) {
    p->target_pos = new_target;

    float position_error = p->target_pos - p->current_pos;
    float abs_error = fabsf(position_error);

    // 计算减速所需距离
    float brake_distance = (p->current_vel * p->current_vel) /
                          (2.0f * p->acceleration);

    // 决策：加速/匀速/减速
    if (abs_error > brake_distance) {
        // 距离足够，可以继续加速或保持最大速度
        if (fabsf(p->current_vel) < p->max_vel) {
            // 加速阶段
            float accel_sign = (position_error > 0) ? 1.0f : -1.0f;
            p->current_vel += accel_sign * p->acceleration * p->dt;

            // 限制最大速度
            if (fabsf(p->current_vel) > p->max_vel) {
                p->current_vel = accel_sign * p->max_vel;
            }
        }
    } else {
        // 需要减速
        if (fabsf(p->current_vel) > 0.01f) {
            float decel_sign = (p->current_vel > 0) ? -1.0f : 1.0f;
            p->current_vel += decel_sign * p->acceleration * p->dt;

            // 防止过冲
            if ((position_error > 0 && p->current_vel < 0) ||
                (position_error < 0 && p->current_vel > 0)) {
                p->current_vel = 0;
            }
        } else {
            // 速度接近零，直接到达目标
            p->current_vel = 0;
            p->current_pos = p->target_pos;
            return p->current_pos;
        }
    }

    // 位置积分
    p->current_pos += p->current_vel * p->dt;

    return p->current_pos;
}

/**
 * @brief 初始化梯形规划器
 */
void trapezoid_planner_init(TrapezoidPlanner_t *p,
                           float max_velocity,
                           float accel,
                           float dt) {
    p->current_pos = 0;
    p->current_vel = 0;
    p->target_pos = 0;
    p->max_vel = max_velocity;
    p->acceleration = accel;
    p->dt = dt;
}
```

---

## 7️⃣ 主控制循环

### 7.1 FreeRTOS任务配置

```c
/**
 * @brief 任务优先级定义
 */
#define PRIORITY_CONTROL    4   // 控制任务（最高）
#define PRIORITY_COMM       3   // 通信任务
#define PRIORITY_MONITOR    1   // 监控任务
#define PRIORITY_IDLE       0   // 空闲任务

/**
 * @brief 任务堆栈大小
 */
#define STACK_SIZE_CONTROL  1024
#define STACK_SIZE_COMM     512
#define STACK_SIZE_MONITOR  512
```

### 7.2 控制任务实现

```c
/**
 * @brief 控制任务 - 1kHz
 */
void Control_Task(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1);  // 1ms周期

    // 初始化轨迹规划器
    TrapezoidPlanner_t planner_pitch, planner_yaw;
    trapezoid_planner_init(&planner_pitch, 150.0f, 800.0f, 0.001f);
    trapezoid_planner_init(&planner_yaw, 200.0f, 1000.0f, 0.001f);

    // 初始化舵机
    servo_pwm_init();

    while(1) {
        // 1. 更新信号状态
        control_update_signal_status();

        // 2. 状态机转换
        control_state_machine();

        // 3. 获取目标角度
        float target_pitch, target_yaw;
        bool laser_enable;
        control_get_target(&target_pitch, &target_yaw, &laser_enable);

        // 4. 轨迹规划（平滑运动）
        float planned_pitch = trapezoid_planner_update(&planner_pitch, target_pitch);
        float planned_yaw = trapezoid_planner_update(&planner_yaw, target_yaw);

        // 5. 斜率限制（模式切换时平滑过渡）
        float smooth_pitch = slew_rate_limit_update(&g_slew_pitch, planned_pitch);
        float smooth_yaw = slew_rate_limit_update(&g_slew_yaw, planned_yaw);

        // 6. 输出到舵机
        servo_set_angle(TIM_CHANNEL_1, smooth_pitch);
        servo_set_angle(TIM_CHANNEL_2, smooth_yaw);

        // 7. 激光控制
        laser_set_state(laser_enable);

        // 8. 等待下一周期
        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

/**
 * @brief 通信任务 - 100Hz
 */
void Communication_Task(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(10);  // 10ms周期

    while(1) {
        // 处理Jetson串口数据（如有）
        jetson_process_commands();

        // 定期发送状态上报
        jetson_send_status();

        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

/**
 * @brief 监控任务 - 10Hz
 */
void Monitor_Task(void *pvParameters) {
    while(1) {
        // 温度采集
        float temperature = adc_read_temperature();

        // 电流采集
        float current_mA = adc_read_current();

        // 温度保护
        if (temperature > 75.0f) {
            // 过温保护：切换到失控模式
            control_mode_enter(CTRL_MODE_FAILSAFE);
        }

        // LED状态更新
        led_update();

        // 调试输出（可选）
        // printf("Mode:%d T:%.1f I:%.0f\r\n",
        //        g_ctrl_sys.current_mode, temperature, current_mA);

        vTaskDelay(pdMS_TO_TICKS(100));  // 100ms
    }
}
```

---

## 8️⃣ 性能优化

### 8.1 内存优化

```c
/* 使用DTCM快速内存（64KB，零等待） */
__attribute__((section(".dtcm"))) float control_data_buffer[128];

/* 使用ITCM存放关键函数 */
__attribute__((section(".itcm")))
void critical_control_function(void) {
    // 关键控制代码
}
```

### 8.2 DMA优化

```c
/**
 * @brief DMA配置优化
 */
void dma_optimize_config(void) {
    // UART1 DMA优先级设置
    HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 1, 0);

    // UART3 DMA（CRSF）- 最高优先级
    HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 0, 0);
}
```

---

## 9️⃣ 调试与测试

### 9.1 串口调试输出

```c
/**
 * @brief 重定向printf到UART2
 */
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}

/**
 * @brief 调试信息输出
 */
void debug_print_status(void) {
    printf("=== System Status ===\r\n");
    printf("Mode: %d\r\n", g_ctrl_sys.current_mode);
    printf("RC Valid: %d\r\n", g_ctrl_sys.rc_signal_valid);
    printf("Jetson Valid: %d\r\n", g_ctrl_sys.jetson_signal_valid);
    printf("Pitch: %.1f deg\r\n", g_slew_pitch.current_value);
    printf("Yaw: %.1f deg\r\n", g_slew_yaw.current_value);
    printf("LQ: %d%%\r\n", g_crsf_channels.link_quality);
    printf("=====================\r\n");
}
```

### 9.2 测试清单

```
[ ] 硬件测试
    [ ] TIM2 PWM波形验证（示波器）
    [ ] UART3波特率验证（420000bps）
    [ ] 舵机角度标定
    [ ] LED指示功能

[ ] 通信测试
    [ ] CRSF协议解析正确性
    [ ] 通道值映射验证
    [ ] 信号超时检测
    [ ] Jetson通信协议

[ ] 控制测试
    [ ] 三模式切换功能
    [ ] 平滑过渡效果
    [ ] 轨迹规划精度
    [ ] 失控保护触发

[ ] 性能测试
    [ ] 控制环频率（应为1kHz）
    [ ] CPU占用率（<60%）
    [ ] 端到端延迟（<50ms）
    [ ] 长时间稳定性（8小时+）
```

---

## 🔟 开发检查清单

```
Phase 1: 基础功能（Week 1）
[ ] STM32CubeIDE工程创建
[ ] 时钟配置（400MHz）
[ ] TIM2 PWM输出（50Hz）
[ ] 舵机驱动测试
[ ] UART3 CRSF接收

Phase 2: 核心算法（Week 2）
[ ] CRSF协议完整解析
[ ] 控制权状态机
[ ] 梯形轨迹规划
[ ] 平滑过渡算法

Phase 3: 系统集成（Week 3）
[ ] Jetson通信对接
[ ] 三模式联调
[ ] 温度监控集成
[ ] 失控保护验证

Phase 4: 测试优化（Week 4）
[ ] 各场景测试
[ ] 性能调优
[ ] 稳定性验证
[ ] 文档完善
```

---

## 📚 参考资料

1. **STM32H750 Reference Manual** - RM0433
2. **CRSF Protocol Specification** - TBS Crossfire
3. **ExpressLRS Documentation** - https://www.expresslrs.org/
4. **FreeRTOS Kernel Guide** - https://www.freertos.org/

---

**文档版本**: V2.0
**编写日期**: 2025-10
**作者**: 幽浮喵 (浮浮酱) ฅ'ω'ฅ
**审核**: 待审核

---

## 附录A：常用宏定义

```c
/* config.h - 全局配置文件 */
#ifndef __CONFIG_H
#define __CONFIG_H

/* 功能开关 */
#define ENABLE_TRAJECTORY_PLANNING  1   // 轨迹规划
#define ENABLE_FEEDFORWARD          0   // 前馈补偿（预留）
#define ENABLE_DEBUG_OUTPUT         1   // 调试输出

/* 控制参数 */
#define CONTROL_FREQ_HZ             1000
#define CONTROL_PERIOD_MS           1

/* 舵机参数 */
#define SERVO_MAX_VELOCITY_PITCH    150.0f  // °/s
#define SERVO_MAX_VELOCITY_YAW      200.0f  // °/s
#define SERVO_ACCELERATION_PITCH    800.0f  // °/s²
#define SERVO_ACCELERATION_YAW      1000.0f // °/s²

/* 超时参数 */
#define RC_TIMEOUT_MS               1000
#define JETSON_TIMEOUT_MS           500

#endif
```

---

## 附录B：故障排查指南

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 舵机不动 | PWM频率错误 | 检查TIM2配置，确保50Hz |
| CRSF无数据 | 波特率不匹配 | 确认420000bps配置正确 |
| 模式不切换 | 信号超时未触发 | 检查超时时间设置 |
| 抖动严重 | 轨迹规划参数过大 | 降低加速度参数 |
| CPU占用过高 | 任务优先级冲突 | 检查FreeRTOS配置 |

---

**END OF DOCUMENT**
