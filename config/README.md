# 配置文件目录 (Config)

## 用途

存放系统配置文件（YAML格式）。

## 配置文件清单

### system_config.yaml ⭐ (主配置文件)
**描述：** 系统全局配置

**包含内容：**
- 项目信息（名称、版本、日志级别）
- 相机配置引用
- 模型配置（引擎路径、阈值）
- 串口配置（端口、波特率）
- 控制参数（最大角速度、斜率限制）
- 追踪参数（最大丢失帧数、置信度阈值）
- 性能配置（GPU预处理、CUDA流、多线程）
- 调试选项（显示图像、保存检测结果、性能分析）

**状态：** ⏳ 待创建

---

### camera_config.yaml
**描述：** 海康相机参数配置

**包含内容：**
- 曝光时间 (us)
- 增益
- 触发模式
- 像素格式
- 伽马值
- 锐化
- 黑电平

**状态：** ⏳ 待创建

---

### camera_intrinsics.yaml
**描述：** 相机内参（标定结果）

**包含内容：**
- fx, fy (焦距)
- cx, cy (光心)
- k1, k2, k3 (径向畸变)
- p1, p2 (切向畸变)
- 图像分辨率

**状态：** ⏳ 待标定后生成

---

## 配置文件格式示例

### system_config.yaml (简化版)
```yaml
project:
  name: "gimbal_tracker"
  version: "1.0.0"
  log_level: "INFO"

camera:
  config_path: "config/camera_config.yaml"
  intrinsics_path: "config/camera_intrinsics.yaml"

model:
  engine_path: "models/yolov8n_fp16.engine"
  conf_threshold: 0.5
  nms_threshold: 0.45
  classes: [0]  # COCO person class

serial:
  port: "/dev/ttyTHS0"
  baudrate: 460800
  timeout: 0.1

control:
  max_velocity_pitch: 150.0  # °/s
  max_velocity_yaw: 200.0
  slew_rate_limit: 300.0     # °/s²

debug:
  show_image: false
  print_fps: true
```

### camera_config.yaml (简化版)
```yaml
hikvision:
  exposure_time: 5000  # us
  gain: 8.0
  trigger_mode: "off"
  pixel_format: "RGB8"
  gamma: 1.0
  sharpness: 128
```

### camera_intrinsics.yaml (示例)
```yaml
# 相机内参矩阵
intrinsics:
  fx: 1000.0
  fy: 1000.0
  cx: 960.0
  cy: 540.0

# 畸变系数
distortion:
  k1: -0.12
  k2: 0.08
  k3: -0.02
  p1: 0.001
  p2: -0.001

# 图像尺寸
resolution:
  width: 1920
  height: 1080
```

---

## 配置加载方式

```python
from src.utils.config import ConfigManager

# 加载主配置
config = ConfigManager("config/system_config.yaml")

# 访问配置项
model_path = config.get("model.engine_path")
conf_thresh = config.get("model.conf_threshold", default=0.5)

# 验证必要字段
config.validate(required_fields=[
    "camera.config_path",
    "model.engine_path",
    "serial.port"
])
```

---

## 注意事项

1. **路径配置**
   - 相对路径基于项目根目录 `~/yolo_ws`
   - 建议使用相对路径而非绝对路径

2. **安全性**
   - 不要在配置文件中存储敏感信息
   - 配置文件应加入版本控制

3. **环境差异**
   - 开发环境：`system_config_dev.yaml`
   - 生产环境：`system_config_prod.yaml`
   - 测试环境：`system_config_test.yaml`

---

## 状态
⏳ 所有配置文件待创建
