# 视觉模块 (Vision Module)

## 模块职责

负责图像采集和相机管理。

## 待实现文件

### camera.py
- **CameraInterface (抽象基类)**
  - `open()` - 打开相机
  - `close()` - 关闭相机
  - `capture()` - 采集图像
  - `get_intrinsics()` - 获取相机内参

- **HIKCamera (海康相机实现)**
  - 继承 CameraInterface
  - 使用海康MVS SDK
  - 支持参数配置（曝光、增益等）

## 依赖
- MVS Python SDK (海康威视)
- NumPy
- OpenCV (可选，用于格式转换)

## 状态
⏳ 待实现
