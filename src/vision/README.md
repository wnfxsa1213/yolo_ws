# 视觉模块 (Vision Module)

## 模块职责

负责图像采集和相机管理。

## 待实现文件

### camera.py
- **CameraInterface** 抽象出采集、控制接口
- **AravisCamera** 基于 PyGObject Aravis 完成 GigE 采集
- **CameraManager** 多线程抓帧，提供最新帧队列

## 依赖
- `gir1.2-aravis-0.8` / `libaravis-0.8-dev`
- `python3-gi`
- NumPy
- OpenCV（Bayer 转 BGR）

## 状态
✅ 已实现 (Aravis)
