# 算法模块 (Algorithms Module)

## 模块职责

C++实现的高性能算法模块，通过pybind11绑定到Python。

## 待实现文件

### detector.hpp / detector.cpp
- **YOLODetector 类**
  - TensorRT引擎加载
  - 图像预处理（Resize, Normalize, HWC→CHW）
  - 推理执行
  - 后处理（NMS, 置信度过滤）
  - 输出格式：`[[x1, y1, x2, y2, conf, cls], ...]`

### tracker.hpp / tracker.cpp
- **ByteTracker 类**
  - 卡尔曼滤波器
  - 匈牙利匹配算法
  - Track管理（new, tracked, lost）
  - 输出格式：`[[track_id, x1, y1, x2, y2, conf], ...]`

### coordinate.hpp / coordinate.cpp
- **CoordinateTransformer 类**
  - 像素坐标 → 相机坐标
  - 相机坐标 → 云台角度
  - 相机内参管理
  - 畸变校正（可选）

### bindings.cpp
- **pybind11绑定**
  - 导出YOLODetector类到Python
  - 导出ByteTracker类到Python
  - 导出CoordinateTransformer类到Python
  - NumPy数组自动转换

### CMakeLists.txt
- CUDA工具链配置
- TensorRT库链接
- OpenCV库链接
- pybind11配置
- 编译输出：`algorithms.cpython-310-aarch64-linux-gnu.so`

## 依赖
- CUDA 12.6
- TensorRT 10.3.0
- OpenCV 4.10.0 (CUDA版本)
- pybind11
- Eigen3 (用于矩阵运算)

## 编译
```bash
cd src/algorithms
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 状态
⏳ 待实现
