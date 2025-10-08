#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "coordinate.hpp"
#include "detector.hpp"

namespace py = pybind11;
using algorithms::CoordinateTransformer;
using algorithms::Detection;
using algorithms::GimbalOffsets;
using algorithms::Intrinsics;
using algorithms::YOLODetector;

namespace {

cv::Mat numpy_uint8_to_mat(const py::array_t<uint8_t>& array) {
    py::buffer_info info = array.request();
    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::runtime_error("图像应为 HxWx3 的 uint8 数组");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw std::runtime_error("图像数组必须是连续的 C-style 内存");
    }
    auto* data = static_cast<uint8_t*>(info.ptr);
    return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC3, data,
                   static_cast<size_t>(info.strides[0]));
}

}  // namespace

PYBIND11_MODULE(detection_core, m) {
    m.doc() = "TensorRT YOLOv8 detection core";

    py::class_<Detection>(m, "Detection")
        .def_readwrite("x1", &Detection::x1)
        .def_readwrite("y1", &Detection::y1)
        .def_readwrite("x2", &Detection::x2)
        .def_readwrite("y2", &Detection::y2)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("class_id", &Detection::class_id);

    py::class_<YOLODetector>(m, "YOLODetector")
        .def(py::init<const std::string&, float, float, bool>(), py::arg("engine_path"),
             py::arg("confidence_threshold") = 0.5f, py::arg("nms_threshold") = 0.45f,
             py::arg("use_letterbox") = true)
        .def("detect",
             [](YOLODetector& self, const py::array_t<uint8_t>& image) {
                 cv::Mat mat = numpy_uint8_to_mat(image);
                 return self.detect(mat);
             },
             py::arg("image"))
        .def_property_readonly("inference_time_ms", &YOLODetector::get_inference_time_ms)
        .def_property_readonly("input_width", &YOLODetector::get_input_width)
        .def_property_readonly("input_height", &YOLODetector::get_input_height)
        .def_property_readonly("num_classes", &YOLODetector::num_classes);

    py::class_<Intrinsics>(m, "Intrinsics")
        .def(py::init<>())
        .def(py::init<float, float, float, float>(), py::arg("fx"), py::arg("fy"), py::arg("cx"),
             py::arg("cy"))
        .def_readwrite("fx", &Intrinsics::fx)
        .def_readwrite("fy", &Intrinsics::fy)
        .def_readwrite("cx", &Intrinsics::cx)
        .def_readwrite("cy", &Intrinsics::cy);

    py::class_<GimbalOffsets>(m, "GimbalOffsets")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("pitch_offset_deg"), py::arg("yaw_offset_deg"))
        .def_readwrite("pitch_offset_deg", &GimbalOffsets::pitch_offset_deg)
        .def_readwrite("yaw_offset_deg", &GimbalOffsets::yaw_offset_deg);

    py::class_<CoordinateTransformer>(m, "CoordinateTransformer")
        .def(py::init<>())
        .def(py::init<Intrinsics, GimbalOffsets>(), py::arg("intrinsics"),
             py::arg("offsets") = GimbalOffsets{})
        .def("set_intrinsics", &CoordinateTransformer::set_intrinsics)
        .def("set_offsets", &CoordinateTransformer::set_offsets)
        .def("pixel_to_angle", &CoordinateTransformer::pixel_to_angle, py::arg("pixel_x"),
             py::arg("pixel_y"), py::arg("image_width"), py::arg("image_height"));
}
