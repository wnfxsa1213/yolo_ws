#include "coordinate.hpp"

#include <cmath>
#include <stdexcept>

namespace algorithms {

CoordinateTransformer::CoordinateTransformer(Intrinsics intr, GimbalOffsets offsets)
    : intrinsics_(intr), offsets_(offsets) {}

void CoordinateTransformer::set_intrinsics(Intrinsics intr) { intrinsics_ = intr; }

void CoordinateTransformer::set_offsets(GimbalOffsets offsets) { offsets_ = offsets; }

std::pair<float, float> CoordinateTransformer::pixel_to_angle(float pixel_x, float pixel_y,
                                                              int image_width,
                                                              int image_height) const {
    if (image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("图像尺寸必须为正数");
    }

    Intrinsics intr = intrinsics_;
    if (intr.cx == 0.0f && intr.cy == 0.0f) {
        intr.cx = image_width / 2.0f;
        intr.cy = image_height / 2.0f;
    }

    if (std::abs(intr.fx) < 1e-6f || std::abs(intr.fy) < 1e-6f) {
        throw std::invalid_argument("焦距 fx/fy 不能为零");
    }

    const float x_norm = (pixel_x - intr.cx) / intr.fx;
    const float y_norm = (pixel_y - intr.cy) / intr.fy;

    constexpr float rad2deg = 57.29577951308232f;

    float yaw = std::atan(x_norm) * rad2deg;
    float pitch = std::atan(y_norm) * rad2deg;

    yaw += offsets_.yaw_offset_deg;
    pitch += offsets_.pitch_offset_deg;
    return {pitch, yaw};
}

}  // namespace algorithms
