#pragma once

#include <array>
#include <optional>
#include <utility>

namespace algorithms {

struct Intrinsics {
    float fx{1000.0f};
    float fy{1000.0f};
    float cx{0.0f};
    float cy{0.0f};
};

struct GimbalOffsets {
    float pitch_offset_deg{0.0f};
    float yaw_offset_deg{0.0f};
};

class CoordinateTransformer {
public:
    CoordinateTransformer() = default;
    CoordinateTransformer(Intrinsics intr, GimbalOffsets offsets = {});

    void set_intrinsics(Intrinsics intr);
    void set_offsets(GimbalOffsets offsets);

    [[nodiscard]] Intrinsics intrinsics() const { return intrinsics_; }
    [[nodiscard]] GimbalOffsets offsets() const { return offsets_; }

    std::pair<float, float> pixel_to_angle(float pixel_x, float pixel_y, int image_width,
                                           int image_height) const;

private:
    Intrinsics intrinsics_{};
    GimbalOffsets offsets_{};
};

}  // namespace algorithms
