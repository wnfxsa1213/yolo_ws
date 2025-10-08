#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace algorithms {

struct Detection {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float confidence{};
    int class_id{};
};

class YOLODetector {
public:
    explicit YOLODetector(const std::string& engine_path, float confidence_threshold = 0.5f,
                          float nms_threshold = 0.45f, bool use_letterbox = true);
    ~YOLODetector();

    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;
    YOLODetector(YOLODetector&&) = delete;
    YOLODetector& operator=(YOLODetector&&) = delete;

    std::vector<Detection> detect(const cv::Mat& image);

    float get_inference_time_ms() const { return inference_time_ms_; }
    int get_input_width() const { return input_w_; }
    int get_input_height() const { return input_h_; }
    int num_classes() const { return num_classes_; }

private:
    void load_engine(const std::string& engine_path);
    void allocate_buffers();
    void preprocess(const cv::Mat& image, std::vector<float>& host_buffer, float& scale,
                    int& pad_w, int& pad_h);
    std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Size& orig_size,
                                       float scale, int pad_w, int pad_h);

    static void letterbox_resize(const cv::Mat& image, cv::Mat& resized, int target_w,
                                 int target_h, float& scale, int& pad_w, int& pad_h);
    static void check_cuda(cudaError_t status);

    struct TrtDeleter {
        void operator()(nvinfer1::IRuntime* ptr) const noexcept;
        void operator()(nvinfer1::ICudaEngine* ptr) const noexcept;
        void operator()(nvinfer1::IExecutionContext* ptr) const noexcept;
    };

    struct CudaMemDeleter {
        void operator()(void* ptr) const noexcept;
    };

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};

    using DeviceBuffer = std::unique_ptr<void, CudaMemDeleter>;
    std::array<DeviceBuffer, 2> device_buffers_{};
    cudaStream_t stream_{};

    int input_c_{3};
    int input_h_{640};
    int input_w_{640};
    int max_output_boxes_{8400};
    int num_classes_{80};
    size_t input_buffer_size_{0};
    size_t output_buffer_size_{0};
    std::string input_tensor_name_{"images"};
    std::string output_tensor_name_{"output0"};

    float confidence_threshold_;
    float nms_threshold_;
    bool use_letterbox_;
    float inference_time_ms_{0.f};
};

}  // namespace algorithms
