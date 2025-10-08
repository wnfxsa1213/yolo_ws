#include "detector.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace algorithms {
namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

TrtLogger g_logger;

float iou(const Detection& a, const Detection& b) {
    const float x_left = std::max(a.x1, b.x1);
    const float y_top = std::max(a.y1, b.y1);
    const float x_right = std::min(a.x2, b.x2);
    const float y_bottom = std::min(a.y2, b.y2);

    if (x_right <= x_left || y_bottom <= y_top) {
        return 0.0f;
    }
    const float intersection = (x_right - x_left) * (y_bottom - y_top);
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return intersection / (area_a + area_b - intersection + 1e-6f);
}

}  // namespace

namespace {
template <typename T>
void destroy_trt_object(T* ptr) noexcept {
    if (!ptr) {
        return;
    }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    delete ptr;
#else
    ptr->destroy();
#endif
}
}  // namespace

void YOLODetector::TrtDeleter::operator()(nvinfer1::IRuntime* ptr) const noexcept {
    destroy_trt_object(ptr);
}

void YOLODetector::TrtDeleter::operator()(nvinfer1::ICudaEngine* ptr) const noexcept {
    destroy_trt_object(ptr);
}

void YOLODetector::TrtDeleter::operator()(nvinfer1::IExecutionContext* ptr) const noexcept {
    destroy_trt_object(ptr);
}

void YOLODetector::CudaMemDeleter::operator()(void* ptr) const noexcept {
    if (ptr) {
        cudaFree(ptr);
    }
}

void YOLODetector::check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
}

YOLODetector::YOLODetector(const std::string& engine_path, float confidence_threshold,
                           float nms_threshold, bool use_letterbox)
    : confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold),
      use_letterbox_(use_letterbox) {
    load_engine(engine_path);
    allocate_buffers();
}

YOLODetector::~YOLODetector() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void YOLODetector::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开TensorRT引擎文件: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    const size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), static_cast<std::streamsize>(size));
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>(
        nvinfer1::createInferRuntime(g_logger));
    if (!runtime) {
        throw std::runtime_error("创建TensorRT runtime失败");
    }

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>(
        runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!engine) {
        throw std::runtime_error("反序列化TensorRT引擎失败");
    }

    auto context = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>(
        engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("创建TensorRT执行上下文失败");
    }

    const int32_t nb_tensors = engine->getNbIOTensors();
    bool found_input = false;
    bool found_output = false;
    for (int32_t i = 0; i < nb_tensors; ++i) {
        const char* tensor_name = engine->getIOTensorName(i);
        if (!tensor_name) {
            continue;
        }
        const std::string name_str{tensor_name};
        const auto mode = engine->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && name_str == input_tensor_name_) {
            found_input = true;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && name_str == output_tensor_name_) {
            found_output = true;
        }
    }
    if (!found_input || !found_output) {
        throw std::runtime_error("TensorRT引擎缺少指定的输入/输出张量名称");
    }

    nvinfer1::Dims input_dims = engine->getTensorShape(input_tensor_name_.c_str());
    if (input_dims.nbDims == 0 || input_dims.nbDims > 4) {
        throw std::runtime_error("输入维度不合法");
    }

    if (input_dims.nbDims == 1) {
        input_dims = nvinfer1::Dims4{1, input_c_, input_h_, input_w_};
    } else {
        input_c_ = input_dims.nbDims > 1 && input_dims.d[1] > 0 ? input_dims.d[1] : input_c_;
        input_h_ = input_dims.nbDims > 2 && input_dims.d[2] > 0 ? input_dims.d[2] : input_h_;
        input_w_ = input_dims.nbDims > 3 && input_dims.d[3] > 0 ? input_dims.d[3] : input_w_;
    }

    const nvinfer1::Dims4 static_input{1, input_c_, input_h_, input_w_};
    if (!context->setInputShape(input_tensor_name_.c_str(), static_input)) {
        throw std::runtime_error("设置输入张量形状失败");
    }

    nvinfer1::Dims output_dims = context->getTensorShape(output_tensor_name_.c_str());
    if (output_dims.nbDims < 2) {
        throw std::runtime_error("输出维度不合法");
    }

    int values = 0;
    int boxes = 0;
    if (output_dims.nbDims == 3) {
        const int dim1 = output_dims.d[1];
        const int dim2 = output_dims.d[2];
        if (dim1 <= dim2) {
            values = dim1;
            boxes = dim2;
        } else {
            boxes = dim1;
            values = dim2;
        }
    } else if (output_dims.nbDims == 2) {
        boxes = output_dims.d[0];
        values = output_dims.d[1];
    } else {
        boxes = output_dims.d[output_dims.nbDims - 2];
        values = output_dims.d[output_dims.nbDims - 1];
    }

    if (boxes <= 0 || values <= 0) {
        throw std::runtime_error("输出张量形状非法");
    }

    num_classes_ = values > 4 ? (values - 4) : num_classes_;
    max_output_boxes_ = boxes;

    runtime_ = std::move(runtime);
    engine_ = std::move(engine);
    context_ = std::move(context);
}

void YOLODetector::allocate_buffers() {
    input_buffer_size_ =
        static_cast<size_t>(input_c_) * input_h_ * input_w_ * sizeof(float);
    output_buffer_size_ =
        static_cast<size_t>(max_output_boxes_) * (num_classes_ + 4) * sizeof(float);

    try {
        void* input_ptr = nullptr;
        check_cuda(cudaMalloc(&input_ptr, input_buffer_size_));
        device_buffers_[0].reset(input_ptr);

        void* output_ptr = nullptr;
        check_cuda(cudaMalloc(&output_ptr, output_buffer_size_));
        device_buffers_[1].reset(output_ptr);

        check_cuda(cudaStreamCreate(&stream_));
    } catch (...) {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        device_buffers_[0].reset();
        device_buffers_[1].reset();
        throw;
    }
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("输入图像为空");
    }

    float scale = 1.0f;
    int pad_w = 0;
    int pad_h = 0;
    std::vector<float> host_input(input_buffer_size_ / sizeof(float));
    preprocess(image, host_input, scale, pad_w, pad_h);

    std::vector<float> host_output(output_buffer_size_ / sizeof(float));

    const auto t0 = std::chrono::high_resolution_clock::now();

    if (!context_->setInputShape(input_tensor_name_.c_str(),
                                 nvinfer1::Dims4{1, input_c_, input_h_, input_w_})) {
        throw std::runtime_error("设置输入形状失败");
    }
    if (!context_->setTensorAddress(input_tensor_name_.c_str(), device_buffers_[0].get())) {
        throw std::runtime_error("绑定输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_tensor_name_.c_str(), device_buffers_[1].get())) {
        throw std::runtime_error("绑定输出张量地址失败");
    }

    check_cuda(cudaMemcpyAsync(device_buffers_[0].get(), host_input.data(), input_buffer_size_,
                               cudaMemcpyHostToDevice, stream_));

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT推理执行失败");
    }

    check_cuda(cudaMemcpyAsync(host_output.data(), device_buffers_[1].get(), output_buffer_size_,
                               cudaMemcpyDeviceToHost, stream_));
    check_cuda(cudaStreamSynchronize(stream_));

    const auto t1 = std::chrono::high_resolution_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return postprocess(host_output, image.size(), scale, pad_w, pad_h);
}

void YOLODetector::preprocess(const cv::Mat& image, std::vector<float>& host_buffer, float& scale,
                              int& pad_w, int& pad_h) {
    cv::Mat converted;
    if (use_letterbox_) {
        letterbox_resize(image, converted, input_w_, input_h_, scale, pad_w, pad_h);
    } else {
        cv::resize(image, converted, cv::Size(input_w_, input_h_));
        scale = std::min(static_cast<float>(input_w_) / image.cols,
                         static_cast<float>(input_h_) / image.rows);
        pad_w = pad_h = 0;
    }

    cv::cvtColor(converted, converted, cv::COLOR_BGR2RGB);
    converted.convertTo(converted, CV_32FC3, 1.0f / 255.0f);

    const int area = input_h_ * input_w_;
    const float* data = converted.ptr<float>();

    // HWC -> CHW
    for (int c = 0; c < input_c_; ++c) {
        for (int y = 0; y < input_h_; ++y) {
            for (int x = 0; x < input_w_; ++x) {
                host_buffer[c * area + y * input_w_ + x] = data[y * input_w_ * input_c_ + x * input_c_ + c];
            }
        }
    }
}

std::vector<Detection> YOLODetector::postprocess(const std::vector<float>& output,
                                                 const cv::Size& orig_size, float scale,
                                                 int pad_w, int pad_h) {
    std::vector<Detection> proposals;
    proposals.reserve(max_output_boxes_);

    const int boxes = max_output_boxes_;
    const int values = num_classes_ + 4;
    const bool channels_first = output.size() == static_cast<size_t>(boxes * values) &&
                                boxes > values;

    auto get_value = [&](int box_id, int idx) -> float {
        if (channels_first) {
            return output[idx * boxes + box_id];
        }
        return output[box_id * values + idx];
    };

    const float pad_w_half = pad_w / 2.0f;
    const float pad_h_half = pad_h / 2.0f;

    for (int i = 0; i < boxes; ++i) {
        float bx = get_value(i, 0);
        float by = get_value(i, 1);
        float bw = get_value(i, 2);
        float bh = get_value(i, 3);

        int best_class = -1;
        float best_score = 0.0f;
        for (int cls = 0; cls < num_classes_; ++cls) {
            float score = get_value(i, 4 + cls);
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }

        if (best_score < confidence_threshold_) {
            continue;
        }

        const float x_center = (bx - pad_w_half) / scale;
        const float y_center = (by - pad_h_half) / scale;
        const float box_w = bw / scale;
        const float box_h = bh / scale;

        Detection det{};
        det.x1 = std::max(0.0f, x_center - box_w / 2.0f);
        det.y1 = std::max(0.0f, y_center - box_h / 2.0f);
        det.x2 = std::min(static_cast<float>(orig_size.width), x_center + box_w / 2.0f);
        det.y2 = std::min(static_cast<float>(orig_size.height), y_center + box_h / 2.0f);
        det.confidence = best_score;
        det.class_id = best_class;
        proposals.push_back(det);
    }

    // NMS
    std::sort(proposals.begin(), proposals.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<Detection> results;
    std::vector<bool> removed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        const Detection& det = proposals[i];
        results.push_back(det);

        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (removed[j]) {
                continue;
            }
            if (det.class_id == proposals[j].class_id && iou(det, proposals[j]) > nms_threshold_) {
                removed[j] = true;
            }
        }
    }

    return results;
}

void YOLODetector::letterbox_resize(const cv::Mat& image, cv::Mat& output, int target_w,
                                    int target_h, float& scale, int& pad_w, int& pad_h) {
    const float scale_w = static_cast<float>(target_w) / static_cast<float>(image.cols);
    const float scale_h = static_cast<float>(target_h) / static_cast<float>(image.rows);
    scale = std::min(scale_w, scale_h);
    const int new_w = static_cast<int>(std::round(image.cols * scale));
    const int new_h = static_cast<int>(std::round(image.rows * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    pad_w = target_w - new_w;
    pad_h = target_h - new_h;
    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    cv::copyMakeBorder(resized, output, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

}  // namespace algorithms
