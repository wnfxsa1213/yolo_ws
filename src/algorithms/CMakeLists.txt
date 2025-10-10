cmake_minimum_required(VERSION 3.18)
project(detection_core LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          HINTS /usr/include /usr/include/aarch64-linux-gnu /usr/local/include)

find_library(TENSORRT_LIBRARY nvinfer
             HINTS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib/aarch64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or libraries未找到，请确认已安装TensorRT SDK。")
endif()

if(NOT Python3_EXTENSION_MODULE_SUFFIX)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
        OUTPUT_VARIABLE Python3_EXTENSION_MODULE_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT Python3_EXTENSION_MODULE_SUFFIX)
        message(FATAL_ERROR "艹，Python扩展模块后缀没拿到，确认Python环境没出幺蛾子。")
    endif()
endif()

add_library(detection_core MODULE
    src/detector.cpp
    src/coordinate.cpp
    src/bindings.cpp
)

target_include_directories(detection_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(detection_core
    PRIVATE
        pybind11::module
        ${OpenCV_LIBS}
        CUDA::cudart
        ${TENSORRT_LIBRARY}
)

target_compile_options(detection_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native>
)

set_target_properties(detection_core PROPERTIES
    PREFIX ""
    SUFFIX "${Python3_EXTENSION_MODULE_SUFFIX}"
)

install(TARGETS detection_core
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
