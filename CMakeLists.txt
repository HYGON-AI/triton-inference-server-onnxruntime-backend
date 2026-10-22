# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(triton_onnx_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# 1. 查找 ROCm 与 HIP 环境
# ==============================================================================
if(NOT DEFINED ENV{DTK_PATH})
  set(DTK_PATH "/opt/dtk" CACHE PATH "Path to ROCm installation")
else()
  set(DTK_PATH $ENV{DTK_PATH} CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED)
message(STATUS "Found HIP: ${hip_VERSION} at ${ROCM_PATH}")

# ==============================================================================
# 2. 查找 Onnxruntime
# ==============================================================================
# 1. 查找 Python3 解释器
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# 2. 从 Python 中直接获取 onnxruntime/capi 所在的绝对路径
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import onnxruntime as ort, os; print(os.path.dirname(ort._file_) if hasattr(ort, '_file_') else os.path.join(os.path.dirname(ort.__file__), 'capi'))"
    OUTPUT_VARIABLE ORT_PYTHON_CAPI_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ORT_FIND_STATUS
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import onnxruntime; print(onnxruntime.__version__)"
    OUTPUT_VARIABLE ORT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ORT_VER_STATUS
)
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ORT_HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime_c_api.h")

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
	        "${ORT_PYTHON_CAPI_DIR}/libonnxruntime.so.${ORT_VERSION}" "${ORT_PYTHON_CAPI_DIR}/libonnxruntime.so.1"
		 RESULT_VARIABLE ORT_SYMLINK_RES)

if (NOT EXISTS ${ORT_HEADER_FILE} OR NOT ${ORT_VERSION} VERSION_EQUAL "1.19.2")
    message(STATUS "正在下载版本 v${ORT_VERSION} 的 onnxruntime_c_api.h ...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/include/onnxruntime/core/session/onnxruntime_c_api.h"
        "${ORT_HEADER_FILE}"
        STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(FATAL_ERROR "头文件下载失败，请检查网络或版本 Tag 是否存在！")
    endif()
endif()

# ==============================================================================
# 3. 引入 Triton Backend 辅助库 (triton-backend-utils)
# ==============================================================================
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Triton Server release branch/tag")

#include(FetchContent)

# 依赖 1: Triton 通用后端基类仓库 (BackendModel, BackendModelInstance 等)
#FetchContent_Declare(
#  TRIton-backend-repo
#  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
#  git_TAG ${TRITON_CORE_REPO_TAG}
#)

#FetchContent_MakeAvailable(triton-backend-repo)

# ==============================================================================
# 4. 构建 MIGraphX 后端共享库
# ==============================================================================
# Triton 约定的后端共享库命名格式为 libtriton_<backend_name>.so
add_library(triton_onnx SHARED
  src/onnx_backend.cc
)

# 设置头文件包含目录
target_include_directories(triton_onnx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${DTK_PATH}/include
  ${TRITON_BACKEND_PATH}/include
  ${TRITON_CORE_PATH}/include/
)

message(STATUS "libonnxruntime:"${ORT_PYTHON_CAPI_DIR}/libonnxruntime.so.${ORT_VERSION})

add_library(onnx_runtime SHARED IMPORTED)
set_target_properties(onnx_runtime PROPERTIES
	IMPORTED_LOCATION ${ORT_PYTHON_CAPI_DIR}/libonnxruntime.so.${ORT_VERSION})


add_library(triton-backend-utils STATIC IMPORTED GLOBAL)
set_target_properties(triton-backend-utils PROPERTIES
	IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritonbackendutils.a)
add_library(triton-common-async-work-queue STATIC IMPORTED GLOBAL)
set_target_properties(triton-common-async-work-queue PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritonasyncworkqueue.a)
add_library(triton-common-logging STATIC IMPORTED GLOBAL)
set_target_properties(triton-common-logging PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritoncommonlogging.a)

add_library(kernel_library_new STATIC IMPORTED GLOBAL)
set_target_properties(kernel_library_new PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libkernel_library_new.a)

# 链接依赖项：MIGraphX C API、ROCm HIP 运行时、Triton 后端辅助工具库
target_link_libraries(triton_onnx PRIVATE
  hip::host
  onnx_runtime
  -Wl,--whole-archive
  triton-backend-utils
  triton-common-async-work-queue
  kernel_library_new
  -Wl,--no-whole-archive
  triton-common-logging
)

# 去除 lib 前缀，保持输出为 libtriton_migraphx.so
set_target_properties(triton_onnx PROPERTIES
  OUTPUT_NAME "triton_onnx"
  PREFIX "lib"
  POSITION_INDEPENDENT_CODE ON
  )
set_target_properties(triton_onnx PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH ${ORT_PYTHON_CAPI_DIR}
)

# ==============================================================================
# 5. 安装规则 (符合 Triton 后端目录结构)
# ==============================================================================
# 安装目录结构: <install_prefix>/backends/onnx/libonnx_runtime.so
install(TARGETS triton_onnx
  DESTINATION backends/onnx
)
