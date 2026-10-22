/**************************************************************************************************
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
**************************************************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <algorithm>

#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

// Triton Backend API
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_output_responder.h>
#include <triton/core/tritonserver.h>
#include <triton/common/triton_json.h>

// ONNX Runtime C API
#include <include/onnxruntime_c_api.h>

// ROCm / HIP Runtime API
#include <hip/hip_runtime.h>

// -----------------------------------------------------------------------------
// 全局 API 表与错误处理宏
// -----------------------------------------------------------------------------
static const OrtApi* g_ort = nullptr;

#define CHECK_ORT(cmd)                                                                  \
  do {                                                                                  \
    OrtStatus* status = (cmd);                                                          \
    if (status != nullptr) {                                                            \
      const char* msg = g_ort->GetErrorMessage(status);                                 \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, \
          (std::string("ONNX Runtime Error: ") + (msg ? msg : "unknown")).c_str());                      \
      g_ort->ReleaseStatus(status);                                                     \
    }                                                                                   \
  } while (0)

#define CHECK_HIP(cmd)                                                                  \
  do {                                                                                  \
    hipError_t err = (cmd);                                                             \
    if (err != hipSuccess) {                                                            \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, \
          (std::string("HIP Error: ") + hipGetErrorString(err)).c_str());                 \
    }                                                                                   \
  } while (0)

// -----------------------------------------------------------------------------
// 数据类型双向转换辅助函数
// -----------------------------------------------------------------------------
static ONNXTensorElementDataType TritonToOrtType(TRITONSERVER_DataType dtype) {
  switch (dtype) {
    case TRITONSERVER_TYPE_FP32:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TRITONSERVER_TYPE_FP16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TRITONSERVER_TYPE_INT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TRITONSERVER_TYPE_INT8:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TRITONSERVER_TYPE_UINT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TRITONSERVER_TYPE_BOOL:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case TRITONSERVER_TYPE_FP64:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    default:                      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
}

static TRITONSERVER_DataType OrtToTritonType(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return TRITONSERVER_TYPE_FP32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TRITONSERVER_TYPE_FP16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return TRITONSERVER_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return TRITONSERVER_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return TRITONSERVER_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return TRITONSERVER_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return TRITONSERVER_TYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return TRITONSERVER_TYPE_FP64;
    default:                                    return TRITONSERVER_TYPE_FP32;
  }
}

enum class ModelFormat {
  UNKNOWN,
  ONNX
};

struct ModelPathInfo {
  std::string path;
  ModelFormat format;
};

static bool PathExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

ModelPathInfo DetectModelPathAndFormat(TRITONBACKEND_ModelInstance* instance) {
  TRITONBACKEND_Model* model = nullptr;
  TRITONBACKEND_ModelInstanceModel(instance, &model);

  uint64_t model_version = 0;
  TRITONBACKEND_ModelVersion(model, &model_version);

  const char* repo_path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  TRITONBACKEND_ModelRepository(model, &artifact_type, &repo_path);

  std::string base_dir = std::string(repo_path) + "/" + std::to_string(model_version);
  std::string onnx_file = base_dir + "/model.onnx";
  if (PathExists(onnx_file)) {
    return {onnx_file, ModelFormat::ONNX};
  }

  return {"", ModelFormat::UNKNOWN};
}

// -----------------------------------------------------------------------------
// 专用于输入的持久化显存块结构
// -----------------------------------------------------------------------------
struct PersistentInputBuffer {
  void* device_ptr = nullptr;
  size_t allocated_bytes = 0;

  void EnsureCapacity(size_t required_bytes) {
    if (required_bytes <= allocated_bytes) {
      return; // 容量充足，直接就地复用，无分配开销
    }

    if (device_ptr != nullptr) {
      CHECK_HIP(hipFree(device_ptr));
      device_ptr = nullptr;
    }

    // 采用 1.5 倍增策略避免频繁小幅度重分配
    size_t new_capacity = std::max(allocated_bytes + (allocated_bytes >> 1), required_bytes);
    CHECK_HIP(hipMalloc(&device_ptr, new_capacity));
    allocated_bytes = new_capacity;
  }

  void Release() {
    if (device_ptr != nullptr) {
      CHECK_HIP(hipFree(device_ptr));
      device_ptr = nullptr;
      allocated_bytes = 0;
    }
  }
};

class ModelInstanceState {
 public:
  ModelInstanceState(TRITONBACKEND_ModelInstance* triton_model_instance);
  ~ModelInstanceState();

  void ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  void InitOrt();

  TRITONBACKEND_ModelInstance* triton_model_instance_;
  int32_t device_id_ = 0;
  uint64_t max_batch_size_ = 0;  // 从 model config 读取

  OrtEnv* env_ = nullptr;
  OrtSession* session_ = nullptr;
  OrtMemoryInfo* gpu_mem_info_ = nullptr;
  OrtAllocator* default_allocator_ = nullptr;

  std::vector<std::string> model_output_names_;

  // 预分配/持久化的输入显存池：TensorName -> PersistentInputBuffer
  std::unordered_map<std::string, PersistentInputBuffer> input_buffers_;
};

ModelInstanceState::ModelInstanceState(TRITONBACKEND_ModelInstance* triton_model_instance)
    : triton_model_instance_(triton_model_instance) {
  InitOrt();
}

ModelInstanceState::~ModelInstanceState() {
  CHECK_HIP(hipSetDevice(device_id_));

  // 释放持久化输入显存池
  for (auto& kv : input_buffers_) {
    kv.second.Release();
  }
  input_buffers_.clear();

  if (gpu_mem_info_ != nullptr) {
    g_ort->ReleaseMemoryInfo(gpu_mem_info_);
    gpu_mem_info_ = nullptr;
  }
  if (session_ != nullptr) {
    g_ort->ReleaseSession(session_);
    session_ = nullptr;
  }
  if (env_ != nullptr) {
    g_ort->ReleaseEnv(env_);
    env_ = nullptr;
  }
}

void ModelInstanceState::InitOrt() {
  if (g_ort == nullptr) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  }

  TRITONSERVER_InstanceGroupKind kind;
  TRITONBACKEND_ModelInstanceKind(triton_model_instance_, &kind);
  TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance_, &device_id_);
  CHECK_HIP(hipSetDevice(device_id_));
  // 读取模型配置的 max_batch_size: 区分 batching (max_batch_size>0) 与
  // non-batching (max_batch_size==0, 此时请求 dims 已含完整 shape)。
  {
    TRITONBACKEND_Model* triton_model = nullptr;
    TRITONBACKEND_ModelInstanceModel(triton_model_instance_, &triton_model);
    if (triton_model != nullptr) {
      TRITONSERVER_Message* config_message = nullptr;
      if (TRITONBACKEND_ModelConfig(triton_model, 1, &config_message) == nullptr &&
          config_message != nullptr) {
        const char* config_json = nullptr;
        size_t config_json_size = 0;
        if (TRITONSERVER_MessageSerializeToJson(
                config_message, &config_json, &config_json_size) == nullptr) {
          triton::common::TritonJson::Value cfg;
          if (cfg.Parse(config_json, config_json_size) == nullptr) {
            int64_t mb = 0;
            if (cfg.MemberAsInt("max_batch_size", &mb) == nullptr && mb > 0) {
              max_batch_size_ = static_cast<uint64_t>(mb);
            }
          }
        }
      }
    }
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("[BACKEND] max_batch_size=") + std::to_string(max_batch_size_) +
         (max_batch_size_ > 0 ? " (batching)" : " (non-batching)")).c_str());
  }


  ModelPathInfo model_info = DetectModelPathAndFormat(triton_model_instance_);
  if (model_info.format == ModelFormat::UNKNOWN) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to find supported model (model.onnx)");
    return;
  }

  // 1. 初始化 Environment 与 SessionOptions
  CHECK_ORT(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Triton_ORT_ROCm", &env_));

  OrtSessionOptions* session_options = nullptr;
  CHECK_ORT(g_ort->CreateSessionOptions(&session_options));

  // 2. 启用 ROCm Execution Provider
  OrtROCMProviderOptions rocm_options;
  std::memset(&rocm_options, 0, sizeof(rocm_options));
  rocm_options.device_id = device_id_;
  rocm_options.miopen_conv_exhaustive_search = 0;
  rocm_options.gpu_mem_limit = 0;
  rocm_options.arena_extend_strategy = 0;

  OrtStatus* rocm_status = g_ort->SessionOptionsAppendExecutionProvider_ROCM(session_options, &rocm_options);
  if (rocm_status != nullptr) {
    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
        (std::string("Failed to append ROCm EP, falling back to CPU: ") +
         (g_ort->GetErrorMessage(rocm_status) ? g_ort->GetErrorMessage(rocm_status) : "unknown")).c_str());
    g_ort->ReleaseStatus(rocm_status);
  }

  // 3. 构建 Session
  CHECK_ORT(g_ort->CreateSession(env_, model_info.path.c_str(), session_options, &session_));
  g_ort->ReleaseSessionOptions(session_options);

  // 4. 显存抽象句柄
  CHECK_ORT(g_ort->GetAllocatorWithDefaultOptions(&default_allocator_));
  CHECK_ORT(g_ort->CreateMemoryInfo(
      "Hip", OrtDeviceAllocator, device_id_, OrtMemTypeDefault, &gpu_mem_info_));

  // 5. 获取并缓存输出节点名称
  size_t output_count = 0;
  CHECK_ORT(g_ort->SessionGetOutputCount(session_, &output_count));
  model_output_names_.resize(output_count);

  for (size_t i = 0; i < output_count; ++i) {
    char* out_name = nullptr;
    CHECK_ORT(g_ort->SessionGetOutputName(session_, i, default_allocator_, &out_name));
    model_output_names_[i] = out_name;
    g_ort->AllocatorFree(default_allocator_, out_name);
  }
}

void ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count) {

  CHECK_HIP(hipSetDevice(device_id_));

  std::vector<TRITONBACKEND_Request*> valid_requests;
  valid_requests.reserve(request_count);

  // 仅 batching 模式 (max_batch_size>0) 下, Triton 才会在请求 dims 的第 0 维
  // 前置 batch 维; non-batching 模式请求 dims 即完整 shape, 不含 batch。
  const bool batching = (max_batch_size_ > 0);
  uint64_t total_batch_size = 0;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    if (request == nullptr) continue;
    valid_requests.push_back(request);
    if (batching) {
      TRITONBACKEND_Input* first_in = nullptr;
      TRITONBACKEND_RequestInputByIndex(request, 0, &first_in);
      if (first_in != nullptr) {
        const int64_t* req_dims = nullptr;
        uint32_t req_dims_count = 0;
        TRITONBACKEND_InputProperties(
            first_in, nullptr, nullptr, &req_dims, &req_dims_count, nullptr, nullptr);
        total_batch_size += (req_dims_count > 0) ? req_dims[0] : 1;
      } else {
        total_batch_size += 1;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // 批处理可观测性: 每次 execute 收到的请求数及合并后总 batch
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
      (std::string("[BATCH] execute: request_count=") +
       std::to_string(valid_requests.size()) +
       " total_batch_size=" + std::to_string(total_batch_size) +
       (batching ? " (batching)" : " (non-batching)")).c_str());

  if (valid_requests.empty()) return;

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(valid_requests.size());
  for (auto* req : valid_requests) {
    TRITONBACKEND_Response* resp = nullptr;
    TRITONBACKEND_ResponseNew(&resp, req);
    responses.push_back(resp);
  }

  TRITONBACKEND_MemoryManager* memory_manager = nullptr;
  TRITONBACKEND_Model* triton_model = nullptr;
  TRITONBACKEND_ModelInstanceModel(triton_model_instance_, &triton_model);
  TRITONBACKEND_Backend* triton_backend = nullptr;
  TRITONBACKEND_ModelBackend(triton_model, &triton_backend);
  TRITONBACKEND_BackendMemoryManager(triton_backend, &memory_manager);

  std::vector<std::string> input_name_strings;
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_tensors;

  // --------------------------------------------------------------------------
  // 1. 输入数据归集 (零 hipMalloc 路径)
  // --------------------------------------------------------------------------
  uint32_t input_count = 0;
  TRITONBACKEND_RequestInputCount(valid_requests[0], &input_count);

  for (uint32_t i = 0; i < input_count; ++i) {
    TRITONBACKEND_Input* in = nullptr;
    TRITONBACKEND_RequestInputByIndex(valid_requests[0], i, &in);

    const char* in_name_cstr = nullptr;
    TRITONSERVER_DataType in_dtype;
    const int64_t* in_dims = nullptr;
    uint32_t in_dims_count = 0;
    uint64_t per_req_byte_size = 0;
    uint32_t dummy_buf_count = 0;

    TRITONBACKEND_InputProperties(
        in, &in_name_cstr, &in_dtype, &in_dims, &in_dims_count, &per_req_byte_size, &dummy_buf_count);

    std::string tensor_name(in_name_cstr);
    input_name_strings.push_back(tensor_name);

    // 计算当前批次该张量的实际需求显存
    size_t total_tensor_bytes = 0;
    for (auto* req : valid_requests) {
      TRITONBACKEND_Input* req_in = nullptr;
      TRITONBACKEND_RequestInput(req, tensor_name.c_str(), &req_in);
      uint64_t req_bytes = 0;
      TRITONBACKEND_InputProperties(req_in, nullptr, nullptr, nullptr, nullptr, &req_bytes, nullptr);
      total_tensor_bytes += req_bytes;
    }

    // 从常驻池中取用，若容量不够才扩容；平时直接复用已分配指针
    PersistentInputBuffer& persistent_buf = input_buffers_[tensor_name];
    persistent_buf.EnsureCapacity(total_tensor_bytes);
    void* gpu_input_buffer = persistent_buf.device_ptr;

    // 拼接聚合各个请求的数据
    size_t current_offset = 0;
    for (auto* req : valid_requests) {
      TRITONBACKEND_Input* req_in = nullptr;
      TRITONBACKEND_RequestInput(req, tensor_name.c_str(), &req_in);

      uint32_t buf_count = 0;
      TRITONBACKEND_InputProperties(req_in, nullptr, nullptr, nullptr, nullptr, nullptr, &buf_count);

      for (uint32_t b = 0; b < buf_count; ++b) {
        const void* src_buffer = nullptr;
        uint64_t src_byte_size = 0;
        TRITONSERVER_MemoryType src_mem_type;
        int64_t src_mem_id = 0;

        TRITONBACKEND_InputBuffer(
            req_in, b, &src_buffer, &src_byte_size, &src_mem_type, &src_mem_id);

        char* dst_ptr = reinterpret_cast<char*>(gpu_input_buffer) + current_offset;

        if (src_mem_type == TRITONSERVER_MEMORY_GPU) {
          CHECK_HIP(hipMemcpyDtoD(dst_ptr, const_cast<void*>(src_buffer), src_byte_size));
        } else {
          CHECK_HIP(hipMemcpyHtoD(dst_ptr, const_cast<void*>(src_buffer), src_byte_size));
        }
        current_offset += src_byte_size;
      }
    }

    // 组合输入 Shape
    std::vector<int64_t> in_shape(in_dims, in_dims + in_dims_count);
    if (batching && !in_shape.empty()) {
      // batching: 用请求累计 batch 覆盖 dims[0]
      in_shape[0] = static_cast<int64_t>(total_batch_size);
    }

    // 用预分配缓冲区的指针包装为 OrtValue
    OrtValue* input_val = nullptr;
    CHECK_ORT(g_ort->CreateTensorWithDataAsOrtValue(
        gpu_mem_info_,
        gpu_input_buffer,
        total_tensor_bytes,
        in_shape.data(),
        in_shape.size(),
        TritonToOrtType(in_dtype),
        &input_val));

    input_tensors.push_back(input_val);
  }

  for (const auto& name : input_name_strings) {
    input_names.push_back(name.c_str());
  }

  // --------------------------------------------------------------------------
  // 2. 准备输出参数
  // --------------------------------------------------------------------------
  std::vector<const char*> output_names;
  for (const auto& out_name : model_output_names_) {
    output_names.push_back(out_name.c_str());
  }

  std::vector<OrtValue*> output_tensors(output_names.size(), nullptr);

  // --------------------------------------------------------------------------
  // 3. 执行推理 (ONNX Runtime 内部通过 ROCm EP 内存池管理输出)
  // --------------------------------------------------------------------------
  CHECK_ORT(g_ort->Run(
      session_,
      nullptr,
      input_names.data(),
      input_tensors.data(),
      input_tensors.size(),
      output_names.data(),
      output_names.size(),
      output_tensors.data()));

  // --------------------------------------------------------------------------
  // 4. 写回响应
  // --------------------------------------------------------------------------
  std::unique_ptr<triton::backend::BackendOutputResponder> responder(
      new triton::backend::BackendOutputResponder(
          valid_requests.data(), valid_requests.size(), &responses,
          static_cast<int>(max_batch_size_) /* max_batch_size */,
          memory_manager, false, nullptr /* stream */));

  for (size_t out_idx = 0; out_idx < output_tensors.size(); ++out_idx) {
    OrtValue* out_val = output_tensors[out_idx];
    if (out_val == nullptr) continue;

    void* out_device_data = nullptr;
    CHECK_ORT(g_ort->GetTensorMutableData(out_val, &out_device_data));

    OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
    CHECK_ORT(g_ort->GetTensorTypeAndShape(out_val, &type_shape_info));

    ONNXTensorElementDataType ort_type;
    CHECK_ORT(g_ort->GetTensorElementType(type_shape_info, &ort_type));

    size_t dim_count = 0;
    CHECK_ORT(g_ort->GetDimensionsCount(type_shape_info, &dim_count));
    std::vector<int64_t> out_dims(dim_count);
    CHECK_ORT(g_ort->GetDimensions(type_shape_info, out_dims.data(), dim_count));

    g_ort->ReleaseTensorTypeAndShapeInfo(type_shape_info);

    TRITONSERVER_DataType triton_out_dtype = OrtToTritonType(ort_type);
    std::string out_name = model_output_names_[out_idx];
    
    // 获取张量的物理内存信息
    const OrtMemoryInfo* mem_info = nullptr;
    CHECK_ORT(g_ort->GetTensorMemoryInfo(out_val, &mem_info));
    
    OrtMemType mem_type;
    CHECK_ORT(g_ort->MemoryInfoGetMemType(mem_info, &mem_type));
    
    // 也可以获取 allocator 名称判断是 CPU 还是 GPU
    const char* alloc_name = nullptr;
    CHECK_ORT(g_ort->MemoryInfoGetName(mem_info, &alloc_name));
    
    TRITONSERVER_MemoryType triton_mem_type = TRITONSERVER_MEMORY_GPU;
    int64_t actual_device_id = device_id_;
    
    // 如果分配器名称包含 Cpu，或者内存类型是 CPU 内存
    if (std::string(alloc_name).find("Cpu") != std::string::npos ||
        std::string(alloc_name).find("CPU") != std::string::npos) {
      triton_mem_type = TRITONSERVER_MEMORY_CPU;
      actual_device_id = 0;
    }

    responder->ProcessTensor(
        out_name, triton_out_dtype, out_dims,
        reinterpret_cast<const char*>(out_device_data), triton_mem_type, device_id_);
  }

  CHECK_HIP(hipStreamSynchronize(hipStreamDefault));

  // --------------------------------------------------------------------------
  // 5. 触发响应与释放 Triton Request
  // --------------------------------------------------------------------------
  for (size_t i = 0; i < responses.size(); ++i) {
    if (responses[i] != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[i], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(
            valid_requests[i], TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release request");
  }

  // --------------------------------------------------------------------------
  // 6. 仅回收轻量级 OrtValue 句柄（底层 GPU Buffer 保留在 input_buffers_ 中）
  // --------------------------------------------------------------------------
  for (auto* tensor : output_tensors) {
    if (tensor != nullptr) g_ort->ReleaseValue(tensor);
  }
  for (auto* tensor : input_tensors) {
    if (tensor != nullptr) g_ort->ReleaseValue(tensor);
  }
}

extern "C" {

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state = nullptr;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

  instance_state->ProcessRequests(requests, request_count);
  return nullptr;
}

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  auto instance_state = new ModelInstanceState(instance);
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));
  return nullptr;
}

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  if (vstate != nullptr) {
    delete reinterpret_cast<ModelInstanceState*>(vstate);
  }
  return nullptr;
}

} // extern "C"
