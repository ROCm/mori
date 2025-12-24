// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Do not compile device-side code since we do not have any kernels here..
#ifndef __HIP_DEVICE_COMPILE__

#include "src/pybind/mori.hpp"

#ifdef MORI_ENABLE_TORCH
#include <ATen/hip/HIPContext.h>
#endif
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef MORI_ENABLE_TORCH
#include <torch/python.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "src/pybind/torch_utils.hpp"
#else
namespace py = pybind11;
#endif

#include "mori/application/application.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
// #include "xla/pjrt/c/pjrt_c_api_helpers.h"

using namespace xla::ffi;
using mori::moe::EpDispatchCombineConfig;
using mori::moe::EpDispatchCombineHandle;
using mori::moe::KernelType;
using mori::moe::index_t;

namespace pjrt {

template <typename ExtType, typename InputType>
ExtType* FindExtension(InputType* in, PJRT_Extension_Type type) {
  PJRT_Extension_Base* ext = in->extension_start;
  while (ext != nullptr) {
    if (ext->type == type) {
      return reinterpret_cast<ExtType*>(ext);
    }
    ext = ext->next;
  }
  // 'type' wasn't found in extension chain
  return nullptr;
}
} // namespace pjrt


/* ---------------------------------------------------------------------------------------------- */
/*                                          XLA Ops APIs                                          */
/* ---------------------------------------------------------------------------------------------- */
namespace {

hipDataType FFIType2HipType(DataType dtype) {
#define XX(A, B) case DataType::A: return B;
  switch (dtype) {
    XX(S32, HIP_R_32F) // we interpret as float32!
    XX(F32, HIP_R_32F)
    XX(BF16, HIP_R_16BF)
    XX(F8E4M3FN, HIP_R_8F_E4M3)
    XX(F8E4M3FNUZ, HIP_R_8F_E4M3_FNUZ)
    default:
      throw std::runtime_error("Unsupported scalar type: " + std::to_string((int)dtype));
  }
#undef XX
}

template <typename T>
T get_attr_value(Dictionary& attrs, std::string attr_name) {
  auto attr = attrs.get<T>(attr_name);
  if (attr.has_error()) {
    MORI_OPS_ERROR("Failure in getting attribute value of '{}'", attr_name);
    return attr.error();
  }
  return attr.value();
}

void GpuCopy(void* dst, const void* src, size_t bytes, hipStream_t stream, 
        hipMemcpyKind copy_dir = hipMemcpyDeviceToDevice) {
  HIP_RUNTIME_CHECK(hipMemcpyAsync(dst, src, bytes, copy_dir, stream));
}

Error MoriDispatchImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    int32_t has_scales,
    int32_t has_weights,
    int32_t kernel_type,
    int32_t block_num,
    int32_t warp_per_block,
    AnyBuffer input,
    BufferR2<F32> weights,
    AnyBuffer scales,
    BufferR2<S32> topk_ids,
    Result<AnyBuffer> out,
    Result<BufferR2<F32>> out_weights,
    Result<AnyBuffer> out_scales,
    Result<BufferR2<S32>> out_indices,
    Result<BufferR0<S32>> total_recv_token_num) {
  
  // XPUT("MoriDispatchImpl handle: %d, kernel_type=%d input=%d weights=%d block_num=%d stream: %p",
  //     h->config.rank, kernel_type, (int)input.size_bytes(), (int)weights.size_bytes(), block_num, stream);
  
  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t) &&
         ByteWidth(out_indices->element_type()) == sizeof(index_t)); 
  
  float *weightsPtr = has_weights ? weights.typed_data() : nullptr;

  uint8_t* scalesPtr = nullptr;
  if (has_scales && h->config.scaleDim > 0) {
    assert(/*scales->is_contiguous() &&*/ 
      ByteWidth(scales.element_type()) == h->config.scaleTypeSize);
    scalesPtr = static_cast< uint8_t *>(scales.untyped_data());
  }

  // NOTE: why output is set to NULL??
  h->PrepareInference(FFIType2HipType(input.element_type()), 
        input.untyped_data(), nullptr, weightsPtr, scalesPtr, 
        topk_ids.typed_data(), input.dimensions()[0]);
  h->LaunchDispatch(static_cast< KernelType >(kernel_type), block_num, 
                                                      warp_per_block, stream);

  GpuCopy(out->untyped_data(), h->shmemDispatchOutTokMemObj->Get(), 
        out->size_bytes(), stream);

  if (weightsPtr) {
    GpuCopy(out_weights->untyped_data(), h->shmemDispatchOutWeightsMemObj->Get(), 
        out_weights->size_bytes(), stream);
  }
  if (scalesPtr) {
    GpuCopy(out_scales->untyped_data(), h->shmemOutScalesMemObj->Get(), 
        out_scales->size_bytes(), stream);
  }

  GpuCopy(out_indices->untyped_data(), h->shmemOutIndicesMemObj->Get(), 
        out_indices->size_bytes(), stream);

  GpuCopy(total_recv_token_num->untyped_data(), h->totalRecvTokenNum, 
        sizeof(index_t), stream);

  return Error::Success();
}

// if this does not work, we will have to send Handle fields via Ctx params..
XLA_FFI_DEFINE_HANDLER(
    MoriDispatchHandler, MoriDispatchImpl,
    // Explicit binding to ensure attrs/args order
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
        .Attr<int32_t>("has_scales")
        .Attr<int32_t>("has_weights")
        .Attr<int32_t>("kernel_type")
        .Attr<int32_t>("block_num")
        .Attr<int32_t>("warp_per_block")
        // .Attrs()
        //.Ctx<UserData<EpDispatchCombineHandle>>()
        .Arg<AnyBuffer>()          // input
        .Arg<BufferR2<F32>>()      // weights optional
        .Arg<AnyBuffer>()          // scales optional
        .Arg<BufferR2<S32>>()      // topk_ids 
        .Ret<AnyBuffer>()          // out
        .Ret<BufferR2<F32>>()      // out_weights optional
        .Ret<AnyBuffer>()          // out_scales optional
        .Ret<BufferR2<S32>>()      // out_indices 
        .Ret<BufferR0<S32>>()      // total_recv_token_num
);

Error MoriCombineImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    int32_t has_weights,
    int32_t kernel_type,
    int32_t block_num,
    int32_t warp_per_block,
    AnyBuffer input,
    BufferR2<F32> weights,
    BufferR2<S32> topk_ids,
    Result<AnyBuffer> out,
    Result<BufferR2<F32>> out_weights) {
  
  XPUT("MoriCombineImpl handle: %d, kernel_type=%d input=%d weights=%d block_num=%d",
      h->config.rank, kernel_type, (int)input.size_bytes(), (int)weights.size_bytes(), block_num);
  
  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t)); 
  
  float *weightsPtr = has_weights ? weights.typed_data() : nullptr;

  // NOTE reading directly from GPU mem!!
  index_t total_recv_token_num = h->totalRecvTokenNum[0];

  // we need to copy data to shmemCombineInpTokMemObj directly
  if (!h->config.useExternalInpBuffer) {
    GpuCopy(h->shmemCombineInpTokMemObj->Get(), input.untyped_data(), 
        out_weights->size_bytes(), stream);
  }
  // NOTE: why output is set to NULL??
  h->PrepareInference(FFIType2HipType(input.element_type()), 
        input.untyped_data(), nullptr, weightsPtr, 
        topk_ids.typed_data(), h->curRankNumToken);
  h->LaunchCombine(static_cast< KernelType >(kernel_type), block_num, 
                                                      warp_per_block, stream);

  GpuCopy(out->untyped_data(), h->shmemCombineOutTokMemObj->Get(), 
        out->size_bytes(), stream);
  // {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim},

  if (weightsPtr) {
    //{handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
    GpuCopy(out_weights->untyped_data(), h->shmemCombineOutWeightsMemObj->Get(), 
        out_weights->size_bytes(), stream);
  }
  return Error::Success();
} 

// if this does not work, we will have to send Handle fields via Ctx params..
XLA_FFI_DEFINE_HANDLER(
    MoriCombineHandler, MoriCombineImpl,
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
        .Attr<int32_t>("has_weights")
        .Attr<int32_t>("kernel_type")
        .Attr<int32_t>("block_num")
        .Attr<int32_t>("warp_per_block")
        .Arg<AnyBuffer>()          // input
        .Arg<BufferR2<F32>>()      // weights optional
        .Arg<BufferR2<S32>>()      // topk_ids 
        .Ret<AnyBuffer>()          // out
        .Ret<BufferR2<F32>>()      // out_weights optional
);

Error MoriResetImpl(hipStream_t stream, EpDispatchCombineHandle *h) {
  h->LaunchReset(stream);
  return Error::Success();
}
 
XLA_FFI_DEFINE_HANDLER(
    MoriResetHandler, MoriResetImpl,
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
);

Error GetDispatchSrcTokenIdJax(hipStream_t stream, EpDispatchCombineHandle *h,
    BufferR0<S32> total_recv_token_num,
    Result<BufferR1<S32>> out) {
  //XPUT("GetDispatchSrcTokenIdJax stream: %p", stream);
  // NOTE here we read the whole buffer but the actual # of tokens received could be less
  // we do nto want to read it since it requires explitic stream syncrhonize otherwise
  GpuCopy(out->untyped_data(), h->dispTokIdToSrcTokIdMemObj->Get(), 
              out->size_bytes(), stream);
  return Error::Success();
} 

// if this does not work, we will have to send Handle fields via Ctx params..
XLA_FFI_DEFINE_HANDLER(
    GetDispatchSrcTokenIdHandler, GetDispatchSrcTokenIdJax,
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
        // this buffer is actually not used by we need it in order to ensure
        // correct order of FFI calls
        .Arg<BufferR0<S32>>()
        .Ret<BufferR1<S32>>()
);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */

#ifdef MORI_ENABLE_TORCH
std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, torch::Tensor,
           torch::Tensor>
LaunchDispatch(EpDispatchCombineHandle& handle, int kernelType,
               const torch::Tensor& input, const std::optional<torch::Tensor>& weights,
               const std::optional<torch::Tensor>& scales, const torch::Tensor& topkIds,
               int blockNum = -1, int warpPerBlock = -1) {
  assert(input.is_contiguous() && topkIds.is_contiguous());

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    assert(weights->is_contiguous() && weights->element_size() == sizeof(float));
    weightPtr = weights->data_ptr<float>();
  }

  uint8_t* scalePtr = nullptr;
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    assert(scales->is_contiguous() && scales->element_size() == handle.config.scaleTypeSize);
    scalePtr = reinterpret_cast<uint8_t*>(scales->data_ptr());
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightPtr, scalePtr, topkIds.data_ptr<index_t>(),
                          input.size(0));
  handle.LaunchDispatch((KernelType)kernelType, blockNum, warpPerBlock,
                        at::cuda::getCurrentHIPStream());

  torch::Tensor out =
      torch::from_blob(handle.shmemDispatchOutTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                       torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA));

  torch::Tensor outWeights = torch::from_blob(
      handle.shmemDispatchOutWeightsMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
      torch::TensorOptions().dtype(mori::GetTorchDataType<float>()).device(torch::kCUDA));

  std::optional<torch::Tensor> outScales{std::nullopt};
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    outScales =
        torch::from_blob(handle.shmemOutScalesMemObj->Get(),
                         {handle.config.MaxNumTokensToRecv(), handle.config.scaleDim},
                         torch::TensorOptions().dtype(scales->scalar_type()).device(torch::kCUDA));
  }

  torch::Tensor outIndices =
      torch::from_blob(handle.shmemOutIndicesMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
                       torch::TensorOptions()
                           .dtype(mori::GetTorchDataType<index_t>())
                           .device(torch::kCUDA));

  torch::Tensor totalRecvTokenNum =
      torch::from_blob(handle.totalRecvTokenNum, {1},
                       torch::TensorOptions()
                           .dtype(mori::GetTorchDataType<index_t>())
                           .device(torch::kCUDA));
  return {out, outWeights, outScales, outIndices, totalRecvTokenNum};
}

// TODO: translate data type
// template <typename T>
std::tuple<torch::Tensor, std::optional<torch::Tensor>> LaunchCombine(
    EpDispatchCombineHandle& handle, int kernelType, const torch::Tensor& input,
    const std::optional<torch::Tensor>& weights, const torch::Tensor& topkIds, int blockNum,
    int warpPerBlock) {
  assert(input.is_contiguous() && topkIds.is_contiguous());

  float* weightsPtr = nullptr;
  if (weights.has_value() && weights->size(0) != 0) {
    assert(weights->is_contiguous());
    weightsPtr = weights->data_ptr<float>();
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightsPtr, topkIds.data_ptr<index_t>(),
                          handle.curRankNumToken);
  handle.LaunchCombine((KernelType)kernelType, blockNum, warpPerBlock,
                       at::cuda::getCurrentHIPStream());

  auto options = torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  if (weightsPtr) {
    outWeights =
        torch::from_blob(handle.shmemCombineOutWeightsMemObj->Get(),
                         {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
                         torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  }

  return {out, outWeights};
}

void LaunchReset(EpDispatchCombineHandle& handle) {
  handle.LaunchReset(at::cuda::getCurrentHIPStream());
}

torch::Tensor GetDispatchSrcTokenId(EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<index_t>())
                     .device(torch::kCUDA);

  XPUT("GetDispatchSrcTokenId recv_num: %d", (int)handle.totalRecvTokenNum[0]);
  torch::Tensor tensor =
      torch::from_blob(handle.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(),
                       {*handle.totalRecvTokenNum}, options);
  return tensor;
}

torch::Tensor GetDispatchSenderTokenIdxMap(EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor = torch::from_blob(
      handle.dispSenderIdxMap, {handle.curRankNumToken * handle.config.numExpertPerToken}, options);
  return tensor;
}

torch::Tensor GetDispatchReceiverTokenIdxMap(EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispReceiverIdxMap, {*handle.localPeTokenCounter}, options);
  return tensor;
}

torch::Tensor GetRegisteredCombineInputBuffer(EpDispatchCombineHandle& handle,
                                              at::ScalarType scalarType) {
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineInpTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                       torch::TensorOptions().dtype(scalarType).device(torch::kCUDA));
  return out;
}
#endif // MORI_ENABLE_TORCH

void JaxPluginSetup(py::capsule pyc_api) {
  if (std::string_view(pyc_api.name()) != "pjrt_c_api") {
    throw std::runtime_error(
              "Argument to user_data_plugin was not a pjrt_c_api capsule.");
  }
  auto* c_api = pyc_api.get_pointer<PJRT_Api>();
  const auto* ffi_ext = pjrt::FindExtension<PJRT_FFI_Extension>(
      c_api, PJRT_Extension_Type::PJRT_Extension_Type_FFI);
  const auto* call_ext = pjrt::FindExtension<PJRT_Gpu_Custom_Call>(
            c_api, PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call);
  if (call_ext == nullptr || ffi_ext == nullptr) {
    throw std::runtime_error("PJRT FFI and/or custom call extension is not available!");
  }

  auto register_ffi = [&](std::string_view name, XLA_FFI_Handler *handler){
#if 0
    PJRT_FFI_Register_Handler_Args args {
      .struct_size = sizeof(PJRT_FFI_Register_Handler_Args),
      .target_name = name.data(),
      .target_name_size = name.length(),
      .handler = reinterpret_cast<void*>(handler),
      .platform_name = "ROCM",
      .platform_name_size = 4,
      .traits = PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE,
    };
    auto *err = std::invoke(ffi_ext->register_handler, &args);
#else
    PJRT_Gpu_Register_Custom_Call_Args args {
      .struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE,
      .function_name = name.data(),
      .function_name_size = name.length(),
      .api_version = 1,
      .handler_instantiate = nullptr,
      .handler_prepare = nullptr,
      .handler_initialize = nullptr,
      .handler_execute = reinterpret_cast<void*>(handler),
    };
    auto *err = std::invoke(call_ext->custom_call, &args);
#endif
    if (err != nullptr) {
      throw std::runtime_error("Unable to register FFI handler for " + 
                std::string(name));
    }
  }; // register_ffi

  register_ffi("launch_dispatch", MoriDispatchHandler);
  register_ffi("launch_combine", MoriCombineHandler);
  register_ffi("launch_reset", MoriResetHandler);
  register_ffi("get_dispatch_src_token_id", GetDispatchSrcTokenIdHandler);
}

void DeclareEpDispatchCombineHandle(pybind11::module& m) {
  std::string className = std::string("EpDispatchCombineHandle");
  pybind11::class_<EpDispatchCombineHandle>(m, className.c_str())
      .def(pybind11::init<EpDispatchCombineConfig>(),
           py::arg("config") = EpDispatchCombineConfig{})
      .def("ptr", [](EpDispatchCombineHandle *p) 
            { return reinterpret_cast<uintptr_t>(p); });

  m.def("pjrt_plugin_setup", &JaxPluginSetup, py::arg("c_api"));

  m.def("get_cur_rank_num_token", &EpDispatchCombineHandle::GetCurRankNumToken);

#ifdef MORI_ENABLE_TORCH
  std::string funcName;
  m.def("launch_dispatch", &LaunchDispatch);
  m.def("launch_combine", &LaunchCombine);
  m.def("launch_reset", &LaunchReset);
  m.def("get_dispatch_src_token_pos", &GetDispatchSrcTokenId);
  m.def("get_dispatch_sender_token_idx_map", &GetDispatchSenderTokenIdxMap);
  m.def("get_dispatch_receiver_token_idx_map", &GetDispatchReceiverTokenIdxMap);
  m.def("get_registered_combine_input_buffer", &GetRegisteredCombineInputBuffer);
#endif // MORI_ENABLE_TORCH
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                           Shmem APIs                                           */
/* ---------------------------------------------------------------------------------------------- */
namespace {

#ifdef MORI_ENABLE_TORCH
int64_t ShmemTorchProcessGroupInit(const std::string& groupName) {
  return mori::shmem::ShmemTorchProcessGroupInit(groupName);
}
#endif

int64_t ShmemFinalize() { return mori::shmem::ShmemFinalize(); }

int64_t ShmemModuleInit(uint64_t hipModule) {
  return mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(hipModule));
}

int64_t ShmemMyPe() { return mori::shmem::ShmemMyPe(); }

int64_t ShmemNPes() { return mori::shmem::ShmemNPes(); }

// UniqueId-based initialization APIs
py::bytes ShmemGetUniqueId() {
  mori::shmem::mori_shmem_uniqueid_t uid;
  mori::shmem::ShmemGetUniqueId(&uid);
  return py::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
}

int64_t ShmemInitAttr(unsigned int flags, int32_t rank, int32_t nranks,
                      const py::bytes& uid_bytes) {
  mori::shmem::mori_shmem_init_attr_t attr;
  mori::shmem::mori_shmem_uniqueid_t uid;

  // Convert Python bytes to uniqueid
  Py_ssize_t len = PyBytes_Size(uid_bytes.ptr());
  const char* data = PyBytes_AsString(uid_bytes.ptr());
  if (len != MORI_SHMEM_UNIQUE_ID_BYTES) {
    throw std::runtime_error("Invalid unique ID size");
  }
  std::memcpy(uid.data(), data, MORI_SHMEM_UNIQUE_ID_BYTES);

  // Set attributes
  mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &attr);

  return mori::shmem::ShmemInitAttr(flags, &attr);
}

void ShmemBarrierAll() { mori::shmem::ShmemBarrierAll(); }

// Symmetric memory APIs
uintptr_t ShmemMalloc(size_t size) {
  void* ptr = mori::shmem::ShmemMalloc(size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemMallocAlign(size_t alignment, size_t size) {
  void* ptr = mori::shmem::ShmemMallocAlign(alignment, size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = mori::shmem::ShmemExtMallocWithFlags(size, flags);
  return reinterpret_cast<uintptr_t>(ptr);
}

void ShmemFree(uintptr_t ptr) { mori::shmem::ShmemFree(reinterpret_cast<void*>(ptr)); }

int64_t ShmemBufferRegister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferRegister(reinterpret_cast<void*>(ptr), size);
}

int64_t ShmemBufferDeregister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferDeregister(reinterpret_cast<void*>(ptr), size);
}

// P2P address translation
uint64_t ShmemPtrP2p(uint64_t destPtr, int myPe, int destPe) {
  return mori::shmem::ShmemPtrP2p(destPtr, myPe, destPe);
}

int64_t ShmemNumQpPerPe() { return mori::shmem::ShmemNumQpPerPe(); }

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                             IO APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {}

namespace mori {

void RegisterMoriOps(py::module_& m) {
  pybind11::enum_<KernelType>(m, "EpDispatchCombineKernelType")
      .value("IntraNode", KernelType::IntraNode)
      .value("InterNode", KernelType::InterNode)
      .value("InterNodeV1", KernelType::InterNodeV1)
      .value("InterNodeV1LL", KernelType::InterNodeV1LL)
      .export_values();

#define OO(X) def(#X, &EpDispatchCombineConfig::X)
  pybind11::class_<EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool,
                          moe::KernelType, int, int, int>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true,
           py::arg("kernel_type") = moe::KernelType::IntraNode, py::arg("gpu_per_node") = 8,
           py::arg("rdma_block_num") = 0, py::arg("num_qp_per_pe") = 1)
      .def_readwrite("rank", &EpDispatchCombineConfig::rank)
      .def_readwrite("world_size", &EpDispatchCombineConfig::worldSize)
      .def_readwrite("hidden_dim", &EpDispatchCombineConfig::hiddenDim)
      .def_readwrite("scale_dim", &EpDispatchCombineConfig::scaleDim)
      .def_readwrite("scale_type_size", &EpDispatchCombineConfig::scaleTypeSize)
      .def_readwrite("max_token_type_size", &EpDispatchCombineConfig::maxTokenTypeSize)
      .def_readwrite("max_num_inp_token_per_rank",
                     &EpDispatchCombineConfig::maxNumInpTokenPerRank)
      .def_readwrite("num_experts_per_rank", &EpDispatchCombineConfig::numExpertPerRank)
      .def_readwrite("num_experts_per_token",
                     &EpDispatchCombineConfig::numExpertPerToken)
      .def_readwrite("warp_num_per_block", &EpDispatchCombineConfig::warpNumPerBlock)
      .def_readwrite("block_num", &EpDispatchCombineConfig::blockNum)
      .def_readwrite("kernel_type", &EpDispatchCombineConfig::kernelType)
      .def_readwrite("gpu_per_node", &EpDispatchCombineConfig::gpuPerNode)
      .def_readwrite("rdma_block_num", &EpDispatchCombineConfig::rdmaBlockNum)
      .def_readwrite("num_qp_per_pe", &EpDispatchCombineConfig::numQpPerPe)
      .OO(MaxNumTokensToSendPerRank)
      .OO(MaxNumTokensToSend)
      .OO(MaxNumTokensToRecvPerRank)
      .OO(MaxNumTokensToRecv);
#undef OO
  DeclareEpDispatchCombineHandle(m);
}

void RegisterMoriShmem(py::module_& m) {
  // Initialization flags
  m.attr("MORI_SHMEM_INIT_WITH_MPI_COMM") = mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM;
  m.attr("MORI_SHMEM_INIT_WITH_UNIQUEID") = mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;

#ifdef MORI_ENABLE_TORCH
  // Traditional initialization APIs
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit, py::arg("group_name"),
        "Initialize shmem from PyTorch process group");
#endif 

  // UniqueId-based initialization APIs (nvshmem/rocshmem compatible)
  m.def("shmem_get_unique_id", &ShmemGetUniqueId,
        "Get a unique ID for shmem initialization (returns bytes)");

  m.def("shmem_init_attr", &ShmemInitAttr, py::arg("flags"), py::arg("rank"), py::arg("nranks"),
        py::arg("unique_id"),
        "Initialize shmem with attributes (unique_id should be bytes from shmem_get_unique_id)");

  m.def("shmem_finalize", &ShmemFinalize, "Finalize shmem");

  //  Module-specific initialization (for Triton kernels)
  m.def("shmem_module_init", &ShmemModuleInit, py::arg("hip_module"),
        "Initialize globalGpuStates in a specific HIP module (for Triton kernels)");

  // Query APIs
  m.def("shmem_mype", &ShmemMyPe, "Get my PE (process element) ID");

  m.def("shmem_npes", &ShmemNPes, "Get number of PEs");

  // Collective operations
  m.def("shmem_barrier_all", &ShmemBarrierAll, "Global barrier synchronization");

  // Symmetric memory management
  m.def("shmem_malloc", &ShmemMalloc, py::arg("size"),
        "Allocate symmetric memory (returns address as int)");

  m.def("shmem_malloc_align", &ShmemMallocAlign, py::arg("alignment"), py::arg("size"),
        "Allocate aligned symmetric memory (returns address as int)");

  m.def("shmem_ext_malloc_with_flags", &ShmemExtMallocWithFlags, py::arg("size"), py::arg("flags"),
        "Allocate symmetric memory with flags (returns address as int)");

  m.def("shmem_free", &ShmemFree, py::arg("ptr"),
        "Free symmetric memory (ptr should be int address)");

  // Buffer registration
  m.def("shmem_buffer_register", &ShmemBufferRegister, py::arg("ptr"), py::arg("size"),
        "Register an existing buffer for RDMA (ptr should be int address)");

  m.def("shmem_buffer_deregister", &ShmemBufferDeregister, py::arg("ptr"), py::arg("size"),
        "Deregister a buffer from RDMA (ptr should be int address)");

  // P2P address translation
  m.def("shmem_ptr_p2p", &ShmemPtrP2p, py::arg("dest_ptr"), py::arg("my_pe"), py::arg("dest_pe"),
        "Convert local symmetric memory pointer to remote P2P address. "
        "Returns 0 if connection uses RDMA or if pointer is invalid. "
        "Returns P2P accessible address if connection uses P2P transport.");
#ifdef MORI_ENABLE_TORCH
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
#endif
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
  m.def("shmem_num_qp_per_pe", &ShmemNumQpPerPe);
}

void RegisterMoriIo(pybind11::module_& m) {
  m.def("set_log_level", &mori::io::SetLogLevel);

  py::enum_<mori::io::BackendType>(m, "BackendType")
      .value("Unknown", mori::io::BackendType::Unknown)
      .value("XGMI", mori::io::BackendType::XGMI)
      .value("RDMA", mori::io::BackendType::RDMA)
      .value("TCP", mori::io::BackendType::TCP)
      .export_values();

  py::enum_<mori::io::MemoryLocationType>(m, "MemoryLocationType")
      .value("Unknown", mori::io::MemoryLocationType::Unknown)
      .value("CPU", mori::io::MemoryLocationType::CPU)
      .value("GPU", mori::io::MemoryLocationType::GPU)
      .export_values();

  py::enum_<mori::io::StatusCode>(m, "StatusCode")
      .value("SUCCESS", mori::io::StatusCode::SUCCESS)
      .value("INIT", mori::io::StatusCode::INIT)
      .value("IN_PROGRESS", mori::io::StatusCode::IN_PROGRESS)
      .value("ERR_INVALID_ARGS", mori::io::StatusCode::ERR_INVALID_ARGS)
      .value("ERR_NOT_FOUND", mori::io::StatusCode::ERR_NOT_FOUND)
      .value("ERR_RDMA_OP", mori::io::StatusCode::ERR_RDMA_OP)
      .value("ERR_BAD_STATE", mori::io::StatusCode::ERR_BAD_STATE)
      .export_values();

  py::enum_<mori::io::PollCqMode>(m, "PollCqMode")
      .value("POLLING", mori::io::PollCqMode::POLLING)
      .value("EVENT", mori::io::PollCqMode::EVENT);

  py::class_<mori::io::BackendConfig>(m, "BackendConfig");

  py::class_<mori::io::RdmaBackendConfig, mori::io::BackendConfig>(m, "RdmaBackendConfig")
      .def(py::init<int, int, int, mori::io::PollCqMode, bool>(), py::arg("qp_per_transfer") = 1,
           py::arg("post_batch_size") = -1, py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING,
           py::arg("enable_notification") = true)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode)
      .def_readwrite("enable_notification", &mori::io::RdmaBackendConfig::enableNotification);

  py::class_<mori::io::IOEngineConfig>(m, "IOEngineConfig")
      .def(py::init<std::string, uint16_t>(), py::arg("host") = "", py::arg("port") = 0)
      .def_readwrite("host", &mori::io::IOEngineConfig::host)
      .def_readwrite("port", &mori::io::IOEngineConfig::port);

  py::class_<mori::io::TransferStatus>(m, "TransferStatus")
      .def(py::init<>())
      .def("Code", &mori::io::TransferStatus::Code)
      .def("Message", &mori::io::TransferStatus::Message)
      .def("Update", &mori::io::TransferStatus::Update)
      .def("Init", &mori::io::TransferStatus::Init)
      .def("InProgress", &mori::io::TransferStatus::InProgress)
      .def("Succeeded", &mori::io::TransferStatus::Succeeded)
      .def("Failed", &mori::io::TransferStatus::Failed)
      .def("SetCode", &mori::io::TransferStatus::SetCode)
      .def("SetMessage", &mori::io::TransferStatus::SetMessage)
      .def("Wait", &mori::io ::TransferStatus::Wait);

  py::class_<mori::io::EngineDesc>(m, "EngineDesc")
      .def_readonly("key", &mori::io::EngineDesc::key)
      .def_readonly("hostname", &mori::io::EngineDesc::hostname)
      .def_readonly("host", &mori::io::EngineDesc::host)
      .def_readonly("port", &mori::io::EngineDesc::port)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::EngineDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::EngineDesc>();
      });

  py::class_<mori::io::MemoryDesc>(m, "MemoryDesc")
      .def(py::init<>())
      .def_readonly("engine_key", &mori::io::MemoryDesc::engineKey)
      .def_readonly("id", &mori::io::MemoryDesc::id)
      .def_readonly("device_id", &mori::io::MemoryDesc::deviceId)
      .def_property_readonly("data",
                             [](const mori::io::MemoryDesc& desc) -> uintptr_t {
                               return reinterpret_cast<uintptr_t>(desc.data);
                             })
      .def_readonly("size", &mori::io::MemoryDesc::size)
      .def_readonly("loc", &mori::io::MemoryDesc::loc)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::MemoryDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::MemoryDesc>();
      });

  py::class_<mori::io::IOEngineSession>(m, "IOEngineSession")
      .def("AllocateTransferUniqueId", &mori::io ::IOEngineSession::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngineSession::Read)
      .def("BatchRead", &mori::io ::IOEngineSession::BatchRead)
      .def("Write", &mori::io ::IOEngineSession::Write)
      .def("BatchWrite", &mori::io ::IOEngineSession::BatchWrite)
      .def("Alive", &mori::io ::IOEngineSession::Alive);

  py::class_<mori::io::IOEngine>(m, "IOEngine")
      .def(py::init<const mori::io::EngineKey&, const mori::io::IOEngineConfig&>())
      .def("GetEngineDesc", &mori::io ::IOEngine::GetEngineDesc)
      .def("CreateBackend", &mori::io::IOEngine::CreateBackend)
      .def("RemoveBackend", &mori::io ::IOEngine::RemoveBackend)
      .def("RegisterRemoteEngine", &mori::io ::IOEngine::RegisterRemoteEngine)
      .def("DeregisterRemoteEngine", &mori::io ::IOEngine::DeregisterRemoteEngine)
      .def("RegisterMemory", &mori::io ::IOEngine::RegisterMemory)
      .def("DeregisterMemory", &mori::io ::IOEngine::DeregisterMemory)
      .def("AllocateTransferUniqueId", &mori::io ::IOEngine::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngine::Read)
      .def("BatchRead", &mori::io ::IOEngine::BatchRead)
      .def("Write", &mori::io ::IOEngine::Write)
      .def("BatchWrite", &mori::io ::IOEngine::BatchWrite)
      .def("CreateSession", &mori::io::IOEngine::CreateSession)
      .def("PopInboundTransferStatus", &mori::io::IOEngine::PopInboundTransferStatus);
}

}  // namespace mori

#endif // __HIP_DEVICE_COMPILE__