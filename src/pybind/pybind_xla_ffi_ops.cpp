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
// #ifndef __HIP_DEVICE_COMPILE__

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/ops/ops.hpp"
#include "mori/pybind/profiler_registry.hpp"
#include "mori/utils/hip_helper.hpp"
#include "src/pybind/mori.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
// #include "xla/pjrt/c/pjrt_c_api_helpers.h"

namespace py = pybind11;

#define XPUT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

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

struct EpDispatchCombineState {
  static TypeId id;

  explicit EpDispatchCombineState(EpDispatchCombineConfig cfg) : handle(cfg) {}

  EpDispatchCombineHandle handle;
};

TypeId EpDispatchCombineState::id = {};

hipDataType FFIType2HipType(DataType dtype) {
#define XX(A, B) case DataType::A: return B;
  switch (dtype) {
    XX(F32, HIP_R_32F)
    XX(BF16, HIP_R_16BF)
    XX(F8E4M3FN, HIP_R_8F_E4M3)
    XX(F8E4M3FNUZ, HIP_R_8F_E4M3_FNUZ)
    default:
      throw std::runtime_error("Unsupported scalar type");
  }
#undef XX
}

// template <typename T>
// T get_attr_value(Dictionary& attrs, std::string attr_name) {
//   auto attr = attrs.get<T>(attr_name);
//   if (attr.has_error()) {
//     MORI_OPS_ERROR("Failure in getting attribute value of '{}'", attr_name);
//     return attr.error();
//   }
//   return attr.value();
// }

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
  
  XPUT("MoriDispatchImpl handle: %d, kernel_type=%d input=%d weights=%d block_num=%d stream: %p",
      h->config.rank, kernel_type, (int)input.size_bytes(), (int)weights.size_bytes(), block_num, stream);
  
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

#if 0 // NOTE: we do not use this anymore   
  h->LaunchDispatch(static_cast< KernelType >(kernel_type), block_num, 
                                                      warp_per_block, stream);
#endif
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
#if 0 // NOTE: we do not use this anymore   
  h->LaunchCombine(static_cast< KernelType >(kernel_type), block_num, 
                                                      warp_per_block, stream);
#endif
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

ErrorOr<std::unique_ptr<EpDispatchCombineState>> MoriLaunchTestInstantiate(
    Span<const int32_t> ep_config) {
  auto cfg = EpDispatchCombineConfig::FromPackedI32Array(
      ep_config.begin(), ep_config.size());
  return std::make_unique<EpDispatchCombineState>(cfg);
}

XLA_FFI_DEFINE_HANDLER(
    MoriLaunchTestInstantiateHandler, MoriLaunchTestInstantiate,
    Ffi::BindInstantiate().Attr<Span<const int32_t>>("ep_config"));

Error MoriLaunchTestImpl(
    hipStream_t stream, EpDispatchCombineState* state, 
    Span<const int32_t> ep_config) {
  std::vector<int32_t> packed = state->handle.config.ToPackedI32Array();
  XPUT(
      "MoriLaunchTestImpl stream=%p state.rank=%d ep_config_i32_size=%d "
      "state_config_size=%d",
      stream, state->handle.config.rank, static_cast<int>(ep_config.size()),
      static_cast<int>(packed.size()));
  return Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    MoriLaunchTestHandler, MoriLaunchTestImpl,
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Ctx<State<EpDispatchCombineState>>()
        .Attr<Span<const int32_t>>("ep_config"));

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

  auto register_type_id = [&](std::string_view name, XLA_FFI_TypeId* type_id_ptr) {
    PJRT_FFI_TypeID_Register_Args args {
      .struct_size = sizeof(PJRT_FFI_TypeID_Register_Args),
      .type_name = name.data(),
      .type_name_size = name.length(),
      .type_id = type_id_ptr->type_id,
    };
    auto *err = std::invoke(ffi_ext->type_id_register, &args);
    if (err != nullptr) {
      throw std::runtime_error("Unable to register type ID for " + 
                std::string(name));
    }
    // Update the type id pointer if it was assigned by XLA
    type_id_ptr->type_id = args.type_id;
  };

  auto register_ffi = [&](std::string_view name, XLA_FFI_Handler* handler_execute,
                          XLA_FFI_Handler* handler_instantiate = nullptr) {
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
      .handler_instantiate = reinterpret_cast<void*>(handler_instantiate),
      .handler_prepare = nullptr,
      .handler_initialize = nullptr,
      .handler_execute = reinterpret_cast<void*>(handler_execute),
    };
    auto *err = std::invoke(call_ext->custom_call, &args);
#endif
    if (err != nullptr) {
      throw std::runtime_error("Unable to register FFI handler for " + 
                std::string(name));
    }
    // XPUT("Registered FFI handler for %s", name.data());
  }; // register_ffi

  register_ffi("launch_dispatch", MoriDispatchHandler);
  register_ffi("launch_combine", MoriCombineHandler);
  register_ffi("launch_reset", MoriResetHandler);
  register_ffi("get_dispatch_src_token_id", GetDispatchSrcTokenIdHandler);
  register_ffi("launch_test", MoriLaunchTestHandler,
               MoriLaunchTestInstantiateHandler);

  register_type_id("EpDispatchCombineState", &EpDispatchCombineState::id);
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                             IO APIs                                            */
/* ---------------------------------------------------------------------------------------------- */

namespace mori {

void RegisterXLAFFIOps(py::module_& m) {

// #define OO(X) def(#X, &EpDispatchCombineConfig::X)
//       .OO(MaxNumTokensToSendPerRank)
//       .OO(MaxNumTokensToSend)
//       .OO(MaxNumTokensToRecvPerRank)
//       .OO(MaxNumTokensToRecv);
// #undef OO
  m.def("pjrt_plugin_setup", &JaxPluginSetup, py::arg("c_api"));
//   m.def("get_cur_rank_num_token", &EpDispatchCombineHandle::GetCurRankNumToken);
}

}  // namespace mori

// #endif // __HIP_DEVICE_COMPILE__