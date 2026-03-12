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

namespace py = pybind11;

#define XPUT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define unlikely(x) __builtin_expect(!!(x), 0)

using namespace xla::ffi;
using mori::moe::EpDispatchCombineConfig;
using mori::moe::EpDispatchCombineHandle;
using mori::moe::KernelType;
using mori::moe::index_t;

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

void GpuCopy(void* dst, const void* src, size_t bytes, hipStream_t stream, 
        hipMemcpyKind copy_dir = hipMemcpyDeviceToDevice) {
  HIP_RUNTIME_CHECK(hipMemcpyAsync(dst, src, bytes, copy_dir, stream));
}

template <class T, class Container>
T GetArg(const Container& container, size_t index) {
  auto result = container.template get<T>(index);
  if (unlikely(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

template <class T, class Container>
Result<T> GetRet(const Container& container, size_t index) {
  auto result = container.template get<T>(index);
  if (unlikely(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

template <class T, class Container>
T GetAttr(const Container& container, std::string_view name) {
  auto result = container.template get<T>(name);
  if (unlikely(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

Error MoriDispatchImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    Dictionary attrs, RemainingArgs args, RemainingRets rets
    /*int32_t has_scales,
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
    Result<BufferR0<S32>> total_recv_token_num*/) {
  auto input = GetArg<AnyBuffer>(args, 0);
  auto weights = GetArg<BufferR2<F32>>(args, 1);
  auto topk_ids = GetArg<BufferR2<S32>>(args, 2);
  auto out = GetRet<AnyBuffer>(rets, 0);
  
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



// #if 0 // NOTE: we do not use this anymore   
//   h->LaunchDispatch(static_cast< KernelType >(kernel_type), block_num, 
//                                                       warp_per_block, stream);
// #endif
//   GpuCopy(out->untyped_data(), h->shmemDispatchOutTokMemObj->Get(), 
//         out->size_bytes(), stream);

//   if (weightsPtr) {
//     GpuCopy(out_weights->untyped_data(), h->shmemDispatchOutWeightsMemObj->Get(), 
//         out_weights->size_bytes(), stream);
//   }
//   if (scalesPtr) {
//     GpuCopy(out_scales->untyped_data(), h->shmemOutScalesMemObj->Get(), 
//         out_scales->size_bytes(), stream);
//   }

//   GpuCopy(out_indices->untyped_data(), h->shmemOutIndicesMemObj->Get(), 
//         out_indices->size_bytes(), stream);

//   GpuCopy(total_recv_token_num->untyped_data(), h->totalRecvTokenNum, 
//         sizeof(index_t), stream);

  return Error::Success();
}

// if this does not work, we will have to send Handle fields via Ctx params..
// XLA_FFI_DEFINE_HANDLER(
//     MoriDispatchHandler, MoriDispatchImpl,
//     // Explicit binding to ensure attrs/args order
//     Ffi::Bind()
//         .Ctx<PlatformStream<hipStream_t>>()
//         .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
//         .Attr<int32_t>("has_scales")
//         .Attr<int32_t>("has_weights")
//         .Attr<int32_t>("kernel_type")
//         .Attr<int32_t>("block_num")
//         .Attr<int32_t>("warp_per_block")
//         .Arg<AnyBuffer>()          // input
//         .Arg<BufferR2<F32>>()      // weights optional
//         .Arg<AnyBuffer>()          // scales optional
//         .Arg<BufferR2<S32>>()      // topk_ids 
//         .Ret<AnyBuffer>()          // out
//         .Ret<BufferR2<F32>>()      // out_weights optional
//         .Ret<AnyBuffer>()          // out_scales optional
//         .Ret<BufferR2<S32>>()      // out_indices 
//         .Ret<BufferR0<S32>>()      // total_recv_token_num
// );

Error MoriCombineImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    Dictionary attrs, RemainingArgs args, RemainingRets rets
    /*int32_t has_weights,
    int32_t kernel_type,
    int32_t block_num,
    int32_t warp_per_block,
    AnyBuffer input,
    BufferR2<F32> weights,
    BufferR2<S32> topk_ids,
    Result<AnyBuffer> out,
    Result<BufferR2<F32>> out_weights*/) {

      auto input = GetArg<AnyBuffer>(args, 0);
      auto weights = GetArg<BufferR2<F32>>(args, 1);
      auto topk_ids = GetArg<BufferR2<S32>>(args, 2);
      auto out = GetRet<AnyBuffer>(rets, 0);
  
//   XPUT("MoriCombineImpl handle: %d, kernel_type=%d input=%d weights=%d block_num=%d",
//       h->config.rank, kernel_type, (int)input.size_bytes(), (int)weights.size_bytes(), block_num);
  
//   assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t)); 
  
//   float *weightsPtr = has_weights ? weights.typed_data() : nullptr;

//   // NOTE reading directly from GPU mem!!
//   index_t total_recv_token_num = h->totalRecvTokenNum[0];

//   // we need to copy data to shmemCombineInpTokMemObj directly
//   if (!h->config.useExternalInpBuffer) {
//     GpuCopy(h->shmemCombineInpTokMemObj->Get(), input.untyped_data(), 
//         out_weights->size_bytes(), stream);
//   }
//   // NOTE: why output is set to NULL??
//   h->PrepareInference(FFIType2HipType(input.element_type()), 
//         input.untyped_data(), nullptr, weightsPtr, 
//         topk_ids.typed_data(), h->curRankNumToken);
// #if 0 // NOTE: we do not use this anymore   
//   h->LaunchCombine(static_cast< KernelType >(kernel_type), block_num, 
//                                                       warp_per_block, stream);
// #endif
//   GpuCopy(out->untyped_data(), h->shmemCombineOutTokMemObj->Get(), 
//         out->size_bytes(), stream);
//   // {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim},

//   if (weightsPtr) {
//     //{handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
//     GpuCopy(out_weights->untyped_data(), h->shmemCombineOutWeightsMemObj->Get(), 
//         out_weights->size_bytes(), stream);
//   }
  return Error::Success();
} 

// if this does not work, we will have to send Handle fields via Ctx params..
// XLA_FFI_DEFINE_HANDLER(
//     MoriCombineHandler, MoriCombineImpl,
//     Ffi::Bind()
//         .Ctx<PlatformStream<hipStream_t>>()
//         .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
//         .Attr<int32_t>("has_weights")
//         .Attr<int32_t>("kernel_type")
//         .Attr<int32_t>("block_num")
//         .Attr<int32_t>("warp_per_block")
//         .Arg<AnyBuffer>()          // input
//         .Arg<BufferR2<F32>>()      // weights optional
//         .Arg<BufferR2<S32>>()      // topk_ids 
//         .Ret<AnyBuffer>()          // out
//         .Ret<BufferR2<F32>>()      // out_weights optional
// );

Error MoriResetImpl(hipStream_t stream, EpDispatchCombineHandle *h) {
  XPUT("MoriResetImpl stream: %p", stream);
  h->LaunchReset(stream);
  return Error::Success();
}
 
// XLA_FFI_DEFINE_HANDLER(
//     MoriResetHandler, MoriResetImpl,
//     Ffi::Bind()
//         .Ctx<PlatformStream<hipStream_t>>()
//         .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
// );

Error GetDispatchSrcTokenId(hipStream_t stream, EpDispatchCombineHandle *h,
    RemainingRets rets) {
  //XPUT("GetDispatchSrcTokenId stream: %p", stream);
  
  auto out = rets.get<BufferR1<S32>>(0);
  if (out.has_error()) return out.error();
  
  // NOTE here we read the whole buffer but the actual # of tokens received could be less
  // we do nto want to read it since it requires explitic stream syncrhonize otherwise
  GpuCopy(out.value()->untyped_data(), h->dispTokIdToSrcTokIdMemObj->Get(), 
              out.value()->size_bytes(), stream);
  return Error::Success();
} 

// // if this does not work, we will have to send Handle fields via Ctx params..
// XLA_FFI_DEFINE_HANDLER(
//     GetDispatchSrcTokenIdHandler, GetDispatchSrcTokenIdJax,
//     Ffi::Bind()
//         .Ctx<PlatformStream<hipStream_t>>()
//         .Attr<Pointer<EpDispatchCombineHandle>>("handle_ptr")
//         // this buffer is actually not used by we need it in order to ensure
//         // correct order of FFI calls
//         .Arg<BufferR0<S32>>()
//         .Ret<BufferR1<S32>>()
// );

ErrorOr<std::unique_ptr<EpDispatchCombineState>> EpDispatchCombineInstantiate(
    Dictionary attrs) {

  auto ep_config = attrs.get<Span<const int32_t>>("ep_config");
  if (ep_config.has_error()) {
    return ErrorOr<std::unique_ptr<EpDispatchCombineState>>(ep_config.error());
  }
  auto cfg = EpDispatchCombineConfig::FromPackedI32Array(
    ep_config->begin(), ep_config->size());
  return std::make_unique<EpDispatchCombineState>(cfg);
}

XLA_FFI_DEFINE_HANDLER(
  EpDispatchCombineInstHandler, EpDispatchCombineInstantiate,
    Ffi::BindInstantiate().Attrs());

Error EpDispatchCombineImpl(
    hipStream_t stream, EpDispatchCombineState* state, 
    Dictionary attrs, 
    RemainingArgs args,
    RemainingRets rets) try {
  XPUT(
      "EpDispatchCombineImpl stream=%p rank=%d  attrs: %zu",
      stream, state->handle.config.rank, attrs.size());
  if (attrs.contains("dispatch_op")) {
    return MoriDispatchImpl(stream, &state->handle, attrs, args, rets);
  }
  if (attrs.contains("combine_op")) {
    return MoriCombineImpl(stream, &state->handle, attrs, args, rets);
  }
  if (attrs.contains("reset_op")) {
    return MoriResetImpl(stream, &state->handle);
  }
  if (attrs.contains("get_src_token_id")) {
    return GetDispatchSrcTokenId(stream, &state->handle, rets);
  }
  return Error::Internal("Invalid operation type");
} catch (const std::exception& e) {
  return Error::Internal(e.what());
}

XLA_FFI_DEFINE_HANDLER(
    EpDispatchCombineHandler, EpDispatchCombineImpl,
    Ffi::Bind()
        .Ctx<PlatformStream<hipStream_t>>()
        .Ctx<State<EpDispatchCombineState>>()
        .Attrs()
        .RemainingArgs()
        .RemainingRets());

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
  m.def("mori_ep_type_id",
        []() { return py::capsule(reinterpret_cast<void*>(&EpDispatchCombineState::id)); });
  m.def("mori_ep_handler", []() {
    py::dict d;
    d["instantiate"] =
        py::capsule(reinterpret_cast<void*>(EpDispatchCombineInstHandler));
    d["execute"] = py::capsule(reinterpret_cast<void*>(EpDispatchCombineHandler));
    return d;
  });
//   m.def("get_cur_rank_num_token", &EpDispatchCombineHandle::GetCurRankNumToken);
}

}  // namespace mori

// #endif // __HIP_DEVICE_COMPILE__