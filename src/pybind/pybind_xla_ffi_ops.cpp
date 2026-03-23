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
#include <mutex>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/ops/ops.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "src/pybind/mori.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

using namespace xla::ffi;
using mori::moe::EpDispatchCombineConfig;
using mori::moe::EpDispatchCombineHandle;
using mori::moe::KernelType;
using mori::moe::index_t;

/* ---------------------------------------------------------------------------------------------- */
/*                                          XLA Ops APIs                                          */
/* ---------------------------------------------------------------------------------------------- */
namespace {

// Global cache: maps packed ep_config → shared handle.
// All call sites with identical ep_config reuse the same EpDispatchCombineHandle.
struct VecI32Hash {
  size_t operator()(const std::vector<int32_t>& v) const noexcept {
    size_t seed = v.size();
    for (auto x : v) {
      // boost::hash_combine style mixing
      seed ^= std::hash<int32_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

static std::mutex g_handle_cache_mu;
static std::unordered_map<std::vector<int32_t>,
    std::shared_ptr<EpDispatchCombineHandle>, VecI32Hash> g_handle_cache;

struct EpDispatchCombineState {
  static TypeId id;

  explicit EpDispatchCombineState(
      std::shared_ptr<EpDispatchCombineHandle> h) : handle(std::move(h)) {}

  std::shared_ptr<EpDispatchCombineHandle> handle;
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
  if (XLA_FFI_PREDICT_FALSE(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

template <class T, class Container>
Result<T> GetRet(const Container& container, size_t index) {
  auto result = container.template get<T>(index);
  if (XLA_FFI_PREDICT_FALSE(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

template <class T, class Container>
T GetAttr(const Container& container, std::string_view name) {
  auto result = container.template get<T>(name);
  if (XLA_FFI_PREDICT_FALSE(result.has_error())) {
    throw std::runtime_error(result.error().message());
  }
  return result.value();
}

template <class T, class Container>
T GetAttrOr(const Container& container, std::string_view name, T def) {
  auto result = container.template get<T>(name);
  if (XLA_FFI_PREDICT_FALSE(result.has_error())) {
    return def;
  }
  return result.value();
}

Error MoriDispatchImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    Dictionary attrs, RemainingArgs args, RemainingRets rets) {

  auto input = GetArg<AnyBuffer>(args, 0);
  auto topk_ids = GetArg<BufferR2<S32>>(args, 1);
  auto out = GetRet<AnyBuffer>(rets, 0);
  auto out_indices = GetRet<BufferR2<S32>>(rets, 1);
  auto total_recv_token_num = GetRet<BufferR0<S32>>(rets, 2);

  auto block_num = GetAttr<int32_t>(attrs, "block_num");
  auto rdma_block_num = GetAttr<int32_t>(attrs, "rdma_block_num");
  auto warp_per_block = GetAttr<int32_t>(attrs, "warp_per_block");
  auto has_scales = GetAttr<int32_t>(attrs, "has_scales");
  
  const int hiddenDim = static_cast<int>(input.dimensions()[1]);
  assert(hiddenDim > 0 && hiddenDim <= h->config.hiddenDim);
  
  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t) &&
         ByteWidth(out_indices->element_type()) == sizeof(index_t)); 
  
  float *weightsPtr = nullptr;
  if ((!has_scales && args.size() >= 3) || 
       (has_scales && args.size() >= 4)) {
    auto weights = GetArg<BufferR2<F32>>(args, 2);
    weightsPtr = weights.typed_data();
  }
  
  uint8_t* scalesPtr = nullptr;
  if (has_scales && h->config.scaleDim > 0) {
    auto scales = GetArg<AnyBuffer>(args, weightsPtr ? 3 : 2);
    assert(/*scales->is_contiguous() &&*/ 
      ByteWidth(scales.element_type()) == h->config.scaleTypeSize);
    scalesPtr = static_cast< uint8_t *>(scales.untyped_data());
  }
  XPUT("MoriDispatch h: %p, input=%d hiddenDim=%d weights=%p scales=%p",
    h, (int)input.size_bytes(), hiddenDim, weightsPtr, scalesPtr);

  mori::moe::LaunchDispatch(*h, input.untyped_data(), weightsPtr, 
      scalesPtr, topk_ids.typed_data(), input.dimensions()[0], 
      FFIType2HipType(input.element_type()), block_num, rdma_block_num,
      warp_per_block, stream, hiddenDim);

  GpuCopy(out->untyped_data(), h->shmemDispatchOutTokMemObj->Get(), 
        out->size_bytes(), stream);

  if (weightsPtr) {
    auto out_weights = GetRet<BufferR2<F32>>(rets, 3);
    GpuCopy(out_weights->untyped_data(), h->shmemDispatchOutWeightsMemObj->Get(), 
        out_weights->size_bytes(), stream);
  }
  if (scalesPtr) {
    auto out_scales = GetRet<AnyBuffer>(rets, weightsPtr ? 4 : 3);
    GpuCopy(out_scales->untyped_data(), h->shmemOutScalesMemObj->Get(), 
        out_scales->size_bytes(), stream);
  }

  GpuCopy(out_indices->untyped_data(), h->shmemOutIndicesMemObj->Get(), 
        out_indices->size_bytes(), stream);

  GpuCopy(total_recv_token_num->untyped_data(), h->totalRecvTokenNum, 
        sizeof(index_t), stream);

  // HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  // XPUT("rank %d MoriDispatch done", h->config.rank);
  return Error::Success();
}

Error MoriCombineImpl(
    hipStream_t stream,
    EpDispatchCombineHandle *h,
    Dictionary attrs, RemainingArgs args, RemainingRets rets) {

  auto input = GetArg<AnyBuffer>(args, 0);
  auto topk_ids = GetArg<BufferR2<S32>>(args, 1);
  auto out = GetRet<AnyBuffer>(rets, 0);

  auto block_num = GetAttr<int32_t>(attrs, "block_num");
  auto rdma_block_num = GetAttr<int32_t>(attrs, "rdma_block_num");
  auto warp_per_block = GetAttr<int32_t>(attrs, "warp_per_block");
  const int hiddenDim = static_cast<int>(input.dimensions()[1]);
  assert(hiddenDim > 0 && hiddenDim <= h->config.hiddenDim);
  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t)); 
  
  float *weightsPtr = nullptr;
  if (args.size() > 2) {
    auto weights = GetArg<BufferR2<F32>>(args, 2);
    weightsPtr = weights.typed_data();
  }
  XPUT("MoriCombine h: %p, input=%d topk_ids=%d hiddenDim: %d weights=%p useExternalInpBuffer=%d",
    h, (int)input.size_bytes(), (int)topk_ids.size_bytes(), hiddenDim, weightsPtr,
    h->config.useExternalInpBuffer);

//   // NOTE reading directly from GPU mem!!
//   index_t total_recv_token_num = h->totalRecvTokenNum[0];
//   // we need to copy data to shmemCombineInpTokMemObj directly
  if (!h->config.useExternalInpBuffer) {
    // GpuCopy(h->shmemCombineInpTokMemObj->Get(), input.untyped_data(), 
    //     out_weights->size_bytes(), // should this be input.size_bytes()?
    //     stream);
  }
  mori::moe::LaunchCombine(*h, input.untyped_data(), weightsPtr,
      topk_ids.typed_data(), //input.dimensions()[0], 
      h->curRankNumToken,
      FFIType2HipType(input.element_type()), block_num, rdma_block_num,
      warp_per_block, h->config.useExternalInpBuffer ? 1 : 0, 
      stream, hiddenDim);

  GpuCopy(out->untyped_data(), h->shmemCombineOutTokMemObj->Get(), 
        out->size_bytes(), stream);
  // {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim},

  if (weightsPtr) {
    auto out_weights = GetRet<BufferR2<F32>>(rets, 1);
    //{handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
    GpuCopy(out_weights->untyped_data(), h->shmemCombineOutWeightsMemObj->Get(), 
        out_weights->size_bytes(), stream);
  }
  // HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  // XPUT("rank %d MoriCombine done", h->config.rank);
  return Error::Success();
} 

Error MoriResetImpl(hipStream_t stream, EpDispatchCombineHandle *h) {
  XPUT("MoriResetImpl stream: %p", stream);
  h->LaunchReset(stream);
  return Error::Success();
}

Error GetDispatchSrcTokenId(hipStream_t stream, EpDispatchCombineHandle *h,
    RemainingRets rets) {
  // XPUT("GetDispatchSrcTokenId h: %p", h);
  auto out = GetRet<BufferR1<S32>>(rets, 0);
  // NOTE here we read the whole buffer but the actual # of tokens received could be less
  // we do not want to read it since it requires explicit stream synchronization otherwise
  GpuCopy(out->untyped_data(), h->dispTokIdToSrcTokIdMemObj->Get(), 
              out->size_bytes(), stream);
  return Error::Success();
} 

ErrorOr<std::unique_ptr<EpDispatchCombineState>> EpDispatchCombineInstantiate(
    Dictionary attrs) {

  auto ep_config = attrs.get<Span<const int32_t>>("ep_config");

  if (ep_config.has_error()) {
    return ErrorOr<std::unique_ptr<EpDispatchCombineState>>(ep_config.error());
  }

  // Use the packed config as cache key so all call sites with the same
  // ep_config share a single EpDispatchCombineHandle.
  std::vector<int32_t> key(ep_config->begin(), ep_config->end());

  std::lock_guard<std::mutex> lock(g_handle_cache_mu);
  auto& entry = g_handle_cache[key];
  if (!entry) {
    auto cfg = EpDispatchCombineConfig::FromPackedI32Array(
        key.data(), key.size());
    XPUT("EpDispatchCombineInstantiate: creating new handle for rank %d "
         "(#attrs: %zu)", cfg.rank, attrs.size());
    entry = std::make_shared<EpDispatchCombineHandle>(cfg);
  } else {
    XPUT("EpDispatchCombineInstantiate: reusing cached handle for rank %d "
         "(#attrs: %zu)", entry->config.rank, attrs.size());
  }
  return std::make_unique<EpDispatchCombineState>(entry);
}

XLA_FFI_DEFINE_HANDLER(
  EpDispatchCombineInstHandler, EpDispatchCombineInstantiate,
    Ffi::BindInstantiate().Attrs());

Error EpDispatchCombineImpl(
    hipStream_t stream, EpDispatchCombineState* state, 
    Dictionary attrs, 
    RemainingArgs args,
    RemainingRets rets) try {
  auto& h = *state->handle;
  // XPUT("EpDispatchCombineImpl stream=%p rank=%d  attrs: %zu",
  //     stream, h.config.rank, attrs.size());
  if (attrs.contains("dispatch_op")) {
    return MoriDispatchImpl(stream, &h, attrs, args, rets);
  }
  if (attrs.contains("combine_op")) {
    return MoriCombineImpl(stream, &h, attrs, args, rets);
  }
  if (attrs.contains("reset_op")) {
    return MoriResetImpl(stream, &h);
  }
  if (attrs.contains("get_src_token_id")) {
    return GetDispatchSrcTokenId(stream, &h, rets);
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
  m.def("mori_ep_type_info", []() {
    // In earlier versions of XLA:FFI, the `MakeTypeInfo` helper was not
    // available. In latest XLF:FFI `TypeInfo` is an alias for C API struct.
    static auto kStateTypeInfo = 
#if XLA_FFI_API_MINOR >= 2
          MakeTypeInfo<EpDispatchCombineState>();
#else
          TypeInfo<EpDispatchCombineState>();
#endif
    py::dict d;
    d["type_id"] = py::capsule(reinterpret_cast<void*>(&EpDispatchCombineState::id));
    d["type_info"] = py::capsule(reinterpret_cast<void*>(&kStateTypeInfo));
    return d;
  });

  m.def("mori_ep_handler", []() {
    py::dict d;
    d["instantiate"] =
        py::capsule(reinterpret_cast<void*>(EpDispatchCombineInstHandler));
    d["execute"] = py::capsule(reinterpret_cast<void*>(EpDispatchCombineHandler));
    return d;
  });
  m.def("preload_kernels", []() {
    mori::moe::KernelRegistry::Instance().AutoLoad();
  });
  m.def("clear_ep_handle_cache", []() {
    std::lock_guard<std::mutex> lock(g_handle_cache_mu);
    g_handle_cache.clear();
  });
}

}  // namespace mori

// #endif // __HIP_DEVICE_COMPILE__