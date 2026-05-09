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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "mori/application/utils/check.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/limits.hpp"
#include "src/pybind/mori.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

using namespace xla::ffi;
using mori::moe::EpDispatchCombineConfig;
using mori::moe::EpDispatchCombineHandle;
using mori::moe::index_t;
using mori::moe::KernelType;

/* ---------------------------------------------------------------------------------------------- */
/*                                          XLA Ops APIs                                          */
/* ---------------------------------------------------------------------------------------------- */
namespace {

// Cache: maps packed ep_config → shared handle.
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

using HandleCacheMap =
    std::unordered_map<std::vector<int32_t>, std::unique_ptr<EpDispatchCombineHandle>, VecI32Hash>;

struct HandleCacheSlot {
  std::mutex mu;
  HandleCacheMap map;
};

#ifdef MORI_MULTITHREAD_SUPPORT
// SPMT: per-GPU cache slot. Each thread is bound to its own device, so each
// gets its own (mu, map). This avoids a deadlock that would occur if a single
// process-global mutex were held while calling the cross-PE Barrier() below —
// thread A would hold the mutex during Barrier(), thread B would block on the
// mutex and never reach its own Barrier(). In multi-process mode every process
// sees its single GPU as device 0, so this collapses to slot[0].
static std::array<HandleCacheSlot, mori::kMaxGpusPerNode> g_handle_cache_slots;

static HandleCacheSlot& GetHandleCacheSlot() {
  int id = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&id));
  if (id < 0 || id >= mori::kMaxGpusPerNode) {
    throw std::runtime_error("EpHandleCache: hipGetDevice() out of range: " + std::to_string(id));
  }
  return g_handle_cache_slots[static_cast<size_t>(id)];
}

// RAII helper: temporarily switch the calling thread to `target` and restore
// the previous device on scope exit. Used in XLA FFI handlers so that
// hipSetDevice does not leak out into XLA's worker-thread state.
class ScopedDevice {
 public:
  explicit ScopedDevice(int target) : saved_(-1), restore_(false) {
    if (target < 0) return;
    if (hipGetDevice(&saved_) != hipSuccess) return;
    if (saved_ == target) return;  // nothing to do
    if (hipSetDevice(target) == hipSuccess) restore_ = true;
  }
  ~ScopedDevice() {
    if (restore_) (void)hipSetDevice(saved_);
  }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int saved_;
  bool restore_;
};
#else
static HandleCacheSlot g_handle_cache_singleton;
static HandleCacheSlot& GetHandleCacheSlot() { return g_handle_cache_singleton; }
#endif

struct EpDispatchCombineState {
  static TypeId id;

  explicit EpDispatchCombineState(EpDispatchCombineHandle* h) : handle(h) {}

  EpDispatchCombineHandle* handle = nullptr;
};

TypeId EpDispatchCombineState::id = {};

hipDataType FFIType2HipType(DataType dtype) {
#define XX(A, B)    \
  case DataType::A: \
    return B;
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

Error MoriDispatchImpl(hipStream_t stream, EpDispatchCombineHandle* h, Dictionary attrs,
                       RemainingArgs args, RemainingRets rets) {
  auto input = GetArg<AnyBuffer>(args, 0);
  auto topk_ids = GetArg<BufferR2<S32>>(args, 1);
  auto out = GetRet<AnyBuffer>(rets, 0);
  auto out_indices = GetRet<BufferR2<S32>>(rets, 1);
  auto total_recv_token_num = GetRet<BufferR0<S32>>(rets, 2);

  auto block_num = GetAttr<int32_t>(attrs, "block_num");
  auto rdma_block_num = GetAttr<int32_t>(attrs, "rdma_block_num");
  auto warp_per_block = GetAttr<int32_t>(attrs, "warp_per_block");
  auto has_scales = GetAttr<int32_t>(attrs, "has_scales");

  if (XLA_FFI_PREDICT_FALSE(input.dimensions().size() != 2)) {
    return Error::InvalidArgument("input must be a 2D tensor");
  }
  const int hiddenDim = static_cast<int>(input.dimensions()[1]);
  assert(hiddenDim > 0 && hiddenDim <= h->config.hiddenDim);

  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t) &&
         ByteWidth(out_indices->element_type()) == sizeof(index_t));

  float* weightsPtr = nullptr;
  if ((!has_scales && args.size() >= 3) || (has_scales && args.size() >= 4)) {
    auto weights = GetArg<BufferR2<F32>>(args, 2);
    weightsPtr = weights.typed_data();
  }

  uint8_t* scalesPtr = nullptr;
  if (has_scales && h->config.scaleDim > 0) {
    auto scales = GetArg<AnyBuffer>(args, weightsPtr ? 3 : 2);
    assert(/*scales->is_contiguous() &&*/
           ByteWidth(scales.element_type()) == h->config.scaleTypeSize);
    scalesPtr = static_cast<uint8_t*>(scales.untyped_data());
  }
  // XPUT("MoriDispatch h: %p, input=%d hiddenDim=%d weights=%p scales=%p",
  //   h, (int)input.size_bytes(), hiddenDim, weightsPtr, scalesPtr);

  mori::moe::LaunchDispatch(*h, input.untyped_data(), weightsPtr, scalesPtr, topk_ids.typed_data(),
                            input.dimensions()[0], FFIType2HipType(input.element_type()), block_num,
                            rdma_block_num, warp_per_block, stream, hiddenDim);

  GpuCopy(out->untyped_data(), h->GetShmemDispatchOutTokMemObj()->Get(), out->size_bytes(), stream);

  if (weightsPtr) {
    auto out_weights = GetRet<BufferR2<F32>>(rets, 3);
    GpuCopy(out_weights->untyped_data(), h->shmemDispatchOutWeightsMemObj->Get(),
            out_weights->size_bytes(), stream);
  }
  if (scalesPtr) {
    auto out_scales = GetRet<AnyBuffer>(rets, weightsPtr ? 4 : 3);
    GpuCopy(out_scales->untyped_data(), h->shmemOutScalesMemObj->Get(), out_scales->size_bytes(),
            stream);
  }

  GpuCopy(out_indices->untyped_data(), h->shmemOutIndicesMemObj->Get(), out_indices->size_bytes(),
          stream);

  GpuCopy(total_recv_token_num->untyped_data(), h->totalRecvTokenNum, sizeof(index_t), stream);
  return Error::Success();
}

Error MoriCombineImpl(hipStream_t stream, EpDispatchCombineHandle* h, Dictionary attrs,
                      RemainingArgs args, RemainingRets rets) {
  auto input = GetArg<AnyBuffer>(args, 0);
  auto topk_ids = GetArg<BufferR2<S32>>(args, 1);
  auto out = GetRet<AnyBuffer>(rets, 0);

  auto block_num = GetAttr<int32_t>(attrs, "block_num");
  auto rdma_block_num = GetAttr<int32_t>(attrs, "rdma_block_num");
  auto warp_per_block = GetAttr<int32_t>(attrs, "warp_per_block");
  if (XLA_FFI_PREDICT_FALSE(input.dimensions().size() != 2)) {
    return Error::InvalidArgument("input must be a 2D tensor");
  }
  const int hiddenDim = static_cast<int>(input.dimensions()[1]);
  assert(hiddenDim > 0 && hiddenDim <= h->config.hiddenDim);
  assert(ByteWidth(topk_ids.element_type()) == sizeof(index_t));

  float* weightsPtr = nullptr;
  if (args.size() > 2) {
    auto weights = GetArg<BufferR2<F32>>(args, 2);
    weightsPtr = weights.typed_data();
  }
  // XPUT("MoriCombine h: %p, input=%d topk_ids=%d hiddenDim: %d weights=%p
  // useExternalInpBuffer=%d",
  //   h, (int)input.size_bytes(), (int)topk_ids.size_bytes(), hiddenDim, weightsPtr,
  //   h->config.useExternalInpBuffer);

  //   // NOTE reading directly from GPU mem!!
  //   index_t total_recv_token_num = h->totalRecvTokenNum[0];
  //   // we need to copy data to shmemCombineInpTokMemObj directly
  if (!h->config.useExternalInpBuffer) {
    return Error::Internal("useExternalInpBuffer=false is not supported");
    // GpuCopy(h->shmemCombineInpTokMemObj->Get(), input.untyped_data(),
    //     out_weights->size_bytes(), // should this be input.size_bytes()?
    //     stream);
  }
  mori::moe::LaunchCombine(*h, input.untyped_data(), weightsPtr,
                           topk_ids.typed_data(),  // input.dimensions()[0],
                           h->curRankNumToken, FFIType2HipType(input.element_type()), block_num,
                           rdma_block_num, warp_per_block, h->config.useExternalInpBuffer ? 1 : 0,
                           stream, hiddenDim);

  GpuCopy(out->untyped_data(), h->GetShmemCombineOutTokMemObj()->Get(), out->size_bytes(), stream);
  // {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim},

  if (weightsPtr) {
    auto out_weights = GetRet<BufferR2<F32>>(rets, 1);
    //{handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
    GpuCopy(out_weights->untyped_data(), h->shmemCombineOutWeightsMemObj->Get(),
            out_weights->size_bytes(), stream);
  }
  return Error::Success();
}

Error MoriResetImpl(hipStream_t stream, EpDispatchCombineHandle* h) {
  // XPUT("MoriResetImpl stream: %p", stream);
  h->LaunchReset(stream);
  return Error::Success();
}

Error GetDispatchSrcTokenId(hipStream_t stream, EpDispatchCombineHandle* h, RemainingRets rets) {
  // XPUT("GetDispatchSrcTokenId h: %p", h);
  auto out = GetRet<BufferR1<S32>>(rets, 0);
  // NOTE here we read the whole buffer but the actual # of tokens received could be less
  // we do not want to read it since it requires explicit stream synchronization otherwise
  GpuCopy(out->untyped_data(), h->dispTokIdToSrcTokIdMemObj->Get(), out->size_bytes(), stream);
  return Error::Success();
}

ErrorOr<std::unique_ptr<EpDispatchCombineState>> EpDispatchCombineInstantiate(
    Dictionary attrs) try {
  auto ep_config = attrs.get<Span<const int32_t>>("ep_config");
  using ErrOr = ErrorOr<std::unique_ptr<EpDispatchCombineState>>;

  if (ep_config.has_error()) {
    return ErrOr(ep_config.error());
  }

  // Use the packed config as cache key so all call sites with the same
  // ep_config share a single EpDispatchCombineHandle.
  std::vector<int32_t> key(ep_config->begin(), ep_config->end());

  // Decode once; reused below for both rank-based device routing (SPMT) and
  // (on cache miss) handle construction.
  auto cfg = EpDispatchCombineConfig::FromPackedI32Array(key.data(), key.size());

#ifdef MORI_MULTITHREAD_SUPPORT
  // SPMT: XLA FFI handlers run on framework worker threads where
  // hipGetDevice() does NOT match the rank's device. Look up the device
  // recorded at ShmemInit time and bind it on this thread (RAII-restored on
  // exit so XLA's other state isn't disturbed) before any HIP call. This
  // ensures GetHandleCacheSlot() and ShmemStatesSingleton::GetInstance()
  // (both keyed by hipGetDevice()) hit the right slot.
  ScopedDevice _dev_guard(mori::shmem::ShmemStatesSingleton::GetDeviceByRank(cfg.rank));
#endif

  auto& slot = GetHandleCacheSlot();
  std::lock_guard<std::mutex> lock(slot.mu);
  auto& entry = slot.map[key];
  if (!entry) {
    auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
    states->CheckStatusValid();
    states->bootStates->bootNet->Barrier();
    // XPUT("EpDispatchCombineInstantiate: creating new handle for rank %d "
    //      "(#attrs: %zu)", cfg.rank, attrs.size());
    entry = std::make_unique<EpDispatchCombineHandle>(cfg);
  }
  return std::make_unique<EpDispatchCombineState>(entry.get());
} catch (const std::exception& e) {
  return ErrorOr<std::unique_ptr<EpDispatchCombineState>>(Error::Internal(e.what()));
}

XLA_FFI_DEFINE_HANDLER(EpDispatchCombineInstHandler, EpDispatchCombineInstantiate,
                       Ffi::BindInstantiate().Attrs());

Error EpDispatchCombineImpl(hipStream_t stream, EpDispatchCombineState* state, Dictionary attrs,
                            RemainingArgs args, RemainingRets rets) try {
  auto& h = *state->handle;
  // XPUT("EpDispatchCombineImpl stream=%p rank=%d  attrs: %zu",
  //     stream, h.config.rank, attrs.size());
#ifdef MORI_MULTITHREAD_SUPPORT
  // SPMT: bind this thread to the rank's device for the duration of this
  // FFI call (RAII-restored). See EpDispatchCombineInstantiate for rationale.
  ScopedDevice _dev_guard(mori::shmem::ShmemStatesSingleton::GetDeviceByRank(h.config.rank));
#endif
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

XLA_FFI_DEFINE_HANDLER(EpDispatchCombineHandler, EpDispatchCombineImpl,
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
    d["instantiate"] = py::capsule(reinterpret_cast<void*>(EpDispatchCombineInstHandler));
    d["execute"] = py::capsule(reinterpret_cast<void*>(EpDispatchCombineHandler));
    return d;
  });
  m.def("preload_kernels", []() { mori::moe::KernelRegistry::Instance().AutoLoad(); });
  m.def("clear_ep_handle_cache", []() {
    // Clear only the calling thread's slot. Under SPMT, each thread is
    // bound to its own GPU and owns one slot; clearing all slots from one
    // thread would invoke ~EpDispatchCombineHandle on OTHER GPUs' handles
    // while the calling thread's hipDevice is still set to its own GPU,
    // and ShmemFree would then look up addresses in the wrong VA manager.
    auto& slot = GetHandleCacheSlot();
    std::lock_guard<std::mutex> lock(slot.mu);
    slot.map.clear();
  });
}

}  // namespace mori
