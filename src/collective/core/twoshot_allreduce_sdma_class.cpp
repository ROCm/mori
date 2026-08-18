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

#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

namespace {
template <typename T>
size_t FixedSlotElements(size_t input_buffer_size, int npes) {
  if (npes <= 0) throw std::invalid_argument("npes must be positive");
  if (input_buffer_size == 0 || input_buffer_size % sizeof(T) != 0)
    throw std::invalid_argument("input_buffer_size must be a positive whole number of elements");
  constexpr size_t pack_size = 16 / sizeof(T);
  const size_t elements = input_buffer_size / sizeof(T);
  const size_t shard = (elements + static_cast<size_t>(npes) - 1) / static_cast<size_t>(npes);
  const size_t rounded = ((shard + pack_size - 1) / pack_size) * pack_size;
  if (rounded > std::numeric_limits<size_t>::max() - 64)
    throw std::overflow_error("AllReduce scratch size overflow");
  return rounded + 64;
}

template <typename T>
size_t FixedScratchBytes(size_t input_buffer_size, int npes) {
  const size_t slot = FixedSlotElements<T>(input_buffer_size, npes);
  if (slot > std::numeric_limits<size_t>::max() / (static_cast<size_t>(npes) + 1) / sizeof(T))
    throw std::overflow_error("AllReduce scratch size overflow");
  return (static_cast<size_t>(npes) + 1) * slot * sizeof(T);
}
}  // namespace

// ---------------------------------------------------------------------------
// Delegating constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t transit_buffer_size,
                                bool copy_output_to_user, bool /*use_graph_mode*/)
    : AllreduceSdma(myPe, npes, transit_buffer_size, transit_buffer_size, copy_output_to_user,
                    false) {}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t input_buffer_size,
                                size_t output_buffer_size, bool copy_output_to_user,
                                bool /*use_graph_mode*/)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      max_blocks_(getDeviceMaxBlocks()),
      flags_(nullptr, ShmemDeleter()),
      barrierPtr_(nullptr),
      barrierMem_(nullptr, ShmemDeleter()),
      input_transit_buffer_(nullptr),
      input_transit_buffer_size_(FixedScratchBytes<T>(input_buffer_size, npes)),
      slot_stride_elements_(FixedSlotElements<T>(input_buffer_size, npes)),
      input_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      async_in_progress_(false),
      async_input_(nullptr),
      async_output_(nullptr),
      async_total_count_(0),
      async_stream_(nullptr),
      async_start_event_(nullptr),
      async_start_time_(0.0),
      copy_output_to_user_(copy_output_to_user) {
  (void)output_buffer_size;
  if (myPe_ < 0 || myPe_ >= npes_) throw std::invalid_argument("myPe must be in [0, npes)");
  if (max_blocks_ > kSdmaMaxBlocks)
    throw std::runtime_error("GPU block count exceeds one-shot barrier capacity");
  // 1. Allocate SDMA completion flags
  size_t flagsSize = 2 * npes_ * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (!flags) throw std::runtime_error("Failed to allocate flags memory");
  flags_.reset(static_cast<uint64_t*>(flags));
  memset(flags_.get(), 0, flagsSize);
  flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_.get());
  if (!flagsObj_.IsValid()) throw std::runtime_error("Failed to get valid flags memory object");

  // 2. Allocate CrossPeBarrier (device-scope broadcast flag, ~128 bytes)
  size_t barrierSize = sizeof(CrossPeBarrier);
  void* bMem = shmem::ShmemMalloc(barrierSize);
  if (!bMem) throw std::runtime_error("Failed to allocate barrier memory");
  barrierMem_.reset(bMem);
  barrierPtr_ = reinterpret_cast<CrossPeBarrier*>(bMem);
  hipError_t me = hipMemset(bMem, 0, barrierSize);
  if (me != hipSuccess) throw std::runtime_error("Failed to zero-init barrier memory");

  // 3. Allocate a fixed-stride reduce-scatter scratch buffer. Keeping rank
  // slots at stable offsets prevents a region written by a prior CU reduce
  // from becoming an SDMA-written peer slot when graph message sizes change.
  input_transit_buffer_ = shmem::ShmemMalloc(input_transit_buffer_size_);
  if (!input_transit_buffer_) throw std::runtime_error("Failed to allocate input transit buffer");
  input_transit_buffer_ptr_.reset(input_transit_buffer_);
  input_transit_buffer_obj_ =
      shmem::ShmemSymmetricRegister(input_transit_buffer_, input_transit_buffer_size_);
  if (!input_transit_buffer_obj_.IsValid())
    throw std::runtime_error("Failed to register input transit buffer");

  hipError_t event_err = hipEventCreateWithFlags(&async_start_event_, hipEventDisableTiming);
  if (event_err != hipSuccess)
    throw std::runtime_error("Failed to create async stream-ordering event");

  printf("AllreduceSdma(SDMA) initialized: PE %d of %d, max_blocks=%d\n", myPe_, npes_,
         max_blocks_);
  printf("  Flags: %zu bytes at %p\n", flagsSize, flags_.get());
  printf("  Barrier: %zu bytes at %p\n", barrierSize, bMem);
  printf("  Reduce scratch buffer: %.2f MB at %p\n", input_transit_buffer_size_ / (1024.0 * 1024.0),
         input_transit_buffer_);
}

// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
  if (async_in_progress_) {
    cancel_async();
  }
  if (async_start_event_ != nullptr) {
    (void)hipEventDestroy(async_start_event_);
  }
  if (flags_) {
    printf("AllreduceSdma destroyed: PE %d\n", myPe_);
  }
}

// copy_input_to_transit implementation
template <typename T>
void AllreduceSdma<T>::copy_input_to_transit(T* input, size_t total_count, hipStream_t stream) {
  size_t input_bytes = total_count * dtype_size_;

  // Verify pointer validity
  if (input == nullptr) {
    fprintf(stderr, "PE %d: Input pointer is null\n", myPe_);
    throw std::runtime_error("Input pointer is null");
  }

  if (input_transit_buffer_ == nullptr) {
    fprintf(stderr, "PE %d: Input transit buffer is null\n", myPe_);
    throw std::runtime_error("Input transit buffer is null");
  }

  // Copy from user input buffer to input transit buffer
  // No explicit sync needed — same-stream operations are ordered by the GPU
  hipError_t err = hipSuccess;
  if (stream != nullptr) {
    err =
        hipMemcpyAsync(input_transit_buffer_, input, input_bytes, hipMemcpyDeviceToDevice, stream);
  } else {
    err = hipMemcpy(input_transit_buffer_, input, input_bytes, hipMemcpyDeviceToDevice);
  }

  if (err != hipSuccess) {
    fprintf(stderr, "PE %d: Failed to copy input to transit buffer: %s\n", myPe_,
            hipGetErrorString(err));
    throw std::runtime_error("Input copy failed");
  }
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
  (void)input;
  (void)output;
  (void)total_count;
  (void)stream;
  throw std::runtime_error("AllreduceSdma::operator() removed — use Python JIT launch path");
}

// ================ Async API Implementations ================

template <typename T>
bool AllreduceSdma<T>::start_async(T* input, T* output, size_t total_count, hipStream_t stream) {
  (void)input;
  (void)output;
  (void)total_count;
  (void)stream;
  throw std::runtime_error("AllreduceSdma::start_async removed — use Python JIT launch path");
}

template <typename T>
double AllreduceSdma<T>::wait_async(hipStream_t stream) {
  (void)stream;
  throw std::runtime_error("AllreduceSdma::wait_async removed — use Python JIT launch path");
}

template <typename T>
void AllreduceSdma<T>::cancel_async() {
  if (async_in_progress_) {
    printf("PE %d: Cancelling async operation\n", myPe_);
    async_in_progress_ = false;
    async_input_ = nullptr;
    async_output_ = nullptr;
    async_total_count_ = 0;
    async_stream_ = nullptr;
    async_start_time_ = 0.0;
  }
}

// ================ END: Async API Implementations ================

// allreduce_inplace — removed; use prepare/finish JIT path
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::allreduce_inplace(T* /*data*/, size_t /*total_count*/,
                                         hipStream_t /*stream*/) {
  throw std::runtime_error("AllreduceSdma::allreduce_inplace removed — use Python JIT launch path");
}

// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::resetFlags() {
  if (flags_) {
    memset(flags_.get(), 0, 2 * npes_ * sizeof(uint64_t));
  }
}

// ---------------------------------------------------------------------------
// JIT launch helpers
// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::fill_jit_args_(const T* input, size_t total_count) {
  jit_args_.myPe = myPe_;
  jit_args_.npes = npes_;
  jit_args_.input = input;
  jit_args_.output = nullptr;
  jit_args_.dstMemObj = input_transit_buffer_obj_;
  jit_args_.outputMemObj = input_transit_buffer_obj_;
  jit_args_.flagsMemObj = flagsObj_;
  jit_args_.barrier = barrierPtr_;
  jit_args_.elementCount = total_count;
  jit_args_.slotStrideElements = slot_stride_elements_;
  jit_args_.outputBaseOffsetBytes = 0;
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_reduce_scatter(const T* input, T* output, size_t total_count,
                                                 hipStream_t stream) {
  if (async_in_progress_) throw std::runtime_error("Async operation in progress");
  (void)stream;
  constexpr size_t pack_size = 16 / sizeof(T);
  const size_t collective_pack = static_cast<size_t>(npes_) * pack_size;
  if (input == nullptr || output == nullptr) throw std::invalid_argument("input/output is null");
  if (total_count == 0 || total_count % collective_pack != 0)
    throw std::invalid_argument("AllReduce count must be divisible by npes * pack_size");
  if (total_count > std::numeric_limits<size_t>::max() / dtype_size_)
    throw std::overflow_error("AllReduce byte count overflow");
  const size_t required_slot = total_count / npes_;
  if (required_slot > slot_stride_elements_)
    throw std::runtime_error("AllReduce input exceeds fixed reduce scratch capacity");
  fill_jit_args_(input, total_count);
  jit_args_.output = output;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
std::tuple<int, int> AllreduceSdma<T>::get_reduce_scatter_grid(size_t total_count) const {
  constexpr size_t pack_size = 16 / sizeof(T);
  size_t packedPerRank = (total_count / npes_ + pack_size - 1) / pack_size;
  const size_t total_bytes = total_count * dtype_size_;
  int threads = total_bytes <= (1U << 20) || total_bytes > (16U << 20)
                    ? (packedPerRank >= 1024 ? 1024 : 512)
                    : 512;
  int blocks = std::min(max_blocks_, static_cast<int>((packedPerRank + threads - 1) / threads));
  if (blocks < 1) blocks = 1;
  return {blocks, threads};
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_allgather(size_t total_count, hipStream_t stream) {
  jit_args_.input = nullptr;
  jit_args_.elementCount = total_count;
  const size_t shard_bytes = total_count / npes_ * dtype_size_;
  const size_t output_bytes = total_count * dtype_size_;
  const size_t slot_bytes = slot_stride_elements_ * dtype_size_;
  jit_args_.outputBaseOffsetBytes =
      output_bytes <= slot_bytes
          ? static_cast<size_t>(npes_) * slot_bytes - static_cast<size_t>(myPe_) * shard_bytes
          : 0;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double AllreduceSdma<T>::finish_sync(T* output, size_t total_count, hipStream_t stream,
                                     bool force_copy_output_to_user, bool direct_output) {
  (void)force_copy_output_to_user;
  if (!direct_output) {
    auto* source = static_cast<uint8_t*>(input_transit_buffer_) + jit_args_.outputBaseOffsetBytes;
    hipError_t err =
        hipMemcpyAsync(output, source, total_count * dtype_size_, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) throw std::runtime_error("Failed to copy AllReduce output");
  }
  return 0.0;
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_async_reduce_scatter(const T* input, T* output,
                                                       size_t total_count, hipStream_t stream) {
  bool expected = false;
  if (!async_in_progress_.compare_exchange_strong(expected, true))
    throw std::runtime_error("Another async operation is already in progress");

  async_input_ = const_cast<T*>(input);
  async_output_ = output;
  async_total_count_ = total_count;
  async_stream_ = stream;
  async_start_time_ = CollectiveWallTime();

  try {
    constexpr size_t pack_size = 16 / sizeof(T);
    const size_t collective_pack = static_cast<size_t>(npes_) * pack_size;
    if (input == nullptr || output == nullptr) throw std::invalid_argument("input/output is null");
    if (total_count == 0 || total_count % collective_pack != 0)
      throw std::invalid_argument("AllReduce count must be divisible by npes * pack_size");
    if (total_count > std::numeric_limits<size_t>::max() / dtype_size_)
      throw std::overflow_error("AllReduce byte count overflow");
    const size_t required_slot = total_count / npes_;
    if (required_slot > slot_stride_elements_)
      throw std::runtime_error("AllReduce input exceeds fixed reduce scratch capacity");
    fill_jit_args_(input, total_count);
    jit_args_.output = output;
  } catch (...) {
    async_in_progress_ = false;
    throw;
  }
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_async_allgather_put(size_t total_count, hipStream_t /*stream*/) {
  jit_args_.input = nullptr;
  jit_args_.elementCount = total_count;
  const size_t shard_bytes = total_count / npes_ * dtype_size_;
  const size_t output_bytes = total_count * dtype_size_;
  const size_t slot_bytes = slot_stride_elements_ * dtype_size_;
  jit_args_.outputBaseOffsetBytes =
      output_bytes <= slot_bytes
          ? static_cast<size_t>(npes_) * slot_bytes - static_cast<size_t>(myPe_) * shard_bytes
          : 0;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
void AllreduceSdma<T>::after_async_start(bool capturing) {
  if (!capturing) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      async_in_progress_ = false;
      throw std::runtime_error("Async kernel launch failed");
    }
  }
  hipError_t err = hipEventRecord(async_start_event_, async_stream_);
  if (err != hipSuccess) {
    async_in_progress_ = false;
    throw std::runtime_error("Failed to record async start event");
  }
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_async_wait(hipStream_t stream) {
  if (!async_in_progress_) throw std::runtime_error("No async operation in progress");
  jit_args_.input = nullptr;
  jit_args_.elementCount = async_total_count_;
  const size_t shard_bytes = async_total_count_ / npes_ * dtype_size_;
  const size_t output_bytes = async_total_count_ * dtype_size_;
  const size_t slot_bytes = slot_stride_elements_ * dtype_size_;
  jit_args_.outputBaseOffsetBytes =
      output_bytes <= slot_bytes
          ? static_cast<size_t>(npes_) * slot_bytes - static_cast<size_t>(myPe_) * shard_bytes
          : 0;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double AllreduceSdma<T>::finish_async_wait(hipStream_t stream, bool capturing, bool direct_output) {
  hipStream_t ws = (stream != nullptr) ? stream : async_stream_;
  if (ws != async_stream_) {
    hipError_t err = hipStreamWaitEvent(ws, async_start_event_, 0);
    if (err != hipSuccess) throw std::runtime_error("Failed to order async wait stream");
  }
  if (!direct_output) {
    auto* source = static_cast<uint8_t*>(input_transit_buffer_) + jit_args_.outputBaseOffsetBytes;
    hipError_t err = hipMemcpyAsync(async_output_, source, async_total_count_ * dtype_size_,
                                    hipMemcpyDeviceToDevice, ws);
    if (err != hipSuccess) throw std::runtime_error("Failed to copy async AllReduce output");
  }
  if (!capturing) {
    hipError_t err = ws ? hipStreamSynchronize(ws) : hipDeviceSynchronize();
    if (err != hipSuccess) throw std::runtime_error("Synchronization failed");
  }

  double duration = CollectiveWallTime() - async_start_time_;
  async_in_progress_ = false;
  async_input_ = nullptr;
  async_output_ = nullptr;
  async_total_count_ = 0;
  async_stream_ = nullptr;
  async_start_time_ = 0.0;
  return duration;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class AllreduceSdma<uint32_t>;
template class AllreduceSdma<uint64_t>;
template class AllreduceSdma<int32_t>;
template class AllreduceSdma<int64_t>;
template class AllreduceSdma<float>;
template class AllreduceSdma<double>;
template class AllreduceSdma<half>;
template class AllreduceSdma<hip_bfloat16>;

}  // namespace collective
}  // namespace mori
