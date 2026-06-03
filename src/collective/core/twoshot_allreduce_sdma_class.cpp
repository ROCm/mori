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
#include <stdexcept>

#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// Delegating constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t transit_buffer_size,
                                bool copy_output_to_user, bool /*use_graph_mode*/)
    : AllreduceSdma(myPe, npes, 0, transit_buffer_size, copy_output_to_user, false) {}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t /*input_buffer_size*/,
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
      input_transit_buffer_size_(0),
      input_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      output_transit_buffer_(nullptr),
      output_transit_buffer_size_(output_buffer_size),
      output_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      async_in_progress_(false),
      async_input_(nullptr),
      async_output_(nullptr),
      async_total_count_(0),
      async_stream_(nullptr),
      async_start_time_(0.0),
      copy_output_to_user_(copy_output_to_user) {
  // 1. Allocate SDMA completion flags
  size_t flagsSize = npes_ * sizeof(uint64_t);
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

  // 3. Allocate output transit buffer (gather + reduce + allgather)
  output_transit_buffer_ = shmem::ShmemMalloc(output_transit_buffer_size_);
  if (!output_transit_buffer_) throw std::runtime_error("Failed to allocate output transit buffer");
  output_transit_buffer_ptr_.reset(output_transit_buffer_);

  output_transit_buffer_obj_ =
      shmem::ShmemSymmetricRegister(output_transit_buffer_, output_transit_buffer_size_);
  if (!output_transit_buffer_obj_.IsValid())
    throw std::runtime_error("Failed to register output transit buffer");

  printf("AllreduceSdma(SDMA) initialized: PE %d of %d, max_blocks=%d\n", myPe_, npes_,
         max_blocks_);
  printf("  Flags: %zu bytes at %p\n", flagsSize, flags_.get());
  printf("  Barrier: %zu bytes at %p\n", barrierSize, bMem);
  printf("  Output transit buffer: %.2f MB at %p\n",
         output_transit_buffer_size_ / (1024.0 * 1024.0), output_transit_buffer_);
}

// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
  if (async_in_progress_) {
    cancel_async();
  }
  if (flags_) {
    printf("AllreduceSdma destroyed: PE %d\n", myPe_);
  }
}

// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::ensure_buffer_size(void*& buffer,
                                          std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                                          size_t& current_size,
                                          application::SymmMemObjPtr& buffer_obj,
                                          size_t required_size, const char* buffer_name) {
  if (required_size <= current_size) {
    return true;
  }

  // If buffer is not large enough, reallocate
  printf("PE %d: %s too small: required %.2f MB, current %.2f MB\n", myPe_, buffer_name,
         required_size / (1024.0 * 1024.0), current_size / (1024.0 * 1024.0));

  // First release the old one
  buffer_ptr.reset();

  // Allocate new one
  current_size = required_size;
  buffer = shmem::ShmemMalloc(current_size);
  if (buffer == nullptr) {
    fprintf(stderr, "PE %d: Failed to reallocate %s of size %.2f MB\n", myPe_, buffer_name,
            current_size / (1024.0 * 1024.0));
    return false;
  }
  buffer_ptr.reset(buffer);

  // Re-register
  buffer_obj = shmem::ShmemSymmetricRegister(buffer, current_size);
  if (!buffer_obj.IsValid()) {
    fprintf(stderr, "PE %d: Failed to re-register %s\n", myPe_, buffer_name);
    return false;
  }

  printf("PE %d: %s reallocated to %.2f MB\n", myPe_, buffer_name,
         current_size / (1024.0 * 1024.0));
  return true;
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

// copy_output_to_user implementation
// For AllReduce: output is total_count elements (same size as input, NOT npes * total_count)
template <typename T>
void AllreduceSdma<T>::copy_output_to_user(T* output, size_t total_count, hipStream_t stream) {
  size_t bytes = total_count * dtype_size_;
  if (!output) throw std::runtime_error("Output pointer is null");
  if (!output_transit_buffer_) throw std::runtime_error("Output transit buffer is null");

  hipError_t err =
      stream
          ? hipMemcpyAsync(output, output_transit_buffer_, bytes, hipMemcpyDeviceToDevice, stream)
          : hipMemcpy(output, output_transit_buffer_, bytes, hipMemcpyDeviceToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "PE %d: copy_output_to_user failed: %s\n", myPe_, hipGetErrorString(err));
    throw std::runtime_error("Output copy failed");
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
    memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
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
  jit_args_.dstMemObj = output_transit_buffer_obj_;
  jit_args_.flagsMemObj = flagsObj_;
  jit_args_.barrier = barrierPtr_;
  jit_args_.elementCount = total_count;
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_reduce_scatter(const T* input, T* output, size_t total_count,
                                                 hipStream_t stream) {
  if (async_in_progress_) throw std::runtime_error("Async operation in progress");
  (void)output;
  (void)stream;
  fill_jit_args_(input, total_count);
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
std::tuple<int, int> AllreduceSdma<T>::get_reduce_scatter_grid(size_t total_count) const {
  constexpr size_t pack_size = 16 / sizeof(T);
  size_t packedPerRank = (total_count / npes_ + pack_size - 1) / pack_size;
  int threads = 512;
  int blocks = std::min(max_blocks_, static_cast<int>((packedPerRank + threads - 1) / threads));
  if (blocks < 1) blocks = 1;
  return {blocks, threads};
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_allgather(size_t total_count, hipStream_t /*stream*/) {
  jit_args_.input = nullptr;
  jit_args_.elementCount = total_count;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double AllreduceSdma<T>::finish_sync(T* output, size_t total_count, hipStream_t stream,
                                     bool force_copy_output_to_user) {
  if (copy_output_to_user_ || force_copy_output_to_user) {
    copy_output_to_user(output, total_count, stream);
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

  size_t required = (total_count / npes_) * npes_ * dtype_size_;
  if (!ensure_buffer_size(output_transit_buffer_, output_transit_buffer_ptr_,
                          output_transit_buffer_size_, output_transit_buffer_obj_, required,
                          "output transit buffer")) {
    async_in_progress_ = false;
    throw std::runtime_error("Buffer allocation failed");
  }

  fill_jit_args_(input, total_count);
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_async_allgather_put(size_t total_count, hipStream_t /*stream*/) {
  jit_args_.input = nullptr;
  jit_args_.elementCount = total_count;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
void AllreduceSdma<T>::after_async_start() {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    async_in_progress_ = false;
    throw std::runtime_error("Async kernel launch failed");
  }
}

template <typename T>
int64_t AllreduceSdma<T>::prepare_async_wait(hipStream_t stream) {
  if (!async_in_progress_) throw std::runtime_error("No async operation in progress");
  (void)stream;
  jit_args_.input = nullptr;
  jit_args_.elementCount = async_total_count_;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double AllreduceSdma<T>::finish_async_wait(hipStream_t stream) {
  hipStream_t ws = (stream != nullptr) ? stream : async_stream_;
  hipError_t err = ws ? hipStreamSynchronize(ws) : hipDeviceSynchronize();
  if (err != hipSuccess) throw std::runtime_error("Synchronization failed");

  if (copy_output_to_user_) {
    copy_output_to_user(async_output_, async_total_count_, ws);
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
