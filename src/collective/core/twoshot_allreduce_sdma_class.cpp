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
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "mori/collective/allreduce/twoshot_sdma_async_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp"
#include "mori/collective/inter_node/executors/ring_1d.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

namespace {

template <typename T>
size_t SdmaTransitUsedBytes(size_t total_count, int npes, size_t dtype_size) {
    constexpr int pack_size = packed_t<T>::P::size;
    const size_t element_count_per_rank =
        ((total_count / static_cast<size_t>(npes) + static_cast<size_t>(pack_size) - 1U) /
         static_cast<size_t>(pack_size)) *
        static_cast<size_t>(pack_size);
    return element_count_per_rank * static_cast<size_t>(npes) * dtype_size;
}

inline bool SdmaShouldZeroTransit() {
    const char* e = std::getenv("MORI_SDMA_ZERO_TRANSIT");
    if (e && e[0] == '1' && e[1] == '\0') return true;
    return false;
}

inline bool UseCopyKernel() {
    const char* e = std::getenv("MORI_COPY_KERNEL");
    return e && e[0] == '1' && e[1] == '\0';
}

inline bool SdmaSeparateAgBuffer() {
    const char* e = std::getenv("MORI_SEPARATE_AG_BUFFER");
    return e && e[0] == '1' && e[1] == '\0';
}

inline bool UseRingExecutorProbe() {
    const char* e = std::getenv("MORI_RING_EXECUTOR");
    return e && e[0] == '1' && e[1] == '\0';
}

struct DirectLaneSpec {
    int forward;
    int reverse;
};

inline DirectLaneSpec ParseDirectLaneSpec(const char* s) {
    DirectLaneSpec spec{3, 3};
    if (s == nullptr || s[0] == '\0') return spec;
    int f = 0;
    int r = 0;
    if (std::sscanf(s, "%dF%dR", &f, &r) == 2 ||
        std::sscanf(s, "%df%dr", &f, &r) == 2) {
        if (f + r > 0) return DirectLaneSpec{f, r};
    }
    if (std::sscanf(s, "%d", &f) == 1 && f > 0) {
        return DirectLaneSpec{f, 0};
    }
    return spec;
}

inline int CopyKernelBlocks() {
    if (const char* e = std::getenv("MORI_COPY_KERNEL_BLOCKS")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 1024;
}

inline int CopyKernelThreads() {
    if (const char* e = std::getenv("MORI_COPY_KERNEL_THREADS")) {
        int v = std::atoi(e);
        if (v == 128 || v == 256 || v == 512) return v;
    }
    return 256;
}

__global__ void D2DVectorCopyKernel(const void* __restrict__ src_void,
                                    void* __restrict__ dst_void,
                                    size_t bytes,
                                    bool aligned16) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const uint8_t* __restrict__ src_b = static_cast<const uint8_t*>(src_void);
    uint8_t* __restrict__ dst_b = static_cast<uint8_t*>(dst_void);

    if (aligned16) {
        const size_t n_vec = bytes / sizeof(uint4);
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(src_b);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(dst_b);
        for (size_t i = tid; i < n_vec; i += stride) {
            dst[i] = src[i];
        }
        const size_t tail = bytes - n_vec * sizeof(uint4);
        const size_t base = n_vec * sizeof(uint4);
        for (size_t i = tid; i < tail; i += stride) {
            dst_b[base + i] = src_b[base + i];
        }
    } else {
        for (size_t i = tid; i < bytes; i += stride) {
            dst_b[i] = src_b[i];
        }
    }
}

}  // namespace

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

  {
    HSAuint64* gpuSig = nullptr;
    uint32_t   gpuNumQ = 0;
    hipError_t sigCopy = hipMemcpy(&gpuSig, &(output_transit_buffer_obj_.gpu->signalPtrs),
                                   sizeof(HSAuint64*), hipMemcpyDeviceToHost);
    hipError_t numQCopy = hipMemcpy(&gpuNumQ, &(output_transit_buffer_obj_.gpu->sdmaNumQueue),
                                    sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (sigCopy != hipSuccess || numQCopy != hipSuccess) {
      throw std::runtime_error("Failed to read output transit SDMA metadata");
    }
    if (gpuSig && gpuNumQ > 0) {
      sdma_num_queue_ = gpuNumQ;
      size_t sigSize = static_cast<size_t>(npes_) * gpuNumQ * sizeof(HSAuint64);
      hipError_t sigZero = hipMemset(gpuSig, 0, sigSize);
      if (sigZero != hipSuccess) {
        throw std::runtime_error("Failed to zero output transit SDMA signals");
      }
    }
    hipError_t agBaseAlloc =
        hipMalloc(&pipeline_ag_gen_by_q_d_,
                  kMaxTrackedSdmaQueues * sizeof(uint64_t));
    if (agBaseAlloc != hipSuccess) {
      throw std::runtime_error("Failed to allocate pipeline_ag_gen_by_q_d_");
    }
    hipError_t agBaseZero = hipMemset(pipeline_ag_gen_by_q_d_, 0,
                                      kMaxTrackedSdmaQueues * sizeof(uint64_t));
    if (agBaseZero != hipSuccess) {
      throw std::runtime_error("Failed to zero pipeline_ag_gen_by_q_d_");
    }
  }

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
  // Drain all GPU work (including in-flight SDMA transfers) before
  // ShmemDeleter frees the symmetric memory regions they reference.
  (void)hipDeviceSynchronize();
  if (phase_ts_d_) {
    (void)hipFree(phase_ts_d_);
    phase_ts_d_ = nullptr;
  }
  if (pipeline_ag_gen_by_q_d_) {
    (void)hipFree(pipeline_ag_gen_by_q_d_);
    pipeline_ag_gen_by_q_d_ = nullptr;
  }
  if (copy_start_event_) {
    (void)hipEventDestroy(copy_start_event_);
    copy_start_event_ = nullptr;
  }
  if (copy_end_event_) {
    (void)hipEventDestroy(copy_end_event_);
    copy_end_event_ = nullptr;
  }
  if (post_ag_flag_d_) {
    (void)hipFree(post_ag_flag_d_);
    post_ag_flag_d_ = nullptr;
  }
  if (flags_) {
    printf("AllreduceSdma destroyed: PE %d\n", myPe_);
  }
}

// ---------------------------------------------------------------------------
// D' prototype: lazy shmem-register user output buffer
// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::enable_register_user_output(bool on) {
  if (on == register_user_output_enabled_) return;
  register_user_output_enabled_ = on;
  if (on) {
    printf("PE %d: register_user_output ENABLED (D' lazy shmem register, "
           "cache cap=%zu)\n", myPe_, kUserOutputCacheCap);
  }
}

template <typename T>
bool AllreduceSdma<T>::register_user_output(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    last_register_us_ = 0.0;
    last_register_was_hit_ = false;
    return false;
  }
  UserOutputCacheKey key{ptr, size};

  auto it = user_output_cache_.find(key);
  if (it != user_output_cache_.end() && it->second.IsValid()) {
    // Cache hit: move to MRU front
    user_output_cache_lru_.remove(key);
    user_output_cache_lru_.push_front(key);
    last_register_us_ = 0.0;
    last_register_was_hit_ = true;
    cache_hits_++;
    return true;
  }

  // Cache miss: collective register.
  auto t0 = std::chrono::steady_clock::now();
  application::SymmMemObjPtr obj = shmem::ShmemSymmetricRegister(ptr, size);
  auto t1 = std::chrono::steady_clock::now();
  last_register_us_ =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  last_register_was_hit_ = false;
  cache_misses_++;

  if (!obj.IsValid()) {
    fprintf(stderr, "PE %d: ShmemSymmetricRegister(user output %p, %zu bytes) "
                    "failed (invalid obj after %.2f us)\n",
            myPe_, ptr, size, last_register_us_);
    return false;
  }

  // Insert into cache with LRU eviction.
  while (user_output_cache_.size() >= kUserOutputCacheCap
         && !user_output_cache_lru_.empty()) {
    auto evict_key = user_output_cache_lru_.back();
    user_output_cache_lru_.pop_back();
    auto ev = user_output_cache_.find(evict_key);
    if (ev != user_output_cache_.end()) {
      // Deregistering is a collective; for safety in this prototype we
      // just drop the handle (shmem will still hold peer mappings until
      // shutdown). An explicit Deregister call would be nicer but
      // complicates the host path; leave as TODO.
      user_output_cache_.erase(ev);
    }
  }

  user_output_cache_[key] = obj;
  user_output_cache_lru_.push_front(key);
  if (myPe_ == 0) {
    printf("PE 0: user output register MISS (ptr=%p, size=%zu, %.1f us); "
           "cache now size=%zu, hits=%llu, misses=%llu\n",
           ptr, size, last_register_us_, user_output_cache_.size(),
           (unsigned long long)cache_hits_,
           (unsigned long long)cache_misses_);
  }
  return true;
}

template <typename T>
void AllreduceSdma<T>::enable_direct_output(bool on) {
  if (on == direct_output_enabled_) return;
  direct_output_enabled_ = on;
  if (on) {
    printf("PE %d: direct_output ENABLED (plan A — CU XGMI AG + direct "
           "write user_output; skips SDMA AG and external hipMemcpyAsync; "
           "MULTI_CHUNK only, copy_output_to_user=%d)\n",
           myPe_, int(copy_output_to_user_));
    if (!copy_output_to_user_) {
      fprintf(stderr,
              "PE %d: WARNING direct_output requires copy_output_to_user=on; "
              "currently off — direct_output will be a no-op.\n",
              myPe_);
    }
  } else {
    printf("PE %d: direct_output DISABLED\n", myPe_);
  }
}

template <typename T>
void AllreduceSdma<T>::enable_post_ag_wait(bool on) {
  if (on == post_ag_wait_enabled_) return;
  if (on) {
    if (post_ag_flag_d_ == nullptr) {
      hipError_t e = hipMalloc(&post_ag_flag_d_, sizeof(uint32_t));
      if (e != hipSuccess) {
        fprintf(stderr, "PE %d: hipMalloc(post_ag_flag) failed: %s\n",
                myPe_, hipGetErrorString(e));
        return;
      }
      hipError_t z = hipMemset(post_ag_flag_d_, 0, sizeof(uint32_t));
      if (z != hipSuccess) {
        fprintf(stderr, "PE %d: hipMemset(post_ag_flag) failed: %s\n",
                myPe_, hipGetErrorString(z));
        (void)hipFree(post_ag_flag_d_);
        post_ag_flag_d_ = nullptr;
        return;
      }
    }
    post_ag_wait_enabled_ = true;
    printf("PE %d: post_ag_wait ENABLED (Stage 1 prototype, flag at %p)\n",
           myPe_, post_ag_flag_d_);
  } else {
    post_ag_wait_enabled_ = false;
  }
}

// ---------------------------------------------------------------------------
// Copy-path instrumentation (baseline: single hipMemcpyAsync)
// Rule R0: we cannot change the copy strategy without having numbers for
// the current (baseline) copy first.  This block establishes those numbers.
// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::enable_copy_timing(bool on) {
  if (on == copy_timing_enabled_) return;
  if (on) {
    if (copy_start_event_ == nullptr) {
      hipError_t e = hipEventCreateWithFlags(&copy_start_event_, hipEventDefault);
      if (e != hipSuccess) {
        fprintf(stderr, "PE %d: copy_start_event create failed: %s\n",
                myPe_, hipGetErrorString(e));
        return;
      }
    }
    if (copy_end_event_ == nullptr) {
      hipError_t e = hipEventCreateWithFlags(&copy_end_event_, hipEventDefault);
      if (e != hipSuccess) {
        fprintf(stderr, "PE %d: copy_end_event create failed: %s\n",
                myPe_, hipGetErrorString(e));
        (void)hipEventDestroy(copy_start_event_);
        copy_start_event_ = nullptr;
        return;
      }
    }
    copy_timing_enabled_ = true;
    copy_timing_recorded_ = false;
    printf("PE %d: copy timing ENABLED (baseline: single hipMemcpyAsync)\n", myPe_);
  } else {
    copy_timing_enabled_ = false;
  }
}

template <typename T>
std::vector<double> AllreduceSdma<T>::get_copy_timing_ms() {
  std::vector<double> out;
  // Do NOT gate on copy_timing_enabled_ here: callers typically disable the
  // flag before reading (to stop subsequent kernels from overwriting events)
  // and only the "recorded" flag reflects whether events hold valid data.
  if (!copy_timing_recorded_) return out;
  if (copy_start_event_ == nullptr || copy_end_event_ == nullptr) return out;
  float ms = 0.0f;
  hipError_t e = hipEventElapsedTime(&ms, copy_start_event_, copy_end_event_);
  out.push_back(copy_timing_host_us_);
  out.push_back(e == hipSuccess ? static_cast<double>(ms) : -1.0);
  return out;
}

// ---------------------------------------------------------------------------
// Phase-level timestamp instrumentation
// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::enable_phase_timing(bool on) {
  if (on == phase_timing_enabled_) return;
  if (on) {
    if (phase_ts_d_ == nullptr) {
      hipError_t err = hipMalloc(&phase_ts_d_, kPhaseTsCapacity * sizeof(uint64_t));
      if (err != hipSuccess) {
        fprintf(stderr, "PE %d: enable_phase_timing: hipMalloc failed: %s\n",
                myPe_, hipGetErrorString(err));
        phase_ts_d_ = nullptr;
        return;
      }
      hipError_t z = hipMemset(phase_ts_d_, 0, kPhaseTsCapacity * sizeof(uint64_t));
      if (z != hipSuccess) {
        fprintf(stderr, "PE %d: enable_phase_timing: initial hipMemset failed: %s\n",
                myPe_, hipGetErrorString(z));
        (void)hipFree(phase_ts_d_);
        phase_ts_d_ = nullptr;
        return;
      }
    }
    // Clear on every enable, not only first allocation. Phase timing may be
    // toggled for one selected launch inside a continuous stream; stale slots
    // from a prior launch make the host-side breakdown meaningless.
    hipError_t z = hipMemset(phase_ts_d_, 0, kPhaseTsCapacity * sizeof(uint64_t));
    if (z != hipSuccess) {
      fprintf(stderr, "PE %d: enable_phase_timing: hipMemset failed: %s\n",
              myPe_, hipGetErrorString(z));
      return;
    }
    phase_timing_enabled_ = true;
    printf("PE %d: phase timing ENABLED (device buf %p, cap=%zu slots)\n",
           myPe_, phase_ts_d_, kPhaseTsCapacity);
  } else {
    phase_timing_enabled_ = false;
    printf("PE %d: phase timing DISABLED\n", myPe_);
  }
}

template <typename T>
std::vector<uint64_t> AllreduceSdma<T>::get_phase_timestamps() {
  std::vector<uint64_t> result(kPhaseTsCapacity, 0);
  if (phase_ts_d_ == nullptr) return result;
  hipError_t err = hipMemcpy(result.data(), phase_ts_d_,
                             kPhaseTsCapacity * sizeof(uint64_t),
                             hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "PE %d: get_phase_timestamps: hipMemcpy failed: %s\n",
            myPe_, hipGetErrorString(err));
  }
  return result;
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

  {
    HSAuint64* gpuSig = nullptr;
    uint32_t   gpuNumQ = 0;
    hipError_t sigCopy = hipMemcpy(&gpuSig, &(buffer_obj.gpu->signalPtrs),
                                   sizeof(HSAuint64*), hipMemcpyDeviceToHost);
    hipError_t numQCopy = hipMemcpy(&gpuNumQ, &(buffer_obj.gpu->sdmaNumQueue),
                                    sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (sigCopy != hipSuccess || numQCopy != hipSuccess) {
      fprintf(stderr, "PE %d: Failed to read %s SDMA metadata\n", myPe_, buffer_name);
      return false;
    }
    if (gpuSig && gpuNumQ > 0) {
      sdma_num_queue_ = gpuNumQ;
      size_t sigSize = static_cast<size_t>(npes_) * gpuNumQ * sizeof(HSAuint64);
      hipError_t sigZero = hipMemset(gpuSig, 0, sigSize);
      if (sigZero != hipSuccess) {
        fprintf(stderr, "PE %d: Failed to zero %s SDMA signals: %s\n",
                myPe_, buffer_name, hipGetErrorString(sigZero));
        return false;
      }
      pipeline_scatter_gen_ = 0;
      pipeline_ag_gen_ = 0;
      pipeline_ag_gen_by_q_.fill(0);
      if (pipeline_ag_gen_by_q_d_) {
        hipError_t genZero = hipMemset(pipeline_ag_gen_by_q_d_, 0,
                                       kMaxTrackedSdmaQueues * sizeof(uint64_t));
        if (genZero != hipSuccess) {
          fprintf(stderr, "PE %d: Failed to zero multi-q AG generations: %s\n",
                  myPe_, hipGetErrorString(genZero));
          return false;
        }
      }
      pipeline_reduce_gen_ = 0;
      if (flags_) {
        hipError_t flagZero =
            hipMemset(flags_.get(), 0, static_cast<size_t>(npes_) * sizeof(uint64_t));
        if (flagZero != hipSuccess) {
          fprintf(stderr, "PE %d: Failed to zero flags after %s realloc: %s\n",
                  myPe_, buffer_name, hipGetErrorString(flagZero));
          return false;
        }
      }
    }
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
  void* copy_src = (SdmaSeparateAgBuffer() && input_transit_buffer_ != nullptr)
      ? input_transit_buffer_ : output_transit_buffer_;

  const bool do_timing = copy_timing_enabled_ && stream != nullptr;
  if (do_timing) {
    hipError_t rec = hipEventRecord(copy_start_event_, stream);
    if (rec != hipSuccess) {
      fprintf(stderr, "PE %d: copy_start_event record failed: %s\n",
              myPe_, hipGetErrorString(rec));
      throw std::runtime_error("Copy timing start event failed");
    }
  }
  auto host_t0 = do_timing ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};

  hipError_t err = hipSuccess;
  if (UseCopyKernel()) {
    const uintptr_t src_addr = reinterpret_cast<uintptr_t>(copy_src);
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(output);
    const bool aligned16 = ((src_addr | dst_addr | bytes) & (sizeof(uint4) - 1)) == 0;
    D2DVectorCopyKernel<<<CopyKernelBlocks(), CopyKernelThreads(), 0, stream>>>(
        copy_src, output, bytes, aligned16);
    err = hipGetLastError();
    if (err == hipSuccess && stream == nullptr) {
      err = hipDeviceSynchronize();
    }
  } else {
    err = stream
        ? hipMemcpyAsync(output, copy_src, bytes, hipMemcpyDeviceToDevice, stream)
        : hipMemcpy(output, copy_src, bytes, hipMemcpyDeviceToDevice);
  }

  if (do_timing) {
    auto host_t1 = std::chrono::steady_clock::now();
    copy_timing_host_us_ =
        std::chrono::duration<double, std::micro>(host_t1 - host_t0).count();
    hipError_t rec = hipEventRecord(copy_end_event_, stream);
    if (rec != hipSuccess) {
      fprintf(stderr, "PE %d: copy_end_event record failed: %s\n",
              myPe_, hipGetErrorString(rec));
      throw std::runtime_error("Copy timing end event failed");
    }
    copy_timing_recorded_ = true;
  }

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
  if (UseRingExecutorProbe()) {
    static thread_local bool s_ring_announced = false;
    if (!s_ring_announced) {
      printf("PE %d: MORI_RING_EXECUTOR=1 — using Ring1DAllReduceExecutor probe\n",
             myPe_);
      s_ring_announced = true;
    }
    try {
      AllReduceConfig cfg;
      cfg.threadsPerBlock = 512;
      cfg.maxBlocks = max_blocks_;
      Ring1DAllReduceExecutor<T> exec(npes_, myPe_, cfg);
      int status = exec.Execute(input, output, total_count, stream);
      if (status != 0) {
        fprintf(stderr, "PE %d: Ring1DAllReduceExecutor failed with status %d\n",
                myPe_, status);
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      fprintf(stderr, "PE %d: Ring1DAllReduceExecutor exception: %s\n",
              myPe_, e.what());
      return false;
    }
  }
  static const bool fused = []() -> bool {
      if (const char* e = std::getenv("MORI_FULLMESH_PIPE")) {
          if (std::atoi(e) == 1) return true;
      }
      const char* e = std::getenv("MORI_PIPELINE_FUSED");
      return e != nullptr && std::atoi(e) == 1;
  }();
  return pipelined(input, output, total_count, 0, 0, stream,
                   /*external_scatter=*/!fused);
}

// ================ Async API Implementations ================

template <typename T>
bool AllreduceSdma<T>::start_async(T* input, T* output, size_t total_count, hipStream_t stream) {
  bool expected = false;
  if (!async_in_progress_.compare_exchange_strong(expected, true)) {
    printf("PE %d: Another async operation is already in progress\n", myPe_);
    return false;
  }

  async_input_ = input;
  async_output_ = output;
  async_total_count_ = total_count;
  async_stream_ = stream;
  async_start_time_ = MPI_Wtime();

  try {
    size_t elementCountPerRank = total_count / npes_;
    size_t required_output_size = elementCountPerRank * npes_ * dtype_size_;
    if (!ensure_buffer_size(output_transit_buffer_, output_transit_buffer_ptr_,
                            output_transit_buffer_size_, output_transit_buffer_obj_,
                            required_output_size, "output transit buffer")) {
      async_in_progress_ = false;
      return false;
    }

    // Step 1: SdmaReduceScatter — same as operator()
    constexpr int pack_size = packed_t<T>::P::size;
    int threads = 512;
    int packedPerRank = static_cast<int>(((total_count / npes_ + pack_size - 1) / pack_size));
    int blocks = std::min(max_blocks_, (packedPerRank + threads - 1) / threads);
    if (blocks < 1) blocks = 1;

    SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
        myPe_, npes_, input, output_transit_buffer_obj_, flagsObj_, barrierPtr_, total_count);

    // Step 2: AllGather PUT only — sends data, returns immediately
    // The wait is deferred to wait_async so the user can run GEMM on CU
    AllGatherAsyncPutKernel<T><<<1, 512, 0, stream>>>(myPe_, npes_, output_transit_buffer_obj_,
                                                      flagsObj_, barrierPtr_, total_count);

    hipError_t kernel_err = hipGetLastError();
    if (kernel_err != hipSuccess) {
      fprintf(stderr, "PE %d: Async kernel launch failed: %s\n", myPe_,
              hipGetErrorString(kernel_err));
      throw std::runtime_error("Kernel launch failed");
    }

    pipeline_scatter_gen_ += 1;  // RS does one ATOMIC_INC on qId=0
    return true;

  } catch (const std::exception& e) {
    fprintf(stderr, "PE %d: Failed to start async operation: %s\n", myPe_, e.what());
    async_in_progress_ = false;
    return false;
  }
}

template <typename T>
double AllreduceSdma<T>::wait_async(hipStream_t stream) {
  if (!async_in_progress_) {
    printf("PE %d: No async operation in progress\n", myPe_);
    return -1.0;
  }

  try {
    hipStream_t wait_stream = (stream != nullptr) ? stream : async_stream_;

    // Wait for AllGather SDMA transfers to complete + invalidate L2
    AllGatherAsyncWaitKernel<<<1, 64, 0, wait_stream>>>(myPe_, npes_, output_transit_buffer_obj_,
                                                        barrierPtr_, async_total_count_);

    // Copy result to user buffer (if enabled)
    if (copy_output_to_user_) {
      copy_output_to_user(async_output_, async_total_count_, wait_stream);
    }

    // Single synchronization at the end
    if (wait_stream != nullptr) {
      hipError_t err = hipStreamSynchronize(wait_stream);
      if (err != hipSuccess) {
        fprintf(stderr, "PE %d: Stream synchronization failed: %s\n", myPe_,
                hipGetErrorString(err));
        throw std::runtime_error("Stream synchronization failed");
      }
    } else {
      hipError_t err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        fprintf(stderr, "PE %d: Device synchronization failed: %s\n", myPe_,
                hipGetErrorString(err));
        throw std::runtime_error("Device synchronization failed");
      }
    }

    double end_time = MPI_Wtime();
    double duration = end_time - async_start_time_;

    pipeline_scatter_gen_ += 1;  // AG does one ATOMIC_INC on qId=0
    async_in_progress_ = false;
    async_input_ = nullptr;
    async_output_ = nullptr;
    async_total_count_ = 0;
    async_stream_ = nullptr;
    async_start_time_ = 0.0;

    return duration;

  } catch (const std::exception& e) {
    fprintf(stderr, "PE %d: Async wait failed: %s\n", myPe_, e.what());
    cancel_async();
    return -1.0;
  }
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

// allreduce_inplace implementation
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::allreduce_inplace(T* data, size_t total_count, hipStream_t stream) {
  bool saved = copy_output_to_user_;
  copy_output_to_user_ = true;
  bool ok = (*this)(data, data, total_count, stream);
  copy_output_to_user_ = saved;
  return ok;
}

// ---------------------------------------------------------------------------
// pipelined() — V2: figo's single-kernel SDMA pipeline.
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::pipelined(T* input, T* output, size_t total_count,
                                 size_t chunk_elems, int scatter_mode,
                                 hipStream_t stream, bool external_scatter) {
    try {
        if (total_count == 0) return true;
        if (!output_transit_buffer_) {
            fprintf(stderr, "PE %d: pipelined: output transit buffer is null\n", myPe_);
            return false;
        }

        constexpr int pack_size = packed_t<T>::P::size;
        const size_t transit_used = SdmaTransitUsedBytes<T>(total_count, npes_, dtype_size_);
        if (transit_used > output_transit_buffer_size_) {
            fprintf(stderr,
                    "PE %d: pipelined: transit need %zu B > allocated %zu B\n",
                    myPe_, transit_used, output_transit_buffer_size_);
            return false;
        }
        if (SdmaShouldZeroTransit()) {
            hipError_t zerr = stream ? hipMemsetAsync(output_transit_buffer_, 0, transit_used, stream)
                                     : hipMemset(output_transit_buffer_, 0, transit_used);
            if (zerr != hipSuccess) {
                fprintf(stderr, "PE %d: pipelined hipMemset(output transit) failed: %s\n",
                        myPe_, hipGetErrorString(zerr));
                return false;
            }
        }

        struct PipelineConfig {
            int threads;
            int cu_limit;
            int target_chunks;
            PipelineConfig() : threads(512), cu_limit(0), target_chunks(2) {
                if (const char* e = std::getenv("MORI_PIPELINE_THREADS")) {
                    int v = std::atoi(e);
                    if (v == 256 || v == 512 || v == 1024) threads = v;
                }
                if (const char* e = std::getenv("MORI_PIPELINE_CU")) {
                    int v = std::atoi(e);
                    if (v > 0) cu_limit = v;
                }
                if (const char* e = std::getenv("MORI_PIPELINE_CHUNKS")) {
                    int v = std::atoi(e);
                    if (v >= 1 && v <= 16) target_chunks = v;
                }
            }
        };
        static const PipelineConfig cfg;

        int threads = cfg.threads;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (scatter_mode == 0) {
            int comp = std::min(blocks, kMaxPipelineBlocks - 1);
            comp = std::min(comp, max_blocks_ - 1);
            if (cfg.cu_limit > 0 && cfg.cu_limit < comp)
                comp = cfg.cu_limit;
            blocks = comp + 1;
        }

        if (chunk_elems == 0) {
            if (scatter_mode == 1) {
                chunk_elems = total_count;
            } else if (direct_output_enabled_ && copy_output_to_user_) {
                // Plan B (perf_history Entry 18): β-pipeline needs multi
                // chunks to hide CU copy behind next chunk's reduce+AG.
                // User-specified default = 8. Skip the 8 MB/chunk-shard
                // minimum check (that heuristic was for SDMA AG cold-path
                // cost, not applicable to our CU AG path). MORI_DIRECT_CHUNKS
                // env can override (range 2..16).
                int target = 8;
                if (const char* e = std::getenv("MORI_DIRECT_CHUNKS")) {
                    int v = std::atoi(e);
                    if (v >= 2 && v <= 16) target = v;
                }
                chunk_elems = total_count / target;
                // Minimum sanity: at least pack_size * npes per chunk.
                size_t min_chunk = static_cast<size_t>(pack_size) * npes_;
                if (chunk_elems < min_chunk) chunk_elems = min_chunk;
            } else {
                size_t min_chunk = std::max<size_t>(
                    static_cast<size_t>(pack_size) * npes_,
                    (512ULL * 1024 / dtype_size_) * npes_);
                int kTargetChunks = cfg.target_chunks;
                constexpr size_t kMinChunkShardBytes = 8ULL * 1024 * 1024;
                const size_t shard_bytes =
                    (total_count / npes_) * dtype_size_;
                const size_t chunk_shard_bytes =
                    shard_bytes / kTargetChunks;
                if (chunk_shard_bytes >= kMinChunkShardBytes) {
                    chunk_elems = total_count / kTargetChunks;
                } else {
                    chunk_elems = total_count;
                }
                if (chunk_elems < min_chunk)
                    chunk_elems = min_chunk;
            }
        }
        size_t align = static_cast<size_t>(pack_size * npes_);
        chunk_elems = ((chunk_elems + align - 1) / align) * align;

        application::SymmMemObjPtr agDstObj = {};
        if (SdmaSeparateAgBuffer() && scatter_mode == 0) {
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    transit_used, "separate AG output buffer")) {
                return false;
            }
            agDstObj = input_transit_buffer_obj_;
            static thread_local bool s_sep_ag_announced = false;
            if (!s_sep_ag_announced) {
                printf("PE %d: MORI_SEPARATE_AG_BUFFER=1 (AG writes to separate "
                       "internal symm buffer; copy/no-copy reads from it)\n",
                       myPe_);
                s_sep_ag_announced = true;
            }
        }

        application::SymmMemObjPtr inputSymmObj = {};
        if (scatter_mode == 1) {
            size_t required_input_size = total_count * dtype_size_;
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    required_input_size, "input transit buffer")) {
                return false;
            }
            copy_input_to_transit(input, total_count, stream);
            inputSymmObj = input_transit_buffer_obj_;
            {
                hipError_t se = (stream != nullptr)
                    ? hipStreamSynchronize(stream)
                    : hipDeviceSynchronize();
                if (se != hipSuccess) {
                    fprintf(stderr,
                            "PE %d: sync before P2P kernel failed: %s\n",
                            myPe_, hipGetErrorString(se));
                    return false;
                }
            }
        }

        if (scatter_mode == 1) {
            hipError_t br =
                stream ? hipMemsetAsync(barrierPtr_, 0, sizeof(CrossPeBarrier), stream)
                       : hipMemset(barrierPtr_, 0, sizeof(CrossPeBarrier));
            if (br != hipSuccess) {
                fprintf(stderr, "PE %d: pipelined hipMemset(barrier) failed: %s\n",
                        myPe_, hipGetErrorString(br));
                return false;
            }
        } else {
            // SCATTER_MODE=0: zero only chunks_complete (cheap 4 bytes).
            // Must be done BEFORE kernel launch on same stream so kernel
            // sees chunks_complete=0 at entry.  Prevents race where compute
            // blocks increment before block 0 reads it as a baseline.
            hipError_t br = stream
                ? hipMemsetAsync(&barrierPtr_->chunks_complete, 0,
                                 sizeof(uint32_t), stream)
                : hipMemset(&barrierPtr_->chunks_complete, 0, sizeof(uint32_t));
            if (br != hipSuccess) {
                fprintf(stderr,
                        "PE %d: pipelined hipMemset(chunks_complete) failed: %s\n",
                        myPe_, hipGetErrorString(br));
                return false;
            }
        }

        const bool multi_chunk = (chunk_elems < total_count);

        const int numChunks_host = (chunk_elems >= total_count) ? 1
            : static_cast<int>((total_count + chunk_elems - 1) / chunk_elems);

        uint64_t scatter_base = pipeline_scatter_gen_;
        uint64_t ag_base      = pipeline_ag_gen_;
        uint64_t reduce_complete_base = pipeline_reduce_gen_;
        const bool multi_q_ag = []() -> bool {
            const char* e = std::getenv("MORI_MULTI_Q_AG");
            return e != nullptr && std::atoi(e) == 1;
        }();
        if (multi_q_ag && pipeline_ag_gen_by_q_d_ != nullptr) {
            hipError_t gen_copy = hipMemcpyAsync(
                pipeline_ag_gen_by_q_d_, pipeline_ag_gen_by_q_.data(),
                kMaxTrackedSdmaQueues * sizeof(uint64_t),
                hipMemcpyHostToDevice, stream);
            if (gen_copy != hipSuccess) {
                fprintf(stderr,
                        "PE %d: pipelined multi-q AG generation copy failed: %s\n",
                        myPe_, hipGetErrorString(gen_copy));
                return false;
            }
        }

        // Phase timing buffer (nullptr unless instrumentation enabled).
        // Block 0 thread 0 writes __builtin_amdgcn_s_memtime() at each phase.
        uint64_t* phase_ts_ptr = phase_timing_enabled_ ? phase_ts_d_ : nullptr;
        last_num_chunks_ = numChunks_host;

        // Stage 1 E' prototype: pass post_ag_flag to kernel when enabled.
        // Reset the flag to 0 on the same stream before kernel launch so the
        // kernel's compute blocks observe 0 initially and only release when
        // block 0 stores 1 after AG wait done.
        uint32_t* post_ag_flag_ptr = nullptr;
        if (post_ag_wait_enabled_ && post_ag_flag_d_ != nullptr) {
            hipError_t post_ag_reset =
                hipMemsetAsync(post_ag_flag_d_, 0, sizeof(uint32_t), stream);
            if (post_ag_reset != hipSuccess) {
                fprintf(stderr,
                        "PE %d: post-AG wait flag reset failed: %s\n",
                        myPe_, hipGetErrorString(post_ag_reset));
                return false;
            }
            post_ag_flag_ptr = post_ag_flag_d_;
        }

        // Plan A (2-kernel, XGMI-pull AG, strict per user spec — see
        // perf_history Entry 19): when direct_output is on AND we're on
        // the copy_output_to_user MULTI_CHUNK path, split the AR into:
        //   Kernel 1: ScatterSdmaOnlyKernel (SDMA scatter only, no CU).
        //   Kernel 2: PipelinedXGMIPullKernel (CU reduce → cross-PE
        //              barrier → CU XGMI pull + direct write user_output;
        //              no SDMA AG, no external hipMemcpyAsync).
        // Replaces Plan B (R/C-group β partition, CU push AG + CU copy)
        // which failed by +1.16 ms wall (Entry 18) — CU-GEMM contention
        // and CU XGMI push is slower than SDMA AG.
        const bool use_plan_a = direct_output_enabled_
            && copy_output_to_user_ && multi_chunk;
        const bool use_fullmesh_chan = []() -> bool {
            const char* e = std::getenv("MORI_FULLMESH_CHAN");
            return e != nullptr && std::atoi(e) == 1;
        }() && copy_output_to_user_ && multi_chunk && scatter_mode == 0;
        const bool request_oneshot_direct = []() -> bool {
            const char* e = std::getenv("MORI_ONESHOT_DIRECT");
            return e != nullptr && std::atoi(e) == 1;
        }();
        const bool use_chunked_direct = []() -> bool {
            const char* e = std::getenv("MORI_CHUNKED_DIRECT");
            return e != nullptr && std::atoi(e) == 1;
        }() && copy_output_to_user_ && scatter_mode == 0;
        const bool use_multilane_direct = []() -> bool {
            const char* e = std::getenv("MORI_MULTILANE_DIRECT");
            return e != nullptr && std::atoi(e) == 1;
        }() && copy_output_to_user_ && scatter_mode == 0;
        const bool use_ring_shard_direct = []() -> bool {
            const char* e = std::getenv("MORI_RING_SHARD_DIRECT");
            return e != nullptr && std::atoi(e) == 1;
        }() && copy_output_to_user_ && scatter_mode == 0;
        const bool allow_failed_oneshot_direct = []() -> bool {
            const char* e = std::getenv("MORI_ALLOW_FAILED_ONESHOT_DIRECT");
            return e != nullptr && std::atoi(e) == 1;
        }();
        if (request_oneshot_direct && !allow_failed_oneshot_direct) {
            static thread_local bool s_oneshot_disabled_announced = false;
            if (!s_oneshot_disabled_announced) {
                printf("PE %d: MORI_ONESHOT_DIRECT=1 ignored: Entry 70 showed "
                       "full-peer-read direct output has seq_ar ~28.5 ms. Set "
                       "MORI_ALLOW_FAILED_ONESHOT_DIRECT=1 only for targeted "
                       "debugging.\n",
                       myPe_);
                s_oneshot_disabled_announced = true;
            }
        }
        const bool use_oneshot_direct = request_oneshot_direct
            && allow_failed_oneshot_direct && copy_output_to_user_ && scatter_mode == 0;
        const bool request_sdma_ag_copy_pipe = []() -> bool {
            const char* e = std::getenv("MORI_SDMA_AG_COPY_PIPE");
            return e != nullptr && std::atoi(e) == 1;
        }();
        const bool allow_failed_copy_pipe = []() -> bool {
            const char* e = std::getenv("MORI_ALLOW_FAILED_COPY_PIPE");
            return e != nullptr && std::atoi(e) == 1;
        }();
        if (request_sdma_ag_copy_pipe && !allow_failed_copy_pipe) {
            static thread_local bool s_pipe_disabled_announced = false;
            if (!s_pipe_disabled_announced) {
                printf("PE %d: MORI_SDMA_AG_COPY_PIPE=1 ignored: Entry 56 "
                       "closed this path after K1 scatter hung at chunk=3 "
                       "expected=4 got=3. Set MORI_ALLOW_FAILED_COPY_PIPE=1 "
                       "only for targeted debugging.\n",
                       myPe_);
                s_pipe_disabled_announced = true;
            }
        }
        const bool use_sdma_ag_copy_pipe = request_sdma_ag_copy_pipe
            && allow_failed_copy_pipe && copy_output_to_user_ && multi_chunk
            && scatter_mode == 0;
        bool skip_external_copy =
            use_plan_a || use_fullmesh_chan || use_sdma_ag_copy_pipe ||
            use_oneshot_direct || use_chunked_direct || use_multilane_direct ||
            use_ring_shard_direct;

        if (use_ring_shard_direct) {
            const size_t required_input_size = total_count * dtype_size_;
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    required_input_size, "ring shard accum buffer")) {
                return false;
            }
            copy_input_to_transit(input, total_count, stream);
            const int rs_threads = threads;
            int rs_blocks = 1;
            if (const char* e = std::getenv("MORI_RING_SHARD_BLOCKS")) {
                int v = std::atoi(e);
                if (v > 0) rs_blocks = v;
            }
            if (rs_blocks > max_blocks_) rs_blocks = max_blocks_;
            static thread_local bool s_rs_announced = false;
            if (!s_rs_announced) {
                printf("PE %d: MORI_RING_SHARD_DIRECT=1 — dedicated ring/shard "
                       "direct-output probe; blocks=%d threads=%d\n",
                       myPe_, rs_blocks, rs_threads);
                s_rs_announced = true;
            }
            RingShardDirectKernel<T><<<rs_blocks, rs_threads, 0, stream>>>(
                myPe_, npes_, input_transit_buffer_obj_, output_transit_buffer_obj_,
                output, total_count, pipeline_scatter_gen_, pipeline_ag_gen_, phase_ts_ptr);
            hipError_t final_copy = stream
                ? hipMemcpyAsync(output, input_transit_buffer_, total_count * dtype_size_,
                                 hipMemcpyDeviceToDevice, stream)
                : hipMemcpy(output, input_transit_buffer_, total_count * dtype_size_,
                            hipMemcpyDeviceToDevice);
            if (final_copy != hipSuccess) {
                fprintf(stderr, "PE %d: ring shard final output copy failed: %s\n",
                        myPe_, hipGetErrorString(final_copy));
                return false;
            }
        } else if (use_multilane_direct) {
            const size_t required_input_size = total_count * dtype_size_;
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    required_input_size, "multilane direct input buffer")) {
                return false;
            }
            copy_input_to_transit(input, total_count, stream);
            const DirectLaneSpec lanes =
                ParseDirectLaneSpec(std::getenv("MORI_MULTILANE_DIRECT_LANES"));
            int blocks_per_lane = 8;
            if (const char* e = std::getenv("MORI_MULTILANE_BLOCKS_PER_LANE")) {
                int v = std::atoi(e);
                if (v > 0) blocks_per_lane = v;
            }
            const int lane_count = lanes.forward + lanes.reverse;
            const int ml_threads = threads;
            int ml_blocks = lane_count * blocks_per_lane;
            if (ml_blocks < lane_count) ml_blocks = lane_count;
            if (ml_blocks > max_blocks_) ml_blocks = max_blocks_;
            static thread_local bool s_ml_announced = false;
            if (!s_ml_announced) {
                printf("PE %d: MORI_MULTILANE_DIRECT=1 — direct-output "
                       "multi-lane P2P-read allreduce; lanes=%dF/%dR blocks=%d threads=%d\n",
                       myPe_, lanes.forward, lanes.reverse, ml_blocks, ml_threads);
                s_ml_announced = true;
            }
            MultiLaneDirectOutputKernel<T><<<ml_blocks, ml_threads, 0, stream>>>(
                myPe_, npes_, input_transit_buffer_obj_, flagsObj_, output,
                total_count, lanes.forward, lanes.reverse, blocks_per_lane,
                pipeline_reduce_gen_, phase_ts_ptr);
        } else if (use_chunked_direct) {
            const size_t required_input_size = total_count * dtype_size_;
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    required_input_size, "chunked direct input buffer")) {
                return false;
            }
            copy_input_to_transit(input, total_count, stream);
            const int cd_threads = threads;
            int cd_blocks = std::min(max_blocks_,
                                     (packedPerRank + cd_threads - 1) / cd_threads);
            if (cd_blocks < 1) cd_blocks = 1;
            if (cfg.cu_limit > 0 && cfg.cu_limit < cd_blocks) cd_blocks = cfg.cu_limit;
            static thread_local bool s_cd_announced = false;
            if (!s_cd_announced) {
                printf("PE %d: MORI_CHUNKED_DIRECT=1 — chunked direct-output "
                       "P2P-read allreduce; chunks=%d blocks=%d threads=%d\n",
                       myPe_, numChunks_host, cd_blocks, cd_threads);
                s_cd_announced = true;
            }
            for (int c = 0; c < numChunks_host; ++c) {
                ChunkedDirectOutputKernel<T><<<cd_blocks, cd_threads, 0, stream>>>(
                    myPe_, npes_, input_transit_buffer_obj_, flagsObj_, output,
                    total_count, chunk_elems, c, pipeline_reduce_gen_, phase_ts_ptr);
            }
        } else if (use_oneshot_direct) {
            const size_t required_input_size = total_count * dtype_size_;
            if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                    input_transit_buffer_size_, input_transit_buffer_obj_,
                                    required_input_size, "oneshot direct input buffer")) {
                return false;
            }
            copy_input_to_transit(input, total_count, stream);
            const int od_threads = threads;
            int od_blocks = std::min(max_blocks_,
                                     (packedPerRank + od_threads - 1) / od_threads);
            if (od_blocks < 1) od_blocks = 1;
            if (cfg.cu_limit > 0 && cfg.cu_limit < od_blocks) od_blocks = cfg.cu_limit;
            static thread_local bool s_od_announced = false;
            if (!s_od_announced) {
                printf("PE %d: MORI_ONESHOT_DIRECT=1 — one-kernel direct-output "
                       "P2P-read allreduce; blocks=%d threads=%d\n",
                       myPe_, od_blocks, od_threads);
                s_od_announced = true;
            }
            OneShotDirectOutputKernel<T><<<od_blocks, od_threads, 0, stream>>>(
                myPe_, npes_, input_transit_buffer_obj_, flagsObj_, output,
                total_count, pipeline_reduce_gen_, phase_ts_ptr);
        } else if (use_sdma_ag_copy_pipe) {
            if (!agDstObj.IsValid()) {
                if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                        input_transit_buffer_size_, input_transit_buffer_obj_,
                                        transit_used, "SDMA AG copy-pipe buffer")) {
                    return false;
                }
                agDstObj = input_transit_buffer_obj_;
            }
            hipError_t br_pipe = stream
                ? hipMemsetAsync(barrierPtr_, 0, sizeof(CrossPeBarrier), stream)
                : hipMemset(barrierPtr_, 0, sizeof(CrossPeBarrier));
            if (br_pipe != hipSuccess) {
                fprintf(stderr, "PE %d: SDMA AG copy-pipe barrier reset failed: %s\n",
                        myPe_, hipGetErrorString(br_pipe));
                return false;
            }
            int pipe_comp = blocks - 1;
            int pipe_nR = pipe_comp / 2;
            if (const char* e = std::getenv("MORI_SDMA_AG_COPY_NR")) {
                int v = std::atoi(e);
                if (v > 0) pipe_nR = v;
            }
            if (pipe_nR < 1) pipe_nR = 1;
            if (pipe_nR > pipe_comp - 1) pipe_nR = pipe_comp - 1;
            static thread_local bool s_pipe_announced = false;
            if (!s_pipe_announced) {
                printf("PE %d: MORI_SDMA_AG_COPY_PIPE=1 — K1 scatter + K2 "
                       "R-reduce / SDMA-AG / C-copy pipeline; chunks=%d, "
                       "blocks=%d, nR=%d, nC=%d\n",
                       myPe_, numChunks_host, blocks, pipe_nR, pipe_comp - pipe_nR);
                s_pipe_announced = true;
            }
            ScatterSdmaOnlyWaitEachChunkKernel<T><<<1, 512, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_,
                total_count, chunk_elems, scatter_base);
            PipelinedSdmaAgCopyKernel<T><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_, agDstObj,
                barrierPtr_, output, total_count, chunk_elems, ag_base, pipe_nR);
        } else if (use_fullmesh_chan) {
            if (!agDstObj.IsValid()) {
                if (!ensure_buffer_size(input_transit_buffer_, input_transit_buffer_ptr_,
                                        input_transit_buffer_size_, input_transit_buffer_obj_,
                                        transit_used, "fullmesh channel AG buffer")) {
                    return false;
                }
                agDstObj = input_transit_buffer_obj_;
            }
            hipError_t br_full = stream
                ? hipMemsetAsync(barrierPtr_, 0, sizeof(CrossPeBarrier), stream)
                : hipMemset(barrierPtr_, 0, sizeof(CrossPeBarrier));
            if (br_full != hipSuccess) {
                fprintf(stderr, "PE %d: fullmesh channel barrier reset failed: %s\n",
                        myPe_, hipGetErrorString(br_full));
                return false;
            }
            static thread_local bool s_fullmesh_chan_announced = false;
            if (!s_fullmesh_chan_announced) {
                printf("PE %d: MORI_FULLMESH_CHAN=1 — chunked scatter/reduce/AG/copy "
                       "kernel; chunks=%d, blocks=%d, threads=%d\n",
                       myPe_, numChunks_host, blocks, threads);
                s_fullmesh_chan_announced = true;
            }
            FullMeshChannelizedAllReduceKernel<T><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_, agDstObj,
                barrierPtr_, output, total_count, chunk_elems, scatter_base, ag_base);
        } else if (use_plan_a) {
            int plan_a_comp = blocks - 1;
            // Plan A v2 uses CU for both reduce and AG-pull. Do not default
            // to all 160 CUs: that starves the overlapped GEMM. Keep a
            // tunable cap so we can sweep the AR/GEMM balance.
            int plan_a_cu = 96;
            if (const char* e = std::getenv("MORI_PLAN_A_CU")) {
                int v = std::atoi(e);
                if (v >= 2) plan_a_cu = v;
            }
            if (plan_a_cu < plan_a_comp) plan_a_comp = plan_a_cu;
            // Need at least 1 R block and npes A blocks (one per output slot).
            if (plan_a_comp < npes_ + 1) plan_a_comp = npes_ + 1;

            int plan_a_nR = plan_a_comp / 3;  // default R:A ~= 1:2
            if (const char* e = std::getenv("MORI_PLAN_A_NR")) {
                int v = std::atoi(e);
                if (v > 0) plan_a_nR = v;
            }
            if (plan_a_nR < 1) plan_a_nR = 1;
            if (plan_a_nR > plan_a_comp - npes_) plan_a_nR = plan_a_comp - npes_;
            const int plan_a_nA = plan_a_comp - plan_a_nR;
            const int plan_a_blocks = plan_a_comp + 1;

            // Plan A needs both chunks_complete AND ag_sync reset to 0.
            // ag_sync is block 0 → compute blocks signal for cross-PE
            // barrier completion. Expand memset to cover both fields.
            // (SCATTER_MODE=0 else-branch only zeroed chunks_complete.)
            hipError_t br2 = stream
                ? hipMemsetAsync(&barrierPtr_->ag_sync, 0,
                                 sizeof(uint32_t), stream)
                : hipMemset(&barrierPtr_->ag_sync, 0, sizeof(uint32_t));
            if (br2 != hipSuccess) {
                fprintf(stderr,
                        "PE %d: Plan A hipMemset(ag_sync) failed: %s\n",
                        myPe_, hipGetErrorString(br2));
                return false;
            }

            static thread_local bool s_plan_a_announced = false;
            if (!s_plan_a_announced) {
                printf("PE %d: Plan A active — ScatterSdmaOnlyKernel + "
                       "PipelinedXGMIPullKernel; chunks=%d, blocks=%d, "
                       "threads=%d, chunk_elems=%zu, nR=%d, nA=%d "
                       "(MORI_PLAN_A_CU=%d)\n",
                       myPe_, numChunks_host, plan_a_blocks, threads,
                       static_cast<size_t>(chunk_elems), plan_a_nR, plan_a_nA,
                       plan_a_comp);
                s_plan_a_announced = true;
            }
            ScatterSdmaOnlyKernel<T><<<1, 512, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_,
                total_count, chunk_elems, scatter_base);
            PipelinedXGMIPullKernel<T><<<plan_a_blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, flagsObj_, barrierPtr_,
                output,
                total_count, chunk_elems, scatter_base, ag_base,
                reduce_complete_base, plan_a_nR, phase_ts_ptr);
        } else if (scatter_mode == 1) {
            PipelinedAllReduceSdmaKernel<T, 1><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, agDstObj, flagsObj_,
                barrierPtr_, inputSymmObj, total_count, chunk_elems,
                scatter_base, ag_base, pipeline_ag_gen_by_q_d_,
                reduce_complete_base, phase_ts_ptr, multi_q_ag,
                post_ag_flag_ptr);
        } else if (external_scatter) {
            ScatterSdmaOnlyKernel<T><<<1, 512, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_,
                total_count, chunk_elems, scatter_base);
            if (multi_chunk) {
                PipelinedAllReduceSdmaKernel<T, 0, true, true>
                    <<<blocks, threads, 0, stream>>>(
                    myPe_, npes_, input,
                    output_transit_buffer_obj_, agDstObj, flagsObj_,
                    barrierPtr_, application::SymmMemObjPtr{},
                    total_count, chunk_elems, scatter_base, ag_base,
                    pipeline_ag_gen_by_q_d_, reduce_complete_base, phase_ts_ptr,
                    multi_q_ag,
                    post_ag_flag_ptr);
            } else {
                PipelinedAllReduceSdmaKernel<T, 0, false, true>
                    <<<blocks, threads, 0, stream>>>(
                    myPe_, npes_, input,
                    output_transit_buffer_obj_, agDstObj, flagsObj_,
                    barrierPtr_, application::SymmMemObjPtr{},
                    total_count, chunk_elems, scatter_base, ag_base,
                    pipeline_ag_gen_by_q_d_, reduce_complete_base, phase_ts_ptr,
                    multi_q_ag,
                    post_ag_flag_ptr);
            }
        } else if (multi_chunk) {
            PipelinedAllReduceSdmaKernel<T, 0, true><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, agDstObj, flagsObj_,
                barrierPtr_, application::SymmMemObjPtr{}, total_count, chunk_elems,
                scatter_base, ag_base, pipeline_ag_gen_by_q_d_,
                reduce_complete_base, phase_ts_ptr, multi_q_ag,
                post_ag_flag_ptr);
        } else {
            PipelinedAllReduceSdmaKernel<T, 0, false><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, agDstObj, flagsObj_,
                barrierPtr_, application::SymmMemObjPtr{}, total_count, chunk_elems,
                scatter_base, ag_base, pipeline_ag_gen_by_q_d_,
                reduce_complete_base, phase_ts_ptr, multi_q_ag,
                post_ag_flag_ptr);
        }

        if (use_oneshot_direct || use_multilane_direct) {
            pipeline_reduce_gen_ += 1;  // ready flag in flagsObj_
        } else if (use_ring_shard_direct) {
            pipeline_scatter_gen_ += static_cast<uint64_t>(npes_ - 1);
            pipeline_ag_gen_ += static_cast<uint64_t>(npes_ - 1);
        } else if (use_chunked_direct) {
            pipeline_reduce_gen_ += static_cast<uint64_t>(numChunks_host);
        } else {
            pipeline_scatter_gen_ += numChunks_host;   // scatter SDMA (qId=0)
            // Plan A does NOT use SDMA AG signal (qId=1); AG is replaced by CU
            // XGMI pull which is synchronized via CrossPeBarrier.ag_sync (per
            // launch). Still advance pipeline_ag_gen_ so baseline path's
            // sub-launches (if interleaved) keep consistent counters.
            pipeline_ag_gen_      += numChunks_host;   // AG (qId=1, unused in Plan A)
            if (multi_q_ag && sdma_num_queue_ > 1) {
                const uint32_t usable_q = std::min<uint32_t>(
                    sdma_num_queue_ - 1,
                    static_cast<uint32_t>(kMaxTrackedSdmaQueues - 1));
                for (int c = 0; c < numChunks_host; c++) {
                    const uint32_t q = 1u + static_cast<uint32_t>(c) % usable_q;
                    pipeline_ag_gen_by_q_[q]++;
                }
            } else {
                pipeline_ag_gen_by_q_[1] += static_cast<uint64_t>(numChunks_host);
            }
            pipeline_reduce_gen_  += numChunks_host;   // reduce_complete via flags
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: PipelinedAllReduce launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        if (copy_output_to_user_ && !skip_external_copy) {
            copy_output_to_user(output, total_count, stream);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: PipelinedAllReduce failed: %s\n", myPe_, e.what());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::resetFlags() {
  if (flags_) {
    memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
  }
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
