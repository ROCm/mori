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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "mori/collective/allreduce/twoshot_sdma_async_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp"
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
    hipMemcpy(&gpuSig, &(output_transit_buffer_obj_.gpu->signalPtrs),
              sizeof(HSAuint64*), hipMemcpyDeviceToHost);
    hipMemcpy(&gpuNumQ, &(output_transit_buffer_obj_.gpu->sdmaNumQueue),
              sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (gpuSig && gpuNumQ > 0) {
      size_t sigSize = static_cast<size_t>(npes_) * gpuNumQ * sizeof(HSAuint64);
      hipMemset(gpuSig, 0, sigSize);
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

  {
    HSAuint64* gpuSig = nullptr;
    uint32_t   gpuNumQ = 0;
    hipMemcpy(&gpuSig, &(buffer_obj.gpu->signalPtrs),
              sizeof(HSAuint64*), hipMemcpyDeviceToHost);
    hipMemcpy(&gpuNumQ, &(buffer_obj.gpu->sdmaNumQueue),
              sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (gpuSig && gpuNumQ > 0) {
      size_t sigSize = static_cast<size_t>(npes_) * gpuNumQ * sizeof(HSAuint64);
      hipMemset(gpuSig, 0, sigSize);
      pipeline_scatter_gen_ = 0;
      pipeline_ag_gen_ = 0;
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
  return pipelined(input, output, total_count, 0, 0, stream);
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
                                 hipStream_t stream) {
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

        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (scatter_mode == 0) {
            int comp = std::min(blocks, kMaxPipelineBlocks - 1);
            blocks = comp + 1;
        }

        if (chunk_elems == 0) {
            size_t min_chunk = std::max<size_t>(
                static_cast<size_t>(pack_size) * npes_,
                (512ULL * 1024 / dtype_size_) * npes_);
            if (scatter_mode == 1) {
                chunk_elems = total_count;
            } else {
                constexpr int    kTargetChunks      = 2;
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
            }
            if (chunk_elems < min_chunk)
                chunk_elems = min_chunk;
        }
        size_t align = static_cast<size_t>(pack_size * npes_);
        chunk_elems = ((chunk_elems + align - 1) / align) * align;

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
        }

        const bool multi_chunk = (chunk_elems < total_count);

        const int numChunks_host = (chunk_elems >= total_count) ? 1
            : static_cast<int>((total_count + chunk_elems - 1) / chunk_elems);

        uint64_t scatter_base = pipeline_scatter_gen_;
        uint64_t ag_base      = pipeline_ag_gen_;

        if (scatter_mode == 1) {
            PipelinedAllReduceSdmaKernel<T, 1><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, flagsObj_,
                barrierPtr_, inputSymmObj, total_count, chunk_elems,
                scatter_base, ag_base);
        } else if (multi_chunk) {
            PipelinedAllReduceSdmaKernel<T, 0, true><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, flagsObj_,
                barrierPtr_, application::SymmMemObjPtr{}, total_count, chunk_elems,
                scatter_base, ag_base);
        } else {
            PipelinedAllReduceSdmaKernel<T, 0, false><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input,
                output_transit_buffer_obj_, flagsObj_,
                barrierPtr_, application::SymmMemObjPtr{}, total_count, chunk_elems,
                scatter_base, ag_base);
        }

        pipeline_scatter_gen_ += numChunks_host;
        pipeline_ag_gen_      += numChunks_host;

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: PipelinedAllReduce launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        if (copy_output_to_user_) {
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
