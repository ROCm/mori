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
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_async_kernel.hpp"
#include "mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace mori {
namespace collective {

namespace {

inline bool MoriSdmaVerbose() {
    const char* e = std::getenv("MORI_SDMA_VERBOSE");
    return e && e[0] == '1' && e[1] == '\0';
}

// Bytes of output transit touched by SdmaReduceScatter/AllGather for this
// elementCount (must match twoshot_sdma_kernel.hpp elementCountPerRank).
template <typename T>
size_t SdmaTransitUsedBytes(size_t total_count, int npes, size_t dtype_size) {
    constexpr int pack_size = packed_t<T>::P::size;
    const size_t element_count_per_rank =
        ((total_count / static_cast<size_t>(npes) + static_cast<size_t>(pack_size) - 1U) /
         static_cast<size_t>(pack_size)) *
        static_cast<size_t>(pack_size);
    return element_count_per_rank * static_cast<size_t>(npes) * dtype_size;
}

// Transit zeroing is off by default: every slot is overwritten (SDMA scatter +
// Phase 2.5 CU copy) before the reduce reads it, so zeroing is redundant.
// Zeroing also races with remote PEs' SDMA scatter when callers don't place a
// cross-PE barrier before each AllReduce call.
// Set MORI_SDMA_ZERO_TRANSIT=1 to re-enable (useful for debugging dropped puts).
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
    : AllreduceSdma(myPe, npes, 0, transit_buffer_size,
                    copy_output_to_user, false) {
}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes,
                                size_t input_buffer_size,
                                size_t output_buffer_size,
                                bool copy_output_to_user,
                                bool /*use_graph_mode*/)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      max_blocks_(getDeviceMaxBlocks()),
      flags_(nullptr, ShmemDeleter()),
      barrierPtr_(nullptr),
      barrierMem_(nullptr, ShmemDeleter()),
      input_transit_buffer_(nullptr),
      input_transit_buffer_size_(input_buffer_size),
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
    if (!flagsObj_.IsValid())
        throw std::runtime_error("Failed to get valid flags memory object");

    // 2. Allocate CrossPeBarrier (flag + ag_sync + block_done[] for pipeline)
    size_t barrierSize = sizeof(CrossPeBarrier);
    void* bMem = shmem::ShmemMalloc(barrierSize);
    if (!bMem) throw std::runtime_error("Failed to allocate barrier memory");
    barrierMem_.reset(bMem);
    barrierPtr_ = reinterpret_cast<CrossPeBarrier*>(bMem);
    hipError_t me = hipMemset(bMem, 0, barrierSize);
    if (me != hipSuccess)
        throw std::runtime_error("Failed to zero-init barrier memory");

    // 3. Allocate output transit buffer (gather + reduce + allgather)
    output_transit_buffer_ = shmem::ShmemMalloc(output_transit_buffer_size_);
    if (!output_transit_buffer_)
        throw std::runtime_error("Failed to allocate output transit buffer");
    output_transit_buffer_ptr_.reset(output_transit_buffer_);

    output_transit_buffer_obj_ =
        shmem::ShmemSymmetricRegister(output_transit_buffer_,
                                      output_transit_buffer_size_);
    if (!output_transit_buffer_obj_.IsValid())
        throw std::runtime_error("Failed to register output transit buffer");

    if (input_transit_buffer_size_ > 0) {
        input_transit_buffer_ = shmem::ShmemMalloc(input_transit_buffer_size_);
        if (!input_transit_buffer_)
            throw std::runtime_error("Failed to allocate input transit buffer");
        input_transit_buffer_ptr_.reset(input_transit_buffer_);
        input_transit_buffer_obj_ =
            shmem::ShmemSymmetricRegister(input_transit_buffer_,
                                          input_transit_buffer_size_);
        if (!input_transit_buffer_obj_.IsValid())
            throw std::runtime_error("Failed to register input transit buffer");
    }

    if (myPe_ == 0) {
        if (MoriSdmaVerbose()) {
            printf("AllreduceSdma initialized: PE %d/%d max_blocks=%d\n",
                   myPe_, npes_, max_blocks_);
            printf("  flags %zu B @%p  barrier %zu B @%p\n",
                   flagsSize, flags_.get(), barrierSize, bMem);
            printf("  output %.2f MB @%p\n",
                   output_transit_buffer_size_ / (1024.0 * 1024.0),
                   output_transit_buffer_);
            if (input_transit_buffer_size_ > 0) {
                printf("  input  %.2f MB @%p\n",
                       input_transit_buffer_size_ / (1024.0 * 1024.0),
                       input_transit_buffer_);
            }
        } else {
            printf("AllreduceSdma: %d PEs  out %.0f MiB", npes_,
                   output_transit_buffer_size_ / (1024.0 * 1024.0));
            if (input_transit_buffer_size_ > 0) {
                printf("  in %.0f MiB",
                       input_transit_buffer_size_ / (1024.0 * 1024.0));
            }
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
    if (async_in_progress_) {
        cancel_async();
    }
    (void)flags_;
}

// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::ensure_buffer_size(void*& buffer,
                                         std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                                         size_t& current_size,
                                         application::SymmMemObjPtr& buffer_obj,
                                         size_t required_size,
                                         const char* buffer_name) {
    if (required_size <= current_size) {
        return true;
    }

    if (MoriSdmaVerbose()) {
        printf("PE %d: %s too small: need %.2f MB, have %.2f MB\n",
               myPe_, buffer_name,
               required_size / (1024.0 * 1024.0),
               current_size / (1024.0 * 1024.0));
    }

    // First release the old one
    buffer_ptr.reset();

    // Allocate new one
    current_size = required_size;
    buffer = shmem::ShmemMalloc(current_size);
    if (buffer == nullptr) {
        fprintf(stderr, "PE %d: Failed to reallocate %s of size %.2f MB\n",
                myPe_, buffer_name, current_size / (1024.0 * 1024.0));
        return false;
    }
    buffer_ptr.reset(buffer);

    // Re-register
    buffer_obj = shmem::ShmemSymmetricRegister(buffer, current_size);
    if (!buffer_obj.IsValid()) {
        fprintf(stderr, "PE %d: Failed to re-register %s\n", myPe_, buffer_name);
        return false;
    }

    if (MoriSdmaVerbose()) {
        printf("PE %d: %s -> %.2f MB\n",
               myPe_, buffer_name, current_size / (1024.0 * 1024.0));
    }
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
        err = hipMemcpyAsync(input_transit_buffer_, input, input_bytes,
                           hipMemcpyDeviceToDevice, stream);
    } else {
        err = hipMemcpy(input_transit_buffer_, input, input_bytes,
                       hipMemcpyDeviceToDevice);
    }

    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: Failed to copy input to transit buffer: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Input copy failed");
    }
}

// copy_output_to_user implementation
// For AllReduce: output is total_count elements (same size as input, NOT npes * total_count)
template <typename T>
void AllreduceSdma<T>::copy_output_to_user(T* output, size_t total_count, hipStream_t stream) {
    size_t bytes = total_count * dtype_size_;
    if (!output)  throw std::runtime_error("Output pointer is null");
    if (!output_transit_buffer_)
        throw std::runtime_error("Output transit buffer is null");

    hipError_t err = stream
        ? hipMemcpyAsync(output, output_transit_buffer_, bytes,
                         hipMemcpyDeviceToDevice, stream)
        : hipMemcpy(output, output_transit_buffer_, bytes,
                    hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: copy_output_to_user failed: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Output copy failed");
    }
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    try {
        if (total_count == 0) return true;
        if (!output_transit_buffer_) {
            fprintf(stderr, "PE %d: operator(): output transit buffer is null\n", myPe_);
            return false;
        }

        constexpr int pack_size = packed_t<T>::P::size;
        const size_t transit_used = SdmaTransitUsedBytes<T>(total_count, npes_, dtype_size_);
        if (transit_used > output_transit_buffer_size_) {
            fprintf(stderr,
                    "PE %d: operator(): transit need %zu B > allocated %zu B\n",
                    myPe_, transit_used, output_transit_buffer_size_);
            return false;
        }
        if (SdmaShouldZeroTransit()) {
            hipError_t zerr = stream ? hipMemsetAsync(output_transit_buffer_, 0, transit_used, stream)
                                     : hipMemset(output_transit_buffer_, 0, transit_used);
            if (zerr != hipSuccess) {
                fprintf(stderr, "PE %d: hipMemset(output transit) failed: %s\n",
                        myPe_, hipGetErrorString(zerr));
                return false;
            }
        }

        // Step 1: SdmaReduceScatter — SDMA scatter + local reduce
        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;

        SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: SdmaReduceScatter launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Retire ReduceScatter (CU reduce + fence) before AllGather SDMA reads transit.
        err = stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr,
                    "PE %d: sync after ReduceScatter failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 2: AllGather via SDMA
        AllGatherSdmaKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: AllGather launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 3: Copy result to user buffer
        if (copy_output_to_user_) {
            copy_output_to_user(output, total_count, stream);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: AllReduce failed: %s\n", myPe_, e.what());
        return false;
    }
    return true;
}

// ================ Pipelined AllReduce ================

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
            // Block 0 = management; remaining blocks = compute (reduce).
            // Use up to kMaxPipelineBlocks-1 for large tensors (was capped at 32).
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
                // Auto-select chunk count for the 3-stage pipeline
                // scatter(i) | reduce(i-1) | AG(i-2).
                // Target 4 chunks: fills the 3-deep pipeline + 1 steady-state step.
                // Remote-read AG uses inbound XGMI (scatter uses outbound),
                // enabling true bidirectional overlap.  Per-chunk shard >= 2 MB
                // keeps the per-chunk sync overhead (~15-25us) well amortised.
                constexpr size_t kMinShardBytes = 2ULL * 1024 * 1024;
                constexpr int    kTargetChunks  = 4;
                const size_t shard_bytes =
                    (total_count / npes_) * dtype_size_;
                const size_t chunk_shard_bytes =
                    shard_bytes / kTargetChunks;
                if (chunk_shard_bytes >= kMinShardBytes) {
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

        // SDMA queue completion counters live on the symmetric object (not shmem flags).
        // Zero them every launch so compute CTAs' baseline reads cannot race block 0's
        // first scatter or inherit mismatched totals from a prior collective.
        {
            application::SymmMemObj* cpuMo = output_transit_buffer_obj_.cpu;
            if (cpuMo && cpuMo->signalPtrs) {
                const uint32_t nq = cpuMo->sdmaNumQueue ? cpuMo->sdmaNumQueue : 8u;
                const size_t sig_bytes =
                    static_cast<size_t>(npes_) * static_cast<size_t>(nq) * sizeof(uint64_t);
                hipError_t sg =
                    stream ? hipMemsetAsync(cpuMo->signalPtrs, 0, sig_bytes, stream)
                           : hipMemset(cpuMo->signalPtrs, 0, sig_bytes);
                if (sg != hipSuccess) {
                    fprintf(stderr, "PE %d: pipelined hipMemset(SDMA signals) failed: %s\n",
                            myPe_, hipGetErrorString(sg));
                    return false;
                }
            }
        }
        {
            hipError_t br =
                stream ? hipMemsetAsync(barrierPtr_, 0, sizeof(CrossPeBarrier), stream)
                       : hipMemset(barrierPtr_, 0, sizeof(CrossPeBarrier));
            if (br != hipSuccess) {
                fprintf(stderr, "PE %d: pipelined hipMemset(barrier) failed: %s\n",
                        myPe_, hipGetErrorString(br));
                return false;
            }
        }

        if (scatter_mode == 1) {
            PipelinedAllReduceSdmaKernel<T, 1><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_, flagsObj_,
                barrierPtr_, inputSymmObj, total_count, chunk_elems);
        } else {
            PipelinedAllReduceSdmaKernel<T, 0><<<blocks, threads, 0, stream>>>(
                myPe_, npes_, input, output_transit_buffer_obj_, flagsObj_,
                barrierPtr_, application::SymmMemObjPtr{}, total_count, chunk_elems);
        }

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
        const size_t transit_used = SdmaTransitUsedBytes<T>(total_count, npes_, dtype_size_);
        if (!ensure_buffer_size(output_transit_buffer_, output_transit_buffer_ptr_,
                                output_transit_buffer_size_, output_transit_buffer_obj_,
                                transit_used, "output transit buffer")) {
            async_in_progress_ = false;
            return false;
        }

        if (SdmaShouldZeroTransit()) {
            hipError_t zerr = stream ? hipMemsetAsync(output_transit_buffer_, 0, transit_used, stream)
                                     : hipMemset(output_transit_buffer_, 0, transit_used);
            if (zerr != hipSuccess) {
                fprintf(stderr, "PE %d: start_async hipMemset(transit) failed: %s\n",
                        myPe_, hipGetErrorString(zerr));
                async_in_progress_ = false;
                return false;
            }
        }

        // Step 1: SdmaReduceScatter — same as operator()
        constexpr int pack_size = packed_t<T>::P::size;
        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;

        SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t rs_err = hipGetLastError();
        if (rs_err != hipSuccess) {
            fprintf(stderr, "PE %d: Async ReduceScatter launch failed: %s\n",
                    myPe_, hipGetErrorString(rs_err));
            async_in_progress_ = false;
            return false;
        }
        rs_err = stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
        if (rs_err != hipSuccess) {
            fprintf(stderr, "PE %d: sync after Async ReduceScatter failed: %s\n",
                    myPe_, hipGetErrorString(rs_err));
            async_in_progress_ = false;
            return false;
        }

        // Step 2: AllGather PUT only — sends data, returns immediately
        // The wait is deferred to wait_async so the user can run GEMM on CU
        AllGatherAsyncPutKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t kernel_err = hipGetLastError();
        if (kernel_err != hipSuccess) {
            fprintf(stderr, "PE %d: Async kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(kernel_err));
            throw std::runtime_error("Kernel launch failed");
        }

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

        // Wait for AllGather SDMA transfers to complete
        // Remote PEs wrote to our local signalPtrs via SDMA ATOMIC
        AllGatherAsyncWaitKernel<<<1, 64, 0, wait_stream>>>(
            myPe_, npes_, output_transit_buffer_obj_, barrierPtr_, async_total_count_);

        // Copy result to user buffer (if enabled)
        if (copy_output_to_user_) {
            copy_output_to_user(async_output_, async_total_count_, wait_stream);
        }

        // Single synchronization at the end
        if (wait_stream != nullptr) {
            hipError_t err = hipStreamSynchronize(wait_stream);
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: Stream synchronization failed: %s\n",
                        myPe_, hipGetErrorString(err));
                throw std::runtime_error("Stream synchronization failed");
            }
        } else {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: Device synchronization failed: %s\n",
                        myPe_, hipGetErrorString(err));
                throw std::runtime_error("Device synchronization failed");
            }
        }

        double end_time = MPI_Wtime();
        double duration = end_time - async_start_time_;

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
bool AllreduceSdma<T>::allreduce_inplace(T* data, size_t total_count,
                                          hipStream_t stream) {
    bool saved = copy_output_to_user_;
    copy_output_to_user_ = true;
    bool ok = (*this)(data, data, total_count, stream);
    copy_output_to_user_ = saved;
    return ok;
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
template class AllreduceSdma<__hip_bfloat16>;

} // namespace collective
} // namespace mori
