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

#include "mori/collective/reducescatter/reducescatter_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// Delegating constructor
// ---------------------------------------------------------------------------
template <typename T>
ReduceScatterSdma<T>::ReduceScatterSdma(int myPe, int npes,
                                         size_t transit_buffer_size,
                                         bool copy_output_to_user)
    : ReduceScatterSdma(myPe, npes, 0, transit_buffer_size, copy_output_to_user) {
}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
ReduceScatterSdma<T>::ReduceScatterSdma(int myPe, int npes,
                                         size_t /*input_buffer_size*/,
                                         size_t output_buffer_size,
                                         bool copy_output_to_user)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      max_blocks_(getDeviceMaxBlocks()),
      flags_(nullptr, ShmemDeleter()),
      barrierPtr_(nullptr),
      barrierMem_(nullptr, ShmemDeleter()),
      transit_buffer_(nullptr),
      transit_buffer_size_(output_buffer_size),
      transit_buffer_ptr_(nullptr, ShmemDeleter()),
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

    // 2. Allocate CrossPeBarrier
    size_t barrierSize = sizeof(CrossPeBarrier);
    void* bMem = shmem::ShmemMalloc(barrierSize);
    if (!bMem) throw std::runtime_error("Failed to allocate barrier memory");
    barrierMem_.reset(bMem);
    barrierPtr_ = reinterpret_cast<CrossPeBarrier*>(bMem);
    hipError_t me = hipMemset(bMem, 0, barrierSize);
    if (me != hipSuccess)
        throw std::runtime_error("Failed to zero-init barrier memory");

    // 3. Allocate transit buffer (gather buffer for SDMA scatter + reduce)
    transit_buffer_ = shmem::ShmemMalloc(transit_buffer_size_);
    if (!transit_buffer_)
        throw std::runtime_error("Failed to allocate transit buffer");
    transit_buffer_ptr_.reset(transit_buffer_);

    transit_buffer_obj_ =
        shmem::ShmemSymmetricRegister(transit_buffer_, transit_buffer_size_);
    if (!transit_buffer_obj_.IsValid())
        throw std::runtime_error("Failed to register transit buffer");

    printf("ReduceScatterSdma(SDMA) initialized: PE %d of %d, max_blocks=%d\n",
           myPe_, npes_, max_blocks_);
    printf("  Flags: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Barrier: %zu bytes at %p\n", barrierSize, bMem);
    printf("  Transit buffer: %.2f MB at %p\n",
           transit_buffer_size_ / (1024.0 * 1024.0), transit_buffer_);
}

// ---------------------------------------------------------------------------
template <typename T>
ReduceScatterSdma<T>::~ReduceScatterSdma() {
    if (async_in_progress_) {
        cancel_async();
    }
    if (flags_) {
        printf("ReduceScatterSdma destroyed: PE %d\n", myPe_);
    }
}

// ---------------------------------------------------------------------------
template <typename T>
bool ReduceScatterSdma<T>::ensure_buffer_size(void*& buffer,
                                              std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                                              size_t& current_size,
                                              application::SymmMemObjPtr& buffer_obj,
                                              size_t required_size,
                                              const char* buffer_name) {
    if (required_size <= current_size) {
        return true;
    }

    printf("PE %d: %s too small: required %.2f MB, current %.2f MB\n",
           myPe_, buffer_name,
           required_size / (1024.0 * 1024.0),
           current_size / (1024.0 * 1024.0));

    buffer_ptr.reset();

    current_size = required_size;
    buffer = shmem::ShmemMalloc(current_size);
    if (buffer == nullptr) {
        fprintf(stderr, "PE %d: Failed to reallocate %s of size %.2f MB\n",
                myPe_, buffer_name, current_size / (1024.0 * 1024.0));
        return false;
    }
    buffer_ptr.reset(buffer);

    buffer_obj = shmem::ShmemSymmetricRegister(buffer, current_size);
    if (!buffer_obj.IsValid()) {
        fprintf(stderr, "PE %d: Failed to re-register %s\n", myPe_, buffer_name);
        return false;
    }

    printf("PE %d: %s reallocated to %.2f MB\n",
           myPe_, buffer_name, current_size / (1024.0 * 1024.0));
    return true;
}

// ---------------------------------------------------------------------------
// Copy the reduced shard (slot[myPe]) from transit buffer to user output.
// ReduceScatter output = total_count / npes elements per rank.
// ---------------------------------------------------------------------------
template <typename T>
void ReduceScatterSdma<T>::copy_result_to_user(T* output, size_t total_count, hipStream_t stream) {
    using P = typename packed_t<T>::P;
    constexpr int pack_size = P::size;
    const size_t elementCountPerRank =
        ((total_count / npes_ + pack_size - 1) / pack_size) * pack_size;
    const size_t bytes = elementCountPerRank * dtype_size_;

    if (!output) throw std::runtime_error("Output pointer is null");
    if (!transit_buffer_) throw std::runtime_error("Transit buffer is null");

    uint8_t* src = reinterpret_cast<uint8_t*>(transit_buffer_)
                   + static_cast<size_t>(myPe_) * bytes;

    hipError_t err = stream
        ? hipMemcpyAsync(output, src, bytes, hipMemcpyDeviceToDevice, stream)
        : hipMemcpy(output, src, bytes, hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: copy_result_to_user failed: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Output copy failed");
    }
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template <typename T>
bool ReduceScatterSdma<T>::operator()(T* input, T* output,
                                      size_t total_count, hipStream_t stream) {
    if (async_in_progress_) {
        printf("PE %d: Cannot execute sync operation while async is in progress\n", myPe_);
        return false;
    }

    try {
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
            transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: SdmaReduceScatter launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        if (copy_output_to_user_) {
            copy_result_to_user(output, total_count, stream);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: ReduceScatter failed: %s\n", myPe_, e.what());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Async API
// ---------------------------------------------------------------------------
template <typename T>
bool ReduceScatterSdma<T>::start_async(T* input, T* output,
                                       size_t total_count, hipStream_t stream) {
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
        size_t required_size = total_count * dtype_size_;
        if (!ensure_buffer_size(transit_buffer_, transit_buffer_ptr_,
                                transit_buffer_size_, transit_buffer_obj_,
                                required_size, "transit buffer")) {
            async_in_progress_ = false;
            return false;
        }

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
            transit_buffer_obj_,
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
double ReduceScatterSdma<T>::wait_async(hipStream_t stream) {
    if (!async_in_progress_) {
        printf("PE %d: No async operation in progress\n", myPe_);
        return -1.0;
    }

    try {
        hipStream_t wait_stream = (stream != nullptr) ? stream : async_stream_;

        if (copy_output_to_user_) {
            copy_result_to_user(async_output_, async_total_count_, wait_stream);
        }

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
void ReduceScatterSdma<T>::cancel_async() {
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

// ---------------------------------------------------------------------------
template <typename T>
void ReduceScatterSdma<T>::resetFlags() {
    if (flags_) {
        memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class ReduceScatterSdma<uint32_t>;
template class ReduceScatterSdma<uint64_t>;
template class ReduceScatterSdma<int32_t>;
template class ReduceScatterSdma<int64_t>;
template class ReduceScatterSdma<float>;
template class ReduceScatterSdma<double>;
template class ReduceScatterSdma<half>;
template class ReduceScatterSdma<__hip_bfloat16>;

} // namespace collective
} // namespace mori
