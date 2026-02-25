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
#include "mori/shmem/shmem.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace mori {
namespace collective {

// Constructor implementation - delegating version
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t transit_buffer_size, bool copy_output_to_user)
    : AllreduceSdma(myPe, npes, transit_buffer_size / 2, transit_buffer_size / 2, copy_output_to_user) {
    // Delegated to another constructor
}

// Main constructor implementation
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size, bool copy_output_to_user)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      flags_(nullptr, ShmemDeleter()),
      input_transit_buffer_(nullptr),
      input_transit_buffer_size_(input_buffer_size),
      input_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      output_transit_buffer_(nullptr),
      output_transit_buffer_size_(output_buffer_size),
      output_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      copy_output_to_user_(copy_output_to_user) {

    // 1. Allocate and initialize flags memory
    size_t flagsSize = npes_ * sizeof(uint64_t);
    void* flags = shmem::ShmemMalloc(flagsSize);
    if (flags == nullptr) {
        throw std::runtime_error("Failed to allocate flags memory");
    }
    flags_.reset(static_cast<uint64_t*>(flags));
    memset(flags_.get(), 0, flagsSize);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_.get());
    if (!flagsObj_.IsValid()) {
        throw std::runtime_error("Failed to get valid flags memory object");
    }

    // 2. Allocate input transit buffer (srcMemObj)
    input_transit_buffer_ = shmem::ShmemMalloc(input_transit_buffer_size_);
    if (input_transit_buffer_ == nullptr) {
        throw std::runtime_error("Failed to allocate input transit buffer");
    }
    input_transit_buffer_ptr_.reset(input_transit_buffer_);

    // Register input transit buffer
    input_transit_buffer_obj_ = shmem::ShmemSymmetricRegister(input_transit_buffer_, input_transit_buffer_size_);
    if (!input_transit_buffer_obj_.IsValid()) {
        throw std::runtime_error("Failed to register input transit buffer");
    }

    // 3. Allocate output transit buffer (dstMemObj — used for gather, reduce, and allgather)
    output_transit_buffer_ = shmem::ShmemMalloc(output_transit_buffer_size_);
    if (output_transit_buffer_ == nullptr) {
        throw std::runtime_error("Failed to allocate output transit buffer");
    }
    output_transit_buffer_ptr_.reset(output_transit_buffer_);

    // Register output transit buffer
    output_transit_buffer_obj_ = shmem::ShmemSymmetricRegister(output_transit_buffer_, output_transit_buffer_size_);
    if (!output_transit_buffer_obj_.IsValid()) {
        throw std::runtime_error("Failed to register output transit buffer");
    }

    // 4. Print initialization information
    printf("AllreduceSdma initialized: PE %d of %d\n", myPe_, npes_);
    printf("  Flags allocated: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Input transit buffer: %.2f MB at %p\n",
           input_transit_buffer_size_ / (1024.0 * 1024.0), input_transit_buffer_);
    printf("  Output transit buffer: %.2f MB at %p\n",
           output_transit_buffer_size_ / (1024.0 * 1024.0), output_transit_buffer_);
}

// Destructor
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
    // Memory is automatically managed by unique_ptr, ShmemDeleter will auto-free during destruction
    if (flags_) {
        printf("AllreduceSdma destroyed: PE %d\n", myPe_);
    }
}

// ensure_buffer_size implementation
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

    // If buffer is not large enough, reallocate
    printf("PE %d: %s too small: required %.2f MB, current %.2f MB\n",
           myPe_, buffer_name,
           required_size / (1024.0 * 1024.0),
           current_size / (1024.0 * 1024.0));

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

    printf("PE %d: %s reallocated to %.2f MB\n",
           myPe_, buffer_name, current_size / (1024.0 * 1024.0));
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
    hipError_t err = hipSuccess;
    if (stream != nullptr) {
        err = hipMemcpyAsync(input_transit_buffer_, input, input_bytes,
                           hipMemcpyDeviceToDevice, stream);
        // Immediately synchronize to ensure copy completes
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            fprintf(stderr, "PE %d: Stream synchronization failed: %s\n",
                    myPe_, hipGetErrorString(sync_err));
        }
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
    size_t output_bytes = total_count * dtype_size_;

    // Verify pointer validity
    if (output == nullptr) {
        fprintf(stderr, "PE %d: Output pointer is null\n", myPe_);
        throw std::runtime_error("Output pointer is null");
    }

    if (output_transit_buffer_ == nullptr) {
        fprintf(stderr, "PE %d: Output transit buffer is null\n", myPe_);
        throw std::runtime_error("Output transit buffer is null");
    }

    // Copy from output transit buffer to user output buffer
    // Only the first total_count elements are valid result data
    hipError_t err = hipSuccess;
    if (stream != nullptr) {
        err = hipMemcpyAsync(output, output_transit_buffer_, output_bytes,
                           hipMemcpyDeviceToDevice, stream);
    } else {
        err = hipMemcpy(output, output_transit_buffer_, output_bytes,
                       hipMemcpyDeviceToDevice);
    }

    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: Failed to copy from transit buffer to output: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Output copy failed");
    }
}

// operator() implementation
// Returns true on success, false on failure
// Synchronization must be done by caller
template <typename T>
bool AllreduceSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    try {
        // Step 1: Copy user input to input transit buffer (symmetric memory)
        // ReduceScatterKernel reads from srcMemObj->peerPtrs[pe].
        copy_input_to_transit(input, total_count, stream);

        // Step 2: ReduceScatter — each rank reduces its partition via direct reads
        constexpr int pack_size = packed_t<T>::P::size;
        constexpr int kMaxBlocks = 80;
        int threads = 512;
        int packedSize = static_cast<int>(total_count) / pack_size;
        int threadsPerRank = threads / npes_;
        int partSize = packedSize / npes_;
        int blocks = std::min(kMaxBlocks, (partSize + threadsPerRank - 1) / threadsPerRank);
        if (blocks < 1) blocks = 1;

        ReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input_transit_buffer_obj_,
            output_transit_buffer_obj_,
            total_count);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: ReduceScatter kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 3: AllGather via SDMA — broadcast reduced shards to all ranks
        AllGatherSdmaKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_, total_count);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: AllGather kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 4: Copy from output transit buffer to user output buffer (if enabled)
        // The result in dstMemObj is laid out with elementCountPerRank-stride shards;
        // the first total_count elements form the complete allreduce result.
        if (copy_output_to_user_) {
            copy_output_to_user(output, total_count, stream);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: AllReduce operation failed: %s\n", myPe_, e.what());
        return false;
    }

    return true;
}

// resetFlags implementation
template <typename T>
void AllreduceSdma<T>::resetFlags() {
    if (flags_) {
        size_t flagsSize = npes_ * sizeof(uint64_t);
        memset(flags_.get(), 0, flagsSize);
    }
}

// Explicit instantiation of common types
template class AllreduceSdma<uint32_t>;
template class AllreduceSdma<uint64_t>;
template class AllreduceSdma<int32_t>;
template class AllreduceSdma<int64_t>;
template class AllreduceSdma<float>;
template class AllreduceSdma<double>;

} // namespace collective
} // namespace mori
