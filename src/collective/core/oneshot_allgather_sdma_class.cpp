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

#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"
#include "mori/collective/allgather/oneshot_sdma_kernel.hpp"
#include "mori/collective/allgather/oneshot_sdma_async_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace mori {
namespace collective {
#if 0
// Implementation of ShmemDeleter::operator()
void ShmemDeleter::operator()(void* ptr) const {
    if (ptr) {
        shmem::ShmemFree(ptr);
    }
}
#endif
// Constructor implementation - delegating version
template <typename T>
AllgatherSdma<T>::AllgatherSdma(int myPe, int npes, size_t transit_buffer_size)
    : AllgatherSdma(myPe, npes, transit_buffer_size / 2, transit_buffer_size / 2) {
    // Delegated to another constructor
}

// Main constructor implementation
template <typename T>
AllgatherSdma<T>::AllgatherSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size)
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
      async_in_progress_(false),
      async_input_(nullptr),
      async_output_(nullptr),
      async_total_count_(0),
      async_stream_(nullptr),
      async_start_time_(0.0) {

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

    // 2. Allocate input transit buffer
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

    // 3. Allocate output transit buffer
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
    printf("AllgatherSdma initialized: PE %d of %d\n", myPe_, npes_);
    printf("  Flags allocated: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Input transit buffer: %.2f MB at %p\n",
           input_transit_buffer_size_ / (1024.0 * 1024.0), input_transit_buffer_);
    printf("  Output transit buffer: %.2f MB at %p\n",
           output_transit_buffer_size_ / (1024.0 * 1024.0), output_transit_buffer_);
}

// Destructor
template <typename T>
AllgatherSdma<T>::~AllgatherSdma() {
    // Cancel any ongoing async operation
    if (async_in_progress_) {
        cancel_async();
    }

    // Memory is automatically managed by unique_ptr, ShmemDeleter will auto-free during destruction
    if (flags_) {
        printf("AllgatherSdma destroyed: PE %d\n", myPe_);
    }
}

// ================ NEW: Async API Implementations ================

template <typename T>
bool AllgatherSdma<T>::start_async(T* input, T* output, size_t total_count, hipStream_t stream) {
    // Check if another async operation is in progress
    bool expected = false;
    if (!async_in_progress_.compare_exchange_strong(expected, true)) {
        printf("PE %d: Another async operation is already in progress\n", myPe_);
        return false;
    }

    // Save parameters for async operation
    async_input_ = input;
    async_output_ = output;
    async_total_count_ = total_count;
    async_stream_ = stream;
    async_start_time_ = MPI_Wtime();

    try {
        // Step 1: Copy input data to input transit buffer
        printf("PE %d: Starting async AllGATHER (PUT phase)\n", myPe_);
        copy_input_to_transit(input, total_count, stream);

        // Step 2: Reset flags
        //resetFlags();

        // Step 3: Execute Allgather kernel (PUT operation)
        printf("PE %d: Launching async PUT kernel...\n", myPe_);

        int block_size = 256;
        int grid_size = (total_count * npes_ + block_size - 1) / block_size;
        if (grid_size < 1) grid_size = 1;
        if (grid_size > 65535) grid_size = 65535;

        printf("  Grid size: %d, Block size: %d\n", grid_size, block_size);

        // Launch the kernel - this runs asynchronously
        OneShotAllGatherSdmaAsyncPutKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            input,
            input_transit_buffer_obj_,
            output_transit_buffer_obj_,
            flagsObj_, total_count);

        hipError_t kernel_err = hipGetLastError();
        if (kernel_err != hipSuccess) {
            printf("PE %d: Async kernel launch failed: %s\n",
                   myPe_, hipGetErrorString(kernel_err));
            throw std::runtime_error("Kernel launch failed");
        }

        printf("PE %d: Async PUT operation started successfully\n", myPe_);
        return true;

    } catch (const std::exception& e) {
        printf("PE %d: Failed to start async operation: %s\n", myPe_, e.what());
        async_in_progress_ = false;
        return false;
    }
}

template <typename T>
double AllgatherSdma<T>::wait_async(hipStream_t stream) {
    if (!async_in_progress_) {
        printf("PE %d: No async operation in progress\n", myPe_);
        return -1.0;
    }

    try {
        printf("PE %d: Waiting for async Allgather completion (WAIT phase)\n", myPe_);

        // Use provided stream or the one from start_async
        hipStream_t wait_stream = (stream != nullptr) ? stream : async_stream_;

        OneShotAllGatherSdmaAsyncWaitKernel<<<1, 64, 0, wait_stream>>>(myPe_, npes_, output_transit_buffer_obj_, flagsObj_);

        // Step 1: Synchronize to ensure PUT kernel is completed
        printf("PE %d: Synchronizing to ensure PUT kernel completion\n", myPe_);

        if (wait_stream != nullptr) {
            hipError_t err = hipStreamSynchronize(wait_stream);
            if (err != hipSuccess) {
                printf("PE %d: Stream synchronization failed: %s\n",
                      myPe_, hipGetErrorString(err));
                throw std::runtime_error("Stream synchronization failed");
            }
        } else {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                printf("PE %d: Device synchronization failed: %s\n",
                      myPe_, hipGetErrorString(err));
                throw std::runtime_error("Device synchronization failed");
            }
        }

        // Step 2: Copy from output transit buffer to user output buffer
        printf("PE %d: Copying results to user output buffer\n", myPe_);
        copy_output_to_user(async_output_, async_total_count_, wait_stream);

        // Final synchronization
        if (wait_stream != nullptr) {
            (void)hipStreamSynchronize(wait_stream);
        } else {
            (void)hipDeviceSynchronize();
        }

        // Calculate total execution time
        double end_time = MPI_Wtime();
        double duration = end_time - async_start_time_;

        printf("PE %d: Async Allgather completed in %.6f seconds\n", myPe_, duration);

        // Reset async state
        async_in_progress_ = false;
        async_input_ = nullptr;
        async_output_ = nullptr;
        async_total_count_ = 0;
        async_stream_ = nullptr;
        async_start_time_ = 0.0;

        return duration;

    } catch (const std::exception& e) {
        printf("PE %d: Async wait failed: %s\n", myPe_, e.what());
        cancel_async();
        return -1.0;
    }
}

template <typename T>
void AllgatherSdma<T>::cancel_async() {
    if (async_in_progress_) {
        printf("PE %d: Cancelling async operation\n", myPe_);

        // Reset async state
        async_in_progress_ = false;
        async_input_ = nullptr;
        async_output_ = nullptr;
        async_total_count_ = 0;
        async_stream_ = nullptr;
        async_start_time_ = 0.0;

        // Reset flags
        //resetFlags();
    }
}

// ================ END: Async API Implementations ================

// ensure_buffer_size implementation (unchanged)
template <typename T>
bool AllgatherSdma<T>::ensure_buffer_size(void*& buffer,
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

// copy_input_to_transit implementation (unchanged)
template <typename T>
void AllgatherSdma<T>::copy_input_to_transit(T* input, size_t total_count, hipStream_t stream) {
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

// copy_output_to_user implementation (unchanged)
template <typename T>
void AllgatherSdma<T>::copy_output_to_user(T* output, size_t total_count, hipStream_t stream) {
    size_t total_elements = total_count * npes_;
    size_t output_bytes = total_elements * dtype_size_;

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
    hipError_t err = hipSuccess;
    if (stream != nullptr) {
        err = hipMemcpyAsync(output, output_transit_buffer_, output_bytes,
                           hipMemcpyDeviceToDevice, stream);
        // Immediately synchronize to ensure copy completes
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            fprintf(stderr, "PE %d: Stream synchronization failed: %s\n",
                    myPe_, hipGetErrorString(sync_err));
        }
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

// operator() implementation (modified to check async status)
// Returns true on success, false on failure
// Synchronization must be done by caller (Python layer)
template <typename T>
bool AllgatherSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    // Check if async operation is in progress
    if (async_in_progress_) {
        printf("PE %d: Cannot execute sync operation while async is in progress\n", myPe_);
        printf("  Call cancel_async() first or wait for async to complete\n");
        return false;
    }

    //hipError_t err = hipSuccess;
    //try {
        // Step 1: Copy input data to input transit buffer
    //    copy_input_to_transit(input, total_count, stream);

        // Step 2: Reset flags
    //    resetFlags();

        // Step 3: Execute Allgather kernel
    //    int block_size = 256;
    //    int grid_size = (total_count * npes_ + block_size - 1) / block_size;
    //    if (grid_size < 1) grid_size = 1;
    //    if (grid_size > 65535) grid_size = 65535;

        OneShotAllGatherSdmaKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            input,
            input_transit_buffer_obj_,
            output_transit_buffer_obj_,
            flagsObj_, total_count);

    //    err = hipGetLastError();
    //    if (err != hipSuccess) {
    //        fprintf(stderr, "PE %d: Kernel launch failed: %s\n",
    //                myPe_, hipGetErrorString(err));
    //        return false;
    //    }

        // Step 4: Copy from output transit buffer to user output buffer
        // Note: Synchronization is handled by Python layer
    //    copy_output_to_user(output, total_count, stream);

    //} catch (const std::exception& e) {
    //    fprintf(stderr, "PE %d: Allgather operation failed: %s\n", myPe_, e.what());
    //    return false;
    //}

    return true;
}

// resetFlags implementation (unchanged)
template <typename T>
void AllgatherSdma<T>::resetFlags() {
    if (flags_) {
        size_t flagsSize = npes_ * sizeof(uint64_t);
        memset(flags_.get(), 0, flagsSize);
    }
}

// Explicit instantiation of common types
template class AllgatherSdma<uint32_t>;
template class AllgatherSdma<uint64_t>;
template class AllgatherSdma<int32_t>;
template class AllgatherSdma<int64_t>;
template class AllgatherSdma<float>;
template class AllgatherSdma<double>;

} // namespace collective
} // namespace mori
