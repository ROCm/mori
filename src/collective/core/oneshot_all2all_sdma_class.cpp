#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace mori {
namespace collective {

// 实现ShmemDeleter的operator()
void ShmemDeleter::operator()(void* ptr) const {
    if (ptr) {
        shmem::ShmemFree(ptr);
    }
}

// Constructor implementation - delegating version
template <typename T>
All2allSdma<T>::All2allSdma(int myPe, int npes, size_t transit_buffer_size)
    : All2allSdma(myPe, npes, transit_buffer_size / 2, transit_buffer_size / 2) {
    // Delegated to another constructor
}

// Main constructor implementation
template <typename T>
All2allSdma<T>::All2allSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size)
    : myPe_(myPe), 
      npes_(npes), 
      dtype_size_(sizeof(T)),
      flags_(nullptr, ShmemDeleter()),
      input_transit_buffer_(nullptr),
      input_transit_buffer_size_(input_buffer_size),
      input_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      output_transit_buffer_(nullptr),
      output_transit_buffer_size_(output_buffer_size),
      output_transit_buffer_ptr_(nullptr, ShmemDeleter()) {

    // 1. Allocate and initialize flags memory
    size_t flagsSize = npes_ * sizeof(uint64_t);
    void* flags = shmem::ShmemMalloc(flagsSize);
    if (flags == nullptr) {
        throw std::runtime_error("Failed to allocate flags memory");
    }
    flags_.reset(static_cast<uint64_t*>(flags));

    // 初始化flags为0
    memset(flags_.get(), 0, flagsSize);

    // 获取对称内存对象指针
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
    printf("All2allSdma initialized: PE %d of %d\n", myPe_, npes_);
    printf("  Flags allocated: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Input transit buffer: %.2f MB at %p\n", 
           input_transit_buffer_size_ / (1024.0 * 1024.0), input_transit_buffer_);
    printf("  Output transit buffer: %.2f MB at %p\n", 
           output_transit_buffer_size_ / (1024.0 * 1024.0), output_transit_buffer_);
}

// Destructor
template <typename T>
All2allSdma<T>::~All2allSdma() {
    // Memory is automatically managed by unique_ptr, ShmemDeleter will auto-free during destruction
    // ShmemDeleter会在unique_ptr析构时自动调用ShmemFree
    if (flags_) {
        printf("All2allSdma destroyed: PE %d\n", myPe_);
    }
}

// ensure_buffer_size implementation
template <typename T>
bool All2allSdma<T>::ensure_buffer_size(void*& buffer, 
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

template <typename T>
void All2allSdma<T>::copy_input_to_transit(T* input, size_t total_count, hipStream_t stream) {
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
    #if 0
    printf("PE %d: Copying %.2f MB from input to transit buffer\n",
           myPe_, input_bytes / (1024.0 * 1024.0));
    printf("  Source: %p, Destination: %p, Size: %zu bytes\n",
           input, input_transit_buffer_, input_bytes);
    #endif 
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
    
    //printf("PE %d: Input copy completed successfully\n", myPe_);
}

template <typename T>
void All2allSdma<T>::copy_output_to_user(T* output, size_t total_count, hipStream_t stream) {
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
    #if 0
    printf("PE %d: Copying %.2f MB from transit buffer to output\n",
           myPe_, output_bytes / (1024.0 * 1024.0));
    printf("  Source: %p, Destination: %p, Size: %zu bytes\n",
           output_transit_buffer_, output, output_bytes);
    #endif
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
    
    //printf("PE %d: Output copy completed successfully\n", myPe_);
}

// Simplified operator() implementation, removed verification steps
template <typename T>
double All2allSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    // ... parameter checks ...

    // Execute All2All operation
    double start = MPI_Wtime();

    hipError_t sync_err = hipSuccess;
    try {
        // Step 1: Copy input data to input transit buffer
        copy_input_to_transit(input, total_count * npes_, stream);

        // Step 2: Reset flags
        resetFlags();

        // Step 3: Execute All2All kernel
        //printf("PE %d: Launching All2All kernel...\n", myPe_);

        int block_size = 256;
        int grid_size = (total_count * npes_ + block_size - 1) / block_size;
        if (grid_size < 1) grid_size = 1;
        if (grid_size > 65535) grid_size = 65535;

        //printf("  Grid size: %d, Block size: %d\n", grid_size, block_size);

        OneShotAll2allSdmaKernel<T><<<1, 64, 0, stream>>>(
            myPe_, npes_,
            input_transit_buffer_obj_,
            output_transit_buffer_obj_,
            flagsObj_, total_count);

        sync_err = hipGetLastError();
        if (sync_err != hipSuccess) {
            fprintf(stderr, "PE %d: Kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(sync_err));
            throw std::runtime_error("Kernel launch failed");
        }

        // Synchronize GPU to ensure kernel completion
        if (stream != nullptr) {
            sync_err = hipStreamSynchronize(stream);
        } else {
            sync_err = hipDeviceSynchronize();
        }

        if (sync_err != hipSuccess) {
            fprintf(stderr, "PE %d: Failed to synchronize: %s\n",
                    myPe_, hipGetErrorString(sync_err));
            throw std::runtime_error("Synchronization failed");
        }

        //printf("PE %d: Kernel execution completed\n", myPe_);

        // Step 4: Copy from output transit buffer to user output buffer
        copy_output_to_user(output, total_count, stream);

        // Final synchronization
        if (stream != nullptr) {
            sync_err = hipStreamSynchronize(stream);
        } else {
            sync_err = hipDeviceSynchronize();
        }

        if (sync_err != hipSuccess) {
            fprintf(stderr, "PE %d: Final synchronization failed: %s\n",
                    myPe_, hipGetErrorString(sync_err));
            throw std::runtime_error("Final synchronization failed");
        }

        //printf("PE %d: All2All operation completed successfully\n", myPe_);

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: All2All operation failed: %s\n", myPe_, e.what());
        return -1.0;
    }

    double end = MPI_Wtime();
    double duration = end - start;

    //printf("PE %d: All2all_sdma completed in %.6f seconds\n", myPe_, duration);

    return duration;
}

// resetFlags implementation
template <typename T>
void All2allSdma<T>::resetFlags() {
    if (flags_) {
        size_t flagsSize = npes_ * sizeof(uint64_t);
        memset(flags_.get(), 0, flagsSize);
    }
}

// Explicit instantiation of common types
template class All2allSdma<uint32_t>;
template class All2allSdma<uint64_t>;
template class All2allSdma<int32_t>;
template class All2allSdma<int64_t>;
template class All2allSdma<float>;
template class All2allSdma<double>;

} // namespace collective
} // namespace mori