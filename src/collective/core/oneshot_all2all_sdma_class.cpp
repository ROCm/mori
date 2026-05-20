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

#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "mori/shmem/shmem.hpp"

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
All2allSdma<T>::All2allSdma(int myPe, int npes, size_t transit_buffer_size,
                            bool copy_output_to_user)
    : All2allSdma(myPe, npes, transit_buffer_size / 2, transit_buffer_size / 2,
                  copy_output_to_user) {
  // Delegated to another constructor
}

// Main constructor implementation
template <typename T>
All2allSdma<T>::All2allSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size,
                            bool copy_output_to_user)
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
      async_start_time_(0.0),
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

  // 2. Allocate input transit buffer
  input_transit_buffer_ = shmem::ShmemMalloc(input_transit_buffer_size_);
  if (input_transit_buffer_ == nullptr) {
    throw std::runtime_error("Failed to allocate input transit buffer");
  }
  input_transit_buffer_ptr_.reset(input_transit_buffer_);

  // Register input transit buffer
  input_transit_buffer_obj_ =
      shmem::ShmemSymmetricRegister(input_transit_buffer_, input_transit_buffer_size_);
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
  output_transit_buffer_obj_ =
      shmem::ShmemSymmetricRegister(output_transit_buffer_, output_transit_buffer_size_);
  if (!output_transit_buffer_obj_.IsValid()) {
    throw std::runtime_error("Failed to register output transit buffer");
  }

  // 4. Print initialization information
  printf("All2allSdma initialized: PE %d of %d\n", myPe_, npes_);
  printf("  Flags allocated: %zu bytes at %p\n", flagsSize, flags_.get());
  printf("  Input transit buffer: %.2f MB at %p\n", input_transit_buffer_size_ / (1024.0 * 1024.0),
         input_transit_buffer_);
  printf("  Output transit buffer: %.2f MB at %p\n",
         output_transit_buffer_size_ / (1024.0 * 1024.0), output_transit_buffer_);
}

// Destructor
template <typename T>
All2allSdma<T>::~All2allSdma() {
  // Cancel any ongoing async operation
  if (async_in_progress_) {
    cancel_async();
  }

  // Memory is automatically managed by unique_ptr, ShmemDeleter will auto-free during destruction
  if (flags_) {
    printf("All2allSdma destroyed: PE %d\n", myPe_);
  }
}

// ================ NEW: Async API Implementations ================

template <typename T>
bool All2allSdma<T>::start_async(T* input, T* output, size_t total_count, hipStream_t stream) {
  // Kernel launch moved to Python JIT path.
  // Use prepare_async_start / after_async_start instead.
  throw std::runtime_error(
      "All2allSdma::start_async() is deprecated; use Python JIT launch path "
      "(prepare_async_start + after_async_start)");
}

template <typename T>
double All2allSdma<T>::wait_async(hipStream_t stream) {
  // Kernel launch moved to Python JIT path.
  // Use prepare_async_wait / finish_async_wait instead.
  throw std::runtime_error(
      "All2allSdma::wait_async() is deprecated; use Python JIT launch path "
      "(prepare_async_wait + finish_async_wait)");
}

template <typename T>
void All2allSdma<T>::cancel_async() {
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
    // resetFlags();
  }
}

// ================ END: Async API Implementations ================

// ensure_buffer_size implementation (unchanged)
template <typename T>
bool All2allSdma<T>::ensure_buffer_size(void*& buffer,
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

// copy_input_to_transit implementation (unchanged)
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

  // Copy from user input buffer to input transit buffer
  hipError_t err = hipSuccess;
  if (stream != nullptr) {
    err =
        hipMemcpyAsync(input_transit_buffer_, input, input_bytes, hipMemcpyDeviceToDevice, stream);
    // Immediately synchronize to ensure copy completes
    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
      fprintf(stderr, "PE %d: Stream synchronization failed: %s\n", myPe_,
              hipGetErrorString(sync_err));
    }
  } else {
    err = hipMemcpy(input_transit_buffer_, input, input_bytes, hipMemcpyDeviceToDevice);
  }

  if (err != hipSuccess) {
    fprintf(stderr, "PE %d: Failed to copy input to transit buffer: %s\n", myPe_,
            hipGetErrorString(err));
    throw std::runtime_error("Input copy failed");
  }
}

// copy_output_to_user implementation (unchanged)
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

  // Copy from output transit buffer to user output buffer
  hipError_t err = hipSuccess;
  if (stream != nullptr) {
    err = hipMemcpyAsync(output, output_transit_buffer_, output_bytes, hipMemcpyDeviceToDevice,
                         stream);
    // Immediately synchronize to ensure copy completes
    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
      fprintf(stderr, "PE %d: Stream synchronization failed: %s\n", myPe_,
              hipGetErrorString(sync_err));
    }
  } else {
    err = hipMemcpy(output, output_transit_buffer_, output_bytes, hipMemcpyDeviceToDevice);
  }

  if (err != hipSuccess) {
    fprintf(stderr, "PE %d: Failed to copy from transit buffer to output: %s\n", myPe_,
            hipGetErrorString(err));
    throw std::runtime_error("Output copy failed");
  }
}

// operator() — kernel launch moved to Python JIT path
template <typename T>
double All2allSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
  // Kernel launch moved to Python JIT path.
  // Use prepare_sync / finish_sync instead.
  throw std::runtime_error(
      "All2allSdma::operator() is deprecated; use Python JIT launch path "
      "(prepare_sync + finish_sync)");
}

// ================ JIT launch support ================

template <typename T>
int64_t All2allSdma<T>::prepare_sync(T* input, T* output, size_t total_count, hipStream_t stream) {
  if (async_in_progress_) {
    throw std::runtime_error("Cannot execute sync operation while async is in progress");
  }
  jit_args_.myPe = myPe_;
  jit_args_.npes = npes_;
  jit_args_.input = input;
  jit_args_.inputTransitMemObj = input_transit_buffer_obj_;
  jit_args_.outputTransitMemObj = output_transit_buffer_obj_;
  jit_args_.flagsMemObj = flagsObj_;
  jit_args_.elementCount = total_count;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double All2allSdma<T>::finish_sync(T* output, size_t total_count, hipStream_t stream) {
  hipError_t err;
  if (stream != nullptr) {
    err = hipStreamSynchronize(stream);
  } else {
    err = hipDeviceSynchronize();
  }
  if (err != hipSuccess) {
    throw std::runtime_error("Synchronization failed");
  }
  if (copy_output_to_user_) {
    copy_output_to_user(output, total_count, stream);
  }
  if (stream != nullptr) {
    (void)hipStreamSynchronize(stream);
  } else {
    (void)hipDeviceSynchronize();
  }
  return 0.0;
}

template <typename T>
int64_t All2allSdma<T>::prepare_async_start(T* input, T* output, size_t total_count,
                                            hipStream_t stream) {
  bool expected = false;
  if (!async_in_progress_.compare_exchange_strong(expected, true)) {
    throw std::runtime_error("Another async operation is already in progress");
  }
  async_input_ = input;
  async_output_ = output;
  async_total_count_ = total_count;
  async_stream_ = stream;
  async_start_time_ = CollectiveWallTime();

  jit_args_.myPe = myPe_;
  jit_args_.npes = npes_;
  jit_args_.input = input;
  jit_args_.inputTransitMemObj = input_transit_buffer_obj_;
  jit_args_.outputTransitMemObj = output_transit_buffer_obj_;
  jit_args_.flagsMemObj = flagsObj_;
  jit_args_.elementCount = total_count;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
void All2allSdma<T>::after_async_start() {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    async_in_progress_ = false;
    throw std::runtime_error("Async kernel launch failed");
  }
}

template <typename T>
int64_t All2allSdma<T>::prepare_async_wait(hipStream_t stream) {
  if (!async_in_progress_) {
    throw std::runtime_error("No async operation in progress");
  }
  jit_args_.myPe = myPe_;
  jit_args_.npes = npes_;
  jit_args_.input = nullptr;
  jit_args_.inputTransitMemObj = {};
  jit_args_.outputTransitMemObj = output_transit_buffer_obj_;
  jit_args_.flagsMemObj = flagsObj_;
  jit_args_.elementCount = 0;
  return reinterpret_cast<int64_t>(&jit_args_);
}

template <typename T>
double All2allSdma<T>::finish_async_wait(hipStream_t stream) {
  hipStream_t wait_stream = (stream != nullptr) ? stream : async_stream_;
  hipError_t err;
  if (wait_stream != nullptr) {
    err = hipStreamSynchronize(wait_stream);
  } else {
    err = hipDeviceSynchronize();
  }
  if (err != hipSuccess) {
    throw std::runtime_error("Synchronization failed");
  }

  if (copy_output_to_user_) {
    copy_output_to_user(async_output_, async_total_count_, wait_stream);
  }
  if (wait_stream != nullptr) {
    (void)hipStreamSynchronize(wait_stream);
  } else {
    (void)hipDeviceSynchronize();
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

// ================ END: JIT launch support ================

// resetFlags implementation (unchanged)
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

}  // namespace collective
}  // namespace mori
