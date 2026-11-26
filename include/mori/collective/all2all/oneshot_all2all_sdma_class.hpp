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

#ifndef ONESHOT_ALL2ALL_SDMA_CLASS_HPP
#define ONESHOT_ALL2ALL_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// Include necessary headers
#include "mori/application/application.hpp"
#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/collective/collective_pub.hpp"
#include "mori/collective/core/wall_time.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {
template <typename T>
class All2allSdma {
 private:
  int myPe_;
  int npes_;
  size_t dtype_size_;

  // Flag memory
  application::SymmMemObjPtr flagsObj_;
  std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

  // Input transit buffer
  void* input_transit_buffer_;
  size_t input_transit_buffer_size_;
  application::SymmMemObjPtr input_transit_buffer_obj_;
  std::unique_ptr<void, ShmemDeleter> input_transit_buffer_ptr_;

  // Output transit buffer
  void* output_transit_buffer_;
  size_t output_transit_buffer_size_;
  application::SymmMemObjPtr output_transit_buffer_obj_;
  std::unique_ptr<void, ShmemDeleter> output_transit_buffer_ptr_;

  // ================ NEW: Async state variables ================
  std::atomic<bool> async_in_progress_;  // Flag indicating async operation is active
  T* async_input_;                       // Saved input pointer for async operation
  T* async_output_;                      // Saved output pointer for async operation
  size_t async_total_count_;             // Saved element count for async operation
  hipStream_t async_stream_;             // Saved stream for async operation
  double async_start_time_;              // Start time for async operation
  // ============================================================

  // Copy mode flag: if true, copy output_transit_buffer to user output buffer
  // if false, user should directly use output_transit_buffer
  bool copy_output_to_user_;

  // Not reentrant: previous launch must complete before next prepare_*.
  CclAll2allArgs<T> jit_args_;

  // Caller outputs registered for direct delivery, by base pointer: the peers write their chunks
  // straight into them, so nothing is copied out of the output transit buffer.
  struct RegisteredOutput {
    application::SymmMemObjPtr obj;
    size_t size;
    // The number of the collective registration that made obj, from 1; the same on every rank.
    uint64_t id;
    // Whether the peers reach this rank's buffer through their obj. False when the runtime
    // could not export the buffer and a stand-in was registered in its place.
    bool direct;
    // The stand-in, or nullptr.
    void* standIn;
  };
  std::map<uintptr_t, RegisteredOutput> registered_outputs_;
  uint64_t register_count_ = 0;

  // Whether the runtime can export this memory to other processes. For some allocations
  // hipIpcGetMemHandle fails every time ("invalid argument"), and the symmetric registration
  // treats that as fatal, so it is asked first.
  static bool can_export(void* ptr) {
    hipIpcMemHandle_t probe;
    if (hipIpcGetMemHandle(&probe, ptr) == hipSuccess) return true;
    (void)hipGetLastError();
    return false;
  }

  void drop_registered(typename std::map<uintptr_t, RegisteredOutput>::iterator it) {
    void* registered =
        it->second.standIn != nullptr ? it->second.standIn : reinterpret_cast<void*>(it->first);
    shmem::ShmemSymmetricDeregister(
        registered, it->second.standIn != nullptr ? kStandInBytes : it->second.size);
    if (it->second.standIn != nullptr) (void)hipFree(it->second.standIn);
  }

  static constexpr size_t kStandInBytes = 2 * 1024 * 1024;

  // Disable copy constructor and assignment operator
  All2allSdma(const All2allSdma&) = delete;
  All2allSdma& operator=(const All2allSdma&) = delete;

  // Internal methods
  bool ensure_buffer_size(void*& buffer, std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                          size_t& current_size, application::SymmMemObjPtr& buffer_obj,
                          size_t required_size, const char* buffer_name);

  void copy_input_to_transit(T* input, size_t total_count, hipStream_t stream);
  void copy_output_to_user(T* output, size_t total_count, hipStream_t stream);

 public:
  /**
   * @brief Constructor, initializes All2allSdma class
   * @param myPe Current PE ID
   * @param npes Total number of PEs
   * @param transit_buffer_size Transit buffer size in bytes (default 512MB), half for input and
   * half for output
   * @param copy_output_to_user If true, copy output_transit_buffer to user output buffer (default
   * true)
   */
  All2allSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024,
              bool copy_output_to_user = true);

  /**
   * @brief Constructor, specifying input and output transit buffer sizes separately
   * @param myPe Current PE ID
   * @param npes Total number of PEs
   * @param input_buffer_size Input transit buffer size in bytes
   * @param output_buffer_size Output transit buffer size in bytes
   * @param copy_output_to_user If true, copy output_transit_buffer to user output buffer (default
   * true)
   */
  All2allSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size,
              bool copy_output_to_user = true);

  /**
   * @brief Destructor, cleans up resources
   */
  ~All2allSdma();

  /**
   * @brief 执行All2All SDMA操作
   * @param input Input data pointer
   * @param output Output data pointer
   * @param total_count Number of data elements per PE
   * @param stream HIP stream
   * @return true if successful, false otherwise
   */
  bool start_async(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

  /**
   * @brief Wait for asynchronous All2All operation to complete (WAIT phase)
   * @param stream HIP stream (optional, can be different from start_async stream)
   * @return Execution time in seconds, -1.0 if failed
   */
  double wait_async(hipStream_t stream = nullptr);

  /**
   * @brief Check if async operation is in progress
   * @return true if async operation is active
   */
  bool is_async_in_progress() const { return async_in_progress_; }

  /**
   * @brief Cancel ongoing async operation
   */
  void cancel_async();
  // =======================================================

  // JIT launch support: prepare args struct and return pointer.
  // Python calls prepare_*, then launches the kernel via JIT, then calls finish_*.

  int64_t prepare_sync(T* input, T* output, size_t total_count, hipStream_t stream);
  double finish_sync(T* output, size_t total_count, hipStream_t stream);

  int64_t prepare_async_start(T* input, T* output, size_t total_count, hipStream_t stream);
  void after_async_start();

  int64_t prepare_async_wait(hipStream_t stream);
  double finish_async_wait(hipStream_t stream);

  /**
   * @brief Executes synchronous All2All SDMA operation
   * @param input Input data pointer
   * @param output Output data pointer
   * @param total_count Number of data elements per PE
   * @param stream HIP stream
   * @return Execution time in seconds, returns -1.0 if failed
   */
  double operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

  /**
   * @brief Gets flag symmetric memory object
   */
  application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }

  /**
   * @brief Gets input transit buffer pointer
   */
  void* getInputTransitBuffer() const { return input_transit_buffer_; }

  /**
   * @brief Gets input transit buffer size in bytes
   */
  size_t getInputTransitBufferSize() const { return input_transit_buffer_size_; }

  /**
   * @brief Gets input transit buffer symmetric memory object
   */
  application::SymmMemObjPtr getInputTransitBufferObj() const { return input_transit_buffer_obj_; }

  /**
   * @brief Gets output transit buffer pointer
   */
  void* getOutputTransitBuffer() const { return output_transit_buffer_; }

  /**
   * @brief Gets output transit buffer size in bytes
   */
  size_t getOutputTransitBufferSize() const { return output_transit_buffer_size_; }

  /**
   * @brief Gets output transit buffer symmetric memory object
   */
  application::SymmMemObjPtr getOutputTransitBufferObj() const {
    return output_transit_buffer_obj_;
  }

  /**
   * @brief Resets flags (sets to 0)
   */
  void resetFlags();

  /**
   * @brief What register_output_buffer hands to the kernel launch.
   */
  struct RegisteredOutputInfo {
    application::SymmMemObjPtr obj;  // reaches the buffers the peers registered in the same call
    uint64_t id;                     // number of that collective registration, from 1
    bool direct;                     // the peers reach this rank's buffer too
  };

  /**
   * @brief Registers a caller output for direct delivery. COLLECTIVE: every rank calls it at the
   * same point with its own buffer of the same size. Cached: an output already registered with
   * this base and size is returned as is; entries that overlap a new one are stale (the allocator
   * reused the memory) and dropped.
   *
   * A buffer the runtime cannot export is not an error: this rank registers a stand-in so that
   * the collective completes, and reports direct = false. It can then still write straight into
   * the peers' buffers, and the peers send its share through the output transit buffer.
   */
  RegisteredOutputInfo register_output_buffer(void* ptr, size_t size) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    auto exact = registered_outputs_.find(base);
    if (exact != registered_outputs_.end() && exact->second.size == size) {
      return {exact->second.obj, exact->second.id, exact->second.direct};
    }
    for (auto it = registered_outputs_.begin(); it != registered_outputs_.end();) {
      const uintptr_t b = it->first;
      const size_t s = it->second.size;
      if (base < b + s && b < base + size) {
        drop_registered(it);
        it = registered_outputs_.erase(it);
      } else {
        ++it;
      }
    }
    const bool direct = can_export(ptr);
    void* standIn = nullptr;
    if (!direct) {
      // Allocations that cannot be exported are kept until one that can is found, so that the
      // allocator does not hand the same one out again.
      std::vector<void*> refused;
      for (int tries = 0; tries < 8 && standIn == nullptr; ++tries) {
        void* candidate = nullptr;
        if (hipMalloc(&candidate, kStandInBytes) != hipSuccess) break;
        if (can_export(candidate)) {
          standIn = candidate;
        } else {
          refused.push_back(candidate);
        }
      }
      for (void* p : refused) (void)hipFree(p);
      if (standIn == nullptr) {
        throw std::runtime_error("All2allSdma: no exportable memory for an output registration");
      }
    }
    // Reached over P2P only, never an RDMA source or destination.
    application::SymmMemObjPtr obj = shmem::ShmemSymmetricRegister(
        standIn != nullptr ? standIn : ptr, standIn != nullptr ? kStandInBytes : size, false);
    if (!obj.IsValid()) throw std::runtime_error("All2allSdma: output register failed");
    const uint64_t id = ++register_count_;
    registered_outputs_[base] = {obj, size, id, direct, standIn};
    return {obj, id, direct};
  }

  /**
   * @brief Drops a registered output. COLLECTIVE like register_output_buffer.
   */
  void deregister_output_buffer(void* ptr) {
    auto it = registered_outputs_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == registered_outputs_.end()) return;
    drop_registered(it);
    registered_outputs_.erase(it);
  }
};

}  // namespace collective
}  // namespace mori

#endif  // ONESHOT_ALL2ALL_SDMA_CLASS_HPP
