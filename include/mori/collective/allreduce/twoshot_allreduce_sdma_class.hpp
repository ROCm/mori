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

#ifndef TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
#define TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>

#include "mori/application/application.hpp"
#include "mori/collective/allreduce/twoshot_sdma_common.hpp"
#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/collective/collective_pub.hpp"
#include "mori/collective/core/wall_time.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

template <typename T>
class AllreduceSdma {
 private:
  int myPe_;
  int npes_;
  size_t dtype_size_;
  int max_blocks_;

  // SDMA completion flags (shared by SdmaReduceScatter and AllGather phases;
  // each phase resets flags before handing off to the next).
  application::SymmMemObjPtr flagsObj_;
  std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

  // Device-side generation and per-block rendezvous state.
  CrossPeBarrier* barrierPtr_;
  std::unique_ptr<void, ShmemDeleter> barrierMem_;

  // Fixed-stride symmetric scratch for SDMA scatter + local reduce. Rank slots
  // keep stable offsets across graph replays with different message sizes.
  void* input_transit_buffer_;
  size_t input_transit_buffer_size_;
  size_t slot_stride_elements_;
  application::SymmMemObjPtr input_transit_buffer_obj_;
  std::unique_ptr<void, ShmemDeleter> input_transit_buffer_ptr_;

  // Async state variables
  std::atomic<bool> async_in_progress_;
  T* async_input_;
  T* async_output_;
  size_t async_total_count_;
  hipStream_t async_stream_;
  double async_start_time_;

  // Kept for source compatibility. Results are always materialized in the
  // caller output; two-shot operations copy from compact symmetric scratch.
  bool copy_output_to_user_;

  AllreduceSdma(const AllreduceSdma&) = delete;
  AllreduceSdma& operator=(const AllreduceSdma&) = delete;

  void copy_input_to_transit(T* input, size_t total_count, hipStream_t stream);
  void fill_jit_args_(const T* input, size_t total_count);

 public:
  /**
   * @brief Constructor
   * @param myPe Current PE ID
   * @param npes Total number of PEs
   * @param transit_buffer_size Output transit buffer size in bytes (default 512MB)
   * @param copy_output_to_user Kept for compatibility; caller output is always populated.
   * @param use_graph_mode Kept for API compat — ignored (SDMA always reads
   *        input directly, no IPC registration needed).
   */
  AllreduceSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024,
                bool copy_output_to_user = true, bool use_graph_mode = false);

  AllreduceSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size,
                bool copy_output_to_user = true, bool use_graph_mode = false);

  ~AllreduceSdma();

  bool operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

  /**
   * @brief Start asynchronous AllReduce operation (AllGather PUT phase)
   * @param input Input data pointer
   * @param output Output data pointer
   * @param total_count Number of data elements per PE
   * @param stream HIP stream
   * @return true if successful, false otherwise
   */
  bool start_async(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

  /**
   * @brief Wait for asynchronous AllReduce operation to complete
   *        (WAIT phase + local reduce + optional copy)
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

  /**
   * @brief Executes in-place AllReduce SDMA operation (result overwrites input)
   * @param data Input/output data pointer (elementCount elements on each rank)
   * @param total_count Number of data elements per PE
   * @param stream HIP stream
   * @return true if successful, false if failed
   * @note Synchronization must be handled by the caller
   */
  bool allreduce_inplace(T* data, size_t total_count, hipStream_t stream = nullptr);

  application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }

  void resetFlags();

  // JIT launch support: Python calls prepare_* to get args pointer,
  // launches the kernel via HipModule, then calls finish_*.
  // Not reentrant: previous launch must complete before next prepare_*.
  CclAllreduceArgs<T> jit_args_;

  int64_t prepare_reduce_scatter(const T* input, T* output, size_t total_count, hipStream_t stream);
  std::tuple<int, int> get_reduce_scatter_grid(size_t total_count) const;
  int64_t prepare_allgather(size_t total_count, hipStream_t stream);
  double finish_sync(T* output, size_t total_count, hipStream_t stream,
                     bool force_copy_output_to_user = false, bool direct_output = false);

  int64_t prepare_async_reduce_scatter(const T* input, T* output, size_t total_count,
                                       hipStream_t stream);
  int64_t prepare_async_allgather_put(size_t total_count, hipStream_t stream);
  void after_async_start(bool capturing = false);

  int64_t prepare_async_wait(hipStream_t stream);
  double finish_async_wait(hipStream_t stream, bool capturing = false, bool direct_output = false);

  int max_blocks() const { return max_blocks_; }
  int npes() const { return npes_; }
};

}  // namespace collective
}  // namespace mori

#endif  // TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
