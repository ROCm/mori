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
#include <mpi.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/collective/collective_pub.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

struct CrossPeBarrier;

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

  // Device-scope barrier for block-0-to-all broadcast inside
  // SdmaReduceScatterKernel (generation counter, ~128 bytes).
  CrossPeBarrier* barrierPtr_;
  std::unique_ptr<void, ShmemDeleter> barrierMem_;

  // Input transit buffer (symmetric memory for P2P reads)
  void* input_transit_buffer_;
  size_t input_transit_buffer_size_;
  application::SymmMemObjPtr input_transit_buffer_obj_;
  std::unique_ptr<void, ShmemDeleter> input_transit_buffer_ptr_;

  // Output transit buffer — serves as:
  //   1. SDMA scatter destination (gather buffer, npes * chunkSize)
  //   2. Local reduce output (myPe's slot)
  //   3. AllGather source / final result
  void* output_transit_buffer_;
  size_t output_transit_buffer_size_;
  application::SymmMemObjPtr output_transit_buffer_obj_;
  std::unique_ptr<void, ShmemDeleter> output_transit_buffer_ptr_;

  // Async state variables
  std::atomic<bool> async_in_progress_;
  T* async_input_;
  T* async_output_;
  size_t async_total_count_;
  hipStream_t async_stream_;
  double async_start_time_;

  // Copy mode flag: if true, copy output_transit_buffer to user output buffer
  // if false, user should directly use output_transit_buffer
  bool copy_output_to_user_;

  // Host-side generation counters for pipeline signal expectations.
  // Avoids reading signal memory at kernel start (inter-GPU race).
  // scatter shares qId=0 with serial; counter tracks both serial and pipeline increments.
  // Signals are zeroed in constructor, so counters start at 0.
  uint64_t pipeline_scatter_gen_ = 0;  // total SDMA ATOMIC_INC on qId=0 (scatter only)
  uint64_t pipeline_ag_gen_ = 0;       // total SDMA ATOMIC_INC on qId=1 (pipeline AG only)
  uint64_t pipeline_reduce_gen_ = 0;   // reduce_complete counter via flagsMemObj (per-chunk barrier)

  // Phase-level timestamp instrumentation (optional, diagnostic).
  // When enabled, block 0 thread 0 of PipelinedAllReduceSdmaKernel writes
  // __builtin_amdgcn_s_memtime() at each phase boundary to phase_ts_d_.
  // Host calls get_phase_timestamps() to read back. See kernel header for slot layout.
  static constexpr size_t kPhaseTsCapacity = 32;
  bool phase_timing_enabled_ = false;
  uint64_t* phase_ts_d_ = nullptr;  // device buffer, capacity kPhaseTsCapacity * uint64_t
  int last_num_chunks_ = 0;  // numChunks used by the most recent pipelined() call

  // ---------------------------------------------------------------------
  // Copy-path instrumentation (diagnostic; baseline uses a single
  // hipMemcpyAsync in copy_output_to_user()). When enabled, we record a
  // hipEvent_t before and after that call, plus chrono around the host API
  // invocation. This is the ONLY way to prove how much time the copy
  // contributes, *before* we change the copy strategy (see rule R0).
  // ---------------------------------------------------------------------
  bool copy_timing_enabled_ = false;
  hipEvent_t copy_start_event_ = nullptr;
  hipEvent_t copy_end_event_ = nullptr;
  double copy_timing_host_us_ = 0.0;   // populated each copy_output_to_user() call
  bool copy_timing_recorded_ = false;  // indicates events were submitted

  // ---------------------------------------------------------------------
  // Post-AG in-kernel copy prototype (E' path).
  // When post_ag_wait_enabled_ is true, the AR kernel receives a per-chunk
  // device flag array (size = kMaxPostAgChunks, reset to 0 before each
  // launch) plus the user's output tensor pointer. Block 0 sets
  // post_ag_flag[c] = 1 when chunk c's AG completes on all peers;
  // compute blocks wait on each flag[c] in turn and copy the chunk's
  // transit region to the user output buffer in-kernel. This overlaps
  // chunk c's CU-side copy with block 0's wait on chunk c+1's AG,
  // eliminating the external hipMemcpyAsync in copy_output_to_user().
  //
  // Stage 2b: adds per-chunk flag + in-kernel copy to Stage 1's
  // spin-wait; kernel wall becomes max(AG wait + last chunk's CU copy).
  // When enabled AND copy_output_to_user_ is true, host skips the
  // external hipMemcpyAsync at pipelined() exit.
  // ---------------------------------------------------------------------
  bool post_ag_wait_enabled_ = false;
  static constexpr int kMaxPostAgChunks = 32;
  uint32_t* post_ag_flag_d_ = nullptr;  // device array of kMaxPostAgChunks uint32

  AllreduceSdma(const AllreduceSdma&) = delete;
  AllreduceSdma& operator=(const AllreduceSdma&) = delete;

  bool ensure_buffer_size(void*& buffer, std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                          size_t& current_size, application::SymmMemObjPtr& buffer_obj,
                          size_t required_size, const char* buffer_name);

  void copy_input_to_transit(T* input, size_t total_count, hipStream_t stream);
  void copy_output_to_user(T* output, size_t total_count, hipStream_t stream);

 public:
  /**
   * @brief Constructor
   * @param myPe Current PE ID
   * @param npes Total number of PEs
   * @param transit_buffer_size Output transit buffer size in bytes (default 512MB)
   * @param copy_output_to_user If true, copy result to user output buffer
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

  /**
   * @brief Pipelined AllReduce: overlapped SDMA scatter + reduce + SDMA AG.
   * @param input   Input data pointer (total_count elements on each rank)
   * @param output  Output data pointer (total_count elements, may alias input)
   * @param total_count Number of data elements per PE
   * @param chunk_elems Chunk size in elements (0 = auto)
   * @param scatter_mode 0 = SDMA scatter, 1 = P2P scatter
   * @param stream  HIP stream
   * @return true if successful
   */
  bool pipelined(T* input, T* output, size_t total_count,
                 size_t chunk_elems = 0, int scatter_mode = 0,
                 hipStream_t stream = nullptr,
                 bool external_scatter = false);

  application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }
  void* getOutputTransitBuffer() const { return output_transit_buffer_; }
  size_t getOutputTransitBufferSize() const { return output_transit_buffer_size_; }
  application::SymmMemObjPtr getOutputTransitBufferObj() const {
    return output_transit_buffer_obj_;
  }

  void resetFlags();

  // --- Phase-level timestamp instrumentation ---
  // Call enable(true) once before benchmarking; each subsequent pipelined()
  // call will write per-phase timestamps into a device buffer. Call
  // get_phase_timestamps() after the kernel completes (ensure stream synced)
  // to read back the most recent call's timestamps.
  void enable_phase_timing(bool on);
  bool is_phase_timing_enabled() const { return phase_timing_enabled_; }
  int get_last_num_chunks() const { return last_num_chunks_; }

  // Returns a vector of length kPhaseTsCapacity. Each slot is a raw
  // __builtin_amdgcn_s_memtime() value in GPU memory-clock cycles.
  // Slot layout (see pipelined_allreduce_sdma_kernel.hpp for details):
  //   0: kernel entry
  //   1: scatter submit done
  //   2 + 3*c + {0,1,2}: chunk c {compute-wait, cross-PE-barrier, AG-submit} done
  //   2 + 3*numChunks:   AG wait done
  //   3 + 3*numChunks:   block 0 exit
  std::vector<uint64_t> get_phase_timestamps();

  // --- Copy-path instrumentation (baseline: single hipMemcpyAsync) ---
  // Enable to record a {start,end} hipEvent around the hipMemcpyAsync in
  // copy_output_to_user(), plus chrono around the host API call. Read back
  // via get_copy_timing_ms() after the stream is synchronized.
  // Layout of the returned vector (length 2):
  //   [0]: host-side microseconds spent inside hipMemcpyAsync() host call
  //   [1]: GPU-side milliseconds between pre/post events (copy kernel wall)
  // get_copy_timing_last_num_chunks() returns 1 (baseline is a single copy).
  void enable_copy_timing(bool on);
  bool is_copy_timing_enabled() const { return copy_timing_enabled_; }
  std::vector<double> get_copy_timing_ms();
  int get_copy_timing_last_num_chunks() const { return 1; }

  // --- Post-AG wait prototype (Stage 1 of E') ---
  // Toggle whether compute blocks stay alive waiting for AG done.
  // Stage 1: compute blocks just spin and exit (cost measurement).
  // Stage 2 (future): compute blocks do in-kernel copy during the spin.
  void enable_post_ag_wait(bool on);
  bool is_post_ag_wait_enabled() const { return post_ag_wait_enabled_; }
};

}  // namespace collective
}  // namespace mori

#endif  // TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
