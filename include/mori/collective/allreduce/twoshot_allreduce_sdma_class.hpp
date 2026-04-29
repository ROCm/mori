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
#include <array>
#include <list>
#include <memory>
#include <unordered_map>
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
  static constexpr int kMaxTrackedSdmaQueues = 16;
  std::array<uint64_t, kMaxTrackedSdmaQueues> pipeline_ag_gen_by_q_{};  // for MORI_MULTI_Q_AG
  uint64_t* pipeline_ag_gen_by_q_d_ = nullptr;
  uint32_t sdma_num_queue_ = 0;

  // Phase-level timestamp instrumentation (optional, diagnostic).
  // When enabled, block 0 thread 0 of PipelinedAllReduceSdmaKernel writes
  // __builtin_amdgcn_s_memtime() at each phase boundary to phase_ts_d_.
  // Host calls get_phase_timestamps() to read back. See kernel header for slot layout.
  // Capacity must hold disjoint phase ranges from
  // pipelined_allreduce_sdma_kernel.hpp:
  //   block0 range: historical slots 0..3+3*numChunks
  //   AG-done range: base 64
  //   first compute/R-block range: base 88
  //   first Plan-A A-block range: base 144
  // 256 leaves headroom for up to ~32 chunks across all ranges. Was 32 which
  // OOB'd at numChunks=8 (Plan A) and broke barrier memory → deadlock.
  // MUST stay in sync with kArPhaseTsCapacity in
  // pipelined_allreduce_sdma_kernel.hpp.
  static constexpr size_t kPhaseTsCapacity = 256;
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
  // Post-AG wait prototype (Stage 1 of E' — in-kernel post-AG CU copy).
  // When post_ag_wait_enabled_ is true, kernel launches are given a device
  // uint32 flag buffer. Compute blocks wait on this flag after their
  // reduce phase (instead of exiting early); block 0 sets the flag once
  // AG wait completes. Stage 1 does NOT do any copy yet — it only measures
  // how much "compute blocks stay alive during AG wait" costs in terms of
  // wall time and GEMM interference. If the cost is acceptable, Stage 2
  // adds the in-kernel transit→user_output copy inside the post-AG phase.
  // ---------------------------------------------------------------------
  bool post_ag_wait_enabled_ = false;
  uint32_t* post_ag_flag_d_ = nullptr;  // device uint32, reset to 0 each call

  // Plan A (perf_history Entry 18): when direct_output_enabled_ is true
  // AND copy_output_to_user_ is true AND the call is MULTI_CHUNK,
  // pipelined() passes the user output pointer directly to the kernel,
  // block 0 SKIPS SDMA AG, and compute blocks do CU XGMI AG (read peer
  // transit via P2P pointer) + direct write to user_output. The external
  // hipMemcpyAsync is SKIPPED.
  // Requires post_ag_wait mechanism (to gate compute blocks until all
  // peers' reduce barriers are complete) — enabled automatically when
  // direct_output is on.
  bool direct_output_enabled_ = false;

  // ---------------------------------------------------------------------
  // D' prototype: lazy-register user output buffer as shmem symmetric
  // memory so AR kernel can AG directly to it, skipping the transit
  // buffer AND the external hipMemcpyAsync entirely.
  //
  // Register is a collective call (3 allgathers + IPC open over N peers),
  // so it is expensive (~ms). We maintain a small LRU cache keyed by
  // the user output ptr + size. When the same output is reused across
  // calls (typical in training loops), the cache hits and the fast path
  // costs 0 extra host time.
  //
  // When register_user_output_enabled_ is true AND the cache lookup
  // returns a valid SymmMemObj, pipelined() uses it as the kernel's
  // destination symm obj instead of output_transit_buffer_obj_. Otherwise
  // falls back to the transit + copy_output_to_user path (baseline).
  //
  // Cache size is intentionally small (LRU) to bound VRAM metadata
  // overhead; only a few distinct output ptrs are usually seen.
  // ---------------------------------------------------------------------
  bool register_user_output_enabled_ = false;
  static constexpr size_t kUserOutputCacheCap = 4;
  struct UserOutputCacheKey {
    void* ptr;
    size_t size;
    bool operator==(const UserOutputCacheKey& o) const {
      return ptr == o.ptr && size == o.size;
    }
  };
  struct UserOutputCacheKeyHash {
    size_t operator()(const UserOutputCacheKey& k) const noexcept {
      return std::hash<void*>()(k.ptr) ^ (std::hash<size_t>()(k.size) << 1);
    }
  };
  std::unordered_map<UserOutputCacheKey,
                     application::SymmMemObjPtr,
                     UserOutputCacheKeyHash> user_output_cache_;
  std::list<UserOutputCacheKey> user_output_cache_lru_;  // front = MRU

  // Instrumentation: last register() / lookup stats
  double last_register_us_ = 0.0;       // host wall of most recent register
  bool last_register_was_hit_ = false;  // true = cache hit, false = miss (did register)
  uint64_t cache_hits_ = 0;
  uint64_t cache_misses_ = 0;

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

  // --- Plan A: direct-output CU XGMI AG (perf_history Entry 18) ---
  // When on AND copy_output_to_user is on, the AR kernel's compute
  // blocks replace SDMA AG + external hipMemcpyAsync with CU XGMI reads
  // from peer transits + direct writes to user_output. Measured CU
  // XGMI BW = 370 GB/s (at 16 blocks/peer, 32 MB/peer shard), so the
  // 1.18 ms AG+copy phase should drop to ~0.61 ms per AR.
  // Only applies to MULTI_CHUNK path; single-chunk falls back to
  // legacy SDMA AG regardless of this flag.
  void enable_direct_output(bool on);
  bool is_direct_output_enabled() const { return direct_output_enabled_; }

  // --- D' prototype: lazy-register user output as symm memory --------
  // Turn on/off the fast path. When on, pipelined() tries to use the
  // user output buffer directly as the AR kernel's destination symm
  // memory. Register is a collective call — all ranks must enable/call
  // with the same size for this to work (AR semantics already require
  // symmetric output sizes, so this lines up).
  void enable_register_user_output(bool on);
  bool is_register_user_output_enabled() const { return register_user_output_enabled_; }

  // Pre-register a user output buffer. All ranks must call collectively
  // with the same size. Returns true on success. Caller can call this
  // before entering the hot loop to guarantee cache hits during benchmarking.
  bool register_user_output(void* ptr, size_t size);

  // Diagnostic: read back the last register() host us (0 on cache hit),
  // whether last lookup was a hit, and cumulative hit/miss counters.
  double last_register_us() const { return last_register_us_; }
  bool last_register_was_hit() const { return last_register_was_hit_; }
  uint64_t cache_hits() const { return cache_hits_; }
  uint64_t cache_misses() const { return cache_misses_; }
};

}  // namespace collective
}  // namespace mori

#endif  // TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
