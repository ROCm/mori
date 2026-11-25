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
/**
 * @file mpsc_device_to_host_queue.hip.hpp
 * @brief Queue<Direction::DeviceToHost, Concurrency::MPSC>: many GPU
 *        wavefronts push, one CPU thread pops
 *
 * Same ring format, push API and consumer as the SPSC queue
 * (device_to_host_queue.hip.hpp, detail/chunk.hip.hpp); the queues differ
 * only in how a producer finds where to write.
 *
 * Producer. All producers share one producer index (in lines) in device
 * memory. Per push, lane 0 claims message_lines(B) contiguous lines with one
 * agent-scope fetch-add and broadcasts the first line. If the handle's cached
 * credit does not cover the claim, lane 0 waits on the published consumer
 * index (host memory) with s_sleep. A claim cannot be given back, so a push
 * on a full ring waits instead of failing. Lane c then stores chunk c with
 * one 16-byte store, as in the SPSC queue. push_lane_masked() is the
 * exception: each active lane pushes its own message, the call claims all of
 * their lines with one fetch-add, and each lane stores all of its own
 * message's chunks.
 * push_active_lanes() is push_lane_masked() over the executing lanes, from divergent code
 * too; its first executing lane takes lane 0's role, and it waits for each claim's result
 * before returning, so its claims also follow the wavefront's program order.
 *
 * Consumer. pop() is the SPSC pop: it keeps its position privately, consumes
 * a message only when every chunk of its lines carries the lap of its own
 * line, and publishes its position after every pop. A producer that claimed
 * lines but has not written them yet holds up only the consumer. A claim of
 * line i + capacity waits until the published index passed i, so a line
 * holds either the expected lap or the one before it, as in the SPSC queue.
 *
 * Liveness, with a running consumer and any number of producers. A claim,
 * collective or lane-masked, takes at most `capacity` lines (a longer one
 * traps before claiming), and the fetch-add orders all claims. Every claim
 * before the oldest unwritten one is written, so the consumer pops up to
 * that claim's first line and publishes exactly that position; its producer,
 * which is running since it made the claim, then has credit for the whole
 * claim and writes it. By induction every claim is written: a waiting claim
 * depends only on older claims, never on a newer one.
 *
 * Ordering. Messages are popped in claim order. Every claim of a wavefront,
 * collective or lane-masked, is one relaxed fetch-add by lane 0, so a
 * wavefront's claims follow lane 0's program order (one thread's
 * read-modify-writes of one location are coherence-ordered); this needs no
 * ordering between atomics of different lanes. One wavefront's messages are
 * therefore popped in program order, and the active lanes of one
 * push_lane_masked() call in lane order; messages of different wavefronts
 * interleave.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "detail/chunk.hip.hpp"
#include "error.hpp"
#include "queue.hip.hpp"

namespace mori::cco::proxy::queue {

/**
 * @brief Multi-producer single-consumer device-to-host queue
 *
 * Any number of wavefronts may push concurrently through copies of the
 * DeviceHandle; exactly one CPU thread pops. Capacity is in 64-byte lines.
 */
template <>
class Queue<Direction::DeviceToHost, Concurrency::MPSC> {
 public:
  /**
   * @brief Device-side producer handle, passed by value to kernels
   *
   * All copies share the device-memory producer index, so handles need no
   * write-back between launches. `consumer_index_cached` is a copy's
   * credit cache; any earlier published value is a safe lower bound.
   * Device operations are wavefront-collective: all 64 lanes call them,
   * with identical arguments except push_lane_masked()'s per-lane
   * arguments; push_active_lanes() is the exception.
   */
  struct DeviceHandle : detail::PushApi<DeviceHandle> {
    detail::global_ptr<detail::Chunk> chunks;           ///< ring, extended-scope host memory
    detail::global_ptr<const uint64_t> consumer_index;  ///< published consumer line, host memory
    detail::global_ptr<uint64_t> producer_index;        ///< device memory, fetch-add target (lines)
    uint64_t consumer_index_cached;                     ///< credit cache (register)
    uint64_t divergent_consumer_index_cached;           ///< push_active_lanes()' cache, per lane
    uint32_t log2_capacity;                             ///< log2(capacity in lines)

    /**
     * @brief Push one message whose payload word i is lane i's `word`
     *
     * Same contract as the SPSC push_lanes(), except that a full ring
     * makes the push wait for the consumer: it always returns true.
     */
    __device__ bool push_lanes(uint64_t word, uint32_t bytes, uint32_t metadata) {
#ifdef MORI_CCO_PROXY_PUSH_UNSUPPORTED
      __builtin_trap();
      return false;
#else
      detail::assert_full_wave();
      bytes = __builtin_amdgcn_readfirstlane(bytes);
      if (bytes > kMaxMessageBytes) {
        __builtin_trap();
      }
      const uint64_t lines = message_lines(bytes);
      const uint64_t cap = uint64_t{1} << log2_capacity;
      if (lines > cap) {
        __builtin_trap();  // could never fit; trap before claiming lines
      }
      uint64_t base = 0;
      if (__lane_id() == 0) {
        base = __hip_atomic_fetch_add(producer_index, lines, __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT);
      }
      base = detail::broadcast_lane0(base);
      // base >= consumer position >= published >= cached: no wrap.
      if (base + lines - consumer_index_cached > cap) {
        uint64_t fresh = 0;
        if (__lane_id() == 0) {
          // The chunk stores below depend on this loop's exit, so
          // they cannot overwrite lines the consumer still reads.
          while (base + lines - (fresh = detail::load_published(consumer_index)) > cap) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        consumer_index_cached = detail::broadcast_lane0(fresh);
      }
      detail::write_message(chunks, log2_capacity, base, word, bytes, metadata, lines);
      return true;
#endif
    }

    /**
     * @brief Every active lane pushes its own `message`
     *
     * The SPSC push_lane_masked() contract: wavefront-collective (all
     * 64 lanes, converged; debug builds trap otherwise), lanes with
     * `active` false push nothing and their other arguments' values are
     * ignored (`message` must still be a valid, readable reference),
     * `bytes` > sizeof(T) traps, a call whose active lanes need more
     * lines than the ring holds traps, and the active lanes' messages
     * take consecutive lines in lane order. Lane 0 claims all of them
     * with one fetch-add, so they are popped in lane order, after this
     * wavefront's earlier pushes, and waits as in push_lanes() until the
     * consumer has freed the whole claim; each lane then stores every
     * chunk of its own message. Like push_lanes(), it always returns
     * true, and the handle's credit cache is updated as in it.
     */
    template <typename T>
    __device__ bool push_lane_masked(const T& message, uint32_t bytes, uint32_t metadata,
                                     bool active) {
      return push_own_messages</*kFullWave=*/true>(message, bytes, metadata, active);
    }

    /// push_lane_masked() over the executing lanes, so callable from divergent code; the first
    /// executing lane takes lane 0's role. Always returns true.
    template <typename T>
    __device__ bool push_active_lanes(const T& message, uint32_t bytes, uint32_t metadata) {
      return push_own_messages</*kFullWave=*/false>(message, bytes, metadata, true);
    }

   private:
    // Body of push_lane_masked() (kFullWave) and push_active_lanes().
    template <bool kFullWave, typename T>
    __device__ bool push_own_messages(const T& message, uint32_t bytes, uint32_t metadata,
                                      bool active) {
#ifdef MORI_CCO_PROXY_PUSH_UNSUPPORTED
      __builtin_trap();
      return false;
#else
      if constexpr (kFullWave) detail::assert_full_wave();
      const detail::LaneClaim c = detail::lane_masked_claim<T>(bytes, active, log2_capacity);
      if (c.total == 0) {
        return true;  // no active lane: claim nothing
      }
      // With a partial exec, broadcast_lane0() and lane_masked_claim() see only executing lanes.
      uint32_t leader = 0;
      if constexpr (!kFullWave) leader = __builtin_ctzll(__builtin_amdgcn_read_exec());
      uint64_t base = 0;
      if (__lane_id() == leader) {
        base = __hip_atomic_fetch_add(producer_index, c.total, __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT);
      }
      base = detail::broadcast_lane0(base);
      // The credit wait of push_lanes(), for the whole claim. Divergent calls keep their own
      // cache, so consumer_index_cached stays wave-uniform; the leader's copy decides.
      uint64_t& cached = kFullWave ? consumer_index_cached : divergent_consumer_index_cached;
      const uint64_t known = kFullWave ? cached : detail::broadcast_lane0(cached);
      const uint64_t cap = uint64_t{1} << log2_capacity;
      if (base + c.total - known > cap) {
        uint64_t fresh = 0;
        if (__lane_id() == leader) {
          // The chunk stores below depend on this loop's exit.
          while (base + c.total - (fresh = detail::load_published(consumer_index)) > cap) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        cached = detail::broadcast_lane0(fresh);
      }
      detail::write_lane_message(chunks, log2_capacity, base + c.prefix, message, bytes, metadata,
                                 c.lines);
      // Compiler fence: keeps the stores in this call. Sunk past a branch's join, they
      // would follow the other arm's claim and credit wait, which may wait on them.
      if constexpr (!kFullWave) __builtin_amdgcn_wave_barrier();
      return true;
#endif
    }
  };

  /// Cached readiness state for the single host consumer.
  using HostPollState = detail::HostPollState<Queue>;

  /// @param capacity ring size in 64-byte lines, a non-zero power of two.
  ///        It must hold the largest message pushed (message_lines(bytes)).
  __host__ explicit Queue(size_t capacity);
  __host__ ~Queue();

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  Queue(Queue&&) noexcept = delete;
  Queue& operator=(Queue&&) noexcept = delete;

  /**
   * @brief Pop one whole message (CPU-side, non-blocking, one thread)
   *
   * Same contract as the SPSC pop().
   */
  __host__ bool pop(void* buf, uint32_t* metadata = nullptr, size_t* bytes = nullptr);

  /// Readiness state synchronized to the current consumer position.
  __host__ HostPollState host_poll_state() const { return HostPollState{this}; }

  /// Ring capacity in 64-byte lines.
  __host__ size_t capacity() const { return capacity_; }

  /**
   * @brief Device handle for passing to kernels by value
   *
   * Copies may be used concurrently and across launches; the queue need
   * not be drained. The credit cache starts at the published index.
   */
  __host__ DeviceHandle device_handle() const;

 private:
  friend HostPollState;

  __host__ uint64_t consumer_position() const { return head_; }

  size_t capacity_;  ///< lines
  uint32_t log2_capacity_;
  uint64_t capacity_mask_;
  detail::Chunk* chunks_;     ///< extended-scope host memory, capacity_ * 4 chunks
  uint64_t* producer_index_;  ///< device memory (lines)
  uint64_t* consumer_index_;  ///< extended-scope host memory, published position
  uint64_t head_;             ///< private consumer position (lines)
};

}  // namespace mori::cco::proxy::queue
