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
 * one 16-byte store, as in the SPSC queue.
 *
 * Consumer. pop() is the SPSC pop: it keeps its position privately, consumes
 * a message only when every chunk of its lines carries the lap of its own
 * line, and publishes its position in batches. A producer that claimed lines
 * but has not written them yet holds up only the consumer. A claim of line
 * i + capacity waits until the published index passed i, so a line holds
 * either the expected lap or the one before it, as in the SPSC queue.
 *
 * Liveness. The published index is at least position - interval + 1, so the
 * producer of the oldest unconsumed message always has credit; claims ahead
 * of it get credit as the consumer moves on.
 *
 * Ordering. Messages are popped in claim order. Lane 0 of a wavefront claims
 * in program order, so one wavefront's messages are popped in program order;
 * messages of different wavefronts interleave.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "detail/chunk.hip.hpp"
#include "error.hpp"
#include "queue.hip.hpp"

namespace mori::kiwi::queue {

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
   * Device operations are wavefront-collective: all 64 lanes call them
   * with identical arguments.
   */
  struct DeviceHandle : detail::PushApi<DeviceHandle> {
    detail::global_ptr<detail::Chunk> chunks;           ///< ring, extended-scope host memory
    detail::global_ptr<const uint64_t> consumer_index;  ///< published consumer line, host memory
    detail::global_ptr<uint64_t> producer_index;        ///< device memory, fetch-add target (lines)
    uint64_t consumer_index_cached;                     ///< credit cache (register)
    uint32_t log2_capacity;                             ///< log2(capacity in lines)

    /**
     * @brief Push one message whose payload word i is lane i's `word`
     *
     * Same contract as the SPSC push_lanes(), except that a full ring
     * makes the push wait for the consumer: it always returns true.
     */
    __device__ bool push_lanes(uint64_t word, uint32_t bytes, uint32_t metadata) {
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
  uint64_t publish_interval_;  ///< lines between consumer index publications
  detail::Chunk* chunks_;      ///< extended-scope host memory, capacity_ * 4 chunks
  uint64_t* producer_index_;   ///< device memory (lines)
  uint64_t* consumer_index_;   ///< extended-scope host memory, published position
  uint64_t head_;              ///< private consumer position (lines)
};

}  // namespace mori::kiwi::queue
