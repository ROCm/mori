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
 * @file device_to_host_queue.hip.hpp
 * @brief Queue<Direction::DeviceToHost, Concurrency::SPSC>: GPU-push, CPU-pop
 *
 * One GPU wavefront pushes, one CPU thread pops, in the LL8 chunk format
 * of detail/chunk.hip.hpp. The producer keeps its index and a cached copy of
 * the consumer index in registers and reads the published consumer index
 * over PCIe only when the ring looks full. No read-modify-write atomics.
 *
 * The consumer keeps its position privately and publishes it to the shared
 * consumer index in batches (each time it crosses a multiple of the publish
 * interval), so the producer sees credit at most interval - 1 lines late.
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
 * @brief Single-producer single-consumer device-to-host queue
 *
 * Capacity is in 64-byte lines. Messages are 0 to kMaxMessageBytes bytes
 * plus 32-bit metadata.
 */
template <>
class Queue<Direction::DeviceToHost, Concurrency::SPSC> {
 public:
  /**
   * @brief Device-side producer handle, passed by value to kernels
   *
   * The indices stay in registers during a kernel and are not written
   * back; take a new handle (or save it, as invoke does) for the next
   * launch. Device operations are wavefront-collective: all 64 lanes call
   * them with identical arguments. Every push returns false if the ring is
   * full (retry later); nothing is written then.
   */
  struct DeviceHandle : detail::PushApi<DeviceHandle> {
    detail::global_ptr<detail::Chunk> chunks;           ///< ring, extended-scope host memory
    detail::global_ptr<const uint64_t> consumer_index;  ///< published consumer line, host memory
    uint64_t producer_index;                            ///< next line to write (register)
    uint64_t consumer_index_cached;                     ///< credit cache (register)
    uint32_t log2_capacity;                             ///< log2(capacity in lines)

    /**
     * @brief Push one message whose payload word i is lane i's `word`
     *
     * The core push; push_reg(), push_shared() and push_global() call it.
     * Words of lanes at or past `bytes` are ignored. `bytes` is taken
     * from lane 0; bytes > kMaxMessageBytes traps, and so does a message
     * longer than the ring, which could never fit.
     */
    __device__ bool push_lanes(uint64_t word, uint32_t bytes, uint32_t metadata) {
      detail::assert_full_wave();
      bytes = __builtin_amdgcn_readfirstlane(bytes);
      if (bytes > kMaxMessageBytes) {
        __builtin_trap();
      }
      const uint64_t lines = message_lines(bytes);
      const uint64_t cap = uint64_t{1} << log2_capacity;
      // Credit check in 32 bits: lines in flight never exceed the
      // capacity, which the host keeps at most kMaxCapacityLines, so
      // used + lines cannot wrap. A 64-bit unsigned compare has no SALU
      // form and would run on the VALU with a vcc round trip per push.
      const uint32_t used = static_cast<uint32_t>(producer_index - consumer_index_cached);
      bool ok = true;
      if (__builtin_expect(used + static_cast<uint32_t>(lines) > static_cast<uint32_t>(cap), 0)) {
        // Rare path: lane 0 refreshes the credit over PCIe.
        if (lines > cap) {
          __builtin_trap();
        }
        uint64_t fresh = 0;
        if (__lane_id() == 0) {
          fresh = detail::load_published(consumer_index);
        }
        consumer_index_cached = detail::broadcast_lane0(fresh);
        ok = static_cast<uint32_t>(producer_index - consumer_index_cached) +
                 static_cast<uint32_t>(lines) <=
             static_cast<uint32_t>(cap);
      }
      // Write under ok instead of returning early from the refresh
      // path: a caller's while (!push) loop then has one exit test, and
      // its steady state runs straight through with one taken branch.
      if (__builtin_expect(ok, 1)) {
        detail::write_message(chunks, log2_capacity, producer_index, word, bytes, metadata, lines);
        producer_index += lines;
      }
      return ok;
    }
  };

  /// Cached readiness state for the single host consumer.
  using HostPollState = detail::HostPollState<Queue>;

  /// @param capacity ring size in 64-byte lines, a non-zero power of two of
  ///        at most kMaxCapacityLines.
  ///        It must hold the largest message pushed (message_lines(bytes)).
  __host__ explicit Queue(size_t capacity);
  __host__ ~Queue();

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  Queue(Queue&&) noexcept = delete;
  Queue& operator=(Queue&&) noexcept = delete;

  /**
   * @brief Pop one whole message (CPU-side, non-blocking)
   *
   * Returns false and consumes nothing if the next message, or any chunk
   * of it, is not visible yet (the contents of `buf` are then unspecified).
   * Otherwise copies exactly the message's bytes into `buf`.
   *
   * @param buf Output buffer of at least the largest message pushed
   *        (at most kMaxMessageBytes bytes).
   * @param metadata Optional out: the message's 32-bit metadata.
   * @param bytes Optional out: the message's length.
   * @throws queue_error on a corrupted ring (a visible length above
   *         kMaxMessageBytes); nothing is consumed.
   */
  __host__ bool pop(void* buf, uint32_t* metadata = nullptr, size_t* bytes = nullptr);

  /// Readiness state synchronized to the current consumer position.
  __host__ HostPollState host_poll_state() const { return HostPollState{this}; }

  /// Ring capacity in 64-byte lines.
  __host__ size_t capacity() const { return capacity_; }

  /**
   * @brief Device handle for passing to kernels by value
   *
   * Publishes the exact consumer position and starts the producer there,
   * so the queue must be drained (every message of earlier launches
   * popped, no pop() in progress) before a new handle is taken.
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
  uint64_t* consumer_index_;   ///< extended-scope host memory, published position
  uint64_t head_;              ///< private consumer position (lines)
};

}  // namespace mori::kiwi::queue
