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
 * consumer index after every pop, so the producer sees credit for every line
 * already popped.
 *
 * Lane-masked pushes. push_lane_masked() lets each lane push its own
 * message, but it stays wavefront-collective like every push: all 64 lanes
 * call it from converged code, and a lane with nothing to send passes
 * active = false. The producer index is a register that every lane must keep
 * identical, and a call from a divergent branch or a lane subset would
 * advance it in some lanes only. Neither can be repaired by sharing the index
 * through memory: lanes on the two sides of a divergent branch are
 * concurrent threads to the compiler, which may sink one side's plain store
 * past the other side's load, so both sides claim the same lines.
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
   * them, with identical arguments except push_lane_masked()'s per-lane
   * arguments. Every push returns false if the ring is full (retry later);
   * nothing is written then.
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
#endif
    }

    /**
     * @brief Every active lane pushes its own `message`
     *
     * Wavefront-collective: all 64 lanes call it, from converged code
     * (debug builds trap otherwise), each with its own message, length,
     * metadata and `active` flag. A lane with `active` false pushes
     * nothing, and the values of its other arguments are ignored; its
     * `message` must still be a valid reference to a readable T, as any
     * reference must, and may be read (clamp the index or pass a dummy
     * object for lanes past the end of an array). An active lane sends
     * the first `bytes` bytes of `message`; `bytes` > sizeof(T) traps in
     * every build. The active lanes' messages take consecutive lines in
     * lane order, so they are popped in lane order, after this
     * wavefront's earlier pushes; each lane stores every chunk of its own
     * message. The producer index advances in registers, as in the other
     * pushes, with which it mixes freely; no atomics.
     *
     * All-or-nothing like push_lanes(): if the ring lacks credit for the
     * lines of all active lanes, nothing is written and it returns false
     * (retry later). A retried call of up to `capacity` lines succeeds
     * once the consumer has popped every earlier message, because the
     * consumer publishes its position on every pop: the published index
     * then equals the producer index and the whole ring is free. A
     * longer call could never fit and traps in every build, like a
     * message longer than the ring; 64 messages take up to
     * 64 * kMaxMessageLines = 1024 lines.
     */
    template <typename T>
    __device__ bool push_lane_masked(const T& message, uint32_t bytes, uint32_t metadata,
                                     bool active) {
#ifdef MORI_CCO_PROXY_PUSH_UNSUPPORTED
      __builtin_trap();
      return false;
#else
      detail::assert_full_wave();
      const detail::LaneClaim c = detail::lane_masked_claim<T>(bytes, active, log2_capacity);
      // The credit check of push_lanes(), in 32 bits; c.total is at
      // most the capacity.
      const uint32_t cap = static_cast<uint32_t>(uint64_t{1} << log2_capacity);
      const uint32_t total = static_cast<uint32_t>(c.total);
      bool ok = true;
      if (__builtin_expect(
              static_cast<uint32_t>(producer_index - consumer_index_cached) + total > cap, 0)) {
        uint64_t fresh = 0;
        if (__lane_id() == 0) {
          fresh = detail::load_published(consumer_index);
        }
        consumer_index_cached = detail::broadcast_lane0(fresh);
        ok = static_cast<uint32_t>(producer_index - consumer_index_cached) + total <= cap;
      }
      if (__builtin_expect(ok, 1)) {
        detail::write_lane_message(chunks, log2_capacity, producer_index + c.prefix, message, bytes,
                                   metadata, c.lines);
        producer_index += c.total;
      }
      return ok;
#endif
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
  detail::Chunk* chunks_;     ///< extended-scope host memory, capacity_ * 4 chunks
  uint64_t* consumer_index_;  ///< extended-scope host memory, published position
  uint64_t head_;             ///< private consumer position (lines)
};

}  // namespace mori::cco::proxy::queue
