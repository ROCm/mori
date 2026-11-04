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
 * @file chunk.hip.hpp
 * @brief Device-to-host wire format and device push API
 *
 * Ring layout. A device-to-host ring is `capacity` 64-byte lines of four
 * 16-byte chunks. Lane c of the pushing wavefront writes chunk c of a message
 * with one 16-byte store (global_store_dwordx4). A chunk is two 8-byte
 * halves, each 4 B of data and a 4 B signal (the LL8 format):
 *
 *   chunk = 16 B  +---------+---------+---------+---------+
 *                 | data lo |  sig 0  | data hi |  sig 1  |
 *                 |   4 B   |   4 B   |   4 B   |   4 B   |
 *                 +---------+---------+---------+---------+
 *
 *   data:  payload bytes [8c, 8c + 4) and [8c + 4, 8c + 8) of the message.
 *          Chunks wholly past the payload carry zero; the bytes of the last
 *          word past the payload are unspecified (push_shared/push_global/
 *          push_lanes pass them through) and never returned by pop().
 *   sig 0: bits  0-7   epoch = (line index >> log2(capacity)) & 0xff, in every
 *                      signal, including the padding chunks of the last line
 *          bits  8-15  metadata bits 0-7   (chunk 0 only)
 *          bits 16-31  payload length in bytes  (chunk 0 only)
 *   sig 1: bits  0-7   epoch, as in sig 0
 *          bits  8-31  metadata bits 8-31  (chunk 0 only)
 *
 * The 32-bit metadata is e.g. the invoke callback id. The code handles a
 * chunk's two signals as one 64-bit signal pair, sig 0 | sig 1 << 32.
 *
 * A message of B bytes uses message_chunks(B) = max(1, ceil(B / 8)) chunks,
 * at most one per lane, so B <= kMaxMessageBytes = 512. It starts on a line
 * and occupies message_lines(B) whole lines; a message that reaches the end
 * of the ring continues at line 0 of the next lap.
 *
 * The host consumes a message only when every signal of its lines, padding
 * included, carries the lap epoch of its own line, so a partly visible
 * message is never consumed. Each signal guards the 4 data bytes beside it,
 * so this relies only on aligned 8-byte halves of a GPU store to host memory
 * and of a host load being single-copy atomic, not on 16-byte atomicity.
 * Producers never write line i + capacity before the consumer has published
 * a position past line i, so a line holds either the expected lap or the one
 * before it, and an 8-bit epoch tells them apart. Empty lines start as lap -1
 * (epoch 0xff).
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../error.hpp"
#include "address_space.hpp"

// Every device-to-host push is wavefront-collective over 64 lanes (one chunk
// per lane). GFX9 (CDNA: gfx90a, gfx942, gfx950) runs only wave64.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX9__)
#error "KIWI device-to-host queues require a wave64 GFX9 target"
#endif

namespace mori::kiwi::queue {

/// Largest device-to-host message: one 8-byte payload word per lane.
inline constexpr size_t kMaxMessageBytes = 512;

/// Ring lines a kMaxMessageBytes message occupies. A ring must hold at least
/// message_lines(B) lines for the largest message B it carries.
inline constexpr size_t kMaxMessageLines = 16;

/// Largest device-to-host ring, in lines. Lines in flight then fit in 32
/// bits, which keeps the producer's credit check on the scalar unit.
inline constexpr size_t kMaxCapacityLines = size_t{1} << 31;

namespace detail {

/// One 16-byte chunk: {data lo, sig 0, data hi, sig 1}. A vector type, so a
/// chunk is moved with one 16-byte access.
using Chunk = uint32_t __attribute__((ext_vector_type(4)));

inline constexpr size_t kChunkPayloadBytes = 8;
inline constexpr size_t kChunksPerLine = 4;
inline constexpr size_t kLineBytes = kChunksPerLine * sizeof(Chunk);

/// Chunks a message of `bytes` payload bytes uses (a header-only message
/// still uses one).
__host__ __device__ constexpr size_t message_chunks(size_t bytes) {
  return bytes == 0 ? 1 : (bytes + kChunkPayloadBytes - 1) / kChunkPayloadBytes;
}

}  // namespace detail

/// Ring lines a message of `bytes` payload bytes occupies.
__host__ __device__ constexpr size_t message_lines(size_t bytes) {
  return (detail::message_chunks(bytes) + detail::kChunksPerLine - 1) / detail::kChunksPerLine;
}

static_assert(message_lines(kMaxMessageBytes) == kMaxMessageLines);

namespace detail {

/// Signal pair (sig 0 | sig 1 << 32) of a chunk; `bytes` and `metadata` are
/// nonzero only in chunk 0.
__host__ __device__ constexpr uint64_t make_signal(uint8_t epoch, uint32_t bytes,
                                                   uint32_t metadata) {
  return static_cast<uint64_t>(epoch) * 0x100000001u |
         (static_cast<uint64_t>(metadata & 0xffu) << 8) |
         (static_cast<uint64_t>(bytes & 0xffffu) << 16) |
         (static_cast<uint64_t>(metadata >> 8) << 40);
}
/// Epoch of sig 0.
__host__ __device__ constexpr uint8_t signal_epoch(uint64_t signal) {
  return static_cast<uint8_t>(signal);
}
__host__ __device__ constexpr uint32_t signal_bytes(uint64_t signal) {
  return static_cast<uint32_t>(signal >> 16) & 0xffffu;
}
__host__ __device__ constexpr uint32_t signal_metadata(uint64_t signal) {
  return (static_cast<uint32_t>(signal >> 8) & 0xffu) | static_cast<uint32_t>(signal >> 40) << 8;
}

/// The chunk carrying payload word `data` and signal pair `signal`.
__host__ __device__ constexpr Chunk make_chunk(uint64_t data, uint64_t signal) {
  return Chunk{static_cast<uint32_t>(data), static_cast<uint32_t>(signal),
               static_cast<uint32_t>(data >> 32), static_cast<uint32_t>(signal >> 32)};
}
/// Payload word of a chunk.
__host__ __device__ constexpr uint64_t chunk_data(const Chunk& c) {
  return c[0] | static_cast<uint64_t>(c[2]) << 32;
}
/// Signal pair of a chunk.
__host__ __device__ constexpr uint64_t chunk_signal(const Chunk& c) {
  return c[1] | static_cast<uint64_t>(c[3]) << 32;
}

/// Lap epoch of absolute line index `line`.
__host__ __device__ constexpr uint8_t line_epoch(uint64_t line, uint32_t log2_capacity) {
  return static_cast<uint8_t>(line >> log2_capacity);
}

/// Lane 0's value in every lane, kept wave-uniform (v_readfirstlane_b32, so
/// the result lives in SGPRs). The whole wavefront must be active.
__device__ __forceinline__ uint64_t broadcast_lane0(uint64_t value) {
  const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(value));
  const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(value >> 32));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

/// Lane 0's copy of `value` in every lane, word by word, so the result lives
/// in SGPRs. Use it on a device handle loaded from memory at a per-wave
/// index: the compiler cannot prove such a load wave-uniform, so the handle
/// (queue indices, pointers) would otherwise stay in VGPRs and every push
/// would run per-lane compares and branches. One wave must own the value and
/// the whole wavefront must be active.
template <typename T>
__device__ __forceinline__ T wave_uniform(const T& value) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0, "whole 32-bit words only");
  uint32_t words[sizeof(T) / sizeof(uint32_t)];
  __builtin_memcpy(words, &value, sizeof(T));
#pragma unroll
  for (uint32_t& w : words) w = __builtin_amdgcn_readfirstlane(w);
  T out;
  __builtin_memcpy(&out, words, sizeof(T));
  return out;
}

/// Published consumer index, re-read from host memory on every call.
__device__ __forceinline__ uint64_t load_published(global_ptr<const uint64_t> index) {
  return __hip_atomic_load(index, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

/// Payload word `__lane_id()` of `message`. Every word is read at a constant
/// offset and picked with selects, so a message held in registers stays there
/// (no dynamically indexed private array, no scratch).
template <typename T>
__device__ __forceinline__ uint64_t select_lane_word(const T& message) {
  constexpr size_t bytes = sizeof(T);
  const unsigned char* src = reinterpret_cast<const unsigned char*>(&message);
  const size_t lane = __lane_id();
  uint64_t selected = 0;
#pragma unroll
  for (size_t k = 0; k < message_chunks(bytes); ++k) {
    uint64_t word = 0;
    if (k < bytes / kChunkPayloadBytes) {
      __builtin_memcpy(&word, src + k * kChunkPayloadBytes, kChunkPayloadBytes);
    } else {
      __builtin_memcpy(&word, src + k * kChunkPayloadBytes, bytes % kChunkPayloadBytes);
    }
    selected = lane == k ? word : selected;
  }
  return selected;
}

/// Lane c stores chunk c of a `bytes`-byte message of `lines` lines starting
/// at absolute line `base`; lanes past the payload write zero data, lanes past
/// the last line write nothing.
__device__ __forceinline__ void write_message(global_ptr<Chunk> ring, uint32_t log2_capacity,
                                              uint64_t base, uint64_t word, uint32_t bytes,
                                              uint32_t metadata, uint64_t lines) {
  const size_t lane = __lane_id();
  if (lane < lines * kChunksPerLine) {
    const uint64_t line = base + lane / kChunksPerLine;
    const uint8_t epoch = line_epoch(line, log2_capacity);
    const uint64_t data = lane * kChunkPayloadBytes < bytes ? word : 0;
    // Lane 0's header carries no epoch, so it is loop-invariant in a push
    // loop; every lane ORs in its line's epoch (the low byte of each
    // signal).
    const uint64_t header = lane == 0 ? make_signal(0, bytes, metadata) : 0;
    const uint64_t signal = header | make_signal(epoch, 0, 0);
    const uint64_t mask = (uint64_t{1} << log2_capacity) - 1;
    ring[(line & mask) * kChunksPerLine + lane % kChunksPerLine] = make_chunk(data, signal);
  }
}

/// Debug builds trap unless all 64 lanes call the push (the wavefront
/// contract); release builds check nothing.
__device__ __forceinline__ void assert_full_wave() {
  KIWI_QUEUE_ASSERT(__builtin_amdgcn_read_exec() == ~uint64_t{0});
}

/// Wave-uniform message length; traps on a message that can never be framed.
__device__ __forceinline__ uint32_t checked_bytes(size_t bytes) {
  if (bytes > kMaxMessageBytes) {
    __builtin_trap();
  }
  return static_cast<uint32_t>(bytes);
}

/**
 * @brief Payload-source front ends of a device-to-host queue handle
 *
 * `Handle` provides push_lanes(word, bytes, metadata), the core push. All
 * device operations are wavefront-collective: all 64 lanes call them with
 * identical arguments.
 */
template <typename Handle>
struct PushApi {
  /**
   * @brief Push `message` (sizeof(T) bytes) held in registers
   *
   * Lane c selects payload word c with compile-time offsets, so the
   * message is never placed in scratch. The whole message lives in every
   * lane's registers; prefer push_shared() for large messages built in LDS.
   */
  template <typename T>
  __device__ bool push_reg(const T& message, uint32_t metadata = 0) {
    static_assert(std::is_trivially_copyable_v<T>, "push_reg() copies the message bytes");
    static_assert(sizeof(T) <= kMaxMessageBytes, "message exceeds kMaxMessageBytes");
    return handle().push_lanes(select_lane_word(message), sizeof(T), metadata);
  }

  /**
   * @brief Push `bytes` bytes staged in shared memory
   *
   * Lane c reads payload word c (one ds_read_b64). `lds` is 8-byte aligned
   * shared memory readable for round_up(bytes, 8) bytes and published (LDS
   * barrier) before the call. Bytes of the last word past `bytes` are
   * written to the ring but never returned by pop().
   */
  __device__ bool push_shared(const void* lds, size_t bytes, uint32_t metadata = 0) {
    using LdsWord = const __attribute__((address_space(3))) uint64_t;
    return push_words((LdsWord*)lds, bytes, metadata);  // address-space cast
  }

  /**
   * @brief Push `bytes` bytes from global memory
   *
   * Same contract as push_shared() for an 8-byte aligned global pointer
   * (one global_load_dwordx2 per lane).
   */
  __device__ bool push_global(const void* ptr, size_t bytes, uint32_t metadata = 0) {
    return push_words((global_ptr<const uint64_t>)ptr, bytes, metadata);  // address-space cast
  }

 private:
  __device__ Handle& handle() { return static_cast<Handle&>(*this); }

  template <typename WordPtr>
  __device__ bool push_words(WordPtr words, size_t bytes, uint32_t metadata) {
    const uint32_t n = checked_bytes(bytes);
    const size_t lane = __lane_id();
    const uint64_t word = lane * kChunkPayloadBytes < n ? words[lane] : 0;
    return handle().push_lanes(word, n, metadata);
  }
};

/**
 * @brief Cached readiness state for a device-to-host queue's host consumer
 *
 * ready() is one load of the next message's chunk-0 sig 0 and an epoch
 * compare; it never reads the shared consumer index. Call refresh() after
 * any operation that may have consumed messages (pop(), invoke
 * Context::progress()). ready() is a hint: it can be true while pop() still
 * returns false, when chunk 0 is visible before the message's other chunks.
 */
template <typename Queue>
class HostPollState {
 public:
  __host__ bool ready() const { return signal_epoch(*signal_) == expected_epoch_; }

  /// Rebind to the queue's current consumer position.
  __host__ void refresh() {
    const uint64_t head = queue_->consumer_position();
    signal_ = reinterpret_cast<const volatile uint32_t*>(
                  queue_->chunks_ + (head & queue_->capacity_mask_) * kChunksPerLine) +
              1;
    expected_epoch_ = line_epoch(head, queue_->log2_capacity_);
  }

 private:
  friend Queue;

  __host__ explicit HostPollState(const Queue* queue) : queue_(queue) { refresh(); }

  const Queue* queue_;
  const volatile uint32_t* signal_ = nullptr;  ///< sig 0; forces a fresh coherent load
  uint8_t expected_epoch_ = 0;
};

}  // namespace detail
}  // namespace mori::kiwi::queue
