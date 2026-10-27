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
 * @file chunk_consumer.hpp
 * @brief Host consumer side of the device-to-host chunk format (private)
 *
 * Used by the host consumer of the SPSC and MPSC device-to-host queues.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mori/kiwi/queue/detail/chunk.hip.hpp>
#include <mori/kiwi/queue/detail/host_memory.hpp>
#include <mori/kiwi/queue/error.hpp>

#if !defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__)
#error "The device-to-host consumer requires single-copy-atomic 16-byte loads (x86-64)"
#endif

namespace mori::kiwi::queue::detail {

/// One aligned 16-byte load (movdqa), single-copy atomic on AVX-capable
/// x86-64 processors. The volatile asm is re-executed on every call because
/// the GPU writes the ring, and its memory clobber keeps the compiler from
/// moving other memory accesses (the consumer index publish) across it.
inline Chunk load_chunk(const Chunk* p) {
  Chunk c;
  asm volatile("movdqa %1, %0" : "=x"(c) : "m"(*p) : "memory");
  return c;
}

/**
 * @brief Lines between publications of the consumer index
 *
 * The consumer publishes when its position crosses a multiple of the
 * interval K, so the published index is at least position - K + 1. The
 * producer of the oldest unconsumed message (at most kMaxMessageLines lines)
 * then has credit while K + kMaxMessageLines - 1 <= capacity. K is the
 * largest power of two up to 64 that satisfies this, or 1 for small rings.
 */
inline uint64_t publish_interval(size_t capacity) {
  uint64_t k = 64;
  while (k > 1 && k + kMaxMessageLines - 1 > capacity) {
    k >>= 1;
  }
  return k;
}

/// True if moving the consumer from line `from` to line `to` crosses a
/// multiple of `interval` (a power of two).
inline bool crosses_publish_point(uint64_t from, uint64_t to, uint64_t interval) {
  return ((from ^ to) & ~(interval - 1)) != 0;
}

/// Allocates a ring of `capacity` lines in extended-scope host memory; every
/// chunk starts as lap -1 (epoch 0xff), which no consumer position expects.
inline Chunk* allocate_ring(size_t capacity) {
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    throw queue_error("Queue capacity must be a power of 2");
  }
  if (capacity > kMaxCapacityLines) {
    throw queue_error("Queue capacity exceeds kMaxCapacityLines");
  }
  Chunk* ring = host_malloc_uc<Chunk>(capacity * kChunksPerLine);
  for (size_t i = 0; i < capacity * kChunksPerLine; ++i) {
    ring[i] = Chunk{0, make_signal(0xff, 0, 0)};
  }
  return ring;
}

/// Raises the published consumer index to `value` unless it is already past
/// it, so an SPSC device_handle() that races pop() never moves it backwards.
inline void publish_max(uint64_t* index, uint64_t value) {
  uint64_t published = __atomic_load_n(index, __ATOMIC_RELAXED);
  while (published < value && !__atomic_compare_exchange_n(index, &published, value, /*weak=*/false,
                                                           __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
  }
}

struct MessageInfo {
  uint32_t bytes;
  uint32_t metadata;
};

/// Out of line and cold, so that read_message() stays small enough to inline
/// into pop().
[[noreturn]] __attribute__((noinline, cold)) inline void throw_bad_message_length() {
  throw queue_error("device-to-host queue: message length exceeds kMaxMessageBytes");
}

/// Copies the low `n` (0-7) bytes of `word` to `dst` with constant-size
/// moves (x86-64 is little-endian); a variable-length memcpy would be a libc
/// call, which costs more than the rest of a small pop.
inline void copy_partial_word(unsigned char* dst, uint64_t word, size_t n) {
  if (n & 4) {
    const uint32_t part = static_cast<uint32_t>(word);
    std::memcpy(dst, &part, 4);
    dst += 4;
    word >>= 32;
  }
  if (n & 2) {
    const uint16_t part = static_cast<uint16_t>(word);
    std::memcpy(dst, &part, 2);
    dst += 2;
    word >>= 16;
  }
  if (n & 1) {
    *dst = static_cast<unsigned char>(word);
  }
}

/// Loads the four chunks of absolute line `line` into `c`; true if all of
/// them carry the line's lap epoch.
inline bool load_line(const Chunk* ring, uint64_t mask, uint32_t log2_capacity, uint64_t line,
                      Chunk (&c)[kChunksPerLine]) {
  const Chunk* chunks = ring + (line & mask) * kChunksPerLine;
  const uint8_t epoch = line_epoch(line, log2_capacity);
  for (size_t j = 0; j < kChunksPerLine; ++j) {
    c[j] = load_chunk(chunks + j);
  }
  // The four signal words as two vectors, all epochs compared at once.
  const Chunk signals01 = {c[0][1], c[1][1]};
  const Chunk signals23 = {c[2][1], c[3][1]};
  const Chunk stale = ((signals01 ^ epoch) | (signals23 ^ epoch)) & 0xff;
  return (stale[0] | stale[1]) == 0;
}

/**
 * @brief Copy the message starting at absolute line `head` if it is whole
 *
 * Validates chunk 0, then every other chunk of the message's lines, padding
 * included, against the lap of its own line. Returns false if any of them
 * is not visible yet. Otherwise copies exactly the message's bytes into
 * `buf`, fills `info`, and returns true.
 *
 * Checking the padding too means the consumer moves past a line only once
 * all four of its chunks show the line's lap. A producer of the next lap
 * writes the line only after that, so a late padding store from one
 * wavefront can never land on top of another wavefront's next-lap chunk.
 *
 * This runs once per message on the proxy thread, so it loads a whole line
 * at a time, compares its four epochs at once, and copies with
 * constant-size moves only.
 *
 * @throws queue_error if chunk 0 is visible but its length exceeds
 *         kMaxMessageBytes (a corrupted ring).
 */
inline bool read_message(const Chunk* ring, uint64_t mask, uint32_t log2_capacity, uint64_t head,
                         void* buf, MessageInfo& info) {
  Chunk c[kChunksPerLine];
  const bool line_visible = load_line(ring, mask, log2_capacity, head, c);
  const uint64_t signal = c[0][1];
  if (signal_epoch(signal) != line_epoch(head, log2_capacity)) {
    return false;
  }
  const uint32_t len = signal_bytes(signal);
  if (len > kMaxMessageBytes) {
    throw_bad_message_length();
  }
  if (!line_visible) {
    return false;  // chunk 0 is visible, another chunk of its line not yet
  }

  // Whole payload words straight from the validated lines. The partial
  // last word, if any, is re-read at the end; that is still the validated
  // chunk, because a producer rewrites a line only after the consumer
  // position has moved past it.
  unsigned char* out = static_cast<unsigned char*>(buf);
  const size_t whole_words = len / kChunkPayloadBytes;
  const size_t lines = message_lines(len);
  for (size_t l = 0;;) {
    for (size_t j = 0; j < kChunksPerLine; ++j) {
      const size_t k = l * kChunksPerLine + j;
      if (k < whole_words) {
        const uint64_t word = c[j][0];
        std::memcpy(out + k * kChunkPayloadBytes, &word, kChunkPayloadBytes);
      }
    }
    if (++l == lines) {
      break;
    }
    if (!load_line(ring, mask, log2_capacity, head + l, c)) {
      return false;  // a later line is not visible yet
    }
  }
  if (const size_t tail = len % kChunkPayloadBytes; tail != 0) {
    const uint64_t line = head + whole_words / kChunksPerLine;
    const Chunk last =
        load_chunk(ring + (line & mask) * kChunksPerLine + whole_words % kChunksPerLine);
    copy_partial_word(out + whole_words * kChunkPayloadBytes, last[0], tail);
  }
  info = MessageInfo{len, signal_metadata(signal)};
  return true;
}

/**
 * @brief The single-consumer pop of the device-to-host queues
 *
 * Copies the message at the private position `head` if it is whole, moves
 * `head` past it, and publishes the new position to `consumer_index` when it
 * crosses a multiple of `interval`. Returns false and changes nothing if the
 * message is not fully visible. Forced inline: pop() is the proxy thread's
 * hot path, and a call would also keep `head` in memory.
 */
__attribute__((always_inline)) inline bool pop_message(const Chunk* ring, uint64_t mask,
                                                       uint32_t log2_capacity, uint64_t interval,
                                                       uint64_t& head, uint64_t* consumer_index,
                                                       void* buf, uint32_t* metadata,
                                                       size_t* bytes) {
  const uint64_t from = head;
  MessageInfo info;
  if (!read_message(ring, mask, log2_capacity, from, buf, info)) {
    return false;
  }
  const uint64_t next = from + message_lines(info.bytes);
  head = next;
  // x86 does not reorder this store before the ring loads above.
  if (crosses_publish_point(from, next, interval)) {
    __atomic_store_n(consumer_index, next, __ATOMIC_RELEASE);
  }
  if (metadata != nullptr) {
    *metadata = info.metadata;
  }
  if (bytes != nullptr) {
    *bytes = info.bytes;
  }
  return true;
}

}  // namespace mori::kiwi::queue::detail
