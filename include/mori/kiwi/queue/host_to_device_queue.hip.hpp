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
 * @file host_to_device_queue.hip.hpp
 * @brief Queue<Direction::HostToDevice, Concurrency::SPSC> — CPU-push, GPU-pop queue
 *
 * Host->device communication queue using:
 * - 64-byte wave-slots (56-byte payload in lanes 0-6, dedicated epoch in lane 7)
 * - Runtime slot memory selection (device_slots constructor parameter):
 *     true:           fine-grained device VRAM, GPU reads locally, CPU writes via WC+mfence
 *     false (default): extended-scope host memory, GPU reads via PCIe, CPU writes locally
 * - Full-wavefront (64 threads) cooperative pop
 * - Epoch-based signaling (eliminates need to clear slots)
 * - Cached consumer progress on CPU (reduces host memory reads)
 * - Batched consumer notification from GPU to CPU
 * - No atomics (single producer/consumer guarantee)
 *
 * Constraints:
 * - Single CPU thread pushes (single producer)
 * - Single GPU wavefront pops (single consumer)
 * - Payload limited to 56 bytes per wave-slot
 * - GPU kernel MUST call flush_consumer_index() before returning
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "error.hpp"
#include "queue.hip.hpp"
#include "slot.hip.hpp"

namespace mori::kiwi::queue {

/**
 * @brief Queue specialization for host->device communication
 *
 * Single-producer (CPU thread) single-consumer (GPU wavefront) queue.
 * Push is a host method. Pop requires a full 64-thread wavefront.
 */
template <>
class Queue<Direction::HostToDevice, Concurrency::SPSC> {
 public:
  /**
   * @brief Lightweight device-side handle for kernel arguments
   *
   * Passed BY VALUE to kernels so all scalar fields land in SGPRs.
   * Pop uses wavefront-cooperative reads with __shfl for data gathering.
   *
   * IMPORTANT: Call flush_consumer_index() before the kernel returns,
   * otherwise the CPU will not see updated consumer progress and may
   * consider the queue full.
   */
  struct DeviceHandle {
    volatile uint64_t* slots;         ///< queue slots (volatile: CPU writes concurrently)
    uint64_t* consumer_index_host;    ///< extended-scope host memory
    uint64_t consumer_index;          ///< value, lives in SGPR
    size_t log2_capacity;             ///< log2(capacity)
    size_t consumer_notify_interval;  ///< pops between consumer_index notifications
    size_t pops_since_notify;         ///< counter, lives in SGPR

    /**
     * @brief Pop one 56-byte wave-slot from the queue (GPU-side, full wavefront)
     *
     * Lanes 0-7 each read one uint64_t from the slot (coalesced device mem read).
     * Lane 7's value is checked for epoch match. On match, payload is gathered
     * from lanes 0-6 via __shfl and unpacked into buf.
     *
     * @param buf Output buffer (must be at least 56 bytes)
     * @return true if popped successfully, false if queue is empty
     */
    __device__ bool pop(void* buf) {
      int lid = __lane_id();
      const size_t cap = 1ULL << log2_capacity;
      uint64_t cons_idx = consumer_index;

      size_t slot_base = (cons_idx & (cap - 1)) << 3;
      uint8_t expected_epoch = static_cast<uint8_t>((cons_idx >> log2_capacity) & 0xFF);

      // Lanes 0-7 each read one uint64_t (coalesced 64B device mem read)
      uint64_t my_val = 0;
      if (lid < 8) {
        my_val = slots[slot_base + lid];
      }

      // Check epoch: broadcast lane 7's value to all lanes
      uint64_t epoch_raw = __shfl(my_val, 7, 64);
      if (get_epoch(epoch_raw) != expected_epoch) {
        return false;  // Queue empty
      }

      // Gather payload from lanes 0-6 via __shfl and unpack into buf
      uint8_t* dst = static_cast<uint8_t*>(buf);
      for (int lane = 0; lane < 7; ++lane) {
        uint64_t val = __shfl(my_val, lane, 64);
        __builtin_memcpy(dst + lane * 8, &val, 8);
      }

      // Advance consumer position (all lanes, uniform — keeps in SGPR)
      consumer_index++;

      // Batched notification to CPU
      pops_since_notify++;
      if (pops_since_notify >= consumer_notify_interval) {
        if (lid == 0) {
          *consumer_index_host = consumer_index;
        }
        pops_since_notify = 0;
      }

      return true;
    }

    /**
     * @brief Flush consumer_index to host memory
     *
     * MUST be called before kernel returns so the CPU sees final
     * consumer progress. Without this, the CPU may think the queue
     * is still full on subsequent pushes.
     */
    __device__ void flush_consumer_index() {
      if (__lane_id() == 0) {
        *consumer_index_host = consumer_index;
      }
      pops_since_notify = 0;
    }
  };

  /**
   * @brief Construct a CPU->GPU queue
   *
   * @param capacity Number of wave-slots (must be power of 2)
   * @param notify_interval Pops between GPU->CPU consumer_index notifications.
   *                        0 = default (capacity / 4). Smaller values reduce
   *                        CPU stalls but increase PCIe writes from GPU.
   * @param device_slots If true, allocate slots in fine-grained device VRAM
   *                     (GPU reads locally, CPU writes via WC+mfence).
   *                     If false, use extended-scope host memory
   *                     (GPU reads via PCIe, CPU writes locally).
   * @throws queue_error if capacity is not a power of 2 or allocation fails
   */
  __host__ explicit Queue(size_t capacity, size_t notify_interval = 0, bool device_slots = false);
  __host__ ~Queue();

  // Disable copy and move
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  Queue(Queue&&) noexcept = delete;
  Queue& operator=(Queue&&) noexcept = delete;

  /**
   * @brief Push one 56-byte wave-slot to the queue (CPU-side)
   *
   * Writes payload lanes 0-6, then epoch lane 7.
   * With device VRAM slots: seq_cst fence (mfence) after all writes flushes WC buffer.
   * With extended-scope host slots: release fence (compiler barrier) between
   * payload and epoch.
   *
   * @param data Input buffer (must be at least 56 bytes)
   * @return true if pushed successfully, false if queue is full
   */
  __host__ bool push(const void* data);

  /**
   * @brief Get the capacity of the queue
   *
   * @return Number of wave-slots in the queue
   */
  __host__ __device__ size_t capacity() const { return capacity_; }

  /**
   * @brief Get a lightweight device handle for passing to kernels by value
   *
   * The handle's consumer_index is initialized from the current
   * consumer progress in host memory.
   *
   * @return DeviceHandle with pointers to queue storage
   */
  __host__ DeviceHandle device_handle() const;

 private:
  // Configuration
  size_t capacity_;         ///< Number of wave-slots
  size_t log2_capacity_;    ///< log2(capacity_) for bitwise division
  size_t capacity_mask_;    ///< capacity_ - 1 for bitwise modulo
  size_t notify_interval_;  ///< Pops between GPU->CPU notifications

  // Ring buffer storage (device VRAM or extended-scope host memory)
  // Array of capacity_ * 8 uint64_t elements (8 per wave-slot)
  uint64_t* slots_;
  bool device_slots_;  ///< true = device VRAM, false = extended-scope host

  // Consumer progress (extended-scope host memory)
  uint64_t* consumer_index_;

  // Producer state (host-only, not shared)
  uint64_t producer_index_;         ///< Next slot to write
  uint64_t consumer_index_cached_;  ///< CPU's cached copy of consumer progress
};

}  // namespace mori::kiwi::queue
