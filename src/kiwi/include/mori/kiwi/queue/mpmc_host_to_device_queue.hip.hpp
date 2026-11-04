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
 * @file mpmc_host_to_device_queue.hip.hpp
 * @brief Queue<HostToDevice, MPMC> — multi-producer multi-consumer CPU-push, GPU-pop queue
 *
 * This queue supports multiple CPU threads pushing concurrently and multiple
 * GPU wavefronts popping concurrently. It uses:
 * - 64-byte wave-slots (56-byte payload in lanes 0-6, sequence number in lane 7)
 * - Vyukov-style sequence numbers for slot coordination
 * - CAS on host memory for producer slot claiming (allows returning false on full)
 * - atomicAdd on device memory for consumer slot claiming
 * - Two-level pre-check: L1 register compare (zero PCIe cost, passes ~99% of
 *   pops in throughput mode), L2 slot sequence ground truth (1 PCIe read,
 *   advances horizon by cap/4 to amortize future checks)
 * - Cached producer progress on GPU (reduces PCIe reads)
 * - Batched consumer progress notification from GPU to CPU
 * - s_sleep in slot spin-wait to reduce PCIe polling frequency
 *
 * Sequence protocol:
 *   seq == pos          -> slot is free, producer with this pos may write
 *   seq == pos + 1      -> slot contains data, consumer with this pos may read
 *   After consumer reads -> seq = pos + capacity (recycle for next round)
 */

#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>

#include "error.hpp"
#include "queue.hip.hpp"
#include "slot.hip.hpp"

namespace mori::kiwi::queue {

/**
 * @brief Multi-producer multi-consumer host-to-device queue
 *
 * Multiple CPU threads can push concurrently. Multiple GPU wavefronts
 * can pop concurrently. Slot coordination uses Vyukov-style sequence
 * numbers stored in lane 7 of each wave-slot.
 *
 * GPU pop uses a two-level pre-check: L1 is a register compare against
 * cached producer_progress (zero PCIe cost, passes on ~99% of pops in
 * throughput mode). L2 checks the actual slot sequence (1 PCIe read)
 * and advances the cached horizon by cap/4 to amortize future checks.
 *
 * CPU push returns false if queue is full (Vyukov CAS-based, no wasted
 * positions).
 *
 * IMPORTANT: Each GPU block MUST call flush_consumer_index() before
 * returning from the kernel, otherwise the CPU will not see updated
 * consumer progress and may consider the queue full.
 */
template <>
class Queue<Direction::HostToDevice, Concurrency::MPMC> {
 public:
  /**
   * @brief Lightweight device-side handle for kernel arguments
   *
   * Passed BY VALUE to kernels so scalar fields land in SGPRs.
   * The consumer_index field is a POINTER to device memory because
   * multiple wavefronts share it via atomicAdd.
   */
  struct DeviceHandle {
    volatile uint64_t* slots;              ///< extended-scope host memory
    uint64_t* consumer_progress_host;      ///< extended-scope host memory
    volatile uint64_t* producer_progress;  ///< extended-scope host memory
    uint64_t*
        consumer_index;  ///< DEVICE memory pointer (atomicAdd target, shared by all wavefronts)
    uint64_t consumer_index_estimate;   ///< register, lower bound on consumer_index (avoids device
                                        ///< mem read)
    uint64_t producer_progress_cached;  ///< register, cached copy of producer_progress
    size_t log2_capacity;               ///< log2(capacity); capacity and mask derived from this
    size_t consumer_notify_interval;    ///< pops between consumer_progress notifications
    size_t pops_since_notify;           ///< counter, lives in SGPR

    /**
     * @brief Pop one wave-slot from the queue (GPU-side, full wavefront)
     *
     * Two-level pre-check minimizes PCIe traffic:
     *   Level 1: Register compare (consumer_index_estimate vs producer_progress_cached).
     *            Zero cost — passes on ~99% of pops in throughput mode.
     *   Level 2: Slot sequence ground truth + producer_progress refresh.
     *            Only when L1 fails (consumer caught up to known progress).
     *            Checks slot seq first (cheap false return), then refreshes
     *            producer_progress to push L1 horizon forward.
     *
     * After pre-check, atomicAdd claims a slot, lane 0 spins on slot
     * sequence (8-byte PCIe reads with s_sleep), then all lanes read
     * the 64-byte slot data cooperatively.
     *
     * @param buf Output buffer (must be at least 56 bytes)
     * @return true if popped, false if queue is empty
     */
    __device__ bool pop(void* buf) {
      int lid = __lane_id();
      const size_t cap = 1ULL << log2_capacity;

      // Level 1 pre-check: register compare (zero PCIe cost).
      // In throughput mode, producer_progress_cached is far ahead of
      // consumer_index_estimate, so this passes on nearly every pop.
      if (static_cast<int64_t>(consumer_index_estimate - producer_progress_cached) >= 0) {
        // Level 2: ground truth check + progress refresh.
        // 1) Read consumer_index (device mem, fast) + slot seq (PCIe).
        //    If no data → return false immediately (cheap: 1 device + 1 PCIe).
        // 2) If data present, refresh producer_progress (PCIe) to push
        //    the L1 horizon forward. This ensures L1 passes for the next
        //    (pp - ci) pops without entering L2 again.
        uint64_t ci = consumer_index_estimate;
        uint64_t seq = 0;
        if (lid == 0) {
          ci = *consumer_index;  // device memory read
          size_t sb = (ci & (cap - 1)) << 3;
          seq = slots[sb + 7];  // 8-byte PCIe read
        }
        ci = __shfl(ci, 0, 64);
        seq = __shfl(seq, 0, 64);
        consumer_index_estimate = ci;

        if (seq != ci + 1) {
          return false;  // Queue truly empty
        }

        // Data present. Refresh producer_progress to get accurate horizon.
        uint64_t pp = producer_progress_cached;
        if (lid == 0) {
          pp = *producer_progress;  // PCIe read
        }
        pp = __shfl(pp, 0, 64);
        producer_progress_cached = pp;
      }

      // Claim slot via atomicAdd on device memory
      uint64_t claimed_pos;
      if (lid == 0) {
        claimed_pos = atomicAdd(reinterpret_cast<unsigned long long*>(consumer_index), 1ULL);
      }
      claimed_pos = __shfl(claimed_pos, 0, 64);
      consumer_index_estimate = claimed_pos + 1;

      // Spin on lane 7 until data is ready (8-byte PCIe reads with s_sleep)
      size_t slot_base = (claimed_pos & (cap - 1)) << 3;
      if (lid == 0) {
        while (slots[slot_base + 7] != claimed_pos + 1) {
          __builtin_amdgcn_s_sleep(1);
        }
      }

      // Read all 8 lanes (coalesced 64-byte PCIe read)
      uint64_t my_val = 0;
      if (lid < 8) {
        my_val = slots[slot_base + lid];
      }

      // Gather payload from lanes 0-6 via __shfl
      uint8_t* dst = static_cast<uint8_t*>(buf);
      for (int lane = 0; lane < 7; ++lane) {
        uint64_t val = __shfl(my_val, lane, 64);
        __builtin_memcpy(dst + lane * 8, &val, 8);
      }

      // Recycle slot: sequence = claimed_pos + capacity
      if (lid == 7) {
        slots[slot_base + 7] = claimed_pos + cap;
      }

      // Batched consumer_progress notification
      pops_since_notify++;
      if (pops_since_notify >= consumer_notify_interval) {
        if (lid == 0) {
          *consumer_progress_host = consumer_index_estimate;
        }
        pops_since_notify = 0;
      }

      return true;
    }

    /**
     * @brief Flush consumer_index to host memory
     *
     * MUST be called by each block before the kernel returns so the CPU
     * sees final consumer progress. Without this, the CPU may think the
     * queue is still full on subsequent pushes.
     *
     * With multiple blocks, each reads the global consumer_index from
     * device memory and writes it to consumer_progress_host. The last
     * block to execute will write the highest (most recent) value.
     */
    __device__ void flush_consumer_index() {
      if (__lane_id() == 0) {
        uint64_t ci = *consumer_index;  // read from device memory (fast)
        *consumer_progress_host = ci;   // write to host memory
      }
      pops_since_notify = 0;
    }
  };

  /**
   * @brief Construct a multi-producer multi-consumer CPU->GPU queue
   *
   * @param capacity Number of wave-slots (must be power of 2)
   * @param notify_interval Pops between GPU->CPU consumer_progress notifications.
   *                        0 = default (capacity / 4). Smaller values reduce
   *                        CPU stalls but increase PCIe writes from GPU.
   * @throws queue_error if capacity is not a power of 2 or allocation fails
   */
  __host__ explicit Queue(size_t capacity, size_t notify_interval = 0);
  __host__ ~Queue();

  // Non-copyable, non-movable
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  Queue(Queue&&) noexcept = delete;
  Queue& operator=(Queue&&) noexcept = delete;

  /**
   * @brief Push one 56-byte wave-slot to the queue (CPU-side, thread-safe)
   *
   * Multiple CPU threads may call push() concurrently. Each thread claims
   * a slot via CAS on producer_index, then writes payload and sequence.
   * Uses Vyukov-style protocol: no wasted positions on failure.
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
  __host__ size_t capacity() const { return capacity_; }

  /**
   * @brief Get a lightweight device handle for passing to kernels by value
   *
   * The handle contains pointers and capacity metadata.
   * Pass it by value to kernels so fields land in SGPRs.
   *
   * @return DeviceHandle with pointers to queue storage
   */
  __host__ DeviceHandle device_handle() const;

 private:
  // Configuration
  size_t capacity_;         ///< Number of wave-slots (power of 2)
  size_t log2_capacity_;    ///< log2(capacity_) for bitwise division
  size_t capacity_mask_;    ///< capacity_ - 1 for bitwise modulo
  size_t notify_interval_;  ///< Pops between GPU->CPU notifications

  // Ring buffer storage (extended-scope host memory)
  // Array of capacity_ * 8 uint64_t elements (8 per wave-slot)
  uint64_t* slots_;

  // Producer index (extended-scope host memory, CAS target for multi-producer)
  // Accessed via __atomic builtins in push(); stored as raw pointer because
  // The extended-scope host allocator manages this storage.
  uint64_t* producer_index_;

  // Consumer index (device memory, shared across wavefronts via atomicAdd)
  uint64_t* consumer_index_;

  // Consumer progress (extended-scope host memory, GPU writes batched, CPU reads)
  // Updated by GPU after pop() to notify CPU of freed slots.
  // May lag behind consumer_index (batched notification).
  uint64_t* consumer_progress_;

  // Producer progress (extended-scope host memory, CPU writes batched, GPU reads)
  // Updated by CPU after push() to notify GPU of available data.
  // May lag behind producer_index (batched notification).
  uint64_t* producer_progress_;
};

}  // namespace mori::kiwi::queue
