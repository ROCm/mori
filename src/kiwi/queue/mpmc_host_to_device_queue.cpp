/**
 * @file mpmc_host_to_device_queue.cpp
 * @brief Host-side implementation of MPMC H2D Queue
 *
 * Implements constructor, destructor, push(), and device_handle().
 * Uses Vyukov-style MPMC queue with CAS-based multi-producer enqueue.
 *
 * Push uses the standard Vyukov bounded MPMC pattern:
 *   1. Load producer_index
 *   2. Read sequence from target slot
 *   3. If seq == pos: CAS producer_index to claim, write data, set seq = pos + 1
 *   4. If seq < pos: queue full (slot not recycled), return false
 *   5. If seq > pos: another producer took this slot, retry
 */

#include <mori/kiwi/queue/mpmc_host_to_device_queue.hip.hpp>
#include <mori/kiwi/queue/detail/host_memory.hpp>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstring>
#include <cassert>
#include <string>

namespace mori::kiwi::queue {

Queue<Direction::HostToDevice, Concurrency::MPMC>::Queue(
    size_t capacity, size_t notify_interval)
    : capacity_(capacity)
    , log2_capacity_(__builtin_ctzll(capacity))
    , capacity_mask_(capacity - 1)
    , notify_interval_(notify_interval == 0 ? capacity / 4 : notify_interval)
    , slots_(nullptr)
    , producer_index_(nullptr)
    , consumer_index_(nullptr)
    , consumer_progress_(nullptr)
    , producer_progress_(nullptr)
{
    // Validate capacity: must be a non-zero power of 2
    if (capacity_ == 0 || (capacity_ & (capacity_ - 1)) != 0) {
        throw queue_error("Queue capacity must be a power of 2");
    }

    // Ensure notify_interval is at least 1
    if (notify_interval_ == 0) {
        notify_interval_ = 1;
    }

    // Allocate extended-scope host memory for wave-slot array.
    // Each wave-slot is 8 consecutive uint64_t values (64 bytes)
    slots_ = detail::host_malloc_uc<uint64_t>(capacity_ * 8);

    // Verify 64-byte alignment
    assert(reinterpret_cast<uintptr_t>(slots_) % 64 == 0 &&
           "Slots must be 64-byte aligned");

    // Initialize all wave-slots: lanes 0-6 = 0, lane 7 = slot index
    // Lane 7 holds the sequence number for Vyukov-style MPMC.
    // Initial sequence = slot index, meaning producer at position i can claim slot i.
    for (size_t i = 0; i < capacity_; i++) {
        for (int lane = 0; lane < 7; lane++) {
            slots_[i * 8 + lane] = 0;
        }
        slots_[i * 8 + 7] = i;  // sequence number = slot index
    }

    // Allocate extended-scope host memory for producer index.
    try {
        producer_index_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        detail::host_free_uc(slots_);
        throw;
    }
    *producer_index_ = 0;

    // Allocate device memory for consumer index (GPU atomicAdd target)
    try {
        HIP_CHECK(hipMalloc(&consumer_index_, sizeof(uint64_t)));
    } catch (...) {
        detail::host_free_uc(producer_index_);
        detail::host_free_uc(slots_);
        throw;
    }
    HIP_CHECK(hipMemset(consumer_index_, 0, sizeof(uint64_t)));

    // Allocate extended-scope host memory for consumer progress.
    try {
        consumer_progress_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        HIP_CHECK_NOEXCEPT(hipFree(consumer_index_));
        detail::host_free_uc(producer_index_);
        detail::host_free_uc(slots_);
        throw;
    }
    *consumer_progress_ = 0;

    // Allocate extended-scope host memory for producer progress.
    try {
        producer_progress_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        detail::host_free_uc(consumer_progress_);
        HIP_CHECK_NOEXCEPT(hipFree(consumer_index_));
        detail::host_free_uc(producer_index_);
        detail::host_free_uc(slots_);
        throw;
    }
    *producer_progress_ = 0;
}

Queue<Direction::HostToDevice, Concurrency::MPMC>::~Queue() {
    if (producer_progress_) {
        detail::host_free_uc(producer_progress_);
    }
    if (consumer_progress_) {
        detail::host_free_uc(consumer_progress_);
    }
    if (consumer_index_) {
        HIP_CHECK_NOEXCEPT(hipFree(consumer_index_));
    }
    if (producer_index_) {
        detail::host_free_uc(producer_index_);
    }
    if (slots_) {
        detail::host_free_uc(slots_);
    }
}

bool Queue<Direction::HostToDevice, Concurrency::MPMC>::push(const void* data) {
    while (true) {
        uint64_t pos = __atomic_load_n(producer_index_, __ATOMIC_RELAXED);
        size_t slot_idx = pos & capacity_mask_;
        size_t slot_base = slot_idx << 3;

        // Read sequence from lane 7
        uint64_t seq = __atomic_load_n(&slots_[slot_base + 7], __ATOMIC_ACQUIRE);
        int64_t dif = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);

        if (dif == 0) {
            // Slot is free for this producer position. Try to claim it.
            if (__atomic_compare_exchange_n(producer_index_, &pos, pos + 1,
                                            /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                // Successfully claimed position pos.
                // Write payload lanes 0-6.
                for (int lane = 0; lane < 7; ++lane) {
                    slots_[slot_base + lane] = pack_wave_lane(data, lane, 0);
                }

                // Write sequence = pos + 1 (signals "data ready for consumer")
                __atomic_store_n(&slots_[slot_base + 7], pos + 1, __ATOMIC_RELEASE);

                // Batch producer_progress updates: only update when pos crosses
                // a notification boundary (every capacity/4 pushes).
                // Use monotonic CAS to prevent backwards writes when producers
                // complete out of order.
                size_t notify_mask = (capacity_ >> 2) - 1;  // capacity/4 - 1
                if ((pos & notify_mask) == 0) {
                    uint64_t new_pp = pos + 1;
                    uint64_t old_pp = __atomic_load_n(producer_progress_, __ATOMIC_RELAXED);
                    while (old_pp < new_pp) {
                        if (__atomic_compare_exchange_n(producer_progress_, &old_pp, new_pp,
                                                        /*weak=*/true, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
                            break;
                    }
                }

                return true;
            }
            // CAS failed — another producer claimed this position, retry
        } else if (dif < 0) {
            return false;  // Queue full (slot not recycled yet by consumer)
        }
        // dif > 0: another producer already wrote here, retry with new pos
    }
}

Queue<Direction::HostToDevice, Concurrency::MPMC>::DeviceHandle
Queue<Direction::HostToDevice, Concurrency::MPMC>::device_handle() const {
    uint64_t cp = *consumer_progress_;
    uint64_t pp = *producer_progress_;
    return DeviceHandle{
        .slots = slots_,
        .consumer_progress_host = consumer_progress_,
        .producer_progress = producer_progress_,
        .consumer_index = consumer_index_,
        .consumer_index_estimate = cp,   // start at consumer_progress (safe lower bound)
        .producer_progress_cached = pp,  // start at current producer_progress
        .log2_capacity = log2_capacity_,
        .consumer_notify_interval = notify_interval_,
        .pops_since_notify = 0
    };
}

} // namespace kiwi::queue
