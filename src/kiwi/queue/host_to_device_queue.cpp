/**
 * @file host_to_device_queue.cpp
 * @brief Host-side implementation of Queue<Direction::HostToDevice, Concurrency::SPSC>
 *
 * Implements constructor, destructor, push(), and device_handle().
 *
 * Slot memory type is runtime-selectable via the device_slots constructor parameter:
 *   true:  Fine-grained device VRAM. GPU reads locally. CPU writes via
 *          WC mapping + seq_cst fence (mfence) to flush WC buffer.
 *   false (default): Extended-scope host memory. GPU reads over PCIe.
 *          CPU writes with release fence (compiler barrier on x86, TSO suffices).
 *
 * All 8 stores per wave-slot (lanes 0-6 payload + lane 7 epoch) target the same
 * 64-byte cache line, so the WC buffer coalesces them into one PCIe write.
 */

#include <mori/kiwi/queue/host_to_device_queue.hip.hpp>
#include <mori/kiwi/queue/detail/host_memory.hpp>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstring>
#include <cassert>
#include <string>

namespace mori::kiwi::queue {

Queue<Direction::HostToDevice, Concurrency::SPSC>::Queue(
    size_t capacity, size_t notify_interval, bool device_slots)
    : capacity_(capacity)
    , log2_capacity_(__builtin_ctzll(capacity))
    , capacity_mask_(capacity - 1)
    , notify_interval_(notify_interval == 0 ? capacity / 4 : notify_interval)
    , slots_(nullptr)
    , device_slots_(device_slots)
    , consumer_index_(nullptr)
    , producer_index_(0)
    , consumer_index_cached_(0)
{
    // Validate capacity: must be a non-zero power of 2
    if (capacity_ == 0 || (capacity_ & (capacity_ - 1)) != 0) {
        throw queue_error("Queue capacity must be a power of 2");
    }

    // Ensure notify_interval is at least 1
    if (notify_interval_ == 0) {
        notify_interval_ = 1;
    }

    if (device_slots_) {
        // Fine-grained device VRAM: GPU reads locally (fast), CPU writes via PCIe/WC mapping.
        // Requires seq_cst fence after writes to flush WC buffer (~1.6us visibility).
        HIP_CHECK(hipExtMallocWithFlags(
            reinterpret_cast<void**>(&slots_),
            capacity_ * 8 * sizeof(uint64_t),
            hipDeviceMallocFinegrained));
    } else {
        // Extended-scope host memory: uncached GPU mapping with system scope.
        slots_ = detail::host_malloc_uc<uint64_t>(capacity_ * 8);
    }

    // Verify 64-byte alignment (ensures each wave-slot sits on one cache line)
    assert(reinterpret_cast<uintptr_t>(slots_) % 64 == 0 &&
           "Slots must be 64-byte aligned");

    // Initialize all wave-slots: lanes 0-6 = 0, lane 7 = epoch 255.
    // Epoch 255 ensures empty slots don't match expected epoch 0 on first pass.
    for (size_t i = 0; i < capacity_; i++) {
        for (int lane = 0; lane < 7; lane++) {
            slots_[i * 8 + lane] = 0;
        }
        slots_[i * 8 + 7] = 255;
    }

    if (device_slots_) {
        // Flush WC buffer so GPU sees initialized slots before any kernel launch
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    // Allocate extended-scope host memory for consumer progress.
    try {
        consumer_index_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        if (device_slots_) {
            HIP_CHECK_NOEXCEPT(hipFree(slots_));
        } else {
            detail::host_free_uc(slots_);
        }
        throw;
    }

    // Initialize consumer progress to 0
    *consumer_index_ = 0;
}

Queue<Direction::HostToDevice, Concurrency::SPSC>::~Queue() {
    if (consumer_index_) {
        detail::host_free_uc(consumer_index_);
    }
    if (slots_) {
        if (device_slots_) {
            HIP_CHECK_NOEXCEPT(hipFree(slots_));
        } else {
            detail::host_free_uc(slots_);
        }
    }
}

bool Queue<Direction::HostToDevice, Concurrency::SPSC>::push(const void* data) {
    // Check space using cached consumer progress
    if (producer_index_ - consumer_index_cached_ >= capacity_) {
        // Refresh from host memory (local read, fast)
        consumer_index_cached_ = *consumer_index_;
        if (producer_index_ - consumer_index_cached_ >= capacity_) {
            return false;  // Queue full
        }
    }

    // Compute slot position and epoch
    size_t ws_base = (producer_index_ & capacity_mask_) << 3;
    uint8_t epoch = static_cast<uint8_t>((producer_index_ >> log2_capacity_) & 0xFF);

    // Write payload lanes 0-6
    for (int lane = 0; lane < 7; ++lane) {
        slots_[ws_base + lane] = pack_wave_lane(data, lane, epoch);
    }

    if (device_slots_) {
        // Write epoch lane 7. All 8 stores target the same 64-byte cache line;
        // the WC buffer coalesces them into one PCIe write. A single seq_cst fence
        // flushes the combined write — GPU sees all-old or all-new.
        slots_[ws_base + 7] = static_cast<uint64_t>(epoch);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    } else {
        // Compiler barrier: ensure payload stores are not reordered past epoch.
        // Extended-scope host memory is UC for GPU access; x86 TSO preserves
        // store ordering at the hardware level, but we still need this to
        // prevent compiler reordering.
        std::atomic_thread_fence(std::memory_order_release);
        slots_[ws_base + 7] = static_cast<uint64_t>(epoch);
    }

    // Advance producer
    producer_index_++;
    return true;
}

Queue<Direction::HostToDevice, Concurrency::SPSC>::DeviceHandle Queue<Direction::HostToDevice, Concurrency::SPSC>::device_handle() const {
    uint64_t ci = *consumer_index_;
    return DeviceHandle{
        .slots = slots_,
        .consumer_index_host = consumer_index_,
        .consumer_index = ci,
        .log2_capacity = log2_capacity_,
        .consumer_notify_interval = notify_interval_,
        .pops_since_notify = 0
    };
}

} // namespace kiwi::queue
