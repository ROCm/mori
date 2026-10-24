/**
 * @file mpsc_device_to_host_queue.cpp
 * @brief Host side of Queue<Direction::DeviceToHost, Concurrency::MPSC>
 */

#include <mori/kiwi/queue/mpsc_device_to_host_queue.hip.hpp>
#include "chunk_consumer.hpp"

#include <hip/hip_runtime.h>

namespace mori::kiwi::queue {

Queue<Direction::DeviceToHost, Concurrency::MPSC>::Queue(size_t capacity)
    : capacity_(capacity)
    , log2_capacity_(capacity == 0 ? 0 : __builtin_ctzll(capacity))
    , capacity_mask_(capacity - 1)
    , publish_interval_(detail::publish_interval(capacity))
    , chunks_(detail::allocate_ring(capacity))
    , producer_index_(nullptr)
    , consumer_index_(nullptr)
    , head_(0)
{
    try {
        HIP_CHECK(hipMalloc(&producer_index_, sizeof(uint64_t)));
        HIP_CHECK(hipMemset(producer_index_, 0, sizeof(uint64_t)));
        HIP_CHECK(hipDeviceSynchronize());
        consumer_index_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        if (producer_index_ != nullptr) {
            HIP_CHECK_NOEXCEPT(hipFree(producer_index_));
        }
        detail::host_free_uc(chunks_);
        throw;
    }
    *consumer_index_ = 0;
}

Queue<Direction::DeviceToHost, Concurrency::MPSC>::~Queue() {
    detail::host_free_uc(consumer_index_);
    HIP_CHECK_NOEXCEPT(hipFree(producer_index_));
    detail::host_free_uc(chunks_);
}

bool Queue<Direction::DeviceToHost, Concurrency::MPSC>::pop(void* buf, uint32_t* metadata,
                                                            size_t* bytes) {
    return detail::pop_message(chunks_, capacity_mask_, log2_capacity_, publish_interval_, head_,
                               consumer_index_, buf, metadata, bytes);
}

Queue<Direction::DeviceToHost, Concurrency::MPSC>::DeviceHandle
Queue<Direction::DeviceToHost, Concurrency::MPSC>::device_handle() const {
    // head_ belongs to the consumer thread; start from the published index,
    // which only lags it.
    DeviceHandle h{};
    h.chunks = detail::to_global(chunks_);
    h.consumer_index = detail::to_global<const uint64_t>(consumer_index_);
    h.producer_index = detail::to_global(producer_index_);
    h.consumer_index_cached = __atomic_load_n(consumer_index_, __ATOMIC_ACQUIRE);
    h.log2_capacity = log2_capacity_;
    return h;
}

} // namespace kiwi::queue
