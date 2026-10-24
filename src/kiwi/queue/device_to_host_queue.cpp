/**
 * @file device_to_host_queue.cpp
 * @brief Host side of Queue<Direction::DeviceToHost, Concurrency::SPSC>
 */

#include <mori/kiwi/queue/device_to_host_queue.hip.hpp>
#include "chunk_consumer.hpp"

#include <hip/hip_runtime.h>

namespace mori::kiwi::queue {

Queue<Direction::DeviceToHost, Concurrency::SPSC>::Queue(size_t capacity)
    : capacity_(capacity)
    , log2_capacity_(capacity == 0 ? 0 : __builtin_ctzll(capacity))
    , capacity_mask_(capacity - 1)
    , publish_interval_(detail::publish_interval(capacity))
    , chunks_(detail::allocate_ring(capacity))
    , consumer_index_(nullptr)
    , head_(0)
{
    try {
        consumer_index_ = detail::host_malloc_uc<uint64_t>(1);
    } catch (...) {
        detail::host_free_uc(chunks_);
        throw;
    }
    *consumer_index_ = 0;
}

Queue<Direction::DeviceToHost, Concurrency::SPSC>::~Queue() {
    detail::host_free_uc(consumer_index_);
    detail::host_free_uc(chunks_);
}

bool Queue<Direction::DeviceToHost, Concurrency::SPSC>::pop(void* buf, uint32_t* metadata,
                                                            size_t* bytes) {
    return detail::pop_message(chunks_, capacity_mask_, log2_capacity_, publish_interval_, head_,
                               consumer_index_, buf, metadata, bytes);
}

Queue<Direction::DeviceToHost, Concurrency::SPSC>::DeviceHandle
Queue<Direction::DeviceToHost, Concurrency::SPSC>::device_handle() const {
    // Publish the exact position: on a drained queue the producer starts
    // where the consumer stands. A max, so a call that races with pop()
    // (outside the drained contract) cannot move the published index back.
    const uint64_t head = head_;
    detail::publish_max(consumer_index_, head);
    DeviceHandle h{};
    h.chunks = detail::to_global(chunks_);
    h.consumer_index = detail::to_global<const uint64_t>(consumer_index_);
    h.producer_index = head;
    h.consumer_index_cached = head;
    h.log2_capacity = log2_capacity_;
    return h;
}

} // namespace kiwi::queue
