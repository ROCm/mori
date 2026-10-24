#pragma once

#include "dispatcher.hpp"
#include "callback_registry.hpp"
#include <mori/kiwi/queue/queue.hpp>
#include <memory>

namespace mori::kiwi::invoke {
namespace internal {

// Endpoint encapsulates a queue and its dispatcher
template<typename QueueType = kiwi::queue::SPSCD2HQueue>
class Endpoint {
public:
    Endpoint(size_t queue_capacity, CallbackRegistry& registry)
        : queue_(nullptr)
        , dispatcher_(nullptr)
    {
        // Create queue with specified capacity
        queue_ = std::make_unique<QueueType>(queue_capacity);

        // Create dispatcher
        dispatcher_ = std::make_unique<CallbackDispatcher<QueueType>>(registry, queue_.get());
    }

    // Disable copy/move (owns unique resources)
    Endpoint(const Endpoint&) = delete;
    Endpoint& operator=(const Endpoint&) = delete;
    Endpoint(Endpoint&&) = default;
    Endpoint& operator=(Endpoint&&) = default;

    // CPU-side: process callbacks from this endpoint's queue
    size_t progress(int max_callbacks = -1) {
        return dispatcher_->progress(max_callbacks);
    }

    // Get queue capacity
    size_t capacity() const {
        return queue_->capacity();
    }

    // Get raw queue pointer (for benchmarking bypass)
    QueueType* raw_queue() const {
        return queue_.get();
    }

private:
    std::unique_ptr<QueueType> queue_;
    std::unique_ptr<CallbackDispatcher<QueueType>> dispatcher_;
};

}  // namespace internal
}  // namespace kiwi::invoke
