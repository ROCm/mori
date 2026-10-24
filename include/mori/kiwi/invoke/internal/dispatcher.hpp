#pragma once

#include "callback_registry.hpp"
#include <mori/kiwi/queue/queue.hpp>
#include <cstddef>
#include <cstdint>

namespace mori::kiwi::invoke {
namespace internal {

// CPU-side callback dispatcher. Each queue message is one whole callback:
// its metadata is the callback_id and its bytes are the serialized
// arguments. The dispatcher keeps no state between messages, so concurrent
// progress() calls are safe whenever the queue's pop() is.
template<typename QueueType = kiwi::queue::SPSCD2HQueue>
class CallbackDispatcher {
public:
    CallbackDispatcher(CallbackRegistry& registry, QueueType* queue)
        : registry_(registry), queue_(queue) {}

    // Process pending callbacks, return count executed
    // max_callbacks = -1 means process all available
    size_t progress(int max_callbacks = -1) {
        size_t executed = 0;
        alignas(8) uint8_t args[kiwi::queue::kMaxMessageBytes];

        while (max_callbacks < 0 || static_cast<int>(executed) < max_callbacks) {
            uint32_t meta = 0;
            size_t arg_bytes = 0;
            if (!queue_->pop(args, &meta, &arg_bytes)) {
                break;  // No complete message available (non-blocking)
            }
            // execute() throws if arg_bytes differs from the registered size.
            registry_.get(static_cast<uint8_t>(meta)).execute(args, arg_bytes);
            executed++;
        }

        return executed;
    }

private:
    CallbackRegistry& registry_;
    QueueType* queue_;
};

}  // namespace internal
}  // namespace kiwi::invoke
