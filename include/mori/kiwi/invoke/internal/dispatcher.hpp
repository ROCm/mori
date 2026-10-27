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
#pragma once

#include <cstddef>
#include <cstdint>
#include <mori/kiwi/queue/queue.hpp>

#include "callback_registry.hpp"

namespace mori::kiwi::invoke {
namespace internal {

// CPU-side callback dispatcher. Each queue message is one whole callback:
// its metadata is the callback_id and its bytes are the serialized
// arguments. The dispatcher keeps no state between messages, so concurrent
// progress() calls are safe whenever the queue's pop() is.
template <typename QueueType = kiwi::queue::SPSCD2HQueue>
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
}  // namespace mori::kiwi::invoke
