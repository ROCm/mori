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

#include <memory>
#include <mori/kiwi/queue/queue.hpp>

#include "callback_registry.hpp"
#include "dispatcher.hpp"

namespace mori::kiwi::invoke {
namespace internal {

// Endpoint encapsulates a queue and its dispatcher
template <typename QueueType = kiwi::queue::SPSCD2HQueue>
class Endpoint {
 public:
  Endpoint(size_t queue_capacity, CallbackRegistry& registry)
      : queue_(nullptr), dispatcher_(nullptr) {
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
  size_t progress(int max_callbacks = -1) { return dispatcher_->progress(max_callbacks); }

  // Get queue capacity
  size_t capacity() const { return queue_->capacity(); }

  // Get raw queue pointer (for benchmarking bypass)
  QueueType* raw_queue() const { return queue_.get(); }

 private:
  std::unique_ptr<QueueType> queue_;
  std::unique_ptr<CallbackDispatcher<QueueType>> dispatcher_;
};

}  // namespace internal
}  // namespace mori::kiwi::invoke
