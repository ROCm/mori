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

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "callback_wrapper.hpp"

namespace mori::kiwi::invoke {
namespace internal {

// Callback registration and lookup manager
class CallbackRegistry {
 public:
  CallbackRegistry() = default;

  // Register a callback, returns callback ID (0-255)
  template <typename... Args>
  uint8_t register_callback(std::function<void(Args...)> func) {
    if (callbacks_.size() >= 256) {
      throw std::runtime_error("Callback registry full (max 256 callbacks)");
    }

    uint8_t callback_id = static_cast<uint8_t>(callbacks_.size());
    callbacks_.push_back(CallbackWrapper::create<Args...>(std::move(func)));
    return callback_id;
  }

  // Get callback by ID
  const CallbackWrapper& get(uint8_t callback_id) const {
    if (callback_id >= callbacks_.size()) {
      throw std::out_of_range("Invalid callback_id: " + std::to_string(callback_id) +
                              " (registry size: " + std::to_string(callbacks_.size()) + ")");
    }
    return callbacks_[callback_id];
  }

  // Get expected argument size for callback
  size_t expected_arg_bytes(uint8_t callback_id) const {
    return get(callback_id).expected_arg_bytes();
  }

  // Get number of registered callbacks
  size_t size() const { return callbacks_.size(); }

 private:
  std::vector<CallbackWrapper> callbacks_;
};

}  // namespace internal
}  // namespace mori::kiwi::invoke
