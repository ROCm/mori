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
#include <functional>
#include <stdexcept>
#include <string>

#include "serializer.hip.hpp"

namespace mori::kiwi::invoke {
namespace internal {

// Type-erased callback wrapper
class CallbackWrapper {
 public:
  CallbackWrapper() : expected_arg_bytes_(0) {}

  // Create wrapper from typed callback function
  template <typename... Args>
  static CallbackWrapper create(std::function<void(Args...)> func) {
    CallbackWrapper wrapper;

    // Calculate total argument size
    if constexpr (sizeof...(Args) == 0) {
      wrapper.expected_arg_bytes_ = 0;
    } else {
      wrapper.expected_arg_bytes_ = (sizeof(Args) + ...);
    }

    // Type-erased executor: deserialize arguments and invoke callback
    wrapper.executor_ = [func](const uint8_t* data, size_t bytes) {
      (void)bytes;  // Unused: size validated by caller
      auto args = ArgumentSerializer<Args...>::deserialize(data);
      std::apply(func, args);
    };

    return wrapper;
  }

  // Execute callback with serialized argument data
  void execute(const uint8_t* arg_data, size_t arg_bytes) const {
    if (arg_bytes != expected_arg_bytes_) {
      throw std::runtime_error("Argument size mismatch: expected " +
                               std::to_string(expected_arg_bytes_) + " bytes, got " +
                               std::to_string(arg_bytes));
    }
    executor_(arg_data, arg_bytes);
  }

  // Get expected argument size in bytes
  size_t expected_arg_bytes() const { return expected_arg_bytes_; }

 private:
  std::function<void(const uint8_t*, size_t)> executor_;
  size_t expected_arg_bytes_;
};

}  // namespace internal
}  // namespace mori::kiwi::invoke
