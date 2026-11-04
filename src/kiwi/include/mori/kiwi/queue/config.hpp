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
/**
 * @file config.hpp
 * @brief Configuration structure for queue
 *
 * Defines QueueConfig with capacity.
 */

#pragma once

#include <cstddef>

#include "error.hpp"

namespace mori::kiwi::queue {

/**
 * @brief Configuration parameters for Queue
 */
struct QueueConfig {
  /**
   * @brief Queue capacity: 64-byte lines for device-to-host queues,
   *        64-byte wave-slots for host-to-device queues
   *
   * Must be a power of 2 for efficient bitwise index operations.
   * Typical values: 1024-4096
   */
  size_t capacity;

  /**
   * @brief Create default configuration for given capacity
   *
   * @param cap Queue capacity (lines or wave-slots, must be power of 2)
   * @return QueueConfig with default settings
   * @throws queue_error if capacity is not a power of 2
   */
  static QueueConfig default_config(size_t cap) {
    if (cap == 0 || (cap & (cap - 1)) != 0) {
      throw queue_error("Queue capacity must be a power of 2");
    }
    return QueueConfig{.capacity = cap};
  }
};

}  // namespace mori::kiwi::queue
