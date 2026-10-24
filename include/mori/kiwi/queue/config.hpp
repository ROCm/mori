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

} // namespace kiwi::queue
