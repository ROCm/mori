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
 * @file queue.hpp
 * @brief Main header for queue library
 *
 * The KIWI queue layer is a high-performance library for
 * control-message transport between AMD GPUs and the host CPU.
 *
 * Features:
 * - Device-to-host and host-to-device queues
 * - SPSC queues in both directions, an MPSC device-to-host queue and an
 *   MPMC host-to-device queue
 * - Device-to-host: 16-byte chunks (8 B payload + 8 B signal per lane),
 *   messages of up to 512 bytes plus 32-bit metadata, and one push API
 *   (push_reg, push_shared, push_global, push_lanes)
 * - Host-to-device: 64-byte wave-slots (56-byte payload in lanes 0-6)
 * - Full-wavefront (64 threads) cooperative device-side operations
 * - Lap epochs instead of clearing; cached producer/consumer progress to
 *   reduce PCIe traffic
 *
 * Quick Start:
 * @code
 * #include <mori/kiwi/queue/queue.hpp>
 *
 * // Create queue (capacity in 64-byte lines, a power of 2)
 * auto queue = std::make_unique<kiwi::queue::SPSCD2HQueue>(1024);
 *
 * // GPU kernel (launched with full wavefront)
 * __global__ void producer(kiwi::queue::SPSCD2HQueue::DeviceHandle q) {
 *     uint32_t value = 42;
 *     while (!q.push_reg(value)) {}
 * }
 *
 * // Launch kernel
 * producer<<<1, 64>>>(queue->device_handle());
 *
 * // Pop on CPU (buffer of at least the largest message)
 * uint32_t value;
 * while (queue->pop(&value)) {
 *     process(value);
 * }
 * @endcode
 */

#pragma once

// Version macros
#define KIWI_QUEUE_VERSION_MAJOR 0
#define KIWI_QUEUE_VERSION_MINOR 2
#define KIWI_QUEUE_VERSION_PATCH 0

// Core headers
#include "device_to_host_queue.hip.hpp"
#include "error.hpp"
#include "host_to_device_queue.hip.hpp"
#include "mpmc_host_to_device_queue.hip.hpp"
#include "mpsc_device_to_host_queue.hip.hpp"
#include "queue.hip.hpp"
#include "slot.hip.hpp"

/**
 * @namespace kiwi::queue
 * @brief Main namespace for queue library
 *
 * All classes, functions, and types are defined in this namespace.
 */
