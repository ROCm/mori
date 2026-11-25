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

// Copied from AMDResearch/kiwi d7bb0d1 (include/kiwi -> include/mori/cco/proxy,
// src/queue -> src/cco/proxy). Local changes:
// - namespace mori::cco::proxy; KIWI_* -> CCO_PROXY_*, KIWI_QUEUE_ASSERT -> CCO_PROXY_ASSERT;
//   KIWI_QUEUE_VERSION_* dropped; license header, include paths and formatting follow mori.
// - Only the device-to-host SPSC/MPSC queues and invoke; no host-to-device queues, slot.hip.hpp
//   or upstream CMake.
// - Pushes compile only for gfx942/gfx950; elsewhere they trap (MORI_CCO_PROXY_PUSH_UNSUPPORTED).
// - MPSC push_active_lanes(): push_lane_masked()'s body moved to push_own_messages(), plus
//   divergent_consumer_index_cached; invoke_active_lanes() in device_context.hip.hpp.

/**
 * @file queue.hpp
 * @brief Main header for queue library
 *
 * The KIWI queue layer is a high-performance library for
 * control-message transport between AMD GPUs and the host CPU.
 *
 * Features:
 * - SPSC and MPSC device-to-host queues
 * - Device-to-host: 16-byte chunks per lane, each two halves of 4 B payload
 *   + 4 B signal (LL8),
 *   messages of up to 512 bytes plus 32-bit metadata, and one push API
 *   (push_reg, push_shared, push_global, push_lanes, and push_lane_masked,
 *   where each active lane pushes its own message)
 * - Full-wavefront (64 threads) cooperative device-side operations
 * - Lap epochs instead of clearing; cached producer/consumer progress to
 *   reduce PCIe traffic
 *
 * Quick Start:
 * @code
 * #include "mori/cco/proxy/queue/queue.hpp"
 *
 * // Create queue (capacity in 64-byte lines, a power of 2)
 * auto queue = std::make_unique<mori::cco::proxy::queue::SPSCD2HQueue>(1024);
 *
 * // GPU kernel (launched with full wavefront)
 * __global__ void producer(mori::cco::proxy::queue::SPSCD2HQueue::DeviceHandle q) {
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

// Core headers
#include "device_to_host_queue.hip.hpp"
#include "error.hpp"
#include "mpsc_device_to_host_queue.hip.hpp"
#include "queue.hip.hpp"

/**
 * @namespace mori::cco::proxy::queue
 * @brief Main namespace for queue library
 *
 * All classes, functions, and types are defined in this namespace.
 */
