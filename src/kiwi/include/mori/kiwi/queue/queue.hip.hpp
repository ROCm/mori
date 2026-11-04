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
 * @file queue.hip.hpp
 * @brief Unified queue template: Queue<Direction, Concurrency>
 *
 * Forward-declares the primary Queue template and provides canonical
 * type aliases. Include the specialization headers for full definitions:
 *   - device_to_host_queue.hip.hpp       ->  Queue<DeviceToHost, SPSC>
 *   - mpsc_device_to_host_queue.hip.hpp  ->  Queue<DeviceToHost, MPSC>
 *   - host_to_device_queue.hip.hpp       ->  Queue<HostToDevice, SPSC>
 *   - mpmc_host_to_device_queue.hip.hpp  ->  Queue<HostToDevice, MPMC>
 */

#pragma once

namespace mori::kiwi::queue {

/// Data-flow direction for Queue<Dir, Con>.
enum class Direction {
  DeviceToHost,  ///< GPU pushes, CPU pops
  HostToDevice   ///< CPU pushes, GPU pops
};

/// Concurrency mode for Queue<Dir, Con>.
enum class Concurrency {
  SPSC,  ///< Single-producer single-consumer (no atomics)
  MPSC,  ///< Multi-producer single-consumer (device fetch-add claim)
  MPMC   ///< Multi-producer multi-consumer (atomicAdd + CAS)
};

/// Primary template — defined only via explicit specializations.
template <Direction Dir, Concurrency Con = Concurrency::SPSC>
class Queue;

/// Canonical aliases
using SPSCD2HQueue = Queue<Direction::DeviceToHost, Concurrency::SPSC>;
using MPSCD2HQueue = Queue<Direction::DeviceToHost, Concurrency::MPSC>;
using SPSCH2DQueue = Queue<Direction::HostToDevice, Concurrency::SPSC>;
using MPMCH2DQueue = Queue<Direction::HostToDevice, Concurrency::MPMC>;

}  // namespace mori::kiwi::queue
