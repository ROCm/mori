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
    DeviceToHost,   ///< GPU pushes, CPU pops
    HostToDevice    ///< CPU pushes, GPU pops
};

/// Concurrency mode for Queue<Dir, Con>.
enum class Concurrency {
    SPSC,   ///< Single-producer single-consumer (no atomics)
    MPSC,   ///< Multi-producer single-consumer (device fetch-add claim)
    MPMC    ///< Multi-producer multi-consumer (atomicAdd + CAS)
};

/// Primary template — defined only via explicit specializations.
template<Direction Dir, Concurrency Con = Concurrency::SPSC>
class Queue;

/// Canonical aliases
using SPSCD2HQueue = Queue<Direction::DeviceToHost, Concurrency::SPSC>;
using MPSCD2HQueue = Queue<Direction::DeviceToHost, Concurrency::MPSC>;
using SPSCH2DQueue = Queue<Direction::HostToDevice, Concurrency::SPSC>;
using MPMCH2DQueue = Queue<Direction::HostToDevice, Concurrency::MPMC>;

} // namespace kiwi::queue
