#pragma once

#include "internal/serializer.hip.hpp"
#include <mori/kiwi/queue/queue.hpp>
#include <cstdint>

namespace mori::kiwi::invoke {

// GPU-side types for invoking callbacks from kernels.
//
// Two usage patterns:
// 1. Pass DeviceContext* to a kernel and call get_device_handle() inside.
//    - Required for multi-workgroup kernels (each WG gets its own handle).
//    - One-time VRAM load at kernel start, then all state lives in registers.
// 2. Pass DeviceHandle by value (legacy, for raw queue benchmarks only).
template<typename QueueType = kiwi::queue::SPSCD2HQueue>
class DeviceContext {
public:
    /**
     * @brief Lightweight register-passed handle for a single endpoint
     *
     * Passed BY VALUE to kernels so all fields land in SGPRs (registers).
     * Eliminates device memory pointer indirection: no DeviceContext* load,
     * no endpoint_queues_[] load, no QueueImpl* load.
     *
     * State (producer_index, consumer_index_cached) lives in registers
     * during kernel execution and is NOT written back.
     */
    struct DeviceHandle {
        typename QueueType::DeviceHandle queue;

        // Invoke callback with no arguments (wave-level: full wavefront).
        // A header-only message; callback_id rides in the message metadata.
        __device__ bool invoke(uint8_t callback_id) {
            return queue.push_lanes(0, 0, callback_id);
        }

        // Invoke callback with arguments (wave-level: full wavefront).
        // The arguments are serialized in registers and sent as one message
        // of exactly their size; callback_id rides in the message metadata.
        template<typename... Args>
        __device__ bool invoke(uint8_t callback_id, const Args&... args) {
            constexpr size_t arg_bytes = internal::ArgumentSerializer<Args...>::total_bytes;
            static_assert(arg_bytes <= kiwi::queue::kMaxMessageBytes,
                          "invoke arguments exceed kiwi::queue::kMaxMessageBytes");
            struct Payload {
                uint8_t bytes[arg_bytes];
            } payload;
            internal::ArgumentSerializer<Args...>::serialize(payload.bytes, args...);
            return queue.push_reg(payload, callback_id);
        }

        // Invoke a callback whose argument bytes already live in SHARED memory.
        // Wave-collective: all 64 lanes call with identical arguments. The
        // message is identical to invoke()'s. `lds_args` must be 8-byte
        // aligned, readable for round_up(arg_bytes, 8) bytes, and published
        // by the caller (LDS barrier) beforehand.
        __device__ bool invoke_shared(uint8_t callback_id, const void* lds_args,
                                      size_t arg_bytes) {
            return queue.push_shared(lds_args, arg_bytes, callback_id);
        }
    };

    // Returns a by-value copy of the DeviceHandle for the given endpoint.
    // IMPORTANT: The returned handle contains mutable state (the SPSC
    // producer_index, the credit cache) as of the last save_device_handle().
    // You MUST cache the returned handle in a local variable and reuse it for
    // all pushes to that endpoint within the kernel. Calling
    // get_device_handle() again returns a stale copy, and an SPSC producer
    // would rewrite lines it already pushed, silently corrupting the queue.
    //
    // The handle is returned wave-uniform (in SGPRs): one wave owns an
    // endpoint, and the whole wavefront must be active when calling these.

    // Default: blockIdx.x → endpoint (1:1 mapping for multi-WG kernels)
    __device__ DeviceHandle get_device_handle() {
        size_t idx = blockIdx.x;
        KIWI_QUEUE_ASSERT(idx < num_endpoints_);
        return kiwi::queue::detail::wave_uniform(handles_[idx]);
    }

    // Explicit: user picks endpoint directly (for single-WG kernels)
    __device__ DeviceHandle get_device_handle(size_t endpoint_id) {
        KIWI_QUEUE_ASSERT(endpoint_id < num_endpoints_);
        return kiwi::queue::detail::wave_uniform(handles_[endpoint_id]);
    }

    // Write back a modified handle to device memory (persists producer_index across kernels).
    // Only lane 0 performs the store to avoid 64x redundant VRAM writes.
    __device__ void save_device_handle(size_t endpoint_id, const DeviceHandle& handle) {
        KIWI_QUEUE_ASSERT(endpoint_id < num_endpoints_);
        if (__lane_id() == 0) {
            handles_[endpoint_id] = handle;
        }
    }

    // Constructor (called on host, object copied to device via hipMemcpy)
    __host__ __device__ DeviceContext(DeviceHandle* handles, size_t num_endpoints)
        : handles_(handles), num_endpoints_(num_endpoints) {}

private:
    DeviceHandle* handles_;    // device-allocated array
    size_t num_endpoints_;
};

}  // namespace kiwi::invoke
