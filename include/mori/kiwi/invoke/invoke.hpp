#pragma once

#include "mori/kiwi/invoke/internal/endpoint.hip.hpp"
#include "mori/kiwi/invoke/internal/callback_registry.hpp"
#include "mori/kiwi/invoke/device_context.hip.hpp"
#include <mori/kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <string>

namespace mori::kiwi::invoke {

// Main public API for GPU-to-CPU callback bridge
template<typename QueueType = kiwi::queue::SPSCD2HQueue>
class Context {
public:
    using DeviceHandle = typename DeviceContext<QueueType>::DeviceHandle;

    // Constructor: creates multiple endpoints and builds device-side context
    // num_endpoints: number of parallel queues for GPU-CPU communication
    // queue_capacity: capacity for each endpoint's queue
    Context(size_t num_endpoints, size_t queue_capacity) {
        if (num_endpoints == 0) {
            throw std::invalid_argument("num_endpoints must be > 0");
        }

        // Create endpoints (each with its own queue and dispatcher)
        endpoints_.reserve(num_endpoints);
        for (size_t i = 0; i < num_endpoints; ++i) {
            endpoints_.push_back(
                std::make_unique<internal::Endpoint<QueueType>>(queue_capacity, registry_)
            );
        }

        // Build device-side handle array
        std::vector<DeviceHandle> host_handles;
        host_handles.reserve(num_endpoints);
        for (size_t i = 0; i < num_endpoints; ++i) {
            host_handles.push_back(DeviceHandle{endpoints_[i]->raw_queue()->device_handle()});
        }

        HIP_CHECK(hipMalloc(&device_handles_, num_endpoints * sizeof(DeviceHandle)));
        HIP_CHECK(hipMemcpy(device_handles_, host_handles.data(),
                            num_endpoints * sizeof(DeviceHandle), hipMemcpyHostToDevice));

        // Build DeviceContext on host, then copy to device
        DeviceContext<QueueType> host_ctx(device_handles_, num_endpoints);
        HIP_CHECK(hipMalloc(&device_context_, sizeof(DeviceContext<QueueType>)));
        HIP_CHECK(hipMemcpy(device_context_, &host_ctx,
                            sizeof(DeviceContext<QueueType>), hipMemcpyHostToDevice));
    }

    ~Context() {
        if (device_context_) (void)hipFree(device_context_);
        if (device_handles_) (void)hipFree(device_handles_);
    }

    // Disable copy (owns GPU resources)
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Enable move
    Context(Context&& other) noexcept
        : registry_(std::move(other.registry_))
        , endpoints_(std::move(other.endpoints_))
        , device_handles_(other.device_handles_)
        , device_context_(other.device_context_)
    {
        other.device_handles_ = nullptr;
        other.device_context_ = nullptr;
    }

    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            if (device_context_) (void)hipFree(device_context_);
            if (device_handles_) (void)hipFree(device_handles_);
            registry_ = std::move(other.registry_);
            endpoints_ = std::move(other.endpoints_);
            device_handles_ = other.device_handles_;
            device_context_ = other.device_context_;
            other.device_handles_ = nullptr;
            other.device_context_ = nullptr;
        }
        return *this;
    }

    // Register callback (CPU-side) - global across all endpoints
    // Returns callback ID (0-255)
    // Each invoke is one queue message of the arguments' size, so every
    // endpoint ring must hold message_lines(argument bytes) lines; throws
    // std::invalid_argument if it cannot (the push would trap on the device).
    template<typename... Args>
    uint8_t register_callback(std::function<void(Args...)> func) {
        constexpr size_t arg_bytes = internal::ArgumentSerializer<Args...>::total_bytes;
        static_assert(arg_bytes <= kiwi::queue::kMaxMessageBytes,
                      "callback arguments exceed kiwi::queue::kMaxMessageBytes");
        const size_t lines = kiwi::queue::message_lines(arg_bytes);
        const size_t capacity = endpoints_.front()->capacity();
        if (lines > capacity) {
            throw std::invalid_argument(
                "invoke callback arguments of " + std::to_string(arg_bytes) +
                " bytes need " + std::to_string(lines) +
                " queue lines, but the queue capacity is " + std::to_string(capacity) +
                " lines");
        }
        return registry_.register_callback<Args...>(std::move(func));
    }

    // Process callbacks from specific endpoint (CPU-side)
    // max_callbacks = -1 means process all available
    // Returns number of callbacks executed
    size_t progress(size_t endpoint_id, int max_callbacks = -1) {
        if (endpoint_id >= endpoints_.size()) {
            throw std::out_of_range(
                "Invalid endpoint_id: " + std::to_string(endpoint_id) +
                " (num_endpoints: " + std::to_string(endpoints_.size()) + ")"
            );
        }
        return endpoints_[endpoint_id]->progress(max_callbacks);
    }

    // Get number of endpoints
    size_t num_endpoints() const {
        return endpoints_.size();
    }

    // Get device-side context pointer (pass to kernels, call get_device_handle() inside)
    DeviceContext<QueueType>* device_context() const { return device_context_; }

    // Get raw queue pointer for an endpoint (for benchmarking bypass)
    QueueType* raw_queue(size_t endpoint_id) const {
        if (endpoint_id >= endpoints_.size()) {
            throw std::out_of_range(
                "Invalid endpoint_id: " + std::to_string(endpoint_id) +
                " (num_endpoints: " + std::to_string(endpoints_.size()) + ")"
            );
        }
        return endpoints_[endpoint_id]->raw_queue();
    }

private:
    internal::CallbackRegistry registry_;
    std::vector<std::unique_ptr<internal::Endpoint<QueueType>>> endpoints_;
    DeviceHandle* device_handles_ = nullptr;           // device-allocated array
    DeviceContext<QueueType>* device_context_ = nullptr; // device-allocated context
};

}  // namespace kiwi::invoke
