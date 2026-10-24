#pragma once

#include "callback_wrapper.hpp"
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace mori::kiwi::invoke {
namespace internal {

// Callback registration and lookup manager
class CallbackRegistry {
public:
    CallbackRegistry() = default;

    // Register a callback, returns callback ID (0-255)
    template<typename... Args>
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
            throw std::out_of_range(
                "Invalid callback_id: " + std::to_string(callback_id) +
                " (registry size: " + std::to_string(callbacks_.size()) + ")"
            );
        }
        return callbacks_[callback_id];
    }

    // Get expected argument size for callback
    size_t expected_arg_bytes(uint8_t callback_id) const {
        return get(callback_id).expected_arg_bytes();
    }

    // Get number of registered callbacks
    size_t size() const {
        return callbacks_.size();
    }

private:
    std::vector<CallbackWrapper> callbacks_;
};

}  // namespace internal
}  // namespace kiwi::invoke
