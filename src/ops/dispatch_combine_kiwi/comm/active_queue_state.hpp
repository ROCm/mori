#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace mori::kiwi::ep {
namespace internal {

// Callbacks one proxy visit may run from a ready endpoint queue. Draining a few
// back-to-back messages saves a sweep and a network progress call per message,
// while the small bound keeps sweeps fair across endpoints. Per-queue FIFO
// order is unchanged, and a post that finds the send queue full polls the
// network before it retries.
inline constexpr int kEpMaxCallbacksPerEndpointSweep = 4;

// A callback publishes one requested active-prefix count for all proxy
// threads. A generation changes only when the requested count changes, so
// repeated publications cannot invalidate an in-progress shrink.
class active_queue_request_t {
public:
    struct snapshot_t {
        int active_count;
        uint32_t generation;
    };

    static uint64_t pack_snapshot(snapshot_t snapshot) {
        return (static_cast<uint64_t>(snapshot.generation) << 32)
             | static_cast<uint32_t>(snapshot.active_count);
    }

    static snapshot_t unpack_snapshot(uint64_t value) {
        return snapshot_t{
            static_cast<int>(static_cast<uint32_t>(value)),
            static_cast<uint32_t>(value >> 32),
        };
    }

    explicit active_queue_request_t(int initial_active_count = 1)
        : value_(pack_snapshot(snapshot_t{initial_active_count, 0})) {}

    // Endpoint 0's proxy thread is the sole publisher.
    void publish(int active_count) {
        const snapshot_t previous = load();
        if (previous.active_count == active_count) {
            return;
        }
        value_.store(
            pack_snapshot(
                snapshot_t{active_count, previous.generation + 1u}),
            std::memory_order_relaxed);
    }

    snapshot_t load() const {
        return unpack_snapshot(value_.load(std::memory_order_relaxed));
    }

private:
    std::atomic<uint64_t> value_;
};

// Per-proxy polling-range state. Grows take effect immediately. Shrinks retain
// the old range until this proxy has drained its retiring queues, and only
// commit if the request generation has not changed while it drained them.
class proxy_active_queue_state_t {
public:
    explicit proxy_active_queue_state_t(int initial_active_count = 1)
        : polling_count_(initial_active_count),
          shrinking_to_(initial_active_count) {}

    void observe(active_queue_request_t::snapshot_t request) {
        if (request.active_count >= polling_count_) {
            polling_count_ = request.active_count;
            shrinking_to_ = polling_count_;
            shrink_generation_ = request.generation;
            return;
        }

        if (request.generation != shrink_generation_
            || request.active_count != shrinking_to_) {
            shrinking_to_ = request.active_count;
            shrink_generation_ = request.generation;
        }
    }

    int polling_count() const { return polling_count_; }

    bool has_retiring_queues() const {
        return shrinking_to_ < polling_count_;
    }

    bool is_retiring_queue(int queue_index) const {
        return has_retiring_queues()
            && queue_index >= shrinking_to_
            && queue_index < polling_count_;
    }

    void complete_shrink_if_current(uint32_t latest_generation) {
        if (latest_generation != shrink_generation_) {
            return;
        }

        polling_count_ = shrinking_to_;
    }

private:
    int polling_count_;
    int shrinking_to_;
    uint32_t shrink_generation_ = 0;
};

// Radix-fold endpoint ownership:
// owner(q, P) = sum of the base-P digits of q modulo P.
inline int proxy_queue_owner(int endpoint_id, int num_proxy_threads) {
    if (num_proxy_threads <= 0) {
        throw std::invalid_argument(
            "kiwi::ep::proxy_queue_owner: num_proxy_threads must be positive");
    }
    if (num_proxy_threads == 1) {
        return 0;
    }

    int owner = 0;
    while (endpoint_id > 0) {
        owner = (owner + endpoint_id % num_proxy_threads)
              % num_proxy_threads;
        endpoint_id /= num_proxy_threads;
    }
    return owner;
}

} // namespace internal
} // namespace kiwi::ep
