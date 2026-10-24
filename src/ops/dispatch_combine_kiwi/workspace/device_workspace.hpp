#pragma once

#include <cstddef>
#include <cstdint>
#include "ep_config.hpp"

namespace mori::kiwi::ep {

// GPU-local EP scratch/workspace layout.  Keep this separate from the
// RDMA-visible symmetric heap and from host-polled pinned/coherent state: these
// fields are only consumed by GPU kernels or stream-ordered setup kernels.
// A single typed layout makes the ownership/semantics of hot scratch explicit
// without merging it into the communication heap.
struct ep_device_workspace_layout_t {
    size_t dest_token_counts_offset = 0;
    size_t dispatch_dest_token_counts_snapshot_offset = 0;
    size_t dispatch_recv_counts_offset = 0;
    size_t recv_count_offset = 0;
    size_t grid_barrier_offset = 0;
    size_t recv_src_rank_map_offset = 0;
    size_t recv_local_slot_map_offset = 0;
    size_t recv_begin_offset = 0;
    size_t token_to_producer_rank_offset = 0;
    size_t token_to_dispatch_recv_slot_offset = 0;
    size_t barrier_counter_offset = 0;
    size_t all2all_counts_recv_offset = 0;
    size_t num_bytes = 0;

    ep_device_workspace_layout_t() = default;
    explicit ep_device_workspace_layout_t(const config_t& cfg) {
        size_t off = 0;
        auto reserve = [&off](size_t bytes, size_t alignment) {
            off = align_up(off, alignment);
            const size_t result = off;
            off += bytes;
            return result;
        };

        dest_token_counts_offset =
            reserve(static_cast<size_t>(cfg.world_size) * sizeof(int32_t),
                    alignof(int32_t));
        dispatch_dest_token_counts_snapshot_offset =
            reserve(static_cast<size_t>(cfg.world_size) * sizeof(int32_t),
                    alignof(int32_t));
        dispatch_recv_counts_offset =
            reserve(static_cast<size_t>(cfg.world_size + 1) * sizeof(int32_t),
                    alignof(int32_t));
        recv_count_offset = reserve(sizeof(int32_t), alignof(int32_t));
        grid_barrier_offset = reserve(sizeof(uint64_t), alignof(uint64_t));
        recv_src_rank_map_offset =
            reserve(static_cast<size_t>(cfg.max_slots()) * sizeof(int32_t),
                    alignof(int32_t));
        recv_local_slot_map_offset =
            reserve(static_cast<size_t>(cfg.max_slots()) * sizeof(int32_t),
                    alignof(int32_t));
        recv_begin_offset =
            reserve(static_cast<size_t>(cfg.world_size + 1) * sizeof(int32_t),
                    alignof(int32_t));
        token_to_producer_rank_offset =
            reserve(cfg.token_to_producer_rank_size(), alignof(int32_t));
        token_to_dispatch_recv_slot_offset =
            reserve(cfg.token_to_dispatch_recv_slot_size(), alignof(int32_t));
        barrier_counter_offset =
            reserve(sizeof(uint64_t), alignof(uint64_t));
        all2all_counts_recv_offset =
            reserve(static_cast<size_t>(cfg.world_size) * sizeof(uint64_t),
                    alignof(uint64_t));
        num_bytes = align_up(off, 16);
    }

    size_t get_num_bytes() const { return num_bytes; }

    template <typename T>
    static __host__ __device__ T* ptr(char* base, size_t offset) {
        return reinterpret_cast<T*>(base + offset);
    }
};

// Concrete GPU workspace handle passed to kernels.  The handle is intentionally
// just a base pointer plus layout offsets: user input/output tensors stay as
// explicit kernel arguments, while hot scratch/control state is accessed through
// typed helpers here.
struct ep_device_workspace_t {
    char* base = nullptr;
    ep_device_workspace_layout_t layout;

    template <typename T>
    __host__ __device__ T* ptr(size_t offset) const {
        return reinterpret_cast<T*>(base + offset);
    }

    __host__ __device__ int32_t* dest_token_counts() const {
        return ptr<int32_t>(layout.dest_token_counts_offset);
    }
    __host__ __device__ int32_t* dispatch_dest_token_counts_snapshot() const {
        return ptr<int32_t>(layout.dispatch_dest_token_counts_snapshot_offset);
    }
    __host__ __device__ int32_t* dispatch_recv_counts() const {
        return ptr<int32_t>(layout.dispatch_recv_counts_offset);
    }
    __host__ __device__ int32_t* recv_count() const {
        return ptr<int32_t>(layout.recv_count_offset);
    }
    __host__ __device__ int32_t* token_to_producer_rank() const {
        return ptr<int32_t>(layout.token_to_producer_rank_offset);
    }
    __host__ __device__ int32_t* token_to_dispatch_recv_slot() const {
        return ptr<int32_t>(layout.token_to_dispatch_recv_slot_offset);
    }
    __host__ __device__ uint64_t* grid_barrier() const {
        return ptr<uint64_t>(layout.grid_barrier_offset);
    }
    __host__ __device__ int32_t* recv_src_rank_map() const {
        return ptr<int32_t>(layout.recv_src_rank_map_offset);
    }
    __host__ __device__ int32_t* recv_local_slot_map() const {
        return ptr<int32_t>(layout.recv_local_slot_map_offset);
    }
    __host__ __device__ int32_t* recv_begin() const {
        return ptr<int32_t>(layout.recv_begin_offset);
    }
    __host__ __device__ uint64_t* barrier_counter() const {
        return ptr<uint64_t>(layout.barrier_counter_offset);
    }
    __host__ __device__ uint64_t* all2all_counts_recv() const {
        return ptr<uint64_t>(layout.all2all_counts_recv_offset);
    }
};


} // namespace kiwi::ep
