// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Shared memory region management for SPDK Proxy IPC.
// Server (proxy daemon) creates; clients (rank processes) attach.
//
#pragma once

#include <cstddef>
#include <string>

#include "umbp/proxy/spdk_proxy_protocol.h"

namespace umbp {
namespace proxy {

class ProxyShmRegion {
   public:
    ProxyShmRegion() = default;
    ~ProxyShmRegion();

    ProxyShmRegion(const ProxyShmRegion&) = delete;
    ProxyShmRegion& operator=(const ProxyShmRegion&) = delete;

    // Server: create and initialize shared memory.
    // If try_hugepage=true, attempts /dev/hugepages/ for DMA-capable zero-copy.
    // Falls back to regular shm_open if hugepages unavailable.
    // cache_budget_mb: total cache budget in MB (ring buffer + hash index).
    //                  0 = no shared cache.
    // Returns 0 on success, -errno on failure.
    int Create(const std::string& name, uint32_t max_ranks,
               size_t data_per_rank = kDefaultDataRegionPerRank,
               bool try_hugepage = true,
               size_t cache_budget_mb = 0);

    // Client: attach to existing shared memory created by server.
    // Returns 0 on success, -errno on failure.
    int Attach(const std::string& name);

    // Detach (unmap). Server also unlinks the name.
    void Detach();

    bool IsValid() const { return base_ != nullptr; }
    bool IsServer() const { return is_server_; }
    bool IsHugepage() const { return is_hugepage_; }

    // Access the header
    ProxyShmHeader* Header() {
        return static_cast<ProxyShmHeader*>(base_);
    }
    const ProxyShmHeader* Header() const {
        return static_cast<const ProxyShmHeader*>(base_);
    }

    // Access a rank channel
    RankChannel* Channel(uint32_t rank) {
        return GetChannel(base_, Header(), rank);
    }

    // Access a rank's data region
    void* DataRegion(uint32_t rank) {
        return proxy::GetDataRegion(base_, Header(), rank);
    }

    void* Base() { return base_; }
    size_t Size() const { return size_; }

    // ---- Lifecycle helpers (static, for use before attach) ----

    // Probe whether a live proxy already owns SHM |name|.
    //   1 = proxy alive and READY
    //   0 = no SHM or proxy is dead (caller should clean stale SHM)
    //  -1 = SHM exists, proxy alive but not yet READY
    static int ProbeExisting(const std::string& name);

    // Forcefully unlink any stale SHM (hugepage file + shm_open).
    static void CleanupStale(const std::string& name);

   private:
    void* base_ = nullptr;
    size_t size_ = 0;
    std::string name_;
    std::string hp_path_;       // hugepage file path (empty if using shm_open)
    bool is_server_ = false;
    bool is_hugepage_ = false;
    int fd_ = -1;
};

}  // namespace proxy
}  // namespace umbp
