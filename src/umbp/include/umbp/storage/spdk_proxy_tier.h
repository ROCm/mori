// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkProxyTier: TierBackend that communicates with an external spdk_proxy
// daemon via POSIX shared memory. Does NOT depend on SPDK headers/libraries.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/proxy/spdk_proxy_protocol.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#include "umbp/storage/tier_backend.h"

class SpdkProxyTier : public TierBackend {
   public:
    explicit SpdkProxyTier(const UMBPConfig& config);
    ~SpdkProxyTier() override;

    bool IsValid() const { return connected_; }
    uint32_t rank_id() const { return rank_id_; }

    bool Write(const std::string& key, const void* data, size_t size) override;
    bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
    bool Exists(const std::string& key) const override;
    bool Evict(const std::string& key) override;
    std::pair<size_t, size_t> Capacity() const override;
    void Clear() override;

    std::vector<bool> BatchWrite(
        const std::vector<std::string>& keys,
        const std::vector<const void*>& data_ptrs,
        const std::vector<size_t>& sizes) override;

    std::vector<bool> BatchReadIntoPtr(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dst_ptrs,
        const std::vector<size_t>& sizes) override;

    std::string GetLRUKey() const override;
    std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

    // Wait for the proxy daemon to become READY (polls SHM).
    // Returns true if READY within timeout, false on timeout or ERROR.
    static bool WaitForProxy(const std::string& shm_name, int timeout_ms);

   private:
    umbp::proxy::ResultCode SubmitAndWait(
        umbp::proxy::RequestType type,
        const std::string& key,
        const void* write_data, size_t write_size,
        void* read_buf, size_t read_buf_size,
        uint64_t* out_result_size = nullptr,
        uint64_t* out_result_aux = nullptr) const;

    std::vector<bool> SubmitBatch(
        umbp::proxy::RequestType type,
        const std::vector<std::string>& keys,
        const std::vector<const void*>& data_ptrs,
        const std::vector<uintptr_t>& dst_ptrs,
        const std::vector<size_t>& sizes) const;

    uint32_t NextSeqId() const;

    // Check if the proxy daemon is still alive by examining its heartbeat.
    bool IsProxyAlive() const;

    // Try to serve a batch read entirely from the SHM shared cache.
    // Returns true if ALL keys hit; false on any miss (caller must fall back).
    bool TryShmCacheBatchRead(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dst_ptrs,
        const std::vector<size_t>& sizes) const;

    bool connected_ = false;
    uint32_t rank_id_ = 0;
    mutable umbp::proxy::ProxyShmRegion shm_;
    mutable std::mutex submit_mu_;
    mutable uint64_t seq_counter_ = 0;
};
