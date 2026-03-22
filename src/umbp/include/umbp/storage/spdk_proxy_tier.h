// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkProxyTier: TierBackend that communicates with an external spdk_proxy
// daemon via POSIX shared memory. Does NOT depend on SPDK headers/libraries.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

    bool Flush() override;

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
    bool IsProxyAlive() const;

    // Per-item ShmCache read — seqlock, returns true on hit.
    bool TryShmCacheReadOne(const std::string& key, uintptr_t dst, size_t size) const;

    // ---- Per-client heap read cache ----
    // Controlled by UMBP_SPDK_READ_CACHE=1 (default OFF).
    // When enabled, first read back-fills the cache; subsequent reads hit.
    struct HeapEntry {
        std::unique_ptr<char[]> data;
        size_t size;
        std::list<std::string>::iterator lru_pos;
    };
    void HeapCachePut(const std::string& key, const void* data, size_t size);
    bool HeapCacheGet(const std::string& key, uintptr_t dst, size_t size) const;

    bool heap_cache_enabled_ = false;
    mutable std::unordered_map<std::string, HeapEntry> heap_cache_;
    mutable std::list<std::string> heap_lru_;
    mutable size_t heap_cache_bytes_ = 0;
    size_t heap_cache_max_bytes_ = 0;

    bool connected_ = false;
    bool cold_read_ = false;    // UMBP_SPDK_COLD_READ=1: skip ring cache, always read from NVMe
    uint32_t rank_id_ = 0;
    mutable umbp::proxy::ProxyShmRegion shm_;
    mutable std::mutex submit_mu_;
    mutable uint64_t seq_counter_ = 0;
};
