// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// spdk_proxy: Standalone daemon that exclusively owns SPDK + NVMe device.
// Rank processes communicate via POSIX shared memory (hugepage-backed for
// zero-copy DMA when available).
//
// SHM shared cache: on write, data is persisted to NVMe and simultaneously
// cached in seqlock-protected SHM slots.  Client ranks can read these slots
// directly (single memcpy, no daemon IPC) for cache hits.  Cache misses fall
// through to NVMe.  Controlled by UMBP_SPDK_PROXY_CACHE_MB (default 8192).
//
// Self-managed lifecycle:
//   - Stores proxy_pid in SHM header for liveness detection.
//   - Updates proxy_heartbeat_ms ~every 500ms so clients detect crashes.
//   - Periodically checks rank PIDs; dead ranks are reaped automatically.
//   - Exits when: (shutdown_requested OR spawner dead) AND active_ranks==0.
//   - Accepts --spawner-pid for auto-fork mode (monitors parent process).
//
// Usage (manual):
//   UMBP_SPDK_NVME_PCI=0000:07:00.0 UMBP_SPDK_MEM_MB=4096 ./spdk_proxy
//
// Usage (auto-fork — launched by LocalStorageManager):
//   UMBP_SPDK_NVME_PCI=... ./spdk_proxy --spawner-pid 12345

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "umbp/common/config.h"
#include "umbp/common/log.h"
#include "umbp/proxy/spdk_proxy_protocol.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#include "umbp/spdk/spdk_env.h"
#include "umbp/storage/spdk_ssd_tier.h"

#ifdef __linux__
#include <sys/prctl.h>
#include <unistd.h>
#endif

using namespace umbp::proxy;

static std::atomic<bool> g_running{true};
static std::string g_shm_name;

// Staged data for a single Write-Back deferred NVMe flush.
struct WbFlushTask {
    std::vector<std::string> keys;
    std::vector<std::vector<char>> staged_bufs;
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    std::vector<size_t> offsets;
};

// Per-rank background flush queue.
// Allows batch_worker to return immediately after ShmCache + COMPLETED,
// while NVMe I/O runs asynchronously in a dedicated flush thread.
// Flush() drains the queue, guaranteeing cross-rank read-after-write visibility.
class WbFlushQueue {
   public:
    void Start(SpdkSsdTier* tier) {
        tier_ = tier;
        thread_ = std::thread([this] { Run(); });
    }
    void Stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }
    void Push(WbFlushTask&& task) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(std::move(task));
            ++pending_;
            ++submit_seq_;
        }
        cv_.notify_one();
    }
    // Block until all tasks submitted before this call are flushed to NVMe.
    void Drain() {
        std::unique_lock<std::mutex> lk(mu_);
        uint64_t target = submit_seq_;
        done_cv_.wait(lk, [&] { return done_seq_ >= target || stop_; });
    }

   private:
    void Run() {
        while (true) {
            WbFlushTask task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !queue_.empty() || stop_; });
                if (stop_ && queue_.empty()) break;
                task = std::move(queue_.front());
                queue_.pop();
            }
            tier_->BatchWriteStreaming(
                task.keys, task.ptrs, task.sizes,
                nullptr, task.offsets, nullptr, 0);
            {
                std::lock_guard<std::mutex> lk(mu_);
                --pending_;
                ++done_seq_;
            }
            done_cv_.notify_all();
        }
    }

    SpdkSsdTier* tier_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;       // wakes flush thread
    std::condition_variable done_cv_;  // wakes Drain() callers
    std::queue<WbFlushTask> queue_;
    int pending_ = 0;
    uint64_t submit_seq_ = 0;   // incremented on Push
    uint64_t done_seq_ = 0;     // incremented on flush complete
    bool stop_ = false;
};

static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

static void atexit_cleanup() {
#ifdef __linux__
    if (!g_shm_name.empty())
        ProxyShmRegion::CleanupStale(g_shm_name);
#endif
}

// ---------------------------------------------------------------------------
// Write data into a specific SHM cache slot (seqlock-protected).
// ---------------------------------------------------------------------------
static void WriteShmCacheSlotAt(ProxyShmRegion& shm, uint32_t idx,
                                const std::string& key,
                                const void* data, size_t size) {
    auto* hdr = shm.Header();
    auto* slot = GetCacheSlot(shm.Base(), hdr, idx);
    char* slot_data = GetCacheSlotData(shm.Base(), hdr, idx);

    uint64_t old = slot->gen.load(std::memory_order_relaxed);
    slot->gen.store(old + 1, std::memory_order_release);          // odd → writing

    slot->key_len = static_cast<uint32_t>(key.size());
    std::memcpy(slot->key, key.data(), key.size());
    slot->data_size = size;
    std::memcpy(slot_data, data, size);

    slot->gen.store(old + 2, std::memory_order_release);          // even → stable
}

// ---------------------------------------------------------------------------
// Write a key/value into the SHM shared cache.
// Items ≤ slot capacity go into a single slot (hash-indexed).
// Larger items are chunked into ≤cap pieces placed at contiguous slots
// starting from hash(original_key), eliminating self-collision.
// ---------------------------------------------------------------------------
static void WriteShmCache(ProxyShmRegion& shm, const std::string& key,
                          const void* data, size_t size) {
    auto* hdr = shm.Header();
    if (hdr->cache_num_slots == 0 || size == 0) return;
    size_t cap = CacheSlotDataCapacity(hdr);
    uint32_t ns = hdr->cache_num_slots;

    if (size <= cap) {
        uint32_t idx = CacheSlotIndex(key.data(),
                                      static_cast<uint32_t>(key.size()), ns);
        WriteShmCacheSlotAt(shm, idx, key, data, size);
        return;
    }

    uint32_t base = CacheSlotIndex(key.data(),
                                   static_cast<uint32_t>(key.size()), ns);
    const char* src = static_cast<const char*>(data);
    size_t remaining = size;
    for (uint32_t ci = 0; remaining > 0; ++ci) {
        size_t chunk_sz = std::min(cap, remaining);
        std::string ck = key + ":c" + std::to_string(ci);
        if (ck.size() > kMaxKeyLen) return;
        uint32_t idx = (base + ci) % ns;
        WriteShmCacheSlotAt(shm, idx, ck, src, chunk_sz);
        src += chunk_sz;
        remaining -= chunk_sz;
    }
}

// ---------------------------------------------------------------------------
// Process a single non-batch request
// ---------------------------------------------------------------------------
static void ProcessSingleRequest(SpdkSsdTier& tier,
                                 ProxyShmRegion& shm,
                                 RingSlot& slot,
                                 void* data_region, size_t region_size) {
    std::string key(slot.key, slot.key_len);
    auto type = static_cast<RequestType>(slot.type);

    switch (type) {
        case RequestType::WRITE: {
            if (slot.data_size > region_size) {
                slot.result = static_cast<int32_t>(ResultCode::ERROR);
                break;
            }
            bool ok = tier.Write(key, data_region, slot.data_size);
            slot.result = ok ? static_cast<int32_t>(ResultCode::OK)
                             : static_cast<int32_t>(ResultCode::NO_SPACE);
            if (ok)
                WriteShmCache(shm, key, data_region, slot.data_size);
            break;
        }
        case RequestType::READ: {
            size_t max_read = std::min(static_cast<size_t>(slot.data_size),
                                       region_size);
            if (max_read == 0) max_read = region_size;
            auto dst = reinterpret_cast<uintptr_t>(data_region);
            bool ok = tier.ReadIntoPtr(key, dst, max_read);
            slot.result = ok ? static_cast<int32_t>(ResultCode::OK)
                             : static_cast<int32_t>(ResultCode::NOT_FOUND);
            if (ok) {
                slot.result_size = max_read;
                WriteShmCache(shm, key, data_region, max_read);
            }
            break;
        }
        case RequestType::EXISTS: {
            bool ok = tier.Exists(key);
            slot.result = ok ? static_cast<int32_t>(ResultCode::OK)
                             : static_cast<int32_t>(ResultCode::NOT_FOUND);
            break;
        }
        case RequestType::EVICT: {
            bool ok = tier.Evict(key);
            slot.result = ok ? static_cast<int32_t>(ResultCode::OK)
                             : static_cast<int32_t>(ResultCode::NOT_FOUND);
            break;
        }
        case RequestType::CLEAR: {
            tier.Clear();
            slot.result = static_cast<int32_t>(ResultCode::OK);
            break;
        }
        case RequestType::CAPACITY: {
            auto [used, total] = tier.Capacity();
            slot.result = static_cast<int32_t>(ResultCode::OK);
            slot.result_size = used;
            slot.result_aux = total;
            break;
        }
        case RequestType::FLUSH: {
            // The actual drain happens in PollLoop before this is called.
            slot.result = static_cast<int32_t>(ResultCode::OK);
            break;
        }
        case RequestType::GET_LRU_KEY: {
            std::string lru = tier.GetLRUKey();
            if (lru.empty()) {
                slot.result = static_cast<int32_t>(ResultCode::NOT_FOUND);
                slot.result_size = 0;
            } else {
                size_t copy_len = std::min(lru.size(), region_size - 1);
                std::memcpy(data_region, lru.data(), copy_len);
                static_cast<char*>(data_region)[copy_len] = '\0';
                slot.result = static_cast<int32_t>(ResultCode::OK);
                slot.result_size = copy_len;
            }
            break;
        }
        default:
            slot.result = static_cast<int32_t>(ResultCode::ERROR);
            break;
    }
}

// ---------------------------------------------------------------------------
// Process a batch request with optional cache acceleration.
// ---------------------------------------------------------------------------
static void ProcessBatchRequest(SpdkSsdTier& tier,
                                ProxyShmRegion& shm,
                                RingSlot& slot,
                                void* data_region, size_t region_size,
                                bool write_back,
                                void** dma_bufs = nullptr, int dma_count = 0,
                                WbFlushTask* wb_out = nullptr) {
    auto type = static_cast<RequestType>(slot.type);
    auto* desc = static_cast<BatchDescriptor*>(data_region);
    uint32_t count = desc->count;

    if (count == 0) {
        slot.result = static_cast<int32_t>(ResultCode::OK);
        return;
    }

    size_t desc_total = sizeof(BatchDescriptor) + count * sizeof(BatchEntry);
    size_t data_base_offset = (desc_total + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
    char* data_base = static_cast<char*>(data_region) + data_base_offset;

    if (type == RequestType::BATCH_WRITE) {
        std::vector<std::string> keys(count);
        std::vector<const void*> cptrs(count);
        std::vector<size_t> sizes(count);
        std::vector<size_t> shm_offsets(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& e = desc->entries[i];
            keys[i] = std::string(e.key, e.key_len);
            cptrs[i] = data_base + e.data_offset;
            sizes[i] = e.data_size;
            shm_offsets[i] = e.data_offset;
        }

        if (write_back && wb_out) {
            // Write-Back fast path: ShmCache + results + stage data to heap.
            // The caller pushes the staged task to a background flush queue,
            // then immediately releases the rank (batch_inflight=false).
            // NVMe persistence runs asynchronously; Flush() drains the queue.
            uint64_t total = desc->total_data_size;
            while (desc->bytes_ready.load(std::memory_order_acquire) < total)
                ;
            for (uint32_t i = 0; i < count; ++i) {
                WriteShmCache(shm, keys[i], cptrs[i], sizes[i]);
                desc->entries[i].result = 1;
            }

            wb_out->keys = std::move(keys);
            wb_out->sizes = std::move(sizes);
            wb_out->offsets = std::move(shm_offsets);
            wb_out->staged_bufs.resize(count);
            wb_out->ptrs.resize(count);
            for (uint32_t i = 0; i < count; ++i) {
                wb_out->staged_bufs[i].assign(
                    static_cast<const char*>(cptrs[i]),
                    static_cast<const char*>(cptrs[i]) + wb_out->sizes[i]);
                wb_out->ptrs[i] = wb_out->staged_bufs[i].data();
            }
        } else {
            // Write-Through: NVMe write + parallel ShmCache population
            std::thread shm_cache_thread([&shm, &keys, &cptrs, &sizes, count, desc]() {
                uint64_t total = desc->total_data_size;
                while (desc->bytes_ready.load(std::memory_order_acquire) < total)
                    ;
                for (uint32_t i = 0; i < count; ++i)
                    WriteShmCache(shm, keys[i], cptrs[i], sizes[i]);
            });

            auto results = tier.BatchWriteStreaming(
                keys, cptrs, sizes, &desc->bytes_ready, shm_offsets,
                dma_bufs, dma_count);

            shm_cache_thread.join();

            int write_ok = 0;
            for (uint32_t i = 0; i < count; ++i) {
                desc->entries[i].result = results[i] ? 1 : 0;
                if (results[i]) ++write_ok;
            }
            if (write_ok < static_cast<int>(count)) {
                auto [used, total] = tier.Capacity();
                UMBP_LOG_WARN("spdk_proxy: BATCH_WRITE %u keys — only %d succeeded "
                              "(cap %zuMB/%zuMB)",
                              count, write_ok, used / (1024*1024), total / (1024*1024));
            }
        }
    } else {
        // BATCH_READ — NVMe streaming read + ShmCache back-fill
        std::vector<std::string> keys(count);
        std::vector<void*> dst_ptrs(count);
        std::vector<size_t> sizes(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& e = desc->entries[i];
            keys[i] = std::string(e.key, e.key_len);
            dst_ptrs[i] = static_cast<void*>(data_base + e.data_offset);
            sizes[i] = e.data_size;
        }

        desc->bytes_done.store(0, std::memory_order_relaxed);
        desc->items_done.store(0, std::memory_order_relaxed);

        std::vector<uintptr_t> dma_ptrs(count);
        std::vector<size_t> shm_offsets(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& e = desc->entries[i];
            dma_ptrs[i] = reinterpret_cast<uintptr_t>(
                data_base + e.data_offset);
            shm_offsets[i] = e.data_offset;
        }

        auto results = tier.BatchReadIntoPtrStreaming(
            keys, dma_ptrs, sizes, &desc->items_done,
            &desc->bytes_done, &shm_offsets,
            dma_bufs, dma_count);
        desc->bytes_done.store(desc->total_data_size,
                               std::memory_order_release);

        int read_hits = 0;
        for (uint32_t i = 0; i < count; ++i) {
            desc->entries[i].result = results[i] ? 1 : 0;
            if (results[i]) ++read_hits;
        }
        if (read_hits == 0 && count > 0) {
            auto [used, total] = tier.Capacity();
            bool first_exists = tier.Exists(keys[0]);
            UMBP_LOG_WARN("spdk_proxy: BATCH_READ %u keys — 0 NVMe hits! "
                          "capacity=%zuMB/%zuMB exists('%s')=%d",
                          count, used / (1024*1024), total / (1024*1024),
                          keys[0].substr(0, 64).c_str(), first_exists);
        }

        // Read-back-fill: populate ShmCache so subsequent reads by other
        // ranks can hit the SHM shared cache directly.
        for (uint32_t i = 0; i < count; ++i) {
            if (results[i])
                WriteShmCache(shm, keys[i], dst_ptrs[i], sizes[i]);
        }
    }

    slot.result = static_cast<int32_t>(ResultCode::OK);
}

// ---------------------------------------------------------------------------
// Per-rank DMA buffer pool — allows concurrent batch I/O without lock contention
// on SpdkSsdTier's global dma_ring_mu_.
// ---------------------------------------------------------------------------
static constexpr int kDmaBufsPerRank = 128;
static constexpr size_t kDmaBufSize = 2ULL * 1024 * 1024;

struct RankDmaPool {
    void** bufs = nullptr;
    int count = 0;
};

static void AllocRankDmaPools(RankDmaPool* pools, int max_ranks) {
    static constexpr int kTotalDmaBudget = 1024;
    int per_rank = std::min(kDmaBufsPerRank,
                            std::max(16, kTotalDmaBudget / std::max(1, max_ranks)));

    auto& env = umbp::SpdkEnv::Instance();
    for (int r = 0; r < max_ranks; ++r) {
        pools[r].bufs = new void*[per_rank];
        int got = env.DmaPoolAllocBatch(pools[r].bufs, kDmaBufSize, per_rank);
        pools[r].count = got;
        if (got < per_rank) {
            for (int i = got; i < per_rank; ++i) pools[r].bufs[i] = nullptr;
            if (got == 0)
                UMBP_LOG_WARN("spdk_proxy: rank %d: 0 DMA bufs allocated!", r);
        }
    }
    UMBP_LOG_INFO("spdk_proxy: allocated %d DMA bufs/rank × %d ranks (%zuMB each)",
                  per_rank, max_ranks, kDmaBufSize / (1024 * 1024));
}

static void FreeRankDmaPools(RankDmaPool* pools, int max_ranks) {
    auto& env = umbp::SpdkEnv::Instance();
    for (int r = 0; r < max_ranks; ++r) {
        if (pools[r].bufs && pools[r].count > 0)
            env.DmaPoolFreeBatch(pools[r].bufs, kDmaBufSize, pools[r].count);
        delete[] pools[r].bufs;
        pools[r].bufs = nullptr;
        pools[r].count = 0;
    }
}

// ---------------------------------------------------------------------------
// Main poll loop
// ---------------------------------------------------------------------------
static constexpr int kMaxConcurrentRanks = 64;

static void PollLoop(SpdkSsdTier& tier,
                     ProxyShmRegion& shm,
                     pid_t spawner_pid, RankDmaPool* rank_dma,
                     bool write_back) {
    auto* hdr = shm.Header();
    uint32_t max_ranks = std::min(hdr->max_ranks,
                                  static_cast<uint32_t>(kMaxConcurrentRanks));

    std::atomic<bool> batch_inflight[kMaxConcurrentRanks] = {};
    std::thread       batch_worker[kMaxConcurrentRanks];

    // Per-rank background flush queues (WB mode only).
    WbFlushQueue wb_queues[kMaxConcurrentRanks];
    if (write_back) {
        for (uint32_t r = 0; r < max_ranks; ++r)
            wb_queues[r].Start(&tier);
    }

    UMBP_LOG_INFO("spdk_proxy: entering poll loop (max_ranks=%u, spawner=%d)",
                  max_ranks, static_cast<int>(spawner_pid));

    auto last_heartbeat = std::chrono::steady_clock::now();
    auto last_reap = last_heartbeat;

    while (g_running.load(std::memory_order_relaxed)) {
        bool any_work = false;

        for (uint32_t r = 0; r < max_ranks; ++r) {
            if (batch_inflight[r].load(std::memory_order_acquire)) continue;

            auto* ch = shm.Channel(r);
            if (!ch->connected) continue;

            uint32_t t = ch->tail.load(std::memory_order_relaxed);
            uint32_t h = ch->head.load(std::memory_order_acquire);
            if (t == h) continue;

            auto& slot = ch->slots[t % kRingSize];
            uint32_t st = slot.state.load(std::memory_order_acquire);
            if (st != static_cast<uint32_t>(SlotState::PENDING)) continue;

            any_work = true;

            void* data_region = shm.DataRegion(r);
            size_t region_size = hdr->data_region_per_rank;

            auto rtype = static_cast<RequestType>(slot.type);
            if (rtype == RequestType::BATCH_WRITE ||
                rtype == RequestType::BATCH_READ) {
                if (batch_worker[r].joinable()) batch_worker[r].join();
                batch_inflight[r].store(true, std::memory_order_relaxed);
                void** dma_bufs = rank_dma[r].bufs;
                int dma_count = rank_dma[r].count;
                WbFlushQueue* wbq = write_back ? &wb_queues[r] : nullptr;
                batch_worker[r] = std::thread(
                    [&tier, &shm, &slot, data_region, region_size, ch, t, r,
                     &batch_inflight, dma_bufs, dma_count, write_back, rtype,
                     wbq]() {
                        bool is_wb_write = write_back &&
                            rtype == RequestType::BATCH_WRITE;
                        WbFlushTask wb_task;
                        ProcessBatchRequest(tier, shm, slot, data_region,
                                            region_size, write_back,
                                            dma_bufs, dma_count,
                                            is_wb_write ? &wb_task : nullptr);
                        slot.state.store(
                            static_cast<uint32_t>(SlotState::COMPLETED),
                            std::memory_order_release);
                        ch->tail.store((t + 1) % kRingSize,
                                       std::memory_order_release);
                        if (is_wb_write && wbq && !wb_task.keys.empty())
                            wbq->Push(std::move(wb_task));
                        batch_inflight[r].store(false,
                                                std::memory_order_release);
                    });
            } else {
                if (rtype == RequestType::FLUSH && write_back) {
                    // Drain all pending WB NVMe flushes for this rank
                    // before signaling COMPLETED.
                    wb_queues[r].Drain();
                }
                ProcessSingleRequest(tier, shm, slot, data_region, region_size);
                slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED),
                                 std::memory_order_release);
                ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
            }
        }

        auto now = std::chrono::steady_clock::now();

        auto since_hb = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_heartbeat);
        if (since_hb.count() > 500) {
            hdr->proxy_heartbeat_ms.store(NowEpochMs(),
                                          std::memory_order_relaxed);
            last_heartbeat = now;
        }

        auto since_reap = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_reap);
        if (since_reap.count() >= 5) {
#ifdef __linux__
            for (uint32_t r = 0; r < max_ranks; ++r) {
                if (batch_inflight[r].load(std::memory_order_relaxed)) continue;
                uint32_t rpid = hdr->rank_pids[r].load(std::memory_order_relaxed);
                if (rpid > 0 && kill(static_cast<pid_t>(rpid), 0) != 0) {
                    UMBP_LOG_WARN("spdk_proxy: rank %u pid %u appears dead, cleaning up",
                                  r, rpid);
                    hdr->rank_pids[r].store(0, std::memory_order_relaxed);
                    hdr->active_ranks.fetch_sub(1, std::memory_order_relaxed);
                    auto* ch = shm.Channel(r);
                    ch->connected = 0;
                    ch->owner_pid.store(0, std::memory_order_release);
                    uint32_t t = ch->tail.load(std::memory_order_relaxed);
                    uint32_t h = ch->head.load(std::memory_order_relaxed);
                    while (t != h) {
                        auto& s = ch->slots[t % kRingSize];
                        s.result = static_cast<int32_t>(ResultCode::ERROR);
                        s.state.store(static_cast<uint32_t>(SlotState::COMPLETED),
                                      std::memory_order_release);
                        t = (t + 1) % kRingSize;
                    }
                    ch->tail.store(t, std::memory_order_release);
                }
            }
#endif
            last_reap = now;
        }

        uint32_t active = hdr->active_ranks.load(std::memory_order_acquire);
        bool shutdown = hdr->shutdown_requested.load(std::memory_order_acquire) != 0;

        if (shutdown && active == 0) {
            UMBP_LOG_INFO("spdk_proxy: shutdown requested and all ranks disconnected");
            break;
        }

#ifdef __linux__
        if (spawner_pid > 0 && kill(spawner_pid, 0) != 0 && active == 0) {
            UMBP_LOG_INFO("spdk_proxy: spawner (pid=%d) is dead and no active ranks",
                          static_cast<int>(spawner_pid));
            break;
        }

        {
            static auto orphan_start = std::chrono::steady_clock::time_point{};
            if (spawner_pid > 0 && kill(spawner_pid, 0) != 0) {
                if (orphan_start == std::chrono::steady_clock::time_point{}) {
                    orphan_start = now;
                    UMBP_LOG_WARN("spdk_proxy: spawner dead, starting 60s orphan countdown");
                }
                auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                    now - orphan_start).count();
                if (secs >= 60) {
                    UMBP_LOG_WARN("spdk_proxy: orphaned 60s, force exit");
                    break;
                }
            } else {
                orphan_start = std::chrono::steady_clock::time_point{};
            }
        }
#endif

        if (!any_work) {
#if defined(__x86_64__) || defined(_M_X64)
            for (int i = 0; i < 32; ++i) _mm_pause();
#endif
        }
    }

    for (uint32_t r = 0; r < max_ranks; ++r) {
        if (batch_worker[r].joinable()) batch_worker[r].join();
    }

    if (write_back) {
        for (uint32_t r = 0; r < max_ranks; ++r)
            wb_queues[r].Stop();
    }

    UMBP_LOG_INFO("spdk_proxy: exiting poll loop");
}

// ---------------------------------------------------------------------------
// Parse command-line arguments
// ---------------------------------------------------------------------------
static pid_t ParseSpawnerPid(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--spawner-pid") == 0 && i + 1 < argc) {
            return static_cast<pid_t>(std::atoi(argv[++i]));
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    pid_t spawner_pid = ParseSpawnerPid(argc, argv);

#ifdef __linux__
    if (spawner_pid > 0) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() != spawner_pid) {
            fprintf(stderr, "spdk_proxy: spawner already dead before prctl, exiting\n");
            return 1;
        }
    }
#endif

    auto getenv_str = [](const char* name, const char* def) -> std::string {
        const char* v = std::getenv(name);
        return v ? v : def;
    };
    auto getenv_int = [](const char* name, int def) -> int {
        const char* v = std::getenv(name);
        return v ? std::atoi(v) : def;
    };
    auto getenv_size = [](const char* name, size_t def) -> size_t {
        const char* v = std::getenv(name);
        return v ? static_cast<size_t>(std::stoull(v)) : def;
    };

    std::string nvme_pci = getenv_str("UMBP_SPDK_NVME_PCI", "");
    std::string nvme_ctrl = getenv_str("UMBP_SPDK_NVME_CTRL", "NVMe0");
    std::string bdev_name = getenv_str("UMBP_SPDK_BDEV", "");
    std::string reactor_mask = getenv_str("UMBP_SPDK_REACTOR_MASK", "0x3");
    int mem_mb = getenv_int("UMBP_SPDK_MEM_MB", 4096);
    int io_workers = getenv_int("UMBP_SPDK_IO_WORKERS", 4);
    size_t ssd_cap = getenv_size("UMBP_SSD_CAPACITY", 0);

    std::string shm_name = getenv_str("UMBP_SPDK_PROXY_SHM", kDefaultShmName);
    int max_ranks = getenv_int("UMBP_SPDK_PROXY_MAX_RANKS", 8);
    size_t data_per_rank_mb = getenv_size("UMBP_SPDK_PROXY_DATA_MB", 2048);
    size_t data_per_rank = data_per_rank_mb * 1024 * 1024;
    bool write_back = getenv_int("UMBP_SPDK_PROXY_WRITE_BACK", 0) != 0;

    if (nvme_pci.empty() && bdev_name.empty()) {
        fprintf(stderr,
                "spdk_proxy: UMBP_SPDK_NVME_PCI or UMBP_SPDK_BDEV required\n");
        return 1;
    }

    g_shm_name = shm_name;
    std::atexit(atexit_cleanup);

    // SHM shared cache: direct-mapped, seqlock per slot.
    // Slot total = 2MB data capacity + 4KB metadata so that 2MB items fit.
    static constexpr size_t kDefaultCacheSlotSize =
        2ULL * 1024 * 1024 + umbp::proxy::kCacheSlotMetaSize;
    size_t shm_cache_mb = getenv_size("UMBP_SPDK_PROXY_CACHE_MB", 8192);
    size_t cache_slot_sz = kDefaultCacheSlotSize;
    uint32_t cache_slots = (shm_cache_mb > 0)
        ? static_cast<uint32_t>((shm_cache_mb * 1024 * 1024) / cache_slot_sz)
        : 0;

    ProxyShmRegion shm;
    int rc = shm.Create(shm_name, max_ranks, data_per_rank, /*try_hugepage=*/true,
                        cache_slot_sz, cache_slots);
    if (rc != 0) {
        fprintf(stderr, "spdk_proxy: failed to create SHM '%s' rc=%d\n",
                shm_name.c_str(), rc);
        return 1;
    }

    auto* hdr = shm.Header();
    hdr->proxy_pid.store(static_cast<uint32_t>(getpid()), std::memory_order_relaxed);
    hdr->spawner_pid.store(static_cast<uint32_t>(spawner_pid), std::memory_order_relaxed);
    hdr->proxy_heartbeat_ms.store(NowEpochMs(), std::memory_order_relaxed);

    UMBP_LOG_INFO("spdk_proxy: SHM created '%s' — %u ranks, %zuMB/rank, total=%zuMB, hugepage=%s, pid=%d",
                  shm_name.c_str(), max_ranks,
                  data_per_rank / (1024 * 1024),
                  shm.Size() / (1024 * 1024),
                  shm.IsHugepage() ? "YES" : "NO",
                  static_cast<int>(getpid()));

    UMBPConfig cfg;
    cfg.ssd_backend = "spdk";
    cfg.spdk_bdev_name = bdev_name;
    cfg.spdk_reactor_mask = reactor_mask;
    cfg.spdk_mem_size_mb = mem_mb;
    cfg.spdk_nvme_pci_addr = nvme_pci;
    cfg.spdk_nvme_ctrl_name = nvme_ctrl;
    cfg.spdk_io_workers = io_workers;
    if (ssd_cap > 0) cfg.ssd_capacity_bytes = ssd_cap;
    else cfg.ssd_capacity_bytes = static_cast<size_t>(-1);

    SpdkSsdTier tier(cfg);
    if (!tier.IsValid()) {
        fprintf(stderr, "spdk_proxy: SpdkSsdTier init failed\n");
        hdr->state.store(static_cast<uint32_t>(ProxyState::ERROR),
                         std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        shm.Detach();
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    auto [used, total] = tier.Capacity();
    hdr->bdev_size = total;
    hdr->block_size = 4096;
    hdr->capacity_used.store(used, std::memory_order_relaxed);
    hdr->capacity_total.store(total, std::memory_order_relaxed);
    hdr->proxy_heartbeat_ms.store(NowEpochMs(), std::memory_order_relaxed);
    hdr->state.store(static_cast<uint32_t>(ProxyState::READY),
                     std::memory_order_release);

    auto* rank_dma = new RankDmaPool[max_ranks];
    AllocRankDmaPools(rank_dma, max_ranks);

    UMBP_LOG_INFO("spdk_proxy: READY — capacity=%zuMB, shm_cache=%u×%zuMB=%zuMB, write_back=%s",
                  total / (1024 * 1024),
                  cache_slots, cache_slot_sz / (1024 * 1024),
                  (size_t)cache_slots * cache_slot_sz / (1024 * 1024),
                  write_back ? "ON" : "OFF");
    fflush(stdout);

    PollLoop(tier, shm, spawner_pid, rank_dma, write_back);

    FreeRankDmaPools(rank_dma, max_ranks);
    delete[] rank_dma;

    hdr->state.store(static_cast<uint32_t>(ProxyState::SHUTDOWN),
                     std::memory_order_release);

    UMBP_LOG_INFO("spdk_proxy: shutting down");
    shm.Detach();

    return 0;
}
