// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// spdk_proxy: Standalone daemon that exclusively owns SPDK + NVMe device.
// Rank processes communicate via POSIX shared memory (hugepage-backed for
// zero-copy DMA when available).
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
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "umbp/storage/spdk_ssd_tier.h"

#ifdef __linux__
#include <sys/prctl.h>
#include <unistd.h>
#endif

using namespace umbp::proxy;

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// ---------------------------------------------------------------------------
// Process a single non-batch request
// ---------------------------------------------------------------------------
static void ProcessSingleRequest(SpdkSsdTier& tier, RingSlot& slot,
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
            break;
        }
        case RequestType::READ: {
            auto dst = reinterpret_cast<uintptr_t>(data_region);
            size_t max_read = std::min(static_cast<size_t>(slot.data_size),
                                       region_size);
            if (max_read == 0) max_read = region_size;
            bool ok = tier.ReadIntoPtr(key, dst, max_read);
            slot.result = ok ? static_cast<int32_t>(ResultCode::OK)
                             : static_cast<int32_t>(ResultCode::NOT_FOUND);
            if (ok) slot.result_size = max_read;
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
// Process a batch request.
//
// Read:  uses BatchReadIntoPtrStreaming — single pipeline call with per-item
//        items_done signaling so client overlaps SHM→user copy with NVMe I/O.
// Write: waits for all client data, then single BatchWrite call.
// ---------------------------------------------------------------------------
static void ProcessBatchRequest(SpdkSsdTier& tier, RingSlot& slot,
                                void* data_region, size_t region_size) {
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

    std::vector<std::string> keys(count);
    std::vector<size_t> sizes(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& e = desc->entries[i];
        keys[i] = std::string(e.key, e.key_len);
        sizes[i] = e.data_size;
    }

    if (type == RequestType::BATCH_WRITE) {
        while (desc->items_ready.load(std::memory_order_acquire) < count) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }

        std::vector<const void*> cptrs(count);
        for (uint32_t i = 0; i < count; ++i)
            cptrs[i] = data_base + desc->entries[i].data_offset;

        auto results = tier.BatchWrite(keys, cptrs, sizes);
        for (uint32_t i = 0; i < count; ++i)
            desc->entries[i].result = results[i] ? 1 : 0;
    } else {
        std::vector<uintptr_t> dma_ptrs(count);
        for (uint32_t i = 0; i < count; ++i)
            dma_ptrs[i] = reinterpret_cast<uintptr_t>(
                data_base + desc->entries[i].data_offset);

        auto results = tier.BatchReadIntoPtrStreaming(
            keys, dma_ptrs, sizes, &desc->items_done);
        for (uint32_t i = 0; i < count; ++i)
            desc->entries[i].result = results[i] ? 1 : 0;
    }

    slot.result = static_cast<int32_t>(ResultCode::OK);
}

// ---------------------------------------------------------------------------
// Main poll loop — with heartbeat, dead-rank reap, and shutdown logic.
// Batch requests are dispatched to per-rank worker threads so that multiple
// ranks can perform disk I/O concurrently without head-of-line blocking.
// ---------------------------------------------------------------------------
static constexpr int kMaxConcurrentRanks = 64;

static void PollLoop(SpdkSsdTier& tier, ProxyShmRegion& shm,
                     pid_t spawner_pid) {
    auto* hdr = shm.Header();
    uint32_t max_ranks = std::min(hdr->max_ranks,
                                  static_cast<uint32_t>(kMaxConcurrentRanks));

    std::atomic<bool> batch_inflight[kMaxConcurrentRanks] = {};
    std::thread       batch_worker[kMaxConcurrentRanks];

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
                batch_worker[r] = std::thread(
                    [&tier, &slot, data_region, region_size, ch, t, r,
                     &batch_inflight]() {
                        ProcessBatchRequest(tier, slot, data_region,
                                            region_size);
                        slot.state.store(
                            static_cast<uint32_t>(SlotState::COMPLETED),
                            std::memory_order_release);
                        ch->tail.store((t + 1) % kRingSize,
                                       std::memory_order_release);
                        batch_inflight[r].store(false,
                                                std::memory_order_release);
                    });
            } else {
                ProcessSingleRequest(tier, slot, data_region, region_size);
                slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED),
                                 std::memory_order_release);
                ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
            }
        }

        auto now = std::chrono::steady_clock::now();

        // ---- Heartbeat update (~every 500ms) ----
        auto since_hb = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_heartbeat);
        if (since_hb.count() > 500) {
            hdr->proxy_heartbeat_ms.store(NowEpochMs(),
                                          std::memory_order_relaxed);
            last_heartbeat = now;
        }

        // ---- Dead-rank reaping (~every 5 seconds) ----
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

        // ---- Check exit conditions ----
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
    // If auto-spawned, arrange to receive SIGTERM when parent dies.
    // This is a safety net — the poll loop also checks spawner liveness.
    if (spawner_pid > 0) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        // Re-check: if parent already died between fork and prctl,
        // getppid() will have changed (to 1 or a subreaper).
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
    size_t data_per_rank_mb = getenv_size("UMBP_SPDK_PROXY_DATA_MB", 512);
    size_t data_per_rank = data_per_rank_mb * 1024 * 1024;

    if (nvme_pci.empty() && bdev_name.empty()) {
        fprintf(stderr,
                "spdk_proxy: UMBP_SPDK_NVME_PCI or UMBP_SPDK_BDEV required\n");
        return 1;
    }

    // Step 1: Create shared memory
    ProxyShmRegion shm;
    int rc = shm.Create(shm_name, max_ranks, data_per_rank, /*try_hugepage=*/true);
    if (rc != 0) {
        fprintf(stderr, "spdk_proxy: failed to create SHM '%s' rc=%d\n",
                shm_name.c_str(), rc);
        return 1;
    }

    // Record proxy PID and spawner PID in header immediately so that
    // clients can detect our process even during SPDK initialization.
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

    // Step 2: Initialize SPDK via SpdkSsdTier
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
        // Keep SHM alive briefly so clients can read ERROR state
        std::this_thread::sleep_for(std::chrono::seconds(2));
        shm.Detach();
        return 1;
    }

    // Re-register signal handlers — spdk_app_start() overrides them
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Step 3: Publish READY state
    auto [used, total] = tier.Capacity();
    hdr->bdev_size = total;
    hdr->block_size = 4096;
    hdr->capacity_used.store(used, std::memory_order_relaxed);
    hdr->capacity_total.store(total, std::memory_order_relaxed);
    hdr->proxy_heartbeat_ms.store(NowEpochMs(), std::memory_order_relaxed);
    hdr->state.store(static_cast<uint32_t>(ProxyState::READY),
                     std::memory_order_release);

    UMBP_LOG_INFO("spdk_proxy: READY — capacity=%zuMB, waiting for connections...",
                  total / (1024 * 1024));
    fflush(stdout);

    // Step 4: Enter main poll loop (returns when shutdown conditions met)
    PollLoop(tier, shm, spawner_pid);

    // Step 5: Shutdown
    hdr->state.store(static_cast<uint32_t>(ProxyState::SHUTDOWN),
                     std::memory_order_release);

    UMBP_LOG_INFO("spdk_proxy: shutting down");
    shm.Detach();

    return 0;
}
