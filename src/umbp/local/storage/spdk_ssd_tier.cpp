// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// SpdkSsdTier: deep-queue NVMe pipeline via SPDK.
//
// Metadata management follows mooncake-store's OffsetAllocatorStorageBackend:
//   - Sharded map with shared_mutex for concurrent reads
//   - RefCounted allocation handles for safe concurrent access
//   - Auto LRU eviction on allocation failure

#include "umbp/local/storage/spdk_ssd_tier.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "umbp/common/log.h"
#include "umbp/spdk/spdk_env.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
SpdkSsdTier::DmaPool::~DmaPool() {
  if (!bufs || count <= 0) return;
  auto& env = umbp::SpdkEnv::Instance();
  if (env.IsInitialized()) {
    env.DmaPoolFreeBatch(bufs, buf_size, count);
  }
  delete[] bufs;
  bufs = nullptr;
  count = 0;
}

SpdkSsdTier::SharedDmaPool SpdkSsdTier::CreateSharedDmaPool(size_t buf_size, int count) {
  if (count <= 0 || buf_size == 0) return nullptr;

  auto& env = umbp::SpdkEnv::Instance();
  if (!env.IsInitialized()) return nullptr;

  auto pool = std::make_shared<DmaPool>();
  pool->buf_size = buf_size;
  pool->count = count;
  pool->bufs = new void*[count];
  int got = env.DmaPoolAllocBatch(pool->bufs, buf_size, count);
  if (got < count) {
    for (int i = got; i < count; ++i) pool->bufs[i] = nullptr;
    pool->count = got;
  }
  if (pool->count <= 0) {
    delete[] pool->bufs;
    pool->bufs = nullptr;
    return nullptr;
  }
  return pool;
}

SpdkSsdTier::SharedDmaPool SpdkSsdTier::EnsureDmaPool() {
  if (dma_pool_) return dma_pool_;

  size_t ring_buf_size = 2ULL * 1024 * 1024;
  int ring_count = kMaxQueueDepth * std::max(1, num_io_workers_);
  dma_pool_ = CreateSharedDmaPool(ring_buf_size, ring_count);
  return dma_pool_;
}

SpdkSsdTier::SpdkSsdTier(const UMBPConfig& config)
    : SpdkSsdTier(config, 0, config.ssd.capacity_bytes, nullptr) {}

SpdkSsdTier::SpdkSsdTier(const UMBPConfig& config, uint64_t base_offset, size_t capacity_bytes,
                         SharedDmaPool shared_dma_pool)
    : TierBackend(StorageTier::LOCAL_SSD) {
  auto& env = umbp::SpdkEnv::Instance();
  if (!env.IsInitialized()) {
    umbp::SpdkEnvConfig ecfg;
    ecfg.bdev_name = config.spdk_bdev_name;
    ecfg.reactor_mask = config.spdk_reactor_mask;
    ecfg.mem_size_mb = config.spdk_mem_size_mb;
    ecfg.nvme_pci_addr = config.spdk_nvme_pci_addr;
    ecfg.nvme_ctrl_name = config.spdk_nvme_ctrl_name;

    int rc = env.Init(ecfg);
    if (rc != 0) {
      UMBP_LOG_ERROR("SpdkSsdTier: SpdkEnv init failed rc=%d, falling back", rc);
      return;
    }
  }

  block_size_ = env.GetBlockSize();
  if (block_size_ == 0) block_size_ = 4096;

  uint64_t device_size = env.GetBdevSize();
  if (base_offset >= device_size) {
    UMBP_LOG_ERROR("SpdkSsdTier: base offset %lu beyond device size %lu",
                   static_cast<unsigned long>(base_offset),
                   static_cast<unsigned long>(device_size));
    return;
  }

  base_offset_ = base_offset;
  size_t max_capacity = static_cast<size_t>(device_size - base_offset_);
  capacity_ = std::min(capacity_bytes, max_capacity);
  if (capacity_ == 0) {
    UMBP_LOG_ERROR("SpdkSsdTier: capacity is zero after range clamp");
    return;
  }

  allocator_ =
      umbp::offset_allocator::OffsetAllocator::createAligned(base_offset_, capacity_, block_size_);
  if (!allocator_) {
    UMBP_LOG_ERROR("SpdkSsdTier: OffsetAllocator creation failed");
    return;
  }

  num_io_workers_ = std::max(1, config.spdk_io_workers);
  dma_pool_ = std::move(shared_dma_pool);
  EnsureDmaPool();

  initialized_ = true;
  UMBP_LOG_INFO(
      "SpdkSsdTier: ready — base=%zuMB capacity=%zuMB block_size=%u "
      "dma_pool=%d×%zuKB io_workers=%d shards=%zu",
      static_cast<size_t>(base_offset_ / (1024 * 1024)), capacity_ / (1024 * 1024), block_size_,
      dma_pool_ ? dma_pool_->count : 0, dma_pool_ ? dma_pool_->buf_size / 1024 : 0, num_io_workers_,
      kNumShards);
}

SpdkSsdTier::~SpdkSsdTier() { Clear(); }

SpdkSsdTier::Stats SpdkSsdTier::GetStats() const {
  Stats stats;
  stats.hit_count = hit_count_.load(std::memory_order_relaxed);
  stats.miss_count = miss_count_.load(std::memory_order_relaxed);
  stats.evicted_bytes = evicted_bytes_.load(std::memory_order_relaxed);
  return stats;
}

void SpdkSsdTier::RecordHit(uint64_t count) {
  hit_count_.fetch_add(count, std::memory_order_relaxed);
}

void SpdkSsdTier::RecordMiss(uint64_t count) {
  miss_count_.fetch_add(count, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Single-key wrappers
// ---------------------------------------------------------------------------
bool SpdkSsdTier::Write(const std::string& key, const void* data, size_t size) {
  std::vector<std::string> keys = {key};
  std::vector<const void*> ptrs = {data};
  std::vector<size_t> sizes = {size};
  auto results = BatchWrite(keys, ptrs, sizes);
  return !results.empty() && results[0];
}

bool SpdkSsdTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  std::vector<std::string> keys = {key};
  std::vector<uintptr_t> ptrs = {dst_ptr};
  std::vector<size_t> sizes = {size};
  auto results = BatchReadIntoPtr(keys, ptrs, sizes);
  return !results.empty() && results[0];
}

// ---------------------------------------------------------------------------
// Exists / Evict / Capacity / Clear / LRU — sharded
// ---------------------------------------------------------------------------
bool SpdkSsdTier::Exists(const std::string& key) const {
  auto& shard = shards_[ShardForKey(key)];
  std::shared_lock<std::shared_mutex> slk(shard.mutex);
  return shard.map.count(key) > 0;
}

bool SpdkSsdTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lru_lk(lru_mu_);
  auto& shard = shards_[ShardForKey(key)];
  std::unique_lock<std::shared_mutex> slk(shard.mutex);
  auto it = shard.map.find(key);
  if (it == shard.map.end()) return false;
  evicted_bytes_.fetch_add(AlignUp(it->second.data_size), std::memory_order_relaxed);
  lru_list_.erase(it->second.lru_pos);
  shard.map.erase(it);
  return true;
}

std::pair<size_t, size_t> SpdkSsdTier::Capacity() const {
  if (!allocator_) return {0, capacity_};
  auto metrics = allocator_->get_metrics();
  return {metrics.allocated_size_, capacity_};
}

void SpdkSsdTier::Clear() {
  std::lock_guard<std::mutex> lru_lk(lru_mu_);
  for (size_t s = 0; s < kNumShards; ++s) {
    std::unique_lock<std::shared_mutex> slk(shards_[s].mutex);
    shards_[s].map.clear();
  }
  lru_list_.clear();
}

std::string SpdkSsdTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lk(lru_mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::vector<std::string> SpdkSsdTier::GetLRUCandidates(size_t max_candidates) const {
  std::lock_guard<std::mutex> lk(lru_mu_);
  std::vector<std::string> result;
  if (max_candidates == 0) max_candidates = 1;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  for (auto it = lru_list_.rbegin(); it != lru_list_.rend() && result.size() < max_candidates;
       ++it) {
    result.push_back(*it);
  }
  return result;
}

// ===========================================================================
// LRU eviction — evicts entries until `needed` bytes are freed.
// Lock ordering: lru_mu_ → shard.mutex (consistent with all other methods).
// Returns actual bytes freed (may be 0 if LRU is empty).
// ===========================================================================
size_t SpdkSsdTier::EvictLRU(size_t needed) {
  size_t freed = 0;
  int evicted = 0;
  while (freed < needed) {
    std::lock_guard<std::mutex> lru_lk(lru_mu_);
    if (lru_list_.empty()) break;

    std::string key = lru_list_.back();
    auto& shard = shards_[ShardForKey(key)];
    std::unique_lock<std::shared_mutex> slk(shard.mutex);
    auto it = shard.map.find(key);
    if (it != shard.map.end()) {
      size_t entry_size = AlignUp(it->second.data_size);
      bool immediate = (it->second.allocation.use_count() == 1);
      freed += entry_size;
      evicted_bytes_.fetch_add(entry_size, std::memory_order_relaxed);
      shard.map.erase(it);
      if (!immediate) {
        UMBP_LOG_WARN(
            "EvictLRU: key '%s' (%zuKB) has in-flight readers, "
            "space reclaim deferred",
            key.c_str(), entry_size / 1024);
      }
    }
    lru_list_.pop_back();
    ++evicted;
  }
  if (evicted > 0) {
    UMBP_LOG_INFO("EvictLRU: evicted %d entries, freed %zuMB (requested %zuMB)", evicted,
                  freed / (1024 * 1024), needed / (1024 * 1024));
  }
  return freed;
}

// ===========================================================================
// PrepareWriteAlloc — common Phase 1 for all BatchWrite* variants.
//
// 1) Checks existing keys (shard shared lock + LRU update)
// 2) Batch allocates space for new keys
// 3) On allocation failure, evicts LRU entries and retries
// ===========================================================================
std::vector<SpdkSsdTier::PendingWrite> SpdkSsdTier::PrepareWriteAlloc(
    const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
    std::vector<bool>& results) {
  const int count = static_cast<int>(keys.size());
  std::vector<PendingWrite> pending;
  pending.reserve(count);

  std::vector<size_t> alloc_sizes;
  std::vector<int> new_indices;
  alloc_sizes.reserve(count);
  new_indices.reserve(count);

  {
    std::lock_guard<std::mutex> lru_lk(lru_mu_);
    for (int i = 0; i < count; ++i) {
      if (sizes[i] == 0) {
        results[i] = false;
        continue;
      }
      auto& shard = shards_[ShardForKey(keys[i])];
      std::unique_lock<std::shared_mutex> slk(shard.mutex);
      auto it = shard.map.find(keys[i]);
      if (it != shard.map.end()) {
        results[i] = true;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_pos);
        continue;
      }
      new_indices.push_back(i);
      alloc_sizes.push_back(AlignUp(sizes[i]));
    }
  }

  if (alloc_sizes.empty()) return pending;

  // Allocate with auto-eviction retry
  size_t success_count = 0;
  constexpr int kMaxEvictRetries = 3;

  for (int retry = 0; retry <= kMaxEvictRetries; ++retry) {
    std::vector<size_t> remaining(alloc_sizes.begin() + static_cast<ptrdiff_t>(success_count),
                                  alloc_sizes.end());

    auto handles = allocator_->batch_allocate(remaining);

    for (size_t j = 0; j < handles.size(); ++j) {
      if (!handles[j].has_value()) break;
      PendingWrite pw;
      pw.idx = new_indices[success_count + j];
      pw.aligned_size = alloc_sizes[success_count + j];
      pw.allocation = std::make_shared<RefCountedAllocationHandle>(std::move(handles[j].value()));
      pending.push_back(std::move(pw));
    }

    success_count = pending.size();
    if (success_count >= new_indices.size()) break;

    size_t needed = 0;
    for (size_t k = success_count; k < alloc_sizes.size(); ++k) needed += alloc_sizes[k];

    size_t freed = EvictLRU(needed);
    if (freed == 0) break;
  }

  if (success_count < new_indices.size()) {
    size_t failed = new_indices.size() - success_count;
    auto metrics = allocator_->get_metrics();
    UMBP_LOG_WARN(
        "PrepareWriteAlloc: %zu/%zu keys failed to allocate "
        "(free=%zuMB, largest_free=%zuMB)",
        failed, new_indices.size(), metrics.total_free_space_ / (1024 * 1024),
        metrics.largest_free_region_ / (1024 * 1024));
    for (size_t k = success_count; k < new_indices.size(); ++k) results[new_indices[k]] = false;
  }

  return pending;
}

// ===========================================================================
// CommitWriteEntries — common Phase 3 for all BatchWrite* variants.
//
// Inserts successfully-written entries into sharded map + LRU.
// ===========================================================================
void SpdkSsdTier::CommitWriteEntries(const std::vector<std::string>& keys,
                                     const std::vector<size_t>& sizes,
                                     std::vector<PendingWrite>& pending,
                                     const std::vector<bool>& item_ok, std::vector<bool>& results) {
  const int pending_count = static_cast<int>(pending.size());

  std::lock_guard<std::mutex> lru_lk(lru_mu_);
  for (int j = 0; j < pending_count; ++j) {
    if (!item_ok[static_cast<size_t>(j)]) continue;
    int idx = pending[j].idx;

    auto& shard = shards_[ShardForKey(keys[idx])];
    std::unique_lock<std::shared_mutex> slk(shard.mutex);

    Entry entry;
    entry.allocation = std::move(pending[j].allocation);
    entry.data_size = sizes[idx];

    auto [it, inserted] = shard.map.try_emplace(keys[idx], std::move(entry));
    if (inserted) {
      lru_list_.push_front(keys[idx]);
      it->second.lru_pos = lru_list_.begin();
    } else {
      it->second.allocation = std::move(entry.allocation);
      it->second.data_size = entry.data_size;
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_pos);
    }
    results[idx] = true;
  }
}

// ===========================================================================
// PrepareReadLookup — common Phase 1 for all BatchRead* variants.
//
// Looks up entries, copies AllocationPtr (keeps NVMe extent alive during DMA),
// and updates LRU.
// ===========================================================================
std::vector<SpdkSsdTier::ReadInfo> SpdkSsdTier::PrepareReadLookup(
    const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
    std::vector<bool>& results) {
  const int count = static_cast<int>(keys.size());
  std::vector<ReadInfo> items;
  items.reserve(count);

  std::lock_guard<std::mutex> lru_lk(lru_mu_);
  for (int i = 0; i < count; ++i) {
    auto& shard = shards_[ShardForKey(keys[i])];
    std::shared_lock<std::shared_mutex> slk(shard.mutex);
    auto it = shard.map.find(keys[i]);
    if (it == shard.map.end()) continue;

    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_pos);

    size_t rd = std::min(sizes[i], it->second.data_size);
    ReadInfo ri;
    ri.idx = i;
    ri.offset = it->second.allocation->handle.address();
    ri.aligned_size = AlignUp(rd);
    ri.data_size = rd;
    ri.guard = it->second.allocation;
    items.push_back(std::move(ri));
  }

  return items;
}

// ===========================================================================
// BatchWrite — deep-queue NVMe write pipeline
//
// Phase 1: PrepareWriteAlloc (check existing, allocate with auto-eviction)
// Phase 2: memcpy + submit + drain pipeline on calling thread
// Phase 3: CommitWriteEntries (update sharded map + LRU)
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchWrite(const std::vector<std::string>& keys,
                                          const std::vector<const void*>& data_ptrs,
                                          const std::vector<size_t>& sizes) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();
  auto pool = EnsureDmaPool();
  if (!pool || pool->count <= 0) return results;

  auto pending = PrepareWriteAlloc(keys, sizes, results);
  if (pending.empty()) return results;

  // --- Phase 2: Chunked deep-queue I/O pipeline (no lock) ---
  const int pending_count = static_cast<int>(pending.size());
  const size_t chunk_sz = pool->buf_size;

  struct WriteChunk {
    int item_idx;
    uint64_t offset;
    size_t nbytes;
    size_t data_offset;
    size_t data_bytes;
  };
  std::vector<WriteChunk> chunks;
  chunks.reserve(pending_count);
  for (int i = 0; i < pending_count; ++i) {
    auto& p = pending[i];
    size_t rem_aligned = p.aligned_size;
    size_t rem_data = sizes[p.idx];
    size_t src_off = 0;
    uint64_t dev_off = p.allocation->handle.address();
    while (rem_aligned > 0) {
      size_t ca = std::min(rem_aligned, chunk_sz);
      size_t cd = std::min(rem_data, ca);
      chunks.push_back({i, dev_off, ca, src_off, cd});
      rem_aligned -= ca;
      rem_data = (rem_data > cd) ? rem_data - cd : 0;
      src_off += cd;
      dev_off += ca;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    std::lock_guard<std::mutex> dma_lk(pool->mutex);

    constexpr int kMinChunksPerWorker = 16;
    int max_w = std::min(num_io_workers_, pool->count / 2);
    int num_workers = std::clamp(chunk_count / kMinChunksPerWorker, 1, max_w);
    int bufs_per = pool->count / num_workers;

    auto run_pipeline = [&](int c_begin, int c_end, void** bufs, int local_qd) {
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          int slot = (head - c_begin) % local_qd;
          auto& c = chunks[head];
          int idx = pending[c.item_idx].idx;

          const char* src = static_cast<const char*>(data_ptrs[idx]) + c.data_offset;
          std::memcpy(bufs[slot], src, c.data_bytes);
          if (c.nbytes > c.data_bytes)
            std::memset(static_cast<char*>(bufs[slot]) + c.data_bytes, 0, c.nbytes - c.data_bytes);

          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::WRITE;
          req.buf = bufs[slot];
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;

          lbatch[bc++] = &req;
          ++head;

          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
          chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      int qd = std::min({chunk_count, kMaxQueueDepth, pool->count});
      run_pipeline(0, chunk_count, pool->bufs, qd);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        void** wb = pool->bufs + w * bufs_per;
        int wq = (w == num_workers - 1) ? (pool->count - w * bufs_per) : bufs_per;
        if (w < num_workers - 1) {
          workers.emplace_back([&, cb, ce, wb, wq]() { run_pipeline(cb, ce, wb, wq); });
        } else {
          run_pipeline(cb, ce, wb, wq);
        }
      }
      for (auto& t : workers) t.join();
    }
  }

  // --- Phase 3: Update metadata ---
  std::vector<bool> item_ok(pending_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;

  CommitWriteEntries(keys, sizes, pending, item_ok, results);
  return results;
}

// ===========================================================================
// BatchWriteStreaming — byte-level streaming write from shared memory.
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchWriteStreaming(const std::vector<std::string>& keys,
                                                   const std::vector<const void*>& data_ptrs,
                                                   const std::vector<size_t>& sizes,
                                                   std::atomic<uint64_t>* bytes_ready,
                                                   const std::vector<size_t>& item_shm_offsets,
                                                   void** ext_dma_bufs, int ext_dma_count) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();
  auto pool = EnsureDmaPool();

  auto pending = PrepareWriteAlloc(keys, sizes, results);
  if (pending.empty()) return results;

  // --- Phase 2: Chunked pipeline with bytes_ready gating ---
  const int pending_count = static_cast<int>(pending.size());
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);

  struct WriteChunk {
    int item_idx;
    uint64_t offset;
    size_t nbytes;
    size_t data_offset;
    size_t data_bytes;
  };
  std::vector<WriteChunk> chunks;
  chunks.reserve(pending_count);
  for (int i = 0; i < pending_count; ++i) {
    auto& p = pending[i];
    size_t rem_aligned = p.aligned_size;
    size_t rem_data = sizes[p.idx];
    size_t src_off = 0;
    uint64_t dev_off = p.allocation->handle.address();
    while (rem_aligned > 0) {
      size_t ca = std::min(rem_aligned, chunk_sz);
      size_t cd = std::min(rem_data, ca);
      chunks.push_back({i, dev_off, ca, src_off, cd});
      rem_aligned -= ca;
      rem_data = (rem_data > cd) ? rem_data - cd : 0;
      src_off += cd;
      dev_off += ca;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  std::unique_lock<std::mutex> dma_lk;
  void** bufs;
  int total_bufs;
  if (ext_dma_bufs && ext_dma_count > 0) {
    bufs = ext_dma_bufs;
    total_bufs = ext_dma_count;
  } else {
    if (!pool || pool->count <= 0) return results;
    dma_lk = std::unique_lock<std::mutex>(pool->mutex);
    bufs = pool->bufs;
    total_bufs = pool->count;
  }

  {
    constexpr int kMinChunksPerWorker = 16;
    int max_w = std::min(num_io_workers_, total_bufs / 2);
    int num_workers = std::clamp(chunk_count / kMinChunksPerWorker, 1, max_w);
    int bufs_per = total_bufs / num_workers;

    auto run_pipeline = [&](int c_begin, int c_end, void** wbufs, int local_qd) {
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          int slot = (head - c_begin) % local_qd;
          auto& c = chunks[head];
          int idx = pending[c.item_idx].idx;

          if (bytes_ready && c.data_bytes > 0) {
            uint64_t abs_end = item_shm_offsets[idx] + c.data_offset + c.data_bytes;
            while (bytes_ready->load(std::memory_order_acquire) < abs_end) CPU_PAUSE();
          }

          const char* src = static_cast<const char*>(data_ptrs[idx]) + c.data_offset;
          std::memcpy(wbufs[slot], src, c.data_bytes);
          if (c.nbytes > c.data_bytes)
            std::memset(static_cast<char*>(wbufs[slot]) + c.data_bytes, 0, c.nbytes - c.data_bytes);

          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::WRITE;
          req.buf = wbufs[slot];
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;

          lbatch[bc++] = &req;
          ++head;

          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
          chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      int qd = std::min({chunk_count, kMaxQueueDepth, total_bufs});
      run_pipeline(0, chunk_count, bufs, qd);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        void** wb = bufs + w * bufs_per;
        int wq = (w == num_workers - 1) ? (total_bufs - w * bufs_per) : bufs_per;
        if (w < num_workers - 1) {
          workers.emplace_back([&, cb, ce, wb, wq]() { run_pipeline(cb, ce, wb, wq); });
        } else {
          run_pipeline(cb, ce, wb, wq);
        }
      }
      for (auto& t : workers) t.join();
    }
  }

  if (dma_lk.owns_lock()) dma_lk.unlock();

  // --- Phase 3: Update metadata ---
  std::vector<bool> item_ok(pending_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;

  CommitWriteEntries(keys, sizes, pending, item_ok, results);
  return results;
}

// ===========================================================================
// BatchReadIntoPtr — deep-queue NVMe read pipeline
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                                const std::vector<uintptr_t>& dst_ptrs,
                                                const std::vector<size_t>& sizes) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();
  auto pool = EnsureDmaPool();
  if (!pool || pool->count <= 0) return results;

  auto items = PrepareReadLookup(keys, sizes, results);
  if (items.empty()) return results;

  // --- Phase 2: Chunked deep-queue I/O pipeline (no lock) ---
  const int item_count = static_cast<int>(items.size());
  const size_t chunk_sz = pool->buf_size;

  struct ReadChunk {
    int item_idx;
    uint64_t offset;
    size_t nbytes;
    size_t data_offset;
    size_t data_bytes;
  };
  std::vector<ReadChunk> chunks;
  chunks.reserve(item_count);
  for (int i = 0; i < item_count; ++i) {
    auto& ri = items[i];
    size_t rem_aligned = ri.aligned_size;
    size_t rem_data = ri.data_size;
    size_t dst_off = 0;
    uint64_t dev_off = ri.offset;
    while (rem_aligned > 0) {
      size_t ca = std::min(rem_aligned, chunk_sz);
      size_t cd = std::min(rem_data, ca);
      chunks.push_back({i, dev_off, ca, dst_off, cd});
      rem_aligned -= ca;
      rem_data = (rem_data > cd) ? rem_data - cd : 0;
      dst_off += cd;
      dev_off += ca;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    std::lock_guard<std::mutex> dma_lk(pool->mutex);

    constexpr int kMinChunksPerWorker = 16;
    int max_w = std::min(num_io_workers_, pool->count / 2);
    int num_workers = std::clamp(chunk_count / kMinChunksPerWorker, 1, max_w);
    int bufs_per = pool->count / num_workers;

    auto run_pipeline = [&](int c_begin, int c_end, void** bufs, int local_qd) {
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          int slot = (head - c_begin) % local_qd;
          auto& c = chunks[head];

          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::READ;
          req.buf = bufs[slot];
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;

          lbatch[bc++] = &req;
          ++head;

          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;

          if (lreqs[slot].success) {
            auto& c = chunks[tail];
            auto& ri = items[c.item_idx];
            char* dst = reinterpret_cast<char*>(dst_ptrs[ri.idx]) + c.data_offset;
            std::memcpy(dst, bufs[slot], c.data_bytes);
            chunk_ok[tail] = 1;
          }
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      int qd = std::min({chunk_count, kMaxQueueDepth, pool->count});
      run_pipeline(0, chunk_count, pool->bufs, qd);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        void** wb = pool->bufs + w * bufs_per;
        int wq = (w == num_workers - 1) ? (pool->count - w * bufs_per) : bufs_per;
        if (w < num_workers - 1) {
          workers.emplace_back([&, cb, ce, wb, wq]() { run_pipeline(cb, ce, wb, wq); });
        } else {
          run_pipeline(cb, ce, wb, wq);
        }
      }
      for (auto& t : workers) t.join();
    }
  }

  // Phase 3: mark results
  std::vector<bool> item_ok(item_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;
  for (int j = 0; j < item_count; ++j)
    if (item_ok[j]) results[items[j].idx] = true;

  return results;
}

// ===========================================================================
// BatchReadIntoPtrStreaming — DMA-ring read with per-item progress signaling
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchReadIntoPtrStreaming(
    const std::vector<std::string>& keys, const std::vector<uintptr_t>& dst_ptrs,
    const std::vector<size_t>& sizes, std::atomic<uint32_t>* items_done,
    std::atomic<uint64_t>* bytes_done, const std::vector<size_t>* item_shm_offsets,
    void** ext_dma_bufs, int ext_dma_count) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();
  auto pool = EnsureDmaPool();

  auto items = PrepareReadLookup(keys, sizes, results);

  if (items.empty()) {
    if (items_done) items_done->store(count, std::memory_order_release);
    return results;
  }

  const int item_count = static_cast<int>(items.size());
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);

  struct ReadChunk {
    int item_idx;
    uint64_t offset;
    size_t nbytes;
    size_t data_offset;
    size_t data_bytes;
  };
  std::vector<ReadChunk> chunks;
  chunks.reserve(item_count);
  for (int i = 0; i < item_count; ++i) {
    auto& ri = items[i];
    size_t rem_a = ri.aligned_size, rem_d = ri.data_size, dst_off = 0;
    uint64_t dev_off = ri.offset;
    while (rem_a > 0) {
      size_t ca = std::min(rem_a, chunk_sz);
      size_t cd = std::min(rem_d, ca);
      chunks.push_back({i, dev_off, ca, dst_off, cd});
      rem_a -= ca;
      rem_d = (rem_d > cd) ? rem_d - cd : 0;
      dst_off += cd;
      dev_off += ca;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  std::unique_lock<std::mutex> dma_lk;
  void** bufs;
  int total_bufs;
  if (ext_dma_bufs && ext_dma_count > 0) {
    bufs = ext_dma_bufs;
    total_bufs = ext_dma_count;
  } else {
    if (!pool || pool->count <= 0) return results;
    dma_lk = std::unique_lock<std::mutex>(pool->mutex);
    bufs = pool->bufs;
    total_bufs = pool->count;
  }

  {
    int qd = std::min({chunk_count, kMaxQueueDepth, total_bufs});
    auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(qd);
    auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(qd);
    int head = 0, tail = 0;
    int items_signaled = 0;

    while (tail < chunk_count) {
      int bc = 0;
      while (head < chunk_count && (head - tail) < qd) {
        int slot = head % qd;
        auto& c = chunks[head];
        auto& req = lreqs[slot];
        req.op = umbp::SpdkIoRequest::READ;
        req.buf = bufs[slot];
        req.offset = c.offset;
        req.nbytes = c.nbytes;
        req.src_data = nullptr;
        req.src_iov = nullptr;
        req.src_iovcnt = 0;
        req.dst_iov = nullptr;
        req.dst_iovcnt = 0;
        req.completed.store(false, std::memory_order_release);
        req.success = false;
        lbatch[bc++] = &req;
        ++head;
        if (bc >= 8) {
          env.SubmitIoBatchAsync(lbatch.get(), bc);
          bc = 0;
        }
      }
      if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

      while (tail < head) {
        int slot = tail % qd;
        if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
        if (lreqs[slot].success) {
          auto& c = chunks[tail];
          auto& ri = items[c.item_idx];
          char* dst = reinterpret_cast<char*>(dst_ptrs[ri.idx]) + c.data_offset;
          std::memcpy(dst, bufs[slot], c.data_bytes);
          chunk_ok[tail] = 1;

          if (bytes_done && item_shm_offsets) {
            uint64_t abs = (*item_shm_offsets)[ri.idx] + c.data_offset + c.data_bytes;
            bytes_done->store(abs, std::memory_order_release);
          }
        }
        ++tail;

        int cur_item = chunks[tail - 1].item_idx;
        bool item_done = (tail >= chunk_count) || (chunks[tail].item_idx != cur_item);
        if (item_done && cur_item >= items_signaled) {
          items_signaled = cur_item + 1;
          if (items_done)
            items_done->store(static_cast<uint32_t>(items_signaled), std::memory_order_release);
        }
      }
    }
  }

  if (items_done) items_done->store(count, std::memory_order_release);

  std::vector<bool> item_ok(item_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;
  for (int j = 0; j < item_count; ++j)
    if (item_ok[j]) results[items[j].idx] = true;

  return results;
}

// ===========================================================================
// BatchWriteDmaDirect — zero-copy write pipeline
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchWriteDmaDirect(const std::vector<std::string>& keys,
                                                   const std::vector<void*>& dma_ptrs,
                                                   const std::vector<size_t>& sizes) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();

  auto pending = PrepareWriteAlloc(keys, sizes, results);
  if (pending.empty()) return results;

  // --- Phase 2: Zero-copy I/O pipeline ---
  const int pending_count = static_cast<int>(pending.size());

  for (auto& p : pending) {
    int idx = p.idx;
    if (p.aligned_size > sizes[idx]) {
      char* buf = static_cast<char*>(dma_ptrs[idx]);
      std::memset(buf + sizes[idx], 0, p.aligned_size - sizes[idx]);
    }
  }

  auto pool = EnsureDmaPool();
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);
  struct DmaChunk {
    int item_idx;
    void* buf;
    uint64_t offset;
    size_t nbytes;
  };
  std::vector<DmaChunk> chunks;
  chunks.reserve(pending_count);

  for (int i = 0; i < pending_count; ++i) {
    auto& p = pending[i];
    char* base = static_cast<char*>(dma_ptrs[p.idx]);
    size_t rem = p.aligned_size;
    size_t off = 0;
    uint64_t dev_off = p.allocation->handle.address();
    while (rem > 0) {
      size_t cs = std::min(rem, chunk_sz);
      chunks.push_back({i, base + off, dev_off, cs});
      rem -= cs;
      off += cs;
      dev_off += cs;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    constexpr int kMinChunksPerWorker = 16;
    int num_workers =
        std::clamp(chunk_count / kMinChunksPerWorker, 1, std::max(1, num_io_workers_));

    auto run_pipeline = [&](int c_begin, int c_end) {
      int local_qd = std::min(kMaxQueueDepth, c_end - c_begin);
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          int slot = (head - c_begin) % local_qd;
          auto& c = chunks[head];
          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::WRITE;
          req.buf = c.buf;
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;
          lbatch[bc++] = &req;
          ++head;
          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
          chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      run_pipeline(0, chunk_count);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        if (w < num_workers - 1) {
          workers.emplace_back([&, cb, ce]() { run_pipeline(cb, ce); });
        } else {
          run_pipeline(cb, ce);
        }
      }
      for (auto& t : workers) t.join();
    }
  }

  // --- Phase 3: Update metadata ---
  std::vector<bool> item_ok(pending_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;

  CommitWriteEntries(keys, sizes, pending, item_ok, results);
  return results;
}

// ===========================================================================
// BatchReadDmaDirect — zero-copy read pipeline
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchReadDmaDirect(const std::vector<std::string>& keys,
                                                  const std::vector<uintptr_t>& dma_ptrs,
                                                  const std::vector<size_t>& sizes) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();

  auto items = PrepareReadLookup(keys, sizes, results);
  if (items.empty()) return results;

  // --- Phase 2: Zero-copy read pipeline ---
  const int item_count = static_cast<int>(items.size());
  auto pool = EnsureDmaPool();
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);

  struct DmaReadChunk {
    int item_idx;
    void* buf;
    uint64_t offset;
    size_t nbytes;
  };
  std::vector<DmaReadChunk> chunks;
  chunks.reserve(item_count);

  for (int i = 0; i < item_count; ++i) {
    auto& ri = items[i];
    char* base = reinterpret_cast<char*>(dma_ptrs[ri.idx]);
    size_t rem = ri.aligned_size;
    size_t off = 0;
    uint64_t dev_off = ri.offset;
    while (rem > 0) {
      size_t cs = std::min(rem, chunk_sz);
      chunks.push_back({i, base + off, dev_off, cs});
      rem -= cs;
      off += cs;
      dev_off += cs;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    constexpr int kMinChunksPerWorker = 16;
    int num_workers =
        std::clamp(chunk_count / kMinChunksPerWorker, 1, std::max(1, num_io_workers_));

    auto run_pipeline = [&](int c_begin, int c_end) {
      int local_qd = std::min(kMaxQueueDepth, c_end - c_begin);
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          int slot = (head - c_begin) % local_qd;
          auto& c = chunks[head];
          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::READ;
          req.buf = c.buf;
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;
          lbatch[bc++] = &req;
          ++head;
          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
          chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      run_pipeline(0, chunk_count);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        if (w < num_workers - 1) {
          workers.emplace_back([&, cb, ce]() { run_pipeline(cb, ce); });
        } else {
          run_pipeline(cb, ce);
        }
      }
      for (auto& t : workers) t.join();
    }
  }

  // Phase 3: mark results
  std::vector<bool> item_ok(item_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;
  for (int j = 0; j < item_count; ++j)
    if (item_ok[j]) results[items[j].idx] = true;

  return results;
}

// ===========================================================================
// BatchWriteDmaStreaming — zero-copy + pipelined memcpy
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchWriteDmaStreaming(const std::vector<std::string>& keys,
                                                      const std::vector<void*>& dma_ptrs,
                                                      const std::vector<size_t>& sizes,
                                                      std::atomic<uint32_t>* items_ready) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();

  auto pending = PrepareWriteAlloc(keys, sizes, results);
  if (pending.empty()) return results;

  const int pending_count = static_cast<int>(pending.size());
  auto pool = EnsureDmaPool();
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);
  struct DmaChunk {
    int item_idx;
    void* buf;
    uint64_t offset;
    size_t nbytes;
  };
  std::vector<DmaChunk> chunks;
  chunks.reserve(pending_count);

  for (int i = 0; i < pending_count; ++i) {
    auto& p = pending[i];
    char* base = static_cast<char*>(dma_ptrs[p.idx]);
    size_t rem = p.aligned_size;
    size_t off = 0;
    uint64_t dev_off = p.allocation->handle.address();
    while (rem > 0) {
      size_t cs = std::min(rem, chunk_sz);
      chunks.push_back({i, base + off, dev_off, cs});
      rem -= cs;
      off += cs;
      dev_off += cs;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    constexpr int kMinChunksPerWorker = 16;
    int num_workers =
        std::clamp(chunk_count / kMinChunksPerWorker, 1, std::max(1, num_io_workers_));

    auto run_streaming = [&](int c_begin, int c_end) {
      int local_qd = std::min(kMaxQueueDepth, c_end - c_begin);
      auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
      auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
      int head = c_begin, tail = c_begin;
      int last_ready_item = -1;

      while (tail < c_end) {
        int bc = 0;
        while (head < c_end && (head - tail) < local_qd) {
          auto& c = chunks[head];
          int key_idx = pending[c.item_idx].idx;

          if (c.item_idx != last_ready_item) {
            while (items_ready->load(std::memory_order_acquire) <= static_cast<uint32_t>(key_idx)) {
              CPU_PAUSE();
            }
            auto& p = pending[c.item_idx];
            if (p.aligned_size > sizes[key_idx]) {
              char* buf = static_cast<char*>(dma_ptrs[key_idx]);
              std::memset(buf + sizes[key_idx], 0, p.aligned_size - sizes[key_idx]);
            }
            last_ready_item = c.item_idx;
          }

          int slot = (head - c_begin) % local_qd;
          auto& req = lreqs[slot];
          req.op = umbp::SpdkIoRequest::WRITE;
          req.buf = c.buf;
          req.offset = c.offset;
          req.nbytes = c.nbytes;
          req.src_data = nullptr;
          req.src_iov = nullptr;
          req.src_iovcnt = 0;
          req.dst_iov = nullptr;
          req.dst_iovcnt = 0;
          req.completed.store(false, std::memory_order_release);
          req.success = false;
          lbatch[bc++] = &req;
          ++head;
          if (bc >= 8) {
            env.SubmitIoBatchAsync(lbatch.get(), bc);
            bc = 0;
          }
        }
        if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

        while (tail < head) {
          int slot = (tail - c_begin) % local_qd;
          if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
          chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
          ++tail;
        }
      }
    };

    if (num_workers <= 1) {
      run_streaming(0, chunk_count);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers - 1);
      for (int w = 0; w < num_workers; ++w) {
        int cb = chunk_count * w / num_workers;
        int ce = chunk_count * (w + 1) / num_workers;
        if (w < num_workers - 1)
          workers.emplace_back([&, cb, ce]() { run_streaming(cb, ce); });
        else
          run_streaming(cb, ce);
      }
      for (auto& t : workers) t.join();
    }
  }

  // --- Phase 3: Update metadata ---
  std::vector<bool> item_ok(pending_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;

  CommitWriteEntries(keys, sizes, pending, item_ok, results);
  return results;
}

// ===========================================================================
// BatchReadDmaStreaming — zero-copy + streaming completion notification
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchReadDmaStreaming(const std::vector<std::string>& keys,
                                                     const std::vector<uintptr_t>& dma_ptrs,
                                                     const std::vector<size_t>& sizes,
                                                     std::atomic<uint32_t>* items_done) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!initialized_ || count == 0) return results;

  auto& env = umbp::SpdkEnv::Instance();

  auto items = PrepareReadLookup(keys, sizes, results);
  if (items.empty()) return results;

  const int item_count = static_cast<int>(items.size());
  auto pool = EnsureDmaPool();
  const size_t chunk_sz = pool ? pool->buf_size : (2ULL * 1024 * 1024);
  struct DmaRC {
    int item_idx;
    void* buf;
    uint64_t offset;
    size_t nbytes;
  };
  std::vector<DmaRC> chunks;
  chunks.reserve(item_count);

  for (int i = 0; i < item_count; ++i) {
    auto& ri = items[i];
    char* base = reinterpret_cast<char*>(dma_ptrs[ri.idx]);
    size_t rem = ri.aligned_size;
    size_t off = 0;
    uint64_t dev_off = ri.offset;
    while (rem > 0) {
      size_t cs = std::min(rem, chunk_sz);
      chunks.push_back({i, base + off, dev_off, cs});
      rem -= cs;
      off += cs;
      dev_off += cs;
    }
  }

  const int chunk_count = static_cast<int>(chunks.size());
  auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
  std::memset(chunk_ok.get(), 0, chunk_count);

  {
    int local_qd = std::min(kMaxQueueDepth, chunk_count);
    auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
    auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
    int head = 0, tail = 0;
    int items_signaled = 0;

    while (tail < chunk_count) {
      int bc = 0;
      while (head < chunk_count && (head - tail) < local_qd) {
        int slot = head % local_qd;
        auto& c = chunks[head];
        auto& req = lreqs[slot];
        req.op = umbp::SpdkIoRequest::READ;
        req.buf = c.buf;
        req.offset = c.offset;
        req.nbytes = c.nbytes;
        req.src_data = nullptr;
        req.src_iov = nullptr;
        req.src_iovcnt = 0;
        req.dst_iov = nullptr;
        req.dst_iovcnt = 0;
        req.completed.store(false, std::memory_order_release);
        req.success = false;
        lbatch[bc++] = &req;
        ++head;
        if (bc >= 8) {
          env.SubmitIoBatchAsync(lbatch.get(), bc);
          bc = 0;
        }
      }
      if (bc > 0) env.SubmitIoBatchAsync(lbatch.get(), bc);

      while (tail < head) {
        int slot = tail % local_qd;
        if (!lreqs[slot].completed.load(std::memory_order_acquire)) break;
        chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
        ++tail;

        int cur = chunks[tail - 1].item_idx;
        bool done = (tail >= chunk_count) || (chunks[tail].item_idx != cur);
        if (done && cur >= items_signaled) {
          items_signaled = cur + 1;
          items_done->store(static_cast<uint32_t>(items_signaled), std::memory_order_release);
        }
      }
    }
  }

  std::vector<bool> item_ok(item_count, true);
  for (int j = 0; j < chunk_count; ++j)
    if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;
  for (int j = 0; j < item_count; ++j)
    if (item_ok[j]) results[items[j].idx] = true;
  return results;
}
