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
#include "umbp/distributed/distributed_client.h"

#include <map>
#include <stdexcept>

#include "mori/io/engine.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/config.h"
#include "umbp/distributed/config.h"

namespace mori::umbp {

DistributedClient::DistributedClient(const UMBPConfig& config) : config_(config) {
  if (!config.distributed.has_value()) {
    throw std::runtime_error("DistributedClient requires UMBPConfig::distributed to be set");
  }

  const auto& dc = config.distributed.value();

  // Ownership (ptr, hugepages, NUMA, prefault) moved into PoolClient's DRAM
  // PageBackend — it self-allocates at Init() instead of DistributedClient
  // calling HostMemAllocator and handing over a buffer pointer
  // (backend-agnostic refactor Phase 2b, design doc §1 item 4).  Only the
  // sizing/policy knobs cross this boundary now.
  DramOwnershipConfig dram_ownership;
  dram_ownership.buffer_sizes = {config.dram.capacity_bytes};
  dram_ownership.use_hugepages = config.dram.use_hugepages;
  dram_ownership.hugepage_size = config.dram.hugepage_size;
  dram_ownership.numa_node = config.dram.numa_node;
  dram_ownership.prefault = config.dram.prefault;

  // SSD is unwired from the distributed data plane (backend-agnostic refactor
  // Phase 0, see design-backend-agnostic-refactor.md): PoolClient no longer
  // builds a PeerSsdManager or serves SSD reads, so this node advertises only
  // its DRAM capacity — advertising SSD here without anything behind it would
  // route Gets/Puts into a dead path.
  //
  // HBM is different: dc.hbm carries everything PoolClient's HBM PageBackend
  // needs (enabled/device/capacity), and ToPoolClientConfig lowers it
  // directly from dc — no ownership struct crosses this boundary the way
  // dram's hugepages/NUMA/prefault knobs do, because hipMalloc has none of
  // those dimensions.
  auto pc_config = ToPoolClientConfig(dc, std::move(dram_ownership));
  pc_config.copy_pipeline = config_.copy_pipeline;

  pool_client_ = std::make_unique<PoolClient>(std::move(pc_config));
  if (!pool_client_->Init()) {
    pool_client_.reset();
    throw std::runtime_error("DistributedClient: PoolClient::Init() failed");
  }

  std::string tags_str;
  for (const auto& t : dc.master_config.tags) {
    if (!tags_str.empty()) tags_str += ',';
    tags_str += t;
  }

  MORI_UMBP_INFO(
      "[DistributedClient] initialized — "
      "node_id={} node_address={} master={} "
      "dram_pool={}MB hugepages={} hugepage_size={}MB numa_node={} "
      "hbm_pool={}MB hbm_device={} "
      "dram_page_size={}KB staging_buffer={}MB peer_port={} cache_remote={} "
      "io_engine={}:{} tags=[{}]",
      dc.master_config.node_id, dc.master_config.node_address, dc.master_config.master_address,
      config.dram.capacity_bytes / (1024 * 1024), config_.dram.use_hugepages,
      config_.dram.hugepage_size / (1024 * 1024), config_.dram.numa_node,
      dc.hbm.enabled ? dc.hbm.capacity_bytes / (1024 * 1024) : 0,
      dc.hbm.enabled ? dc.hbm.device : -1, dc.dram_page_size / 1024,
      dc.staging_buffer_size / (1024 * 1024), dc.peer_service_port, dc.cache_remote_fetches,
      dc.io_engine.host, dc.io_engine.port, tags_str);
}

DistributedClient::~DistributedClient() { Close(); }

// ---------------------------------------------------------------------------
// Core KV Operations
// ---------------------------------------------------------------------------

bool DistributedClient::Put(const std::string& key, uintptr_t src, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  return pool_client_->Put(key, reinterpret_cast<const void*>(src), size);
}

bool DistributedClient::Get(const std::string& key, uintptr_t dst, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  return pool_client_->Get(key, reinterpret_cast<void*>(dst), size);
}

bool DistributedClient::Exists(const std::string& key) const {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  return pool_client_->Exists(key);
}

// ---------------------------------------------------------------------------
// Batch Operations
// ---------------------------------------------------------------------------

std::vector<bool> DistributedClient::BatchPut(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& srcs,
                                              const std::vector<size_t>& sizes) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);

  std::vector<const void*> src_ptrs(srcs.size());
  for (size_t i = 0; i < srcs.size(); ++i) {
    src_ptrs[i] = reinterpret_cast<const void*>(srcs[i]);
  }
  return pool_client_->BatchPut(keys, src_ptrs, sizes);
}

std::vector<bool> DistributedClient::BatchPutWithDepth(const std::vector<std::string>& keys,
                                                       const std::vector<uintptr_t>& srcs,
                                                       const std::vector<size_t>& sizes,
                                                       const std::vector<int>& /*depths*/) {
  // Depth was a master-side hint for the prior allocator; in the
  // master-as-advisor design master no longer tracks per-key depth.
  // Forward to the depth-less BatchPut and silently drop the hint.
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);
  std::vector<const void*> src_ptrs(srcs.size());
  for (size_t i = 0; i < srcs.size(); ++i) {
    src_ptrs[i] = reinterpret_cast<const void*>(srcs[i]);
  }
  return pool_client_->BatchPut(keys, src_ptrs, sizes);
}

std::vector<bool> DistributedClient::BatchGet(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& dsts,
                                              const std::vector<size_t>& sizes) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);

  std::vector<void*> dst_ptrs(dsts.size());
  for (size_t i = 0; i < dsts.size(); ++i) {
    dst_ptrs[i] = reinterpret_cast<void*>(dsts[i]);
  }
  return pool_client_->BatchGet(keys, dst_ptrs, sizes);
}

std::vector<bool> DistributedClient::BatchExists(const std::vector<std::string>& keys) const {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);

  // Single batched gRPC instead of N per-key Lookup RPCs (was the #5
  // bottleneck — sglang probes with batch_size=128 used to emit 128
  // roundtrips per BatchExists call).
  return pool_client_->BatchExists(keys);
}

size_t DistributedClient::BatchExistsConsecutive(const std::vector<std::string>& keys) const {
  if (closing_) return 0;
  std::shared_lock lk(op_mutex_);
  if (closed_) return 0;

  // One batched gRPC, then scan the parallel result vector for the first
  // missing key.  A wire failure or size mismatch surfaces as an all-false
  // vector from BatchExists and we return 0 (same failure posture as
  // the old loop-over-Exists path).
  auto found = pool_client_->BatchExists(keys);
  for (size_t i = 0; i < found.size(); ++i) {
    if (!found[i]) return i;
  }
  return keys.size();
}

// ---------------------------------------------------------------------------
// RegisterMemory / DeregisterMemory
// ---------------------------------------------------------------------------

bool DistributedClient::RegisterMemory(uintptr_t ptr, size_t size, mori::io::MemoryLocationType loc,
                                       int device) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  return pool_client_->RegisterMemory(reinterpret_cast<void*>(ptr), size, loc, device);
}

void DistributedClient::DeregisterMemory(uintptr_t ptr) {
  if (closing_) return;
  std::shared_lock lk(op_mutex_);
  if (closed_) return;
  pool_client_->DeregisterMemory(reinterpret_cast<void*>(ptr));
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool DistributedClient::Clear() {
  // Vacuously done during shutdown / teardown: there is no live client to
  // converge with master, so callers in close paths should not see a
  // spurious failure.
  if (closing_) return true;
  // Exclusive lock: Clear races with every Put/Get/Batch* (which take
  // shared_lock) and with Close (which takes unique_lock).  Holding it
  // here keeps local in-flight public API calls out of the clear
  // window — remote in-flight RDMA reads are not in scope (best
  // effort; see distributed-clear-full-sync-plan-zh.md).
  std::unique_lock lk(op_mutex_);
  if (closed_ || !pool_client_) return true;
  const bool ok = pool_client_->Clear();
  if (ok) {
    MORI_UMBP_INFO("[DistributedClient] Clear() completed full-sync empty snapshot");
  } else {
    MORI_UMBP_WARN("[DistributedClient] Clear() full-sync empty snapshot failed");
  }
  return ok;
}

bool DistributedClient::Flush() {
  if (closing_) return true;
  std::shared_lock lk(op_mutex_);
  if (closed_ || !pool_client_) return true;
  pool_client_->Master().FlushHeartbeat();
  return true;
}

void DistributedClient::Close() {
  closing_ = true;
  std::unique_lock lk(op_mutex_);
  if (closed_) return;
  closed_ = true;

  if (pool_client_) {
    pool_client_->Shutdown();
    pool_client_.reset();
  }

  MORI_UMBP_INFO("[DistributedClient] closed");
}

bool DistributedClient::IsDistributed() const { return true; }

bool DistributedClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes,
                                               TierType tier) {
  if (!pool_client_) return false;
  return pool_client_->ReportExternalKvBlocks(hashes, tier);
}

bool DistributedClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes,
                                               TierType tier) {
  if (!pool_client_) return false;
  return pool_client_->RevokeExternalKvBlocks(hashes, tier);
}

bool DistributedClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  if (!pool_client_) return false;
  return pool_client_->RevokeAllExternalKvBlocksAtTier(tier);
}

std::vector<IUMBPClient::ExternalKvMatch> DistributedClient::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit) {
  if (!pool_client_) return {};

  std::vector<MasterClient::ExternalKvNodeMatch> raw;
  if (!pool_client_->MatchExternalKv(hashes, &raw, count_as_hit)) return {};

  std::vector<IUMBPClient::ExternalKvMatch> result;
  result.reserve(raw.size());
  for (auto& r : raw) {
    IUMBPClient::ExternalKvMatch m;
    m.node_id = std::move(r.node_id);
    m.peer_address = std::move(r.peer_address);
    m.hashes_by_tier = std::move(r.hashes_by_tier);
    result.push_back(std::move(m));
  }
  return result;
}

std::vector<IUMBPClient::ExternalKvHitCountEntry> DistributedClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes) {
  if (!pool_client_) return {};

  std::vector<MasterClient::ExternalKvHitCountEntry> raw;
  if (!pool_client_->GetExternalKvHitCounts(hashes, &raw)) return {};

  std::vector<IUMBPClient::ExternalKvHitCountEntry> result;
  result.reserve(raw.size());
  for (auto& r : raw) {
    IUMBPClient::ExternalKvHitCountEntry entry;
    entry.hash = std::move(r.hash);
    entry.hit_count_total = r.hit_count_total;
    result.push_back(std::move(entry));
  }
  return result;
}

}  // namespace mori::umbp
