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

#include <sys/mman.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/config.h"
#include "umbp/distributed/config.h"

namespace mori::umbp {

DistributedClient::DistributedClient(const UMBPConfig& config) : config_(config) {
  if (!config.distributed.has_value()) {
    throw std::runtime_error("DistributedClient requires UMBPConfig::distributed to be set");
  }

  // Apply runtime env override on top of the programmatically-set
  // max_mr_chunk_size before lowering to PoolClientConfig.  We mutate the
  // local copy `dc_effective` rather than the caller's config to keep the
  // original UMBPConfig observably const.  The split itself happens later
  // inside PoolClient::Init(), where the IOEngine is available to query
  // GetMaxMemoryRegionSize().  Env var format: bytes (decimal).  0 / unset
  // = fall through to the device-reported cap.
  UMBPDistributedConfig dc_effective = config.distributed.value();
  if (const char* env_chunk = std::getenv("UMBP_MAX_MR_CHUNK_SIZE")) {
    try {
      dc_effective.max_mr_chunk_size = static_cast<size_t>(std::stoull(env_chunk));
    } catch (const std::exception& e) {
      MORI_UMBP_WARN(
          "[DistributedClient] UMBP_MAX_MR_CHUNK_SIZE='{}' is not a valid byte count: {} "
          "(falling back to config / device cap)",
          env_chunk, e.what());
    }
  }
  const auto& dc = dc_effective;

  dram_pool_size_ = config.dram.capacity_bytes;
  dram_pool_ =
      mmap(nullptr, dram_pool_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (dram_pool_ == MAP_FAILED) {
    dram_pool_ = nullptr;
    throw std::runtime_error("DistributedClient: mmap failed for DRAM pool");
  }

  // Hand the full pool to PoolClient as a SINGLE logical exportable
  // buffer.  PoolClient::Init() will (post-IOEngine-creation) split this
  // entry into ceil(dram_pool_size_ / effective_chunk) sub-buffers and
  // register each as its own MemoryDesc — that's what spreads pin pressure
  // across NICs and stays under the per-context CPU pin cap on ionic.
  auto pc_config = ToPoolClientConfig(
      dc,
      /*dram_buffers=*/{{dram_pool_, dram_pool_size_}},
      /*tier_capacities=*/{{TierType::DRAM, {dram_pool_size_, dram_pool_size_}}});

  pool_client_ = std::make_unique<PoolClient>(std::move(pc_config));
  if (!pool_client_->Init()) {
    pool_client_.reset();
    munmap(dram_pool_, dram_pool_size_);
    dram_pool_ = nullptr;
    throw std::runtime_error("DistributedClient: PoolClient::Init() failed");
  }

  MORI_UMBP_INFO("[DistributedClient] initialized — node_id={} dram_pool={}MB",
                 dc.master_config.node_id, dram_pool_size_ / (1024 * 1024));
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
                                                       const std::vector<int>& depths) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);
  std::vector<const void*> src_ptrs(srcs.size());
  for (size_t i = 0; i < srcs.size(); ++i) {
    src_ptrs[i] = reinterpret_cast<const void*>(srcs[i]);
  }
  return pool_client_->BatchPut(keys, src_ptrs, sizes, depths);
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

bool DistributedClient::RegisterMemory(void* ptr, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  return pool_client_->RegisterMemory(ptr, size);
}

void DistributedClient::DeregisterMemory(void* ptr) {
  if (closing_) return;
  std::shared_lock lk(op_mutex_);
  if (closed_) return;
  pool_client_->DeregisterMemory(ptr);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void DistributedClient::Clear() { MORI_UMBP_DEBUG("[DistributedClient] Clear() — no-op"); }

bool DistributedClient::Flush() {
  MORI_UMBP_DEBUG("[DistributedClient] Flush() — no-op");
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

  if (dram_pool_) {
    munmap(dram_pool_, dram_pool_size_);
    dram_pool_ = nullptr;
  }

  MORI_UMBP_INFO("[DistributedClient] closed");
}

bool DistributedClient::IsDistributed() const { return true; }

}  // namespace mori::umbp
