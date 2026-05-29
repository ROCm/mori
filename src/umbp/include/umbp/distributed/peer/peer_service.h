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
#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mori::umbp {

class LocalStorageManager;
class LocalBlockIndex;
class PeerDramAllocator;
class MasterClient;
class SsdCopyPipeline;

struct StagingMetrics {
  std::atomic<uint64_t> expired_reclaims{0};
  std::atomic<uint64_t> invalid_lease_rejects{0};
  std::atomic<uint64_t> slot_full_rejects{0};
};

class PeerServiceServer {
 public:
  // `dram_alloc` is non-owning and may be null when the host process has
  // no DRAM/HBM tier (SSD-only deployments).  When null, the
  // AllocateSlot/CommitSlot/AbortSlot/ResolveKey/EvictKey handlers
  // respond with success=false / found=false; the SSD read-staging RPCs
  // continue to work unchanged.  The allocator's outbox is where owned-tier
  // ADD/REMOVE events are queued for heartbeat shipment.
  // (Direct-SSD-put RPCs were removed in the SSD-tier redesign; only the SSD
  // read-staging RPCs remain, pending Phase 3 refactor.)
  PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                    const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                    LocalStorageManager& storage, LocalBlockIndex& index,
                    PeerDramAllocator* dram_alloc = nullptr, int num_read_slots = 8,
                    int lease_timeout_s = 10, std::vector<uint8_t> engine_desc_bytes = {},
                    MasterClient* master_client = nullptr);
  // `copy_pipeline` (non-owning, may be null when SSD is disabled) receives an
  // SsdCopyTask after each successful DRAM commit so the owner peer copies the
  // committed bytes to its local SSD tier asynchronously.
  PeerServiceServer(PeerDramAllocator* dram_alloc, int num_read_slots = 8, int lease_timeout_s = 10,
                    std::vector<uint8_t> engine_desc_bytes = {},
                    MasterClient* master_client = nullptr,
                    SsdCopyPipeline* copy_pipeline = nullptr);
  ~PeerServiceServer();

  bool Start(uint16_t port);
  void Stop();

  const StagingMetrics& Metrics() const { return metrics_; }

  // Read-only access for the heartbeat shipper (lives in MasterClient
  // / PoolClient).  Never null after construction with a non-null
  // allocator argument.
  PeerDramAllocator* DramAllocator() const { return dram_alloc_; }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  LocalStorageManager* storage_;
  LocalBlockIndex* index_;
  PeerDramAllocator* dram_alloc_;
  MasterClient* master_client_;
  SsdCopyPipeline* copy_pipeline_ = nullptr;

  StagingMetrics metrics_;

  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;
  std::vector<uint8_t> engine_desc_bytes_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
