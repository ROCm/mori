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
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mori::umbp {

class PeerDramAllocator;
class PeerSsdManager;
class MasterClient;
class SsdCopyPipeline;

// Prometheus-only observability counters for the SSD read-staging slots.  NOT
// correctness state (the slot state machine in peer_service.cpp is the source
// of truth) — relaxed atomics incremented at discrete events and read once per
// metrics tick by PoolClient::PublishSsdMetrics.
struct StagingMetrics {
  std::atomic<uint64_t> expired_reclaims{0};   // leased slot reclaimed past TTL
  std::atomic<uint64_t> slot_full_rejects{0};  // PrepareSsdRead -> NO_SLOT
  // Direct-SSD put counterparts (write staging region).
  std::atomic<uint64_t> write_expired_reclaims{0};
  std::atomic<uint64_t> write_slot_full_rejects{0};  // AllocateSsdWriteSlots -> NO_SLOT
};

// Write-staging geometry for the direct-SSD put path.  slots == 0 disables the
// path entirely: the peer advertises no write slots and rejects the RPCs, which
// is exactly the pre-existing behavior for DRAM-backed deployments.
//
// The region lives ABOVE the read region inside the same RDMA-registered
// staging buffer, so `region_size` bytes must have been allocated on top of the
// read region's size — the read region's slot count and slot size are never
// changed by enabling writes.
struct SsdWriteStagingConfig {
  int slots = 0;
  size_t region_base = 0;  // byte offset of the write region within the buffer
  size_t region_size = 0;
};

class PeerServiceServer {
 public:
  // `dram_alloc` is non-owning and may be null when the host process has
  // no DRAM/HBM tier (SSD-only deployments).  When null, the
  // AllocateSlot/CommitSlot/AbortSlot/ResolveKey/EvictKey handlers
  // respond with success=false / found=false.
  //
  // `peer_ssd` + the SSD staging region (base / size / packed MemoryDesc) drive
  // the SSD read RPCs: when all are present the peer serves PrepareSsdRead out
  // of `peer_ssd` into the staging buffer.  When `peer_ssd` is null (SSD
  // disabled) the staging args are typically null/0 and the SSD read RPCs
  // report SSD_READ_ERROR (SsdRpcAvailable() == false).  The staging buffer
  // must be RDMA-registered by the caller; its MemoryDesc is published via
  // GetPeerInfo so readers can RDMA out of it.
  //
  // `copy_pipeline` (non-owning, may be null when SSD is disabled) receives an
  // SsdCopyTask after each successful DRAM commit so the owner peer copies the
  // committed bytes to its local SSD tier asynchronously.
  //
  // `write_staging` carves the direct-SSD put region out of the same buffer
  // (see SsdWriteStagingConfig); default-constructed means the peer accepts no
  // direct SSD writes.
  PeerServiceServer(PeerDramAllocator* dram_alloc, PeerSsdManager* peer_ssd = nullptr,
                    void* ssd_staging_base = nullptr, size_t ssd_staging_size = 0,
                    std::vector<uint8_t> ssd_staging_mem_desc_bytes = {}, int num_read_slots = 16,
                    std::chrono::milliseconds lease_timeout = std::chrono::milliseconds{3000},
                    std::vector<uint8_t> engine_desc_bytes = {},
                    MasterClient* master_client = nullptr, SsdCopyPipeline* copy_pipeline = nullptr,
                    SsdWriteStagingConfig write_staging = {});
  ~PeerServiceServer();

  bool Start(uint16_t port);
  void Stop();

  const StagingMetrics& Metrics() const { return metrics_; }

  // SSD read staging slots currently in use (Preparing or Leased).  Sampled
  // once per metrics flush by PoolClient's metrics provider for a gauge.
  size_t SnapshotReadSlotsInUse() const;

  // Same, for the direct-SSD put write-staging region.  Always 0 when the path
  // is disabled.
  size_t SnapshotWriteSlotsInUse() const;

  // Read-only access for the heartbeat shipper (lives in MasterClient
  // / PoolClient).  Never null after construction with a non-null
  // allocator argument.
  PeerDramAllocator* DramAllocator() const { return dram_alloc_; }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  SsdWriteStagingConfig write_staging_;
  PeerSsdManager* peer_ssd_;
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
