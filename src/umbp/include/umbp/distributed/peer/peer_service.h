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

struct StagingMetrics {
  std::atomic<uint64_t> expired_reclaims{0};
  std::atomic<uint64_t> invalid_lease_rejects{0};
  std::atomic<uint64_t> slot_full_rejects{0};
};

class PeerServiceServer {
 public:
  // `dram_alloc` is non-owning and may be null when the host process has
  // no DRAM/HBM tier (SSD-only deployments).  When null, the new
  // AllocateSlot/CommitSlot/AbortSlot/ResolveKey/EvictKey handlers
  // respond with success=false / found=false; SSD staging RPCs continue
  // to work unchanged.  The allocator's outbox is also where the SSD
  // commit path queues its ADD events for heartbeat shipment.
  PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                    const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                    LocalStorageManager& storage, LocalBlockIndex& index,
                    PeerDramAllocator* dram_alloc = nullptr, int num_read_slots = 8,
                    int num_write_slots = 8, int lease_timeout_s = 10,
                    std::vector<uint8_t> engine_desc_bytes = {});
  PeerServiceServer(PeerDramAllocator* dram_alloc, int num_read_slots = 8, int num_write_slots = 8,
                    int lease_timeout_s = 10, std::vector<uint8_t> engine_desc_bytes = {});
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

  StagingMetrics metrics_;

  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;
  std::vector<uint8_t> engine_desc_bytes_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
