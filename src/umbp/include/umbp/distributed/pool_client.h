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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  bool Init();
  void Shutdown();

  const std::string& NodeId() const { return config_.master_config.node_id; }

  bool RegisterMemory(void* ptr, size_t size);
  void DeregisterMemory(void* ptr);

  // Phase 2: DRAM-only methods for UMBPClient integration.
  // UMBPClient handles local storage directly and calls these for cluster
  // interactions only. PoolClient never touches local storage.

  // Register an already-written local block with the Master so remote nodes
  // can discover it. UMBPClient provides the location_id (e.g. "0:<offset>").
  bool RegisterWithMaster(const std::string& key, size_t size, const std::string& location_id,
                          TierType tier);
  bool FinalizeAllocation(const std::string& key, size_t size, const std::string& location_id,
                          TierType tier, const std::string& allocation_id);
  bool PublishLocalBlock(const std::string& key, size_t size, const std::string& location_id,
                         TierType tier);
  bool AbortAllocation(const std::string& node_id, TierType tier, const std::string& allocation_id,
                       uint64_t size);

  // Check whether a block exists on any remote node (RouteGet without RDMA).
  bool ExistsRemote(const std::string& key);

  bool IsRegistered(const std::string& key) const;

  // Fetch a block from a remote node via RDMA.
  // DRAM: RouteGet -> direct RDMA read.
  // SSD: RouteGet -> PeerService PrepareSsdRead (SSD->staging slot) -> RDMA read.
  bool GetRemote(const std::string& key, void* dst, size_t size);

  // Write a block to a remote node via RDMA.
  // DRAM: RoutePut -> direct RDMA write.
  // SSD: RoutePut -> AllocateWriteSlot -> RDMA write -> CommitSsdWrite.
  bool PutRemote(const std::string& key, const void* src, size_t size);

  // Batch write: single gRPC for routing + batched RDMA + single gRPC for finalize.
  std::vector<bool> BatchPutRemote(const std::vector<std::string>& keys,
                                   const std::vector<const void*>& srcs,
                                   const std::vector<size_t>& sizes,
                                   const std::vector<int>& depths = {});

  // Batch read: single gRPC for routing + batched RDMA.
  std::vector<bool> BatchGetRemote(const std::vector<std::string>& keys,
                                   const std::vector<void*>& dsts,
                                   const std::vector<size_t>& sizes);

  // Unregister a block from the Master (block no longer remotely accessible).
  bool UnregisterFromMaster(const std::string& key);

  void* SsdStagingPtr() const { return ssd_staging_buffer_.get(); }
  size_t SsdStagingSize() const { return config_.staging_buffer_size; }
  const std::vector<uint8_t>& SsdStagingMemDescBytes() const { return ssd_staging_mem_desc_bytes_; }

  MasterClient& Master();
  bool IsInitialized() const;

 private:
  PoolClientConfig config_;
  std::atomic<bool> initialized_{false};

  std::unique_ptr<MasterClient> master_client_;

  // IO Engine (data plane)
  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_{};
  std::vector<mori::io::MemoryDesc> export_dram_mems_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // SSD staging buffer — separate from DRAM exportable buffers so that
  // PeerService SSD staging traffic does not conflict with Master-managed
  // DRAM tier offset allocations.
  std::unique_ptr<char[]> ssd_staging_buffer_;
  mori::io::MemoryDesc ssd_staging_mem_{};
  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;

  // Peer connections (lazy init, keyed by node_id)
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    std::vector<mori::io::MemoryDesc> dram_memories;
    bool engine_registered = false;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;

    // Dedicated SSD staging MemoryDesc, independent of dram_memories to avoid
    // offset conflicts between DRAM tier allocations and SSD staging traffic.
    mori::io::MemoryDesc ssd_staging_mem{};
    size_t ssd_staging_size = 0;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  PeerConnection& GetOrConnectPeer(const std::string& node_id, const std::string& peer_address,
                                   const std::vector<uint8_t>& engine_desc_bytes,
                                   const std::vector<uint8_t>& dram_memory_desc_bytes,
                                   uint32_t buffer_index = 0);

  bool RemoteDramWrite(PeerConnection& peer, uint32_t buffer_index, const void* src, size_t size,
                       uint64_t offset, bool zero_copy);
  bool RemoteDramRead(PeerConnection& peer, uint32_t buffer_index, void* dst, size_t size,
                      uint64_t offset, bool zero_copy);
  bool EnsurePeerServiceConnection(PeerConnection& peer);
  bool RemoteSsdWrite(PeerConnection& peer, const std::string& key, const void* src, size_t size,
                      bool zero_copy, uint32_t store_index = 0,
                      const std::string& allocation_id = "");
  bool RemoteSsdRead(PeerConnection& peer, const std::string& key, const std::string& location_id,
                     void* dst, size_t size, bool zero_copy);

  // Zero-copy registered memory regions
  struct RegisteredRegion {
    void* base;
    size_t size;
    mori::io::MemoryDesc mem_desc;
  };
  std::mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;

  std::optional<std::pair<mori::io::MemoryDesc, size_t>> FindRegisteredMemory(const void* ptr,
                                                                              size_t size);

  mutable std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> cluster_locations_;
};

}  // namespace mori::umbp
