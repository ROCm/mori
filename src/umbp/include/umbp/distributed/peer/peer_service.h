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

namespace mori::io {
class IOEngine;
}

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

// Sync-server poller/handler thread pool sizing for the peer service.
//
// gRPC's defaults (MIN_POLLERS=1, MAX_POLLERS=2) assume short handlers.  The
// peer's are the opposite: BatchPrepareSsdRead holds its thread for the whole
// SSD read (tens of ms), and with single-flight on, every follower blocks in
// the handler until its leader finishes.  A ThreadManager thread that completes
// work only resumes polling while pollers < max_pollers and otherwise exits, so
// at MAX_POLLERS=2 every concurrent request past the second costs a thread
// spawn — not once at warmup, but on every request, forever.
//
// That is invisible on an idle storage node and expensive on a node that is
// also running inference: measured on a 2-node pure-SSD run, reads served by a
// sibling rank on the *same* host carried ~8ms more RPC overhead than reads
// pulled across the network from an idle storage node, flat over the whole run.
// Sizing the pool to the expected fan-in (>= tp_size, since MLA + TP has every
// rank ask for the same key) keeps threads resident and removes the spawn.
struct PeerPollerConfig {
  int min_pollers = 8;
  int max_pollers = 64;
};

// Resolve from UMBP_PEER_MIN_POLLERS / UMBP_PEER_MAX_POLLERS, defaults above.
// max is raised to min when a caller sets them inconsistently, since gRPC
// requires max >= min.  Re-reads the environment on every call (not cached):
// Start() runs once per process, and the tests depend on it.
PeerPollerConfig ResolvePeerPollerConfig();

// How long a client's push registration (engine + destination buffers, learned
// from its GetPeerInfo) stays usable without being refreshed.  Every
// BatchPrepareSsdRead from that client refreshes it, so an active client never
// ages out; only one that has gone silent — crashed, killed, restarted — does.
//
// This bound matters because the push path makes the peer an RDMA initiator
// against the client role, which is the short-lived, restart-prone side: a
// client that dies and comes back with the same node_id gets the same
// per-process MemoryUniqueId sequence, so without expiry the peer could resolve
// a push against a registration belonging to the dead instance.  Expiry fails
// safe (fall back to memcpy + pull), never into a misdirected write.
//
// Resolved from UMBP_PUSH_TARGET_TTL_MS.  Re-reads the environment per call.
std::chrono::milliseconds ResolveClientPushTtl();

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
  //
  // `io_engine` (non-owning, may be null) is what lets the peer *initiate* RDMA
  // instead of only serving as a target — the SSD read fan-out push.  It is used
  // for two things: registering a client's engine at GetPeerInfo time, and
  // (indirectly, via the same engine handed to PeerSsdManager) posting the
  // pushes.  Null leaves every read on the classic staging+pull path.  On the
  // PoolClient side the engine is constructed well before this server, so
  // passing it costs nothing.
  PeerServiceServer(PeerDramAllocator* dram_alloc, PeerSsdManager* peer_ssd = nullptr,
                    void* ssd_staging_base = nullptr, size_t ssd_staging_size = 0,
                    std::vector<uint8_t> ssd_staging_mem_desc_bytes = {}, int num_read_slots = 16,
                    std::chrono::milliseconds lease_timeout = std::chrono::milliseconds{3000},
                    std::vector<uint8_t> engine_desc_bytes = {},
                    MasterClient* master_client = nullptr, SsdCopyPipeline* copy_pipeline = nullptr,
                    SsdWriteStagingConfig write_staging = {},
                    mori::io::IOEngine* io_engine = nullptr);
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

  // Live (non-expired) client push registrations.  Test/observability hook: it
  // is the only external way to see that a handshake landed and that a dead
  // client's entry actually ages out.
  size_t SnapshotClientPushTargets() const;

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
  mori::io::IOEngine* io_engine_ = nullptr;

  StagingMetrics metrics_;

  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;
  std::vector<uint8_t> engine_desc_bytes_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
