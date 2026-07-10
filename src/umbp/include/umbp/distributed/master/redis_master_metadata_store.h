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

// RedisMasterMetadataStore — a RESP-protocol-compatible IMasterMetadataStore.
//
// Makes the master stateless: all client/block metadata lives in an external
// RESP store (Redis / Dragonfly / Valkey), so any number of master replicas can
// serve traffic and a crashed master reads the full picture back on restart.
//
// PHASE 1 SCOPE (this file): the six hot-path methods that decide the whole
// backend's performance are implemented against real Lua/pipelines, plus the
// handful of methods the running master calls on background threads
// (UnregisterClient, ExpireStaleClients, GarbageCollectHits) and the cheap
// single-command client reads. The external-KV methods and the eviction
// candidate enumeration are Phase 2 and throw std::logic_error if called; they
// are not exercised by the default hot-path benchmark. See
// design-redis-metadata-store.md.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/master/master_metadata_store.h"
#include "umbp/distributed/master/redis/key_schema.h"
#include "umbp/distributed/master/redis/resp_client.h"
#include "umbp/distributed/master/redis/thread_pool.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class RedisMasterMetadataStore : public IMasterMetadataStore {
 public:
  struct Config {
    std::string uri = "tcp://127.0.0.1:6379";
    // Multi-endpoint mode: one Redis instance per entry, block shards spread one
    // per instance so their scripts run on independent server threads (the only
    // way past a single instance's single-thread ceiling). Empty or size 1 =>
    // single-endpoint mode using `uri`. clients_[0] is the control instance
    // (client records, alive set, peer view, extkv all live there).
    std::vector<std::string> shard_uris;
    std::string namespace_id = "default";
    std::string password;
    int connect_timeout_ms = 1000;
    int socket_timeout_ms = 1000;
    std::size_t pool_size = 8;
    // Single-endpoint only: number of hash-tag shards the block keyspace is
    // spread over. 1 keeps the legacy single-tag layout (byte-identical keys,
    // whole-batch-atomic reads). In multi-endpoint mode the shard count is fixed
    // to the number of endpoints (one shard per instance). See KeySchema.
    std::size_t block_shards = 1;
  };

  explicit RedisMasterMetadataStore(const Config& config);
  ~RedisMasterMetadataStore() override = default;

  RedisMasterMetadataStore(const RedisMasterMetadataStore&) = delete;
  RedisMasterMetadataStore& operator=(const RedisMasterMetadataStore&) = delete;

  // --- Cross-store writes ---
  bool RegisterClient(const ClientRegistration& registration,
                      std::chrono::system_clock::time_point now,
                      std::chrono::system_clock::duration stale_after) override;
  void UnregisterClient(const std::string& node_id) override;
  HeartbeatResult ApplyHeartbeat(const std::string& node_id, uint64_t seq,
                                 std::chrono::system_clock::time_point now,
                                 const std::map<TierType, TierCapacity>& caps,
                                 const std::vector<KvEvent>& events, bool is_full_sync) override;
  std::vector<std::string> ExpireStaleClients(
      std::chrono::system_clock::time_point cutoff) override;

  // --- External-KV writes (Phase 2) ---
  bool RegisterExternalKvIfAlive(const std::string& node_id, const std::vector<std::string>& hashes,
                                 TierType tier) override;
  void UnregisterExternalKv(const std::string& node_id, const std::vector<std::string>& hashes,
                            TierType tier) override;
  void UnregisterExternalKvByTier(const std::string& node_id, TierType tier) override;
  void UnregisterExternalKvByNode(const std::string& node_id) override;
  std::size_t GarbageCollectHits(std::chrono::system_clock::time_point cutoff) override;

  // --- Block reads ---
  std::vector<Location> LookupBlock(const std::string& key) const override;
  std::vector<Location> LookupBlockForRouteGet(
      const std::string& key, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) override;
  std::vector<std::vector<Location>> BatchLookupBlockForRouteGet(
      const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) override;
  std::vector<bool> BatchExistsBlock(const std::vector<std::string>& keys) const override;
  std::map<NodeTierKey, std::vector<EvictionCandidate>> EnumerateEvictionCandidates(
      const std::vector<NodeTierKey>& buckets, EvictionOrder order, size_t max_per_bucket,
      std::chrono::system_clock::time_point now) const override;

  // --- Client reads ---
  std::optional<ClientRecord> GetClient(const std::string& node_id) const override;
  bool IsClientAlive(const std::string& node_id) const override;
  std::optional<std::string> GetPeerAddress(const std::string& node_id) const override;
  std::vector<ClientRecord> ListAliveClients() const override;
  std::unordered_map<std::string, std::string> GetAlivePeerView() const override;
  std::size_t AliveClientCount() const override;
  std::vector<std::string> GetClientTags(const std::string& node_id) const override;

  // --- External-KV reads (Phase 2) ---
  std::vector<NodeMatch> MatchExternalKv(const std::vector<std::string>& hashes, bool count_as_hit,
                                         std::chrono::system_clock::time_point now) override;
  std::vector<ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes) const override;
  std::size_t GetExternalKvCount(const std::string& node_id) const override;

  // Attach the master's Prometheus server so hot ops export
  // mori_umbp_store_op_latency_seconds{op,backend="redis"}. Null (default) =
  // no metrics (e.g. the standalone microbench).
  void SetMetricsSink(mori::metrics::MetricsServer* metrics) override { metrics_ = metrics; }

  // Best-effort connectivity probe (PING). Every endpoint must answer.
  bool Ping() const {
    for (const auto& c : clients_) {
      if (!c->Ping()) return false;
    }
    return true;
  }

 private:
  // How many block shards to build for a config (one per endpoint in
  // multi-endpoint mode, else Config::block_shards).
  static std::size_t ResolveNumShards(const Config& config);

  std::size_t num_endpoints() const { return clients_.size(); }
  bool multi_endpoint() const { return clients_.size() > 1; }
  redis::IRespClient& control() const { return *clients_[0]; }
  std::size_t endpoint_of_shard(std::size_t shard) const { return shard % clients_.size(); }
  redis::IRespClient& client_for_shard(std::size_t shard) const {
    return *clients_[endpoint_of_shard(shard)];
  }

  // Multi-endpoint write helpers (see .cpp). Each runs the per-shard block
  // script on the shard's own instance; idempotent so retries are safe.
  void ApplyBlockEventsMulti(const std::string& node_id,
                             const std::vector<KvEvent>& events, bool is_full_sync,
                             std::chrono::system_clock::time_point now);
  void WipeNodeBlocksMulti(const std::string& node_id);

  redis::KeySchema keys_;
  // clients_[0] is the control instance; clients_[s] backs block shard s.
  // Held as IRespClient so the same store logic drives the hiredis single-node
  // client (single / multi-endpoint) and the redis-plus-plus cluster client.
  // mutable so const read methods can borrow a pooled connection.
  mutable std::vector<std::unique_ptr<redis::IRespClient>> clients_;
  // Workers that issue a multi-endpoint read fan-out's per-instance calls
  // concurrently (one round trip instead of N). Null in single-endpoint mode.
  std::unique_ptr<redis::ThreadPool> fanout_pool_;
  // Optional Prometheus sink for per-op latency; null = no metrics.
  mori::metrics::MetricsServer* metrics_ = nullptr;
};

}  // namespace mori::umbp
