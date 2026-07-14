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
#include "umbp/distributed/master/redis_master_metadata_store.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/redis/lua_scripts.h"
#include "umbp/distributed/master/redis/resp_cluster_client.h"
#include "umbp/distributed/master/redis/sharded_read.h"

namespace mori::umbp {

namespace {

using redis::RespValue;

// Records store-op latency into mori_umbp_store_op_latency_seconds{op,backend=redis}
// on scope exit when a metrics sink is attached; a cheap no-op otherwise. Lets a
// dashboard separate backend round-trip cost per store method.
class ScopedStoreOp {
 public:
  ScopedStoreOp(mori::metrics::MetricsServer* metrics, const char* op)
      : metrics_(metrics), op_(op) {
    if (metrics_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedStoreOp() {
    if (!metrics_) return;
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    static const std::vector<double> kBounds = {0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01,
                                                0.025,  0.05,    0.1,    0.25,  0.5,    1.0};
    metrics_->observe("mori_umbp_store_op_latency_seconds",
                      "Latency of IMasterMetadataStore operations against the backend",
                      {{"op", op_}, {"backend", "redis"}}, kBounds, sec);
  }
  ScopedStoreOp(const ScopedStoreOp&) = delete;
  ScopedStoreOp& operator=(const ScopedStoreOp&) = delete;

 private:
  mori::metrics::MetricsServer* metrics_;
  const char* op_;
  std::chrono::steady_clock::time_point t0_;
};

int64_t ToEpochMs(std::chrono::system_clock::time_point tp) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

int64_t ToMs(std::chrono::system_clock::duration d) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

std::chrono::system_clock::time_point FromEpochMs(int64_t ms) {
  return std::chrono::system_clock::time_point(std::chrono::milliseconds(ms));
}

// caps: "tier:total:avail;tier:total:avail;..."
std::string EncodeCaps(const std::map<TierType, TierCapacity>& caps) {
  std::string out;
  for (const auto& [tier, cap] : caps) {
    out += std::to_string(static_cast<int>(tier));
    out += ':';
    out += std::to_string(cap.total_bytes);
    out += ':';
    out += std::to_string(cap.available_bytes);
    out += ';';
  }
  return out;
}

std::map<TierType, TierCapacity> DecodeCaps(const std::string& blob) {
  std::map<TierType, TierCapacity> caps;
  size_t pos = 0;
  while (pos < blob.size()) {
    const size_t semi = blob.find(';', pos);
    const size_t end = (semi == std::string::npos) ? blob.size() : semi;
    const std::string tok = blob.substr(pos, end - pos);
    pos = (semi == std::string::npos) ? blob.size() : semi + 1;
    if (tok.empty()) continue;
    const size_t c1 = tok.find(':');
    const size_t c2 = (c1 == std::string::npos) ? std::string::npos : tok.find(':', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos) continue;
    try {
      const int tier = std::stoi(tok.substr(0, c1));
      const uint64_t total = std::stoull(tok.substr(c1 + 1, c2 - c1 - 1));
      const uint64_t avail = std::stoull(tok.substr(c2 + 1));
      caps[static_cast<TierType>(tier)] = TierCapacity{total, avail};
    } catch (const std::exception&) {
      // best-effort decode; skip malformed token
    }
  }
  return caps;
}

std::string JoinTags(const std::vector<std::string>& tags) {
  std::string out;
  for (size_t i = 0; i < tags.size(); ++i) {
    if (i) out += '\n';
    out += tags[i];
  }
  return out;
}

std::vector<std::string> SplitTags(const std::string& blob) {
  std::vector<std::string> out;
  if (blob.empty()) return out;
  size_t pos = 0;
  while (pos <= blob.size()) {
    const size_t nl = blob.find('\n', pos);
    const size_t end = (nl == std::string::npos) ? blob.size() : nl;
    out.push_back(blob.substr(pos, end - pos));
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  return out;
}

// Decode a flat HGETALL reply [f,v,f,v,...] into a field->value map.
std::unordered_map<std::string, std::string> FlatToMap(const RespValue& flat) {
  std::unordered_map<std::string, std::string> m;
  if (!flat.is_array()) return m;
  for (size_t i = 0; i + 1 < flat.elements.size(); i += 2) {
    m.emplace(flat.elements[i].str, flat.elements[i + 1].str);
  }
  return m;
}

ClientRecord DecodeRecord(const std::string& node_id,
                          const std::unordered_map<std::string, std::string>& f) {
  ClientRecord rec;
  rec.node_id = node_id;
  auto get = [&](const char* k) -> const std::string* {
    auto it = f.find(k);
    return it == f.end() ? nullptr : &it->second;
  };
  if (auto* v = get("addr")) rec.node_address = *v;
  if (auto* v = get("peer")) rec.peer_address = *v;
  if (auto* v = get("status")) {
    try {
      rec.status = static_cast<ClientStatus>(std::stoi(*v));
    } catch (...) {
      rec.status = ClientStatus::UNKNOWN;
    }
  }
  if (auto* v = get("last_hb")) {
    try {
      rec.last_heartbeat = FromEpochMs(std::stoll(*v));
    } catch (...) {
    }
  }
  if (auto* v = get("reg_at")) {
    try {
      rec.registered_at = FromEpochMs(std::stoll(*v));
    } catch (...) {
    }
  }
  if (auto* v = get("seq")) {
    try {
      rec.last_applied_seq = std::stoull(*v);
    } catch (...) {
    }
  }
  if (auto* v = get("caps")) rec.tier_capacities = DecodeCaps(*v);
  if (auto* v = get("engine")) rec.engine_desc_bytes.assign(v->begin(), v->end());
  if (auto* v = get("tags")) rec.tags = SplitTags(*v);
  return rec;
}

// ShardedBatch / GroupKeysByShard / RunShardedRead moved to
// redis/sharded_read.h so the batch fan-out + fault-tolerance + reply-scatter
// logic can be unit-tested against a fake IRespClient (see test_sharded_read).

}  // namespace

std::size_t RedisMasterMetadataStore::ResolveNumShards(const Config& config) {
  if (config.cluster)
    return config.block_shards == 0 ? 1 : config.block_shards;        // slot-spread tags
  if (config.shard_uris.size() > 1) return config.shard_uris.size();  // one shard per endpoint
  return config.block_shards == 0 ? 1 : config.block_shards;          // single-endpoint knob
}

redis::KeySchema RedisMasterMetadataStore::BuildKeySchema(const Config& config) {
  // Cluster balanced placement: use the explicit per-master tags when the
  // factory supplied them; otherwise fall back to formulaic shard tags.
  if (config.cluster && !config.cluster_block_tags.empty()) {
    return redis::KeySchema(config.namespace_id, config.cluster_block_tags);
  }
  return redis::KeySchema(config.namespace_id, ResolveNumShards(config));
}

RedisMasterMetadataStore::RedisMasterMetadataStore(const Config& config)
    : keys_(BuildKeySchema(config)),
      mode_(config.cluster                 ? Mode::kCluster
            : config.shard_uris.size() > 1 ? Mode::kMulti
                                           : Mode::kSingle) {
  if (mode_ == Mode::kCluster) {
    // One Redis Cluster client routes every shard by slot; the store still uses
    // the split control + per-shard-block write path (their keys live in
    // different slots), driven by split_writes().
    redis::RespClusterClient::Options opts;
    opts.seeds =
        config.cluster_seeds.empty() ? std::vector<std::string>{config.uri} : config.cluster_seeds;
    opts.password = config.password;
    opts.connect_timeout_ms = config.connect_timeout_ms;
    opts.socket_timeout_ms = config.socket_timeout_ms;
    opts.pool_size = config.pool_size;
    clients_.push_back(std::make_unique<redis::RespClusterClient>(std::move(opts)));
    MORI_UMBP_INFO("[RedisStore] cluster namespace={} pool={} seeds={} block_shards={}",
                   config.namespace_id, config.pool_size, opts.seeds.size(), keys_.NumShards());
    return;
  }

  // Endpoints: the shard_uris list when given (multi-endpoint), else the single
  // `uri`. clients_[0] is the control instance and also backs block shard 0.
  std::vector<std::string> uris =
      config.shard_uris.size() > 1 ? config.shard_uris : std::vector<std::string>{config.uri};

  clients_.reserve(uris.size());
  for (const auto& uri : uris) {
    redis::RespClient::Options opts;
    opts.uri = uri;
    opts.password = config.password;
    opts.connect_timeout_ms = config.connect_timeout_ms;
    opts.socket_timeout_ms = config.socket_timeout_ms;
    opts.pool_size = config.pool_size;
    clients_.push_back(std::make_unique<redis::RespClient>(std::move(opts)));
  }

  if (multi_endpoint()) {
    // Workers to fan the per-instance read calls out concurrently. Sized so a
    // handful of in-flight RouteGets can each parallelize their N instance calls
    // without per-call thread churn; workers block on Redis I/O so a generous
    // count is cheap. Capped to keep it bounded.
    const std::size_t pool_threads = std::min<std::size_t>(64, clients_.size() * 8);
    fanout_pool_ = std::make_unique<redis::ThreadPool>(pool_threads);

    std::string joined;
    for (size_t i = 0; i < uris.size(); ++i) joined += (i ? "," : "") + uris[i];
    MORI_UMBP_INFO("[RedisStore] multi-endpoint namespace={} pool={} endpoints={} fanout={} [{}]",
                   config.namespace_id, config.pool_size, uris.size(), pool_threads, joined);
  } else {
    MORI_UMBP_INFO("[RedisStore] backend at {} namespace={} pool={} block_shards={}", config.uri,
                   config.namespace_id, config.pool_size, keys_.NumShards());
  }
}

// =====================================================================
// Multi-endpoint write helpers: run per-shard block scripts on each shard's own
// instance. Idempotent (ADD overwrites / REMOVE no-op / full_sync clear+replay)
// so a thrown/retried step is safe; a partial failure surfaces as an exception,
// the peer retries, seq-gaps, and heals via full_sync.
// =====================================================================

void RedisMasterMetadataStore::ApplyBlockEventsMulti(const std::string& node_id,
                                                     const std::vector<KvEvent>& events,
                                                     bool is_full_sync,
                                                     std::chrono::system_clock::time_point now) {
  const std::string node_prefix = "l|" + node_id + "|";
  const std::string now_ms = std::to_string(ToEpochMs(now));

  // Group events by their shard so each instance gets exactly its own events.
  std::vector<std::vector<const KvEvent*>> events_by_shard(keys_.NumShards());
  for (const auto& ev : events) events_by_shard[keys_.ShardOf(ev.key)].push_back(&ev);

  // full_sync must clear the node on EVERY shard (to drop stale locations), even
  // shards with no new ADDs; a delta only touches shards that have events.
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    const auto& shard_events = events_by_shard[shard];
    if (!is_full_sync && shard_events.empty()) continue;

    std::vector<std::string> args;
    args.reserve(5 + shard_events.size() * 4);
    args.push_back(keys_.NodeBlocks(node_id, shard));
    args.push_back(node_prefix);
    args.push_back(is_full_sync ? "1" : "0");
    args.push_back(now_ms);
    args.push_back(std::to_string(shard_events.size()));
    for (const KvEvent* ev : shard_events) {
      args.push_back(ev->kind == KvEvent::Kind::ADD ? "0" : "1");
      args.push_back(keys_.Block(ev->key));
      args.push_back(std::to_string(static_cast<int>(ev->tier)));
      args.push_back(std::to_string(ev->size));
    }
    // KEYS[1] = this shard's reverse-index key (a shard-tag key) so the cluster
    // client can route to the shard's slot; the script reads all its keys from
    // ARGV (same slot). Harmless in single / multi-endpoint mode (a standalone
    // Redis does not slot-check, and the script ignores KEYS).
    RespValue r = client_for_shard(shard).Eval(redis::kApplyBlockEventsLua,
                                               {keys_.NodeBlocks(node_id, shard)}, args);
    if (r.is_error()) throw std::runtime_error("[RedisStore] ApplyBlockEvents: " + r.str);
  }
}

void RedisMasterMetadataStore::WipeNodeBlocksMulti(const std::string& node_id) {
  const std::string node_prefix = "l|" + node_id + "|";
  // Best-effort per shard: a down shard must not block wiping the reachable
  // ones. The node is already gone/EXPIRED on the control instance, so any
  // locations lingering on an unreachable shard point at a dead node and are
  // filtered out of reads by GetAlivePeerView until that shard returns.
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    try {
      RespValue r = client_for_shard(shard).Eval(redis::kWipeNodeBlocksLua,
                                                 {keys_.NodeBlocks(node_id, shard)},
                                                 {keys_.NodeBlocks(node_id, shard), node_prefix});
      if (r.is_error()) {
        MORI_UMBP_WARN("[RedisStore] WipeNodeBlocks: shard {} script error, skipped: {}", shard,
                       r.str);
      }
    } catch (const std::exception& e) {
      MORI_UMBP_WARN("[RedisStore] WipeNodeBlocks: shard {} unavailable, skipped: {}", shard,
                     e.what());
    }
  }
}

// =====================================================================
// Cross-store writes
// =====================================================================

bool RedisMasterMetadataStore::RegisterClient(const ClientRegistration& registration,
                                              std::chrono::system_clock::time_point now,
                                              std::chrono::system_clock::duration stale_after) {
  ScopedStoreOp _op(metrics_, "RegisterClient");
  const std::string engine(registration.engine_desc_bytes.begin(),
                           registration.engine_desc_bytes.end());
  RespValue r = control().Eval(
      redis::kRegisterClientLua, {keys_.Node(registration.node_id)},
      {keys_.Tag(), registration.node_id, std::to_string(ToEpochMs(now)),
       std::to_string(ToMs(stale_after)), registration.node_address, registration.peer_address,
       EncodeCaps(registration.tier_capacities), engine, JoinTags(registration.tags)});
  if (r.is_error()) throw std::runtime_error("[RedisStore] RegisterClient: " + r.str);
  return r.integer == 1;
}

void RedisMasterMetadataStore::UnregisterClient(const std::string& node_id) {
  ScopedStoreOp _op(metrics_, "UnregisterClient");
  if (!split_writes()) {
    RespValue r =
        control().Eval(redis::kUnregisterClientLua, {keys_.Node(node_id)}, {keys_.Tag(), node_id});
    if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterClient: " + r.str);
    return;
  }
  // Multi-endpoint: control record first (so the router stops routing to it),
  // then wipe its block locations on every shard's instance. A lingering
  // location for the now-gone node is filtered out by GetAlivePeerView, and the
  // wipe is idempotent, so a mid-way failure + retry is safe.
  RespValue r =
      control().Eval(redis::kUnregisterControlLua, {keys_.Node(node_id)}, {keys_.Tag(), node_id});
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterClient(control): " + r.str);
  WipeNodeBlocksMulti(node_id);
}

HeartbeatResult RedisMasterMetadataStore::ApplyHeartbeat(
    const std::string& node_id, uint64_t seq, std::chrono::system_clock::time_point now,
    const std::map<TierType, TierCapacity>& caps, const std::vector<KvEvent>& events,
    bool is_full_sync) {
  ScopedStoreOp _op(metrics_, "ApplyHeartbeat");
  RespValue r;
  if (!split_writes()) {
    // Single instance: one atomic script does seq-CAS + record + blocks.
    std::vector<std::string> args;
    args.reserve(7 + events.size() * 4);
    args.push_back(keys_.Tag());
    args.push_back(node_id);
    args.push_back(std::to_string(seq));
    args.push_back(std::to_string(ToEpochMs(now)));
    args.push_back(is_full_sync ? "1" : "0");
    args.push_back(EncodeCaps(caps));
    args.push_back(std::to_string(events.size()));
    for (const auto& ev : events) {
      args.push_back(ev.kind == KvEvent::Kind::ADD ? "0" : "1");
      // Pass the fully composed (sharded) block key so the Lua script never has
      // to know the shard layout; it also becomes the reverse-index member.
      args.push_back(keys_.Block(ev.key));
      args.push_back(std::to_string(static_cast<int>(ev.tier)));
      args.push_back(std::to_string(ev.size));
    }
    r = control().Eval(redis::kApplyHeartbeatLua, {keys_.Node(node_id)}, args);
  } else {
    // Multi-endpoint: control step (seq-CAS + record + alive/peers) only; block
    // events are applied per shard afterwards if this heartbeat is APPLIED.
    r = control().Eval(redis::kApplyHeartbeatControlLua, {keys_.Node(node_id)},
                       {keys_.Tag(), node_id, std::to_string(seq), std::to_string(ToEpochMs(now)),
                        is_full_sync ? "1" : "0", EncodeCaps(caps)});
  }

  if (r.is_error()) throw std::runtime_error("[RedisStore] ApplyHeartbeat: " + r.str);
  if (!r.is_array() || r.elements.size() < 2) {
    throw std::runtime_error("[RedisStore] ApplyHeartbeat: malformed reply");
  }
  const std::string& status = r.elements[0].str;
  uint64_t acked = 0;
  try {
    acked = std::stoull(r.elements[1].str);
  } catch (...) {
  }
  if (status == "UNKNOWN") return HeartbeatResult{HeartbeatResult::UNKNOWN, 0};
  if (status == "SEQ_GAP") return HeartbeatResult{HeartbeatResult::SEQ_GAP, acked};

  // APPLIED. In multi-endpoint mode the control record has advanced (so the node
  // is already marked ALIVE and won't be reaped); now apply the block events on
  // each shard's instance. If a shard is down, don't fail the RPC — return
  // SEQ_GAP so the peer full_syncs and self-heals via the existing recovery path
  // once the shard is back (block ops are idempotent, so the replay is safe).
  if (split_writes()) {
    try {
      ApplyBlockEventsMulti(node_id, events, is_full_sync, now);
    } catch (const std::exception& e) {
      MORI_UMBP_WARN(
          "[RedisStore] ApplyHeartbeat: block apply degraded for node {} (a shard is "
          "unavailable); requesting full_sync to self-heal: {}",
          node_id, e.what());
      return HeartbeatResult{HeartbeatResult::SEQ_GAP, acked};
    }
  }
  return HeartbeatResult{HeartbeatResult::APPLIED, acked};
}

std::vector<std::string> RedisMasterMetadataStore::ExpireStaleClients(
    std::chrono::system_clock::time_point cutoff) {
  ScopedStoreOp _op(metrics_, "ExpireStaleClients");
  // Bind a reference (not const char*) so the SHA cache's pointer-identity key
  // (&script) stays stable — see lua_scripts.h.
  const std::string& script = split_writes() ? redis::kExpireControlLua : redis::kExpireStaleLua;
  // KEYS[1] = a control-tag key (nodes:alive) to route + fix the slot in cluster
  // mode; the script reads tag from ARGV and touches only control-tag keys.
  RespValue r = control().Eval(script, {keys_.NodesAlive()},
                               {keys_.Tag(), std::to_string(ToEpochMs(cutoff))});
  if (r.is_error()) throw std::runtime_error("[RedisStore] ExpireStaleClients: " + r.str);
  std::vector<std::string> dead;
  if (r.is_array()) {
    dead.reserve(r.elements.size());
    for (const auto& e : r.elements) dead.push_back(e.str);
  }
  // Split-write modes: the control step only flipped status + returned the dead
  // ids; wipe each dead node's block locations on every shard.
  if (split_writes()) {
    for (const auto& id : dead) WipeNodeBlocksMulti(id);
  }
  return dead;
}

// =====================================================================
// External-KV writes. extkv + hit live on the control tag, so each is one
// single-slot Lua on the control instance in every mode (single / multi-endpoint
// control instance / cluster control slot). See design §4/§5.
// =====================================================================

bool RedisMasterMetadataStore::RegisterExternalKvIfAlive(const std::string& node_id,
                                                         const std::vector<std::string>& hashes,
                                                         TierType tier) {
  ScopedStoreOp _op(metrics_, "RegisterExternalKvIfAlive");
  std::vector<std::string> args;
  args.reserve(4 + hashes.size());
  args.push_back(keys_.Tag());
  args.push_back(node_id);
  args.push_back(std::to_string(static_cast<int>(tier)));
  args.push_back(std::to_string(hashes.size()));
  for (const auto& h : hashes) args.push_back(h);
  RespValue r = control().Eval(redis::kRegisterExternalKvLua, {keys_.Node(node_id)}, args);
  if (r.is_error()) throw std::runtime_error("[RedisStore] RegisterExternalKvIfAlive: " + r.str);
  return r.integer == 1;
}

void RedisMasterMetadataStore::UnregisterExternalKv(const std::string& node_id,
                                                    const std::vector<std::string>& hashes,
                                                    TierType tier) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKv");
  std::vector<std::string> args;
  args.reserve(4 + hashes.size());
  args.push_back(keys_.Tag());
  args.push_back(node_id);
  args.push_back(std::to_string(static_cast<int>(tier)));
  args.push_back(std::to_string(hashes.size()));
  for (const auto& h : hashes) args.push_back(h);
  RespValue r = control().Eval(redis::kUnregisterExternalKvLua, {keys_.ExtKvNode(node_id)}, args);
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterExternalKv: " + r.str);
}

void RedisMasterMetadataStore::UnregisterExternalKvByTier(const std::string& node_id,
                                                          TierType tier) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKvByTier");
  RespValue r = control().Eval(redis::kUnregisterExternalKvByTierLua, {keys_.ExtKvNode(node_id)},
                               {keys_.Tag(), node_id, std::to_string(static_cast<int>(tier))});
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterExternalKvByTier: " + r.str);
}

void RedisMasterMetadataStore::UnregisterExternalKvByNode(const std::string& node_id) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKvByNode");
  RespValue r = control().Eval(redis::kUnregisterExternalKvByNodeLua, {keys_.ExtKvNode(node_id)},
                               {keys_.Tag(), node_id});
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterExternalKvByNode: " + r.str);
}

std::size_t RedisMasterMetadataStore::GarbageCollectHits(
    std::chrono::system_clock::time_point cutoff) {
  ScopedStoreOp _op(metrics_, "GarbageCollectHits");
  // Drop every hit counter whose last_seen < cutoff via the hit:index reverse
  // set (no SCAN — cluster-routable by the index key). Runs on the slow hit-GC
  // timer, not the hot path.
  RespValue r = control().Eval(redis::kGarbageCollectHitsLua, {keys_.HitIndex()},
                               {keys_.Tag(), std::to_string(ToEpochMs(cutoff))});
  if (r.is_error()) throw std::runtime_error("[RedisStore] GarbageCollectHits: " + r.str);
  return r.type == RespValue::Type::Integer ? static_cast<std::size_t>(r.integer) : 0;
}

// =====================================================================
// Block reads
// =====================================================================

std::vector<Location> RedisMasterMetadataStore::LookupBlock(const std::string& key) const {
  ScopedStoreOp _op(metrics_, "LookupBlock");
  RespValue r = client_for_shard(keys_.ShardOf(key)).Command({"HGETALL", keys_.Block(key)});
  std::vector<Location> out;
  if (!r.is_array()) return out;
  for (size_t i = 0; i + 1 < r.elements.size(); i += 2) {
    const std::string& f = r.elements[i].str;
    if (f.rfind("l|", 0) != 0) continue;
    const std::string rest = f.substr(2);
    const size_t sep = rest.find('|');
    if (sep == std::string::npos) continue;
    Location loc;
    loc.node_id = rest.substr(0, sep);
    try {
      loc.tier = static_cast<TierType>(std::stoi(rest.substr(sep + 1)));
      loc.size = std::stoull(r.elements[i + 1].str);
    } catch (...) {
      continue;
    }
    out.push_back(std::move(loc));
  }
  return out;
}

std::vector<Location> RedisMasterMetadataStore::LookupBlockForRouteGet(
    const std::string& key, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::system_clock::time_point now, std::chrono::system_clock::duration lease_duration) {
  auto batch = BatchLookupBlockForRouteGet({key}, exclude_nodes, now, lease_duration);
  return batch.empty() ? std::vector<Location>{} : std::move(batch.front());
}

std::vector<std::vector<Location>> RedisMasterMetadataStore::BatchLookupBlockForRouteGet(
    const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::system_clock::time_point now, std::chrono::system_clock::duration lease_duration) {
  std::vector<std::vector<Location>> out(keys.size());
  if (keys.empty()) return out;
  ScopedStoreOp _op(metrics_, "BatchLookupBlockForRouteGet");

  std::vector<std::string> args;
  args.reserve(3 + exclude_nodes.size());
  args.push_back(std::to_string(ToEpochMs(now)));
  args.push_back(std::to_string(ToMs(lease_duration)));
  args.push_back(std::to_string(exclude_nodes.size()));
  for (const auto& n : exclude_nodes) args.push_back(n);

  // Fan out one single-slot route_get_batch per shard; groups on the same
  // instance share one pipeline, groups on different instances run against
  // their own instance. Each key's reply is a flat [node, size, tier, ...] list.
  const redis::ShardedBatch batch = redis::GroupKeysByShard(keys_, keys);
  redis::RunShardedRead(
      batch, redis::kRouteGetBatchLua, args, "BatchLookupBlockForRouteGet", fanout_pool_.get(),
      /*tolerate_shard_failures=*/mode_ != Mode::kSingle, metrics_,
      [&](size_t group) { return &client_for_shard(batch.shard_of_group[group]); },
      [&](size_t orig_index, const RespValue& locs) {
        if (!locs.is_array()) return;
        for (size_t j = 0; j + 3 <= locs.elements.size(); j += 3) {
          Location loc;
          loc.node_id = locs.elements[j].str;
          try {
            loc.size = std::stoull(locs.elements[j + 1].str);
            loc.tier = static_cast<TierType>(std::stoi(locs.elements[j + 2].str));
          } catch (...) {
            continue;
          }
          out[orig_index].push_back(std::move(loc));
        }
      });
  return out;
}

std::vector<bool> RedisMasterMetadataStore::BatchExistsBlock(
    const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;
  ScopedStoreOp _op(metrics_, "BatchExistsBlock");

  const redis::ShardedBatch batch = redis::GroupKeysByShard(keys_, keys);
  redis::RunShardedRead(
      batch, redis::kExistsBatchLua, {}, "BatchExistsBlock", fanout_pool_.get(),
      /*tolerate_shard_failures=*/mode_ != Mode::kSingle, metrics_,
      [&](size_t group) { return &client_for_shard(batch.shard_of_group[group]); },
      [&](size_t orig_index, const RespValue& has) { results[orig_index] = has.integer != 0; });
  return results;
}

std::map<NodeTierKey, std::vector<EvictionCandidate>>
RedisMasterMetadataStore::EnumerateEvictionCandidates(const std::vector<NodeTierKey>&,
                                                      EvictionOrder, size_t,
                                                      std::chrono::system_clock::time_point) const {
  // Phase 1: master-driven eviction (the per-(node,tier) LRU index + candidate
  // enumeration) is Phase 2. Returning no candidates makes the eviction tick a
  // safe no-op; the hot-path benchmark does not cross watermark. See
  // design-redis-metadata-store.md.
  return {};
}

// =====================================================================
// Client reads
// =====================================================================

std::optional<ClientRecord> RedisMasterMetadataStore::GetClient(const std::string& node_id) const {
  RespValue r = control().Command({"HGETALL", keys_.Node(node_id)});
  if (!r.is_array() || r.elements.empty()) return std::nullopt;
  return DecodeRecord(node_id, FlatToMap(r));
}

bool RedisMasterMetadataStore::IsClientAlive(const std::string& node_id) const {
  RespValue r = control().Command({"HGET", keys_.Node(node_id), "status"});
  return r.type == RespValue::Type::String && r.str == "1";
}

std::optional<std::string> RedisMasterMetadataStore::GetPeerAddress(
    const std::string& node_id) const {
  RespValue r = control().Command({"HGET", keys_.Node(node_id), "peer"});
  if (r.is_nil()) return std::nullopt;
  return r.str;
}

std::vector<ClientRecord> RedisMasterMetadataStore::ListAliveClients() const {
  ScopedStoreOp _op(metrics_, "ListAliveClients");
  // KEYS[1] = a control-tag key (nodes:alive) to route + fix the slot in cluster
  // mode; the script reads tag from ARGV and touches only control-tag keys.
  RespValue r = control().Eval(redis::kListAliveLua, {keys_.NodesAlive()}, {keys_.Tag()});
  if (r.is_error()) throw std::runtime_error("[RedisStore] ListAliveClients: " + r.str);
  std::vector<ClientRecord> out;
  if (!r.is_array()) return out;
  out.reserve(r.elements.size());
  for (const auto& entry : r.elements) {
    if (!entry.is_array() || entry.elements.size() < 2) continue;
    const std::string& id = entry.elements[0].str;
    out.push_back(DecodeRecord(id, FlatToMap(entry.elements[1])));
  }
  return out;
}

std::unordered_map<std::string, std::string> RedisMasterMetadataStore::GetAlivePeerView() const {
  ScopedStoreOp _op(metrics_, "GetAlivePeerView");
  RespValue r = control().Command({"HGETALL", keys_.AlivePeers()});
  std::unordered_map<std::string, std::string> view;
  if (!r.is_array()) return view;
  for (size_t i = 0; i + 1 < r.elements.size(); i += 2) {
    view.emplace(r.elements[i].str, r.elements[i + 1].str);
  }
  return view;
}

std::size_t RedisMasterMetadataStore::AliveClientCount() const {
  RespValue r = control().Command({"SCARD", keys_.NodesAlive()});
  return r.type == RespValue::Type::Integer ? static_cast<std::size_t>(r.integer) : 0;
}

std::vector<std::string> RedisMasterMetadataStore::GetClientTags(const std::string& node_id) const {
  RespValue r = control().Command({"HGET", keys_.Node(node_id), "tags"});
  if (r.type != RespValue::Type::String) return {};
  return SplitTags(r.str);
}

// =====================================================================
// External-KV reads. extkv + hit live on the control tag (single-slot Lua).
// =====================================================================

std::vector<NodeMatch> RedisMasterMetadataStore::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit,
    std::chrono::system_clock::time_point now) {
  ScopedStoreOp _op(metrics_, "MatchExternalKv");
  std::vector<NodeMatch> result;
  if (hashes.empty()) return result;

  std::vector<std::string> args;
  args.reserve(4 + hashes.size());
  args.push_back(keys_.Tag());
  args.push_back(count_as_hit ? "1" : "0");
  args.push_back(std::to_string(ToEpochMs(now)));
  args.push_back(std::to_string(hashes.size()));
  for (const auto& h : hashes) args.push_back(h);
  RespValue r = control().Eval(redis::kMatchExternalKvLua, {keys_.NodesAlive()}, args);
  if (r.is_error()) throw std::runtime_error("[RedisStore] MatchExternalKv: " + r.str);
  if (!r.is_array()) return result;

  // Reply: array of { hash, flat_hgetall[node, mask, node, mask, ...] }.
  // Group by node, decoding each node's tier bitmask (bit == 1<<TierType) into
  // the tiers that hold the hash — a hash mirrored across tiers lands in several
  // buckets (MatchedHashCount dedupes it), matching the in-memory backend.
  std::unordered_map<std::string, std::map<TierType, std::vector<std::string>>> acc;
  for (const auto& entry : r.elements) {
    if (!entry.is_array() || entry.elements.size() < 2) continue;
    const std::string& hash = entry.elements[0].str;
    const RespValue& flat = entry.elements[1];
    if (!flat.is_array()) continue;
    for (size_t i = 0; i + 1 < flat.elements.size(); i += 2) {
      const std::string& node = flat.elements[i].str;
      long long mask = 0;
      try {
        mask = std::stoll(flat.elements[i + 1].str);
      } catch (...) {
        continue;
      }
      auto& by_tier = acc[node];
      for (int t = 0; t < 16; ++t) {
        if ((mask >> t) & 1) by_tier[static_cast<TierType>(t)].push_back(hash);
      }
    }
  }
  result.reserve(acc.size());
  for (auto& [node_id, by_tier] : acc) {
    NodeMatch m;
    m.node_id = node_id;
    m.hashes_by_tier = std::move(by_tier);
    result.push_back(std::move(m));
  }
  return result;
}

std::vector<ExternalKvHitCountEntry> RedisMasterMetadataStore::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes) const {
  std::vector<ExternalKvHitCountEntry> out;
  if (hashes.empty()) return out;
  std::vector<std::string> args;
  args.reserve(2 + hashes.size());
  args.push_back(keys_.Tag());
  args.push_back(std::to_string(hashes.size()));
  for (const auto& h : hashes) args.push_back(h);
  RespValue r = control().Eval(redis::kGetHitCountsLua, {keys_.NodesAlive()}, args);
  if (r.is_error()) throw std::runtime_error("[RedisStore] GetExternalKvHitCounts: " + r.str);
  if (!r.is_array()) return out;
  out.reserve(r.elements.size());
  for (const auto& entry : r.elements) {
    if (!entry.is_array() || entry.elements.size() < 2) continue;
    ExternalKvHitCountEntry e;
    e.hash = entry.elements[0].str;
    try {
      e.hit_count_total = std::stoull(entry.elements[1].str);
    } catch (...) {
      continue;
    }
    out.push_back(std::move(e));
  }
  return out;
}

std::size_t RedisMasterMetadataStore::GetExternalKvCount(const std::string& node_id) const {
  // O(1) via the per-node reverse index — the in-memory backend scans its whole
  // map here only because it lacks a by-node index; Redis has one.
  RespValue r = control().Command({"SCARD", keys_.ExtKvNode(node_id)});
  return r.type == RespValue::Type::Integer ? static_cast<std::size_t>(r.integer) : 0;
}

}  // namespace mori::umbp
