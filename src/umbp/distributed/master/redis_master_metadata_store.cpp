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
#include <unordered_set>
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

// Dedup a hash list preserving first-seen order. The external-KV read scripts
// used to dedup with a Lua `seen` table; doing it here lets every touched key be
// declared in KEYS[] (no allow-undeclared-keys) and lets a match/hit result be
// scattered back by first-seen position.
std::vector<std::string> DedupPreserveOrder(const std::vector<std::string>& in) {
  std::vector<std::string> out;
  out.reserve(in.size());
  std::unordered_set<std::string> seen;
  seen.reserve(in.size() * 2);
  for (const auto& s : in) {
    if (seen.insert(s).second) out.push_back(s);
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

void RedisMasterMetadataStore::WipeNodeExtKvMulti(const std::string& node_id) {
  // Best-effort per shard: a down shard must not block wiping the reachable ones.
  // The node record is already gone/EXPIRED on the control instance, so any extkv
  // entry lingering on an unreachable shard points at a dead node and is filtered
  // out of matches by GetAlivePeerView until that shard returns. Members of the
  // per-(node, shard) reverse index are full extkv keys, so the script HDELs them
  // directly. Idempotent.
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    try {
      RespValue r = client_for_shard(shard).Eval(redis::kUnregisterExternalKvByNodeLua,
                                                 {keys_.ExtKvNode(node_id, shard)}, {node_id});
      if (r.is_error()) {
        MORI_UMBP_WARN("[RedisStore] WipeNodeExtKv: shard {} script error, skipped: {}", shard,
                       r.str);
      }
    } catch (const std::exception& e) {
      MORI_UMBP_WARN("[RedisStore] WipeNodeExtKv: shard {} unavailable, skipped: {}", shard,
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
  // wipe_extkv=1 wipes the node's external-kv inline in the control/single script
  // (num_shards==1: extkv lives on the control slot). When extkv is sharded it is
  // wiped per shard afterwards (WipeNodeExtKvMulti); the inline step is skipped.
  const std::string wipe_extkv = extkv_sharded() ? "0" : "1";
  if (!split_writes()) {
    RespValue r = control().Eval(redis::kUnregisterClientLua, {keys_.Node(node_id)},
                                 {keys_.Tag(), node_id, wipe_extkv});
    if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterClient: " + r.str);
    if (extkv_sharded()) WipeNodeExtKvMulti(node_id);
    return;
  }
  // Multi-endpoint: control record first (so the router stops routing to it),
  // then wipe its block locations on every shard's instance. A lingering
  // location for the now-gone node is filtered out by GetAlivePeerView, and the
  // wipe is idempotent, so a mid-way failure + retry is safe.
  RespValue r = control().Eval(redis::kUnregisterControlLua, {keys_.Node(node_id)},
                               {keys_.Tag(), node_id, wipe_extkv});
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterClient(control): " + r.str);
  WipeNodeBlocksMulti(node_id);
  if (extkv_sharded()) WipeNodeExtKvMulti(node_id);
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
  const std::string wipe_extkv = extkv_sharded() ? "0" : "1";
  // KEYS[1] = a control-tag key (nodes:alive) to route + fix the slot in cluster
  // mode; the script reads tag from ARGV and touches only control-tag keys.
  RespValue r = control().Eval(script, {keys_.NodesAlive()},
                               {keys_.Tag(), std::to_string(ToEpochMs(cutoff)), wipe_extkv});
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
  // Sharded extkv: wipe each dead node's external-kv per shard (the control step
  // skipped it when wipe_extkv==0).
  if (extkv_sharded()) {
    for (const auto& id : dead) WipeNodeExtKvMulti(id);
  }
  return dead;
}

// =====================================================================
// External-KV writes. The extkv/hit keyspace is sharded by ExtKvShardOf(hash),
// so every op fans out one single-slot script per touched shard (grouped like
// the block read hot path). For num_shards == 1 this collapses onto the control
// slot — one atomic script, byte-identical to the legacy layout. See design
// §4/§5 + the PHASE 2 note in lua_scripts.h.
// =====================================================================

bool RedisMasterMetadataStore::RegisterExternalKvIfAlive(const std::string& node_id,
                                                         const std::vector<std::string>& hashes,
                                                         TierType tier) {
  ScopedStoreOp _op(metrics_, "RegisterExternalKvIfAlive");
  if (hashes.empty()) return IsClientAlive(node_id);
  const std::string tier_str = std::to_string(static_cast<int>(tier));

  if (!extkv_sharded()) {
    // num_shards == 1: alive-gate + write in one atomic script (KEYS declared, no
    // allow-undeclared-keys flag — Dragonfly-single would still run it global, but
    // that path is num_shards>1). extkv reverse index + extkv keys all on shard 0
    // (== control slot).
    std::vector<std::string> keys;
    keys.reserve(2 + hashes.size());
    keys.push_back(keys_.Node(node_id));                          // KEYS[1] alive-check
    keys.push_back(keys_.ExtKvNode(node_id, 0));                  // KEYS[2] reverse index
    for (const auto& h : hashes) keys.push_back(keys_.ExtKv(h));  // KEYS[3..]
    RespValue r = control().Eval(redis::kRegisterExternalKvLua, keys, {node_id, tier_str});
    if (r.is_error()) throw std::runtime_error("[RedisStore] RegisterExternalKvIfAlive: " + r.str);
    return r.integer == 1;
  }

  // Sharded: alive-check on the control instance first, then fan out the write to
  // each touched shard. The check + write are not one atomic step across shards
  // (the node key and the sharded extkv keys live in different slots), a TOCTOU
  // window consistent with the split block-write path — a node that dies between
  // the check and the write leaves extkv entries the unregister/expire cascade
  // reaps. Idempotent, so a retried shard is safe.
  if (!IsClientAlive(node_id)) return false;

  // Group hashes by shard; each group's KEYS = [reverse index] ++ [extkv keys].
  redis::ShardedBatch batch;
  std::vector<int> group_of_shard(keys_.NumShards(), -1);
  for (const auto& h : hashes) {
    const std::size_t shard = keys_.ExtKvShardOf(h);
    int& g = group_of_shard[shard];
    if (g < 0) {
      g = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back(1, keys_.ExtKvNode(node_id, shard));  // KEYS[1]
      batch.orig_index_by_shard.emplace_back();
      batch.shard_of_group.push_back(shard);
    }
    batch.keys_by_shard[g].push_back(keys_.ExtKv(h));
  }
  redis::RunShardedRead(
      batch, redis::kRegisterExternalKvWriteLua, {node_id, tier_str}, "RegisterExternalKvIfAlive",
      fanout_pool_.get(), /*tolerate_shard_failures=*/false, metrics_,
      [&](size_t g) { return &client_for_shard(batch.shard_of_group[g]); },
      [](size_t, const RespValue&) {});
  return true;
}

void RedisMasterMetadataStore::UnregisterExternalKv(const std::string& node_id,
                                                    const std::vector<std::string>& hashes,
                                                    TierType tier) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKv");
  if (hashes.empty()) return;
  const std::string tier_str = std::to_string(static_cast<int>(tier));

  // Group hashes by shard; each group's KEYS = [reverse index] ++ [extkv keys].
  // num_shards == 1 => one group on the control slot (one atomic script).
  redis::ShardedBatch batch;
  std::vector<int> group_of_shard(keys_.NumShards(), -1);
  for (const auto& h : hashes) {
    const std::size_t shard = keys_.ExtKvShardOf(h);
    int& g = group_of_shard[shard];
    if (g < 0) {
      g = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back(1, keys_.ExtKvNode(node_id, shard));
      batch.orig_index_by_shard.emplace_back();
      batch.shard_of_group.push_back(shard);
    }
    batch.keys_by_shard[g].push_back(keys_.ExtKv(h));
  }
  redis::RunShardedRead(
      batch, redis::kUnregisterExternalKvLua, {node_id, tier_str}, "UnregisterExternalKv",
      fanout_pool_.get(), /*tolerate_shard_failures=*/false, metrics_,
      [&](size_t g) { return &client_for_shard(batch.shard_of_group[g]); },
      [](size_t, const RespValue&) {});
}

void RedisMasterMetadataStore::UnregisterExternalKvByTier(const std::string& node_id,
                                                          TierType tier) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKvByTier");
  // Enumerates via the per-(node, shard) reverse index, so it must touch every
  // shard. One script per shard, routed to that shard.
  const std::string tier_str = std::to_string(static_cast<int>(tier));
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    RespValue r =
        client_for_shard(shard).Eval(redis::kUnregisterExternalKvByTierLua,
                                     {keys_.ExtKvNode(node_id, shard)}, {node_id, tier_str});
    if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterExternalKvByTier: " + r.str);
  }
}

void RedisMasterMetadataStore::UnregisterExternalKvByNode(const std::string& node_id) {
  ScopedStoreOp _op(metrics_, "UnregisterExternalKvByNode");
  // Same per-shard wipe the client-record cascades use.
  WipeNodeExtKvMulti(node_id);
}

std::size_t RedisMasterMetadataStore::GarbageCollectHits(
    std::chrono::system_clock::time_point cutoff) {
  ScopedStoreOp _op(metrics_, "GarbageCollectHits");
  // Drop every hit counter whose last_seen < cutoff via each shard's hit index
  // (no SCAN — cluster-routable by the index key). One script per shard; runs on
  // the slow hit-GC timer, not the hot path. num_shards == 1 => one call.
  const std::string cutoff_str = std::to_string(ToEpochMs(cutoff));
  std::size_t dropped = 0;
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    RespValue r = client_for_shard(shard).Eval(redis::kGarbageCollectHitsLua,
                                               {keys_.HitIndex(shard)}, {cutoff_str});
    if (r.is_error()) throw std::runtime_error("[RedisStore] GarbageCollectHits: " + r.str);
    if (r.type == RespValue::Type::Integer) dropped += static_cast<std::size_t>(r.integer);
  }
  return dropped;
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
RedisMasterMetadataStore::EnumerateEvictionCandidates(
    const std::vector<NodeTierKey>& buckets, EvictionOrder order, size_t max_per_bucket,
    std::chrono::system_clock::time_point now) const {
  std::map<NodeTierKey, std::vector<EvictionCandidate>> result;
  if (buckets.empty()) return result;
  ScopedStoreOp _op(metrics_, "EnumerateEvictionCandidates");

  // The read hot path is untouched: eviction reuses the per-node block reverse
  // index + the _lease/_lacc already maintained on each block hash. Candidates
  // are enumerated per shard by kEnumerateEvictionLua and aggregated here; the
  // ordering / per-bucket cap is applied in C++ (an eviction tick is seconds,
  // not a hot path), mirroring the in-memory backend's scan-and-sort.

  // Distinct nodes to scan, and the wanted (node, tier) pairs the script filters
  // on (passed as ARGV pairs, so a node id may contain any byte).
  std::vector<std::string> nodes;
  {
    std::unordered_set<std::string> seen;
    for (const auto& b : buckets) {
      if (seen.insert(b.node_id).second) nodes.push_back(b.node_id);
    }
  }
  std::vector<std::string> shared_args;
  shared_args.reserve(2 + buckets.size() * 2);
  shared_args.push_back(std::to_string(ToEpochMs(now)));
  shared_args.push_back(std::to_string(buckets.size()));
  for (const auto& b : buckets) {
    shared_args.push_back(b.node_id);
    shared_args.push_back(std::to_string(static_cast<int>(b.tier)));
  }

  // Build the per-shard groups. split_writes() (multi-endpoint / cluster) keeps a
  // reverse index per (node, shard) on that shard's instance/slot, so one group
  // per shard routed to that shard. Single mode keeps one control-tag reverse
  // index per node, so one group on the control instance.
  redis::ShardedBatch batch;
  auto add_group = [&](std::size_t shard, std::vector<std::string> node_block_keys) {
    const std::size_t g = batch.keys_by_shard.size();
    batch.orig_index_by_shard.emplace_back();
    for (std::size_t j = 0; j < node_block_keys.size(); ++j) {
      batch.orig_index_by_shard[g].push_back(j);  // unused (node is in each tuple)
    }
    batch.keys_by_shard.push_back(std::move(node_block_keys));
    batch.shard_of_group.push_back(shard);
  };
  if (split_writes()) {
    for (std::size_t s = 0; s < keys_.NumShards(); ++s) {
      std::vector<std::string> ks;
      ks.reserve(nodes.size());
      for (const auto& n : nodes) ks.push_back(keys_.NodeBlocks(n, s));
      add_group(s, std::move(ks));
    }
  } else {
    std::vector<std::string> ks;
    ks.reserve(nodes.size());
    for (const auto& n : nodes) ks.push_back(keys_.NodeBlocks(n));
    add_group(/*shard=*/0, std::move(ks));
  }

  // Strip "<shardtag>:block:" from a redis block key to recover the user key.
  auto user_key_of = [](const std::string& block_key) -> std::string {
    const size_t pos = block_key.find(":block:");
    return pos == std::string::npos ? block_key : block_key.substr(pos + 7);
  };

  redis::RunShardedRead(
      batch, redis::kEnumerateEvictionLua, shared_args, "EnumerateEvictionCandidates",
      fanout_pool_.get(), /*tolerate_shard_failures=*/mode_ != Mode::kSingle, metrics_,
      [&](size_t group) { return &client_for_shard(batch.shard_of_group[group]); },
      [&](size_t /*orig_index*/, const RespValue& cands) {
        if (!cands.is_array()) return;
        // Flat 5-tuples: [block_key, node, tier, size, last_accessed_ms].
        for (size_t j = 0; j + 5 <= cands.elements.size(); j += 5) {
          EvictionCandidate c;
          c.key = user_key_of(cands.elements[j].str);
          c.location.node_id = cands.elements[j + 1].str;
          try {
            c.location.tier = static_cast<TierType>(std::stoi(cands.elements[j + 2].str));
            c.location.size = std::stoull(cands.elements[j + 3].str);
            c.last_accessed_at = FromEpochMs(std::stoll(cands.elements[j + 4].str));
          } catch (...) {
            continue;
          }
          c.size = c.location.size;
          result[NodeTierKey{c.location.node_id, c.location.tier}].push_back(std::move(c));
        }
      });

  // Honor the ordering hint and the per-bucket cap (same policy as in-memory).
  const auto older_first = [](const EvictionCandidate& a, const EvictionCandidate& b) {
    return a.last_accessed_at < b.last_accessed_at;
  };
  for (auto& [ntk, candidates] : result) {
    (void)ntk;
    const bool cap = max_per_bucket > 0 && candidates.size() > max_per_bucket;
    if (order == EvictionOrder::kLeastRecentlyAccessed) {
      if (cap) {
        std::partial_sort(candidates.begin(), candidates.begin() + max_per_bucket, candidates.end(),
                          older_first);
        candidates.resize(max_per_bucket);
      } else {
        std::sort(candidates.begin(), candidates.end(), older_first);
      }
    } else if (cap) {
      candidates.resize(max_per_bucket);
    }
  }
  return result;
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
// External-KV reads. The extkv/hit keyspace is sharded by ExtKvShardOf(hash), so
// a batch fans out one single-slot script per touched shard (grouped + scattered
// by RunShardedRead like the block read hot path). num_shards == 1 collapses to
// one group on the control instance.
// =====================================================================

std::vector<NodeMatch> RedisMasterMetadataStore::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit,
    std::chrono::system_clock::time_point now) {
  ScopedStoreOp _op(metrics_, "MatchExternalKv");
  std::vector<NodeMatch> result;
  if (hashes.empty()) return result;

  // Dedup preserving first-seen order (lets each touched key be declared in
  // KEYS[] — no allow-undeclared-keys — and lets a per-hash reply be scattered
  // back by first-seen position).
  const std::vector<std::string> uniq = DedupPreserveOrder(hashes);

  // Group unique hashes by their extkv shard. Each group's KEYS starts as the k
  // extkv keys (reply is parallel to these, scattered back by orig index); when
  // counting hits, the k hit keys + the shard's hit index are appended so the
  // script can bump the counter in the same single-slot call.
  redis::ShardedBatch batch;
  std::vector<int> group_of_shard(keys_.NumShards(), -1);
  for (std::size_t i = 0; i < uniq.size(); ++i) {
    const std::size_t shard = keys_.ExtKvShardOf(uniq[i]);
    int& g = group_of_shard[shard];
    if (g < 0) {
      g = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back();
      batch.orig_index_by_shard.emplace_back();
      batch.shard_of_group.push_back(shard);
    }
    batch.keys_by_shard[g].push_back(keys_.ExtKv(uniq[i]));
    batch.orig_index_by_shard[g].push_back(i);
  }
  if (count_as_hit) {
    for (std::size_t g = 0; g < batch.keys_by_shard.size(); ++g) {
      auto& ks = batch.keys_by_shard[g];
      const auto& idx = batch.orig_index_by_shard[g];
      const std::size_t kg = idx.size();
      ks.reserve(2 * kg + 1);
      for (std::size_t j = 0; j < kg; ++j) ks.push_back(keys_.Hit(uniq[idx[j]]));
      ks.push_back(keys_.HitIndex(batch.shard_of_group[g]));
    }
  }

  const std::vector<std::string> shared_args = {count_as_hit ? "1" : "0",
                                                std::to_string(ToEpochMs(now))};

  // Group by node, decoding each node's tier bitmask (bit == 1<<TierType) into the
  // tiers that hold the hash — a hash mirrored across tiers lands in several
  // buckets (MatchedHashCount dedupes it), matching the in-memory backend.
  std::unordered_map<std::string, std::map<TierType, std::vector<std::string>>> acc;
  redis::RunShardedRead(
      batch, redis::kMatchExternalKvLua, shared_args, "MatchExternalKv", fanout_pool_.get(),
      /*tolerate_shard_failures=*/mode_ != Mode::kSingle, metrics_,
      [&](size_t g) { return &client_for_shard(batch.shard_of_group[g]); },
      [&](size_t orig_index, const RespValue& flat) {
        if (!flat.is_array()) return;
        const std::string& hash = uniq[orig_index];
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
      });
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
  const std::vector<std::string> uniq = DedupPreserveOrder(hashes);

  // Group hit keys by shard; reply per shard is parallel to that shard's hit
  // keys, scattered back by orig index.
  redis::ShardedBatch batch;
  std::vector<int> group_of_shard(keys_.NumShards(), -1);
  for (std::size_t i = 0; i < uniq.size(); ++i) {
    const std::size_t shard = keys_.ExtKvShardOf(uniq[i]);
    int& g = group_of_shard[shard];
    if (g < 0) {
      g = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back();
      batch.orig_index_by_shard.emplace_back();
      batch.shard_of_group.push_back(shard);
    }
    batch.keys_by_shard[g].push_back(keys_.Hit(uniq[i]));
    batch.orig_index_by_shard[g].push_back(i);
  }

  std::vector<ExternalKvHitCountEntry> scratch(uniq.size());
  std::vector<bool> present(uniq.size(), false);
  redis::RunShardedRead(
      batch, redis::kGetHitCountsLua, {}, "GetExternalKvHitCounts", fanout_pool_.get(),
      /*tolerate_shard_failures=*/mode_ != Mode::kSingle, metrics_,
      [&](size_t g) { return &client_for_shard(batch.shard_of_group[g]); },
      [&](size_t orig_index, const RespValue& c) {
        if (c.type != RespValue::Type::String) return;
        try {
          scratch[orig_index].hit_count_total = std::stoull(c.str);
        } catch (...) {
          return;
        }
        scratch[orig_index].hash = uniq[orig_index];
        present[orig_index] = true;
      });
  out.reserve(uniq.size());
  for (std::size_t i = 0; i < uniq.size(); ++i) {
    if (present[i]) out.push_back(std::move(scratch[i]));
  }
  return out;
}

std::size_t RedisMasterMetadataStore::GetExternalKvCount(const std::string& node_id) const {
  // Sum the per-(node, shard) reverse-index cardinalities. num_shards == 1 => one
  // SCARD on the control slot (byte-identical to the legacy single index).
  std::size_t total = 0;
  for (std::size_t shard = 0; shard < keys_.NumShards(); ++shard) {
    RespValue r = client_for_shard(shard).Command({"SCARD", keys_.ExtKvNode(node_id, shard)});
    if (r.type == RespValue::Type::Integer) total += static_cast<std::size_t>(r.integer);
  }
  return total;
}

}  // namespace mori::umbp
