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

#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/redis/lua_scripts.h"

namespace mori::umbp {

namespace {

using redis::RespValue;

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

// A batch's block keys bucketed by shard, plus the mapping to scatter each
// shard's reply back to the caller's original key order. Only shards the batch
// actually touches get a group, so a single-shard batch is one group.
struct ShardedBatch {
  std::vector<std::vector<std::string>> keys_by_shard;   // group -> block keys
  std::vector<std::vector<size_t>> orig_index_by_shard;  // group -> caller indices
};

// Bucket `user_keys` by shard, composing the full block key for each and
// remembering each key's original position for the scatter step.
ShardedBatch GroupKeysByShard(const redis::KeySchema& schema,
                              const std::vector<std::string>& user_keys) {
  ShardedBatch batch;
  // shard index -> group slot, lazily assigned so we only build touched shards.
  std::vector<int> group_of_shard(schema.NumShards(), -1);
  for (size_t i = 0; i < user_keys.size(); ++i) {
    const size_t shard = schema.ShardOf(user_keys[i]);
    int& group = group_of_shard[shard];
    if (group < 0) {
      group = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back();
      batch.orig_index_by_shard.emplace_back();
    }
    batch.keys_by_shard[group].push_back(schema.Block(user_keys[i]));
    batch.orig_index_by_shard[group].push_back(i);
  }
  return batch;
}

// Walk every shard reply (parallel to batch.keys_by_shard) and invoke
// on_key(original_index, reply_element_for_that_key). Throws if any shard's
// script returned an error.
template <typename Fn>
void ForEachShardReply(const ShardedBatch& batch, const std::vector<RespValue>& replies,
                       const char* method, Fn&& on_key) {
  for (size_t group = 0; group < batch.keys_by_shard.size() && group < replies.size(); ++group) {
    const RespValue& reply = replies[group];
    if (reply.is_error()) {
      throw std::runtime_error(std::string("[RedisStore] ") + method + ": " + reply.str);
    }
    if (!reply.is_array()) continue;
    const auto& orig_indices = batch.orig_index_by_shard[group];
    for (size_t j = 0; j < orig_indices.size() && j < reply.elements.size(); ++j) {
      on_key(orig_indices[j], reply.elements[j]);
    }
  }
}

}  // namespace

RedisMasterMetadataStore::RedisMasterMetadataStore(const Config& config)
    : keys_(config.namespace_id, config.block_shards) {
  redis::RespClient::Options opts;
  opts.uri = config.uri;
  opts.password = config.password;
  opts.connect_timeout_ms = config.connect_timeout_ms;
  opts.socket_timeout_ms = config.socket_timeout_ms;
  opts.pool_size = config.pool_size;
  client_ = std::make_unique<redis::RespClient>(std::move(opts));
  MORI_UMBP_INFO("[RedisStore] backend at {} namespace={} pool={} block_shards={}", config.uri,
                 config.namespace_id, config.pool_size, keys_.NumShards());
}

// =====================================================================
// Cross-store writes
// =====================================================================

bool RedisMasterMetadataStore::RegisterClient(const ClientRegistration& registration,
                                              std::chrono::system_clock::time_point now,
                                              std::chrono::system_clock::duration stale_after) {
  const std::string engine(registration.engine_desc_bytes.begin(),
                           registration.engine_desc_bytes.end());
  RespValue r = client_->Eval(
      redis::kRegisterClientLua, {keys_.Node(registration.node_id)},
      {keys_.Tag(), registration.node_id, std::to_string(ToEpochMs(now)),
       std::to_string(ToMs(stale_after)), registration.node_address, registration.peer_address,
       EncodeCaps(registration.tier_capacities), engine, JoinTags(registration.tags)});
  if (r.is_error()) throw std::runtime_error("[RedisStore] RegisterClient: " + r.str);
  return r.integer == 1;
}

void RedisMasterMetadataStore::UnregisterClient(const std::string& node_id) {
  RespValue r =
      client_->Eval(redis::kUnregisterClientLua, {keys_.Node(node_id)}, {keys_.Tag(), node_id});
  if (r.is_error()) throw std::runtime_error("[RedisStore] UnregisterClient: " + r.str);
}

HeartbeatResult RedisMasterMetadataStore::ApplyHeartbeat(
    const std::string& node_id, uint64_t seq, std::chrono::system_clock::time_point now,
    const std::map<TierType, TierCapacity>& caps, const std::vector<KvEvent>& events,
    bool is_full_sync) {
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

  RespValue r = client_->Eval(redis::kApplyHeartbeatLua, {keys_.Node(node_id)}, args);
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
  return HeartbeatResult{HeartbeatResult::APPLIED, acked};
}

std::vector<std::string> RedisMasterMetadataStore::ExpireStaleClients(
    std::chrono::system_clock::time_point cutoff) {
  RespValue r =
      client_->Eval(redis::kExpireStaleLua, {}, {keys_.Tag(), std::to_string(ToEpochMs(cutoff))});
  if (r.is_error()) throw std::runtime_error("[RedisStore] ExpireStaleClients: " + r.str);
  std::vector<std::string> dead;
  if (r.is_array()) {
    dead.reserve(r.elements.size());
    for (const auto& e : r.elements) dead.push_back(e.str);
  }
  return dead;
}

// =====================================================================
// External-KV writes — Phase 2 (not exercised by the hot-path benchmark).
// GarbageCollectHits is a safe no-op so the master's hit-GC thread never
// throws while the external-KV hit path is unimplemented.
// =====================================================================

bool RedisMasterMetadataStore::RegisterExternalKvIfAlive(const std::string&,
                                                         const std::vector<std::string>&,
                                                         TierType) {
  throw std::logic_error(
      "RedisMasterMetadataStore::RegisterExternalKvIfAlive unimplemented (phase 1)");
}

void RedisMasterMetadataStore::UnregisterExternalKv(const std::string&,
                                                    const std::vector<std::string>&, TierType) {
  throw std::logic_error("RedisMasterMetadataStore::UnregisterExternalKv unimplemented (phase 1)");
}

void RedisMasterMetadataStore::UnregisterExternalKvByTier(const std::string&, TierType) {
  throw std::logic_error(
      "RedisMasterMetadataStore::UnregisterExternalKvByTier unimplemented (phase 1)");
}

void RedisMasterMetadataStore::UnregisterExternalKvByNode(const std::string&) {
  throw std::logic_error(
      "RedisMasterMetadataStore::UnregisterExternalKvByNode unimplemented (phase 1)");
}

std::size_t RedisMasterMetadataStore::GarbageCollectHits(std::chrono::system_clock::time_point) {
  // Phase 1: external-KV hit counts are not written, so there is nothing to GC.
  // Implemented as a safe no-op (rather than throwing) because the master's
  // hit-index GC thread calls this on every tick.
  return 0;
}

// =====================================================================
// Block reads
// =====================================================================

std::vector<Location> RedisMasterMetadataStore::LookupBlock(const std::string& key) const {
  RespValue r = client_->Command({"HGETALL", keys_.Block(key)});
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

  std::vector<std::string> args;
  args.reserve(3 + exclude_nodes.size());
  args.push_back(std::to_string(ToEpochMs(now)));
  args.push_back(std::to_string(ToMs(lease_duration)));
  args.push_back(std::to_string(exclude_nodes.size()));
  for (const auto& n : exclude_nodes) args.push_back(n);

  // Fan out one single-slot route_get_batch per shard, in one pipeline; each
  // shard's reply is a per-key array of flat [node, size, tier, ...] triplets.
  const ShardedBatch batch = GroupKeysByShard(keys_, keys);
  const std::vector<RespValue> replies =
      client_->EvalPipeline(redis::kRouteGetBatchLua, batch.keys_by_shard, args);

  ForEachShardReply(batch, replies, "BatchLookupBlockForRouteGet",
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

  const ShardedBatch batch = GroupKeysByShard(keys_, keys);
  const std::vector<RespValue> replies =
      client_->EvalPipeline(redis::kExistsBatchLua, batch.keys_by_shard, {});

  ForEachShardReply(
      batch, replies, "BatchExistsBlock",
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
  RespValue r = client_->Command({"HGETALL", keys_.Node(node_id)});
  if (!r.is_array() || r.elements.empty()) return std::nullopt;
  return DecodeRecord(node_id, FlatToMap(r));
}

bool RedisMasterMetadataStore::IsClientAlive(const std::string& node_id) const {
  RespValue r = client_->Command({"HGET", keys_.Node(node_id), "status"});
  return r.type == RespValue::Type::String && r.str == "1";
}

std::optional<std::string> RedisMasterMetadataStore::GetPeerAddress(
    const std::string& node_id) const {
  RespValue r = client_->Command({"HGET", keys_.Node(node_id), "peer"});
  if (r.is_nil()) return std::nullopt;
  return r.str;
}

std::vector<ClientRecord> RedisMasterMetadataStore::ListAliveClients() const {
  RespValue r = client_->Eval(redis::kListAliveLua, {}, {keys_.Tag()});
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
  RespValue r = client_->Command({"HGETALL", keys_.AlivePeers()});
  std::unordered_map<std::string, std::string> view;
  if (!r.is_array()) return view;
  for (size_t i = 0; i + 1 < r.elements.size(); i += 2) {
    view.emplace(r.elements[i].str, r.elements[i + 1].str);
  }
  return view;
}

std::size_t RedisMasterMetadataStore::AliveClientCount() const {
  RespValue r = client_->Command({"SCARD", keys_.NodesAlive()});
  return r.type == RespValue::Type::Integer ? static_cast<std::size_t>(r.integer) : 0;
}

std::vector<std::string> RedisMasterMetadataStore::GetClientTags(const std::string& node_id) const {
  RespValue r = client_->Command({"HGET", keys_.Node(node_id), "tags"});
  if (r.type != RespValue::Type::String) return {};
  return SplitTags(r.str);
}

// =====================================================================
// External-KV reads — Phase 2.
// =====================================================================

std::vector<NodeMatch> RedisMasterMetadataStore::MatchExternalKv(
    const std::vector<std::string>&, bool, std::chrono::system_clock::time_point) {
  throw std::logic_error("RedisMasterMetadataStore::MatchExternalKv unimplemented (phase 1)");
}

std::vector<ExternalKvHitCountEntry> RedisMasterMetadataStore::GetExternalKvHitCounts(
    const std::vector<std::string>&) const {
  throw std::logic_error(
      "RedisMasterMetadataStore::GetExternalKvHitCounts unimplemented (phase 1)");
}

std::size_t RedisMasterMetadataStore::GetExternalKvCount(const std::string&) const {
  throw std::logic_error("RedisMasterMetadataStore::GetExternalKvCount unimplemented (phase 1)");
}

}  // namespace mori::umbp
