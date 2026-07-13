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
#include "umbp/distributed/master/master_metadata_store_factory.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/in_memory_master_metadata_store.h"

#ifdef USE_REDIS_BACKEND
#include "umbp/distributed/master/redis/resp_cluster_client.h"
#include "umbp/distributed/master/redis_master_metadata_store.h"
#endif

namespace mori::umbp {

namespace {

std::string GetEnvStr(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string(v) : def;
}

int GetEnvInt(const char* key, int def) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return def;
  char* end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || n <= 0) {
    MORI_UMBP_WARN("[MetadataStore] ignoring invalid {}='{}'", key, v);
    return def;
  }
  return static_cast<int>(n);
}

bool GetEnvBool(const char* key, bool def) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return def;
  const std::string s(v);
  if (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on") return true;
  if (s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "off") return false;
  MORI_UMBP_WARN("[MetadataStore] ignoring invalid {}='{}', using default {}", key, v, def);
  return def;
}

// Split a comma-separated list, dropping empty entries.
std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    const size_t end = (comma == std::string::npos) ? s.size() : comma;
    std::string tok = s.substr(pos, end - pos);
    if (!tok.empty()) out.push_back(tok);
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

#ifdef USE_REDIS_BACKEND
// Best-effort probe: does the single-endpoint server report itself as Dragonfly?
// (`INFO server` carries "dragonfly_version" on Dragonfly; Redis/Valkey do not.)
// Only used to nudge operators to shard the block keyspace so Dragonfly's
// proactor threads are actually used; any error just returns false (the real
// readiness probe below surfaces an outage).
bool ServerLooksLikeDragonfly(const std::string& uri, const std::string& password,
                              int connect_timeout_ms, int socket_timeout_ms) {
  try {
    redis::RespClient::Options o;
    o.uri = uri;
    o.password = password;
    o.connect_timeout_ms = connect_timeout_ms;
    o.socket_timeout_ms = socket_timeout_ms;
    o.pool_size = 1;
    redis::RespClient client(std::move(o));
    const redis::RespValue r = client.Command({"INFO", "server"});
    return r.str.find("dragonfly") != std::string::npos;
  } catch (const std::exception&) {
    return false;
  }
}
#endif

}  // namespace

std::unique_ptr<IMasterMetadataStore> MakeMasterMetadataStore() {
  const std::string backend = GetEnvStr("UMBP_METADATA_BACKEND", "inmemory");

  if (backend == "redis") {
#ifdef USE_REDIS_BACKEND
    RedisMasterMetadataStore::Config cfg;
    cfg.uri = GetEnvStr("UMBP_REDIS_URI", "tcp://127.0.0.1:6379");
    cfg.namespace_id = GetEnvStr("UMBP_REDIS_NAMESPACE", "default");
    cfg.password = GetEnvStr("UMBP_REDIS_PASSWORD", "");
    cfg.connect_timeout_ms = GetEnvInt("UMBP_REDIS_CONNECT_TIMEOUT_MS", 1000);
    cfg.socket_timeout_ms = GetEnvInt("UMBP_REDIS_SOCKET_TIMEOUT_MS", 1000);
    // Cap the default pool: a single-threaded Redis serializes every command,
    // so a pool sized to a big host's CPU count (e.g. 448 on 224 cores) only
    // deepens the queue and pushes tail latency past the socket timeout. 32 is
    // a healthy default; override with UMBP_REDIS_POOL_SIZE for sharded/cluster
    // deployments.
    unsigned hw = std::thread::hardware_concurrency();
    const int default_pool = static_cast<int>(std::min(32u, std::max(4u, hw ? hw * 2u : 8u)));
    cfg.pool_size = static_cast<std::size_t>(GetEnvInt("UMBP_REDIS_POOL_SIZE", default_pool));
    // Redis Cluster mode (UMBP_REDIS_CLUSTER=1): one redis-plus-plus client
    // routes by hash-tag slot with MOVED/ASK + master-failover handled for us.
    // Read early because it changes the block-shard default.
    const bool cluster = GetEnvBool("UMBP_REDIS_CLUSTER", false);

    // Block-keyspace shards. Default 1 = legacy single-tag layout (no change) for
    // single/multi. In cluster mode, if left unset it is auto-sized to ~2x the
    // discovered master count (below) so a RouteGet batch spreads across nodes
    // without over-splitting into too many per-shard scripts; 16 is only the
    // fallback if discovery fails. Override explicitly with UMBP_REDIS_BLOCK_SHARDS.
    const char* bs_env = std::getenv("UMBP_REDIS_BLOCK_SHARDS");
    const bool bs_explicit = (bs_env != nullptr && *bs_env != '\0');
    constexpr int kMaxBlockShards = 4096;
    int block_shards = GetEnvInt("UMBP_REDIS_BLOCK_SHARDS", cluster ? 16 : 1);
    if (block_shards > kMaxBlockShards) {
      MORI_UMBP_WARN("[MetadataStore] clamping UMBP_REDIS_BLOCK_SHARDS={} to max {}", block_shards,
                     kMaxBlockShards);
      block_shards = kMaxBlockShards;
    }
    cfg.block_shards = static_cast<std::size_t>(block_shards);
    // Multi-endpoint mode: comma-separated Redis URIs, one instance per block
    // shard, so their scripts run on independent server threads (the way past a
    // single instance's single-thread ceiling). When set, it supersedes the
    // single-endpoint block_shards knob.
    cfg.shard_uris = SplitCsv(GetEnvStr("UMBP_REDIS_SHARD_URIS", ""));

    // single-endpoint / multi-endpoint / cluster are mutually exclusive.
    if (cluster && cfg.shard_uris.size() > 1) {
      throw std::runtime_error(
          "UMBP_REDIS_CLUSTER=1 and UMBP_REDIS_SHARD_URIS are mutually exclusive; set only one");
    }

    if (cluster) {
      cfg.cluster = true;
      // Seeds: the UMBP_REDIS_URI comma list (any reachable node bootstraps the
      // rest via CLUSTER SLOTS).
      cfg.cluster_seeds = SplitCsv(GetEnvStr("UMBP_REDIS_URI", "tcp://127.0.0.1:6379"));
      // Balanced placement (unless the operator set UMBP_REDIS_BLOCK_SHARDS):
      // read CLUSTER SLOTS and put exactly one block-shard tag on each master
      // (a tag whose CRC16 slot that node owns), so a RouteGet batch spreads
      // evenly — one EVALSHA per node — instead of piling onto whichever node the
      // formulaic tags happen to hash to. Measured to lift cluster throughput
      // from ~single-node to ~85% of multi-endpoint. Falls back to formulaic
      // shards if discovery / tag search fails (readiness probe surfaces a real
      // outage).
      if (!bs_explicit) {
        try {
          redis::RespClusterClient::Options opts;
          opts.seeds = cfg.cluster_seeds;
          opts.password = cfg.password;
          opts.connect_timeout_ms = cfg.connect_timeout_ms;
          opts.socket_timeout_ms = cfg.socket_timeout_ms;
          opts.pool_size = cfg.pool_size;
          const auto ranges = redis::RespClusterClient::DiscoverMasterSlotRanges(opts);
          std::vector<std::string> tags;
          tags.reserve(ranges.size());
          for (const auto& node_ranges : ranges) {
            std::string tag;
            if (redis::FindTagForRanges(cfg.namespace_id, node_ranges, &tag)) tags.push_back(tag);
          }
          if (!tags.empty() && tags.size() == ranges.size()) {
            cfg.cluster_block_tags = tags;
            cfg.block_shards = tags.size();  // one balanced tag per master
            MORI_UMBP_INFO("[MetadataStore] cluster balanced placement: {} tags, one per master",
                           tags.size());
          } else {
            const std::size_t fallback = std::min<std::size_t>(
                kMaxBlockShards, std::max<std::size_t>(1, 2 * std::max<std::size_t>(1, ranges.size())));
            cfg.block_shards = fallback;
            MORI_UMBP_WARN(
                "[MetadataStore] cluster balanced tag search matched {}/{} masters; "
                "using formulaic block_shards={}",
                tags.size(), ranges.size(), fallback);
          }
        } catch (const std::exception& e) {
          MORI_UMBP_WARN(
              "[MetadataStore] cluster topology discovery failed ({}); using block_shards={}",
              e.what(), cfg.block_shards);
        }
      }
      MORI_UMBP_INFO("[MetadataStore] backend=redis namespace={} seeds={} block_shards={} (cluster)",
                     cfg.namespace_id, cfg.cluster_seeds.size(), cfg.block_shards);
    } else if (cfg.shard_uris.size() > 1) {
      MORI_UMBP_INFO("[MetadataStore] backend=redis namespace={} endpoints={} (multi-endpoint)",
                     cfg.namespace_id, cfg.shard_uris.size());
    } else {
      // Single-endpoint. Dragonfly is multi-threaded, but with the default
      // block_shards=1 every block key shares one hash tag => one proactor
      // thread, so the RouteGet hot path does not scale. We do NOT silently bump
      // the default (block_shards is fixed for a deployment's lifetime — changing
      // it strands existing block keys), but nudge the operator to set it.
      if (!bs_explicit && ServerLooksLikeDragonfly(cfg.uri, cfg.password, cfg.connect_timeout_ms,
                                                   cfg.socket_timeout_ms)) {
        MORI_UMBP_WARN(
            "[MetadataStore] Dragonfly at {} with default UMBP_REDIS_BLOCK_SHARDS=1: block reads "
            "run on a single proactor thread. Set UMBP_REDIS_BLOCK_SHARDS to ~the Dragonfly "
            "--proactor_threads count (e.g. 8) to parallelize the RouteGet hot path. NOTE: this "
            "value is fixed for the deployment's lifetime — pick one and keep it (changing it "
            "strands existing block keys).",
            cfg.uri);
      }
      MORI_UMBP_INFO("[MetadataStore] backend=redis uri={} namespace={} block_shards={}", cfg.uri,
                     cfg.namespace_id, cfg.block_shards);
    }

    auto store = std::make_unique<RedisMasterMetadataStore>(cfg);

    // Startup readiness probe. Pinging every configured endpoint turns a
    // misconfigured / unreachable store into a clear, immediate failure instead
    // of a master that starts "healthy" and then returns UNAVAILABLE on every
    // RPC. Gated by UMBP_REDIS_REQUIRED (default true): set 0 to start in a
    // degraded state (e.g. when some shards are expected to come up later) and
    // rely on the runtime reconnect path.
    const std::string where = cfg.shard_uris.size() > 1
                                  ? ("endpoints=" + std::to_string(cfg.shard_uris.size()))
                                  : cfg.uri;
    if (store->Ping()) {
      MORI_UMBP_INFO("[MetadataStore] backend=redis readiness probe OK ({})", where);
    } else if (GetEnvBool("UMBP_REDIS_REQUIRED", true)) {
      throw std::runtime_error(
          "backend=redis but the store is unreachable at " + where +
          " (readiness probe failed). Fix the store/connection, or set UMBP_REDIS_REQUIRED=0 to "
          "start in a degraded state.");
    } else {
      MORI_UMBP_WARN(
          "[MetadataStore] backend=redis store unreachable at {} at startup; starting degraded "
          "(UMBP_REDIS_REQUIRED=0) — RPCs return UNAVAILABLE until it recovers.",
          where);
    }
    return store;
#else
    throw std::runtime_error(
        "UMBP_METADATA_BACKEND=redis but the Redis backend was not compiled in; "
        "rebuild with -DUSE_REDIS_BACKEND=ON");
#endif
  }

  if (backend != "inmemory") {
    MORI_UMBP_WARN("[MetadataStore] unknown UMBP_METADATA_BACKEND='{}', using inmemory", backend);
  }
  MORI_UMBP_INFO("[MetadataStore] backend=inmemory");
  return std::make_unique<InMemoryMasterMetadataStore>();
}

}  // namespace mori::umbp
