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

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/in_memory_master_metadata_store.h"

#ifdef USE_REDIS_BACKEND
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
    // Block-keyspace shards. Default 1 = legacy single-tag layout (no change).
    // Set >1 to spread block lookups across slots (see KeySchema); the win
    // shows up on a threaded store (Dragonfly) or a sharded/cluster deployment.
    constexpr int kMaxBlockShards = 4096;
    int block_shards = GetEnvInt("UMBP_REDIS_BLOCK_SHARDS", 1);
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
    const std::string shard_uris = GetEnvStr("UMBP_REDIS_SHARD_URIS", "");
    if (!shard_uris.empty()) {
      size_t pos = 0;
      while (pos <= shard_uris.size()) {
        const size_t comma = shard_uris.find(',', pos);
        const size_t end = (comma == std::string::npos) ? shard_uris.size() : comma;
        std::string u = shard_uris.substr(pos, end - pos);
        if (!u.empty()) cfg.shard_uris.push_back(u);
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }

    // Mode selection + validation. single-endpoint (UMBP_REDIS_URI),
    // multi-endpoint (UMBP_REDIS_SHARD_URIS) and Redis Cluster
    // (UMBP_REDIS_CLUSTER) are mutually exclusive. Cluster mode is on the
    // roadmap but not wired into the store yet, so recognize the flag and fail
    // clearly instead of silently falling back to single-endpoint.
    const bool cluster = GetEnvBool("UMBP_REDIS_CLUSTER", false);
    if (cluster && cfg.shard_uris.size() > 1) {
      throw std::runtime_error(
          "UMBP_REDIS_CLUSTER=1 and UMBP_REDIS_SHARD_URIS are mutually exclusive; set only one");
    }
    if (cluster) {
      throw std::runtime_error(
          "UMBP_REDIS_CLUSTER=1 is not supported yet (Redis Cluster mode is on the roadmap); "
          "use single-endpoint (UMBP_REDIS_URI) or multi-endpoint (UMBP_REDIS_SHARD_URIS)");
    }

    if (cfg.shard_uris.size() > 1) {
      MORI_UMBP_INFO("[MetadataStore] backend=redis namespace={} endpoints={} (multi-endpoint)",
                     cfg.namespace_id, cfg.shard_uris.size());
    } else {
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
