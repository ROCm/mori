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
    MORI_UMBP_INFO("[MetadataStore] backend=redis uri={} namespace={} block_shards={}", cfg.uri,
                   cfg.namespace_id, cfg.block_shards);
    return std::make_unique<RedisMasterMetadataStore>(cfg);
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
