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

// RespClusterClient — the IRespClient implementation for Redis Cluster, built on
// redis-plus-plus (sw::redis::RedisCluster).
//
// It routes each command / script to the node owning the relevant hash-tag slot
// and delegates MOVED / ASK redirection, slot-map refresh, and master-failover
// reconnect to redis-plus-plus (its RedisCluster::command redirection loop). The
// store programs against IRespClient, so single / multi-endpoint (hiredis
// RespClient) and cluster (this class) share the same hot-path logic.
//
// Every scripted call must carry at least one key (KEYS[0]) so we can route it;
// all keys touched by one script share a hash tag (single slot), which is what
// keeps the script cross-key-atomic and cluster-legal.
//
// redis-plus-plus (and hiredis) are confined to resp_cluster_client.cpp; this
// header forward-declares RedisCluster so includers stay library-agnostic.

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/master/redis/resp_value.h"

namespace sw::redis {
class RedisCluster;
}

namespace mori::umbp::redis {

class ThreadPool;

class RespClusterClient : public IRespClient {
 public:
  struct Options {
    // Cluster seeds, e.g. {"tcp://127.0.0.1:7000", ...}. Only one reachable seed
    // is needed; redis-plus-plus discovers the rest via CLUSTER SLOTS. We try
    // seeds in order until one connects.
    std::vector<std::string> seeds;
    std::string password;
    int connect_timeout_ms = 1000;
    int socket_timeout_ms = 1000;
    std::size_t pool_size = 8;
  };

  explicit RespClusterClient(Options options);
  ~RespClusterClient() override;

  RespClusterClient(const RespClusterClient&) = delete;
  RespClusterClient& operator=(const RespClusterClient&) = delete;

  // Connect to the cluster and return how many master nodes it has (from
  // CLUSTER SLOTS). Used to auto-size the block-shard count (~2x masters) so a
  // RouteGet batch spreads across nodes without over-splitting into too many
  // per-shard scripts. Throws RespError if no seed is reachable.
  static std::size_t DiscoverMasterCount(const Options& options);

  // Single command; routed by the key at args[1] (all store commands put the key
  // there: HGETALL/HGET/SCARD <key> ...). Returns the reply (Error value on a
  // server error), throws RespError on a transport/redirection failure.
  RespValue Command(const std::vector<std::string>& args) override;

  // EVALSHA routed by keys[0]'s slot, with a transparent SCRIPT LOAD + NOSCRIPT
  // retry (a newly-promoted replica or resharded node may not have the script
  // cached). keys must be non-empty. The SHA is loaded once and cached.
  RespValue Eval(const std::string& script, const std::vector<std::string>& keys,
                 const std::vector<std::string>& args) override;

  // Run `script` over several single-slot KEYS groups, each routed by its own
  // keys[0]. Groups are issued concurrently through a small worker pool (each is
  // an independent redirection-aware call), so a batch spanning several nodes is
  // ~one round trip per node rather than N sequential ones.
  std::vector<RespValue> EvalPipeline(
      const std::string& script, const std::vector<std::vector<std::string>>& keys_per_call,
      const std::vector<std::string>& shared_args) override;

  bool Ping() override;

 private:
  // Send `argv` to the node owning `routing_key`'s slot via redis-plus-plus's
  // redirection loop (handles MOVED/ASK/failover), decode the reply. Server
  // errors become an Error RespValue; transport failures throw RespError.
  RespValue RunArgvRouted(const std::vector<std::string>& argv, const std::string& routing_key);

  // EVALSHA of `script` routed by keys[0], reloading the script on the target
  // node and retrying once on NOSCRIPT. keys must be non-empty.
  RespValue EvalShaRouted(const std::string& script, const std::vector<std::string>& keys,
                          const std::vector<std::string>& args);

  // SHA1 of `script`, obtained via SCRIPT LOAD once and cached (the digest is
  // content-addressed, so it is the same on every node).
  std::string GetOrLoadSha(const std::string& script);

  Options options_;
  std::unique_ptr<sw::redis::RedisCluster> cluster_;
  std::unique_ptr<ThreadPool> pool_;

  std::mutex sha_mu_;
  std::unordered_map<std::string, std::string> sha_cache_;  // script body -> sha1
};

}  // namespace mori::umbp::redis
