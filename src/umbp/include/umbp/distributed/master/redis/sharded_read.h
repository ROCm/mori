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

// Sharded batch-read fan-out for the block hot path (RouteGet / Exists).
//
// Split out of redis_master_metadata_store.cpp so the batch bucketing + fan-out +
// fault-tolerance + reply-scatter logic — the trickiest, most order-sensitive
// part of the backend — can be unit-tested directly against a fake IRespClient,
// without a live Redis. The store just supplies the KeySchema, the routing
// (client-per-group), and the per-key decoder.

#pragma once

#include <cstddef>
#include <future>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/redis/key_schema.h"
#include "umbp/distributed/master/redis/resp_value.h"
#include "umbp/distributed/master/redis/thread_pool.h"

namespace mori::umbp::redis {

// A batch's block keys bucketed by shard, plus the mapping to scatter each
// shard's reply back to the caller's original key order. Only shards the batch
// actually touches get a group, so a single-shard batch is one group.
struct ShardedBatch {
  std::vector<std::vector<std::string>> keys_by_shard;        // group -> block keys
  std::vector<std::vector<std::size_t>> orig_index_by_shard;  // group -> caller indices
  std::vector<std::size_t> shard_of_group;                    // group -> shard index
};

// Bucket `user_keys` by shard, composing the full block key for each and
// remembering each key's original position for the scatter step.
inline ShardedBatch GroupKeysByShard(const KeySchema& schema,
                                     const std::vector<std::string>& user_keys) {
  ShardedBatch batch;
  // shard index -> group slot, lazily assigned so we only build touched shards.
  std::vector<int> group_of_shard(schema.NumShards(), -1);
  for (std::size_t i = 0; i < user_keys.size(); ++i) {
    const std::size_t shard = schema.ShardOf(user_keys[i]);
    int& group = group_of_shard[shard];
    if (group < 0) {
      group = static_cast<int>(batch.keys_by_shard.size());
      batch.keys_by_shard.emplace_back();
      batch.orig_index_by_shard.emplace_back();
      batch.shard_of_group.push_back(shard);
    }
    batch.keys_by_shard[group].push_back(schema.Block(user_keys[i]));
    batch.orig_index_by_shard[group].push_back(i);
  }
  return batch;
}

// One instance's slice of a sharded batch: which groups it serves, the block
// keys per group, and (after the call) that instance's replies or an error.
struct ShardCall {
  IRespClient* client = nullptr;
  std::vector<std::size_t> groups;
  std::vector<std::vector<std::string>> keys;
  std::vector<RespValue> replies;
  bool ok = true;
  std::string error;
};

// Run `script` over a sharded batch, routing each shard-group to the client
// returned by client_for_group(group) — one EvalPipeline per distinct client.
// A single-endpoint deployment is one pipelined round trip on the calling
// thread (no pool use). A multi-endpoint one issues each instance's pipeline
// CONCURRENTLY via `pool` (one round trip instead of N — the win on high-RTT
// remote stores), then scatters replies on the calling thread so on_key needs
// no locking.
//
// Fault tolerance (tolerate_shard_failures): when the keyspace is spread across
// multiple nodes/instances (multi-endpoint OR cluster), a single node that is
// down / erroring does NOT fail the whole batch — its keys are simply left
// untouched (the caller's default is a miss: empty locations / exists=false), so
// RouteGet keeps serving keys on the healthy shards. A WARN is logged per failed
// shard. In single-endpoint mode there is nothing to fall back to, so a failure
// propagates (a total outage must surface, not masquerade as misses).
//
// NOTE: this must be decided by the caller (topology), NOT inferred from the
// number of distinct clients — in cluster mode a single cluster client fronts
// every node, so `calls.size()` is always 1 even though the keys are spread
// across nodes and per-node failures SHOULD degrade to misses.
template <typename ClientForGroup, typename Fn>
void RunShardedRead(const ShardedBatch& batch, const std::string& script,
                    const std::vector<std::string>& shared_args, const char* method,
                    ThreadPool* pool, bool tolerate_shard_failures,
                    mori::metrics::MetricsServer* metrics, ClientForGroup client_for_group,
                    Fn&& on_key) {
  // Cold path only: count a shard whose keys we degraded to miss (node down /
  // script error). No-op when no metrics sink is attached.
  auto count_degraded = [&] {
    if (metrics == nullptr) return;
    metrics->addCounter("mori_umbp_redis_degraded_shard_total",
                        "Shard reads degraded to miss because a node was down / erroring",
                        {{"backend", "redis"}, {"method", method}});
  };
  // Bucket group indices by the client that serves them.
  std::unordered_map<IRespClient*, std::vector<std::size_t>> groups_by_client;
  for (std::size_t g = 0; g < batch.keys_by_shard.size(); ++g) {
    groups_by_client[client_for_group(g)].push_back(g);
  }

  std::vector<ShardCall> calls;
  calls.reserve(groups_by_client.size());
  for (auto& [client, group_indices] : groups_by_client) {
    ShardCall c;
    c.client = client;
    c.groups = std::move(group_indices);
    c.keys.reserve(c.groups.size());
    for (std::size_t g : c.groups) c.keys.push_back(batch.keys_by_shard[g]);
    calls.push_back(std::move(c));
  }

  // Capture a transport failure into the ShardCall rather than throwing, so one
  // dead instance can be tolerated below instead of aborting the fan-out.
  auto do_eval = [&](ShardCall& c) {
    try {
      c.replies = c.client->EvalPipeline(script, c.keys, shared_args);
    } catch (const std::exception& e) {
      c.ok = false;
      c.error = e.what();
    }
  };

  if (calls.size() <= 1 || pool == nullptr) {
    for (auto& c : calls) do_eval(c);  // single instance / no pool: inline.
  } else {
    // Fan the extra instances out to the pool, run the first inline, then join
    // every future so no task outlives this scope (do_eval never throws).
    std::vector<std::future<void>> futs;
    futs.reserve(calls.size() - 1);
    for (std::size_t i = 1; i < calls.size(); ++i) {
      futs.push_back(pool->Enqueue([&do_eval, &calls, i] { do_eval(calls[i]); }));
    }
    do_eval(calls[0]);
    for (auto& f : futs) f.get();
  }

  const bool tolerate =
      tolerate_shard_failures;  // multi-endpoint/cluster: degrade a down shard to misses.

  // Scatter replies back to caller order (single-threaded; on_key is unlocked).
  for (const ShardCall& c : calls) {
    if (!c.ok) {
      if (!tolerate)
        throw std::runtime_error(std::string("[RedisStore] ") + method + ": " + c.error);
      MORI_UMBP_WARN("[RedisStore] {}: shard instance unavailable, its keys read as miss: {}",
                     method, c.error);
      count_degraded();
      continue;  // leave this shard's keys at the caller's default (miss).
    }
    for (std::size_t k = 0; k < c.groups.size() && k < c.replies.size(); ++k) {
      const RespValue& reply = c.replies[k];
      if (reply.is_error()) {
        if (!tolerate) {
          throw std::runtime_error(std::string("[RedisStore] ") + method + ": " + reply.str);
        }
        MORI_UMBP_WARN("[RedisStore] {}: shard script error, its keys read as miss: {}", method,
                       reply.str);
        count_degraded();
        continue;
      }
      if (!reply.is_array()) continue;
      const auto& orig_indices = batch.orig_index_by_shard[c.groups[k]];
      for (std::size_t j = 0; j < orig_indices.size() && j < reply.elements.size(); ++j) {
        on_key(orig_indices[j], reply.elements[j]);
      }
    }
  }
}

}  // namespace mori::umbp::redis
