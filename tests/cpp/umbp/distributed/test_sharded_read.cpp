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

// Deterministic unit tests for the block-read fan-out (redis/sharded_read.h)
// against a fake IRespClient — no live Redis. Covers the order-sensitive,
// fault-tolerance-critical parts that the live integration test can only reach
// with a running store: reply scatter back to caller order, multi-instance
// fan-out, and the tolerate-shard-failures degrade-to-miss behaviour.

#include <gtest/gtest.h>

#include <functional>
#include <set>
#include <string>
#include <vector>

#include "umbp/distributed/master/redis/key_schema.h"
#include "umbp/distributed/master/redis/resp_value.h"
#include "umbp/distributed/master/redis/sharded_read.h"
#include "umbp/distributed/master/redis/thread_pool.h"

namespace mori::umbp::redis {
namespace {

// A configurable IRespClient stand-in. Only EvalPipeline is exercised by
// RunShardedRead; the rest satisfy the interface.
class FakeRespClient : public IRespClient {
 public:
  // Per-group reply builder; default echoes each key as a String element, so a
  // group's reply is an Array parallel to its keys (what route_get_batch shapes).
  std::function<RespValue(const std::vector<std::string>&)> reply_fn;
  bool throw_transport = false;  // simulate a dead / unreachable instance
  int eval_pipeline_calls = 0;

  RespValue Command(const std::vector<std::string>&) override { return {}; }
  RespValue Eval(const std::string&, const std::vector<std::string>&,
                 const std::vector<std::string>&) override {
    return {};
  }
  std::vector<RespValue> EvalPipeline(const std::string&,
                                      const std::vector<std::vector<std::string>>& keys_per_call,
                                      const std::vector<std::string>&) override {
    ++eval_pipeline_calls;
    if (throw_transport) throw std::runtime_error("fake: transport down");
    std::vector<RespValue> out;
    out.reserve(keys_per_call.size());
    for (const auto& keys : keys_per_call) {
      out.push_back(reply_fn ? reply_fn(keys) : EchoArray(keys));
    }
    return out;
  }
  bool Ping() override { return true; }

  static RespValue EchoArray(const std::vector<std::string>& keys) {
    RespValue arr;
    arr.type = RespValue::Type::Array;
    for (const auto& k : keys) {
      RespValue e;
      e.type = RespValue::Type::String;
      e.str = k;
      arr.elements.push_back(std::move(e));
    }
    return arr;
  }
  static RespValue ErrorReply(const std::string& msg) {
    RespValue v;
    v.type = RespValue::Type::Error;
    v.str = msg;
    return v;
  }
};

// Build a ShardedBatch by hand for full control over grouping (independent of the
// key hash), mirroring what GroupKeysByShard produces.
ShardedBatch MakeBatch(std::vector<std::vector<std::string>> keys_by_shard,
                       std::vector<std::vector<std::size_t>> orig_index_by_shard,
                       std::vector<std::size_t> shard_of_group) {
  ShardedBatch b;
  b.keys_by_shard = std::move(keys_by_shard);
  b.orig_index_by_shard = std::move(orig_index_by_shard);
  b.shard_of_group = std::move(shard_of_group);
  return b;
}

// Collect on_key(orig_index, String reply) into a result vector for assertions.
std::vector<std::string> RunAndCollect(const ShardedBatch& batch, std::size_t n_keys,
                                       ThreadPool* pool, bool tolerate,
                                       std::function<IRespClient*(std::size_t)> route) {
  std::vector<std::string> out(n_keys);
  RunShardedRead(
      batch, "script", /*shared_args=*/{}, "TestOp", pool, tolerate, /*metrics=*/nullptr,
      [&](std::size_t group) { return route(group); },
      [&](std::size_t orig, const RespValue& v) { out[orig] = v.str; });
  return out;
}

// ---- GroupKeysByShard --------------------------------------------------------

TEST(GroupKeysByShard, BucketsPreserveEveryOriginalIndexOncePerShard) {
  KeySchema schema("ns", /*num_shards=*/4);
  std::vector<std::string> user_keys = {"a", "b", "c", "d", "e", "f", "g", "h"};
  const ShardedBatch batch = GroupKeysByShard(schema, user_keys);

  ASSERT_EQ(batch.keys_by_shard.size(), batch.orig_index_by_shard.size());
  ASSERT_EQ(batch.keys_by_shard.size(), batch.shard_of_group.size());

  std::set<std::size_t> seen;
  for (std::size_t g = 0; g < batch.keys_by_shard.size(); ++g) {
    ASSERT_EQ(batch.keys_by_shard[g].size(), batch.orig_index_by_shard[g].size());
    for (std::size_t j = 0; j < batch.keys_by_shard[g].size(); ++j) {
      const std::size_t orig = batch.orig_index_by_shard[g][j];
      // Every original index appears exactly once across all groups.
      EXPECT_TRUE(seen.insert(orig).second) << "duplicate orig index " << orig;
      // The composed block key matches KeySchema, and its shard matches the group.
      EXPECT_EQ(batch.keys_by_shard[g][j], schema.Block(user_keys[orig]));
      EXPECT_EQ(schema.ShardOf(user_keys[orig]), batch.shard_of_group[g]);
    }
  }
  EXPECT_EQ(seen.size(), user_keys.size());
}

TEST(GroupKeysByShard, SingleShardYieldsOneGroup) {
  KeySchema schema("ns", /*num_shards=*/1);
  const ShardedBatch batch = GroupKeysByShard(schema, {"a", "b", "c"});
  ASSERT_EQ(batch.keys_by_shard.size(), 1u);
  EXPECT_EQ(batch.keys_by_shard[0].size(), 3u);
  EXPECT_EQ(batch.shard_of_group[0], 0u);
}

// ---- RunShardedRead: scatter -------------------------------------------------

TEST(RunShardedRead, SingleClientScattersRepliesToOriginalOrder) {
  // Two groups, interleaved original indices, one client (inline path, no pool).
  const ShardedBatch batch = MakeBatch(/*keys=*/{{"blk0", "blk2"}, {"blk1", "blk3"}},
                                       /*orig=*/{{0, 2}, {1, 3}}, /*shard=*/{0, 1});
  FakeRespClient client;  // echoes each key
  const auto out = RunAndCollect(batch, 4, /*pool=*/nullptr, /*tolerate=*/false,
                                 [&](std::size_t) { return &client; });
  EXPECT_EQ(out, (std::vector<std::string>{"blk0", "blk1", "blk2", "blk3"}));
  EXPECT_EQ(client.eval_pipeline_calls, 1);  // one client => one pipelined call
}

TEST(RunShardedRead, MultiClientFanOutPreservesOrder) {
  // Groups routed to two different clients; a real pool drives the fan-out.
  const ShardedBatch batch =
      MakeBatch({{"blk0", "blk3"}, {"blk1"}, {"blk2"}}, {{0, 3}, {1}, {2}}, {0, 1, 2});
  FakeRespClient a, b;
  ThreadPool pool(4);
  // group 0 -> a, groups 1,2 -> b.
  const auto out = RunAndCollect(batch, 4, &pool, /*tolerate=*/false,
                                 [&](std::size_t g) -> IRespClient* { return g == 0 ? &a : &b; });
  EXPECT_EQ(out, (std::vector<std::string>{"blk0", "blk1", "blk2", "blk3"}));
  EXPECT_EQ(a.eval_pipeline_calls, 1);
  EXPECT_EQ(b.eval_pipeline_calls, 1);  // b's two groups share one pipeline
}

// ---- RunShardedRead: fault tolerance ----------------------------------------

TEST(RunShardedRead, TolerateDegradesDeadInstanceToMiss) {
  const ShardedBatch batch = MakeBatch({{"blk0"}, {"blk1"}}, {{0}, {1}}, {0, 1});
  FakeRespClient a;  // healthy
  FakeRespClient b;  // dead
  b.throw_transport = true;
  ThreadPool pool(2);
  const auto out = RunAndCollect(batch, 2, &pool, /*tolerate=*/true,
                                 [&](std::size_t g) -> IRespClient* { return g == 0 ? &a : &b; });
  EXPECT_EQ(out[0], "blk0");  // healthy shard delivered
  EXPECT_EQ(out[1], "");      // dead shard left at default (miss), no throw
}

TEST(RunShardedRead, NoToleratePropagatesDeadInstance) {
  const ShardedBatch batch = MakeBatch({{"blk0"}}, {{0}}, {0});
  FakeRespClient dead;
  dead.throw_transport = true;
  EXPECT_THROW(RunAndCollect(batch, 1, /*pool=*/nullptr, /*tolerate=*/false,
                             [&](std::size_t) { return &dead; }),
               std::runtime_error);
}

TEST(RunShardedRead, TolerateSkipsScriptErrorGroupButDeliversRest) {
  const ShardedBatch batch = MakeBatch({{"blk0"}, {"blk1"}}, {{0}, {1}}, {0, 1});
  FakeRespClient client;
  // Group whose key is "blk1" returns a server-side error reply; others echo.
  client.reply_fn = [](const std::vector<std::string>& keys) {
    if (!keys.empty() && keys[0] == "blk1") return FakeRespClient::ErrorReply("boom");
    return FakeRespClient::EchoArray(keys);
  };
  const auto out = RunAndCollect(batch, 2, /*pool=*/nullptr, /*tolerate=*/true,
                                 [&](std::size_t) { return &client; });
  EXPECT_EQ(out[0], "blk0");  // healthy group delivered
  EXPECT_EQ(out[1], "");      // errored group skipped (miss)
}

TEST(RunShardedRead, NoToleratePropagatesScriptError) {
  const ShardedBatch batch = MakeBatch({{"blk0"}}, {{0}}, {0});
  FakeRespClient client;
  client.reply_fn = [](const std::vector<std::string>&) {
    return FakeRespClient::ErrorReply("boom");
  };
  EXPECT_THROW(RunAndCollect(batch, 1, /*pool=*/nullptr, /*tolerate=*/false,
                             [&](std::size_t) { return &client; }),
               std::runtime_error);
}

}  // namespace
}  // namespace mori::umbp::redis
