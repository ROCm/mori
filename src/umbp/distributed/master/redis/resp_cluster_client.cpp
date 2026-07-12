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
#include "umbp/distributed/master/redis/resp_cluster_client.h"

#include <sw/redis++/redis++.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/redis/resp_client.h"
#include "umbp/distributed/master/redis/thread_pool.h"

namespace mori::umbp::redis {

namespace {

// Parse "tcp://host:port" (or "host:port") into a redis-plus-plus
// ConnectionOptions host/port. Throws RespError on a malformed seed.
void ParseSeed(const std::string& uri, std::string* host, int* port) {
  std::string rest = uri;
  const std::string scheme = "tcp://";
  if (rest.rfind(scheme, 0) == 0) rest = rest.substr(scheme.size());
  const auto slash = rest.find('/');
  if (slash != std::string::npos) rest = rest.substr(0, slash);
  const auto colon = rest.rfind(':');
  if (colon == std::string::npos) {
    *host = rest;
    *port = 6379;
    return;
  }
  *host = rest.substr(0, colon);
  try {
    *port = std::stoi(rest.substr(colon + 1));
  } catch (const std::exception&) {
    throw RespError("RespClusterClient: invalid port in seed '" + uri + "'");
  }
}

}  // namespace

RespClusterClient::RespClusterClient(Options options) : options_(std::move(options)) {
  if (options_.seeds.empty()) {
    throw RespError("RespClusterClient: no cluster seeds configured");
  }
  if (options_.pool_size == 0) options_.pool_size = 1;

  sw::redis::ConnectionPoolOptions pool_opts;
  pool_opts.size = options_.pool_size;

  // Try each seed until one lets us discover the cluster topology (RedisCluster's
  // constructor fetches CLUSTER SLOTS from the seed and throws if it is down).
  std::string last_err;
  for (const auto& seed : options_.seeds) {
    sw::redis::ConnectionOptions co;
    ParseSeed(seed, &co.host, &co.port);
    if (!options_.password.empty()) co.password = options_.password;
    co.connect_timeout = std::chrono::milliseconds(options_.connect_timeout_ms);
    co.socket_timeout = std::chrono::milliseconds(options_.socket_timeout_ms);
    try {
      cluster_ = std::make_unique<sw::redis::RedisCluster>(co, pool_opts);
      break;
    } catch (const sw::redis::Error& e) {
      last_err = e.what();
      MORI_UMBP_WARN("[RedisCluster] seed {} unreachable, trying next: {}", seed, e.what());
    }
  }
  if (!cluster_) {
    throw RespError("RespClusterClient: no reachable cluster seed (last error: " + last_err + ")");
  }

  // Workers to fan a multi-slot EvalPipeline out concurrently (one round trip
  // per node instead of N sequential). Each task is an independent
  // redirection-aware call, so a slow/failing node cannot block the others.
  const std::size_t threads = std::min<std::size_t>(64, std::max<std::size_t>(4, options_.pool_size));
  pool_ = std::make_unique<ThreadPool>(threads);

  MORI_UMBP_INFO("[RedisCluster] connected via {} seed(s), pool={}, fanout={}", options_.seeds.size(),
                 options_.pool_size, threads);
}

RespClusterClient::~RespClusterClient() = default;

RespValue RespClusterClient::RunArgvRouted(const std::vector<std::string>& argv,
                                           const std::string& routing_key) {
  // Send the raw argv to the node owning routing_key's slot. redis-plus-plus's
  // RedisCluster::command redirection loop resolves the slot, follows MOVED/ASK,
  // and refreshes the slot map on a moved slot or a failed-over master.
  auto sender = [&argv](sw::redis::Connection& conn, const sw::redis::StringView&) {
    std::vector<const char*> ptrs;
    std::vector<std::size_t> lens;
    ptrs.reserve(argv.size());
    lens.reserve(argv.size());
    for (const auto& a : argv) {
      ptrs.push_back(a.data());
      lens.push_back(a.size());
    }
    conn.send(static_cast<int>(ptrs.size()), ptrs.data(), lens.data());
  };

  try {
    auto reply = cluster_->command(sender, sw::redis::StringView(routing_key));
    return RespClient::Convert(reply.get());
  } catch (const sw::redis::ReplyError& e) {
    // A server-side logical error (e.g. a Lua runtime error). Surface it as an
    // Error value so callers keep the error-as-value contract. (MOVED/ASK are
    // handled inside the redirection loop and do not reach here.)
    RespValue v;
    v.type = RespValue::Type::Error;
    v.str = e.what();
    return v;
  } catch (const sw::redis::Error& e) {
    // Transport / redirection-exhausted / cluster-down: a real failure.
    throw RespError(std::string("RespClusterClient: ") + e.what());
  }
}

std::string RespClusterClient::GetOrLoadSha(const std::string& script) {
  {
    std::lock_guard<std::mutex> lk(sha_mu_);
    auto it = sha_cache_.find(script);
    if (it != sha_cache_.end()) return it->second;
  }
  // SCRIPT LOAD on any node returns the content-addressed SHA (same everywhere).
  RespValue r = RunArgvRouted({"SCRIPT", "LOAD", script}, "sha");
  if (r.type != RespValue::Type::String) {
    throw RespError("RespClusterClient: SCRIPT LOAD did not return a sha: " + r.str);
  }
  {
    std::lock_guard<std::mutex> lk(sha_mu_);
    sha_cache_[script] = r.str;
  }
  return r.str;
}

RespValue RespClusterClient::EvalShaRouted(const std::string& script,
                                           const std::vector<std::string>& keys,
                                           const std::vector<std::string>& args) {
  const std::string sha = GetOrLoadSha(script);
  std::vector<std::string> argv;
  argv.reserve(3 + keys.size() + args.size());
  argv.push_back("EVALSHA");
  argv.push_back(sha);
  argv.push_back(std::to_string(keys.size()));
  for (const auto& k : keys) argv.push_back(k);
  for (const auto& a : args) argv.push_back(a);

  RespValue r = RunArgvRouted(argv, keys.front());
  if (r.is_error() && r.str.rfind("NOSCRIPT", 0) == 0) {
    // The node serving this slot doesn't have the script cached (e.g. a promoted
    // replica or a resharded node). Load it there (routed by the same key) and
    // retry once. EVALSHA has no side effects until it runs, so this is safe.
    RunArgvRouted({"SCRIPT", "LOAD", script}, keys.front());
    r = RunArgvRouted(argv, keys.front());
  }
  return r;
}

RespValue RespClusterClient::Command(const std::vector<std::string>& args) {
  if (args.empty()) throw RespError("RespClusterClient::Command: empty argv");
  // Store commands put the key at args[1] (HGETALL/HGET/SCARD <key> ...); fall
  // back to args[0] for keyless commands.
  const std::string& routing_key = args.size() > 1 ? args[1] : args[0];
  return RunArgvRouted(args, routing_key);
}

RespValue RespClusterClient::Eval(const std::string& script, const std::vector<std::string>& keys,
                                  const std::vector<std::string>& args) {
  if (keys.empty()) {
    throw RespError("RespClusterClient::Eval: a routing key (KEYS[1]) is required in cluster mode");
  }
  return EvalShaRouted(script, keys, args);
}

std::vector<RespValue> RespClusterClient::EvalPipeline(
    const std::string& script, const std::vector<std::vector<std::string>>& keys_per_call,
    const std::vector<std::string>& shared_args) {
  std::vector<RespValue> replies(keys_per_call.size());
  if (keys_per_call.empty()) return replies;

  // One independent, redirection-aware EVAL per group; groups are single-slot
  // (the caller groups a batch by shard tag) so each routes cleanly by keys[0].
  auto run_one = [&](std::size_t i) {
    const auto& keys = keys_per_call[i];
    if (keys.empty()) {
      RespValue v;
      v.type = RespValue::Type::Error;
      v.str = "RespClusterClient::EvalPipeline: empty KEYS group";
      replies[i] = std::move(v);
      return;
    }
    try {
      replies[i] = EvalShaRouted(script, keys, shared_args);
    } catch (const std::exception& e) {
      // Keep parallel-to-input: a failed group becomes an Error value the caller
      // can inspect (mirrors the hiredis EvalPipeline's per-call replies).
      RespValue v;
      v.type = RespValue::Type::Error;
      v.str = e.what();
      replies[i] = std::move(v);
    }
  };

  if (keys_per_call.size() == 1 || pool_ == nullptr) {
    for (std::size_t i = 0; i < keys_per_call.size(); ++i) run_one(i);
    return replies;
  }

  // Fan the extra groups out to the pool, run the first inline, then join all.
  std::vector<std::future<void>> futs;
  futs.reserve(keys_per_call.size() - 1);
  for (std::size_t i = 1; i < keys_per_call.size(); ++i) {
    futs.push_back(pool_->Enqueue([&run_one, i] { run_one(i); }));
  }
  run_one(0);
  for (auto& f : futs) f.get();
  return replies;
}

bool RespClusterClient::Ping() {
  try {
    // PING ignores its "key"; routing by an arbitrary tag just picks a node.
    RespValue r = RunArgvRouted({"PING"}, "ping");
    return r.type == RespValue::Type::Status || r.type == RespValue::Type::String;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace mori::umbp::redis
