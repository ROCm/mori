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

// RespClient — a thin, RESP-protocol-compatible client seam for the master's
// Redis metadata backend.
//
// This is deliberately the ONLY file in the store that knows which client
// library is used underneath. Phase 1 implements it on hiredis (the library
// already present in the build image); the store code above it depends only on
// the small RespValue / RespClient surface here, so a raw-hiredis client can be
// swapped for redis-plus-plus (or a cluster client) without touching the store.
//
// It speaks only the portable RESP subset (STRING/HASH/SET/ZSET commands plus
// EVAL/EVALSHA/SCRIPT LOAD), so the same binary connects unchanged to Redis,
// Dragonfly, and Valkey — selection is connection config only.
//
// Threading: a fixed-size connection pool fronts a shared instance. The gRPC
// handler thread pool calls Command/Pipeline/Eval concurrently; each call
// borrows one pooled connection for its duration.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/master/redis/resp_detail.h"
#include "umbp/distributed/master/redis/resp_value.h"

// Forward-declare the hiredis context so this header stays library-agnostic to
// its includers (only resp_client.cpp includes <hiredis/hiredis.h>).
struct redisContext;

namespace mori::umbp::redis {

// RespError and RespValue live in resp_value.h so the redis-plus-plus cluster
// client can share them without pulling in hiredis.

class RespClient : public IRespClient {
 public:
  struct Options {
    // e.g. "tcp://127.0.0.1:6379". Only tcp is supported in Phase 1.
    std::string uri = "tcp://127.0.0.1:6379";
    std::string password;
    int connect_timeout_ms = 1000;
    int socket_timeout_ms = 1000;
    std::size_t pool_size = 8;
  };

  explicit RespClient(Options options);
  ~RespClient();

  RespClient(const RespClient&) = delete;
  RespClient& operator=(const RespClient&) = delete;

  // One command. `args` is the full argv (command name first); values are
  // binary-safe. Returns the server's reply (which may be an Error value).
  RespValue Command(const std::vector<std::string>& args) override;

  // Pipeline: append every command, then read all replies in order. One round
  // trip for the whole batch. (Not part of IRespClient; single-endpoint only.)
  // NOTE: currently has no in-tree caller — retained as the single-endpoint
  // general-purpose batch primitive (the hot path uses EvalPipeline instead). If
  // it stays unused long-term, consider dropping it.
  std::vector<RespValue> Pipeline(const std::vector<std::vector<std::string>>& commands);

  // EVALSHA with transparent SCRIPT LOAD + NOSCRIPT fallback. `script` is the
  // Lua body; its SHA is loaded once and cached. `keys` then `args` are passed
  // to the script as KEYS[] / ARGV[].
  RespValue Eval(const std::string& script, const std::vector<std::string>& keys,
                 const std::vector<std::string>& args) override;

  // Pipeline the SAME script over several KEYS groups in one round trip. Call i
  // runs `script` with KEYS = keys_per_call[i] and ARGV = shared_args; the
  // returned vector is parallel to keys_per_call. Used by the sharded read path
  // to fan one batch out to one single-slot EVALSHA per shard.
  //
  // On NOSCRIPT the script is reloaded and ONLY the affected calls are retried,
  // so a script with side effects (e.g. route_get_batch's lease/access bump) is
  // never applied twice for a call that already ran.
  std::vector<RespValue> EvalPipeline(const std::string& script,
                                      const std::vector<std::vector<std::string>>& keys_per_call,
                                      const std::vector<std::string>& shared_args) override;

  // Liveness probe (PING). Returns false if no connection can be established.
  bool Ping() override;

  void SetMetrics(mori::metrics::MetricsServer* metrics) override { metrics_ = metrics; }

  const Options& options() const { return options_; }

  // Convert a raw redisReply (void* to avoid leaking the hiredis type into this
  // header) into a RespValue. Public + static so the redis-plus-plus cluster
  // client can reuse the exact same reply decoding (its ReplyUPtr is a
  // hiredis redisReply under the hood).
  static RespValue Convert(void* reply);

 private:
  // RAII borrow of one pooled connection.
  class Lease {
   public:
    Lease(RespClient* owner, redisContext* ctx) : owner_(owner), ctx_(ctx) {}
    ~Lease();
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    redisContext* get() const { return ctx_; }
    void MarkBroken() { healthy_ = false; }

   private:
    RespClient* owner_;
    redisContext* ctx_;
    bool healthy_ = true;
  };

  redisContext* Connect();  // create + AUTH; throws on failure
  Lease Acquire();          // borrow (blocks if pool exhausted)
  void Release(redisContext* ctx, bool healthy);

  // Run one argv command on a specific connection; sets *broke on transport
  // failure. Returns a RespValue (Error type on server error).
  RespValue RunArgv(redisContext* ctx, const std::vector<std::string>& args, bool* broke);

  // Append one EVALSHA (using `sha`) per index in `indices`, then read the
  // replies back in order into (*replies)[idx]. Returns the subset of `indices`
  // whose reply was a NOSCRIPT error, so the caller can reload + retry just
  // those. Sets *broke on transport failure.
  std::vector<std::size_t> PipelineEvalshaBatch(
      redisContext* ctx, const std::string& sha,
      const std::vector<std::vector<std::string>>& keys_per_call,
      const std::vector<std::string>& shared_args, const std::vector<std::size_t>& indices,
      std::vector<RespValue>* replies, bool* broke);

  // SHA for `script`, loaded + cached on first use (deterministic across
  // connections, so one cache is correct).
  std::string GetOrLoadSha(redisContext* ctx, const std::string& script, bool* broke);

  // Cold-path metric helpers (no-op when metrics_ is null). Kept off the hot
  // success path — called only on a transport failure / NOSCRIPT reload.
  void CountTransportError();
  void CountNoscriptReload();

  Options options_;
  std::string host_;
  int port_ = 6379;

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<redisContext*> idle_;
  std::size_t created_ = 0;

  // Optional cold-path metrics sink; null = no metrics. Set once at startup.
  mori::metrics::MetricsServer* metrics_ = nullptr;

  // EVALSHA SHA cache, keyed by script identity (see ScriptCache / lua_scripts.h
  // for the pointer-identity contract: never pass a freshly-constructed string).
  ScriptCache script_cache_;
};

}  // namespace mori::umbp::redis
