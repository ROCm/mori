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

// RespValue / RespError / IRespClient — the library-agnostic surface the store
// depends on, split out of resp_client.h so a client that is NOT built on raw
// hiredis (e.g. the redis-plus-plus cluster client) can implement the same
// interface without pulling in hiredis.
//
// IRespClient is the small seam the RedisMasterMetadataStore programs against.
// Two implementations live behind it:
//   * RespClient        (resp_client.h)         — raw hiredis, single endpoint;
//                                                  used for single / multi-endpoint modes.
//   * RespClusterClient (resp_cluster_client.h) — redis-plus-plus RedisCluster;
//                                                  slot routing + MOVED/ASK/failover.
//
// Error model: server-returned errors (e.g. a Lua runtime error, NOSCRIPT) are
// surfaced as a RespValue with type == Error so callers can inspect the message
// (error-as-value). Only transport-level failures throw RespError.

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

// Forward-declared so the interface can carry an optional metrics sink without
// this lightweight header pulling in the Prometheus server.
namespace mori::metrics {
class MetricsServer;
}

namespace mori::umbp::redis {

// Thrown on a transport-level failure (connect/socket error, protocol error).
// Command errors returned BY the server are NOT thrown — they surface as a
// RespValue with type == Error.
class RespError : public std::runtime_error {
 public:
  explicit RespError(const std::string& what) : std::runtime_error(what) {}
};

// Owned, recursive mirror of a redisReply. Owning it (rather than exposing the
// raw redisReply*) keeps the client library out of every includer of this
// header.
struct RespValue {
  enum class Type { Nil, Status, Error, Integer, String, Array };

  Type type = Type::Nil;
  long long integer = 0;            // Integer
  std::string str;                  // Status / Error / String (binary-safe)
  std::vector<RespValue> elements;  // Array

  bool is_nil() const { return type == Type::Nil; }
  bool is_error() const { return type == Type::Error; }
  bool is_array() const { return type == Type::Array; }
  bool ok() const { return type != Type::Error; }
};

// The small RESP surface the store programs against. Implementations own their
// own connection management (pool, cluster slot map, ...); the store only sees
// these four operations, so single-endpoint, multi-endpoint, and cluster modes
// share the same hot-path logic.
class IRespClient {
 public:
  virtual ~IRespClient() = default;

  // One command (full argv, command name first; binary-safe). Returns the
  // server reply (may be an Error value).
  virtual RespValue Command(const std::vector<std::string>& args) = 0;

  // EVALSHA with transparent SCRIPT LOAD + NOSCRIPT fallback. `keys` then `args`
  // become KEYS[] / ARGV[]. `keys` must carry at least one key per touched slot
  // so a cluster implementation can route the script.
  virtual RespValue Eval(const std::string& script, const std::vector<std::string>& keys,
                         const std::vector<std::string>& args) = 0;

  // Run `script` over several KEYS groups; returned vector is parallel to
  // keys_per_call. A cluster implementation may internally regroup calls by node
  // (different groups can land on different slots/nodes) and fan them out.
  virtual std::vector<RespValue> EvalPipeline(
      const std::string& script, const std::vector<std::vector<std::string>>& keys_per_call,
      const std::vector<std::string>& shared_args) = 0;

  // Liveness probe. Returns false if the endpoint(s) cannot be reached.
  virtual bool Ping() = 0;

  // Attach an optional metrics sink for COLD-PATH counters/histograms only
  // (transport errors, NOSCRIPT reloads, connection-pool waits). Default no-op.
  // Set once at startup, before serving traffic; the hot success path records
  // nothing (every emit is gated on a non-null sink and only on rare paths).
  virtual void SetMetrics(mori::metrics::MetricsServer* /*metrics*/) {}
};

}  // namespace mori::umbp::redis
