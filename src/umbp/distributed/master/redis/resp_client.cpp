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
#include "umbp/distributed/master/redis/resp_client.h"

#include <hiredis/hiredis.h>

#include <chrono>
#include <cstring>
#include <utility>

#include "mori/metrics/prometheus_metrics_server.hpp"

namespace mori::umbp::redis {

namespace {

// Backend label shared by every RESP-client cold-path metric.
const mori::metrics::MetricsServer::Labels kRedisLabels = {{"backend", "redis"}};

// Parse "tcp://host:port" (or "host:port") into (host, port). Throws on a
// malformed URI so a misconfigured master fails fast at startup.
void ParseUri(const std::string& uri, std::string* host, int* port) {
  std::string rest = uri;
  const std::string scheme = "tcp://";
  if (rest.rfind(scheme, 0) == 0) rest = rest.substr(scheme.size());
  // Strip any comma-separated cluster seeds; Phase 1 uses the first only.
  const auto comma = rest.find(',');
  if (comma != std::string::npos) rest = rest.substr(0, comma);
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
    throw RespError("RespClient: invalid port in URI '" + uri + "'");
  }
}

}  // namespace

RespClient::Lease::~Lease() { owner_->Release(ctx_, healthy_); }

RespClient::RespClient(Options options) : options_(std::move(options)) {
  ParseUri(options_.uri, &host_, &port_);
  if (options_.pool_size == 0) options_.pool_size = 1;
  idle_.reserve(options_.pool_size);
}

RespClient::~RespClient() {
  std::lock_guard<std::mutex> lk(mu_);
  for (redisContext* c : idle_) {
    if (c) redisFree(c);
  }
  idle_.clear();
}

void RespClient::CountTransportError() {
  if (metrics_ == nullptr) return;
  metrics_->addCounter("mori_umbp_redis_transport_errors_total",
                       "RESP client transport failures (connect/socket/protocol)", kRedisLabels);
}

void RespClient::CountNoscriptReload() {
  if (metrics_ == nullptr) return;
  metrics_->addCounter("mori_umbp_redis_noscript_reload_total",
                       "EVALSHA NOSCRIPT reloads (server evicted the cached script)", kRedisLabels);
}

redisContext* RespClient::Connect() {
  timeval tv{};
  tv.tv_sec = options_.connect_timeout_ms / 1000;
  tv.tv_usec = (options_.connect_timeout_ms % 1000) * 1000;
  redisContext* ctx = redisConnectWithTimeout(host_.c_str(), port_, tv);
  if (ctx == nullptr || ctx->err) {
    const std::string msg =
        ctx ? std::string(ctx->errstr) : std::string("redisConnectWithTimeout returned null");
    if (ctx) redisFree(ctx);
    throw RespError("RespClient: connect to " + host_ + ":" + std::to_string(port_) +
                    " failed: " + msg);
  }
  timeval sock{};
  sock.tv_sec = options_.socket_timeout_ms / 1000;
  sock.tv_usec = (options_.socket_timeout_ms % 1000) * 1000;
  redisSetTimeout(ctx, sock);

  if (!options_.password.empty()) {
    redisReply* r =
        static_cast<redisReply*>(redisCommand(ctx, "AUTH %s", options_.password.c_str()));
    const bool bad = (r == nullptr) || (r->type == REDIS_REPLY_ERROR);
    if (r) freeReplyObject(r);
    if (bad) {
      redisFree(ctx);
      throw RespError("RespClient: AUTH failed");
    }
  }
  return ctx;
}

RespClient::Lease RespClient::Acquire() {
  std::unique_lock<std::mutex> lk(mu_);
  // Pool-wait timing is armed lazily: only when we actually block on an exhausted
  // pool (the else branch below). The fast path (idle conn available or room to
  // create one) reads no clock and touches no metric.
  bool waited = false;
  std::chrono::steady_clock::time_point wait_start;
  auto observe_wait = [&] {
    if (!waited || metrics_ == nullptr) return;
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
    static const std::vector<double> kBounds = {0.0001, 0.00025, 0.0005, 0.001, 0.0025,
                                                0.005,  0.01,    0.025,  0.05,  0.1};
    metrics_->observe("mori_umbp_redis_pool_wait_seconds",
                      "Time a caller blocked waiting for a free pooled RESP connection",
                      kRedisLabels, kBounds, sec);
  };
  for (;;) {
    if (!idle_.empty()) {
      redisContext* c = idle_.back();
      idle_.pop_back();
      observe_wait();
      return Lease(this, c);
    }
    if (created_ < options_.pool_size) {
      ++created_;
      lk.unlock();
      redisContext* c = nullptr;
      try {
        c = Connect();
      } catch (...) {
        std::lock_guard<std::mutex> relk(mu_);
        --created_;
        cv_.notify_one();
        throw;
      }
      observe_wait();
      return Lease(this, c);
    }
    if (metrics_ != nullptr && !waited) {
      waited = true;
      wait_start = std::chrono::steady_clock::now();
      metrics_->addCounter("mori_umbp_redis_pool_exhausted_total",
                           "Times a caller found the RESP connection pool exhausted and blocked",
                           kRedisLabels);
    }
    cv_.wait(lk);
  }
}

void RespClient::Release(redisContext* ctx, bool healthy) {
  std::lock_guard<std::mutex> lk(mu_);
  if (healthy && ctx != nullptr && ctx->err == 0) {
    idle_.push_back(ctx);
  } else {
    if (ctx) redisFree(ctx);
    if (created_ > 0) --created_;
  }
  cv_.notify_one();
}

RespValue RespClient::Convert(void* reply_ptr) {
  RespValue out;
  auto* reply = static_cast<redisReply*>(reply_ptr);
  if (reply == nullptr) {
    out.type = RespValue::Type::Nil;
    return out;
  }
  switch (reply->type) {
    case REDIS_REPLY_STRING:
      out.type = RespValue::Type::String;
      out.str.assign(reply->str, reply->len);
      break;
    case REDIS_REPLY_STATUS:
      out.type = RespValue::Type::Status;
      out.str.assign(reply->str, reply->len);
      break;
    case REDIS_REPLY_ERROR:
      out.type = RespValue::Type::Error;
      out.str.assign(reply->str, reply->len);
      break;
    case REDIS_REPLY_INTEGER:
      out.type = RespValue::Type::Integer;
      out.integer = reply->integer;
      break;
    case REDIS_REPLY_NIL:
      out.type = RespValue::Type::Nil;
      break;
    case REDIS_REPLY_ARRAY:
      out.type = RespValue::Type::Array;
      out.elements.reserve(reply->elements);
      for (size_t i = 0; i < reply->elements; ++i) {
        out.elements.push_back(Convert(reply->element[i]));
      }
      break;
    default:
      // REDIS_REPLY_DOUBLE / MAP / SET / etc. (RESP3): treat as string/array
      // best-effort. Phase 1 uses RESP2, so this is rarely hit.
      if (reply->str != nullptr) {
        out.type = RespValue::Type::String;
        out.str.assign(reply->str, reply->len);
      } else {
        out.type = RespValue::Type::Nil;
      }
      break;
  }
  return out;
}

RespValue RespClient::RunArgv(redisContext* ctx, const std::vector<std::string>& args,
                              bool* broke) {
  std::vector<const char*> argv;
  std::vector<size_t> argvlen;
  ToArgv(args, argv, argvlen);
  auto* reply = static_cast<redisReply*>(
      redisCommandArgv(ctx, static_cast<int>(argv.size()), argv.data(), argvlen.data()));
  if (reply == nullptr) {
    *broke = true;
    CountTransportError();
    const std::string err = ctx ? std::string(ctx->errstr) : std::string("null reply");
    throw RespError("RespClient: command failed (transport): " + err);
  }
  RespValue out = Convert(reply);
  freeReplyObject(reply);
  return out;
}

RespValue RespClient::Command(const std::vector<std::string>& args) {
  // Command backs only pure-READ metadata lookups (HGETALL/HGET/SCARD), so a
  // transport failure on a stale pooled connection — a server/LB drops an idle
  // socket and hiredis trips ctx->err only on the next use — is safe to retry
  // once on a fresh connection. This is deliberately NOT done for Eval /
  // EvalPipeline: route_get_batch bumps _lease/_lacc/_acnt, so a blind retry of a
  // script that may already have run server-side could double-apply a side
  // effect. A connect failure (Acquire throws) is a different case and propagates
  // immediately. A second transport failure propagates.
  for (int attempt = 0; attempt < 2; ++attempt) {
    Lease lease = Acquire();
    bool broke = false;
    try {
      return RunArgv(lease.get(), args, &broke);
    } catch (const RespError&) {
      if (broke) lease.MarkBroken();
      if (broke && attempt == 0) continue;  // stale socket: retry once, fresh conn.
      throw;
    }
  }
  throw RespError("RespClient: command retry exhausted");  // unreachable
}

std::vector<RespValue> RespClient::Pipeline(const std::vector<std::vector<std::string>>& commands) {
  std::vector<RespValue> replies;
  replies.reserve(commands.size());
  Lease lease = Acquire();
  redisContext* ctx = lease.get();

  for (const auto& cmd : commands) {
    std::vector<const char*> argv;
    std::vector<size_t> argvlen;
    ToArgv(cmd, argv, argvlen);
    if (redisAppendCommandArgv(ctx, static_cast<int>(argv.size()), argv.data(), argvlen.data()) !=
        REDIS_OK) {
      lease.MarkBroken();
      throw RespError("RespClient: pipeline append failed");
    }
  }
  for (size_t i = 0; i < commands.size(); ++i) {
    void* r = nullptr;
    if (redisGetReply(ctx, &r) != REDIS_OK) {
      lease.MarkBroken();
      const std::string err = ctx ? std::string(ctx->errstr) : std::string("null");
      throw RespError("RespClient: pipeline read failed: " + err);
    }
    replies.push_back(Convert(r));
    if (r) freeReplyObject(static_cast<redisReply*>(r));
  }
  return replies;
}

std::string RespClient::GetOrLoadSha(redisContext* ctx, const std::string& script, bool* broke) {
  return script_cache_.GetOrLoad(script, [&](const std::string& s) {
    RespValue r = RunArgv(ctx, {"SCRIPT", "LOAD", s}, broke);
    if (r.type != RespValue::Type::String) {
      throw RespError("RespClient: SCRIPT LOAD did not return a sha: " + r.str);
    }
    return r.str;
  });
}

RespValue RespClient::Eval(const std::string& script, const std::vector<std::string>& keys,
                           const std::vector<std::string>& args) {
  Lease lease = Acquire();
  redisContext* ctx = lease.get();
  bool broke = false;
  try {
    const std::string sha = GetOrLoadSha(ctx, script, &broke);
    std::vector<std::string> cmd = BuildEvalshaArgv(sha, keys, args);

    RespValue r = RunArgv(ctx, cmd, &broke);
    if (r.is_error() && r.str.rfind("NOSCRIPT", 0) == 0) {
      // Script evicted from the server cache; reload and retry once.
      CountNoscriptReload();
      script_cache_.Invalidate(script);
      cmd[1] = GetOrLoadSha(ctx, script, &broke);
      r = RunArgv(ctx, cmd, &broke);
    }
    return r;
  } catch (...) {
    if (broke) lease.MarkBroken();
    throw;
  }
}

std::vector<std::size_t> RespClient::PipelineEvalshaBatch(
    redisContext* ctx, const std::string& sha,
    const std::vector<std::vector<std::string>>& keys_per_call,
    const std::vector<std::string>& shared_args, const std::vector<std::size_t>& indices,
    std::vector<RespValue>* replies, bool* broke) {
  // Queue one EVALSHA <sha> <nkeys> <keys...> <shared_args...> per index.
  for (const std::size_t idx : indices) {
    std::vector<std::string> cmd = BuildEvalshaArgv(sha, keys_per_call[idx], shared_args);
    std::vector<const char*> argv;
    std::vector<size_t> argvlen;
    ToArgv(cmd, argv, argvlen);
    if (redisAppendCommandArgv(ctx, static_cast<int>(argv.size()), argv.data(), argvlen.data()) !=
        REDIS_OK) {
      *broke = true;
      CountTransportError();
      throw RespError("RespClient: EvalPipeline append failed");
    }
  }

  // Read replies back in the same order; flag the ones the server didn't have.
  std::vector<std::size_t> noscript;
  for (const std::size_t idx : indices) {
    void* r = nullptr;
    if (redisGetReply(ctx, &r) != REDIS_OK) {
      *broke = true;
      CountTransportError();
      const std::string err = ctx ? std::string(ctx->errstr) : std::string("null");
      throw RespError("RespClient: EvalPipeline read failed: " + err);
    }
    RespValue val = Convert(r);
    if (r) freeReplyObject(static_cast<redisReply*>(r));
    if (val.is_error() && val.str.rfind("NOSCRIPT", 0) == 0) noscript.push_back(idx);
    (*replies)[idx] = std::move(val);
  }
  return noscript;
}

std::vector<RespValue> RespClient::EvalPipeline(
    const std::string& script, const std::vector<std::vector<std::string>>& keys_per_call,
    const std::vector<std::string>& shared_args) {
  std::vector<RespValue> replies(keys_per_call.size());
  if (keys_per_call.empty()) return replies;

  Lease lease = Acquire();
  redisContext* ctx = lease.get();
  bool broke = false;
  try {
    const std::string sha = GetOrLoadSha(ctx, script, &broke);

    std::vector<std::size_t> all(keys_per_call.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
    std::vector<std::size_t> missing =
        PipelineEvalshaBatch(ctx, sha, keys_per_call, shared_args, all, &replies, &broke);

    // NOSCRIPT => the server evicted the script from its cache. Reload once and
    // retry ONLY the calls that missed, so calls that already executed (and, for
    // route_get_batch, already bumped lease/access) are not run a second time.
    if (!missing.empty()) {
      CountNoscriptReload();
      script_cache_.Invalidate(script);
      const std::string sha2 = GetOrLoadSha(ctx, script, &broke);
      PipelineEvalshaBatch(ctx, sha2, keys_per_call, shared_args, missing, &replies, &broke);
    }
    return replies;
  } catch (...) {
    if (broke) lease.MarkBroken();
    throw;
  }
}

bool RespClient::Ping() {
  try {
    RespValue r = Command({"PING"});
    return r.type == RespValue::Type::Status || r.type == RespValue::Type::String;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace mori::umbp::redis
