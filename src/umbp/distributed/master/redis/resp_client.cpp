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

#include <cstring>
#include <utility>

namespace mori::umbp::redis {

namespace {

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
  for (;;) {
    if (!idle_.empty()) {
      redisContext* c = idle_.back();
      idle_.pop_back();
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
      return Lease(this, c);
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
  argv.reserve(args.size());
  argvlen.reserve(args.size());
  for (const auto& a : args) {
    argv.push_back(a.data());
    argvlen.push_back(a.size());
  }
  auto* reply = static_cast<redisReply*>(
      redisCommandArgv(ctx, static_cast<int>(argv.size()), argv.data(), argvlen.data()));
  if (reply == nullptr) {
    *broke = true;
    const std::string err = ctx ? std::string(ctx->errstr) : std::string("null reply");
    throw RespError("RespClient: command failed (transport): " + err);
  }
  RespValue out = Convert(reply);
  freeReplyObject(reply);
  return out;
}

RespValue RespClient::Command(const std::vector<std::string>& args) {
  Lease lease = Acquire();
  bool broke = false;
  try {
    return RunArgv(lease.get(), args, &broke);
  } catch (...) {
    if (broke) lease.MarkBroken();
    throw;
  }
}

std::vector<RespValue> RespClient::Pipeline(const std::vector<std::vector<std::string>>& commands) {
  std::vector<RespValue> replies;
  replies.reserve(commands.size());
  Lease lease = Acquire();
  redisContext* ctx = lease.get();

  for (const auto& cmd : commands) {
    std::vector<const char*> argv;
    std::vector<size_t> argvlen;
    argv.reserve(cmd.size());
    argvlen.reserve(cmd.size());
    for (const auto& a : cmd) {
      argv.push_back(a.data());
      argvlen.push_back(a.size());
    }
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
  {
    std::lock_guard<std::mutex> lk(sha_mu_);
    auto it = sha_cache_.find(script);
    if (it != sha_cache_.end()) return it->second;
  }
  RespValue r = RunArgv(ctx, {"SCRIPT", "LOAD", script}, broke);
  if (r.type != RespValue::Type::String) {
    throw RespError("RespClient: SCRIPT LOAD did not return a sha: " + r.str);
  }
  {
    std::lock_guard<std::mutex> lk(sha_mu_);
    sha_cache_[script] = r.str;
  }
  return r.str;
}

RespValue RespClient::Eval(const std::string& script, const std::vector<std::string>& keys,
                           const std::vector<std::string>& args) {
  Lease lease = Acquire();
  redisContext* ctx = lease.get();
  bool broke = false;
  try {
    const std::string sha = GetOrLoadSha(ctx, script, &broke);

    std::vector<std::string> cmd;
    cmd.reserve(3 + keys.size() + args.size());
    cmd.push_back("EVALSHA");
    cmd.push_back(sha);
    cmd.push_back(std::to_string(keys.size()));
    for (const auto& k : keys) cmd.push_back(k);
    for (const auto& a : args) cmd.push_back(a);

    RespValue r = RunArgv(ctx, cmd, &broke);
    if (r.is_error() && r.str.rfind("NOSCRIPT", 0) == 0) {
      // Script evicted from the server cache; reload and retry once.
      {
        std::lock_guard<std::mutex> lk(sha_mu_);
        sha_cache_.erase(script);
      }
      const std::string sha2 = GetOrLoadSha(ctx, script, &broke);
      cmd[1] = sha2;
      r = RunArgv(ctx, cmd, &broke);
    }
    return r;
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
