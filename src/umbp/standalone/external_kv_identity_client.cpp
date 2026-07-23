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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/standalone/external_kv_identity_client.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdlib>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp.grpc.pb.h"

namespace mori::umbp::standalone {
namespace {

::umbp::UMBPMaster::Stub* Stub(void* ptr) { return static_cast<::umbp::UMBPMaster::Stub*>(ptr); }

::umbp::TierType TierToProto(TierType tier) {
  switch (tier) {
    case TierType::HBM:
      return ::umbp::TIER_HBM;
    case TierType::DRAM:
      return ::umbp::TIER_DRAM;
    case TierType::SSD:
      return ::umbp::TIER_SSD;
    default:
      return ::umbp::TIER_UNKNOWN;
  }
}

TierType TierFromProto(::umbp::TierType tier) {
  switch (tier) {
    case ::umbp::TIER_HBM:
      return TierType::HBM;
    case ::umbp::TIER_DRAM:
      return TierType::DRAM;
    case ::umbp::TIER_SSD:
      return TierType::SSD;
    default:
      return TierType::UNKNOWN;
  }
}

int RpcShutdownTimeoutMs() {
  const char* raw = std::getenv("UMBP_RPC_SHUTDOWN_TIMEOUT_MS");
  if (!raw || raw[0] == '\0') return 3000;
  const int parsed = std::atoi(raw);
  return parsed > 0 ? parsed : 3000;
}

void SetDeadline(grpc::ClientContext* ctx) {
  ctx->set_deadline(std::chrono::system_clock::now() +
                    std::chrono::milliseconds(RpcShutdownTimeoutMs()));
}

}  // namespace

ExternalKvIdentityClient::ExternalKvIdentityClient(Config config)
    : config_(std::move(config)),
      stub_(nullptr, [](void* p) { delete static_cast<::umbp::UMBPMaster::Stub*>(p); }) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  args.SetMaxSendMessageSize(64 * 1024 * 1024);
  auto channel =
      grpc::CreateCustomChannel(config_.master_address, grpc::InsecureChannelCredentials(), args);
  channel_ = channel;
  stub_.reset(::umbp::UMBPMaster::NewStub(channel).release());
}

ExternalKvIdentityClient::~ExternalKvIdentityClient() { Stop(); }

bool ExternalKvIdentityClient::Start() {
  if (config_.master_address.empty() || config_.node_id.empty() || config_.node_address.empty()) {
    MORI_UMBP_WARN("[ExternalKvIdentity] invalid config: master/node identity is empty");
    return false;
  }

  {
    std::lock_guard lock(rpc_mu_);
    if (!RegisterLocked()) return false;
  }

  running_.store(true);
  heartbeat_thread_ = std::thread(&ExternalKvIdentityClient::HeartbeatLoop, this);
  return true;
}

void ExternalKvIdentityClient::Stop() {
  bool was_running = running_.exchange(false);
  if (was_running) {
    cv_.notify_one();
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
  }

  std::lock_guard lock(rpc_mu_);
  UnregisterLocked();
}

bool ExternalKvIdentityClient::RegisterLocked() {
  ::umbp::RegisterClientRequest req;
  req.set_node_id(config_.node_id);
  req.set_node_address(config_.node_address);
  req.set_peer_address(config_.peer_address);
  req.set_engine_desc(config_.engine_desc_bytes.data(), config_.engine_desc_bytes.size());
  for (const auto& tag : config_.tags) req.add_tags(tag);

  ::umbp::RegisterClientResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->RegisterClient(&ctx, req, &resp);
  if (!status.ok() && status.error_code() != grpc::StatusCode::ALREADY_EXISTS) {
    MORI_UMBP_WARN("[ExternalKvIdentity] RegisterClient failed node_id={} error={}",
                   config_.node_id, status.error_message());
    registered_ = false;
    return false;
  }
  if (resp.heartbeat_interval_ms() > 0) heartbeat_interval_ms_ = resp.heartbeat_interval_ms();
  registered_ = true;
  MORI_UMBP_INFO("[ExternalKvIdentity] registered node_id={} peer={}", config_.node_id,
                 config_.peer_address);
  return true;
}

void ExternalKvIdentityClient::UnregisterLocked() {
  if (!registered_) return;
  ::umbp::UnregisterClientRequest req;
  req.set_node_id(config_.node_id);
  ::umbp::UnregisterClientResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->UnregisterClient(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[ExternalKvIdentity] UnregisterClient failed node_id={} error={}",
                   config_.node_id, status.error_message());
  }
  registered_ = false;
}

bool ExternalKvIdentityClient::SendHeartbeatOnceLocked() {
  if (!registered_) return false;

  ::umbp::HeartbeatRequest req;
  req.set_node_id(config_.node_id);
  req.set_is_full_sync(false);

  ::umbp::HeartbeatResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->Heartbeat(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[ExternalKvIdentity] Heartbeat failed node_id={} error={}", config_.node_id,
                   status.error_message());
    return false;
  }
  if (resp.status() == ::umbp::CLIENT_STATUS_UNKNOWN) {
    MORI_UMBP_WARN("[ExternalKvIdentity] master forgot node_id={}, re-registering",
                   config_.node_id);
    registered_ = false;
    return RegisterLocked();
  }
  return true;
}

void ExternalKvIdentityClient::HeartbeatLoop() {
  while (running_.load()) {
    std::unique_lock lock(cv_mu_);
    cv_.wait_for(lock, std::chrono::milliseconds(heartbeat_interval_ms_),
                 [this] { return !running_.load(); });
    lock.unlock();
    if (!running_.load()) break;

    std::lock_guard rpc_lock(rpc_mu_);
    SendHeartbeatOnceLocked();
  }
}

bool ExternalKvIdentityClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes,
                                                      TierType tier) {
  if (hashes.empty()) return true;
  std::lock_guard lock(rpc_mu_);
  if (!registered_) return false;
  ::umbp::ReportExternalKvBlocksRequest req;
  req.set_node_id(config_.node_id);
  req.set_tier(TierToProto(tier));
  for (const auto& hash : hashes) req.add_hashes(hash);
  ::umbp::ReportExternalKvBlocksResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->ReportExternalKvBlocks(&ctx, req, &resp);
  return status.ok();
}

bool ExternalKvIdentityClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes,
                                                      TierType tier) {
  if (hashes.empty()) return true;
  std::lock_guard lock(rpc_mu_);
  if (!registered_) return false;
  ::umbp::RevokeExternalKvBlocksRequest req;
  req.set_node_id(config_.node_id);
  req.set_tier(TierToProto(tier));
  for (const auto& hash : hashes) req.add_hashes(hash);
  ::umbp::RevokeExternalKvBlocksResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->RevokeExternalKvBlocks(&ctx, req, &resp);
  return status.ok();
}

bool ExternalKvIdentityClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  std::lock_guard lock(rpc_mu_);
  if (!registered_) return false;
  ::umbp::RevokeAllExternalKvBlocksAtTierRequest req;
  req.set_node_id(config_.node_id);
  req.set_tier(TierToProto(tier));
  ::umbp::RevokeAllExternalKvBlocksAtTierResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->RevokeAllExternalKvBlocksAtTier(&ctx, req, &resp);
  return status.ok();
}

std::vector<IUMBPClient::ExternalKvMatch> ExternalKvIdentityClient::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit) {
  std::vector<IUMBPClient::ExternalKvMatch> out;
  if (hashes.empty()) return out;
  std::lock_guard lock(rpc_mu_);
  if (!registered_) return out;

  ::umbp::MatchExternalKvRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  req.set_count_as_hit(count_as_hit);
  ::umbp::MatchExternalKvResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->MatchExternalKv(&ctx, req, &resp);
  if (!status.ok()) return out;

  out.reserve(resp.matches_size());
  for (const auto& m : resp.matches()) {
    IUMBPClient::ExternalKvMatch match;
    match.node_id = m.node_id();
    match.peer_address = m.peer_address();
    for (const auto& bucket : m.hashes_by_tier()) {
      std::vector<std::string> values(bucket.hashes().begin(), bucket.hashes().end());
      match.hashes_by_tier[TierFromProto(bucket.tier())] = std::move(values);
    }
    out.push_back(std::move(match));
  }
  return out;
}

std::vector<IUMBPClient::ExternalKvHitCountEntry> ExternalKvIdentityClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes) {
  std::vector<IUMBPClient::ExternalKvHitCountEntry> out;
  if (hashes.empty()) return out;
  std::lock_guard lock(rpc_mu_);
  if (!registered_) return out;

  ::umbp::GetExternalKvHitCountsRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  ::umbp::GetExternalKvHitCountsResponse resp;
  grpc::ClientContext ctx;
  SetDeadline(&ctx);
  auto status = Stub(stub_.get())->GetExternalKvHitCounts(&ctx, req, &resp);
  if (!status.ok()) return out;

  out.reserve(resp.entries_size());
  for (const auto& e : resp.entries()) {
    IUMBPClient::ExternalKvHitCountEntry entry;
    entry.hash = e.hash();
    entry.hit_count_total = e.hit_count_total();
    out.push_back(std::move(entry));
  }
  return out;
}

}  // namespace mori::umbp::standalone
