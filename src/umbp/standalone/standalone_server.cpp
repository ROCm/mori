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
#include "umbp/standalone/standalone_server.h"

#include <grpcpp/grpcpp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/standalone/external_kv_identity_client.h"
#include "umbp/standalone/ipc.h"
#include "umbp/umbp_client.h"
#include "umbp_standalone.grpc.pb.h"

namespace mori::umbp::standalone {
namespace {

std::chrono::seconds ShutdownDeadline() {
  const char* v = std::getenv("UMBP_STANDALONE_GRPC_SHUTDOWN_DEADLINE_SEC");
  if (!v) v = std::getenv("UMBP_GRPC_SHUTDOWN_DEADLINE_SEC");
  if (!v) return std::chrono::seconds(5);
  int seconds = std::atoi(v);
  return std::chrono::seconds(seconds > 0 ? seconds : 5);
}

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

bool FillSockaddr(const std::string& path, sockaddr_un* addr, socklen_t* addr_len) {
  if (path.empty() || path.size() >= sizeof(addr->sun_path)) return false;
  std::memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;
  std::strncpy(addr->sun_path, path.c_str(), sizeof(addr->sun_path) - 1);
  *addr_len = static_cast<socklen_t>(sizeof(sa_family_t) + path.size() + 1);
  return true;
}

void SetBool(::umbp::BoolResponse* response, bool ok, const std::string& error = {}) {
  response->set_ok(ok);
  if (!error.empty()) response->set_error(error);
}

bool SetFdSocketTimeouts(int fd, std::chrono::milliseconds timeout) {
  if (fd < 0 || timeout.count() <= 0) return true;
  timeval tv;
  tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
  return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0 &&
         setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
}

UMBPConfig NormalizeBackendConfig(UMBPConfig config) {
  // Older callers pass the worker-facing standalone_process field to the
  // server constructor. The server backend must never consume that field,
  // otherwise CreateUMBPClient would recursively create a client to itself.
  config.standalone_process.reset();
  return config;
}

::umbp::StandaloneBackendMode BackendModeToProto(UMBPDeploymentMode mode) {
  switch (mode) {
    case UMBPDeploymentMode::Distributed:
      return ::umbp::STANDALONE_BACKEND_DISTRIBUTED;
    case UMBPDeploymentMode::Local:
      return ::umbp::STANDALONE_BACKEND_LOCAL;
    case UMBPDeploymentMode::StandaloneProcess:
    default:
      return ::umbp::STANDALONE_BACKEND_UNKNOWN;
  }
}

std::chrono::milliseconds FdHandshakeTimeout() {
  const char* raw = std::getenv("UMBP_STANDALONE_FD_HANDSHAKE_TIMEOUT_MS");
  if (!raw || raw[0] == '\0') return std::chrono::milliseconds(5000);
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || value <= 0) return std::chrono::milliseconds(5000);
  if (value > INT_MAX) value = INT_MAX;
  return std::chrono::milliseconds(value);
}

}  // namespace

class StandaloneServer::Impl final : public ::umbp::UMBPStandalone::Service {
 public:
  Impl(const UMBPConfig& config, std::string address)
      : backend_config_(NormalizeBackendConfig(config)),
        client_(CreateUMBPClient(backend_config_)),
        address_(std::move(address)),
        fd_socket_path_(DeriveFdSocketPath(address_)) {}

  ~Impl() override { Shutdown(); }

  bool Start() {
    std::string error;
    const std::string grpc_path = UnixPathFromGrpcAddress(address_);
    if (!EnsureParentDirectory(grpc_path, &error)) {
      MORI_UMBP_ERROR("[StandaloneServer] {}", error);
      return false;
    }
    if (!EnsureParentDirectory(fd_socket_path_, &error)) {
      MORI_UMBP_ERROR("[StandaloneServer] {}", error);
      return false;
    }

    unlink(grpc_path.c_str());
    unlink(fd_socket_path_.c_str());

    if (!StartFdListener()) return false;

    grpc::ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    builder.SetMaxSendMessageSize(64 * 1024 * 1024);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 4);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 32);

    int selected_port = 0;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(this);
    mode_t old_umask = umask(0077);
    server_ = builder.BuildAndStart();
    umask(old_umask);
    if (!server_) {
      MORI_UMBP_ERROR("[StandaloneServer] failed to start gRPC server on {}", address_);
      StopFdListener();
      return false;
    }

    chmod(grpc_path.c_str(), 0600);
    MORI_UMBP_INFO("[StandaloneServer] listening grpc={} fd_socket={}", address_, fd_socket_path_);
    return true;
  }

  void Run() {
    if (server_) server_->Wait();
  }

  void Shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;

    StopFdListener();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + ShutdownDeadline());
    }
    UnregisterAllExternalIdentities();
    {
      std::lock_guard<std::mutex> lock(client_mu_);
      client_->Flush();
    }
    UnmapAll();
    {
      std::lock_guard<std::mutex> lock(client_mu_);
      client_->Close();
    }
    unlink(UnixPathFromGrpcAddress(address_).c_str());
    unlink(fd_socket_path_.c_str());
  }

  grpc::Status Ping(grpc::ServerContext*, const ::umbp::Empty*,
                    ::umbp::PingResponse* response) override {
    response->set_ready(!shutdown_.load());
    response->set_deployment_mode(BackendModeToProto(client_->GetDeploymentMode()));
    return grpc::Status::OK;
  }

  grpc::Status Put(grpc::ServerContext*, const ::umbp::PutRequest* request,
                   ::umbp::BoolResponse* response) override {
    uintptr_t ptr = 0;
    if (!ResolveRange(request->client_id(), request->region_base(), request->shm_offset(),
                      request->size(), &ptr)) {
      SetBool(response, false, "unregistered or out-of-range shm buffer");
      return grpc::Status::OK;
    }
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    SetBool(response, client_->Put(request->key(), ptr, static_cast<size_t>(request->size())));
    return grpc::Status::OK;
  }

  grpc::Status Get(grpc::ServerContext*, const ::umbp::GetRequest* request,
                   ::umbp::BoolResponse* response) override {
    uintptr_t ptr = 0;
    if (!ResolveRange(request->client_id(), request->region_base(), request->shm_offset(),
                      request->size(), &ptr)) {
      SetBool(response, false, "unregistered or out-of-range shm buffer");
      return grpc::Status::OK;
    }
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    SetBool(response, client_->Get(request->key(), ptr, static_cast<size_t>(request->size())));
    return grpc::Status::OK;
  }

  grpc::Status BatchPut(grpc::ServerContext*, const ::umbp::BatchDataRequest* request,
                        ::umbp::BatchBoolResponse* response) override {
    std::vector<uintptr_t> ptrs;
    if (!ResolveBatch(*request, &ptrs)) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::vector<size_t> sizes = Sizes(*request);
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    FillResults(client_->BatchPut(keys, ptrs, sizes), response);
    return grpc::Status::OK;
  }

  grpc::Status BatchPutWithDepth(grpc::ServerContext*,
                                 const ::umbp::BatchDataWithDepthRequest* request,
                                 ::umbp::BatchBoolResponse* response) override {
    // region_bases is optional for legacy single-region callers; when present it
    // must be parallel to keys (BatchPutWithDepth resolves inline, not through
    // ResolveBatch).
    const bool has_region_bases = request->region_bases_size() > 0;
    if (request->keys_size() != request->shm_offsets_size() ||
        request->keys_size() != request->sizes_size() ||
        (has_region_bases && request->keys_size() != request->region_bases_size())) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    std::vector<uintptr_t> ptrs;
    ptrs.reserve(request->keys_size());
    for (int i = 0; i < request->keys_size(); ++i) {
      uintptr_t ptr = 0;
      uint64_t region_base = has_region_bases ? request->region_bases(i) : 0;
      if (!ResolveRange(request->client_id(), region_base, request->shm_offsets(i),
                        request->sizes(i), &ptr)) {
        FillFalse(request->keys_size(), response);
        return grpc::Status::OK;
      }
      ptrs.push_back(ptr);
    }
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::vector<size_t> sizes;
    sizes.reserve(request->sizes_size());
    for (uint64_t size : request->sizes()) sizes.push_back(static_cast<size_t>(size));
    std::vector<int> depths(request->depths().begin(), request->depths().end());
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    FillResults(client_->BatchPutWithDepth(keys, ptrs, sizes, depths), response);
    return grpc::Status::OK;
  }

  grpc::Status BatchGet(grpc::ServerContext*, const ::umbp::BatchDataRequest* request,
                        ::umbp::BatchBoolResponse* response) override {
    std::vector<uintptr_t> ptrs;
    if (!ResolveBatch(*request, &ptrs)) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::vector<size_t> sizes = Sizes(*request);
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    FillResults(client_->BatchGet(keys, ptrs, sizes), response);
    return grpc::Status::OK;
  }

  grpc::Status Exists(grpc::ServerContext*, const ::umbp::KeyRequest* request,
                      ::umbp::BoolResponse* response) override {
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    SetBool(response, client_->Exists(request->key()));
    return grpc::Status::OK;
  }

  grpc::Status BatchExists(grpc::ServerContext*, const ::umbp::BatchKeysRequest* request,
                           ::umbp::BatchBoolResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      FillFalse(request->keys_size(), response);
      return grpc::Status::OK;
    }
    FillResults(client_->BatchExists(keys), response);
    return grpc::Status::OK;
  }

  grpc::Status BatchExistsConsecutive(grpc::ServerContext*, const ::umbp::BatchKeysRequest* request,
                                      ::umbp::CountResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      response->set_count(0);
      return grpc::Status::OK;
    }
    response->set_count(client_->BatchExistsConsecutive(keys));
    return grpc::Status::OK;
  }

  grpc::Status Clear(grpc::ServerContext*, const ::umbp::Empty*,
                     ::umbp::BoolResponse* response) override {
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    SetBool(response, client_->Clear());
    return grpc::Status::OK;
  }

  grpc::Status Flush(grpc::ServerContext*, const ::umbp::Empty*,
                     ::umbp::BoolResponse* response) override {
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    SetBool(response, client_->Flush());
    return grpc::Status::OK;
  }

  grpc::Status RegisterMemory(grpc::ServerContext*, const ::umbp::RegisterMemoryRequest* request,
                              ::umbp::BoolResponse* response) override {
    if (shutdown_.load()) {
      SetBool(response, false, "server is shutting down");
      return grpc::Status::OK;
    }
    std::lock_guard<std::mutex> lifecycle_lock(external_identity_lifecycle_mu_);
    bool ok = false;
    {
      std::lock_guard<std::mutex> lock(memory_mu_);
      auto it = memory_.find(request->client_id());
      if (it != memory_.end()) {
        ok = std::any_of(it->second.begin(), it->second.end(), [&](const RegisteredMemory& mem) {
          return mem.worker_base == request->worker_base() && mem.size >= request->size();
        });
      }
    }
    if (!ok) {
      SetBool(response, false, "fd handoff registration was not found");
      return grpc::Status::OK;
    }

    if (!EnsureExternalIdentity(*request)) {
      MORI_UMBP_WARN(
          "[StandaloneServer] external-KV identity registration failed for client_id={} "
          "worker_node_id={}; continuing with core memory registration",
          request->client_id(), request->worker_node_id());
    }
    SetBool(response, true);
    return grpc::Status::OK;
  }

  grpc::Status DeregisterMemory(grpc::ServerContext*,
                                const ::umbp::DeregisterMemoryRequest* request,
                                ::umbp::Empty*) override {
    std::lock_guard<std::mutex> lifecycle_lock(external_identity_lifecycle_mu_);
    RemoveExternalIdentity(request->client_id());
    UnmapClient(request->client_id());
    return grpc::Status::OK;
  }

  grpc::Status ReportExternalKvBlocks(grpc::ServerContext*,
                                      const ::umbp::StandaloneExternalKvMutationRequest* request,
                                      ::umbp::BoolResponse* response) override {
    if (!BackendIsDistributed()) {
      SetBool(response, true);
      return grpc::Status::OK;
    }
    auto identity = GetExternalIdentity(request->client_id());
    if (!identity) {
      SetBool(response, false, "external-KV identity is not registered for client_id");
      return grpc::Status::OK;
    }
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    SetBool(response, identity->ReportExternalKvBlocks(hashes, TierFromProto(request->tier())));
    return grpc::Status::OK;
  }

  grpc::Status RevokeExternalKvBlocks(grpc::ServerContext*,
                                      const ::umbp::StandaloneExternalKvMutationRequest* request,
                                      ::umbp::BoolResponse* response) override {
    if (!BackendIsDistributed()) {
      SetBool(response, true);
      return grpc::Status::OK;
    }
    auto identity = GetExternalIdentity(request->client_id());
    if (!identity) {
      SetBool(response, false, "external-KV identity is not registered for client_id");
      return grpc::Status::OK;
    }
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    SetBool(response, identity->RevokeExternalKvBlocks(hashes, TierFromProto(request->tier())));
    return grpc::Status::OK;
  }

  grpc::Status RevokeAllExternalKvBlocksAtTier(
      grpc::ServerContext*, const ::umbp::StandaloneExternalKvTierRequest* request,
      ::umbp::BoolResponse* response) override {
    if (!BackendIsDistributed()) {
      SetBool(response, true);
      return grpc::Status::OK;
    }
    auto identity = GetExternalIdentity(request->client_id());
    if (!identity) {
      SetBool(response, false, "external-KV identity is not registered for client_id");
      return grpc::Status::OK;
    }
    SetBool(response, identity->RevokeAllExternalKvBlocksAtTier(TierFromProto(request->tier())));
    return grpc::Status::OK;
  }

  grpc::Status MatchExternalKv(grpc::ServerContext*,
                               const ::umbp::StandaloneMatchExternalKvRequest* request,
                               ::umbp::StandaloneMatchExternalKvResponse* response) override {
    if (!BackendIsDistributed()) return grpc::Status::OK;
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    std::vector<IUMBPClient::ExternalKvMatch> matches;
    if (auto identity = GetExternalIdentity(request->client_id())) {
      matches = identity->MatchExternalKv(hashes, request->count_as_hit());
    } else {
      std::lock_guard<std::mutex> lock(client_mu_);
      if (!shutdown_.load()) matches = client_->MatchExternalKv(hashes, request->count_as_hit());
    }
    FillExternalKvMatches(matches, response);
    return grpc::Status::OK;
  }

  grpc::Status GetExternalKvHitCounts(
      grpc::ServerContext*, const ::umbp::StandaloneExternalKvHitCountsRequest* request,
      ::umbp::StandaloneExternalKvHitCountsResponse* response) override {
    if (!BackendIsDistributed()) return grpc::Status::OK;
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    std::vector<IUMBPClient::ExternalKvHitCountEntry> entries;
    if (auto identity = GetExternalIdentity(request->client_id())) {
      entries = identity->GetExternalKvHitCounts(hashes);
    } else {
      std::lock_guard<std::mutex> lock(client_mu_);
      if (!shutdown_.load()) entries = client_->GetExternalKvHitCounts(hashes);
    }
    FillExternalKvHitCounts(entries, response);
    return grpc::Status::OK;
  }

 private:
  struct RegisteredMemory {
    void* base = nullptr;
    uint64_t worker_base = 0;
    uint64_t size = 0;
  };

  bool BackendIsDistributed() const {
    return client_ && client_->GetDeploymentMode() == UMBPDeploymentMode::Distributed;
  }

  std::string BackendPeerAddress() const {
    if (!backend_config_.distributed.has_value()) return "";
    const auto& dist = backend_config_.distributed.value();
    if (dist.peer_service_port == 0 || dist.master_config.node_address.empty()) return "";
    return dist.master_config.node_address + ":" + std::to_string(dist.peer_service_port);
  }

  bool EnsureExternalIdentity(const ::umbp::RegisterMemoryRequest& request) {
    if (!BackendIsDistributed()) return true;
    if (request.worker_node_id().empty()) return true;
    if (!backend_config_.distributed.has_value()) return false;

    std::shared_ptr<ExternalKvIdentityClient> old;
    {
      std::lock_guard<std::mutex> lock(external_identity_mu_);
      auto it = external_identities_.find(request.client_id());
      if (it != external_identities_.end()) {
        if (it->second && it->second->node_id() == request.worker_node_id()) return true;
        old = std::move(it->second);
        external_identities_.erase(it);
      }
    }
    if (old) old->Stop();

    const auto& dist = backend_config_.distributed.value();
    ExternalKvIdentityClient::Config cfg;
    cfg.master_address = dist.master_config.master_address;
    cfg.node_id = request.worker_node_id();
    cfg.node_address = request.worker_node_address();
    cfg.peer_address = BackendPeerAddress();
    cfg.tags.assign(request.tags().begin(), request.tags().end());

    auto identity = std::make_shared<ExternalKvIdentityClient>(std::move(cfg));
    if (!identity->Start()) return false;

    {
      std::lock_guard<std::mutex> lock(external_identity_mu_);
      external_identities_[request.client_id()] = identity;
    }
    return true;
  }

  std::shared_ptr<ExternalKvIdentityClient> GetExternalIdentity(const std::string& client_id) {
    std::lock_guard<std::mutex> lock(external_identity_mu_);
    auto it = external_identities_.find(client_id);
    return it == external_identities_.end() ? nullptr : it->second;
  }

  void RemoveExternalIdentity(const std::string& client_id) {
    std::shared_ptr<ExternalKvIdentityClient> identity;
    {
      std::lock_guard<std::mutex> lock(external_identity_mu_);
      auto it = external_identities_.find(client_id);
      if (it == external_identities_.end()) return;
      identity = std::move(it->second);
      external_identities_.erase(it);
    }
    if (identity) identity->Stop();
  }

  void UnregisterAllExternalIdentities() {
    std::lock_guard<std::mutex> lifecycle_lock(external_identity_lifecycle_mu_);
    std::vector<std::shared_ptr<ExternalKvIdentityClient>> identities;
    {
      std::lock_guard<std::mutex> lock(external_identity_mu_);
      for (auto& kv : external_identities_) identities.push_back(std::move(kv.second));
      external_identities_.clear();
    }
    for (auto& identity : identities) {
      if (identity) identity->Stop();
    }
  }

  bool StartFdListener() {
    listen_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd_ < 0) {
      MORI_UMBP_ERROR("[StandaloneServer] fd socket() failed: {}", std::strerror(errno));
      return false;
    }

    sockaddr_un addr;
    socklen_t addr_len = 0;
    if (!FillSockaddr(fd_socket_path_, &addr, &addr_len)) {
      MORI_UMBP_ERROR("[StandaloneServer] invalid fd socket path {}", fd_socket_path_);
      close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }

    mode_t old_umask = umask(0077);
    int bind_rc = bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), addr_len);
    umask(old_umask);
    if (bind_rc != 0) {
      MORI_UMBP_ERROR("[StandaloneServer] bind('{}') failed: {}", fd_socket_path_,
                      std::strerror(errno));
      close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    chmod(fd_socket_path_.c_str(), 0600);

    if (listen(listen_fd_, 16) != 0) {
      MORI_UMBP_ERROR("[StandaloneServer] listen('{}') failed: {}", fd_socket_path_,
                      std::strerror(errno));
      close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }

    fd_running_.store(true);
    fd_thread_ = std::thread([this]() { FdAcceptLoop(); });
    return true;
  }

  void StopFdListener() {
    if (!fd_running_.exchange(false)) return;
    if (listen_fd_ >= 0) shutdown(listen_fd_, SHUT_RDWR);
    int active_fd = active_fd_client_.load();
    if (active_fd >= 0) shutdown(active_fd, SHUT_RDWR);
    if (fd_thread_.joinable()) fd_thread_.join();
    if (listen_fd_ >= 0) {
      close(listen_fd_);
      listen_fd_ = -1;
    }
  }

  void FdAcceptLoop() {
    while (fd_running_.load()) {
      int client_fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
      if (client_fd < 0) {
        if (fd_running_.load()) {
          MORI_UMBP_WARN("[StandaloneServer] accept fd socket failed: {}", std::strerror(errno));
        }
        continue;
      }
      active_fd_client_.store(client_fd);
      if (!SetFdSocketTimeouts(client_fd, FdHandshakeTimeout())) {
        MORI_UMBP_WARN("[StandaloneServer] failed to set fd socket timeout: {}",
                       std::strerror(errno));
      }
      HandleFdConnection(client_fd);
      active_fd_client_.store(-1);
      close(client_fd);
    }
  }

  void HandleFdConnection(int client_fd) {
    FdRegistrationMessage msg;
    std::string error;
    int fd = RecvFdRegistration(client_fd, &msg, &error);
    if (fd < 0) {
      MORI_UMBP_WARN("[StandaloneServer] fd registration receive failed: {}", error);
      SendStatus(client_fd, -1);
      return;
    }

    int32_t status = RegisterFd(fd, msg) ? 0 : -1;
    SendStatus(client_fd, status);
  }

  bool RegisterFd(int fd, const FdRegistrationMessage& msg) {
    void* mapped =
        mmap(nullptr, static_cast<size_t>(msg.size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
      MORI_UMBP_WARN("[StandaloneServer] mmap received fd failed: {}", std::strerror(errno));
      return false;
    }
    if (!RegisterBackendMemory(mapped, static_cast<size_t>(msg.size))) {
      MORI_UMBP_WARN("[StandaloneServer] backend RegisterMemory failed for client_id={}",
                     msg.client_id);
      munmap(mapped, static_cast<size_t>(msg.size));
      return false;
    }

    std::string client_id(msg.client_id);
    RegisteredMemory entry{mapped, static_cast<uint64_t>(msg.worker_base), msg.size};
    std::optional<RegisteredMemory> old_mem;
    {
      std::lock_guard<std::mutex> lock(memory_mu_);
      auto& regions = memory_[client_id];
      auto existing = std::find_if(
          regions.begin(), regions.end(),
          [&](const RegisteredMemory& mem) { return mem.worker_base == entry.worker_base; });
      if (existing != regions.end()) {
        // Same region re-registered: replace the mapping, release the old one.
        old_mem = *existing;
        *existing = entry;
      } else {
        regions.push_back(entry);
      }
    }
    // Release outside the lock, and via ReleaseRegisteredMemory so the backend
    // deregisters the region before its mapping is munmap'd (required by the
    // distributed backend; see design-standalone-process-mode.md §5.3/§6.3).
    if (old_mem.has_value()) ReleaseRegisteredMemory(*old_mem);
    MORI_UMBP_INFO("[StandaloneServer] registered shm client_id={} worker_base=0x{:x} size={}MB",
                   client_id, msg.worker_base, msg.size / (1024 * 1024));
    return true;
  }

  void UnmapClient(const std::string& client_id) {
    std::vector<RegisteredMemory> mems;
    {
      std::lock_guard<std::mutex> lock(memory_mu_);
      auto it = memory_.find(client_id);
      if (it == memory_.end()) return;
      mems.swap(it->second);
      memory_.erase(it);
    }
    for (const auto& mem : mems) ReleaseRegisteredMemory(mem);
  }

  void UnmapAll() {
    std::vector<RegisteredMemory> entries;
    {
      std::lock_guard<std::mutex> lock(memory_mu_);
      for (auto& kv : memory_) {
        for (const auto& mem : kv.second) entries.push_back(mem);
      }
      memory_.clear();
    }
    for (const auto& mem : entries) ReleaseRegisteredMemory(mem);
  }

  bool RegisterBackendMemory(void* base, size_t size) {
    std::lock_guard<std::mutex> lock(client_mu_);
    if (shutdown_.load()) return false;
    return client_->RegisterMemory(reinterpret_cast<uintptr_t>(base), size);
  }

  void ReleaseRegisteredMemory(const RegisteredMemory& mem) {
    if (!mem.base) return;
    {
      std::lock_guard<std::mutex> lock(client_mu_);
      client_->DeregisterMemory(reinterpret_cast<uintptr_t>(mem.base));
    }
    munmap(mem.base, static_cast<size_t>(mem.size));
  }

  // Resolves (client_id, region_base, offset) to a server-local pointer.
  // region_base selects which of the client's regions the offset is relative to;
  // 0 means legacy/single-region and falls back to the first region that fits.
  bool ResolveRange(const std::string& client_id, uint64_t region_base, uint64_t offset,
                    uint64_t size, uintptr_t* out_ptr) {
    if (!out_ptr || size == 0) return false;
    std::lock_guard<std::mutex> lock(memory_mu_);
    auto it = memory_.find(client_id);
    if (it == memory_.end()) return false;
    for (const auto& mem : it->second) {
      if (region_base != 0 && mem.worker_base != region_base) continue;
      if (offset > mem.size || size > mem.size - offset) {
        if (region_base != 0) return false;
        continue;
      }
      *out_ptr = reinterpret_cast<uintptr_t>(mem.base) + static_cast<uintptr_t>(offset);
      return true;
    }
    return false;
  }

  bool ResolveBatch(const ::umbp::BatchDataRequest& request, std::vector<uintptr_t>* ptrs) {
    // region_bases is optional for legacy single-region callers; when present it
    // must be parallel to keys.
    const bool has_region_bases = request.region_bases_size() > 0;
    if (!ptrs || request.keys_size() != request.shm_offsets_size() ||
        request.keys_size() != request.sizes_size() ||
        (has_region_bases && request.keys_size() != request.region_bases_size())) {
      return false;
    }
    ptrs->clear();
    ptrs->reserve(request.keys_size());
    for (int i = 0; i < request.keys_size(); ++i) {
      uintptr_t ptr = 0;
      uint64_t region_base = has_region_bases ? request.region_bases(i) : 0;
      if (!ResolveRange(request.client_id(), region_base, request.shm_offsets(i), request.sizes(i),
                        &ptr)) {
        return false;
      }
      ptrs->push_back(ptr);
    }
    return true;
  }

  static std::vector<size_t> Sizes(const ::umbp::BatchDataRequest& request) {
    std::vector<size_t> sizes;
    sizes.reserve(request.sizes_size());
    for (uint64_t size : request.sizes()) sizes.push_back(static_cast<size_t>(size));
    return sizes;
  }

  static void FillResults(const std::vector<bool>& results, ::umbp::BatchBoolResponse* response) {
    response->mutable_ok()->Reserve(static_cast<int>(results.size()));
    for (bool ok : results) response->add_ok(ok);
  }

  static void FillFalse(int n, ::umbp::BatchBoolResponse* response) {
    response->mutable_ok()->Reserve(n);
    for (int i = 0; i < n; ++i) response->add_ok(false);
  }

  static void FillExternalKvMatches(const std::vector<IUMBPClient::ExternalKvMatch>& matches,
                                    ::umbp::StandaloneMatchExternalKvResponse* response) {
    for (const auto& match : matches) {
      auto* out = response->add_matches();
      out->set_node_id(match.node_id);
      out->set_peer_address(match.peer_address);
      for (const auto& [tier, hashes] : match.hashes_by_tier) {
        auto* bucket = out->add_hashes_by_tier();
        bucket->set_tier(TierToProto(tier));
        for (const auto& hash : hashes) bucket->add_hashes(hash);
      }
    }
  }

  static void FillExternalKvHitCounts(
      const std::vector<IUMBPClient::ExternalKvHitCountEntry>& entries,
      ::umbp::StandaloneExternalKvHitCountsResponse* response) {
    for (const auto& entry : entries) {
      auto* out = response->add_entries();
      out->set_hash(entry.hash);
      out->set_hit_count_total(entry.hit_count_total);
    }
  }

  UMBPConfig backend_config_;
  std::unique_ptr<IUMBPClient> client_;
  std::string address_;
  std::string fd_socket_path_;
  std::unique_ptr<grpc::Server> server_;
  std::atomic<bool> shutdown_{false};

  std::mutex client_mu_;
  std::mutex memory_mu_;
  // A worker registers N non-contiguous host regions (e.g. DeepSeek-V4's KV
  // side pools), so each client_id maps to a list of regions, resolved by
  // worker_base at data-op time.
  std::map<std::string, std::vector<RegisteredMemory>> memory_;
  std::mutex external_identity_lifecycle_mu_;
  std::mutex external_identity_mu_;
  std::map<std::string, std::shared_ptr<ExternalKvIdentityClient>> external_identities_;

  std::atomic<bool> fd_running_{false};
  int listen_fd_ = -1;
  std::atomic<int> active_fd_client_{-1};
  std::thread fd_thread_;
};

StandaloneServer::StandaloneServer(UMBPConfig config, std::string address)
    : config_(std::move(config)), address_(std::move(address)) {
  impl_ = std::make_unique<Impl>(config_, address_);
}

StandaloneServer::~StandaloneServer() { Shutdown(); }

bool StandaloneServer::Start() { return impl_->Start(); }

void StandaloneServer::Run() { impl_->Run(); }

void StandaloneServer::Shutdown() {
  if (impl_) impl_->Shutdown();
}

}  // namespace mori::umbp::standalone
