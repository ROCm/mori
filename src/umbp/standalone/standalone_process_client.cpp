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
#include "umbp/standalone/standalone_process_client.h"

#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"

namespace mori::umbp::standalone {
namespace {

std::atomic<uint64_t> g_client_counter{0};

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

bool IsLocalRankZero() {
  for (const char* name :
       {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID"}) {
    const char* value = std::getenv(name);
    if (value) return std::atoi(value) == 0;
  }
  return true;
}

std::string BootstrapLockPath() {
  const char* dir = std::getenv("UMBP_STANDALONE_SHM_DIR");
  std::string base = (dir && dir[0] != '\0') ? dir : "/tmp";
  if (!base.empty() && base.back() == '/') base.pop_back();
  return base + "/umbp_standalone_bootstrap.lock";
}

std::string FindStandaloneServerBinary() {
  const char* env = std::getenv("UMBP_STANDALONE_BIN");
  return (env && env[0] != '\0') ? env : "umbp_standalone_server";
}

void SetEnv(const char* name, const std::string& value) {
  if (!value.empty()) setenv(name, value.c_str(), 1);
}

void SetEnv(const char* name, size_t value) { setenv(name, std::to_string(value).c_str(), 1); }

void SetEnv(const char* name, int value) { setenv(name, std::to_string(value).c_str(), 1); }

void SetEnv(const char* name, bool value) { setenv(name, value ? "1" : "0", 1); }

void SetEnv(const char* name, double value) { setenv(name, std::to_string(value).c_str(), 1); }

void ExportServerEnv(const UMBPConfig& config, const std::string& address) {
  SetEnv("UMBP_STANDALONE_ADDRESS", address);
  SetEnv("UMBP_ROLE", "standalone");
  SetEnv("UMBP_DRAM_CAPACITY", config.dram.capacity_bytes);
  SetEnv("UMBP_DRAM_USE_HUGEPAGES", config.dram.use_hugepages);
  SetEnv("UMBP_DRAM_HUGEPAGE_SIZE", config.dram.hugepage_size);
  SetEnv("UMBP_DRAM_NUMA_NODE", config.dram.numa_node);
  SetEnv("UMBP_DRAM_PREFAULT", config.dram.prefault);
  SetEnv("UMBP_DRAM_HIGH_WM", config.dram.high_watermark);
  SetEnv("UMBP_DRAM_LOW_WM", config.dram.low_watermark);
  SetEnv("UMBP_SSD_ENABLED", config.ssd.enabled);
  SetEnv("UMBP_SSD_DIR", config.ssd.storage_dir);
  SetEnv("UMBP_SSD_CAPACITY", config.ssd.capacity_bytes);
  SetEnv("UMBP_SSD_BACKEND", config.ssd.ssd_backend);
  SetEnv("UMBP_SSD_HIGH_WM", config.ssd.high_watermark);
  SetEnv("UMBP_SSD_LOW_WM", config.ssd.low_watermark);
  SetEnv("UMBP_EVICTION_POLICY", config.eviction.policy);
  SetEnv("UMBP_SPDK_BDEV", config.ssd.spdk_bdev_name);
  SetEnv("UMBP_SPDK_REACTOR_MASK", config.ssd.spdk_reactor_mask);
  SetEnv("UMBP_SPDK_MEM_MB", config.ssd.spdk_mem_size_mb);
  SetEnv("UMBP_SPDK_NVME_PCI", config.ssd.spdk_nvme_pci_addr);
  SetEnv("UMBP_SPDK_NVME_CTRL", config.ssd.spdk_nvme_ctrl_name);
  SetEnv("UMBP_SPDK_IO_WORKERS", config.ssd.spdk_io_workers);
  SetEnv("UMBP_SPDK_PROXY_SHM", config.ssd.spdk_proxy_shm_name);
  SetEnv("UMBP_SPDK_PROXY_BIN", config.ssd.spdk_proxy_bin);
  SetEnv("UMBP_SPDK_PROXY_TENANT_ID", static_cast<int>(config.ssd.spdk_proxy_tenant_id));
  SetEnv("UMBP_SPDK_PROXY_TENANT_QUOTA_BYTES", config.ssd.spdk_proxy_tenant_quota_bytes);
  SetEnv("UMBP_SPDK_PROXY_MAX_CHANNELS", static_cast<int>(config.ssd.spdk_proxy_max_channels));
  SetEnv("UMBP_SPDK_PROXY_DATA_PER_CHANNEL_MB", config.ssd.spdk_proxy_data_per_channel_mb);
  SetEnv("UMBP_SPDK_PROXY_TIMEOUT_MS", config.ssd.spdk_proxy_startup_timeout_ms);
  SetEnv("UMBP_SPDK_PROXY_AUTO_START", config.ssd.spdk_proxy_auto_start);
  SetEnv("UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS", config.ssd.spdk_proxy_idle_exit_timeout_ms);
  SetEnv("UMBP_SPDK_PROXY_ALLOW_BORROW", config.ssd.spdk_proxy_allow_borrow);
  SetEnv("UMBP_SPDK_PROXY_RESERVED_SHARED_BYTES", config.ssd.spdk_proxy_reserved_shared_bytes);
}

class ScopedBootstrapLock {
 public:
  ScopedBootstrapLock() {
    std::string path = BootstrapLockPath();
    fd_ = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ >= 0 && flock(fd_, LOCK_EX) != 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  ~ScopedBootstrapLock() {
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
  }

  bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

}  // namespace

StandaloneProcessClient::StandaloneProcessClient(const UMBPConfig& config) : config_(config) {
  if (!config_.standalone_process.has_value()) {
    throw std::runtime_error("StandaloneProcessClient requires UMBPConfig::standalone_process");
  }
  standalone_config_ = config_.standalone_process.value();
  std::string error_message;
  if (!config_.Validate(&error_message)) {
    throw std::runtime_error("invalid UMBP config: " + error_message);
  }

  address_ = standalone_config_.address;
  fd_socket_path_ = DeriveFdSocketPath(address_);
  channel_ = grpc::CreateChannel(address_, grpc::InsecureChannelCredentials());
  stub_ = ::umbp::UMBPStandalone::NewStub(channel_);

  MaybeAutoStart();
  if (!WaitReady(standalone_config_.startup_timeout_ms)) {
    throw std::runtime_error("StandaloneProcessClient: server is not ready at " + address_);
  }

  MORI_UMBP_INFO("[StandaloneProcessClient] connected address={} fd_socket={}", address_,
                 fd_socket_path_);
}

StandaloneProcessClient::~StandaloneProcessClient() { Close(); }

std::string StandaloneProcessClient::ClientId() {
  std::lock_guard<std::mutex> lock(registration_mu_);
  if (!client_id_.empty()) return client_id_;
  std::ostringstream oss;
  oss << "umbp-" << getpid() << "-" << g_client_counter.fetch_add(1);
  client_id_ = oss.str();
  return client_id_;
}

bool StandaloneProcessClient::WaitReady(int timeout_ms) const {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
    ::umbp::Empty req;
    ::umbp::PingResponse resp;
    grpc::Status status = stub_->Ping(&ctx, req, &resp);
    if (status.ok() && resp.ready()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

void StandaloneProcessClient::MaybeAutoStart() {
  if (WaitReady(200)) return;
  if (!standalone_config_.auto_start) return;

  ScopedBootstrapLock lock;
  if (!lock.valid()) {
    MORI_UMBP_WARN("[StandaloneProcessClient] bootstrap lock unavailable; waiting for server");
    return;
  }
  if (WaitReady(200)) return;
  if (!IsLocalRankZero()) return;

  std::string bin = FindStandaloneServerBinary();
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("StandaloneProcessClient: fork() failed: " +
                             std::string(std::strerror(errno)));
  }
  if (pid == 0) {
    setsid();
    ExportServerEnv(config_, address_);
    execlp(bin.c_str(), "umbp_standalone_server", address_.c_str(), static_cast<char*>(nullptr));
    fprintf(stderr, "[UMBP ERROR] execlp('%s') failed: %s\n", bin.c_str(), std::strerror(errno));
    _exit(127);
  }
  MORI_UMBP_INFO(
      "[StandaloneProcessClient] spawned umbp_standalone_server pid={} bin={} address={}", pid, bin,
      address_);
}

bool StandaloneProcessClient::OffsetFor(uintptr_t ptr, size_t size, uint64_t* offset) const {
  std::lock_guard<std::mutex> lock(registration_mu_);
  if (!registered_ || ptr < registered_base_) return false;
  uintptr_t rel = ptr - registered_base_;
  if (rel > registered_size_ || size > registered_size_ - rel) return false;
  *offset = static_cast<uint64_t>(rel);
  return true;
}

bool StandaloneProcessClient::Put(const std::string& key, uintptr_t src, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  uint64_t offset = 0;
  if (!OffsetFor(src, size, &offset)) return false;
  grpc::ClientContext ctx;
  ::umbp::PutRequest req;
  req.set_key(key);
  req.set_client_id(ClientId());
  req.set_shm_offset(offset);
  req.set_size(size);
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Put(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::Get(const std::string& key, uintptr_t dst, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  uint64_t offset = 0;
  if (!OffsetFor(dst, size, &offset)) return false;
  grpc::ClientContext ctx;
  ::umbp::GetRequest req;
  req.set_key(key);
  req.set_client_id(ClientId());
  req.set_shm_offset(offset);
  req.set_size(size);
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Get(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::Exists(const std::string& key) const {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  grpc::ClientContext ctx;
  ::umbp::KeyRequest req;
  req.set_key(key);
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Exists(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

std::vector<bool> StandaloneProcessClient::BatchPut(const std::vector<std::string>& keys,
                                                    const std::vector<uintptr_t>& srcs,
                                                    const std::vector<size_t>& sizes) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_ || keys.size() != srcs.size() || keys.size() != sizes.size()) {
    return std::vector<bool>(keys.size(), false);
  }
  ::umbp::BatchDataRequest req;
  req.set_client_id(ClientId());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t offset = 0;
    if (!OffsetFor(srcs[i], sizes[i], &offset)) return std::vector<bool>(keys.size(), false);
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_sizes(sizes[i]);
  }
  grpc::ClientContext ctx;
  ::umbp::BatchBoolResponse resp;
  grpc::Status status = stub_->BatchPut(&ctx, req, &resp);
  if (!status.ok() || resp.ok_size() != static_cast<int>(keys.size())) {
    return std::vector<bool>(keys.size(), false);
  }
  return std::vector<bool>(resp.ok().begin(), resp.ok().end());
}

std::vector<bool> StandaloneProcessClient::BatchPutWithDepth(const std::vector<std::string>& keys,
                                                             const std::vector<uintptr_t>& srcs,
                                                             const std::vector<size_t>& sizes,
                                                             const std::vector<int>& depths) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_ || keys.size() != srcs.size() || keys.size() != sizes.size()) {
    return std::vector<bool>(keys.size(), false);
  }
  ::umbp::BatchDataWithDepthRequest req;
  req.set_client_id(ClientId());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t offset = 0;
    if (!OffsetFor(srcs[i], sizes[i], &offset)) return std::vector<bool>(keys.size(), false);
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_sizes(sizes[i]);
    req.add_depths(i < depths.size() ? depths[i] : -1);
  }
  grpc::ClientContext ctx;
  ::umbp::BatchBoolResponse resp;
  grpc::Status status = stub_->BatchPutWithDepth(&ctx, req, &resp);
  if (!status.ok() || resp.ok_size() != static_cast<int>(keys.size())) {
    return std::vector<bool>(keys.size(), false);
  }
  return std::vector<bool>(resp.ok().begin(), resp.ok().end());
}

std::vector<bool> StandaloneProcessClient::BatchGet(const std::vector<std::string>& keys,
                                                    const std::vector<uintptr_t>& dsts,
                                                    const std::vector<size_t>& sizes) {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_ || keys.size() != dsts.size() || keys.size() != sizes.size()) {
    return std::vector<bool>(keys.size(), false);
  }
  ::umbp::BatchDataRequest req;
  req.set_client_id(ClientId());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t offset = 0;
    if (!OffsetFor(dsts[i], sizes[i], &offset)) return std::vector<bool>(keys.size(), false);
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_sizes(sizes[i]);
  }
  grpc::ClientContext ctx;
  ::umbp::BatchBoolResponse resp;
  grpc::Status status = stub_->BatchGet(&ctx, req, &resp);
  if (!status.ok() || resp.ok_size() != static_cast<int>(keys.size())) {
    return std::vector<bool>(keys.size(), false);
  }
  return std::vector<bool>(resp.ok().begin(), resp.ok().end());
}

std::vector<bool> StandaloneProcessClient::BatchExists(const std::vector<std::string>& keys) const {
  if (closing_) return std::vector<bool>(keys.size(), false);
  std::shared_lock lk(op_mutex_);
  if (closed_) return std::vector<bool>(keys.size(), false);
  grpc::ClientContext ctx;
  ::umbp::BatchKeysRequest req;
  for (const auto& key : keys) req.add_keys(key);
  ::umbp::BatchBoolResponse resp;
  grpc::Status status = stub_->BatchExists(&ctx, req, &resp);
  if (!status.ok() || resp.ok_size() != static_cast<int>(keys.size())) {
    return std::vector<bool>(keys.size(), false);
  }
  return std::vector<bool>(resp.ok().begin(), resp.ok().end());
}

size_t StandaloneProcessClient::BatchExistsConsecutive(const std::vector<std::string>& keys) const {
  if (closing_) return 0;
  std::shared_lock lk(op_mutex_);
  if (closed_) return 0;
  grpc::ClientContext ctx;
  ::umbp::BatchKeysRequest req;
  for (const auto& key : keys) req.add_keys(key);
  ::umbp::CountResponse resp;
  grpc::Status status = stub_->BatchExistsConsecutive(&ctx, req, &resp);
  return status.ok() ? static_cast<size_t>(resp.count()) : 0;
}

bool StandaloneProcessClient::Clear() {
  if (closing_) return true;
  std::unique_lock lk(op_mutex_);
  if (closed_) return true;
  grpc::ClientContext ctx;
  ::umbp::Empty req;
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Clear(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::Flush() {
  if (closing_) return true;
  std::shared_lock lk(op_mutex_);
  if (closed_) return true;
  grpc::ClientContext ctx;
  ::umbp::Empty req;
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Flush(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

void StandaloneProcessClient::Close() {
  closing_ = true;
  std::unique_lock lk(op_mutex_);
  if (closed_) return;
  try {
    DeregisterMemoryLocked();
  } catch (...) {
  }
  closed_ = true;
  stub_.reset();
  channel_.reset();
}

bool StandaloneProcessClient::RegisterMemory(uintptr_t ptr, size_t size) {
  if (closing_) return false;
  std::unique_lock lk(op_mutex_);
  if (closed_) return false;

  auto allocation = HostMemAllocator::AcquireShmAllocation(ptr, size);
  if (!allocation.has_value()) {
    throw std::runtime_error(
        "StandaloneProcessClient::RegisterMemory requires an AnonymousShm-backed host buffer");
  }

  bool acquired_kept = false;
  const std::string client_id = ClientId();
  try {
    std::string error;
    int status = SendFdRegistration(
        fd_socket_path_, allocation->fd, client_id, reinterpret_cast<uintptr_t>(allocation->base),
        allocation->mapped_size, standalone_config_.startup_timeout_ms, &error);
    if (status != 0) {
      throw std::runtime_error("fd handoff failed: " + error);
    }

    grpc::ClientContext ctx;
    ::umbp::RegisterMemoryRequest req;
    req.set_client_id(client_id);
    req.set_worker_base(reinterpret_cast<uintptr_t>(allocation->base));
    req.set_size(allocation->mapped_size);
    req.set_worker_node_id(standalone_config_.worker_node_id);
    req.set_worker_node_address(standalone_config_.worker_node_address);
    for (const auto& tag : standalone_config_.tags) req.add_tags(tag);
    ::umbp::BoolResponse resp;
    grpc::Status rpc_status = stub_->RegisterMemory(&ctx, req, &resp);
    if (!rpc_status.ok() || !resp.ok()) {
      throw std::runtime_error("standalone RegisterMemory RPC failed: " +
                               (rpc_status.ok() ? resp.error() : rpc_status.error_message()));
    }

    std::lock_guard<std::mutex> lock(registration_mu_);
    if (registered_) HostMemAllocator::ReleaseShmAllocation(registered_base_);
    registered_base_ = reinterpret_cast<uintptr_t>(allocation->base);
    registered_size_ = allocation->mapped_size;
    registered_ = true;
    acquired_kept = true;
  } catch (...) {
    if (!acquired_kept) {
      HostMemAllocator::ReleaseShmAllocation(reinterpret_cast<uintptr_t>(allocation->base));
    }
    throw;
  }
  return true;
}

void StandaloneProcessClient::DeregisterMemoryLocked() {
  std::string client_id;
  uintptr_t base = 0;
  {
    std::lock_guard<std::mutex> lock(registration_mu_);
    if (!registered_) return;
    client_id = client_id_;
    base = registered_base_;
    registered_ = false;
    registered_base_ = 0;
    registered_size_ = 0;
  }

  grpc::ClientContext ctx;
  ::umbp::DeregisterMemoryRequest req;
  req.set_client_id(client_id);
  ::umbp::Empty resp;
  if (stub_) stub_->DeregisterMemory(&ctx, req, &resp);
  HostMemAllocator::ReleaseShmAllocation(base);
}

void StandaloneProcessClient::DeregisterMemory(uintptr_t /*ptr*/) {
  if (closing_) return;
  std::unique_lock lk(op_mutex_);
  if (closed_) return;
  DeregisterMemoryLocked();
}

bool StandaloneProcessClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes,
                                                     TierType tier) {
  grpc::ClientContext ctx;
  ::umbp::StandaloneExternalKvMutationRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  req.set_tier(TierToProto(tier));
  req.set_client_id(ClientId());
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->ReportExternalKvBlocks(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes,
                                                     TierType tier) {
  grpc::ClientContext ctx;
  ::umbp::StandaloneExternalKvMutationRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  req.set_tier(TierToProto(tier));
  req.set_client_id(ClientId());
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->RevokeExternalKvBlocks(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  grpc::ClientContext ctx;
  ::umbp::StandaloneExternalKvTierRequest req;
  req.set_tier(TierToProto(tier));
  req.set_client_id(ClientId());
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->RevokeAllExternalKvBlocksAtTier(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

std::vector<IUMBPClient::ExternalKvMatch> StandaloneProcessClient::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit) {
  grpc::ClientContext ctx;
  ::umbp::StandaloneMatchExternalKvRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  req.set_count_as_hit(count_as_hit);
  req.set_client_id(ClientId());
  ::umbp::StandaloneMatchExternalKvResponse resp;
  grpc::Status status = stub_->MatchExternalKv(&ctx, req, &resp);
  if (!status.ok()) return {};

  std::vector<IUMBPClient::ExternalKvMatch> out;
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

std::vector<IUMBPClient::ExternalKvHitCountEntry> StandaloneProcessClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes) {
  grpc::ClientContext ctx;
  ::umbp::StandaloneExternalKvHitCountsRequest req;
  for (const auto& hash : hashes) req.add_hashes(hash);
  req.set_client_id(ClientId());
  ::umbp::StandaloneExternalKvHitCountsResponse resp;
  grpc::Status status = stub_->GetExternalKvHitCounts(&ctx, req, &resp);
  if (!status.ok()) return {};
  std::vector<IUMBPClient::ExternalKvHitCountEntry> out;
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
