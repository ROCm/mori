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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_copy.h"
#include "umbp/common/range_utils.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"

namespace mori::umbp::standalone {
namespace {

std::atomic<uint64_t> g_client_counter{0};

// Travels with a handle so the server can check it still stands for the list the
// caller means. A check, not the lookup: the client compares the keys in full,
// so a collision costs a resend and never a wrong read.
//
// Computed once per key set, on the call that already pays to serialize it.
// Zero is reserved to mean "do not bother remembering this set".
//
// A word at a time: byte-wise FNV-1a retires one byte per multiply latency,
// which at a thousand ~128-byte keys is tens of microseconds.
uint64_t FingerprintKeys(const std::vector<std::string>& keys) {
  constexpr uint64_t kMul1 = 0x9e3779b97f4a7c15ULL;
  constexpr uint64_t kMul2 = 0xc2b2ae3d27d4eb4fULL;
  uint64_t hash = 1469598103934665603ULL;
  const auto mix_word = [&hash](uint64_t word) {
    hash ^= word * kMul1;
    hash = ((hash << 31) | (hash >> 33)) * kMul2;
  };
  mix_word(keys.size());
  for (const std::string& key : keys) {
    // Length-delimited, so that concatenations that happen to agree do not.
    mix_word(key.size());
    const char* data = key.data();
    size_t i = 0;
    for (; i + 8 <= key.size(); i += 8) {
      uint64_t word;
      std::memcpy(&word, data + i, sizeof(word));  // unaligned-safe, no aliasing UB
      mix_word(word);
    }
    if (i < key.size()) {
      uint64_t tail = 0;
      std::memcpy(&tail, data + i, key.size() - i);
      mix_word(tail);
    }
  }
  return hash == 0 ? kMul1 : hash;
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

UMBPDeploymentMode BackendModeFromProto(::umbp::StandaloneBackendMode mode) {
  switch (mode) {
    case ::umbp::STANDALONE_BACKEND_LOCAL:
      return UMBPDeploymentMode::Local;
    case ::umbp::STANDALONE_BACKEND_DISTRIBUTED:
      return UMBPDeploymentMode::Distributed;
    case ::umbp::STANDALONE_BACKEND_UNKNOWN:
    default:
      return UMBPDeploymentMode::StandaloneProcess;
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

bool StandaloneProcessClient::WaitReady(int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
    ::umbp::Empty req;
    ::umbp::PingResponse resp;
    grpc::Status status = stub_->Ping(&ctx, req, &resp);
    if (status.ok() && resp.ready()) {
      backend_mode_ = BackendModeFromProto(resp.deployment_mode());
      supports_ranged_io_ = resp.supports_ranged_io();
      return true;
    }
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

bool StandaloneProcessClient::OffsetFor(uintptr_t ptr, size_t size, uint64_t* offset,
                                        uint64_t* region_base) const {
  std::lock_guard<std::mutex> lock(registration_mu_);
  for (const auto& region : regions_) {
    if (ptr < region.base) continue;
    uintptr_t rel = ptr - region.base;
    if (rel > region.size || size > region.size - rel) continue;
    *offset = static_cast<uint64_t>(rel);
    *region_base = static_cast<uint64_t>(region.base);
    return true;
  }
  return false;
}

bool StandaloneProcessClient::Put(const std::string& key, uintptr_t src, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  uint64_t offset = 0;
  uint64_t region_base = 0;
  if (!OffsetFor(src, size, &offset, &region_base)) return false;
  grpc::ClientContext ctx;
  ::umbp::PutRequest req;
  req.set_key(key);
  req.set_client_id(ClientId());
  req.set_shm_offset(offset);
  req.set_size(size);
  req.set_region_base(region_base);
  ::umbp::BoolResponse resp;
  grpc::Status status = stub_->Put(&ctx, req, &resp);
  return status.ok() && resp.ok();
}

bool StandaloneProcessClient::Get(const std::string& key, uintptr_t dst, size_t size) {
  if (closing_) return false;
  std::shared_lock lk(op_mutex_);
  if (closed_) return false;
  uint64_t offset = 0;
  uint64_t region_base = 0;
  if (!OffsetFor(dst, size, &offset, &region_base)) return false;
  grpc::ClientContext ctx;
  ::umbp::GetRequest req;
  req.set_key(key);
  req.set_client_id(ClientId());
  req.set_shm_offset(offset);
  req.set_size(size);
  req.set_region_base(region_base);
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
    uint64_t region_base = 0;
    if (!OffsetFor(srcs[i], sizes[i], &offset, &region_base)) {
      return std::vector<bool>(keys.size(), false);
    }
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_region_bases(region_base);
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
    uint64_t region_base = 0;
    if (!OffsetFor(srcs[i], sizes[i], &offset, &region_base)) {
      return std::vector<bool>(keys.size(), false);
    }
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_region_bases(region_base);
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
    uint64_t region_base = 0;
    if (!OffsetFor(dsts[i], sizes[i], &offset, &region_base)) {
      return std::vector<bool>(keys.size(), false);
    }
    req.add_keys(keys[i]);
    req.add_shm_offsets(offset);
    req.add_region_bases(region_base);
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

uint64_t StandaloneProcessClient::LookupKeyHandle(const std::vector<std::string>& keys,
                                                  uint64_t* fingerprint) {
  // Nothing to name, and nothing worth remembering: a zero fingerprint tells
  // the server not to mint a handle for it.
  if (keys.empty()) {
    *fingerprint = 0;
    return 0;
  }
  {
    std::lock_guard<std::mutex> lock(key_handle_mu_);
    for (size_t i = 0; i < key_handles_.size(); ++i) {
      // Size, then the two ends, then the whole thing: the cheap tests reject a
      // different set outright, so the full compare runs only for a hit.
      const KeyHandle& entry = key_handles_[i];
      if (entry.keys.size() != keys.size()) continue;
      if (entry.keys.front() != keys.front() || entry.keys.back() != keys.back()) continue;
      if (entry.keys != keys) continue;
      *fingerprint = entry.fingerprint;
      const uint64_t handle = entry.handle;
      if (i != 0)
        std::rotate(key_handles_.begin(), key_handles_.begin() + i, key_handles_.begin() + i + 1);
      return handle;
    }
  }
  *fingerprint = FingerprintKeys(keys);
  return 0;
}

void StandaloneProcessClient::RememberKeyHandle(const std::vector<std::string>& keys,
                                                uint64_t handle, uint64_t fingerprint) {
  if (handle == 0) return;
  std::lock_guard<std::mutex> lock(key_handle_mu_);
  key_handles_.insert(key_handles_.begin(), KeyHandle{keys, handle, fingerprint});
  if (key_handles_.size() > kKeyHandleSlots) key_handles_.resize(kKeyHandleSlots);
}

void StandaloneProcessClient::ForgetKeyHandle(uint64_t handle) {
  std::lock_guard<std::mutex> lock(key_handle_mu_);
  key_handles_.erase(std::remove_if(key_handles_.begin(), key_handles_.end(),
                                    [handle](const KeyHandle& e) { return e.handle == handle; }),
                     key_handles_.end());
}

std::vector<bool> StandaloneProcessClient::BatchGetRanges(
    const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dsts,
    const std::vector<std::vector<size_t>>& sizes,
    const std::vector<std::vector<size_t>>& src_offsets) {
  std::vector<bool> failed(keys.size(), false);
  if (closing_) return failed;
  std::shared_lock lk(op_mutex_);
  if (closed_ || !RangeBatchShapeValid(keys.size(), dsts, sizes, src_offsets)) return failed;

  uint64_t fingerprint = 0;
  uint64_t handle = LookupKeyHandle(keys, &fingerprint);

  ::umbp::BatchRangeDataRequest req;
  req.set_client_id(ClientId());
  req.set_key_fingerprint(fingerprint);
  for (size_t i = 0; i < keys.size(); ++i) {
    if (dsts[i].size() > std::numeric_limits<uint32_t>::max()) return failed;
    req.add_range_counts(static_cast<uint32_t>(dsts[i].size()));
    for (size_t j = 0; j < dsts[i].size(); ++j) {
      uint64_t shm_offset = 0;
      uint64_t region_base = 0;
      if (!OffsetFor(dsts[i][j], sizes[i][j], &shm_offset, &region_base)) return failed;
      req.add_shm_offsets(shm_offset);
      req.add_region_bases(region_base);
      req.add_sizes(sizes[i][j]);
      req.add_object_offsets(src_offsets[i][j]);
    }
  }

  // At most twice: once naming a handle, then -- only if the server no longer
  // holds it -- once carrying the keys.
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (handle != 0) {
      req.set_key_handle(handle);
      req.clear_keys();
    } else {
      req.set_key_handle(0);
      req.mutable_keys()->Reserve(static_cast<int>(keys.size()));
      for (const auto& key : keys) req.add_keys(key);
    }

    grpc::ClientContext ctx;
    ::umbp::BatchBoolResponse resp;
    const grpc::Status status = stub_->BatchGetRanges(&ctx, req, &resp);
    if (!status.ok()) return failed;
    if (resp.key_handle_unknown()) {
      ForgetKeyHandle(handle);
      handle = 0;
      continue;
    }
    if (resp.ok_size() != static_cast<int>(keys.size())) return failed;
    RememberKeyHandle(keys, resp.key_handle(), fingerprint);
    return std::vector<bool>(resp.ok().begin(), resp.ok().end());
  }
  return failed;
}

std::vector<bool> StandaloneProcessClient::BatchPutRanges(
    const std::vector<std::string>& keys, const std::vector<size_t>& object_sizes,
    const std::vector<std::vector<uintptr_t>>& srcs, const std::vector<std::vector<size_t>>& sizes,
    const std::vector<std::vector<size_t>>& dst_offsets) {
  std::vector<bool> failed(keys.size(), false);
  if (closing_) return failed;
  std::shared_lock lk(op_mutex_);
  if (closed_ || object_sizes.size() != keys.size() ||
      !RangeBatchShapeValid(keys.size(), srcs, sizes, dst_offsets)) {
    return failed;
  }

  ::umbp::BatchRangeDataRequest req;
  req.set_client_id(ClientId());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (srcs[i].size() > std::numeric_limits<uint32_t>::max()) return failed;
    req.add_keys(keys[i]);
    req.add_object_sizes(object_sizes[i]);
    req.add_range_counts(static_cast<uint32_t>(srcs[i].size()));
    for (size_t j = 0; j < srcs[i].size(); ++j) {
      uint64_t shm_offset = 0;
      uint64_t region_base = 0;
      if (!OffsetFor(srcs[i][j], sizes[i][j], &shm_offset, &region_base)) return failed;
      req.add_shm_offsets(shm_offset);
      req.add_region_bases(region_base);
      req.add_sizes(sizes[i][j]);
      req.add_object_offsets(dst_offsets[i][j]);
    }
  }

  grpc::ClientContext ctx;
  ::umbp::BatchBoolResponse resp;
  const grpc::Status status = stub_->BatchPutRanges(&ctx, req, &resp);
  if (!status.ok() || resp.ok_size() != static_cast<int>(keys.size())) return failed;
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
  } catch (const std::exception& error) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] deregistration during close failed: {}",
                    error.what());
  } catch (...) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] deregistration during close failed");
  }
  closed_ = true;
  stub_.reset();
  channel_.reset();
}

bool StandaloneProcessClient::RegisterMemory(uintptr_t ptr, size_t size,
                                             mori::io::MemoryLocationType loc, int device,
                                             MemoryRegistration /*mode*/) {
  if (closing_) return false;
  std::unique_lock lk(op_mutex_);
  if (closed_) return false;

  // Classification is authoritative, and `loc`/`device` only fill in what it
  // cannot know.  Two callers have to work here: ours, which states the
  // location explicitly, and a connector written against the two-argument
  // upstream API, which does not.  Trusting the pointer over the argument
  // means a caller who leaves `loc` at its CPU default cannot silently get a
  // device buffer registered as host memory — the failure that would then show
  // up as a memcpy from device memory, far from here.
  PointerLocation location = DetectPointerLocation(reinterpret_cast<void*>(ptr));
  if (loc == mori::io::MemoryLocationType::GPU && !location.IsDevice()) {
    MORI_UMBP_ERROR(
        "[StandaloneProcessClient] RegisterMemory: caller declared GPU for ptr=0x{:x} but it is "
        "not device memory",
        ptr);
    return false;
  }
  // A device ordinal from the caller is honoured when classification could not
  // supply one (hipPointerGetAttributes reports -1 for some allocations).
  if (location.IsDevice() && location.device_id < 0) location.device_id = device;

  if (location.IsDevice()) return RegisterDeviceMemory(ptr, size, location.device_id);
  return RegisterHostShmMemory(ptr, size);
}

bool StandaloneProcessClient::RegisterDeviceMemory(uintptr_t ptr, size_t size, int device_id) {
  if (ptr == 0 || size == 0) return false;
  ScopedHipDevice device_guard(device_id);
  if (!device_guard.IsValid()) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] failed to select GPU device {}", device_id);
    return false;
  }

  void* allocation_base = nullptr;
  size_t allocation_size = 0;
  const hipError_t range_status =
      hipMemGetAddressRange(reinterpret_cast<hipDeviceptr_t*>(&allocation_base), &allocation_size,
                            reinterpret_cast<hipDeviceptr_t>(ptr));
  if (range_status != hipSuccess || allocation_base == nullptr) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] hipMemGetAddressRange failed for ptr=0x{:x}: {}",
                    ptr, hipGetErrorString(range_status));
    (void)hipGetLastError();
    return false;
  }

  const uintptr_t allocation_address = reinterpret_cast<uintptr_t>(allocation_base);
  if (ptr < allocation_address) return false;
  const uint64_t ipc_offset = static_cast<uint64_t>(ptr - allocation_address);
  if (ipc_offset > allocation_size || size > allocation_size - ipc_offset) {
    MORI_UMBP_ERROR(
        "[StandaloneProcessClient] GPU registration range exceeds allocation: ptr=0x{:x} "
        "size={} alloc_base=0x{:x} alloc_size={}",
        ptr, size, allocation_address, allocation_size);
    return false;
  }

  hipIpcMemHandle_t handle{};
  const hipError_t handle_status = hipIpcGetMemHandle(&handle, allocation_base);
  if (handle_status != hipSuccess) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] hipIpcGetMemHandle failed for ptr=0x{:x}: {}", ptr,
                    hipGetErrorString(handle_status));
    (void)hipGetLastError();
    return false;
  }

  ::umbp::RegisterMemoryRequest req;
  req.set_client_id(ClientId());
  req.set_worker_base(ptr);
  req.set_size(size);
  req.set_worker_node_id(standalone_config_.worker_node_id);
  req.set_worker_node_address(standalone_config_.worker_node_address);
  for (const auto& tag : standalone_config_.tags) req.add_tags(tag);
  req.set_kind(::umbp::MEMORY_KIND_GPU_IPC);
  req.set_device_id(device_id);
  req.set_ipc_handle(reinterpret_cast<const char*>(&handle), sizeof(handle));
  req.set_ipc_offset(ipc_offset);
  req.set_alloc_base(allocation_address);

  grpc::ClientContext ctx;
  ::umbp::BoolResponse resp;
  const grpc::Status rpc_status = stub_->RegisterMemory(&ctx, req, &resp);
  if (!rpc_status.ok() || !resp.ok()) {
    MORI_UMBP_ERROR("[StandaloneProcessClient] GPU RegisterMemory failed: {}",
                    rpc_status.ok() ? resp.error() : rpc_status.error_message());
    return false;
  }

  std::lock_guard<std::mutex> lock(registration_mu_);
  auto existing = std::find_if(regions_.begin(), regions_.end(), [&](const RegisteredRegion& r) {
    return r.base == ptr && r.kind == RegionKind::kGpuIpc;
  });
  if (existing != regions_.end()) {
    existing->size = size;
  } else {
    regions_.push_back({ptr, size, RegionKind::kGpuIpc});
  }
  return true;
}

bool StandaloneProcessClient::RegisterHostShmMemory(uintptr_t ptr, size_t size) {
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

    const uintptr_t base = reinterpret_cast<uintptr_t>(allocation->base);
    std::lock_guard<std::mutex> lock(registration_mu_);
    auto existing = std::find_if(regions_.begin(), regions_.end(),
                                 [&](const RegisteredRegion& r) { return r.base == base; });
    if (existing != regions_.end()) {
      // Same region re-registered: keep the existing entry and drop the freshly
      // acquired duplicate allocation (its refcount is balanced by the Release).
      existing->size = allocation->mapped_size;
      HostMemAllocator::ReleaseShmAllocation(base);
    } else {
      regions_.push_back({base, allocation->mapped_size, RegionKind::kHostShm});
    }
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
  std::vector<RegisteredRegion> regions;
  {
    std::lock_guard<std::mutex> lock(registration_mu_);
    if (regions_.empty()) return;
    client_id = client_id_;
    regions = regions_;
  }

  // One RPC tears down all of this client's regions server-side (UnmapClient),
  // so a single DeregisterMemory call covers every region.
  grpc::ClientContext ctx;
  ::umbp::DeregisterMemoryRequest req;
  req.set_client_id(client_id);
  ::umbp::Empty resp;
  if (!stub_) throw std::runtime_error("standalone DeregisterMemory RPC has no active stub");
  const grpc::Status status = stub_->DeregisterMemory(&ctx, req, &resp);
  if (!status.ok()) {
    throw std::runtime_error("standalone DeregisterMemory RPC failed: " + status.error_message());
  }

  {
    std::lock_guard<std::mutex> lock(registration_mu_);
    regions_.clear();
  }
  for (const auto& region : regions) {
    if (region.kind == RegionKind::kHostShm) {
      HostMemAllocator::ReleaseShmAllocation(region.base);
    }
  }
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
