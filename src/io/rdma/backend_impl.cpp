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
#include "src/io/rdma/backend_impl.hpp"

#include <sys/epoll.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"
#include "src/io/rdma/protocol.hpp"
namespace mori {
namespace io {

static void ValidateRdmaNotificationConfig(const RdmaBackendConfig& config) {
  if (config.enableNotification && config.notifPerQp == 0) {
    MORI_IO_ERROR(
        "Invalid RDMA config: notifPerQp must be >= 1 when notification is enabled; got {}",
        config.notifPerQp);
    throw std::runtime_error(
        "Invalid RDMA config: notifPerQp must be >= 1 when notification is "
        "enabled");
  }
}

enum class CqeFailureOrigin : uint8_t {
  BatchTransfer = 0,
  NotificationSend,
  NotificationRecv,
  Unknown,
};

struct CqeFailureAdvice {
  const char* statusText{nullptr};
  std::string hint;

  bool HasHint() const { return !hint.empty(); }

  std::string ComposeStatusMessage() const {
    std::string message = statusText != nullptr ? statusText : "unknown";
    if (hint.empty()) return message;
    message += " Hint: ";
    message += hint;
    return message;
  }
};

static void LogAsyncTransferFailureIfNeeded(internal::IoCallDiagnostics* diagnostics, uint32_t code,
                                            const std::string& message) {
  if (diagnostics == nullptr || diagnostics->Label() == nullptr) return;

  internal::IoFailureKind failureKind = diagnostics->CurrentFailureKind();
  if (failureKind != internal::IoFailureKind::FlushCascade ||
      !diagnostics->TryMarkLogged(failureKind)) {
    return;
  }

  MORI_IO_DEBUG("{} error {} message {}", diagnostics->Label(), code, message);
}

static CqeFailureOrigin ClassifyCqeFailureOrigin(uint64_t wrId, uint32_t notifPerQp) {
  if (IsNotifSendWrId(wrId)) return CqeFailureOrigin::NotificationSend;
  if (wrId < notifPerQp) return CqeFailureOrigin::NotificationRecv;
  return CqeFailureOrigin::BatchTransfer;
}

static CqeFailureAdvice DescribeCqeFailure(ibv_wc_status status, CqeFailureOrigin origin,
                                           const RdmaBackendConfig& config) {
  CqeFailureAdvice advice{ibv_wc_status_str(status), {}};
  switch (status) {
    case IBV_WC_RETRY_EXC_ERR:
      advice.hint =
          "transport retry limit exceeded; check peer liveness/connectivity, verify GID "
          "selection (unset or correct MORI_IB_GID_INDEX), and if running RoCE verify QoS "
          "settings such as MORI_IO_SL/MORI_IO_TC or MORI_RDMA_SL/MORI_RDMA_TC.";
      break;
    case IBV_WC_RNR_RETRY_EXC_ERR:
      if (origin == CqeFailureOrigin::NotificationSend) {
        advice.hint =
            "receiver not ready for SEND completions; if notifications are enabled, ensure the "
            "peer pre-posts enough RECV WRs. Try increasing notifPerQp / MORI_IO_QP_MAX_RECV_WR "
            "(current notifPerQp=" +
            std::to_string(config.notifPerQp) +
            "), or set MORI_IO_ENABLE_NOTIFICATION=0 if inbound notification is not required.";
      } else {
        advice.hint =
            "receiver not ready; check the peer receive path. If this is related to MORI "
            "notifications, increase notifPerQp / MORI_IO_QP_MAX_RECV_WR or disable "
            "MORI_IO_ENABLE_NOTIFICATION when inbound notification is not required.";
      }
      break;
    case IBV_WC_LOC_PROT_ERR:
      advice.hint =
          "local protection error; verify the local buffer is still registered with MORI, lkey "
          "matches the posted WR, and transfer offsets/lengths stay within the registered range.";
      break;
    case IBV_WC_LOC_LEN_ERR:
      advice.hint =
          "local length error; verify SGE lengths and transfer offsets stay within the registered "
          "local MR bounds.";
      break;
    case IBV_WC_REM_ACCESS_ERR:
      advice.hint =
          "remote access error; verify the remote buffer is still registered, rkey/permissions "
          "allow this operation, and remote offsets/lengths stay within the registered range.";
      break;
    case IBV_WC_REM_OP_ERR:
      advice.hint =
          "remote operation error; verify both peers use compatible verbs/QP state and the remote "
          "endpoint supports the requested RDMA operation.";
      break;
    default:
      break;
  }
  return advice;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */

RdmaManager::RdmaManager(const RdmaBackendConfig cfg, application::RdmaContext* ctx)
    : config(cfg), ctx(ctx) {
  application::RdmaDeviceList devices = ctx->GetRdmaDeviceList();
  availDevices = GetActiveDevicePortList(devices);
  assert(availDevices.size() > 0);

  deviceCtxs.resize(availDevices.size(), nullptr);
  topo.reset(new application::TopoSystem());
}

RdmaManager::~RdmaManager() {
  localRegistrations_.clear();

  for (auto* devCtx : deviceCtxs) {
    if (devCtx != nullptr) {
      delete devCtx;
    }
  }
  deviceCtxs.clear();

  if (ctx != nullptr) {
    delete ctx;
    ctx = nullptr;
  }
}

std::vector<std::pair<int, int>> RdmaManager::Search(TopoKey key) {
  if (key.loc == MemoryLocationType::GPU) {
    std::string nicName = topo->MatchGpuAndNic(key.deviceId);
    assert(!nicName.empty());
    for (int i = 0; i < availDevices.size(); i++) {
      if (availDevices[i].first->Name() == nicName) {
        return {{i, 1}};
      }
    }
    MORI_IO_WARN("No matching NIC found for GPU {}, nicName: {}", key.deviceId, nicName);
  } else if (key.loc == MemoryLocationType::CPU) {
    if (availDevices.empty()) return {};
    const char* envNic = std::getenv("MORI_IO_RDMA_NIC_IDX");
    if (envNic) {
      int idx = std::atoi(envNic);
      if (idx >= 0 && idx < static_cast<int>(availDevices.size())) {
        return {{idx, 1}};
      }
      MORI_IO_WARN("MORI_IO_RDMA_NIC_IDX={} out of range [0, {}), falling back to round-robin", idx,
                   availDevices.size());
    }
    int idx = (roundRobinCounter.fetch_add(1, std::memory_order_relaxed) % availDevices.size());
    return {{idx, 1}};
  }
  MORI_IO_ERROR(
      "topo searching for device other than CPU/GPU is not implemented yet, returning default "
      "device 0");
  return {{0, 1}};
}

/* ----------------------------------- Local Memory Management ---------------------------------- */
int RdmaManager::PickRdmaDeviceForMemory(const MemoryDesc& desc) {
  TopoKey tkey{desc.deviceId, desc.loc};
  auto candidates = Search(tkey);
  assert(!candidates.empty());
  return candidates[0].first;
}

size_t RdmaManager::GetEffectiveChunkSize(application::RdmaDeviceContext* devCtx) {
  auto* device = devCtx->GetRdmaDevice();
  const auto* attr = device->GetDeviceAttr();
  size_t deviceMaxMrSize = static_cast<size_t>(attr->orig_attr.max_mr_size);

  size_t effective = deviceMaxMrSize;
  const char* envOverride = std::getenv("MORI_IO_RDMA_MR_CHUNK_SIZE");
  if (envOverride) {
    size_t overrideVal = std::strtoull(envOverride, nullptr, 10);
    if (overrideVal > 0) {
      effective = (deviceMaxMrSize > 0) ? std::min(deviceMaxMrSize, overrideVal) : overrideVal;
    }
  }
  return effective;
}

std::shared_ptr<const RdmaLocalMemoryRegistration> RdmaManager::GetOrMaterializeLocalRegistration(
    const MemoryDesc& desc) {
  // Fast path: check if already materialized
  {
    std::shared_lock<std::shared_mutex> lock(mu);
    auto it = localRegistrations_.find(desc.id);
    if (it != localRegistrations_.end()) return it->second;
  }

  // Get or create per-memory mutex for single-flight materialization
  std::shared_ptr<std::mutex> matMu;
  {
    std::unique_lock<std::shared_mutex> lock(mu);
    auto& ptr = materializationMutexes_[desc.id];
    if (!ptr) ptr = std::make_shared<std::mutex>();
    matMu = ptr;
  }

  std::lock_guard<std::mutex> matLock(*matMu);

  // Re-check after acquiring the per-memory lock
  {
    std::shared_lock<std::shared_mutex> lock(mu);
    auto it = localRegistrations_.find(desc.id);
    if (it != localRegistrations_.end()) return it->second;
  }

  // Pin device affinity
  int rdmaDevId;
  {
    std::unique_lock<std::shared_mutex> lock(mu);
    auto affIt = memoryDeviceAffinity_.find(desc.id);
    if (affIt != memoryDeviceAffinity_.end()) {
      rdmaDevId = affIt->second;
    } else {
      rdmaDevId = PickRdmaDeviceForMemory(desc);
      memoryDeviceAffinity_[desc.id] = rdmaDevId;
    }
  }

  // Compute chunk plan and register MRs outside the manager lock
  application::RdmaDeviceContext* devCtx;
  {
    std::unique_lock<std::shared_mutex> lock(mu);
    devCtx = GetOrCreateDeviceContext(rdmaDevId);
  }

  size_t chunkSize = GetEffectiveChunkSize(devCtx);
  if (chunkSize == 0 || desc.size <= chunkSize) {
    // Try single MR first
    auto ownedMr = devCtx->TryRegisterOwnedMr(reinterpret_cast<void*>(desc.data), desc.size);
    if (ownedMr.has_value()) {
      RdmaMemoryLayout layout;
      layout.rdmaDevId = rdmaDevId;
      layout.baseAddr = desc.data;
      layout.logicalSize = desc.size;
      layout.chunks.push_back({0, ownedMr->Region()});

      std::vector<application::RdmaOwnedMr> ownedMrs;
      ownedMrs.push_back(std::move(*ownedMr));

      auto reg =
          std::make_shared<RdmaLocalMemoryRegistration>(std::move(layout), std::move(ownedMrs));
      std::unique_lock<std::shared_mutex> lock(mu);
      localRegistrations_[desc.id] = reg;
      return reg;
    }
    if (chunkSize == 0) {
      MORI_IO_ERROR("Failed to register single MR for memory {} and no chunk size available",
                    desc.id);
      throw std::runtime_error("Failed to register RDMA memory region");
    }
    // Fall through to chunked registration
  }

  // Chunked registration
  //
  // On AMD AINIC (ionic, vendor_id 0x1dd8 / Pensando), the firmware silently
  // drops the last ~64 bytes of any RDMA WRITE that lands on a chunked MR's
  // tail when the user-supplied buffer's base VA is NOT 4 KiB page-aligned.
  // The CQE reports IBV_WC_SUCCESS but the destination bytes retain their
  // pre-write content. Triggered specifically when:
  //   1. CPU memory (kernel pin path; GPU goes through dmabuf importer)
  //   2. Per-chunk MR size >= 1 GiB (firmware path switch)
  //   3. Buffer base not page-aligned, so MR_a's last bytes share a 4 KiB
  //      kernel page with MR_b's first bytes (the chunked path produces
  //      adjacent MRs from one buffer)
  //
  // Cross-platform verification: Mellanox mlx5 (vendor 0x02c9) and Broadcom
  // bnxt_re (vendor 0x14e4) tested CLEAN with the same workload, so this
  // guard is gated on Pensando vendor id.
  //
  // PyTorch's CPU caching allocator returns 64-byte-aligned but non-page-
  // aligned pointers (typically ending in 0x...040), which trips the bug.
  // mmap(2), shm_open + mmap, and posix_memalign(PAGESIZE, ...) all return
  // page-aligned pointers and are safe.
  if (desc.loc == MemoryLocationType::CPU) {
    const bool isIonic = (devCtx->GetRdmaDevice()->GetDeviceAttr()->orig_attr.vendor_id ==
                          static_cast<uint32_t>(application::RdmaDeviceVendorId::Pensando));
    if (isIonic && (desc.data % PAGESIZE) != 0) {
      MORI_IO_ERROR(
          "Chunked CPU registration on ionic requires a 4 KiB page-aligned "
          "base address. Memory id={} starts at 0x{:x} (offset 0x{:x} into "
          "a 4 KiB page). Allocate via mmap, shm_open + mmap, or "
          "posix_memalign(PAGESIZE, ...). PyTorch CPU tensors are NOT page-"
          "aligned and trigger an ionic firmware bug that silently corrupts "
          "cross-MR writes.",
          desc.id, desc.data, desc.data % PAGESIZE);
      throw std::runtime_error(
          "Chunked CPU registration on ionic requires page-aligned base address");
    }
  }

  size_t numChunks = (desc.size + chunkSize - 1) / chunkSize;
  std::vector<RdmaMemoryChunk> chunks;
  std::vector<application::RdmaOwnedMr> ownedMrs;
  chunks.reserve(numChunks);
  ownedMrs.reserve(numChunks);

  for (size_t i = 0; i < numChunks; ++i) {
    size_t offset = i * chunkSize;
    size_t len = std::min(chunkSize, desc.size - offset);
    void* ptr = reinterpret_cast<void*>(desc.data + offset);

    auto ownedMr = devCtx->TryRegisterOwnedMr(ptr, len);
    if (!ownedMr.has_value()) {
      MORI_IO_ERROR("Chunk registration failed for memory {}: chunk {}/{} offset={} len={}",
                    desc.id, i, numChunks, offset, len);
      throw std::runtime_error("Failed to register RDMA memory chunk");
    }
    chunks.push_back({offset, ownedMr->Region()});
    ownedMrs.push_back(std::move(*ownedMr));
  }

  RdmaMemoryLayout layout;
  layout.rdmaDevId = rdmaDevId;
  layout.baseAddr = desc.data;
  layout.logicalSize = desc.size;
  layout.chunks = std::move(chunks);

  auto reg = std::make_shared<RdmaLocalMemoryRegistration>(std::move(layout), std::move(ownedMrs));
  std::unique_lock<std::shared_mutex> lock(mu);
  localRegistrations_[desc.id] = reg;
  MORI_IO_INFO("Materialized local registration for memory {}: {} chunk(s) on rdmaDevId {}",
               desc.id, reg->Layout().chunks.size(), rdmaDevId);
  return reg;
}

void RdmaManager::DeregisterLocalRegistration(const MemoryDesc& desc) {
  std::unique_lock<std::shared_mutex> lock(mu);
  auto it = localRegistrations_.find(desc.id);
  if (it != localRegistrations_.end()) {
    it->second->MarkInvalidated();
    localRegistrations_.erase(it);
  }
  memoryDeviceAffinity_.erase(desc.id);
}

/* ---------------------------------- Remote Layout Management ---------------------------------- */
std::shared_ptr<const RdmaRemoteMemoryLayout> RdmaManager::GetRemoteLayout(EngineKey ekey,
                                                                           MemoryUniqueId id) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto remIt = remotes.find(ekey);
  if (remIt == remotes.end()) return nullptr;
  auto it = remIt->second.remoteLayouts.find(id);
  if (it == remIt->second.remoteLayouts.end()) return nullptr;
  return it->second;
}

void RdmaManager::RegisterRemoteLayout(EngineKey ekey, MemoryUniqueId id,
                                       std::shared_ptr<const RdmaRemoteMemoryLayout> layout) {
  std::unique_lock<std::shared_mutex> lock(mu);
  remotes[ekey].remoteLayouts[id] = std::move(layout);
}

void RdmaManager::DeregisterRemoteLayout(EngineKey ekey, MemoryUniqueId id) {
  std::unique_lock<std::shared_mutex> lock(mu);
  auto remIt = remotes.find(ekey);
  if (remIt != remotes.end()) {
    remIt->second.remoteLayouts.erase(id);
  }
}

/* ------------------------------------- Endpoint Management ------------------------------------ */
int RdmaManager::CountEndpoint(EngineKey engine, const ExactRouteKey& key) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto remoteIt = remotes.find(engine);
  if (remoteIt == remotes.end()) return 0;
  auto routeIt = remoteIt->second.rTable.find(key);
  if (routeIt == remoteIt->second.rTable.end()) return 0;
  return routeIt->second.size();
}

EpPairVec RdmaManager::GetAllEndpoint(EngineKey engine, const ExactRouteKey& key) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto remoteIt = remotes.find(engine);
  if (remoteIt == remotes.end()) return {};
  auto routeIt = remoteIt->second.rTable.find(key);
  if (routeIt == remoteIt->second.rTable.end()) return {};
  return routeIt->second;
}

application::RdmaEndpointConfig RdmaManager::GetRdmaEndpointConfig(int devId) {
  const auto& [device, portId] = availDevices[devId];
  const auto* deviceAttr = device->GetDeviceAttr();

  application::RdmaEndpointConfig epConfig{};
  epConfig.portId = portId;
  epConfig.gidIdx = -1;
  const char* envGidIdx = std::getenv("MORI_IB_GID_INDEX");
  if (envGidIdx != nullptr) {
    epConfig.gidIdx = std::atoi(envGidIdx);
  }

  epConfig.enableSrq = false;
  epConfig.alignment = PAGESIZE;
  epConfig.withCompChannel = (config.pollCqMode == PollCqMode::EVENT);

  uint32_t maxQpWr = static_cast<uint32_t>(deviceAttr->orig_attr.max_qp_wr);
  uint32_t maxCqe = static_cast<uint32_t>(deviceAttr->orig_attr.max_cqe);
  uint32_t maxSge = static_cast<uint32_t>(deviceAttr->orig_attr.max_sge);

  if (config.enableNotification && maxQpWr < config.notifPerQp) {
    MORI_IO_ERROR(
        "Device max_qp_wr={} is less than notifPerQp={}; notification requires at least "
        "notifPerQp RQ slots. Either reduce notifPerQp or disable notification.",
        maxQpWr, config.notifPerQp);
    throw std::runtime_error("Device RQ capacity insufficient for configured notifPerQp");
  }

  uint32_t desiredSendWr = config.maxSendWr > 0 ? static_cast<uint32_t>(config.maxSendWr) : 8192u;
  uint32_t desiredRecvWr = config.enableNotification ? config.notifPerQp : 0u;
  uint32_t desiredCqe = config.maxCqeNum > 0 ? static_cast<uint32_t>(config.maxCqeNum) : 16384u;
  std::optional<uint32_t> desiredMsgSge =
      config.maxMsgSge > 0 ? std::optional<uint32_t>(static_cast<uint32_t>(config.maxMsgSge))
                           : std::nullopt;

  env::Override("MORI_IO_QP_MAX_SEND_WR", desiredSendWr, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_RECV_WR", desiredRecvWr, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_CQE", desiredCqe, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_MSG_SGE", desiredMsgSge, mori::env::detail::ParsePositiveU32);
  // Alias for convenience: keep both MORI_IO_QP_MAX_MSG_SGE and MORI_IO_QP_MAX_SGE.
  env::Override("MORI_IO_QP_MAX_SGE", desiredMsgSge, mori::env::detail::ParsePositiveU32);

  if (config.enableNotification && desiredRecvWr < config.notifPerQp) {
    MORI_IO_WARN("MORI_IO_QP_MAX_RECV_WR={} is less than notifPerQp={}; clamping to notifPerQp",
                 desiredRecvWr, config.notifPerQp);
    desiredRecvWr = config.notifPerQp;
  }

  epConfig.maxMsgsNum = std::min(desiredSendWr, maxQpWr);
  // RQ must fit NotifManager's pre-posted recv WQEs (config.notifPerQp) when notification is
  // enabled. MORI_IO_QP_MAX_RECV_WR can raise this baseline, but not lower it.
  epConfig.maxRecvWr = desiredRecvWr > 0 ? std::min(desiredRecvWr, maxQpWr) : 0;
  epConfig.maxCqeNum = std::min(desiredCqe, maxCqe);
  uint32_t minRequiredCqe = epConfig.maxMsgsNum + epConfig.maxRecvWr;
  if (epConfig.maxCqeNum < minRequiredCqe) {
    uint32_t newCqeNum = std::min(minRequiredCqe, maxCqe);
    MORI_IO_WARN(
        "maxCqeNum ({}) is smaller than SQ+RQ depth ({}+{}={}); increasing maxCqeNum to {}",
        epConfig.maxCqeNum, epConfig.maxMsgsNum, epConfig.maxRecvWr, minRequiredCqe, newCqeNum);
    epConfig.maxCqeNum = newCqeNum;
  }
  if (desiredMsgSge.has_value()) {
    epConfig.maxMsgSge = std::min(*desiredMsgSge, maxSge);
  } else {
    bool is_ionic = (deviceAttr->orig_attr.vendor_id ==
                     static_cast<uint32_t>(application::RdmaDeviceVendorId::Pensando));
    epConfig.maxMsgSge = std::min(maxSge, is_ionic ? 2u : 4u);
  }
  return epConfig;
}

application::RdmaEndpoint RdmaManager::CreateEndpoint(int devId) {
  std::unique_lock<std::shared_mutex> lock(mu);

  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);

  application::RdmaEndpoint rdmaEp = devCtx->CreateRdmaEndpoint(GetRdmaEndpointConfig(devId));
  if (config.pollCqMode == PollCqMode::EVENT)
    SYSCALL_RETURN_ZERO(ibv_req_notify_cq(rdmaEp.ibvHandle.cq, 0));
  return rdmaEp;
}

bool RdmaManager::IsValidRdmaDeviceId(int devId) const {
  std::shared_lock<std::shared_mutex> lock(mu);
  return devId >= 0 && devId < static_cast<int>(availDevices.size());
}

void RdmaManager::ConnectLocalEndpoint(int ldevId, const application::RdmaEndpoint& local,
                                       const application::RdmaEndpointHandle& remote) {
  std::unique_lock<std::shared_mutex> lock(mu);
  deviceCtxs[ldevId]->ConnectEndpoint(local.handle, remote);
}

void RdmaManager::DestroyEndpoint(int devId, const application::RdmaEndpoint& endpoint) {
  std::unique_lock<std::shared_mutex> lock(mu);
  if (devId < 0 || devId >= static_cast<int>(deviceCtxs.size()) || deviceCtxs[devId] == nullptr) {
    return;
  }
  deviceCtxs[devId]->DestroyRdmaEndpoint(endpoint);
}

EndpointId RdmaManager::PublishEndpoint(EngineKey remoteKey, const ExactRouteKey& key,
                                        application::RdmaEndpoint local,
                                        application::RdmaEndpointHandle remote, int weight) {
  std::unique_lock<std::shared_mutex> lock(mu);
  RemoteEngineMeta& meta = remotes[remoteKey];
  auto epConfig = GetRdmaEndpointConfig(key.ldevId);
  EpPair ep{weight,
            key.ldevId,
            key.rdevId,
            remoteKey,
            local,
            remote,
            std::make_shared<std::atomic<int>>(0),
            static_cast<int>(epConfig.maxMsgsNum),
            std::make_shared<std::atomic<bool>>(false),
            std::make_shared<SubmissionLedger>(config.notifPerQp)};
  meta.rTable[key].push_back(ep);

  EndpointId id = nextEndpointId_.fetch_add(1);
  auto rt = std::make_shared<EndpointRuntime>(id, ep);
  endpointsById_[id] = rt;
  return id;
}

bool RdmaManager::UnpublishEndpoint(EngineKey remoteKey, const ExactRouteKey& key, EndpointId id) {
  std::unique_lock<std::shared_mutex> lock(mu);

  auto rtIt = endpointsById_.find(id);
  if (rtIt == endpointsById_.end()) return false;
  uint32_t localQpn = rtIt->second->ep.local.handle.qpn;
  endpointsById_.erase(rtIt);

  auto remoteIt = remotes.find(remoteKey);
  if (remoteIt == remotes.end()) return false;
  auto routeIt = remoteIt->second.rTable.find(key);
  if (routeIt == remoteIt->second.rTable.end()) return false;

  auto& eps = routeIt->second;
  auto epIt = std::remove_if(eps.begin(), eps.end(), [localQpn](const EpPair& ep) {
    return ep.local.handle.qpn == localQpn;
  });
  bool removed = epIt != eps.end();
  eps.erase(epIt, eps.end());
  if (eps.empty()) remoteIt->second.rTable.erase(routeIt);
  return removed;
}

std::shared_ptr<EndpointRuntime> RdmaManager::GetEndpointRuntime(EndpointId id) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto it = endpointsById_.find(id);
  if (it == endpointsById_.end()) return nullptr;
  return it->second;
}

application::RdmaDeviceContext* RdmaManager::GetRdmaDeviceContext(int devId) {
  std::shared_lock<std::shared_mutex> lock(mu);
  return deviceCtxs[devId];
}

std::vector<std::shared_ptr<EndpointRuntime>> RdmaManager::SnapshotEndpointRuntimes() {
  std::shared_lock<std::shared_mutex> lock(mu);
  std::vector<std::shared_ptr<EndpointRuntime>> result;
  result.reserve(endpointsById_.size());
  for (auto& [_, rt] : endpointsById_) {
    result.push_back(rt);
  }
  return result;
}

application::RdmaDeviceContext* RdmaManager::GetOrCreateDeviceContext(int devId) {
  assert(devId < deviceCtxs.size());
  application::RdmaDeviceContext* devCtx = deviceCtxs[devId];
  if (devCtx == nullptr) {
    devCtx = availDevices[devId].first->CreateRdmaDeviceContext();
    deviceCtxs[devId] = devCtx;
  }
  return devCtx;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
NotifManager::NotifManager(RdmaManager* rdmaMgr, const RdmaBackendConfig& cfg)
    : rdma(rdmaMgr), config(cfg) {}

NotifManager::~NotifManager() { Shutdown(); }

void NotifManager::RegisterEndpoint(const std::shared_ptr<EndpointRuntime>& rt) {
  if (!rt) return;

  if (config.pollCqMode == PollCqMode::EVENT) {
    epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = rt->id;
    assert(rt->ep.local.ibvHandle.compCh);
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, rt->ep.local.ibvHandle.compCh->fd, &ev));
  }

  // Skip notification setup if disabled
  if (!config.enableNotification) {
    std::lock_guard<std::mutex> lock(mu);
    registeredRuntimes_[rt->id] = rt;
    return;
  }

  std::lock_guard<std::mutex> lock(mu);
  if (notifCtxById_.find(rt->id) != notifCtxById_.end()) return;

  registeredRuntimes_[rt->id] = rt;

  application::RdmaDeviceContext* devCtx = rdma->GetRdmaDeviceContext(rt->ep.ldevId);
  assert(devCtx);

  void* buf;
  SYSCALL_RETURN_ZERO(
      posix_memalign(reinterpret_cast<void**>(&buf), PAGESIZE,
                     static_cast<size_t>(config.notifPerQp) * sizeof(NotifMessage)));
  application::RdmaMemoryRegion mr =
      devCtx->RegisterRdmaMemoryRegion(buf, config.notifPerQp * sizeof(NotifMessage));

  notifCtxById_.insert({rt->id, {mr, buf}});

  struct ibv_qp* qp = rt->ep.local.ibvHandle.qp;
  assert(qp);

  for (uint64_t i = 0; i < config.notifPerQp; i++) {
    struct ibv_sge sge{};
    sge.addr = mr.addr + i * sizeof(NotifMessage);
    sge.length = sizeof(NotifMessage);
    sge.lkey = mr.lkey;

    struct ibv_recv_wr wr{};
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr* bad = nullptr;
    SYSCALL_RETURN_ZERO(ibv_post_recv(qp, &wr, &bad));
  }
}

void NotifManager::UnregisterEndpoint(const std::shared_ptr<EndpointRuntime>& rt) {
  if (!rt) return;

  if (config.pollCqMode == PollCqMode::EVENT && rt->ep.local.ibvHandle.compCh != nullptr) {
    SYSCALL_RETURN_ZERO_IGNORE_ERROR(
        epoll_ctl(epfd, EPOLL_CTL_DEL, rt->ep.local.ibvHandle.compCh->fd, NULL), ENOENT);
  }

  QpNotifContext notifCtx{};
  bool hasNotifCtx = false;
  {
    std::lock_guard<std::mutex> lock(mu);
    registeredRuntimes_.erase(rt->id);
    auto it = notifCtxById_.find(rt->id);
    if (it != notifCtxById_.end()) {
      notifCtx = it->second;
      notifCtxById_.erase(it);
      hasNotifCtx = true;
    }
  }

  if (!hasNotifCtx) return;

  application::RdmaDeviceContext* devCtx = rdma->GetRdmaDeviceContext(rt->ep.ldevId);
  if (devCtx != nullptr) {
    devCtx->DeregisterRdmaMemoryRegion(reinterpret_cast<void*>(notifCtx.mr.addr));
  }
  free(notifCtx.buf);
}

NotifManager::FlushDrainStats NotifManager::ProcessOneCqe(
    const std::shared_ptr<EndpointRuntime>& rt) {
  const EpPair& ep = rt->ep;
  ibv_cq* cq = ep.local.ibvHandle.cq;
  FlushDrainStats flushDrain;

  // Resolve notif context once before the CQ drain loop.
  QpNotifContext* notifCtxPtr = nullptr;
  if (config.enableNotification) {
    std::lock_guard<std::mutex> lock(mu);
    auto nit = notifCtxById_.find(rt->id);
    if (nit != notifCtxById_.end()) notifCtxPtr = &nit->second;
  }

  const int batchSize = 32;
  struct ibv_wc wc[batchSize];
  int n = 0;

  while ((n = ibv_poll_cq(cq, batchSize, wc)) > 0) {
    for (int i = 0; i < n; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        const bool isFlush = (wc[i].status == IBV_WC_WR_FLUSH_ERR);
        const CqeFailureOrigin failureOrigin =
            ClassifyCqeFailureOrigin(wc[i].wr_id, config.notifPerQp);
        const CqeFailureAdvice failureAdvice =
            isFlush ? CqeFailureAdvice{ibv_wc_status_str(wc[i].status), {}}
                    : DescribeCqeFailure(wc[i].status, failureOrigin, config);

        if (isFlush) {
          flushDrain.Record(wc[i].qp_num);
          MORI_IO_DEBUG("ProcessOneCqe: flush error #{}: wr_id={} qp_num={}", flushDrain.count,
                        wc[i].wr_id, wc[i].qp_num);
        } else {
          // Non-flush error: this is the root cause — always log at ERROR.
          if (failureAdvice.HasHint()) {
            MORI_IO_ERROR(
                "ProcessOneCqe: [ROOT CAUSE] CQE error: wr_id={} status={}({}) qp_num={} "
                "vendor_err={} hint={}",
                wc[i].wr_id, static_cast<uint32_t>(wc[i].status), failureAdvice.statusText,
                wc[i].qp_num, wc[i].vendor_err, failureAdvice.hint);
          } else {
            MORI_IO_ERROR(
                "ProcessOneCqe: [ROOT CAUSE] CQE error: wr_id={} status={}({}) qp_num={} "
                "vendor_err={}",
                wc[i].wr_id, static_cast<uint32_t>(wc[i].status), failureAdvice.statusText,
                wc[i].qp_num, wc[i].vendor_err);
          }
        }

        int mergedBatchSize = 0;
        auto meta = ep.ledger
                        ? ep.ledger->ReleaseByCqe(wc[i].wr_id, ep.sqDepth.get(), &mergedBatchSize)
                        : nullptr;
        if (meta) {
          (void)meta->finishedBatchSize.fetch_add(mergedBatchSize);
          if (isFlush) {
            meta->diagnostics.MarkFlushCascade();
          } else {
            meta->diagnostics.MarkRootCause();
          }
          LogAsyncTransferFailureIfNeeded(&meta->diagnostics,
                                          static_cast<uint32_t>(StatusCode::ERR_RDMA_OP),
                                          failureAdvice.ComposeStatusMessage());
          TransferStatus* statusPtr = meta->status;
          if (statusPtr != nullptr) {
            statusPtr->Update(StatusCode::ERR_RDMA_OP, failureAdvice.ComposeStatusMessage());
            meta->status = nullptr;
          }
          if (ep.degraded && ep.degraded->load(std::memory_order_relaxed) && ep.ledger) {
            const int orphanedReleased = ep.ledger->ReleaseOrphanedByRecovery(ep.sqDepth.get());
            ep.degraded->store(false, std::memory_order_relaxed);
            MORI_IO_WARN(
                "ProcessOneCqe: recovered degraded EP eid={} qpn={} by releasing {} orphaned WRs",
                rt->id, ep.local.handle.qpn, orphanedReleased);
          }
        } else if (IsNotifSendWrId(wc[i].wr_id)) {
          if (ep.sqDepth) ep.sqDepth->fetch_sub(1, std::memory_order_relaxed);
          if (!isFlush) {
            MORI_IO_WARN(
                "ProcessOneCqe: failed notification SEND CQE, transfer_id={}, released 1 sqDepth",
                ExtractTransferIdFromWrId(wc[i].wr_id));
          }
        } else if (wc[i].wr_id < config.notifPerQp) {
          if (!isFlush) {
            MORI_IO_WARN("ProcessOneCqe: failed notification RECV CQE, wr_id={} (recv_idx)",
                         wc[i].wr_id);
          }
        } else {
          if (!isFlush) {
            MORI_IO_WARN(
                "ProcessOneCqe: failed CQE wr_id={} in ledger range but no record found, "
                "sqDepth may be stale",
                wc[i].wr_id);
          }
        }
        continue;
      }

      if (wc[i].opcode == IBV_WC_RECV) {
        // Skip RECV processing if notification is disabled
        if (!config.enableNotification) {
          MORI_IO_WARN("Received unexpected RECV completion when notification is disabled");
          continue;
        }

        std::lock_guard<std::mutex> lock(mu);

        assert(notifCtxPtr != nullptr);
        QpNotifContext& ctx = *notifCtxPtr;

        // FIXME: this notif mechenism has bug when notif index is wrapped around
        uint64_t idx = wc[i].wr_id;
        NotifMessage msg = reinterpret_cast<NotifMessage*>(ctx.mr.addr)[idx];
        assert(msg.totalNum > 0);

        EngineKey ekey = ep.remoteEngineKey;
        if (notifPool[ekey].find(msg.id) == notifPool[ekey].end()) {
          notifPool[ekey][msg.id] = msg.totalNum;
        }
        notifPool[ekey][msg.id] -= 1;
        MORI_IO_TRACE(
            "NotifManager receive notif message from engine {} id {} qp {} total num {} cur num {}",
            ekey.c_str(), msg.id, msg.qpIndex, msg.totalNum, notifPool[ekey][msg.id]);
        // replenish recv wr
        struct ibv_sge sge{};
        sge.addr = ctx.mr.addr + idx * sizeof(NotifMessage);
        sge.length = sizeof(NotifMessage);
        sge.lkey = ctx.mr.lkey;

        struct ibv_recv_wr wr{};
        wr.wr_id = idx;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        struct ibv_recv_wr* bad = nullptr;
        SYSCALL_RETURN_ZERO(ibv_post_recv(ep.local.ibvHandle.qp, &wr, &bad));
      } else if (wc[i].opcode == IBV_WC_SEND) {
        if (!IsNotifSendWrId(wc[i].wr_id)) {
          MORI_IO_WARN(
              "ProcessOneCqe: unexpected SEND completion with non-notification wr_id {}; "
              "releasing 1 sqDepth under current SEND invariant",
              wc[i].wr_id);
        }
        if (ep.sqDepth) ep.sqDepth->fetch_sub(1, std::memory_order_relaxed);
      } else {
        // Batch path: wr_id carries a recordId from the SubmissionLedger.
        uint64_t recordId = wc[i].wr_id;
        int mergedBatchSize = 0;
        auto meta = ep.ledger
                        ? ep.ledger->ReleaseByCqe(recordId, ep.sqDepth.get(), &mergedBatchSize)
                        : nullptr;
        if (meta) {
          uint32_t finishedBefore = meta->finishedBatchSize.fetch_add(mergedBatchSize);
          TransferStatus* statusPtr = meta->status;
          if (statusPtr != nullptr && (finishedBefore + mergedBatchSize) == meta->totalBatchSize) {
            statusPtr->Update(StatusCode::SUCCESS, ibv_wc_status_str(wc[i].status));
          }
          MORI_IO_TRACE("ProcessOneCqe: batch CQE for task {} total={} finished={} cur={}",
                        meta->id, meta->totalBatchSize, finishedBefore, mergedBatchSize);
        } else {
          MORI_IO_WARN(
              "ProcessOneCqe: no ledger record for wr_id {} (recordId {}); sqDepth may be stale",
              wc[i].wr_id, recordId);
        }
      }
    }
  }

  if (!flushDrain.Empty()) {
    MORI_IO_DEBUG("ProcessOneCqe: drain — {} flush errors on eid={} qp_num={}", flushDrain.count,
                  rt->id, flushDrain.firstQpNum);
  }
  return flushDrain;
}

void NotifManager::EmitFlushSummaryIfNeeded(const FlushRoundStats& roundStats) {
  if (roundStats.Empty()) {
    flushSummaryStreak_ = 0;
    return;
  }

  flushSummaryStreak_++;
  const bool shouldLog =
      (flushSummaryStreak_ == 1) ||
      (flushSummaryStreak_ < 64 && (flushSummaryStreak_ & (flushSummaryStreak_ - 1)) == 0) ||
      (flushSummaryStreak_ % 1000 == 0);

  if (shouldLog) {
    if (flushSummaryStreak_ == 1) {
      MORI_IO_ERROR(
          "CQ poll round summary: {} flush errors across {} endpoint(s); "
          "representative eid={} qp_num={}. "
          "Flush errors are cascaded from QP(s) entering Error State. "
          "Check: (1) peer process alive, (2) PFC / network congestion, "
          "(3) ibv_devinfo / dmesg for HW errors",
          roundStats.total, roundStats.endpointCount, roundStats.sampleEndpointId,
          roundStats.sampleQpNum);
    } else {
      MORI_IO_WARN(
          "CQ poll round summary: {} flush errors across {} endpoint(s); "
          "representative eid={} qp_num={}; in "
          "consecutive flush round #{} (rate-limited). "
          "Flush errors are cascaded from QP(s) entering Error State. "
          "Check: (1) peer process alive, (2) PFC / network congestion, "
          "(3) ibv_devinfo / dmesg for HW errors",
          roundStats.total, roundStats.endpointCount, roundStats.sampleEndpointId,
          roundStats.sampleQpNum, flushSummaryStreak_);
    }
  }
}

void NotifManager::MainLoop() {
  if (config.pollCqMode == PollCqMode::EVENT) {
    constexpr int maxEvents = 128;
    epoll_event events[maxEvents];
    while (running.load()) {
      FlushRoundStats roundStats;
      bool handledCqEvent = false;
      int nfds = epoll_wait(epfd, events, maxEvents, 0 /*ms*/);
      for (int i = 0; i < nfds; ++i) {
        EndpointId eid = events[i].data.u64;

        std::shared_ptr<EndpointRuntime> rt;
        {
          std::lock_guard<std::mutex> lock(mu);
          auto it = registeredRuntimes_.find(eid);
          if (it == registeredRuntimes_.end()) continue;
          rt = it->second;
        }

        struct ibv_comp_channel* ch = rt->ep.local.ibvHandle.compCh;

        struct ibv_cq* cq = nullptr;
        void* evCtx = nullptr;
        if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
        ibv_ack_cq_events(cq, 1);
        ibv_req_notify_cq(cq, 0);

        handledCqEvent = true;
        roundStats.Merge(rt->id, ProcessOneCqe(rt));
      }
      if (handledCqEvent) {
        EmitFlushSummaryIfNeeded(roundStats);
      }
    }
  } else {
    while (running.load()) {
      auto snapshot = rdma->SnapshotEndpointRuntimes();
      if (snapshot.empty()) {
        EmitFlushSummaryIfNeeded(FlushRoundStats{});
        std::this_thread::yield();
        continue;
      }
      FlushRoundStats roundStats;
      for (auto& rt : snapshot) {
        roundStats.Merge(rt->id, ProcessOneCqe(rt));
      }
      EmitFlushSummaryIfNeeded(roundStats);
    }
  }
}

bool NotifManager::PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id,
                                            TransferStatus* status) {
  std::lock_guard<std::mutex> lock(mu);
  if (notifPool[remote].find(id) != notifPool[remote].end()) {
    if (notifPool[remote][id] == 0) {
      status->SetCode(StatusCode::SUCCESS);
      return true;
    }
  }
  return false;
}

void NotifManager::Start() {
  if (running.load()) return;
  if (config.pollCqMode == PollCqMode::EVENT) {
    epfd = epoll_create1(EPOLL_CLOEXEC);
    assert(epfd >= 0);
  }
  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void NotifManager::Shutdown() {
  running.store(false);
  if (config.pollCqMode == PollCqMode::EVENT) {
    epfd = close(epfd);
  }
  if (thd.joinable()) thd.join();
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                      Control Plane Server */
/* ----------------------------------------------------------------------------------------------
 */
ControlPlaneServer::ControlPlaneServer(const std::string& k, const std::string& host, int port,
                                       RdmaManager* rdmaMgr, NotifManager* notifMgr)
    : myEngKey(k) {
  ctx.reset(new application::TCPContext(host, port));
  rdma = rdmaMgr;
  notif = notifMgr;
}

ControlPlaneServer::~ControlPlaneServer() { Shutdown(); }

void ControlPlaneServer::RegisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  engines[rdesc.key] = rdesc;
}

void ControlPlaneServer::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  engines.erase(rdesc.key);
}

void ControlPlaneServer::BuildRdmaConn(EngineKey ekey, const ExactRouteKey& key) {
  application::TCPEndpointHandle tcph{};
  bool tcpConnected = false;
  bool localEpCreated = false;
  bool published = false;
  EndpointId eid = 0;
  application::RdmaEndpoint lep;
  std::shared_ptr<EndpointRuntime> ert;

  try {
    {
      std::lock_guard<std::mutex> lock(mu);
      assert((engines.find(ekey) != engines.end()) && "register engine first");
      EngineDesc& rdesc = engines[ekey];
      tcph = ctx->Connect(rdesc.host, rdesc.port);
      tcpConnected = true;
    }

    if (!rdma->IsValidRdmaDeviceId(key.ldevId)) {
      throw std::runtime_error("invalid initiator local RDMA device id " +
                               std::to_string(key.ldevId));
    }
    if (!rdma->IsValidRdmaDeviceId(key.rdevId)) {
      throw std::runtime_error("invalid responder local RDMA device id " +
                               std::to_string(key.rdevId));
    }

    lep = rdma->CreateEndpoint(key.ldevId);
    localEpCreated = true;

    Protocol p(tcph);
    p.WriteMessageRegEndpointRequest({myEngKey, key.topo, key.ldevId, key.rdevId, lep.handle});
    MessageHeader hdr = p.ReadMessageHeader();
    if (hdr.type != MessageType::RegEndpoint) {
      throw std::runtime_error("unexpected control-plane response type " +
                               std::to_string(static_cast<uint8_t>(hdr.type)));
    }

    MessageRegEndpointResponse msg = p.ReadMessageRegEndpointResponse(hdr.len);
    if (msg.code != StatusCode::SUCCESS) {
      throw std::runtime_error("RegEndpoint failed: " + msg.message);
    }
    if (msg.responderLocalDevId != key.rdevId) {
      throw std::runtime_error("RegEndpoint responder local dev mismatch: expected " +
                               std::to_string(key.rdevId) + " got " +
                               std::to_string(msg.responderLocalDevId));
    }

    rdma->ConnectLocalEndpoint(key.ldevId, lep, msg.responderEph);
    eid = rdma->PublishEndpoint(ekey, key, lep, msg.responderEph, 1);
    published = true;
    ert = rdma->GetEndpointRuntime(eid);
    if (!ert) throw std::runtime_error("failed to resolve published endpoint runtime");
    notif->RegisterEndpoint(ert);
    localEpCreated = false;
    MORI_IO_INFO("Built exact RdmaConn for engine {} local({},{}) remote({},{}) pair({}->{})", ekey,
                 key.topo.local.deviceId, key.topo.local.loc, key.topo.remote.deviceId,
                 key.topo.remote.loc, key.ldevId, key.rdevId);
  } catch (...) {
    if (published) {
      notif->UnregisterEndpoint(ert);
      if (!rdma->UnpublishEndpoint(ekey, key, eid)) {
        MORI_IO_WARN("Failed to unpublish initiator endpoint {} for exact route cleanup", eid);
      }
    }
    if (localEpCreated) rdma->DestroyEndpoint(key.ldevId, lep);
    if (tcpConnected) ctx->CloseEndpoint(tcph);
    throw;
  }

  if (tcpConnected) ctx->CloseEndpoint(tcph);
}

void ControlPlaneServer::RegisterMemory(MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  mems[desc.id] = desc;
}

void ControlPlaneServer::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  mems.erase(desc.id);
}

std::shared_ptr<const RdmaRemoteMemoryLayout> ControlPlaneServer::AskRemoteMemoryLayout(
    EngineKey ekey, MemoryUniqueId id, size_t expectedSize) {
  application::TCPEndpointHandle tcph;
  {
    std::lock_guard<std::mutex> lock(mu);
    assert((engines.find(ekey) != engines.end()) && "register engine first");
    EngineDesc& rdesc = engines[ekey];
    tcph = ctx->Connect(rdesc.host, rdesc.port);
  }

  Protocol p(tcph);
  p.WriteMessageAskMemoryLayoutRequest({ekey, id});
  MessageHeader hdr = p.ReadMessageHeader();
  if (hdr.type != MessageType::AskMemoryLayout) {
    ctx->CloseEndpoint(tcph);
    throw std::runtime_error("AskRemoteMemoryLayout: unexpected message type " +
                             std::to_string(static_cast<uint8_t>(hdr.type)));
  }
  MessageAskMemoryLayoutResponse msg = p.ReadMessageAskMemoryLayoutResponse(hdr.len);
  ctx->CloseEndpoint(tcph);

  if (msg.code != StatusCode::SUCCESS) {
    MORI_IO_ERROR("AskRemoteMemoryLayout failed for memory {}: code={} message={}", id,
                  static_cast<uint32_t>(msg.code), msg.message);
    throw std::runtime_error("Remote memory layout materialization failed: " + msg.message);
  }

  // Validate chunk coverage
  RdmaMemoryLayout layout;
  layout.rdmaDevId = msg.rdmaDevId;
  layout.logicalSize = 0;
  layout.baseAddr = 0;
  for (auto& cw : msg.chunks) {
    application::RdmaMemoryRegion mr;
    mr.addr = static_cast<uintptr_t>(cw.addr);
    mr.rkey = cw.rkey;
    mr.lkey = 0;
    mr.length = static_cast<size_t>(cw.length);
    layout.chunks.push_back({static_cast<size_t>(cw.offset), mr});
    if (layout.baseAddr == 0) layout.baseAddr = mr.addr;
    layout.logicalSize += mr.length;
  }

  if (layout.chunks.empty()) {
    throw std::runtime_error("Remote memory layout for id " + std::to_string(id) +
                             " returned zero chunks");
  }
  if (layout.logicalSize != expectedSize) {
    throw std::runtime_error("Remote memory layout size mismatch for id " + std::to_string(id) +
                             ": expected " + std::to_string(expectedSize) + " got " +
                             std::to_string(layout.logicalSize));
  }
  size_t expectedOffset = 0;
  for (const auto& c : layout.chunks) {
    if (c.offset != expectedOffset) {
      throw std::runtime_error("Remote memory layout for id " + std::to_string(id) +
                               " has non-contiguous chunks: expected offset " +
                               std::to_string(expectedOffset) + " got " + std::to_string(c.offset));
    }
    expectedOffset += c.mr.length;
  }

  return std::make_shared<RdmaRemoteMemoryLayout>(RdmaRemoteMemoryLayout{std::move(layout)});
}

void ControlPlaneServer::AcceptRemoteEngineConn() {
  application::TCPEndpointHandleVec newEps = ctx->Accept();
  for (auto& ep : newEps) {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = ep.fd;
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, ep.fd, &ev));
    eps.insert({ep.fd, ep});
  }
}

void ControlPlaneServer::HandleControlPlaneProtocol(int fd) {
  assert(eps.find(fd) != eps.end());
  application::TCPEndpointHandle tcph = eps[fd];

  Protocol p(tcph);
  MessageHeader hdr = p.ReadMessageHeader();

  switch (hdr.type) {
    case MessageType::RegEndpoint: {
      MessageRegEndpointRequest msg = p.ReadMessageRegEndpointRequest(hdr.len);
      application::RdmaEndpoint lep;
      bool localEpCreated = false;
      bool published = false;
      int devId = -1;
      EndpointId eid = 0;
      ExactRouteKey reverseKey{};
      std::shared_ptr<EndpointRuntime> ert;

      try {
        if (msg.initiatorLocalDevId < 0) {
          throw std::runtime_error("invalid initiator local RDMA device id");
        }
        devId = msg.expectedResponderLocalDevId;
        if (!rdma->IsValidRdmaDeviceId(devId)) {
          throw std::runtime_error("invalid responder local RDMA device id " +
                                   std::to_string(devId));
        }

        reverseKey.topo = {msg.topo.remote, msg.topo.local};
        reverseKey.ldevId = devId;
        reverseKey.rdevId = msg.initiatorLocalDevId;

        lep = rdma->CreateEndpoint(devId);
        localEpCreated = true;
        rdma->ConnectLocalEndpoint(devId, lep, msg.initiatorEph);
        p.WriteMessageRegEndpointResponse({StatusCode::SUCCESS, "", devId, lep.handle});
      } catch (const std::exception& e) {
        if (localEpCreated) rdma->DestroyEndpoint(devId, lep);
        try {
          p.WriteMessageRegEndpointResponse({StatusCode::ERR_RDMA_OP, e.what(), -1, {}});
        } catch (...) {
        }
        SYSCALL_RETURN_ZERO_IGNORE_ERROR(epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL), ENOENT);
        break;
      }

      try {
        eid = rdma->PublishEndpoint(msg.ekey, reverseKey, lep, msg.initiatorEph, 1);
        published = true;
        ert = rdma->GetEndpointRuntime(eid);
        if (!ert) throw std::runtime_error("failed to resolve published endpoint runtime");
        notif->RegisterEndpoint(ert);
        localEpCreated = false;
      } catch (const std::exception& e) {
        if (published) {
          notif->UnregisterEndpoint(ert);
          if (!rdma->UnpublishEndpoint(msg.ekey, reverseKey, eid)) {
            MORI_IO_WARN("Failed to unpublish responder endpoint {} for exact route cleanup", eid);
          }
        }
        if (localEpCreated) rdma->DestroyEndpoint(devId, lep);
        MORI_IO_ERROR("RegEndpoint responder post-response setup failed: {}", e.what());
      }
      SYSCALL_RETURN_ZERO_IGNORE_ERROR(epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL), ENOENT);
      break;
    }
    case MessageType::AskMemoryLayout: {
      MessageAskMemoryLayoutRequest req = p.ReadMessageAskMemoryLayoutRequest(hdr.len);
      MemoryDesc desc;
      bool found = false;
      {
        std::lock_guard<std::mutex> lock(mu);
        auto it = mems.find(req.id);
        if (it != mems.end()) {
          desc = it->second;
          found = true;
        }
      }
      if (!found) {
        p.WriteMessageAskMemoryLayoutResponse(
            {req.ekey, req.id, StatusCode::ERR_NOT_FOUND, "memory not found", -1, {}});
        break;
      }
      try {
        auto reg = rdma->GetOrMaterializeLocalRegistration(desc);
        const auto& layout = reg->Layout();
        std::vector<RdmaRemoteMemoryChunkWire> wireChunks;
        wireChunks.reserve(layout.chunks.size());
        for (const auto& c : layout.chunks) {
          wireChunks.push_back({static_cast<uint64_t>(c.offset), static_cast<uint64_t>(c.mr.addr),
                                c.mr.rkey, static_cast<uint64_t>(c.mr.length)});
        }
        p.WriteMessageAskMemoryLayoutResponse(
            {req.ekey, req.id, StatusCode::SUCCESS, "", layout.rdmaDevId, std::move(wireChunks)});
      } catch (const std::exception& e) {
        p.WriteMessageAskMemoryLayoutResponse(
            {req.ekey, req.id, StatusCode::ERR_RDMA_OP, e.what(), -1, {}});
      }
      break;
    }
    default:
      assert(false && "not implemented");
  }

  ctx->CloseEndpoint(tcph);
  eps.erase(fd);
}

void ControlPlaneServer::MainLoop() {
  constexpr int maxEvents = 128;
  epoll_event events[maxEvents];

  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 5 /*ms*/);

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      // Add new endpoints into epoll list
      if (fd == ctx->GetListenFd()) {
        AcceptRemoteEngineConn();
        continue;
      }

      HandleControlPlaneProtocol(fd);
    }
  }
}

void ControlPlaneServer::Start() {
  if (running.load()) return;

  // Create epoll fd
  epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);

  // Add TCP listen fd
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ctx->Listen();
  ev.data.fd = ctx->GetListenFd();
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, ctx->GetListenFd(), &ev));

  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void ControlPlaneServer::Shutdown() {
  running.store(false);
  if (thd.joinable()) thd.join();
  if (epfd >= 0) {
    close(epfd);
    epfd = -1;
  }
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                       RdmaBackendSession */
/* ----------------------------------------------------------------------------------------------
 */
RdmaBackendSession::RdmaBackendSession(const RdmaBackendConfig& config, RdmaResolvedLocalMemory l,
                                       RdmaResolvedRemoteMemory r, const EpPairVec& e,
                                       Executor* exec)
    : config(config), local(std::move(l)), remote(std::move(r)), eps(e), executor(exec) {}

bool RdmaBackendSession::UseFastPath(const SizeVec& sizes) const {
  if (!local.singleMrFastPath || !remote.singleMrFastPath) return false;
  for (auto s : sizes) {
    if (s > std::numeric_limits<uint32_t>::max()) return false;
  }
  return true;
}

void RdmaBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  if (local.registration && local.registration->Invalidated()) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "local memory has been deregistered");
    return;
  }
  status->SetCode(StatusCode::IN_PROGRESS);

  if (local.singleMrFastPath && remote.singleMrFastPath &&
      size <= std::numeric_limits<uint32_t>::max()) {
    auto callbackMeta = std::make_shared<CqCallbackMeta>(status, id, 1);
    callbackMeta->localRegRef = local.registration;
    internal::PublishCurrentIoCallDiagnostics(callbackMeta);
    RdmaOpRet ret = RdmaReadWrite(eps, local.singleMr, localOffset, remote.singleMr, remoteOffset,
                                  size, callbackMeta, id, isRead);
    assert(!ret.Init());
    if (ret.Failed() || ret.Succeeded()) status->Update(ret.code, ret.message);
  } else {
    size_t maxResolvedLanes =
        executor ? std::min(eps.size(), static_cast<size_t>(executor->MaxParallelTasks()))
                 : eps.size();
    auto plan =
        BuildResolvedTransferPlan(eps, local.registration->Layout(), {localOffset},
                                  remote.layout->layout, {remoteOffset}, {size}, maxResolvedLanes);
    if (plan.slices.empty() && size > 0) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "length out of range");
      return;
    }
    auto callbackMeta =
        std::make_shared<CqCallbackMeta>(status, id, static_cast<int>(plan.slices.size()));
    callbackMeta->localRegRef = local.registration;
    internal::PublishCurrentIoCallDiagnostics(callbackMeta);
    RdmaOpRet ret;
    if (executor) {
      ResolvedExecutorReq req{eps, plan.slices,          plan.lanes, callbackMeta,
                              id,  config.postBatchSize, isRead};
      ret = executor->RdmaBatchReadWriteResolved(req);
    } else {
      ret = RdmaSubmitResolvedTransferPlanInline(eps, plan.slices, plan.lanes, callbackMeta, id,
                                                 isRead, config.postBatchSize);
    }
    assert(!ret.Init());
    if (ret.Failed() || ret.Succeeded()) status->Update(ret.code, ret.message);
  }

  if (!status->Failed() && config.enableNotification) {
    RdmaOpRet notifRet = RdmaNotifyTransfer(eps, status, id);
    if (notifRet.Failed()) status->Update(notifRet.code, notifRet.message);
  }
}

void RdmaBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  if (local.registration && local.registration->Invalidated()) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "local memory has been deregistered");
    return;
  }
  status->SetCode(StatusCode::IN_PROGRESS);

  if (UseFastPath(sizes)) {
    auto callbackMeta = std::make_shared<CqCallbackMeta>(status, id, sizes.size());
    callbackMeta->localRegRef = local.registration;
    internal::PublishCurrentIoCallDiagnostics(callbackMeta);
    RdmaOpRet ret;
    if (executor) {
      ExecutorReq req{eps,   local.singleMr, localOffsets, remote.singleMr,      remoteOffsets,
                      sizes, callbackMeta,   id,           config.postBatchSize, isRead};
      ret = executor->RdmaBatchReadWrite(req);
    } else {
      ret = RdmaBatchReadWrite(eps, local.singleMr, localOffsets, remote.singleMr, remoteOffsets,
                               sizes, callbackMeta, id, isRead, config.postBatchSize);
    }
    assert(!ret.Init());
    if (ret.Failed() || ret.Succeeded()) status->Update(ret.code, ret.message);
  } else {
    size_t maxResolvedLanes =
        executor ? std::min(eps.size(), static_cast<size_t>(executor->MaxParallelTasks()))
                 : eps.size();
    auto plan =
        BuildResolvedTransferPlan(eps, local.registration->Layout(), localOffsets,
                                  remote.layout->layout, remoteOffsets, sizes, maxResolvedLanes);
    if (plan.slices.empty() &&
        std::any_of(sizes.begin(), sizes.end(), [](size_t sz) { return sz > 0; })) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "length out of range");
      return;
    }
    auto callbackMeta =
        std::make_shared<CqCallbackMeta>(status, id, static_cast<int>(plan.slices.size()));
    callbackMeta->localRegRef = local.registration;
    internal::PublishCurrentIoCallDiagnostics(callbackMeta);
    RdmaOpRet ret;
    if (executor) {
      ResolvedExecutorReq req{eps, plan.slices,          plan.lanes, callbackMeta,
                              id,  config.postBatchSize, isRead};
      ret = executor->RdmaBatchReadWriteResolved(req);
    } else {
      ret = RdmaSubmitResolvedTransferPlanInline(eps, plan.slices, plan.lanes, callbackMeta, id,
                                                 isRead, config.postBatchSize);
    }
    assert(!ret.Init());
    if (ret.Failed() || ret.Succeeded()) status->Update(ret.code, ret.message);
  }

  if (!status->Failed() && config.enableNotification) {
    RdmaOpRet notifRet = RdmaNotifyTransfer(eps, status, id);
    if (notifRet.Failed()) status->Update(notifRet.code, notifRet.message);
  }
}

bool RdmaBackendSession::Alive() const {
  return local.registration != nullptr && !local.registration->Invalidated() &&
         remote.layout != nullptr;
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                           RdmaBackend */
/* ----------------------------------------------------------------------------------------------
 */

RdmaBackend::RdmaBackend(EngineKey k, const IOEngineConfig& engConfig,
                         const RdmaBackendConfig& beConfig)
    : myEngKey(k), config(beConfig) {
  env::Override("MORI_IO_ENABLE_NOTIFICATION", config.enableNotification,
                mori::env::detail::ParseBool);
  ValidateRdmaNotificationConfig(config);

  application::RdmaContext* ctx =
      new application::RdmaContext(application::RdmaBackendType::IBVerbs);
  rdma.reset(new mori::io::RdmaManager(config, ctx));

  notif.reset(new NotifManager(rdma.get(), config));
  notif->Start();

  server.reset(
      new ControlPlaneServer(myEngKey, engConfig.host, engConfig.port, rdma.get(), notif.get()));
  server->Start();

  if (config.numWorkerThreads > 1) {
    executor.reset(
        new MultithreadExecutor(std::min(config.qpPerTransfer, config.numWorkerThreads)));
    executor->Start();
  }

  std::stringstream ss;
  ss << config;
  MORI_IO_INFO("RdmaBackend created with config: {}", ss.str().c_str());
}

RdmaBackend::~RdmaBackend() {
  notif->Shutdown();
  server->Shutdown();
  if (executor.get() != nullptr) {
    executor->Shutdown();
  }
  sessionCache.clear();
}

void RdmaBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  server->RegisterRemoteEngine(rdesc);
}

void RdmaBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  server->DeregisterRemoteEngine(rdesc);
}

void RdmaBackend::RegisterMemory(MemoryDesc& desc) { server->RegisterMemory(desc); }

void RdmaBackend::DeregisterMemory(const MemoryDesc& desc) {
  server->DeregisterMemory(desc);
  rdma->DeregisterLocalRegistration(desc);
  InvalidateSessionsForMemory(desc.id);
}

void RdmaBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                            const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  try {
    RdmaBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
    sess->ReadWrite(localOffset, remoteOffset, size, status, id, isRead);
  } catch (const std::exception& e) {
    MORI_IO_ERROR("RdmaBackend::ReadWrite failed: {}", e.what());
    status->Update(StatusCode::ERR_RDMA_OP, e.what());
  }
}

void RdmaBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                 const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                 const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                 bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  assert(localOffsets.size() == remoteOffsets.size());
  assert(sizes.size() == remoteOffsets.size());
  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  try {
    RdmaBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
    sess->BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, isRead);
  } catch (const std::exception& e) {
    MORI_IO_ERROR("RdmaBackend::BatchReadWrite failed: {}", e.what());
    status->Update(StatusCode::ERR_RDMA_OP, e.what());
  }
}

BackendSession* RdmaBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  try {
    auto sess = CreateSessionImpl(local, remote);
    return new RdmaBackendSession(std::move(*sess));
  } catch (const std::exception& e) {
    MORI_IO_ERROR("RdmaBackend::CreateSession failed: {}", e.what());
    return nullptr;
  }
}

std::shared_ptr<RdmaBackendSession> RdmaBackend::CreateSessionImpl(const MemoryDesc& local,
                                                                   const MemoryDesc& remote) {
  // 1. Materialize local registration (pins local.rdmaDevId)
  auto localReg = rdma->GetOrMaterializeLocalRegistration(local);
  const auto& localLayout = localReg->Layout();

  // 2. Fetch remote layout (gets remote.rdmaDevId)
  EngineKey ekey = remote.engineKey;
  auto remoteLayout = rdma->GetRemoteLayout(ekey, remote.id);
  if (!remoteLayout) {
    remoteLayout = server->AskRemoteMemoryLayout(ekey, remote.id, remote.size);
    rdma->RegisterRemoteLayout(ekey, remote.id, remoteLayout);
  }

  // 3. Create/reuse endpoints for the exact (ldevId, rdevId) pair
  TopoKey localKey{local.deviceId, local.loc};
  TopoKey remoteKey{remote.deviceId, remote.loc};
  TopoKeyPair kp{localKey, remoteKey};
  ExactRouteKey exactKey{kp, localLayout.rdmaDevId, remoteLayout->layout.rdmaDevId};

  EpPairVec epSet = rdma->GetAllEndpoint(ekey, exactKey);
  while (static_cast<int>(epSet.size()) < config.qpPerTransfer) {
    server->BuildRdmaConn(ekey, exactKey);
    epSet = rdma->GetAllEndpoint(ekey, exactKey);
  }
  epSet.resize(config.qpPerTransfer);

  // 4. Build resolved memory descriptors with single-MR fast path detection
  RdmaResolvedLocalMemory resolvedLocal;
  resolvedLocal.registration = localReg;
  if (localLayout.IsSingleChunk()) {
    resolvedLocal.singleMrFastPath = true;
    resolvedLocal.singleMr = localLayout.chunks[0].mr;
  }

  RdmaResolvedRemoteMemory resolvedRemote;
  resolvedRemote.layout = remoteLayout;
  if (remoteLayout->layout.IsSingleChunk()) {
    resolvedRemote.singleMrFastPath = true;
    resolvedRemote.singleMr = remoteLayout->layout.chunks[0].mr;
  }

  return std::make_shared<RdmaBackendSession>(config, std::move(resolvedLocal),
                                              std::move(resolvedRemote), epSet, executor.get());
}

bool RdmaBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                           TransferStatus* status) {
  return notif->PopInboundTransferStatus(remote, id, status);
}

RdmaBackendSession* RdmaBackend::GetOrCreateSessionCached(const MemoryDesc& local,
                                                          const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};
  {
    std::lock_guard<std::mutex> lock(sessionCacheMu);
    auto it = sessionCache.find(key);
    if (it != sessionCache.end()) return it->second.get();
  }
  auto newSess = CreateSessionImpl(local, remote);
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) return it->second.get();
  auto [emplacedIt, inserted] = sessionCache.emplace(key, std::move(newSess));
  return emplacedIt->second.get();
}

void RdmaBackend::InvalidateSessionsForMemory(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  for (auto it = sessionCache.begin(); it != sessionCache.end();) {
    if (it->first.localMemId == id || it->first.remoteMemId == id) {
      it = sessionCache.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace io
}  // namespace mori
