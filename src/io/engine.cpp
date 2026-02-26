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
#include "mori/io/engine.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>

#include "mori/io/logging.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/tcp/backend_impl.hpp"
#include "src/io/xgmi/backend_impl.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                         IOEngineSession                                        */
/* ---------------------------------------------------------------------------------------------- */
TransferUniqueId IOEngineSession::AllocateTransferUniqueId() {
  return engine->AllocateTransferUniqueId();
}

void IOEngineSession::Read(size_t localOffset, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->Read(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session read error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngineSession::Write(size_t localOffset, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->Write(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session write error {} message {}", status->CodeUint32(), status->Message());
  }
  return;
}

void IOEngineSession::BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->BatchRead(localOffsets, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session batch read error {} message {}", status->CodeUint32(),
                  status->Message());
  }
}

void IOEngineSession::BatchWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                 const SizeVec& sizes, TransferStatus* status,
                                 TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->BatchWrite(localOffsets, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session batch write error {} message {}", status->CodeUint32(),
                  status->Message());
  }
}

bool IOEngineSession::Alive() { return backendSess->Alive(); }

/* ---------------------------------------------------------------------------------------------- */
/*                                            IOEngine                                            */
/* ---------------------------------------------------------------------------------------------- */

IOEngine::IOEngine(EngineKey key, IOEngineConfig config) : config(config) {
  // Initialize descriptor
  desc.key = key;
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  desc.nodeId = ResolveNodeId(hostname);
  desc.hostname = std::string(hostname);
  desc.host = config.host;
  desc.port = config.port;
  MORI_IO_INFO("Create engine key {} node_id {} hostname {}", key, desc.nodeId, hostname);
}

IOEngine::~IOEngine() {}

void IOEngine::CreateBackend(BackendType type, const BackendConfig& beConfig) {
  if (backends.find(type) != backends.end()) {
    MORI_IO_WARN("Backend type {} already exists, skip duplicate creation",
                 static_cast<uint32_t>(type));
    return;
  }

  if (type == BackendType::RDMA) {
    auto backend = std::make_unique<RdmaBackend>(desc.key, config,
                                                 static_cast<const RdmaBackendConfig&>(beConfig));

    if (config.port == 0) {
      auto bound_port_opt = backend->GetListenPort();
      if (!bound_port_opt.has_value() || bound_port_opt.value() == 0) {
        MORI_IO_ERROR("IOEngine key {} failed to retrieve bound port after RDMA backend init",
                      desc.key);
        assert(false && "Failed to retrieve bound port after RDMA backend init");
      } else {
        uint16_t bound_port = bound_port_opt.value();
        desc.port = bound_port;
        this->config.port = bound_port;
        MORI_IO_INFO("IOEngine key {} bound ephemeral port {}", desc.key, bound_port);
      }
    }

    backends.insert({type, std::move(backend)});
    InvalidateRouteCache();
    EnsureXgmiBackendCreatedIfSupported();
  } else if (type == BackendType::XGMI) {
    auto backend = std::make_unique<XgmiBackend>(desc.key, config,
                                                 static_cast<const XgmiBackendConfig&>(beConfig));
    backends.insert({type, std::move(backend)});
    InvalidateRouteCache();
  } else if (type == BackendType::TCP) {
    assert(backends.find(type) == backends.end());
    auto backend = std::make_unique<TcpBackend>(desc.key, config,
                                                static_cast<const TcpBackendConfig&>(beConfig));

    if (config.port == 0) {
      auto bound_port_opt = backend->GetListenPort();
      if (!bound_port_opt.has_value() || bound_port_opt.value() == 0) {
        MORI_IO_ERROR("IOEngine key {} failed to retrieve bound port after TCP backend init",
                      desc.key);
        assert(false && "Failed to retrieve bound port after TCP backend init");
      } else {
        uint16_t bound_port = bound_port_opt.value();
        desc.port = bound_port;
        this->config.port = bound_port;
        MORI_IO_INFO("IOEngine key {} bound ephemeral port {}", desc.key, bound_port);
      }
    }

    backends.insert({type, std::move(backend)});
  } else {
    assert(false && "not implemented");
  }
  MORI_IO_INFO("Create backend type {}", static_cast<uint32_t>(type));
}

bool IOEngine::SupportsXgmiBackendByP2P() const {
  int numDevices = 0;
  hipError_t err = hipGetDeviceCount(&numDevices);
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI probe skipped: hipGetDeviceCount failed: {}", hipGetErrorString(err));
    return false;
  }

  if (numDevices <= 0) {
    MORI_IO_INFO("XGMI probe skipped: no GPU device found");
    return false;
  }

  if (numDevices == 1) {
    MORI_IO_INFO("XGMI probe succeeded with single GPU device");
    return true;
  }

  for (int src = 0; src < numDevices; ++src) {
    for (int dst = 0; dst < numDevices; ++dst) {
      if (src == dst) continue;
      int canAccess = 0;
      err = hipDeviceCanAccessPeer(&canAccess, src, dst);
      if (err != hipSuccess) {
        MORI_IO_WARN("XGMI probe cannot query P2P from device {} to {}: {}", src, dst,
                     hipGetErrorString(err));
        continue;
      }
      if (canAccess != 0) {
        MORI_IO_INFO("XGMI probe succeeded: P2P is available between GPU {} and {}", src, dst);
        return true;
      }
    }
  }

  MORI_IO_INFO("XGMI probe skipped: no GPU peer access found");
  return false;
}

void IOEngine::EnsureXgmiBackendCreatedIfSupported() {
  bool isAutoXgmiDisabled = false;
  const char* disableAutoXgmi = std::getenv("MORI_DISABLE_AUTO_XGMI");
  if (disableAutoXgmi != nullptr) {
    isAutoXgmiDisabled = disableAutoXgmi[0] != '\0' && disableAutoXgmi[0] != '0';
    if (isAutoXgmiDisabled) {
      MORI_IO_INFO("Auto XGMI creation is disabled by MORI_DISABLE_AUTO_XGMI");
      return;
    }
  }

  if (backends.find(BackendType::XGMI) != backends.end()) {
    return;
  }

  if (!SupportsXgmiBackendByP2P()) {
    return;
  }

  try {
    XgmiBackendConfig xgmiConfig{};
    auto backend = std::make_unique<XgmiBackend>(desc.key, config, xgmiConfig);
    backends.insert({BackendType::XGMI, std::move(backend)});
    InvalidateRouteCache();
    MORI_IO_INFO("Auto-created XGMI backend after RDMA initialization");
  } catch (const std::exception& e) {
    MORI_IO_WARN("Auto-create XGMI backend failed: {}", e.what());
  } catch (...) {
    MORI_IO_WARN("Auto-create XGMI backend failed due to unknown error");
  }
}

void IOEngine::RemoveBackend(BackendType type) {
  backends.erase(type);
  InvalidateRouteCache();
}

void IOEngine::RegisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->RegisterRemoteEngine(remote);
  }
  InvalidateRouteCache();
  MORI_IO_INFO("Register remote engine {} node_id {} hostname {}", remote.key.c_str(),
               remote.nodeId.c_str(), remote.hostname.c_str());
}

void IOEngine::DeregisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->DeregisterRemoteEngine(remote);
  }
  InvalidateRouteCache();
  MORI_IO_INFO("Deregister remote engine {}", remote.key.c_str());
}

MemoryDesc IOEngine::RegisterMemory(void* data, size_t size, int device, MemoryLocationType loc) {
  MemoryDesc memDesc;
  memDesc.engineKey = desc.key;
  memDesc.id = nextMemUid.fetch_add(1, std::memory_order_relaxed);
  memDesc.deviceId = device;
  memDesc.data = reinterpret_cast<uintptr_t>(data);
  memDesc.size = size;
  memDesc.loc = loc;

  for (auto& it : backends) {
    it.second->RegisterMemory(memDesc);
  }

  memPool.insert({memDesc.id, memDesc});
  MORI_IO_TRACE("Register memory address {} size {} device {} loc {} with id {}", data, size,
                device, static_cast<uint32_t>(loc), memDesc.id);
  return memDesc;
}

void IOEngine::DeregisterMemory(const MemoryDesc& desc) {
  for (auto& it : backends) {
    it.second->DeregisterMemory(desc);
  }
  memPool.erase(desc.id);
  MORI_IO_TRACE("Deregister memory {} at address {}", desc.id, desc.data);
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  TransferUniqueId id = nextTransferUid.fetch_add(1, std::memory_order_relaxed);
  MORI_IO_TRACE("Allocate transfer uid {}", id);
  return id;
}

Backend* IOEngine::SelectBackend(const MemoryDesc& local, const MemoryDesc& remote) {
  if (backends.empty()) {
    return nullptr;
  }

  RouteCacheKey routeKey{remote.engineKey, local.loc, remote.loc, local.deviceId, remote.deviceId};

  if (auto cachedType = QueryRouteCache(routeKey); cachedType.has_value()) {
    auto cachedBackend = backends.find(cachedType.value());
    if (cachedBackend != backends.end()) {
      if (cachedType.value() != BackendType::XGMI ||
          cachedBackend->second->CanHandle(local, remote)) {
        return cachedBackend->second.get();
      }
    }
  }

  auto xgmiIt = backends.find(BackendType::XGMI);
  bool isIntraNodeCapable = xgmiIt != backends.end() && xgmiIt->second->CanHandle(local, remote);
  // For intra-node GPU transfers, prefer XGMI when it can handle this pair.
  if (isIntraNodeCapable) {
    UpdateRouteCache(routeKey, BackendType::XGMI);
    return xgmiIt->second.get();
  }

  auto rdmaIt = backends.find(BackendType::RDMA);
  if (rdmaIt != backends.end()) {
    UpdateRouteCache(routeKey, BackendType::RDMA);
    return rdmaIt->second.get();
  }

  BackendType fallbackType = backends.begin()->first;
  UpdateRouteCache(routeKey, fallbackType);
  return backends.begin()->second.get();
}

void IOEngine::InvalidateRouteCache() {
  std::unique_lock<std::shared_mutex> lock(routeCacheMu);
  routeCache.clear();
}

void IOEngine::UpdateRouteCache(const RouteCacheKey& key, BackendType backendType) {
  std::unique_lock<std::shared_mutex> lock(routeCacheMu);
  routeCache[key] = backendType;
}

std::optional<BackendType> IOEngine::QueryRouteCache(const RouteCacheKey& key) const {
  std::shared_lock<std::shared_mutex> lock(routeCacheMu);
  auto it = routeCache.find(key);
  if (it == routeCache.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string IOEngine::ResolveNodeId(const std::string& hostname) const {
  const char* nodeIdEnv = std::getenv("MORI_IO_NODE_ID");
  if (nodeIdEnv != nullptr && nodeIdEnv[0] != '\0') {
    return std::string(nodeIdEnv);
  }
  return hostname;
}

#define SELECT_BACKEND_AND_RETURN_IF_NONE(local, remote, status, backend)     \
  backend = SelectBackend(local, remote);                                     \
  if (backend == nullptr) {                                                   \
    if (status != nullptr) {                                                  \
      status->Update(StatusCode::ERR_BAD_STATE,                               \
                     "No available backend found, create backend first");     \
    }                                                                         \
    MORI_IO_ERROR("No available backend found, please create backend first"); \
    return;                                                                   \
  }

void IOEngine::Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localDest, remoteSrc, status, backend);
  backend->Read(localDest, localOffset, remoteSrc, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Engine read error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngine::Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localSrc, remoteDest, status, backend);
  backend->Write(localSrc, localOffset, remoteDest, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Engine write error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngine::BatchRead(const MemDescVec& localDest, const BatchSizeVec& localOffsets,
                         const MemDescVec& remoteSrc, const BatchSizeVec& remoteOffsets,
                         const BatchSizeVec& sizes, TransferStatusPtrVec& status,
                         TransferUniqueIdVec& ids) {
  MORI_IO_FUNCTION_TIMER;
  size_t batchSize = localDest.size();
  assert(batchSize == remoteSrc.size());
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());
  assert(batchSize == sizes.size());
  assert(batchSize == status.size());
  assert(batchSize == ids.size());

  for (size_t i = 0; i < batchSize; i++) {
    Backend* backend = nullptr;
    SELECT_BACKEND_AND_RETURN_IF_NONE(localDest[i], remoteSrc[i], status[i], backend);
    backend->BatchRead(localDest[i], localOffsets[i], remoteSrc[i], remoteOffsets[i], sizes[i],
                       status[i], ids[i]);
    if (status[i]->Failed()) {
      MORI_IO_ERROR("Engine batch read error {} message {}", status[i]->CodeUint32(),
                    status[i]->Message());
    }
  }
}

void IOEngine::BatchWrite(const MemDescVec& localSrc, const BatchSizeVec& localOffsets,
                          const MemDescVec& remoteDest, const BatchSizeVec& remoteOffsets,
                          const BatchSizeVec& sizes, TransferStatusPtrVec& status,
                          TransferUniqueIdVec& ids) {
  MORI_IO_FUNCTION_TIMER;
  size_t batchSize = localSrc.size();
  assert(batchSize == remoteDest.size());
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());
  assert(batchSize == sizes.size());
  assert(batchSize == status.size());
  assert(batchSize == ids.size());

  for (size_t i = 0; i < batchSize; i++) {
    Backend* backend = nullptr;
    SELECT_BACKEND_AND_RETURN_IF_NONE(localSrc[i], remoteDest[i], status[i], backend);
    backend->BatchWrite(localSrc[i], localOffsets[i], remoteDest[i], remoteOffsets[i], sizes[i],
                        status[i], ids[i]);
    if (status[i]->Failed()) {
      MORI_IO_ERROR("Engine batch write error {} message {}", status[i]->CodeUint32(),
                    status[i]->Message());
    }
  }
}

std::optional<IOEngineSession> IOEngine::CreateSession(const MemoryDesc& local,
                                                       const MemoryDesc& remote) {
  IOEngineSession sess{};
  sess.engine = this;

  Backend* backend = SelectBackend(local, remote);
  if (backend == nullptr) {
    return std::nullopt;
  }
  sess.backendSess.reset(backend->CreateSession(local, remote));

  return sess;
}

bool IOEngine::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) {
  // status->SetCode(StatusCode::SUCCESS);
  for (auto& it : backends) {
    bool popped = it.second->PopInboundTransferStatus(remote, id, status);
    if (popped) return true;
  }
  return false;
}

}  // namespace io
}  // namespace mori
