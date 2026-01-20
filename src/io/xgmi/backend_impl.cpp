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
#include "src/io/xgmi/backend_impl.hpp"

#include <limits.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                        XgmiBackendSession                                      */
/* ---------------------------------------------------------------------------------------------- */

XgmiBackendSession::XgmiBackendSession(const XgmiBackendConfig& config, void* localAddr,
                                       void* remoteAddr, int localDevice, int remoteDevice,
                                       StreamPool* streamPool, EventPool* eventPool)
    : config(config),
      localAddr(localAddr),
      remoteAddr(remoteAddr),
      localDevice(localDevice),
      remoteDevice(remoteDevice),
      streamPool(streamPool),
      eventPool(eventPool) {}

void XgmiBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  const int srcDevice = isRead ? remoteDevice : localDevice;
  const int dstDevice = isRead ? localDevice : remoteDevice;

  hipError_t err = hipSetDevice(dstDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_RDMA_OP,
                   std::string("XGMI: Failed to set device: ") + hipGetErrorString(err));
    return;
  }

  hipStream_t stream = streamPool->GetNextStream(dstDevice);
  hipEvent_t event = eventPool->GetEvent(dstDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "XGMI: Failed to get stream or event from pool");
    if (event != nullptr) {
      eventPool->PutEvent(event, dstDevice);
    }
    return;
  }

  void* src = isRead ? static_cast<char*>(remoteAddr) + remoteOffset
                     : static_cast<char*>(localAddr) + localOffset;
  void* dst = isRead ? static_cast<char*>(localAddr) + localOffset
                     : static_cast<char*>(remoteAddr) + remoteOffset;

  if (srcDevice == dstDevice) {
    err = hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
      status->Update(StatusCode::ERR_RDMA_OP,
                     std::string("XGMI: hipMemcpyAsync failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, dstDevice);
      return;
    }
  } else {
    err = hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
    if (err != hipSuccess) {
      status->Update(StatusCode::ERR_RDMA_OP,
                     std::string("XGMI: hipMemcpyPeerAsync failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, dstDevice);
      return;
    }
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_RDMA_OP,
                   std::string("XGMI: hipEventRecord failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, dstDevice);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  status->SetWaitCallback([status, event, dstDevice, pool = eventPool]() {
    hipError_t err = hipEventSynchronize(event);
    if (err == hipSuccess) {
      status->SetCode(StatusCode::SUCCESS);
    } else {
      status->Update(StatusCode::ERR_RDMA_OP,
                     std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(err));
    }
    pool->PutEvent(event, dstDevice);
  });
  MORI_IO_TRACE("XGMI: Transfer issued, id={}, size={}, isRead={}", id, size, isRead);
}

void XgmiBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  size_t batchSize = sizes.size();
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());

  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  const int srcDevice = isRead ? remoteDevice : localDevice;
  const int dstDevice = isRead ? localDevice : remoteDevice;

  hipError_t err = hipSetDevice(dstDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_RDMA_OP,
                   std::string("XGMI: Failed to set device: ") + hipGetErrorString(err));
    return;
  }

  hipStream_t stream = streamPool->GetNextStream(dstDevice);
  hipEvent_t event = eventPool->GetEvent(dstDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "XGMI: Failed to get stream or event from pool");
    if (event != nullptr) {
      eventPool->PutEvent(event, dstDevice);
    }
    return;
  }

  size_t runStartIdx = 0;
  size_t runLocalOff = 0;
  size_t runRemoteOff = 0;
  size_t runSize = 0;
  bool hasRun = false;

  auto flush_run = [&](size_t failedAtIdx) -> bool {
    if (!hasRun) return true;

    void* src = isRead ? static_cast<char*>(remoteAddr) + runRemoteOff
                       : static_cast<char*>(localAddr) + runLocalOff;
    void* dst = isRead ? static_cast<char*>(localAddr) + runLocalOff
                       : static_cast<char*>(remoteAddr) + runRemoteOff;

    if (srcDevice == dstDevice) {
      err = hipMemcpyAsync(dst, src, runSize, hipMemcpyDeviceToDevice, stream);
      if (err != hipSuccess) {
        status->Update(StatusCode::ERR_RDMA_OP,
                       std::string("XGMI: hipMemcpyAsync failed at batch ") +
                           std::to_string(failedAtIdx) + ": " + hipGetErrorString(err));
        return false;
      }
    } else {
      err = hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, runSize, stream);
      if (err != hipSuccess) {
        status->Update(StatusCode::ERR_RDMA_OP,
                       std::string("XGMI: hipMemcpyPeerAsync failed at batch ") +
                           std::to_string(failedAtIdx) + ": " + hipGetErrorString(err));
        return false;
      }
    }
    return true;
  };

  for (size_t i = 0; i < batchSize; ++i) {
    const size_t sz = sizes[i];
    if (sz == 0) continue;

    if (!hasRun) {
      runStartIdx = i;
      runLocalOff = localOffsets[i];
      runRemoteOff = remoteOffsets[i];
      runSize = sz;
      hasRun = true;
      continue;
    }

    const bool remoteContiguous = (runRemoteOff + runSize) == remoteOffsets[i];
    const bool localContiguous = (runLocalOff + runSize) == localOffsets[i];

    if (remoteContiguous && localContiguous) {
      runSize += sz;
      continue;
    }

    if (!flush_run(runStartIdx)) {
      eventPool->PutEvent(event, dstDevice);
      return;
    }

    runStartIdx = i;
    runLocalOff = localOffsets[i];
    runRemoteOff = remoteOffsets[i];
    runSize = sz;
    hasRun = true;
  }

  if (!flush_run(runStartIdx)) {
    eventPool->PutEvent(event, dstDevice);
    return;
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_RDMA_OP,
                   std::string("XGMI: hipEventRecord failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, dstDevice);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  status->SetWaitCallback([status, event, dstDevice, pool = eventPool]() {
    hipError_t err = hipEventSynchronize(event);
    if (err == hipSuccess) {
      status->SetCode(StatusCode::SUCCESS);
    } else {
      status->Update(StatusCode::ERR_RDMA_OP,
                     std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(err));
    }
    pool->PutEvent(event, dstDevice);
  });
  MORI_IO_TRACE("XGMI: Batch transfer issued, id={}, batchSize={}, isRead={}", id, batchSize,
                isRead);
}

bool XgmiBackendSession::Alive() const { return true; }

/* ---------------------------------------------------------------------------------------------- */
/*                                           XgmiBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

XgmiBackend::XgmiBackend(EngineKey k, const IOEngineConfig& engConfig,
                         const XgmiBackendConfig& beConfig)
    : myEngKey(k), config(beConfig) {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  myHostname = std::string(hostname);

  streamPool = std::make_unique<StreamPool>(config.numStreams);
  eventPool = std::make_unique<EventPool>(config.numEvents);

  InitializeP2PAccess();

  std::stringstream ss;
  ss << config;
  MORI_IO_INFO("XgmiBackend created with config: {} hostname: {}", ss.str().c_str(),
               myHostname.c_str());
}

XgmiBackend::~XgmiBackend() {
  std::unique_lock<std::shared_mutex> lock(ipcMutex);
  for (auto& entry : remoteIpcHandles) {
    if (entry.second.remappedAddr != nullptr) {
      hipIpcCloseMemHandle(entry.second.remappedAddr);
    }
  }
  remoteIpcHandles.clear();
  localIpcHandles.clear();
}

void XgmiBackend::InitializeP2PAccess() {
  hipError_t err = hipGetDeviceCount(&numDevices);
  if (err != hipSuccess || numDevices <= 0) {
    MORI_IO_WARN("XGMI: Failed to get device count or no devices found");
    numDevices = 0;
    return;
  }

  p2pMatrix.resize(numDevices, std::vector<bool>(numDevices, false));

  for (int i = 0; i < numDevices; ++i) {
    err = hipSetDevice(i);
    if (err != hipSuccess) {
      MORI_IO_WARN("XGMI: Failed to set device {}", i);
      continue;
    }

    for (int j = 0; j < numDevices; ++j) {
      if (i == j) {
        p2pMatrix[i][j] = true;
        continue;
      }

      int canAccess = 0;
      err = hipDeviceCanAccessPeer(&canAccess, i, j);
      if (err != hipSuccess) {
        MORI_IO_WARN("XGMI: Failed to query P2P access from device {} to {}", i, j);
        continue;
      }

      if (canAccess) {
        hipError_t enableErr = hipDeviceEnablePeerAccess(j, 0);
        if (enableErr == hipErrorPeerAccessAlreadyEnabled) {
          hipGetLastError();
          p2pMatrix[i][j] = true;
          MORI_IO_TRACE("XGMI: P2P access already enabled from device {} to {}", i, j);
        } else if (enableErr != hipSuccess) {
          MORI_IO_WARN("XGMI: Failed to enable P2P access from device {} to {}: {}", i, j,
                       hipGetErrorString(enableErr));
        } else {
          p2pMatrix[i][j] = true;
          MORI_IO_TRACE("XGMI: Enabled P2P access from device {} to {}", i, j);
        }
      } else {
        MORI_IO_TRACE("XGMI: P2P access not available from device {} to {}", i, j);
      }
    }
  }
}

bool XgmiBackend::IsP2PAccessible(int srcDevice, int dstDevice) const {
  if (srcDevice < 0 || srcDevice >= numDevices || dstDevice < 0 || dstDevice >= numDevices) {
    return false;
  }
  return p2pMatrix[srcDevice][dstDevice];
}

void XgmiBackend::RegisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  remoteEngines[desc.key] = desc;
  MORI_IO_TRACE("XGMI: Registered remote engine {} hostname {}", desc.key, desc.hostname);
}

void XgmiBackend::DeregisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  remoteEngines.erase(desc.key);
  MORI_IO_TRACE("XGMI: Deregistered remote engine {}", desc.key);
}

void XgmiBackend::RegisterMemory(MemoryDesc& desc) {
  if (desc.loc != MemoryLocationType::GPU) {
    MORI_IO_TRACE("XGMI: Skipping non-GPU memory registration for id={}", desc.id);
    return;
  }

  hipIpcMemHandle_t handle;
  hipError_t err = hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(desc.data));
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI: Failed to get IPC handle for memory id={}: {}", desc.id,
                 hipGetErrorString(err));
    return;
  }

  static_assert(sizeof(handle) == kIpcHandleSize, "IPC handle size mismatch");
  std::memcpy(desc.ipcHandle.data(), &handle, sizeof(handle));

  std::unique_lock<std::shared_mutex> lock(ipcMutex);
  localIpcHandles[desc.id] = handle;
  MORI_IO_TRACE("XGMI: Registered memory id={}, addr={}, size={}", desc.id, desc.data, desc.size);
}

void XgmiBackend::DeregisterMemory(const MemoryDesc& desc) {
  std::unique_lock<std::shared_mutex> lock(ipcMutex);
  localIpcHandles.erase(desc.id);
  InvalidateSessionsForMemory(desc.id);
  MORI_IO_TRACE("XGMI: Deregistered memory id={}", desc.id);
}

void* XgmiBackend::GetRemappedAddress(const MemoryDesc& desc, int localDeviceId) {
  if (desc.engineKey == myEngKey) {
    return reinterpret_cast<void*>(desc.data);
  }

  {
    std::shared_lock<std::shared_mutex> rlock(ipcMutex);
    auto it = remoteIpcHandles.find(desc.id);
    if (it != remoteIpcHandles.end() && it->second.remappedAddr != nullptr) {
      return it->second.remappedAddr;
    }
  }

  hipIpcMemHandle_t handle;
  static_assert(sizeof(handle) == kIpcHandleSize, "IPC handle size mismatch");
  std::memcpy(&handle, desc.ipcHandle.data(), sizeof(handle));

  hipError_t err = hipSetDevice(localDeviceId);
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI: Failed to set device {} for IPC open: {}", localDeviceId,
                 hipGetErrorString(err));
    return nullptr;
  }

  void* remappedAddr = nullptr;
  err = hipIpcOpenMemHandle(&remappedAddr, handle, hipIpcMemLazyEnablePeerAccess);
  if (err != hipSuccess) {
    hipGetLastError();
    if (IsP2PAccessible(localDeviceId, desc.deviceId)) {
      MORI_IO_TRACE("XGMI: IPC failed, using direct P2P pointer for id={}", desc.id);
      return reinterpret_cast<void*>(desc.data);
    }
    MORI_IO_WARN("XGMI: Failed to open IPC handle for id={}: {}", desc.id, hipGetErrorString(err));
    return nullptr;
  }

  std::unique_lock<std::shared_mutex> wlock(ipcMutex);
  remoteIpcHandles[desc.id] = {handle, remappedAddr, desc.size};
  MORI_IO_TRACE("XGMI: Opened IPC handle for id={}, remapped={}", desc.id,
                reinterpret_cast<uintptr_t>(remappedAddr));
  return remappedAddr;
}

void XgmiBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                            const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id, bool isRead) {
  XgmiBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  if (sess == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "XGMI: Failed to create session");
    return;
  }

  sess->ReadWrite(localOffset, remoteOffset, size, status, id, isRead);
}

void XgmiBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                 const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                 const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                 bool isRead) {
  XgmiBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  if (sess == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "XGMI: Failed to create session");
    return;
  }

  sess->BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* XgmiBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  int localDevice = local.deviceId;
  void* localAddr = GetRemappedAddress(local, localDevice);
  void* remoteAddr = GetRemappedAddress(remote, localDevice);
  int remoteDevice = remote.deviceId;

  if (!IsP2PAccessible(localDevice, remoteDevice)) {
    MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                 remoteDevice);
  }

  return new XgmiBackendSession(config, localAddr, remoteAddr, localDevice, remoteDevice,
                                streamPool.get(), eventPool.get());
}

XgmiBackendSession* XgmiBackend::GetOrCreateSessionCached(const MemoryDesc& local,
                                                          const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};

  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) {
    return it->second.get();
  }

  void* localAddr = reinterpret_cast<void*>(local.data);
  void* remoteAddr = GetRemappedAddress(remote, local.deviceId);
  int localDevice = local.deviceId;
  int remoteDevice = remote.deviceId;

  if (!IsP2PAccessible(localDevice, remoteDevice)) {
    MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                 remoteDevice);
  }

  auto sess = std::make_unique<XgmiBackendSession>(config, localAddr, remoteAddr, localDevice,
                                                   remoteDevice, streamPool.get(), eventPool.get());

  XgmiBackendSession* rawPtr = sess.get();
  sessionCache[key] = std::move(sess);

  MORI_IO_TRACE("XGMI: Created session for local.id={}, remote.id={}", local.id, remote.id);
  return rawPtr;
}

void XgmiBackend::InvalidateSessionsForMemory(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  for (auto it = sessionCache.begin(); it != sessionCache.end();) {
    if (it->first.localMemId == id || it->first.remoteMemId == id) {
      it = sessionCache.erase(it);
    } else {
      ++it;
    }
  }
}

bool XgmiBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                           TransferStatus* status) {
  return false;
}

bool XgmiBackend::CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const {
  if (local.loc != MemoryLocationType::GPU || remote.loc != MemoryLocationType::GPU) {
    return false;
  }

  if (!IsP2PAccessible(local.deviceId, remote.deviceId)) {
    return false;
  }

  if (remote.engineKey == myEngKey) {
    return true;
  }

  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  auto it = remoteEngines.find(remote.engineKey);
  if (it == remoteEngines.end()) {
    return false;
  }
  return it->second.hostname == myHostname;
}

}  // namespace io
}  // namespace mori
