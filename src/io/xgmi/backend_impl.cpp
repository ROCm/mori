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

#include <errno.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {
namespace {

int getScatterGatherKernelThreshold() {
  static const int threshold = []() {
    const char* env = std::getenv("MORI_IO_XGMI_SCATTER_GATHER_THRESHOLD");
    if (env != nullptr && env[0] != '\0') {
      errno = 0;
      char* end = nullptr;
      long val = std::strtol(env, &end, 10);
      if (errno == 0 && end != env && *end == '\0' && val >= 0 && val <= INT_MAX) {
        MORI_IO_WARN(
            "XGMI: Experimental scatter/gather batch-copy optimization enabled via "
            "MORI_IO_XGMI_SCATTER_GATHER_THRESHOLD={}.",
            val);
        return static_cast<int>(val);
      }
      MORI_IO_WARN(
          "XGMI: Ignoring invalid MORI_IO_XGMI_SCATTER_GATHER_THRESHOLD='{}'. "
          "Scatter/gather remains disabled.",
          env);
    }
    return INT_MAX;
  }();
  return threshold;
}

// Keep caller-visible HIP current device unchanged across MORI internals.
class ScopedHipDeviceGuard {
 public:
  ScopedHipDeviceGuard() {
    hipError_t err = hipGetDevice(&originalDevice_);
    if (err != hipSuccess) {
      valid_ = false;
      MORI_IO_WARN("XGMI: Failed to query current device for guard: {}", hipGetErrorString(err));
    }
  }

  ~ScopedHipDeviceGuard() {
    if (!valid_) return;
    hipError_t err = hipSetDevice(originalDevice_);
    if (err != hipSuccess) {
      MORI_IO_WARN("XGMI: Failed to restore current device {}: {}", originalDevice_,
                   hipGetErrorString(err));
    }
  }

 private:
  int originalDevice_{0};
  bool valid_{true};
};

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                        XgmiBackendSession                                      */
/* ---------------------------------------------------------------------------------------------- */

XgmiBackendSession::XgmiBackendSession(const XgmiBackendConfig& config, void* localAddr,
                                       void* remoteAddr, int localDevice, int remoteDevice,
                                       bool isIpcSession, XgmiBackend* backend,
                                       StreamPool* streamPool, EventPool* eventPool)
    : config(config),
      localAddr(localAddr),
      remoteAddr(remoteAddr),
      localDevice(localDevice),
      remoteDevice(remoteDevice),
      isIpcSession(isIpcSession),
      backend(backend),
      streamPool(streamPool),
      eventPool(eventPool) {}

void XgmiBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  ScopedHipDeviceGuard deviceGuard;
  const int srcDevice = isRead ? remoteDevice : localDevice;
  const int dstDevice = isRead ? localDevice : remoteDevice;

  hipError_t err = hipSetDevice(dstDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
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
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("XGMI: hipMemcpyAsync failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, dstDevice);
      return;
    }
  } else {
    err = hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
    if (err != hipSuccess) {
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("XGMI: hipMemcpyPeerAsync failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, dstDevice);
      return;
    }
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
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
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(err));
    }
    pool->PutEvent(event, dstDevice);
  });
  MORI_IO_TRACE("XGMI: Transfer issued, id={}, size={}, isRead={}", id, size, isRead);
}

void XgmiBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  ScopedHipDeviceGuard deviceGuard;
  size_t batchSize = sizes.size();
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());

  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  const int srcDevice = isRead ? remoteDevice : localDevice;
  const int dstDevice = isRead ? localDevice : remoteDevice;

  // For IPC writes the scatter/gather kernel must run on localDevice because
  // remoteAddr was IPC-opened in localDevice's context.  The hipMemcpyPeerAsync
  // fallback handles cross-device routing internally so it stays on dstDevice.
  const bool kernelOnLocal = isIpcSession && !isRead;
  const int kernelDevice = kernelOnLocal ? localDevice : dstDevice;

  hipError_t err = hipSetDevice(kernelDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("XGMI: Failed to set device: ") + hipGetErrorString(err));
    return;
  }

  hipStream_t stream = streamPool->GetNextStream(kernelDevice);
  hipEvent_t event = eventPool->GetEvent(kernelDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "XGMI: Failed to get stream or event from pool");
    if (event != nullptr) {
      eventPool->PutEvent(event, kernelDevice);
    }
    return;
  }

  // Sort indices by remote offset to maximize contiguous-run merging
  std::vector<size_t> indices(batchSize);
  std::iota(indices.begin(), indices.end(), 0);
  if (!std::is_sorted(remoteOffsets.begin(), remoteOffsets.end())) {
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return remoteOffsets[a] < remoteOffsets[b]; });
  }

  struct MergedSeg {
    size_t localOff;
    size_t remoteOff;
    size_t sz;
  };
  std::vector<MergedSeg> segments;
  segments.reserve(batchSize);

  for (size_t i = 0; i < batchSize; ++i) {
    size_t idx = indices[i];
    if (sizes[idx] == 0) continue;

    if (!segments.empty()) {
      MergedSeg& last = segments.back();
      bool localContig = (last.localOff + last.sz) == localOffsets[idx];
      bool remoteContig = (last.remoteOff + last.sz) == remoteOffsets[idx];
      if (localContig && remoteContig) {
        last.sz += sizes[idx];
        continue;
      }
    }
    segments.push_back({localOffsets[idx], remoteOffsets[idx], sizes[idx]});
  }

  if (segments.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    eventPool->PutEvent(event, kernelDevice);
    return;
  }

  void* srcBase = isRead ? remoteAddr : localAddr;
  void* dstBase = isRead ? localAddr : remoteAddr;

  hipFunction_t sgFunc =
      backend != nullptr ? backend->GetScatterGatherFunc(kernelDevice) : nullptr;
  bool useKernel =
      sgFunc != nullptr && static_cast<int>(segments.size()) > getScatterGatherKernelThreshold();

  if (useKernel) {
    size_t numSegs = segments.size();
    size_t metaBytes = numSegs * sizeof(size_t) * 3;

    std::vector<size_t> hostMeta(numSegs * 3);
    size_t* hSrcOff = hostMeta.data();
    size_t* hDstOff = hostMeta.data() + numSegs;
    size_t* hSizes = hostMeta.data() + numSegs * 2;
    for (size_t i = 0; i < numSegs; ++i) {
      hSrcOff[i] = isRead ? segments[i].remoteOff : segments[i].localOff;
      hDstOff[i] = isRead ? segments[i].localOff : segments[i].remoteOff;
      hSizes[i] = segments[i].sz;
    }

    size_t* dMeta = nullptr;
    err = hipMalloc(&dMeta, metaBytes);
    if (err != hipSuccess) {
      MORI_IO_WARN("XGMI: scatter/gather metadata alloc failed, falling back to hipMemcpy");
      useKernel = false;
    }

    if (useKernel) {
      err = hipMemcpyAsync(dMeta, hostMeta.data(), metaBytes, hipMemcpyHostToDevice, stream);
      if (err != hipSuccess) {
        (void)hipFree(dMeta);
        MORI_IO_WARN("XGMI: scatter/gather metadata upload failed, falling back to hipMemcpy");
        useKernel = false;
      }
    }

    if (useKernel) {
      size_t* dSrcOff = dMeta;
      size_t* dDstOff = dMeta + numSegs;
      size_t* dSizes = dMeta + numSegs * 2;

      int threadsPerBlock = 256;
      int numBlocks = std::min(static_cast<int>(numSegs), 1024);

      const char* srcPtr = reinterpret_cast<const char*>(srcBase);
      char* dstPtr = reinterpret_cast<char*>(dstBase);
      int numSegsInt = static_cast<int>(numSegs);
      void* kernelArgs[] = {&srcPtr, &dstPtr, &dSrcOff, &dDstOff, &dSizes, &numSegsInt};
      err = hipModuleLaunchKernel(sgFunc, numBlocks, 1, 1, threadsPerBlock, 1, 1, 0, stream,
                                  kernelArgs, nullptr);
      if (err != hipSuccess) {
        status->Update(
            StatusCode::ERR_GPU_OP,
            std::string("XGMI: scatter/gather kernel launch failed: ") + hipGetErrorString(err));
        (void)hipFree(dMeta);
        eventPool->PutEvent(event, kernelDevice);
        return;
      }

      err = hipEventRecord(event, stream);
      if (err != hipSuccess) {
        status->Update(StatusCode::ERR_GPU_OP,
                       std::string("XGMI: hipEventRecord failed: ") + hipGetErrorString(err));
        (void)hipFree(dMeta);
        eventPool->PutEvent(event, kernelDevice);
        return;
      }

      status->SetCode(StatusCode::IN_PROGRESS);
      status->SetWaitCallback([status, event, kernelDevice, pool = eventPool, dMeta]() {
        hipError_t e = hipEventSynchronize(event);
        if (e == hipSuccess) {
          status->SetCode(StatusCode::SUCCESS);
        } else {
          status->Update(StatusCode::ERR_GPU_OP,
                         std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(e));
        }
        (void)hipFree(dMeta);
        pool->PutEvent(event, kernelDevice);
      });
      MORI_IO_TRACE("XGMI: Batch transfer via scatter/gather kernel, id={}, segments={}, isRead={}",
                    id, numSegs, isRead);
      return;
    }
  }

  // Fallback: individual hipMemcpy per merged segment — always uses dstDevice
  // because hipMemcpyPeerAsync handles cross-device routing internally.
  int eventDevice = kernelDevice;
  if (kernelOnLocal) {
    (void)hipSetDevice(dstDevice);
    hipStream_t memcpyStream = streamPool->GetNextStream(dstDevice);
    hipEvent_t memcpyEvent = eventPool->GetEvent(dstDevice);
    if (memcpyStream != nullptr && memcpyEvent != nullptr) {
      eventPool->PutEvent(event, kernelDevice);
      stream = memcpyStream;
      event = memcpyEvent;
      eventDevice = dstDevice;
    } else {
      // Cannot obtain dstDevice resources; stay on kernelDevice.
      (void)hipSetDevice(kernelDevice);
    }
  }

  for (auto& seg : segments) {
    void* src = isRead ? static_cast<char*>(remoteAddr) + seg.remoteOff
                       : static_cast<char*>(localAddr) + seg.localOff;
    void* dst = isRead ? static_cast<char*>(localAddr) + seg.localOff
                       : static_cast<char*>(remoteAddr) + seg.remoteOff;

    if (srcDevice == dstDevice) {
      err = hipMemcpyAsync(dst, src, seg.sz, hipMemcpyDeviceToDevice, stream);
    } else {
      err = hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, seg.sz, stream);
    }
    if (err != hipSuccess) {
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("XGMI: memcpy failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, eventDevice);
      return;
    }
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("XGMI: hipEventRecord failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, eventDevice);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  status->SetWaitCallback([status, event, eventDevice, pool = eventPool]() {
    hipError_t e = hipEventSynchronize(event);
    if (e == hipSuccess) {
      status->SetCode(StatusCode::SUCCESS);
    } else {
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(e));
    }
    pool->PutEvent(event, eventDevice);
  });
  MORI_IO_TRACE("XGMI: Batch transfer via hipMemcpy, id={}, segments={}, isRead={}", id,
                segments.size(), isRead);
}

bool XgmiBackendSession::Alive() const { return true; }

/* ---------------------------------------------------------------------------------------------- */
/*                                           XgmiBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

XgmiBackend::XgmiBackend(EngineKey k, const IOEngineConfig& engConfig,
                         const XgmiBackendConfig& beConfig)
    : myEngKey(k), config(beConfig) {
  if (auto nodeId = mori::env::GetString("MORI_IO_NODE_ID"); nodeId.has_value()) {
    myNodeId = *nodeId;
  }
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  myHostname = std::string(hostname);
  if (myNodeId.empty()) {
    myNodeId = myHostname;
  }

  streamPool = std::make_unique<StreamPool>(config.numStreams);
  eventPool = std::make_unique<EventPool>(config.numEvents);

  InitializeP2PAccess();

  std::stringstream ss;
  ss << config;
  MORI_IO_INFO("XgmiBackend created with config: {} node_id: {} hostname: {}", ss.str().c_str(),
               myNodeId.c_str(), myHostname.c_str());
}

XgmiBackend::~XgmiBackend() {
  std::unique_lock<std::shared_mutex> lock(ipcMutex);
  for (auto& entry : remoteIpcHandles) {
    if (entry.second.remappedAddr != nullptr) {
      hipError_t closeErr = hipIpcCloseMemHandle(entry.second.remappedAddr);
      if (closeErr != hipSuccess) {
        MORI_IO_WARN("XGMI: Failed to close IPC mem handle: {}", hipGetErrorString(closeErr));
      }
    }
  }
  remoteIpcHandles.clear();
  localIpcHandles.clear();

  for (auto& mod : scatterGatherModules_) {
    if (mod != nullptr) {
      hipModuleUnload(mod);
    }
  }
  scatterGatherModules_.clear();
  scatterGatherFuncs_.clear();
}

void XgmiBackend::LoadScatterGatherModule(const std::string& hsacoPath) {
  scatterGatherHsacoPath_ = hsacoPath;
  scatterGatherModules_.resize(numDevices, nullptr);
  scatterGatherFuncs_.resize(numDevices, nullptr);
  MORI_IO_INFO("XGMI: Scatter/gather kernel registered from {}", hsacoPath);
}

hipFunction_t XgmiBackend::GetScatterGatherFunc(int deviceId) {
  if (scatterGatherHsacoPath_.empty() || deviceId < 0 || deviceId >= numDevices) {
    return nullptr;
  }
  if (scatterGatherFuncs_[deviceId] != nullptr) {
    return scatterGatherFuncs_[deviceId];
  }
  ScopedHipDeviceGuard deviceGuard;
  hipError_t err = hipSetDevice(deviceId);
  if (err != hipSuccess) {
    return nullptr;
  }
  err = hipModuleLoad(&scatterGatherModules_[deviceId], scatterGatherHsacoPath_.c_str());
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI: Failed to load scatter/gather module on device {}: {}", deviceId,
                 hipGetErrorString(err));
    return nullptr;
  }
  err = hipModuleGetFunction(&scatterGatherFuncs_[deviceId], scatterGatherModules_[deviceId],
                             "scatterGatherCopyKernel");
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI: Failed to get scatterGatherCopyKernel on device {}: {}", deviceId,
                 hipGetErrorString(err));
    hipModuleUnload(scatterGatherModules_[deviceId]);
    scatterGatherModules_[deviceId] = nullptr;
    return nullptr;
  }
  MORI_IO_INFO("XGMI: Loaded scatter/gather kernel on device {}", deviceId);
  return scatterGatherFuncs_[deviceId];
}

void XgmiBackend::InitializeP2PAccess() {
  hipError_t err = hipGetDeviceCount(&numDevices);
  if (err != hipSuccess || numDevices <= 0) {
    MORI_IO_WARN("XGMI: Failed to get device count or no devices found");
    numDevices = 0;
    return;
  }

  p2pMatrix.resize(numDevices, std::vector<bool>(numDevices, false));
  ScopedHipDeviceGuard deviceGuard;

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
          hipError_t clearErr = hipGetLastError();
          if (clearErr != hipSuccess) {
            MORI_IO_WARN("XGMI: Failed to clear peer access error: {}",
                         hipGetErrorString(clearErr));
          }
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

  IpcCacheKey cacheKey{desc.id, localDeviceId};
  {
    std::shared_lock<std::shared_mutex> rlock(ipcMutex);
    auto it = remoteIpcHandles.find(cacheKey);
    if (it != remoteIpcHandles.end() && it->second.remappedAddr != nullptr) {
      return it->second.remappedAddr;
    }
  }

  hipIpcMemHandle_t handle;
  static_assert(sizeof(handle) == kIpcHandleSize, "IPC handle size mismatch");
  std::memcpy(&handle, desc.ipcHandle.data(), sizeof(handle));

  ScopedHipDeviceGuard deviceGuard;
  hipError_t err = hipSetDevice(localDeviceId);
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI: Failed to set device {} for IPC open: {}", localDeviceId,
                 hipGetErrorString(err));
    return nullptr;
  }

  void* remappedAddr = nullptr;
  err = hipIpcOpenMemHandle(&remappedAddr, handle, hipIpcMemLazyEnablePeerAccess);
  if (err != hipSuccess) {
    hipError_t clearErr = hipGetLastError();
    if (clearErr != hipSuccess) {
      MORI_IO_WARN("XGMI: Failed to clear IPC open error: {}", hipGetErrorString(clearErr));
    }
    if (IsP2PAccessible(localDeviceId, desc.deviceId)) {
      MORI_IO_TRACE("XGMI: IPC failed, using direct P2P pointer for id={}", desc.id);
      return reinterpret_cast<void*>(desc.data);
    }
    MORI_IO_WARN("XGMI: Failed to open IPC handle for id={} on device {}: {}", desc.id,
                 localDeviceId, hipGetErrorString(err));
    return nullptr;
  }

  std::unique_lock<std::shared_mutex> wlock(ipcMutex);
  remoteIpcHandles[cacheKey] = {handle, remappedAddr, desc.size};
  MORI_IO_TRACE("XGMI: Opened IPC handle for id={} on device {}, remapped={}", desc.id,
                localDeviceId, reinterpret_cast<uintptr_t>(remappedAddr));
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
  int remoteDevice = remote.deviceId;
  void* localAddr = GetRemappedAddress(local, localDevice);
  void* remoteAddr = GetRemappedAddress(remote, localDevice);
  bool ipcSession = (remote.engineKey != myEngKey);

  if (!IsP2PAccessible(localDevice, remoteDevice)) {
    MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                 remoteDevice);
  }

  return new XgmiBackendSession(config, localAddr, remoteAddr, localDevice, remoteDevice,
                                ipcSession, this, streamPool.get(), eventPool.get());
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
  int localDevice = local.deviceId;
  int remoteDevice = remote.deviceId;
  void* remoteAddr = GetRemappedAddress(remote, localDevice);
  bool ipcSession = (remote.engineKey != myEngKey);

  if (!IsP2PAccessible(localDevice, remoteDevice)) {
    MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                 remoteDevice);
  }

  auto sess =
      std::make_unique<XgmiBackendSession>(config, localAddr, remoteAddr, localDevice, remoteDevice,
                                           ipcSession, this, streamPool.get(), eventPool.get());

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
  const EngineDesc& remoteEngine = it->second;
  if (!myNodeId.empty() && !remoteEngine.nodeId.empty()) {
    return remoteEngine.nodeId == myNodeId;
  }
  return remoteEngine.hostname == myHostname;
}

}  // namespace io
}  // namespace mori
