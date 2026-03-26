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
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"
#include "mori/utils/hip_helper.hpp"

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

// Lowercase a PCI bus ID for consistent comparison across HIP APIs and sysfs.
// hipDeviceGetPCIBusId may return uppercase hex (e.g. "0000:C1:00.0") while
// sysfs directory names are lowercase ("0000:c1:00.0").
std::string NormalizeBusId(const std::string& busId) {
  std::string result = busId;
  for (auto& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

bool IsIpcHandleEmpty(const std::array<char, kIpcHandleSize>& handle) {
  return std::all_of(handle.begin(), handle.end(), [](char c) { return c == 0; });
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

struct TransferCompletion {
  TransferStatus* status{nullptr};
  hipEvent_t event{nullptr};
  int eventDevice{-1};
  EventPool* eventPool{nullptr};
  std::function<void()> cleanup;
  std::once_flag completionOnce;
  std::atomic<bool> completed{false};
  std::mutex eventMu;

  void FinalizeBlocking() {
    std::lock_guard<std::mutex> lock(eventMu);
    std::call_once(completionOnce, [this]() {
      hipError_t err = hipEventSynchronize(event);
      if (err == hipSuccess) {
        status->SetCode(StatusCode::SUCCESS);
      } else {
        status->Update(StatusCode::ERR_GPU_OP,
                       std::string("XGMI: hipEventSynchronize failed: ") + hipGetErrorString(err));
      }
      if (cleanup) cleanup();
      eventPool->PutEvent(event, eventDevice);
      completed.store(true, std::memory_order_release);
    });
  }

  void FinalizeNonBlocking() {
    std::lock_guard<std::mutex> lock(eventMu);
    if (completed.load(std::memory_order_acquire)) {
      return;
    }

    hipError_t err = hipEventQuery(event);
    if (err == hipErrorNotReady) {
      (void)hipGetLastError();
      return;
    }

    std::call_once(completionOnce, [this, err]() {
      if (err == hipSuccess) {
        status->SetCode(StatusCode::SUCCESS);
      } else {
        status->Update(StatusCode::ERR_GPU_OP,
                       std::string("XGMI: hipEventQuery failed: ") + hipGetErrorString(err));
      }
      if (cleanup) cleanup();
      eventPool->PutEvent(event, eventDevice);
      completed.store(true, std::memory_order_release);
    });
  }
};

void ArmTransferCompletion(TransferStatus* status, hipEvent_t event, int eventDevice,
                           EventPool* eventPool, std::function<void()> cleanup = {}) {
  auto completion = std::make_shared<TransferCompletion>();
  completion->status = status;
  completion->event = event;
  completion->eventDevice = eventDevice;
  completion->eventPool = eventPool;
  completion->cleanup = std::move(cleanup);
  status->SetWaitCallback([completion]() { completion->FinalizeBlocking(); });
  status->SetProgressCallback([completion]() { completion->FinalizeNonBlocking(); });
}

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
  ArmTransferCompletion(status, event, dstDevice, eventPool);
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

  hipFunction_t sgFunc = backend != nullptr ? backend->GetScatterGatherFunc(kernelDevice) : nullptr;
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
      ArmTransferCompletion(status, event, kernelDevice, eventPool,
                            [dMeta]() { (void)hipFree(dMeta); });
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
  ArmTransferCompletion(status, event, eventDevice, eventPool);
  MORI_IO_TRACE("XGMI: Batch transfer via hipMemcpy, id={}, segments={}, isRead={}", id,
                segments.size(), isRead);
}

bool XgmiBackendSession::Alive() const { return true; }

/* ---------------------------------------------------------------------------------------------- */
/*                                           XgmiBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

XgmiBackend::XgmiBackend(EngineKey k, const IOEngineConfig& engConfig,
                         const XgmiBackendConfig& beConfig)
    : myEngKey(k), config(beConfig), myPid(static_cast<int>(getpid())) {
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

    char busId[32] = {0};
    err = hipDeviceGetPCIBusId(busId, sizeof(busId), i);
    if (err == hipSuccess) {
      localDeviceByBusId[NormalizeBusId(std::string(busId))] = i;
    } else {
      MORI_IO_WARN("XGMI: Failed to query PCI bus id for device {}: {}", i, hipGetErrorString(err));
    }

    for (int j = 0; j < numDevices; ++j) {
      if (i == j) {
        p2pMatrix[i][j] = true;
        continue;
      }

      if (mori::MoriEnablePeerAccess(i, j)) {
        p2pMatrix[i][j] = true;
      }
    }
  }

  BuildTopologyMap();
}

bool XgmiBackend::IsP2PAccessible(int srcDevice, int dstDevice) const {
  if (srcDevice < 0 || srcDevice >= numDevices || dstDevice < 0 || dstDevice >= numDevices) {
    return false;
  }
  return p2pMatrix[srcDevice][dstDevice];
}

void XgmiBackend::BuildTopologyMap() {
  namespace fs = std::filesystem;

  // Primary: parse KFD topology to get hive_id per GPU.
  // KFD provides hive_id (shared across all GPUs in the same XGMI mesh) and
  // location_id + domain (which encode the PCI BDF address).
  const fs::path kfdNodes("/sys/class/kfd/kfd/topology/nodes");
  std::error_code ec;
  if (fs::is_directory(kfdNodes, ec)) {
    for (const auto& nodeEntry : fs::directory_iterator(kfdNodes, ec)) {
      if (ec) break;
      fs::path propsFile = nodeEntry.path() / "properties";
      std::ifstream f(propsFile);
      if (!f.is_open()) continue;

      uint64_t hiveId = 0;
      uint64_t locationId = 0;
      uint64_t domain = 0;
      std::string key;
      uint64_t val;
      while (f >> key >> val) {
        if (key == "hive_id")
          hiveId = val;
        else if (key == "location_id")
          locationId = val;
        else if (key == "domain")
          domain = val;
      }

      // hive_id == 0 means CPU node or non-XGMI device
      if (hiveId == 0 || locationId == 0) continue;

      // Decode PCI BDF from location_id: bus[15:8] device[7:3] function[2:0]
      unsigned bus = (locationId >> 8) & 0xff;
      unsigned device = (locationId >> 3) & 0x1f;
      unsigned function = locationId & 0x7;
      char bdf[24];
      std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", static_cast<unsigned>(domain), bus,
                    device, function);
      gpuTopoByBusId[NormalizeBusId(std::string(bdf))] = hiveId;
    }
  }

  if (!gpuTopoByBusId.empty()) {
    MORI_IO_INFO("XGMI: Topology map built from KFD with {} GPU entries", gpuTopoByBusId.size());
    return;
  }

  // Fallback: if KFD is unavailable, use xgmi_physical_id existence as a
  // coarse hive indicator.  All devices that have xgmi_physical_id are treated
  // as belonging to the same hive (assigned a synthetic shared hive ID of 1).
  // This is correct for single-node MI300X where all GPUs are fully connected.
  const fs::path sysDevices("/sys/bus/pci/devices");
  if (!fs::is_directory(sysDevices, ec)) {
    MORI_IO_TRACE("XGMI: sysfs PCI device directory not available, hidden-device XGMI disabled");
    return;
  }

  for (const auto& entry : fs::directory_iterator(sysDevices, ec)) {
    if (ec) break;
    fs::path xgmiFile = entry.path() / "xgmi_physical_id";
    if (!fs::exists(xgmiFile, ec)) continue;

    std::ifstream f(xgmiFile);
    int physicalId;
    if (f.is_open() && (f >> physicalId)) {
      std::string busId = NormalizeBusId(entry.path().filename().string());
      gpuTopoByBusId[busId] = 1;  // synthetic shared hive ID
    }
  }

  if (!gpuTopoByBusId.empty()) {
    MORI_IO_INFO("XGMI: Topology map built from xgmi_physical_id fallback with {} GPU entries",
                 gpuTopoByBusId.size());
  }
}

std::optional<int> XgmiBackend::LookupVisibleDevice(const std::string& busId) const {
  auto it = localDeviceByBusId.find(NormalizeBusId(busId));
  if (it == localDeviceByBusId.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool XgmiBackend::IsTopologyEligible(int localDeviceId, const std::string& remoteBusId) const {
  if (gpuTopoByBusId.empty()) {
    return false;
  }

  // Find local device's bus ID by reverse lookup
  std::string localBusId;
  for (const auto& [busId, devId] : localDeviceByBusId) {
    if (devId == localDeviceId) {
      localBusId = busId;
      break;
    }
  }
  if (localBusId.empty()) {
    return false;
  }

  std::string normalizedRemote = NormalizeBusId(remoteBusId);
  auto localIt = gpuTopoByBusId.find(localBusId);
  auto remoteIt = gpuTopoByBusId.find(normalizedRemote);
  if (localIt == gpuTopoByBusId.end() || remoteIt == gpuTopoByBusId.end()) {
    return false;
  }

  return localIt->second == remoteIt->second;
}

std::optional<int> XgmiBackend::ResolveVisibleDeviceId(const MemoryDesc& desc) const {
  if (desc.loc != MemoryLocationType::GPU) {
    return std::nullopt;
  }

  if (desc.engineKey == myEngKey) {
    return desc.deviceId;
  }

  if (desc.deviceBusId.empty()) {
    return std::nullopt;
  }

  auto it = localDeviceByBusId.find(NormalizeBusId(desc.deviceBusId));
  if (it == localDeviceByBusId.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool XgmiBackend::IsSameProcessEngine(const EngineKey& engineKey) const {
  if (engineKey == myEngKey) {
    return true;
  }

  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  auto it = remoteEngines.find(engineKey);
  return it != remoteEngines.end() && it->second.pid == myPid;
}

bool XgmiBackend::IsSameNodeEngine(const EngineKey& engineKey) const {
  if (engineKey == myEngKey) {
    return true;
  }

  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  auto it = remoteEngines.find(engineKey);
  if (it == remoteEngines.end()) {
    return false;
  }
  const EngineDesc& remoteEngine = it->second;
  if (!myNodeId.empty() && !remoteEngine.nodeId.empty()) {
    return remoteEngine.nodeId == myNodeId;
  }
  return remoteEngine.hostname == myHostname;
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

  IpcCacheKey cacheKey{desc.engineKey, desc.id, localDeviceId};
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
    auto visibleRemoteDevice = ResolveVisibleDeviceId(desc);
    if (visibleRemoteDevice.has_value() && IsSameProcessEngine(desc.engineKey) &&
        IsP2PAccessible(localDeviceId, visibleRemoteDevice.value())) {
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
  void* localAddr = GetRemappedAddress(local, localDevice);

  auto visibleRemote = LookupVisibleDevice(remote.deviceBusId);
  if (visibleRemote.has_value()) {
    // Visible path: remote GPU is in HIP_VISIBLE_DEVICES
    int remoteDevice = visibleRemote.value();
    void* remoteAddr = GetRemappedAddress(remote, localDevice);
    if (localAddr == nullptr || remoteAddr == nullptr) {
      MORI_IO_WARN("XGMI: Failed to remap memory (local id={}, remote id={})", local.id, remote.id);
      return nullptr;
    }
    bool ipcSession = (remote.engineKey != myEngKey);

    if (!IsP2PAccessible(localDevice, remoteDevice)) {
      MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                   remoteDevice);
    }

    return new XgmiBackendSession(config, localAddr, remoteAddr, localDevice, remoteDevice,
                                  ipcSession, this, streamPool.get(), eventPool.get());
  }

  // Hidden-device path: remote GPU is not visible, use IPC on localDevice
  void* remoteAddr = GetRemappedAddress(remote, localDevice);
  if (localAddr == nullptr || remoteAddr == nullptr) {
    MORI_IO_WARN("XGMI: Failed to remap hidden-device memory (local id={}, remote id={})", local.id,
                 remote.id);
    return nullptr;
  }

  MORI_IO_INFO("XGMI: Created hidden-device IPC session (local id={}, remote id={}, device={})",
               local.id, remote.id, localDevice);
  return new XgmiBackendSession(config, localAddr, remoteAddr, localDevice, localDevice, true, this,
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
  int localDevice = local.deviceId;

  int remoteDevice;
  bool ipcSession;
  auto visibleRemote = LookupVisibleDevice(remote.deviceBusId);
  if (visibleRemote.has_value()) {
    remoteDevice = visibleRemote.value();
    ipcSession = (remote.engineKey != myEngKey);
    if (!IsP2PAccessible(localDevice, remoteDevice)) {
      MORI_IO_WARN("XGMI: P2P access not available between devices {} and {}", localDevice,
                   remoteDevice);
    }
  } else {
    // Hidden-device path
    remoteDevice = localDevice;
    ipcSession = true;
  }

  void* remoteAddr = GetRemappedAddress(remote, localDevice);
  if (remoteAddr == nullptr) {
    MORI_IO_WARN(
        "XGMI: Failed to remap remote memory for session creation (local.id={}, remote.id={}, "
        "localDevice={}, remoteDevice={})",
        local.id, remote.id, localDevice, remoteDevice);
    return nullptr;
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

  if (!IsSameNodeEngine(remote.engineKey)) {
    return false;
  }

  if (remote.deviceBusId.empty()) {
    return false;
  }

  // Visible fast path: remote GPU is in this process's HIP_VISIBLE_DEVICES
  auto visibleRemote = LookupVisibleDevice(remote.deviceBusId);
  if (visibleRemote.has_value()) {
    return IsP2PAccessible(local.deviceId, visibleRemote.value());
  }

  // Hidden-device path: remote GPU is not visible but may be on the same XGMI hive
  if (IsIpcHandleEmpty(remote.ipcHandle)) {
    return false;
  }
  return IsTopologyEligible(local.deviceId, remote.deviceBusId);
}

}  // namespace io
}  // namespace mori
