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
#include "src/io/fabric/backend_impl.hpp"

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>

#include "mori/io/logging.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/host_utils.hpp"

namespace mori {
namespace io {
namespace {

// Lowercase a PCI bus ID so HIP (may return uppercase) and sysfs (lowercase)
// spellings compare equal.
std::string NormalizeBusId(const std::string& busId) {
  std::string result = busId;
  for (auto& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

bool IsFabricHandleEmpty(const std::array<char, kFabricHandleSize>& handle) {
  return std::all_of(handle.begin(), handle.end(), [](char c) { return c == 0; });
}

// Keep caller-visible HIP current device unchanged across MORI internals.
class ScopedHipDeviceGuard {
 public:
  ScopedHipDeviceGuard() {
    if (hipGetDevice(&originalDevice_) != hipSuccess) {
      valid_ = false;
    }
  }
  ~ScopedHipDeviceGuard() {
    if (valid_) (void)hipSetDevice(originalDevice_);
  }

 private:
  int originalDevice_{0};
  bool valid_{true};
};

// Round a user size up to the device's fabric-allocation granularity, so an
// imported handle can be reserved/mapped as a whole allocation.
size_t AlignToFabricGranularity(int device, size_t size) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = hipMemHandleTypeFabricCompat;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t gran = 0;
  hipError_t err =
      hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
  if (err != hipSuccess || gran == 0) {
    (void)hipGetLastError();
    gran = 2ull * 1024 * 1024;  // conservative 2 MiB fallback
  }
  return ((size + gran - 1) / gran) * gran;
}

// Deferred completion for an event-backed transfer. Mirrors the XGMI backend:
// the status polls/waits on the HIP event and finalizes exactly once.
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
        status->Update(StatusCode::ERR_GPU_OP, std::string("FABRIC: hipEventSynchronize failed: ") +
                                                   hipGetErrorString(err));
      }
      if (cleanup) cleanup();
      eventPool->PutEvent(event, eventDevice);
      completed.store(true, std::memory_order_release);
    });
  }

  void FinalizeNonBlocking() {
    std::lock_guard<std::mutex> lock(eventMu);
    if (completed.load(std::memory_order_acquire)) return;

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
                       std::string("FABRIC: hipEventQuery failed: ") + hipGetErrorString(err));
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

// Registry backing FabricMalloc / FabricFree.
struct FabricAllocation {
  hipMemGenericAllocationHandle_t handle{};
  size_t size{0};
};
std::mutex g_fabricAllocMu;
std::unordered_map<void*, FabricAllocation> g_fabricAllocs;

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                  Fabric allocation helpers                                     */
/* ---------------------------------------------------------------------------------------------- */
void* FabricMalloc(size_t size, int device) {
  if (size == 0) return nullptr;

  ScopedHipDeviceGuard guard;
  if (hipSetDevice(device) != hipSuccess) {
    MORI_IO_WARN("FABRIC: FabricMalloc failed to set device {}", device);
    return nullptr;
  }

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = hipMemHandleTypeFabricCompat;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended) !=
          hipSuccess ||
      gran == 0) {
    MORI_IO_WARN("FABRIC: FabricMalloc granularity query failed on device {}", device);
    return nullptr;
  }
  size_t aligned = ((size + gran - 1) / gran) * gran;

  hipMemGenericAllocationHandle_t handle{};
  hipError_t err = hipMemCreate(&handle, aligned, &prop, 0);
  if (err != hipSuccess) {
    MORI_IO_WARN("FABRIC: FabricMalloc hipMemCreate failed: {}", hipGetErrorString(err));
    return nullptr;
  }

  void* ptr = nullptr;
  err = hipMemAddressReserve(&ptr, aligned, 0, 0, 0);
  if (err != hipSuccess) {
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: FabricMalloc hipMemAddressReserve failed: {}", hipGetErrorString(err));
    return nullptr;
  }
  err = hipMemMap(ptr, aligned, 0, handle, 0);
  if (err != hipSuccess) {
    (void)hipMemAddressFree(ptr, aligned);
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: FabricMalloc hipMemMap failed: {}", hipGetErrorString(err));
    return nullptr;
  }
  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  err = hipMemSetAccess(ptr, aligned, &access, 1);
  if (err != hipSuccess) {
    (void)hipMemUnmap(ptr, aligned);
    (void)hipMemAddressFree(ptr, aligned);
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: FabricMalloc hipMemSetAccess failed: {}", hipGetErrorString(err));
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(g_fabricAllocMu);
    g_fabricAllocs[ptr] = {handle, aligned};
  }
  MORI_IO_TRACE("FABRIC: FabricMalloc device={} size={} aligned={} ptr={}", device, size, aligned,
                ptr);
  return ptr;
}

void FabricFree(void* ptr) {
  if (ptr == nullptr) return;
  FabricAllocation alloc;
  {
    std::lock_guard<std::mutex> lock(g_fabricAllocMu);
    auto it = g_fabricAllocs.find(ptr);
    if (it == g_fabricAllocs.end()) {
      MORI_IO_WARN("FABRIC: FabricFree unknown pointer {}", ptr);
      return;
    }
    alloc = it->second;
    g_fabricAllocs.erase(it);
  }
  (void)hipMemUnmap(ptr, alloc.size);
  (void)hipMemRelease(alloc.handle);
  (void)hipMemAddressFree(ptr, alloc.size);
  MORI_IO_TRACE("FABRIC: FabricFree ptr={} size={}", ptr, alloc.size);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     FabricBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
FabricBackendSession::FabricBackendSession(const FabricBackendConfig& config, void* localAddr,
                                           void* remoteAddr, int localDevice,
                                           FabricBackend* backend, StreamPool* streamPool,
                                           EventPool* eventPool)
    : config(config),
      localAddr(localAddr),
      remoteAddr(remoteAddr),
      localDevice(localDevice),
      backend(backend),
      streamPool(streamPool),
      eventPool(eventPool) {}

hipError_t FabricBackendSession::LaunchCopy(void* dst, const void* src, size_t size,
                                            hipStream_t stream) {
  if (size == 0) return hipSuccess;

  hipFunction_t fn = backend != nullptr ? backend->GetFabricCopyFunc(localDevice) : nullptr;
  if (fn == nullptr) {
    // Kernel unavailable: fall back to hipMemcpyAsync. Correct for small
    // payloads; large imported-fabric-pointer copies should always have the
    // kernel loaded (see fabric_copy.hip note).
    return hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream);
  }

  char* dstPtr = static_cast<char*>(dst);
  const char* srcPtr = static_cast<const char*>(src);
  size_t nbytes = size;
  void* args[] = {&dstPtr, &srcPtr, &nbytes};

  const int threads = 256;
  size_t units = (size + 15) / 16;  // 16B chunks (grid-stride handles remainder)
  long long blocks = static_cast<long long>((units + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;

  return hipModuleLaunchKernel(fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1, 0, stream,
                               args, nullptr);
}

void FabricBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                     TransferStatus* status, TransferUniqueId id, bool isRead) {
  ScopedHipDeviceGuard deviceGuard;
  hipError_t err = hipSetDevice(localDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("FABRIC: Failed to set device: ") + hipGetErrorString(err));
    return;
  }

  hipStream_t stream = streamPool->GetNextStream(localDevice);
  hipEvent_t event = eventPool->GetEvent(localDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to get stream or event from pool");
    if (event != nullptr) eventPool->PutEvent(event, localDevice);
    return;
  }

  void* src = isRead ? static_cast<char*>(remoteAddr) + remoteOffset
                     : static_cast<char*>(localAddr) + localOffset;
  void* dst = isRead ? static_cast<char*>(localAddr) + localOffset
                     : static_cast<char*>(remoteAddr) + remoteOffset;

  err = LaunchCopy(dst, src, size, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("FABRIC: copy launch failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, localDevice);
    return;
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("FABRIC: hipEventRecord failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, localDevice);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  ArmTransferCompletion(status, event, localDevice, eventPool);
  MORI_IO_TRACE("FABRIC: Transfer issued, id={}, size={}, isRead={}", id, size, isRead);
}

void FabricBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
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

  hipError_t err = hipSetDevice(localDevice);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("FABRIC: Failed to set device: ") + hipGetErrorString(err));
    return;
  }

  hipStream_t stream = streamPool->GetNextStream(localDevice);
  hipEvent_t event = eventPool->GetEvent(localDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to get stream or event from pool");
    if (event != nullptr) eventPool->PutEvent(event, localDevice);
    return;
  }

  // Sort by remote offset then merge contiguous runs (both sides contiguous) to
  // cut the number of kernel launches.
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
    eventPool->PutEvent(event, localDevice);
    return;
  }

  for (const auto& seg : segments) {
    void* src = isRead ? static_cast<char*>(remoteAddr) + seg.remoteOff
                       : static_cast<char*>(localAddr) + seg.localOff;
    void* dst = isRead ? static_cast<char*>(localAddr) + seg.localOff
                       : static_cast<char*>(remoteAddr) + seg.remoteOff;
    err = LaunchCopy(dst, src, seg.sz, stream);
    if (err != hipSuccess) {
      status->Update(StatusCode::ERR_GPU_OP,
                     std::string("FABRIC: batch copy launch failed: ") + hipGetErrorString(err));
      eventPool->PutEvent(event, localDevice);
      return;
    }
  }

  err = hipEventRecord(event, stream);
  if (err != hipSuccess) {
    status->Update(StatusCode::ERR_GPU_OP,
                   std::string("FABRIC: hipEventRecord failed: ") + hipGetErrorString(err));
    eventPool->PutEvent(event, localDevice);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  ArmTransferCompletion(status, event, localDevice, eventPool);
  MORI_IO_TRACE("FABRIC: Batch transfer issued, id={}, segments={}, isRead={}", id, segments.size(),
                isRead);
}

bool FabricBackendSession::Alive() const { return true; }

/* ---------------------------------------------------------------------------------------------- */
/*                                         FabricBackend                                          */
/* ---------------------------------------------------------------------------------------------- */
FabricBackend::FabricBackend(EngineKey k, const IOEngineConfig& /*engConfig*/,
                             const FabricBackendConfig& beConfig)
    : myEngKey(k), config(beConfig), myPid(static_cast<int>(getpid())) {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  myHostname = std::string(hostname);
  myNodeId = mori::ResolveNodeId(myHostname);

  streamPool = std::make_unique<StreamPool>(config.numStreams);
  eventPool = std::make_unique<EventPool>(config.numEvents);

  Initialize();

  std::stringstream ss;
  ss << config;
  MORI_IO_INFO("FabricBackend created with config: {} node_id: {} hostname: {} numDevices: {}",
               ss.str().c_str(), myNodeId.c_str(), myHostname.c_str(), numDevices);
}

FabricBackend::~FabricBackend() {
  {
    std::unique_lock<std::shared_mutex> lock(importMutex);
    for (auto& entry : importedRegions) {
      ImportedRegion& region = entry.second;
      if (region.mappedAddr != nullptr) {
        (void)hipMemUnmap(region.mappedAddr, region.size);
        (void)hipMemRelease(region.handle);
        (void)hipMemAddressFree(region.mappedAddr, region.size);
      }
    }
    importedRegions.clear();
  }

  std::lock_guard<std::mutex> lock(moduleMu_);
  for (auto& mod : fabricCopyModules_) {
    if (mod != nullptr) (void)hipModuleUnload(mod);
  }
  fabricCopyModules_.clear();
  fabricCopyFuncs_.clear();
}

void FabricBackend::Initialize() {
  hipError_t err = hipGetDeviceCount(&numDevices);
  if (err != hipSuccess || numDevices <= 0) {
    MORI_IO_WARN("FABRIC: Failed to get device count or no devices found");
    numDevices = 0;
    return;
  }

  deviceSupportsFabric.assign(numDevices, 0);
  deviceVpodKeys.assign(numDevices, fabric::VpodKey{});

  for (int i = 0; i < numDevices; ++i) {
    char busId[32] = {0};
    if (hipDeviceGetPCIBusId(busId, sizeof(busId), i) == hipSuccess) {
      localDeviceByBusId[NormalizeBusId(std::string(busId))] = i;
    }
    deviceSupportsFabric[i] = fabric::DeviceSupportsFabric(i) ? 1 : 0;
    deviceVpodKeys[i] = fabric::ReadVpodKey(i);
    if (deviceSupportsFabric[i]) {
      MORI_IO_INFO("FABRIC: device {} supports fabric (vpod valid={} ppod_id={} vpod_id={})", i,
                   deviceVpodKeys[i].valid, deviceVpodKeys[i].ppodId, deviceVpodKeys[i].vpodId);
    }
  }
}

std::optional<int> FabricBackend::LookupVisibleDevice(const std::string& busId) const {
  auto it = localDeviceByBusId.find(NormalizeBusId(busId));
  if (it == localDeviceByBusId.end()) return std::nullopt;
  return it->second;
}

fabric::VpodKey FabricBackend::LocalVpodKey(int deviceId) const {
  if (deviceId < 0 || deviceId >= numDevices) return fabric::VpodKey{};
  return deviceVpodKeys[deviceId];
}

void FabricBackend::LoadFabricCopyModule(const std::string& hsacoPath) {
  std::lock_guard<std::mutex> lock(moduleMu_);
  fabricCopyHsacoPath_ = hsacoPath;
  fabricCopyModules_.assign(numDevices, nullptr);
  fabricCopyFuncs_.assign(numDevices, nullptr);
  MORI_IO_INFO("FABRIC: copy kernel registered from {}", hsacoPath);
}

hipFunction_t FabricBackend::GetFabricCopyFunc(int deviceId) {
  std::lock_guard<std::mutex> lock(moduleMu_);
  if (fabricCopyHsacoPath_.empty() || deviceId < 0 || deviceId >= numDevices) {
    return nullptr;
  }
  if (fabricCopyFuncs_[deviceId] != nullptr) {
    return fabricCopyFuncs_[deviceId];
  }

  ScopedHipDeviceGuard deviceGuard;
  if (hipSetDevice(deviceId) != hipSuccess) return nullptr;

  hipError_t err = hipModuleLoad(&fabricCopyModules_[deviceId], fabricCopyHsacoPath_.c_str());
  if (err != hipSuccess) {
    MORI_IO_WARN("FABRIC: Failed to load copy module on device {}: {}", deviceId,
                 hipGetErrorString(err));
    return nullptr;
  }
  err = hipModuleGetFunction(&fabricCopyFuncs_[deviceId], fabricCopyModules_[deviceId],
                             "fabricCopyKernel");
  if (err != hipSuccess) {
    MORI_IO_WARN("FABRIC: Failed to get fabricCopyKernel on device {}: {}", deviceId,
                 hipGetErrorString(err));
    (void)hipModuleUnload(fabricCopyModules_[deviceId]);
    fabricCopyModules_[deviceId] = nullptr;
    return nullptr;
  }
  MORI_IO_INFO("FABRIC: Loaded copy kernel on device {}", deviceId);
  return fabricCopyFuncs_[deviceId];
}

void FabricBackend::RegisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  remoteEngines[desc.key] = desc;
  MORI_IO_TRACE("FABRIC: Registered remote engine {} hostname {}", desc.key, desc.hostname);
}

void FabricBackend::DeregisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  remoteEngines.erase(desc.key);
  MORI_IO_TRACE("FABRIC: Deregistered remote engine {}", desc.key);
}

void FabricBackend::RegisterMemory(MemoryDesc& desc) {
  if (desc.loc != MemoryLocationType::GPU) {
    MORI_IO_TRACE("FABRIC: Skipping non-GPU memory registration for id={}", desc.id);
    return;
  }

  int dev = desc.deviceId;
  // Advertise the owning device's vPOD identity so the peer's CanHandle can
  // decide whether this memory is reachable over the same scale-up fabric.
  if (dev >= 0 && dev < numDevices) {
    const fabric::VpodKey& key = deviceVpodKeys[dev];
    if (key.valid) {
      desc.vpodId = key.vpodId;
      desc.vpodPpodId = key.ppodId;
    }
    if (!deviceSupportsFabric[dev]) {
      MORI_IO_TRACE("FABRIC: device {} does not support fabric; memory id={} not exportable", dev,
                    desc.id);
      return;
    }
  }

  // Export a fabric shareable handle by retaining the allocation backing the
  // registered pointer. This only succeeds for VMM allocations created with the
  // fabric handle type (e.g. via FabricMalloc); plain hipMalloc memory fails
  // here, leaving fabricHandle empty so CanHandle falls back to RDMA.
  ScopedHipDeviceGuard deviceGuard;
  if (dev >= 0) (void)hipSetDevice(dev);

  hipMemGenericAllocationHandle_t handle{};
  hipError_t err = hipMemRetainAllocationHandle(&handle, reinterpret_cast<void*>(desc.data));
  if (err != hipSuccess) {
    (void)hipGetLastError();
    MORI_IO_TRACE("FABRIC: memory id={} is not a VMM allocation ({}), fabric export skipped",
                  desc.id, hipGetErrorString(err));
    return;
  }

  hipMemFabricHandle_compat_t fh;
  err = hipMemExportToShareableHandle(&fh, handle, hipMemHandleTypeFabricCompat, 0);
  (void)hipMemRelease(handle);  // drop the extra ref taken by Retain
  if (err != hipSuccess) {
    (void)hipGetLastError();
    MORI_IO_TRACE("FABRIC: fabric export failed for memory id={}: {}", desc.id,
                  hipGetErrorString(err));
    return;
  }

  static_assert(sizeof(fh) == kFabricHandleSize, "fabric handle size mismatch");
  std::memcpy(desc.fabricHandle.data(), &fh, sizeof(fh));

  // The registered pointer may be a sub-allocation inside a larger fabric VMM
  // allocation (e.g. a torch tensor carved out of a MemPool segment). Record its
  // offset + the full allocation size so the peer maps the whole allocation and
  // offsets to the right address.
  void* base = nullptr;
  size_t rangeSize = 0;
  hipError_t rangeErr =
      hipMemGetAddressRange(&base, &rangeSize, reinterpret_cast<hipDeviceptr_t>(desc.data));
  if (rangeErr == hipSuccess && base != nullptr && rangeSize > 0) {
    desc.fabricOffset = desc.data - reinterpret_cast<uintptr_t>(base);
    desc.fabricAllocSize = rangeSize;
  } else {
    (void)hipGetLastError();
    desc.fabricOffset = 0;
    desc.fabricAllocSize = desc.size;
  }

  MORI_IO_TRACE(
      "FABRIC: Registered memory id={}, addr={}, size={}, fabricOffset={}, allocSize={} "
      "(fabric-exportable)",
      desc.id, desc.data, desc.size, desc.fabricOffset, desc.fabricAllocSize);
}

void FabricBackend::DeregisterMemory(const MemoryDesc& desc) {
  {
    std::unique_lock<std::shared_mutex> lock(importMutex);
    for (auto it = importedRegions.begin(); it != importedRegions.end();) {
      if (it->first.memId == desc.id && it->first.engineKey == desc.engineKey) {
        if (it->second.mappedAddr != nullptr) {
          (void)hipMemUnmap(it->second.mappedAddr, it->second.size);
          (void)hipMemRelease(it->second.handle);
          (void)hipMemAddressFree(it->second.mappedAddr, it->second.size);
        }
        it = importedRegions.erase(it);
      } else {
        ++it;
      }
    }
  }
  InvalidateSessionsForMemory(desc.id);
  MORI_IO_TRACE("FABRIC: Deregistered memory id={}", desc.id);
}

void* FabricBackend::GetImportedAddress(const MemoryDesc& desc, int localDeviceId) {
  if (desc.engineKey == myEngKey) {
    return reinterpret_cast<void*>(desc.data);
  }
  if (IsFabricHandleEmpty(desc.fabricHandle)) {
    return nullptr;
  }

  ImportCacheKey cacheKey{desc.engineKey, desc.id, localDeviceId};
  {
    std::shared_lock<std::shared_mutex> rlock(importMutex);
    auto it = importedRegions.find(cacheKey);
    if (it != importedRegions.end() && it->second.mappedAddr != nullptr) {
      return static_cast<char*>(it->second.mappedAddr) + desc.fabricOffset;
    }
  }

  hipMemFabricHandle_compat_t fh;
  static_assert(sizeof(fh) == kFabricHandleSize, "fabric handle size mismatch");
  std::memcpy(&fh, desc.fabricHandle.data(), sizeof(fh));

  ScopedHipDeviceGuard deviceGuard;
  hipError_t err = hipSetDevice(localDeviceId);
  if (err != hipSuccess) {
    MORI_IO_WARN("FABRIC: Failed to set device {} for fabric import: {}", localDeviceId,
                 hipGetErrorString(err));
    return nullptr;
  }

  hipMemGenericAllocationHandle_t handle{};
  err = hipMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(&fh),
                                        hipMemHandleTypeFabricCompat);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    MORI_IO_WARN("FABRIC: hipMemImportFromShareableHandle failed for id={}: {}", desc.id,
                 hipGetErrorString(err));
    return nullptr;
  }

  // Map the whole exported allocation, then offset to `data` within it.
  size_t allocBytes = desc.fabricAllocSize != 0 ? desc.fabricAllocSize : desc.size;
  size_t mapSize = AlignToFabricGranularity(localDeviceId, allocBytes);
  void* va = nullptr;
  err = hipMemAddressReserve(&va, mapSize, 0, 0, 0);
  if (err != hipSuccess) {
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: hipMemAddressReserve failed for id={}: {}", desc.id,
                 hipGetErrorString(err));
    return nullptr;
  }
  err = hipMemMap(va, mapSize, 0, handle, 0);
  if (err != hipSuccess) {
    (void)hipMemAddressFree(va, mapSize);
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: hipMemMap failed for id={}: {}", desc.id, hipGetErrorString(err));
    return nullptr;
  }
  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = localDeviceId;
  access.flags = hipMemAccessFlagsProtReadWrite;
  err = hipMemSetAccess(va, mapSize, &access, 1);
  if (err != hipSuccess) {
    (void)hipMemUnmap(va, mapSize);
    (void)hipMemAddressFree(va, mapSize);
    (void)hipMemRelease(handle);
    MORI_IO_WARN("FABRIC: hipMemSetAccess failed for id={}: {}", desc.id, hipGetErrorString(err));
    return nullptr;
  }

  std::unique_lock<std::shared_mutex> wlock(importMutex);
  // Another thread may have imported concurrently; keep the first winner.
  auto existing = importedRegions.find(cacheKey);
  if (existing != importedRegions.end() && existing->second.mappedAddr != nullptr) {
    void* winner = existing->second.mappedAddr;
    wlock.unlock();
    (void)hipMemUnmap(va, mapSize);
    (void)hipMemAddressFree(va, mapSize);
    (void)hipMemRelease(handle);
    return static_cast<char*>(winner) + desc.fabricOffset;
  }
  importedRegions[cacheKey] = {handle, va, mapSize};
  MORI_IO_TRACE("FABRIC: Imported fabric handle for id={} on device {}, mapped={} offset={}",
                desc.id, localDeviceId, va, desc.fabricOffset);
  return static_cast<char*>(va) + desc.fabricOffset;
}

void FabricBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                              const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                              TransferStatus* status, TransferUniqueId id, bool isRead) {
  FabricBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  if (sess == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to create session");
    return;
  }
  sess->ReadWrite(localOffset, remoteOffset, size, status, id, isRead);
}

void FabricBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                   const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                   const SizeVec& sizes, TransferStatus* status,
                                   TransferUniqueId id, bool isRead) {
  FabricBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  if (sess == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to create session");
    return;
  }
  sess->BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* FabricBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  int localDevice = local.deviceId;
  void* localAddr = reinterpret_cast<void*>(local.data);
  void* remoteAddr = GetImportedAddress(remote, localDevice);
  if (localAddr == nullptr || remoteAddr == nullptr) {
    MORI_IO_WARN("FABRIC: Failed to map memory for session (local id={}, remote id={})", local.id,
                 remote.id);
    return nullptr;
  }
  return new FabricBackendSession(config, localAddr, remoteAddr, localDevice, this,
                                  streamPool.get(), eventPool.get());
}

FabricBackendSession* FabricBackend::GetOrCreateSessionCached(const MemoryDesc& local,
                                                              const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};

  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) {
    return it->second.get();
  }

  int localDevice = local.deviceId;
  void* localAddr = reinterpret_cast<void*>(local.data);
  void* remoteAddr = GetImportedAddress(remote, localDevice);
  if (remoteAddr == nullptr) {
    MORI_IO_WARN("FABRIC: Failed to import remote memory for session (local.id={}, remote.id={})",
                 local.id, remote.id);
    return nullptr;
  }

  auto sess = std::make_unique<FabricBackendSession>(config, localAddr, remoteAddr, localDevice,
                                                     this, streamPool.get(), eventPool.get());
  FabricBackendSession* rawPtr = sess.get();
  sessionCache[key] = std::move(sess);
  MORI_IO_TRACE("FABRIC: Created session for local.id={}, remote.id={}", local.id, remote.id);
  return rawPtr;
}

void FabricBackend::InvalidateSessionsForMemory(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  for (auto it = sessionCache.begin(); it != sessionCache.end();) {
    if (it->first.localMemId == id || it->first.remoteMemId == id) {
      it = sessionCache.erase(it);
    } else {
      ++it;
    }
  }
}

bool FabricBackend::PopInboundTransferStatus(EngineKey /*remote*/, TransferUniqueId /*id*/,
                                             TransferStatus* /*status*/) {
  return false;
}

bool FabricBackend::CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const {
  if (local.loc != MemoryLocationType::GPU || remote.loc != MemoryLocationType::GPU) {
    return false;
  }
  if (local.deviceId < 0 || local.deviceId >= numDevices) {
    return false;
  }
  if (!deviceSupportsFabric[local.deviceId]) {
    return false;
  }
  // Remote memory must be fabric-exportable (non-empty handle).
  if (IsFabricHandleEmpty(remote.fabricHandle)) {
    return false;
  }

  // Both endpoints must sit in the same scale-up fabric domain (vPOD).
  fabric::VpodKey localKey = LocalVpodKey(local.deviceId);
  fabric::VpodKey remoteKey;
  remoteKey.vpodId = remote.vpodId;
  remoteKey.ppodId = remote.vpodPpodId;
  remoteKey.valid = (remote.vpodId >= 0) || !remote.vpodPpodId.empty();
  return fabric::VpodKeySame(localKey, remoteKey);
}

}  // namespace io
}  // namespace mori
