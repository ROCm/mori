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
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mori/io/logging.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/host_utils.hpp"
#include "src/io/fabric/completion_guard.hpp"

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
  if (size > std::numeric_limits<size_t>::max() - (gran - 1)) return 0;
  return ((size + gran - 1) / gran) * gran;
}

// Deferred completion for an event-backed transfer. Mirrors the XGMI backend:
// the status polls/waits on the HIP event and finalizes exactly once.
struct TransferCompletion {
  TransferStatus* status{nullptr};
  hipEvent_t event{nullptr};
  hipStream_t stream{nullptr};
  int eventDevice{-1};
  EventPool* eventPool{nullptr};
  std::unique_ptr<fabric::internal::QuarantineEntry> quarantineEntry;
  std::once_flag completionOnce;
  std::atomic<bool> completed{false};
  bool submitted{false};
  bool recorded{false};
  std::mutex eventMu;

  void ReturnEvent() noexcept {
    if (event == nullptr) return;
    try {
      eventPool->PutEvent(event, eventDevice);
    } catch (...) {
      // Completion has already been proved. Pool growth must not strand a
      // terminal status when host allocation fails.
      (void)hipEventDestroy(event);
    }
    event = nullptr;
  }

  ~TransferCompletion() {
    if (completed.load(std::memory_order_acquire)) return;
    if (submitted) {
      fabric::internal::QuarantineResources(std::move(quarantineEntry));
    } else {
      ReturnEvent();
    }
  }

  void Finish(hipError_t error, bool drained) {
    if (drained) {
      quarantineEntry.reset();
      ReturnEvent();
    } else {
      // This entry and its callback were allocated before enqueue. Retaining
      // imports/descriptors here does not allocate, including on bad_alloc.
      fabric::internal::QuarantineResources(std::move(quarantineEntry));
      status->SetResourcesQuarantined();
    }
    completed.store(true, std::memory_order_release);
    // Publish a terminal code before allocating a diagnostic message.
    status->SetCode(error == hipSuccess ? StatusCode::SUCCESS : StatusCode::ERR_GPU_OP);
  }

  void Finalize(hipError_t error, const char* operation) {
    Finish(error, error == hipSuccess);
    if (error != hipSuccess) {
      status->SetMessage(std::string("FABRIC: ") + operation + " failed: " +
                         hipGetErrorString(error) + "; DRAIN_UNCONFIRMED: resources quarantined");
    }
  }

  // The submission path holds eventMu, including the submitted/recorded flags.
  void FailSubmissionLocked(hipError_t error, const char* operation) {
    const bool drained = hipStreamSynchronize(stream) == hipSuccess;
    Finish(error, drained);
    status->SetMessage(std::string("FABRIC: ") + operation +
                       " failed: " + hipGetErrorString(error) +
                       (drained ? "" : "; DRAIN_UNCONFIRMED: resources quarantined"));
  }

  void FailSubmissionException() noexcept {
    std::lock_guard<std::mutex> lock(eventMu);
    if (completed.load(std::memory_order_acquire)) return;
    const bool drained = hipStreamSynchronize(stream) == hipSuccess;
    Finish(hipErrorUnknown, drained);
  }

  void FinalizeBlocking() {
    std::lock_guard<std::mutex> lock(eventMu);
    if (completed.load(std::memory_order_acquire)) return;
    std::call_once(completionOnce, [this]() {
      ScopedHipDeviceGuard guard;
      (void)hipSetDevice(eventDevice);
      Finalize(recorded ? hipEventSynchronize(event) : hipStreamSynchronize(stream),
               recorded ? "hipEventSynchronize" : "hipStreamSynchronize");
    });
  }

  void FinalizeNonBlocking() {
    std::lock_guard<std::mutex> lock(eventMu);
    if (completed.load(std::memory_order_acquire) || !recorded) return;
    ScopedHipDeviceGuard guard;
    (void)hipSetDevice(eventDevice);
    hipError_t error = hipEventQuery(event);
    if (error == hipErrorNotReady) {
      (void)hipGetLastError();
      return;
    }
    std::call_once(completionOnce, [this, error]() { Finalize(error, "hipEventQuery"); });
  }
};

std::shared_ptr<TransferCompletion> PrepareTransferCompletion(TransferStatus* status,
                                                              hipEvent_t event, hipStream_t stream,
                                                              int eventDevice, EventPool* eventPool,
                                                              std::shared_ptr<void> region) {
  std::shared_ptr<TransferCompletion> completion;
  try {
    completion = std::make_shared<TransferCompletion>();
  } catch (...) {
    try {
      eventPool->PutEvent(event, eventDevice);
    } catch (...) {
      (void)hipEventDestroy(event);
    }
    throw;
  }
  completion->status = status;
  completion->event = event;
  completion->stream = stream;
  completion->eventDevice = eventDevice;
  completion->eventPool = eventPool;
  completion->quarantineEntry = std::make_unique<fabric::internal::QuarantineEntry>();
  completion->quarantineEntry->region = std::move(region);
  // All callback allocations precede the first HIP enqueue and IN_PROGRESS.
  status->SetStreamWaitCallback([completion](void* stream) {
    std::lock_guard<std::mutex> lock(completion->eventMu);
    if (completion->completed.load(std::memory_order_acquire)) return completion->status->Code();
    if (!completion->recorded) return StatusCode::ERR_BAD_STATE;
    ScopedHipDeviceGuard guard;
    if (hipSetDevice(completion->eventDevice) != hipSuccess) return StatusCode::ERR_GPU_OP;
    // hipStreamWaitEvent captures the current record; later event reuse does
    // not change this dependency. The consumer stream must belong to this GPU.
    hipError_t error =
        hipStreamWaitEvent(reinterpret_cast<hipStream_t>(stream), completion->event, 0);
    return error == hipSuccess ? StatusCode::SUCCESS : StatusCode::ERR_GPU_OP;
  });
  status->SetWaitCallback([completion]() { completion->FinalizeBlocking(); });
  status->SetProgressCallback([completion]() { completion->FinalizeNonBlocking(); });
  return completion;
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
  if (size > std::numeric_limits<size_t>::max() - (gran - 1)) return nullptr;
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
                                           EventPool* eventPool,
                                           std::shared_ptr<void> regionLifetime,
                                           std::shared_ptr<std::atomic<bool>> validity)
    : config(config),
      localAddr(localAddr),
      remoteAddr(remoteAddr),
      localDevice(localDevice),
      backend(backend),
      streamPool(streamPool),
      eventPool(eventPool),
      regionLifetime(std::move(regionLifetime)),
      validity(std::move(validity)) {}

hipError_t FabricBackendSession::LaunchCopy(void* dst, const void* src, size_t size,
                                            hipStream_t stream) {
  if (size == 0) return hipSuccess;

  hipFunction_t fn = backend != nullptr ? backend->GetFabricCopyFunc(localDevice) : nullptr;
  if (fn == nullptr) {
    // hipMemcpyAsync livelocks on imported fabric pointers for large payloads
    // (see fabric_copy.hip); fail fast instead of hanging, only allow it small.
    constexpr size_t kFabricMemcpyFallbackMax = 4ull * 1024 * 1024;
    if (size >= kFabricMemcpyFallbackMax) {
      MORI_IO_WARN(
          "FABRIC: fabric_copy kernel not loaded; refusing hipMemcpyAsync "
          "fallback for {} byte transfer (>= {} B may livelock on imported "
          "fabric pointers). Load the fabric_copy module first.",
          size, kFabricMemcpyFallbackMax);
      return hipErrorNotSupported;
    }
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
  if (!Alive()) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: remote session invalidated");
    return;
  }
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

  auto completion =
      PrepareTransferCompletion(status, event, stream, localDevice, eventPool, regionLifetime);
  void* src = isRead ? static_cast<char*>(remoteAddr) + remoteOffset
                     : static_cast<char*>(localAddr) + localOffset;
  void* dst = isRead ? static_cast<char*>(localAddr) + localOffset
                     : static_cast<char*>(remoteAddr) + remoteOffset;
  try {
    std::lock_guard<std::mutex> submissionLock(completion->eventMu);
    completion->submitted = true;
    err = LaunchCopy(dst, src, size, stream);
    if (err != hipSuccess) {
      completion->FailSubmissionLocked(err, "copy launch");
      return;
    }
    err = hipEventRecord(event, stream);
    if (err != hipSuccess) {
      completion->FailSubmissionLocked(err, "hipEventRecord");
      return;
    }
    completion->recorded = true;
    status->SetCode(StatusCode::IN_PROGRESS);
    MORI_IO_TRACE("FABRIC: Transfer issued, id={}, size={}, isRead={}", id, size, isRead);
  } catch (...) {
    completion->FailSubmissionException();
    throw;
  }
}

void FabricBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                          const SizeVec& sizes, TransferStatus* status,
                                          TransferUniqueId id, bool isRead) {
  if (!Alive()) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: remote session invalidated");
    return;
  }
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
      bool localContig = last.sz <= std::numeric_limits<size_t>::max() - last.localOff &&
                         (last.localOff + last.sz) == localOffsets[idx];
      bool remoteContig = last.sz <= std::numeric_limits<size_t>::max() - last.remoteOff &&
                          (last.remoteOff + last.sz) == remoteOffsets[idx];
      if (localContig && remoteContig) {
        last.sz += sizes[idx];
        continue;
      }
    }
    segments.push_back({localOffsets[idx], remoteOffsets[idx], sizes[idx]});
  }

  if (segments.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  const bool useBatch = segments.size() > 1 && backend != nullptr &&
                        backend->GetFabricBatchCopyFunc(localDevice) != nullptr;
  std::vector<FabricCopySegment> descriptors;
  if (useBatch) {
    descriptors.reserve(segments.size());
    for (const auto& seg : segments) {
      const char* src = isRead ? static_cast<char*>(remoteAddr) + seg.remoteOff
                               : static_cast<char*>(localAddr) + seg.localOff;
      char* dst = isRead ? static_cast<char*>(localAddr) + seg.localOff
                         : static_cast<char*>(remoteAddr) + seg.remoteOff;
      descriptors.push_back({src, dst, seg.sz});
    }
  }

  hipStream_t stream = streamPool->GetNextStream(localDevice);
  hipEvent_t event = eventPool->GetEvent(localDevice);
  if (stream == nullptr || event == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to get stream or event from pool");
    if (event != nullptr) eventPool->PutEvent(event, localDevice);
    return;
  }
  auto completion =
      PrepareTransferCompletion(status, event, stream, localDevice, eventPool, regionLifetime);
  try {
    std::lock_guard<std::mutex> submissionLock(completion->eventMu);
    completion->submitted = true;
    if (useBatch) {
      err = backend->LaunchBatchCopy(descriptors, localDevice, stream,
                                     &completion->quarantineEntry->staging);
      if (err != hipSuccess) {
        completion->FailSubmissionLocked(err, "batch copy launch");
        return;
      }
    } else {
      // Old modules without the optional batch entrypoint remain supported.
      for (const auto& seg : segments) {
        void* src = isRead ? static_cast<char*>(remoteAddr) + seg.remoteOff
                           : static_cast<char*>(localAddr) + seg.localOff;
        void* dst = isRead ? static_cast<char*>(localAddr) + seg.localOff
                           : static_cast<char*>(remoteAddr) + seg.remoteOff;
        err = LaunchCopy(dst, src, seg.sz, stream);
        if (err != hipSuccess) {
          completion->FailSubmissionLocked(err, "batch copy launch");
          return;
        }
      }
    }
    err = hipEventRecord(event, stream);
    if (err != hipSuccess) {
      completion->FailSubmissionLocked(err, "hipEventRecord");
      return;
    }
    completion->recorded = true;
    status->SetCode(StatusCode::IN_PROGRESS);
    MORI_IO_TRACE("FABRIC: Batch transfer issued, id={}, segments={}, isRead={}", id,
                  segments.size(), isRead);
  } catch (...) {
    completion->FailSubmissionException();
    throw;
  }
}

bool FabricBackendSession::Alive() const {
  return !validity || validity->load(std::memory_order_acquire);
}

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

FabricBackend::ImportedRegion::~ImportedRegion() {
  if (!ownsMapping || mappedAddr == nullptr) return;
  ScopedHipDeviceGuard guard;
  if (device >= 0) (void)hipSetDevice(device);
  (void)hipMemUnmap(mappedAddr, size);
  (void)hipMemRelease(handle);
  (void)hipMemAddressFree(mappedAddr, size);
}

FabricBackend::~FabricBackend() {
  // The engine must outlive submitted statuses. Session/mapping references in
  // each completion also prevent invalidation from unmapping in-flight reads.
  sessionCache.clear();
  importedRegions.clear();
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
  if (!fabricCopyHsacoPath_.empty() && fabricCopyHsacoPath_ != hsacoPath) {
    throw std::logic_error("Cannot replace a live FABRIC copy module");
  }
  if (fabricCopyHsacoPath_ == hsacoPath && !fabricCopyModules_.empty()) return;
  fabricCopyHsacoPath_ = hsacoPath;
  fabricCopyModules_.assign(numDevices, nullptr);
  fabricCopyFuncs_.assign(numDevices, nullptr);
  fabricBatchCopyFuncs_.assign(numDevices, nullptr);
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
  // Optional symbol: preserve compatibility with deployed contiguous-only modules.
  if (hipModuleGetFunction(&fabricBatchCopyFuncs_[deviceId], fabricCopyModules_[deviceId],
                           "fabricBatchCopyKernel") != hipSuccess) {
    fabricBatchCopyFuncs_[deviceId] = nullptr;
    (void)hipGetLastError();
  }
  MORI_IO_INFO("FABRIC: Loaded copy kernel on device {}", deviceId);
  return fabricCopyFuncs_[deviceId];
}

hipFunction_t FabricBackend::GetFabricBatchCopyFunc(int deviceId) {
  if (GetFabricCopyFunc(deviceId) == nullptr) return nullptr;
  std::lock_guard<std::mutex> lock(moduleMu_);
  return fabricBatchCopyFuncs_[deviceId];
}

bool FabricBackend::IsReady(int deviceId) {
  return deviceId >= 0 && deviceId < numDevices && deviceSupportsFabric[deviceId] &&
         GetFabricCopyFunc(deviceId) != nullptr;
}

FabricBackend::BatchStaging::~BatchStaging() {
  ScopedHipDeviceGuard guard;
  if (device >= 0) (void)hipSetDevice(device);
  if (devicePtr) (void)hipFree(devicePtr);
  if (host) (void)hipHostFree(host);
}

hipError_t FabricBackend::LaunchBatchCopy(const std::vector<FabricCopySegment>& segments,
                                          int device, hipStream_t stream,
                                          std::shared_ptr<void>* lifetime) {
  hipFunction_t fn = GetFabricBatchCopyFunc(device);
  if (fn == nullptr || segments.empty()) return hipErrorNotSupported;
  if (segments.size() > std::numeric_limits<size_t>::max() / sizeof(FabricCopySegment)) {
    return hipErrorInvalidValue;
  }
  std::shared_ptr<BatchStaging> staging;
  {
    std::lock_guard<std::mutex> lock(stagingMu_);
    for (auto& candidate : batchStaging_) {
      if (candidate.use_count() == 1 && candidate->device == device &&
          candidate->capacity >= segments.size()) {
        staging = candidate;
        break;
      }
    }
    if (!staging) {
      staging = std::make_shared<BatchStaging>();
      staging->device = device;
      staging->capacity = segments.size();
      const size_t bytes = segments.size() * sizeof(FabricCopySegment);
      hipError_t err =
          hipHostMalloc(reinterpret_cast<void**>(&staging->host), bytes, hipHostMallocDefault);
      if (err != hipSuccess) return err;
      err = hipMalloc(reinterpret_cast<void**>(&staging->devicePtr), bytes);
      if (err != hipSuccess) return err;
      batchStaging_.push_back(staging);
    }
  }
  // Returned before any enqueue so the caller can retain even failed uploads.
  *lifetime = staging;
  std::memcpy(staging->host, segments.data(), segments.size() * sizeof(FabricCopySegment));
  hipError_t err =
      hipMemcpyAsync(staging->devicePtr, staging->host, segments.size() * sizeof(FabricCopySegment),
                     hipMemcpyHostToDevice, stream);
  if (err != hipSuccess) return err;
  size_t maxBytes = 0;
  for (const auto& segment : segments) maxBytes = std::max(maxBytes, segment.size);
  unsigned blocksPerSegment =
      static_cast<unsigned>(std::clamp<size_t>(((maxBytes + 15) / 16 + 255) / 256, 1, 8));
  if (segments.size() > std::numeric_limits<unsigned>::max() / blocksPerSegment) {
    return hipErrorInvalidValue;
  }
  size_t count = segments.size();
  auto* descriptors = staging->devicePtr;
  void* args[] = {&descriptors, &count, &blocksPerSegment};
  return hipModuleLaunchKernel(fn, static_cast<unsigned>(count) * blocksPerSegment, 1, 1, 256, 1, 1,
                               0, stream, args, nullptr);
}

void FabricBackend::RegisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteEnginesMu);
  remoteEngines[desc.key] = desc;
  if (!remoteValidity.count(desc.key)) {
    remoteValidity.emplace(desc.key, std::make_shared<std::atomic<bool>>(true));
  }
  MORI_IO_TRACE("FABRIC: Registered remote engine {} hostname {}", desc.key, desc.hostname);
}

void FabricBackend::DeregisterRemoteEngine(const EngineDesc& desc) {
  {
    std::lock_guard<std::mutex> lock(remoteEnginesMu);
    remoteEngines.erase(desc.key);
    auto it = remoteValidity.find(desc.key);
    if (it != remoteValidity.end()) {
      it->second->store(false, std::memory_order_release);
      remoteValidity.erase(it);
    }
  }
  {
    std::lock_guard<std::mutex> lock(sessionCacheMu);
    for (auto it = sessionCache.begin(); it != sessionCache.end();) {
      if (it->first.remoteEngineKey == desc.key)
        it = sessionCache.erase(it);
      else
        ++it;
    }
  }
  std::unique_lock<std::shared_mutex> lock(importMutex);
  for (auto it = importedRegions.begin(); it != importedRegions.end();) {
    if (it->first.engineKey == desc.key)
      it = importedRegions.erase(it);
    else
      ++it;
  }
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
  InvalidateSessionsForMemory(desc);
  std::unique_lock<std::shared_mutex> lock(importMutex);
  for (auto it = importedRegions.begin(); it != importedRegions.end();) {
    if (it->first.memId == desc.id && it->first.engineKey == desc.engineKey) {
      it = importedRegions.erase(it);
    } else
      ++it;
  }
}

void FabricBackend::InvalidateRemoteMemory(const MemoryDesc& desc) {
  if (desc.engineKey != myEngKey) DeregisterMemory(desc);
}

std::shared_ptr<FabricBackend::ImportedRegion> FabricBackend::GetImportedRegion(
    const MemoryDesc& desc, int localDeviceId) {
  if (desc.engineKey == myEngKey) {
    if (desc.fabricOffset > desc.data) return nullptr;
    auto region = std::make_shared<ImportedRegion>();
    region->mappedAddr = reinterpret_cast<void*>(desc.data - desc.fabricOffset);
    region->ownsMapping = false;
    return region;
  }
  if (IsFabricHandleEmpty(desc.fabricHandle)) {
    return nullptr;
  }

  ImportCacheKey cacheKey{desc.engineKey, desc.id, localDeviceId};
  {
    std::shared_lock<std::shared_mutex> rlock(importMutex);
    auto it = importedRegions.find(cacheKey);
    if (it != importedRegions.end() && it->second->mappedAddr != nullptr) {
      return it->second;
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
  if (mapSize == 0 || desc.fabricOffset > allocBytes ||
      desc.size > allocBytes - desc.fabricOffset) {
    (void)hipMemRelease(handle);
    return nullptr;
  }
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

  auto region = std::make_shared<ImportedRegion>();
  region->handle = handle;
  region->mappedAddr = va;
  region->size = mapSize;
  region->device = localDeviceId;
  std::unique_lock<std::shared_mutex> wlock(importMutex);
  auto [it, inserted] = importedRegions.emplace(cacheKey, region);
  return it->second;
}

void FabricBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                              const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                              TransferStatus* status, TransferUniqueId id, bool isRead) {
  if (size > localDest.size || localOffset > localDest.size - size || size > remoteSrc.size ||
      remoteOffset > remoteSrc.size - size) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "FABRIC: transfer outside registered region");
    return;
  }
  auto sess = GetOrCreateSessionCached(localDest, remoteSrc);
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
  if (localOffsets.size() != sizes.size() || remoteOffsets.size() != sizes.size()) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "FABRIC: inconsistent batch vectors");
    return;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > localDest.size || localOffsets[i] > localDest.size - sizes[i] ||
        sizes[i] > remoteSrc.size || remoteOffsets[i] > remoteSrc.size - sizes[i]) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "FABRIC: batch outside registered region");
      return;
    }
  }
  auto sess = GetOrCreateSessionCached(localDest, remoteSrc);
  if (sess == nullptr) {
    status->Update(StatusCode::ERR_BAD_STATE, "FABRIC: Failed to create session");
    return;
  }
  sess->BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* FabricBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  std::lock_guard<std::mutex> peerLock(remoteEnginesMu);
  if (remote.engineKey != myEngKey && !remoteEngines.count(remote.engineKey)) return nullptr;
  auto validity = remote.engineKey == myEngKey ? std::shared_ptr<std::atomic<bool>>{}
                                               : remoteValidity.at(remote.engineKey);
  int localDevice = local.deviceId;
  void* localAddr = reinterpret_cast<void*>(local.data);
  auto region = GetImportedRegion(remote, localDevice);
  void* remoteAddr =
      region ? static_cast<char*>(region->mappedAddr) + remote.fabricOffset : nullptr;
  if (localAddr == nullptr || remoteAddr == nullptr) {
    MORI_IO_WARN("FABRIC: Failed to map memory for session (local id={}, remote id={})", local.id,
                 remote.id);
    return nullptr;
  }
  return new FabricBackendSession(config, localAddr, remoteAddr, localDevice, this,
                                  streamPool.get(), eventPool.get(), region, validity);
}

std::shared_ptr<FabricBackendSession> FabricBackend::GetOrCreateSessionCached(
    const MemoryDesc& local, const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};
  std::lock_guard<std::mutex> peerLock(remoteEnginesMu);
  if (remote.engineKey != myEngKey && !remoteEngines.count(remote.engineKey)) return nullptr;
  auto validity = remote.engineKey == myEngKey ? std::shared_ptr<std::atomic<bool>>{}
                                               : remoteValidity.at(remote.engineKey);

  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) {
    return it->second;
  }

  int localDevice = local.deviceId;
  void* localAddr = reinterpret_cast<void*>(local.data);
  auto region = GetImportedRegion(remote, localDevice);
  void* remoteAddr =
      region ? static_cast<char*>(region->mappedAddr) + remote.fabricOffset : nullptr;
  if (remoteAddr == nullptr) {
    MORI_IO_WARN("FABRIC: Failed to import remote memory for session (local.id={}, remote.id={})",
                 local.id, remote.id);
    return nullptr;
  }

  auto sess =
      std::make_shared<FabricBackendSession>(config, localAddr, remoteAddr, localDevice, this,
                                             streamPool.get(), eventPool.get(), region, validity);
  sessionCache[key] = sess;
  MORI_IO_TRACE("FABRIC: Created session for local.id={}, remote.id={}", local.id, remote.id);
  return sess;
}

void FabricBackend::InvalidateSessionsForMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  for (auto it = sessionCache.begin(); it != sessionCache.end();) {
    if ((desc.engineKey == myEngKey && it->first.localMemId == desc.id) ||
        (it->first.remoteEngineKey == desc.engineKey && it->first.remoteMemId == desc.id)) {
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
  if (remote.engineKey != myEngKey) {
    std::lock_guard<std::mutex> lock(remoteEnginesMu);
    if (!remoteEngines.count(remote.engineKey)) return false;
  }
  if (local.loc != MemoryLocationType::GPU || remote.loc != MemoryLocationType::GPU) {
    return false;
  }
  if (local.deviceId < 0 || local.deviceId >= numDevices) {
    return false;
  }
  if (!const_cast<FabricBackend*>(this)->IsReady(local.deviceId)) {
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
