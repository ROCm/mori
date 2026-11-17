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
//
// The MORI-CCO host C ABI (include/mori/cco/cco_c.h) on top of the C++ API in
// cco.hpp. Handles are ids looked up in a registry, so a stale or foreign handle
// is an error rather than a crash. No C++ exception leaves this file.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/cco/cco.hpp"
#include "mori/cco/cco_c.h"
#include "src/cco/cco_check.hpp"

#ifndef MORI_VERSION_STRING
#define MORI_VERSION_STRING "unknown"
#endif

namespace mori {
namespace cco {
namespace {

static_assert(sizeof(moriCcoUniqueId_t) == sizeof(ccoUniqueId), "unique id size drifted");
// Fails when ccoDevComm changes size: bump CCO_DEV_COMM_LAYOUT_VERSION, then update this.
static_assert(CCO_DEV_COMM_LAYOUT_VERSION == 1 && sizeof(ccoDevComm) == 200,
              "ccoDevComm layout changed without a CCO_DEV_COMM_LAYOUT_VERSION bump");

// Bytes of each struct up to its last ABI 1.0 field: the least a caller may pass.
#define MORI_CCO_END_OF(type, field) \
  (offsetof(type, field) + sizeof(static_cast<type*>(nullptr)->field))
constexpr size_t kApiSize10 = MORI_CCO_END_OF(moriCcoApi_t, devCommGetInfo);
constexpr size_t kConfigSize10 = MORI_CCO_END_OF(moriCcoCommConfig_t, reserved);
constexpr size_t kCommInfoSize10 = MORI_CCO_END_OF(moriCcoCommInfo_t, flags);
constexpr size_t kWindowInfoSize10 = MORI_CCO_END_OF(moriCcoWindowInfo_t, bytes);
constexpr size_t kReqsSize10 = MORI_CCO_END_OF(moriCcoDevCommRequirements_t, sdmaQueueCount);
constexpr size_t kDevCommInfoSize10 = MORI_CCO_END_OF(moriCcoDevCommInfo_t, sdmaQueueCount);
#undef MORI_CCO_END_OF

constexpr uint64_t kKnownCommFlags = MORI_CCO_COMM_NO_CPU_AFFINITY | MORI_CCO_COMM_LSA_ONLY;

/* ---------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------- */

thread_local detail::TlsText<4096> tlsLastError;

int Fail(int code, const char* entry, const char* msg) noexcept {
  tlsLastError.Clear();
  tlsLastError.Append(entry, std::strlen(entry));
  tlsLastError.Append(": ", 2);
  tlsLastError.Append(msg, std::strlen(msg));
  return code;
}

int Fail(int code, const char* entry, const std::string& msg) noexcept {
  return Fail(code, entry, msg.c_str());
}

// A cco call failed: report what it logged.
int CcoFail(const char* entry, const char* call) {
  const std::string why = detail::TakeErrors();
  return Fail(MORI_CCO_ERROR_INTERNAL, entry,
              why.empty() ? fmt::format("{} failed (see the MORI shmem log)", call)
                          : fmt::format("{} failed: {}", call, why));
}

int HipFail(const char* entry, const char* call, hipError_t err) {
  return Fail(MORI_CCO_ERROR_HIP, entry,
              fmt::format("{} failed: {}", call, hipGetErrorString(err)));
}

// Runs one entry: clears the last error, and turns any exception into a code.
template <typename Body>
int Guard(const char* entry, Body&& body) noexcept {
  tlsLastError.Clear();
  try {
    (void)detail::TakeErrors();
    return body();
  } catch (const std::bad_alloc&) {
    return Fail(MORI_CCO_ERROR_INTERNAL, entry, "out of host memory");
  } catch (const std::exception& e) {
    return Fail(MORI_CCO_ERROR_INTERNAL, entry, e.what());
  } catch (...) {
    return Fail(MORI_CCO_ERROR_INTERNAL, entry, "unknown exception");
  }
}

/* ---------------------------------------------------------------------------
 * Size-prefixed structs
 * ------------------------------------------------------------------------- */

// Copies a caller's struct over `out`, which holds the defaults. Bytes past this
// library's struct are fields from a newer minor and must be 0.
template <typename T>
int ReadSized(const char* entry, const char* what, const T* in, size_t minSize, T* out) {
  if (in == nullptr) return MORI_CCO_SUCCESS;
  if (in->size < minSize) {
    return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, entry,
                fmt::format("{}->size is {}, below the {} bytes of ABI 1.0; start from its _INIT "
                            "macro",
                            what, in->size, minSize));
  }
  std::memcpy(out, in, std::min(in->size, sizeof(T)));
  const auto* bytes = reinterpret_cast<const unsigned char*>(in);
  for (size_t i = sizeof(T); i < in->size; ++i) {
    if (bytes[i] != 0) {
      return Fail(MORI_CCO_ERROR_NOT_SUPPORTED, entry,
                  fmt::format("{} sets a field newer than this library's ABI {}.{}", what,
                              MORI_CCO_ABI_MAJOR, MORI_CCO_ABI_MINOR));
    }
  }
  return MORI_CCO_SUCCESS;
}

// Writes `value` into a caller's struct of out->size bytes; zero past ours.
template <typename T>
int WriteSized(const char* entry, const char* what, T value, size_t minSize, T* out) {
  if (out == nullptr) {
    return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, entry, fmt::format("{} is NULL", what));
  }
  const size_t callerSize = out->size;
  if (callerSize < minSize) {
    return Fail(
        MORI_CCO_ERROR_INVALID_ARGUMENT, entry,
        fmt::format("{}->size is {}, below the {} bytes of ABI 1.0", what, callerSize, minSize));
  }
  value.size = callerSize;
  std::memcpy(out, &value, std::min(callerSize, sizeof(T)));
  if (callerSize > sizeof(T)) {
    std::memset(reinterpret_cast<char*>(out) + sizeof(T), 0, callerSize - sizeof(T));
  }
  return MORI_CCO_SUCCESS;
}

template <typename T>
T Zeroed() {
  T v;
  std::memset(&v, 0, sizeof(v));
  v.size = sizeof(T);
  return v;
}

/* ---------------------------------------------------------------------------
 * Handles
 * ------------------------------------------------------------------------- */

struct WindowState;

struct CommState {
  std::mutex mu;            // serialises every call on this comm
  ccoComm* comm = nullptr;  // null once destroyed
  int device = -1;
  uint64_t flags = 0;
  struct DevComm {
    ccoDevComm host;
    ccoDevComm* device;
  };
  std::unordered_map<uintptr_t, std::shared_ptr<WindowState>> windows;
  std::unordered_map<const void*, DevComm> devComms;  // by device pointer
};

struct WindowState {
  std::weak_ptr<CommState> comm;  // set once; weak, since the comm owns its windows
  ccoWindow_t win = nullptr;
  void* localPtr = nullptr;
  size_t bytes = 0;
  std::vector<void*> peers;  // per world rank; null outside the LSA team
  bool released = false;     // guarded by the comm's mu
};

// Ids are never reused, so a stale handle cannot alias a live object.
class Registry {
 public:
  uintptr_t Add(std::shared_ptr<CommState> c) {
    std::lock_guard<std::mutex> lock(mu_);
    comms_.emplace(next_, std::move(c));
    return next_++;
  }
  uintptr_t Add(std::shared_ptr<WindowState> w) {
    std::lock_guard<std::mutex> lock(mu_);
    windows_.emplace(next_, std::move(w));
    return next_++;
  }
  std::shared_ptr<CommState> Comm(const void* h, bool take = false) {
    return Find(comms_, h, take);
  }
  std::shared_ptr<WindowState> Window(const void* h, bool take = false) {
    return Find(windows_, h, take);
  }
  void DropWindow(uintptr_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    windows_.erase(id);
  }

 private:
  template <typename T>
  std::shared_ptr<T> Find(std::unordered_map<uintptr_t, std::shared_ptr<T>>& map, const void* h,
                          bool take) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map.find(reinterpret_cast<uintptr_t>(h));
    if (it == map.end()) return nullptr;
    std::shared_ptr<T> out = it->second;
    if (take) map.erase(it);
    return out;
  }

  std::mutex mu_;
  uintptr_t next_ = 1;
  std::unordered_map<uintptr_t, std::shared_ptr<CommState>> comms_;
  std::unordered_map<uintptr_t, std::shared_ptr<WindowState>> windows_;
};

// Never destroyed, like the error text: handles may be released from atexit hooks.
Registry& Reg() {
  static Registry* r = new Registry();
  return *r;
}

// One entry's HIP calls: switches to a device and restores the caller's on exit.
// Leaves HIP's last error as the caller had it: an error this entry hit is cleared
// unless one was already pending.
class DeviceScope {
 public:
  // Call before the entry's first HIP call.
  void Begin() {
    if (begun_) return;
    begun_ = true;
    callerError_ = hipPeekAtLastError() != hipSuccess;
  }
  hipError_t Enter(int dev) {
    Begin();
    hipError_t err = hipGetDevice(&prev_);
    if (err != hipSuccess) return err;
    if (prev_ == dev) return hipSuccess;
    err = hipSetDevice(dev);
    restore_ = err == hipSuccess;
    return err;
  }
  ~DeviceScope() {
    if (restore_) (void)hipSetDevice(prev_);
    if (begun_ && !callerError_) (void)hipGetLastError();
  }

 private:
  int prev_ = -1;
  bool restore_ = false;
  bool begun_ = false;
  bool callerError_ = false;
};

// A comm call in progress: the comm is alive and locked, and its device current.
struct CommCall {
  std::shared_ptr<CommState> state;
  std::unique_lock<std::mutex> lock;
  DeviceScope device;
  ccoComm* comm() const { return state->comm; }
};

int BeginCall(const char* entry, moriCcoComm_t handle, bool needDevice, CommCall* call) {
  call->state = Reg().Comm(handle);
  if (!call->state) {
    return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, entry, "unknown or destroyed comm handle");
  }
  call->lock = std::unique_lock<std::mutex>(call->state->mu);
  if (call->state->comm == nullptr) {
    return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, entry, "comm was destroyed");
  }
  if (needDevice) {
    const hipError_t err = call->device.Enter(call->state->device);
    if (err != hipSuccess) return HipFail(entry, "hipSetDevice", err);
  }
  return MORI_CCO_SUCCESS;
}

// Caller holds the comm's lock and is on its device.
int ReleaseWindowLocked(ccoComm* comm, WindowState& w) {
  w.released = true;
  int rc = ccoWindowDeregister(comm, w.win);
  if (ccoMemFree(comm, w.localPtr) != 0) rc = -1;
  return rc;
}

/* ---------------------------------------------------------------------------
 * Entries
 * ------------------------------------------------------------------------- */

int GetVersion(int* abiMajor, int* abiMinor, const char** libraryVersion) {
  tlsLastError.Clear();
  if (abiMajor) *abiMajor = MORI_CCO_ABI_MAJOR;
  if (abiMinor) *abiMinor = MORI_CCO_ABI_MINOR;
  if (libraryVersion) *libraryVersion = MORI_VERSION_STRING;
  return MORI_CCO_SUCCESS;
}

const char* GetLastError() { return tlsLastError.text; }

int GetUniqueId(moriCcoUniqueId_t* uniqueId) {
  constexpr const char* kEntry = "getUniqueId";
  return Guard(kEntry, [&]() -> int {
    if (uniqueId == nullptr) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "uniqueId is NULL");
    }
    if (ccoGetUniqueId(reinterpret_cast<ccoUniqueId*>(uniqueId)) != 0) {
      return CcoFail(kEntry, "ccoGetUniqueId");
    }
    return MORI_CCO_SUCCESS;
  });
}

int CommCreate(const moriCcoUniqueId_t* uniqueId, int nRanks, int rank,
               const moriCcoCommConfig_t* config, moriCcoComm_t* out) {
  constexpr const char* kEntry = "commCreate";
  return Guard(kEntry, [&]() -> int {
    if (out == nullptr) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "comm is NULL");
    *out = nullptr;
    moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
    if (int rc = ReadSized(kEntry, "config", config, kConfigSize10, &cfg)) return rc;
    if (cfg.bootstrap != nullptr || cfg.reserved != 0) {
      return Fail(MORI_CCO_ERROR_NOT_SUPPORTED, kEntry,
                  "config->bootstrap and config->reserved must be NULL/0 in ABI 1.0");
    }
    if (cfg.flags & ~kKnownCommFlags) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  fmt::format("unknown config->flags bits {:#x}", cfg.flags & ~kKnownCommFlags));
    }
    if (uniqueId == nullptr) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "uniqueId is NULL");
    }
    if (nRanks <= 0 || rank < 0 || rank >= nRanks) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  fmt::format("rank {} is not in [0, nRanks={})", rank, nRanks));
    }

    DeviceScope scope;
    scope.Begin();
    int device = cfg.device;
    if (device == -1) {
      const hipError_t err = hipGetDevice(&device);
      if (err != hipSuccess) return HipFail(kEntry, "hipGetDevice", err);
    } else {
      int count = 0;
      const hipError_t err = hipGetDeviceCount(&count);
      if (err != hipSuccess) return HipFail(kEntry, "hipGetDeviceCount", err);
      if (device < 0 || device >= count) {
        return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                    fmt::format("config->device {} is not in [0, {})", device, count));
      }
    }
    if (const hipError_t err = scope.Enter(device); err != hipSuccess) {
      return HipFail(kEntry, "hipSetDevice", err);
    }

    ccoComm* comm = nullptr;
    const ccoUniqueId& id = *reinterpret_cast<const ccoUniqueId*>(uniqueId);
    ccoCommConfig commConfig = CCO_COMM_CONFIG_INITIALIZER;
    commConfig.cpuAffinity = (cfg.flags & MORI_CCO_COMM_NO_CPU_AFFINITY) ? 0 : 1;
    commConfig.lsaOnly = (cfg.flags & MORI_CCO_COMM_LSA_ONLY) ? 1 : 0;
    const int rc = ccoCommCreate(id, nRanks, rank, cfg.perRankVaBytes, &commConfig, &comm);
    if (rc != 0 || comm == nullptr) return CcoFail(kEntry, "ccoCommCreate");

    auto state = std::make_shared<CommState>();
    state->comm = comm;
    state->device = device;
    state->flags = cfg.flags;
    try {
      *out = reinterpret_cast<moriCcoComm_t>(Reg().Add(state));
    } catch (...) {
      (void)ccoCommDestroy(comm);
      throw;
    }
    return MORI_CCO_SUCCESS;
  });
}

int CommDestroy(moriCcoComm_t handle) {
  constexpr const char* kEntry = "commDestroy";
  return Guard(kEntry, [&]() -> int {
    auto state = Reg().Comm(handle, /*take=*/true);
    if (!state) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "unknown comm handle");
    std::lock_guard<std::mutex> lock(state->mu);
    for (auto& [id, w] : state->windows) Reg().DropWindow(id);

    DeviceScope scope;
    if (const hipError_t err = scope.Enter(state->device); err != hipSuccess) {
      // The runtime is unusable (e.g. process teardown): leak rather than crash.
      for (auto& [id, w] : state->windows) w->released = true;
      state->windows.clear();
      state->devComms.clear();
      state->comm = nullptr;
      return HipFail(kEntry, "hipSetDevice", err);
    }
    bool ok = true;
    // Dev comms first: their window tables point at the windows.
    for (auto& [ptr, d] : state->devComms) {
      ccoDevCommFreeDeviceCopy(d.device);
      if (ccoDevCommDestroy(state->comm, &d.host) != 0) ok = false;
    }
    state->devComms.clear();
    for (auto& [id, w] : state->windows) {
      if (ReleaseWindowLocked(state->comm, *w) != 0) ok = false;
    }
    state->windows.clear();
    if (ccoCommDestroy(state->comm) != 0) ok = false;
    state->comm = nullptr;
    return ok ? MORI_CCO_SUCCESS : CcoFail(kEntry, "teardown");
  });
}

int CommGetInfo(moriCcoComm_t handle, moriCcoCommInfo_t* info) {
  constexpr const char* kEntry = "commGetInfo";
  return Guard(kEntry, [&]() -> int {
    CommCall call;
    if (int rc = BeginCall(kEntry, handle, /*needDevice=*/false, &call)) return rc;
    const ccoComm* c = call.comm();
    auto v = Zeroed<moriCcoCommInfo_t>();
    v.rank = c->rank;
    v.nRanks = c->worldSize;
    v.lsaRank = c->lsaRank;
    v.lsaSize = c->lsaSize;
    v.lsaFirstRank = c->myNodeStart;
    v.device = call.state->device;
    v.memHandleType = c->handleType;
    v.perRankVaBytes = c->perRankSize;
    v.flags = call.state->flags;
    return WriteSized(kEntry, "info", v, kCommInfoSize10, info);
  });
}

int WindowImport(moriCcoComm_t handle, void* ptr, size_t bytes, uint64_t flags,
                 moriCcoWindow_t* window, moriCcoWindowInfo_t* info) {
  constexpr const char* kEntry = "windowImport";
  return Guard(kEntry, [&]() -> int {
    // Everything that can fail locally is checked before the collective.
    if (window == nullptr) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "window is NULL");
    *window = nullptr;
    if (ptr == nullptr || bytes == 0) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "ptr is NULL or bytes is 0");
    }
    if (flags != 0) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "flags must be 0");
    if (info != nullptr && info->size < kWindowInfoSize10) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  fmt::format("info->size is {}, below the {} bytes of ABI 1.0", info->size,
                              kWindowInfoSize10));
    }
    CommCall call;
    if (int rc = BeginCall(kEntry, handle, /*needDevice=*/true, &call)) return rc;
    ccoComm* c = call.comm();

    ccoWindow_t win = nullptr;
    void* localPtr = nullptr;
    if (ccoWindowRegister(c, ptr, bytes, &win, &localPtr) != 0) {
      return CcoFail(kEntry, "ccoWindowRegister");
    }
    auto w = std::make_shared<WindowState>();
    w->comm = call.state;
    w->win = win;
    w->localPtr = localPtr;
    w->bytes = bytes;
    uintptr_t id = 0;
    try {
      w->peers.resize(c->worldSize);
      for (int r = 0; r < c->worldSize; ++r) w->peers[r] = ccoGetPeerPtr(c, localPtr, r);
      id = Reg().Add(w);
      call.state->windows.emplace(id, w);
    } catch (...) {
      if (id) Reg().DropWindow(id);
      (void)ReleaseWindowLocked(c, *w);
      throw;
    }
    *window = reinterpret_cast<moriCcoWindow_t>(id);
    if (info != nullptr) {
      auto v = Zeroed<moriCcoWindowInfo_t>();
      v.localPtr = localPtr;
      v.deviceWindow = win;
      v.bytes = bytes;
      (void)WriteSized(kEntry, "info", v, kWindowInfoSize10, info);
    }
    return MORI_CCO_SUCCESS;
  });
}

int WindowRelease(moriCcoWindow_t handle) {
  constexpr const char* kEntry = "windowRelease";
  return Guard(kEntry, [&]() -> int {
    auto w = Reg().Window(handle, /*take=*/true);
    if (!w) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "unknown or released window");
    auto state = w->comm.lock();
    if (!state) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "its comm was destroyed");
    std::lock_guard<std::mutex> lock(state->mu);
    state->windows.erase(reinterpret_cast<uintptr_t>(handle));
    if (w->released || state->comm == nullptr) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "window or its comm was released");
    }
    DeviceScope scope;
    if (const hipError_t err = scope.Enter(state->device); err != hipSuccess) {
      w->released = true;  // leak rather than crash
      return HipFail(kEntry, "hipSetDevice", err);
    }
    if (ReleaseWindowLocked(state->comm, *w) != 0) return CcoFail(kEntry, "window teardown");
    return MORI_CCO_SUCCESS;
  });
}

int WindowPeerPtr(moriCcoWindow_t handle, int peer, void** ptr) {
  constexpr const char* kEntry = "windowPeerPtr";
  return Guard(kEntry, [&]() -> int {
    if (ptr == nullptr) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "ptr is NULL");
    *ptr = nullptr;
    // Lock-free: the peer table is fixed at import, so a collective on the comm cannot stall this.
    auto w = Reg().Window(handle);
    if (!w) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "unknown or released window");
    if (peer < 0 || peer >= static_cast<int>(w->peers.size())) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  fmt::format("peer {} is not in [0, {})", peer, w->peers.size()));
    }
    *ptr = w->peers[peer];
    return MORI_CCO_SUCCESS;
  });
}

int DevCommCreate(moriCcoComm_t handle, const moriCcoDevCommRequirements_t* requirements,
                  void** devComm) {
  constexpr const char* kEntry = "devCommCreate";
  return Guard(kEntry, [&]() -> int {
    if (devComm == nullptr) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "devComm is NULL");
    *devComm = nullptr;
    moriCcoDevCommRequirements_t req = MORI_CCO_DEV_COMM_REQUIREMENTS_INIT;
    if (int rc = ReadSized(kEntry, "requirements", requirements, kReqsSize10, &req)) return rc;
    if (req.lsaBarrierCount < 0 || req.sdmaQueueCount < 0) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "negative requirement count");
    }
    CommCall call;
    if (int rc = BeginCall(kEntry, handle, /*needDevice=*/true, &call)) return rc;
    ccoComm* c = call.comm();

    ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;  // ABI 1.0 is intra-node (LSA) only
    reqs.lsaBarrierCount = req.lsaBarrierCount;
    reqs.sdmaQueueCount = req.sdmaQueueCount;
    ccoDevComm host{};
    if (ccoDevCommCreate(c, &reqs, &host) != 0) return CcoFail(kEntry, "ccoDevCommCreate");
    ccoDevComm* device = ccoDevCommCopyToDevice(&host);
    if (device == nullptr) {
      (void)ccoDevCommDestroy(c, &host);
      return CcoFail(kEntry, "ccoDevCommCopyToDevice");
    }
    try {
      call.state->devComms.emplace(device, CommState::DevComm{host, device});
    } catch (...) {
      ccoDevCommFreeDeviceCopy(device);
      (void)ccoDevCommDestroy(c, &host);
      throw;
    }
    *devComm = device;
    return MORI_CCO_SUCCESS;
  });
}

int DevCommDestroy(moriCcoComm_t handle, void* devComm) {
  constexpr const char* kEntry = "devCommDestroy";
  return Guard(kEntry, [&]() -> int {
    CommCall call;
    if (int rc = BeginCall(kEntry, handle, /*needDevice=*/true, &call)) return rc;
    auto it = call.state->devComms.find(devComm);
    if (it == call.state->devComms.end()) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  "devComm is not a dev comm of this comm");
    }
    CommState::DevComm d = it->second;
    call.state->devComms.erase(it);
    ccoDevCommFreeDeviceCopy(d.device);
    if (ccoDevCommDestroy(call.comm(), &d.host) != 0) return CcoFail(kEntry, "ccoDevCommDestroy");
    return MORI_CCO_SUCCESS;
  });
}

int DevCommGetInfo(moriCcoComm_t handle, const void* devComm, moriCcoDevCommInfo_t* info) {
  constexpr const char* kEntry = "devCommGetInfo";
  return Guard(kEntry, [&]() -> int {
    CommCall call;
    if (int rc = BeginCall(kEntry, handle, /*needDevice=*/false, &call)) return rc;
    auto it = call.state->devComms.find(devComm);
    if (it == call.state->devComms.end()) {
      return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
                  "devComm is not a dev comm of this comm");
    }
    const ccoDevComm& host = it->second.host;
    auto v = Zeroed<moriCcoDevCommInfo_t>();
    v.devCommBytes = sizeof(ccoDevComm);
    v.layoutVersion = CCO_DEV_COMM_LAYOUT_VERSION;
    v.lsaBarrierCount = host.lsaBarrier.nBarriers;
    v.sdmaQueueCount = static_cast<int>(host.sdma.sdmaNumQueue);
    return WriteSized(kEntry, "info", v, kDevCommInfoSize10, info);
  });
}

}  // namespace
}  // namespace cco
}  // namespace mori

extern "C" __attribute__((visibility("default"))) int moriCcoGetApi(int abiMajor,
                                                                    moriCcoApi_t* table,
                                                                    size_t tableSize) {
  using namespace mori::cco;
  constexpr const char* kEntry = "moriCcoGetApi";
  return Guard(kEntry, [&]() -> int {
    if (abiMajor != MORI_CCO_ABI_MAJOR) {
      return Fail(MORI_CCO_ERROR_ABI_MISMATCH, kEntry,
                  fmt::format("ABI major {} requested; this library implements {}.{}", abiMajor,
                              MORI_CCO_ABI_MAJOR, MORI_CCO_ABI_MINOR));
    }
    if (table == nullptr) return Fail(MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry, "table is NULL");
    if (tableSize < kApiSize10) {
      return Fail(
          MORI_CCO_ERROR_INVALID_ARGUMENT, kEntry,
          fmt::format("tableSize {} is below the {} bytes of ABI 1.0", tableSize, kApiSize10));
    }
    moriCcoApi_t api;
    std::memset(&api, 0, sizeof(api));
    api.size = std::min(tableSize, sizeof(api));
    api.abiMajor = MORI_CCO_ABI_MAJOR;
    api.abiMinor = MORI_CCO_ABI_MINOR;
    api.getVersion = GetVersion;
    api.getLastError = GetLastError;
    api.getUniqueId = GetUniqueId;
    api.commCreate = CommCreate;
    api.commDestroy = CommDestroy;
    api.commGetInfo = CommGetInfo;
    api.windowImport = WindowImport;
    api.windowRelease = WindowRelease;
    api.windowPeerPtr = WindowPeerPtr;
    api.devCommCreate = DevCommCreate;
    api.devCommDestroy = DevCommDestroy;
    api.devCommGetInfo = DevCommGetInfo;
    std::memcpy(table, &api, api.size);
    if (tableSize > sizeof(api)) {
      std::memset(reinterpret_cast<char*>(table) + sizeof(api), 0, tableSize - sizeof(api));
    }
    return MORI_CCO_SUCCESS;
  });
}
