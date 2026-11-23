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
#include "umbp/l1/l1_pool.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/l1/l1_fabric.h"

namespace mori::umbp::l1 {

namespace {

// How long Allocate waits for a slot that is free but pinned by a copy.
constexpr std::chrono::milliseconds kPinnedWait{2000};
// How long a fetch into held slots waits for their owner's writes to settle.
constexpr std::chrono::milliseconds kSettleWait{1000};

std::string HipError(const char* what, hipError_t err) {
  return std::string(what) + " failed: " + hipGetErrorString(err);
}

// Current device is per thread; these calls arrive on arbitrary RPC threads.
class ScopedDevice {
 public:
  explicit ScopedDevice(int device) {
    if (hipGetDevice(&previous_) != hipSuccess) previous_ = -1;
    if (device == previous_) {
      ok_ = true;
      return;
    }
    ok_ = hipSetDevice(device) == hipSuccess;
    restore_ = ok_ && previous_ >= 0;
  }
  ~ScopedDevice() {
    if (restore_) (void)hipSetDevice(previous_);
  }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;
  bool ok() const { return ok_; }

 private:
  int previous_ = -1;
  bool ok_ = false;
  bool restore_ = false;
};

// Allocation, import and their undoing are made under one exclusive lock.
// Attaches for several devices arrive at once on different RPC threads, and
// concurrent allocate / zero-fill / synchronize calls from those threads have
// been seen to park forever inside the runtime. Copies share that lock, so they
// never run beside an attach, and otherwise only serialize per destination
// device: fetches into different pools copy at once, each on its device's own
// stream, which is all they synchronize.
class HipMemoryApi final : public MemoryApi {
 public:
  bool Allocate(int device, uint64_t bytes, void** base, ExportedMemory* exported,
                std::string* error) override {
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(device) + ") failed";
      return false;
    }
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, static_cast<size_t>(bytes));
    if (err != hipSuccess || ptr == nullptr) {
      *error = HipError("hipMalloc", err);
      return false;
    }
    err = hipMemset(ptr, 0, static_cast<size_t>(bytes));
    if (err == hipSuccess) err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      *error = HipError("hipMemset", err);
      (void)hipFree(ptr);
      return false;
    }
    hipIpcMemHandle_t handle;
    err = hipIpcGetMemHandle(&handle, ptr);
    if (err != hipSuccess) {
      *error = HipError("hipIpcGetMemHandle", err);
      (void)hipFree(ptr);
      return false;
    }
    exported->kind = "hip_ipc";
    exported->handle.assign(handle.reserved, sizeof(handle.reserved));
    *base = ptr;
    return true;
  }

  void Free(int device, void* base, uint64_t /*bytes*/) override {
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    (void)hipFree(base);
  }

  bool Import(int device, const std::string& handle, uint64_t offset, void** base, void** mapped,
              std::string* error) override {
    hipIpcMemHandle_t ipc;
    if (handle.size() != sizeof(ipc.reserved)) {
      *error = "external handle is not a hip_ipc handle";
      return false;
    }
    std::memcpy(ipc.reserved, handle.data(), sizeof(ipc.reserved));
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(device) + ") failed";
      return false;
    }
    void* ptr = nullptr;
    const hipError_t err = hipIpcOpenMemHandle(&ptr, ipc, hipIpcMemLazyEnablePeerAccess);
    if (err != hipSuccess || ptr == nullptr) {
      *error = HipError("hipIpcOpenMemHandle", err);
      return false;
    }
    *mapped = ptr;
    *base = static_cast<char*>(ptr) + offset;
    return true;
  }

  void Unimport(int device, void* mapped) override {
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    (void)hipIpcCloseMemHandle(mapped);
  }

  bool Copy(int dst_device, void* dst, int /*src_device*/, const void* src, uint64_t bytes,
            std::string* error) override {
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(dst_device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(dst_device) + ") failed";
      return false;
    }
    const hipError_t err = hipMemcpy(dst, src, static_cast<size_t>(bytes), hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
      *error = HipError("hipMemcpy", err);
      return false;
    }
    return true;
  }

  // Every fragment on the destination device's own stream, waited for once.
  // The copy engine, not a kernel: the engine that owns both pools is running
  // its model on these GPUs, and a gather kernel there takes compute units
  // from it (measured: the engine's decode slowed more than the copies sped
  // up).
  bool CopyBatch(int dst_device, const std::vector<Fragment>& fragments,
                 std::string* error) override {
    if (fragments.empty()) return true;
    std::shared_lock<std::shared_mutex> shared(hip_mutex_);
    std::lock_guard<std::mutex> device_lock(DeviceMutex(dst_device));
    ScopedDevice scope(dst_device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(dst_device) + ") failed";
      return false;
    }
    hipStream_t stream = StreamFor(dst_device);
    if (stream == nullptr) {
      *error = "no copy stream on device " + std::to_string(dst_device);
      return false;
    }
    for (const Fragment& f : fragments) {
      const hipError_t err = hipMemcpyAsync(f.dst, f.src, static_cast<size_t>(f.bytes),
                                            hipMemcpyDeviceToDevice, stream);
      if (err != hipSuccess) {
        *error = HipError("hipMemcpyAsync", err);
        return false;
      }
    }
    const hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
      *error = HipError("hipStreamSynchronize", err);
      return false;
    }
    return true;
  }

  // A device-to-device copy is not guaranteed to have landed when hipMemcpy
  // returns, and the caller hands these bytes to another process next.
  bool Synchronize(int device, std::string* error) override {
    std::unique_lock<std::shared_mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    const hipError_t err = scope.ok() ? hipDeviceSynchronize() : hipErrorInvalidDevice;
    if (err != hipSuccess) {
      *error = HipError("hipDeviceSynchronize", err);
      return false;
    }
    return true;
  }

 private:
  std::mutex& DeviceMutex(int device) {
    std::lock_guard<std::mutex> lock(table_mutex_);
    return device_mutex_[device];
  }

  // The device is current. Created once and kept: a pool's copies all go here.
  hipStream_t StreamFor(int device) {
    std::lock_guard<std::mutex> lock(table_mutex_);
    auto found = streams_.find(device);
    if (found != streams_.end()) return found->second;
    hipStream_t stream = nullptr;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) stream = nullptr;
    streams_[device] = stream;
    return stream;
  }

  std::shared_mutex hip_mutex_;
  std::mutex table_mutex_;
  std::map<int, std::mutex> device_mutex_;
  std::map<int, hipStream_t> streams_;
};

class HostMemoryApi final : public MemoryApi {
 public:
  bool Allocate(int /*device*/, uint64_t bytes, void** base, ExportedMemory* exported,
                std::string* error) override {
    void* ptr = std::calloc(1, static_cast<size_t>(bytes));
    if (ptr == nullptr) {
      *error = "calloc failed";
      return false;
    }
    exported->kind = "host";
    exported->handle.clear();
    *base = ptr;
    return true;
  }
  void Free(int /*device*/, void* base, uint64_t /*bytes*/) override { std::free(base); }
  // In-process stand-in for an exported allocation: the handle carries the
  // allocation's address (8 bytes, little-endian).
  bool Import(int /*device*/, const std::string& handle, uint64_t offset, void** base,
              void** mapped, std::string* error) override {
    uint64_t address = 0;
    if (handle.size() != sizeof(address)) {
      *error = "a host handle is an 8-byte address";
      return false;
    }
    std::memcpy(&address, handle.data(), sizeof(address));
    *mapped = reinterpret_cast<void*>(address);
    *base = static_cast<char*>(*mapped) + offset;
    return true;
  }
  void Unimport(int /*device*/, void* /*mapped*/) override {}
  bool Copy(int /*dst_device*/, void* dst, int /*src_device*/, const void* src, uint64_t bytes,
            std::string* /*error*/) override {
    std::memcpy(dst, src, static_cast<size_t>(bytes));
    return true;
  }
  bool Synchronize(int /*device*/, std::string* /*error*/) override { return true; }
};

bool SameLayout(const Layout& a, const Layout& b) {
  if (a.num_slots != b.num_slots || a.regions.size() != b.regions.size()) return false;
  for (size_t i = 0; i < a.regions.size(); ++i) {
    if (a.regions[i].offset != b.regions[i].offset ||
        a.regions[i].unit_bytes != b.regions[i].unit_bytes) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool MemoryApi::CopyBatch(int dst_device, const std::vector<Fragment>& fragments,
                          std::string* error) {
  for (const Fragment& f : fragments) {
    if (!Copy(dst_device, f.dst, f.src_device, f.src, f.bytes, error)) return false;
  }
  return fragments.empty() || Synchronize(dst_device, error);
}

std::shared_ptr<MemoryApi> MakeHipMemoryApi() { return std::make_shared<HipMemoryApi>(); }
std::shared_ptr<MemoryApi> MakeHostMemoryApi() { return std::make_shared<HostMemoryApi>(); }

CopyOutcome MemoryApi::CopyBatchTracked(int device, const std::vector<Fragment>& fragments) {
  std::string error;
  if (CopyBatch(device, fragments, &error)) return {true, true, {}};
  // CopyBatch can fail after earlier submissions. A successful drain, not the
  // error itself, is what authorizes the caller to release pins and reuse slots.
  std::string drain_error;
  const bool drained = Synchronize(device, &drain_error);
  if (!drained) error += "; drain failed: " + drain_error;
  return {false, drained, error};
}

namespace {

uint32_t EnvCount(const char* name, uint32_t fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const unsigned long value = std::strtoul(raw, &end, 10);
  return (end == raw || *end != '\0') ? fallback : static_cast<uint32_t>(value);
}

// One line per this many evictions of a pool: how much it evicts and how much
// of that another pool still holds.
constexpr uint64_t kEvictionLogEvery = 16384;

std::string NewIncarnation() {
  std::random_device random;
  std::string id;
  for (int i = 0; i < 4; ++i) id += std::to_string(random()) + "-";
  return id;
}
constexpr size_t kPeerKeys = 128;
constexpr size_t kPeerRegions = 64;
// Receipts/tombstones are not expired. Bound their memory instead of allowing a
// late Acquire to resurrect a released operation. Restart only after draining.
constexpr size_t kPeerReceipts = 65536;
bool ValidPeerShape(const std::vector<std::string>& keys, const std::vector<uint64_t>& units) {
  return keys.size() <= kPeerKeys && !units.empty() && units.size() <= kPeerRegions &&
         std::all_of(units.begin(), units.end(), [](uint64_t n) { return n != 0; }) &&
         std::all_of(keys.begin(), keys.end(),
                     [](const std::string& k) { return !k.empty() && k.size() <= 1024; });
}

}  // namespace

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api)
    : api_(std::move(api)), incarnation_(NewIncarnation()) {
  SetEvictionPolicy(EnvCount("UMBP_L1_KEEP_COPIES", 0));
}

void PoolManager::SetEvictionPolicy(uint32_t keep_copies) {
  std::lock_guard<std::mutex> lock(mutex_);
  keep_copies_ = keep_copies;
  MORI_UMBP_INFO("[L1Pool] eviction policy: keep_copies={}", keep_copies_);
}

void PoolManager::UncacheLocked(Pool* pool, uint32_t slot) {
  Slot& s = pool->slots[slot];
  (s.kept ? pool->kept : pool->lru).erase(s.lru_it);
  s.kept = false;
}

bool PoolManager::IsSpareLocked(const Pool* pool, uint32_t slot) const {
  const Slot& s = pool->slots[slot];
  if (!s.has_key) return true;
  const auto group = groups_.find(pool->group);
  if (group == groups_.end()) return true;
  const auto holders = group->second.holders.find(s.key);
  if (holders == group->second.holders.end()) return true;
  // Holders stand oldest first (`HolderAddedLocked`, `HolderUsedLocked`): the
  // last `keep_copies_` are the copies worth keeping, every older one is spare.
  const std::vector<Holder>& list = holders->second;
  if (list.size() <= keep_copies_) return false;
  for (size_t i = list.size() - keep_copies_; i < list.size(); ++i) {
    if (list[i].pool == pool && list[i].slot == slot) return false;
  }
  return true;
}

void PoolManager::HolderAddedLocked(Group* group, const std::string& key) {
  if (keep_copies_ == 0) return;
  const auto holders = group->holders.find(key);
  if (holders == group->holders.end() || holders->second.size() <= keep_copies_) return;
  // The newest holder stands last. The one it pushed out of the newest
  // `keep_copies_` is spare from now on: if its pool had set it aside as a copy
  // to keep, it is that pool's coldest spare copy.
  const Holder& h = holders->second[holders->second.size() - 1 - keep_copies_];
  Slot& s = h.pool->slots[h.slot];
  if (s.state == SlotState::kCached && s.kept) {
    h.pool->lru.splice(h.pool->lru.begin(), h.pool->kept, s.lru_it);
    s.kept = false;
  }
}

void PoolManager::HolderUsedLocked(Pool* pool, uint32_t slot) {
  if (keep_copies_ == 0) return;
  const Slot& s = pool->slots[slot];
  if (!s.has_key) return;
  const auto group = groups_.find(pool->group);
  if (group == groups_.end()) return;
  const auto holders = group->second.holders.find(s.key);
  if (holders == group->second.holders.end() || holders->second.size() < 2) return;
  std::vector<Holder>& list = holders->second;
  const auto mine = std::find_if(list.begin(), list.end(),
                                 [&](const Holder& h) { return h.pool == pool && h.slot == slot; });
  if (mine == list.end() || std::next(mine) == list.end()) return;
  // The copy an engine has just taken for a request is the newest one.
  std::rotate(mine, std::next(mine), list.end());
  HolderAddedLocked(&group->second, s.key);
}

PoolManager::~PoolManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  bool retained = false;
  for (auto& entry : pools_) {
    Pool& pool = *entry.second;
    if (pool.busy != 0 || pool.quarantined) {
      retained = true;
      continue;  // No completion proof: do not unmap memory still used by a peer.
    }
    if (pool.external)
      api_->Unimport(pool.device, pool.mapped);
    else
      api_->Free(pool.device, pool.base, pool.bytes);
  }
  if (retained) {
    // Deliberately retain the mappings on graceful shutdown. Forced process or
    // GPU teardown requires coordinated peer shutdown; no automatic recovery.
    (void)new std::shared_ptr<MemoryApi>(api_);
  }
  pools_.clear();
}

PoolManager::Pool* PoolManager::FindLocked(const std::string& pool_id) {
  auto it = pools_.find(pool_id);
  return it == pools_.end() ? nullptr : it->second.get();
}

void PoolManager::UnindexLocked(Pool* pool, uint32_t slot) {
  Slot& s = pool->slots[slot];
  if (!s.has_key) return;
  pool->index.erase(s.key);
  auto group = groups_.find(pool->group);
  if (group != groups_.end()) {
    auto holders = group->second.holders.find(s.key);
    if (holders != group->second.holders.end()) {
      auto& list = holders->second;
      list.erase(std::remove_if(list.begin(), list.end(),
                                [&](const Holder& h) { return h.pool == pool && h.slot == slot; }),
                 list.end());
      if (list.empty()) group->second.holders.erase(holders);
    }
  }
  s.has_key = false;
  s.key.clear();
}

void PoolManager::ResetSlotsLocked(Pool* pool) {
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  pool->index.clear();
  pool->lru.clear();
  pool->kept.clear();
  pool->vacant.clear();
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) {
    pool->slots[slot] = Slot{};
    pool->vacant.insert(slot);
  }
  pool->held = 0;
  pool->committed = 0;
}

bool PoolManager::TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                                 bool settled_only) {
  for (auto it = pool->vacant.begin(); it != pool->vacant.end(); ++it) {
    if (pool->slots[*it].pins != 0 || pool->slots[*it].writers != 0) continue;
    if (settled_only && !Fillable(*pool, pool->slots[*it])) continue;
    *slot = *it;
    pool->vacant.erase(it);
    return true;
  }
  if (evicted == nullptr) return false;
  const auto takeable = [&](const Slot& s) {
    return s.pins == 0 && s.writers == 0 && (!settled_only || Fillable(*pool, s));
  };
  const auto group = groups_.find(pool->group);
  // How many pools of the group hold `s`'s key, this one included.
  const auto copies = [&](const Slot& s) -> size_t {
    if (!s.has_key || group == groups_.end()) return 0;
    const auto holders = group->second.holders.find(s.key);
    return holders == group->second.holders.end() ? 0 : holders->second.size();
  };
  // The coldest slot that may be taken. With `keep_copies_` it is the coldest
  // spare copy: a slot whose key `keep_copies_` pools or more hold a newer
  // copy of (`IsSpareLocked`). The copies to keep met on the way are set
  // aside in `kept`, in their age order, so each is looked at once and not on
  // every eviction (a prefix is a run of hundreds of slots of the same age,
  // all to keep or all spare). `kept` is evicted from only when `lru` holds
  // nothing to take.
  bool found = false;
  uint32_t victim = 0;
  for (auto it = pool->lru.begin(); it != pool->lru.end();) {
    Slot& s = pool->slots[*it];
    if (!takeable(s)) {
      ++it;
      continue;
    }
    if (keep_copies_ == 0 || IsSpareLocked(pool, *it)) {
      victim = *it;
      found = true;
      break;
    }
    const auto next = std::next(it);
    pool->kept.splice(pool->kept.end(), pool->lru, it);
    s.kept = true;
    it = next;
  }
  if (!found) {
    for (uint32_t candidate : pool->kept) {
      if (takeable(pool->slots[candidate])) {
        victim = candidate;
        found = true;
        break;
      }
    }
  }
  if (!found) return false;
  Slot& s = pool->slots[victim];
  *slot = victim;
  UncacheLocked(pool, victim);
  if (s.has_key) {
    if (copies(s) > 1) ++pool->evicted_spare;
    evicted->push_back(s.key);
    UnindexLocked(pool, *slot);
    if (++pool->evicted % kEvictionLogEvery == 0) {
      MORI_UMBP_INFO(
          "[L1Pool] evictions pool={} total={} spare_copy={} cached={} last_copies_kept={} "
          "vacant={}",
          pool->id, pool->evicted, pool->evicted_spare, pool->lru.size() + pool->kept.size(),
          pool->kept.size(), pool->vacant.size());
    }
  }
  return true;
}

bool PoolManager::HasPinnedFreeLocked(const Pool* pool) const {
  for (uint32_t slot : pool->vacant) {
    if (pool->slots[slot].pins != 0) return true;
  }
  for (uint32_t slot : pool->lru) {
    if (pool->slots[slot].pins != 0) return true;
  }
  for (uint32_t slot : pool->kept) {
    if (pool->slots[slot].pins != 0) return true;
  }
  return false;
}

Status PoolManager::WaitIdleLocked(std::unique_lock<std::mutex>& lock, const std::string& pool_id,
                                   const char* operation) {
  // A peer may disappear while holding a read grant. Never strand a control
  // RPC indefinitely and never interpret a timeout as permission to unmap.
  Pool* original = FindLocked(pool_id);
  const uint64_t generation = original != nullptr ? original->generation : 0;
  const bool ready = idle_.wait_for(lock, kPinnedWait, [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->generation != generation || p->quarantined || p->busy == 0;
  });
  Pool* pool = FindLocked(pool_id);
  if (pool != nullptr && pool->generation != generation) {
    return Status::Error(std::string(operation) + ": pool " + pool_id +
                         " was replaced while waiting; the replacement was not modified");
  }
  if (pool != nullptr && pool->quarantined) {
    return Status::Error(std::string(operation) + ": pool " + pool_id +
                         " is quarantined; retain the allocation until coordinated peer drain");
  }
  if (!ready) {
    return Status::Error(std::string(operation) + ": pool " + pool_id +
                         " still has in-flight reads or writes; retain its allocation and retry "
                         "only after the peer completes and releases its read");
  }
  return {};
}

Status PoolManager::DestroyLocked(std::unique_lock<std::mutex>& lock, Pool* pool) {
  const std::string pool_id = pool->id;
  Status idle = WaitIdleLocked(lock, pool_id, "detach");
  if (!idle.ok) return idle;
  pool = FindLocked(pool_id);
  if (pool == nullptr) return {};
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  if (pool->external) {
    api_->Unimport(pool->device, pool->mapped);
  } else {
    api_->Free(pool->device, pool->base, pool->bytes);
  }
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} device={} bytes={}{} evicted={} spare_copy={}",
                 pool_id, pool->device, pool->bytes, pool->external ? " (adopted, unmapped)" : "",
                 pool->evicted, pool->evicted_spare);
  pools_.erase(pool_id);
  return {};
}

AttachResult PoolManager::Attach(const std::string& pool_id, const std::string& group, int device,
                                 uint64_t bytes, const ExternalMemory& external) {
  const bool adopt = !external.handle.empty();
  AttachResult result;
  result.incarnation = incarnation_;
  if (external.kind == "hip_fabric") {
    std::string error;
    if (!api_->SupportsFabric() || !IsFabricHandle(external.handle) ||
        !ValidateFabricRange(external.handle, external.offset, bytes, &error)) {
      result.status = Status::Error("invalid FABRIC attachment: " + error);
      return result;
    }
  } else if (!external.kind.empty() && external.kind != "hip_ipc") {
    result.status = Status::Error("unknown external memory kind");
    return result;
  }
  if (api_->SupportsFabric() && external.kind != "hip_fabric") {
    result.status = Status::Error("FABRIC L1 only adopts exported native HBM");
    return result;
  }
  if (pool_id.empty() || bytes == 0 || device < 0) {
    result.status = Status::Error("attach needs a pool id, a device and a non-zero size");
    return result;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (attaching_.count(pool_id) != 0) {
    result.status = Status::Error("pool " + pool_id + " is already being attached");
    return result;
  }
  if (Pool* existing = FindLocked(pool_id)) {
    if (existing->quarantined) {
      result.status = Status::Error("pool is quarantined after an unconfirmed copy");
      return result;
    }
    if (existing->group == group && existing->device == device && existing->bytes == bytes &&
        existing->external == adopt && existing->external_handle == external.handle &&
        existing->external_kind == external.kind && existing->external_offset == external.offset) {
      result.memory = existing->exported;
      result.bytes = existing->bytes;
      result.created = false;
      return result;
    }
    // Reserve the id before the wait releases mutex_: a second Attach must
    // not race this replacement and publish a different allocation under it.
    attaching_.insert(pool_id);
    result.status = DestroyLocked(lock, existing);
    if (!result.status.ok) {
      attaching_.erase(pool_id);
      return result;
    }
  }
  // The allocation zero-fills hundreds of GB; do it without the lock so pools
  // already serving are not stalled behind it.
  attaching_.insert(pool_id);
  lock.unlock();
  void* base = nullptr;
  void* mapped = nullptr;
  ExportedMemory exported;
  std::string error;
  const bool ok =
      adopt ? api_->Import(device, external.handle, external.offset, &base, &mapped, &error)
            : api_->Allocate(device, bytes, &base, &exported, &error);
  lock.lock();
  attaching_.erase(pool_id);
  if (!ok) {
    MORI_UMBP_ERROR("[L1Pool] attach pool={} device={} bytes={} failed: {}", pool_id, device, bytes,
                    error);
    result.status = Status::Error(error);
    return result;
  }
  auto pool = std::make_unique<Pool>();
  pool->id = pool_id;
  pool->generation = ++next_pool_;
  pool->group = group;
  pool->device = device;
  pool->base = base;
  pool->bytes = bytes;
  pool->exported = exported;
  pool->external = adopt;
  pool->mapped = mapped;
  pool->external_handle = external.handle;
  pool->external_kind = external.kind;
  pool->external_offset = external.offset;
  pools_[pool_id] = std::move(pool);
  MORI_UMBP_INFO("[L1Pool] attached pool={} group={} device={} bytes={}{}", pool_id, group, device,
                 bytes, adopt ? " (adopted from its owner)" : "");
  result.memory = std::move(exported);
  result.bytes = bytes;
  result.created = true;
  return result;
}

Status PoolManager::BindLayout(const std::string& pool_id, const Layout& layout) {
  if (layout.num_slots == 0 || layout.regions.empty()) {
    return Status::Error("a layout needs slots and at least one region");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status::Error("unknown pool " + pool_id);
  if (pool->quarantined) return Status::Error("pool is quarantined");
  if (pool->bound && SameLayout(pool->layout, layout)) return Status{};
  if (pool->bound) {
    // Wait before validating allocation bounds or group compatibility: those
    // facts may change while the mutex is released by the wait.
    Status idle = WaitIdleLocked(lock, pool_id, "bind layout");
    if (!idle.ok) return idle;
    pool = FindLocked(pool_id);
    if (pool == nullptr) return Status::Error("pool " + pool_id + " was detached");
  }
  std::vector<uint64_t> units;
  std::vector<std::pair<uint64_t, uint64_t>> spans;
  for (const Region& region : layout.regions) {
    if (region.unit_bytes == 0 || region.offset > pool->bytes ||
        layout.num_slots > (pool->bytes - region.offset) / region.unit_bytes) {
      return Status::Error("a region does not fit in pool " + pool_id);
    }
    units.push_back(region.unit_bytes);
    spans.emplace_back(region.offset,
                       region.offset + uint64_t{layout.num_slots} * region.unit_bytes);
  }
  std::sort(spans.begin(), spans.end());
  for (size_t i = 1; i < spans.size(); ++i) {
    if (spans[i].first < spans[i - 1].second) {
      return Status::Error("layout regions overlap in pool " + pool_id);
    }
  }
  Group& group = groups_[pool->group];
  if (group.has_layout && group.unit_bytes != units) {
    for (const auto& entry : pools_) {
      const Pool* other = entry.second.get();
      if (other != pool && other->bound && other->group == pool->group) {
        return Status::Error("pool " + pool_id + " declares slot sizes that differ from group " +
                             pool->group);
      }
    }
  }
  if (pool->bound) ResetSlotsLocked(pool);
  Group& current = groups_[pool->group];
  current.has_layout = true;
  current.unit_bytes = std::move(units);
  pool->layout = layout;
  pool->slots.assign(layout.num_slots, Slot{});
  pool->bound = true;
  ResetSlotsLocked(pool);
  MORI_UMBP_INFO("[L1Pool] bound pool={} slots={} regions={}", pool_id, layout.num_slots,
                 layout.regions.size());
  return Status{};
}

OpenResult PoolManager::Open(const std::string& pool_id) {
  OpenResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  result.status = WaitIdleLocked(lock, pool_id, "open");
  if (!result.status.ok) return result;
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  ResetSlotsLocked(pool);
  result.num_slots = pool->layout.num_slots;
  MORI_UMBP_INFO("[L1Pool] opened pool={} slots={}", pool_id, result.num_slots);
  return result;
}

AllocateResult PoolManager::Allocate(const std::string& pool_id, uint32_t count,
                                     uint32_t min_count) {
  AllocateResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  ++pool->busy;  // Retain partial grants/layout while waiting on pinned slots.
  const auto deadline = std::chrono::steady_clock::now() + kPinnedWait;
  while (result.slots.size() < count) {
    uint32_t slot = 0;
    const bool required = result.slots.size() < min_count;
    if (TakeFreeLocked(pool, &slot, required ? &result.evicted_keys : nullptr)) {
      pool->slots[slot].state = SlotState::kHeld;
      ++pool->held;
      result.slots.push_back(slot);
      continue;
    }
    if (!required) break;
    // Every remaining free slot is being read by a copy; those finish quickly.
    if (!HasPinnedFreeLocked(pool)) break;
    if (idle_.wait_until(lock, deadline) == std::cv_status::timeout) break;
    pool = FindLocked(pool_id);
    if (pool == nullptr || !pool->bound || pool->quarantined) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      if (pool != nullptr) --pool->busy;
      idle_.notify_all();
      return result;
    }
  }
  --pool->busy;
  idle_.notify_all();
  return result;
}

Status PoolManager::Commit(const std::string& pool_id, uint64_t committed) {
  std::lock_guard<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status::Error("unknown pool " + pool_id);
  pool->committed = std::max(pool->committed, committed);
  // A fetch into this pool's held slots may be waiting for them to settle.
  idle_.notify_all();
  return Status{};
}

Status PoolManager::Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    return Status::Error("pool " + pool_id + " is not attached and bound");
  }
  // Reject a whole batch before mutating it if it would reuse an in-flight
  // destination or change bytes pinned by another tray.
  for (const Op& op : ops) {
    if (op.slot >= pool->slots.size()) continue;
    const Slot& slot = pool->slots[op.slot];
    // Hold of cached content is a read-only prefix claim. Reuse for different
    // content must go through Allocate, which excludes pinned slots.
    if (slot.writers != 0 ||
        (slot.pins != 0 && op.kind == OpKind::kPublish && (!slot.has_key || slot.key != op.key))) {
      return Status::Error("apply would reuse a slot with an in-flight copy");
    }
  }
  for (size_t i = 0; i < ops.size(); ++i) {
    const Op& op = ops[i];
    auto fail = [&](const char* why) {
      return Status::Error("op " + std::to_string(i) + " on slot " + std::to_string(op.slot) +
                           " of pool " + pool_id + ": " + why);
    };
    if (op.slot >= pool->slots.size()) return fail("slot out of range");
    Slot& s = pool->slots[op.slot];
    switch (op.kind) {
      case OpKind::kHold:
        if (s.state == SlotState::kHeld) break;
        if (s.state == SlotState::kFilling) return fail("hold of a slot being filled");
        if (s.pins != 0 && !s.has_key) return fail("reuse of a slot pinned by a reader");
        if (s.state == SlotState::kCached) {
          UncacheLocked(pool, op.slot);
          HolderUsedLocked(pool, op.slot);
        } else {
          pool->vacant.erase(op.slot);
        }
        s.state = SlotState::kHeld;
        ++pool->held;
        break;
      case OpKind::kRelease:
        if (s.state != SlotState::kHeld) return fail("release of a slot that is not held");
        --pool->held;
        s.released = stamp;
        if (s.has_key) {
          s.state = SlotState::kCached;
          s.lru_it = pool->lru.insert(pool->lru.end(), op.slot);
        } else {
          s.state = SlotState::kVacant;
          pool->vacant.insert(op.slot);
        }
        break;
      case OpKind::kPublish: {
        if (s.state != SlotState::kHeld) return fail("publish of a slot that is not held");
        if (op.key.empty()) return fail("publish without a key");
        if (s.has_key && s.key == op.key) break;
        if (s.pins != 0) return fail("change of content pinned by a reader");
        UnindexLocked(pool, op.slot);
        auto previous = pool->index.find(op.key);
        if (previous != pool->index.end()) {
          // The same content, now named by this slot. The older copy stops
          // being reachable; if nobody holds it, it is simply free.
          const uint32_t old_slot = previous->second;
          Slot& old = pool->slots[old_slot];
          UnindexLocked(pool, old_slot);
          old.released = stamp;
          if (old.state == SlotState::kCached) {
            UncacheLocked(pool, old_slot);
            old.state = SlotState::kVacant;
            pool->vacant.insert(old_slot);
          }
        }
        s.has_key = true;
        s.key = op.key;
        s.written = stamp;
        pool->index[op.key] = op.slot;
        Group& group = groups_[pool->group];
        group.holders[op.key].push_back(Holder{pool, op.slot});
        HolderAddedLocked(&group, op.key);
        break;
      }
      default:
        return fail("unknown op");
    }
  }
  return Status{};
}

FetchResult PoolManager::FetchLocal(const std::string& pool_id,
                                    const std::vector<std::string>& keys,
                                    const FetchOptions& options) {
  FetchResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  Group& group = groups_[pool->group];
  // Another pool's settled copy of `key`, or null.
  auto source_of = [&](const std::string& key) -> const Holder* {
    auto holders = group.holders.find(key);
    if (holders == group.holders.end()) return nullptr;
    for (const Holder& h : holders->second) {
      // Not this pool, and not a slot its owner may still be writing.
      if (h.pool != pool && !h.pool->quarantined && h.pool->slots[h.slot].writers == 0 &&
          Readable(*h.pool, h.pool->slots[h.slot]))
        return &h;
    }
    return nullptr;
  };

  if (options.probe_only) {
    for (const std::string& key : keys) {
      if (pool->index.find(key) == pool->index.end() && source_of(key) == nullptr) break;
      ++result.available;
    }
    return result;
  }
  const bool into = !options.dst_slots.empty();
  if (into && options.dst_slots.size() < keys.size()) {
    result.status = Status::Error("fetch into pool " + pool_id + ": fewer slots than keys");
    return result;
  }
  // Retain the allocation and its layout through the settle wait as well as
  // the unlocked copy, including calls with only local hits or no copy steps.
  ++pool->busy;
  if (into) {
    // A regranted slot may still have its owner's writes in flight. They land
    // within a forward or two, and the owner's next Commit says so: wait for
    // that rather than leave the caller to recompute what it asked for. Past
    // the bound, the check below stops at the first unsettled slot.
    const auto settle_deadline = std::chrono::steady_clock::now() + kSettleWait;
    idle_.wait_until(lock, settle_deadline, [&] {
      pool = FindLocked(pool_id);
      if (pool == nullptr || !pool->bound || pool->quarantined) return true;
      for (size_t i = 0; i < keys.size(); ++i) {
        const uint32_t dst = options.dst_slots[i];
        if (dst < pool->slots.size() && !Fillable(*pool, pool->slots[dst])) return false;
      }
      return true;
    });
    if (pool == nullptr || !pool->bound || pool->quarantined) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      if (pool != nullptr) --pool->busy;
      idle_.notify_all();
      return result;
    }
  }

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists -- or, copying
  // into the caller's slots, check that it holds them.
  std::vector<CopyStep> steps;
  std::vector<uint32_t> local_pins;
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string& key = keys[i];
    if (into) {
      const uint32_t dst = options.dst_slots[i];
      if (dst >= pool->slots.size() || pool->slots[dst].state != SlotState::kHeld ||
          pool->slots[dst].has_key) {
        result.status = Status::Error("fetch into pool " + pool_id + ": slot " +
                                      std::to_string(dst) + " is not an unindexed held slot");
        break;
      }
      const Holder* source = source_of(key);
      if (source == nullptr) break;
      if (pool->slots[dst].pins != 0 || pool->slots[dst].writers != 0 ||
          !Fillable(*pool, pool->slots[dst])) {
        // Allocate may regrant a slot before its owner's writes complete:
        // owner-local reuse is ordered on that stream, but this copy is not.
        // Stop at the landed prefix; the caller still holds every destination.
        break;
      }
      ++source->pool->slots[source->slot].pins;
      ++source->pool->busy;
      ++pool->busy;
      ++pool->slots[dst].writers;
      steps.push_back(CopyStep{source->pool, source->slot, dst, i});
      result.slots.push_back(dst);
      continue;
    }
    auto local = pool->index.find(key);
    if (local != pool->index.end() && pool->slots[local->second].writers == 0) {
      // Pinned so a later destination in this same call cannot evict it.
      ++pool->slots[local->second].pins;
      local_pins.push_back(local->second);
      result.slots.push_back(local->second);
      continue;
    }
    const Holder* source = nullptr;
    auto holders = group.holders.find(key);
    if (holders != group.holders.end()) {
      for (const Holder& h : holders->second) {
        // Not this pool, and not a slot its owner may still be writing.
        if (h.pool != pool && !h.pool->quarantined && h.pool->slots[h.slot].writers == 0 &&
            Readable(*h.pool, h.pool->slots[h.slot])) {
          source = &h;
          break;
        }
      }
    }
    if (source == nullptr) break;
    uint32_t dst = 0;
    if (!TakeFreeLocked(pool, &dst, &result.evicted_keys, /*settled_only=*/true)) break;
    pool->slots[dst].state = SlotState::kFilling;
    ++source->pool->slots[source->slot].pins;
    ++source->pool->busy;
    ++pool->busy;
    steps.push_back(CopyStep{source->pool, source->slot, dst, i});
    result.slots.push_back(dst);
  }

  // Copy without the lock. Pool pointers stay valid: Detach, Open and a
  // re-bind all wait for `busy` to reach zero.
  lock.unlock();
  const auto started = std::chrono::steady_clock::now();
  std::vector<bool> copied(steps.size(), false);
  std::string copy_error;
  std::vector<MemoryApi::Fragment> fragments;
  size_t run_start = 0;
  bool failed = false;
  bool drained = true;
  while (run_start < steps.size()) {
    size_t run_end = run_start + 1;
    while (run_end < steps.size() && steps[run_end].src == steps[run_start].src &&
           steps[run_end].src_slot == steps[run_end - 1].src_slot + 1 &&
           steps[run_end].dst_slot == steps[run_end - 1].dst_slot + 1) {
      ++run_end;
    }
    const CopyStep& first = steps[run_start];
    const uint64_t run_slots = run_end - run_start;
    for (size_t r = 0; r < pool->layout.regions.size(); ++r) {
      const Region& dst_region = pool->layout.regions[r];
      const Region& src_region = first.src->layout.regions[r];
      const uint64_t unit = dst_region.unit_bytes;
      void* dst =
          static_cast<char*>(pool->base) + dst_region.offset + uint64_t{first.dst_slot} * unit;
      const void* src = static_cast<const char*>(first.src->base) + src_region.offset +
                        uint64_t{first.src_slot} * unit;
      fragments.push_back(MemoryApi::Fragment{dst, src, run_slots * unit, first.src->device});
      result.copied_bytes += run_slots * unit;
    }
    run_start = run_end;
  }
  // One batch for the whole call, landed before it returns: nothing is
  // handed to the requester before then.
  if (!fragments.empty()) {
    CopyOutcome outcome;
    try {
      outcome = api_->CopyBatchTracked(pool->device, fragments);
    } catch (const std::exception& e) {
      outcome = {false, false, e.what()};
    } catch (...) {
      outcome = {false, false, "copy backend threw without a completion proof"};
    }
    failed = !outcome.ok;
    drained = outcome.drained;
    copy_error = outcome.error;
    if (failed) result.copied_bytes = 0;
    std::fill(copied.begin(), copied.end(), !failed);
  }
  result.copy_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  lock.lock();

  if (!drained) {
    pool->quarantined = true;
    result.completion_unknown = true;
    result.status = Status::Error("copy completion unknown: " + copy_error);
    result.slots.clear();
    // Keep every source pin, destination writer and pool busy reference.
    idle_.notify_all();
    return result;
  }

  // Commit: a copied slot becomes cached content of this pool.
  size_t valid_keys = result.slots.size();
  std::vector<std::pair<std::string, uint32_t>> sources;  // runs, for the log
  for (size_t i = 0; i < steps.size(); ++i) {
    const CopyStep& step = steps[i];
    --step.src->slots[step.src_slot].pins;
    --step.src->busy;
    --pool->busy;
    Slot& d = pool->slots[step.dst_slot];
    const std::string& key = keys[step.key_index];
    const bool usable = copied[i] && step.key_index < valid_keys;
    if (into) {
      --d.writers;
      // The caller's slot, which it publishes itself: only the count moves.
      if (usable) {
        ++step.src->served_out_slots;
        ++pool->fetched_in_slots;
        ++result.copied_slots;
        if (sources.empty() || sources.back().first != step.src->id) {
          sources.emplace_back(step.src->id, 0u);
        }
        ++sources.back().second;
      } else {
        valid_keys = std::min(valid_keys, step.key_index);
      }
      continue;
    }
    auto existing = pool->index.find(key);
    if (usable && existing == pool->index.end()) {
      d.has_key = true;
      d.key = key;
      // The copy has landed and the owner has never used this content.
      d.written = 0;
      d.released = 0;
      d.state = SlotState::kCached;
      d.lru_it = pool->lru.insert(pool->lru.end(), step.dst_slot);
      pool->index[key] = step.dst_slot;
      group.holders[key].push_back(Holder{pool, step.dst_slot});
      HolderAddedLocked(&group, key);
      ++step.src->served_out_slots;
      ++pool->fetched_in_slots;
      ++result.copied_slots;
      if (sources.empty() || sources.back().first != step.src->id) {
        sources.emplace_back(step.src->id, 0u);
      }
      ++sources.back().second;
      continue;
    }
    d.state = SlotState::kVacant;
    pool->vacant.insert(step.dst_slot);
    if (usable) {
      // Published here while the copy ran; that slot is the one to report.
      result.slots[step.key_index] = existing->second;
    } else {
      valid_keys = std::min(valid_keys, step.key_index);
    }
  }
  for (uint32_t slot : local_pins) --pool->slots[slot].pins;
  --pool->busy;
  result.slots.resize(valid_keys);
  pool->fetched_in_bytes += result.copied_bytes;
  pool->copy_micros += result.copy_micros;
  idle_.notify_all();

  if (failed) {
    MORI_UMBP_ERROR("[L1Pool] fetch into pool={} stopped after {} of {} keys: {}", pool_id,
                    valid_keys, keys.size(), copy_error);
  } else if (result.copied_slots != 0) {
    std::string from;
    for (const auto& source : sources) {
      from += (from.empty() ? "" : ",") + source.first + "x" + std::to_string(source.second);
    }
    MORI_UMBP_INFO(
        "[L1Pool] fetch pool={} keys={} resident={} copied_slots={} bytes={} micros={} from={}",
        pool_id, keys.size(), result.slots.size(), result.copied_slots, result.copied_bytes,
        result.copy_micros, from);
  }
  return result;
}

Status PoolManager::Clear(const std::string& pool_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  Status idle = WaitIdleLocked(lock, pool_id, "clear");
  if (!idle.ok) return idle;
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    return Status::Error("pool " + pool_id + " is not attached and bound");
  }
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) {
    Slot& s = pool->slots[slot];
    UnindexLocked(pool, slot);
    if (s.state == SlotState::kCached) {
      UncacheLocked(pool, slot);
      s.state = SlotState::kVacant;
      pool->vacant.insert(slot);
    }
  }
  return Status{};
}

Status PoolManager::Detach(const std::string& pool_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (attaching_.count(pool_id) != 0) {
    return Status::Error("detach: pool attachment is in progress; retain the allocation");
  }
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status{};
  if (pool->quarantined) return Status::Error("pool is quarantined; completion is unknown");
  return DestroyLocked(lock, pool);
}

std::vector<PoolStats> PoolManager::Stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PoolStats> out;
  out.reserve(pools_.size());
  for (const auto& entry : pools_) {
    const Pool& pool = *entry.second;
    PoolStats stats;
    stats.pool_id = pool.id;
    stats.group = pool.group;
    stats.device = pool.device;
    stats.bytes = pool.bytes;
    stats.num_slots = static_cast<uint32_t>(pool.slots.size());
    stats.held = pool.held;
    stats.cached = static_cast<uint32_t>(pool.lru.size() + pool.kept.size());
    stats.vacant = static_cast<uint32_t>(pool.vacant.size());
    stats.evicted = pool.evicted;
    stats.fetched_in_slots = pool.fetched_in_slots;
    stats.fetched_in_bytes = pool.fetched_in_bytes;
    stats.served_out_slots = pool.served_out_slots;
    stats.copy_micros = pool.copy_micros;
    out.push_back(std::move(stats));
  }
  std::sort(out.begin(), out.end(),
            [](const PoolStats& a, const PoolStats& b) { return a.pool_id < b.pool_id; });
  return out;
}

void PoolManager::ConfigurePeers(std::vector<std::shared_ptr<PeerApi>> peers) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pools_.empty()) throw std::logic_error("configure peers before attaching pools");
  if (!peers.empty() && !api_->SupportsFabric()) {
    throw std::invalid_argument("native L1 peers require direct FABRIC memory");
  }
  if (peers.size() > 8 ||
      std::any_of(peers.begin(), peers.end(), [](const auto& p) { return p == nullptr; })) {
    throw std::invalid_argument("invalid native L1 peer list");
  }
  peers_ = std::move(peers);
}

const PoolManager::Holder* PoolManager::PeerSourceLocked(const std::string& group,
                                                         const std::string& key) {
  auto g = groups_.find(group);
  if (g == groups_.end()) return nullptr;
  auto found = g->second.holders.find(key);
  if (found == g->second.holders.end()) return nullptr;
  for (const Holder& h : found->second) {
    const Pool& p = *h.pool;
    if (p.bound && !p.quarantined && p.external_kind == "hip_fabric" &&
        p.slots[h.slot].writers == 0 && Readable(p, p.slots[h.slot]))
      return &h;
  }
  return nullptr;
}

PeerProbeResult PoolManager::Probe(const std::string& group, const std::vector<std::string>& keys,
                                   const std::vector<uint64_t>& unit_bytes) {
  PeerProbeResult result;
  result.incarnation = incarnation_;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ValidPeerShape(keys, unit_bytes)) {
    result.status = Status::Error("invalid or oversized native L1 peer request");
    return result;
  }
  auto g = groups_.find(group);
  if (g == groups_.end()) return result;
  if (g->second.unit_bytes != unit_bytes) {
    result.status = Status::Error("native L1 peer layout does not match group");
    return result;
  }
  for (const std::string& key : keys) {
    if (PeerSourceLocked(group, key) == nullptr) break;
    ++result.available;
  }
  return result;
}

PeerReadResult PoolManager::AcquireRead(const PeerReadRequest& request) {
  PeerReadResult result;
  result.incarnation = incarnation_;
  std::lock_guard<std::mutex> lock(mutex_);
  if (request.expected_incarnation != incarnation_ || request.operation_id.empty() ||
      request.operation_id.size() > 256 || !ValidPeerShape(request.keys, request.unit_bytes)) {
    result.status = Status::Error("invalid native L1 read or stale owner incarnation");
    return result;
  }
  auto old = read_grants_.find(request.operation_id);
  if (old != read_grants_.end()) {
    const ReadGrant& grant = old->second;
    if (grant.released) {
      result.status = Status::Error("native L1 read operation was already released");
      return result;
    }
    if (grant.request.group != request.group || grant.request.keys != request.keys ||
        grant.request.unit_bytes != request.unit_bytes) {
      result.status = Status::Error("native L1 operation id reused for a different request");
      return result;
    }
    return grant.result;
  }
  if (read_grants_.size() >= kPeerReceipts) {
    result.status = Status::Error("native L1 read receipt budget exhausted; drain before restart");
    return result;
  }
  ReadGrant grant;
  grant.request = request;
  auto group = groups_.find(request.group);
  if (group != groups_.end() && group->second.unit_bytes != request.unit_bytes) {
    result.status = Status::Error("native L1 peer layout does not match group");
    return result;
  }
  // Lookup and pin occur under the same mutex; Probe was only a hint. Pool
  // replacement/rebinding waits on busy, so this grant needs no cached locator.
  for (const std::string& key : request.keys) {
    const Holder* source = PeerSourceLocked(request.group, key);
    if (source == nullptr) break;
    const Pool& p = *source->pool;
    for (const Region& r : p.layout.regions) {
      result.regions.push_back(PeerReadRegion{
          p.external_handle, p.external_offset + r.offset + uint64_t{source->slot} * r.unit_bytes,
          r.unit_bytes});
    }
    ++source->pool->slots[source->slot].pins;
    ++source->pool->busy;
    grant.sources.push_back(*source);
    ++result.available;
  }
  grant.result = result;
  read_grants_.emplace(request.operation_id, std::move(grant));
  return result;
}

Status PoolManager::ReleaseRead(const std::string& operation_id,
                                const std::string& expected_incarnation, bool drained) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (expected_incarnation != incarnation_ || operation_id.empty() || operation_id.size() > 256 ||
      !drained) {
    return Status::Error("read release requires matching owner and confirmed drain");
  }
  auto found = read_grants_.find(operation_id);
  if (found == read_grants_.end()) {
    // Release may beat a delayed Acquire whose reply was lost. Close the id
    // before acknowledging; it may never acquire pins after this response.
    if (read_grants_.size() >= kPeerReceipts) {
      return Status::Error("native L1 read receipt budget exhausted");
    }
    ReadGrant closed;
    closed.released = true;
    read_grants_.emplace(operation_id, std::move(closed));
    return {};
  }
  ReadGrant& grant = found->second;
  if (grant.released) return {};
  for (const Holder& source : grant.sources) {
    --source.pool->slots[source.slot].pins;
    --source.pool->busy;
  }
  grant = ReadGrant{};
  grant.released = true;
  idle_.notify_all();
  return {};
}

FetchResult PoolManager::Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                               const FetchOptions& options) {
  // Reject overlapping caller destinations before either local or remote work.
  if (!options.probe_only && !options.dst_slots.empty()) {
    std::unordered_set<uint32_t> unique;
    if (options.dst_slots.size() < keys.size()) {
      FetchResult bad;
      bad.status = Status::Error("fewer destination slots than keys");
      return bad;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!unique.insert(options.dst_slots[i]).second) {
        FetchResult bad;
        bad.status = Status::Error("overlapping native L1 destination slots");
        return bad;
      }
    }
  }
  std::unique_lock<std::mutex> lock(mutex_);
  // Snapshot configuration under the same lock as ConfigurePeers.
  const auto peers = peers_;
  if (peers.empty() || (!options.probe_only && options.dst_slots.empty()) || keys.empty()) {
    lock.unlock();
    return FetchLocal(pool_id, keys, options);
  }
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    FetchResult bad;
    bad.status = Status::Error("native L1 pool is unavailable or quarantined");
    return bad;
  }
  const std::string group = pool->group;
  const std::vector<uint64_t> units = groups_[group].unit_bytes;
  ++pool->busy;  // Protect destination allocation/layout across local work and RPC.
  lock.unlock();
  FetchResult result = FetchLocal(pool_id, keys, options);
  size_t leading = options.probe_only ? result.available : result.slots.size();
  auto finish = [&] {
    std::lock_guard<std::mutex> done(mutex_);
    --pool->busy;
    idle_.notify_all();
  };
  if (!result.status.ok || leading == keys.size()) {
    finish();
    return result;
  }
  const size_t count = std::min(kPeerKeys, keys.size() - leading);
  std::vector<std::string> suffix(keys.begin() + leading, keys.begin() + leading + count);
  if (!ValidPeerShape(suffix, units)) {
    finish();
    return result;
  }
  // Use one peer per suffix. No metadata cache: resolve and pin again for every
  // Fetch, preserving owner-local timestamp semantics across different hosts.
  std::shared_ptr<PeerApi> chosen;
  PeerProbeResult best;
  for (const auto& peer : peers) {
    PeerProbeResult probe = peer->Probe(group, suffix, units);
    if (probe.status.ok && !probe.incarnation.empty() && probe.available <= count &&
        probe.available > best.available) {
      chosen = peer;
      best = std::move(probe);
    }
  }
  if (!chosen) {
    finish();
    return result;
  }
  if (options.probe_only) {
    result.available += best.available;
    finish();
    return result;
  }
  lock.lock();
  size_t reserved = 0;
  for (; reserved < best.available; ++reserved) {
    const uint32_t id = options.dst_slots[leading + reserved];
    if (id >= pool->slots.size() || pool->slots[id].state != SlotState::kHeld ||
        pool->slots[id].has_key ||
        pool->slots[id].writers != 0 || pool->slots[id].pins != 0 ||
        !Fillable(*pool, pool->slots[id]))
      break;
    ++pool->slots[id].writers;
  }
  const std::string operation = incarnation_ + std::to_string(++next_read_);
  lock.unlock();
  if (reserved == 0) {
    finish();
    return result;
  }
  suffix.resize(reserved);
  PeerReadRequest request{operation, best.incarnation, group, suffix, units};
  PeerReadResult read = chosen->AcquireRead(request);
  std::vector<void*> mappings;
  std::vector<MemoryApi::Fragment> fragments;
  std::string error;
  bool valid = read.status.ok && read.incarnation == best.incarnation &&
               read.available <= reserved && read.regions.size() == read.available * units.size();
  uint64_t bytes = 0;
  if (valid) {
    // Descriptor imports are reused within this call only. Mapping a descriptor
    // creates an address view, not a payload allocation/copy or a staging pool.
    std::unordered_map<std::string, void*> bases;
    for (size_t i = 0; i < read.regions.size(); ++i) {
      const PeerReadRegion& region = read.regions[i];
      if (region.bytes != units[i % units.size()] ||
          !ValidateFabricRange(region.handle, region.offset, region.bytes, &error)) {
        valid = false;
        break;
      }
      auto entry = bases.find(region.handle);
      if (entry == bases.end()) {
        void* base = nullptr;
        void* mapped = nullptr;
        if (!api_->Import(pool->device, region.handle, 0, &base, &mapped, &error)) {
          valid = false;
          break;
        }
        mappings.push_back(mapped);
        entry = bases.emplace(region.handle, base).first;
      }
      const uint32_t dst = options.dst_slots[leading + i / units.size()];
      const Region& layout = pool->layout.regions[i % units.size()];
      fragments.push_back(MemoryApi::Fragment{
          static_cast<char*>(pool->base) + layout.offset + uint64_t{dst} * layout.unit_bytes,
          static_cast<const char*>(entry->second) + region.offset, region.bytes, pool->device});
      bytes += region.bytes;
    }
  }
  const auto started = std::chrono::steady_clock::now();
  CopyOutcome outcome{false, true, "remote read unavailable: " + error};
  if (valid) {
    try {
      outcome = api_->CopyBatchTracked(pool->device, fragments);
    } catch (const std::exception& e) {
      outcome = {false, false, e.what()};
    } catch (...) {
      outcome = {false, false, "copy backend threw without a completion proof"};
    }
  }
  const auto micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - started)
                                                .count());
  if (!outcome.drained) {
    lock.lock();
    pool->quarantined = true;
    result.status = Status::Error("FABRIC copy completion unknown: " + outcome.error);
    result.completion_unknown = true;
    result.slots.clear();
    // Keep the remote source pin, mappings, destination writers and busy guard.
    // No fallback or recomputation into these destinations is permitted.
    idle_.notify_all();
    return result;
  }
  for (void* mapped : mappings) api_->Unimport(pool->device, mapped);
  // Even a lost Acquire reply can have pinned an owner slot. No submission or
  // confirmed drain makes a release safe; owner tombstones fence delayed RPCs.
  Status release = chosen->ReleaseRead(operation, best.incarnation, true);
  if (!release.ok) {
    MORI_UMBP_WARN(
        "[L1Peer] release unconfirmed operation={} owner={} error={}; source stays pinned",
        operation, best.incarnation, release.error);
  }
  lock.lock();
  for (size_t i = 0; i < reserved; ++i) --pool->slots[options.dst_slots[leading + i]].writers;
  if (outcome.ok) {
    for (size_t i = 0; i < read.available; ++i)
      result.slots.push_back(options.dst_slots[leading + i]);
    result.copied_slots += read.available;
    result.copied_bytes += bytes;
    result.copy_micros += micros;
    pool->fetched_in_slots += read.available;
    pool->fetched_in_bytes += bytes;
    pool->copy_micros += micros;
    MORI_UMBP_INFO("[L1Peer] fabric completed pool={} owner={} operation={} slots={} bytes={}",
                   pool_id, best.incarnation, operation, read.available, bytes);
  }
  --pool->busy;
  idle_.notify_all();
  return result;
}

}  // namespace mori::umbp::l1
