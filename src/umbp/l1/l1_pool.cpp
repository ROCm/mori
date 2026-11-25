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
#include <shared_mutex>
#include <utility>

#include "mori/utils/mori_log.hpp"

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
        // The fragments already queued are still copying. The caller gives up
        // their sources and destinations as soon as this returns.
        (void)hipStreamSynchronize(stream);
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
  // A failed creation is not remembered, so the next copy tries again.
  hipStream_t StreamFor(int device) {
    std::lock_guard<std::mutex> lock(table_mutex_);
    auto found = streams_.find(device);
    if (found != streams_.end()) return found->second;
    hipStream_t stream = nullptr;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) return nullptr;
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

}  // namespace

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api) : api_(std::move(api)) {
  // On by default: with ranks copying each other's prefixes, plain LRU evicts
  // a conversation's only up-to-date copy while older duplicates of it stay.
  SetEvictionPolicy(EnvCount("UMBP_L1_KEEP_COPIES", 1));
  settled_first_ = EnvCount("UMBP_L1_GRANT_SETTLED_FIRST", 1) != 0;
  MORI_UMBP_INFO("[L1Pool] grants take settled slots first: {}", settled_first_ ? "on" : "off");
}

void PoolManager::SetEvictionPolicy(uint32_t keep_copies) {
  std::lock_guard<std::mutex> lock(mutex_);
  keep_copies_ = keep_copies;
  MORI_UMBP_INFO("[L1Pool] eviction policy: keep_copies={}", keep_copies_);
}

void PoolManager::Holders::push_back(const Holder& holder) {
  if (more_.empty() && size_ < kInline) {
    inline_[size_++] = holder;
    return;
  }
  if (more_.empty()) more_.assign(inline_, inline_ + size_);
  more_.push_back(holder);
  ++size_;
}

void PoolManager::Holders::erase(size_t i) {
  std::move(begin() + i + 1, end(), begin() + i);
  --size_;
  if (!more_.empty()) more_.pop_back();
}

void PoolManager::Holders::move_to_back(size_t i) {
  std::rotate(begin() + i, begin() + i + 1, end());
}

size_t PoolManager::Holders::find(const Pool* pool) const {
  for (size_t i = 0; i < size_; ++i) {
    if ((*this)[i].pool == pool) return i;
  }
  return size_;
}

void PoolManager::PushBack(Pool* pool, SlotList* list, uint32_t slot) {
  Slot& s = pool->slots[slot];
  s.prev = list->tail;
  s.next = kNoSlot;
  if (list->tail != kNoSlot) {
    pool->slots[list->tail].next = slot;
  } else {
    list->head = slot;
  }
  list->tail = slot;
  ++list->size;
}

void PoolManager::PushFront(Pool* pool, SlotList* list, uint32_t slot) {
  Slot& s = pool->slots[slot];
  s.prev = kNoSlot;
  s.next = list->head;
  if (list->head != kNoSlot) {
    pool->slots[list->head].prev = slot;
  } else {
    list->tail = slot;
  }
  list->head = slot;
  ++list->size;
}

void PoolManager::Unlink(Pool* pool, SlotList* list, uint32_t slot) {
  Slot& s = pool->slots[slot];
  if (s.prev != kNoSlot) {
    pool->slots[s.prev].next = s.next;
  } else {
    list->head = s.next;
  }
  if (s.next != kNoSlot) {
    pool->slots[s.next].prev = s.prev;
  } else {
    list->tail = s.prev;
  }
  s.prev = s.next = kNoSlot;
  --list->size;
}

void PoolManager::SetGrantSettledFirst(bool on) {
  std::lock_guard<std::mutex> lock(mutex_);
  settled_first_ = on;
}

void PoolManager::UncacheLocked(Pool* pool, uint32_t slot) {
  Slot& s = pool->slots[slot];
  Unlink(pool, s.kept ? &pool->kept : &pool->lru, slot);
  s.kept = false;
}

bool PoolManager::IsSpareLocked(const Pool* pool, uint32_t slot) const {
  const Slot& s = pool->slots[slot];
  if (s.content == nullptr) return true;
  // Holders stand oldest first (`HolderAddedLocked`, `HolderUsedLocked`): the
  // last `keep_copies_` are the copies worth keeping, every older one is spare.
  const Holders& list = s.content->second;
  if (list.size() <= keep_copies_) return false;
  for (size_t i = list.size() - keep_copies_; i < list.size(); ++i) {
    if (list[i].pool == pool && list[i].slot == slot) return false;
  }
  return true;
}

void PoolManager::HolderAddedLocked(Holders* holders) {
  if (keep_copies_ == 0 || holders->size() <= keep_copies_) return;
  // The newest holder stands last. The one it pushed out of the newest
  // `keep_copies_` is spare from now on: if its pool had set it aside as a copy
  // to keep, it is that pool's coldest spare copy.
  const Holder& h = (*holders)[holders->size() - 1 - keep_copies_];
  Slot& s = h.pool->slots[h.slot];
  if (s.state == SlotState::kCached && s.kept) {
    Unlink(h.pool, &h.pool->kept, h.slot);
    PushFront(h.pool, &h.pool->lru, h.slot);
    s.kept = false;
  }
}

void PoolManager::HolderUsedLocked(Pool* pool, uint32_t slot) {
  if (keep_copies_ == 0) return;
  const Slot& s = pool->slots[slot];
  if (s.content == nullptr) return;
  Holders& list = s.content->second;
  if (list.size() < 2) return;
  size_t mine = 0;
  while (mine < list.size() && !(list[mine].pool == pool && list[mine].slot == slot)) ++mine;
  if (mine + 1 >= list.size()) return;
  // The copy an engine has just taken for a request is the newest one.
  list.move_to_back(mine);
  HolderAddedLocked(&list);
}

PoolManager::~PoolManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) ReleaseMemoryLocked(entry.second.get());
  pools_.clear();
}

void PoolManager::ReleaseMemoryLocked(Pool* pool) {
  if (pool->external) {
    api_->Unimport(pool->device, pool->mapped);
  } else {
    api_->Free(pool->device, pool->base, pool->bytes);
  }
}

PoolManager::Pool* PoolManager::FindLocked(const std::string& pool_id) {
  auto it = pools_.find(pool_id);
  return it == pools_.end() ? nullptr : it->second.get();
}

void PoolManager::UnindexLocked(Pool* pool, uint32_t slot) {
  Slot& s = pool->slots[slot];
  if (s.content == nullptr) return;
  Holders& list = s.content->second;
  for (size_t i = 0; i < list.size(); ++i) {
    if (list[i].pool == pool && list[i].slot == slot) {
      list.erase(i);
      break;
    }
  }
  if (list.empty()) {
    // By iterator: the key passed to a lookup must outlive it.
    Content& content = pool->content_group->content;
    content.erase(content.find(s.content->first));
  }
  s.content = nullptr;
}

void PoolManager::ResetSlotsLocked(Pool* pool) {
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  pool->lru = SlotList{};
  pool->kept = SlotList{};
  pool->vacant.clear();
  pool->unsettled.clear();
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) {
    pool->slots[slot] = Slot{};
    pool->vacant.insert(slot);
  }
  pool->held = 0;
  pool->committed = 0;
}

void PoolManager::VacateLocked(Pool* pool, uint32_t slot) {
  Slot& s = pool->slots[slot];
  s.state = SlotState::kVacant;
  // A pool whose owner has never reported progress has no settled slots at
  // all: nothing in it can be filled from outside, so nothing is held back.
  if (settled_first_ && pool->committed != 0 && !Fillable(*pool, s)) {
    pool->unsettled.emplace(std::max(s.released, s.written), slot);
  } else {
    pool->vacant.insert(slot);
  }
}

void PoolManager::UnvacateLocked(Pool* pool, uint32_t slot) {
  if (pool->vacant.erase(slot) != 0) return;
  const Slot& s = pool->slots[slot];
  pool->unsettled.erase({std::max(s.released, s.written), slot});
}

bool PoolManager::TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                                 bool settled_only) {
  // Vacant slots whose owner's writes have landed since they were released.
  while (!pool->unsettled.empty() && pool->unsettled.begin()->first <= pool->committed) {
    pool->vacant.insert(pool->unsettled.begin()->second);
    pool->unsettled.erase(pool->unsettled.begin());
  }
  for (auto it = pool->vacant.begin(); it != pool->vacant.end(); ++it) {
    if (pool->slots[*it].pins != 0) continue;
    if (settled_only && !Fillable(*pool, pool->slots[*it])) continue;
    *slot = *it;
    pool->vacant.erase(it);
    return true;
  }
  // A slot its owner released a moment ago is the last one to hand out: what
  // is put there from outside -- a fetch into it -- has to wait for the
  // owner's device, a forward or two, and holds up everything fetched with
  // it. It is settled, and first in line, by the time of the next grants.
  const auto take_unsettled = [&]() {
    if (settled_only) return false;
    for (auto it = pool->unsettled.begin(); it != pool->unsettled.end(); ++it) {
      if (pool->slots[it->second].pins != 0) continue;
      *slot = it->second;
      pool->unsettled.erase(it);
      return true;
    }
    return false;
  };
  // Asked ahead of need: nothing is evicted, and nothing unsettled given.
  if (evicted == nullptr) return false;
  const auto takeable = [&](const Slot& s) {
    return s.pins == 0 && (!settled_only || Fillable(*pool, s));
  };
  // How many pools of the group hold `s`'s key, this one included.
  const auto copies = [&](const Slot& s) -> size_t {
    return s.content == nullptr ? 0 : s.content->second.size();
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
  for (uint32_t it = pool->lru.head; it != kNoSlot;) {
    Slot& s = pool->slots[it];
    const uint32_t next = s.next;
    if (!takeable(s)) {
      it = next;
      continue;
    }
    if (keep_copies_ == 0 || IsSpareLocked(pool, it)) {
      victim = it;
      found = true;
      break;
    }
    Unlink(pool, &pool->lru, it);
    PushBack(pool, &pool->kept, it);
    s.kept = true;
    it = next;
  }
  if (!found) {
    for (uint32_t candidate = pool->kept.head; candidate != kNoSlot;
         candidate = pool->slots[candidate].next) {
      if (takeable(pool->slots[candidate])) {
        victim = candidate;
        found = true;
        break;
      }
    }
  }
  if (!found) return take_unsettled();
  Slot& s = pool->slots[victim];
  *slot = victim;
  UncacheLocked(pool, victim);
  if (s.content != nullptr) {
    if (copies(s) > 1) ++pool->evicted_spare;
    evicted->push_back(s.content->first);
    UnindexLocked(pool, *slot);
    if (++pool->evicted % kEvictionLogEvery == 0) {
      MORI_UMBP_INFO(
          "[L1Pool] evictions pool={} total={} spare_copy={} cached={} last_copies_kept={} "
          "vacant={}",
          pool->id, pool->evicted, pool->evicted_spare, pool->lru.size + pool->kept.size,
          pool->kept.size, pool->vacant.size() + pool->unsettled.size());
    }
  }
  return true;
}

bool PoolManager::HasPinnedFreeLocked(const Pool* pool) const {
  for (uint32_t slot : pool->vacant) {
    if (pool->slots[slot].pins != 0) return true;
  }
  for (const auto& entry : pool->unsettled) {
    if (pool->slots[entry.second].pins != 0) return true;
  }
  for (const SlotList* list : {&pool->lru, &pool->kept}) {
    for (uint32_t slot = list->head; slot != kNoSlot; slot = pool->slots[slot].next) {
      if (pool->slots[slot].pins != 0) return true;
    }
  }
  return false;
}

void PoolManager::DestroyLocked(std::unique_lock<std::mutex>& lock, Pool* pool) {
  const std::string pool_id = pool->id;
  const uint64_t serial = pool->serial;
  idle_.wait(lock, [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->serial != serial || p->busy == 0;
  });
  pool = FindLocked(pool_id);
  // Someone else destroyed it during the wait, and may have attached a new
  // pool under the id: that one is not this caller's to destroy.
  if (pool == nullptr || pool->serial != serial) return;
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  ReleaseMemoryLocked(pool);
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} device={} bytes={}{} evicted={} spare_copy={}",
                 pool_id, pool->device, pool->bytes, pool->external ? " (adopted, unmapped)" : "",
                 pool->evicted, pool->evicted_spare);
  pools_.erase(pool_id);
}

AttachResult PoolManager::Attach(const std::string& pool_id, const std::string& group, int device,
                                 uint64_t bytes, const ExternalMemory& external) {
  const bool adopt = !external.handle.empty();
  AttachResult result;
  if (pool_id.empty() || bytes == 0 || device < 0) {
    result.status = Status::Error("attach needs a pool id, a device and a non-zero size");
    return result;
  }
  const uint64_t offset = adopt ? external.offset : 0;
  std::unique_lock<std::mutex> lock(mutex_);
  // DestroyLocked gives up the lock while the pool's copies finish, so another
  // Attach of this id may have replaced the pool, or be allocating it, by the
  // time it returns: look again each time round.
  for (;;) {
    if (attaching_.count(pool_id) != 0) {
      result.status = Status::Error("pool " + pool_id + " is already being attached");
      return result;
    }
    Pool* existing = FindLocked(pool_id);
    if (existing == nullptr) break;
    if (existing->group == group && existing->device == device && existing->bytes == bytes &&
        existing->external == adopt && existing->external_handle == external.handle &&
        existing->external_offset == offset) {
      result.memory = existing->exported;
      result.bytes = existing->bytes;
      result.created = false;
      return result;
    }
    DestroyLocked(lock, existing);
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
  pool->serial = next_serial_++;
  pool->group = group;
  pool->device = device;
  pool->base = base;
  pool->bytes = bytes;
  pool->exported = exported;
  pool->external = adopt;
  pool->mapped = mapped;
  pool->external_handle = external.handle;
  pool->external_offset = offset;
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
  std::vector<uint64_t> units;
  for (const Region& region : layout.regions) {
    // Divided rather than multiplied out: offset + slots * unit can wrap.
    if (region.unit_bytes == 0 || region.offset > pool->bytes ||
        region.unit_bytes > (pool->bytes - region.offset) / layout.num_slots) {
      return Status::Error("a region does not fit in pool " + pool_id);
    }
    units.push_back(region.unit_bytes);
  }
  if (pool->bound && SameLayout(pool->layout, layout)) return Status{};

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
  if (pool->bound) {
    idle_.wait(lock, [&] {
      Pool* p = FindLocked(pool_id);
      return p == nullptr || p->busy == 0;
    });
    pool = FindLocked(pool_id);
    if (pool == nullptr) return Status::Error("pool " + pool_id + " was detached");
    ResetSlotsLocked(pool);
  }
  Group& current = groups_[pool->group];
  current.has_layout = true;
  current.unit_bytes = std::move(units);
  // Room for every slot's key now, so the index never rehashes -- a pause of
  // tens of milliseconds at this size -- while engines are being served.
  if (pool->bound) current.slots -= pool->layout.num_slots;
  current.slots += layout.num_slots;
  current.content.reserve(current.slots);
  pool->content_group = &current;
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
  idle_.wait(lock, [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->busy == 0;
  });
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
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
  if (pool == nullptr || !pool->bound) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
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
    if (pool == nullptr || !pool->bound) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      return result;
    }
  }
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
  if (pool == nullptr || !pool->bound) {
    return Status::Error("pool " + pool_id + " is not attached and bound");
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
        if (s.state == SlotState::kCached) {
          UncacheLocked(pool, op.slot);
          HolderUsedLocked(pool, op.slot);
        } else {
          UnvacateLocked(pool, op.slot);
        }
        s.state = SlotState::kHeld;
        ++pool->held;
        break;
      case OpKind::kRelease:
        if (s.state != SlotState::kHeld) return fail("release of a slot that is not held");
        --pool->held;
        s.released = stamp;
        if (s.content != nullptr) {
          s.state = SlotState::kCached;
          PushBack(pool, &pool->lru, op.slot);
        } else {
          VacateLocked(pool, op.slot);
        }
        break;
      case OpKind::kPublish: {
        if (s.state != SlotState::kHeld) return fail("publish of a slot that is not held");
        if (op.key.empty()) return fail("publish without a key");
        if (s.content != nullptr && s.content->first == op.key) break;
        UnindexLocked(pool, op.slot);
        // One lookup: the key's entry, made if this is its first holder.
        ContentEntry& entry = *pool->content_group->content.try_emplace(op.key).first;
        Holders& holders = entry.second;
        const size_t previous = holders.find(pool);
        if (previous != holders.size()) {
          // The same content, now named by this slot. The older copy stops
          // being reachable; if nobody holds it, it is simply free.
          const uint32_t old_slot = holders[previous].slot;
          Slot& old = pool->slots[old_slot];
          holders.erase(previous);
          old.content = nullptr;
          old.released = stamp;
          if (old.state == SlotState::kCached) {
            UncacheLocked(pool, old_slot);
            VacateLocked(pool, old_slot);
          }
        }
        s.content = &entry;
        s.written = stamp;
        holders.push_back(Holder{pool, op.slot});
        HolderAddedLocked(&holders);
        break;
      }
      default:
        return fail("unknown op");
    }
  }
  return Status{};
}

FetchResult PoolManager::Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                               const FetchOptions& options) {
  FetchResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  // Read again after any wait: the pool may have been replaced meanwhile.
  Content* content = &pool->content_group->content;
  // Who holds `key`, or null: one lookup answers both questions below.
  auto holders_of = [&](const std::string& key) -> const Holders* {
    auto entry = content->find(key);
    return entry == content->end() ? nullptr : &entry->second;
  };
  // This pool's slot for the key, or kNoSlot.
  auto local_of = [&](const Holders* holders) -> uint32_t {
    if (holders == nullptr) return kNoSlot;
    const size_t mine = holders->find(pool);
    return mine == holders->size() ? kNoSlot : (*holders)[mine].slot;
  };
  // Another pool's settled copy, or null.
  auto source_of = [&](const Holders* holders) -> const Holder* {
    if (holders == nullptr) return nullptr;
    for (const Holder& h : *holders) {
      // Not this pool, and not a slot its owner may still be writing.
      if (h.pool != pool && Readable(*h.pool, h.pool->slots[h.slot])) return &h;
    }
    return nullptr;
  };

  if (options.probe_only) {
    for (const std::string& key : keys) {
      const Holders* holders = holders_of(key);
      if (local_of(holders) == kNoSlot && source_of(holders) == nullptr) break;
      ++result.available;
    }
    return result;
  }
  const bool into = !options.dst_slots.empty();
  uint64_t settle_micros = 0;  // waiting for the destinations' owner to commit
  if (into && options.dst_slots.size() < keys.size()) {
    result.status = Status::Error("fetch into pool " + pool_id + ": fewer slots than keys");
    return result;
  }
  if (into) {
    // A regranted slot may still have its owner's writes in flight. They land
    // within a forward or two, and the owner's next Commit says so: wait for
    // that rather than leave the caller to recompute what it asked for. Past
    // the bound, the check below stops at the first unsettled slot.
    const auto settle_started = std::chrono::steady_clock::now();
    const auto settle_deadline = settle_started + kSettleWait;
    idle_.wait_until(lock, settle_deadline, [&] {
      pool = FindLocked(pool_id);
      if (pool == nullptr || !pool->bound) return true;
      for (size_t i = 0; i < keys.size(); ++i) {
        const uint32_t dst = options.dst_slots[i];
        if (dst < pool->slots.size() && !Fillable(*pool, pool->slots[dst])) return false;
      }
      return true;
    });
    if (pool == nullptr || !pool->bound) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      return result;
    }
    content = &pool->content_group->content;
    settle_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - settle_started)
                                              .count());
  }

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists -- or, copying
  // into the caller's slots, check that it holds them.
  std::vector<CopyStep> steps;
  std::vector<uint32_t> local_pins;
  if (!into) {
    // Every key of the call this pool already holds is pinned before any
    // destination is taken: room for an earlier key is never made by evicting
    // a later one.
    for (const std::string& key : keys) {
      const uint32_t local = local_of(holders_of(key));
      if (local == kNoSlot) continue;
      ++pool->slots[local].pins;
      local_pins.push_back(local);
    }
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string& key = keys[i];
    if (into) {
      const uint32_t dst = options.dst_slots[i];
      if (dst >= pool->slots.size() || pool->slots[dst].state != SlotState::kHeld) {
        result.status = Status::Error("fetch into pool " + pool_id + ": slot " +
                                      std::to_string(dst) + " is not held by its owner");
        break;
      }
      const Holder* source = source_of(holders_of(key));
      if (source == nullptr) break;
      if (!Fillable(*pool, pool->slots[dst])) {
        // Allocate may regrant a slot before its owner's writes complete:
        // owner-local reuse is ordered on that stream, but this copy is not.
        // Stop at the landed prefix; the caller still holds every destination.
        break;
      }
      ++source->pool->slots[source->slot].pins;
      ++source->pool->busy;
      ++pool->busy;
      steps.push_back(CopyStep{source->pool, source->slot, dst, i});
      result.slots.push_back(dst);
      continue;
    }
    const Holders* holders = holders_of(key);
    const uint32_t local = local_of(holders);
    if (local != kNoSlot) {
      result.slots.push_back(local);
      continue;
    }
    const Holder* found = source_of(holders);
    if (found == nullptr) break;
    // By value: making room below edits the index.
    const Holder source = *found;
    uint32_t dst = 0;
    if (!TakeFreeLocked(pool, &dst, &result.evicted_keys, /*settled_only=*/true)) break;
    pool->slots[dst].state = SlotState::kFilling;
    ++source.pool->slots[source.slot].pins;
    ++source.pool->busy;
    ++pool->busy;
    steps.push_back(CopyStep{source.pool, source.slot, dst, i});
    result.slots.push_back(dst);
  }

  // Copy without the lock. Pool pointers stay valid: each step holds `busy`
  // on both of its pools, which Detach, Open and a re-bind wait to reach zero.
  // A call with nothing to copy holds no `busy`, so it keeps the lock: without
  // it this pool could be detached, or re-opened with this call's pins out.
  const bool copying = !steps.empty();
  if (copying) lock.unlock();
  const auto started = std::chrono::steady_clock::now();
  std::vector<bool> copied(steps.size(), false);
  std::string copy_error;
  std::vector<MemoryApi::Fragment> fragments;
  size_t run_start = 0;
  bool failed = false;
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
    failed = !api_->CopyBatch(pool->device, fragments, &copy_error);
    if (failed) result.copied_bytes = 0;
    std::fill(copied.begin(), copied.end(), !failed);
  }
  result.copy_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  if (copying) lock.lock();

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
    ContentEntry* entry = nullptr;
    uint32_t existing = kNoSlot;
    if (usable) {
      entry = &*content->try_emplace(key).first;
      existing = local_of(&entry->second);
    }
    if (usable && existing == kNoSlot) {
      d.content = entry;
      // The copy has landed and the owner has never used this content.
      d.written = 0;
      d.released = 0;
      d.state = SlotState::kCached;
      PushBack(pool, &pool->lru, step.dst_slot);
      entry->second.push_back(Holder{pool, step.dst_slot});
      HolderAddedLocked(&entry->second);
      ++step.src->served_out_slots;
      ++pool->fetched_in_slots;
      ++result.copied_slots;
      if (sources.empty() || sources.back().first != step.src->id) {
        sources.emplace_back(step.src->id, 0u);
      }
      ++sources.back().second;
      continue;
    }
    VacateLocked(pool, step.dst_slot);
    if (usable) {
      // Published here while the copy ran; that slot is the one to report.
      result.slots[step.key_index] = existing;
    } else {
      valid_keys = std::min(valid_keys, step.key_index);
    }
  }
  for (uint32_t slot : local_pins) --pool->slots[slot].pins;
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
        "[L1Pool] fetch pool={} keys={} resident={} copied_slots={} bytes={} micros={} "
        "settle_wait_micros={} from={}",
        pool_id, keys.size(), result.slots.size(), result.copied_slots, result.copied_bytes,
        result.copy_micros, settle_micros, from);
  }
  return result;
}

Status PoolManager::Clear(const std::string& pool_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  idle_.wait(lock, [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->busy == 0;
  });
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
    return Status::Error("pool " + pool_id + " is not attached and bound");
  }
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) {
    Slot& s = pool->slots[slot];
    UnindexLocked(pool, slot);
    if (s.state == SlotState::kCached) {
      UncacheLocked(pool, slot);
      VacateLocked(pool, slot);
    }
  }
  return Status{};
}

Status PoolManager::Detach(const std::string& pool_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status{};
  DestroyLocked(lock, pool);
  return Status{};
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
    stats.cached = pool.lru.size + pool.kept.size;
    stats.vacant = static_cast<uint32_t>(pool.vacant.size() + pool.unsettled.size());
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

}  // namespace mori::umbp::l1
