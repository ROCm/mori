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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

double EnvFraction(const char* name, double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  return end == raw ? fallback : std::min(1.0, std::max(0.0, value));
}

// One line per this many evictions of a pool: how much it evicts and how much
// of that another pool still holds.
constexpr uint64_t kEvictionLogEvery = 16384;
// One line per this many slots of a pool written to the backing store.
constexpr uint64_t kStoreLogEvery = 8192;
// Slots one round of writing ahead copies.
constexpr size_t kAheadBatch = 128;
// A store call carries at least this many slots before a write-back or a
// load is split over several.
constexpr size_t kStoreCallSlots = 16;
// How long the writer waits when it found nothing to write, and when the
// store took nothing of what it offered.
constexpr std::chrono::milliseconds kAheadIdle{20};
constexpr std::chrono::milliseconds kAheadRefused{500};
// Cached slots the writer ahead walks before it gives the lock up, and how
// much longer than a walk it then leaves the lock alone: a pool holds
// hundreds of thousands of slots, nearly all of them written already, and
// every call of every engine needs the lock the walk is made under.
constexpr size_t kAheadSlice = 2048;
constexpr int kAheadRest = 9;
// Published slots the writer looks at in one round for ones that have landed,
// and the run of ones that have not at which it stops looking.
constexpr size_t kFreshScan = 8192;
constexpr size_t kFreshUnsettledRun = 64;
// How long the report of a stuck call tries for the pool manager's lock.
constexpr int kLockProbes = 20;
constexpr std::chrono::milliseconds kLockProbeGap{10};

// What tells one cut of a slot from another in the store's key space.
uint64_t HashUnits(uint64_t hash, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    hash ^= (value >> (8 * i)) & 0xFF;
    hash *= 1099511628211ULL;  // FNV-1a
  }
  return hash;
}

}  // namespace

// A store call or a copy between pools, from its start to its return.
class PoolManager::Flight {
 public:
  Flight(PoolManager* manager, const char* what, const std::string& pool, size_t slots,
         uint64_t bytes)
      : manager_(manager), id_(manager->BeginFlight(what, pool, slots, bytes)) {}
  ~Flight() { manager_->EndFlight(id_); }
  Flight(const Flight&) = delete;
  Flight& operator=(const Flight&) = delete;

 private:
  PoolManager* const manager_;
  const uint64_t id_;
};

class PoolManager::CallScope {
 public:
  CallScope(PoolManager* manager, CallKind kind) : count_(manager->calls_[kind]) {
    count_.fetch_add(1, std::memory_order_relaxed);
  }
  ~CallScope() { count_.fetch_sub(1, std::memory_order_relaxed); }
  CallScope(const CallScope&) = delete;
  CallScope& operator=(const CallScope&) = delete;

 private:
  std::atomic<int>& count_;
};

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api) : api_(std::move(api)) {
  // On by default: with ranks copying each other's prefixes, plain LRU evicts
  // a conversation's only up-to-date copy while older duplicates of it stay.
  SetEvictionPolicy(EnvCount("UMBP_L1_KEEP_COPIES", 1));
  settled_first_ = EnvCount("UMBP_L1_GRANT_SETTLED_FIRST", 1) != 0;
  MORI_UMBP_INFO("[L1Pool] grants take settled slots first: {}", settled_first_ ? "on" : "off");
  SetPeerSource(EnvCount("UMBP_L1_PEER_SOURCE", 1) != 0);
  watchdog_ = std::thread(&PoolManager::Watchdog, this);
}

BackingStoreOptions BackingStoreOptionsFromEnv() {
  BackingStoreOptions options;
  options.ahead = EnvFraction("UMBP_L1_DRAM_AHEAD", 0.0);
  options.write_at_eviction = EnvCount("UMBP_L1_DRAM_WRITE_AT_EVICTION", 1) != 0;
  options.threads = std::max<uint32_t>(1, EnvCount("UMBP_L1_DRAM_THREADS", options.threads));
  const char* write = std::getenv("UMBP_L1_DRAM_WRITE");
  if (write != nullptr && *write != '\0') {
    const std::string policy(write);
    if (policy == "publish") {
      options.write_at_publish = true;
    } else if (policy != "ahead") {
      MORI_UMBP_WARN("[L1Pool] UMBP_L1_DRAM_WRITE={} is neither `ahead` nor `publish`: ahead",
                     policy);
    }
  }
  return options;
}

void PoolManager::SetBackingStore(std::shared_ptr<BackingStore> store,
                                  const BackingStoreOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  store_ = std::move(store);
  store_on_ = store_ != nullptr;
  store_options_ = options;
  store_options_.threads = std::max<uint32_t>(1, options.threads);
  if (!store_on_) return;
  MORI_UMBP_INFO(
      "[L1Pool] backing store: {}a key's last L1 copy is written {}{}, over {} store calls at "
      "once",
      store_options_.write_at_publish
          ? "every block is written once its publish has landed (UMBP_L1_DRAM_WRITE=publish); "
            "what that misses, "
          : "",
      store_options_.write_at_eviction ? "at its eviction" : "only ahead of its eviction",
      store_options_.write_at_eviction && store_options_.ahead > 0 ? " and ahead of it" : "",
      store_options_.threads);
  if ((store_options_.ahead > 0 || store_options_.write_at_publish) && !store_worker_.joinable()) {
    stopping_ = false;
    store_worker_ = std::thread(&PoolManager::StoreWorker, this);
  }
}

void PoolManager::StopStoreWorker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  store_wake_.notify_all();
  if (store_worker_.joinable()) store_worker_.join();
}

void PoolManager::StopBackingStore() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    store_on_ = false;
  }
  StopStoreWorker();
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

void PoolManager::SetPeerSource(bool on) {
  std::lock_guard<std::mutex> lock(mutex_);
  peer_source_ = on;
  MORI_UMBP_INFO(
      "[L1Pool] pools as each other's fetch sources: {}",
      on ? "on"
         : "off (UMBP_L1_PEER_SOURCE=0): a key is fetched from the backing store or not at "
           "all, every copy is written there as a last copy, eviction is least recently released");
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
  const uint32_t keep = KeepCopiesLocked();
  if (list.size() <= keep) return false;
  for (size_t i = list.size() - keep; i < list.size(); ++i) {
    if (list[i].pool == pool && list[i].slot == slot) return false;
  }
  return true;
}

void PoolManager::HolderAddedLocked(Holders* holders) {
  const uint32_t keep = KeepCopiesLocked();
  if (keep == 0 || holders->size() <= keep) return;
  // The newest holder stands last. The one it pushed out of the newest
  // `keep_copies_` is spare from now on: if its pool had set it aside as a copy
  // to keep, it is that pool's coldest spare copy.
  const Holder& h = (*holders)[holders->size() - 1 - keep];
  Slot& s = h.pool->slots[h.slot];
  if (s.state == SlotState::kCached && s.kept) {
    Unlink(h.pool, &h.pool->kept, h.slot);
    PushFront(h.pool, &h.pool->lru, h.slot);
    s.kept = false;
  }
}

void PoolManager::HolderUsedLocked(Pool* pool, uint32_t slot) {
  if (KeepCopiesLocked() == 0) return;
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
  StopStoreWorker();
  {
    std::lock_guard<std::mutex> lock(flights_mutex_);
    watchdog_stop_ = true;
  }
  watchdog_wake_.notify_all();
  if (watchdog_.joinable()) watchdog_.join();
  std::vector<Part> parts;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : pools_) {
      for (Part& part : entry.second->parts) parts.push_back(std::move(part));
    }
    pools_.clear();
  }
  for (const Part& part : parts) ReleasePart(part);
}

// ----------------------------- backing store ------------------------------ //

PoolManager::StoreCall PoolManager::StoreCallLocked(const Pool* pool) const {
  StoreCall call;
  if (!BackedLocked(pool) || pool->content_group == nullptr) return call;
  call.store = store_;
  call.generation = pool->content_group->generation;
  call.space = pool->space_prefix + "/" + std::to_string(call.generation);
  call.threads = store_options_.threads;
  return call;
}

bool PoolManager::LastCopyLocked(const Slot& slot) const {
  return slot.content != nullptr && (!peer_source_ || slot.content->second.size() <= 1);
}

bool PoolManager::NeedsWriteBackLocked(const Pool* pool, const Slot& slot) const {
  // A write in flight counts as done: the slot it reads is pinned, so this is
  // another holder's copy of a key about to be in the store.
  return BackedLocked(pool) && LastCopyLocked(slot) && !slot.content->second.stored &&
         !slot.content->second.writing;
}

void PoolManager::AppendFragments(const Pool& pool, uint32_t slot,
                                  std::vector<BackingStore::Fragment>* out) {
  for (const Part& part : pool.parts) {
    for (const Region& region : part.layout.regions) {
      out->push_back(BackingStore::Fragment{
          static_cast<char*>(part.base) + region.offset + uint64_t{slot} * region.unit_bytes,
          region.unit_bytes});
    }
  }
}

// The store copies a call's blocks on the thread that makes the call, so a
// burst of slots goes out as a few calls side by side.
std::vector<bool> PoolManager::StoreBlocks(const StoreCall& call,
                                           std::vector<BackingStore::Block> blocks, bool write) {
  const size_t count = blocks.size();
  const auto one_call = [&](const std::vector<BackingStore::Block>& part) {
    std::vector<bool> done =
        write ? call.store->Write(call.space, part) : call.store->Read(call.space, part);
    done.resize(part.size(), false);
    return done;
  };
  const size_t calls = std::min<size_t>(call.threads, std::max<size_t>(1, count / kStoreCallSlots));
  if (calls <= 1) return one_call(blocks);
  std::vector<std::vector<BackingStore::Block>> parts(calls);
  std::vector<size_t> first(calls + 1, 0);
  for (size_t c = 0; c < calls; ++c) {
    first[c + 1] = first[c] + count / calls + (c < count % calls ? 1 : 0);
    parts[c].assign(std::make_move_iterator(blocks.begin() + first[c]),
                    std::make_move_iterator(blocks.begin() + first[c + 1]));
  }
  std::vector<std::vector<bool>> done(calls);
  std::vector<std::thread> workers;
  for (size_t c = 1; c < calls; ++c) {
    workers.emplace_back([&, c] { done[c] = one_call(parts[c]); });
  }
  done[0] = one_call(parts[0]);
  for (std::thread& worker : workers) worker.join();
  std::vector<bool> all;
  all.reserve(count);
  for (const std::vector<bool>& part : done) all.insert(all.end(), part.begin(), part.end());
  return all;
}

std::vector<bool> PoolManager::WriteBack(Pool* pool, const std::vector<StoreSlot>& slots,
                                         const StoreCall& call, bool ahead) {
  std::vector<bool> stored(slots.size(), false);
  if (slots.empty()) return stored;
  const auto started = std::chrono::steady_clock::now();
  if (call.store != nullptr) {
    std::vector<BackingStore::Block> blocks(slots.size());
    uint64_t bytes = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
      blocks[i].key = slots[i].key;
      AppendFragments(*pool, slots[i].slot, &blocks[i].fragments);
      for (const BackingStore::Fragment& fragment : blocks[i].fragments) bytes += fragment.bytes;
    }
    Flight flight(this, ahead ? "store write" : "store write at eviction", pool->id, slots.size(),
                  bytes);
    stored = StoreBlocks(call, std::move(blocks), /*write=*/true);
  }
  const size_t written = static_cast<size_t>(std::count(stored.begin(), stored.end(), true));
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t before = pool->written_back_slots + pool->written_ahead_slots;
  (ahead ? pool->written_ahead_slots : pool->written_back_slots) += written;
  pool->offered_slots += slots.size();
  if (!ahead) pool->lost_slots += slots.size() - written;
  pool->tier_micros += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  // Remembered with the key, for every pool that holds it: the one this slot
  // is in, when the write was ahead of its eviction; another pool, when pools
  // are not each other's sources and so every copy is written, or when one
  // published the key while the write ran. None of those copies needs a
  // write. Not after a Clear, which moved the group to another key space:
  // what was just written is not where a fetch would look.
  if (pool->content_group != nullptr) {
    const bool current = pool->content_group->generation == call.generation;
    Content& content = pool->content_group->content;
    for (size_t i = 0; i < slots.size(); ++i) {
      const auto entry = content.find(slots[i].key);
      if (entry == content.end()) continue;
      entry->second.writing = false;
      if (stored[i] && current) MarkStoredLocked(&entry->second);
    }
  }
  if (written != slots.size() && !ahead) {
    MORI_UMBP_WARN("[L1Pool] the backing store took {} of {} evicted slots of pool={}: {} lost",
                   written, slots.size(), pool->id, pool->lost_slots);
  }
  if ((before + written) / kStoreLogEvery != before / kStoreLogEvery) {
    MORI_UMBP_INFO(
        "[L1Pool] backing store pool={} published={} offered={} written_at_eviction={} "
        "written_ahead={} unstored={} loaded={} lost={} micros={}",
        pool->id, pool->published_slots, pool->offered_slots, pool->written_back_slots,
        pool->written_ahead_slots, pool->unstored, pool->loaded_slots, pool->lost_slots,
        pool->tier_micros);
  }
  return stored;
}

void PoolManager::HolderGainedLocked(Pool* pool, const Holders& holders) {
  if (!holders.stored) ++pool->unstored;
}

void PoolManager::HolderLostLocked(Pool* pool, const Holders& holders) {
  if (!holders.stored) --pool->unstored;
}

void PoolManager::MarkStoredLocked(Holders* holders) {
  if (holders->stored) return;
  for (const Holder& holder : *holders) --holder.pool->unstored;
  holders->stored = true;
}

void PoolManager::ForgetStoredLocked(Group* group) {
  for (auto& entry : group->content) {
    Holders& holders = entry.second;
    if (!holders.stored) continue;
    for (const Holder& holder : holders) ++holder.pool->unstored;
    holders.stored = false;
  }
}

namespace {

// The pools in the order they were attached, starting after `serial`: each
// writer serves them in turn.
template <typename Pools>
auto PoolsAfter(Pools& pools, uint64_t serial) {
  std::vector<decltype(pools.begin()->second.get())> order;
  order.reserve(pools.size());
  for (auto& entry : pools) order.push_back(entry.second.get());
  std::sort(order.begin(), order.end(),
            [](const auto* a, const auto* b) { return a->serial < b->serial; });
  const auto next = std::find_if(order.begin(), order.end(),
                                 [&](const auto* pool) { return pool->serial > serial; });
  std::rotate(order.begin(), next, order.end());
  return order;
}

}  // namespace

int PoolManager::WritePinnedLocked(std::unique_lock<std::mutex>& lock, Pool* pool,
                                   const std::vector<StoreSlot>& slots) {
  // Pinned: an eviction leaves them alone until the copy is out. A held slot
  // is read as a fetch from another pool reads it: its content has landed,
  // and its owner does not write a block it has published.
  for (const StoreSlot& slot : slots) {
    Slot& s = pool->slots[slot.slot];
    ++s.pins;
    s.content->second.writing = true;
  }
  ++pool->busy;
  const StoreCall call = StoreCallLocked(pool);
  lock.unlock();
  const std::vector<bool> stored = WriteBack(pool, slots, call, /*ahead=*/true);
  lock.lock();
  for (const StoreSlot& slot : slots) --pool->slots[slot.slot].pins;
  --pool->busy;
  idle_.notify_all();
  return static_cast<int>(std::count(stored.begin(), stored.end(), true));
}

// Write-through. A published slot waits in `fresh` until its owner's device
// has got past its stamp, then in `ready` until the writer takes it: the
// newest first, since the block another rank asks for next is the one a
// request has just produced. What is still waiting when the queue overflows,
// and what the store refuses, is left to the writer ahead and to eviction.
PoolManager::WriterRound PoolManager::WriteFresh() {
  WriterRound round;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!store_on_ || !store_options_.write_at_publish) return round;
  const auto worth_writing = [](const Pool& pool, uint32_t slot) {
    if (slot >= pool.slots.size()) return false;
    const Slot& s = pool.slots[slot];
    return s.content != nullptr && (s.state == SlotState::kHeld || s.state == SlotState::kCached) &&
           !s.content->second.stored && !s.content->second.writing;
  };
  for (Pool* pool : PoolsAfter(pools_, fresh_last_serial_)) {
    if (!pool->bound || !BackedLocked(pool)) {
      pool->fresh.clear();
      pool->ready.clear();
      continue;
    }
    // What has landed since the last round. The rest keeps its place; slots
    // settle in the order they were published, so a run of them that has not
    // is the end of what has.
    std::vector<uint32_t> waiting;
    size_t unsettled_run = 0;
    for (size_t looked = 0;
         !pool->fresh.empty() && looked < kFreshScan && unsettled_run < kFreshUnsettledRun;
         ++looked) {
      const uint32_t slot = pool->fresh.front();
      pool->fresh.pop_front();
      if (!worth_writing(*pool, slot)) continue;
      if (Readable(*pool, pool->slots[slot])) {
        pool->ready.push_back(slot);
        unsettled_run = 0;
      } else {
        waiting.push_back(slot);
        ++unsettled_run;
      }
    }
    pool->fresh.insert(pool->fresh.begin(), waiting.begin(), waiting.end());
    while (pool->ready.size() > pool->num_slots) pool->ready.pop_front();

    std::vector<StoreSlot> slots;
    while (!pool->ready.empty() && slots.size() < kAheadBatch) {
      const uint32_t slot = pool->ready.back();
      pool->ready.pop_back();
      if (!worth_writing(*pool, slot) || !Readable(*pool, pool->slots[slot])) continue;
      Slot& s = pool->slots[slot];
      s.content->second.writing = true;  // a slot queued twice is taken once
      slots.push_back(StoreSlot{slot, s.content->first});
    }
    if (slots.empty()) continue;
    fresh_last_serial_ = pool->serial;
    round.written = WritePinnedLocked(lock, pool, slots);
    return round;
  }
  return round;
}

// The slots an eviction takes next are a pool's coldest last copies: write the
// content of up to `ahead` of its slots to the store before they are asked
// for, so the Allocate that evicts them has nothing to write.
//
// A slice of one pool at a time. Nearly every slot the walk passes has been
// written already, a pool has hundreds of thousands of them, and the walk is
// made under the lock every call of every engine takes: a pool walked whole
// for every batch holds that lock most of the time and finds a hundred slots
// a pass (two pools of 400k slots: 17k slots a second, and a Commit waiting
// 17 ms for the lock). So a walk gives the lock up after `kAheadSlice` slots,
// comes back to where it was, and after a slice that found nothing stays
// away for several times as long as the slice took.
PoolManager::WriterRound PoolManager::WriteAhead() {
  WriterRound round;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!store_on_ || store_options_.ahead <= 0) return round;
  const auto started = std::chrono::steady_clock::now();
  if (started < ahead_not_before_) {
    round.sweep_done = false;
    return round;
  }
  for (Pool* pool : PoolsAfter(pools_, ahead_last_serial_)) {
    if (!pool->bound || !BackedLocked(pool)) continue;
    Pool::AheadPass& pass = pool->ahead;
    if (pass.sweep == ahead_sweep_ && !pass.active) continue;  // walked to its end already
    ahead_last_serial_ = pool->serial;
    round.sweep_done = false;
    if (pass.sweep != ahead_sweep_) {
      // In the order an eviction takes last copies: those it has passed over
      // first, then the coldest of the rest.
      pass.sweep = ahead_sweep_;
      pass.active = true;
      pass.in_lru = false;
      pass.cursor = pool->kept.head;
      pass.budget = static_cast<size_t>(store_options_.ahead * pool->num_slots);
    } else if (pass.cursor != kNoSlot) {
      // The slot the walk stopped at was taken meanwhile: this pool's walk
      // ends here, and the next sweep starts it again.
      const Slot& at = pool->slots[pass.cursor];
      if (at.state != SlotState::kCached || at.kept == pass.in_lru) {
        pass.active = false;
        continue;
      }
    }
    std::vector<StoreSlot> slots;
    size_t walked = 0;
    while (walked < kAheadSlice && slots.size() < kAheadBatch) {
      if (pass.cursor == kNoSlot) {
        if (pass.in_lru) {
          pass.active = false;
          break;
        }
        pass.in_lru = true;
        pass.cursor = pool->lru.head;
        continue;
      }
      const uint32_t slot = pass.cursor;
      const Slot& s = pool->slots[slot];
      pass.cursor = s.next;
      ++walked;
      // A copy another pool also holds, and can be fetched from, is dropped
      // at eviction, not written.
      if (!LastCopyLocked(s)) continue;
      if (pass.budget == 0) {
        pass.active = false;
        break;
      }
      --pass.budget;
      if (s.pins != 0 || !Readable(*pool, s) || s.content->second.stored ||
          s.content->second.writing) {
        continue;
      }
      slots.push_back(StoreSlot{slot, s.content->first});
    }
    const auto walked_for = std::chrono::steady_clock::now() - started;
    round.scan_micros = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(walked_for).count());
    if (!slots.empty()) {
      round.written = WritePinnedLocked(lock, pool, slots);
    } else {
      ahead_not_before_ = std::chrono::steady_clock::now() + kAheadRest * walked_for;
    }
    return round;
  }
  // Every pool has been walked to its end: the next call starts over.
  ++ahead_sweep_;
  return round;
}

void PoolManager::StoreWorker() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopping_) {
    lock.unlock();
    const WriterRound fresh = WriteFresh();
    // Write-through comes first; the walk ahead of eviction only runs when it
    // has nothing to do.
    const WriterRound ahead = fresh.written < 0 ? WriteAhead() : WriterRound{};
    lock.lock();
    if (stopping_) break;
    if (fresh.written > 0 || ahead.written > 0) continue;
    if (fresh.written == 0 || ahead.written == 0) {
      // A store that took none of what it was offered: full of content it may
      // not drop, or closing. Offered again, but not at once.
      store_wake_.wait_for(lock, kAheadRefused);
    } else if (ahead.sweep_done) {
      // Nothing to write anywhere. A publish or a commit wakes the writer.
      store_wake_.wait_for(lock, kAheadIdle);
    } else if (std::chrono::steady_clock::now() < ahead_not_before_) {
      store_wake_.wait_until(lock, ahead_not_before_);
    }
  }
}

// ------------------------------- diagnostics ------------------------------ //

namespace {

double SecondsSince(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

}  // namespace

uint64_t PoolManager::BeginFlight(const char* what, const std::string& pool, size_t slots,
                                  uint64_t bytes) {
  FlightRecord record;
  record.what = what;
  record.pool = pool;
  record.slots = slots;
  record.bytes = bytes;
  record.since = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(flights_mutex_);
  const uint64_t id = next_flight_++;
  flights_.emplace(id, std::move(record));
  return id;
}

void PoolManager::EndFlight(uint64_t id) {
  std::string line;
  {
    std::lock_guard<std::mutex> lock(flights_mutex_);
    const auto found = flights_.find(id);
    if (found == flights_.end()) return;
    const FlightRecord record = std::move(found->second);
    flights_.erase(found);
    const double seconds = SecondsSince(record.since);
    if (seconds < std::chrono::duration<double>(slow_call_).count()) return;
    char took[32];
    std::snprintf(took, sizeof(took), "%.3f", seconds);
    line = std::string("slow call: ") + record.what + " pool=" + record.pool +
           " slots=" + std::to_string(record.slots) + " bytes=" + std::to_string(record.bytes) +
           " took " + took + " s";
    ++diagnostics_.slow_calls;
    diagnostics_.last = line;
  }
  MORI_UMBP_WARN("[L1Pool] {}", line);
}

// Reports, once, a store call or a copy between pools that has not come back,
// with what a stall is told apart by: what else is in flight, which calls
// are inside the pool manager, whether its lock can be had, and what the
// store says its own calls are waiting for.
void PoolManager::Watchdog() {
  static const char* const kNames[kCallKinds] = {
      "Attach", "BindLayout", "Open", "Allocate", "Apply", "Commit", "Fetch", "Clear", "Detach"};
  std::unique_lock<std::mutex> lock(flights_mutex_);
  while (!watchdog_stop_) {
    const auto period = std::min<std::chrono::milliseconds>(
        std::chrono::milliseconds(1000),
        std::max<std::chrono::milliseconds>(std::chrono::milliseconds(5), stuck_call_ / 4));
    watchdog_wake_.wait_for(lock, period);
    if (watchdog_stop_) break;
    const auto now = std::chrono::steady_clock::now();
    std::vector<FlightRecord> stuck;
    std::map<std::string, std::pair<int, double>> in_flight;  // what -> how many, the oldest
    for (auto& entry : flights_) {
      FlightRecord& record = entry.second;
      const double age = std::chrono::duration<double>(now - record.since).count();
      std::pair<int, double>& kind = in_flight[record.what];
      ++kind.first;
      kind.second = std::max(kind.second, age);
      if (record.reported || now - record.since < stuck_call_) continue;
      record.reported = true;
      stuck.push_back(record);
    }
    if (stuck.empty()) continue;
    lock.unlock();

    std::string others;
    char number[32];
    for (const auto& kind : in_flight) {
      std::snprintf(number, sizeof(number), "%.1f", kind.second.second);
      others += (others.empty() ? "" : ", ") + std::to_string(kind.second.first) + " " +
                kind.first + " (oldest " + number + " s)";
    }
    std::string calls;
    for (int kind = 0; kind < kCallKinds; ++kind) {
      const int count = calls_[kind].load(std::memory_order_relaxed);
      if (count == 0) continue;
      calls +=
          (calls.empty() ? "" : ", ") + std::string(kNames[kind]) + " " + std::to_string(count);
    }
    // Never waited for: this thread must be able to say that the lock is the
    // thing that is stuck. Held for well under a millisecond at a time when
    // all is well, so a try that fails twenty times over is a finding.
    bool lock_free = false;
    std::shared_ptr<BackingStore> store;
    for (int probe = 0; probe < kLockProbes && !lock_free; ++probe) {
      if (mutex_.try_lock()) {
        store = store_;
        mutex_.unlock();
        lock_free = true;
      } else {
        std::this_thread::sleep_for(kLockProbeGap);
      }
    }
    const std::string describes = store != nullptr ? store->Describe() : std::string();
    std::vector<std::string> lines;
    for (const FlightRecord& record : stuck) {
      std::snprintf(number, sizeof(number), "%.1f", SecondsSince(record.since));
      lines.push_back(std::string("call in flight for ") + number + " s: " + record.what +
                      " pool=" + record.pool + " slots=" + std::to_string(record.slots) +
                      " bytes=" + std::to_string(record.bytes) + "; in flight now: " + others +
                      "; calls inside the pool manager: " + (calls.empty() ? "none" : calls) +
                      "; the pool manager's lock " +
                      (lock_free
                           ? "can be had"
                           : "could not be had in " +
                                 std::to_string(kLockProbes * kLockProbeGap.count()) + " ms") +
                      (describes.empty() ? "" : "; store: " + describes));
    }
    for (const std::string& line : lines) MORI_UMBP_WARN("[L1Pool] {}", line);
    lock.lock();
    diagnostics_.stuck_calls += lines.size();
    diagnostics_.last = lines.back();
  }
}

void PoolManager::SetStallThresholds(std::chrono::milliseconds slow,
                                     std::chrono::milliseconds stuck) {
  {
    std::lock_guard<std::mutex> lock(flights_mutex_);
    slow_call_ = slow;
    stuck_call_ = stuck;
  }
  watchdog_wake_.notify_all();
}

StallDiagnostics PoolManager::Diagnostics() const {
  std::lock_guard<std::mutex> lock(flights_mutex_);
  return diagnostics_;
}

void PoolManager::ReleasePart(const Part& part) {
  if (!part.attached) return;
  if (part.registered) {
    std::shared_ptr<BackingStore> store;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      store = store_;
    }
    if (store != nullptr) store->RemoveMemory(part.base);
  }
  if (part.external) {
    api_->Unimport(part.device, part.mapped);
  } else {
    api_->Free(part.device, part.base, part.bytes);
  }
}

// Pools of one group declare the same slot sizes part by part (`BindLayout`),
// so two of them differ in shape only by how many parts they have.
bool PoolManager::SameShape(const Pool& src, const Pool& dst) {
  return src.parts.size() == dst.parts.size();
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
      HolderLostLocked(pool, list);
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
  for (Part& part : pool->parts) part.committed = 0;
  pool->fresh.clear();
  pool->ready.clear();
  pool->ahead = Pool::AheadPass{};
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
                                 bool settled_only, std::vector<StoreSlot>* victims) {
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
    if (KeepCopiesLocked() == 0 || IsSpareLocked(pool, it)) {
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
    if (victims != nullptr && NeedsWriteBackLocked(pool, s)) {
      // Content its owner is still writing cannot be read out: it is lost. So
      // is content not written ahead when evictions do not wait for the store.
      if (store_options_.write_at_eviction && Readable(*pool, s)) {
        victims->push_back(StoreSlot{victim, s.content->first});
      } else {
        ++pool->lost_slots;
      }
    }
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
  const std::vector<Part> parts = std::move(pool->parts);
  uint64_t bytes = 0;
  for (const Part& part : parts) bytes += part.attached ? part.bytes : 0;
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} parts={} bytes={} evicted={} spare_copy={}", pool_id,
                 parts.size(), bytes, pool->evicted, pool->evicted_spare);
  pools_.erase(pool_id);
  // The pool is gone from the table and nothing copies into or out of it: its
  // memory is given up without the lock, which the store may make wait.
  lock.unlock();
  for (const Part& part : parts) ReleasePart(part);
  lock.lock();
}

AttachResult PoolManager::Attach(const std::string& pool_id, const std::string& group, int device,
                                 uint64_t bytes, const ExternalMemory& external,
                                 const PartSpec& spec) {
  CallScope in_call(this, kAttach);
  const bool adopt = !external.handle.empty();
  AttachResult result;
  if (pool_id.empty() || bytes == 0 || device < 0) {
    result.status = Status::Error("attach needs a pool id, a device and a non-zero size");
    return result;
  }
  if (spec.count == 0 || spec.count > kMaxParts || spec.index >= spec.count) {
    result.status = Status::Error("attach of pool " + pool_id + " names part " +
                                  std::to_string(spec.index) + " of " + std::to_string(spec.count));
    return result;
  }
  const uint64_t offset = adopt ? external.offset : 0;
  const std::string attach_key = pool_id + "#" + std::to_string(spec.index);
  std::unique_lock<std::mutex> lock(mutex_);
  // DestroyLocked gives up the lock while the pool's copies finish, so another
  // Attach of this id may have replaced the pool, or be allocating it, by the
  // time it returns: look again each time round.
  for (;;) {
    if (attaching_.count(attach_key) != 0) {
      result.status = Status::Error("pool " + pool_id + " is already being attached");
      return result;
    }
    Pool* existing = FindLocked(pool_id);
    if (existing == nullptr) break;
    if (existing->group == group && existing->parts.size() == spec.count) {
      const Part& have = existing->parts[spec.index];
      // The pool's other parts came first and it is waiting for this one.
      if (!have.attached) break;
      if (have.device == device && have.bytes == bytes && have.external == adopt &&
          have.external_handle == external.handle && have.external_offset == offset) {
        result.memory = have.exported;
        result.bytes = have.bytes;
        result.created = false;
        return result;
      }
    }
    // Another shape under this id: the pool goes, every part of it. Its owner
    // restarted, and the other parts' memory went with the old process too.
    DestroyLocked(lock, existing);
  }
  // The allocation zero-fills hundreds of GB; do it without the lock so pools
  // already serving are not stalled behind it.
  attaching_.insert(attach_key);
  const std::shared_ptr<BackingStore> store = store_on_ ? store_ : nullptr;
  lock.unlock();
  Part part;
  std::string error;
  const bool ok = adopt ? api_->Import(device, external.handle, external.offset, &part.base,
                                       &part.mapped, &error)
                        : api_->Allocate(device, bytes, &part.base, &part.exported, &error);
  // Here, once, and not when a slot is copied: the store learns where this
  // memory is and on which device, so that its copies are planned against the
  // allocation rather than fragment by fragment.
  if (ok && store != nullptr) {
    std::string why;
    part.registered = store->AddMemory(part.base, bytes, device, &why);
    if (!part.registered) {
      MORI_UMBP_WARN(
          "[L1Pool] the backing store did not take pool={} part={}: {}; the pool's evicted "
          "content is not kept",
          pool_id, spec.index, why);
    }
  }
  lock.lock();
  attaching_.erase(attach_key);
  if (!ok) {
    MORI_UMBP_ERROR("[L1Pool] attach pool={} part={} device={} bytes={} failed: {}", pool_id,
                    spec.index, device, bytes, error);
    result.status = Status::Error(error);
    return result;
  }
  part.attached = true;
  part.device = device;
  part.bytes = bytes;
  part.external = adopt;
  part.external_handle = external.handle;
  part.external_offset = offset;
  // The pool's other parts attach on their own threads: by now one of them may
  // have created the pool this part belongs to.
  Pool* pool = FindLocked(pool_id);
  if (pool != nullptr && (pool->group != group || pool->parts.size() != spec.count ||
                          pool->parts[spec.index].attached)) {
    lock.unlock();
    ReleasePart(part);
    result.status = Status::Error("pool " + pool_id + " was replaced while part " +
                                  std::to_string(spec.index) + " was attaching");
    return result;
  }
  if (pool == nullptr) {
    auto created = std::make_unique<Pool>();
    created->id = pool_id;
    created->serial = next_serial_++;
    created->group = group;
    created->parts.resize(spec.count);
    pool = created.get();
    pools_[pool_id] = std::move(created);
  }
  result.memory = part.exported;
  pool->parts[spec.index] = std::move(part);
  MORI_UMBP_INFO("[L1Pool] attached pool={} part={}/{} group={} device={} bytes={}{}", pool_id,
                 spec.index, spec.count, group, device, bytes,
                 adopt ? " (adopted from its owner)" : "");
  result.bytes = bytes;
  result.created = true;
  return result;
}

Status PoolManager::BindLayout(const std::string& pool_id, const Layout& layout,
                               uint32_t part_index) {
  CallScope in_call(this, kBindLayout);
  if (layout.num_slots == 0 || layout.regions.empty()) {
    return Status::Error("a layout needs slots and at least one region");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status::Error("unknown pool " + pool_id);
  const auto part_of = [&](Pool* p) -> Part* {
    return part_index < p->parts.size() && p->parts[part_index].attached ? &p->parts[part_index]
                                                                         : nullptr;
  };
  Part* part = part_of(pool);
  if (part == nullptr) {
    return Status::Error("pool " + pool_id + " has no part " + std::to_string(part_index) +
                         " attached");
  }
  std::vector<uint64_t> units;
  for (const Region& region : layout.regions) {
    // Divided rather than multiplied out: offset + slots * unit can wrap.
    if (region.unit_bytes == 0 || region.offset > part->bytes ||
        region.unit_bytes > (part->bytes - region.offset) / layout.num_slots) {
      return Status::Error("a region does not fit in pool " + pool_id);
    }
    units.push_back(region.unit_bytes);
  }
  if (part->bound && SameLayout(part->layout, layout)) return Status{};
  // One slot table for every part: slot i is the same slot on each of them.
  for (size_t i = 0; i < pool->parts.size(); ++i) {
    const Part& other = pool->parts[i];
    if (i != part_index && other.bound && other.layout.num_slots != layout.num_slots) {
      return Status::Error("part " + std::to_string(part_index) + " of pool " + pool_id +
                           " declares " + std::to_string(layout.num_slots) + " slots, part " +
                           std::to_string(i) + " " + std::to_string(other.layout.num_slots));
    }
  }

  Group& group = groups_[pool->group];
  if (part_index < group.unit_bytes.size() && !group.unit_bytes[part_index].empty() &&
      group.unit_bytes[part_index] != units) {
    for (const auto& entry : pools_) {
      const Pool* other = entry.second.get();
      if (other != pool && other->group == pool->group && part_index < other->parts.size() &&
          other->parts[part_index].bound) {
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
    part = pool == nullptr ? nullptr : part_of(pool);
    if (part == nullptr) return Status::Error("pool " + pool_id + " was detached");
    ResetSlotsLocked(pool);
  }
  Group& current = groups_[pool->group];
  if (current.unit_bytes.size() <= part_index) current.unit_bytes.resize(part_index + 1);
  current.unit_bytes[part_index] = std::move(units);
  const bool was_bound = pool->bound;
  part->layout = layout;
  part->bound = true;
  pool->bound = std::all_of(pool->parts.begin(), pool->parts.end(),
                            [](const Part& p) { return p.attached && p.bound; });
  MORI_UMBP_INFO("[L1Pool] bound pool={} part={}/{} slots={} regions={}{}", pool_id, part_index,
                 pool->parts.size(), layout.num_slots, layout.regions.size(),
                 pool->bound ? "" : " (waiting for the other parts)");
  if (!pool->bound) return Status{};
  // Room for every slot's key now, so the index never rehashes -- a pause of
  // tens of milliseconds at this size -- while engines are being served.
  if (was_bound) current.slots -= pool->num_slots;
  current.slots += layout.num_slots;
  current.content.reserve(current.slots);
  pool->content_group = &current;
  pool->num_slots = layout.num_slots;
  // Pools write and find each other's evicted content in the store only when
  // their slots are cut the same way, part by part and region by region.
  uint64_t shape = 14695981039346656037ULL;
  for (const Part& p : pool->parts) {
    shape = HashUnits(shape, p.layout.regions.size());
    for (const Region& region : p.layout.regions) shape = HashUnits(shape, region.unit_bytes);
  }
  char shape_text[17];
  std::snprintf(shape_text, sizeof(shape_text), "%016llx", static_cast<unsigned long long>(shape));
  pool->space_prefix = pool->group + "/" + std::to_string(pool->parts.size()) + "x" + shape_text;
  pool->backed = std::all_of(pool->parts.begin(), pool->parts.end(),
                             [](const Part& p) { return p.registered; });
  pool->slots.assign(layout.num_slots, Slot{});
  ResetSlotsLocked(pool);
  return Status{};
}

OpenResult PoolManager::Open(const std::string& pool_id) {
  CallScope in_call(this, kOpen);
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
  result.num_slots = pool->num_slots;
  MORI_UMBP_INFO("[L1Pool] opened pool={} slots={}", pool_id, result.num_slots);
  return result;
}

AllocateResult PoolManager::Allocate(const std::string& pool_id, uint32_t count,
                                     uint32_t min_count) {
  CallScope in_call(this, kAllocate);
  AllocateResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  const auto deadline = std::chrono::steady_clock::now() + kPinnedWait;
  // Evicted slots whose content has to reach the backing store before the
  // owner is given them: until this call returns they are held, and nobody
  // writes them.
  std::vector<StoreSlot> victims;
  while (result.slots.size() < count) {
    uint32_t slot = 0;
    const bool required = result.slots.size() < min_count;
    if (TakeFreeLocked(pool, &slot, required ? &result.evicted_keys : nullptr,
                       /*settled_only=*/false, &victims)) {
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
  if (!victims.empty()) {
    const StoreCall call = StoreCallLocked(pool);
    ++pool->busy;  // Open, a re-bind and Detach wait for the copy
    lock.unlock();
    WriteBack(pool, victims, call, /*ahead=*/false);
    lock.lock();
    --pool->busy;
    idle_.notify_all();
  }
  return result;
}

Status PoolManager::Commit(const std::string& pool_id, uint64_t committed, uint32_t part) {
  CallScope in_call(this, kCommit);
  std::lock_guard<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status::Error("unknown pool " + pool_id);
  if (part >= pool->parts.size()) {
    return Status::Error("pool " + pool_id + " has no part " + std::to_string(part));
  }
  pool->parts[part].committed = std::max(pool->parts[part].committed, committed);
  // A slot is one slot on every part: it has landed when the slowest has.
  uint64_t slowest = pool->parts[0].committed;
  for (const Part& p : pool->parts) slowest = std::min(slowest, p.committed);
  const bool advanced = slowest > pool->committed;
  pool->committed = slowest;
  // A fetch into this pool's held slots may be waiting for them to settle.
  idle_.notify_all();
  // And so may published slots bound for the store.
  if (advanced && !pool->fresh.empty()) store_wake_.notify_one();
  return Status{};
}

Status PoolManager::Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp) {
  CallScope in_call(this, kApply);
  std::lock_guard<std::mutex> lock(mutex_);
  // A published slot that has landed already goes to the store at once; one
  // that has not waits for the Commit that says so.
  bool queued = false;
  struct WakeWriter {
    ~WakeWriter() {
      if (*queued) wake->notify_one();
    }
    const bool* queued;
    std::condition_variable* wake;
  } wake_writer{&queued, &store_wake_};
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
          HolderLostLocked(pool, holders);
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
        HolderGainedLocked(pool, holders);
        HolderAddedLocked(&holders);
        ++pool->published_slots;
        // Write-through: to the store as soon as it has landed, unless the
        // store was given this key already.
        if (store_options_.write_at_publish && BackedLocked(pool) && !holders.stored) {
          if (pool->fresh.size() >= pool->num_slots) pool->fresh.pop_front();
          pool->fresh.push_back(op.slot);
          queued = queued || Readable(*pool, s);
        }
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
  CallScope in_call(this, kFetch);
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
  // Another pool's settled copy, or null; always null when pools are not each
  // other's sources.
  auto source_of = [&](const Holders* holders) -> const Holder* {
    if (holders == nullptr || !peer_source_) return nullptr;
    for (const Holder& h : *holders) {
      // Not this pool, and not a slot its owner may still be writing.
      if (h.pool != pool && SameShape(*h.pool, *pool) && Readable(*h.pool, h.pool->slots[h.slot])) {
        return &h;
      }
    }
    return nullptr;
  };

  const bool into = !options.dst_slots.empty();
  // Whether a pool of the group can supply `key` to this call: a fetch into
  // the caller's slots does not look in this pool.
  auto in_pools = [&](const std::string& key, bool local_counts) {
    const Holders* holders = holders_of(key);
    return (local_counts && local_of(holders) != kNoSlot) || source_of(holders) != nullptr;
  };

  // The keys no pool can supply are looked up in the backing store, in one
  // call and without the lock, before anything is taken: `in_store[i]` then
  // says the store held keys[i] a moment ago. A call whose every key a pool
  // holds asks the store nothing.
  StoreCall store_call = StoreCallLocked(pool);
  std::vector<char> in_store;
  uint64_t lookup_micros = 0;
  const auto look_in_store = [&](bool local_counts) {
    std::vector<std::string> ask;
    std::vector<size_t> ask_index;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (in_pools(keys[i], local_counts)) continue;
      ask.push_back(keys[i]);
      ask_index.push_back(i);
    }
    if (ask.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    lock.unlock();
    std::vector<bool> found;
    {
      Flight flight(this, "store lookup", pool_id, ask.size(), 0);
      found = store_call.store->Contains(store_call.space, ask);
    }
    lock.lock();
    lookup_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - started)
                                              .count());
    found.resize(ask.size(), false);
    in_store.assign(keys.size(), 0);
    for (size_t j = 0; j < ask.size(); ++j) in_store[ask_index[j]] = found[j] ? 1 : 0;
  };
  // The lock was given up for the lookup: the pool may be gone, or its group
  // cleared, and then the answer is for a key space nobody reads any more.
  const auto after_lookup = [&]() {
    pool = FindLocked(pool_id);
    if (pool == nullptr || !pool->bound) return false;
    content = &pool->content_group->content;
    const StoreCall now = StoreCallLocked(pool);
    if (now.store == nullptr || now.space != store_call.space) in_store.clear();
    store_call = now;
    pool->tier_micros += lookup_micros;
    return true;
  };
  auto held_by_store = [&](size_t i) { return i < in_store.size() && in_store[i] != 0; };

  if (options.probe_only) {
    if (store_call.store != nullptr) {
      look_in_store(/*local_counts=*/true);
      if (!after_lookup()) {
        result.status = Status::Error("pool " + pool_id + " was detached");
        return result;
      }
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!in_pools(keys[i], /*local_counts=*/true) && !held_by_store(i)) break;
      ++result.available;
    }
    return result;
  }
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
    store_call = StoreCallLocked(pool);
    settle_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - settle_started)
                                              .count());
  }

  if (store_call.store != nullptr) {
    look_in_store(/*local_counts=*/!into);
    if (!after_lookup()) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      return result;
    }
  }

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists -- or, copying
  // into the caller's slots, check that it holds them.
  std::vector<CopyStep> steps;
  std::vector<StoreSlot> victims;  // slots evicted for this call, bound for the store
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
      if (source == nullptr && !held_by_store(i)) break;
      if (!Fillable(*pool, pool->slots[dst])) {
        // Allocate may regrant a slot before its owner's writes complete:
        // owner-local reuse is ordered on that stream, but this copy is not.
        // Stop at the landed prefix; the caller still holds every destination.
        break;
      }
      if (source != nullptr) {
        ++source->pool->slots[source->slot].pins;
        ++source->pool->busy;
        steps.push_back(CopyStep{source->pool, source->slot, dst, i});
      } else {
        // From the store, which may have dropped the key by the time it is
        // read: the call then ends before this key.
        steps.push_back(CopyStep{nullptr, 0, dst, i});
      }
      ++pool->busy;
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
    if (found == nullptr && !held_by_store(i)) break;
    // By value: making room below edits the index. Without one, from the store.
    const bool from_pool = found != nullptr;
    const Holder source = from_pool ? *found : Holder{};
    uint32_t dst = 0;
    if (!TakeFreeLocked(pool, &dst, &result.evicted_keys, /*settled_only=*/true, &victims)) break;
    pool->slots[dst].state = SlotState::kFilling;
    if (from_pool) {
      ++source.pool->slots[source.slot].pins;
      ++source.pool->busy;
      steps.push_back(CopyStep{source.pool, source.slot, dst, i});
    } else {
      steps.push_back(CopyStep{nullptr, 0, dst, i});
    }
    ++pool->busy;
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
  // Part p of the source slots to part p of the destination slots.
  const size_t num_parts = pool->parts.size();
  std::vector<std::vector<MemoryApi::Fragment>> fragments(num_parts);
  // The slots this call evicted to make room go to the store before it fills
  // them.
  if (!victims.empty()) WriteBack(pool, victims, store_call, /*ahead=*/false);
  uint64_t slot_bytes = 0;
  for (const Part& part : pool->parts) {
    for (const Region& region : part.layout.regions) slot_bytes += region.unit_bytes;
  }
  // The steps the store supplies: it reads each key into its destination slot.
  std::vector<size_t> store_steps;
  std::vector<BackingStore::Block> store_blocks;
  uint64_t store_micros = 0;
  size_t store_dropped = 0;  // keys the store held at the lookup and not at the read
  size_t run_start = 0;
  bool failed = false;
  while (run_start < steps.size()) {
    const CopyStep& first = steps[run_start];
    if (first.src == nullptr) {
      store_steps.push_back(run_start);
      store_blocks.emplace_back();
      store_blocks.back().key = keys[first.key_index];
      AppendFragments(*pool, first.dst_slot, &store_blocks.back().fragments);
      ++run_start;
      continue;
    }
    size_t run_end = run_start + 1;
    while (run_end < steps.size() && steps[run_end].src == first.src &&
           steps[run_end].src_slot == steps[run_end - 1].src_slot + 1 &&
           steps[run_end].dst_slot == steps[run_end - 1].dst_slot + 1) {
      ++run_end;
    }
    const uint64_t run_slots = run_end - run_start;
    for (size_t p = 0; p < num_parts; ++p) {
      const Part& dst_part = pool->parts[p];
      const Part& src_part = first.src->parts[p];
      for (size_t r = 0; r < dst_part.layout.regions.size(); ++r) {
        const Region& dst_region = dst_part.layout.regions[r];
        const Region& src_region = src_part.layout.regions[r];
        const uint64_t unit = dst_region.unit_bytes;
        void* dst =
            static_cast<char*>(dst_part.base) + dst_region.offset + uint64_t{first.dst_slot} * unit;
        const void* src = static_cast<const char*>(src_part.base) + src_region.offset +
                          uint64_t{first.src_slot} * unit;
        fragments[p].push_back(MemoryApi::Fragment{dst, src, run_slots * unit, src_part.device});
      }
    }
    run_start = run_end;
  }
  // One batch a part for the whole call, landed before it returns: nothing is
  // handed to the requester before then. The parts are on different devices,
  // so their batches run side by side, and the store's reads beside them; a
  // slot copied from a pool counts only if every part of it landed, and one
  // read from the store only if the store filled it.
  if (!steps.empty()) {
    std::vector<bool> read(store_steps.size(), false);
    std::thread reader;
    const bool from_pools = store_steps.size() != steps.size();
    const auto read_store = [&] {
      const auto read_started = std::chrono::steady_clock::now();
      Flight flight(this, "store read", pool_id, store_steps.size(),
                    store_steps.size() * slot_bytes);
      read = StoreBlocks(store_call, std::move(store_blocks), /*write=*/false);
      store_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - read_started)
                                               .count());
    };
    if (!store_steps.empty() && store_call.store != nullptr) {
      if (from_pools) {
        reader = std::thread(read_store);
      } else {
        read_store();
      }
    }
    if (from_pools) {
      const size_t pool_steps = steps.size() - store_steps.size();
      Flight flight(this, "copy between pools", pool_id, pool_steps, pool_steps * slot_bytes);
      std::vector<std::string> errors(num_parts);
      std::vector<char> landed(num_parts, 0);
      const auto copy_part = [&](size_t p) {
        const int device = pool->parts[p].device;
        landed[p] =
            fragments[p].empty() || api_->CopyBatch(device, fragments[p], &errors[p]) ? 1 : 0;
      };
      std::vector<std::thread> workers;
      for (size_t p = 1; p < num_parts; ++p) workers.emplace_back(copy_part, p);
      copy_part(0);
      for (std::thread& worker : workers) worker.join();
      for (size_t p = 0; p < num_parts; ++p) {
        if (landed[p] != 0) continue;
        if (!failed) copy_error = "part " + std::to_string(p) + ": " + errors[p];
        failed = true;
      }
    }
    if (reader.joinable()) reader.join();
    for (size_t i = 0; i < steps.size(); ++i) copied[i] = steps[i].src != nullptr && !failed;
    for (size_t j = 0; j < store_steps.size(); ++j) {
      copied[store_steps[j]] = read[j];
      if (!read[j]) ++store_dropped;
    }
  }
  result.copy_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  if (copying) lock.lock();

  // Commit: a copied slot becomes cached content of this pool.
  const uint64_t store_generation_now = pool->content_group->generation;
  size_t valid_keys = result.slots.size();
  std::vector<std::pair<std::string, uint32_t>> sources;  // runs, for the log
  for (size_t i = 0; i < steps.size(); ++i) {
    const CopyStep& step = steps[i];
    if (step.src != nullptr) {
      --step.src->slots[step.src_slot].pins;
      --step.src->busy;
    }
    --pool->busy;
    // Who a landed slot came from: another pool, or the backing store.
    const auto count_source = [&] {
      const std::string from = step.src != nullptr ? step.src->id : std::string("store");
      if (step.src != nullptr) {
        ++step.src->served_out_slots;
      } else {
        ++pool->loaded_slots;
      }
      if (sources.empty() || sources.back().first != from) sources.emplace_back(from, 0u);
      ++sources.back().second;
      result.copied_bytes += slot_bytes;
    };
    Slot& d = pool->slots[step.dst_slot];
    const std::string& key = keys[step.key_index];
    const bool usable = copied[i] && step.key_index < valid_keys;
    if (into) {
      // The caller's slot, which it publishes itself: only the count moves.
      if (usable) {
        count_source();
        ++pool->fetched_in_slots;
        ++result.copied_slots;
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
      HolderGainedLocked(pool, entry->second);
      HolderAddedLocked(&entry->second);
      // What was just read from the store is there: evicting this copy, the
      // last one or not, writes nothing. Not so after a Clear during the read.
      if (step.src == nullptr && store_generation_now == store_call.generation) {
        MarkStoredLocked(&entry->second);
      }
      count_source();
      ++pool->fetched_in_slots;
      ++result.copied_slots;
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
  pool->tier_micros += store_micros;
  idle_.notify_all();

  if (store_dropped != 0) {
    MORI_UMBP_INFO(
        "[L1Pool] fetch pool={}: the backing store no longer had {} of the {} keys it was "
        "asked for",
        pool_id, store_dropped, store_steps.size());
  }

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
  CallScope in_call(this, kClear);
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
  // What the owner asked to forget must not come back from the backing store
  // either (an engine clears its pool when the content no longer matches its
  // weights). The store has no way to delete a key, so the group moves to a
  // key space of its own from here on -- one no earlier Clear used, in this
  // process or one before it: what the group wrote stays in the store,
  // unreachable, until the store evicts it. That holds for every pool of the
  // group, whose content means the same thing as this pool's.
  if (store_ != nullptr) {
    Group* group = pool->content_group;
    const uint64_t now =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());
    group->generation = std::max(group->generation + 1, now);
    ForgetStoredLocked(group);
    // Write-through: what the group's other pools still hold is written
    // again, where the group now looks.
    if (store_options_.write_at_publish) {
      for (auto& entry : group->content) {
        for (const Holder& holder : entry.second) {
          if (BackedLocked(holder.pool)) holder.pool->ready.push_back(holder.slot);
        }
      }
      store_wake_.notify_one();
    }
  }
  return Status{};
}

Status PoolManager::Detach(const std::string& pool_id) {
  CallScope in_call(this, kDetach);
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
    stats.device = pool.parts[0].device;
    for (const Part& part : pool.parts) stats.bytes += part.attached ? part.bytes : 0;
    stats.parts = static_cast<uint32_t>(pool.parts.size());
    stats.num_slots = static_cast<uint32_t>(pool.slots.size());
    stats.held = pool.held;
    stats.cached = pool.lru.size + pool.kept.size;
    stats.vacant = static_cast<uint32_t>(pool.vacant.size() + pool.unsettled.size());
    stats.evicted = pool.evicted;
    stats.fetched_in_slots = pool.fetched_in_slots;
    stats.fetched_in_bytes = pool.fetched_in_bytes;
    stats.served_out_slots = pool.served_out_slots;
    stats.copy_micros = pool.copy_micros;
    stats.written_back_slots = pool.written_back_slots;
    stats.written_ahead_slots = pool.written_ahead_slots;
    stats.loaded_slots = pool.loaded_slots;
    stats.lost_slots = pool.lost_slots;
    stats.tier_micros = pool.tier_micros;
    stats.published_slots = pool.published_slots;
    stats.offered_slots = pool.offered_slots;
    stats.unstored_slots = pool.unstored;
    out.push_back(std::move(stats));
  }
  std::sort(out.begin(), out.end(),
            [](const PoolStats& a, const PoolStats& b) { return a.pool_id < b.pool_id; });
  return out;
}

}  // namespace mori::umbp::l1
