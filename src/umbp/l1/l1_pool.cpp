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
#include <charconv>
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

CopyOutcome TryCopyBatch(MemoryApi& api, int device,
                         const std::vector<MemoryApi::Fragment>& fragments) {
  try {
    return api.CopyBatchTracked(device, fragments);
  } catch (const std::exception& e) {
    return {false, false, e.what()};
  } catch (...) {
    return {false, false, "copy backend threw without a completion proof"};
  }
}

Status TryUnimport(MemoryApi& api, int device, void* mapped) {
  try {
    api.Unimport(device, mapped);
    return {};
  } catch (const std::exception& e) {
    return Status::Error(e.what());
  } catch (...) {
    return Status::Error("unmap failed without a completion acknowledgement");
  }
}

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
// Bound concurrent receipts, not the lifetime number of reads. Admission epochs
// fence late Acquire calls before released tombstones are reclaimed.
constexpr size_t kPeerReceipts = 65536;
constexpr size_t kPendingReleases = 1024;
constexpr auto kReadEpochDuration = std::chrono::seconds(60);
uint64_t OperationEpoch(const std::string& operation) {
  const size_t colon = operation.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == operation.size()) return 0;
  uint64_t epoch = 0;
  const auto parsed = std::from_chars(operation.data(), operation.data() + colon, epoch);
  return parsed.ec == std::errc{} && parsed.ptr == operation.data() + colon ? epoch : 0;
}
PeerReleaseResult ReleaseDrained(PeerApi& peer, const std::string& operation,
                                 const std::string& incarnation) {
  try {
    return peer.ReleaseRead(operation, incarnation, true);
  } catch (const std::exception& error) {
    return {Status::Error(error.what()), {}};
  } catch (...) {
    return {Status::Error("release threw without an acknowledgement"), {}};
  }
}
bool ValidPeerShape(const std::vector<std::string>& keys, const std::vector<uint64_t>& units) {
  return keys.size() <= kPeerKeys && !units.empty() && units.size() <= kPeerRegions &&
         std::all_of(units.begin(), units.end(), [](uint64_t n) { return n != 0; }) &&
         std::all_of(keys.begin(), keys.end(),
                     [](const std::string& k) { return !k.empty() && k.size() <= 1024; });
}

}  // namespace

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api, std::function<Clock::time_point()> now)
    : api_(std::move(api)),
      incarnation_(NewIncarnation()),
      now_(std::move(now)),
      epoch_expires_(now_() + kReadEpochDuration) {
  // On by default: with ranks copying each other's prefixes, plain LRU evicts
  // a conversation's only up-to-date copy while older duplicates of it stay.
  SetEvictionPolicy(EnvCount("UMBP_L1_KEEP_COPIES", 1));
  settled_first_ = EnvCount("UMBP_L1_GRANT_SETTLED_FIRST", 1) != 0;
  MORI_UMBP_INFO("[L1Pool] grants take settled slots first: {}", settled_first_ ? "on" : "off");
}

uint64_t PoolManager::ReadEpochLocked() {
  const auto now = now_();
  if (now >= epoch_expires_) {
    ++read_epoch_;
    epoch_expires_ = now + kReadEpochDuration;
    for (auto it = read_grants_.begin(); it != read_grants_.end();) {
      if (it->second.released || it->second.sources.empty())
        it = read_grants_.erase(it);
      else
        ++it;  // An admission deadline is never permission to release a live pin.
    }
  }
  return read_epoch_;
}

bool PoolManager::ReserveRelease(const std::string& operation, std::shared_ptr<PeerApi> peer,
                                 const std::string& incarnation) {
  std::lock_guard<std::mutex> lock(releases_mutex_);
  if (stop_releases_ || pending_releases_.size() >= kPendingReleases) return false;
  pending_releases_.emplace(operation, PendingRelease{std::move(peer), incarnation});
  if (!release_thread_.joinable()) release_thread_ = std::thread(&PoolManager::RetryReleases, this);
  return true;
}

void PoolManager::FinishRelease(const std::string& operation, const PeerReleaseResult& result) {
  std::lock_guard<std::mutex> lock(releases_mutex_);
  auto found = pending_releases_.find(operation);
  if (found == pending_releases_.end()) return;
  auto& pending = found->second;
  if (result.status.ok ||
      (!result.incarnation.empty() && result.incarnation != pending.incarnation)) {
    pending_releases_.erase(found);
  } else {
    pending.ready = true;
    pending.retry_at =
        Clock::now() + std::chrono::milliseconds(10u << std::min(pending.attempts++, 7u));
  }
  releases_changed_.notify_all();
}

void PoolManager::RetryReleases() {
  std::unique_lock<std::mutex> lock(releases_mutex_);
  while (!stop_releases_) {
    auto next = pending_releases_.end();
    for (auto it = pending_releases_.begin(); it != pending_releases_.end(); ++it) {
      if (it->second.ready &&
          (next == pending_releases_.end() || it->second.retry_at < next->second.retry_at))
        next = it;
    }
    if (next == pending_releases_.end()) {
      releases_changed_.wait(lock);
      continue;
    }
    if (next->second.retry_at > Clock::now()) {
      const auto retry_at = next->second.retry_at;
      releases_changed_.wait_until(lock, retry_at);
      continue;
    }
    const std::string operation = next->first;
    const PendingRelease pending = next->second;
    next->second.ready = false;
    lock.unlock();
    const auto result = ReleaseDrained(*pending.peer, operation, pending.incarnation);
    FinishRelease(operation, result);
    lock.lock();
  }
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
  {
    std::unique_lock<std::mutex> lock(releases_mutex_);
    // Give confirmed-drained releases a bounded final retry opportunity. Unknown
    // copies are never made ready and remain pinned on their owners.
    releases_changed_.wait_for(lock, std::chrono::seconds(2),
                               [&] { return pending_releases_.empty(); });
    stop_releases_ = true;
    releases_changed_.notify_all();
  }
  if (release_thread_.joinable()) release_thread_.join();
  if (!pending_releases_.empty()) {
    MORI_UMBP_WARN(
        "[L1Peer] shutdown with {} unconfirmed reads; owners retain pins until coordinated drain",
        pending_releases_.size());
  }
  std::lock_guard<std::mutex> lock(mutex_);
  bool retained = false;
  for (auto& entry : pools_) {
    Pool& pool = *entry.second;
    if (pool.busy != 0 || pool.quarantined) {
      retained = true;
      continue;  // No completion proof: do not unmap memory still used by a peer.
    }
    if (pool.external) {
      const Status unmapped = TryUnimport(*api_, pool.device, pool.mapped);
      if (!unmapped.ok) {
        retained = true;
        MORI_UMBP_WARN("[L1Pool] shutdown retaining pool={} after unmap failure: {}", pool.id,
                       unmapped.error);
      }
    } else {
      api_->Free(pool.device, pool.base, pool.bytes);
    }
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
  pool->probe_hint = {};
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
    if (pool->slots[*it].pins != 0 || pool->slots[*it].writers != 0) continue;
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
      if (pool->slots[it->second].pins != 0 || pool->slots[it->second].writers != 0) continue;
      *slot = it->second;
      pool->unsettled.erase(it);
      return true;
    }
    return false;
  };
  // Asked ahead of need: nothing is evicted, and nothing unsettled given.
  if (evicted == nullptr) return false;
  const auto takeable = [&](const Slot& s) {
    return s.pins == 0 && s.writers == 0 && (!settled_only || Fillable(*pool, s));
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

Status PoolManager::WaitIdleLocked(std::unique_lock<std::mutex>& lock, const std::string& pool_id,
                                   const char* operation) {
  // A peer may disappear while holding a read grant. Never strand a control
  // RPC indefinitely and never interpret a timeout as permission to unmap.
  Pool* original = FindLocked(pool_id);
  const uint64_t serial = original != nullptr ? original->serial : 0;
  const auto idle = [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->serial != serial || p->quarantined || p->busy == 0;
  };
  bool ready = true;
  if (original != nullptr && original->external_kind == "hip_fabric")
    ready = idle_.wait_for(lock, kPinnedWait, idle);
  else
    idle_.wait(lock, idle);  // Preserve the default IPC path's wait-for-idle contract.
  Pool* pool = FindLocked(pool_id);
  if (pool != nullptr && pool->serial != serial) {
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
  if (pool->external_kind == "hip_fabric") {
    // Keep this generation unavailable until Unimport confirms completion.
    // It may wait for another GPU's copy; unrelated control RPCs must not.
    pool->quarantined = true;
    idle_.notify_all();
    lock.unlock();
    const Status unmapped = TryUnimport(*api_, pool->device, pool->mapped);
    lock.lock();
    if (!unmapped.ok) {
      return Status::Error("detach: " + unmapped.error + "; retain the allocation");
    }
  } else if (pool->external) {
    api_->Unimport(pool->device, pool->mapped);
  } else {
    api_->Free(pool->device, pool->base, pool->bytes);
  }
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} device={} bytes={}{} evicted={} spare_copy={}",
                 pool_id, pool->device, pool->bytes, pool->external ? " (adopted, unmapped)" : "",
                 pool->evicted, pool->evicted_spare);
  pools_.erase(pool_id);
  idle_.notify_all();
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
  const uint64_t offset = adopt ? external.offset : 0;
  std::unique_lock<std::mutex> lock(mutex_);
  if (attaching_.count(pool_id) != 0) {
    result.status = Status::Error("pool " + pool_id + " is being attached");
    return result;
  }
  if (Pool* existing = FindLocked(pool_id)) {
    if (existing->quarantined) {
      result.status = Status::Error("pool is quarantined; unmap or copy completion is unconfirmed");
      return result;
    }
    if (existing->group == group && existing->device == device && existing->bytes == bytes &&
        existing->external == adopt && existing->external_handle == external.handle &&
        existing->external_kind == external.kind && existing->external_offset == offset) {
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
  pool->serial = next_serial_++;
  pool->group = group;
  pool->device = device;
  pool->base = base;
  pool->bytes = bytes;
  pool->exported = exported;
  pool->external = adopt;
  pool->mapped = mapped;
  pool->external_handle = external.handle;
  pool->external_kind = external.kind;
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
    // Divided rather than multiplied out: offset + slots * unit can wrap.
    if (region.unit_bytes == 0 || region.offset > pool->bytes ||
        region.unit_bytes > (pool->bytes - region.offset) / layout.num_slots) {
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
  result.status = WaitIdleLocked(lock, pool_id, "open");
  if (!result.status.ok) return result;
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  ResetSlotsLocked(pool);
  result.num_slots = pool->layout.num_slots;
  result.memory_kind = pool->external
                           ? (pool->external_kind.empty() ? "hip_ipc" : pool->external_kind)
                           : pool->exported.kind;
  result.incarnation = incarnation_;
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
  const bool fabric = api_->SupportsFabric();
  if (fabric) {
    // Reject a whole batch before mutating it if it would reuse an in-flight
    // destination or change bytes pinned by another tray.
    for (const Op& op : ops) {
      if (op.slot >= pool->slots.size()) continue;
      const Slot& slot = pool->slots[op.slot];
      // Hold of cached content is a read-only prefix claim. Reuse for different
      // content must go through Allocate, which excludes pinned slots.
      if (slot.writers != 0 || (slot.pins != 0 && op.kind == OpKind::kPublish &&
                                (slot.content == nullptr || slot.content->first != op.key))) {
        return Status::Error("apply would reuse a slot with an in-flight copy");
      }
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
        if (fabric && s.pins != 0 && s.content == nullptr) return fail("reuse of a slot pinned by a reader");
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
        if (fabric && s.pins != 0) return fail("change of content pinned by a reader");
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
      if (h.pool != pool && !h.pool->quarantined && h.pool->slots[h.slot].writers == 0 &&
          Readable(*h.pool, h.pool->slots[h.slot]))
        return &h;
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
  const bool fabric = api_->SupportsFabric();
  // Retain the allocation and its layout through the settle wait as well as
  // the unlocked copy, including calls with only local hits or no copy steps.
  ++pool->busy;
  if (into) {
    // A regranted slot may still have its owner's writes in flight. They land
    // within a forward or two, and the owner's next Commit says so: wait for
    // that rather than leave the caller to recompute what it asked for. Past
    // the bound, the check below stops at the first unsettled slot.
    const auto settle_started = std::chrono::steady_clock::now();
    const auto settle_deadline = settle_started + kSettleWait;
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
      if (dst >= pool->slots.size() || pool->slots[dst].state != SlotState::kHeld ||
          (fabric && pool->slots[dst].content != nullptr)) {
        result.status = Status::Error("fetch into pool " + pool_id + ": slot " +
                                      std::to_string(dst) + " is not a writable held slot");
        break;
      }
      const Holder* source = source_of(holders_of(key));
      if (source == nullptr) break;
      if ((fabric && (pool->slots[dst].pins != 0 || pool->slots[dst].writers != 0)) ||
          !Fillable(*pool, pool->slots[dst])) {
        // Allocate may regrant a slot before its owner's writes complete:
        // owner-local reuse is ordered on that stream, but this copy is not.
        // Stop at the landed prefix; the caller still holds every destination.
        break;
      }
      ++source->pool->slots[source->slot].pins;
      ++source->pool->busy;
      if (fabric) ++pool->slots[dst].writers;
      steps.push_back(CopyStep{source->pool, source->slot, dst, i});
      result.slots.push_back(dst);
      continue;
    }
    const Holders* holders = holders_of(key);
    const uint32_t local = local_of(holders);
    if (local != kNoSlot && pool->slots[local].writers == 0) {
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
    steps.push_back(CopyStep{source.pool, source.slot, dst, i});
    result.slots.push_back(dst);
  }

  // Copy without the lock. The call retains the destination and each step
  // retains its source, so Detach, Open and re-bind wait for completion.
  // Keep the lock when there is nothing to copy.
  const bool copying = !steps.empty();
  if (copying) lock.unlock();
  const auto started = std::chrono::steady_clock::now();
  std::vector<MemoryApi::Fragment> fragments;
  size_t run_start = 0;
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
  CopyOutcome outcome{true, true, {}};
  if (!fragments.empty()) {
    outcome = TryCopyBatch(*api_, pool->device, fragments);
    if (!outcome.ok) result.copied_bytes = 0;
  }
  result.copy_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  if (copying) lock.lock();

  if (!outcome.drained) {
    pool->quarantined = true;
    result.completion_unknown = true;
    result.status = Status::Error("copy completion unknown: " + outcome.error);
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
    Slot& d = pool->slots[step.dst_slot];
    const std::string& key = keys[step.key_index];
    const bool usable = outcome.ok && step.key_index < valid_keys;
    if (into) {
      if (fabric) --d.writers;
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
  --pool->busy;
  result.slots.resize(valid_keys);
  pool->fetched_in_bytes += result.copied_bytes;
  pool->copy_micros += result.copy_micros;
  idle_.notify_all();

  if (!outcome.ok) {
    MORI_UMBP_ERROR("[L1Pool] fetch into pool={} stopped after {} of {} keys: {}", pool_id,
                    valid_keys, keys.size(), outcome.error);
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
  Status idle = WaitIdleLocked(lock, pool_id, "clear");
  if (!idle.ok) return idle;
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound || pool->quarantined) {
    return Status::Error("pool " + pool_id + " is not attached and bound");
  }
  pool->probe_hint = {};
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
  auto found = g->second.content.find(key);
  if (found == g->second.content.end()) return nullptr;
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
  result.read_epoch = ReadEpochLocked();
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
  const uint64_t epoch = ReadEpochLocked();
  if (request.expected_incarnation != incarnation_ || request.operation_id.empty() ||
      request.operation_id.size() > 256 || OperationEpoch(request.operation_id) != epoch ||
      !ValidPeerShape(request.keys, request.unit_bytes)) {
    result.status =
        Status::Error("invalid native L1 read, expired admission epoch or stale owner incarnation");
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
    result.status =
        Status::Error("native L1 read receipt budget exhausted in current admission epoch");
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
  std::unordered_map<std::string, uint32_t> handles;
  result.regions.reserve(request.keys.size() * request.unit_bytes.size());
  for (const std::string& key : request.keys) {
    const Holder* source = PeerSourceLocked(request.group, key);
    if (source == nullptr) break;
    const Pool& p = *source->pool;
    const auto [handle, added] = handles.try_emplace(p.external_handle, result.handles.size());
    if (added) result.handles.push_back(p.external_handle);
    for (const Region& r : p.layout.regions) {
      result.regions.push_back(PeerReadRegion{
          handle->second, p.external_offset + r.offset + uint64_t{source->slot} * r.unit_bytes,
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
  const uint64_t epoch = ReadEpochLocked();
  const uint64_t operation_epoch = OperationEpoch(operation_id);
  if (expected_incarnation != incarnation_ || operation_id.empty() || operation_id.size() > 256 ||
      operation_epoch == 0 || operation_epoch > epoch || !drained) {
    return Status::Error("read release requires matching owner and confirmed drain");
  }
  auto found = read_grants_.find(operation_id);
  if (found == read_grants_.end()) {
    if (operation_epoch < epoch) return {};  // Its Acquire can no longer be accepted.
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
  if (operation_epoch < epoch) {
    read_grants_.erase(found);
  } else {
    grant = ReadGrant{};
    grant.released = true;
  }
  idle_.notify_all();
  return {};
}

FetchResult PoolManager::Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                               const FetchOptions& options) {
  // Validate shape before local work and reject overlapping FABRIC writes.
  if (!options.probe_only && !options.dst_slots.empty()) {
    if (options.dst_slots.size() < keys.size()) {
      FetchResult bad;
      bad.status = Status::Error("fewer destination slots than keys");
      return bad;
    }
    if (api_->SupportsFabric()) {
      std::unordered_set<uint32_t> unique;
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!unique.insert(options.dst_slots[i]).second) {
          FetchResult bad;
          bad.status = Status::Error("overlapping native L1 destination slots");
          return bad;
        }
      }
    }
  }
  std::unique_lock<std::mutex> lock(mutex_);
  // Snapshot configuration under the same lock as ConfigurePeers.
  const auto peers = peers_;
  Pool* pool = FindLocked(pool_id);
  PeerHint hint;
  if (pool) {
    if (!options.probe_only) hint = std::move(pool->probe_hint);
    pool->probe_hint = {};
  }
  if (peers.empty() || (!options.probe_only && options.dst_slots.empty()) || keys.empty()) {
    lock.unlock();
    return FetchLocal(pool_id, keys, options);
  }
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
  const size_t local_prefix = leading;
  auto finish = [&] {
    std::lock_guard<std::mutex> done(mutex_);
    --pool->busy;
    idle_.notify_all();
  };
  if (!result.status.ok || leading == keys.size()) {
    finish();
    return result;
  }
  // Bound each read, not the whole fetch. Finish and release one window before
  // acquiring the next, stopping at the first incomplete or failed window.
  while (leading < keys.size()) {
    const size_t count = std::min(kPeerKeys, keys.size() - leading);
    std::vector<std::string> suffix(keys.begin() + leading, keys.begin() + leading + count);
    if (!ValidPeerShape(suffix, units)) break;
    // Use one peer per window. Resolve and pin again for each window, preserving
    // owner-local timestamp semantics across different hosts.
    std::shared_ptr<PeerApi> chosen;
    PeerProbeResult best;
    // A hint may cover a caller-shortened request, but must never truncate a
    // larger one. Consume it once so a stale hint can retry normal discovery.
    const bool hinted = hint.peer && hint.keys.size() >= count &&
                        std::equal(suffix.begin(), suffix.end(), hint.keys.begin());
    if (hinted) {
      chosen = hint.peer;
      best.incarnation = hint.incarnation;
      best.read_epoch = hint.read_epoch;
      best.available = count;
    } else {
      for (const auto& peer : peers) {
        PeerProbeResult probe = peer->Probe(group, suffix, units);
        if (probe.status.ok && !probe.incarnation.empty() && probe.available <= count &&
            probe.available > best.available) {
          chosen = peer;
          best = std::move(probe);
          if (best.available == count) break;
        }
      }
    }
    hint = {};
    if (!chosen) break;
    if (options.probe_only) {
      if (leading == local_prefix) {
        std::lock_guard<std::mutex> save(mutex_);
        pool->probe_hint = {{suffix.begin(), suffix.begin() + best.available},
                            chosen,
                            best.incarnation,
                            best.read_epoch};
      }
      result.available += best.available;
      leading += best.available;
      if (best.available < count) break;
      continue;
    }
    lock.lock();
    size_t reserved = 0;
    for (; reserved < best.available; ++reserved) {
      const uint32_t id = options.dst_slots[leading + reserved];
      if (id >= pool->slots.size() || pool->slots[id].state != SlotState::kHeld ||
          pool->slots[id].content != nullptr || pool->slots[id].writers != 0 ||
          pool->slots[id].pins != 0 ||
          !Fillable(*pool, pool->slots[id]))
        break;
      ++pool->slots[id].writers;
    }
    const std::string operation =
        std::to_string(best.read_epoch) + ":" + incarnation_ + std::to_string(++next_read_);
    lock.unlock();
    if (reserved == 0) break;
    if (!ReserveRelease(operation, chosen, best.incarnation)) {
      lock.lock();
      for (size_t i = 0; i < reserved; ++i) --pool->slots[options.dst_slots[leading + i]].writers;
      lock.unlock();
      break;  // Never Acquire without room to retain its release obligation.
    }
    suffix.resize(reserved);
    PeerReadRequest request{operation, best.incarnation, group, suffix, units};
    PeerReadResult read = chosen->AcquireRead(request);
    std::vector<void*> mappings;
    std::vector<MemoryApi::Fragment> fragments;
    std::string error;
    bool valid = read.status.ok && read.incarnation == best.incarnation &&
                 read.available <= reserved && read.regions.size() == read.available * units.size();
    // A responding new incarnation rejects before creating a grant. Transport
    // failures have no incarnation and still require the conservative release.
    const bool owner_changed = hinted && !read.status.ok && !read.incarnation.empty() &&
                               read.incarnation != best.incarnation && read.available == 0 &&
                               read.regions.empty() && read.handles.empty();
    uint64_t bytes = 0;
    if (valid) {
      // Descriptor imports are reused within this window only. Mapping a descriptor
      // creates an address view, not a payload allocation/copy or a staging pool.
      struct View {
        void* base = nullptr;
        uint64_t bytes = 0;
      };
      std::vector<View> views(read.handles.size());
      // Build runs per layout region before the backend checks and plans copies.
      // Every original range is still validated; never merge descriptor views.
      for (size_t r = 0; valid && r < units.size(); ++r) {
        const PeerReadRegion* previous = nullptr;
        uint32_t previous_dst = 0;
        for (size_t k = 0; k < read.available; ++k) {
          const PeerReadRegion& region = read.regions[k * units.size() + r];
          if (region.handle_index >= views.size() || region.bytes != units[r]) {
            valid = false;
            break;
          }
          View& view = views[region.handle_index];
          const std::string& handle = read.handles[region.handle_index];
          // Decode once per descriptor for range checks, not once per region.
          if (view.bytes == 0 && !GetFabricViewBytes(handle, &view.bytes, &error)) {
            valid = false;
            break;
          }
          if (region.offset > view.bytes || region.bytes > view.bytes - region.offset) {
            error = "native L1 pool range exceeds the exported FABRIC view";
            valid = false;
            break;
          }
          if (view.base == nullptr) {
            void* mapped = nullptr;
            if (!api_->Import(pool->device, handle, 0, &view.base, &mapped, &error)) {
              valid = false;
              break;
            }
            mappings.push_back(mapped);
          }
          const uint32_t dst = options.dst_slots[leading + k];
          const Region& layout = pool->layout.regions[r];
          if (previous && previous->handle_index == region.handle_index &&
              previous->offset + previous->bytes == region.offset &&
              uint64_t{previous_dst} + 1 == dst) {
            fragments.back().bytes += region.bytes;
          } else {
            fragments.push_back(MemoryApi::Fragment{
                static_cast<char*>(pool->base) + layout.offset + uint64_t{dst} * layout.unit_bytes,
                static_cast<const char*>(view.base) + region.offset, region.bytes, pool->device});
          }
          previous = &region;
          previous_dst = dst;
          bytes += region.bytes;
        }
      }
    }
    // A stale hint can fail without an owner restart (for example, a layout
    // rebind). Retry discovery only after the no-copy cleanup and Release below.
    const bool retry_probe = hinted && (owner_changed || !valid || read.available < reserved);
    const auto started = std::chrono::steady_clock::now();
    CopyOutcome outcome{false, true, "remote read unavailable: " + error};
    if (valid && !retry_probe) outcome = TryCopyBatch(*api_, pool->device, fragments);
    const auto micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() - started)
                                                  .count());
    Status unmapped;
    if (outcome.drained) {
      for (void* mapped : mappings) {
        unmapped = TryUnimport(*api_, pool->device, mapped);
        if (!unmapped.ok) break;
      }
    }
    if (!outcome.drained || !unmapped.ok) {
      lock.lock();
      pool->quarantined = true;
      result.status = Status::Error("FABRIC copy or cleanup completion unknown: " +
                                    (outcome.drained ? unmapped.error : outcome.error));
      result.completion_unknown = true;
      result.slots.clear();
      // Keep the remote source pin, mappings, destination writers and busy guard.
      // No fallback or recomputation into these destinations is permitted.
      idle_.notify_all();
      return result;
    }
    // Even a lost Acquire reply can have pinned an owner slot. No submission or
    // confirmed drain makes a release safe; owner tombstones fence delayed RPCs.
    PeerReleaseResult release =
        owner_changed ? PeerReleaseResult{} : ReleaseDrained(*chosen, operation, best.incarnation);
    FinishRelease(operation, release);
    if (!release.status.ok) {
      MORI_UMBP_WARN(
          "[L1Peer] release unconfirmed operation={} owner={} error={}; source stays pinned",
          operation, best.incarnation, release.status.error);
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
    lock.unlock();
    if (retry_probe && release.status.ok)
      continue;  // No copy was submitted; writers and grant are gone.
    // An unconfirmed release may retain the entire grant on the owner. Do not
    // acquire another window; the confirmed copy is still a usable prefix.
    if (!release.status.ok || !outcome.ok || read.available < count) break;
    leading += read.available;
  }
  finish();
  return result;
}

}  // namespace mori::umbp::l1
