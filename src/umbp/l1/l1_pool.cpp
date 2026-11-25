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
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
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

  // Pinned, so a copy to or from a device goes straight to these pages:
  // ordinary memory, written once so that it exists where this thread's
  // memory policy puts it, then registered. (hipHostMalloc was measured at a
  // tenth of a GiB a second; this is several.) The runtime's lock is taken
  // shared: the copies the pools are serving must not wait for gigabytes to
  // be pinned.
  void* AllocateHost(uint64_t bytes, bool pinned, std::string* error) override {
    void* ptr = MemoryApi::AllocateHost(bytes, false, error);
    if (ptr == nullptr || !pinned) return ptr;
    std::memset(ptr, 0, static_cast<size_t>(bytes));
    std::shared_lock<std::shared_mutex> shared(hip_mutex_);
    const hipError_t err = hipHostRegister(ptr, static_cast<size_t>(bytes), hipHostRegisterDefault);
    if (err != hipSuccess) {
      *error = HipError("hipHostRegister", err);
      MemoryApi::FreeHost(ptr, bytes, false);
      return nullptr;
    }
    return ptr;
  }

  void FreeHost(void* base, uint64_t bytes, bool pinned) override {
    if (pinned) {
      std::shared_lock<std::shared_mutex> shared(hip_mutex_);
      (void)hipHostUnregister(base);
    }
    MemoryApi::FreeHost(base, bytes, false);
  }

  bool CopyToHost(int device, const std::vector<Fragment>& fragments, std::string* error) override {
    return HostCopy(device, fragments, hipMemcpyDeviceToHost, error);
  }
  bool CopyFromHost(int device, const std::vector<Fragment>& fragments,
                    std::string* error) override {
    return HostCopy(device, fragments, hipMemcpyHostToDevice, error);
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
  // Every fragment on the device's own stream, waited for once, as CopyBatch.
  bool HostCopy(int device, const std::vector<Fragment>& fragments, hipMemcpyKind kind,
                std::string* error) {
    if (fragments.empty()) return true;
    std::shared_lock<std::shared_mutex> shared(hip_mutex_);
    std::lock_guard<std::mutex> device_lock(DeviceMutex(device));
    ScopedDevice scope(device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(device) + ") failed";
      return false;
    }
    hipStream_t stream = StreamFor(device);
    if (stream == nullptr) {
      *error = "no copy stream on device " + std::to_string(device);
      return false;
    }
    for (const Fragment& f : fragments) {
      const hipError_t err =
          hipMemcpyAsync(f.dst, f.src, static_cast<size_t>(f.bytes), kind, stream);
      if (err != hipSuccess) {
        *error = HipError("hipMemcpyAsync", err);
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

void* MemoryApi::AllocateHost(uint64_t bytes, bool /*pinned*/, std::string* error) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, size_t{2} << 20, static_cast<size_t>(bytes)) != 0 || ptr == nullptr) {
    *error = "host allocation of " + std::to_string(bytes) + " bytes failed";
    return nullptr;
  }
  return ptr;
}

void MemoryApi::FreeHost(void* base, uint64_t /*bytes*/, bool /*pinned*/) { std::free(base); }

bool MemoryApi::CopyToHost(int device, const std::vector<Fragment>& fragments, std::string* error) {
  for (const Fragment& f : fragments) {
    if (!Copy(device, f.dst, device, f.src, f.bytes, error)) return false;
  }
  return true;
}

bool MemoryApi::CopyFromHost(int device, const std::vector<Fragment>& fragments,
                             std::string* error) {
  for (const Fragment& f : fragments) {
    if (!Copy(device, f.dst, device, f.src, f.bytes, error)) return false;
  }
  return true;
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

uint64_t EnvBytes(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  return end == raw ? 0 : static_cast<uint64_t>(value);
}

double EnvFraction(const char* name, double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  return end == raw ? fallback : std::min(1.0, std::max(0.0, value));
}

// The DRAM tier's memory is touched by the thread that allocates it: spread it
// over the NUMA nodes, since the pools it serves are on devices of all of them
// and one node cannot hold a tier sized for the machine. Best effort.
void InterleaveThisThreadsMemory() {
#ifdef SYS_set_mempolicy
  constexpr int kInterleave = 3;  // MPOL_INTERLEAVE
  unsigned long nodes[16];
  std::fill(std::begin(nodes), std::end(nodes), ~0UL);
  (void)syscall(SYS_set_mempolicy, kInterleave, nodes, sizeof(nodes) * 8);
#endif
}

// One line per this many evictions of a pool: how much it evicts and how much
// of that another pool still holds.
constexpr uint64_t kEvictionLogEvery = 16384;
// One line per this many records written to a group's DRAM.
constexpr uint64_t kTierLogEvery = 8192;
// Slots one round of writing ahead copies.
constexpr size_t kAheadBatch = 128;

}  // namespace

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api) : api_(std::move(api)) {
  // On by default: with ranks copying each other's prefixes, plain LRU evicts
  // a conversation's only up-to-date copy while older duplicates of it stay.
  SetEvictionPolicy(EnvCount("UMBP_L1_KEEP_COPIES", 1));
  DramTierOptions tier;
  tier.capacity_bytes = EnvBytes("UMBP_L1_DRAM_BYTES");
  tier.ahead = EnvFraction("UMBP_L1_DRAM_AHEAD", 0.0);
  tier.pinned = EnvCount("UMBP_L1_DRAM_PINNED", 1) != 0;
  tier.write_at_eviction = EnvCount("UMBP_L1_DRAM_WRITE_AT_EVICTION", 1) != 0;
  if (tier.capacity_bytes != 0) SetDramTier(tier);
}

void PoolManager::SetDramTier(const DramTierOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  tier_options_ = options;
  tier_options_.extent_slots = std::max<uint32_t>(1, options.extent_slots);
  MORI_UMBP_INFO(
      "[L1Pool] DRAM tier: {} bytes a group, a key's last L1 copy written {}{}",
      tier_options_.capacity_bytes,
      tier_options_.write_at_eviction ? "at its eviction" : "only ahead of its eviction",
      tier_options_.write_at_eviction && tier_options_.ahead > 0 ? " and ahead of it" : "");
  if (tier_options_.capacity_bytes != 0 && tier_options_.ahead > 0 && tier_threads_.empty()) {
    tier_threads_.emplace_back(&PoolManager::TierWorker, this);
  }
}

void PoolManager::StopTierThreads() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  tier_wake_.notify_all();
  idle_.notify_all();
  for (std::thread& thread : tier_threads_) {
    if (thread.joinable()) thread.join();
  }
  tier_threads_.clear();
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
  StopTierThreads();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) ReleaseMemoryLocked(entry.second.get());
  pools_.clear();
  for (auto& entry : groups_) {
    Tier* tier = entry.second.tier.get();
    if (tier == nullptr) continue;
    for (size_t p = 0; p < tier->chunks.size(); ++p) {
      const uint64_t bytes = tier->chunk_extents * tier->extent_bytes[p];
      for (void* chunk : tier->chunks[p]) api_->FreeHost(chunk, bytes, tier_options_.pinned);
    }
  }
}

// ------------------------------- DRAM tier -------------------------------- //

PoolManager::Tier* PoolManager::TierOfLocked(const Pool* pool) const {
  const auto group = groups_.find(pool->group);
  if (group == groups_.end() || group->second.tier == nullptr) return nullptr;
  Tier* tier = group->second.tier.get();
  // A pool whose slots are cut differently has no records there.
  if (tier->unit_bytes.size() != pool->parts.size()) return nullptr;
  for (size_t p = 0; p < pool->parts.size(); ++p) {
    const std::vector<Region>& regions = pool->parts[p].layout.regions;
    if (regions.size() != tier->unit_bytes[p].size()) return nullptr;
    for (size_t r = 0; r < regions.size(); ++r) {
      if (regions[r].unit_bytes != tier->unit_bytes[p][r]) return nullptr;
    }
  }
  return tier;
}

void PoolManager::MakeTierLocked(Pool* pool) {
  if (tier_options_.capacity_bytes == 0 || stopping_) return;
  Group& group = groups_[pool->group];
  if (group.tier != nullptr) return;
  auto tier = std::make_unique<Tier>();
  tier->extent_slots = tier_options_.extent_slots;
  uint64_t slot_bytes = 0;
  uint64_t largest_extent = 0;
  for (const Part& part : pool->parts) {
    std::vector<uint64_t> starts;
    std::vector<uint64_t> units;
    uint64_t record = 0;
    for (const Region& region : part.layout.regions) {
      starts.push_back(record * tier->extent_slots);
      units.push_back(region.unit_bytes);
      record += region.unit_bytes;
    }
    tier->region_start.push_back(std::move(starts));
    tier->unit_bytes.push_back(std::move(units));
    tier->extent_bytes.push_back(record * tier->extent_slots);
    largest_extent = std::max(largest_extent, record * tier->extent_slots);
    slot_bytes += record;
  }
  // An allocation holds whole extents: about `chunk_bytes` of the largest
  // part, but never more than the tier is to hold.
  const uint64_t extents = tier_options_.capacity_bytes / (slot_bytes * tier->extent_slots);
  tier->chunk_extents = static_cast<uint32_t>(std::max<uint64_t>(
      1, std::min(extents, tier_options_.chunk_bytes / std::max<uint64_t>(1, largest_extent))));
  const uint64_t chunk_records = uint64_t{tier->chunk_extents} * tier->extent_slots;
  const uint64_t chunks = tier_options_.capacity_bytes / (chunk_records * slot_bytes);
  tier->max_records = static_cast<uint32_t>(std::min<uint64_t>(chunks * chunk_records, UINT32_MAX));
  tier->keys.resize(tier->max_records);
  tier->busy.assign(tier->max_records / tier->extent_slots, 0);
  tier->chunks.resize(pool->parts.size());
  // Never reallocated: copies read the chunks already there while more come.
  for (std::vector<void*>& part_chunks : tier->chunks) part_chunks.reserve(chunks);
  MORI_UMBP_INFO(
      "[L1Pool] DRAM tier of group {}: room for {} slots of {} bytes ({} parts), in extents of {}",
      pool->group, tier->max_records, slot_bytes, pool->parts.size(), tier->extent_slots);
  group.tier = std::move(tier);
  if (group.tier->max_records != 0) {
    tier_threads_.emplace_back(&PoolManager::GrowTier, this, pool->group);
  }
}

// Brings the tier's memory in, an allocation at a time: the ring uses what is
// there, so the pools serve from the start and the tier fills as it grows.
void PoolManager::GrowTier(std::string group_id) {
  InterleaveThisThreadsMemory();
  std::unique_lock<std::mutex> lock(mutex_);
  Tier* tier = groups_[group_id].tier.get();
  const uint32_t chunk_records = tier->chunk_extents * tier->extent_slots;
  const auto started = std::chrono::steady_clock::now();
  while (!stopping_ && tier->ready_records < tier->max_records) {
    lock.unlock();
    std::vector<void*> got;
    std::string error;
    for (size_t p = 0; p < tier->extent_bytes.size(); ++p) {
      void* chunk = api_->AllocateHost(tier->chunk_extents * tier->extent_bytes[p],
                                       tier_options_.pinned, &error);
      if (chunk == nullptr) break;
      // Pageable memory is placed when first written: here, not by a copy.
      if (!tier_options_.pinned) std::memset(chunk, 0, tier->chunk_extents * tier->extent_bytes[p]);
      got.push_back(chunk);
    }
    lock.lock();
    if (got.size() != tier->extent_bytes.size()) {
      for (size_t p = 0; p < got.size(); ++p) {
        api_->FreeHost(got[p], tier->chunk_extents * tier->extent_bytes[p], tier_options_.pinned);
      }
      MORI_UMBP_ERROR("[L1Pool] DRAM tier of group {} stops at {} of {} slots: {}", group_id,
                      tier->ready_records, tier->max_records, error);
      return;
    }
    for (size_t p = 0; p < got.size(); ++p) tier->chunks[p].push_back(got[p]);
    tier->ready_records += chunk_records;
  }
  MORI_UMBP_INFO(
      "[L1Pool] DRAM tier of group {}: memory for {} slots is in ({} s)", group_id,
      tier->ready_records,
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started)
          .count());
}

void* PoolManager::RecordAddress(const Tier& tier, size_t part, size_t region, uint32_t record) {
  const uint32_t extent = record / tier.extent_slots;
  const uint32_t row = record % tier.extent_slots;
  char* chunk = static_cast<char*>(tier.chunks[part][extent / tier.chunk_extents]);
  return chunk + uint64_t{extent % tier.chunk_extents} * tier.extent_bytes[part] +
         tier.region_start[part][region] + uint64_t{row} * tier.unit_bytes[part][region];
}

bool PoolManager::LastCopyLocked(const Pool* pool, const Slot& slot) const {
  if (!slot.has_key) return false;
  const auto group = groups_.find(pool->group);
  if (group == groups_.end()) return true;
  const auto holders = group->second.holders.find(slot.key);
  return holders == group->second.holders.end() || holders->second.size() <= 1;
}

bool PoolManager::NeedsWriteBackLocked(const Pool* pool, const Slot& slot) const {
  const Tier* tier = TierOfLocked(pool);
  return tier != nullptr && LastCopyLocked(pool, slot) && tier->index.count(slot.key) == 0;
}

uint32_t PoolManager::ReserveRecordsLocked(Tier* tier, uint32_t count, uint32_t* first) {
  const uint32_t extent_slots = tier->extent_slots;
  const uint32_t extents = tier->ready_records / extent_slots;
  for (uint32_t attempt = 0; attempt < extents; ++attempt) {
    if (tier->cursor >= tier->ready_records) tier->cursor = 0;
    const uint32_t extent = tier->cursor / extent_slots;
    const uint32_t row = tier->cursor % extent_slots;
    if (row == 0) {
      // The ring comes round to this extent: it is reused whole, unless a
      // copy is still reading or writing it.
      if (tier->busy[extent] != 0) {
        tier->cursor = (extent + 1) * extent_slots;
        continue;
      }
      for (uint32_t record = tier->cursor; record < tier->cursor + extent_slots; ++record) {
        std::string& key = tier->keys[record];
        if (key.empty()) continue;
        const auto indexed = tier->index.find(key);
        if (indexed != tier->index.end() && indexed->second == record) tier->index.erase(indexed);
        key.clear();
        ++tier->overwritten;
      }
    }
    const uint32_t got = std::min(count, extent_slots - row);
    *first = tier->cursor;
    tier->cursor += got;
    ++tier->busy[extent];
    return got;
  }
  return 0;
}

size_t PoolManager::WriteBack(Pool* pool, const std::vector<TierSlot>& unordered, bool ahead) {
  if (unordered.empty()) return 0;
  const auto started = std::chrono::steady_clock::now();
  std::vector<TierSlot> slots = unordered;
  std::sort(slots.begin(), slots.end(),
            [](const TierSlot& a, const TierSlot& b) { return a.slot < b.slot; });
  // Neighbouring slots go to neighbouring records of one extent: a run.
  struct Run {
    size_t first = 0;  // in `slots`
    uint32_t count = 0;
    uint32_t record = 0;
  };
  std::vector<Run> runs;
  Tier* tier = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tier = TierOfLocked(pool);
    size_t i = 0;
    while (tier != nullptr && i < slots.size()) {
      size_t n = 1;
      while (i + n < slots.size() && slots[i + n].slot == slots[i + n - 1].slot + 1) ++n;
      size_t done = 0;
      while (done < n) {
        uint32_t record = 0;
        const uint32_t got = ReserveRecordsLocked(tier, static_cast<uint32_t>(n - done), &record);
        if (got == 0) break;
        runs.push_back(Run{i + done, got, record});
        done += got;
      }
      if (done < n) break;  // no memory in the tier yet, or every extent busy
      i += n;
    }
  }
  const size_t num_parts = pool->parts.size();
  std::vector<char> landed(num_parts, 1);
  std::vector<std::string> errors(num_parts);
  if (!runs.empty()) {
    const auto copy_part = [&](size_t p) {
      const Part& part = pool->parts[p];
      std::vector<MemoryApi::Fragment> fragments;
      fragments.reserve(runs.size() * part.layout.regions.size());
      for (const Run& run : runs) {
        for (size_t r = 0; r < part.layout.regions.size(); ++r) {
          const Region& region = part.layout.regions[r];
          const void* src = static_cast<const char*>(part.base) + region.offset +
                            uint64_t{slots[run.first].slot} * region.unit_bytes;
          fragments.push_back(MemoryApi::Fragment{RecordAddress(*tier, p, r, run.record), src,
                                                  run.count * region.unit_bytes, part.device});
        }
      }
      landed[p] = api_->CopyToHost(part.device, fragments, &errors[p]) ? 1 : 0;
    };
    std::vector<std::thread> workers;
    for (size_t p = 1; p < num_parts; ++p) workers.emplace_back(copy_part, p);
    copy_part(0);
    for (std::thread& worker : workers) worker.join();
  }
  const bool ok = std::all_of(landed.begin(), landed.end(), [](char v) { return v != 0; });
  size_t written = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const Run& run : runs) {
    --tier->busy[run.record / tier->extent_slots];
    if (!ok) continue;
    for (uint32_t k = 0; k < run.count; ++k) {
      const std::string& key = slots[run.first + k].key;
      const uint32_t record = run.record + k;
      // The same key written twice (two pools evicting it at once): the newer
      // record is the one found.
      const auto old = tier->index.find(key);
      if (old != tier->index.end()) tier->keys[old->second].clear();
      tier->keys[record] = key;
      tier->index[key] = record;
      ++written;
      if (++tier->written % kTierLogEvery == 0) {
        MORI_UMBP_INFO(
            "[L1Pool] DRAM tier group={} holds={} of {} (memory in for {}) written={} "
            "loaded={} overwritten={}",
            pool->group, tier->index.size(), tier->max_records, tier->ready_records, tier->written,
            tier->loaded, tier->overwritten);
      }
    }
  }
  if (!ok) {
    const auto bad = std::find(landed.begin(), landed.end(), 0) - landed.begin();
    MORI_UMBP_ERROR("[L1Pool] write-back of {} slots of pool={} failed: {}", slots.size(), pool->id,
                    errors[static_cast<size_t>(bad)]);
  }
  (ahead ? pool->written_ahead_slots : pool->written_back_slots) += written;
  if (!ahead) pool->lost_slots += slots.size() - written;
  pool->tier_micros += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  return written;
}

// The slots an eviction takes next are a pool's coldest last copies: write the
// content of up to `ahead` of its slots to DRAM before they are asked for, so
// the Allocate that evicts them has nothing to copy.
bool PoolManager::WriteAhead() {
  Pool* chosen = nullptr;
  std::vector<TierSlot> slots;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : pools_) {
      Pool* pool = entry.second.get();
      const Tier* tier = pool->bound ? TierOfLocked(pool) : nullptr;
      if (tier == nullptr || tier->ready_records == 0) continue;
      size_t budget = static_cast<size_t>(tier_options_.ahead * pool->num_slots);
      // Spare copies stand among the last ones and are passed over: look at
      // no more than a few times the budget, this runs under the pools' lock.
      size_t looked = 4 * budget + kAheadBatch;
      // In the order an eviction takes last copies: those it has passed over
      // first, then the coldest of the rest.
      for (const std::list<uint32_t>* list : {&pool->kept, &pool->lru}) {
        for (uint32_t slot : *list) {
          if (budget == 0 || looked == 0 || slots.size() >= kAheadBatch) break;
          --looked;
          const Slot& s = pool->slots[slot];
          // A copy another pool also holds is dropped at eviction, not written.
          if (!LastCopyLocked(pool, s)) continue;
          --budget;
          if (s.pins != 0 || !Readable(*pool, s) || tier->index.count(s.key) != 0) continue;
          slots.push_back(TierSlot{slot, s.key});
        }
      }
      if (slots.empty()) continue;
      // Pinned: an eviction leaves them alone until the copy is out.
      for (const TierSlot& slot : slots) ++pool->slots[slot.slot].pins;
      ++pool->busy;
      chosen = pool;
      break;
    }
  }
  if (chosen == nullptr) return false;
  WriteBack(chosen, slots, /*ahead=*/true);
  std::lock_guard<std::mutex> lock(mutex_);
  for (const TierSlot& slot : slots) --chosen->slots[slot.slot].pins;
  --chosen->busy;
  idle_.notify_all();
  return true;
}

void PoolManager::TierWorker() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopping_) {
    lock.unlock();
    const bool worked = WriteAhead();
    lock.lock();
    if (!worked && !stopping_) tier_wake_.wait_for(lock, std::chrono::milliseconds(20));
  }
}

void PoolManager::ReleasePart(const Part& part) {
  if (!part.attached) return;
  if (part.external) {
    api_->Unimport(part.device, part.mapped);
  } else {
    api_->Free(part.device, part.base, part.bytes);
  }
}

void PoolManager::ReleaseMemoryLocked(Pool* pool) {
  for (const Part& part : pool->parts) ReleasePart(part);
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
  for (Part& part : pool->parts) part.committed = 0;
}

bool PoolManager::TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                                 bool settled_only, std::vector<TierSlot>* victims) {
  for (auto it = pool->vacant.begin(); it != pool->vacant.end(); ++it) {
    if (pool->slots[*it].pins != 0) continue;
    if (settled_only && !Fillable(*pool, pool->slots[*it])) continue;
    *slot = *it;
    pool->vacant.erase(it);
    return true;
  }
  if (evicted == nullptr) return false;
  const auto takeable = [&](const Slot& s) {
    return s.pins == 0 && (!settled_only || Fillable(*pool, s));
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
    if (victims != nullptr && NeedsWriteBackLocked(pool, s)) {
      // Content its owner is still writing cannot be read out: it is lost. So
      // is content not written ahead when evictions do not wait for DRAM.
      if (tier_options_.write_at_eviction && Readable(*pool, s)) {
        victims->push_back(TierSlot{victim, s.key});
      } else {
        ++pool->lost_slots;
      }
    }
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
  uint64_t bytes = 0;
  for (const Part& part : pool->parts) bytes += part.attached ? part.bytes : 0;
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} parts={} bytes={} evicted={} spare_copy={}", pool_id,
                 pool->parts.size(), bytes, pool->evicted, pool->evicted_spare);
  pools_.erase(pool_id);
}

AttachResult PoolManager::Attach(const std::string& pool_id, const std::string& group, int device,
                                 uint64_t bytes, const ExternalMemory& external,
                                 const PartSpec& spec) {
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
  lock.unlock();
  Part part;
  std::string error;
  const bool ok = adopt ? api_->Import(device, external.handle, external.offset, &part.base,
                                       &part.mapped, &error)
                        : api_->Allocate(device, bytes, &part.base, &part.exported, &error);
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
  part->layout = layout;
  part->bound = true;
  pool->bound = std::all_of(pool->parts.begin(), pool->parts.end(),
                            [](const Part& p) { return p.attached && p.bound; });
  MORI_UMBP_INFO("[L1Pool] bound pool={} part={}/{} slots={} regions={}{}", pool_id, part_index,
                 pool->parts.size(), layout.num_slots, layout.regions.size(),
                 pool->bound ? "" : " (waiting for the other parts)");
  if (!pool->bound) return Status{};
  pool->num_slots = layout.num_slots;
  MakeTierLocked(pool);
  pool->slots.assign(layout.num_slots, Slot{});
  ResetSlotsLocked(pool);
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
  result.num_slots = pool->num_slots;
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
  // Evicted slots whose content has to reach DRAM before the owner is given
  // them: until this call returns they are held, and nobody writes them.
  std::vector<TierSlot> victims;
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
    ++pool->busy;  // Open, a re-bind and Detach wait for the copy
    lock.unlock();
    WriteBack(pool, victims, /*ahead=*/false);
    lock.lock();
    --pool->busy;
    idle_.notify_all();
  }
  return result;
}

Status PoolManager::Commit(const std::string& pool_id, uint64_t committed, uint32_t part) {
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
  pool->committed = slowest;
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

FetchResult PoolManager::Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                               const FetchOptions& options) {
  FetchResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
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
      if (h.pool != pool && SameShape(*h.pool, *pool) && Readable(*h.pool, h.pool->slots[h.slot])) {
        return &h;
      }
    }
    return nullptr;
  };

  // The group's DRAM holds `key`, a record nothing is overwriting.
  Tier* tier = TierOfLocked(pool);
  auto in_tier = [&](const std::string& key, uint32_t* record) {
    if (tier == nullptr) return false;
    const auto found = tier->index.find(key);
    if (found == tier->index.end()) return false;
    if (record != nullptr) *record = found->second;
    return true;
  };

  if (options.probe_only) {
    for (const std::string& key : keys) {
      if (pool->index.find(key) == pool->index.end() && source_of(key) == nullptr &&
          !in_tier(key, nullptr)) {
        break;
      }
      ++result.available;
    }
    return result;
  }
  const bool into = !options.dst_slots.empty();
  if (into && options.dst_slots.size() < keys.size()) {
    result.status = Status::Error("fetch into pool " + pool_id + ": fewer slots than keys");
    return result;
  }
  if (into) {
    // A regranted slot may still have its owner's writes in flight. They land
    // within a forward or two, and the owner's next Commit says so: wait for
    // that rather than leave the caller to recompute what it asked for. Past
    // the bound, the check below stops at the first unsettled slot.
    const auto settle_deadline = std::chrono::steady_clock::now() + kSettleWait;
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
  }

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists -- or, copying
  // into the caller's slots, check that it holds them.
  std::vector<CopyStep> steps;
  std::vector<TierSlot> victims;  // slots evicted for this call, bound for DRAM
  std::vector<uint32_t> local_pins;
  if (!into) {
    // Every key of the call this pool already holds is pinned before any
    // destination is taken: room for an earlier key is never made by evicting
    // a later one.
    for (const std::string& key : keys) {
      auto local = pool->index.find(key);
      if (local == pool->index.end()) continue;
      ++pool->slots[local->second].pins;
      local_pins.push_back(local->second);
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
      const Holder* source = source_of(key);
      uint32_t record = 0;
      if (source == nullptr && !in_tier(key, &record)) break;
      if (!Fillable(*pool, pool->slots[dst])) {
        // Allocate may regrant a slot before its owner's writes complete:
        // owner-local reuse is ordered on that stream, but this copy is not.
        // Stop at the landed prefix; the caller still holds every destination.
        break;
      }
      if (source != nullptr) {
        ++source->pool->slots[source->slot].pins;
        ++source->pool->busy;
        steps.push_back(CopyStep{source->pool, source->slot, dst, i, 0});
      } else {
        // From DRAM: the record's extent is not reused while the copy reads it.
        ++tier->busy[record / tier->extent_slots];
        steps.push_back(CopyStep{nullptr, record, dst, i, record});
      }
      ++pool->busy;
      result.slots.push_back(dst);
      continue;
    }
    auto local = pool->index.find(key);
    if (local != pool->index.end()) {
      result.slots.push_back(local->second);
      continue;
    }
    const Holder* source = nullptr;
    auto holders = group.holders.find(key);
    if (holders != group.holders.end()) {
      for (const Holder& h : holders->second) {
        // Not this pool, and not a slot its owner may still be writing.
        if (h.pool != pool && SameShape(*h.pool, *pool) &&
            Readable(*h.pool, h.pool->slots[h.slot])) {
          source = &h;
          break;
        }
      }
    }
    uint32_t record = 0;
    if (source == nullptr && !in_tier(key, &record)) break;
    uint32_t dst = 0;
    if (!TakeFreeLocked(pool, &dst, &result.evicted_keys, /*settled_only=*/true, &victims)) break;
    pool->slots[dst].state = SlotState::kFilling;
    if (source != nullptr) {
      ++source->pool->slots[source->slot].pins;
      ++source->pool->busy;
      steps.push_back(CopyStep{source->pool, source->slot, dst, i, 0});
    } else {
      ++tier->busy[record / tier->extent_slots];
      steps.push_back(CopyStep{nullptr, record, dst, i, record});
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
  // And the same from DRAM records, which are host memory.
  std::vector<std::vector<MemoryApi::Fragment>> host_fragments(num_parts);
  // The slots this call evicted to make room go to DRAM before it fills them.
  if (!victims.empty()) WriteBack(pool, victims, /*ahead=*/false);
  size_t run_start = 0;
  bool failed = false;
  while (run_start < steps.size()) {
    size_t run_end = run_start + 1;
    // A run from DRAM stays inside one extent, where records are neighbours.
    while (run_end < steps.size() && steps[run_end].src == steps[run_start].src &&
           steps[run_end].src_slot == steps[run_end - 1].src_slot + 1 &&
           steps[run_end].dst_slot == steps[run_end - 1].dst_slot + 1 &&
           (steps[run_start].src != nullptr || steps[run_end].record / tier->extent_slots ==
                                                   steps[run_start].record / tier->extent_slots)) {
      ++run_end;
    }
    const CopyStep& first = steps[run_start];
    const uint64_t run_slots = run_end - run_start;
    for (size_t p = 0; p < num_parts; ++p) {
      const Part& dst_part = pool->parts[p];
      for (size_t r = 0; r < dst_part.layout.regions.size(); ++r) {
        const Region& dst_region = dst_part.layout.regions[r];
        const uint64_t unit = dst_region.unit_bytes;
        void* dst =
            static_cast<char*>(dst_part.base) + dst_region.offset + uint64_t{first.dst_slot} * unit;
        if (first.src == nullptr) {
          host_fragments[p].push_back(MemoryApi::Fragment{
              dst, RecordAddress(*tier, p, r, first.record), run_slots * unit, dst_part.device});
        } else {
          const Part& src_part = first.src->parts[p];
          const Region& src_region = src_part.layout.regions[r];
          const void* src = static_cast<const char*>(src_part.base) + src_region.offset +
                            uint64_t{first.src_slot} * unit;
          fragments[p].push_back(MemoryApi::Fragment{dst, src, run_slots * unit, src_part.device});
        }
        result.copied_bytes += run_slots * unit;
      }
    }
    run_start = run_end;
  }
  // One batch a part for the whole call, landed before it returns: nothing is
  // handed to the requester before then. The parts are on different devices,
  // so their batches run side by side; a slot counts only if every part of it
  // landed.
  if (!steps.empty()) {
    std::vector<std::string> errors(num_parts);
    std::vector<char> landed(num_parts, 0);
    const auto copy_part = [&](size_t p) {
      const int device = pool->parts[p].device;
      landed[p] = (fragments[p].empty() || api_->CopyBatch(device, fragments[p], &errors[p])) &&
                          api_->CopyFromHost(device, host_fragments[p], &errors[p])
                      ? 1
                      : 0;
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
    if (step.src != nullptr) {
      --step.src->slots[step.src_slot].pins;
      --step.src->busy;
    } else {
      --tier->busy[step.record / tier->extent_slots];
    }
    --pool->busy;
    // Who a landed slot came from: another pool, or DRAM.
    const auto count_source = [&] {
      const std::string from = step.src != nullptr ? step.src->id : std::string("dram");
      if (step.src != nullptr) {
        ++step.src->served_out_slots;
      } else {
        ++pool->loaded_slots;
        ++tier->loaded;
      }
      if (sources.empty() || sources.back().first != from) sources.emplace_back(from, 0u);
      ++sources.back().second;
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
      count_source();
      ++pool->fetched_in_slots;
      ++result.copied_slots;
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
      s.state = SlotState::kVacant;
      pool->vacant.insert(slot);
    }
  }
  // What the owner asked to forget must not come back from DRAM either (an
  // engine clears its pool when the content no longer matches its weights).
  if (Tier* tier = TierOfLocked(pool)) {
    for (const auto& entry : tier->index) tier->keys[entry.second].clear();
    tier->index.clear();
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
    stats.device = pool.parts[0].device;
    for (const Part& part : pool.parts) stats.bytes += part.attached ? part.bytes : 0;
    stats.parts = static_cast<uint32_t>(pool.parts.size());
    stats.num_slots = static_cast<uint32_t>(pool.slots.size());
    stats.held = pool.held;
    stats.cached = static_cast<uint32_t>(pool.lru.size() + pool.kept.size());
    stats.vacant = static_cast<uint32_t>(pool.vacant.size());
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
    if (const Tier* tier = TierOfLocked(&pool)) {
      stats.tier_records = tier->index.size();
      stats.tier_capacity = tier->ready_records;
    }
    out.push_back(std::move(stats));
  }
  std::sort(out.begin(), out.end(),
            [](const PoolStats& a, const PoolStats& b) { return a.pool_id < b.pool_id; });
  return out;
}

}  // namespace mori::umbp::l1
