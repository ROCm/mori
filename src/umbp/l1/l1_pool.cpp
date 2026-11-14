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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp::l1 {

namespace {

// How long Allocate waits for a slot that is free but pinned by a copy.
constexpr std::chrono::milliseconds kPinnedWait{2000};
// How long the tier worker sleeps when nothing asks for it.
constexpr std::chrono::milliseconds kWorkerIdle{200};
// Slots one round of offloading ahead copies; small, so an eviction that needs
// one of them does not wait long for its pin.
constexpr size_t kAheadBatch = 16;

// The store is code this file does not own, reaching disks and other nodes. A
// failure there -- an exception included -- must cost the engine a cache miss
// or a dropped copy, never an error: these calls are made in the middle of
// slot bookkeeping that has to be completed either way.
template <typename Fn>
std::vector<bool> Guarded(const char* what, size_t count, Fn&& fn) {
  try {
    std::vector<bool> out = fn();
    out.resize(count, false);
    return out;
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[L1Pool] lower store {} failed: {}", what, e.what());
  } catch (...) {
    MORI_UMBP_ERROR("[L1Pool] lower store {} failed", what);
  }
  return std::vector<bool>(count, false);
}

template <typename Fn>
void GuardedVoid(const char* what, Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[L1Pool] lower store {} failed: {}", what, e.what());
  } catch (...) {
    MORI_UMBP_ERROR("[L1Pool] lower store {} failed", what);
  }
}

// Where each region's unit starts inside a stored slot.
std::vector<uint64_t> RegionStarts(const Layout& layout) {
  std::vector<uint64_t> starts;
  uint64_t at = 0;
  for (const Region& region : layout.regions) {
    starts.push_back(at);
    at += region.unit_bytes;
  }
  return starts;
}

// Regions `rows` long, starting at `first_region`, that have one unit size and
// lie a constant distance apart in both layouts: a slot's pieces in them are
// the rows of one rectangle.
struct Rectangle {
  size_t first_region = 0;
  uint64_t rows = 1;
  uint64_t dst_pitch = 0;
  uint64_t src_pitch = 0;
};

std::vector<Rectangle> Rectangles(const Layout& dst, const Layout& src) {
  std::vector<Rectangle> out;
  const size_t count = dst.regions.size();
  size_t r = 0;
  while (r < count) {
    Rectangle rect;
    rect.first_region = r;
    const uint64_t unit = dst.regions[r].unit_bytes;
    while (r + 1 < count && dst.regions[r + 1].unit_bytes == unit &&
           dst.regions[r + 1].offset > dst.regions[r].offset &&
           src.regions[r + 1].offset > src.regions[r].offset) {
      const uint64_t dst_pitch = dst.regions[r + 1].offset - dst.regions[r].offset;
      const uint64_t src_pitch = src.regions[r + 1].offset - src.regions[r].offset;
      if (rect.rows == 1) {
        rect.dst_pitch = dst_pitch;
        rect.src_pitch = src_pitch;
      } else if (dst_pitch != rect.dst_pitch || src_pitch != rect.src_pitch) {
        break;
      }
      ++rect.rows;
      ++r;
    }
    ++r;
    out.push_back(rect);
  }
  return out;
}

uint64_t MicrosSince(std::chrono::steady_clock::time_point started) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - started)
                                   .count());
}

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

// Every HIP call is made under one lock. Attaches for several devices arrive
// at once on different RPC threads, and concurrent allocate / zero-fill /
// synchronize calls from those threads have been seen to park forever inside
// the runtime. Copies are short, so serializing them costs little for now.
class HipMemoryApi final : public MemoryApi {
 public:
  bool Allocate(int device, uint64_t bytes, void** base, ExportedMemory* exported,
                std::string* error) override {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
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
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    (void)hipFree(base);
  }

  bool Copy(int dst_device, void* dst, int /*src_device*/, const void* src, uint64_t bytes,
            std::string* error) override {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
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

  // A scattered block is one small piece per region (301 of ~30 KB for
  // MiniMax-M3), and a blocking hipMemcpy costs ~13 us whatever its size, so
  // one at a time moves 2.2 GiB/s between two MI355X. Two things lift that:
  // a rectangle job is one 2D copy (41 GiB/s when a block is one rectangle),
  // and what is left is queued on up to eight streams from eight threads
  // (~3.5 us a copy, 8 GiB/s). The API lock is held throughout, so nothing
  // else enters the runtime meanwhile; the workers only queue and wait.
  bool CopyMany(int dst_device, int /*src_device*/, const std::vector<CopyJob>& jobs,
                std::string* error) override {
    return RunJobs(dst_device, jobs, hipMemcpyDeviceToDevice, error);
  }
  bool CopyManyToHost(int device, const std::vector<CopyJob>& jobs, std::string* error) override {
    return RunJobs(device, jobs, hipMemcpyDeviceToHost, error);
  }
  bool CopyManyFromHost(int device, const std::vector<CopyJob>& jobs, std::string* error) override {
    return RunJobs(device, jobs, hipMemcpyHostToDevice, error);
  }

  // A device-to-device copy is not guaranteed to have landed when hipMemcpy
  // returns, and the caller hands these bytes to another process next.
  bool Synchronize(int device, std::string* error) override {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    const hipError_t err = scope.ok() ? hipDeviceSynchronize() : hipErrorInvalidDevice;
    if (err != hipSuccess) {
      *error = HipError("hipDeviceSynchronize", err);
      return false;
    }
    return true;
  }

  // Page-locked, so the copies to and from a device are direct and the lower
  // store can register it with a NIC.
  void* AllocateHost(uint64_t bytes) override {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    void* ptr = nullptr;
    if (hipHostMalloc(&ptr, static_cast<size_t>(bytes), hipHostMallocDefault) != hipSuccess) {
      return nullptr;
    }
    return ptr;
  }

  void FreeHost(void* base) override {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    (void)hipHostFree(base);
  }

  bool CopyToHost(int device, const void* src, void* host_dst, uint64_t bytes,
                  std::string* error) override {
    return HostCopy(device, host_dst, src, bytes, hipMemcpyDeviceToHost, error);
  }

  bool CopyFromHost(int device, void* dst, const void* host_src, uint64_t bytes,
                    std::string* error) override {
    return HostCopy(device, dst, host_src, bytes, hipMemcpyHostToDevice, error);
  }

 private:
  bool HostCopy(int device, void* dst, const void* src, uint64_t bytes, hipMemcpyKind kind,
                std::string* error) {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(device) + ") failed";
      return false;
    }
    const hipError_t err = hipMemcpy(dst, src, static_cast<size_t>(bytes), kind);
    if (err != hipSuccess) {
      *error = HipError("hipMemcpy", err);
      return false;
    }
    return true;
  }

  bool RunJobs(int device, const std::vector<CopyJob>& jobs, hipMemcpyKind kind,
               std::string* error) {
    std::lock_guard<std::mutex> hip_lock(hip_mutex_);
    ScopedDevice scope(device);
    if (!scope.ok()) {
      *error = "hipSetDevice(" + std::to_string(device) + ") failed";
      return false;
    }
    const size_t workers =
        std::max<size_t>(1, std::min(kDeviceCopyThreads, jobs.size() / kMinJobsPerCopyThread));
    std::vector<hipStream_t>& streams = copy_streams_[device];
    while (streams.size() < workers) {
      hipStream_t stream = nullptr;
      const hipError_t err = hipStreamCreate(&stream);
      if (err != hipSuccess) {
        *error = HipError("hipStreamCreate", err);
        return false;
      }
      streams.push_back(stream);
    }
    std::vector<hipError_t> results(workers, hipSuccess);
    auto work = [&](size_t w, bool set_device) {
      hipError_t err = set_device ? hipSetDevice(device) : hipSuccess;
      for (size_t i = w; i < jobs.size() && err == hipSuccess; i += workers) {
        const CopyJob& job = jobs[i];
        err = job.rows == 1 ? hipMemcpyAsync(job.dst, job.src, static_cast<size_t>(job.bytes), kind,
                                             streams[w])
                            : hipMemcpy2DAsync(job.dst, static_cast<size_t>(job.dst_pitch), job.src,
                                               static_cast<size_t>(job.src_pitch),
                                               static_cast<size_t>(job.bytes),
                                               static_cast<size_t>(job.rows), kind, streams[w]);
      }
      // Wait even after a failure: what was queued must not outlive the call.
      const hipError_t waited = hipStreamSynchronize(streams[w]);
      results[w] = err != hipSuccess ? err : waited;
    };
    std::vector<std::thread> threads;
    for (size_t w = 1; w < workers; ++w) threads.emplace_back(work, w, true);
    work(0, false);
    for (std::thread& thread : threads) thread.join();
    for (hipError_t err : results) {
      if (err != hipSuccess) {
        *error = HipError("hipMemcpyAsync", err);
        return false;
      }
    }
    return true;
  }

  static constexpr size_t kDeviceCopyThreads = 8;
  // Under this many copies a thread costs more than it saves.
  static constexpr size_t kMinJobsPerCopyThread = 256;

  std::mutex hip_mutex_;
  // Streams CopyMany queues on, per destination device; under hip_mutex_.
  std::map<int, std::vector<hipStream_t>> copy_streams_;
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
  bool Copy(int /*dst_device*/, void* dst, int /*src_device*/, const void* src, uint64_t bytes,
            std::string* /*error*/) override {
    std::memcpy(dst, src, static_cast<size_t>(bytes));
    return true;
  }
  bool Synchronize(int /*device*/, std::string* /*error*/) override { return true; }
  void* AllocateHost(uint64_t bytes) override { return std::malloc(static_cast<size_t>(bytes)); }
  void FreeHost(void* base) override { std::free(base); }
  bool CopyToHost(int /*device*/, const void* src, void* host_dst, uint64_t bytes,
                  std::string* /*error*/) override {
    std::memcpy(host_dst, src, static_cast<size_t>(bytes));
    return true;
  }
  bool CopyFromHost(int /*device*/, void* dst, const void* host_src, uint64_t bytes,
                    std::string* /*error*/) override {
    std::memcpy(dst, host_src, static_cast<size_t>(bytes));
    return true;
  }
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

std::shared_ptr<MemoryApi> MakeHipMemoryApi() { return std::make_shared<HipMemoryApi>(); }
std::shared_ptr<MemoryApi> MakeHostMemoryApi() { return std::make_shared<HostMemoryApi>(); }

PoolManager::PoolManager(std::shared_ptr<MemoryApi> api) : api_(std::move(api)) {}

PoolManager::~PoolManager() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_.notify_all();
  if (worker_.joinable()) worker_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) {
    api_->Free(entry.second->device, entry.second->base, entry.second->bytes);
  }
  pools_.clear();
  if (staging_ != nullptr) {
    if (store_) store_->DeregisterHostMemory(staging_);
    api_->FreeHost(staging_);
    staging_ = nullptr;
  }
}

void PoolManager::SetLowerStore(std::shared_ptr<LowerStore> store, const TierOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (store_ || !store) return;
  void* staging = api_->AllocateHost(options.staging_bytes);
  if (staging == nullptr) {
    MORI_UMBP_ERROR("[L1Pool] no lower tiers: could not allocate {} B of staging",
                    options.staging_bytes);
    return;
  }
  if (!store->RegisterHostMemory(staging, options.staging_bytes)) {
    MORI_UMBP_ERROR("[L1Pool] no lower tiers: the store refused the staging memory");
    api_->FreeHost(staging);
    return;
  }
  staging_ = staging;
  tier_ = options;
  store_ = std::move(store);
  worker_ = std::thread([this] { TierWorker(); });
  MORI_UMBP_INFO(
      "[L1Pool] lower tiers attached: staging={} B offload_on_evict={} offload_ahead={} load={} "
      "ask_peers={} peer_copy={}",
      tier_.staging_bytes, tier_.offload_on_evict, tier_.offload_ahead, tier_.load, tier_.ask_peers,
      tier_.peer_copy);
}

std::string PoolManager::StoreKey(const std::string& group, const std::string& key) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(3 + group.size() + 1 + key.size() * 2);
  out += "l1/";
  out += group;
  out += '/';
  for (unsigned char c : key) {
    out += kHex[c >> 4];
    out += kHex[c & 0xf];
  }
  return out;
}

bool PoolManager::ParseStoreKey(const std::string& name, std::string* group, std::string* key) {
  // "l1/<group>/<hex>": the key is hex, so the last '/' ends the group.
  if (name.compare(0, 3, "l1/") != 0) return false;
  const size_t slash = name.rfind('/');
  if (slash == std::string::npos || slash < 3) return false;
  const size_t hex = name.size() - slash - 1;
  if (hex == 0 || hex % 2 != 0) return false;
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  std::string out;
  out.reserve(hex / 2);
  for (size_t i = slash + 1; i < name.size(); i += 2) {
    const int hi = nibble(name[i]);
    const int lo = nibble(name[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  *group = name.substr(3, slash - 3);
  *key = std::move(out);
  return true;
}

void PoolManager::QueueAdvertiseLocked(const std::string& group, const std::string& key,
                                       bool added) {
  if (!store_) return;
  advertise_[StoreKey(group, key)] = added;
  work_.notify_one();
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
      if (list.empty()) {
        group->second.holders.erase(holders);
        QueueAdvertiseLocked(pool->group, s.key, false);
      }
    }
  }
  s.has_key = false;
  s.in_store = false;
  s.key.clear();
}

void PoolManager::ResetSlotsLocked(Pool* pool) {
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  pool->index.clear();
  pool->lru.clear();
  pool->vacant.clear();
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) {
    pool->slots[slot] = Slot{};
    pool->vacant.insert(slot);
  }
  pool->held = 0;
  pool->committed = 0;
}

bool PoolManager::TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                                 bool settled_only, std::vector<Victim>* victims) {
  for (auto it = pool->vacant.begin(); it != pool->vacant.end(); ++it) {
    if (pool->slots[*it].pins != 0) continue;
    if (settled_only && !Fillable(*pool, pool->slots[*it])) continue;
    *slot = *it;
    pool->vacant.erase(it);
    return true;
  }
  if (evicted == nullptr) return false;
  for (auto it = pool->lru.begin(); it != pool->lru.end(); ++it) {
    Slot& s = pool->slots[*it];
    if (s.pins != 0) continue;
    if (settled_only && !Fillable(*pool, s)) continue;
    *slot = *it;
    pool->lru.erase(it);
    if (s.has_key) {
      evicted->push_back(s.key);
      if (!s.in_store) {
        // Unsettled content cannot be read from here, so it is dropped.
        if (victims != nullptr && store_ && tier_.offload_on_evict && Readable(*pool, s)) {
          victims->push_back(Victim{*slot, s.key});
        } else {
          ++pool->dropped_slots;
        }
      }
      UnindexLocked(pool, *slot);
      ++pool->evicted;
    }
    return true;
  }
  return false;
}

std::vector<bool> PoolManager::OffloadSlots(Pool* pool, const std::vector<Victim>& slots,
                                            bool evicting) {
  std::vector<bool> stored(slots.size(), false);
  if (slots.empty()) return stored;
  const uint64_t slot_bytes = pool->slot_bytes;
  if (!store_ || staging_ == nullptr || slot_bytes == 0 || slot_bytes > tier_.staging_bytes) {
    return stored;
  }
  const auto started = std::chrono::steady_clock::now();
  const size_t per_round = static_cast<size_t>(tier_.staging_bytes / slot_bytes);
  const std::vector<Region>& regions = pool->layout.regions;
  const std::vector<uint64_t> starts = RegionStarts(pool->layout);
  const std::vector<Rectangle> rectangles = Rectangles(pool->layout, pool->layout);
  std::string error;
  uint64_t done = 0;
  uint64_t wait_micros = 0, read_micros = 0, put_micros = 0;
  for (size_t begin = 0; begin < slots.size(); begin += per_round) {
    const size_t end = std::min(slots.size(), begin + per_round);
    auto mark = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> staging(staging_mutex_);
    wait_micros += MicrosSince(mark);
    mark = std::chrono::steady_clock::now();
    std::vector<std::string> keys;
    std::vector<const void*> srcs;
    std::vector<size_t> which;
    // A slot goes to the device as one rectangle per group of like regions,
    // and lands in the layout the store keeps: a slot's units back to back.
    char* const staging_base = static_cast<char*>(staging_);
    std::vector<MemoryApi::CopyJob> jobs;
    jobs.reserve((end - begin) * rectangles.size());
    for (size_t i = begin; i < end; ++i) {
      char* const slot_base = staging_base + (i - begin) * slot_bytes;
      for (const Rectangle& rect : rectangles) {
        const Region& region = regions[rect.first_region];
        const uint64_t unit = region.unit_bytes;
        const void* src =
            static_cast<const char*>(pool->base) + region.offset + uint64_t{slots[i].slot} * unit;
        jobs.push_back(MemoryApi::CopyJob{slot_base + starts[rect.first_region], src, unit,
                                          rect.rows, unit, rect.src_pitch});
      }
    }
    if (api_->CopyManyToHost(pool->device, jobs, &error)) {
      for (size_t i = begin; i < end; ++i) {
        keys.push_back(StoreKey(pool->group, slots[i].key));
        srcs.push_back(staging_base + (i - begin) * slot_bytes);
        which.push_back(i);
      }
    }
    read_micros += MicrosSince(mark);
    if (keys.empty()) continue;
    mark = std::chrono::steady_clock::now();
    const std::vector<bool> put =
        Guarded("put", keys.size(), [&] { return store_->Put(keys, srcs, slot_bytes); });
    put_micros += MicrosSince(mark);
    for (size_t j = 0; j < which.size() && j < put.size(); ++j) {
      if (put[j]) {
        stored[which[j]] = true;
        ++done;
      }
    }
  }
  const uint64_t micros = MicrosSince(started);
  if (!error.empty()) {
    MORI_UMBP_ERROR("[L1Pool] offload from pool={} read error: {}", pool->id, error);
  }
  MORI_UMBP_INFO(
      "[L1Pool] offload pool={} slots={} stored={} bytes={} micros={} wait={} read={} "
      "put={}",
      pool->id, slots.size(), done, done * slot_bytes, micros, wait_micros, read_micros,
      put_micros);
  std::lock_guard<std::mutex> lock(mutex_);
  pool->offloaded_slots += done;
  // A slot that stays cached here has lost nothing by a failed copy.
  if (evicting) pool->dropped_slots += slots.size() - done;
  pool->offload_micros += micros;
  if (done != 0) {
    // Other nodes learn of the copies from the worker, not from this call:
    // it is on an engine's allocation path.
    publish_pending_ = true;
    work_.notify_one();
  }
  return stored;
}

std::vector<bool> PoolManager::LoadSlots(Pool* pool, const std::vector<Victim>& slots,
                                         uint64_t* bytes) {
  std::vector<bool> loaded(slots.size(), false);
  if (slots.empty()) return loaded;
  const uint64_t slot_bytes = pool->slot_bytes;
  if (!store_ || staging_ == nullptr || slot_bytes == 0 || slot_bytes > tier_.staging_bytes) {
    return loaded;
  }
  const size_t per_round = static_cast<size_t>(tier_.staging_bytes / slot_bytes);
  const std::vector<Region>& regions = pool->layout.regions;
  const std::vector<uint64_t> starts = RegionStarts(pool->layout);
  const std::vector<Rectangle> rectangles = Rectangles(pool->layout, pool->layout);
  std::string error;
  // The caller wants a run from the front, so the first failure ends the work.
  bool stop = false;
  uint64_t wait_micros = 0, get_micros = 0, write_micros = 0;
  for (size_t begin = 0; begin < slots.size() && !stop; begin += per_round) {
    const size_t end = std::min(slots.size(), begin + per_round);
    auto mark = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> staging(staging_mutex_);
    wait_micros += MicrosSince(mark);
    mark = std::chrono::steady_clock::now();
    std::vector<std::string> keys;
    std::vector<void*> dsts;
    for (size_t i = begin; i < end; ++i) {
      keys.push_back(StoreKey(pool->group, slots[i].key));
      dsts.push_back(static_cast<char*>(staging_) + (i - begin) * slot_bytes);
    }
    const std::vector<bool> got =
        Guarded("get", keys.size(), [&] { return store_->Get(keys, dsts, slot_bytes); });
    // The caller wants a run from the front: what follows the first key the
    // store could not give is not used.
    get_micros += MicrosSince(mark);
    mark = std::chrono::steady_clock::now();
    size_t good = begin;
    while (good < end && good - begin < got.size() && got[good - begin]) ++good;
    if (good < end) stop = true;
    char* const staging_base = static_cast<char*>(staging_);
    std::vector<MemoryApi::CopyJob> jobs;
    jobs.reserve((good - begin) * rectangles.size());
    for (size_t i = begin; i < good; ++i) {
      const char* const slot_base = staging_base + (i - begin) * slot_bytes;
      for (const Rectangle& rect : rectangles) {
        const Region& region = regions[rect.first_region];
        const uint64_t unit = region.unit_bytes;
        void* dst = static_cast<char*>(pool->base) + region.offset + uint64_t{slots[i].slot} * unit;
        jobs.push_back(MemoryApi::CopyJob{dst, slot_base + starts[rect.first_region], unit,
                                          rect.rows, rect.dst_pitch, unit});
      }
    }
    if (jobs.empty()) continue;
    if (!api_->CopyManyFromHost(pool->device, jobs, &error)) {
      stop = true;
      break;
    }
    for (size_t i = begin; i < good; ++i) loaded[i] = true;
    *bytes += (good - begin) * slot_bytes;
    write_micros += MicrosSince(mark);
  }
  if (!error.empty()) {
    MORI_UMBP_ERROR("[L1Pool] load into pool={} write error: {}", pool->id, error);
  }
  MORI_UMBP_INFO("[L1Pool] load pool={} slots={} bytes={} wait={} get={} write={}", pool->id,
                 slots.size(), *bytes, wait_micros, get_micros, write_micros);
  return loaded;
}

bool PoolManager::HasPinnedFreeLocked(const Pool* pool) const {
  for (uint32_t slot : pool->vacant) {
    if (pool->slots[slot].pins != 0) return true;
  }
  for (uint32_t slot : pool->lru) {
    if (pool->slots[slot].pins != 0) return true;
  }
  return false;
}

void PoolManager::DestroyLocked(std::unique_lock<std::mutex>& lock, Pool* pool) {
  const std::string pool_id = pool->id;
  idle_.wait(lock, [&] {
    Pool* p = FindLocked(pool_id);
    return p == nullptr || p->busy == 0;
  });
  pool = FindLocked(pool_id);
  if (pool == nullptr) return;
  for (uint32_t slot = 0; slot < pool->slots.size(); ++slot) UnindexLocked(pool, slot);
  api_->Free(pool->device, pool->base, pool->bytes);
  MORI_UMBP_INFO("[L1Pool] destroyed pool={} device={} bytes={}", pool_id, pool->device,
                 pool->bytes);
  pools_.erase(pool_id);
}

AttachResult PoolManager::Attach(const std::string& pool_id, const std::string& group, int device,
                                 uint64_t bytes) {
  AttachResult result;
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
    if (existing->group == group && existing->device == device && existing->bytes == bytes) {
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
  ExportedMemory exported;
  std::string error;
  const bool ok = api_->Allocate(device, bytes, &base, &exported, &error);
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
  pool->group = group;
  pool->device = device;
  pool->base = base;
  pool->bytes = bytes;
  pool->exported = exported;
  pools_[pool_id] = std::move(pool);
  MORI_UMBP_INFO("[L1Pool] attached pool={} group={} device={} bytes={}", pool_id, group, device,
                 bytes);
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
    if (region.unit_bytes == 0 ||
        region.offset + static_cast<uint64_t>(layout.num_slots) * region.unit_bytes > pool->bytes) {
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
  pool->layout = layout;
  pool->slot_bytes = 0;
  for (const Region& region : layout.regions) pool->slot_bytes += region.unit_bytes;
  pool->slots.assign(layout.num_slots, Slot{});
  pool->bound = true;
  ResetSlotsLocked(pool);
  MORI_UMBP_INFO("[L1Pool] bound pool={} slots={} regions={} rectangles={}", pool_id,
                 layout.num_slots, layout.regions.size(), Rectangles(layout, layout).size());
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
  std::vector<Victim> victims;
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
    Pool* now = FindLocked(pool_id);
    if (now != pool || !now->bound) {
      result.status = Status::Error("pool " + pool_id + " was detached");
      return result;
    }
  }
  if (!victims.empty()) {
    // The owner is waiting on this call and the slots are held for it, so
    // their old content stays put until it has been copied below.
    ++pool->busy;
    lock.unlock();
    OffloadSlots(pool, victims, /*evicting=*/true);
    lock.lock();
    --pool->busy;
    idle_.notify_all();
  }
  return result;
}

Status PoolManager::Commit(const std::string& pool_id, uint64_t committed) {
  std::lock_guard<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr) return Status::Error("unknown pool " + pool_id);
  pool->committed = std::max(pool->committed, committed);
  if (store_ && tier_.offload_ahead > 0.0) work_.notify_one();
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
          pool->lru.erase(s.lru_it);
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
            pool->lru.erase(old.lru_it);
            old.state = SlotState::kVacant;
            pool->vacant.insert(old_slot);
          }
        }
        s.has_key = true;
        s.in_store = false;
        s.key = op.key;
        s.written = stamp;
        pool->index[op.key] = op.slot;
        auto& holders = groups_[pool->group].holders[op.key];
        holders.push_back(Holder{pool, op.slot});
        if (holders.size() == 1) QueueAdvertiseLocked(pool->group, op.key, true);
        break;
      }
      default:
        return fail("unknown op");
    }
  }
  return Status{};
}

FetchResult PoolManager::Fetch(const std::string& pool_id, const std::vector<std::string>& keys) {
  FetchResult result;
  std::unique_lock<std::mutex> lock(mutex_);
  Pool* pool = FindLocked(pool_id);
  if (pool == nullptr || !pool->bound) {
    result.status = Status::Error("pool " + pool_id + " is not attached and bound");
    return result;
  }
  Group& group = groups_[pool->group];

  // A settled slot of another pool of this node holding `key`, or null.
  auto find_source = [&](const std::string& key) -> const Holder* {
    if (!tier_.peer_copy) return nullptr;
    auto holders = group.holders.find(key);
    if (holders == group.holders.end()) return nullptr;
    for (const Holder& h : holders->second) {
      // Not this pool, and not a slot its owner may still be writing.
      if (h.pool != pool && Readable(*h.pool, h.pool->slots[h.slot])) return &h;
    }
    return nullptr;
  };

  // Keys no pool of this node can serve are looked for below L1: in the
  // store, and — through the directory — in other nodes' L1, which copy them
  // into the store when asked. Both are calls to other processes, so they are
  // made without the lock and the plan is drawn up afterwards.
  std::vector<bool> below(keys.size(), false);
  if (store_ && tier_.load) {
    std::vector<size_t> unknown;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (pool->index.count(keys[i]) == 0 && find_source(keys[i]) == nullptr) unknown.push_back(i);
    }
    if (!unknown.empty()) {
      std::vector<std::string> names;
      names.reserve(unknown.size());
      for (size_t i : unknown) names.push_back(StoreKey(pool->group, keys[i]));
      ++pool->busy;
      lock.unlock();
      std::vector<bool> found =
          Guarded("lookup", names.size(), [&] { return store_->Exists(names); });
      if (tier_.ask_peers) {
        std::vector<std::string> missing;
        std::vector<size_t> which;
        for (size_t j = 0; j < names.size(); ++j) {
          if (!found[j]) {
            missing.push_back(names[j]);
            which.push_back(j);
          }
        }
        if (!missing.empty()) {
          const std::vector<bool> asked = Guarded(
              "peer request", missing.size(), [&] { return store_->RequestFromPeers(missing); });
          for (size_t j = 0; j < which.size() && j < asked.size(); ++j) {
            if (asked[j]) found[which[j]] = true;
          }
        }
      }
      lock.lock();
      --pool->busy;
      idle_.notify_all();
      for (size_t j = 0; j < unknown.size(); ++j) below[unknown[j]] = found[j];
    }
  }

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists. Keys already
  // here are pinned first, so no destination of this call evicts one.
  std::vector<CopyStep> steps;
  std::vector<uint32_t> local_pins;
  for (const std::string& key : keys) {
    auto local = pool->index.find(key);
    if (local == pool->index.end()) continue;
    ++pool->slots[local->second].pins;
    local_pins.push_back(local->second);
  }
  std::vector<Victim> victims;
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string& key = keys[i];
    auto local = pool->index.find(key);
    if (local != pool->index.end()) {
      result.slots.push_back(local->second);
      continue;
    }
    const Holder* source = find_source(key);
    if (source == nullptr && !below[i]) break;
    uint32_t dst = 0;
    if (!TakeFreeLocked(pool, &dst, &result.evicted_keys, /*settled_only=*/true, &victims)) break;
    pool->slots[dst].state = SlotState::kFilling;
    ++pool->busy;
    if (source != nullptr) {
      ++source->pool->slots[source->slot].pins;
      ++source->pool->busy;
      steps.push_back(CopyStep{source->pool, source->slot, dst, i});
    } else {
      steps.push_back(CopyStep{nullptr, 0, dst, i});
    }
    result.slots.push_back(dst);
  }

  // Copy without the lock. Pool pointers stay valid: Detach, Open and a
  // re-bind all wait for `busy` to reach zero.
  ++pool->busy;
  lock.unlock();
  // What the destinations held goes below before anything overwrites it.
  if (!victims.empty()) OffloadSlots(pool, victims, /*evicting=*/true);

  const auto started = std::chrono::steady_clock::now();
  std::vector<bool> copied(steps.size(), false);
  std::string copy_error;
  bool failed = false;
  std::vector<size_t> peer_steps;
  std::vector<size_t> store_steps;
  for (size_t i = 0; i < steps.size(); ++i) {
    (steps[i].src != nullptr ? peer_steps : store_steps).push_back(i);
  }
  std::vector<MemoryApi::CopyJob> jobs;
  std::vector<Rectangle> rectangles;  // of the source pool `jobs` is for
  std::vector<size_t> jobs_steps;
  uint64_t jobs_bytes = 0;
  Pool* jobs_src = nullptr;
  auto flush_jobs = [&] {
    if (jobs.empty()) return;
    if (!failed && api_->CopyMany(pool->device, jobs_src->device, jobs, &copy_error)) {
      for (size_t step : jobs_steps) copied[step] = true;
      result.copied_bytes += jobs_bytes;
    } else {
      failed = true;
    }
    jobs.clear();
    jobs_steps.clear();
    jobs_bytes = 0;
  };
  size_t run_start = 0;
  while (run_start < peer_steps.size() && !failed) {
    size_t run_end = run_start + 1;
    while (run_end < peer_steps.size()) {
      const CopyStep& prev = steps[peer_steps[run_end - 1]];
      const CopyStep& next = steps[peer_steps[run_end]];
      if (next.src != prev.src || next.src_slot != prev.src_slot + 1 ||
          next.dst_slot != prev.dst_slot + 1) {
        break;
      }
      ++run_end;
    }
    // Runs from one source pool that follow each other go to the device as
    // one list, so scattered slots do not pay a lock and a device switch each.
    const CopyStep& first = steps[peer_steps[run_start]];
    const uint64_t run_slots = run_end - run_start;
    if (jobs_src != first.src) {
      flush_jobs();
      jobs_src = first.src;
    }
    if (jobs.empty()) rectangles = Rectangles(pool->layout, first.src->layout);
    for (const Rectangle& rect : rectangles) {
      const Region& dst_region = pool->layout.regions[rect.first_region];
      const Region& src_region = first.src->layout.regions[rect.first_region];
      const uint64_t unit = dst_region.unit_bytes;
      void* dst =
          static_cast<char*>(pool->base) + dst_region.offset + uint64_t{first.dst_slot} * unit;
      const void* src = static_cast<const char*>(first.src->base) + src_region.offset +
                        uint64_t{first.src_slot} * unit;
      jobs.push_back(MemoryApi::CopyJob{dst, src, run_slots * unit, rect.rows, rect.dst_pitch,
                                        rect.src_pitch});
      jobs_bytes += run_slots * unit * rect.rows;
    }
    for (size_t i = run_start; i < run_end; ++i) jobs_steps.push_back(peer_steps[i]);
    run_start = run_end;
  }
  flush_jobs();
  result.copy_micros = MicrosSince(started);

  const auto load_started = std::chrono::steady_clock::now();
  if (!store_steps.empty() && !failed) {
    std::vector<Victim> wanted;
    wanted.reserve(store_steps.size());
    for (size_t i : store_steps)
      wanted.push_back(Victim{steps[i].dst_slot, keys[steps[i].key_index]});
    const std::vector<bool> loaded = LoadSlots(pool, wanted, &result.loaded_bytes);
    for (size_t j = 0; j < store_steps.size(); ++j) copied[store_steps[j]] = loaded[j];
  }
  // Once per device rather than once per copy: nothing is handed to the
  // requester before this returns.
  if (!steps.empty()) {
    std::set<int> devices{pool->device};
    for (const CopyStep& step : steps) {
      if (step.src != nullptr) devices.insert(step.src->device);
    }
    for (int device : devices) {
      if (!api_->Synchronize(device, &copy_error)) {
        failed = true;
        std::fill(copied.begin(), copied.end(), false);
        break;
      }
    }
  }
  result.load_micros = store_steps.empty() ? 0 : MicrosSince(load_started);
  lock.lock();
  --pool->busy;

  // Commit: a filled slot becomes cached content of this pool.
  size_t valid_keys = result.slots.size();
  std::vector<std::pair<std::string, uint32_t>> sources;  // runs, for the log
  for (size_t i = 0; i < steps.size(); ++i) {
    const CopyStep& step = steps[i];
    bool source_in_store = true;  // content that came from the store is there
    if (step.src != nullptr) {
      Slot& from = step.src->slots[step.src_slot];
      source_in_store = from.in_store;
      --from.pins;
      --step.src->busy;
    }
    --pool->busy;
    Slot& d = pool->slots[step.dst_slot];
    const std::string& key = keys[step.key_index];
    const bool usable = copied[i] && step.key_index < valid_keys;
    auto existing = pool->index.find(key);
    if (usable && existing == pool->index.end()) {
      d.has_key = true;
      d.in_store = source_in_store;
      d.key = key;
      // The copy has landed and the owner has never used this content.
      d.written = 0;
      d.released = 0;
      d.state = SlotState::kCached;
      d.lru_it = pool->lru.insert(pool->lru.end(), step.dst_slot);
      pool->index[key] = step.dst_slot;
      auto& holders = group.holders[key];
      holders.push_back(Holder{pool, step.dst_slot});
      if (holders.size() == 1) QueueAdvertiseLocked(pool->group, key, true);
      const std::string from = step.src != nullptr ? step.src->id : std::string("store");
      if (step.src != nullptr) {
        ++step.src->served_out_slots;
        ++pool->fetched_in_slots;
        ++result.copied_slots;
      } else {
        ++pool->loaded_slots;
        ++result.loaded_slots;
      }
      if (sources.empty() || sources.back().first != from) sources.emplace_back(from, 0u);
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
  result.slots.resize(valid_keys);
  pool->fetched_in_bytes += result.copied_bytes;
  pool->copy_micros += result.copy_micros;
  pool->load_micros += result.load_micros;
  idle_.notify_all();

  if (failed) {
    MORI_UMBP_ERROR("[L1Pool] fetch into pool={} stopped after {} of {} keys: {}", pool_id,
                    valid_keys, keys.size(), copy_error);
  } else if (result.copied_slots != 0 || result.loaded_slots != 0) {
    std::string from;
    for (const auto& source : sources) {
      from += (from.empty() ? "" : ",") + source.first + "x" + std::to_string(source.second);
    }
    MORI_UMBP_INFO(
        "[L1Pool] fetch pool={} keys={} resident={} copied_slots={} bytes={} micros={} "
        "loaded_slots={} loaded_bytes={} load_micros={} from={}",
        pool_id, keys.size(), result.slots.size(), result.copied_slots, result.copied_bytes,
        result.copy_micros, result.loaded_slots, result.loaded_bytes, result.load_micros, from);
  }
  return result;
}

ExportResult PoolManager::Export(const std::string& group_id,
                                 const std::vector<std::string>& keys) {
  ExportResult result;
  result.stored.assign(keys.size(), false);
  std::unique_lock<std::mutex> lock(mutex_);
  if (!store_) {
    result.status = Status::Error("this node has no lower store to export into");
    return result;
  }
  auto found = groups_.find(group_id);
  if (found == groups_.end()) return result;
  Group& group = found->second;

  // One settled holder per key, pinned for the copy and grouped by pool.
  struct Pick {
    Pool* pool = nullptr;
    uint32_t slot = 0;
    size_t key_index = 0;
    bool hinted = false;  // believed to be in the store already
  };
  std::vector<Pick> picks;
  for (size_t i = 0; i < keys.size(); ++i) {
    auto holders = group.holders.find(keys[i]);
    if (holders == group.holders.end()) continue;
    for (const Holder& h : holders->second) {
      Slot& s = h.pool->slots[h.slot];
      if (!Readable(*h.pool, s)) continue;
      ++s.pins;
      ++h.pool->busy;
      picks.push_back(Pick{h.pool, h.slot, i, s.in_store});
      break;
    }
  }
  lock.unlock();

  // The hint can be stale: the store evicts on its own.
  std::vector<std::string> hinted_names;
  std::vector<size_t> hinted;
  for (size_t j = 0; j < picks.size(); ++j) {
    if (!picks[j].hinted) continue;
    hinted_names.push_back(StoreKey(group_id, keys[picks[j].key_index]));
    hinted.push_back(j);
  }
  std::vector<bool> present(picks.size(), false);
  if (!hinted.empty()) {
    const std::vector<bool> there =
        Guarded("lookup", hinted_names.size(), [&] { return store_->Exists(hinted_names); });
    for (size_t j = 0; j < hinted.size() && j < there.size(); ++j) present[hinted[j]] = there[j];
  }
  std::unordered_map<Pool*, std::vector<size_t>> by_pool;
  for (size_t j = 0; j < picks.size(); ++j) {
    if (!present[j]) by_pool[picks[j].pool].push_back(j);
  }
  for (auto& entry : by_pool) {
    std::vector<Victim> slots;
    for (size_t j : entry.second) slots.push_back(Victim{picks[j].slot, keys[picks[j].key_index]});
    const std::vector<bool> stored = OffloadSlots(entry.first, slots, /*evicting=*/false);
    for (size_t k = 0; k < entry.second.size(); ++k) present[entry.second[k]] = stored[k];
  }
  GuardedVoid("publish", [&] { store_->Publish(); });

  lock.lock();
  size_t exported = 0;
  for (size_t j = 0; j < picks.size(); ++j) {
    const Pick& pick = picks[j];
    Slot& s = pick.pool->slots[pick.slot];
    --s.pins;
    --pick.pool->busy;
    if (!present[j]) continue;
    if (s.has_key && s.key == keys[pick.key_index]) s.in_store = true;
    result.stored[pick.key_index] = true;
    ++exported;
  }
  idle_.notify_all();
  MORI_UMBP_INFO("[L1Pool] export group={} keys={} held={} stored={}", group_id, keys.size(),
                 picks.size(), exported);
  return result;
}

bool PoolManager::OffloadAhead() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!store_ || tier_.offload_ahead <= 0.0) return false;
  Pool* chosen = nullptr;
  std::vector<Victim> slots;
  for (auto& entry : pools_) {
    Pool* pool = entry.second.get();
    if (!pool->bound || pool->slot_bytes == 0) continue;
    // Walk from the coldest cached slot: those are the next to be evicted, so
    // they are the ones that should already be below.
    const double want = std::ceil(tier_.offload_ahead * static_cast<double>(pool->slots.size()));
    size_t ready = pool->vacant.size();
    for (uint32_t slot : pool->lru) {
      if (static_cast<double>(ready) >= want || slots.size() >= kAheadBatch) break;
      Slot& s = pool->slots[slot];
      ++ready;
      if (!s.has_key || s.in_store || s.pins != 0 || !Readable(*pool, s)) continue;
      slots.push_back(Victim{slot, s.key});
    }
    if (!slots.empty()) {
      chosen = pool;
      break;
    }
  }
  if (chosen == nullptr) return false;
  for (const Victim& v : slots) ++chosen->slots[v.slot].pins;
  ++chosen->busy;
  lock.unlock();
  const std::vector<bool> stored = OffloadSlots(chosen, slots, /*evicting=*/false);
  lock.lock();
  bool any = false;
  for (size_t i = 0; i < slots.size(); ++i) {
    Slot& s = chosen->slots[slots[i].slot];
    --s.pins;
    if (!stored[i]) continue;
    any = true;
    if (s.has_key && s.key == slots[i].key) s.in_store = true;
  }
  --chosen->busy;
  idle_.notify_all();
  return any;
}

void PoolManager::TierWorker() {
  while (true) {
    std::unordered_map<std::string, bool> changes;
    bool publish = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_.wait_for(lock, kWorkerIdle,
                     [&] { return stopping_ || publish_pending_ || !advertise_.empty(); });
      if (stopping_) return;
      changes.swap(advertise_);
      publish = publish_pending_;
      publish_pending_ = false;
    }
    // Before the directory changes: a block that just left L1 for the store
    // is then never missing from both.
    if (publish) GuardedVoid("publish", [&] { store_->Publish(); });
    if (!changes.empty()) {
      std::vector<std::string> added;
      std::vector<std::string> removed;
      for (auto& change : changes) (change.second ? added : removed).push_back(change.first);
      GuardedVoid("advertise", [&] { store_->Advertise(added, removed); });
    }
    while (OffloadAhead()) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
    }
  }
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
      pool->lru.erase(s.lru_it);
      s.state = SlotState::kVacant;
      pool->vacant.insert(slot);
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
    stats.offloaded_slots = pool.offloaded_slots;
    stats.offload_micros = pool.offload_micros;
    stats.loaded_slots = pool.loaded_slots;
    stats.load_micros = pool.load_micros;
    stats.dropped_slots = pool.dropped_slots;
    stats.held = pool.held;
    stats.cached = static_cast<uint32_t>(pool.lru.size());
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

}  // namespace mori::umbp::l1
