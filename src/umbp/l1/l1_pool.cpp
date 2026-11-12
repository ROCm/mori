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
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp::l1 {

namespace {

// How long Allocate waits for a slot that is free but pinned by a copy.
constexpr std::chrono::milliseconds kPinnedWait{2000};

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

 private:
  std::mutex hip_mutex_;
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
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) {
    api_->Free(entry.second->device, entry.second->base, entry.second->bytes);
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
      UnindexLocked(pool, *slot);
      ++pool->evicted;
    }
    return true;
  }
  return false;
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
        s.key = op.key;
        s.written = stamp;
        pool->index[op.key] = op.slot;
        groups_[pool->group].holders[op.key].push_back(Holder{pool, op.slot});
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

  // Plan under the lock: pick a source and a destination for every key, pin
  // the sources, and take the destinations off the free lists.
  std::vector<CopyStep> steps;
  std::vector<uint32_t> local_pins;
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string& key = keys[i];
    auto local = pool->index.find(key);
    if (local != pool->index.end()) {
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
        if (h.pool != pool && Readable(*h.pool, h.pool->slots[h.slot])) {
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
  size_t run_start = 0;
  bool failed = false;
  while (run_start < steps.size() && !failed) {
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
      if (!api_->Copy(pool->device, dst, first.src->device, src, run_slots * unit, &copy_error)) {
        failed = true;
        break;
      }
      result.copied_bytes += run_slots * unit;
    }
    if (!failed) {
      for (size_t i = run_start; i < run_end; ++i) copied[i] = true;
    }
    run_start = run_end;
  }
  // Once per device rather than once per copy: nothing is handed to the
  // requester before this returns.
  if (!steps.empty()) {
    std::set<int> devices{pool->device};
    for (const CopyStep& step : steps) devices.insert(step.src->device);
    for (int device : devices) {
      if (!api_->Synchronize(device, &copy_error)) {
        failed = true;
        std::fill(copied.begin(), copied.end(), false);
        break;
      }
    }
  }
  result.copy_micros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
  lock.lock();

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
