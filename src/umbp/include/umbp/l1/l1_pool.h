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
#pragma once

#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mori::umbp::l1 {

// UMBP-owned L1 (GPU HBM) block pools.
//
// A pool is one allocation on one device, owned by this process and mapped by
// one engine process, which reads and writes slots in place. This class owns
// what the engine used to own itself: the memory, the free list, eviction, and
// the content index. Pools in the same group share that index, so content one
// engine published can be copied into another engine's pool on request.
//
// Nothing here knows what a slot holds. A slot is `sum(region.unit_bytes)`
// bytes, spread over the pool's regions; a key is opaque bytes.
//
// Ownership model: every state change of a pool is requested by that pool's
// owner (Allocate / Apply / Fetch / Clear). Another pool's Fetch only reads a
// slot, and pins it for the duration of the copy. That is what lets the owner
// keep an exact local view of its own pool without a notification channel.
//
// Settling: an owner decides what a slot holds before its device has finished
// writing it -- an engine publishes a block when the forward that fills it is
// issued. Inside the owner that is safe, because its own readers queue behind
// its own writers. This process is not in that queue. So publishes and
// releases carry a stamp, the owner reports how far its device has got
// (Commit), and a slot is touched from outside -- read as a fetch source or
// filled as a fetch destination -- only once it has settled.

struct Region {
  uint64_t offset = 0;
  uint64_t unit_bytes = 0;
};

struct Layout {
  uint32_t num_slots = 0;
  std::vector<Region> regions;
};

struct ExportedMemory {
  std::string kind;    // "hip_ipc" or "host"
  std::string handle;  // opaque bytes the mapping process opens
};

// The device operations, injectable so the slot logic is testable on CPU.
class MemoryApi {
 public:
  virtual ~MemoryApi() = default;
  // Allocate `bytes` zero-filled on `device` and export it for another process.
  virtual bool Allocate(int device, uint64_t bytes, void** base, ExportedMemory* exported,
                        std::string* error) = 0;
  virtual void Free(int device, void* base, uint64_t bytes) = 0;
  // Copy `bytes`. The bytes are only guaranteed visible at `dst` after a
  // Synchronize of the destination device.
  virtual bool Copy(int dst_device, void* dst, int src_device, const void* src, uint64_t bytes,
                    std::string* error) = 0;
  // Wait until every copy issued on `device` has landed.
  virtual bool Synchronize(int device, std::string* error) = 0;
};

std::shared_ptr<MemoryApi> MakeHipMemoryApi();
// Host memory; the exported handle is not openable by another process.
std::shared_ptr<MemoryApi> MakeHostMemoryApi();

struct Status {
  bool ok = true;
  std::string error;
  static Status Error(std::string message) { return Status{false, std::move(message)}; }
};

enum class OpKind : uint32_t { kHold = 0, kRelease = 1, kPublish = 2 };

struct Op {
  OpKind kind = OpKind::kHold;
  uint32_t slot = 0;
  std::string key;  // kPublish only
};

struct AttachResult {
  Status status;
  ExportedMemory memory;
  uint64_t bytes = 0;
  bool created = false;
};

struct OpenResult {
  Status status;
  uint32_t num_slots = 0;
};

struct AllocateResult {
  Status status;
  std::vector<uint32_t> slots;
  std::vector<std::string> evicted_keys;
};

struct FetchResult {
  Status status;
  std::vector<uint32_t> slots;
  std::vector<std::string> evicted_keys;
  uint32_t copied_slots = 0;
  uint64_t copied_bytes = 0;
  uint64_t copy_micros = 0;
};

struct PoolStats {
  std::string pool_id;
  std::string group;
  int device = -1;
  uint64_t bytes = 0;
  uint32_t num_slots = 0;
  uint32_t held = 0;
  uint32_t cached = 0;
  uint32_t vacant = 0;
  uint64_t evicted = 0;
  uint64_t fetched_in_slots = 0;
  uint64_t fetched_in_bytes = 0;
  uint64_t served_out_slots = 0;
  uint64_t copy_micros = 0;
};

class PoolManager {
 public:
  explicit PoolManager(std::shared_ptr<MemoryApi> api);
  ~PoolManager();

  PoolManager(const PoolManager&) = delete;
  PoolManager& operator=(const PoolManager&) = delete;

  // Create the pool, or return the existing one when group, device and size
  // match. A pool with the same id and a different shape is replaced.
  AttachResult Attach(const std::string& pool_id, const std::string& group, int device,
                      uint64_t bytes);
  // Every pool of a group must declare the same per-region slot sizes.
  Status BindLayout(const std::string& pool_id, const Layout& layout);
  // Reset every slot to vacant and drop the pool's keys.
  OpenResult Open(const std::string& pool_id);
  // Up to `count` slots: vacant ones first (lowest id), then — only until
  // `min_count` slots are produced — the least recently released cached slot.
  // Returned slots are held by the owner and carry no key.
  AllocateResult Allocate(const std::string& pool_id, uint32_t count,
                          uint32_t min_count = kAllRequired);
  static constexpr uint32_t kAllRequired = UINT32_MAX;
  // `stamp` marks the slots these ops publish or release as unsettled until a
  // Commit at or past it; 0 means they are settled already.
  Status Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp = 0);
  // Every write the owner's device was issued before `committed` has landed.
  Status Commit(const std::string& pool_id, uint64_t committed);
  // For each key in order: already in this pool -> its slot; held, settled,
  // by another pool of the group -> copied into a settled free slot here,
  // which becomes cached under that key. Stops at the first key no settled
  // slot holds or no settled slot is free for.
  FetchResult Fetch(const std::string& pool_id, const std::vector<std::string>& keys);
  Status Clear(const std::string& pool_id);
  Status Detach(const std::string& pool_id);
  std::vector<PoolStats> Stats() const;

 private:
  enum class SlotState : uint8_t { kVacant, kCached, kHeld, kFilling };

  struct Slot {
    SlotState state = SlotState::kVacant;
    uint32_t pins = 0;  // in-flight copies reading this slot
    // Each settles once the pool's `committed` reaches it. A slot may be read
    // from outside when its content has landed, and filled from outside when
    // its owner has stopped using it; releasing a published slot does not
    // unsettle what it holds.
    uint64_t written = 0;
    uint64_t released = 0;
    bool has_key = false;
    std::string key;
    std::list<uint32_t>::iterator lru_it;
  };

  struct Pool {
    std::string id;
    std::string group;
    int device = -1;
    void* base = nullptr;
    uint64_t bytes = 0;
    ExportedMemory exported;
    Layout layout;
    bool bound = false;
    std::vector<Slot> slots;
    std::set<uint32_t> vacant;
    std::list<uint32_t> lru;  // cached slots, coldest first
    std::unordered_map<std::string, uint32_t> index;
    uint32_t held = 0;
    uint32_t busy = 0;       // in-flight copies into or out of this pool
    uint64_t committed = 0;  // how far the owner's device has got
    uint64_t evicted = 0;
    uint64_t fetched_in_slots = 0;
    uint64_t fetched_in_bytes = 0;
    uint64_t served_out_slots = 0;
    uint64_t copy_micros = 0;
  };

  struct Holder {
    Pool* pool = nullptr;
    uint32_t slot = 0;
  };

  struct Group {
    bool has_layout = false;
    std::vector<uint64_t> unit_bytes;
    std::unordered_map<std::string, std::vector<Holder>> holders;
  };

  struct CopyStep {
    Pool* src = nullptr;
    uint32_t src_slot = 0;
    uint32_t dst_slot = 0;
    size_t key_index = 0;
  };

  Pool* FindLocked(const std::string& pool_id);
  // Stop advertising `slot`'s key. The slot keeps its state.
  void UnindexLocked(Pool* pool, uint32_t slot);
  void ResetSlotsLocked(Pool* pool);
  // A free slot for new content, or false. Appends the key it destroyed; with
  // `evicted` null only a vacant slot is taken. `settled_only` is for a taker
  // other than the owner, which must not write where the owner still may.
  bool TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                      bool settled_only = false);
  static bool Readable(const Pool& pool, const Slot& slot) {
    return slot.written <= pool.committed;
  }
  static bool Fillable(const Pool& pool, const Slot& slot) {
    return slot.released <= pool.committed && slot.written <= pool.committed;
  }
  bool HasPinnedFreeLocked(const Pool* pool) const;
  void DestroyLocked(std::unique_lock<std::mutex>& lock, Pool* pool);

  std::shared_ptr<MemoryApi> api_;
  mutable std::mutex mutex_;
  std::condition_variable idle_;
  std::unordered_map<std::string, std::unique_ptr<Pool>> pools_;
  std::unordered_map<std::string, Group> groups_;
  std::unordered_set<std::string> attaching_;  // pool ids being allocated outside the lock
};

}  // namespace mori::umbp::l1
