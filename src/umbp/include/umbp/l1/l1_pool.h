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
#include <thread>
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
//
// Lower tiers: with a LowerStore attached, L1 is the top of a hierarchy this
// process manages end to end. A cached slot's content is copied into the store
// before the slot is reused (offload), and a key no L1 pool of this node holds
// is looked for in the store, and through it on other nodes, before a fetch
// gives up (load). The store is UMBP's own: host DRAM, SSD and, with a master,
// the other nodes' stores, moved between by the store's own tier policy.

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
  // Many copies into one device from one device. A fetch whose slots are
  // scattered is hundreds of thousands of small copies; an implementation can
  // take its locks and select the device once for all of them.
  // A job is `rows` pieces of `bytes`, `src_pitch` apart at the source and
  // `dst_pitch` apart at the destination. Regions of one size that follow each
  // other at a constant distance make one such rectangle out of a block, which
  // a device copies in one command instead of one per region.
  struct CopyJob {
    void* dst = nullptr;
    const void* src = nullptr;
    uint64_t bytes = 0;
    uint64_t rows = 1;
    uint64_t dst_pitch = 0;
    uint64_t src_pitch = 0;
  };
  virtual bool CopyMany(int dst_device, int src_device, const std::vector<CopyJob>& jobs,
                        std::string* error) {
    for (const CopyJob& job : jobs) {
      for (uint64_t row = 0; row < job.rows; ++row) {
        if (!Copy(dst_device, static_cast<char*>(job.dst) + row * job.dst_pitch, src_device,
                  static_cast<const char*>(job.src) + row * job.src_pitch, job.bytes, error)) {
          return false;
        }
      }
    }
    return true;
  }
  // Wait until every copy issued on `device` has landed.
  virtual bool Synchronize(int device, std::string* error) = 0;
  // Host memory to stage bytes on their way to or from the lower store.
  virtual void* AllocateHost(uint64_t bytes) = 0;
  virtual void FreeHost(void* base) = 0;
  // Device to host, complete on return; host to device, visible after a
  // Synchronize of the device.
  virtual bool CopyToHost(int device, const void* src, void* host_dst, uint64_t bytes,
                          std::string* error) = 0;
  virtual bool CopyFromHost(int device, void* dst, const void* host_src, uint64_t bytes,
                            std::string* error) = 0;
  // Many copies between one device and host memory, as CopyMany's jobs: a
  // block leaves for the lower store, or comes back, as one rectangle per
  // group of like regions, straight into the layout the store keeps.
  virtual bool CopyManyToHost(int device, const std::vector<CopyJob>& jobs, std::string* error) {
    for (const CopyJob& job : jobs) {
      for (uint64_t row = 0; row < job.rows; ++row) {
        if (!CopyToHost(device, static_cast<const char*>(job.src) + row * job.src_pitch,
                        static_cast<char*>(job.dst) + row * job.dst_pitch, job.bytes, error)) {
          return false;
        }
      }
    }
    return true;
  }
  virtual bool CopyManyFromHost(int device, const std::vector<CopyJob>& jobs, std::string* error) {
    for (const CopyJob& job : jobs) {
      for (uint64_t row = 0; row < job.rows; ++row) {
        if (!CopyFromHost(device, static_cast<char*>(job.dst) + row * job.dst_pitch,
                          static_cast<const char*>(job.src) + row * job.src_pitch, job.bytes,
                          error)) {
          return false;
        }
      }
    }
    return true;
  }
};

// The tiers below L1, reached through this process's UMBP store. Values are
// whole slots, `size` contiguous bytes of host memory each. Every call may
// block on I/O or on another node and is made without the pool lock.
class LowerStore {
 public:
  virtual ~LowerStore() = default;
  // Host memory this store may be handed in Put and Get.
  virtual bool RegisterHostMemory(void* base, uint64_t bytes) = 0;
  virtual void DeregisterHostMemory(void* base) = 0;
  // result[i]: the store can serve keys[i], from this node or another.
  virtual std::vector<bool> Exists(const std::vector<std::string>& keys) = 0;
  virtual std::vector<bool> Put(const std::vector<std::string>& keys,
                                const std::vector<const void*>& srcs, uint64_t size) = 0;
  virtual std::vector<bool> Get(const std::vector<std::string>& keys,
                                const std::vector<void*>& dsts, uint64_t size) = 0;
  // Make what Put stored visible to the other nodes now.
  virtual void Publish() = 0;
  // The L1 directory: tell the other nodes which keys this node holds in HBM
  // only, so they know to ask (RequestFromPeers).
  virtual void Advertise(const std::vector<std::string>& added,
                         const std::vector<std::string>& removed) = 0;
  // Ask the nodes that advertise `keys` to copy them into the store.
  // result[i]: keys[i] can now be served by Get.
  virtual std::vector<bool> RequestFromPeers(const std::vector<std::string>& keys) = 0;
};

struct TierOptions {
  // Host staging for offload and load; a batch moves at most this many bytes.
  uint64_t staging_bytes = 1ull << 30;
  // Copy a cached slot into the store before the slot is reused.
  bool offload_on_evict = true;
  // Keep this fraction of every pool's slots either vacant or already
  // offloaded, by copying the coldest cached slots ahead of need. 0 offloads
  // only at eviction; 1 offloads everything once it has settled.
  double offload_ahead = 0.0;
  // Look in the store for keys no pool of this node holds.
  bool load = true;
  // Ask other nodes' L1 for keys neither this node nor the store holds.
  bool ask_peers = true;
  // Copy a block straight from another pool of this node that holds it. Off,
  // pools of a node share only through the store: what an engine sees when
  // its ranks exchange KV through DRAM and not through HBM.
  bool peer_copy = true;
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
  uint32_t copied_slots = 0;  // from another pool of this node
  uint64_t copied_bytes = 0;
  uint64_t copy_micros = 0;
  uint32_t loaded_slots = 0;  // from the lower store
  uint64_t loaded_bytes = 0;
  uint64_t load_micros = 0;
};

struct ExportResult {
  Status status;
  std::vector<bool> stored;  // per key: now in the lower store
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
  uint64_t offloaded_slots = 0;  // copied into the lower store
  uint64_t offload_micros = 0;
  uint64_t loaded_slots = 0;  // filled from the lower store
  uint64_t load_micros = 0;
  uint64_t dropped_slots = 0;  // evicted without a copy below
};

class PoolManager {
 public:
  explicit PoolManager(std::shared_ptr<MemoryApi> api);
  ~PoolManager();

  // Attach the tiers below L1. Call once, before any pool is attached.
  void SetLowerStore(std::shared_ptr<LowerStore> store, const TierOptions& options);
  // The name a slot's content has in the lower store.
  static std::string StoreKey(const std::string& group, const std::string& key);
  // The inverse; false for a name StoreKey did not produce.
  static bool ParseStoreKey(const std::string& name, std::string* group, std::string* key);

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
  // by another pool of the group -> copied into a settled free slot here; in
  // the lower store (on this node or another, or in another node's L1 once
  // that node has been asked) -> loaded into a settled free slot here. A
  // filled slot becomes cached under that key. Stops at the first key nothing
  // holds or no settled slot is free for.
  FetchResult Fetch(const std::string& pool_id, const std::vector<std::string>& keys);
  // Copy what this node's pools of `group` hold of `keys` into the lower
  // store, for another node to read. The slots stay as they are.
  ExportResult Export(const std::string& group, const std::vector<std::string>& keys);
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
    // The lower store was given this content (or it came from there). A hint:
    // the store may have dropped it since.
    bool in_store = false;
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
    uint64_t offloaded_slots = 0;
    uint64_t offload_micros = 0;
    uint64_t loaded_slots = 0;
    uint64_t load_micros = 0;
    uint64_t dropped_slots = 0;
    uint64_t slot_bytes = 0;  // sum of the regions' unit sizes
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
    Pool* src = nullptr;  // null: the content comes from the lower store
    uint32_t src_slot = 0;
    uint32_t dst_slot = 0;
    size_t key_index = 0;
  };

  // A slot whose content should reach the lower store.
  struct Victim {
    uint32_t slot = 0;
    std::string key;
  };

  Pool* FindLocked(const std::string& pool_id);
  // Stop advertising `slot`'s key. The slot keeps its state.
  void UnindexLocked(Pool* pool, uint32_t slot);
  void ResetSlotsLocked(Pool* pool);
  // A free slot for new content, or false. Appends the key it destroyed; with
  // `evicted` null only a vacant slot is taken. `settled_only` is for a taker
  // other than the owner, which must not write where the owner still may.
  // When the destroyed content should be offloaded first it is appended to
  // `victims`, and the caller copies it out before anything writes the slot.
  bool TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                      bool settled_only = false, std::vector<Victim>* victims = nullptr);
  // Copy slots into the lower store / fill slots from it. Called without the
  // pool lock, on slots nothing else may change meanwhile. result[i] per slot.
  // `evicting`: the slots are being reused, so a failed copy loses content.
  std::vector<bool> OffloadSlots(Pool* pool, const std::vector<Victim>& slots, bool evicting);
  std::vector<bool> LoadSlots(Pool* pool, const std::vector<Victim>& slots, uint64_t* bytes);
  void QueueAdvertiseLocked(const std::string& group, const std::string& key, bool added);
  void TierWorker();
  // One round of offloading ahead of need. True when something was copied.
  bool OffloadAhead();
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

  std::shared_ptr<LowerStore> store_;
  TierOptions tier_;
  // One staging arena; tier transfers take turns on it.
  std::mutex staging_mutex_;
  void* staging_ = nullptr;
  // Same size, for regrouping a batch between the store's layout and a pool's.
  // Directory changes not yet sent: store key -> added (true) or removed.
  std::unordered_map<std::string, bool> advertise_;
  // Offloaded copies the other nodes have not been told of yet.
  bool publish_pending_ = false;
  std::condition_variable work_;
  std::thread worker_;
  bool stopping_ = false;
};

}  // namespace mori::umbp::l1
