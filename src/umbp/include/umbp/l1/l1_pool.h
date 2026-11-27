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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
// A pool may also be several such allocations, its parts: an engine rank that
// runs tensor-parallel attention keeps slot i of its KV on every one of its
// devices, a different shard on each. The parts share one slot table -- one
// free list, one index, one eviction order -- so slot i is the same slot on
// every device; each part has its own memory and layout and reports its own
// device's progress. A copy between two pools copies part p to part p.
//
// Below the pools (write-back): this class manages HBM and nothing else. With
// a backing store set (`SetBackingStore`; in the standalone server, the UMBP
// store the same process hosts, so DRAM on this node or wherever its master
// routes the put), a cached slot's content is written there when the slot is
// taken for new content and no other pool of the group holds that key -- the
// last L1 copy, and only then: content an engine has just computed stays in L1
// alone, and a spare copy another pool also holds is simply dropped. A fetch
// finds a key no pool holds in the store and has it read into the requester's
// slots. What the store keeps, for how long and on which node is the store's
// business; this class only remembers which of the keys it still holds it has
// already written, so that one is not written twice.
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
  // Map memory another process allocated and exported (`handle`) on `device`;
  // the pool starts `offset` bytes into it. `*mapped` is what Unimport takes.
  virtual bool Import(int device, const std::string& handle, uint64_t offset, void** base,
                      void** mapped, std::string* error) = 0;
  virtual void Unimport(int device, void* mapped) = 0;
  // Copy `bytes`. The bytes are only guaranteed visible at `dst` after a
  // Synchronize of the destination device.
  virtual bool Copy(int dst_device, void* dst, int src_device, const void* src, uint64_t bytes,
                    std::string* error) = 0;
  // Wait until every copy issued on `device` has landed.
  virtual bool Synchronize(int device, std::string* error) = 0;

  struct Fragment {
    void* dst;
    const void* src;
    uint64_t bytes;
    int src_device;
  };
  // Copy every fragment onto `dst_device` and return once all have landed.
  // Fragments must not overlap. By default one Copy each, then Synchronize.
  virtual bool CopyBatch(int dst_device, const std::vector<Fragment>& fragments,
                         std::string* error);
};

// Where the last L1 copy of a key goes when its slot is taken, and where a
// fetch looks for a key no pool holds. The pools hand it addresses inside
// their own memory; it moves the bytes, by whatever path it has, and decides
// on its own what it keeps. Injectable: the standalone server supplies the
// UMBP store it hosts, a test a fake.
//
// Every call may come from several threads at once, and none is made with the
// pool manager's lock held.
class BackingStore {
 public:
  struct Fragment {
    void* address = nullptr;
    uint64_t bytes = 0;
  };
  // One slot's content. The fragments stand part by part and, within a part,
  // region by region: the same order and sizes for every pool that shares a
  // `space`, so what one pool wrote another reads.
  struct Block {
    std::string key;
    std::vector<Fragment> fragments;
  };

  virtual ~BackingStore() = default;
  // An allocation of a pool, on `device`, that later blocks point into. Called
  // once when a pool attaches the memory, never on the copy path.
  virtual bool AddMemory(void* base, uint64_t bytes, int device, std::string* error) = 0;
  virtual void RemoveMemory(void* base) = 0;
  // Store every block under (`space`, key) and return once its bytes are out
  // of the pool. result[i]: block i is stored; a key already there counts.
  virtual std::vector<bool> Write(const std::string& space, const std::vector<Block>& blocks) = 0;
  // Fill every block's fragments with what is stored under (`space`, key);
  // the bytes have landed when this returns. result[i]: block i was filled.
  virtual std::vector<bool> Read(const std::string& space, const std::vector<Block>& blocks) = 0;
  // result[i]: keys[i] is stored under `space`. A hint: the store may drop a
  // key before it is read.
  virtual std::vector<bool> Contains(const std::string& space,
                                     const std::vector<std::string>& keys) = 0;
  // What the store's calls are waiting for right now, in a few words, for the
  // report of a call that does not come back. Must not block.
  virtual std::string Describe() { return {}; }
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

// Memory a pool adopts rather than allocates (`PoolManager::Attach`).
struct ExternalMemory {
  std::string handle;  // empty: the server allocates
  uint64_t offset = 0;
};

// Which part of a pool an Attach brings, and how many the pool has. Every
// part of a pool names the same count.
struct PartSpec {
  uint32_t index = 0;
  uint32_t count = 1;
};

// How the pools use their backing store (`PoolManager::SetBackingStore`).
struct BackingStoreOptions {
  // Keep the content of this fraction of every pool's slots -- the cached last
  // copies an eviction takes next -- already written to the store, by a
  // background thread, so the eviction does not wait for the write. 0 writes
  // only at eviction, inside the Allocate that evicts.
  double ahead = 0.0;
  // Whether an eviction writes a last copy that was not written ahead before
  // it hands the slot out. Off, the engine never waits for the store: such
  // content is lost (and counted), which is the trade when the write is slow.
  bool write_at_eviction = true;
  // Write-through: offer every published block to the store as soon as its
  // owner's writes have landed, the newest first, whether or not it is cold,
  // held, or the last copy -- the first holder whose copy has settled writes
  // it, once. For a store other ranks and nodes are to read what this rank
  // just computed from, and one that must already hold a block by the time
  // an eviction that does not wait takes it. The rules above stay as the
  // fallback for what this missed: a block the store refused, or one
  // published before the store was there.
  bool write_at_publish = false;
  // Store calls one write-back or load is split over, side by side: the store
  // copies one call's blocks on the calling thread.
  uint32_t threads = 4;
};
// From UMBP_L1_DRAM_AHEAD, UMBP_L1_DRAM_WRITE_AT_EVICTION, UMBP_L1_DRAM_THREADS
// and UMBP_L1_DRAM_WRITE (`ahead`, the default, or `publish`).
BackingStoreOptions BackingStoreOptionsFromEnv();

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

// How a Fetch places what it copies (`PoolManager::Fetch`).
struct FetchOptions {
  // Copy key i into dst_slots[i], a slot the caller holds, rather than into
  // free slots: nothing here is evicted or indexed; the caller publishes.
  // Regranted destinations must be settled (their old owner's writes
  // committed). An unsettled destination ends the returned leading prefix;
  // every supplied destination remains held by the caller, copied or not.
  std::vector<uint32_t> dst_slots;
  // Copy nothing; count the leading keys the group could supply.
  bool probe_only = false;
};

struct FetchResult {
  Status status;
  std::vector<uint32_t> slots;
  std::vector<std::string> evicted_keys;
  uint32_t copied_slots = 0;
  uint64_t copied_bytes = 0;
  uint64_t copy_micros = 0;
  // probe_only: leading keys this pool or another pool of the group holds.
  uint32_t available = 0;
};

struct PoolStats {
  std::string pool_id;
  std::string group;
  int device = -1;     // of part 0
  uint64_t bytes = 0;  // over every part
  uint32_t parts = 1;
  uint32_t num_slots = 0;
  uint32_t held = 0;
  uint32_t cached = 0;
  uint32_t vacant = 0;
  uint64_t evicted = 0;
  uint64_t fetched_in_slots = 0;
  uint64_t fetched_in_bytes = 0;
  uint64_t served_out_slots = 0;
  uint64_t copy_micros = 0;
  // Backing store: slots of this pool written to it at eviction and ahead of
  // it (write-through counts as ahead), slots filled from it, and evictions
  // of a last copy it did not get.
  uint64_t written_back_slots = 0;
  uint64_t written_ahead_slots = 0;
  uint64_t loaded_slots = 0;
  uint64_t lost_slots = 0;
  // Time this pool's calls spent in the store: lookups, writes and loads.
  // What the store holds and has room for is in the store's own metrics.
  uint64_t tier_micros = 0;
  // Blocks this pool's owner published, and blocks of this pool offered to
  // the store by any of the three paths (taken or refused).
  uint64_t published_slots = 0;
  uint64_t offered_slots = 0;
  // Now: slots of this pool whose key the store has not been given, by this
  // pool or another. With write-through this is the writer's backlog (plus
  // what has not settled yet); otherwise it is mostly copies that are not
  // last copies and are not meant to be written.
  uint64_t unstored_slots = 0;
};

// What the always-on stall diagnostics have seen (`PoolManager::Diagnostics`).
struct StallDiagnostics {
  uint64_t slow_calls = 0;   // store calls and pool copies that took over a second
  uint64_t stuck_calls = 0;  // of those, reported while still in flight
  std::string last;          // the last line logged
};

class PoolManager {
 public:
  explicit PoolManager(std::shared_ptr<MemoryApi> api);
  ~PoolManager();

  PoolManager(const PoolManager&) = delete;
  PoolManager& operator=(const PoolManager&) = delete;

  // Create the pool, or return the existing one when group, device and size
  // match. A pool with the same id and a different shape is replaced.
  // With `external.handle` the pool adopts that memory instead: the owner keeps
  // allocating and freeing it, the server only maps it and manages its slots.
  // With `part.count` > 1 this is one part of the pool: the others arrive in
  // their own calls, in any order, and the pool serves once all are bound. A
  // part attached again with another shape replaces the whole pool.
  AttachResult Attach(const std::string& pool_id, const std::string& group, int device,
                      uint64_t bytes, const ExternalMemory& external = {},
                      const PartSpec& part = {});
  // Every pool of a group must declare the same per-region slot sizes, part by
  // part, and every part of a pool the same number of slots.
  Status BindLayout(const std::string& pool_id, const Layout& layout, uint32_t part = 0);
  // Reset every slot to vacant and drop the pool's keys.
  OpenResult Open(const std::string& pool_id);
  // Up to `count` slots: vacant ones first (lowest id), then — only until
  // `min_count` slots are produced — the least recently released cached slot.
  // A vacant slot the owner's device may still be writing (released since its
  // last Commit) comes last of all, and never beyond `min_count`: a fetch
  // into it would wait for that device. Returned slots are held by the owner
  // and carry no key.
  AllocateResult Allocate(const std::string& pool_id, uint32_t count,
                          uint32_t min_count = kAllRequired);
  static constexpr uint32_t kAllRequired = UINT32_MAX;
  // More parts than a pool has devices to put them on is a caller's mistake.
  static constexpr uint32_t kMaxParts = 64;
  // `stamp` marks the slots these ops publish or release as unsettled until a
  // Commit at or past it; 0 means they are settled already.
  Status Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp = 0);
  // Every write the owner's device was issued before `committed` has landed.
  // A pool of several parts has got as far as its slowest part.
  Status Commit(const std::string& pool_id, uint64_t committed, uint32_t part = 0);
  // For each key in order: already in this pool -> its slot; held, settled,
  // by another pool of the group -> copied into a settled free slot here,
  // which becomes cached under that key. Stops at the first key no settled
  // slot holds or no settled slot is free for.
  //
  // With `options.dst_slots` key i is copied into dst_slots[i] instead, which
  // the caller must hold; keys already in this pool are not looked up here,
  // and nothing in this pool is evicted, indexed or changes state, so the
  // caller may keep allocating and applying while the copy runs. With
  // `options.probe_only` nothing is copied: `available` counts the leading
  // keys this pool or another pool of the group holds readable.
  FetchResult Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                    const FetchOptions& options = {});
  Status Clear(const std::string& pool_id);
  Status Detach(const std::string& pool_id);
  std::vector<PoolStats> Stats() const;

  // Which cached slot an eviction takes. With `keep_copies` 0 it is the least
  // recently released one. With `keep_copies` N > 0 it is the least recently
  // released spare copy: a slot whose key N pools of the group hold a newer
  // copy of, where the newest copy is the one last published, fetched or
  // taken by its engine for a request. A copy to keep goes only when the
  // pool has no spare copy left. Ranks that copy each other's prefixes end
  // up with several copies of the same content: this drops the old ones
  // first and keeps the N newest, which are where the conversation is being
  // served now. Read from UMBP_L1_KEEP_COPIES at construction (default 1;
  // 0 is plain least-recently-released).
  void SetEvictionPolicy(uint32_t keep_copies);
  // Put a store under the pools. Before any pool is attached: a pool's
  // memory is made known to the store as it attaches, and a pool attached
  // earlier does not use the store.
  void SetBackingStore(std::shared_ptr<BackingStore> store,
                       const BackingStoreOptions& options = {});
  // Stop writing to and reading from the store, and wait for the writer
  // ahead. For the store's owner, before it closes the store; the pools'
  // memory is still taken out of it as they go.
  void StopBackingStore();
  // Whether a grant passes over vacant slots their owner's device may still
  // be writing, evicting cold content instead, until nothing else is free.
  // Read from UMBP_L1_GRANT_SETTLED_FIRST at construction (default 1). Set
  // before any pool is in use.
  void SetGrantSettledFirst(bool on);
  // Whether a pool's content may be copied out of it into another pool: a
  // Fetch's source. Off, the pools of a group share nothing but the backing
  // store: a Fetch or a probe finds a key this pool does not hold in the
  // store or not at all, never in another pool. A copy another pool holds
  // then protects nothing, so every copy is written to the store as a last
  // copy is (at its eviction and ahead of it, once: the store's having it is
  // remembered with the key), and eviction takes the least recently released
  // slot whatever `keep_copies` says. For measuring what the exchange between
  // pools adds over the shared store, on one binary. Read from
  // UMBP_L1_PEER_SOURCE at construction (default 1). Set before any pool is
  // in use.
  void SetPeerSource(bool on);

  // A store call (Write, Read, Contains) or a copy between pools that takes
  // longer than `slow` is logged when it returns, with its pool, slots, bytes
  // and duration; one still in flight after `stuck` is logged once, with
  // what else is in flight and whether the pool manager's lock can be had.
  // 1 s and 10 s; settable for tests.
  void SetStallThresholds(std::chrono::milliseconds slow, std::chrono::milliseconds stuck);
  StallDiagnostics Diagnostics() const;

 private:
  enum class SlotState : uint8_t { kVacant, kCached, kHeld, kFilling };

  struct Pool;

  struct Holder {
    Pool* pool = nullptr;
    uint32_t slot = 0;
  };

  // The slots that hold one key, oldest first: nearly always one or two, so
  // they sit in the index entry itself.
  class Holders {
   public:
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    Holder* begin() { return more_.empty() ? inline_ : more_.data(); }
    Holder* end() { return begin() + size_; }
    const Holder* begin() const { return more_.empty() ? inline_ : more_.data(); }
    const Holder* end() const { return begin() + size_; }
    Holder& operator[](size_t i) { return begin()[i]; }
    const Holder& operator[](size_t i) const { return begin()[i]; }
    void push_back(const Holder& holder);
    // Drop element `i`, keeping the others' order.
    void erase(size_t i);
    // Make element `i` the newest.
    void move_to_back(size_t i);
    // The element for `pool`, or size().
    size_t find(const Pool* pool) const;

    // The backing store was given this key's content (written ahead, or it
    // was loaded from there): its last copy needs no write when evicted.
    bool stored = false;
    // A write of this key's content to the store is in flight.
    bool writing = false;

   private:
    static constexpr size_t kInline = 3;
    Holder inline_[kInline];
    uint32_t size_ = 0;
    std::vector<Holder> more_;  // every element, once there are more than kInline
  };

  // A group's content index: key -> who holds it. An entry never moves, so a
  // slot points at the entry of the key it holds and reaches its holders
  // without a lookup.
  using Content = std::unordered_map<std::string, Holders>;
  using ContentEntry = Content::value_type;

  static constexpr uint32_t kNoSlot = UINT32_MAX;

  struct Slot {
    SlotState state = SlotState::kVacant;
    // A cached slot stands in `Pool::lru`, or in `Pool::kept` when `kept`.
    bool kept = false;
    uint32_t pins = 0;  // in-flight copies reading this slot
    // Its neighbours in that list.
    uint32_t prev = kNoSlot;
    uint32_t next = kNoSlot;
    // Each settles once the pool's `committed` reaches it. A slot may be read
    // from outside when its content has landed, and filled from outside when
    // its owner has stopped using it; releasing a published slot does not
    // unsettle what it holds.
    uint64_t written = 0;
    uint64_t released = 0;
    // The key this slot holds, as its entry in the group's index; null: none.
    ContentEntry* content = nullptr;
  };

  // Slots linked through `Slot::prev` / `Slot::next`.
  struct SlotList {
    uint32_t head = kNoSlot;
    uint32_t tail = kNoSlot;
    uint32_t size = 0;
  };

  struct Group;

  // One allocation of a pool, on one device.
  struct Part {
    bool attached = false;
    int device = -1;
    void* base = nullptr;
    uint64_t bytes = 0;
    ExportedMemory exported;
    bool external = false;   // adopted, not allocated: unmap, never free
    void* mapped = nullptr;  // what Import returned, for Unimport
    std::string external_handle;
    uint64_t external_offset = 0;
    bool registered = false;  // known to the backing store (`AddMemory`)
    Layout layout;
    bool bound = false;
    uint64_t committed = 0;  // how far this part's device has got
  };

  struct Pool {
    std::string id;
    // Tells this pool from a later one attached under the same id.
    uint64_t serial = 0;
    std::string group;
    Group* content_group = nullptr;  // set once every part is bound
    std::vector<Part> parts;
    // Every part attached and bound: the pool has slots and can be opened.
    bool bound = false;
    // Every part is known to the backing store: evicted last copies go there.
    bool backed = false;
    // Names the group and how its slots are cut, for the store's key space.
    std::string space_prefix;
    uint32_t num_slots = 0;
    std::vector<Slot> slots;
    std::set<uint32_t> vacant;
    // Vacant slots their owner's device may still be writing, by the stamp
    // that settles each: handed out only when nothing else can be
    // (`settled_first_`), and moved to `vacant` once `committed` passes them.
    std::set<std::pair<uint64_t, uint32_t>> unsettled;
    SlotList lru;  // cached slots, coldest first
    // Cached slots an eviction passed over as last copies (`keep_copies`),
    // oldest first. Evicted from only when `lru` has nothing to take.
    SlotList kept;
    uint32_t held = 0;
    uint32_t busy = 0;       // in-flight copies into or out of this pool
    uint64_t committed = 0;  // how far the owner's devices have got: the slowest part
    uint64_t evicted = 0;
    uint64_t evicted_spare = 0;  // of those, evictions that left another pool holding the key
    uint64_t fetched_in_slots = 0;
    uint64_t fetched_in_bytes = 0;
    uint64_t served_out_slots = 0;
    uint64_t copy_micros = 0;
    uint64_t written_back_slots = 0;
    uint64_t written_ahead_slots = 0;
    uint64_t loaded_slots = 0;
    uint64_t lost_slots = 0;
    uint64_t tier_micros = 0;
    uint64_t published_slots = 0;
    uint64_t offered_slots = 0;
    // Slots holding a key the store has not been given (`Holders::stored`).
    uint64_t unstored = 0;
    // Write-through: slots published and not yet looked at, in the order they
    // were published -- which is the order they settle in -- and those whose
    // content has landed, waiting for the writer, which takes the newest.
    std::deque<uint32_t> fresh;
    std::deque<uint32_t> ready;
    // Where the writer ahead has got to in this pool's cached slots: it walks
    // them a slice at a time, never the whole pool under the lock.
    struct AheadPass {
      uint64_t sweep = 0;         // the sweep this pool was last walked in
      bool active = false;        // a walk is under way
      bool in_lru = false;        // which list `cursor` is in: `kept`, then `lru`
      uint32_t cursor = kNoSlot;  // the next slot to look at; kNoSlot: the list's end
      size_t budget = 0;          // last copies the walk may still look at
    } ahead;
  };

  // A slot whose content goes to the backing store.
  struct StoreSlot {
    uint32_t slot = 0;
    std::string key;
  };

  // What a store call made without the lock needs: taken under it.
  struct StoreCall {
    std::shared_ptr<BackingStore> store;  // null: the pool has no store
    std::string space;
    uint64_t generation = 0;
    uint32_t threads = 1;
  };

  struct Group {
    // Per part index, the slot size of each region; empty until a pool of the
    // group binds that part.
    std::vector<std::vector<uint64_t>> unit_bytes;
    Content content;
    uint64_t slots = 0;  // of every pool bound so far: what `content` is sized for
    // Part of the store's key space. A Clear moves the group to a new one:
    // what it wrote before can no longer be found.
    uint64_t generation = 0;
  };

  struct CopyStep {
    Pool* src = nullptr;  // null: from the backing store
    uint32_t src_slot = 0;
    uint32_t dst_slot = 0;
    size_t key_index = 0;
  };

  Pool* FindLocked(const std::string& pool_id);
  static void PushBack(Pool* pool, SlotList* list, uint32_t slot);
  static void PushFront(Pool* pool, SlotList* list, uint32_t slot);
  static void Unlink(Pool* pool, SlotList* list, uint32_t slot);
  // Stop advertising `slot`'s key. The slot keeps its state.
  void UnindexLocked(Pool* pool, uint32_t slot);
  // Take a cached slot off whichever list it stands in.
  void UncacheLocked(Pool* pool, uint32_t slot);
  // Whether evicting `slot` leaves `keep_copies_` newer copies of its key in
  // the group (or it has no key).
  bool IsSpareLocked(const Pool* pool, uint32_t slot) const;
  // A key gained a holder, its newest: the copy this pushes out of the newest
  // `keep_copies_` is spare, and the coldest spare copy of its pool.
  void HolderAddedLocked(Holders* holders);
  // The owner took `slot` for a request: its copy is the newest of its key.
  void HolderUsedLocked(Pool* pool, uint32_t slot);
  void ResetSlotsLocked(Pool* pool);
  // `slot` becomes vacant / a vacant slot is taken by its owner.
  void VacateLocked(Pool* pool, uint32_t slot);
  void UnvacateLocked(Pool* pool, uint32_t slot);
  // A free slot for new content, or false. Appends the key it destroyed; with
  // `evicted` null only a vacant slot is taken. `settled_only` is for a taker
  // other than the owner, which must not write where the owner still may.
  // With `victims`, an evicted slot whose content is the group's last L1 copy
  // and not yet in the backing store is appended: the caller writes it back
  // before anything writes the slot.
  bool TakeFreeLocked(Pool* pool, uint32_t* slot, std::vector<std::string>* evicted,
                      bool settled_only = false, std::vector<StoreSlot>* victims = nullptr);
  // The pool's store call; `.store` is null when the pool has none.
  StoreCall StoreCallLocked(const Pool* pool) const;
  bool BackedLocked(const Pool* pool) const { return store_on_ && pool->backed; }
  // Whether `slot`'s content is the last a fetch could reach in L1: no other
  // pool of the group holds its key, or pools are not each other's sources.
  bool LastCopyLocked(const Slot& slot) const;
  // How many copies of a key eviction keeps: none to speak of when no pool
  // can read another's.
  uint32_t KeepCopiesLocked() const { return peer_source_ ? keep_copies_ : 0; }
  // Whether `slot`'s content is the group's last L1 copy and the backing
  // store has not been given it.
  bool NeedsWriteBackLocked(const Pool* pool, const Slot& slot) const;
  // Write `slots` of `pool` to the backing store. The caller keeps the slots
  // from changing (it holds them, or has pinned them) and holds `busy` on the
  // pool; called without the lock. result[i]: slots[i] is stored.
  std::vector<bool> WriteBack(Pool* pool, const std::vector<StoreSlot>& slots,
                              const StoreCall& call, bool ahead);
  // The fragments of `slot`, in the order every pool of its shape names them.
  static void AppendFragments(const Pool& pool, uint32_t slot,
                              std::vector<BackingStore::Fragment>* out);
  // One Write (or Read) of `blocks`, split over `call.threads` store calls.
  static std::vector<bool> StoreBlocks(const StoreCall& call,
                                       std::vector<BackingStore::Block> blocks, bool write);
  void StoreWorker();
  // What one round of a writer did.
  struct WriterRound {
    int written = -1;          // slots the store took; -1: none were offered
    bool sweep_done = true;    // (ahead) every pool has been walked to its end
    uint64_t scan_micros = 0;  // (ahead) time spent walking under the lock
  };
  // Write-through: the newest settled published blocks of one pool.
  WriterRound WriteFresh();
  // Ahead of eviction: one slice of one pool's coldest last copies.
  WriterRound WriteAhead();
  // Pin `slots` of `pool` and write them outside the lock (which `lock`
  // holds on entry and on return); mark what the store took as stored.
  int WritePinnedLocked(std::unique_lock<std::mutex>& lock, Pool* pool,
                        const std::vector<StoreSlot>& slots);
  void StopStoreWorker();
  // A key gained or is about to lose `pool` as a holder; the store was given
  // a key; nothing a group wrote can be found any more.
  static void HolderGainedLocked(Pool* pool, const Holders& holders);
  static void HolderLostLocked(Pool* pool, const Holders& holders);
  static void MarkStoredLocked(Holders* holders);
  static void ForgetStoredLocked(Group* group);

  // A store call or a copy between pools in flight, for the stall reports.
  class Flight;
  struct FlightRecord {
    const char* what = "";
    std::string pool;
    size_t slots = 0;
    uint64_t bytes = 0;
    std::chrono::steady_clock::time_point since;
    bool reported = false;
  };
  uint64_t BeginFlight(const char* what, const std::string& pool, size_t slots, uint64_t bytes);
  void EndFlight(uint64_t id);
  void Watchdog();
  // Calls inside the pool manager, by kind, for the same reports.
  enum CallKind {
    kAttach,
    kBindLayout,
    kOpen,
    kAllocate,
    kApply,
    kCommit,
    kFetch,
    kClear,
    kDetach,
    kCallKinds
  };
  class CallScope;
  static bool Readable(const Pool& pool, const Slot& slot) {
    return slot.written <= pool.committed;
  }
  static bool Fillable(const Pool& pool, const Slot& slot) {
    return slot.released <= pool.committed && slot.written <= pool.committed;
  }
  bool HasPinnedFreeLocked(const Pool* pool) const;
  // Waits, without the lock, for the pool's copies to finish. Destroys nothing
  // if the pool was detached or replaced under its id in the meantime.
  void DestroyLocked(std::unique_lock<std::mutex>& lock, Pool* pool);
  // Take a part's memory out of the backing store, then unmap it if adopted
  // or free it if allocated. Without the lock: the store may make this wait
  // for its calls in flight.
  void ReleasePart(const Part& part);
  // Whether a slot of `src` can be copied into `dst`: the same parts, each
  // with the same regions.
  static bool SameShape(const Pool& src, const Pool& dst);

  std::shared_ptr<MemoryApi> api_;
  uint32_t keep_copies_ = 0;
  std::shared_ptr<BackingStore> store_;
  // Off once the store's owner is closing it: `store_` then only takes the
  // pools' memory back.
  bool store_on_ = false;
  BackingStoreOptions store_options_;
  std::thread store_worker_;  // the writer: write-through and ahead of eviction
  std::condition_variable store_wake_;
  bool stopping_ = false;
  // The writer ahead's sweep over the pools, the pool it walked last, and
  // when it may walk again: it leaves the lock alone nine tenths of the time.
  uint64_t ahead_sweep_ = 1;
  uint64_t ahead_last_serial_ = 0;
  std::chrono::steady_clock::time_point ahead_not_before_{};
  uint64_t fresh_last_serial_ = 0;
  // A grant prefers evicting cold content to handing out a slot whose owner's
  // writes have not landed (UMBP_L1_GRANT_SETTLED_FIRST, default on).
  bool settled_first_ = true;
  // Pools are each other's fetch sources (UMBP_L1_PEER_SOURCE, default on).
  bool peer_source_ = true;
  mutable std::mutex mutex_;
  std::condition_variable idle_;
  std::unordered_map<std::string, std::unique_ptr<Pool>> pools_;
  std::unordered_map<std::string, Group> groups_;
  // "pool id#part" of each part being allocated or mapped outside the lock
  std::unordered_set<std::string> attaching_;
  uint64_t next_serial_ = 1;

  // Stall diagnostics. Their own lock: a report must come out when the pool
  // manager's is the thing that is stuck.
  mutable std::mutex flights_mutex_;
  std::unordered_map<uint64_t, FlightRecord> flights_;
  uint64_t next_flight_ = 1;
  std::chrono::milliseconds slow_call_{1000};
  std::chrono::milliseconds stuck_call_{10000};
  StallDiagnostics diagnostics_;
  std::condition_variable watchdog_wake_;
  bool watchdog_stop_ = false;
  std::thread watchdog_;
  std::atomic<int> calls_[kCallKinds] = {};
};

}  // namespace mori::umbp::l1
