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
#include "umbp/distributed/transfer/mori_io_engine.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <msgpack.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {

namespace {

// Do `plan` and `item` name the same two endpoints?  Compares engine keys, not
// just ids: a MemoryUniqueId is allocated per IOEngine, so two different peers
// can both publish buffer id 1.
bool SameEndpoints(const TransferPlan& plan, const TransferRef& src, const TransferRef& dst) {
  return plan.src.mem.id == src.mem.id && plan.dst.mem.id == dst.mem.id &&
         plan.src.mem.engineKey == src.mem.engineKey && plan.dst.mem.engineKey == dst.mem.engineKey;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Handles
// ---------------------------------------------------------------------------

// A handle over posted-but-not-waited mori-io work.
//
// Lifetime contract, inherited verbatim from the pre-refactor
// RemoteDram{Get,Put}InFlight: the RDMA backend keeps a RAW TransferStatus*
// into `statuses`, so each PostSet is heap-owned and its status vector is built
// once at final size and never resized or moved.
class MoriIoEngine::RdmaHandle final : public TransferHandle {
 public:
  // One BatchRead or one BatchWrite.  mori-io's batch calls are direction-typed
  // (local-dest + remote-src vs local-src + remote-dest), so a plan set that
  // mixes directions posts one PostSet per direction.
  struct PostSet {
    mori::io::MemDescVec local_descs;
    mori::io::MemDescVec remote_descs;
    mori::io::BatchSizeVec local_offsets;
    mori::io::BatchSizeVec remote_offsets;
    mori::io::BatchSizeVec sizes;
    std::vector<mori::io::TransferStatus> statuses;  // built once, never resized/moved
    mori::io::TransferStatusPtrVec status_ptrs;
    mori::io::TransferUniqueIdVec ids;
    std::vector<std::vector<size_t>> tags;  // per group
    std::vector<std::string> endpoints;     // per group, remote engine key
    bool is_pull = false;
  };

  ~RdmaHandle() override {
    // Safety net: on an early/exceptional destroy (before Wait) drain so the
    // backend's completion callback never writes freed memory.  No failure
    // mapping here — that is Wait's job.
    if (drained_) return;
    for (auto& set : sets_) {
      for (auto& s : set->statuses) s.Wait();
    }
  }

  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    drained_ = true;
    if (failures != nullptr) {
      failures->insert(failures->end(), std::make_move_iterator(presettled_.begin()),
                       std::make_move_iterator(presettled_.end()));
    }
    presettled_.clear();
    // Wait every group; never break early, so no status is left live.
    for (auto& set : sets_) {
      for (size_t g = 0; g < set->statuses.size(); ++g) {
        set->statuses[g].Wait();
        if (set->statuses[g].Succeeded()) continue;
        if (failures == nullptr) continue;
        TransferFailure f;
        f.tags = set->tags[g];
        f.code = set->statuses[g].CodeUint32();
        f.message = set->statuses[g].Message();
        f.endpoint = set->endpoints[g];
        failures->push_back(std::move(f));
      }
    }
  }

  PostSet* AddPostSet() {
    sets_.push_back(std::make_unique<PostSet>());
    return sets_.back().get();
  }
  bool Empty() const { return sets_.empty(); }
  void AddPresettled(TransferFailure f) { presettled_.push_back(std::move(f)); }

 private:
  std::vector<std::unique_ptr<PostSet>> sets_;
  // Failures from bounce plans that Submit already ran to completion.
  std::vector<TransferFailure> presettled_;
  bool drained_ = false;
};

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

MoriIoEngine::MoriIoEngine(std::string engine_key, mori::io::IOEngineConfig io_config,
                           uint64_t bounce_bytes)
    : local_engine_key_(std::move(engine_key)),
      io_config_(std::move(io_config)),
      bounce_size_(bounce_bytes) {}

MoriIoEngine::~MoriIoEngine() { Shutdown(); }

bool MoriIoEngine::Init() {
  if (io_engine_ != nullptr) return true;
  if (io_config_.host.empty()) return false;

  io_engine_ = std::make_unique<mori::io::IOEngine>(local_engine_key_, io_config_);
  // Adopt whatever key the engine actually published: IsRemoteMemory compares
  // a descriptor's engineKey against this, so a transformed key here would make
  // every local buffer look remote.
  local_engine_key_ = io_engine_->GetEngineDesc().key;

  mori::io::RdmaBackendConfig rdma_cfg;
  rdma_cfg.qpPerTransfer = 4;
  rdma_cfg.enableTransferChunking = true;
  rdma_cfg.numNicsPerTransfer = 4;
  // UMBP only sets the config defaults above.  All RDMA knobs (qpPerTransfer /
  // postBatchSize / numWorkerThreads / pollCqMode / chunking / numNics / ...)
  // are overridable via MORI_IO_* env in the RDMA backend ctor, shared by every
  // IO backend user; no UMBP-specific entry points.
  io_engine_->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);

  if (bounce_size_ > 0) {
    bounce_buffer_ = std::make_unique<char[]>(bounce_size_);
    std::memset(bounce_buffer_.get(), 0, bounce_size_);
    bounce_ref_ = RegisterMemory(bounce_buffer_.get(), bounce_size_,
                                 mori::io::MemoryLocationType::CPU, /*device=*/-1);
    if (!bounce_ref_.HasMemoryDesc()) {
      MORI_UMBP_WARN("[MoriIoEngine] bounce buffer could not be registered; staging disabled");
      bounce_buffer_.reset();
      bounce_ref_ = TransferRef{};
      bounce_size_ = 0;
    }
  }

  MORI_UMBP_INFO("[MoriIoEngine] initialized on {}:{} bounce_bytes={}", io_config_.host,
                 io_config_.port, bounce_size_);
  return true;
}

void MoriIoEngine::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(remotes_mutex_);
    remotes_.clear();
  }
  if (io_engine_ != nullptr && bounce_buffer_ != nullptr) {
    io_engine_->DeregisterMemory(bounce_ref_.mem);
  }
  bounce_ref_ = TransferRef{};
  bounce_buffer_.reset();
  bounce_size_ = 0;
  io_engine_.reset();
}

// ---------------------------------------------------------------------------
//  Registration
// ---------------------------------------------------------------------------

TransferRef MoriIoEngine::RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                                         int device) {
  TransferRef ref = TransferRef::HostBytes(base, size, loc, device);
  if (io_engine_ == nullptr || base == nullptr || size == 0) return ref;
  ref.mem = io_engine_->RegisterMemory(base, size, device, loc);
  return ref;
}

void MoriIoEngine::Deregister(const TransferRef& ref) {
  if (io_engine_ == nullptr || !ref.HasMemoryDesc()) return;
  io_engine_->DeregisterMemory(ref.mem);
}

std::vector<uint8_t> MoriIoEngine::PackedLocalEngineDesc() const {
  if (io_engine_ == nullptr) return {};
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, io_engine_->GetEngineDesc());
  return std::vector<uint8_t>(sbuf.data(), sbuf.data() + sbuf.size());
}

// ---------------------------------------------------------------------------
//  Remote endpoints
// ---------------------------------------------------------------------------

bool MoriIoEngine::EnsureRemoteEngine(const std::string& node_id,
                                      const std::string& packed_engine_desc) {
  if (packed_engine_desc.empty()) {
    // Nothing new to learn: success only if we already know this peer's engine
    // (or if there is no engine at all, in which case there is nothing to
    // register and the caller's transfers will be rejected by CanHandle).
    if (io_engine_ == nullptr) return true;
    return HasRemoteEngine(node_id);
  }

  mori::io::EngineDesc desc;
  try {
    auto handle = msgpack::unpack(packed_engine_desc.data(), packed_engine_desc.size());
    desc = handle.get().as<mori::io::EngineDesc>();
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[MoriIoEngine] EnsureRemoteEngine: bad engine desc from '{}': {}", node_id,
                    e.what());
    return false;
  }

  if (io_engine_ != nullptr) io_engine_->RegisterRemoteEngine(desc);

  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto& remote = remotes_[node_id];
  remote.engine_desc = std::move(desc);
  remote.engine_registered = (io_engine_ != nullptr);
  return true;
}

bool MoriIoEngine::HasRemoteEngine(const std::string& node_id) const {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto it = remotes_.find(node_id);
  return it != remotes_.end() && it->second.engine_registered;
}

void MoriIoEngine::ForgetRemote(const std::string& node_id) {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  remotes_.erase(node_id);
}

void MoriIoEngine::CacheRemoteBuffers(const std::string& node_id,
                                      const std::vector<BufferMemoryDescBytes>& descs) {
  if (descs.empty()) return;
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto& remote = remotes_[node_id];
  for (const auto& d : descs) {
    if (d.backend_id >= kMaxBackendsPerPeer) {
      MORI_UMBP_ERROR("[MoriIoEngine] CacheRemoteBuffers: bad backend_id={} from '{}' (max {})",
                      d.backend_id, node_id, kMaxBackendsPerPeer);
      continue;
    }
    auto& shelf = remote.buffers[d.backend_id];
    if (shelf.size() <= d.buffer_index) shelf.resize(d.buffer_index + 1);
    if (d.desc_bytes.empty()) continue;

    mori::io::MemoryDesc mem;
    try {
      auto handle =
          msgpack::unpack(reinterpret_cast<const char*>(d.desc_bytes.data()), d.desc_bytes.size());
      mem = handle.get().as<mori::io::MemoryDesc>();
    } catch (const std::exception& e) {
      MORI_UMBP_ERROR(
          "[MoriIoEngine] CacheRemoteBuffers: bad desc from '{}' backend={} buffer_index={}: {}",
          node_id, d.backend_id, d.buffer_index, e.what());
      continue;
    }

    if (shelf[d.buffer_index].HasMemoryDesc()) {
      // Already hydrated.  First-write-wins is only safe while a descriptor is
      // immutable for a given address — a CHANGED descriptor means two buffers
      // are claiming one address, which does not fail loudly on its own: the
      // reader would keep planning RDMA against valid-looking memory and return
      // the wrong bytes as a hit.  That is exactly how the pre-backend_id
      // collision stayed invisible, so say so and keep the incumbent.
      const auto& held = shelf[d.buffer_index].mem;
      if (!(held == mem)) {
        MORI_UMBP_ERROR(
            "[MoriIoEngine] CacheRemoteBuffers: conflicting desc from '{}' backend={} "
            "buffer_index={} (held id={} data={:#x} size={}, offered id={} data={:#x} size={}); "
            "keeping the held one — the peer is publishing two buffers at one address",
            node_id, d.backend_id, d.buffer_index, held.id, held.data, held.size, mem.id, mem.data,
            mem.size);
      }
      continue;
    }
    shelf[d.buffer_index] = TransferRef::Remote(std::move(mem));
  }
}

bool MoriIoEngine::HasRemoteBuffers(const std::string& node_id) const {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto it = remotes_.find(node_id);
  if (it == remotes_.end()) return false;
  // True means "this peer's buffers are hydrated", which is what the caller
  // turns into BatchResolveKeysRequest.omit_descs.  GetPeerInfo publishes every
  // backend at once, so any non-empty shelf means the whole handshake landed.
  for (const auto& shelf : it->second.buffers) {
    if (!shelf.empty()) return true;
  }
  return false;
}

std::vector<TransferRef> MoriIoEngine::RemoteBufferSnapshot(const std::string& node_id,
                                                            uint32_t backend_id) const {
  if (backend_id >= kMaxBackendsPerPeer) return {};
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto it = remotes_.find(node_id);
  if (it == remotes_.end()) return {};
  return it->second.buffers[backend_id];
}

// ---------------------------------------------------------------------------
//  Planning
// ---------------------------------------------------------------------------

namespace {

// A ref names memory this engine could reach over the wire.
bool IsRemoteMemory(const TransferRef& ref, const std::string& local_key) {
  return ref.HasMemoryDesc() && ref.mem.engineKey != local_key;
}

// Whether the local endpoint can be staged through the host bounce pool.
//
// Addressability is not enough: staging is a std::memcpy in and out of a
// host-allocated region, so a GPU endpoint cannot take this path at all.  An
// unregistered GPU buffer is therefore rejected by Plan and surfaces as a
// per-key failure, rather than faulting inside the memcpy.  This is the single
// place the decision is made, which is why it belongs here and not at each of
// PoolClient's call sites — the transfer layer is the only layer that knows
// whether staging is involved.
bool IsBounceable(const TransferRef& local) {
  return local.HasHostPtr() && local.loc == mori::io::MemoryLocationType::CPU;
}

}  // namespace

bool MoriIoEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  if (io_engine_ == nullptr) return false;
  const bool src_remote = IsRemoteMemory(src, local_engine_key_);
  const bool dst_remote = IsRemoteMemory(dst, local_engine_key_);
  if (src_remote == dst_remote) return false;  // both local or both remote
  const TransferRef& local = src_remote ? dst : src;
  if (local.HasMemoryDesc()) return true;          // zero copy
  return IsBounceable(local) && bounce_size_ > 0;  // staged through the pool
}

namespace {

// Plans for MoriIoEngine, in first-appearance order so submit order stays
// deterministic.
//
// Zero copy: items between the same two registered buffers share a plan,
// coalescing segments that are exactly contiguous on BOTH sides.  This is NOT
// redundant with the RDMA backend's own WR merging: it shrinks the SG vector
// the backend sorts/merges (O(M log M)) and allocates, which matters for a big
// batch x pages.  A merged segment must fit ibv_sge.length (uint32_t).
//
// Staged: an item whose local side is unregistered rides the bounce pool.  It
// groups by (remote buffer, direction), takes the plan's next bounce offset,
// and once a plan's pool is full it is retired for a fresh one -- several
// plans over one pool are fine because Submit runs each to completion before
// the next.  An item larger than the whole pool is rejected.
struct MoriIoPlanTraits {
  static constexpr PlanGrouping kGrouping = PlanGrouping::kManyPerKey;
  // Memory ids are allocated per IOEngine, so two peers can both publish id 1:
  // a key only buckets endpoints, and Joins tells them apart by engine key.
  struct Key {
    bool bounce;
    TransferDirection dir;
    mori::io::MemoryUniqueId first;   // src id; the remote id when staged
    mori::io::MemoryUniqueId second;  // dst id; 0 when staged
    bool operator==(const Key& o) const noexcept {
      return bounce == o.bounce && dir == o.dir && first == o.first && second == o.second;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      const size_t ids = HashCombine(std::hash<mori::io::MemoryUniqueId>{}(k.first),
                                     std::hash<mori::io::MemoryUniqueId>{}(k.second));
      return HashCombine(ids, (k.bounce ? 2u : 0u) | (k.dir == TransferDirection::kPush ? 1u : 0u));
    }
  };

  const std::string& local_key;
  uint64_t bounce_size;
  const TransferRef& bounce_ref;
  // Per staged plan, by plan index: the next free bounce offset, and whether a
  // full pool retired the plan.
  struct BounceState {
    uint64_t cursor = 0;
    bool retired = false;
  };
  std::vector<BounceState> bounce;

  static constexpr size_t kMaxSgeBytes = std::numeric_limits<uint32_t>::max();

  std::optional<Key> Admit(const TransferItem& item) const {
    const bool src_remote = IsRemoteMemory(item.src, local_key);
    const bool dst_remote = IsRemoteMemory(item.dst, local_key);
    if (src_remote == dst_remote) return std::nullopt;  // both local or both remote
    const TransferDirection dir = dst_remote ? TransferDirection::kPush : TransferDirection::kPull;
    const TransferRef& local = src_remote ? item.dst : item.src;
    if (local.HasMemoryDesc()) return Key{false, dir, item.src.mem.id, item.dst.mem.id};
    if (bounce_size == 0 || !IsBounceable(local) || item.size > bounce_size) return std::nullopt;
    const TransferRef& remote = src_remote ? item.src : item.dst;
    return Key{true, dir, remote.mem.id, 0};
  }

  bool Joins(size_t index, const TransferPlan& plan, const Key& key, const TransferItem& item) {
    if (!key.bounce) return SameEndpoints(plan, item.src, item.dst);
    const bool push = key.dir == TransferDirection::kPush;
    const TransferRef& remote = push ? item.dst : item.src;
    const TransferRef& plan_remote = push ? plan.dst : plan.src;
    if (plan.dir != key.dir || plan_remote.mem.engineKey != remote.mem.engineKey) return false;
    BounceState& state = bounce[index];
    if (state.retired) return false;
    if (state.cursor + item.size <= bounce_size) return true;
    state.retired = true;  // full: the item opens a fresh plan instead
    return false;
  }

  TransferPlan NewPlan(const Key& key, const TransferItem& item) const {
    TransferPlan plan;
    plan.dir = key.dir;
    if (!key.bounce) {
      plan.src = item.src;
      plan.dst = item.dst;
    } else if (key.dir == TransferDirection::kPush) {
      // The local endpoint IS the bounce region; the remote side is unchanged.
      plan.uses_bounce = true;
      plan.src = bounce_ref;
      plan.dst = item.dst;
    } else {
      plan.uses_bounce = true;
      plan.src = item.src;
      plan.dst = bounce_ref;
    }
    return plan;
  }

  void Append(size_t index, TransferPlan* plan, const Key& key, const TransferItem& item) {
    if (!key.bounce) {
      AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag, true,
                    kMaxSgeBytes);
      return;
    }
    if (bounce.size() <= index) bounce.resize(index + 1);
    const uint64_t bounce_off = bounce[index].cursor;
    bounce[index].cursor += item.size;
    const bool push = key.dir == TransferDirection::kPush;
    const TransferRef& local = push ? item.src : item.dst;
    const uint64_t local_off = push ? item.src_offset : item.dst_offset;
    const uint64_t remote_off = push ? item.dst_offset : item.src_offset;
    if (push) {
      AppendSegment(plan, bounce_off, remote_off, item.size, item.tag, true, kMaxSgeBytes);
    } else {
      AppendSegment(plan, remote_off, bounce_off, item.size, item.tag, true, kMaxSgeBytes);
    }
    plan->bounce_copies.push_back(TransferPlan::BounceCopy{
        static_cast<char*>(local.host_ptr) + local_off, bounce_off, item.size});
  }
};

}  // namespace

std::unique_ptr<TransferPlanBuilder> MoriIoEngine::CreatePlanBuilder(size_t expected_items) const {
  return std::make_unique<GroupedPlanBuilder<MoriIoPlanTraits>>(
      MoriIoPlanTraits{local_engine_key_, bounce_size_, bounce_ref_, {}}, expected_items);
}

TransferPlanSet MoriIoEngine::Plan(const std::vector<TransferItem>& items) const {
  return PlanViaBuilder(*this, items);
}

// ---------------------------------------------------------------------------
//  Submission
// ---------------------------------------------------------------------------

bool MoriIoEngine::PostPlans(const std::vector<TransferPlan>& plans, RdmaHandle* handle) {
  // One PostSet per direction: mori-io's batch calls are direction-typed.
  RdmaHandle::PostSet* pull = nullptr;
  RdmaHandle::PostSet* push = nullptr;
  auto set_for = [&](TransferDirection dir) -> RdmaHandle::PostSet* {
    RdmaHandle::PostSet*& slot = (dir == TransferDirection::kPull) ? pull : push;
    if (slot == nullptr) {
      slot = handle->AddPostSet();
      slot->is_pull = (dir == TransferDirection::kPull);
    }
    return slot;
  };

  for (const auto& plan : plans) {
    RdmaHandle::PostSet* set = set_for(plan.dir);
    const bool pulling = (plan.dir == TransferDirection::kPull);
    // local = dst when pulling, src when pushing.
    set->local_descs.push_back(pulling ? plan.dst.mem : plan.src.mem);
    set->remote_descs.push_back(pulling ? plan.src.mem : plan.dst.mem);
    set->local_offsets.push_back(pulling ? plan.dst_offsets : plan.src_offsets);
    set->remote_offsets.push_back(pulling ? plan.src_offsets : plan.dst_offsets);
    set->sizes.push_back(plan.sizes);
    set->tags.push_back(plan.tags);
    set->endpoints.push_back(pulling ? plan.src.mem.engineKey : plan.dst.mem.engineKey);
  }

  bool posted = false;
  for (RdmaHandle::PostSet* set : {pull, push}) {
    if (set == nullptr) continue;
    const size_t g = set->local_descs.size();
    // Built once at final size and never resized/moved: the backend holds raw
    // TransferStatus* into this vector for the life of the post.
    set->statuses = std::vector<mori::io::TransferStatus>(g);
    set->status_ptrs.resize(g);
    set->ids.resize(g);
    for (size_t i = 0; i < g; ++i) {
      set->status_ptrs[i] = &set->statuses[i];
      set->ids[i] = io_engine_->AllocateTransferUniqueId();
    }
    if (set->is_pull) {
      io_engine_->BatchRead(set->local_descs, set->local_offsets, set->remote_descs,
                            set->remote_offsets, set->sizes, set->status_ptrs, set->ids);
    } else {
      io_engine_->BatchWrite(set->local_descs, set->local_offsets, set->remote_descs,
                             set->remote_offsets, set->sizes, set->status_ptrs, set->ids);
    }
    posted = true;
  }
  return posted;
}

bool MoriIoEngine::RunBouncePlanInline(const TransferPlan& plan, TransferFailure* failure) {
  auto fail = [&](uint32_t code, std::string msg) {
    failure->tags = plan.tags;
    failure->code = code;
    failure->message = std::move(msg);
    failure->endpoint =
        (plan.dir == TransferDirection::kPull) ? plan.src.mem.engineKey : plan.dst.mem.engineKey;
    return false;
  };

  // Held for the whole staged round trip.  Never held across a return, which is
  // why a submit-all loop over several peers cannot deadlock on it.
  std::lock_guard<std::mutex> lock(bounce_mutex_);
  if (bounce_buffer_ == nullptr) return fail(0, "bounce buffer not configured");

  const bool pushing = (plan.dir == TransferDirection::kPush);
  if (pushing) {
    for (const auto& c : plan.bounce_copies) {
      if (c.bounce_offset + c.size > bounce_size_) return fail(0, "bounce offset overflow");
      std::memcpy(bounce_buffer_.get() + c.bounce_offset, c.host, c.size);
    }
  }

  RdmaHandle inflight;
  if (!PostPlans({plan}, &inflight)) return fail(0, "post failed");
  std::vector<TransferFailure> failures;
  inflight.Wait(&failures);
  if (!failures.empty()) {
    *failure = std::move(failures.front());
    return false;
  }

  if (!pushing) {
    for (const auto& c : plan.bounce_copies) {
      if (c.bounce_offset + c.size > bounce_size_) return fail(0, "bounce offset overflow");
      std::memcpy(c.host, bounce_buffer_.get() + c.bounce_offset, c.size);
    }
  }
  return true;
}

std::unique_ptr<TransferHandle> MoriIoEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty() || io_engine_ == nullptr) return nullptr;

  // Post the zero-copy plans FIRST (they do not block), then run the staged
  // ones inline so the staged round trips overlap the posted wire rather than
  // serializing behind it.
  std::vector<TransferPlan> direct;
  std::vector<size_t> bounced;
  direct.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    if (plans[i].uses_bounce) {
      bounced.push_back(i);
    } else {
      direct.push_back(std::move(plans[i]));
    }
  }

  auto handle = std::make_unique<RdmaHandle>();
  bool any = false;
  if (!direct.empty()) any = PostPlans(direct, handle.get());

  for (size_t i : bounced) {
    TransferFailure failure;
    if (RunBouncePlanInline(plans[i], &failure)) {
      any = true;
      continue;
    }
    MORI_UMBP_ERROR("[MoriIoEngine] staged transfer failed: code={} msg='{}' peer_engine='{}'",
                    failure.code, failure.message, failure.endpoint);
    handle->AddPresettled(std::move(failure));
    any = true;
  }

  if (!any) return nullptr;
  return handle;
}

}  // namespace mori::umbp
