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

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <msgpack.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {

namespace {

std::string IncarnationKey(const std::string& node_id) {
  static std::atomic<uint64_t> sequence{0};
  std::random_device random;
  std::ostringstream key;
  key << node_id << "/" << getpid() << "/" << std::hex
      << std::chrono::steady_clock::now().time_since_epoch().count() << "/" << random() << "/"
      << sequence.fetch_add(1, std::memory_order_relaxed);
  return key.str();
}

// Same two endpoints?  Ids are only unique per IOEngine, so compare engine keys too.
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

  bool WaitOnStream(void* stream) override {
    bool ready = presettled_.empty();
    for (auto& set : sets_) {
      for (auto& status : set->statuses) {
        if (status.WaitOnStream(stream) != mori::io::StatusCode::SUCCESS) ready = false;
      }
    }
    return ready;
  }

  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    // Finish every posted group before failure reporting can allocate. Marking
    // drained_ or reporting an early failure first would let bad_alloc skip
    // later statuses and make destruction free a still-live backend callback.
    for (auto& set : sets_)
      for (auto& status : set->statuses) status.Wait();
    drained_ = true;
    if (failures != nullptr) {
      failures->insert(failures->end(), std::make_move_iterator(presettled_.begin()),
                       std::make_move_iterator(presettled_.end()));
    }
    presettled_.clear();
    // All groups are terminal; reporting exceptions cannot strand GPU/RDMA work.
    for (auto& set : sets_) {
      for (size_t g = 0; g < set->statuses.size(); ++g) {
        if (set->statuses[g].Succeeded()) continue;
        if (failures == nullptr) continue;
        TransferFailure f;
        f.tags = set->tags[g];
        f.code = set->statuses[g].CodeUint32();
        f.message = set->statuses[g].Message();
        f.endpoint = set->endpoints[g];
        f.resources_quarantined = set->statuses[g].ResourcesQuarantined();
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
                           uint64_t bounce_bytes, UMBPIoEngineConfig transport_config)
    : local_engine_key_(std::move(engine_key)),
      io_config_(std::move(io_config)),
      transport_config_(std::move(transport_config)),
      bounce_size_(transport_config_.mode == UMBPTransferMode::kFabricStrict ? 0 : bounce_bytes) {}

MoriIoEngine::~MoriIoEngine() { Shutdown(); }

bool MoriIoEngine::Init() {
  if (io_engine_ != nullptr) return true;
  const bool strict = transport_config_.mode == UMBPTransferMode::kFabricStrict;
  const bool fabric = transport_config_.mode != UMBPTransferMode::kRdma;
  if (io_config_.host.empty() && !fabric) return false;
  if (strict && transport_config_.fabric_kernel_path.empty()) {
    MORI_UMBP_ERROR("[MoriIoEngine] fabric_strict requires fabric_kernel_path");
    return false;
  }
  // FABRIC has no RDMA control listener; the peer gRPC endpoint carries metadata.
  if (io_config_.host.empty()) io_config_.host = "127.0.0.1";
  try {
    local_engine_key_ = IncarnationKey(local_engine_key_);
    io_engine_ = std::make_unique<mori::io::IOEngine>(local_engine_key_, io_config_);
    local_engine_key_ = io_engine_->GetEngineDesc().key;
    enabled_backend_mask_ = 0;
    if (fabric) {
      try {
        mori::io::FabricBackendConfig config;
        config.maxBlocksPerSegment = transport_config_.fabric_max_blocks_per_segment;
        config.cacheBatchDescriptors = transport_config_.fabric_cache_descriptors;
        io_engine_->CreateBackend(mori::io::BackendType::FABRIC, config);
        enabled_backend_mask_ |= mori::io::BackendBit(mori::io::BackendType::FABRIC);
        if (!transport_config_.fabric_kernel_path.empty()) {
          io_engine_->LoadFabricCopyModule(transport_config_.fabric_kernel_path);
        }
      } catch (const std::exception& e) {
        if (strict) throw;
        MORI_UMBP_WARN("[MoriIoEngine] auto: FABRIC unavailable; retaining RDMA path: {}",
                       e.what());
      }
      int devices = 0;
      (void)hipGetDeviceCount(&devices);
      bool ready = false;
      for (int device = 0; device < devices; ++device) ready |= io_engine_->IsFabricReady(device);
      if (strict && !ready) {
        MORI_UMBP_ERROR(
            "[MoriIoEngine] fabric_strict: no GPU has a ready fabric copy module/domain");
        Shutdown();
        return false;
      }
      if (!ready) {
        MORI_UMBP_WARN(
            "[MoriIoEngine] FABRIC not copy-ready; auto requests require another backend");
      }
    }
    if (!strict) {
      mori::io::RdmaBackendConfig rdma_cfg;
      rdma_cfg.qpPerTransfer = 4;
      rdma_cfg.enableTransferChunking = true;
      rdma_cfg.numNicsPerTransfer = 4;
      rdma_cfg.enableNotification = false;
      try {
        io_engine_->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);
        enabled_backend_mask_ |= mori::io::BackendBit(mori::io::BackendType::RDMA) |
                                 mori::io::BackendBit(mori::io::BackendType::XGMI);
      } catch (const std::exception& e) {
        if (!fabric) throw;
        MORI_UMBP_WARN("[MoriIoEngine] auto: RDMA unavailable; retaining FABRIC: {}", e.what());
      }
    }
    io_engine_->SetTransferBackendMask(strict ? mori::io::BackendBit(mori::io::BackendType::FABRIC)
                                              : enabled_backend_mask_);
    if (strict || !(enabled_backend_mask_ & mori::io::BackendBit(mori::io::BackendType::RDMA))) {
      bounce_size_ = 0;
    }
    if (bounce_size_ > 0) {
      bounce_buffer_ = std::make_unique<char[]>(bounce_size_);
      std::memset(bounce_buffer_.get(), 0, bounce_size_);
      try {
        bounce_ref_ = RegisterMemory(bounce_buffer_.get(), bounce_size_,
                                     mori::io::MemoryLocationType::CPU, /*device=*/-1);
      } catch (const std::exception& e) {
        if (!fabric) throw;
        MORI_UMBP_WARN("[MoriIoEngine] auto: host bounce registration unavailable: {}", e.what());
      }
      if (!bounce_ref_.HasMemoryDesc()) {
        bounce_buffer_.reset();
        bounce_ref_ = TransferRef{};
        bounce_size_ = 0;
      }
    }
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[MoriIoEngine] init failed: {}", e.what());
    Shutdown();
    return false;
  }
  MORI_UMBP_INFO("[MoriIoEngine] initialized key={} mode={} bounce_bytes={}", local_engine_key_,
                 static_cast<int>(transport_config_.mode), bounce_size_);
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
  return RegisterMemoryWithBackends(base, size, loc, device, mori::io::kAllBackendMask);
}

TransferRef MoriIoEngine::RegisterMemoryWithBackends(void* base, size_t size,
                                                     mori::io::MemoryLocationType loc, int device,
                                                     uint32_t allowed_backend_mask) {
  TransferRef ref = TransferRef::HostBytes(base, size, loc, device);
  if (io_engine_ == nullptr || base == nullptr || size == 0) return ref;
  const bool strict = transport_config_.mode == UMBPTransferMode::kFabricStrict;
  uint32_t mask = allowed_backend_mask & enabled_backend_mask_;
  if (strict && loc != mori::io::MemoryLocationType::GPU) return ref;
  if (strict && !io_engine_->IsFabricReady(device)) {
    throw std::runtime_error("fabric_strict registration requires a copy-ready GPU");
  }
  if (mask == 0) return ref;
  try {
    ref.mem = io_engine_->RegisterMemoryWithBackends(base, size, device, loc, mask);
  } catch (const std::exception& e) {
    const uint32_t fabric_mask = mori::io::BackendBit(mori::io::BackendType::FABRIC);
    if (transport_config_.mode != UMBPTransferMode::kAuto ||
        loc != mori::io::MemoryLocationType::GPU || !(mask & fabric_mask) || mask == fabric_mask) {
      throw;
    }
    MORI_UMBP_WARN("[MoriIoEngine] auto: registration fell back to FABRIC-only: {}", e.what());
    ref.mem = io_engine_->RegisterMemoryWithBackends(base, size, device, loc, fabric_mask);
  }
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

  if (desc.key.empty() || desc.key == local_engine_key_) {
    MORI_UMBP_ERROR("[MoriIoEngine] refusing invalid remote identity from '{}'", node_id);
    return false;
  }
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto& remote = remotes_[node_id];
  if (remote.engine_registered && remote.engine_desc == desc) return true;
  if (!remote.engine_desc.key.empty() && !(remote.engine_desc == desc)) {
    if (remote.engine_registered && io_engine_) {
      io_engine_->DeregisterRemoteEngine(remote.engine_desc);
    }
    remote = RemoteNode{};
  }
  try {
    if (io_engine_) io_engine_->RegisterRemoteEngine(desc);
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[MoriIoEngine] remote registration failed for '{}': {}", node_id, e.what());
    return false;
  }
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
  auto it = remotes_.find(node_id);
  if (it == remotes_.end()) return;
  if (io_engine_ && it->second.engine_registered) {
    io_engine_->DeregisterRemoteEngine(it->second.engine_desc);
  }
  remotes_.erase(it);
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

    if (!remote.engine_desc.key.empty() && mem.engineKey != remote.engine_desc.key) {
      MORI_UMBP_ERROR("[MoriIoEngine] discarding stale region for '{}' engine={}", node_id,
                      mem.engineKey);
      continue;
    }
    if (shelf[d.buffer_index].HasMemoryDesc()) {
      const auto& held = shelf[d.buffer_index].mem;
      if (held == mem) continue;
      if (io_engine_) io_engine_->InvalidateRemoteMemory(held);
      MORI_UMBP_INFO("[MoriIoEngine] invalidated changed region '{}' backend={} buffer={}", node_id,
                     d.backend_id, d.buffer_index);
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
    for (const auto& ref : shelf) {
      if (ref.HasMemoryDesc()) return true;
    }
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
  if (local.HasMemoryDesc()) {
    const TransferRef& remote = src_remote ? src : dst;
    return io_engine_->GetSelectedBackend(local.mem, remote.mem).has_value();
  }
  return IsBounceable(local) && bounce_size_ > 0;  // staged through the pool
}

namespace {

// Memory ids are per IOEngine, so two peers can share one; SameEndpoints tells
// a bucket's plans apart.
struct PairKey {
  mori::io::MemoryUniqueId src;
  mori::io::MemoryUniqueId dst;
  bool operator==(const PairKey& o) const noexcept { return src == o.src && dst == o.dst; }
};
struct PairKeyHash {
  size_t operator()(const PairKey& k) const noexcept {
    return HashCombine(std::hash<mori::io::MemoryUniqueId>{}(k.src),
                       std::hash<mori::io::MemoryUniqueId>{}(k.dst));
  }
};

// Zero copy: one plan per registered buffer pair, segments merged up to
// ibv_sge.length.  Staged: unregistered local memory takes bounce-pool offsets;
// a full pool starts a new plan, and an item larger than the pool is rejected.
class MoriIoPlanBuilder final : public TransferPlanBuilder {
 public:
  MoriIoPlanBuilder(const std::string& local_key, uint64_t bounce_size,
                    const TransferRef& bounce_ref, bool fabric_strict = false)
      : local_key_(local_key),
        bounce_size_(bounce_size),
        bounce_ref_(bounce_ref),
        fabric_strict_(fabric_strict) {}

  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    if (fabric_strict_ && (item.src.loc != mori::io::MemoryLocationType::GPU ||
                           item.dst.loc != mori::io::MemoryLocationType::GPU)) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    const bool src_remote = IsRemoteMemory(item.src, local_key_);
    const bool dst_remote = IsRemoteMemory(item.dst, local_key_);
    if (src_remote == dst_remote) {  // both local or both remote
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    const TransferDirection dir = dst_remote ? TransferDirection::kPush : TransferDirection::kPull;
    const TransferRef& local = src_remote ? item.dst : item.src;
    const TransferRef& remote = src_remote ? item.src : item.dst;
    const uint64_t local_off = src_remote ? item.dst_offset : item.src_offset;
    const uint64_t remote_off = src_remote ? item.src_offset : item.dst_offset;

    if (local.HasMemoryDesc()) {
      auto& candidates = pair_to_plans_[PairKey{item.src.mem.id, item.dst.mem.id}];
      size_t pi = out_.plans.size();
      for (size_t cand : candidates) {
        if (SameEndpoints(out_.plans[cand], item.src, item.dst)) {
          pi = cand;
          break;
        }
      }
      if (pi == out_.plans.size()) {
        candidates.push_back(pi);
        out_.plans.push_back(PlanBetween(item, dir));
        bounce_cursor_.push_back(0);
      }
      AppendSegment(&out_.plans[pi], item.src_offset, item.dst_offset, item.size, item.tag, true,
                    kMaxSgeBytes);
      return;
    }

    // Unregistered local memory: stage through the bounce pool.
    if (bounce_size_ == 0 || !IsBounceable(local) || item.size > bounce_size_) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    auto& candidates =
        bounce_to_plans_[PairKey{remote.mem.id, static_cast<mori::io::MemoryUniqueId>(dir)}];
    size_t pi = out_.plans.size();
    bool fresh = true;
    for (size_t& cand : candidates) {
      const TransferPlan& p = out_.plans[cand];
      const TransferRef& p_remote = (dir == TransferDirection::kPush) ? p.dst : p.src;
      if (p.dir != dir || p_remote.mem.engineKey != remote.mem.engineKey) continue;
      // Reuse while the pool has room, else retire it for a new plan.
      if (bounce_cursor_[cand] + item.size <= bounce_size_) {
        pi = cand;
        fresh = false;
      } else {
        cand = pi;  // now names the new plan
      }
      break;
    }
    if (fresh) {
      if (std::find(candidates.begin(), candidates.end(), pi) == candidates.end()) {
        candidates.push_back(pi);
      }
      TransferPlan plan;
      plan.dir = dir;
      plan.uses_bounce = true;
      // The bounce region stands in for the local side.
      plan.src = dir == TransferDirection::kPush ? bounce_ref_ : remote;
      plan.dst = dir == TransferDirection::kPush ? remote : bounce_ref_;
      out_.plans.push_back(std::move(plan));
      bounce_cursor_.push_back(0);
    }
    TransferPlan& plan = out_.plans[pi];
    const uint64_t bounce_off = bounce_cursor_[pi];
    bounce_cursor_[pi] += item.size;
    if (dir == TransferDirection::kPush) {
      AppendSegment(&plan, bounce_off, remote_off, item.size, item.tag, true, kMaxSgeBytes);
    } else {
      AppendSegment(&plan, remote_off, bounce_off, item.size, item.tag, true, kMaxSgeBytes);
    }
    plan.bounce_copies.push_back(TransferPlan::BounceCopy{
        static_cast<char*>(local.host_ptr) + local_off, bounce_off, item.size});
  }

  TransferPlanSet Finish() override { return std::move(out_); }

 private:
  static constexpr size_t kMaxSgeBytes = std::numeric_limits<uint32_t>::max();

  const std::string& local_key_;
  const uint64_t bounce_size_;
  const TransferRef& bounce_ref_;
  const bool fabric_strict_;
  TransferPlanSet out_;
  std::unordered_map<PairKey, std::vector<size_t>, PairKeyHash> pair_to_plans_;
  std::unordered_map<PairKey, std::vector<size_t>, PairKeyHash> bounce_to_plans_;
  std::vector<uint64_t> bounce_cursor_;  // parallel to out_.plans
};

}  // namespace

std::unique_ptr<TransferPlanBuilder> MoriIoEngine::CreatePlanBuilder(
    size_t /*expected_items*/) const {
  return std::make_unique<MoriIoPlanBuilder>(
      local_engine_key_, bounce_size_, bounce_ref_,
      transport_config_.mode == UMBPTransferMode::kFabricStrict);
}

std::unique_ptr<TransferPlanBuilder> MakeRdmaPlanBuilder(const std::string& local_key,
                                                         uint64_t bounce_size,
                                                         const TransferRef& bounce_ref) {
  return std::make_unique<MoriIoPlanBuilder>(local_key, bounce_size, bounce_ref);
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
    const bool pulling = (plan.dir == TransferDirection::kPull);
    const auto& local = pulling ? plan.dst.mem : plan.src.mem;
    const auto& remote = pulling ? plan.src.mem : plan.dst.mem;
    const auto backend = io_engine_->GetSelectedBackend(local, remote);
    if (!backend) {
      TransferFailure failure;
      failure.tags = plan.tags;
      failure.code = static_cast<uint32_t>(mori::io::StatusCode::ERR_NOT_FOUND);
      failure.message = "no permitted backend can reach this endpoint pair";
      failure.endpoint = remote.engineKey;
      handle->AddPresettled(std::move(failure));
      continue;
    }
    const auto backend_index = static_cast<size_t>(*backend);
    if (backend_index < kBackendCount) {
      const size_t direction_index = pulling ? 0 : 1;
      submitted_plans_[backend_index][direction_index].fetch_add(1, std::memory_order_relaxed);
      const uint64_t bytes = std::accumulate(plan.sizes.begin(), plan.sizes.end(), uint64_t{0});
      submitted_bytes_[backend_index][direction_index].fetch_add(bytes, std::memory_order_relaxed);
    }
    RdmaHandle::PostSet* set = set_for(plan.dir);
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

std::vector<MetricSample> MoriIoEngine::SampleMetrics() const {
  static constexpr const char* names[] = {"unknown", "xgmi", "rdma", "tcp", "fabric", "ofi"};
  std::vector<MetricSample> samples;
  for (size_t backend = 0; backend < kBackendCount; ++backend) {
    for (size_t direction = 0; direction < 2; ++direction) {
      MetricLabels labels{{"backend", names[backend]},
                          {"direction", direction == 0 ? "pull" : "push"}};
      samples.push_back(
          MetricSample{"mori_umbp_transport_submitted_bytes_total",
                       "Bytes submitted by the selected MORI-IO backend", labels,
                       submitted_bytes_[backend][direction].load(std::memory_order_relaxed)});
      samples.push_back(
          MetricSample{"mori_umbp_transport_submitted_plans_total",
                       "Plans submitted by the selected MORI-IO backend", std::move(labels),
                       submitted_plans_[backend][direction].load(std::memory_order_relaxed)});
    }
  }
  return samples;
}

}  // namespace mori::umbp
