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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mori/io/common.hpp"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  The transfer layer (design doc §4)
//
//  There is exactly ONE byte-moving path in the system and it is here.  A
//  storage backend owns bytes and publishes endpoints for them; the transfer
//  layer decides how to move bytes between two endpoints.  Admitting a second
//  byte-moving path at the backend level is what produced the coupling this
//  refactor exists to remove.
//
//  mori-io is a MEMORY-to-memory engine (BackendType is {XGMI, RDMA, TCP,
//  FABRIC}, MemoryLocationType is {CPU, GPU}); it has no file or object
//  endpoint.  Rather than teach it one, UMBP abstracts the transfer engine on
//  its own side and makes mori-io one implementation — so mori-io's other
//  consumers see no blast radius.
// ---------------------------------------------------------------------------

// One endpoint of a transfer.
//
// NOT a std::variant, though §4 sketched one.  Registration FANS OUT: the same
// buffer is reachable as a raw process-local pointer AND as an RDMA-registered
// MR at the same time, and which one is used is a property of the (src, dst)
// PAIR, not of the buffer.  A variant would force a buffer to be one or the
// other, and would make the case this phase turns on — memcpy between two
// buffers that also happen to be RDMA-registered — inexpressible.  So a
// TransferRef carries the per-transport handles side by side and each engine
// reads the one it needs.
//
// mori::io::MemoryDesc already works this way, and says so: ipcHandle and
// fabricHandle sit next to each other "so XGMI and Fabric backends can register
// the same memory without clobbering each other's handle."
//
// Empty handles are how an endpoint says "not reachable that way":
//   * host_ptr == nullptr  -> not addressable in this process (a peer's memory)
//   * mem.size == 0        -> not registered with mori-io (a plain user buffer,
//                             or a node with no RDMA configured)
// An endpoint with neither is invalid and no engine will accept it.
//
// DELIBERATELY ABSENT: a file / object endpoint, and MemoryRegistrar::
// RegisterFile.  §4 specified both, but no engine in tree consumes one yet, and
// this plan has already refused to land an abstraction nothing exercises once
// (§3: BackendProperties was proposed and then NOT adopted, because "an
// advertised order would have been scaffolding nothing exercises").  The same
// test applies here: SsdBackend is what needs a FileRef, so SsdBackend brings
// it — adding a `kind` tag and a second handle set is a one-file change to this
// header, and CanHandle already exists to route it.
struct TransferRef {
  // Process-local view.  Set for anything this node allocated or a caller
  // handed us; null for a peer's memory.
  void* host_ptr = nullptr;
  uint64_t size = 0;
  mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU;
  int device = -1;

  // mori-io registration.  Valid iff an IOEngine registered these bytes
  // (local), or they were hydrated from a peer's published descriptor (remote).
  mori::io::MemoryDesc mem{};

  bool HasHostPtr() const { return host_ptr != nullptr && size > 0; }
  bool HasMemoryDesc() const { return mem.size > 0; }
  bool Valid() const { return HasHostPtr() || HasMemoryDesc(); }

  // Unregistered, directly-addressable bytes: a caller's Put src / Get dst that
  // was never passed to RegisterMemory.
  static TransferRef HostBytes(void* p, uint64_t n,
                               mori::io::MemoryLocationType l = mori::io::MemoryLocationType::CPU,
                               int dev = -1) {
    TransferRef r;
    r.host_ptr = p;
    r.size = n;
    r.loc = l;
    r.device = dev;
    return r;
  }

  // A peer's memory: reachable only through mori-io.
  static TransferRef Remote(mori::io::MemoryDesc desc) {
    TransferRef r;
    r.size = desc.size;
    r.loc = desc.loc;
    r.device = desc.deviceId;
    r.mem = std::move(desc);
    return r;
  }
};

// ---------------------------------------------------------------------------
//  MemoryRegistrar — the ONLY view a storage backend gets
//
//  Phase 5 Rule C ("no backend may call Plan/Submit/Wait") was going to be a
//  lint rule because §4 folded registration into TransferEngine, handing
//  backends an object they could transfer with.  That conflated ONE CLASS with
//  ONE INTERFACE.  Keeping one object and giving it two views turns the rule
//  into a compile error at zero runtime cost: MediumBackend::Init takes a
//  MemoryRegistrar*, which has no Submit to call.
// ---------------------------------------------------------------------------
class MemoryRegistrar {
 public:
  virtual ~MemoryRegistrar() = default;

  MemoryRegistrar(const MemoryRegistrar&) = delete;
  MemoryRegistrar& operator=(const MemoryRegistrar&) = delete;

  // Pin [base, base+size) for transfer and return the merged handle set.  The
  // CALLER supplies the facts a descriptor cannot recover — location type,
  // device — because the caller is the allocator (design doc §6, "allocation
  // and registration belong to the backend").
  //
  // Always returns a ref with host_ptr set; whether it also carries an
  // mori-io registration depends on what is configured.  A node with no RDMA
  // still gets a usable local-only endpoint rather than a failure.
  virtual TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                                     int device) = 0;

  // Release every registration held in `ref`.  Tolerates an invalid ref.
  virtual void Deregister(const TransferRef& ref) = 0;

 protected:
  MemoryRegistrar() = default;
};

// ---------------------------------------------------------------------------
//  PeerDirectory — learning how to reach another node
//
//  Some transports cannot address a peer until they have been TOLD about it:
//  mori-io needs the peer's EngineDesc registered and the peer's published
//  MemoryDescs unpacked before a single byte can move.  That handshake is not a
//  transfer, so it does not belong on TransferEngine — but it is also not the
//  client's business HOW it happens, so it must not be reached by naming a
//  concrete engine either.
//
//  So it is its own interface, implemented by the engines that need it.
//  PoolClient obtains one at composition time and drives peer setup through it;
//  a second remote transport implements this and PoolClient does not change.
//  A memcpy engine has no peers and implements nothing.
//
//  Note "directory", not "cache": these calls are the authority on what this
//  process knows about a peer's transport identity, and ForgetRemote is how a
//  failed handshake is undone.
// ---------------------------------------------------------------------------
class PeerDirectory {
 public:
  virtual ~PeerDirectory() = default;

  PeerDirectory(const PeerDirectory&) = delete;
  PeerDirectory& operator=(const PeerDirectory&) = delete;

  // This node's own transport identity, in the opaque form peers exchange.
  // Empty when the transport is not active.
  virtual std::vector<uint8_t> PackedLocalEngineDesc() const = 0;

  // Learn a peer's transport identity from its packed descriptor.  Idempotent.
  // False only when the descriptor cannot be understood.  An empty descriptor
  // means "nothing new" and succeeds iff the peer is already known.
  virtual bool EnsureRemoteEngine(const std::string& node_id, const std::string& packed) = 0;
  virtual bool HasRemoteEngine(const std::string& node_id) const = 0;

  // Drop everything known about a peer, so the next access re-handshakes.
  virtual void ForgetRemote(const std::string& node_id) = 0;

  // Learn the endpoints a peer published, by (backend_id, buffer_index).
  // Idempotent: an address already known is left alone, so a stale descriptor
  // cannot overwrite a live one.
  //
  // The address is the PAIR.  buffer_index is backend-local — every backend on
  // a peer numbers from 0, and each publishes exactly one buffer today — so
  // caching by index alone let one medium's descriptor stand in for another's
  // and served reads out of the wrong memory.  See BufferMemoryDesc in
  // umbp.proto.
  virtual void CacheRemoteBuffers(const std::string& node_id,
                                  const std::vector<BufferMemoryDescBytes>& descs) = 0;
  virtual bool HasRemoteBuffers(const std::string& node_id) const = 0;

  // Every known endpoint for ONE of a peer's backends, indexed by that
  // backend's own buffer_index.  Taken ONCE per batch so a transfer-build loop
  // never locks per page; a batch's pages may span backends, so it is taken per
  // distinct backend_id in the batch rather than once per peer.  An index that
  // was never learned comes back invalid and the caller degrades that key to a
  // miss.
  virtual std::vector<TransferRef> RemoteBufferSnapshot(const std::string& node_id,
                                                        uint32_t backend_id) const = 0;

 protected:
  PeerDirectory() = default;
};

// ---------------------------------------------------------------------------
//  Work
// ---------------------------------------------------------------------------

// One contiguous copy request.  `tag` is the caller's own per-entry index; it
// survives grouping so a per-plan failure maps back to the keys that
// contributed to it.
struct TransferItem {
  TransferRef src;
  uint64_t src_offset = 0;
  TransferRef dst;
  uint64_t dst_offset = 0;
  uint64_t size = 0;
  size_t tag = 0;
};

// Which endpoint of a plan is this node's.  A plan is uniform in direction
// because it is grouped by (src endpoint, dst endpoint).
enum class TransferDirection {
  kPush,  // local -> remote  (mori-io BatchWrite)
  kPull,  // remote -> local  (mori-io BatchRead)
  kLocal  // local -> local   (no wire)
};

// One grouped unit of work, produced by Plan() and consumed by Submit().
//
// Collapsing a scatter into one plan per endpoint pair is an RDMA optimization
// — it cuts CQE and post count — which is exactly why Plan() is virtual and
// not a base-class helper: a memcpy engine gains nothing from pair grouping and
// a file engine would rather group by file and merge adjacent offset ranges.
// Submit-all-then-wait IS universal, and lives with the caller's scheduler.
struct TransferPlan {
  TransferRef src;
  TransferRef dst;
  std::vector<size_t> src_offsets;
  std::vector<size_t> dst_offsets;
  std::vector<size_t> sizes;
  // De-duplicated tags of the items that contributed segments here.
  std::vector<size_t> tags;
  TransferDirection dir = TransferDirection::kLocal;

  // Set when this plan's LOCAL endpoint is unregistered and the engine must
  // stage through its own bounce buffer.  The local-side offsets above are then
  // offsets INTO the bounce buffer, and `bounce_copies` says which user bytes
  // to stage in (kPush) or out (kPull) around the wire.  See
  // MoriIoEngine::Submit for why such a plan completes inline.
  struct BounceCopy {
    void* host = nullptr;
    uint64_t bounce_offset = 0;
    uint64_t size = 0;
  };
  bool uses_bounce = false;
  std::vector<BounceCopy> bounce_copies;

  // Set by CompositeTransferEngine so Submit can route a plan back to the
  // engine that produced it.  Null means "whoever I am submitted to".
  class TransferEngine* engine = nullptr;
};

// What Plan() could not express.  `rejected_tags` is not an error path the
// caller can ignore: those items were dropped and their keys must be failed.
struct TransferPlanSet {
  std::vector<TransferPlan> plans;
  std::vector<size_t> rejected_tags;
};

// One failed plan, mapped back to the keys that rode in it.  Carries enough to
// reproduce the pre-refactor per-transfer error log.
struct TransferFailure {
  std::vector<size_t> tags;
  uint32_t code = 0;
  std::string message;
  std::string endpoint;  // remote engine key, for diagnosis
};

// Posted, not-yet-settled work.  OWNS everything Submit referenced — mori-io's
// RDMA backend keeps a raw TransferStatus* into the handle's status vector, so
// the handle must outlive the post and must never move its statuses.
//
// The destructor is a safety net, not the normal path: on an early or
// exceptional destroy it drains, so a completion callback never writes freed
// memory.  It performs no failure mapping — that is Wait's job.
class TransferHandle {
 public:
  virtual ~TransferHandle() = default;

  TransferHandle(const TransferHandle&) = delete;
  TransferHandle& operator=(const TransferHandle&) = delete;

  // Block until every posted plan settles, appending one TransferFailure per
  // plan that did not succeed.  Never breaks early — every plan is waited, so
  // no status is left live.  Idempotent: a second call appends nothing.
  virtual void Wait(std::vector<TransferFailure>* failures) = 0;

 protected:
  TransferHandle() = default;
};

// ---------------------------------------------------------------------------
//  TransferEngine
// ---------------------------------------------------------------------------
class TransferEngine : public MemoryRegistrar {
 public:
  virtual const char* Name() const = 0;

  // Can this engine move bytes from `src` to `dst`?  Engine selection is a
  // function of the PAIR, never of either endpoint alone.
  virtual bool CanHandle(const TransferRef& src, const TransferRef& dst) const = 0;

  // Group `items` into the units this engine wants to submit.  Pure: no IO is
  // issued and no engine state changes, so a caller may plan on one thread and
  // submit on another.
  virtual TransferPlanSet Plan(const std::vector<TransferItem>& items) const = 0;

  // Post `plans`.  Returns a handle to wait on, or nullptr if nothing was
  // posted (in which case every tag in `plans` must be treated as failed).
  //
  // Submit does NOT block on the wire, with one documented exception: a plan
  // that needs the engine's bounce buffer completes INSIDE Submit and comes
  // back already settled.  That keeps the bounce pool's lock from ever being
  // held across a return, which is what makes a submit-all-then-wait loop over
  // several peers deadlock-free without the caller knowing staging exists.
  virtual std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) = 0;

  // Plan + Submit + Wait, for callers with nothing to overlap.  Returns false
  // and fills `failed_tags` if anything was rejected or failed.
  bool Transfer(const std::vector<TransferItem>& items, std::vector<size_t>* failed_tags);
};

}  // namespace mori::umbp
