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
#include <mutex>
#include <vector>

#include "umbp/common/host_registration.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// The one-endpoint-is-GPU implementation of TransferEngine.
//
// WHY THIS EXISTS.  Before this engine, a local pair with a GPU endpoint was
// servable by NO engine in tree, so an HBM backend's local Put/Get could not
// complete at all:
//
//   LocalCopyEngine::CanHandle  requires BOTH endpoints loc == CPU — a GPU
//                               endpoint is local but not memcpy-able, and the
//                               header says so ("falls to an engine that can
//                               reach it").  This is that engine.
//   MoriIoEngine::CanHandle     requires EXACTLY ONE remote endpoint
//                               (`src_remote == dst_remote` -> false), so a
//                               both-local pair is refused however it is typed.
//
// The REMOTE side of HBM already worked before this engine and still does not
// route here: a peer's HBM is reached through mori-io, whose MemoryDesc carries
// loc=GPU on the wire already (design doc §2, "this is why the remote path
// already works against an HBM peer unmodified, and why the local fast path
// does not").  This engine closes exactly the local half of that sentence.
//
// SELECTION IS DISJOINT, not preference-ordered.  The three engines partition
// the (src, dst) space with no overlap, so CompositeTransferEngine's
// first-match order is documentation here rather than a tie-break:
//
//   both local, both CPU        -> LocalCopyEngine
//   both local, either GPU      -> HbmCopyEngine      (this)
//   exactly one remote          -> MoriIoEngine
//
// That covers H2D (a caller's host Put src into an HBM slot), D2H (an HBM slot
// out to a caller's host Get dst), and D2D (HBM-to-HBM on one node, e.g. a
// mirror between two GPUs).
//
// NOTE ON `host_ptr`.  TransferRef::host_ptr is documented as the "process-local
// view", not "host memory": for an hipMalloc'd buffer the DEVICE pointer is that
// view, since it is what hipMemcpy accepts and what is valid in this process.
// So HbmBackend registers its buffers with host_ptr = device pointer and
// loc = GPU, and HasHostPtr() reads as "addressable here" for both media.  No
// new TransferRef field or `kind` tag was needed — which is the reverse of the
// SSD/FileRef case the base header reserves.
class HbmCopyEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "HbmCopyEngine"; }

  // Nothing to pin.  A device pointer is already a usable endpoint for
  // hipMemcpy, exactly as a host pointer is for memcpy — the registration that
  // DOES cost something (an RDMA MR) is MoriIoEngine's, and the composite fans
  // registration out to it in the same call.
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return TransferRef::HostBytes(base, size, loc, device);
  }
  void Deregister(const TransferRef&) override {}

  // Declare a host region as kernel-addressable, enabling the gather fast path
  // for plans whose host side lies inside it.
  //
  // A gather kernel cannot dereference plain mmap memory — it faults the GPU —
  // so the region must be hipHostRegister'd first, which is what
  // HostTierRegistration does (asynchronously for large pools).  Registration
  // is best-effort and this is purely an optimisation hint: until it completes,
  // and forever if it fails, Submit simply takes the hipMemcpy path it took
  // before.
  //
  // PoolClient calls this once per buffer each medium publishes, after Init.
  // It belongs on the engine rather than at the call sites because the engine
  // is the only layer that sees the segment shape a copy actually decomposes
  // into, which is what decides whether a kernel beats the copy engine.
  void AddHostGatherRegion(void* base, size_t bytes);
  void ClearHostGatherRegions();

  // Both endpoints addressable in this process, at least one of them GPU.
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // Group by (src base, dst base) and coalesce exactly-contiguous segments —
  // same shape as LocalCopyEngine::Plan, and it pays MORE here.  Each hipMemcpy
  // carries a fixed launch/synchronization cost that a memcpy does not, so
  // collapsing a run of adjacent pages into one call removes real overhead
  // rather than just loop iterations.  This is the concrete instance of the
  // reason Plan() is virtual rather than a base-class helper.
  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override;

  // Performs the copies inline; the returned handle is already settled.
  //
  // Synchronous hipMemcpy, deliberately, matching LocalCopyEngine's contract
  // that a local transfer costs no round trip.  An async/stream variant would
  // overlap the segments within one plan, but the parallelism that actually
  // matters here is ACROSS keys and already exists above this engine, in
  // PoolClient's UMBP_DRAM_{READ,WRITE}_THREADS executors.  A shared stream
  // would also need its own synchronization, since Submit is called from those
  // worker threads concurrently — cost with no matching win at KV-block sizes,
  // where a multi-MiB copy dwarfs the per-call launch overhead.
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

 private:
  // Why a plan did not take the gather kernel.  The hot path never computes
  // this: it is filled in only when debug mode is on (UMBP_HBM_COPY_DEBUG), so
  // that the log can say which path carried a batch AND why the other one
  // declined — the two questions are always asked together when a bandwidth
  // number looks wrong.
  enum class GatherSkip : uint8_t {
    kTaken = 0,
    kUnrecorded,                  // debug mode off, or a plan the pass never classified
    kDisabled,                    // UMBP_DRAM_GATHER_KERNEL=0, or latched off by a failure
    kKindNotHostDevice,           // D2D: no host side to register, so no kernel form
    kNoDevice,                    // neither endpoint named a device
    kHostNotRegistered,           // host side outside every HostTierRegistration
    kNoFragments,                 // plan contributed no segments
    kTooFewFragments,             // < 2 in the bucket; one segment is hipMemcpy's best case
    kFragmentAtOrAboveThreshold,  // mean segment >= kGatherFragmentThreshold
    kSetDeviceFailed,
    kNoStream,
    kLaunchFailed,
    kSyncFailed,
  };
  static const char* GatherSkipName(GatherSkip reason);

  // Run the gather kernel over the batch's eligible segments, bucketed by
  // device.  Returns a per-plan flag: 1 means fully copied by the kernel, 0
  // means "not taken" and never "failed" — Submit then runs its hipMemcpy loop
  // for it, which reaches the same result whether or not fragments landed.
  // May leave the current HIP device changed.
  //
  // `skip_reasons`, when non-null, is resized to plans.size() and filled with
  // the per-plan verdict.  Submit passes it only in debug mode.
  std::vector<char> GatherEligiblePlans(const std::vector<TransferPlan>& plans,
                                        std::vector<GatherSkip>* skip_reasons);

  // True when [ptr, ptr + size) is inside a completed registration, and so safe
  // to hand to a kernel.  False is always the safe answer.
  bool HostRegionCovers(const void* ptr, size_t size) const;

  mutable std::mutex host_regions_mutex_;
  std::vector<std::unique_ptr<HostTierRegistration>> host_regions_;
};

}  // namespace mori::umbp
