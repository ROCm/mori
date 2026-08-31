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
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/metrics/component_metrics.h"
#include "umbp/distributed/peer/backend/medium_backend.h"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  InstrumentedBackend — generic storage metrics for EVERY medium, for free
//
//  A MediumBackend decorator that forwards every call and measures the slot
//  lifecycle on the way through: entries by outcome, bytes, batch count, and
//  time spent inside the medium.  It is the answer to "how do we keep metrics
//  working when a backend is added" — the answer is that adding a backend is
//  not a metrics event at all.  PoolClient::Init wraps whatever the medium
//  switch produced, so a medium written next year emits the same series as
//  DRAM does today, lands in the same dashboard panels, and its author never
//  opens this file.
//
//  Why a decorator rather than counters at the call sites: the backends are
//  driven from three places (PeerServiceServer for remote callers, PoolClient
//  for same-node access, the eviction path), and instrumenting callers means
//  every new caller is a new hole.  There is exactly one interface, so there is
//  exactly one place to stand.
//
//  What it deliberately does NOT do: name a medium, look at a TierType, or
//  branch on the wrapped type.  Tier() and Name() are pass-through, and they
//  become the tier= / backend= labels on everything published.
//
//  Cost: one steady_clock pair and a handful of relaxed atomic adds per BATCH
//  call (not per key).  Against a call that touches a medium — a hipMemcpy, a
//  drive read, a bitmap allocation under a mutex — that is not measurable.
// ---------------------------------------------------------------------------
class InstrumentedBackend final : public MediumBackend {
 public:
  explicit InstrumentedBackend(std::unique_ptr<MediumBackend> inner);

  // The wrapped backend, for the few call sites that legitimately need the
  // concrete object (tests asserting medium internals).  Never used to bypass
  // the decorator on a data path.
  MediumBackend* Inner() const { return inner_.get(); }

  // ---- identity: pass-through, and the source of the metric labels ----
  TierType Tier() const override { return inner_->Tier(); }
  const char* Name() const override { return inner_->Name(); }

  // ---- ownership ----
  bool Init(MemoryRegistrar* registrar) override;
  void Shutdown() override { inner_->Shutdown(); }

  // ---- control plane ----
  TierCapacity Capacity() const override { return inner_->Capacity(); }
  uint64_t OwnedKeyCount() const override { return inner_->OwnedKeyCount(); }
  bool Contains(const std::string& key) const override { return inner_->Contains(key); }
  std::vector<KvEvent> DrainPendingEvents() override { return inner_->DrainPendingEvents(); }
  std::vector<KvEvent> SnapshotOwnedKeys() const override { return inner_->SnapshotOwnedKeys(); }
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override {
    return inner_->SnapshotOwnedKeysForFullSync();
  }
  void ClearLocal() override { inner_->ClearLocal(); }
  void ClearFullSyncAcked() override { inner_->ClearFullSyncAcked(); }
  bool IsClearFullSyncPending() const override { return inner_->IsClearFullSyncPending(); }
  void SetAutoFlushHook(size_t threshold, std::function<void()> cb) override {
    inner_->SetAutoFlushHook(threshold, std::move(cb));
  }

  // ---- observability ----
  // The generic series this decorator measured, followed by whatever the
  // wrapped medium publishes about its own internals.  One source, so
  // PoolClient publishes a backend with a single call and cannot forget the
  // medium's half.
  std::vector<MetricSample> SampleMetrics() const override;

  // ---- slot lifecycle: measured, then forwarded ----
  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& reqs) override;
  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& reqs) override;
  std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids) override;
  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                          bool include_descs) override;
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys) override;

  // ---- bootstrap / local endpoints: pass-through ----
  uint64_t PageSize() const override { return inner_->PageSize(); }
  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override {
    return inner_->AllBufferDescs();
  }
  size_t BufferCount() const override { return inner_->BufferCount(); }
  TransferRef BufferRef(uint32_t buffer_index) const override {
    return inner_->BufferRef(buffer_index);
  }

 private:
  // One op's accumulators.  Relaxed atomics: these are read once per metrics
  // tick from another thread and are never correctness state, so ordering
  // against the data plane buys nothing.
  struct OpCounters {
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> nanos{0};
    std::atomic<uint64_t> bytes{0};
  };

  // Per (op, status) entry counts.  A fixed array rather than a map because the
  // set is closed and this is touched per call: kOpCount * kStatusCount is 25
  // atomics, and the indices are the label values.
  enum Op { kAllocate, kCommit, kAbort, kResolve, kEvict, kOpCount };
  enum Status { kOk, kExists, kNoSpace, kMiss, kFailed, kStatusCount };

  void Add(Op op, Status status, uint64_t n) {
    entries_[op][status].fetch_add(n, std::memory_order_relaxed);
  }

  std::unique_ptr<MediumBackend> inner_;
  OpCounters ops_[kOpCount];
  std::atomic<uint64_t> entries_[kOpCount][kStatusCount];
};

// Wrap `inner` so it reports the generic backend metrics.  Null in, null out —
// a failed medium construction stays a failed medium construction.
std::unique_ptr<MediumBackend> MakeInstrumentedBackend(std::unique_ptr<MediumBackend> inner);

}  // namespace mori::umbp
