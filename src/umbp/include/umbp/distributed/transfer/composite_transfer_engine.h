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
#include <cstddef>
#include <memory>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// Holds the concrete engines and IS a TransferEngine, so the rest of the system
// holds exactly one pointer to exactly one type: PoolClient owns it, and
// backends receive the same object narrowed to MemoryRegistrar at Init.
//
// Registration FANS OUT — every engine that can reach a buffer gets to publish
// its handle, and the handles are merged into one TransferRef (see the comment
// on TransferRef for why that is a struct and not a variant).  Transfer
// DISPATCHES — a (src, dst) pair goes to exactly one engine.  That asymmetry is
// the whole reason this class exists; §4 argued it does not need a second class,
// only a second method shape.
//
// Selection is first-match in registration order, so order encodes preference:
// register LocalCopyEngine before MoriIoEngine and a both-endpoints-local pair
// never reaches the wire.  In practice the two are disjoint (mori-io requires
// exactly one remote endpoint), so the order is documentation rather than a
// tie-break — but a future engine pair may genuinely overlap.
//
// NOT implemented: chaining, where no single engine spans the pair and the
// layer composes two hops through a bounce buffer (§4's "3FS file on node A,
// reader on node B").  It cannot arise while every endpoint is memory; it
// arrives with the first non-memory endpoint, which is SsdBackend's FileRef.
// It is also the transfer layer's OBSERVABILITY point, for the same reason it
// is the dispatch point: it is the one place that knows which engine carried
// which plan.  Bytes, plan counts, failures and in-flight time are measured
// here and published under engine= / direction= labels, so an engine added
// with AddEngine() shows up in the existing dashboard panels without writing
// or wiring a single metric (see component_metrics.h).
class CompositeTransferEngine final : public TransferEngine {
 public:
  // Order is preference order.  Null is ignored.
  void AddEngine(std::unique_ptr<TransferEngine> engine);

  const char* Name() const override { return "CompositeTransferEngine"; }

  // Fan out: every sub-engine registers, and the per-transport handles are
  // merged.  Returns an invalid ref only when no engine claimed the memory.
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override;
  // Fan out to the first file-capable sub-engine (GdsEngine): it owns the fd
  // registration outright and returns a File ref carrying its handle.  Invalid
  // when no file engine is configured.
  TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) override;
  void Deregister(const TransferRef& ref) override;

  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // Partition items by the engine that will carry them, delegate, and tag each
  // returned plan with its engine so Submit can route it back.
  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override;

  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

  // Which engine would carry this pair, or null.  Exposed for tests and for
  // callers that want to know whether a transfer is servicable before building
  // a whole item list.
  TransferEngine* SelectEngine(const TransferRef& src, const TransferRef& dst) const;

  // The generic per-engine series measured above, followed by whatever each
  // sub-engine publishes about its own internals (engine= already stamped on
  // both).  One source for the whole transfer layer, so PoolClient publishes it
  // with the same call it uses for a backend.
  std::vector<MetricSample> SampleMetrics() const override;

 private:
  class FanOutHandle;

  // Per (engine, direction) accumulators.  Indexed by the engine's position in
  // engines_, which is fixed after composition, so the hot path is an array
  // index and never a lookup.  Relaxed atomics: read once per metrics tick,
  // never correctness state.
  struct DirectionCounters {
    std::atomic<uint64_t> plans{0};
    std::atomic<uint64_t> failed_plans{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> nanos{0};
  };
  // kPush / kPull / kLocal, indexed by TransferDirection.
  static constexpr size_t kDirectionCount = 3;
  struct EngineCounters {
    DirectionCounters by_direction[kDirectionCount];
  };

  static size_t DirectionIndex(TransferDirection dir) { return static_cast<size_t>(dir); }
  // Position of `engine` in engines_, or engines_.size() when it is not ours.
  size_t IndexOf(const TransferEngine* engine) const;
  // Charge a settled plan back to the engine that carried it.  Called from
  // FanOutHandle::Wait, which is why the counters outlive any single submit.
  void RecordSettled(size_t engine_index, TransferDirection dir, uint64_t bytes, uint64_t nanos,
                     uint64_t failed_plans);

  std::vector<std::unique_ptr<TransferEngine>> engines_;
  // One per engine, in engines_ order.  Held by unique_ptr because the atomics
  // inside are neither copyable nor movable and engines_ grows during
  // composition.
  std::vector<std::unique_ptr<EngineCounters>> counters_;
  // Items Plan() could not route to any engine.  Not charged to an engine —
  // there was none — so it rides engine="none", status="rejected".
  mutable std::atomic<uint64_t> rejected_items_{0};
};

}  // namespace mori::umbp
