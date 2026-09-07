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

// ---------------------------------------------------------------------------
//  The metrics seam: MetricPublisher, InstrumentedBackend, and the transfer
//  layer's per-engine accounting.
//
//  The load-bearing test in this file is InstrumentedBackendMetrics.
//  MetricsAreGenericOverBackends: MockBackend contains NOT ONE LINE of metrics
//  code, and the assertions still hold.  That is the property the refactor is
//  supposed to guarantee — a medium added later reports the same series and
//  lands in the same dashboard panels — and it is the property that silently
//  rots if someone reintroduces per-medium metric names or moves counting back
//  into the call sites.  Keep this test backend-agnostic: if it ever needs to
//  know which medium it wrapped, the property is gone.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/metrics/component_metrics.h"
#include "umbp/distributed/peer/backend/instrumented_backend.h"
#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {

// A metric identity as a dashboard sees it: name plus its sorted label set.
std::string Identity(const char* name, const MetricLabels& labels) {
  std::map<std::string, std::string> sorted(labels.begin(), labels.end());
  std::string out = name;
  for (const auto& [k, v] : sorted) {
    out += ',';
    out += k;
    out += '=';
    out += v;
  }
  return out;
}

// Collects what a publisher emitted, so a test can assert on the wire form
// rather than on a component's internals.
struct Collector {
  std::map<std::string, double> counters;  // identity -> summed delta
  std::map<std::string, double> gauges;    // identity -> last value

  MetricPublisher::Sink Sink() {
    return MetricPublisher::Sink{
        [this](const char* name, const char*, const MetricLabels& labels, double delta) {
          counters[Identity(name, labels)] += delta;
        },
        [this](const char* name, const char*, const MetricLabels& labels, double value) {
          gauges[Identity(name, labels)] = value;
        }};
  }

  bool Has(const std::string& identity) const { return counters.count(identity) > 0; }
  double Counter(const std::string& identity) const {
    auto it = counters.find(identity);
    return it == counters.end() ? -1.0 : it->second;
  }
};

// A MetricSource whose samples a test controls outright.
class FakeSource : public MetricSource {
 public:
  std::vector<MetricSample> samples;
  std::vector<MetricSample> SampleMetrics() const override { return samples; }
};

constexpr const char* kName = "test_metric_total";
constexpr const char* kHelp = "help";

// ---------------------------------------------------------------------------
//  MetricPublisher
// ---------------------------------------------------------------------------

TEST(MetricPublisher, ShipsDeltasNotAbsoluteValues) {
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {}, 10}};

  MetricPublisher pub;
  Collector c;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.Counter(kName), 10.0);

  // Same absolute counter, one tick later: nothing new happened.
  c.counters.clear();
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_FALSE(c.Has(kName)) << "an unchanged counter must not re-ship its total";

  src.samples[0].value = 25;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.Counter(kName), 15.0);
}

TEST(MetricPublisher, RebasesInsteadOfShippingNegativeDeltas) {
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {}, 100}};
  MetricPublisher pub;
  Collector c;
  pub.Publish("src", {}, src, c.Sink());

  // A component that was torn down and rebuilt restarts its counter.  The
  // publisher must absorb that, not emit a negative delta.
  c.counters.clear();
  src.samples[0].value = 5;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_FALSE(c.Has(kName));

  // ...and the next real increment is measured against the NEW baseline.
  src.samples[0].value = 8;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.Counter(kName), 3.0);
}

TEST(MetricPublisher, SourcesDoNotShareABaseline) {
  // Two components reporting the same metric name: if they shared a baseline,
  // one's progress would cancel the other's out.  This is the bug the source id
  // exists to prevent, and it is exactly the shape two backends take.
  FakeSource a;
  FakeSource b;
  a.samples = {MetricSample{kName, kHelp, {}, 10}};
  b.samples = {MetricSample{kName, kHelp, {}, 1000}};

  MetricPublisher pub;
  Collector c;
  pub.Publish("a", {{"tier", "DRAM"}}, a, c.Sink());
  pub.Publish("b", {{"tier", "SSD"}}, b, c.Sink());

  EXPECT_DOUBLE_EQ(c.Counter(Identity(kName, {{"tier", "DRAM"}})), 10.0);
  EXPECT_DOUBLE_EQ(c.Counter(Identity(kName, {{"tier", "SSD"}})), 1000.0);

  c.counters.clear();
  a.samples[0].value = 11;
  pub.Publish("a", {{"tier", "DRAM"}}, a, c.Sink());
  pub.Publish("b", {{"tier", "SSD"}}, b, c.Sink());
  EXPECT_DOUBLE_EQ(c.Counter(Identity(kName, {{"tier", "DRAM"}})), 1.0);
  EXPECT_FALSE(c.Has(Identity(kName, {{"tier", "SSD"}})));
}

TEST(MetricPublisher, GaugesShipEveryTickAndIgnoreBaselines) {
  FakeSource src;
  src.samples = {MetricSample{"test_gauge", kHelp, {}, 7, MetricKind::kGauge}};
  MetricPublisher pub;
  Collector c;

  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.gauges["test_gauge"], 7.0);

  // A gauge that goes DOWN is normal (an arena drains), and it must be
  // reported as-is rather than swallowed the way a counter regression is.
  src.samples[0].value = 2;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.gauges["test_gauge"], 2.0);
}

TEST(MetricPublisher, AppliesScaleSoNamedUnitsStayHonest) {
  // Components accumulate in whatever integer unit an atomic add can carry;
  // the metric still has to mean what its name says.
  FakeSource src;
  src.samples = {
      MetricSample{"test_seconds_total", kHelp, {}, 2'500'000'000ULL, MetricKind::kCounter, 1e-9}};
  MetricPublisher pub;
  Collector c;
  pub.Publish("src", {}, src, c.Sink());
  EXPECT_DOUBLE_EQ(c.Counter("test_seconds_total"), 2.5);
}

TEST(MetricPublisher, PrependsSourceLabelsToEverySample) {
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {{"op", "commit"}}, 4}};
  MetricPublisher pub;
  Collector c;
  pub.Publish("src", {{"tier", "HBM"}, {"backend", "Mock"}}, src, c.Sink());
  EXPECT_DOUBLE_EQ(
      c.Counter(Identity(kName, {{"tier", "HBM"}, {"backend", "Mock"}, {"op", "commit"}})), 4.0);
}

// ---------------------------------------------------------------------------
//  InstrumentedBackend
// ---------------------------------------------------------------------------

class InstrumentedBackendMetrics : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mock = std::make_unique<MockBackend>(TierType::DRAM);
    mock_ = mock.get();
    backend_ = MakeInstrumentedBackend(std::move(mock));
    ASSERT_TRUE(backend_->Init(nullptr));
  }

  // Publish with the same labels PoolClient uses, so the identities asserted
  // below are byte-for-byte the ones a dashboard queries.
  void Publish() {
    collector_.counters.clear();
    collector_.gauges.clear();
    const MetricLabels labels = {{"tier", TierTypeName(backend_->Tier())},
                                 {"backend", backend_->Name()}};
    publisher_.Publish(std::string("backend:") + backend_->Name(), labels, *backend_,
                       collector_.Sink());
  }

  // mori_umbp_backend_ops_total{tier=...,backend=...,op=...,status=...}
  double Ops(const char* op, const char* status) {
    return collector_.Counter(
        Identity(MORI_UMBP_METRIC_BACKEND_OPS_TOTAL, {{"tier", TierTypeName(backend_->Tier())},
                                                      {"backend", backend_->Name()},
                                                      {"op", op},
                                                      {"status", status}}));
  }
  double Bytes(const char* op) {
    return collector_.Counter(Identity(
        MORI_UMBP_METRIC_BACKEND_BYTES_TOTAL,
        {{"tier", TierTypeName(backend_->Tier())}, {"backend", backend_->Name()}, {"op", op}}));
  }
  double Batches(const char* op) {
    return collector_.Counter(Identity(
        MORI_UMBP_METRIC_BACKEND_BATCHES_TOTAL,
        {{"tier", TierTypeName(backend_->Tier())}, {"backend", backend_->Name()}, {"op", op}}));
  }

  // Allocate + commit one key of `size` bytes, the way a writer does.
  void PutKey(const std::string& key, uint64_t size) {
    auto alloc = backend_->BatchAllocate({AllocateRequest{key, size}});
    ASSERT_EQ(alloc.size(), 1u);
    if (alloc[0].outcome != AllocateOutcome::kSuccessAllocated) return;
    backend_->BatchCommit({CommitRequest{alloc[0].slot_id, key}});
  }

  MockBackend* mock_ = nullptr;
  std::unique_ptr<MediumBackend> backend_;
  MetricPublisher publisher_;
  Collector collector_;
};

TEST_F(InstrumentedBackendMetrics, MetricsAreGenericOverBackends) {
  // MockBackend implements no observability of its own.  Everything asserted
  // here was derived from the MediumBackend interface by the decorator, which
  // is the guarantee: a new medium is instrumented by being composed in.
  PutKey("k1", 4096);
  PutKey("k2", 2048);
  backend_->BatchResolve({"k1", "missing"}, /*include_descs=*/false);
  backend_->Evict({"k2"});
  Publish();

  EXPECT_DOUBLE_EQ(Ops("allocate", "ok"), 2.0);
  EXPECT_DOUBLE_EQ(Ops("commit", "ok"), 2.0);
  EXPECT_DOUBLE_EQ(Bytes("commit"), 4096.0 + 2048.0);

  EXPECT_DOUBLE_EQ(Ops("resolve", "ok"), 1.0);
  EXPECT_DOUBLE_EQ(Ops("resolve", "miss"), 1.0);
  EXPECT_DOUBLE_EQ(Bytes("resolve"), 4096.0);

  EXPECT_DOUBLE_EQ(Ops("evict", "ok"), 1.0);
  EXPECT_DOUBLE_EQ(Bytes("evict"), 2048.0);
}

TEST_F(InstrumentedBackendMetrics, LabelsCarryTheMediumSoOnePanelShowsThemAll) {
  // The medium must appear as a LABEL VALUE and never inside a metric name —
  // that is what lets one panel render every backend.
  PutKey("k", 128);
  Publish();

  bool found = false;
  for (const auto& [identity, value] : collector_.counters) {
    (void)value;
    EXPECT_EQ(identity.find("dram"), std::string::npos)
        << "medium leaked into a metric identifier: " << identity;
    EXPECT_EQ(identity.find("_ssd_"), std::string::npos)
        << "medium leaked into a metric identifier: " << identity;
    if (identity.find(",tier=DRAM") != std::string::npos) found = true;
  }
  EXPECT_TRUE(found) << "no series carried the tier label";
}

TEST_F(InstrumentedBackendMetrics, CountsEntriesAndBatchesSeparately) {
  // Batch depth is a workload property worth seeing: one call, three keys.
  backend_->BatchAllocate(
      {AllocateRequest{"a", 16}, AllocateRequest{"b", 16}, AllocateRequest{"c", 16}});
  Publish();
  EXPECT_DOUBLE_EQ(Ops("allocate", "ok"), 3.0);
  EXPECT_DOUBLE_EQ(Batches("allocate"), 1.0);
}

TEST_F(InstrumentedBackendMetrics, SeparatesDedupFromAFreshAllocation) {
  // "already exists" is success to the caller but moves no bytes; a capacity
  // panel that lumps it in with a real allocation reads as write amplification
  // that is not happening.
  PutKey("dup", 64);
  backend_->BatchAllocate({AllocateRequest{"dup", 64}});
  Publish();
  EXPECT_DOUBLE_EQ(Ops("allocate", "exists"), 1.0);
  EXPECT_DOUBLE_EQ(Ops("allocate", "ok"), 1.0);
}

TEST_F(InstrumentedBackendMetrics, ReportsFailedAllocations) {
  backend_->BatchAllocate({AllocateRequest{"zero", 0}});  // MockBackend fails size 0
  Publish();
  EXPECT_DOUBLE_EQ(Ops("allocate", "failed"), 1.0);
}

TEST_F(InstrumentedBackendMetrics, CommitAgainstAReapedSlotCountsAsFailed) {
  auto alloc = backend_->BatchAllocate({AllocateRequest{"k", 32}});
  ASSERT_EQ(alloc.size(), 1u);
  backend_->BatchAbort({alloc[0].slot_id});
  backend_->BatchCommit({CommitRequest{alloc[0].slot_id, "k"}});
  Publish();
  EXPECT_DOUBLE_EQ(Ops("commit", "failed"), 1.0);
  EXPECT_DOUBLE_EQ(Ops("abort", "ok"), 1.0);
}

TEST_F(InstrumentedBackendMetrics, EvictingAnAbsentKeyIsAMissNotAFailure) {
  // Master retries protected keys on a later round; an eviction that freed
  // nothing is not an error and must not read as one.
  backend_->Evict({"never-existed"});
  Publish();
  EXPECT_DOUBLE_EQ(Ops("evict", "miss"), 1.0);
  EXPECT_DOUBLE_EQ(Ops("evict", "ok"), -1.0) << "no successful eviction should have been reported";
}

TEST_F(InstrumentedBackendMetrics, RecordsTimeSpentInsideTheMedium) {
  PutKey("k", 64);
  Publish();
  const double seconds = collector_.Counter(Identity(
      MORI_UMBP_METRIC_BACKEND_OP_SECONDS_TOTAL,
      {{"tier", TierTypeName(backend_->Tier())}, {"backend", backend_->Name()}, {"op", "commit"}}));
  EXPECT_GT(seconds, 0.0);
  // Scaled to seconds, not left in nanoseconds: a commit of one small key on
  // an in-memory mock cannot plausibly take a second.
  EXPECT_LT(seconds, 1.0);
}

TEST_F(InstrumentedBackendMetrics, ForwardsEveryCallToTheWrappedBackend) {
  // The decorator is on the live data path; behaviour must be untouched.
  PutKey("k", 100);
  EXPECT_EQ(backend_->OwnedKeyCount(), 1u);
  EXPECT_EQ(mock_->OwnedKeyCount(), 1u);

  auto resolved = backend_->BatchResolve({"k"}, false);
  ASSERT_EQ(resolved.size(), 1u);
  EXPECT_TRUE(resolved[0].found);
  EXPECT_EQ(resolved[0].size, 100u);

  EXPECT_EQ(backend_->Tier(), mock_->Tier());
  EXPECT_STREQ(backend_->Name(), mock_->Name());
  EXPECT_EQ(backend_->PageSize(), mock_->PageSize());

  auto events = backend_->DrainPendingEvents();
  EXPECT_FALSE(events.empty()) << "the heartbeat's outbox must survive the decorator";
}

TEST_F(InstrumentedBackendMetrics, PassesThroughAMediumsOwnSamples) {
  // A medium may still publish what the interface cannot show.  Those samples
  // must arrive alongside the generic ones, under the same tier/backend labels.
  class ChattyBackend : public MockBackend {
   public:
    ChattyBackend() : MockBackend(TierType::SSD) {}
    std::vector<MetricSample> SampleMetrics() const override {
      return {MetricSample{MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL,
                           MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL_HELP,
                           {{"event", "device_read_ok"}},
                           9}};
    }
  };

  auto wrapped = MakeInstrumentedBackend(std::make_unique<ChattyBackend>());
  MetricPublisher pub;
  Collector c;
  const MetricLabels labels = {{"tier", TierTypeName(wrapped->Tier())},
                               {"backend", wrapped->Name()}};
  pub.Publish("backend:chatty", labels, *wrapped, c.Sink());

  EXPECT_DOUBLE_EQ(c.Counter(Identity(
                       MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL,
                       {{"tier", "SSD"}, {"backend", "MockBackend"}, {"event", "device_read_ok"}})),
                   9.0);
}

// ---------------------------------------------------------------------------
//  Transfer layer
// ---------------------------------------------------------------------------

TEST(TransferEngineMetrics, ChargesBytesAndPlansToTheEngineThatCarriedThem) {
  CompositeTransferEngine composite;
  composite.AddEngine(std::make_unique<LocalCopyEngine>());

  std::vector<uint8_t> src(1024, 7);
  std::vector<uint8_t> dst(1024, 0);

  TransferItem item;
  item.src = TransferRef::HostBytes(src.data(), src.size());
  item.dst = TransferRef::HostBytes(dst.data(), dst.size());
  item.size = src.size();
  item.tag = 0;

  std::vector<size_t> failed;
  ASSERT_TRUE(composite.Transfer({item}, &failed));
  ASSERT_TRUE(failed.empty());
  EXPECT_EQ(dst, src) << "instrumentation must not disturb the transfer itself";

  MetricPublisher pub;
  Collector c;
  pub.Publish("transfer", {}, composite, c.Sink());

  const MetricLabels local = {{"engine", "LocalCopyEngine"}, {"direction", "local"}};
  EXPECT_DOUBLE_EQ(c.Counter(Identity(MORI_UMBP_METRIC_TRANSFER_BYTES_TOTAL, local)), 1024.0);
  MetricLabels ok = local;
  ok.push_back({"status", "ok"});
  EXPECT_DOUBLE_EQ(c.Counter(Identity(MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL, ok)), 1.0);
}

TEST(TransferEngineMetrics, CountsItemsNoEngineWouldTake) {
  // A rejected item is a routing failure, not a transport one, so it must not
  // be blamed on whichever engine happened to be registered.
  CompositeTransferEngine composite;
  composite.AddEngine(std::make_unique<LocalCopyEngine>());

  TransferItem item;  // both endpoints invalid: no engine can claim this pair
  item.size = 64;
  item.tag = 0;
  TransferPlanSet planned = composite.Plan({item});
  EXPECT_EQ(planned.rejected_tags.size(), 1u);

  MetricPublisher pub;
  Collector c;
  pub.Publish("transfer", {}, composite, c.Sink());
  EXPECT_DOUBLE_EQ(
      c.Counter(Identity(MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL,
                         {{"engine", "none"}, {"direction", "none"}, {"status", "rejected"}})),
      1.0);
}

TEST(TransferEngineMetrics, AnEngineNeedsNoMetricsCodeToBeMeasured) {
  // The composite is the measurement point precisely so an engine can be added
  // with AddEngine() alone.  An engine registered but never used reports
  // nothing rather than a row of zeros, keeping the series set honest.
  CompositeTransferEngine composite;
  composite.AddEngine(std::make_unique<LocalCopyEngine>());

  MetricPublisher pub;
  Collector c;
  pub.Publish("transfer", {}, composite, c.Sink());
  EXPECT_TRUE(c.counters.empty());
}

}  // namespace
}  // namespace mori::umbp
