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

// The traffic counters of each module, and how they become registry series
// (mori/metrics/traffic_counters.hpp, section 3 of mori/metrics/metrics.hpp).
// The IO engine cases run on the simulated backends; a case that needs hardware
// (two GPUs, an active RDMA device, a GPU) skips without it.
// Part 2 is last: it initialises HIP, and the fork case of part 1 has to run
// before that.

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"
#include "mori/metrics/metrics.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/protocol.hpp"

// Metrics are decided once per process, on first use: turn them on before main().
static const int kEnableMetrics = [] {
  setenv("MORI_METRICS_ENABLE", "1", 1);
  setenv("MORI_METRICS_HOST_ID", "host-self", 1);
  setenv("MORI_METRICS_INTERVAL_MS", "3600000", 1);  // the cases sample by hand
  unsetenv("MORI_METRICS_PORT");
  return 0;
}();

/* ------------------------------------------------------------------------ */
/* 1. IO                                                                    */
/* ------------------------------------------------------------------------ */

namespace mori::io {
namespace {

using internal::IoMetrics;
using mori::metrics::MetricLabels;
using mori::metrics::MetricSample;
using Labels = std::map<std::string, std::string>;

uint64_t L(const std::atomic<uint64_t>& a) { return a.load(); }

uint64_t Completed(const IoOpCounters& c) {
  return L(c.completedOk) + L(c.completedFailed) + L(c.completedUnobserved);
}

uint64_t LatencyCount(const IoOpCounters& c) {
  uint64_t n = 0;
  for (const auto& b : c.latencyBuckets) n += L(b);
  return n;
}

TEST(PerTransferMeter, FinishesOkWithBytesAndLatency) {
  IoFlowCounters flow;
  TransferStatus st;
  st.StartMeter(&flow, IoOp::kWrite, 4096);
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  EXPECT_EQ(L(w.submitted), 1u);
  EXPECT_EQ(L(w.submittedBytes), 4096u);
  EXPECT_EQ(Completed(w), 0u);

  st.SetCode(StatusCode::SUCCESS);
  EXPECT_EQ(L(w.completedOk), 1u);
  EXPECT_EQ(L(w.completedBytes), 4096u);
  EXPECT_EQ(LatencyCount(w), 1u);
  EXPECT_EQ(L(flow.op[static_cast<int>(IoOp::kRead)].submitted), 0u);
}

TEST(PerTransferMeter, InProgressDoesNotFinish) {
  IoFlowCounters flow;
  TransferStatus st;
  st.StartMeter(&flow, IoOp::kRead, 100);
  st.SetCode(StatusCode::IN_PROGRESS);
  st.Update(StatusCode::IN_PROGRESS, "still going");
  const IoOpCounters& r = flow.op[static_cast<int>(IoOp::kRead)];
  EXPECT_EQ(Completed(r), 0u);
  st.SetCode(StatusCode::SUCCESS);
  EXPECT_EQ(L(r.completedOk), 1u);
}

TEST(PerTransferMeter, FinishesOnlyOnce) {
  IoFlowCounters flow;
  TransferStatus st;
  st.StartMeter(&flow, IoOp::kWrite, 100);
  st.Update(StatusCode::ERR_RDMA_OP, "first");
  st.SetCode(StatusCode::ERR_BAD_STATE);
  st.Update(StatusCode::ERR_GPU_OP, "second");
  st.SetCode(StatusCode::SUCCESS);
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  EXPECT_EQ(L(w.completedFailed), 1u);
  EXPECT_EQ(L(w.completedOk), 0u);
  EXPECT_EQ(L(w.completedBytes), 0u);  // a failed transfer moved no accounted bytes
  EXPECT_EQ(LatencyCount(w), 1u);
}

TEST(PerTransferMeter, SynchronousCompletionIsCounted) {
  IoFlowCounters flow;
  TransferStatus st;
  st.StartMeter(&flow, IoOp::kWrite, 7);
  st.Update(StatusCode::ERR_INVALID_ARGS, "rejected at submit");
  EXPECT_EQ(L(flow.op[static_cast<int>(IoOp::kWrite)].completedFailed), 1u);
}

TEST(PerTransferMeter, DestroyedBeforeATerminalStateIsUnobserved) {
  IoFlowCounters flow;
  {
    TransferStatus st;
    st.StartMeter(&flow, IoOp::kWrite, 100);
    st.SetCode(StatusCode::IN_PROGRESS);
  }
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  EXPECT_EQ(L(w.completedUnobserved), 1u);
  EXPECT_EQ(L(w.completedOk) + L(w.completedFailed), 0u);
  EXPECT_EQ(LatencyCount(w), 0u);
}

TEST(PerTransferMeter, ReusedWhilePendingAbandonsTheEarlierTransfer) {
  IoFlowCounters flow;
  TransferStatus st;
  st.StartMeter(&flow, IoOp::kWrite, 100);
  st.StartMeter(&flow, IoOp::kRead, 50);
  EXPECT_EQ(L(flow.op[static_cast<int>(IoOp::kWrite)].completedUnobserved), 1u);
  st.SetCode(StatusCode::SUCCESS);
  EXPECT_EQ(L(flow.op[static_cast<int>(IoOp::kRead)].completedOk), 1u);
  EXPECT_EQ(L(flow.op[static_cast<int>(IoOp::kRead)].completedBytes), 50u);
}

TEST(PerTransferMeter, NoFlowIsANoOp) {
  TransferStatus st;
  st.StartMeter(nullptr, IoOp::kWrite, 100);
  st.SetCode(StatusCode::SUCCESS);
  EXPECT_TRUE(st.Succeeded());
}

TEST(PerTransferMeter, RacingFinishersFinishOnce) {
  IoFlowCounters flow;
  const int kRounds = 2000;
  for (int i = 0; i < kRounds; ++i) {
    TransferStatus st;
    st.StartMeter(&flow, IoOp::kWrite, 1);
    std::atomic<bool> go{false};
    std::thread a([&] {
      while (!go.load()) {
      }
      st.SetCode(StatusCode::SUCCESS);
    });
    std::thread b([&] {
      while (!go.load()) {
      }
      st.Update(StatusCode::ERR_RDMA_OP, "x");
    });
    go.store(true);
    a.join();
    b.join();
  }
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  EXPECT_EQ(L(w.submitted), static_cast<uint64_t>(kRounds));
  EXPECT_EQ(Completed(w), static_cast<uint64_t>(kRounds));
  EXPECT_EQ(LatencyCount(w), static_cast<uint64_t>(kRounds));
}

void RunTransfers(IoFlowCounters* flow, IoOp op, int n, uint64_t bytes) {
  for (int i = 0; i < n; ++i) {
    TransferStatus st;
    st.StartMeter(flow, op, bytes);
    st.SetCode(StatusCode::SUCCESS);
  }
}

TEST(IoSampling, OneInNCountsNFold) {
  std::atomic<uint32_t> period{8};
  IoFlowCounters flow;
  flow.samplePeriod = &period;
  const int n = 80000;
  RunTransfers(&flow, IoOp::kWrite, n, 100);
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  EXPECT_EQ(L(w.submitted) % 8, 0u);
  EXPECT_EQ(L(w.submittedBytes), L(w.submitted) * 100);
  EXPECT_EQ(L(w.completedOk), L(w.submitted));
  EXPECT_EQ(L(w.completedBytes), L(w.submittedBytes));
  EXPECT_EQ(LatencyCount(w), L(w.submitted));
  // 10000 gaps drawn from 1..15: within a few percent; a tenth is far outside chance.
  EXPECT_GT(L(w.submitted), n * 9 / 10);
  EXPECT_LT(L(w.submitted), n * 11 / 10);
}

TEST(IoSampling, ZeroCountsNothingAndOneEverything) {
  std::atomic<uint32_t> period{0};
  IoFlowCounters flow;
  flow.samplePeriod = &period;
  RunTransfers(&flow, IoOp::kRead, 1000, 64);
  const IoOpCounters& r = flow.op[static_cast<int>(IoOp::kRead)];
  EXPECT_EQ(L(r.submitted), 0u);
  EXPECT_EQ(Completed(r), 0u);
  period = 1;
  RunTransfers(&flow, IoOp::kRead, 1000, 64);
  EXPECT_EQ(L(r.submitted), 1000u);
  EXPECT_EQ(L(r.completedBytes), 64000u);
}

// A fixed gap of 2 would take one kind every time and never the other.
TEST(IoSampling, AlternatingKindsAreBothMetered) {
  for (uint32_t n : {2u, 4u, 16u}) {
    std::atomic<uint32_t> period{n};
    IoFlowCounters flow;
    flow.samplePeriod = &period;
    const int rounds = 40000;
    for (int i = 0; i < rounds; ++i) {
      RunTransfers(&flow, IoOp::kWrite, 1, 1000);
      RunTransfers(&flow, IoOp::kRead, 1, 10);
    }
    const uint64_t writes = L(flow.op[static_cast<int>(IoOp::kWrite)].submitted);
    const uint64_t reads = L(flow.op[static_cast<int>(IoOp::kRead)].submitted);
    EXPECT_GT(writes, static_cast<uint64_t>(rounds) * 8 / 10) << n;
    EXPECT_LT(writes, static_cast<uint64_t>(rounds) * 12 / 10) << n;
    EXPECT_GT(reads, static_cast<uint64_t>(rounds) * 8 / 10) << n;
    EXPECT_LT(reads, static_cast<uint64_t>(rounds) * 12 / 10) << n;
  }
}

TEST(IoSampling, AnInFlightTransferKeepsItsWeight) {
  std::atomic<uint32_t> period{4};
  IoFlowCounters flow;
  flow.samplePeriod = &period;
  const IoOpCounters& w = flow.op[static_cast<int>(IoOp::kWrite)];
  TransferStatus st;
  for (int i = 0; i < 100 && L(w.submitted) == 0; ++i) st.StartMeter(&flow, IoOp::kWrite, 10);
  ASSERT_EQ(L(w.submitted), 4u);  // the one that was taken, pending in `st`
  period = 1;
  st.SetCode(StatusCode::SUCCESS);
  EXPECT_EQ(L(w.completedOk), 4u);
  EXPECT_EQ(L(w.completedBytes), 40u);
  // Lowering the period ends a long gap that was being counted down.
  period = 1000;
  RunTransfers(&flow, IoOp::kWrite, 1, 10);  // draws a gap of up to 1999
  period = 2;
  const uint64_t before = L(w.submitted);
  RunTransfers(&flow, IoOp::kWrite, 8, 10);
  EXPECT_GT(L(w.submitted), before);
}

TEST(IoSampling, ShortLivedThreadsAreNotCountedSeveralTimesOver) {
  std::atomic<uint32_t> period{16};
  IoFlowCounters flow;
  flow.samplePeriod = &period;
  const int threads = 6000, each = 3;
  for (int t = 0; t < threads; ++t) {
    std::thread([&] { RunTransfers(&flow, IoOp::kWrite, each, 10); }).join();
  }
  const uint64_t counted = L(flow.op[static_cast<int>(IoOp::kWrite)].submitted);
  // Some 1100 of 18000 are taken; taking every thread's first one would count 96000.
  EXPECT_GT(counted, static_cast<uint64_t>(threads * each) * 85 / 100);
  EXPECT_LT(counted, static_cast<uint64_t>(threads * each) * 115 / 100);
}

TEST(IoSampling, FlowsOfATableFollowTheProcessSetting) {
  IoMetrics table;
  IoFlowIdentity id{"rdma", "0", "host-b", "1"};
  IoFlowCounters* flow = table.Flow(id);
  ASSERT_EQ(flow->samplePeriod, mori::metrics::IoSamplePeriodPtr());

  mori::metrics::Registry::Instance().SetSampling(-1, 0, -1);
  RunTransfers(flow, IoOp::kWrite, 50, 10);
  EXPECT_EQ(L(flow->op[static_cast<int>(IoOp::kWrite)].submitted), 0u);
  mori::metrics::Registry::Instance().SetSampling(-1, 1, -1);
  RunTransfers(flow, IoOp::kWrite, 50, 10);
  EXPECT_EQ(L(flow->op[static_cast<int>(IoOp::kWrite)].submitted), 50u);
}

TEST(IoFlowCounters, LatencyLandsInTheRightBucket) {
  IoFlowCounters flow;
  flow.Complete(IoOp::kRead, true, 1, 9'000);               // <= 10 us: bucket 0
  flow.Complete(IoOp::kRead, true, 1, 10'000);              // == bound: still bucket 0
  flow.Complete(IoOp::kRead, true, 1, 10'001);              // bucket 1
  flow.Complete(IoOp::kRead, false, 1, 60'000'000'000ULL);  // above every bound
  const IoOpCounters& r = flow.op[static_cast<int>(IoOp::kRead)];
  EXPECT_EQ(L(r.latencyBuckets[0]), 2u);
  EXPECT_EQ(L(r.latencyBuckets[1]), 1u);
  EXPECT_EQ(L(r.latencyBuckets[kIoLatencyBoundCount]), 1u);
  EXPECT_EQ(L(r.latencyNs), 9'000u + 10'000u + 10'001u + 60'000'000'000ULL);
}

Labels ToMap(const MetricLabels& labels) { return Labels(labels.begin(), labels.end()); }

// The one sample named `name` whose labels include all of `want`.
const MetricSample* FindSample(const std::vector<MetricSample>& all, const std::string& name,
                               const Labels& want) {
  const MetricSample* found = nullptr;
  for (const auto& s : all) {
    if (name != s.name) continue;
    Labels have = ToMap(s.labels);
    bool match = true;
    for (const auto& [k, v] : want) {
      auto it = have.find(k);
      if (it == have.end() || it->second != v) match = false;
    }
    if (!match) continue;
    EXPECT_EQ(found, nullptr) << "more than one sample of " << name << " matches";
    found = &s;
  }
  return found;
}

TEST(IoMetricsTable, SameIdentityIsTheSameBlock) {
  IoMetrics table;
  IoFlowIdentity id{"rdma", "3", "node-b", "5"};
  IoFlowCounters* a = table.Flow(id);
  EXPECT_EQ(table.Flow(id), a);
  IoFlowIdentity other = id;
  other.remoteGpu = "6";
  EXPECT_NE(table.Flow(other), a);
  // Blocks do not move when the table grows.
  for (int i = 0; i < 1000; ++i) {
    IoFlowIdentity more = id;
    more.remoteHost = "node-" + std::to_string(i);
    table.Flow(more);
  }
  EXPECT_EQ(table.Flow(id), a);
}

TEST(IoMetricsTable, BlocksCarryTheirIdentity) {
  IoMetrics table;
  const IoFlowCounters* flow = table.Flow({"rdma", "3", "node-b", "5"});
  EXPECT_EQ(flow->id.backend, "rdma");
  EXPECT_EQ(flow->id.localGpu, "3");
  EXPECT_EQ(flow->id.remoteHost, "node-b");
  EXPECT_EQ(flow->id.remoteGpu, "5");

  const NicFlowCounters* nf = table.NicFlow({"ionic_3", "ens3", "node-b", "ionic_5", "ens5"});
  EXPECT_EQ(nf->id.ibdev, "ionic_3");
  EXPECT_EQ(nf->id.netdev, "ens3");
  EXPECT_EQ(nf->id.remoteHost, "node-b");
  EXPECT_EQ(nf->id.remoteIbdev, "ionic_5");
  EXPECT_EQ(nf->id.remoteNetdev, "ens5");
}

// A write flows local -> remote, a read remote -> local.
TEST(IoMetricsTable, FlowIsLabelledByDirection) {
  IoMetrics table;
  IoFlowCounters* flow = table.Flow({"rdma", "3", "node-b", "5"});
  flow->Submit(IoOp::kWrite, 1000);
  flow->Complete(IoOp::kWrite, true, 1000, 20'000);
  flow->Submit(IoOp::kRead, 300);
  flow->Complete(IoOp::kRead, true, 300, 20'000);

  auto samples = table.SampleMetrics();
  const Labels write = {{"backend", "rdma"}, {"op", "write"},        {"src_host", "host-self"},
                        {"src_gpu", "3"},    {"dst_host", "node-b"}, {"dst_gpu", "5"}};
  const Labels read = {{"backend", "rdma"},       {"op", "read"},
                       {"src_host", "node-b"},    {"src_gpu", "5"},
                       {"dst_host", "host-self"}, {"dst_gpu", "3"}};
  const MetricSample* w = FindSample(samples, "mori_io_completed_bytes_total", write);
  const MetricSample* r = FindSample(samples, "mori_io_completed_bytes_total", read);
  ASSERT_NE(w, nullptr);
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(w->value, 1000u);
  EXPECT_EQ(r->value, 300u);
  // A flow sample never carries a bare `gpu`: the registry adds only `host`.
  EXPECT_EQ(ToMap(w->labels).count("gpu"), 0u);

  Labels ok = write;
  ok["status"] = "ok";
  ASSERT_NE(FindSample(samples, "mori_io_completed_total", ok), nullptr);
  EXPECT_EQ(FindSample(samples, "mori_io_completed_total", ok)->value, 1u);
  // Every status exists from the start, at zero.
  Labels failed = write;
  failed["status"] = "failed";
  ASSERT_NE(FindSample(samples, "mori_io_completed_total", failed), nullptr);
  EXPECT_EQ(FindSample(samples, "mori_io_completed_total", failed)->value, 0u);
}

TEST(IoMetricsTable, LatencyAndInflightAreSummedPerLocalGpu) {
  IoMetrics table;
  IoFlowCounters* toB = table.Flow({"rdma", "3", "node-b", "5"});
  IoFlowCounters* toC = table.Flow({"rdma", "3", "node-c", "1"});
  IoFlowCounters* gpu4 = table.Flow({"rdma", "4", "node-b", "5"});
  toB->Submit(IoOp::kWrite, 1);
  toB->Complete(IoOp::kWrite, true, 1, 9'000);  // <= 10 us
  toC->Submit(IoOp::kWrite, 1);
  toC->Complete(IoOp::kWrite, true, 1, 40'000);  // <= 50 us
  toC->Submit(IoOp::kWrite, 1);                  // still in flight
  toC->Submit(IoOp::kRead, 1);                   // still in flight
  gpu4->Submit(IoOp::kWrite, 1);
  gpu4->Complete(IoOp::kWrite, false, 1, 9'000);

  auto samples = table.SampleMetrics();
  const MetricSample* lat = FindSample(samples, "mori_io_latency_seconds",
                                       {{"backend", "rdma"}, {"op", "write"}, {"gpu", "3"}});
  ASSERT_NE(lat, nullptr);
  EXPECT_EQ(lat->kind, mori::metrics::MetricKind::kHistogram);
  ASSERT_EQ(lat->buckets.size(), static_cast<size_t>(kIoLatencyBoundCount));
  EXPECT_EQ(lat->count, 2u);
  EXPECT_EQ(lat->buckets[0], 1u);  // cumulative: <= 10 us
  EXPECT_EQ(lat->buckets[1], 1u);  // <= 25 us
  EXPECT_EQ(lat->buckets[2], 2u);  // <= 50 us
  EXPECT_EQ(lat->buckets.back(), 2u);
  EXPECT_EQ(lat->value, 49'000u);
  EXPECT_DOUBLE_EQ(lat->scale, 1e-9);
  ASSERT_NE(lat->bounds, nullptr);
  EXPECT_DOUBLE_EQ(lat->bounds->front(), 10e-6);

  const MetricSample* inflight =
      FindSample(samples, "mori_io_inflight", {{"backend", "rdma"}, {"gpu", "3"}});
  ASSERT_NE(inflight, nullptr);
  EXPECT_EQ(inflight->value, 2u);  // one write and one read from gpu 3
  EXPECT_EQ(inflight->kind, mori::metrics::MetricKind::kGauge);
  EXPECT_EQ(FindSample(samples, "mori_io_inflight", {{"backend", "rdma"}, {"gpu", "4"}})->value,
            0u);
}

TEST(IoMetricsTable, NicFlowIsLabelledByDirection) {
  IoMetrics table;
  NicFlowCounters* nf = table.NicFlow({"ionic_3", "ens3", "node-b", "ionic_5", "ens5"});
  EXPECT_EQ(table.NicFlow({"ionic_3", "ens3", "node-b", "ionic_5", "ens5"}), nf);
  nf->Posted(/*isRead=*/false, 4000);
  nf->Posted(/*isRead=*/true, 500);

  auto samples = table.SampleMetrics();
  const MetricSample* tx = FindSample(samples, "mori_io_nic_bytes_total",
                                      {{"op", "write"},
                                       {"src_host", "host-self"},
                                       {"src_ibdev", "ionic_3"},
                                       {"src_netdev", "ens3"},
                                       {"dst_host", "node-b"},
                                       {"dst_ibdev", "ionic_5"},
                                       {"dst_netdev", "ens5"}});
  const MetricSample* rx = FindSample(samples, "mori_io_nic_bytes_total",
                                      {{"op", "read"},
                                       {"src_host", "node-b"},
                                       {"src_ibdev", "ionic_5"},
                                       {"dst_host", "host-self"},
                                       {"dst_ibdev", "ionic_3"}});
  ASSERT_NE(tx, nullptr);
  ASSERT_NE(rx, nullptr);
  EXPECT_EQ(tx->value, 4000u);
  EXPECT_EQ(rx->value, 500u);
}

TEST(IoMetricsTable, HostLabelOfAPeer) {
  IoMetrics table;
  char raw[256] = {0};
  ASSERT_EQ(gethostname(raw, sizeof(raw) - 1), 0);
  EXPECT_EQ(table.Host(), "host-self");          // MORI_METRICS_HOST_ID
  EXPECT_EQ(table.HostLabel(raw), "host-self");  // a peer on this machine
  EXPECT_EQ(table.HostLabel("node-b"), "node-b");
  EXPECT_EQ(table.HostLabel(""), "unknown");
}

// `obj` (a msgpack array) without its last `drop` elements: what an older sender packs.
msgpack::object_handle Truncated(const msgpack::object& obj, size_t drop) {
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(&buf);
  const uint32_t n = obj.via.array.size - static_cast<uint32_t>(drop);
  pk.pack_array(n);
  for (uint32_t i = 0; i < n; ++i) pk.pack(obj.via.array.ptr[i]);
  return msgpack::unpack(buf.data(), buf.size());
}

TEST(WireCompat, MemoryDescGpuIndexRoundTripsAndIsOptional) {
  MemoryDesc d;
  d.engineKey = "e";
  d.id = 7;
  d.deviceId = 0;
  d.deviceBusId = "0000:c5:00.0";
  d.loc = MemoryLocationType::GPU;
  d.gpuIndex = 6;
  msgpack::sbuffer buf;
  msgpack::pack(buf, d);
  auto oh = msgpack::unpack(buf.data(), buf.size());
  EXPECT_EQ(oh.get().as<MemoryDesc>().gpuIndex, 6);

  MemoryDesc old = Truncated(oh.get(), 1).get().as<MemoryDesc>();
  EXPECT_EQ(old.gpuIndex, -1);
  EXPECT_EQ(old.deviceBusId, "0000:c5:00.0");
  EXPECT_EQ(old.id, 7u);
  EXPECT_TRUE(old == d);  // gpuIndex is not part of a descriptor's identity
}

TEST(WireCompat, EngineDescMetricsHostRoundTripsAndIsOptional) {
  EngineDesc d;
  d.key = "e";
  d.hostname = "node-b.cluster";
  d.host = "10.0.0.2";
  d.port = 7000;
  d.metricsHost = "rack7-b";
  msgpack::sbuffer buf;
  msgpack::pack(buf, d);
  auto oh = msgpack::unpack(buf.data(), buf.size());
  EXPECT_EQ(oh.get().as<EngineDesc>().MetricsHost(), "rack7-b");

  EngineDesc old = Truncated(oh.get(), 1).get().as<EngineDesc>();
  EXPECT_TRUE(old.metricsHost.empty());
  EXPECT_EQ(old.MetricsHost(), "node-b.cluster");
  EXPECT_EQ(old.port, 7000);
  EXPECT_TRUE(old == d);  // the label is not part of an engine's identity
}

// The handshake as a peer one release older knows it.
struct OldMessageRegEndpoint {
  EngineKey ekey;
  TopoKeyPair topo;
  int devId;
  application::RdmaEndpointHandle eph;
  int nicRank{0};
  int railId{-1};
  MSGPACK_DEFINE(ekey, topo, devId, eph, nicRank, railId);
};

TEST(WireCompat, RegEndpointNicNamesGoBothWaysAcrossVersions) {
  MessageRegEndpoint now{};
  now.ekey = "engine";
  now.devId = 3;
  now.nicRank = 1;
  now.railId = 3;
  now.ibdev = "ionic_3";
  now.netdev = "ens3f0np0";
  msgpack::sbuffer buf;
  msgpack::pack(buf, now);
  auto oh = msgpack::unpack(buf.data(), buf.size());
  MessageRegEndpoint same = oh.get().as<MessageRegEndpoint>();
  EXPECT_EQ(same.ibdev, "ionic_3");
  EXPECT_EQ(same.netdev, "ens3f0np0");

  // New sender, old receiver: the extra fields are ignored.
  OldMessageRegEndpoint older = oh.get().as<OldMessageRegEndpoint>();
  EXPECT_EQ(older.ekey, "engine");
  EXPECT_EQ(older.devId, 3);
  EXPECT_EQ(older.railId, 3);

  // Old sender, new receiver: the names are empty, the rest is intact.
  OldMessageRegEndpoint old{};
  old.ekey = "engine";
  old.devId = 5;
  old.railId = 2;
  msgpack::sbuffer oldBuf;
  msgpack::pack(oldBuf, old);
  MessageRegEndpoint fromOld =
      msgpack::unpack(oldBuf.data(), oldBuf.size()).get().as<MessageRegEndpoint>();
  EXPECT_EQ(fromOld.devId, 5);
  EXPECT_EQ(fromOld.railId, 2);
  EXPECT_TRUE(fromOld.ibdev.empty());
  EXPECT_TRUE(fromOld.netdev.empty());
}

// Sum of the registry series named `name` whose labels include `want`.
double Series(const std::string& name, const Labels& want) {
  double total = 0;
  for (const auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name != name) continue;
    Labels have(s.labels.begin(), s.labels.end());
    bool match = true;
    for (const auto& [k, v] : want) {
      auto it = have.find(k);
      if (it == have.end() || it->second != v) match = false;
    }
    if (match) total += s.value;
  }
  return total;
}

int Count(const std::string& name, const Labels& want) {
  int n = 0;
  for (const auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name != name) continue;
    Labels have(s.labels.begin(), s.labels.end());
    bool match = true;
    for (const auto& [k, v] : want) {
      auto it = have.find(k);
      if (it == have.end() || it->second != v) match = false;
    }
    if (match) ++n;
  }
  return n;
}

bool WaitDone(TransferStatus* st, int timeoutMs = 20000) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!st->Init() && !st->InProgress()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

std::string UniqueKey(const std::string& tag) {
  static std::atomic<int> n{0};
  return tag + "-" + std::to_string(getpid()) + "-" + std::to_string(n.fetch_add(1));
}

// Two engines in this process on the simulated backends. `remoteMetricsHost` "-"
// is a target that sends no host label (an older build).
struct SimPair {
  std::unique_ptr<IOEngine> a;
  std::unique_ptr<IOEngine> b;

  explicit SimPair(const std::string& remoteHostname = "",
                   const std::string& remoteMetricsHost = "") {
    setenv("MORI_IO_SIM_ENABLE", "1", 1);
    setenv("MORI_IO_SIM_TIME_SCALE", "0", 1);
    IOEngineConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 0;
    a = std::make_unique<IOEngine>(UniqueKey("sim-a"), cfg);
    b = std::make_unique<IOEngine>(UniqueKey("sim-b"), cfg);
    RdmaBackendConfig rdmaCfg{};
    a->CreateBackend(BackendType::RDMA, rdmaCfg);
    b->CreateBackend(BackendType::RDMA, rdmaCfg);
    EngineDesc bDesc = b->GetEngineDesc();
    if (!remoteHostname.empty()) {
      bDesc.hostname = remoteHostname;
      bDesc.metricsHost = remoteMetricsHost.empty()  ? remoteHostname
                          : remoteMetricsHost == "-" ? std::string()
                                                     : remoteMetricsHost;
    }
    a->RegisterRemoteEngine(bDesc);
    b->RegisterRemoteEngine(a->GetEngineDesc());
  }
  ~SimPair() {
    a.reset();
    b.reset();
    unsetenv("MORI_IO_SIM_ENABLE");
    unsetenv("MORI_IO_SIM_TIME_SCALE");
  }
};

void* FakePtr() { return reinterpret_cast<void*>(0x10000000ULL); }

TEST(IoMetricsEngine, MetricsAreOnAndRegistered) {
  ASSERT_NE(IoMetrics::Get(), nullptr);
  SimPair pair;
  IoMetrics::Get()->Flow({"rdma", "0", "node-registered", "1"})->Submit(IoOp::kWrite, 1);
  EXPECT_EQ(Count("mori_io_submitted_total",
                  {{"op", "write"}, {"dst_host", "node-registered"}, {"host", "host-self"}}),
            1);
}

// A simulated GPU's physical index is its ordinal.
TEST(IoMetricsEngine, RegisterMemoryFillsGpuIndex) {
  SimPair pair;
  MemoryDesc gpu = pair.a->RegisterMemory(FakePtr(), 4096, 5, MemoryLocationType::GPU);
  EXPECT_EQ(gpu.gpuIndex, 5);
  MemoryDesc cpu = pair.a->RegisterMemory(FakePtr(), 4096, -1, MemoryLocationType::CPU);
  EXPECT_EQ(cpu.gpuIndex, -1);
}

TEST(IoMetricsEngine, EngineEntriesAccountBothDirections) {
  SimPair pair("node-b");
  MemoryDesc local = pair.a->RegisterMemory(FakePtr(), 1 << 20, 2, MemoryLocationType::GPU);
  MemoryDesc remote = pair.b->RegisterMemory(FakePtr(), 1 << 20, 5, MemoryLocationType::GPU);

  const Labels write = {{"host", "host-self"},     {"backend", "rdma"}, {"op", "write"},
                        {"src_host", "host-self"}, {"src_gpu", "2"},    {"dst_host", "node-b"},
                        {"dst_gpu", "5"}};
  const Labels read = {{"host", "host-self"},  {"backend", "rdma"}, {"op", "read"},
                       {"src_host", "node-b"}, {"src_gpu", "5"},    {"dst_host", "host-self"},
                       {"dst_gpu", "2"}};
  Labels writeOk = write, readOk = read;
  writeOk["status"] = "ok";
  readOk["status"] = "ok";

  TransferStatus w1, r1, w2, r2;
  pair.a->Write(local, 0, remote, 0, 1000, &w1, pair.a->AllocateTransferUniqueId());
  pair.a->Read(local, 0, remote, 0, 300, &r1, pair.a->AllocateTransferUniqueId());
  ASSERT_TRUE(WaitDone(&w1));
  ASSERT_TRUE(WaitDone(&r1));
  ASSERT_TRUE(w1.Succeeded()) << w1.Message();
  ASSERT_TRUE(r1.Succeeded()) << r1.Message();

  // A batched transfer is one transfer of the sum of its pieces.
  MemDescVec locals{local}, remotes{remote};
  BatchSizeVec offs{{0, 4096, 8192}}, sizes{{100, 200, 300}};
  TransferStatusPtrVec wst{&w2}, rst{&r2};
  TransferUniqueIdVec ids{pair.a->AllocateTransferUniqueId()};
  pair.a->BatchWrite(locals, offs, remotes, offs, sizes, wst, ids);
  ids[0] = pair.a->AllocateTransferUniqueId();
  pair.a->BatchRead(locals, offs, remotes, offs, sizes, rst, ids);
  ASSERT_TRUE(WaitDone(&w2));
  ASSERT_TRUE(WaitDone(&r2));
  ASSERT_TRUE(w2.Succeeded()) << w2.Message();
  ASSERT_TRUE(r2.Succeeded()) << r2.Message();

  EXPECT_EQ(Series("mori_io_submitted_total", write), 2);
  EXPECT_EQ(Series("mori_io_submitted_bytes_total", write), 1600);
  EXPECT_EQ(Series("mori_io_completed_bytes_total", write), 1600);
  EXPECT_EQ(Series("mori_io_completed_total", writeOk), 2);
  EXPECT_EQ(Series("mori_io_submitted_total", read), 2);
  EXPECT_EQ(Series("mori_io_completed_bytes_total", read), 900);
  EXPECT_EQ(Series("mori_io_completed_total", readOk), 2);
  EXPECT_EQ(Count("mori_io_completed_bytes_total", write), 1);

  const Labels lat = {{"host", "host-self"}, {"backend", "rdma"}, {"op", "write"}, {"gpu", "2"}};
  EXPECT_EQ(Count("mori_io_latency_seconds", lat), 1);
  for (const auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name != "mori_io_latency_seconds") continue;
    Labels have(s.labels.begin(), s.labels.end());
    if (have["gpu"] != "2" || have["op"] != "write") continue;
    EXPECT_EQ(s.count, 2u);
    EXPECT_EQ(s.buckets.back(), 2u);
  }
  EXPECT_EQ(Series("mori_io_inflight", {{"backend", "rdma"}, {"gpu", "2"}}), 0);
  EXPECT_EQ(mori::metrics::Registry::Instance().RejectedSamples(), 0u);
}

TEST(IoMetricsEngine, SessionEntriesAccountToTheSessionsFlow) {
  SimPair pair("node-c");
  MemoryDesc local = pair.a->RegisterMemory(FakePtr(), 1 << 20, 1, MemoryLocationType::GPU);
  MemoryDesc remote = pair.b->RegisterMemory(FakePtr(), 1 << 20, 7, MemoryLocationType::GPU);
  auto sess = pair.a->CreateSession(local, remote);
  ASSERT_TRUE(sess.has_value());

  TransferStatus w, r, bw, br;
  sess->Write(0, 0, 64, &w, sess->AllocateTransferUniqueId());
  sess->Read(0, 0, 32, &r, sess->AllocateTransferUniqueId());
  sess->BatchWrite({0, 100}, {0, 100}, {10, 20}, &bw, sess->AllocateTransferUniqueId());
  sess->BatchRead({0, 100}, {0, 100}, {1, 2}, &br, sess->AllocateTransferUniqueId());
  for (TransferStatus* st : {&w, &r, &bw, &br}) {
    ASSERT_TRUE(WaitDone(st));
    ASSERT_TRUE(st->Succeeded()) << st->Message();
  }
  const Labels write = {{"backend", "rdma"},
                        {"op", "write"},
                        {"src_gpu", "1"},
                        {"dst_host", "node-c"},
                        {"dst_gpu", "7"}};
  const Labels read = {{"backend", "rdma"},
                       {"op", "read"},
                       {"src_host", "node-c"},
                       {"src_gpu", "7"},
                       {"dst_gpu", "1"}};
  EXPECT_EQ(Series("mori_io_completed_bytes_total", write), 94);
  EXPECT_EQ(Series("mori_io_completed_bytes_total", read), 35);
  EXPECT_EQ(Series("mori_io_submitted_total", write), 2);
  EXPECT_EQ(Series("mori_io_submitted_total", read), 2);
}

TEST(IoMetricsEngine, HostMemoryIsCpuAndPeerOnThisHostIsSelf) {
  SimPair pair;  // the target advertises this machine's own hostname
  MemoryDesc local = pair.a->RegisterMemory(FakePtr(), 4096, -1, MemoryLocationType::CPU);
  MemoryDesc remote = pair.b->RegisterMemory(FakePtr(), 4096, 3, MemoryLocationType::GPU);
  const Labels write = {{"backend", "rdma"}, {"op", "write"},           {"src_host", "host-self"},
                        {"src_gpu", "cpu"},  {"dst_host", "host-self"}, {"dst_gpu", "3"}};
  const double before = Series("mori_io_completed_bytes_total", write);
  TransferStatus st;
  pair.a->Write(local, 0, remote, 0, 77, &st, pair.a->AllocateTransferUniqueId());
  ASSERT_TRUE(WaitDone(&st));
  ASSERT_TRUE(st.Succeeded()) << st.Message();
  EXPECT_EQ(Series("mori_io_completed_bytes_total", write) - before, 77);
}

// By the host label the peer sends, or its hostname when its build sends none.
TEST(IoMetricsEngine, RemoteHostIsThePeersOwnHostLabel) {
  {
    SimPair own;  // what this process sends: its MORI_METRICS_HOST_ID
    EXPECT_EQ(own.a->GetEngineDesc().metricsHost, "host-self");
  }
  for (const auto& [metricsHost, want] :
       {std::pair<const char*, const char*>{"rack7-b", "rack7-b"}, {"-", "node-b.cluster"}}) {
    SimPair pair("node-b.cluster", metricsHost);
    MemoryDesc local = pair.a->RegisterMemory(FakePtr(), 4096, 1, MemoryLocationType::GPU);
    MemoryDesc remote = pair.b->RegisterMemory(FakePtr(), 4096, 4, MemoryLocationType::GPU);
    const Labels write = {{"backend", "rdma"}, {"op", "write"},    {"src_host", "host-self"},
                          {"src_gpu", "1"},    {"dst_host", want}, {"dst_gpu", "4"}};
    const double before = Series("mori_io_completed_bytes_total", write);
    TransferStatus st;
    pair.a->Write(local, 0, remote, 0, 33, &st, pair.a->AllocateTransferUniqueId());
    ASSERT_TRUE(WaitDone(&st));
    ASSERT_TRUE(st.Succeeded()) << st.Message();
    EXPECT_EQ(Series("mori_io_completed_bytes_total", write) - before, 33) << want;
  }
}

TEST(IoMetricsEngine, NoBackendIsNotAFlow) {
  setenv("MORI_IO_SIM_ENABLE", "1", 1);
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  IOEngine lonely(UniqueKey("lonely"), cfg);  // no backend at all
  MemoryDesc local = lonely.RegisterMemory(FakePtr(), 4096, 0, MemoryLocationType::GPU);
  MemoryDesc remote = local;
  remote.engineKey = "nobody";
  const double before = Series("mori_io_submitted_total", {});
  TransferStatus st;
  lonely.Write(local, 0, remote, 0, 10, &st, lonely.AllocateTransferUniqueId());
  EXPECT_TRUE(st.Failed());
  EXPECT_EQ(Series("mori_io_submitted_total", {}), before);
  unsetenv("MORI_IO_SIM_ENABLE");
}

TEST(IoMetricsEngine, ForkedChildCountsFromZeroInItsOwnRegistry) {
  SimPair pair("node-fork");
  MemoryDesc local = pair.a->RegisterMemory(FakePtr(), 4096, 0, MemoryLocationType::GPU);
  MemoryDesc remote = pair.b->RegisterMemory(FakePtr(), 4096, 1, MemoryLocationType::GPU);
  TransferStatus st;
  pair.a->Write(local, 0, remote, 0, 123, &st, pair.a->AllocateTransferUniqueId());
  ASSERT_TRUE(WaitDone(&st));
  const Labels flow = {{"dst_host", "node-fork"}, {"op", "write"}};
  ASSERT_EQ(Series("mori_io_completed_bytes_total", flow), 123);
  mori::metrics::Registry* parent = &mori::metrics::Registry::Instance();

  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(20);  // a child stuck on an inherited lock is reported by the parent
    int rc = 0;
    if (&mori::metrics::Registry::Instance() == parent) rc |= 1;
    // Nothing is published in the child until something registers again.
    if (Count("mori_io_completed_bytes_total", flow) != 0) rc |= 2;
    IoMetrics::Get()->EnsureRegistered();  // what creating an engine does
    if (Count("mori_io_completed_bytes_total", flow) != 1) rc |= 4;
    if (Series("mori_io_completed_bytes_total", flow) != 0) rc |= 8;
    if (Series("mori_io_submitted_total", flow) != 0) rc |= 16;
    IoMetrics::Get()->Flow({"rdma", "0", "node-fork", "1"})->Submit(IoOp::kWrite, 9);
    if (Series("mori_io_submitted_bytes_total", flow) != 9) rc |= 32;
    _exit(rc);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status)) << "child killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0) << "child failure bits";
  EXPECT_EQ(&mori::metrics::Registry::Instance(), parent);
  EXPECT_EQ(Series("mori_io_completed_bytes_total", flow), 123);
}

int GpuCount() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n;
}

// A real GPU is labelled by its physical index from sysfs, not by its HIP ordinal.
std::string ExpectedGpuLabel(int hipDevice) {
  char bus[64] = {0};
  EXPECT_EQ(hipDeviceGetPCIBusId(bus, sizeof(bus), hipDevice), hipSuccess);
  return mori::metrics::GpuLabel(mori::metrics::PhysicalGpuIndexForPci(bus), bus);
}

TEST(IoMetricsXgmi, FlowIsBetweenPhysicalGpusAndSettlesWhenObserved) {
  if (GpuCount() < 2) GTEST_SKIP() << "needs two GPUs";
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine a(UniqueKey("xgmi-a"), cfg);
  IOEngine b(UniqueKey("xgmi-b"), cfg);
  XgmiBackendConfig xgmiCfg{};
  a.CreateBackend(BackendType::XGMI, xgmiCfg);
  b.CreateBackend(BackendType::XGMI, xgmiCfg);
  a.RegisterRemoteEngine(b.GetEngineDesc());
  b.RegisterRemoteEngine(a.GetEngineDesc());

  constexpr size_t kSize = 1 << 20;
  void* src = nullptr;
  void* dst = nullptr;
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMalloc(&src, kSize), hipSuccess);
  ASSERT_EQ(hipMemset(src, 0xAB, kSize), hipSuccess);
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  ASSERT_EQ(hipMalloc(&dst, kSize), hipSuccess);
  ASSERT_EQ(hipMemset(dst, 0, kSize), hipSuccess);
  MemoryDesc srcDesc = a.RegisterMemory(src, kSize, 0, MemoryLocationType::GPU);
  MemoryDesc dstDesc = b.RegisterMemory(dst, kSize, 1, MemoryLocationType::GPU);

  const std::string gpu0 = ExpectedGpuLabel(0);
  const std::string gpu1 = ExpectedGpuLabel(1);
  ASSERT_NE(gpu0, gpu1);

  const Labels flow = {{"backend", "xgmi"}, {"op", "write"},           {"src_host", "host-self"},
                       {"src_gpu", gpu0},   {"dst_host", "host-self"}, {"dst_gpu", gpu1}};
  const Labels inflight = {{"backend", "xgmi"}, {"gpu", gpu0}};
  const double submittedBefore = Series("mori_io_submitted_bytes_total", flow);
  const double completedBefore = Series("mori_io_completed_bytes_total", flow);
  const double inflightBefore = Series("mori_io_inflight", inflight);

  {
    TransferStatus st;
    a.Write(srcDesc, 0, dstDesc, 0, kSize, &st, a.AllocateTransferUniqueId());
    // Give the copy all the time it needs; nobody has asked the status yet.
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(Series("mori_io_submitted_bytes_total", flow) - submittedBefore, kSize);
    EXPECT_EQ(Series("mori_io_completed_bytes_total", flow) - completedBefore, 0);
    EXPECT_EQ(Series("mori_io_inflight", inflight) - inflightBefore, 1);

    ASSERT_TRUE(WaitDone(&st));  // looking at it is what settles it
    ASSERT_TRUE(st.Succeeded()) << st.Message();
    EXPECT_EQ(Series("mori_io_completed_bytes_total", flow) - completedBefore, kSize);
    EXPECT_EQ(Series("mori_io_inflight", inflight) - inflightBefore, 0);
  }

  // A caller that never looks: ends as `unobserved` when the status goes away.
  Labels unobserved = flow;
  unobserved["status"] = "unobserved";
  const double unobservedBefore = Series("mori_io_completed_total", unobserved);
  {
    TransferStatus st;
    a.Write(srcDesc, 0, dstDesc, 0, kSize, &st, a.AllocateTransferUniqueId());
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  }
  EXPECT_EQ(Series("mori_io_submitted_bytes_total", flow) - submittedBefore, 2 * kSize);
  EXPECT_EQ(Series("mori_io_completed_bytes_total", flow) - completedBefore, kSize);
  EXPECT_EQ(Series("mori_io_completed_total", unobserved) - unobservedBefore, 1);
  EXPECT_EQ(Series("mori_io_inflight", inflight) - inflightBefore, 0);

  std::vector<uint8_t> host(kSize);
  ASSERT_EQ(hipMemcpy(host.data(), dst, kSize, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(host.front(), 0xAB);
  EXPECT_EQ(host.back(), 0xAB);
  a.DeregisterMemory(srcDesc);
  b.DeregisterMemory(dstDesc);
  hipFree(dst);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  hipFree(src);
}

struct RdmaPair {
  std::unique_ptr<IOEngine> a;
  std::unique_ptr<IOEngine> b;
  std::vector<char> bufA = std::vector<char>(1 << 20, 'a');
  std::vector<char> bufB = std::vector<char>(1 << 20, 'b');
  MemoryDesc memA;
  MemoryDesc memB;

  RdmaPair() {
    IOEngineConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 0;
    a = std::make_unique<IOEngine>(UniqueKey("rdma-a"), cfg);
    b = std::make_unique<IOEngine>(UniqueKey("rdma-b"), cfg);
    RdmaBackendConfig rdmaCfg{};
    rdmaCfg.enableNotification = false;
    a->CreateBackend(BackendType::RDMA, rdmaCfg);
    b->CreateBackend(BackendType::RDMA, rdmaCfg);
    a->RegisterRemoteEngine(b->GetEngineDesc());
    b->RegisterRemoteEngine(a->GetEngineDesc());
    memA = a->RegisterMemory(bufA.data(), bufA.size(), -1, MemoryLocationType::CPU);
    memB = b->RegisterMemory(bufB.data(), bufB.size(), -1, MemoryLocationType::CPU);
  }
};

// Every (src_ibdev, dst_ibdev) the snapshot has for `op`, with its bytes.
std::map<std::pair<std::string, std::string>, double> NicBytes(const std::string& op) {
  std::map<std::pair<std::string, std::string>, double> out;
  for (const auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name != "mori_io_nic_bytes_total") continue;
    Labels have(s.labels.begin(), s.labels.end());
    if (have["op"] != op) continue;
    out[{have["src_ibdev"], have["dst_ibdev"]}] += s.value;
  }
  return out;
}

double Total(const std::map<std::pair<std::string, std::string>, double>& m) {
  double t = 0;
  for (const auto& [k, v] : m) t += v;
  return t;
}

TEST(IoMetricsRdma, DataBytesAreAccountedToTheNicPairThatCarriedThem) {
  if (!RdmaBackend::HasActiveDevices()) GTEST_SKIP() << "needs an active RDMA device";
  RdmaPair pair;

  const auto writeBefore = NicBytes("write");
  const auto readBefore = NicBytes("read");

  TransferStatus w, r;
  pair.a->Write(pair.memA, 0, pair.memB, 0, 200000, &w, pair.a->AllocateTransferUniqueId());
  ASSERT_TRUE(WaitDone(&w));
  ASSERT_TRUE(w.Succeeded()) << w.Message();
  // Read a part of the target the write did not reach.
  pair.a->Read(pair.memA, 0, pair.memB, 500000, 50000, &r, pair.a->AllocateTransferUniqueId());
  ASSERT_TRUE(WaitDone(&r));
  ASSERT_TRUE(r.Succeeded()) << r.Message();
  EXPECT_EQ(pair.bufB[0], 'a');
  EXPECT_EQ(pair.bufB[199999], 'a');
  EXPECT_EQ(pair.bufB[200000], 'b');
  EXPECT_EQ(pair.bufA[0], 'b');
  EXPECT_EQ(pair.bufA[49999], 'b');
  EXPECT_EQ(pair.bufA[50000], 'a');

  const auto writeAfter = NicBytes("write");
  const auto readAfter = NicBytes("read");
  EXPECT_EQ(Total(writeAfter) - Total(writeBefore), 200000);
  EXPECT_EQ(Total(readAfter) - Total(readBefore), 50000);

  // Each end is named by a real device and the network interface sysfs gives it.
  for (const auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name != "mori_io_nic_bytes_total" || s.value == 0) continue;
    Labels have(s.labels.begin(), s.labels.end());
    for (const char* end : {"src", "dst"}) {
      const std::string ibdev = have[std::string(end) + "_ibdev"];
      EXPECT_EQ(access(("/sys/class/infiniband/" + ibdev).c_str(), F_OK), 0) << ibdev;
      EXPECT_EQ(have[std::string(end) + "_netdev"], mori::metrics::NetdevForIbdev(ibdev));
      EXPECT_EQ(have[std::string(end) + "_host"], "host-self");
    }
  }

  const Labels flow = {{"backend", "rdma"}, {"op", "write"},           {"src_gpu", "cpu"},
                       {"dst_gpu", "cpu"},  {"src_host", "host-self"}, {"dst_host", "host-self"}};
  EXPECT_GE(Series("mori_io_completed_bytes_total", flow), 200000);
}

}  // namespace
}  // namespace mori::io

/* ------------------------------------------------------------------------ */
/* 2. RDMA issued by the GPU itself (shmem, cco GDA)                        */
/* ------------------------------------------------------------------------ */

// The device side is stood in for by writing the block as device code would leave it.

namespace mori::application {
namespace {

using mori::metrics::PeIdentity;

using Labels = std::map<std::string, std::string>;

bool HaveGpu() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

// Two nodes of two GPUs. Rank 1 (host-self, GPU 1, ionic_1) is the one under test.
std::vector<PeIdentity> World() {
  std::vector<PeIdentity> w(4);
  w[0] = {"host-self", 0, "0000:05:00.0", "ionic_0", "benic1p1"};
  w[1] = {"host-self", 1, "0000:15:00.0", "ionic_1", "benic2p1"};
  w[2] = {"node-b", 0, "0000:05:00.0", "ionic_0", "benic1p1"};
  // An index that could not be determined: labelled by PCI address.
  w[3] = {"node-b", -1, "0000:15:00.0", "ionic_1", "benic2p1"};
  return w;
}

std::vector<mori::metrics::Registry::Series> Named(const std::string& name) {
  std::vector<mori::metrics::Registry::Series> out;
  for (auto& s : mori::metrics::Registry::Instance().Snapshot()) {
    if (s.name == name) out.push_back(s);
  }
  return out;
}

Labels ToMap(const mori::metrics::MetricLabels& l) { return Labels(l.begin(), l.end()); }

TEST(RdmaTrafficCounters, BlockIsMappedAndSizedByTheWorld) {
  if (!HaveGpu()) GTEST_SKIP() << "needs a GPU";
  ASSERT_TRUE(mori::metrics::TrafficEnabled());
  RdmaTrafficCounters counters("test-size", 1, World());
  ASSERT_NE(counters.DevicePtr(), nullptr);
  EXPECT_EQ(counters.Block().Read().size(), RdmaTrafficSlots(4));
  // The slots of different ops never overlap, whatever the peer.
  EXPECT_EQ(RdmaTrafficSlot(kRdmaTrafficPut, 4, 3) + 1, RdmaTrafficSlot(kRdmaTrafficInline, 4, 0));
  EXPECT_EQ(RdmaTrafficSlot(kRdmaTrafficGet, 4, 3) + 1, RdmaTrafficSlots(4));
}

// A put sends this rank's bytes to the peer; a get brings the peer's bytes here.
TEST(RdmaTrafficCounters, FlowsAreLabelledWithBothEnds) {
  if (!HaveGpu()) GTEST_SKIP() << "needs a GPU";
  RdmaTrafficCounters counters("test-flow", 1, World());
  ASSERT_NE(counters.DevicePtr(), nullptr);
  std::vector<uint64_t> block(RdmaTrafficSlots(4), 0);
  block[RdmaTrafficSlot(kRdmaTrafficPut, 4, 2)] = 1000;
  block[RdmaTrafficSlot(kRdmaTrafficAtomic, 4, 2)] = 16;
  block[RdmaTrafficSlot(kRdmaTrafficGet, 4, 3)] = 300;
  counters.Block().Write(block);

  std::map<std::string, Labels> byOp;
  std::map<std::string, double> value;
  for (const auto& s : Named("mori_gpu_rdma_bytes_total")) {
    Labels l = ToMap(s.labels);
    if (l["api"] != "test-flow") continue;
    byOp[l["op"]] = l;
    value[l["op"]] = s.value;
  }
  // Pairs that carried nothing are not published.
  ASSERT_EQ(byOp.size(), 3u);

  const Labels put = {
      {"host", "host-self"},      {"api", "test-flow"},      {"op", "put"},
      {"src_host", "host-self"},  {"src_gpu", "1"},          {"src_ibdev", "ionic_1"},
      {"src_netdev", "benic2p1"}, {"dst_host", "node-b"},    {"dst_gpu", "0"},
      {"dst_ibdev", "ionic_0"},   {"dst_netdev", "benic1p1"}};
  EXPECT_EQ(byOp["put"], put);
  EXPECT_EQ(value["put"], 1000);
  EXPECT_EQ(byOp["atomic"]["dst_host"], "node-b");
  EXPECT_EQ(value["atomic"], 16);

  const Labels get = {{"host", "host-self"},
                      {"api", "test-flow"},
                      {"op", "get"},
                      {"src_host", "node-b"},
                      {"src_gpu", "pci:0000:15:00.0"},
                      {"src_ibdev", "ionic_1"},
                      {"src_netdev", "benic2p1"},
                      {"dst_host", "host-self"},
                      {"dst_gpu", "1"},
                      {"dst_ibdev", "ionic_1"},
                      {"dst_netdev", "benic2p1"}};
  EXPECT_EQ(byOp["get"], get);
  EXPECT_EQ(value["get"], 300);
  EXPECT_EQ(mori::metrics::Registry::Instance().RejectedSamples(), 0u);
}

TEST(RdmaTrafficCounters, NicInfoNamesTheRanksNicAndGoesAwayWithIt) {
  if (!HaveGpu()) GTEST_SKIP() << "needs a GPU";
  {
    RdmaTrafficCounters counters("test-info", 1, World());
    bool found = false;
    for (const auto& s : Named("mori_gpu_rdma_nic_info")) {
      Labels l = ToMap(s.labels);
      if (l["gpu"] != "1") continue;
      found = true;
      EXPECT_EQ(l["host"], "host-self");
      EXPECT_EQ(l["pci"], "0000:15:00.0");
      EXPECT_EQ(l["ibdev"], "ionic_1");
      EXPECT_EQ(l["netdev"], "benic2p1");
      EXPECT_EQ(s.value, 1);
    }
    EXPECT_TRUE(found);
  }
  EXPECT_TRUE(Named("mori_gpu_rdma_nic_info").empty());
}

TEST(RdmaTrafficCounters, NoNicNoInfo) {
  if (!HaveGpu()) GTEST_SKIP() << "needs a GPU";
  std::vector<PeIdentity> world = World();
  world[1].ibdev.clear();
  world[1].netdev.clear();
  RdmaTrafficCounters counters("test-nonic", 1, world);
  EXPECT_TRUE(Named("mori_gpu_rdma_nic_info").empty());
}

}  // namespace
}  // namespace mori::application
