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

// The common layer of the metrics module (mori/metrics/metrics.hpp, sections 0-2),
// which needs no hardware.
// The cases that change the environment put it back, and the fork case, which
// creates the process's Registry::Instance() from the environment, is last.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "mori/metrics/metrics.hpp"
#include "mori/metrics/prometheus_metrics_server.hpp"

namespace mori::metrics {
namespace {

/* ------------------------------------------------------------------------ */
/* 1. Identities                                                            */
/* ------------------------------------------------------------------------ */

// A sysfs tree on disk: drm/cardN/device is a symlink to the PCI device directory.
class FakeSysfs : public ::testing::Test {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/mori_identity_XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    root_ = tmpl;
    Mkdir(root_ + "/devices");
    Mkdir(root_ + "/drm");
    Mkdir(root_ + "/infiniband");
  }
  void TearDown() override { ASSERT_EQ(std::system(("rm -rf " + root_).c_str()), 0); }

  static void Mkdir(const std::string& p) { ASSERT_EQ(mkdir(p.c_str(), 0755), 0) << p; }
  static void Touch(const std::string& p) { std::ofstream(p) << "1\n"; }

  void AddCard(int n, const std::string& bdf, bool xgmi = true) {
    const std::string dev = root_ + "/devices/" + bdf;
    if (access(dev.c_str(), F_OK) != 0) Mkdir(dev);
    if (xgmi) Touch(dev + "/xgmi_device_id");
    const std::string card = root_ + "/drm/card" + std::to_string(n);
    Mkdir(card);
    ASSERT_EQ(symlink(("../../devices/" + bdf).c_str(), (card + "/device").c_str()), 0);
  }

  std::string Drm() const { return root_ + "/drm"; }
  std::string Ib() const { return root_ + "/infiniband"; }
  std::string root_;
};

TEST(NormalizePciBdf, AddsTheDomainAndLowersCase) {
  EXPECT_EQ(NormalizePciBdf("0000:05:00.0"), "0000:05:00.0");
  EXPECT_EQ(NormalizePciBdf("05:00.0"), "0000:05:00.0");
  EXPECT_EQ(NormalizePciBdf("0000:C5:00.0"), "0000:c5:00.0");
  EXPECT_EQ(NormalizePciBdf("0001:c5:00.0"), "0001:c5:00.0");
}

TEST_F(FakeSysfs, IndexFollowsCardNumberOrder) {
  AddCard(10, "0000:05:00.0");
  AddCard(2, "0000:c5:00.0");
  AddCard(1, "0000:85:00.0");
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:85:00.0", Drm()), 0);
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:c5:00.0", Drm()), 1);
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:05:00.0", Drm()), 2);  // card10 sorts after card2
}

// What hipDeviceGetPCIBusId may hand back: no domain, upper case.
TEST_F(FakeSysfs, AddressIsMatchedAfterNormalizing) {
  AddCard(0, "0000:05:00.0");
  AddCard(1, "0000:c5:00.0");
  EXPECT_EQ(PhysicalGpuIndexForPci("C5:00.0", Drm()), 1);
}

// A BMC's VGA card in front of the GPUs must not shift their numbers.
TEST_F(FakeSysfs, CardsWithoutXgmiAreNotCounted) {
  AddCard(0, "0000:03:00.0", /*xgmi=*/false);
  AddCard(1, "0000:05:00.0");
  AddCard(2, "0000:c5:00.0");
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:05:00.0", Drm()), 0);
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:c5:00.0", Drm()), 1);
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:03:00.0", Drm()), -1);
}

TEST_F(FakeSysfs, ConnectorsAreNotCards) {
  AddCard(1, "0000:05:00.0");
  Mkdir(Drm() + "/card1-DP-1");
  Mkdir(Drm() + "/renderD128");
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:05:00.0", Drm()), 0);
}

TEST_F(FakeSysfs, UnknownAddressAndMissingSysfsGiveMinusOne) {
  AddCard(0, "0000:05:00.0");
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:99:00.0", Drm()), -1);
  EXPECT_EQ(PhysicalGpuIndexForPci("", Drm()), -1);
  EXPECT_EQ(PhysicalGpuIndexForPci("0000:05:00.0", root_ + "/not-there"), -1);
}

TEST(GpuLabel, IndexThenAddressThenUnknown) {
  EXPECT_EQ(GpuLabel(3, "0000:05:00.0"), "3");
  EXPECT_EQ(GpuLabel(0, ""), "0");
  EXPECT_EQ(GpuLabel(-1, "C5:00.0"), "pci:0000:c5:00.0");
  EXPECT_EQ(GpuLabel(-1, ""), "unknown");
}

TEST_F(FakeSysfs, NetdevOfAnRdmaDevice) {
  Mkdir(Ib() + "/ionic_3");
  Mkdir(Ib() + "/ionic_3/device");
  Mkdir(Ib() + "/ionic_3/device/net");
  Mkdir(Ib() + "/ionic_3/device/net/ens3f0np0");
  EXPECT_EQ(NetdevForIbdev("ionic_3", Ib()), "ens3f0np0");
  EXPECT_EQ(NetdevForIbdev("ionic_9", Ib()), "");
  EXPECT_EQ(NetdevForIbdev("", Ib()), "");
}

TEST_F(FakeSysfs, NetdevIsTheFirstByNameWhenThereAreSeveral) {
  Mkdir(Ib() + "/mlx5_0");
  Mkdir(Ib() + "/mlx5_0/device");
  Mkdir(Ib() + "/mlx5_0/device/net");
  Mkdir(Ib() + "/mlx5_0/device/net/eth9");
  Mkdir(Ib() + "/mlx5_0/device/net/eth10");
  EXPECT_EQ(NetdevForIbdev("mlx5_0", Ib()), "eth10");
}

// In bytes per second; 0 when the system does not say.
TEST_F(FakeSysfs, LinkSpeedOfAnRdmaDevice) {
  const std::string ib = root_ + "/ib";
  auto rate = [&](const std::string& dev, const std::string& port, const std::string& text) {
    ASSERT_EQ(std::system(("mkdir -p " + ib + "/" + dev + "/ports/" + port).c_str()), 0);
    std::ofstream(ib + "/" + dev + "/ports/" + port + "/rate") << text << "\n";
  };
  rate("mlx5_4", "1", "400 Gb/sec (4X NDR)");
  rate("ionic_0", "1", "100 Gb/sec (4X EDR)");
  rate("ionic_0", "2", "25 Gb/sec (1X EDR)");
  rate("old_0", "1", "2.5 Gb/sec (1X SDR)");
  rate("odd_0", "1", "unknown");
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("mlx5_4", ib), 50e9);
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("ionic_0", ib), 12.5e9);  // the faster port
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("old_0", ib), 2.5e9 / 8);
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("odd_0", ib), 0);
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("absent_0", ib), 0);
  EXPECT_DOUBLE_EQ(IbdevLinkBytesPerSecond("", ib), 0);
}

TEST(HostId, EnvOverridesGethostname) {
  unsetenv("MORI_METRICS_HOST_ID");
  char buf[256] = {0};
  ASSERT_EQ(gethostname(buf, sizeof(buf) - 1), 0);
  EXPECT_EQ(HostId(), std::string(buf));  // as it is, not cut at the first dot
  setenv("MORI_METRICS_HOST_ID", "node-a", 1);
  EXPECT_EQ(HostId(), "node-a");
  unsetenv("MORI_METRICS_HOST_ID");
}

TEST(EnabledFromEnv, PortOrExplicitSwitch) {
  unsetenv("MORI_METRICS_ENABLE");
  unsetenv("MORI_METRICS_PORT");
  EXPECT_FALSE(EnabledFromEnv());
  setenv("MORI_METRICS_PORT", "9400", 1);
  EXPECT_TRUE(EnabledFromEnv());
  setenv("MORI_METRICS_ENABLE", "0", 1);  // the explicit switch wins
  EXPECT_FALSE(EnabledFromEnv());
  unsetenv("MORI_METRICS_PORT");
  setenv("MORI_METRICS_ENABLE", "1", 1);  // snapshot-only use: no port
  EXPECT_TRUE(EnabledFromEnv());
  unsetenv("MORI_METRICS_ENABLE");
  setenv("MORI_METRICS_PUSH_URL", "http://127.0.0.1:8428", 1);  // push-only use: no port either
  EXPECT_TRUE(EnabledFromEnv());
  unsetenv("MORI_METRICS_PUSH_URL");
  EXPECT_FALSE(EnabledFromEnv());
}

/* ------------------------------------------------------------------------ */
/* 2. MetricPublisher                                                       */
/* ------------------------------------------------------------------------ */

constexpr const char* kName = "test_total";
constexpr const char* kHelp = "help";
const std::vector<double> kBounds = {0.001, 0.01, 0.1};

struct FakeSource : MetricSource {
  std::vector<MetricSample> samples;
  std::vector<MetricSample> SampleMetrics() const override { return samples; }
};

struct Recorder {
  struct Hist {
    std::string name;
    MetricLabels labels;
    std::vector<uint64_t> buckets;
    uint64_t count;
    double sum;
  };
  std::vector<std::tuple<std::string, MetricLabels, double>> counters;
  std::vector<std::tuple<std::string, MetricLabels, double>> gauges;
  std::vector<Hist> hists;

  MetricPublisher::Sink Sink() {
    MetricPublisher::Sink s;
    s.counter = [this](const char* n, const char*, const MetricLabels& l, double d) {
      counters.emplace_back(n, l, d);
    };
    s.gauge = [this](const char* n, const char*, const MetricLabels& l, double v) {
      gauges.emplace_back(n, l, v);
    };
    s.histogram = [this](const char* n, const char*, const MetricLabels& l,
                         const std::vector<double>&, const std::vector<uint64_t>& b, uint64_t c,
                         double sum) { hists.push_back({n, l, b, c, sum}); };
    return s;
  }
};

MetricSample Histogram(std::vector<uint64_t> buckets, uint64_t count, uint64_t sum_ns) {
  MetricSample s;
  s.name = "test_seconds";
  s.help = kHelp;
  s.kind = MetricKind::kHistogram;
  s.scale = 1e-9;
  s.bounds = &kBounds;
  s.buckets = std::move(buckets);
  s.count = count;
  s.value = sum_ns;
  return s;
}

TEST(HasDuplicateLabelNames, FindsARepeatedName) {
  EXPECT_FALSE(HasDuplicateLabelNames({}));
  EXPECT_FALSE(HasDuplicateLabelNames({{"host", "a"}, {"gpu", "0"}}));
  EXPECT_TRUE(HasDuplicateLabelNames({{"gpu", "0"}, {"op", "read"}, {"gpu", "1"}}));
  EXPECT_TRUE(HasDuplicateLabelNames({{"gpu", "0"}, {"gpu", "0"}}));
}

TEST(MetricPublisher, CounterShipsDeltasAndStartsAtZero) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {}, 0}};
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.counters.size(), 1u);  // the series exists before its first event
  EXPECT_EQ(std::get<2>(rec.counters[0]), 0.0);

  src.samples[0].value = 10;
  pub.Publish("s", {}, src, rec.Sink());
  src.samples[0].value = 10;  // nothing new: nothing shipped
  pub.Publish("s", {}, src, rec.Sink());
  src.samples[0].value = 25;
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.counters.size(), 3u);
  EXPECT_EQ(std::get<2>(rec.counters[1]), 10.0);
  EXPECT_EQ(std::get<2>(rec.counters[2]), 15.0);
}

TEST(MetricPublisher, SourceLabelsComeFirst) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {{"op", "read"}}, 1}};
  pub.Publish("s", {{"host", "h"}, {"gpu", "3"}}, src, rec.Sink());
  ASSERT_EQ(rec.counters.size(), 1u);
  MetricLabels want = {{"host", "h"}, {"gpu", "3"}, {"op", "read"}};
  EXPECT_EQ(std::get<1>(rec.counters[0]), want);
}

TEST(MetricPublisher, DuplicateLabelNameDropsTheSampleOnly) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  src.samples = {MetricSample{kName, kHelp, {{"gpu", "1"}}, 5},
                 MetricSample{"other_total", kHelp, {{"op", "read"}}, 7},
                 MetricSample{"dup_gauge", kHelp, {{"gpu", "1"}}, 2, MetricKind::kGauge}};
  pub.Publish("s", {{"gpu", "0"}}, src, rec.Sink());
  ASSERT_EQ(rec.counters.size(), 1u);
  EXPECT_EQ(std::get<0>(rec.counters[0]), "other_total");
  EXPECT_TRUE(rec.gauges.empty());
  EXPECT_EQ(pub.RejectedSamples(), 2u);
}

TEST(MetricPublisher, HistogramShipsBucketDeltas) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  src.samples = {Histogram({0, 0, 0}, 0, 0)};
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.hists.size(), 1u);  // exists at zero
  EXPECT_EQ(rec.hists[0].count, 0u);
  EXPECT_EQ(rec.hists[0].buckets, (std::vector<uint64_t>{0, 0, 0}));

  // 2 observations <= 1 ms, 1 more <= 10 ms, 1 above every bound; 2.5 s total.
  src.samples = {Histogram({2, 3, 3}, 4, 2'500'000'000ULL)};
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.hists.size(), 2u);
  EXPECT_EQ(rec.hists[1].buckets, (std::vector<uint64_t>{2, 3, 3}));
  EXPECT_EQ(rec.hists[1].count, 4u);
  EXPECT_DOUBLE_EQ(rec.hists[1].sum, 2.5);

  pub.Publish("s", {}, src, rec.Sink());  // nothing new
  EXPECT_EQ(rec.hists.size(), 2u);

  src.samples = {Histogram({2, 4, 5}, 6, 2'600'000'000ULL)};
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.hists.size(), 3u);
  EXPECT_EQ(rec.hists[2].buckets, (std::vector<uint64_t>{0, 1, 2}));
  EXPECT_EQ(rec.hists[2].count, 2u);
  EXPECT_NEAR(rec.hists[2].sum, 0.1, 1e-9);
}

TEST(MetricPublisher, HistogramRebasesWhenTheSourceWasRebuilt) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  src.samples = {Histogram({5, 5, 5}, 5, 1000)};
  pub.Publish("s", {}, src, rec.Sink());
  src.samples = {Histogram({1, 1, 1}, 1, 10)};  // went backwards
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.hists.size(), 1u);  // no negative delta, no double count
  src.samples = {Histogram({2, 2, 2}, 2, 20)};
  pub.Publish("s", {}, src, rec.Sink());
  ASSERT_EQ(rec.hists.size(), 2u);
  EXPECT_EQ(rec.hists[1].count, 1u);
}

TEST(MetricPublisher, HistogramWithoutBoundsIsSkipped) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource src;
  MetricSample s = Histogram({1, 1}, 1, 1);  // two buckets, three bounds
  src.samples = {s};
  pub.Publish("s", {}, src, rec.Sink());
  EXPECT_TRUE(rec.hists.empty());
}

TEST(MetricPublisher, ForgetDropsOneSourcesBaselines) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource a, b;
  a.samples = {MetricSample{kName, kHelp, {}, 10}};
  b.samples = {MetricSample{kName, kHelp, {}, 100}};
  pub.Publish("a", {}, a, rec.Sink());
  pub.Publish("b", {}, b, rec.Sink());
  rec.counters.clear();

  pub.Forget("a");
  pub.Publish("a", {}, a, rec.Sink());  // seen as new: ships its full reading
  pub.Publish("b", {}, b, rec.Sink());  // untouched: nothing new
  ASSERT_EQ(rec.counters.size(), 1u);
  EXPECT_EQ(std::get<2>(rec.counters[0]), 10.0);
}

TEST(MetricPublisher, ForgetMatchesTheWholeSourceId) {
  MetricPublisher pub;
  Recorder rec;
  FakeSource a;
  a.samples = {MetricSample{kName, kHelp, {}, 10}};
  pub.Publish("ab", {}, a, rec.Sink());
  rec.counters.clear();
  pub.Forget("a");
  pub.Publish("ab", {}, a, rec.Sink());
  EXPECT_TRUE(rec.counters.empty());
}

/* ------------------------------------------------------------------------ */
/* 3. Registry                                                              */
/* ------------------------------------------------------------------------ */

const std::vector<double> kRegistryBounds = {0.001, 0.01};

// One request, e.g. "GET /metrics": the whole response, headers included.
std::string HttpRequest(int port, const std::string& what) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return "";
  }
  const std::string request = what + " HTTP/1.0\r\n\r\n";
  (void)!::write(fd, request.data(), request.size());
  std::string response;
  char buf[4096];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) response.append(buf, static_cast<std::size_t>(n));
  ::close(fd);
  return response;
}

std::string HttpGetBody(int port) {
  const std::string response = HttpRequest(port, "GET /metrics");
  auto pos = response.find("\r\n\r\n");
  return pos == std::string::npos ? response : response.substr(pos + 4);
}

struct Component : MetricSource {
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> inflight{0};
  std::atomic<uint64_t> lat_buckets[2] = {};
  std::atomic<uint64_t> lat_count{0};
  std::atomic<uint64_t> lat_sum_ns{0};
  MetricLabels extra;  // labels on the samples themselves
  // `bytes` as it was the last time this was sampled, by whoever sampled it.
  mutable std::atomic<uint64_t> sampled{0};

  std::vector<MetricSample> SampleMetrics() const override {
    std::vector<MetricSample> out;
    sampled.store(bytes.load());
    out.push_back(MetricSample{"t_bytes_total", "bytes", extra, bytes.load()});
    out.push_back(
        MetricSample{"t_inflight", "inflight", extra, inflight.load(), MetricKind::kGauge});
    MetricSample h;
    h.name = "t_latency_seconds";
    h.help = "latency";
    h.labels = extra;
    h.kind = MetricKind::kHistogram;
    h.scale = 1e-9;
    h.bounds = &kRegistryBounds;
    h.buckets = {lat_buckets[0].load(), lat_buckets[1].load()};
    h.count = lat_count.load();
    h.value = lat_sum_ns.load();
    out.push_back(std::move(h));
    return out;
  }
};

Registry::Options Quiet() {
  Registry::Options o;
  o.host = "node-a";
  o.interval_ms = 0;  // tests tick by hand
  // The registry's own sampling series are looked at in the Sampling cases only.
  o.sampling_series = false;
  return o;
}

const Registry::Series* Find(const std::vector<Registry::Series>& all, const std::string& name,
                             const MetricLabels& labels) {
  for (const auto& s : all) {
    if (s.name == name && s.labels == labels) return &s;
  }
  return nullptr;
}

TEST(Registry, AddsHostAndNothingElse) {
  Registry reg(Quiet());
  Component c;
  c.bytes = 7;
  uint64_t h = reg.Register("io", {{"gpu", "3"}}, &c);
  ASSERT_NE(h, 0u);
  auto snap = reg.Snapshot();
  const auto* s = Find(snap, "t_bytes_total", {{"host", "node-a"}, {"gpu", "3"}});
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(s->value, 7.0);
  EXPECT_EQ(s->kind, MetricKind::kCounter);
  reg.Unregister(h);
}

TEST(Registry, SnapshotHoldsCumulativeCountersGaugesAndHistograms) {
  Registry reg(Quiet());
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  c.bytes = 100;
  c.inflight = 2;
  c.lat_buckets[0] = 1;
  c.lat_buckets[1] = 3;
  c.lat_count = 4;
  c.lat_sum_ns = 30'000'000;
  reg.Tick();
  c.bytes = 250;
  c.inflight = 0;
  c.lat_buckets[1] = 4;
  c.lat_count = 5;
  c.lat_sum_ns = 35'000'000;
  auto snap = reg.Snapshot();

  const MetricLabels host = {{"host", "node-a"}};
  const auto* bytes = Find(snap, "t_bytes_total", host);
  const auto* inflight = Find(snap, "t_inflight", host);
  const auto* lat = Find(snap, "t_latency_seconds", host);
  ASSERT_NE(bytes, nullptr);
  ASSERT_NE(inflight, nullptr);
  ASSERT_NE(lat, nullptr);
  EXPECT_EQ(bytes->value, 250.0);
  EXPECT_EQ(inflight->value, 0.0);
  EXPECT_EQ(lat->kind, MetricKind::kHistogram);
  EXPECT_EQ(lat->bounds, kRegistryBounds);
  EXPECT_EQ(lat->buckets, (std::vector<uint64_t>{1, 4}));
  EXPECT_EQ(lat->count, 5u);
  EXPECT_NEAR(lat->value, 0.035, 1e-12);
  reg.Unregister(h);
}

TEST(Registry, SourcesSharingASeriesAddUp) {
  Registry reg(Quiet());
  Component a, b;
  uint64_t ha = reg.Register("op", {{"gpu", "0"}}, &a);
  uint64_t hb = reg.Register("op", {{"gpu", "0"}}, &b);  // same name on purpose
  a.bytes = 10;
  b.bytes = 5;
  auto snap = reg.Snapshot();
  const MetricLabels labels = {{"host", "node-a"}, {"gpu", "0"}};
  ASSERT_NE(Find(snap, "t_bytes_total", labels), nullptr);
  EXPECT_EQ(Find(snap, "t_bytes_total", labels)->value, 15.0);

  // One goes away: the series stays, with what both had added.
  reg.Unregister(ha);
  b.bytes = 6;
  snap = reg.Snapshot();
  ASSERT_NE(Find(snap, "t_bytes_total", labels), nullptr);
  EXPECT_EQ(Find(snap, "t_bytes_total", labels)->value, 16.0);

  reg.Unregister(hb);
  EXPECT_TRUE(reg.Snapshot().empty());
}

TEST(Registry, UnregisterPublishesTheLastIncrementThenRemovesTheSeries) {
  Registry::Options o = Quiet();
  o.port = 0;
  Registry reg(o);
  Component keep, gone;
  uint64_t hk = reg.Register("keep", {{"gpu", "0"}}, &keep);
  uint64_t hg = reg.Register("gone", {{"gpu", "1"}}, &gone);
  gone.bytes = 9;  // never ticked
  reg.Unregister(hg);
  auto snap = reg.Snapshot();
  EXPECT_EQ(Find(snap, "t_bytes_total", {{"host", "node-a"}, {"gpu", "1"}}), nullptr);
  EXPECT_NE(Find(snap, "t_bytes_total", {{"host", "node-a"}, {"gpu", "0"}}), nullptr);
  reg.Unregister(hk);
  reg.Unregister(hk);  // twice is harmless
  reg.Unregister(12345);
}

TEST(Registry, DuplicateLabelIsRefusedNotServed) {
  Registry reg(Quiet());
  Component c;
  c.extra = {{"gpu", "1"}};
  uint64_t h = reg.Register("io", {{"gpu", "0"}}, &c);
  EXPECT_TRUE(reg.Snapshot().empty());
  EXPECT_GE(reg.RejectedSamples(), 3u);
  reg.Unregister(h);

  Component host_clash;
  host_clash.extra = {{"host", "other"}};
  h = reg.Register("io", {}, &host_clash);
  EXPECT_TRUE(reg.Snapshot().empty());
  reg.Unregister(h);
}

TEST(Registry, ServesOnThePortAndRemovesSeriesFromIt) {
  MetricsServer probe(0, "127.0.0.1");  // a port known to be free right after
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  o.port = probe.port() + 1;
  Registry reg(o);
  EXPECT_EQ(reg.port(), -1);  // nothing is served before the first source
  Component c;
  uint64_t h = reg.Register("io", {{"gpu", "2"}}, &c);
  ASSERT_GT(reg.port(), 0);
  c.bytes = 42;
  reg.Tick();
  std::string body = HttpGetBody(reg.port());
  EXPECT_NE(body.find("t_bytes_total{host=\"node-a\",gpu=\"2\"} 42"), std::string::npos) << body;
  EXPECT_NE(body.find("t_latency_seconds_count{host=\"node-a\",gpu=\"2\"} 0"), std::string::npos)
      << body;
  reg.Unregister(h);
  body = HttpGetBody(reg.port());
  EXPECT_EQ(body.find("t_bytes_total"), std::string::npos) << body;
}

TEST(Registry, AScrapeSamplesTheSourcesItself) {
  MetricsServer probe(0, "127.0.0.1");
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  o.port = probe.port() + 1;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  ASSERT_GT(reg.port(), 0);
  c.bytes = 41;
  EXPECT_NE(HttpGetBody(reg.port()).find("t_bytes_total{host=\"node-a\"} 41"), std::string::npos);
  c.bytes = 97;
  EXPECT_NE(HttpGetBody(reg.port()).find("t_bytes_total{host=\"node-a\"} 97"), std::string::npos);
  reg.Unregister(h);
}

TEST(Registry, TakenPortMovesToTheNextOneAndWritesTheTargetFile) {
  char tmpl[] = "/tmp/mori_registry_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string dir = tmpl;

  MetricsServer taken(0, "127.0.0.1");
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  o.port = taken.port();
  o.port_dir = dir;
  o.target_labels = {{"local_rank", "3"}};
  // Host and pid: several nodes may share the directory, and pids repeat.
  std::string file = dir + "/node-a-" + std::to_string(getpid()) + ".json";
  {
    Registry reg(o);
    Component c;
    uint64_t h = reg.Register("io", {}, &c);
    ASSERT_GT(reg.port(), taken.port());
    EXPECT_LT(reg.port(), taken.port() + o.port_probe);

    std::ifstream in(file);
    ASSERT_TRUE(in.good()) << file;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string want = "[{\"targets\":[\"127.0.0.1:" + std::to_string(reg.port()) +
                             "\"],\"labels\":{\"pid\":\"" + std::to_string(getpid()) +
                             "\",\"local_rank\":\"3\"}}]\n";
    EXPECT_EQ(ss.str(), want);
    reg.Unregister(h);
  }
  EXPECT_NE(access(file.c_str(), F_OK), 0);  // gone with the registry
  ASSERT_EQ(std::system(("rm -rf " + dir).c_str()), 0);
}

TEST(Registry, TargetFileNameKeepsToItsDirectory) {
  char tmpl[] = "/tmp/mori_registry_name_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string dir = tmpl;
  Registry::Options o = Quiet();
  o.host = "../rack 7/b";
  o.bind_addr = "127.0.0.1";
  MetricsServer probe(0, "127.0.0.1");
  o.port = probe.port() + 1;
  o.port_dir = dir;
  {
    Registry reg(o);
    Component c;
    uint64_t h = reg.Register("io", {}, &c);
    EXPECT_EQ(access((dir + "/.._rack_7_b-" + std::to_string(getpid()) + ".json").c_str(), F_OK),
              0);
    reg.Unregister(h);
  }
  ASSERT_EQ(std::system(("rm -rf " + dir).c_str()), 0);
}

TEST(Registry, HandlesAreUniqueAcrossRegistries) {
  Registry a(Quiet()), b(Quiet());
  Component ca, cb;
  cb.bytes = 4;
  const uint64_t ha = a.Register("x", {}, &ca);
  const uint64_t hb = b.Register("x", {}, &cb);
  EXPECT_NE(ha, hb);
  b.Unregister(ha);  // not b's: nothing happens
  EXPECT_NE(Find(b.Snapshot(), "t_bytes_total", {{"host", "node-a"}}), nullptr);
  a.Unregister(ha);
  b.Unregister(hb);
  EXPECT_EQ(Find(b.Snapshot(), "t_bytes_total", {{"host", "node-a"}}), nullptr);
}

TEST(Registry, NoFreePortIsNotFatal) {
  MetricsServer taken(0, "127.0.0.1");
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  o.port = taken.port();
  o.port_probe = 1;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  EXPECT_EQ(reg.port(), -1);
  c.bytes = 3;
  auto snap = reg.Snapshot();
  ASSERT_NE(Find(snap, "t_bytes_total", {{"host", "node-a"}}), nullptr);
  reg.Unregister(h);
}

TEST(Registry, BackgroundThreadTicks) {
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  MetricsServer probe(0, "127.0.0.1");
  o.port = probe.port() + 1;
  o.interval_ms = 20;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  c.bytes = 11;
  std::string body;
  for (int i = 0; i < 200; ++i) {  // no Tick(): the thread has to do it
    body = HttpGetBody(reg.port());
    if (body.find("t_bytes_total{host=\"node-a\"} 11") != std::string::npos) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_NE(body.find("t_bytes_total{host=\"node-a\"} 11"), std::string::npos) << body;
  reg.Unregister(h);
}

// The periods are the process's: every case leaves them at 1. Not through
// Registry::Instance(): the fork case must be the first to create it.
struct RestoreSampling {
  ~RestoreSampling() { Registry(Quiet()).SetSampling(1, 1, -1); }
};

TEST(Sampling, StartsAtEverythingAndIsPublished) {
  RestoreSampling restore;
  Registry::Options o = Quiet();
  o.sampling_series = true;
  o.interval_ms = 250;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);

  EXPECT_EQ(reg.GetSampling().ep, 1u);
  EXPECT_EQ(reg.GetSampling().io, 1u);
  EXPECT_EQ(reg.GetSampling().interval_ms, 250);
  auto snap = reg.Snapshot();
  const auto* ep = Find(snap, "mori_metrics_sample_period", {{"host", "node-a"}, {"module", "ep"}});
  const auto* io = Find(snap, "mori_metrics_sample_period", {{"host", "node-a"}, {"module", "io"}});
  const auto* interval = Find(snap, "mori_metrics_sample_interval_seconds", {{"host", "node-a"}});
  ASSERT_NE(ep, nullptr);
  ASSERT_NE(io, nullptr);
  ASSERT_NE(interval, nullptr);
  EXPECT_EQ(ep->kind, MetricKind::kGauge);
  EXPECT_EQ(ep->value, 1.0);
  EXPECT_EQ(io->value, 1.0);
  EXPECT_DOUBLE_EQ(interval->value, 0.25);

  std::string error;
  const mori::metrics::Sampling now = reg.SetSampling(8, 64, -1, &error);
  EXPECT_TRUE(error.empty()) << error;
  EXPECT_EQ(now.ep, 8u);
  EXPECT_EQ(now.io, 64u);
  EXPECT_EQ(now.interval_ms, 250);  // not asked for: as it was
  EXPECT_EQ(mori::metrics::EpSamplePeriod(), 8u);
  EXPECT_EQ(mori::metrics::IoSamplePeriodPtr()->load(), 64u);
  snap = reg.Snapshot();
  EXPECT_EQ(Find(snap, "mori_metrics_sample_period", {{"host", "node-a"}, {"module", "ep"}})->value,
            8.0);
  EXPECT_EQ(Find(snap, "mori_metrics_sample_period", {{"host", "node-a"}, {"module", "io"}})->value,
            64.0);

  EXPECT_EQ(reg.SetSampling(-1, 0, -1).io, 0u);
  EXPECT_EQ(reg.GetSampling().ep, 8u);
  reg.Unregister(h);
}

TEST(Sampling, AValueOutOfRangeChangesNothing) {
  RestoreSampling restore;
  Registry::Options o = Quiet();
  o.interval_ms = 250;
  Registry reg(o);
  for (const auto& bad :
       std::vector<std::array<int64_t, 3>>{{int64_t{mori::metrics::kMaxSamplePeriod} + 1, 4, -1},
                                           {4, int64_t{mori::metrics::kMaxSamplePeriod} + 1, -1},
                                           {4, 4, 0},
                                           {4, 4, mori::metrics::kMaxSampleIntervalMs + 1}}) {
    std::string error;
    const mori::metrics::Sampling now = reg.SetSampling(bad[0], bad[1], bad[2], &error);
    EXPECT_FALSE(error.empty());
    // Nothing of a refused change is applied, the valid fields neither.
    EXPECT_EQ(now.ep, 1u);
    EXPECT_EQ(now.io, 1u);
    EXPECT_EQ(now.interval_ms, 250);
  }
  // A registry without a thread has no interval to change.
  Registry still(Quiet());
  std::string error;
  still.SetSampling(-1, -1, 10, &error);
  EXPECT_FALSE(error.empty());
}

// The thread sleeps for an hour, and ticks within moments of being told ten milliseconds.
TEST(Sampling, IntervalIsChangedWhileTheThreadRuns) {
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  MetricsServer probe(0, "127.0.0.1");
  o.port = probe.port() + 1;
  o.interval_ms = 3600000;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  c.bytes = 11;
  EXPECT_EQ(reg.SetSampling(-1, -1, 10).interval_ms, 10);
  // What SetSampling itself sampled is 11; only the thread can bring 12.
  c.bytes = 12;
  for (int i = 0; i < 300 && c.sampled.load() != 12u; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(c.sampled.load(), 12u);
  EXPECT_NE(HttpRequest(reg.port(), "GET /sampling").find("\"interval_ms\":10"), std::string::npos);
  reg.Unregister(h);
}

TEST(Sampling, IsReadAndChangedOnTheMetricsPort) {
  RestoreSampling restore;
  MetricsServer probe(0, "127.0.0.1");
  Registry::Options o = Quiet();
  o.sampling_series = true;
  o.bind_addr = "127.0.0.1";
  o.port = probe.port() + 1;
  o.interval_ms = 250;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  ASSERT_GT(reg.port(), 0);

  std::string response = HttpRequest(reg.port(), "GET /sampling");
  EXPECT_NE(response.find("200 OK"), std::string::npos) << response;
  EXPECT_NE(response.find("{\"ep\":1,\"io\":1,\"interval_ms\":250}"), std::string::npos)
      << response;

  response = HttpRequest(reg.port(), "POST /sampling?ep=16&io=4&interval_ms=100");
  EXPECT_NE(response.find("200 OK"), std::string::npos) << response;
  EXPECT_NE(response.find("{\"ep\":16,\"io\":4,\"interval_ms\":100}"), std::string::npos)
      << response;
  EXPECT_EQ(mori::metrics::EpSamplePeriod(), 16u);
  EXPECT_EQ(mori::metrics::IoSamplePeriodPtr()->load(), 4u);
  EXPECT_EQ(reg.interval_ms(), 100);
  const std::string body = HttpGetBody(reg.port());
  EXPECT_NE(body.find("mori_metrics_sample_period{host=\"node-a\",module=\"ep\"} 16"),
            std::string::npos)
      << body;
  EXPECT_NE(body.find("mori_metrics_sample_interval_seconds{host=\"node-a\"} 0.1"),
            std::string::npos)
      << body;

  response = HttpRequest(reg.port(), "POST /sampling?io=0");
  EXPECT_NE(response.find("{\"ep\":16,\"io\":0,\"interval_ms\":100}"), std::string::npos)
      << response;

  response = HttpRequest(reg.port(), "GET /sampling?ep=2");
  EXPECT_NE(response.find("405"), std::string::npos) << response;
  for (const char* bad : {"POST /sampling?ep=two", "POST /sampling?ep=-3", "POST /sampling?gpu=2",
                          "POST /sampling?ep=2&io=99999999", "POST /sampling?ep"}) {
    response = HttpRequest(reg.port(), bad);
    EXPECT_NE(response.find("400 Bad Request"), std::string::npos) << bad << ": " << response;
  }
  EXPECT_EQ(mori::metrics::EpSamplePeriod(), 16u);
  EXPECT_NE(HttpRequest(reg.port(), "GET /samplings").find("404"), std::string::npos);
  reg.Unregister(h);
}

// With SIGUSR1 blocked in this thread it has to stay pending: a registry thread
// that had it open would get it at once.
std::atomic<int> g_signal_runs{0};
void CountSignal(int) { g_signal_runs.fetch_add(1); }

TEST(Registry, ItsThreadsDoNotTakeProcessSignals) {
  MetricsServer probe(0, "127.0.0.1");
  Registry::Options o = Quiet();
  o.bind_addr = "127.0.0.1";
  o.port = probe.port() + 1;
  o.interval_ms = 10;  // both the tick thread and the server's accept thread
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  ASSERT_GT(reg.port(), 0);

  struct sigaction sa{};
  struct sigaction old_sa{};
  sa.sa_handler = CountSignal;
  ASSERT_EQ(sigaction(SIGUSR1, &sa, &old_sa), 0);
  sigset_t usr1, old_mask;
  sigemptyset(&usr1);
  sigaddset(&usr1, SIGUSR1);
  ASSERT_EQ(pthread_sigmask(SIG_BLOCK, &usr1, &old_mask), 0);

  g_signal_runs.store(0);
  ASSERT_EQ(kill(getpid(), SIGUSR1), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(g_signal_runs.load(), 0) << "a registry thread handled a process signal";

  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, &old_mask, nullptr), 0);  // now it is ours to take
  for (int i = 0; i < 200 && g_signal_runs.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(g_signal_runs.load(), 1);
  sigaction(SIGUSR1, &old_sa, nullptr);
  reg.Unregister(h);
}

TEST(RegistryOptions, PortIsBasePlusLocalRank) {
  for (const char* v : {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                        "MPI_LOCALRANKID", "MORI_METRICS_ADDR", "MORI_METRICS_PORT_DIR",
                        "MORI_METRICS_INTERVAL_MS", "MORI_METRICS_HOST_ID"}) {
    unsetenv(v);
  }
  setenv("MORI_METRICS_PORT", "9400", 1);
  EXPECT_EQ(Registry::Options::FromEnv().port, 9400);  // no rank: the base itself
  EXPECT_TRUE(Registry::Options::FromEnv().target_labels.empty());

  setenv("SLURM_LOCALID", "5", 1);
  EXPECT_EQ(Registry::Options::FromEnv().port, 9405);
  setenv("LOCAL_RANK", "2", 1);  // first in the list wins
  auto o = Registry::Options::FromEnv();
  EXPECT_EQ(o.port, 9402);
  // Not `rank`: a target label of that name makes Prometheus rename the EP series' own.
  EXPECT_EQ(o.target_labels, (MetricLabels{{"local_rank", "2"}}));

  setenv("MORI_METRICS_ADDR", "127.0.0.1", 1);
  setenv("MORI_METRICS_HOST_ID", "node-b", 1);
  setenv("MORI_METRICS_INTERVAL_MS", "250", 1);
  o = Registry::Options::FromEnv();
  EXPECT_EQ(o.bind_addr, "127.0.0.1");
  EXPECT_EQ(o.host, "node-b");
  EXPECT_EQ(o.interval_ms, 250);

  unsetenv("MORI_METRICS_PORT");  // no port: nothing is served
  EXPECT_EQ(Registry::Options::FromEnv().port, 0);

  // A push is for millisecond resolution: every millisecond unless told otherwise.
  setenv("MORI_METRICS_PUSH_URL", "http://127.0.0.1:8428", 1);
  o = Registry::Options::FromEnv();
  EXPECT_EQ(o.push_url, "http://127.0.0.1:8428");
  EXPECT_EQ(o.interval_ms, 250);  // MORI_METRICS_INTERVAL_MS is still set
  unsetenv("MORI_METRICS_INTERVAL_MS");
  EXPECT_EQ(Registry::Options::FromEnv().interval_ms, 1);
  unsetenv("MORI_METRICS_PUSH_URL");
  EXPECT_EQ(Registry::Options::FromEnv().interval_ms, 1000);
  for (const char* v : {"LOCAL_RANK", "SLURM_LOCALID", "MORI_METRICS_ADDR", "MORI_METRICS_HOST_ID",
                        "MORI_METRICS_INTERVAL_MS"}) {
    unsetenv(v);
  }
}

// A stand-in for VictoriaMetrics' /api/v1/import: keeps the bodies, answers `status`.
class PushSink {
 public:
  PushSink() {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    EXPECT_EQ(::listen(fd_, 16), 0);
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }
  ~PushSink() {
    stop_.store(true);
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    thread_.join();
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
  int requests() const { return requests_.load(); }
  std::atomic<int> status{204};

  // The line of the series `name` whose labels contain `label` ("" = any), or "".
  std::string Line(const std::string& name, const std::string& label = "") {
    std::lock_guard<std::mutex> lock(mu_);
    std::stringstream ss(body_);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.find("\"__name__\":\"" + name + "\"") == std::string::npos) continue;
      if (!label.empty() && line.find(label) == std::string::npos) continue;
      return line;
    }
    return "";
  }
  // The numbers of the array `key` ("values" / "timestamps") of a line.
  static std::vector<double> Numbers(const std::string& line, const std::string& key) {
    std::vector<double> out;
    auto pos = line.find("\"" + key + "\":[");
    if (pos == std::string::npos) return out;
    pos += key.size() + 4;
    std::stringstream ss(line.substr(pos, line.find(']', pos) - pos));
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
    return out;
  }
  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    body_.clear();
  }

 private:
  void Serve() {
    while (!stop_.load()) {
      int c = ::accept(fd_, nullptr, nullptr);
      if (c < 0) break;
      std::string req;
      char buf[65536];
      std::size_t want = std::string::npos;
      for (;;) {
        const auto head_end = req.find("\r\n\r\n");
        if (head_end != std::string::npos && want == std::string::npos) {
          const auto cl = req.find("Content-Length: ");
          want = head_end + 4 + (cl == std::string::npos ? 0 : std::stoul(req.substr(cl + 16)));
        }
        if (want != std::string::npos && req.size() >= want) break;
        const ssize_t n = ::read(c, buf, sizeof(buf));
        if (n <= 0) break;
        req.append(buf, static_cast<std::size_t>(n));
      }
      if (req.compare(0, 20, "POST /api/v1/import ") == 0 && want != std::string::npos) {
        std::lock_guard<std::mutex> lock(mu_);
        body_ += req.substr(req.find("\r\n\r\n") + 4);
      }
      requests_.fetch_add(1);
      const std::string resp = "HTTP/1.0 " + std::to_string(status.load()) + " X\r\n\r\n";
      (void)!::write(c, resp.data(), resp.size());
      ::close(c);
    }
  }

  int fd_ = -1;
  int port_ = 0;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<int> requests_{0};
  std::mutex mu_;
  std::string body_;
};

Registry::Options Pushing(const PushSink& sink) {
  Registry::Options o = Quiet();
  o.push_url = sink.url();
  o.push_flush_ms = 0;            // the tests send by hand
  o.push_heartbeat_ms = 3600000;  // and no heartbeat unless a test wants one
  return o;
}

TEST(RegistryPush, KeepsAPointForEverySampleThatMoved) {
  PushSink sink;
  Registry reg(Pushing(sink));
  Component c;
  const int64_t before = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
  uint64_t h = reg.Register("io", {{"gpu", "3"}}, &c);  // first sample: the series at 0
  c.bytes = 5;
  reg.Tick();
  reg.Tick();  // nothing moved: no point
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  c.bytes = 9;
  c.inflight = 2;
  reg.Tick();
  reg.FlushPush();
  ASSERT_EQ(sink.requests(), 1);

  const std::string line = sink.Line("t_bytes_total");
  ASSERT_FALSE(line.empty());
  EXPECT_EQ(PushSink::Numbers(line, "values"), (std::vector<double>{0, 5, 9}));
  const std::vector<double> ts = PushSink::Numbers(line, "timestamps");
  ASSERT_EQ(ts.size(), 3u);
  EXPECT_GE(ts[0], before);  // wall clock, milliseconds
  EXPECT_LE(ts[0], ts[1]);
  EXPECT_GE(ts[2] - ts[1], 3);  // the time each sample was taken, not the time of the send
  // The source's labels, and what a scraper would have added.
  EXPECT_NE(line.find("\"host\":\"node-a\",\"gpu\":\"3\""), std::string::npos) << line;
  EXPECT_NE(line.find("\"instance\":\"node-a:" + std::to_string(getpid()) + "\""),
            std::string::npos)
      << line;
  EXPECT_NE(line.find("\"job\":\"mori_traffic\""), std::string::npos) << line;

  EXPECT_EQ(PushSink::Numbers(sink.Line("t_inflight"), "values"), (std::vector<double>{0, 2}));
  // A histogram goes with the heartbeat only: two buckets, +Inf, sum and count, once.
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_count"), "values").size(), 1u);
  EXPECT_EQ(reg.push_stats().sent, 3u + 2u + 5u);
  EXPECT_EQ(reg.push_stats().dropped, 0u);

  reg.Tick();
  reg.FlushPush();
  EXPECT_EQ(sink.requests(), 1);
  reg.Unregister(h);
}

TEST(RegistryPush, HeartbeatResendsIdleSeriesAndSendsHistograms) {
  PushSink sink;
  Registry::Options o = Pushing(sink);
  o.push_heartbeat_ms = 20;
  Registry reg(o);
  Component c;
  c.bytes = 7;
  c.lat_buckets[0] = 1;
  c.lat_buckets[1] = 3;
  c.lat_count = 4;
  c.lat_sum_ns = 5000000;
  uint64_t h = reg.Register("io", {}, &c);
  reg.Tick();  // the first heartbeat
  reg.FlushPush();
  sink.Clear();

  reg.Tick();  // inside the heartbeat period: nothing moved, nothing kept
  reg.FlushPush();
  EXPECT_TRUE(sink.Line("t_bytes_total").empty());

  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  reg.Tick();
  reg.FlushPush();
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_bytes_total"), "values"), (std::vector<double>{7}));
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_bucket", "\"le\":\"0.001\""), "values"),
            (std::vector<double>{1}));
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_bucket", "\"le\":\"0.01\""), "values"),
            (std::vector<double>{3}));
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_bucket", "\"le\":\"+Inf\""), "values"),
            (std::vector<double>{4}));
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_count"), "values"),
            (std::vector<double>{4}));
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_latency_seconds_sum"), "values"),
            (std::vector<double>{0.005}));
  reg.Unregister(h);
}

TEST(RegistryPush, ADatabaseThatIsAwayCostsPointsNotTime) {
  int closed_port = 0;
  {
    MetricsServer probe(0, "127.0.0.1");
    closed_port = probe.port();
  }
  Registry::Options o = Quiet();
  o.push_url = "http://127.0.0.1:" + std::to_string(closed_port);
  o.push_flush_ms = 0;
  Registry reg(o);
  Component c;
  uint64_t h = reg.Register("io", {}, &c);
  const auto t0 = std::chrono::steady_clock::now();
  reg.FlushPush(500);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(1500));
  EXPECT_EQ(reg.push_stats().failures, 1u);
  EXPECT_EQ(reg.push_stats().dropped, 2u);  // the counter and the gauge at their first sample
  EXPECT_EQ(reg.push_stats().sent, 0u);

  PushSink sink;  // a database that answers, but with an error
  sink.status = 500;
  Registry::Options o2 = Pushing(sink);
  Registry reg2(o2);
  uint64_t h2 = reg2.Register("io", {}, &c);
  reg2.FlushPush();
  EXPECT_EQ(reg2.push_stats().failures, 1u);
  EXPECT_EQ(reg2.push_stats().dropped, 2u);
  reg.Unregister(h);
  reg2.Unregister(h2);
}

TEST(RegistryPush, TheBufferIsBounded) {
  PushSink sink;
  Registry::Options o = Pushing(sink);
  o.push_max_points = 6;
  Registry reg(o);
  struct Counter : MetricSource {
    std::atomic<uint64_t> n{0};
    std::vector<MetricSample> SampleMetrics() const override {
      return {MetricSample{"t_bytes_total", "bytes", {}, n.load()}};
    }
  } c;
  uint64_t h = reg.Register("io", {}, &c);  // the first point
  for (int i = 1; i <= 10; ++i) {
    c.n = i;
    reg.Tick();  // one point each
  }
  EXPECT_EQ(reg.push_stats().dropped, 5u);  // 11 samples moved, 6 fit
  reg.FlushPush();
  EXPECT_EQ(PushSink::Numbers(sink.Line("t_bytes_total"), "values"),
            (std::vector<double>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(reg.push_stats().sent, 6u);
  reg.Unregister(h);
}

TEST(RegistryPush, AUrlThatCannotBeUsedPushesNothing) {
  Registry::Options o = Quiet();
  o.push_url = "victoria:8428";
  o.push_flush_ms = 0;
  Registry reg(o);
  Component c;
  c.bytes = 3;
  uint64_t h = reg.Register("io", {}, &c);
  reg.FlushPush();
  EXPECT_EQ(reg.push_stats().sent + reg.push_stats().dropped, 0u);
  EXPECT_NE(Find(reg.Snapshot(), "t_bytes_total", {{"host", "node-a"}}), nullptr);
  reg.Unregister(h);
}

// How many ticks fit depends on how busy the machine is: loose lower bounds, an exact top.
TEST(RegistryPush, AMillisecondTickIsSentInBatches) {
  PushSink sink;
  Registry::Options o = Pushing(sink);
  o.interval_ms = 1;
  o.push_flush_ms = 100;
  Registry reg(o);
  struct Running : MetricSource {
    std::vector<MetricSample> SampleMetrics() const override {
      return {MetricSample{"t_running_total", "running", {}, ++n}};
    }
    mutable uint64_t n = 0;
  } running;
  uint64_t h = reg.Register("run", {}, &running);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  reg.Unregister(h);
  reg.FlushPush();
  EXPECT_GE(sink.requests(), 2);  // several sends, not one at the end
  const uint64_t sent = reg.push_stats().sent;
  EXPECT_GE(sent, 20u) << "ticks in 600 ms";
  EXPECT_LE(sent, 700u) << "no more than a tick a millisecond";
  EXPECT_EQ(reg.push_stats().dropped, 0u);
  const std::string line = sink.Line("t_running_total");
  const std::vector<double> ts = PushSink::Numbers(line, "timestamps");
  const std::vector<double> values = PushSink::Numbers(line, "values");
  ASSERT_GE(ts.size(), 2u);
  ASSERT_EQ(ts.size(), values.size());
  for (std::size_t i = 1; i < ts.size(); ++i) {
    EXPECT_LE(ts[i - 1], ts[i]);
    EXPECT_EQ(values[i], values[i - 1] + 1);
  }
  EXPECT_GT(ts.back(), ts.front()) << "the time of each sample, not the time of the send";
}

// The parent's lock is HELD across the fork (a tick in progress on another
// thread), which is exactly the state a child must not wait on.
TEST(RegistryFork, ChildStartsItsOwnRegistry) {
  char tmpl[] = "/tmp/mori_registry_fork_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string dir = tmpl;
  MetricsServer probe(0, "127.0.0.1");
  setenv("MORI_METRICS_PORT", std::to_string(probe.port() + 1).c_str(), 1);
  setenv("MORI_METRICS_ADDR", "127.0.0.1", 1);
  setenv("MORI_METRICS_PORT_DIR", dir.c_str(), 1);
  setenv("MORI_METRICS_INTERVAL_MS", "50", 1);
  setenv("MORI_METRICS_HOST_ID", "node-a", 1);

  struct Blocking : MetricSource {
    std::atomic<bool> block{false};
    mutable std::atomic<bool> inside{false};
    std::vector<MetricSample> SampleMetrics() const override {
      while (block.load()) {
        inside.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      inside.store(false);
      return {MetricSample{"t_parent_total", "parent", {}, 1}};
    }
  };
  static Blocking parent_source;  // never unregistered: Instance() is never destroyed

  Registry& parent = Registry::Instance();
  const uint64_t parent_handle = parent.Register("parent", {}, &parent_source);
  const int parent_port = parent.port();
  ASSERT_GT(parent_port, 0);

  parent_source.block.store(true);
  for (int i = 0; i < 2000 && !parent_source.inside.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(parent_source.inside.load());  // the tick thread now holds the lock

  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    // A child that waits on the inherited lock is killed by the alarm and reported below.
    alarm(20);
    Registry& child = Registry::Instance();
    int rc = 0;
    if (&child == &parent) rc |= 1;
    static Component child_source;
    child_source.bytes = 5;
    child.Register("child", {}, &child_source);
    if (child.port() <= 0 || child.port() == parent_port) rc |= 2;
    child.Tick();
    std::string body = HttpGetBody(child.port());
    if (body.find("t_bytes_total{host=\"node-a\"} 5") == std::string::npos) rc |= 4;
    if (body.find("t_parent_total") != std::string::npos) rc |= 8;  // not carried over
    const std::string file = dir + "/node-a-" + std::to_string(getpid()) + ".json";
    if (access(file.c_str(), F_OK) != 0) rc |= 16;
    // A handle the PARENT's registry gave out must name nothing in the child's.
    child.Unregister(parent_handle);
    child_source.bytes = 9;
    child.Tick();
    body = HttpGetBody(child.port());
    if (body.find("t_bytes_total{host=\"node-a\"} 9") == std::string::npos) rc |= 32;
    _exit(rc);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  parent_source.block.store(false);
  ASSERT_TRUE(WIFEXITED(status)) << "child killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0) << "child failure bits";

  EXPECT_EQ(&Registry::Instance(), &parent);
  std::string body;
  for (int i = 0; i < 200; ++i) {
    body = HttpGetBody(parent_port);
    if (body.find("t_parent_total{host=\"node-a\"} 1") != std::string::npos) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_NE(body.find("t_parent_total{host=\"node-a\"} 1"), std::string::npos) << body;
  EXPECT_EQ(access((dir + "/node-a-" + std::to_string(getpid()) + ".json").c_str(), F_OK), 0);
}

}  // namespace
}  // namespace mori::metrics
