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
// Live push: HTTP client, counter selection and the exporter against a fake VictoriaMetrics.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <thread>

#include "exporter.h"
#include "fake_vm.h"
#include "harness.h"
#include "http_client.h"
#include "util.h"

using namespace netmon;

namespace {

uint64_t g_key = 0;  // unique (experiment) per exporter: the same-host lock is process-wide

PushOptions Opts(const nt::FakeVm& vm) {
  PushOptions o;
  o.url = vm.url();
  o.experiment = "t" + std::to_string(getpid()) + "_" + std::to_string(g_key++);
  o.host = "h";
  o.interval_ns = 100000000ull;
  o.all_counters = true;
  o.quiet = true;
  return o;
}

ExportStream Stream1() {
  ExportStream s;
  s.netdev = "n0";
  s.source = "rdma";
  s.interval_ns = 2000000;
  s.snapshot_fraction = 0.5;
  s.names = {"tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes", "other"};
  s.record_words = 4 + s.names.size();
  s.batch_records = 25;
  s.speed_mbps = 400000;
  s.ibdev = "ionic_3";  // not the netdev's number: the name must come from the stream
  s.pci = "0000:79:00.0";
  return s;
}

// `n` records starting at sample index `first`, 2 ms apart, steady traffic.
Batch MakeBatch(size_t first, size_t n) {
  Batch b(0, 7, n);
  for (size_t i = 0; i < n; i++) {
    uint64_t* r = b.Slot();
    const uint64_t t0 = 1000000000000ull + (first + i) * 2000000ull;
    r[0] = t0;
    r[1] = t0 + 100000;
    r[2] = t0 + 1700000000000000000ull;
    r[3] = 0;
    r[4] = (first + i) * 1000;
    r[5] = (first + i) * 2000;
    r[6] = (first + i) * 3;
    b.count++;
  }
  return b;
}

bool WaitPhase(Exporter* e, Exporter::Phase want, uint64_t ms) {
  const uint64_t until = MonoNs() + ms * 1000000ull;
  while (MonoNs() < until) {
    if (e->phase() == want) return true;
    usleep(5000);
  }
  return e->phase() == want;
}

// Records [first, first + n) at the given sample times (ns from the start), steady traffic.
Batch MakeBatchAt(size_t first, size_t n, const std::vector<uint64_t>& at) {
  Batch b(0, 7, n);
  for (size_t i = 0; i < n; i++) {
    uint64_t* r = b.Slot();
    const uint64_t t0 = 1000000000000ull + at[first + i];
    r[0] = t0;
    r[1] = t0 + 100000;
    r[2] = t0 + 1700000000000000000ull;
    r[3] = 0;
    r[4] = at[first + i] / 2000;
    r[5] = at[first + i] / 1000;
    r[6] = first + i;
    b.count++;
  }
  return b;
}

// Pushes `at` (sample times) through an exporter and returns every netmon_step_ms value sent.
std::vector<std::string> StepsPushed(const std::vector<uint64_t>& at) {
  auto& vm = *new nt::FakeVm();      // its threads outlive the test: never freed
  auto* e = new Exporter(Opts(vm));  // leaked on purpose, like in the recorder
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  std::string err;
  CHECK(e->Configure({Stream1()}, 1700001000000000000ull, &err));
  for (size_t k = 0; k + 25 <= at.size(); k += 25) {
    e->Offer(MakeBatchAt(k, 25, at));
    usleep(1000);  // the slot ring holds 200 batches: let the exporter keep up
  }
  e->CloseInput();
  CHECK(e->WaitDone(MonoNs() + 20000000000ull));
  CHECK(e->Snapshot().dropped_samples == 0);
  std::vector<std::string> out;
  for (const auto& body : vm.bodies()) {
    size_t p = 0;
    while ((p = body.find("\"__name__\":\"netmon_step_ms\"", p)) != std::string::npos) {
      const size_t v = body.find("\"values\":[", p) + 10;
      out.push_back(body.substr(v, body.find(']', v) - v));
      p = v;
    }
  }
  return out;
}

}  // namespace

// Same cases as test_import.py DisplayLabelsTest (the importer's rule).
TEST(DisplayLabelsMatchTheImporter) {
  using L = std::vector<std::pair<std::string, std::string>>;
  CHECK((DisplayLabels("n08-25.example.com", "benic3p1", "ionic_3", "0000:79:00.0") ==
         L{{"display_nic", "ionic_3 (benic3p1)"},
           {"display_host_nic", "n08-25 \u00b7 ionic_3"},
           {"ibdev", "ionic_3"},
           {"pci", "0000:79:00.0"}}));
  CHECK((DisplayLabels("plainhost", "eth9", "", "") ==
         L{{"display_nic", "eth9"}, {"display_host_nic", "plainhost \u00b7 eth9"}}));
}

TEST(HttpUrlParsing) {
  HttpUrl u;
  std::string err;
  CHECK(ParseHttpUrl("http://127.0.0.1:8428", &u, &err));
  CHECK(u.host == "127.0.0.1" && u.port == 8428 && u.prefix.empty());
  CHECK(ParseHttpUrl("http://vm/prefix/", &u, &err));
  CHECK(u.host == "vm" && u.port == 80 && u.prefix == "/prefix");
  CHECK(ParseHttpUrl("http://[::1]:9/", &u, &err));
  CHECK(u.host == "::1" && u.port == 9);
  CHECK(!ParseHttpUrl("https://vm:8428", &u, &err));
  CHECK(!ParseHttpUrl("http://vm:0", &u, &err));
  CHECK(!ParseHttpUrl("http://:80", &u, &err));
  CHECK(!ParseHttpUrl("http://vm/x?y", &u, &err));
}

TEST(HttpClientResults) {
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  HttpUrl u;
  std::string err;
  CHECK(ParseHttpUrl(vm.url(), &u, &err));
  HttpClient c;
  CHECK(c.Resolve(u, &err));
  auto r = c.Request("GET", "/flags", "", nullptr, MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kOk && r.status == 200 && r.body == vm.flags);
  vm.raw_mode = 1;  // chunked
  r = c.Request("GET", "/flags", "", nullptr, MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kOk && r.body == vm.flags);
  vm.raw_mode = 2;  // oversized
  r = c.Request("GET", "/flags", "", nullptr, MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kUncertain);
  vm.raw_mode = 0;
  vm.import_mode = static_cast<int>(nt::FakeVm::Import::kFail500);
  r = c.Request("POST", "/api/v1/import", "x", "application/json", MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kOk && r.status == 500);
  vm.import_mode = static_cast<int>(nt::FakeVm::Import::kCloseNoReply);
  r = c.Request("POST", "/api/v1/import", "x", "application/json", MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kUncertain);  // sent, then closed: never retried
  vm.import_mode = static_cast<int>(nt::FakeVm::Import::kBlackHole);
  const uint64_t t0 = MonoNs();
  r = c.Request("POST", "/api/v1/import", "x", "application/json", MonoNs() + 300000000ull);
  CHECK(r.kind == HttpResult::Kind::kUncertain);
  CHECK(MonoNs() - t0 < 1000000000ull);  // the deadline holds
  // Nothing listens: refused before anything was sent (safe to retry).
  HttpUrl dead = u;
  dead.port = 1;
  HttpClient d;
  CHECK(d.Resolve(dead, &err));
  r = d.Request("GET", "/", "", nullptr, MonoNs() + 1000000000ull);
  CHECK(r.kind == HttpResult::Kind::kNotSent);
}

TEST(PushCounterSelection) {
  std::vector<bool> opt;
  auto cols = SelectPushCounters(
      "rdma", {"x", "rx_rdma_ucast_bytes", "tx_rdma_ucast_bytes", "rx_pripause_3_1us_count"}, false,
      &opt);
  // key counters in KEY_COUNTERS order, then the optional ones from their first change
  CHECK(cols.size() == 3 && cols[0] == 2 && cols[1] == 1 && cols[2] == 3);
  CHECK(!opt[0] && !opt[1] && opt[2]);
  cols = SelectPushCounters("rdma", {"x", "tx_rdma_ucast_bytes"}, true, &opt);
  CHECK(cols.size() == 2 && cols[0] == 1 && cols[1] == 0 && opt[1]);
  cols = SelectPushCounters("ethtool", {"tx_rdma_ucast_bytes"}, true, &opt);
  CHECK(cols.empty());
  // ethtool: exactly the port statistics (they define the refresh rule), even with --all
  cols = SelectPushCounters("ethtool", {"x", "octets_rx_ok", "frames_tx_pri_3", "octets_tx_ok"},
                            true, &opt);
  CHECK(cols.size() == 2 && cols[0] == 3 && cols[1] == 1 && !opt[0] && !opt[1]);
  cols = SelectPushCounters("ipc", {"frames_rx_pri_3", "octets_rx_ok"}, false, &opt);
  CHECK(cols.size() == 2 && cols[0] == 1 && opt[1]);
}

TEST(ExporterPushesIntervalsAndMetadata) {
  auto& vm = *new nt::FakeVm();      // its threads outlive the test: never freed
  auto* e = new Exporter(Opts(vm));  // leaked on purpose, like in the recorder
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  std::string err;
  ExportStream eth;  // 1 s, nothing pushed: must not set the step to 3 s
  eth.netdev = "n0";
  eth.source = "ethtool";
  eth.interval_ns = 1000000000ull;
  eth.record_words = 5;
  eth.batch_records = 1;
  eth.names = {"x"};
  CHECK(e->Configure({Stream1(), eth}, 1700001000000000000ull, &err));
  for (size_t k = 0; k < 20; k++) e->Offer(MakeBatch(k * 25, 25));
  e->CloseInput();
  CHECK(e->WaitDone(MonoNs() + 5000000000ull));
  auto st = e->Snapshot();
  CHECK(!st.abandoned);
  CHECK(st.failed_points == 0 && st.dropped_samples == 0);
  CHECK(st.points_sent > 0);
  std::string all;
  for (const auto& b : vm.bodies()) all += b;
  CHECK(all.find("\"__name__\":\"netmon_target_info\"") != std::string::npos);
  CHECK(all.find("\"status\":\"live\"") != std::string::npos);
  CHECK(all.find("\"display_nic\":\"ionic_3 (n0)\"") != std::string::npos);
  CHECK(all.find("\"ibdev\":\"ionic_3\",\"pci\":\"0000:79:00.0\"") != std::string::npos);
  CHECK(all.find("\"__name__\":\"netmon_import_info\"") != std::string::npos);
  CHECK(all.find("\"__name__\":\"netmon_step_ms\"") != std::string::npos);
  CHECK(all.find("\"counter\":\"tx_rdma_ucast_bytes\",\"resolution\":\"normal\"") !=
        std::string::npos);
  // step: 3 x 2 ms (the preliminary value, and the measured baseline once warmed up)
  CHECK(all.find("\"__name__\":\"netmon_step_ms\",\"experiment\":") != std::string::npos);
  CHECK(all.find(",\"values\":[6],") != std::string::npos);
  CHECK(all.find(",\"values\":[3000],") == std::string::npos);
  // the preliminary step is written at the anchor (start - 1 ms)
  CHECK(all.find("\"values\":[6],\"timestamps\":[1700000999999]") != std::string::npos);
  // 2 ms intervals of 1000 bytes = 500000 B/s
  CHECK(all.find("\"values\":[500000,") != std::string::npos);
}

TEST(ExporterRetriesMetadataUntilPublished) {
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  vm.fail_first = 1;             // the first push (with the metadata) fails
  auto* e = new Exporter(Opts(vm));
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  std::string err;
  CHECK(e->Configure({Stream1()}, 1700001000000000000ull, &err));
  for (size_t k = 0; k < 4; k++) e->Offer(MakeBatch(k * 25, 25));
  usleep(1500000);  // first push fails, backoff 1 s
  for (size_t k = 4; k < 8; k++) e->Offer(MakeBatch(k * 25, 25));
  e->CloseInput();
  CHECK(e->WaitDone(MonoNs() + 5000000000ull));
  auto st = e->Snapshot();
  CHECK(st.failed_points > 0);
  CHECK(!st.last_error.empty());
  std::string all;
  for (const auto& b : vm.bodies()) all += b;
  CHECK(all.find("netmon_target_info") != std::string::npos);  // published later
}

TEST(ExporterNeverBlocksTheWriter) {
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  vm.import_mode = static_cast<int>(nt::FakeVm::Import::kBlackHole);
  auto* e = new Exporter(Opts(vm));
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  std::string err;
  CHECK(e->Configure({Stream1()}, 1700001000000000000ull, &err));
  // The exporter hangs in a request; the writer keeps offering far more than the ring holds.
  usleep(300000);
  uint64_t worst = 0;
  for (size_t k = 0; k < 1000; k++) {
    Batch b = MakeBatch(k * 25, 25);
    const uint64_t t0 = MonoNs();
    e->Offer(b);
    worst = std::max(worst, MonoNs() - t0);
  }
  CHECK(worst < 5000000ull);  // never waits (5 ms is generous for a 25-record copy)
  e->CloseInput();
  const uint64_t t0 = MonoNs();
  e->WaitDone(MonoNs() + 500000000ull);
  CHECK(MonoNs() - t0 < 1000000000ull);  // bounded wait
  auto st = e->Snapshot();               // never waits either
  CHECK(st.dropped_samples > 0);
}

TEST(ExporterPreflightRefusesConflicts) {
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  vm.series_conflict = true;
  auto* e = new Exporter(Opts(vm));
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kFailed, 3000));
  CHECK(e->preflight_error().find("already has data") != std::string::npos);

  auto& vm2 = *new nt::FakeVm();  // its threads outlive the test: never freed
  PushOptions o = Opts(vm2);
  auto* a = new Exporter(o);
  a->Start();
  CHECK(WaitPhase(a, Exporter::Phase::kReady, 3000));
  auto* b = new Exporter(o);  // same VM, experiment and host: refused by the local lock
  b->Start();
  CHECK(WaitPhase(b, Exporter::Phase::kFailed, 3000));
  CHECK(b->preflight_error().find("already pushing") != std::string::npos);
  a->Abort();

  PushOptions dead = Opts(vm2);
  dead.url = "http://127.0.0.1:1";
  auto* c = new Exporter(dead);
  c->Start();
  CHECK(WaitPhase(c, Exporter::Phase::kFailed, 5000));
}

TEST(ExporterKeepsAnIntervalsMetricsInOneRequest) {
  // Tiny request budget: many requests; each holds rate, increase and interval of a
  // (counter, resolution) together or none of them.
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  PushOptions o = Opts(vm);
  o.max_body_bytes = 300;
  auto* e = new Exporter(o);
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  std::string err;
  CHECK(e->Configure({Stream1()}, 1700001000000000000ull, &err));
  for (size_t k = 0; k < 8; k++) e->Offer(MakeBatch(k * 25, 25));
  e->CloseInput();
  CHECK(e->WaitDone(MonoNs() + 10000000000ull));
  const auto bodies = vm.bodies();
  CHECK(bodies.size() > 3);
  for (const auto& b : bodies) {
    for (const char* c : {"tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes", "other"}) {
      for (const char* res : {"normal", "low"}) {
        const std::string lab = std::string("\"counter\":\"") + c + "\",\"resolution\":\"" + res;
        int n = 0;
        for (const char* m : {"netmon_rate\"", "netmon_increase\"", "netmon_interval_seconds\""}) {
          size_t p = 0;
          while ((p = b.find(m, p)) != std::string::npos) {
            const size_t eol = b.find('\n', p);
            if (b.substr(p, eol - p).find(lab) != std::string::npos) n++;
            p = eol;
          }
        }
        CHECK(n == 0 || n == 3);
      }
    }
  }
}

TEST(HttpClientFallsBackToTheNextAddress) {
  // "localhost" may resolve to ::1 first; the fake server listens on 127.0.0.1 only.
  auto& vm = *new nt::FakeVm();  // its threads outlive the test: never freed
  HttpUrl u;
  std::string err;
  CHECK(ParseHttpUrl(vm.url(), &u, &err));
  u.host = "localhost";
  HttpClient c;
  CHECK(c.Resolve(u, &err));
  auto r = c.Request("GET", "/flags", "", nullptr, MonoNs() + 2000000000ull);
  CHECK(r.kind == HttpResult::Kind::kOk && r.status == 200);
}

// The measured step needs kStepMinIntervals (10000) intervals and follows the whole-run
// median (CUSTOMER_FIXES_DESIGN §1): a stretch of long intervals at the start, before the
// threshold, leaves only the preliminary step; 3 ms sampling throughout publishes 9 ms.
TEST(ExporterStepFollowsTheWholeRun) {
  std::vector<uint64_t> early;
  uint64_t t = 0;
  for (size_t i = 0; i < 13000; i++, t += (i < 3000 ? 6000000 : 2000000)) early.push_back(t);
  const auto a = StepsPushed(early);
  CHECK(!a.empty());
  for (const auto& v : a) CHECK(v == "6");

  std::vector<uint64_t> slow;
  for (size_t i = 0; i < 12000; i++) slow.push_back(i * 3000000ull);
  const auto b = StepsPushed(slow);
  CHECK(std::find(b.begin(), b.end(), "6") != b.end());  // preliminary, at the anchor
  CHECK(std::find(b.begin(), b.end(), "9") != b.end());  // measured, after 10000 intervals

  std::vector<uint64_t> few;
  for (size_t i = 0; i < 5000; i++) few.push_back(i * 3000000ull);
  for (const auto& v : StepsPushed(few)) CHECK(v == "6");  // too few intervals to measure
}

// netmon_target_info says where each NIC's port statistics come from, as the importer does:
// ipc where there is an ipc stream, ethtool (pushed) where there is none, else none.
TEST(ExporterPortStatsLabel) {
  auto& vm = *new nt::FakeVm();      // its threads outlive the test: never freed
  auto* e = new Exporter(Opts(vm));  // leaked on purpose, like in the recorder
  e->Start();
  CHECK(WaitPhase(e, Exporter::Phase::kReady, 3000));
  auto stream = [](const char* nd, const char* src, std::vector<std::string> names, uint64_t iv) {
    ExportStream s;
    s.netdev = nd;
    s.source = src;
    s.interval_ns = iv;
    s.names = std::move(names);
    s.record_words = 4 + s.names.size();
    s.batch_records = 25;
    return s;
  };
  const std::vector<std::string> eth = {"octets_tx_ok", "octets_rx_ok", "tx_bytes"};
  std::string err;
  CHECK(e->Configure(
      {stream("a", "rdma", {"tx_rdma_ucast_bytes"}, 2000000),
       stream("a", "ipc", {"octets_tx_ok"}, 2000000), stream("a", "ethtool", eth, 1000000000ull),
       stream("b", "rdma", {"tx_rdma_ucast_bytes"}, 2000000),
       stream("b", "ethtool", eth, 1000000000ull),
       stream("c", "rdma", {"tx_rdma_ucast_bytes"}, 2000000),
       stream("c", "ethtool", {"tx_bytes"}, 1000000000ull)},
      1700001000000000000ull, &err));
  e->CloseInput();
  CHECK(e->WaitDone(MonoNs() + 5000000000ull));
  std::string all;
  for (const auto& b : vm.bodies()) all += b;
  auto label = [&](const char* nd) {
    const size_t p = all.find("\"netdev\":\"" + std::string(nd) + "\",\"target\"");
    const size_t q = all.find("\"port_stats\":\"", p);
    return p == std::string::npos || q == std::string::npos
               ? std::string()
               : all.substr(q + 14, all.find('"', q + 14) - q - 14);
  };
  CHECK(label("a") == "ipc");
  CHECK(label("b") == "ethtool");
  CHECK(label("c") == "none");
}
