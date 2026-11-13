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
// Unit tests for xgmi-netmon. No GPU needed: gpu_metrics captures from MI355X (v1.8 from amdgpu
// 6.16.6, v1.9 from amdgpu 6.16.13), a fake /sys/class/drm tree and a fake VictoriaMetrics.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "gpu_metrics.h"
#include "http.h"
#include "import.h"
#include "rates.h"
#include "recorder.h"
#include "recording.h"
#include "ring.h"
#include "topology.h"
#include "vm.h"

namespace fs = std::filesystem;
using namespace xgmi;

namespace {

int g_failures = 0;

#define CHECK(c)                                                            \
  do {                                                                      \
    if (!(c)) {                                                             \
      std::fprintf(stderr, "  %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #c); \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

#define CHECK_STR(a, b)                                                                        \
  do {                                                                                         \
    const std::string a_ = (a), b_ = (b);                                                      \
    if (a_ != b_) {                                                                            \
      std::fprintf(stderr, "  %s:%d: \"%s\"\n    != \"%s\"\n", __FILE__, __LINE__, a_.c_str(), \
                   b_.c_str());                                                                \
      ++g_failures;                                                                            \
    }                                                                                          \
  } while (0)

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string Fixture(const char* name) { return ReadFile(std::string(XGMI_FIXTURES) + "/" + name); }

const uint8_t* U8(const std::string& s) { return reinterpret_cast<const uint8_t*>(s.data()); }

struct TempDir {
  fs::path path;
  TempDir() {
    char tmpl[] = "/tmp/xgmi_netmon_test_XXXXXX";
    path = mkdtemp(tmpl);
  }
  ~TempDir() { fs::remove_all(path); }
};

// A fake VictoriaMetrics: records every request and answers by path.
class FakeVm {
 public:
  struct Request {
    std::string method, path, body;
  };
  std::string series_body = R"({"status":"success","data":[]})";
  std::atomic<bool> hang{false};
  std::atomic<int> import_status{204};

  FakeVm() {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    listen(fd_, 16);
    socklen_t len = sizeof(a);
    getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
    port_ = ntohs(a.sin_port);
    thread_ = std::thread([this] { Loop(); });
  }
  ~FakeVm() {
    stop_ = true;
    thread_.join();
    close(fd_);
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
  std::vector<Request> requests() {
    std::lock_guard<std::mutex> g(mu_);
    return reqs_;
  }
  std::string imported() {
    std::string all;
    for (const Request& r : requests())
      if (r.path == "/api/v1/import/prometheus") all += r.body;
    return all;
  }

 private:
  void Loop() {
    while (!stop_) {
      pollfd p{fd_, POLLIN, 0};
      if (poll(&p, 1, 20) <= 0) continue;
      const int c = accept(fd_, nullptr, nullptr);
      if (c < 0) continue;
      Handle(c);
      close(c);
    }
  }
  void Handle(int c) {
    std::string in;
    char buf[65536];
    size_t need = std::string::npos;
    while (need == std::string::npos || in.size() < need) {
      const ssize_t r = recv(c, buf, sizeof(buf), 0);
      if (r <= 0) return;
      in.append(buf, static_cast<size_t>(r));
      const size_t sep = in.find("\r\n\r\n");
      if (need == std::string::npos && sep != std::string::npos) {
        size_t cl = 0;
        const size_t h = in.find("Content-Length: ");
        if (h != std::string::npos && h < sep) cl = std::stoul(in.substr(h + 16));
        need = sep + 4 + cl;
      }
    }
    Request req;
    std::istringstream line(in.substr(0, in.find("\r\n")));
    line >> req.method >> req.path;
    req.body = in.substr(in.find("\r\n\r\n") + 4);
    {
      std::lock_guard<std::mutex> g(mu_);
      reqs_.push_back(req);
    }
    if (hang) {
      while (!stop_) std::this_thread::sleep_for(std::chrono::milliseconds(5));
      return;
    }
    std::string status = "200 OK", body;
    if (req.path == "/health") {
      body = "OK";
    } else if (req.path.rfind("/api/v1/series", 0) == 0) {
      body = series_body;
    } else if (req.path == "/api/v1/import/prometheus") {
      status =
          std::to_string(import_status.load()) + (import_status == 204 ? " No Content" : " Error");
    } else if (req.path.rfind("/api/v1/admin/tsdb/delete_series", 0) == 0) {
      status = "204 No Content";
    }
    const std::string out = "HTTP/1.1 " + status +
                            "\r\nContent-Length: " + std::to_string(body.size()) +
                            "\r\nConnection: close\r\n\r\n" + body;
    send(c, out.data(), out.size(), MSG_NOSIGNAL);
  }

  int fd_ = -1, port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mu_;
  std::vector<Request> reqs_;
};

Record MakeRecord(uint64_t real_ns, uint64_t fw, uint64_t rd1, uint64_t wr1) {
  Record r{};
  r.real_ns = real_ns;
  r.s.fw = fw;
  r.s.rd[1] = rd1;
  r.s.wr[1] = wr1;
  return r;
}

// A fake /sys/class/drm with two connected GPUs (port 5 <-> port 5) backed by `metrics`.
void MakeFakeDrm(const fs::path& root, const std::string& metrics) {
  const struct {
    const char* card;
    const char* pci;
    const char* ports;
  } cards[] = {{"card0", "0000:05:00.0", "01:05 ->  02:05\n"},
               {"card8", "0000:15:00.0", "02:05 ->  01:05\n"}};
  fs::create_directories(root / "drm");
  for (const auto& c : cards) {
    const fs::path dev = root / "devices" / c.pci;
    fs::create_directories(dev);
    std::ofstream(dev / "gpu_metrics", std::ios::binary) << metrics;
    std::ofstream(dev / "xgmi_device_id") << "1\n";
    std::ofstream(dev / "xgmi_port_num") << c.ports;
    fs::create_directories(root / "drm" / c.card);
    fs::create_directory_symlink(dev, root / "drm" / c.card / "device");
  }
  fs::create_directories(root / "drm" / "card0-DP-1");  // connectors and render nodes are ignored
  fs::create_directories(root / "drm" / "renderD128");
}

// ---------------------------------------------------------------------------------------------

void TestLayoutV18() {
  const std::string raw = Fixture("gpu_metrics_v1_8.bin");
  Layout l;
  std::string err;
  CHECK(ParseLayout(U8(raw), raw.size(), &l, &err));
  CHECK(l.rd == 136 && l.wr == 200 && l.fw == 288 && l.width == 72 && l.speed == 74);
  Snapshot s;
  ReadSnapshot(U8(raw), l, &s);
  CHECK(s.fw == 243647961296168ull);
  CHECK(s.rd[0] == 0 && s.rd[1] == 1233066902317ull && s.rd[7] == 1237512831488ull);
  CHECK(s.wr[5] == 1851049654948ull);
  CHECK(LinkMaxBytesPerSecond(U8(raw), l) == 72e9);
}

void TestLayoutV19() {
  const std::string raw = Fixture("gpu_metrics_v1_9.bin");
  Layout l;
  std::string err;
  CHECK(ParseLayout(U8(raw), raw.size(), &l, &err));
  CHECK(l.rd == 376 && l.wr == 448 && l.fw == 544 && l.width == 228 && l.speed == 238);
  Snapshot s;
  ReadSnapshot(U8(raw), l, &s);
  CHECK(s.fw == 189831201677852ull);
  CHECK(s.rd[3] == 1366248931244ull);
  CHECK(s.wr[1] == 541954904484ull && s.wr[7] == 541451658948ull);
  CHECK(LinkMaxBytesPerSecond(U8(raw), l) == 72e9);
}

void TestLayoutErrors() {
  std::string err;
  Layout l;
  std::string raw = Fixture("gpu_metrics_v1_8.bin");
  raw[3] = 7;
  CHECK(!ParseLayout(U8(raw), raw.size(), &l, &err) && err.find("v1.7") != std::string::npos);

  raw = Fixture("gpu_metrics_v1_9.bin");
  CHECK(!ParseLayout(U8(raw), 600, &l, &err));  // cut inside the attribute list

  raw = Fixture("gpu_metrics_v1_9.bin");
  uint64_t enc;
  std::memcpy(&enc, raw.data() + 376 - 8, 8);    // encoding of xgmi_read_data_acc
  enc = (enc & ~(0xFull << 20)) | (4ull << 20);  // claim u32
  std::memcpy(&raw[376 - 8], &enc, 8);
  CHECK(!ParseLayout(U8(raw), raw.size(), &l, &err) &&
        err.find("xgmi_read_data_acc") != std::string::npos);

  CHECK(!ParseLayout(U8(raw), 3, &l, &err));
}

void TestPortNum() {
  std::vector<PortLine> p;
  std::string err;
  CHECK(ParsePortNum(Fixture("xgmi_port_num_card0.txt"), &p, &err));
  CHECK(p.size() == 7);
  CHECK(p[0].src_node == 1 && p[0].src_port == 5 && p[0].dst_node == 2);
  CHECK(p[6].src_port == 6 && p[6].dst_node == 8);
  p.clear();
  CHECK(!ParsePortNum("01:xx -> 02:05\n", &p, &err));
}

void TestDiscover() {
  TempDir t;
  MakeFakeDrm(t.path, Fixture("gpu_metrics_v1_8.bin"));
  std::vector<Gpu> gpus;
  std::string err;
  CHECK(Discover((t.path / "drm").string(), &gpus, &err));
  CHECK(gpus.size() == 2);
  if (gpus.size() != 2) return;
  CHECK(gpus[0].card == "card0" && gpus[1].card == "card8");
  CHECK(gpus[0].pci == "0000:05:00.0" && gpus[1].pci == "0000:15:00.0");
  CHECK(gpus[0].peer[5] == 1 && gpus[1].peer[5] == 0);
  CHECK(gpus[0].peer[1] == -1);
  CHECK(gpus[0].link_max == 72e9);
}

void TestRates() {
  int peer[kNumLinks] = {-1, 4, -1, -1, -1, -1, -1, -1};
  RateCalc c("exp\"1", "h1", 0, peer, 72e9);
  std::string out;
  const uint64_t t = 1000000000123456789ull;
  c.Add(MakeRecord(t, 1000, 100, 200), &out);
  CHECK(out.empty());

  c.Add(MakeRecord(t + 2000000, 1000 + 200000, 1100, 200), &out);  // 2 ms, 1000 KB received
  const std::string L = "{experiment=\"exp\\\"1\",host=\"h1\",gpu=\"0\"";
  CHECK_STR(out, "xgmi_interval_seconds" + L +
                     "} 0.002000000 1000000000125\n"
                     "xgmi_bytes" +
                     L +
                     ",peer=\"4\",dir=\"rd\"} 1024000 1000000000125\n"
                     "xgmi_rate_bytes_per_second" +
                     L +
                     ",peer=\"4\",dir=\"rd\"} 5.12e+08 1000000000125\n"
                     "xgmi_bytes" +
                     L +
                     ",peer=\"4\",dir=\"wr\"} 0 1000000000125\n"
                     "xgmi_rate_bytes_per_second" +
                     L + ",peer=\"4\",dir=\"wr\"} 0 1000000000125\n");

  out.clear();
  c.Add(MakeRecord(t + 4000000, 1000 + 200000, 1200, 200), &out);  // same firmware_timestamp
  CHECK(out.empty() && c.stats().stale == 1);

  out.clear();
  c.Add(MakeRecord(t + 6000000, 1000 + 400000, 50, 300), &out);  // rd went down
  CHECK(c.stats().decreases == 1);
  CHECK(out.find("dir=\"rd\"") == std::string::npos &&
        out.find("dir=\"wr\"} 102400 ") != std::string::npos);

  out.clear();
  // 2 ms at 2x 72 GB/s = 288 MB = 281250 KB; 400000 KB is not physical.
  c.Add(MakeRecord(t + 8000000, 1000 + 600000, 50, 300 + 400000), &out);
  CHECK(c.stats().implausible == 1);
  CHECK(out.find("dir=\"wr\"") == std::string::npos &&
        out.find("dir=\"rd\"} 0 ") != std::string::npos);
  CHECK(c.stats().intervals == 3);
}

void TestRecordingFile() {
  TempDir t;
  Gpu g;
  g.idx = 3;
  g.card = "card24";
  g.pci = "0000:75:00.0";
  g.link_max = 72e9;
  for (int i = 0; i < kNumLinks; ++i) g.peer[i] = i == 0 ? -1 : 7 - i;
  const FileHeader h = MakeHeader(g, "n06-21", "exp42", 2000000);
  std::vector<Record> recs = {MakeRecord(1, 2, 3, 4), MakeRecord(5, 6, 7, 8),
                              MakeRecord(9, 10, 11, 12)};
  const std::string path = (t.path / "gpu3.bin").string();
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(Record));
    f.write("0123456789", 10);  // killed mid-write
  }
  FileHeader rh;
  std::vector<Record> got;
  size_t partial = 0;
  std::string err;
  CHECK(ReadRecording(path, &rh, &got, &partial, &err));
  CHECK(got.size() == 3 && partial == 10);
  CHECK(std::memcmp(got.data(), recs.data(), 3 * sizeof(Record)) == 0);
  CHECK(rh.gpu_idx == 3 && rh.peer[0] == -1 && rh.peer[1] == 6 && rh.link_max == 72e9);
  CHECK_STR(rh.host, "n06-21");
  CHECK_STR(rh.experiment, "exp42");
  CHECK_STR(rh.pci, "0000:75:00.0");

  std::ofstream(path, std::ios::binary) << std::string(600, 'x');
  CHECK(!ReadRecording(path, &rh, &got, &partial, &err) &&
        err.find("not an xgmi-netmon") != std::string::npos);
}

void TestRing() {
  SpscRing<int> r(3);
  CHECK(r.Push(1) && r.Push(2) && r.Push(3) && !r.Push(4));
  int out[8];
  CHECK(r.PopMany(out, 8) == 3 && out[0] == 1 && out[2] == 3);
  CHECK(r.PopMany(out, 8) == 0);

  SpscRing<uint64_t> q(1000);
  constexpr uint64_t kN = 1000000;
  std::thread producer([&] {
    for (uint64_t i = 0; i < kN;)
      if (q.Push(i)) ++i;
  });
  uint64_t expect = 0, buf[256];
  bool ordered = true;
  while (expect < kN) {
    const size_t n = q.PopMany(buf, 256);
    for (size_t k = 0; k < n; ++k) ordered &= buf[k] == expect++;
  }
  producer.join();
  CHECK(ordered && expect == kN);
}

void TestUrl() {
  Url u;
  std::string err;
  CHECK(ParseUrl("http://127.0.0.1:28428", &u, &err) && u.host == "127.0.0.1" &&
        u.port == "28428" && u.prefix.empty());
  CHECK(ParseUrl("http://vm.example/", &u, &err) && u.host == "vm.example" && u.port == "80" &&
        u.prefix.empty());
  CHECK(ParseUrl("http://[::1]:9/sub/", &u, &err) && u.host == "::1" && u.port == "9" &&
        u.prefix == "/sub");
  CHECK(!ParseUrl("https://x", &u, &err));
  CHECK(UrlEncode("a{b=\"c\"}") == "a%7Bb%3D%22c%22%7D");
}

void TestVmCalls() {
  FakeVm vm;
  Url u;
  std::string err;
  CHECK(ParseUrl(vm.url(), &u, &err));
  CHECK(VmHealthy(u, &err));
  CHECK(VmHasSeries(u, "e", "h", &err) == 0);
  vm.series_body = R"({"status":"success","data":[{"__name__":"xgmi_interval_seconds"}]})";
  CHECK(VmHasSeries(u, "e", "h", &err) == 1);
  const std::string body = std::string(3 << 20, 'x') + "\n";  // larger than socket buffers
  CHECK(VmImport(u, body, 5000, &err));
  CHECK(vm.imported() == body);
  CHECK(VmDeleteSeries(u, "e", "h", &err));
  const auto reqs = vm.requests();
  CHECK(reqs.back().method == "POST" &&
        reqs.back().path.find(
            "delete_series?match[]=%7B__name__%3D~%22xgmi_.%2A%22%2Cexperiment%3D%22e%22") !=
            std::string::npos);
  vm.import_status = 400;
  CHECK(!VmImport(u, "x\n", 2000, &err) && err.find("HTTP 400") != std::string::npos);

  vm.hang = true;  // accepts, never answers
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(!VmImport(u, "x\n", 300, &err));
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CHECK(ms >= 250 && ms < 1000);

  CHECK(!ParseUrl("http://127.0.0.1:1", &u, &err) ||
        !VmHealthy(u, &err));  // nothing listens on port 1
}

void TestImport() {
  TempDir t;
  Gpu g;
  g.idx = 0;
  g.card = "card0";
  g.link_max = 72e9;
  for (int i = 0; i < kNumLinks; ++i) g.peer[i] = i == 1 ? 4 : -1;
  const FileHeader h = MakeHeader(g, "h1", "e1", 2000000);
  std::vector<Record> recs;
  for (uint64_t i = 0; i < 100; ++i)
    recs.push_back(
        MakeRecord(1000000000000000000ull + i * 2000000, 1000 + i * 200000, i * 10, i * 20));
  fs::create_directories(t.path / "e1" / "h1");
  {
    std::ofstream f(t.path / "e1" / "h1" / "gpu0.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(Record));
  }
  std::string expect;
  RateCalc c("e1", "h1", 0, g.peer, g.link_max);
  for (const Record& r : recs) c.Add(r, &expect);

  FakeVm vm;
  ImportOptions o;
  o.dirs = {(t.path / "e1" / "h1").string()};
  o.url = vm.url();
  CHECK(RunImport(o) == 0);
  CHECK(vm.imported() == expect);  // same points as the live push would write

  vm.series_body = R"({"status":"success","data":[{"__name__":"xgmi_interval_seconds"}]})";
  CHECK(RunImport(o) == 1);  // data present, no --replace
  o.replace = true;
  CHECK(RunImport(o) == 0);
  bool deleted = false;
  for (const auto& r : vm.requests()) deleted |= r.path.find("delete_series") != std::string::npos;
  CHECK(deleted);
}

void TestRecordEndToEnd() {
  // Static fixture files: each GPU's firmware_timestamp never changes, so one snapshot per GPU.
  TempDir t;
  MakeFakeDrm(t.path, Fixture("gpu_metrics_v1_9.bin"));
  FakeVm vm;
  RecordOptions o;
  o.out = (t.path / "exp7" / "{host}").string();
  o.duration_s = 0.4;
  o.push_url = vm.url();
  o.host_id = "nodeA";
  o.drm_root = (t.path / "drm").string();
  CHECK(RunRecord(o) == 0);
  const fs::path dir = t.path / "exp7" / "nodeA";
  for (const char* f : {"gpu0.bin", "gpu1.bin"}) {
    FileHeader h;
    std::vector<Record> recs;
    size_t partial = 0;
    std::string err;
    CHECK(ReadRecording((dir / f).string(), &h, &recs, &partial, &err));
    CHECK(recs.size() == 1 && partial == 0);
    CHECK_STR(h.experiment, "exp7");
    CHECK_STR(h.host, "nodeA");
    if (!recs.empty())
      CHECK(recs[0].s.fw == 189831201677852ull && recs[0].t1_mono_ns >= recs[0].t0_mono_ns);
  }
  const std::string meta = ReadFile((dir / "meta.json").string());
  CHECK(meta.find("\"status\": \"complete\"") != std::string::npos);
  CHECK(meta.find("\"peers\": {\"5\": 1}") != std::string::npos);
  CHECK(fs::exists(dir / ".lock"));
  // A recording is never overwritten.
  CHECK(RunRecord(o) == 1);
}

}  // namespace

int main() {
  const std::pair<const char*, std::function<void()>> tests[] = {
      {"layout v1.8", TestLayoutV18},
      {"layout v1.9", TestLayoutV19},
      {"layout errors", TestLayoutErrors},
      {"xgmi_port_num", TestPortNum},
      {"discover", TestDiscover},
      {"rates", TestRates},
      {"recording file", TestRecordingFile},
      {"ring", TestRing},
      {"url", TestUrl},
      {"vm calls", TestVmCalls},
      {"import", TestImport},
      {"record end to end", TestRecordEndToEnd},
  };
  for (const auto& t : tests) {
    const int before = g_failures;
    t.second();
    std::fprintf(stderr, "%s %s\n", g_failures == before ? "ok  " : "FAIL", t.first);
  }
  std::fprintf(stderr, "%s (%d failed checks)\n", g_failures ? "FAILED" : "all passed", g_failures);
  return g_failures ? 1 : 0;
}
