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
#include "exporter.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "live_rates.h"
#include "meta_json.h"
#include "util.h"

namespace netmon {

// Keep in sync with KEY_COUNTERS in report/ainic_netmon_import.py (checked by test_live.py).
const char* const kKeyCountersRdma[] = {"tx_rdma_ucast_bytes",
                                        "rx_rdma_ucast_bytes",
                                        "tx_rdma_ucast_pkts",
                                        "rx_rdma_ucast_pkts",
                                        "tx_rdma_cnp_pkts",
                                        "rx_rdma_cnp_pkts",
                                        "rx_rdma_ecn_pkts",
                                        "tx_rdma_retx_bytes",
                                        "tx_rdma_retx_pkts",
                                        "tx_rdma_ack_timeout",
                                        nullptr};
const char* const kKeyCountersIpc[] = {"octets_tx_ok", "octets_rx_ok", "frames_tx_ok",
                                       "frames_rx_ok", nullptr};

namespace {

constexpr uint64_t kSlotsPerStream = 200;
constexpr uint64_t kMaxRingBytes = 1ull << 30;
constexpr uint64_t kMaxPendingPoints = 500000;
constexpr uint64_t kRequestTimeoutNs = 3000000000ull;
constexpr uint64_t kWarnEveryNs = 10000000000ull;
constexpr const char* kAllTimeStart = "2000-01-01T00:00:00Z";  // VM: start=0 means "unset"

void SleepFor(uint64_t ns) {
  timespec ts{static_cast<time_t>(ns / 1000000000ull), static_cast<long>(ns % 1000000000ull)};
  while (nanosleep(&ts, &ts) != 0) {
  }
}

std::string Q(const std::string& s) { return "\"" + JsonWriter::Escape(s) + "\""; }

std::string Num(double v) {
  char b[40];
  snprintf(b, sizeof(b), "%.17g", v);
  return b;
}

// PromQL / MetricsQL string literal for a selector.
std::string PromStr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') {
      o += "\\n";
      continue;
    }
    o += c;
  }
  return o + "\"";
}

uint64_t Fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// VM flag value is a zero duration ("0", "0s", "0ms", ...).
bool ZeroDuration(const std::string& v) {
  if (v.empty()) return true;
  size_t i = 0;
  while (i < v.size() && (v[i] == '0' || v[i] == '.')) i++;
  return i > 0 && (i == v.size() || isalpha(static_cast<unsigned char>(v[i])));
}

// "-name=value" or -name="value" lines of VM's /flags page.
std::string FlagValue(const std::string& flags, const std::string& name) {
  const std::string key = "-" + name + "=";
  size_t p = 0;
  while ((p = flags.find(key, p)) != std::string::npos) {
    if (p == 0 || flags[p - 1] == '\n') {
      size_t e = flags.find('\n', p);
      std::string v = flags.substr(p + key.size(), e == std::string::npos ? e : e - p - key.size());
      if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
      return v;
    }
    p += key.size();
  }
  return "<default>";
}

}  // namespace

std::vector<size_t> SelectPushCounters(const std::string& source,
                                       const std::vector<std::string>& names, bool all,
                                       std::vector<bool>* optional) {
  static const std::regex kOptional(
      "(pripause_\\d+_1us_count|rx_pause_1us_count|frames_(tx|rx)_pri_\\d)$");
  const char* const* key = source == "rdma"  ? kKeyCountersRdma
                           : source == "ipc" ? kKeyCountersIpc
                                             : nullptr;
  std::vector<size_t> cols;
  optional->clear();
  if (!key) return cols;
  for (const char* const* k = key; *k; k++) {
    for (size_t i = 0; i < names.size(); i++) {
      if (names[i] == *k) {
        cols.push_back(i);
        optional->push_back(false);
      }
    }
  }
  for (size_t i = 0; i < names.size(); i++) {
    if (std::find(cols.begin(), cols.end(), i) != cols.end()) continue;
    if (all || std::regex_search(names[i], kOptional)) {
      cols.push_back(i);
      optional->push_back(true);
    }
  }
  return cols;
}

// ------------------------------------------------------------------ AsyncLog

AsyncLog::AsyncLog() {
  std::thread([this] { Main(); }).detach();  // lives until the process ends
}

void AsyncLog::Post(std::string msg) {
  std::lock_guard<std::mutex> lk(mu_);
  if (q_.size() >= 8) {
    dropped_.fetch_add(1);
    return;
  }
  q_.push_back(std::move(msg));
  cv_.notify_one();
}

bool AsyncLog::Idle() {
  std::lock_guard<std::mutex> lk(mu_);
  return q_.empty() && !writing_;
}

void AsyncLog::Main() {
  for (;;) {
    std::string msg;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return !q_.empty(); });
      msg = std::move(q_.front());
      q_.erase(q_.begin());
      writing_ = true;  // set together with the dequeue: Idle() never sees a gap
    }
    size_t off = 0;
    while (off < msg.size()) {
      const ssize_t w = write(STDERR_FILENO, msg.data() + off, msg.size() - off);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) break;
      off += static_cast<size_t>(w);
    }
    std::lock_guard<std::mutex> lk(mu_);
    writing_ = false;
  }
}

// ------------------------------------------------------------------ Exporter

struct Exporter::StreamState {
  std::unique_ptr<LiveRates> lr;
  size_t series_base = 0;  // 8 series per counter, see Sink
  size_t counters = 0;
};

// Series layout per counter: rate/increase/interval x normal/low (0..5), gap reset (6),
// gap clock (7).
class Exporter::Sink : public LiveSink {
 public:
  Sink(Exporter* e, size_t base) : e_(e), base_(base) {}
  void Interval(size_t c, bool low, int64_t end_ms, double rate, double inc, double dt) override {
    const size_t s = base_ + c * 8 + (low ? 3 : 0);
    Put(s, end_ms, rate);
    Put(s + 1, end_ms, inc);
    Put(s + 2, end_ms, dt);
  }
  void Gap(size_t c, bool clock, int64_t end_ms, double dt) override {
    Put(base_ + c * 8 + (clock ? 7 : 6), end_ms, dt);
  }

 private:
  void Put(size_t s, int64_t ts, double v) {
    e_->series_[s].ts.push_back(ts);
    e_->series_[s].v.push_back(v);
    e_->pending_points_++;
  }
  Exporter* e_;
  size_t base_;
};

std::atomic<bool> Exporter::test_throw{false};

Exporter::Exporter(PushOptions opt) : opt_(std::move(opt)) {}

void Exporter::Start() {
  std::thread([this] { Main(); }).detach();  // see design §6: never joined, never freed
}

Exporter::Phase Exporter::phase() {
  std::lock_guard<std::mutex> lk(mu_);
  return phase_;
}

std::string Exporter::preflight_error() {
  if (crashed_.load()) return "internal error in the live push";
  std::lock_guard<std::mutex> lk(mu_);
  return preflight_error_;
}

void Exporter::SetError(const std::string& e) {
  std::lock_guard<std::mutex> lk(err_mu_);
  last_error_ = e;
}

void Exporter::Warn(const std::string& e) {
  const uint64_t now = MonoNs();
  if (last_warn_ && now - last_warn_ < kWarnEveryNs) return;
  last_warn_ = now;
  log_.Post("warning: push: " + e + "\n");
}

bool Exporter::Preflight(std::string* error) {
  // 1. one live push per (VM, experiment, host) on this network namespace
  const std::string key = opt_.url + "\n" + opt_.experiment + "\n" + opt_.host;
  char name[64];
  snprintf(name, sizeof(name), "ainic-netmon-push:%016llx",
           static_cast<unsigned long long>(Fnv1a(key)));
  lock_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (lock_fd_ < 0) {
    *error = std::string("socket: ") + strerror(errno);
    return false;
  }
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  const size_t n = strlen(name);
  memcpy(a.sun_path + 1, name, n);  // sun_path[0] = 0: abstract namespace, no file
  if (bind(lock_fd_, reinterpret_cast<sockaddr*>(&a),
           static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + n)) != 0) {
    *error = errno == EADDRINUSE ? "another ainic-netmon is already pushing experiment '" +
                                       opt_.experiment + "' host '" + opt_.host + "' to " + opt_.url
                                 : std::string("bind: ") + strerror(errno);
    return false;
  }
  // 2. reach VictoriaMetrics and check its flags
  HttpUrl u;
  if (!ParseHttpUrl(opt_.url, &u, error) || !http_.Resolve(u, error)) return false;
  HttpResult r = http_.Request("GET", "/flags", "", nullptr, MonoNs() + kRequestTimeoutNs);
  if (r.kind != HttpResult::Kind::kOk || r.status != 200) {
    *error = "VictoriaMetrics at " + opt_.url + " not reachable (" +
             (r.error.empty() ? "HTTP " + std::to_string(r.status) : r.error) + ")";
    return false;
  }
  const std::string dedup = FlagValue(r.body, "dedup.minScrapeInterval");
  if (dedup != "<default>" && !ZeroDuration(dedup)) {
    *error = "VictoriaMetrics dedup.minScrapeInterval=" + dedup + " would merge ms samples";
    return false;
  }
  const std::string lat = FlagValue(r.body, "search.latencyOffset");
  if (lat == "<default>" || !ZeroDuration(lat)) {
    log_.Post("warning: VictoriaMetrics search.latencyOffset=" +
              (lat == "<default>" ? std::string("30s (default)") : lat) +
              ": the live view lags by that much (the netmon deployment sets 0s)\n");
  }
  // 3. this host must not have data in this experiment already
  const std::string match = "{__name__=~\"netmon_.+\",experiment=" + PromStr(opt_.experiment) +
                            ",host=" + PromStr(opt_.host) + "}";
  r = http_.Request(
      "GET", "/api/v1/series?match[]=" + UrlEncode(match) + "&start=" + UrlEncode(kAllTimeStart),
      "", nullptr, MonoNs() + kRequestTimeoutNs);
  if (r.kind != HttpResult::Kind::kOk || r.status != 200) {
    *error = "VictoriaMetrics series query failed (" +
             (r.error.empty() ? "HTTP " + std::to_string(r.status) : r.error) + ")";
    return false;
  }
  if (r.body.find("\"data\":[]") == std::string::npos) {
    *error = "experiment '" + opt_.experiment + "' already has data for host '" + opt_.host +
             "' in VictoriaMetrics: use another --experiment (or delete it first)";
    return false;
  }
  return true;
}

bool Exporter::Configure(std::vector<ExportStream> streams, uint64_t start_real_ns,
                         std::string* error) {
  try {
    return ConfigureImpl(std::move(streams), start_real_ns, error);
  } catch (const std::exception& e) {  // e.g. std::bad_alloc: the recording aborts cleanly
    *error = std::string("cannot set up the live push: ") + e.what();
    return false;
  }
}

bool Exporter::ConfigureImpl(std::vector<ExportStream> streams, uint64_t start_real_ns,
                             std::string* error) {
  // Rings first, with an overflow-checked byte bound.
  uint64_t bytes = 0;
  std::vector<std::unique_ptr<Ring>> rings;
  std::vector<std::vector<bool>> optional(streams.size());
  for (size_t i = 0; i < streams.size(); i++) {
    auto r = std::make_unique<Ring>();
    r->cols =
        SelectPushCounters(streams[i].source, streams[i].names, opt_.all_counters, &optional[i]);
    r->slots = kSlotsPerStream;
    r->slot_records = std::max<size_t>(streams[i].batch_records, 1);
    r->width = 4 + r->cols.size();
    const uint64_t b = static_cast<uint64_t>(r->slots) * r->slot_records * r->width * 8;
    if (r->slot_records > (1u << 20) || r->width > 4096 || b > kMaxRingBytes ||
        bytes + b > kMaxRingBytes) {
      *error = "live push buffers would exceed 1 GiB: push fewer streams or counters";
      return false;
    }
    bytes += b;
    rings.push_back(std::move(r));
  }
  for (auto& r : rings) {
    r->data.assign(r->slots * r->slot_records * r->width, 0);  // allocated before sampling
    r->counts.assign(r->slots, 0);
  }
  // Series and online state.
  const std::string& host = opt_.host;
  for (size_t i = 0; i < streams.size(); i++) {
    const ExportStream& s = streams[i];
    auto st = std::make_unique<StreamState>();
    st->counters = rings[i]->cols.size();
    st->series_base = series_.size();
    const size_t pending_cap =
        s.interval_ns ? std::max<uint64_t>(1000000000ull / s.interval_ns, 2) : 1024;
    st->lr = std::make_unique<LiveRates>(s.interval_ns, s.snapshot_fraction, st->counters,
                                         optional[i], pending_cap);
    const std::string base = "\"experiment\":" + Q(opt_.experiment) + ",\"host\":" + Q(host) +
                             ",\"netdev\":" + Q(s.netdev) +
                             ",\"target\":" + Q(host + "/" + s.netdev) +
                             ",\"source\":" + Q(s.source);
    for (size_t c = 0; c < st->counters; c++) {
      const std::string lab = base + ",\"counter\":" + Q(s.names[rings[i]->cols[c]]);
      for (const char* res : {"normal", "low"}) {
        for (const char* m : {"netmon_rate", "netmon_increase", "netmon_interval_seconds"}) {
          Series x;
          x.metric = "{\"__name__\":" + Q(m) + "," + lab + ",\"resolution\":" + Q(res) + "}";
          series_.push_back(std::move(x));
        }
      }
      for (const char* reason : {"reset", "clock"}) {
        Series x;
        x.metric = "{\"__name__\":\"netmon_gap_seconds\"," + lab + ",\"reason\":" + Q(reason) + "}";
        series_.push_back(std::move(x));
      }
    }
    streams_.push_back(std::move(st));
  }
  anchor_ms_ = static_cast<int64_t>(start_real_ns / 1000000ull) - 1;
  auto rings_or_cols_empty = [](const std::vector<std::unique_ptr<Ring>>& r, size_t i) {
    return r[i]->cols.empty();
  };
  int64_t prelim = 0;
  for (size_t i = 0; i < streams.size(); i++) {
    if (rings_or_cols_empty(rings, i)) continue;  // not pushed (e.g. ethtool, 1 s)
    const ExportStream& s = streams[i];
    const double ms = std::max<double>(static_cast<double>(s.interval_ns), 2e6) / 1e6;
    prelim = std::max<int64_t>(prelim, static_cast<int64_t>(std::ceil(3 * ms - 1e-9)));
  }
  step_prelim_ = prelim;
  step_published_ = 0;
  step_pending_ = 0;
  cfg_ = std::move(streams);
  rings_ = std::move(rings);
  {
    std::lock_guard<std::mutex> lk(mu_);
    phase_ = Phase::kRunning;
  }
  cv_.notify_all();
  return true;
}

std::string Exporter::Describe() const {
  uint64_t bytes = 0, pps = 0;
  for (size_t i = 0; i < rings_.size(); i++) {
    bytes += rings_[i]->data.size() * 8;
    const uint64_t hz = cfg_[i].interval_ns ? 1000000000ull / cfg_[i].interval_ns : 600;
    pps += hz * rings_[i]->cols.size() * 3;
  }
  char b[256];
  snprintf(b, sizeof(b),
           "live push to %s (experiment %s, host %s): buffers %.1f MiB, up to ~%llu points/s "
           "(~%.1f MB/s)\n",
           opt_.url.c_str(), opt_.experiment.c_str(), opt_.host.c_str(),
           static_cast<double>(bytes) / (1 << 20), static_cast<unsigned long long>(pps),
           static_cast<double>(pps) * 25 / 1e6);
  return b;
}

std::vector<std::string> Exporter::DashboardUrls(const std::string& grafana) const {
  std::string g = grafana;
  while (!g.empty() && g.back() == '/') g.pop_back();
  const std::string q =
      "?from=now-5m&to=now&refresh=2s&var-experiment=" + UrlEncode(opt_.experiment);
  return {g + "/d/netmon-overview/netmon-overview" + q, g + "/d/netmon-nic/netmon-nic-detail" + q};
}

void Exporter::Abort() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    abort_ = true;
  }
  cv_.notify_all();
}

void Exporter::Offer(const Batch& b) {
  if (b.stream < 0 || static_cast<size_t>(b.stream) >= rings_.size()) return;
  Ring& r = *rings_[static_cast<size_t>(b.stream)];
  if (r.cols.empty()) return;  // nothing pushed from this stream (ethtool)
  const uint64_t h = r.head.load(std::memory_order_relaxed);
  if (h - r.tail.load(std::memory_order_acquire) >= r.slots) {
    r.dropped.fetch_add(b.count, std::memory_order_relaxed);
    return;
  }
  const size_t slot = static_cast<size_t>(h % r.slots);
  uint64_t* dst = &r.data[slot * r.slot_records * r.width];
  const size_t n = std::min(b.count, r.slot_records);
  for (size_t k = 0; k < n; k++) {
    const uint64_t* rec = b.Record(k);
    dst[0] = rec[0];
    dst[1] = rec[1];
    dst[2] = rec[2];
    dst[3] = rec[3];
    for (size_t j = 0; j < r.cols.size(); j++) dst[4 + j] = rec[4 + r.cols[j]];
    dst += r.width;
  }
  if (b.count > n) r.dropped.fetch_add(b.count - n, std::memory_order_relaxed);
  r.counts[slot] = n;
  r.head.store(h + 1, std::memory_order_release);
}

void Exporter::CloseInput() { input_closed_.store(true, std::memory_order_release); }

bool Exporter::WaitDone(uint64_t deadline) {
  std::unique_lock<std::mutex> lk(mu_);
  while (phase_ != Phase::kDone) {
    const uint64_t now = MonoNs();
    if (now >= deadline) return false;
    cv_.wait_for(lk, std::chrono::nanoseconds(deadline - now));
  }
  return true;
}

PushStats Exporter::Snapshot() {
  PushStats s;
  s.points_sent = points_sent_.load();
  s.requests = requests_.load();
  s.failed_points = failed_points_.load();
  for (const auto& r : rings_) s.dropped_samples += r->dropped.load();
  s.unconfirmed_dropped = unconfirmed_.load();
  s.clock_back_intervals = clock_back_.load();
  s.log_dropped = log_.dropped();
  {
    std::unique_lock<std::mutex> lk(mu_, std::try_to_lock);
    s.abandoned = !lk.owns_lock() || phase_ != Phase::kDone;
  }
  if (crashed_.load()) s.last_error = "exporter stopped by an internal error (allocation?)";
  for (int i = 0; i < 10 && s.last_error.empty(); i++) {  // never wait (design §6)
    std::unique_lock<std::mutex> lk(err_mu_, std::try_to_lock);
    if (lk.owns_lock()) {
      s.last_error = last_error_;
      break;
    }
    SleepFor(1000000ull);
  }
  return s;
}

void Exporter::Main() {
  // Nothing may escape this thread (std::terminate would end the recording): any exception
  // only stops the push. The handler does not allocate.
  bool started = false;
  try {
    std::string err;
    const bool ok = Preflight(&err);
    {
      std::unique_lock<std::mutex> lk(mu_);
      phase_ = ok ? Phase::kReady : Phase::kFailed;
      preflight_error_ = err;
      cv_.notify_all();
      if (!ok) return;
      cv_.wait(lk, [&] { return phase_ == Phase::kRunning || abort_; });
      if (phase_ != Phase::kRunning) {
        phase_ = Phase::kDone;
        cv_.notify_all();
        return;
      }
    }
    started = true;
    Run();
  } catch (...) {
    crashed_.store(true);
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    // Reached when Run() returned or anything threw (the early returns above set the phase).
    phase_ = started ? Phase::kDone : Phase::kFailed;
  }
  cv_.notify_all();
}

void Exporter::ProcessRing(size_t i) {
  Ring& r = *rings_[i];
  StreamState& st = *streams_[i];
  Sink sink(this, st.series_base);
  uint64_t t = r.tail.load(std::memory_order_relaxed);
  const uint64_t h = r.head.load(std::memory_order_acquire);
  for (; t < h; t++) {
    const size_t slot = static_cast<size_t>(t % r.slots);
    const uint64_t* rec = &r.data[slot * r.slot_records * r.width];
    for (size_t k = 0; k < r.counts[slot]; k++, rec += r.width) {
      st.lr->Add(rec, &sink);
      if (pending_points_ >= kMaxPendingPoints) Flush(false);
    }
    r.tail.store(t + 1, std::memory_order_release);  // slot may be reused from here on
  }
}

void Exporter::Run() {
  uint64_t next_push = MonoNs() + opt_.interval_ns;
  uint64_t next_status = MonoNs() + kWarnEveryNs;
  uint64_t sent_at_status = 0;
  for (;;) {
    if (test_throw.load()) throw std::runtime_error("test");
    const bool closed = input_closed_.load(std::memory_order_acquire);
    for (size_t i = 0; i < rings_.size(); i++) ProcessRing(i);  // drained if closed
    if (closed) {
      for (size_t i = 0; i < streams_.size(); i++) {
        Sink sink(this, streams_[i]->series_base);
        streams_[i]->lr->Finish(&sink);
      }
      UpdateStep();
      Flush(true);
      return;
    }
    const uint64_t now = MonoNs();
    if (now >= next_push) {
      UpdateStep();
      Flush(false);
      next_push = now + opt_.interval_ns;
    }
    if (!opt_.quiet && now >= next_status) {
      const uint64_t sent = points_sent_.load();
      char b[160];
      snprintf(b, sizeof(b), "[push] %.0f points/s, failed %llu, dropped samples %llu\n",
               static_cast<double>(sent - sent_at_status) * 1e9 /
                   static_cast<double>(now - next_status + kWarnEveryNs),
               static_cast<unsigned long long>(failed_points_.load()),
               static_cast<unsigned long long>(Snapshot().dropped_samples));
      log_.Post(b);
      sent_at_status = sent;
      next_status = now + kWarnEveryNs;
    }
    SleepFor(5000000ull);
  }
}

void Exporter::UpdateStep() {
  int64_t step = 0;
  for (auto& st : streams_) {
    const double b = st->lr->BaselineNs();
    if (b <= 0) continue;  // not warmed up yet
    step = std::max<int64_t>(step, static_cast<int64_t>(std::ceil(3 * b / 1e6 - 1e-9)));
  }
  uint64_t unconfirmed = 0, back = 0;
  for (auto& st : streams_) {
    unconfirmed += st->lr->stats().unconfirmed_dropped;
    back += st->lr->stats().clock_back_intervals;
  }
  unconfirmed_.store(unconfirmed);
  clock_back_.store(back);
  // Only increases are published (design §4.3): a too-small step leaves empty windows.
  if (step > std::max({step_prelim_, step_published_, step_pending_})) {
    step_pending_ = step;
    step_pending_ts_ = static_cast<int64_t>(RealNs() / 1000000ull);
  }
}

void Exporter::MetaLines(std::string* body) {
  const std::string& host = opt_.host;
  auto line = [&](const std::string& metric, double v, int64_t ts) {
    *body += "{\"metric\":" + metric + ",\"values\":[" + Num(v) + "],\"timestamps\":[" +
             std::to_string(ts) + "]}\n";
  };
  if (!meta_published_) {
    std::vector<std::string> seen;
    for (const auto& s : cfg_) {
      if (std::find(seen.begin(), seen.end(), s.netdev) != seen.end()) continue;
      seen.push_back(s.netdev);
      line("{\"__name__\":\"netmon_target_info\",\"experiment\":" + Q(opt_.experiment) +
               ",\"host\":" + Q(host) + ",\"netdev\":" + Q(s.netdev) +
               ",\"target\":" + Q(host + "/" + s.netdev) +
               ",\"status\":\"live\",\"speed_mbps\":" + Q(std::to_string(s.speed_mbps)) + "}",
           1, anchor_ms_);
    }
    line("{\"__name__\":\"netmon_import_info\",\"experiment\":" + Q(opt_.experiment) +
             ",\"host\":" + Q(host) +
             ",\"state\":\"live\",\"anchor_ms\":" + Q(std::to_string(anchor_ms_)) + "}",
         1, anchor_ms_);
    line("{\"__name__\":\"netmon_step_ms\",\"experiment\":" + Q(opt_.experiment) +
             ",\"host\":" + Q(host) + "}",
         static_cast<double>(step_prelim_), anchor_ms_);
  }
  if (step_pending_) {
    line("{\"__name__\":\"netmon_step_ms\",\"experiment\":" + Q(opt_.experiment) +
             ",\"host\":" + Q(host) + "}",
         static_cast<double>(step_pending_), step_pending_ts_);
  }
}

void Exporter::Flush(bool final_push) {
  std::string body;
  uint64_t points = 0;
  bool with_meta = !meta_published_ || step_pending_;
  if (with_meta) MetaLines(&body);
  // Series come in groups of 8 per counter (stream bases are multiples of 8): rate,
  // increase and interval of one resolution (0-2, 3-5) always go into the same request, so
  // the client never splits an increase from its duration (the mean panels divide their
  // sums). A request that fails after sending may still be partly stored by the server; the
  // live mean of such a window is then unreliable (documented; the import is exact).
  // Gaps (6, 7) are single series.
  std::string chunk;
  uint64_t chunk_points = 0;
  for (size_t i = 0; i < series_.size(); i++) {
    Series& s = series_[i];
    if (!s.ts.empty()) {
      chunk += "{\"metric\":" + s.metric + ",\"values\":[";
      for (size_t k = 0; k < s.v.size(); k++) {
        if (k) chunk += ',';
        chunk += Num(s.v[k]);
      }
      chunk += "],\"timestamps\":[";
      for (size_t k = 0; k < s.ts.size(); k++) {
        if (k) chunk += ',';
        chunk += std::to_string(s.ts[k]);
      }
      chunk += "]}\n";
      chunk_points += s.ts.size();
      s.ts.clear();
      s.v.clear();
    }
    const size_t pos = i % 8;
    const bool group_end = pos == 2 || pos == 5 || pos >= 6;
    if (!group_end || chunk.empty()) continue;
    if (!body.empty() && body.size() + chunk.size() > opt_.max_body_bytes) {
      Send(&body, points, with_meta, final_push);
      with_meta = false;
      points = 0;
    }
    body += chunk;
    points += chunk_points;
    chunk.clear();
    chunk_points = 0;
  }
  pending_points_ = 0;
  if (!body.empty()) Send(&body, points, with_meta, final_push);
}

void Exporter::Send(std::string* body, uint64_t points, bool with_meta, bool final_push) {
  const uint64_t now = MonoNs();
  if (!final_push && now < backoff_until_) {
    failed_points_.fetch_add(points);  // backing off: dropped, counted
    body->clear();
    return;
  }
  HttpResult r;
  for (int attempt = 0; attempt < 3; attempt++) {
    r = http_.Request("POST", "/api/v1/import", *body, "application/json",
                      MonoNs() + kRequestTimeoutNs);
    // Retry only what certainly was not sent: the server never saw it (design §2).
    if (r.kind != HttpResult::Kind::kNotSent || attempt == 2) break;
    SleepFor(200000000ull);
  }
  body->clear();
  requests_.fetch_add(1);
  if (r.kind == HttpResult::Kind::kOk && r.status / 100 == 2) {
    points_sent_.fetch_add(points);
    backoff_ns_ = 0;
    backoff_until_ = 0;
    if (with_meta) {
      meta_published_ = true;
      step_published_ = std::max({step_published_, step_pending_, step_prelim_});
      step_pending_ = 0;
    }
    return;
  }
  failed_points_.fetch_add(points);
  const std::string e = r.kind == HttpResult::Kind::kOk
                            ? "HTTP " + std::to_string(r.status) + ": " + r.body.substr(0, 200)
                            : r.error;
  SetError(e);
  backoff_ns_ = backoff_ns_ ? std::min<uint64_t>(backoff_ns_ * 2, 10000000000ull) : 1000000000ull;
  backoff_until_ = MonoNs() + backoff_ns_;
  Warn(e + " (" + std::to_string(points) + " points dropped; retrying in " +
       std::to_string(backoff_ns_ / 1000000000ull) + " s)");
}

}  // namespace netmon
