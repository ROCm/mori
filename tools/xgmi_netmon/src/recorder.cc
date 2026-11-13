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
#include "recorder.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "rates.h"
#include "recording.h"
#include "ring.h"
#include "topology.h"
#include "vm.h"

namespace xgmi {
namespace {

std::atomic<bool> g_signal{false};
void OnSignal(int) { g_signal.store(true); }

uint64_t Now(clockid_t c) {
  timespec ts;
  clock_gettime(c, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

constexpr size_t kRingRecords = 8192;       // ~16 s at one snapshot per 2 ms
constexpr uint64_t kLongInterval = 300000;  // 3 ms in firmware_timestamp units (10 ns)

struct SamplerStats {
  uint64_t reads = 0, snapshots = 0, read_errors = 0, late = 0;
  uint64_t write_drops = 0, push_drops = 0;
  uint64_t long_intervals = 0, max_interval = 0;  // firmware_timestamp units
  uint64_t wake_sum_ns = 0, wake_max_ns = 0;      // read start after the grid point
};

struct Stream {
  Gpu gpu;
  int sys_fd = -1;
  int out_fd = -1;
  std::string file;
  std::unique_ptr<SpscRing<Record>> write_ring, push_ring;
  SamplerStats st;
  std::thread thread;
};

struct PushStats {
  uint64_t requests = 0, failed_requests = 0, pushed_points = 0, failed_points = 0;
  RateStats rates;
  std::string last_error;
};

void SamplerMain(Stream* s, uint64_t first, uint64_t period, const std::atomic<bool>* stop) {
  alignas(8) uint8_t buf[4096];
  const size_t need = RequiredSize(s->gpu.layout);
  uint64_t deadline = first, last_fw = 0;
  bool have = false;
  SamplerStats& st = s->st;
  while (!stop->load(std::memory_order_relaxed)) {
    timespec ts{static_cast<time_t>(deadline / 1000000000ull),
                static_cast<long>(deadline % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
    }
    const uint64_t woke = Now(CLOCK_MONOTONIC);
    if (woke >= deadline + period) {  // overslept: skip the grid points already past
      ++st.late;
      deadline += (woke - deadline) / period * period;
    }
    Record r;
    r.real_ns = Now(CLOCK_REALTIME);
    r.t0_mono_ns = Now(CLOCK_MONOTONIC);
    r.real_err_ns = r.t0_mono_ns - woke;
    const ssize_t n = pread(s->sys_fd, buf, sizeof(buf), 0);
    r.t1_mono_ns = Now(CLOCK_MONOTONIC);
    ++st.reads;
    const uint64_t wake = r.t0_mono_ns - deadline;
    st.wake_sum_ns += wake;
    st.wake_max_ns = std::max(st.wake_max_ns, wake);
    deadline += period;
    if (n < static_cast<ssize_t>(need)) {
      ++st.read_errors;
      continue;
    }
    const uint64_t fw = ReadFirmwareTimestamp(buf, s->gpu.layout);
    if (have && fw == last_fw) continue;
    ReadSnapshot(buf, s->gpu.layout, &r.s);
    if (have && fw > last_fw) {
      const uint64_t d = fw - last_fw;
      st.long_intervals += d > kLongInterval;
      st.max_interval = std::max(st.max_interval, d);
    }
    last_fw = fw;
    have = true;
    ++st.snapshots;
    if (!s->write_ring->Push(r)) ++st.write_drops;
    if (s->push_ring && !s->push_ring->Push(r)) ++st.push_drops;
  }
}

// Drains the write rings every 100 ms; after `stop` it drains once more and returns.
void WriterMain(std::vector<Stream>* streams, const std::atomic<bool>* stop,
                std::atomic<bool>* failed, std::string* error) {
  std::vector<Record> buf(4096);
  for (;;) {
    const bool last = stop->load(std::memory_order_acquire);
    for (Stream& s : *streams) {
      size_t n;
      while ((n = s.write_ring->PopMany(buf.data(), buf.size())) > 0) {
        std::string e;
        if (!WriteAll(s.out_fd, buf.data(), n * sizeof(Record), &e)) {
          *error = s.file + ": " + e;
          failed->store(true);
          return;
        }
      }
    }
    if (last) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void PushMain(std::vector<Stream>* streams, const Url& url, const std::string& experiment,
              const std::string& host, const std::atomic<bool>* stop, PushStats* ps) {
  std::vector<RateCalc> calc;
  for (Stream& s : *streams)
    calc.emplace_back(experiment, host, s.gpu.idx, s.gpu.peer, s.gpu.link_max);
  std::vector<Record> buf(4096);
  std::string body;
  uint64_t last_warn = 0;
  for (;;) {
    const bool last = stop->load(std::memory_order_acquire);
    body.clear();
    for (size_t i = 0; i < streams->size(); ++i) {
      size_t n;
      while ((n = (*streams)[i].push_ring->PopMany(buf.data(), buf.size())) > 0)
        for (size_t k = 0; k < n; ++k) calc[i].Add(buf[k], &body);
    }
    if (!body.empty()) {
      const uint64_t points = static_cast<uint64_t>(std::count(body.begin(), body.end(), '\n'));
      std::string err;
      ++ps->requests;
      if (VmImport(url, body, 2000, &err)) {
        ps->pushed_points += points;
      } else {
        ++ps->failed_requests;
        ps->failed_points += points;
        ps->last_error = err;
        const uint64_t now = Now(CLOCK_MONOTONIC);
        if (now - last_warn > 10000000000ull) {
          std::fprintf(stderr, "push failed, %llu points dropped (recording continues): %s\n",
                       static_cast<unsigned long long>(points), err.c_str());
          last_warn = now;
        }
      }
    }
    if (last) break;
    for (int k = 0; k < 20 && !stop->load(std::memory_order_acquire); ++k)  // 1 s, or until stopped
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  for (const RateCalc& c : calc) {
    ps->rates.intervals += c.stats().intervals;
    ps->rates.stale += c.stats().stale;
    ps->rates.decreases += c.stats().decreases;
    ps->rates.implausible += c.stats().implausible;
  }
}

std::string JsonString(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", c);
      out += b;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

struct Meta {
  std::string status, host, experiment, push_url, error;
  uint64_t period_ns = 0, start_real_ns = 0, end_real_ns = 0;
  double cpu_user_s = 0, cpu_sys_s = 0;
  bool push = false;
  PushStats push_stats;
};

bool WriteMeta(const std::string& dir, const Meta& m, const std::vector<Stream>& streams,
               std::string* err) {
  std::string j = "{\n";
  j += "  \"tool\": \"xgmi-netmon\",\n  \"version\": \"" XGMI_NETMON_VERSION "\",\n";
  j += "  \"status\": " + JsonString(m.status) + ",\n";
  j += "  \"host\": " + JsonString(m.host) + ",\n  \"experiment\": " + JsonString(m.experiment) +
       ",\n";
  j += "  \"period_ns\": " + std::to_string(m.period_ns) + ",\n";
  j += "  \"start_real_ns\": " + std::to_string(m.start_real_ns) + ",\n";
  j += "  \"end_real_ns\": " + std::to_string(m.end_real_ns) + ",\n";
  j += "  \"record_fields\": [\"t0_mono_ns\", \"t1_mono_ns\", \"real_ns\", \"real_err_ns\", "
       "\"firmware_timestamp_10ns\", \"rd0_kb\", \"rd1_kb\", \"rd2_kb\", \"rd3_kb\", \"rd4_kb\", "
       "\"rd5_kb\", "
       "\"rd6_kb\", \"rd7_kb\", \"wr0_kb\", \"wr1_kb\", \"wr2_kb\", \"wr3_kb\", \"wr4_kb\", "
       "\"wr5_kb\", "
       "\"wr6_kb\", \"wr7_kb\"],\n";
  j += "  \"gpus\": [\n";
  for (size_t i = 0; i < streams.size(); ++i) {
    const Stream& s = streams[i];
    const SamplerStats& st = s.st;
    j += "    {\"gpu\": " + std::to_string(s.gpu.idx) + ", \"card\": " + JsonString(s.gpu.card) +
         ", \"pci\": " + JsonString(s.gpu.pci) + ", \"file\": " + JsonString(s.file) +
         ", \"link_max_bytes_per_second\": " +
         std::to_string(static_cast<uint64_t>(s.gpu.link_max)) + ", \"peers\": {";
    bool first = true;
    for (int p = 0; p < kNumLinks; ++p) {
      if (s.gpu.peer[p] < 0) continue;
      j += std::string(first ? "" : ", ") + "\"" + std::to_string(p) +
           "\": " + std::to_string(s.gpu.peer[p]);
      first = false;
    }
    j += "}, \"reads\": " + std::to_string(st.reads) +
         ", \"snapshots\": " + std::to_string(st.snapshots) +
         ", \"intervals_over_3ms\": " + std::to_string(st.long_intervals) +
         ", \"max_interval_ms\": " + std::to_string(st.max_interval * 1e-5) +
         ", \"late_rounds\": " + std::to_string(st.late) +
         ", \"read_errors\": " + std::to_string(st.read_errors) +
         ", \"write_ring_drops\": " + std::to_string(st.write_drops) +
         ", \"push_ring_drops\": " + std::to_string(st.push_drops) + "}" +
         (i + 1 < streams.size() ? ",\n" : "\n");
  }
  j += "  ],\n";
  if (m.push) {
    const PushStats& p = m.push_stats;
    j += "  \"push\": {\"url\": " + JsonString(m.push_url) +
         ", \"requests\": " + std::to_string(p.requests) +
         ", \"failed_requests\": " + std::to_string(p.failed_requests) +
         ", \"pushed_points\": " + std::to_string(p.pushed_points) +
         ", \"failed_points\": " + std::to_string(p.failed_points) +
         ", \"counter_decreases\": " + std::to_string(p.rates.decreases) +
         ", \"implausible\": " + std::to_string(p.rates.implausible) +
         ", \"last_error\": " + JsonString(p.last_error) + "},\n";
  }
  j += "  \"cpu_user_s\": " + std::to_string(m.cpu_user_s) +
       ",\n  \"cpu_sys_s\": " + std::to_string(m.cpu_sys_s) + ",\n";
  j += "  \"error\": " + JsonString(m.error) + "\n}\n";
  const std::string tmp = dir + "/meta.json.tmp";
  const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    *err = tmp + ": " + std::strerror(errno);
    return false;
  }
  const bool ok = WriteAll(fd, j.data(), j.size(), err);
  close(fd);
  if (!ok) return false;
  if (rename(tmp.c_str(), (dir + "/meta.json").c_str()) != 0) {
    *err = std::string("rename meta.json: ") + std::strerror(errno);
    return false;
  }
  return true;
}

std::string ShortHostname() {
  char h[256] = {};
  gethostname(h, sizeof(h) - 1);
  std::string s = h;
  return s.substr(0, s.find('.'));
}

// mkdir -p; the leaf must not exist or be empty (a recording is never overwritten).
bool PrepareDir(const std::string& dir, std::string* err) {
  for (size_t i = 1; i <= dir.size(); ++i) {
    if (i < dir.size() && dir[i] != '/') continue;
    const std::string p = dir.substr(0, i);
    if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
      *err = "mkdir " + p + ": " + std::strerror(errno);
      return false;
    }
  }
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) {
        closedir(d);
        *err = dir + " is not empty (a recording is never overwritten)";
        return false;
      }
    }
    closedir(d);
  }
  return true;
}

std::string Replace(std::string s, const std::string& from, const std::string& to) {
  for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size()))
    s.replace(p, from.size(), to);
  return s;
}

}  // namespace

int RunRecord(const RecordOptions& opt) {
  const std::string host = opt.host_id.empty() ? ShortHostname() : opt.host_id;
  if (host.empty() || host.size() > kMaxHostLen) {
    std::fprintf(stderr, "error: host id must be 1..%zu characters\n", kMaxHostLen);
    return 1;
  }
  std::string dir = Replace(opt.out, "{host}", host);
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  std::string err;
  if (!PrepareDir(dir, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  char abs[PATH_MAX];
  if (!realpath(dir.c_str(), abs)) {
    std::fprintf(stderr, "error: %s: %s\n", dir.c_str(), std::strerror(errno));
    return 1;
  }
  dir = abs;
  std::string experiment = opt.experiment;
  if (experiment.empty()) {
    const std::string parent = dir.substr(0, dir.find_last_of('/'));
    experiment = parent.substr(parent.find_last_of('/') + 1);
  }
  if (experiment.empty() || experiment.size() > kMaxExperimentLen) {
    std::fprintf(stderr, "error: experiment name must be 1..%zu characters (--experiment)\n",
                 kMaxExperimentLen);
    return 1;
  }

  std::vector<Gpu> gpus;
  if (!Discover(opt.drm_root, &gpus, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  if (gpus.empty()) {
    std::fprintf(stderr, "error: no GPU with XGMI under %s\n", opt.drm_root.c_str());
    return 1;
  }

  Url url;
  const bool push = !opt.push_url.empty();
  if (push) {
    if (!ParseUrl(opt.push_url, &url, &err) || !VmHealthy(url, &err)) {
      std::fprintf(stderr, "error: VictoriaMetrics at %s: %s\n", opt.push_url.c_str(), err.c_str());
      return 1;
    }
    const int has = VmHasSeries(url, experiment, host, &err);
    if (has != 0) {
      if (has < 0)
        std::fprintf(stderr, "error: VictoriaMetrics at %s: %s\n", opt.push_url.c_str(),
                     err.c_str());
      else
        std::fprintf(
            stderr,
            "error: VictoriaMetrics already has xgmi data for experiment \"%s\", host \"%s\"; "
            "record into another experiment, or import with --replace\n",
            experiment.c_str(), host.c_str());
      return 1;
    }
  }

  const int lock = open((dir + "/.lock").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (lock < 0) {
    std::fprintf(stderr, "error: %s/.lock: %s\n", dir.c_str(), std::strerror(errno));
    return 1;
  }
  const std::string lock_text = "pid " + std::to_string(getpid()) + "\nhost " + host +
                                "\nstart_real_ns " + std::to_string(Now(CLOCK_REALTIME)) + "\n";
  WriteAll(lock, lock_text.data(), lock_text.size(), &err);
  close(lock);

  std::vector<Stream> streams(gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i) {
    Stream& s = streams[i];
    s.gpu = gpus[i];
    s.file = "gpu" + std::to_string(s.gpu.idx) + ".bin";
    s.sys_fd = open(s.gpu.metrics_path.c_str(), O_RDONLY | O_CLOEXEC);
    s.out_fd = open((dir + "/" + s.file).c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (s.sys_fd < 0 || s.out_fd < 0) {
      std::fprintf(stderr, "error: open %s: %s\n",
                   s.sys_fd < 0 ? s.gpu.metrics_path.c_str() : s.file.c_str(),
                   std::strerror(errno));
      return 1;
    }
    const FileHeader h = MakeHeader(s.gpu, host, experiment, opt.period_ns);
    if (!WriteAll(s.out_fd, &h, sizeof(h), &err)) {
      std::fprintf(stderr, "error: %s: %s\n", s.file.c_str(), err.c_str());
      return 1;
    }
    s.write_ring = std::make_unique<SpscRing<Record>>(kRingRecords);
    if (push) s.push_ring = std::make_unique<SpscRing<Record>>(kRingRecords);
  }

  Meta meta;
  meta.status = "running";
  meta.host = host;
  meta.experiment = experiment;
  meta.period_ns = opt.period_ns;
  meta.push = push;
  meta.push_url = opt.push_url;
  meta.start_real_ns = Now(CLOCK_REALTIME);
  if (!WriteMeta(dir, meta, streams, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  struct sigaction sa{};
  sa.sa_handler = OnSignal;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // First grid point a little ahead, so every sampler thread is waiting when it comes.
  const uint64_t p = opt.period_ns;
  const uint64_t mono = Now(CLOCK_MONOTONIC), real = Now(CLOCK_REALTIME);
  const uint64_t first_real = (real / p + 1) * p + 50 * p;
  const uint64_t first = mono + (first_real - real);

  std::atomic<bool> stop_samplers{false}, stop_writer{false}, stop_push{false}, write_failed{false};
  std::string write_error;
  PushStats push_stats;
  for (Stream& s : streams) s.thread = std::thread(SamplerMain, &s, first, p, &stop_samplers);
  std::thread writer(WriterMain, &streams, &stop_writer, &write_failed, &write_error);
  std::thread pusher;
  if (push)
    pusher = std::thread(PushMain, &streams, url, experiment, host, &stop_push, &push_stats);

  std::fprintf(
      stderr,
      "recording %zu GPUs into %s (experiment \"%s\", host \"%s\", every %.3g ms)%s%s; %s\n",
      streams.size(), dir.c_str(), experiment.c_str(), host.c_str(), p * 1e-6,
      push ? ", pushing to " : "", push ? opt.push_url.c_str() : "",
      opt.duration_s > 0 ? "stops after the duration or on Ctrl-C" : "Ctrl-C stops");
  const uint64_t end = opt.duration_s > 0 ? mono + static_cast<uint64_t>(opt.duration_s * 1e9) : 0;
  while (!g_signal.load() && !write_failed.load() && (end == 0 || Now(CLOCK_MONOTONIC) < end))
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

  stop_samplers.store(true);
  for (Stream& s : streams) s.thread.join();
  stop_writer.store(true, std::memory_order_release);
  writer.join();
  if (push) {
    stop_push.store(true, std::memory_order_release);
    pusher.join();
  }
  for (Stream& s : streams) {
    close(s.sys_fd);
    if (fsync(s.out_fd) != 0 && !write_failed.load()) {
      write_failed.store(true);
      write_error = s.file + ": fsync: " + std::strerror(errno);
    }
    close(s.out_fd);
  }

  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  meta.cpu_user_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
  meta.cpu_sys_s = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
  meta.end_real_ns = Now(CLOCK_REALTIME);
  meta.status = write_failed.load() ? "failed" : "complete";
  meta.error = write_error;
  meta.push_stats = push_stats;
  if (!WriteMeta(dir, meta, streams, &err)) std::fprintf(stderr, "error: %s\n", err.c_str());

  const double secs = (meta.end_real_ns - meta.start_real_ns) * 1e-9;
  std::fprintf(stderr, "%-4s %10s %8s %12s %10s %12s %6s %8s\n", "gpu", "snapshots", "per s",
               "> 3 ms", "max ms", "wake us avg", "late", "dropped");
  for (const Stream& s : streams) {
    const SamplerStats& st = s.st;
    const uint64_t iv = st.snapshots ? st.snapshots - 1 : 0;
    std::fprintf(stderr, "%-4d %10llu %8.1f %6llu %4.1f%% %10.3f %12.1f %6llu %8llu\n", s.gpu.idx,
                 static_cast<unsigned long long>(st.snapshots), st.snapshots / secs,
                 static_cast<unsigned long long>(st.long_intervals),
                 iv ? 100.0 * st.long_intervals / iv : 0.0, st.max_interval * 1e-5,
                 st.reads ? st.wake_sum_ns * 1e-3 / st.reads : 0.0,
                 static_cast<unsigned long long>(st.late),
                 static_cast<unsigned long long>(st.write_drops + st.push_drops + st.read_errors));
  }
  std::fprintf(stderr, "cpu: user %.2f s, sys %.2f s over %.1f s\n", meta.cpu_user_s,
               meta.cpu_sys_s, secs);
  if (push)
    std::fprintf(stderr, "push: %llu points, %llu failed in %llu of %llu requests\n",
                 static_cast<unsigned long long>(push_stats.pushed_points),
                 static_cast<unsigned long long>(push_stats.failed_points),
                 static_cast<unsigned long long>(push_stats.failed_requests),
                 static_cast<unsigned long long>(push_stats.requests));
  if (write_failed.load()) {
    std::fprintf(stderr, "error: recording failed: %s\n", write_error.c_str());
    return 1;
  }
  return 0;
}

}  // namespace xgmi
