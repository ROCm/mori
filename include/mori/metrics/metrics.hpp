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

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/metrics/traffic_counters.hpp"

// ===========================================================================
//  MORI metrics, host side: everything libmori_metrics implements, except the
//  HTTP endpoint itself (prometheus_metrics_server.hpp). Device code calls
//  traffic_counters.hpp, nothing from here. With BUILD_METRICS=OFF there is no
//  MORI_ENABLE_METRICS and section 4 gives the classes of section 3 empty
//  bodies, so that no caller needs an #ifdef.
// ===========================================================================

/* ------------------------------------------------------------------------ */
/*  0. Identity                                                              */
/* ------------------------------------------------------------------------ */

// Label values (host, GPU, NIC) have to be physical and agree with ainic-netmon and xgmi-netmon.
// Inline, all of it, and free of HIP: this section needs no library.

namespace mori::metrics {

// True when MORI_METRICS_ENABLE is truthy, or MORI_METRICS_PORT or MORI_METRICS_PUSH_URL is set.
inline bool EnabledFromEnv() {
  auto truthy = [](const char* v) {
    if (v == nullptr || *v == '\0') return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return !(s == "0" || s == "false" || s == "off" || s == "no");
  };
  const char* enable = std::getenv("MORI_METRICS_ENABLE");
  if (enable != nullptr && *enable != '\0') return truthy(enable);
  const char* port = std::getenv("MORI_METRICS_PORT");
  if (port != nullptr && *port != '\0' && std::atoi(port) > 0) return true;
  const char* push = std::getenv("MORI_METRICS_PUSH_URL");
  return push != nullptr && *push != '\0';
}

inline bool Enabled() {
  static const bool on = EnabledFromEnv();
  return on;
}

// The host label: MORI_METRICS_HOST_ID, else gethostname() as it is (not cut at the first dot).
inline std::string HostId() {
  const char* forced = std::getenv("MORI_METRICS_HOST_ID");
  if (forced != nullptr && *forced != '\0') return forced;
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
  return buf;
}

// "0000:05:00.0": lower case, with the PCI domain.
inline std::string NormalizePciBdf(std::string bdf) {
  std::transform(bdf.begin(), bdf.end(), bdf.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (std::count(bdf.begin(), bdf.end(), ':') == 1) bdf = "0000:" + bdf;
  return bdf;
}

namespace detail {

// "card" followed by digits only: "card3" yes, "card3-DP-1" (a connector) no.
inline bool IsDrmCardName(const std::string& name) {
  if (name.size() <= 4 || name.compare(0, 4, "card") != 0) return false;
  return std::all_of(name.begin() + 4, name.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; });
}

inline std::vector<std::string> ListDir(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (d == nullptr) return out;
  while (dirent* e = readdir(d)) {
    const std::string name = e->d_name;
    if (name != "." && name != "..") out.push_back(name);
  }
  closedir(d);
  return out;
}

}  // namespace detail

// The physical index of the GPU at PCI address `bdf`, or -1. Numbered as xgmi-netmon does: the
// DRM cards that have device/xgmi_device_id, sorted by card number, counted from 0.
inline int PhysicalGpuIndexForPci(const std::string& bdf,
                                  const std::string& drm_root = "/sys/class/drm") {
  if (bdf.empty()) return -1;
  const std::string want = NormalizePciBdf(bdf);
  std::vector<std::pair<int, std::string>> cards;
  for (const std::string& name : detail::ListDir(drm_root)) {
    if (!detail::IsDrmCardName(name)) continue;
    if (access((drm_root + "/" + name + "/device/xgmi_device_id").c_str(), F_OK) != 0) continue;
    cards.emplace_back(std::atoi(name.c_str() + 4), name);
  }
  std::sort(cards.begin(), cards.end());
  for (std::size_t i = 0; i < cards.size(); ++i) {
    char real[PATH_MAX];
    if (realpath((drm_root + "/" + cards[i].second + "/device").c_str(), real) == nullptr) continue;
    std::string path = real;
    if (NormalizePciBdf(path.substr(path.find_last_of('/') + 1)) == want) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Same against the real sysfs, remembered per address.
inline int PhysicalGpuIndexForPciCached(const std::string& bdf) {
  static std::mutex mu;
  static std::map<std::string, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(bdf);
  if (it != cache.end()) return it->second;
  int idx = PhysicalGpuIndexForPci(bdf);
  cache.emplace(bdf, idx);
  return idx;
}

// The value of a `gpu` label: the physical index, else "pci:<bdf>", else "unknown".
inline std::string GpuLabel(int gpu_index, const std::string& bdf) {
  if (gpu_index >= 0) return std::to_string(gpu_index);
  if (!bdf.empty()) return "pci:" + NormalizePciBdf(bdf);
  return "unknown";
}

// The network interface of RDMA device `ibdev` ("ionic_3" -> "ens3f0np0"), or "".
inline std::string NetdevForIbdev(const std::string& ibdev,
                                  const std::string& ib_root = "/sys/class/infiniband") {
  if (ibdev.empty()) return "";
  std::vector<std::string> names = detail::ListDir(ib_root + "/" + ibdev + "/device/net");
  if (names.empty()) return "";
  std::sort(names.begin(), names.end());
  return names.front();
}

// The link speed of RDMA device `ibdev` in bytes per second (the fastest of its ports), or 0.
inline double IbdevLinkBytesPerSecond(const std::string& ibdev,
                                      const std::string& ib_root = "/sys/class/infiniband") {
  if (ibdev.empty()) return 0;
  const std::string ports = ib_root + "/" + ibdev + "/ports";
  double best = 0;
  for (const std::string& port : detail::ListDir(ports)) {
    FILE* f = fopen((ports + "/" + port + "/rate").c_str(), "r");
    if (f == nullptr) continue;
    char line[128] = {0};
    const bool got = fgets(line, sizeof(line), f) != nullptr;
    fclose(f);
    if (!got) continue;
    char* unit = nullptr;
    const double number = strtod(line, &unit);
    while (*unit == ' ') ++unit;
    double bits = 0;
    if (unit[0] == 'G' && unit[1] == 'b') bits = number * 1e9;
    if (unit[0] == 'M' && unit[1] == 'b') bits = number * 1e6;
    if (unit[0] == 'K' && unit[1] == 'b') bits = number * 1e3;
    best = std::max(best, bits / 8);
  }
  return best;
}

// Who a rank is, for metrics labels; exchanged once at init (application::Context::PeIdentities).
struct PeIdentity {
  std::string hostname;  // HostId()
  int gpuIndex{-1};      // as xgmi-netmon numbers GPUs; -1 when it could not be determined
  std::string gpuPci;    // PCI address of the GPU; what is left when gpuIndex is -1
  std::string ibdev;     // RDMA device this rank's QPs are on; empty without RDMA
  std::string netdev;    // its network interface

  std::string GpuLabel() const { return mori::metrics::GpuLabel(gpuIndex, gpuPci); }
};

}  // namespace mori::metrics

/* ------------------------------------------------------------------------ */
/*  1. Sampling                                                              */
/* ------------------------------------------------------------------------ */

// A component implements MetricSource to read its counters out; MetricPublisher
// turns the cumulative readings into deltas and hands them to a sink.

namespace mori::metrics {

enum class MetricKind {
  // Value is MONOTONIC and the publisher ships `value - last`.  A decrease rebases.
  kCounter,
  // Value is the current reading and is shipped as-is every tick.
  kGauge,
  // `buckets` / `count` / `value` (the sum) are MONOTONIC and shipped as deltas.
  kHistogram,
};

using MetricLabels = std::vector<std::pair<std::string, std::string>>;

// One sample from a component.  `name`/`help` MUST be string literals with static storage.
struct MetricSample {
  const char* name = nullptr;
  const char* help = nullptr;
  MetricLabels labels;
  // Counter: cumulative count.  Gauge: current reading.  Histogram: cumulative sum of the values.
  uint64_t value = 0;
  MetricKind kind = MetricKind::kCounter;
  // Applied by the publisher to the delta (counter), the reading (gauge) or the sum (histogram).
  double scale = 1.0;

  // --- kHistogram only ------------------------------------------------------
  // Ascending finite upper bounds, in the METRIC's unit (already scaled).  Must
  // outlive the publisher: point it at a static table.
  const std::vector<double>* bounds = nullptr;
  // CUMULATIVE over the bounds.  Same length as *bounds.
  std::vector<uint64_t> buckets;
  // Total observations (the +Inf bucket).
  uint64_t count = 0;
};

// Anything that can be sampled once per metrics tick.  Never called under a component lock.
class MetricSource {
 public:
  virtual ~MetricSource() = default;
  virtual std::vector<MetricSample> SampleMetrics() const { return {}; }
};

// True when `labels` names the same label twice; the publisher refuses such a sample.
bool HasDuplicateLabelNames(const MetricLabels& labels);

// Turns snapshots into deltas and hands them to a sink; baselines are keyed by (source id, metric
// name, labels).  Not thread-safe: one publisher belongs to one publishing thread.
class MetricPublisher {
 public:
  struct Sink {
    // delta >= 0.  Not called for a zero delta, except once for a series seen for the first time.
    std::function<void(const char* name, const char* help, const MetricLabels&, double delta)>
        counter;
    std::function<void(const char* name, const char* help, const MetricLabels&, double value)>
        gauge;
    // `bucket_delta` is cumulative over `bounds`.  Called for a series seen for
    // the first time, then only when there are new observations.
    std::function<void(const char* name, const char* help, const MetricLabels&,
                       const std::vector<double>& bounds, const std::vector<uint64_t>& bucket_delta,
                       uint64_t count_delta, double sum_delta)>
        histogram;
  };

  // `source_labels` go in front of every sample's own.  `source_id` must be stable across ticks.
  void Publish(const std::string& source_id, const MetricLabels& source_labels,
               const MetricSource& source, const Sink& sink);

  void Publish(const std::string& source_id, const MetricLabels& source_labels,
               const std::vector<MetricSample>& samples, const Sink& sink);

  // Drop every baseline.
  void Reset() {
    last_.clear();
    last_hist_.clear();
  }

  // Drop the baselines of one source.  Call it when the source goes away.
  void Forget(const std::string& source_id);

  // Samples refused because their merged label set named a label twice.
  uint64_t RejectedSamples() const { return rejected_; }

 private:
  struct HistBaseline {
    std::vector<uint64_t> buckets;
    uint64_t count = 0;
    uint64_t sum = 0;
  };

  std::unordered_map<std::string, uint64_t> last_;
  std::unordered_map<std::string, HistBaseline> last_hist_;
  uint64_t rejected_ = 0;
};

}  // namespace mori::metrics

/* ------------------------------------------------------------------------ */
/*  2. Registry                                                              */
/* ------------------------------------------------------------------------ */

// The one metrics endpoint of a process.  One background thread samples every registered source
// each tick into a MetricsServer on MORI_METRICS_PORT and into the table Snapshot() returns; with
// MORI_METRICS_PUSH_URL every sample is also sent with the time it was taken.
// The Registry adds exactly one label to every series: `host`.
// fork(): in a child, Instance() returns a NEW registry and abandons the inherited one without
// touching it; sources have to register again.

namespace mori::metrics {

class MetricsServer;

// How much of the traffic is counted, and how often the counters are read.  Process-wide; starts
// from MORI_METRICS_SAMPLE_EP, MORI_METRICS_SAMPLE_IO and MORI_METRICS_INTERVAL_MS and can be
// changed while the process runs (POST /sampling?ep=8&io=8&interval_ms=10, SetSampling()).
// A period of N counts one event in N, N-fold: 1 counts everything (the default), 0 nothing.
struct Sampling {
  // One EP dispatch and its combine in this many are counted and timed.
  uint32_t ep = 1;
  // One IO transfer in this many is metered.
  uint32_t io = 1;
  // Period of the metrics thread; 0 when the process has no such thread.
  int interval_ms = 0;
};

inline constexpr int kMaxSampleIntervalMs = 60000;

// The periods as the data paths read them (EpTrafficStats, IoMetrics).
uint32_t EpSamplePeriod();
const std::atomic<uint32_t>* IoSamplePeriodPtr();

class Registry {
 public:
  struct Options {
    // First port to try; <= 0 serves nothing.  On "address in use" the next ports are tried, up
    // to port + port_probe - 1.
    int port = 0;
    int port_probe = 32;
    // Dotted IPv4 to listen on; empty = every interface.
    std::string bind_addr;
    // Directory for a Prometheus file_sd target file, <host>-<pid>.json; empty writes none.
    std::string port_dir;
    // Value of the `host` label.
    std::string host;
    // Period of the background thread; <= 0 starts none, and only Tick() / Snapshot() sample.
    int interval_ms = 1000;
    // Publish the sampling in effect among the registry's series.
    bool sampling_series = true;
    // Extra labels for the file_sd target (e.g. local_rank): none may be a label a source uses.
    MetricLabels target_labels;

    // ---- Push ---------------------------------------------------------------
    // Base URL of a VictoriaMetrics instance ("http://host:8428", plain http); empty pushes
    // nothing.  Every changed sample is sent with its wall-clock time to <url>/api/v1/import.
    std::string push_url;
    // How often the kept points are sent; <= 0 starts no sending thread.
    int push_flush_ms = 1000;
    // A series that did not change is sent again this often; histograms only at this pace.
    int push_heartbeat_ms = 1000;
    // Points kept at most while the database is slow or away; beyond it new points are dropped.
    std::size_t push_max_points = 4000000;

    // MORI_METRICS_PORT (+ the local rank, when the launcher exports one), MORI_METRICS_ADDR,
    // MORI_METRICS_PORT_DIR, MORI_METRICS_HOST_ID, MORI_METRICS_PUSH_URL, MORI_METRICS_INTERVAL_MS
    // (1 ms by default with a push URL, else 1 s).
    static Options FromEnv();
  };

  // The process's registry, configured from the environment.  Never destroyed.
  static Registry& Instance();

  explicit Registry(Options options);
  ~Registry();

  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  // Start sampling `source`, which must stay valid until Unregister returns.  `name` must be
  // unique among live sources; `labels` go in front of every sample's own (after `host`).
  // Returns a handle: never 0, and never one given out before, in this process or its fork parents.
  uint64_t Register(const std::string& name, MetricLabels labels, const MetricSource* source);

  // Sample `handle`'s source one last time, then drop it and every series only it was writing.
  void Unregister(uint64_t handle);

  // Sample every source now (what the thread does each tick).
  void Tick();

  struct Series {
    std::string name;
    std::string help;
    MetricKind kind = MetricKind::kCounter;
    MetricLabels labels;  // `host` first
    // Counter: cumulative total.  Gauge: last reading.  Histogram: sum of observations.
    double value = 0.0;
    // Histogram only; `buckets` is cumulative over `bounds`.
    std::vector<double> bounds;
    std::vector<uint64_t> buckets;
    uint64_t count = 0;
  };

  // Tick(), then every series as it stands.
  std::vector<Series> Snapshot();

  // Period of the sampling thread now; 0 when there is none.
  int interval_ms() const { return interval_ms_.load(std::memory_order_relaxed); }
  void SetIntervalMs(int ms);

  // The sampling in effect; the interval is this registry's thread's.
  Sampling GetSampling() const;
  // Changes the settings that are not negative and returns those in effect. A value out of
  // range changes nothing: `error`, when given, says what was wrong.
  Sampling SetSampling(int64_t ep, int64_t io, int64_t interval_ms, std::string* error = nullptr);

  // GET or POST /sampling: the answer's body, and its HTTP status in `status`. `query` is what
  // follows the '?', "ep=8&io=1".
  std::string HandleSamplingRequest(const std::string& method, const std::string& query,
                                    std::string* status);

  // The port being served, or -1.
  int port() const { return port_.load(std::memory_order_acquire); }
  const std::string& host() const { return options_.host; }
  uint64_t RejectedSamples();

  // Send the points kept so far now, waiting at most `timeout_ms`.  A no-op without a push URL.
  void FlushPush(int timeout_ms = 2000);

  struct PushStats {
    uint64_t sent = 0;      // points the database accepted
    uint64_t dropped = 0;   // points lost: the buffer was full, or a send failed
    uint64_t failures = 0;  // sends that failed
  };
  PushStats push_stats() const;

 private:
  struct Source {
    std::string name;
    MetricLabels labels;
    const MetricSource* source = nullptr;
    std::set<std::string> series;  // keys into series_
  };
  struct Stored {
    Series data;
    int refs = 0;  // sources writing this series
    // Push: the JSON this series' lines start with, and when a point of it was last kept.
    std::vector<const std::string*> push_heads;
    int64_t pushed_ms = 0;
  };
  // One kept sample.  `head` points into push_heads_, whose strings never move.
  struct Point {
    const std::string* head;
    int64_t ts_ms;
    double value;
  };

  void StartLocked();
  void StartServerLocked();
  void WriteTargetFileLocked();
  void PublishLocked(Source& src);
  void TickLocked();
  void Loop();
  void PushLoop();
  const std::string* PushHeadLocked(const std::string& name, const MetricLabels& labels,
                                    const char* le = nullptr);
  void KeepLocked(Stored& stored, std::size_t slot, double value, int64_t ts_ms);
  void HeartbeatLocked(int64_t now_ms);

  const Options options_;
  const pid_t owner_pid_;

  std::mutex mu_;
  std::condition_variable cv_;
  bool started_ = false;
  bool stopping_ = false;
  std::map<uint64_t, Source> sources_;
  std::map<std::string, Stored> series_;
  MetricPublisher publisher_;
  std::unique_ptr<MetricsServer> server_;
  std::atomic<int> port_{-1};
  std::atomic<int> interval_ms_{0};
  std::unique_ptr<MetricSource> sampling_source_;
  std::string target_file_;
  std::thread thread_;

  // Push.  pending_, push_heads_ and heartbeat_ms_ are under mu_; a send holds
  // push_mu_ (never together with mu_).
  bool pushing_ = false;
  std::vector<Point> pending_;
  std::deque<std::string> push_heads_;
  int64_t heartbeat_ms_ = 0;
  std::mutex push_mu_;
  std::thread push_thread_;
  std::atomic<uint64_t> push_sent_{0};
  std::atomic<uint64_t> push_dropped_{0};
  std::atomic<uint64_t> push_failures_{0};

  friend struct RegistryForkAccess;
};

}  // namespace mori::metrics

/* ------------------------------------------------------------------------ */
/*  3. Traffic counters, host side                                           */
/* ------------------------------------------------------------------------ */

// Who owns each block of traffic_counters.hpp and makes it series of the registry:
//   DeviceCounterBlock   the two below: the memory device code adds to
//   RdmaTrafficCounters   shmem (per rank), cco (per comm)
//   EpTrafficStats        each EP op (v1 handle, v2 Python op)
//   IoMetrics             the process (one table for every IOEngine)

namespace mori {
namespace metrics {

// True when the build has the metrics module and the user turned metrics on.
inline bool TrafficEnabled() {
#ifdef MORI_ENABLE_METRICS
  return Enabled();
#else
  return false;
#endif
}

}  // namespace metrics

/* ---- 3a. Device counter block ---- */

// uint64 counters that device code adds to.
//   kHostMapped  host memory registered with HIP and mapped into the device: the host reads and
//                pokes it with plain loads and stores, no HIP call. Read() is thread-safe.
//   kDevice      device memory, cheaper to add to. The host does not read it: a HIP read from
//                the metrics thread would invalidate a graph capture on another stream.

namespace application {

class DeviceCounterBlock {
 public:
  enum class Placement { kDevice, kHostMapped };

  // `slots` zeroed uint64 on the calling thread's current device. On failure
  // DevicePtr() is null and Read() returns zeros.
  explicit DeviceCounterBlock(size_t slots, Placement placement = Placement::kHostMapped);
  ~DeviceCounterBlock();

  DeviceCounterBlock(const DeviceCounterBlock&) = delete;
  DeviceCounterBlock& operator=(const DeviceCounterBlock&) = delete;

  // What device code is given.
  uint64_t* DevicePtr() const { return devicePtr_; }

  // kHostMapped only: zeros for a device block.
  std::vector<uint64_t> Read() const;

  // Set one slot while kernels may be running. kHostMapped only.
  void Poke(size_t slot, uint64_t value) const;

  // Set the block's contents, before any kernel runs.
  void Write(const std::vector<uint64_t>& values);

 private:
  size_t slots_;
  Placement placement_;
  int device_{-1};
  uint64_t* devicePtr_{nullptr};
  uint64_t* host_{nullptr};  // kHostMapped: the block itself
  bool hostRegistered_{false};
  mutable std::mutex mu_;
};

/* ---- 3b. RDMA issued by the GPU itself (traffic_counters.hpp, section 3) ---- */

// Owns the block a library's device code adds to, and publishes it as
// mori_gpu_rdma_bytes_total / mori_gpu_rdma_nic_info for as long as it lives.
class RdmaTrafficCounters : public metrics::MetricSource {
 public:
  // `api` names the library ("shmem", "cco"). `peers` is the Context's identity table.
  RdmaTrafficCounters(std::string api, int rank, std::vector<metrics::PeIdentity> peers);
  ~RdmaTrafficCounters() override;

  RdmaTrafficCounters(const RdmaTrafficCounters&) = delete;
  RdmaTrafficCounters& operator=(const RdmaTrafficCounters&) = delete;

  // What device code is given; null if the block could not be set up.
  uint64_t* DevicePtr() const { return block_.DevicePtr(); }
  // bytes[op][pe]. Tests write it as device code would.
  DeviceCounterBlock& Block() { return block_; }

  std::vector<metrics::MetricSample> SampleMetrics() const override;

 private:
  std::string api_;
  int rank_;
  std::vector<metrics::PeIdentity> peers_;
  DeviceCounterBlock block_;
  uint64_t registryHandle_{0};  // 0: not registered
};

}  // namespace application

/* ---- 3c. EP (traffic_counters.hpp, section 4) ---- */

namespace moe {

struct EpTrafficStatsConfig {
  int rank{0};
  int worldSize{0};
  int gpuPerNode{1};
  // "v1", "v2-hip", "v2-flydsl".
  std::string impl;
  // Kernel family, for the calls metric ("intranode", "internode_v1", ...).
  std::string kernel;
  // Identity of every rank, indexed by rank. May be empty (tests): peers are labelled by rank.
  std::vector<metrics::PeIdentity> peers;
};

// Owns the counter block an EP op's kernels add to, and publishes it as
// mori_ep_* for as long as it lives. Create it on the device the op runs on.
class EpTrafficStats : public metrics::MetricSource {
 public:
  explicit EpTrafficStats(EpTrafficStatsConfig config);
  ~EpTrafficStats() override;

  EpTrafficStats(const EpTrafficStats&) = delete;
  EpTrafficStats& operator=(const EpTrafficStats&) = delete;

  // What a kernel is given. Null if the block could not be set up.
  uint64_t* DevicePtr() const { return ready_ ? block_.DevicePtr() : nullptr; }

  // Bytes one token carries on the wire for `op`. The kernels count tokens; the
  // bytes published are tokens times this, taken at each tick.
  void SetBytesPerToken(int op, uint64_t bytes) {
    bytesPerToken_[op].store(bytes, std::memory_order_relaxed);
  }

  // sent[op][pe], recv[op][pe], calls[op], and ticks[op]: kernel time in device wall-clock ticks.
  struct Snapshot {
    std::vector<uint64_t> sent[kEpStatsOpCount];
    std::vector<uint64_t> recv[kEpStatsOpCount];
    uint64_t calls[kEpStatsOpCount] = {0, 0};
    uint64_t ticks[kEpStatsOpCount] = {0, 0};
  };
  Snapshot Read() const;

  // Ticks of the device's wall clock in a second; 0 if the device did not say.
  double TicksPerSecond() const { return ticksPerSecond_; }

  // Called by the registry only, under its lock. Also hands the kernels the process's EP
  // sampling period when it has changed.
  std::vector<metrics::MetricSample> SampleMetrics() const override;

 private:
  EpTrafficStatsConfig config_;
  // What the kernels add to (device memory), and the host memory they copy it to.
  application::DeviceCounterBlock block_;
  application::DeviceCounterBlock mirror_;
  bool ready_{false};
  std::atomic<uint64_t> bytesPerToken_[kEpStatsOpCount] = {};
  mutable std::atomic<uint32_t> samplePeriod_{1};
  double ticksPerSecond_{0};
  // [op][tx|rx][pe]: the tokens last seen, and the bytes they have added up to.
  mutable std::vector<uint64_t> lastTokens_[kEpStatsOpCount][2];
  mutable std::vector<uint64_t> bytes_[kEpStatsOpCount][2];
  uint64_t registryHandle_{0};  // 0: not registered
};

}  // namespace moe

/* ---- 3d. IO (traffic_counters.hpp, section 2) ---- */

// The process's IO traffic counters: one table shared by every IOEngine in it. Blocks are never
// freed and never move: pointers to them are kept in route cache entries, sessions and endpoints.

namespace io {
namespace internal {

class IoMetrics : public mori::metrics::MetricSource {
 public:
  // The table, or nullptr when metrics are off: every caller treats that as "count nothing".
  static IoMetrics* Get();

  // The block of a flow, created on first use.  Slow path: a lock and a map.
  IoFlowCounters* Flow(IoFlowIdentity id);
  NicFlowCounters* NicFlow(NicFlowIdentity id);

  // Register with this process's metric registry; again after a fork, when it is a new one.
  void EnsureRegistered();

  // The host label of this process: the local end of every flow.
  const std::string& Host() const { return host_; }
  // The `*_host` label for a peer that names its host `hostname`: this process's
  // own host label when that is this machine's hostname, "unknown" when empty.
  std::string HostLabel(const std::string& hostname) const;

  std::vector<mori::metrics::MetricSample> SampleMetrics() const override;

  // Tests only: a table that is not the process's, registered nowhere.
  IoMetrics();

 private:
  void ZeroAll();

  std::string host_;
  std::string rawHostname_;  // gethostname(), which is what a peer advertises

  mutable std::mutex mu_;
  // std::deque: growing it never moves the blocks already handed out.
  std::deque<IoFlowCounters> flowBlocks_;
  std::deque<NicFlowCounters> nicFlowBlocks_;
  std::map<IoFlowIdentity, IoFlowCounters*> flows_;
  std::map<NicFlowIdentity, NicFlowCounters*> nicFlows_;
  mutable std::map<std::string, uint64_t> linkSpeed_;  // by ibdev; the sampling thread's

  // The registry this table is registered with (compared by address only).
  std::atomic<const void*> registeredWith_{nullptr};
};

}  // namespace internal
}  // namespace io
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  4. Built without the metrics module                                      */
/* ------------------------------------------------------------------------ */

#ifndef MORI_ENABLE_METRICS
// BUILD_METRICS=OFF: these keep the callers of section 3 linking; none is
// reached past TrafficEnabled() / IoMetrics::Get().
namespace mori {
namespace application {
inline DeviceCounterBlock::DeviceCounterBlock(size_t slots, Placement placement)
    : slots_(slots), placement_(placement) {}
inline DeviceCounterBlock::~DeviceCounterBlock() {}
inline RdmaTrafficCounters::RdmaTrafficCounters(std::string, int, std::vector<metrics::PeIdentity>)
    : rank_(0), block_(0, DeviceCounterBlock::Placement::kDevice) {}
inline RdmaTrafficCounters::~RdmaTrafficCounters() {}
inline std::vector<metrics::MetricSample> RdmaTrafficCounters::SampleMetrics() const { return {}; }
}  // namespace application
namespace moe {
inline EpTrafficStats::EpTrafficStats(EpTrafficStatsConfig config)
    : config_(std::move(config)),
      block_(0, application::DeviceCounterBlock::Placement::kDevice),
      mirror_(0, application::DeviceCounterBlock::Placement::kDevice) {}
inline EpTrafficStats::~EpTrafficStats() {}
inline EpTrafficStats::Snapshot EpTrafficStats::Read() const { return {}; }
inline std::vector<metrics::MetricSample> EpTrafficStats::SampleMetrics() const { return {}; }
}  // namespace moe
namespace io {
namespace internal {
inline IoMetrics* IoMetrics::Get() { return nullptr; }
inline IoFlowCounters* IoMetrics::Flow(IoFlowIdentity) { return nullptr; }
inline NicFlowCounters* IoMetrics::NicFlow(NicFlowIdentity) { return nullptr; }
inline void IoMetrics::EnsureRegistered() {}
inline std::string IoMetrics::HostLabel(const std::string& hostname) const { return hostname; }
inline std::vector<mori::metrics::MetricSample> IoMetrics::SampleMetrics() const { return {}; }
}  // namespace internal
}  // namespace io
}  // namespace mori
#endif  // !MORI_ENABLE_METRICS
