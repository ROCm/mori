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

// Implementation of mori/metrics/metrics.hpp, in the order of its sections.

#include "mori/metrics/metrics.hpp"

#include <hip/hip_runtime_api.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "mori/utils/mori_log.hpp"

/* ------------------------------------------------------------------------ */
/*  1. Sampling: MetricPublisher                                            */
/* ------------------------------------------------------------------------ */

namespace mori::metrics {
namespace {

// '\0' separates the parts.  The source id is the PREFIX, which is what Forget() relies on.
std::string BaselineKey(const std::string& source_id, const char* name,
                        const MetricLabels& source_labels, const MetricLabels& sample_labels) {
  std::string id = source_id;
  id += '\0';
  id += name;
  auto append = [&id](const MetricLabels& labels) {
    for (const auto& [k, v] : labels) {
      id += '\0';
      id += k;
      id += '=';
      id += v;
    }
  };
  append(source_labels);
  append(sample_labels);
  return id;
}

MetricLabels MergeLabels(const MetricLabels& source_labels, const MetricLabels& sample_labels) {
  MetricLabels merged;
  merged.reserve(source_labels.size() + sample_labels.size());
  merged.insert(merged.end(), source_labels.begin(), source_labels.end());
  merged.insert(merged.end(), sample_labels.begin(), sample_labels.end());
  return merged;
}

void WarnDuplicateOnce(const char* name) {
  // Leaked on purpose: the registry's thread may publish while static objects are destroyed.
  static std::mutex& mu = *new std::mutex();
  static std::set<std::string>& seen = *new std::set<std::string>();
  std::lock_guard<std::mutex> lock(mu);
  if (!seen.insert(name).second) return;
  MORI_METRICS_ERROR(
      "metric '{}' names a label twice once the source's labels are merged in; the sample is "
      "dropped (a scraper would reject the whole scrape over it)",
      name);
}

}  // namespace

bool HasDuplicateLabelNames(const MetricLabels& labels) {
  for (std::size_t i = 1; i < labels.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (labels[i].first == labels[j].first) return true;
    }
  }
  return false;
}

void MetricPublisher::Publish(const std::string& source_id, const MetricLabels& source_labels,
                              const MetricSource& source, const Sink& sink) {
  Publish(source_id, source_labels, source.SampleMetrics(), sink);
}

void MetricPublisher::Publish(const std::string& source_id, const MetricLabels& source_labels,
                              const std::vector<MetricSample>& samples, const Sink& sink) {
  for (const MetricSample& s : samples) {
    if (s.name == nullptr) continue;

    MetricLabels merged = MergeLabels(source_labels, s.labels);
    if (HasDuplicateLabelNames(merged)) {
      ++rejected_;
      WarnDuplicateOnce(s.name);
      continue;
    }

    if (s.kind == MetricKind::kGauge) {
      if (sink.gauge) {
        sink.gauge(s.name, s.help, merged, static_cast<double>(s.value) * s.scale);
      }
      continue;
    }

    if (s.kind == MetricKind::kHistogram) {
      if (s.bounds == nullptr || s.buckets.size() != s.bounds->size()) continue;
      auto [it, first_sight] =
          last_hist_.try_emplace(BaselineKey(source_id, s.name, source_labels, s.labels));
      HistBaseline& last = it->second;
      // A rebuilt source reads lower than its baseline: rebase rather than ship a negative delta.
      const bool rebuilt =
          !first_sight && (s.count < last.count || last.buckets.size() != s.buckets.size());
      if (first_sight || rebuilt) last = HistBaseline{std::vector<uint64_t>(s.buckets.size(), 0)};
      if (rebuilt) {
        last.buckets = s.buckets;
        last.count = s.count;
        last.sum = s.value;
        continue;
      }
      if ((first_sight || s.count > last.count) && sink.histogram) {
        std::vector<uint64_t> delta(s.buckets.size(), 0);
        for (std::size_t i = 0; i < delta.size(); ++i) {
          delta[i] = s.buckets[i] > last.buckets[i] ? s.buckets[i] - last.buckets[i] : 0;
        }
        const uint64_t count_delta = s.count - last.count;
        const double sum_delta =
            s.value > last.sum ? static_cast<double>(s.value - last.sum) * s.scale : 0.0;
        sink.histogram(s.name, s.help, merged, *s.bounds, delta, count_delta, sum_delta);
      }
      last.buckets = s.buckets;
      last.count = s.count;
      last.sum = s.value;
      continue;
    }

    // Shipped at first sight even at zero: rate() needs a sample from before the first event.
    auto [it, first_sight] =
        last_.try_emplace(BaselineKey(source_id, s.name, source_labels, s.labels), 0);
    uint64_t& last = it->second;
    if ((first_sight || s.value > last) && sink.counter) {
      sink.counter(s.name, s.help, merged,
                   s.value > last ? static_cast<double>(s.value - last) * s.scale : 0.0);
    }
    // Also on a zero or negative delta: stays correct against a counter rebuilt underneath us.
    last = s.value;
  }
}

void MetricPublisher::Forget(const std::string& source_id) {
  std::string prefix = source_id;
  prefix += '\0';
  auto drop = [&prefix](auto& map) {
    for (auto it = map.begin(); it != map.end();) {
      if (it->first.compare(0, prefix.size(), prefix) == 0) {
        it = map.erase(it);
      } else {
        ++it;
      }
    }
  };
  drop(last_);
  drop(last_hist_);
}

}  // namespace mori::metrics

/* ------------------------------------------------------------------------ */
/*  2. Registry                                                             */
/* ------------------------------------------------------------------------ */

namespace mori::metrics {
namespace {

int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v) return fallback;
  return static_cast<int>(n);
}

std::string EnvStr(const char* name) {
  const char* v = std::getenv(name);
  return v == nullptr ? std::string() : std::string(v);
}

// -1 when the launcher exports none.  Same variables, same order, as UMBP's standalone client.
int LocalRankFromEnv() {
  for (const char* name :
       {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID"}) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') continue;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && n >= 0) return static_cast<int>(n);
  }
  return -1;
}

std::string JsonString(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

std::string SeriesKey(const char* name, const MetricLabels& labels) {
  std::string key = name;
  for (const auto& [k, v] : labels) {
    key += '\0';
    key += k;
    key += '=';
    key += v;
  }
  return key;
}

std::atomic<Registry*> g_instance{nullptr};
std::atomic<Registry*> g_target_owner{nullptr};

int64_t WallMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t SteadyMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void AppendNumber(std::string* out, double v) {
  char buf[40];
  if (v == std::floor(v) && std::fabs(v) < 9e15) {
    std::snprintf(buf, sizeof(buf), "%.0f", v);
  } else {
    std::snprintf(buf, sizeof(buf), "%.9g", v);
  }
  *out += buf;
}

struct PushUrl {
  std::string host;
  std::string port = "80";
  std::string path;  // prefix + /api/v1/import
  bool ok = false;
};

PushUrl ParsePushUrl(const std::string& s) {
  PushUrl u;
  const std::string scheme = "http://";
  if (s.compare(0, scheme.size(), scheme) != 0) return u;
  std::string rest = s.substr(scheme.size());
  const std::size_t slash = rest.find('/');
  std::string hostport = rest.substr(0, slash);
  std::string prefix = slash == std::string::npos ? "" : rest.substr(slash);
  while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
  const std::size_t colon = hostport.rfind(':');
  u.host = hostport.substr(0, colon);
  if (colon != std::string::npos) u.port = hostport.substr(colon + 1);
  u.path = prefix + "/api/v1/import";
  u.ok = !u.host.empty() && !u.port.empty();
  return u;
}

bool WaitFd(int fd, short events, int64_t deadline) {
  for (;;) {
    const int64_t left = deadline - SteadyMs();
    if (left <= 0) return false;
    pollfd p{fd, events, 0};
    const int rc = ::poll(&p, 1, static_cast<int>(std::min<int64_t>(left, 1000)));
    if (rc > 0) return true;
    if (rc < 0 && errno != EINTR) return false;
  }
}

// Returns the HTTP status, or -1 with *err set.  `timeout_ms` bounds the whole request.
int HttpPost(const PushUrl& url, const std::string& body, int timeout_ms, std::string* err) {
  const int64_t deadline = SteadyMs() + timeout_ms;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const int rc = ::getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &res);
  if (rc != 0) {
    *err = "resolve " + url.host + ": " + ::gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
    fd = ::socket(a->ai_family, a->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, a->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    int so = errno;
    if (errno == EINPROGRESS) {
      so = ETIMEDOUT;
      if (WaitFd(fd, POLLOUT, deadline)) {
        socklen_t len = sizeof(so);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &len) == 0 && so == 0) break;
      }
    }
    *err = "connect " + url.host + ":" + url.port + ": " + std::strerror(so);
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) {
    if (err->empty()) *err = "connect " + url.host + ":" + url.port + " failed";
    return -1;
  }

  std::string req =
      "POST " + url.path + " HTTP/1.0\r\nHost: " + url.host + ":" + url.port +
      "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\n\r\n";
  req += body;
  std::size_t sent = 0;
  while (sent < req.size()) {
    const ssize_t w = ::send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
    if (w > 0) {
      sent += static_cast<std::size_t>(w);
    } else if (w < 0 && errno == EINTR) {
      continue;
    } else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
               WaitFd(fd, POLLOUT, deadline)) {
      continue;
    } else {
      *err = std::string("send: ") +
             (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK ? std::strerror(errno) : "timeout");
      ::close(fd);
      return -1;
    }
  }
  std::string resp;
  char buf[512];
  while (resp.find("\r\n") == std::string::npos && resp.size() < 4096) {
    const ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
    if (r > 0) {
      resp.append(buf, static_cast<std::size_t>(r));
    } else if (r == 0) {
      break;
    } else if (errno == EINTR) {
      continue;
    } else if ((errno == EAGAIN || errno == EWOULDBLOCK) && WaitFd(fd, POLLIN, deadline)) {
      continue;
    } else {
      *err = std::string("receive: ") +
             (errno != EAGAIN && errno != EWOULDBLOCK ? std::strerror(errno) : "timeout");
      ::close(fd);
      return -1;
    }
  }
  ::close(fd);
  int status = -1;
  if (std::sscanf(resp.c_str(), "HTTP/%*d.%*d %d", &status) != 1) {
    *err = "no HTTP status in the answer";
    return -1;
  }
  return status;
}
// Leaked on purpose: a CQ thread may meter a transfer while static objects are destroyed.
struct SamplePeriods {
  std::atomic<uint32_t> ep;
  std::atomic<uint32_t> io;
  SamplePeriods()
      : ep(static_cast<uint32_t>(std::clamp(EnvInt("MORI_METRICS_SAMPLE_EP", 1), 0,
                                            static_cast<int>(kMaxSamplePeriod)))),
        io(static_cast<uint32_t>(std::clamp(EnvInt("MORI_METRICS_SAMPLE_IO", 1), 0,
                                            static_cast<int>(kMaxSamplePeriod)))) {}
};
SamplePeriods& Periods() {
  static SamplePeriods* periods = new SamplePeriods();
  return *periods;
}

class SamplingSource : public MetricSource {
 public:
  explicit SamplingSource(const Registry* registry) : registry_(registry) {}
  std::vector<MetricSample> SampleMetrics() const override {
    std::vector<MetricSample> out;
    out.push_back(MetricSample{names::kSamplePeriod,
                               names::kSamplePeriodHelp,
                               {{"module", "ep"}},
                               EpSamplePeriod(),
                               MetricKind::kGauge});
    out.push_back(MetricSample{names::kSamplePeriod,
                               names::kSamplePeriodHelp,
                               {{"module", "io"}},
                               IoSamplePeriodPtr()->load(std::memory_order_relaxed),
                               MetricKind::kGauge});
    MetricSample interval{names::kSampleInterval,
                          names::kSampleIntervalHelp,
                          {},
                          static_cast<uint64_t>(std::max(registry_->interval_ms(), 0)),
                          MetricKind::kGauge};
    interval.scale = 1e-3;
    out.push_back(std::move(interval));
    return out;
  }

 private:
  const Registry* registry_;
};

std::atomic<uint64_t> g_next_handle{1};

std::string FileNamePart(const std::string& s) {
  std::string out = s.empty() ? std::string("unknown") : s;
  for (char& c : out) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (!(std::isalnum(u) != 0 || c == '.' || c == '-' || c == '_')) c = '_';
  }
  return out;
}

}  // namespace

// For a registry that may be an abandoned, inherited one: neither read takes its lock.
struct RegistryForkAccess {
  static pid_t Owner(const Registry* r) { return r->owner_pid_; }
  static void RemoveTargetFile() {
    Registry* r = g_target_owner.load(std::memory_order_acquire);
    if (r == nullptr || r->owner_pid_ != getpid()) return;
    if (!r->target_file_.empty()) ::unlink(r->target_file_.c_str());
  }
  static void FlushPushAtExit() {
    Registry* r = g_instance.load(std::memory_order_acquire);
    if (r == nullptr || r->owner_pid_ != getpid() || !r->pushing_) return;
    r->Tick();
    r->FlushPush(1000);
  }
};

uint32_t EpSamplePeriod() { return Periods().ep.load(std::memory_order_relaxed); }

const std::atomic<uint32_t>* IoSamplePeriodPtr() { return &Periods().io; }

Sampling Registry::GetSampling() const {
  Sampling s;
  s.ep = EpSamplePeriod();
  s.io = Periods().io.load(std::memory_order_relaxed);
  s.interval_ms = interval_ms();
  return s;
}

Sampling Registry::SetSampling(int64_t ep, int64_t io, int64_t interval_ms, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error != nullptr) *error = what;
    return GetSampling();
  };
  const int64_t maxPeriod = static_cast<int64_t>(kMaxSamplePeriod);
  if (ep > maxPeriod) return fail("ep must be 0 to " + std::to_string(maxPeriod));
  if (io > maxPeriod) return fail("io must be 0 to " + std::to_string(maxPeriod));
  if (interval_ms == 0 || interval_ms > kMaxSampleIntervalMs) {
    return fail("interval_ms must be 1 to " + std::to_string(kMaxSampleIntervalMs));
  }
  if (interval_ms > 0 && this->interval_ms() <= 0) {
    return fail("this process has no metrics thread whose interval could be changed");
  }
  if (ep >= 0) Periods().ep.store(static_cast<uint32_t>(ep), std::memory_order_relaxed);
  if (io >= 0) Periods().io.store(static_cast<uint32_t>(io), std::memory_order_relaxed);
  if (interval_ms > 0) SetIntervalMs(static_cast<int>(interval_ms));
  if (ep >= 0 || io >= 0 || interval_ms > 0) {
    MORI_METRICS_INFO("metrics sampling: ep 1/{}, io 1/{}, counters read every {} ms",
                      EpSamplePeriod(), Periods().io.load(std::memory_order_relaxed),
                      this->interval_ms());
    // Now, not at the next tick: an EP op hands the new period to its kernels when sampled.
    Tick();
  }
  return GetSampling();
}

Registry::Options Registry::Options::FromEnv() {
  Options o;
  const int base = EnvInt("MORI_METRICS_PORT", 0);
  const int rank = LocalRankFromEnv();
  o.port = base > 0 ? base + std::max(rank, 0) : 0;
  o.bind_addr = EnvStr("MORI_METRICS_ADDR");
  o.port_dir = EnvStr("MORI_METRICS_PORT_DIR");
  o.host = HostId();
  o.push_url = EnvStr("MORI_METRICS_PUSH_URL");
  const int default_interval = o.push_url.empty() ? 1000 : 1;
  o.interval_ms = EnvInt("MORI_METRICS_INTERVAL_MS", default_interval);
  if (o.interval_ms <= 0) o.interval_ms = default_interval;
  if (rank >= 0) o.target_labels.emplace_back("local_rank", std::to_string(rank));
  return o;
}

Registry& Registry::Instance() {
  const pid_t pid = getpid();
  Registry* current = g_instance.load(std::memory_order_acquire);
  // Decided without a lock: one inside an inherited registry may be held by a thread lost at fork.
  while (current == nullptr || RegistryForkAccess::Owner(current) != pid) {
    auto* fresh = new Registry(Options::FromEnv());
    if (g_instance.compare_exchange_strong(current, fresh, std::memory_order_acq_rel)) {
      // The parent's registry is abandoned: its destructor would join a thread we never had.
      return *fresh;
    }
    delete fresh;  // lost the race; it never started anything
  }
  return *current;
}

Registry::Registry(Options options) : options_(std::move(options)), owner_pid_(getpid()) {
  ModuleLogger::GetInstance().InitModule(modules::METRICS);
  interval_ms_.store(std::max(options_.interval_ms, 0), std::memory_order_relaxed);
  if (!options_.push_url.empty()) {
    pushing_ = ParsePushUrl(options_.push_url).ok;
    if (!pushing_) {
      MORI_METRICS_ERROR(
          "MORI_METRICS_PUSH_URL '{}' is not http://host[:port][/prefix]; nothing "
          "is pushed",
          options_.push_url);
    }
  }
}

Registry::~Registry() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  if (push_thread_.joinable()) push_thread_.join();
  FlushPush(1000);
  if (!target_file_.empty()) {
    ::unlink(target_file_.c_str());
    Registry* self = this;
    g_target_owner.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
  }
}

uint64_t Registry::Register(const std::string& name, MetricLabels labels,
                            const MetricSource* source) {
  if (source == nullptr) return 0;
  std::lock_guard<std::mutex> lock(mu_);
  StartLocked();
  const uint64_t handle = g_next_handle.fetch_add(1, std::memory_order_relaxed);
  Source& src = sources_[handle];
  src.name = name + "#" + std::to_string(handle);
  src.labels.reserve(labels.size() + 1);
  src.labels.emplace_back("host", options_.host);
  for (auto& l : labels) src.labels.push_back(std::move(l));
  src.source = source;
  PublishLocked(src);
  return handle;
}

void Registry::Unregister(uint64_t handle) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = sources_.find(handle);
  if (it == sources_.end()) return;
  Source& src = it->second;
  PublishLocked(src);  // what happened since the last tick
  for (const std::string& key : src.series) {
    auto sit = series_.find(key);
    if (sit == series_.end()) continue;
    if (--sit->second.refs > 0) continue;
    if (server_) server_->removeSeries(sit->second.data.name, sit->second.data.labels);
    series_.erase(sit);
  }
  publisher_.Forget(src.name);
  sources_.erase(it);
}

void Registry::Tick() {
  std::lock_guard<std::mutex> lock(mu_);
  TickLocked();
}

void Registry::TickLocked() {
  for (auto& [handle, src] : sources_) PublishLocked(src);
  if (pushing_) HeartbeatLocked(WallMs());
}

std::vector<Registry::Series> Registry::Snapshot() {
  std::lock_guard<std::mutex> lock(mu_);
  TickLocked();
  std::vector<Series> out;
  out.reserve(series_.size());
  for (const auto& [key, stored] : series_) out.push_back(stored.data);
  return out;
}

uint64_t Registry::RejectedSamples() {
  std::lock_guard<std::mutex> lock(mu_);
  return publisher_.RejectedSamples();
}

void Registry::StartLocked() {
  if (started_) return;
  started_ = true;
  // Signals blocked while the threads are created (they inherit the mask): a process-directed
  // signal must be delivered to one of the application's own threads.
  sigset_t all, old;
  sigfillset(&all);
  const bool masked = pthread_sigmask(SIG_BLOCK, &all, &old) == 0;
  StartServerLocked();
  if (options_.sampling_series) {
    sampling_source_ = std::make_unique<SamplingSource>(this);
    Source& src = sources_[g_next_handle.fetch_add(1, std::memory_order_relaxed)];
    src.name = "mori_metrics_sampling";
    src.labels.emplace_back("host", options_.host);
    src.source = sampling_source_.get();
    PublishLocked(src);
  }
  if (options_.interval_ms > 0) thread_ = std::thread(&Registry::Loop, this);
  if (pushing_) {
    if (options_.push_flush_ms > 0) push_thread_ = std::thread(&Registry::PushLoop, this);
    static std::once_flag once;
    std::call_once(once, [] { std::atexit(&RegistryForkAccess::FlushPushAtExit); });
    MORI_METRICS_INFO("pushing metrics to {}: sampled every {} ms, sent every {} ms",
                      options_.push_url, options_.interval_ms, options_.push_flush_ms);
  }
  if (masked) pthread_sigmask(SIG_SETMASK, &old, nullptr);
}

void Registry::StartServerLocked() {
  if (options_.port <= 0) return;
  const int tries = std::max(options_.port_probe, 1);
  std::string last_error;
  for (int i = 0; i < tries && !server_; ++i) {
    const int port = options_.port + i;
    if (port > 65535) break;
    try {
      server_ = std::make_unique<MetricsServer>(port, options_.bind_addr);
    } catch (const std::exception& e) {
      last_error = e.what();
    }
  }
  if (!server_) {
    MORI_METRICS_ERROR("no metrics port could be bound in [{}, {}): {}. Metrics are not served",
                       options_.port, options_.port + tries, last_error);
    return;
  }
  // Tick-old values would skew a scraper's rates.  Runs on the server's thread; Tick() locks.
  server_->setScrapeHook([this] { Tick(); });
  server_->setControlHandler("/sampling", [this](const std::string& method,
                                                 const std::string& query, std::string* status) {
    return HandleSamplingRequest(method, query, status);
  });
  port_.store(server_->port(), std::memory_order_release);
  if (server_->port() != options_.port) {
    MORI_METRICS_INFO("metrics port {} was taken; serving on {}", options_.port, server_->port());
  }
  WriteTargetFileLocked();
}

void Registry::WriteTargetFileLocked() {
  if (options_.port_dir.empty()) return;
  // Written to a name the *.json glob does not match, then renamed: never read half-written.
  const std::string addr = options_.bind_addr.empty() ? options_.host : options_.bind_addr;
  std::string body = "[{\"targets\":[" + JsonString(addr + ":" + std::to_string(server_->port())) +
                     "],\"labels\":{\"pid\":" + JsonString(std::to_string(owner_pid_));
  for (const auto& [k, v] : options_.target_labels) {
    body += "," + JsonString(k) + ":" + JsonString(v);
  }
  body += "}}]\n";

  const std::string final_path = options_.port_dir + "/" + FileNamePart(options_.host) + "-" +
                                 std::to_string(owner_pid_) + ".json";
  const std::string tmp_path = final_path + ".tmp";
  {
    std::ofstream out(tmp_path, std::ios::trunc);
    out << body;
    if (!out) {
      MORI_METRICS_ERROR("cannot write the metrics target file {}", tmp_path);
      return;
    }
  }
  if (std::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    MORI_METRICS_ERROR("cannot rename {} to {}: {}", tmp_path, final_path, std::strerror(errno));
    ::unlink(tmp_path.c_str());
    return;
  }
  target_file_ = final_path;
  static std::once_flag once;
  std::call_once(once, [] { std::atexit(&RegistryForkAccess::RemoveTargetFile); });
  g_target_owner.store(this, std::memory_order_release);
}

void Registry::PublishLocked(Source& src) {
  const int64_t now_ms = pushing_ ? WallMs() : 0;
  auto touch = [&](const char* name, const char* help, const MetricLabels& labels,
                   MetricKind kind) -> Stored& {
    std::string key = SeriesKey(name, labels);
    auto [it, fresh] = series_.try_emplace(key);
    Stored& stored = it->second;
    if (fresh) {
      stored.data.name = name;
      stored.data.help = help == nullptr ? "" : help;
      stored.data.kind = kind;
      stored.data.labels = labels;
      if (pushing_ && kind != MetricKind::kHistogram) {
        stored.push_heads.push_back(PushHeadLocked(stored.data.name, labels));
      }
    }
    if (src.series.insert(std::move(key)).second) ++stored.refs;
    return stored;
  };

  MetricPublisher::Sink sink;
  sink.counter = [&](const char* name, const char* help, const MetricLabels& labels, double delta) {
    Stored& stored = touch(name, help, labels, MetricKind::kCounter);
    stored.data.value += delta;
    if (pushing_ && (delta != 0 || stored.pushed_ms == 0)) {
      KeepLocked(stored, 0, stored.data.value, now_ms);
    }
    if (server_) server_->addCounter(name, help == nullptr ? "" : help, labels, delta);
  };
  sink.gauge = [&](const char* name, const char* help, const MetricLabels& labels, double value) {
    Stored& stored = touch(name, help, labels, MetricKind::kGauge);
    if (pushing_ && (value != stored.data.value || stored.pushed_ms == 0)) {
      KeepLocked(stored, 0, value, now_ms);
    }
    stored.data.value = value;
    if (server_) server_->setGauge(name, help == nullptr ? "" : help, labels, value);
  };
  sink.histogram = [&](const char* name, const char* help, const MetricLabels& labels,
                       const std::vector<double>& bounds, const std::vector<uint64_t>& bucket_delta,
                       uint64_t count_delta, double sum_delta) {
    Stored& stored = touch(name, help, labels, MetricKind::kHistogram);
    Series& s = stored.data;
    if (s.bounds.empty()) {
      s.bounds = bounds;
      s.buckets.assign(bounds.size(), 0);
      if (pushing_) {
        for (double bound : bounds) {
          char le[40];
          std::snprintf(le, sizeof(le), "%.9g", bound);
          stored.push_heads.push_back(PushHeadLocked(s.name + "_bucket", labels, le));
        }
        stored.push_heads.push_back(PushHeadLocked(s.name + "_bucket", labels, "+Inf"));
        stored.push_heads.push_back(PushHeadLocked(s.name + "_sum", labels));
        stored.push_heads.push_back(PushHeadLocked(s.name + "_count", labels));
      }
    }
    if (s.buckets.size() == bucket_delta.size()) {
      for (std::size_t i = 0; i < bucket_delta.size(); ++i) s.buckets[i] += bucket_delta[i];
      s.count += count_delta;
      s.value += sum_delta;
    }
    if (server_) {
      server_->observeAggregated(name, help == nullptr ? "" : help, labels, bounds, bucket_delta,
                                 count_delta, sum_delta);
    }
  };
  publisher_.Publish(src.name, src.labels, *src.source, sink);
}

void Registry::Loop() {
  std::unique_lock<std::mutex> lock(mu_);
  // Against the clock: at a millisecond the time a tick takes would otherwise stretch the period.
  auto next = std::chrono::steady_clock::now();
  while (!stopping_) {
    const int ms = interval_ms_.load(std::memory_order_relaxed);
    next += std::chrono::milliseconds(ms);
    const auto now = std::chrono::steady_clock::now();
    if (next < now) next = now;  // fell behind: go on from here, no burst of ticks
    const bool interrupted = cv_.wait_until(lock, next, [&] {
      return stopping_ || interval_ms_.load(std::memory_order_relaxed) != ms;
    });
    if (stopping_) break;
    if (interrupted) {
      next = std::chrono::steady_clock::now();
      continue;
    }
    TickLocked();
  }
}

void Registry::SetIntervalMs(int ms) {
  if (ms <= 0) return;
  {
    // Under the lock the thread waits with: it cannot miss the change before going to sleep.
    std::lock_guard<std::mutex> lock(mu_);
    interval_ms_.store(ms, std::memory_order_relaxed);
  }
  cv_.notify_all();
}

std::string Registry::HandleSamplingRequest(const std::string& method, const std::string& query,
                                            std::string* status) {
  auto json = [](const Sampling& s) {
    return "{\"ep\":" + std::to_string(s.ep) + ",\"io\":" + std::to_string(s.io) +
           ",\"interval_ms\":" + std::to_string(s.interval_ms) + "}\n";
  };
  *status = "200 OK";
  if (query.empty()) return json(GetSampling());
  if (method != "POST") {
    *status = "405 Method Not Allowed";
    return "The settings are changed with POST: curl -X POST 'http://<host>:<port>/sampling?" +
           query + "'\n";
  }
  int64_t ep = -1, io = -1, interval_ms = -1;
  std::size_t pos = 0;
  while (pos < query.size()) {
    std::size_t end = query.find('&', pos);
    if (end == std::string::npos) end = query.size();
    const std::string item = query.substr(pos, end - pos);
    pos = end + 1;
    if (item.empty()) continue;
    const std::size_t eq = item.find('=');
    const std::string key = item.substr(0, eq);
    const std::string value = eq == std::string::npos ? "" : item.substr(eq + 1);
    int64_t* field = key == "ep"            ? &ep
                     : key == "io"          ? &io
                     : key == "interval_ms" ? &interval_ms
                                            : nullptr;
    char* rest = nullptr;
    errno = 0;
    const long long n = std::strtoll(value.c_str(), &rest, 10);
    if (field == nullptr || value.empty() || *rest != '\0' || errno != 0 || n < 0) {
      *status = "400 Bad Request";
      return "Cannot use '" + item +
             "'. Settings: ep=<n>, io=<n> (count one event in n; 1 all, 0 none), "
             "interval_ms=<n>\n";
    }
    *field = n;
  }
  std::string error;
  const Sampling now = SetSampling(ep, io, interval_ms, &error);
  if (!error.empty()) {
    *status = "400 Bad Request";
    return error + "\n";
  }
  return json(now);
}

const std::string* Registry::PushHeadLocked(const std::string& name, const MetricLabels& labels,
                                            const char* le) {
  std::string head = "{\"metric\":{\"__name__\":" + JsonString(name);
  for (const auto& [k, v] : labels) head += "," + JsonString(k) + ":" + JsonString(v);
  if (le != nullptr) head += ",\"le\":" + JsonString(le);
  head += ",\"instance\":" + JsonString(options_.host + ":" + std::to_string(owner_pid_)) +
          ",\"job\":\"mori_traffic\"},";
  push_heads_.push_back(std::move(head));
  return &push_heads_.back();
}

void Registry::KeepLocked(Stored& stored, std::size_t slot, double value, int64_t ts_ms) {
  stored.pushed_ms = ts_ms;
  if (slot >= stored.push_heads.size()) return;
  if (pending_.size() >= options_.push_max_points) {
    push_dropped_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  pending_.push_back(Point{stored.push_heads[slot], ts_ms, value});
}

void Registry::HeartbeatLocked(int64_t now_ms) {
  if (now_ms - heartbeat_ms_ < options_.push_heartbeat_ms) return;
  heartbeat_ms_ = now_ms;
  for (auto& [key, stored] : series_) {
    const Series& s = stored.data;
    if (s.kind == MetricKind::kHistogram) {
      // Buckets (cumulative over the bounds), +Inf, sum, count: as a scrape would see them.
      const std::size_t n = s.buckets.size();
      if (stored.push_heads.size() != n + 3) continue;
      for (std::size_t i = 0; i < n; ++i) {
        KeepLocked(stored, i, static_cast<double>(s.buckets[i]), now_ms);
      }
      KeepLocked(stored, n, static_cast<double>(s.count), now_ms);
      KeepLocked(stored, n + 1, s.value, now_ms);
      KeepLocked(stored, n + 2, static_cast<double>(s.count), now_ms);
    } else if (now_ms - stored.pushed_ms >= options_.push_heartbeat_ms) {
      KeepLocked(stored, 0, s.value, now_ms);
    }
  }
}

void Registry::FlushPush(int timeout_ms) {
  if (!pushing_) return;
  std::vector<Point> points;
  {
    std::lock_guard<std::mutex> lock(mu_);
    points.swap(pending_);
  }
  if (points.empty()) return;
  // One line per series, its points in the order they were taken.
  std::stable_sort(points.begin(), points.end(),
                   [](const Point& a, const Point& b) { return a.head < b.head; });
  std::string body;
  body.reserve(points.size() * 30);
  std::string stamps;
  for (std::size_t i = 0; i < points.size();) {
    std::size_t j = i;
    body += *points[i].head;
    body += "\"values\":[";
    stamps.clear();
    for (; j < points.size() && points[j].head == points[i].head; ++j) {
      if (j != i) {
        body += ',';
        stamps += ',';
      }
      AppendNumber(&body, points[j].value);
      stamps += std::to_string(points[j].ts_ms);
    }
    body += "],\"timestamps\":[";
    body += stamps;
    body += "]}\n";
    i = j;
  }

  std::lock_guard<std::mutex> send(push_mu_);
  std::string err;
  const int status = HttpPost(ParsePushUrl(options_.push_url), body, timeout_ms, &err);
  if (status >= 200 && status < 300) {
    push_sent_.fetch_add(points.size(), std::memory_order_relaxed);
    return;
  }
  push_dropped_.fetch_add(points.size(), std::memory_order_relaxed);
  const uint64_t failures = push_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (failures == 1 || failures % 100 == 0) {
    MORI_METRICS_ERROR("metrics push to {} failed ({}); {} points dropped so far, {} failed sends",
                       options_.push_url, status >= 0 ? "HTTP " + std::to_string(status) : err,
                       push_dropped_.load(std::memory_order_relaxed), failures);
  }
}

Registry::PushStats Registry::push_stats() const {
  PushStats s;
  s.sent = push_sent_.load(std::memory_order_relaxed);
  s.dropped = push_dropped_.load(std::memory_order_relaxed);
  s.failures = push_failures_.load(std::memory_order_relaxed);
  return s;
}

void Registry::PushLoop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (cv_.wait_for(lock, std::chrono::milliseconds(options_.push_flush_ms),
                       [this] { return stopping_; })) {
        return;
      }
    }
    FlushPush(2000);
  }
}

}  // namespace mori::metrics

/* ------------------------------------------------------------------------ */
/*  3a. Device counter block                                                */
/* ------------------------------------------------------------------------ */

namespace mori {
namespace application {
namespace {

// Set at exit: from then on a device block is not freed through HIP, which is being torn down.
// The handler is registered after HIP was initialised, so it runs before HIP's teardown.
std::atomic<bool> g_exiting{false};
std::mutex g_exitMu;  // held around the free: the handler waits one out

void MarkExiting() {
  std::lock_guard<std::mutex> lock(g_exitMu);
  g_exiting.store(true, std::memory_order_release);
}

class ScopedDevice {
 public:
  explicit ScopedDevice(int device) {
    if (hipGetDevice(&previous_) != hipSuccess) previous_ = -1;
    const bool ok = previous_ == device || hipSetDevice(device) == hipSuccess;
    switched_ = ok && previous_ != device;
  }
  ~ScopedDevice() {
    if (switched_ && previous_ >= 0) (void)hipSetDevice(previous_);
  }

 private:
  int previous_{-1};
  bool switched_{false};
};

}  // namespace

DeviceCounterBlock::DeviceCounterBlock(size_t slots, Placement placement)
    : slots_(slots), placement_(placement) {
  if (slots_ == 0) return;
  const size_t bytes = slots_ * sizeof(uint64_t);
  if (hipGetDevice(&device_) != hipSuccess) device_ = -1;

  if (placement_ == Placement::kHostMapped) {
    // Whole pages: the registration pins and maps pages.
    const size_t pageBytes = (bytes + 4095) & ~static_cast<size_t>(4095);
    void* mem = nullptr;
    if (posix_memalign(&mem, 4096, pageBytes) != 0 || mem == nullptr) {
      MORI_METRICS_ERROR("traffic counters: cannot allocate {} bytes of host memory", pageBytes);
      return;
    }
    std::memset(mem, 0, pageBytes);
    hipError_t err =
        hipHostRegister(mem, pageBytes, hipHostRegisterMapped | hipHostRegisterPortable);
    void* dev = nullptr;
    if (err == hipSuccess) {
      hostRegistered_ = true;
      err = hipHostGetDevicePointer(&dev, mem, 0);
    }
    if (err != hipSuccess || dev == nullptr) {
      MORI_METRICS_ERROR("traffic counters: cannot map host memory into the device ({})",
                         hipGetErrorString(err));
      if (hostRegistered_) hipHostUnregister(mem);
      hostRegistered_ = false;
      free(mem);
      return;
    }
    host_ = static_cast<uint64_t*>(mem);
    devicePtr_ = static_cast<uint64_t*>(dev);
    return;
  }

  void* dev = nullptr;
  hipError_t err = hipMalloc(&dev, bytes);
  if (err == hipSuccess) err = hipMemset(dev, 0, bytes);
  if (err != hipSuccess) {
    MORI_METRICS_ERROR("traffic counters: cannot set up a device block of {} bytes ({})", bytes,
                       hipGetErrorString(err));
    if (dev != nullptr) (void)hipFree(dev);
    return;
  }
  devicePtr_ = static_cast<uint64_t*>(dev);
  static std::once_flag once;
  std::call_once(once, [] { std::atexit(&MarkExiting); });
}

DeviceCounterBlock::~DeviceCounterBlock() {
  std::lock_guard<std::mutex> lock(mu_);
  if (placement_ == Placement::kHostMapped) {
    if (host_ != nullptr) {
      if (hostRegistered_) hipHostUnregister(host_);
      free(host_);
    }
    return;
  }
  if (devicePtr_ == nullptr) return;
  std::lock_guard<std::mutex> exitLock(g_exitMu);
  if (g_exiting.load(std::memory_order_acquire)) return;  // HIP is going away with the process
  ScopedDevice onDevice(device_);
  (void)hipFree(devicePtr_);
}

std::vector<uint64_t> DeviceCounterBlock::Read() const {
  std::vector<uint64_t> out(slots_, 0);
  std::lock_guard<std::mutex> lock(mu_);
  if (host_ == nullptr) return out;
  // An aligned 8-byte load is one value or the next; volatile keeps each a real load.
  const volatile uint64_t* p = host_;
  for (size_t i = 0; i < slots_; ++i) out[i] = p[i];
  return out;
}

void DeviceCounterBlock::Poke(size_t slot, uint64_t value) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (host_ == nullptr || slot >= slots_) return;
  static_cast<volatile uint64_t*>(host_)[slot] = value;
}

void DeviceCounterBlock::Write(const std::vector<uint64_t>& values) {
  std::lock_guard<std::mutex> lock(mu_);
  if (devicePtr_ == nullptr || values.size() != slots_) return;
  if (placement_ == Placement::kHostMapped) {
    std::memcpy(host_, values.data(), slots_ * sizeof(uint64_t));
    return;
  }
  ScopedDevice onDevice(device_);
  (void)hipMemcpy(devicePtr_, values.data(), slots_ * sizeof(uint64_t), hipMemcpyHostToDevice);
}

}  // namespace application
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  3b. RDMA issued by the GPU itself                                       */
/* ------------------------------------------------------------------------ */

namespace mori {
namespace application {
namespace {

using mori::metrics::MetricKind;
using mori::metrics::MetricSample;
using mori::metrics::PeIdentity;
namespace names = mori::metrics::names;

constexpr const char* kOpLabels[kRdmaTrafficOpCount] = {"put", "inline", "atomic", "get"};

}  // namespace

std::vector<MetricSample> RdmaTrafficCounters::SampleMetrics() const {
  std::vector<MetricSample> out;
  const int n = static_cast<int>(peers_.size());
  const PeIdentity& self = peers_[rank_];
  const std::vector<uint64_t> bytes = block_.Read();

  for (int op = 0; op < kRdmaTrafficOpCount; ++op) {
    const bool localIsSrc = op != kRdmaTrafficGet;
    for (int pe = 0; pe < n; ++pe) {
      const uint64_t value = bytes[RdmaTrafficSlot(op, n, pe)];
      // Published once it has carried something: a rank talks RDMA to few of a large world.
      if (value == 0) continue;
      const PeIdentity& src = localIsSrc ? self : peers_[pe];
      const PeIdentity& dst = localIsSrc ? peers_[pe] : self;
      out.push_back(MetricSample{names::kGpuRdmaBytes,
                                 names::kGpuRdmaBytesHelp,
                                 {{"api", api_},
                                  {"op", kOpLabels[op]},
                                  {"src_host", src.hostname},
                                  {"src_gpu", src.GpuLabel()},
                                  {"src_ibdev", src.ibdev},
                                  {"src_netdev", src.netdev},
                                  {"dst_host", dst.hostname},
                                  {"dst_gpu", dst.GpuLabel()},
                                  {"dst_ibdev", dst.ibdev},
                                  {"dst_netdev", dst.netdev}},
                                 value});
    }
  }
  if (!self.ibdev.empty()) {
    out.push_back(MetricSample{names::kGpuRdmaNicInfo,
                               names::kGpuRdmaNicInfoHelp,
                               {{"gpu", self.GpuLabel()},
                                {"pci", self.gpuPci},
                                {"ibdev", self.ibdev},
                                {"netdev", self.netdev}},
                               1,
                               MetricKind::kGauge});
  }
  return out;
}

RdmaTrafficCounters::RdmaTrafficCounters(std::string api, int rank, std::vector<PeIdentity> peers)
    : api_(std::move(api)),
      rank_(rank),
      peers_(std::move(peers)),
      block_(RdmaTrafficSlots(static_cast<int>(peers_.size()))) {
  if (block_.DevicePtr() == nullptr) {
    MORI_METRICS_ERROR("{} RDMA traffic counters are off: their block could not be set up", api_);
    return;
  }
  registryHandle_ = mori::metrics::Registry::Instance().Register("mori_gpu_rdma_" + api_, {}, this);
}

RdmaTrafficCounters::~RdmaTrafficCounters() {
  // First: after this returns the metrics thread no longer reads the block.
  if (registryHandle_ != 0) mori::metrics::Registry::Instance().Unregister(registryHandle_);
}

}  // namespace application
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  3c. EP                                                                  */
/* ------------------------------------------------------------------------ */

namespace mori {
namespace moe {
namespace {

using mori::metrics::MetricLabels;
using mori::metrics::MetricSample;
using mori::metrics::PeIdentity;
namespace names = mori::metrics::names;

const char* OpLabel(int op) { return op == kEpStatsDispatch ? "dispatch" : "combine"; }

// A launch that starts within this long of the previous one's end is timed from that end.
constexpr double kLaunchGapSeconds = 50e-6;

}  // namespace

std::vector<MetricSample> EpTrafficStats::SampleMetrics() const {
  std::vector<MetricSample> out;
  const EpTrafficStatsConfig& cfg = config_;
  const int n = cfg.worldSize;
  const int perNode = cfg.gpuPerNode > 0 ? cfg.gpuPerNode : 1;

  // Tell the kernels the process's EP sampling period if it changed.
  const uint32_t period = mori::metrics::EpSamplePeriod();
  if (samplePeriod_.exchange(period, std::memory_order_relaxed) != period) {
    mirror_.Poke(EpStatsPeriodOffset(n), period);
  }

  const Snapshot snap = Read();
  const bool known = static_cast<int>(cfg.peers.size()) == n;
  const std::string ibdev = known ? cfg.peers[cfg.rank].ibdev : "";
  const std::string netdev = known ? cfg.peers[cfg.rank].netdev : "";

  for (int op = 0; op < kEpStatsOpCount; ++op) {
    const uint64_t bytesPerToken = bytesPerToken_[op].load(std::memory_order_relaxed);
    for (int dir = 0; dir < 2; ++dir) {
      const std::vector<uint64_t>& tokens = dir == 0 ? snap.sent[op] : snap.recv[op];
      for (int pe = 0; pe < n; ++pe) {
        // Tokens become bytes at the size a token has NOW; earlier ones were charged at theirs.
        uint64_t& last = lastTokens_[op][dir][pe];
        if (tokens[pe] > last) bytes_[op][dir][pe] += (tokens[pe] - last) * bytesPerToken;
        last = tokens[pe];

        // This node's ranks and the other nodes' proxy ranks are published from the start, at
        // zero; any other rank only once it has carried something.
        const bool sameNode = pe / perNode == cfg.rank / perNode;
        const bool proxy = pe % perNode == cfg.rank % perNode;
        if (!sameNode && !proxy && tokens[pe] == 0) continue;

        const std::string path = pe == cfg.rank ? "local" : sameNode ? "xgmi" : "rdma";
        const std::string peerHost = known ? cfg.peers[pe].hostname : "unknown";
        const std::string peerGpu = known ? cfg.peers[pe].GpuLabel() : std::to_string(pe);
        const char* direction = dir == 0 ? "tx" : "rx";

        out.push_back(MetricSample{names::kEpTokens,
                                   names::kEpTokensHelp,
                                   {{"op", OpLabel(op)},
                                    {"direction", direction},
                                    {"path", path},
                                    {"peer_host", peerHost},
                                    {"peer_gpu", peerGpu},
                                    {"impl", cfg.impl}},
                                   tokens[pe]});
        if (path == "xgmi") {
          out.push_back(MetricSample{names::kEpXgmiBytes,
                                     names::kEpXgmiBytesHelp,
                                     {{"peer_gpu", peerGpu},
                                      {"op", OpLabel(op)},
                                      {"direction", direction},
                                      {"impl", cfg.impl}},
                                     bytes_[op][dir][pe]});
        } else if (path == "rdma") {
          out.push_back(MetricSample{names::kEpRdmaBytes,
                                     names::kEpRdmaBytesHelp,
                                     {{"ibdev", ibdev},
                                      {"netdev", netdev},
                                      {"peer_host", peerHost},
                                      {"peer_gpu", peerGpu},
                                      {"op", OpLabel(op)},
                                      {"direction", direction},
                                      {"impl", cfg.impl}},
                                     bytes_[op][dir][pe]});
        }
      }
    }
    out.push_back(MetricSample{names::kEpCalls,
                               names::kEpCallsHelp,
                               {{"op", OpLabel(op)}, {"kernel", cfg.kernel}, {"impl", cfg.impl}},
                               snap.calls[op]});
    if (ticksPerSecond_ > 0) {
      // Ticks of the device clock; the publisher turns each increment into seconds.
      MetricSample seconds{names::kEpKernelSeconds,
                           names::kEpKernelSecondsHelp,
                           {{"op", OpLabel(op)}, {"kernel", cfg.kernel}, {"impl", cfg.impl}},
                           snap.ticks[op]};
      seconds.scale = 1.0 / ticksPerSecond_;
      out.push_back(std::move(seconds));
    }
  }
  return out;
}

EpTrafficStats::EpTrafficStats(EpTrafficStatsConfig config)
    : config_(std::move(config)),
      block_(EpStatsSlots(config_.worldSize), application::DeviceCounterBlock::Placement::kDevice),
      // The whole layout: the kernels read the sampling period from the mirror at the same slot.
      mirror_(EpStatsSlots(config_.worldSize),
              application::DeviceCounterBlock::Placement::kHostMapped) {
  const int n = config_.worldSize;
  if (block_.DevicePtr() == nullptr || mirror_.DevicePtr() == nullptr) {
    MORI_METRICS_ERROR(
        "EP traffic statistics are off for this op: their block could not be set up");
    return;
  }
  int device = -1, khz = 0;
  if (hipGetDevice(&device) == hipSuccess &&
      hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) == hipSuccess &&
      khz > 0) {
    ticksPerSecond_ = khz * 1e3;
  }
  std::vector<uint64_t> init(EpStatsSlots(n), 0);
  init[EpStatsGapLimitOffset(n)] = static_cast<uint64_t>(kLaunchGapSeconds * ticksPerSecond_);
  init[EpStatsMirrorOffset(n)] = reinterpret_cast<uint64_t>(mirror_.DevicePtr());
  // The sampling period, and what a combine that runs before any dispatch goes by.
  const uint32_t period = mori::metrics::EpSamplePeriod();
  samplePeriod_.store(period, std::memory_order_relaxed);
  init[EpStatsPeriodOffset(n)] = period;
  init[EpStatsWeightOffset(n)] = period == 0 ? 0 : 1;
  block_.Write(init);
  mirror_.Poke(EpStatsPeriodOffset(n), period);
  for (auto& perOp : lastTokens_) {
    for (auto& perDir : perOp) perDir.assign(n, 0);
  }
  for (auto& perOp : bytes_) {
    for (auto& perDir : perOp) perDir.assign(n, 0);
  }
  ready_ = true;

  const bool known = static_cast<int>(config_.peers.size()) == config_.worldSize;
  MetricLabels labels = {
      {"gpu", known ? config_.peers[config_.rank].GpuLabel() : std::to_string(config_.rank)},
      {"rank", std::to_string(config_.rank)}};
  registryHandle_ =
      mori::metrics::Registry::Instance().Register("mori_ep", std::move(labels), this);
}

EpTrafficStats::~EpTrafficStats() {
  // First: after this returns the metrics thread no longer reads the block.
  if (registryHandle_ != 0) mori::metrics::Registry::Instance().Unregister(registryHandle_);
}

EpTrafficStats::Snapshot EpTrafficStats::Read() const {
  Snapshot snap;
  const int n = config_.worldSize;
  for (int op = 0; op < kEpStatsOpCount; ++op) {
    snap.sent[op].assign(n, 0);
    snap.recv[op].assign(n, 0);
  }
  // The mirror: reading it calls nothing in HIP.
  const std::vector<uint64_t> p = mirror_.Read();
  for (int op = 0; op < kEpStatsOpCount; ++op) {
    for (int pe = 0; pe < n; ++pe) {
      snap.sent[op][pe] = p[EpStatsSentOffset(op, n) + pe];
      snap.recv[op][pe] = p[EpStatsRecvOffset(op, n) + pe];
    }
    snap.calls[op] = p[EpStatsCallsOffset(n) + op];
    snap.ticks[op] = p[EpStatsTicksOffset(n) + op];
  }
  return snap;
}

}  // namespace moe
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  3d. IO                                                                  */
/* ------------------------------------------------------------------------ */

namespace mori {
namespace io {
namespace internal {
namespace {

using mori::metrics::MetricKind;
using mori::metrics::MetricLabels;
using mori::metrics::MetricSample;
namespace names = mori::metrics::names;

// Leaked on purpose: the metrics thread may be sampling while static objects are destroyed.
const std::vector<double>& LatencyBounds() {
  static const std::vector<double>* bounds = [] {
    auto* b = new std::vector<double>();
    for (uint64_t ns : kIoLatencyBoundsNs) b->push_back(static_cast<double>(ns) * 1e-9);
    return b;
  }();
  return *bounds;
}

const char* OpLabel(int op) { return op == static_cast<int>(IoOp::kRead) ? "read" : "write"; }

uint64_t Load(const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); }

}  // namespace

std::vector<MetricSample> IoMetrics::SampleMetrics() const {
  std::vector<MetricSample> out;
  std::vector<const IoFlowCounters*> flows;
  std::vector<const NicFlowCounters*> nicFlows;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& entry : flows_) flows.push_back(entry.second);
    for (const auto& entry : nicFlows_) nicFlows.push_back(entry.second);
  }

  struct Latency {
    std::vector<uint64_t> buckets = std::vector<uint64_t>(kIoLatencyBoundCount, 0);
    uint64_t count = 0;
    uint64_t sumNs = 0;
  };
  std::map<std::tuple<std::string, int, std::string>, Latency> latency;
  std::map<std::pair<std::string, std::string>, int64_t> inflight;

  for (const IoFlowCounters* flow : flows) {
    const IoFlowIdentity& id = flow->id;
    for (int op = 0; op < kIoOpCount; ++op) {
      const IoOpCounters& c = flow->op[op];
      // A write moves data local -> remote, a read remote -> local.
      const bool write = op == static_cast<int>(IoOp::kWrite);
      MetricLabels labels = {{"backend", id.backend},
                             {"op", OpLabel(op)},
                             {"src_host", write ? host_ : id.remoteHost},
                             {"src_gpu", write ? id.localGpu : id.remoteGpu},
                             {"dst_host", write ? id.remoteHost : host_},
                             {"dst_gpu", write ? id.remoteGpu : id.localGpu}};

      const uint64_t submitted = Load(c.submitted);
      const uint64_t ok = Load(c.completedOk);
      const uint64_t failed = Load(c.completedFailed);
      const uint64_t unobserved = Load(c.completedUnobserved);
      out.push_back(MetricSample{names::kIoSubmitted, names::kIoSubmittedHelp, labels, submitted});
      out.push_back(MetricSample{names::kIoSubmittedBytes, names::kIoSubmittedBytesHelp, labels,
                                 Load(c.submittedBytes)});
      out.push_back(MetricSample{names::kIoCompletedBytes, names::kIoCompletedBytesHelp, labels,
                                 Load(c.completedBytes)});
      for (const auto& [status, n] : {std::pair<const char*, uint64_t>{"ok", ok},
                                      {"failed", failed},
                                      {"unobserved", unobserved}}) {
        MetricLabels withStatus = labels;
        withStatus.emplace_back("status", status);
        out.push_back(
            MetricSample{names::kIoCompleted, names::kIoCompletedHelp, std::move(withStatus), n});
      }

      Latency& lat = latency[{id.backend, op, id.localGpu}];
      uint64_t running = 0;
      for (int b = 0; b < kIoLatencyBoundCount; ++b) {
        running += Load(c.latencyBuckets[b]);
        lat.buckets[b] += running;  // cumulative over the bounds
      }
      lat.count += running + Load(c.latencyBuckets[kIoLatencyBoundCount]);
      lat.sumNs += Load(c.latencyNs);

      // The counters are read one by one: the difference can dip below zero; clamped below.
      inflight[{id.backend, id.localGpu}] +=
          static_cast<int64_t>(submitted) - static_cast<int64_t>(ok + failed + unobserved);
    }
  }

  for (auto& [key, lat] : latency) {
    MetricSample s;
    s.name = names::kIoLatency;
    s.help = names::kIoLatencyHelp;
    s.labels = {{"backend", std::get<0>(key)},
                {"op", OpLabel(std::get<1>(key))},
                {"gpu", std::get<2>(key)}};
    s.kind = MetricKind::kHistogram;
    s.scale = 1e-9;
    s.bounds = &LatencyBounds();
    s.buckets = std::move(lat.buckets);
    s.count = lat.count;
    s.value = lat.sumNs;
    out.push_back(std::move(s));
  }
  for (const auto& [key, n] : inflight) {
    out.push_back(MetricSample{names::kIoInflight,
                               names::kIoInflightHelp,
                               {{"backend", key.first}, {"gpu", key.second}},
                               static_cast<uint64_t>(std::max<int64_t>(n, 0)),
                               MetricKind::kGauge});
  }

  std::set<std::string> localNics;
  for (const NicFlowCounters* flow : nicFlows) {
    const NicFlowIdentity& id = flow->id;
    for (int op = 0; op < kIoOpCount; ++op) {
      const bool write = op == static_cast<int>(IoOp::kWrite);
      out.push_back(MetricSample{names::kIoNicBytes,
                                 names::kIoNicBytesHelp,
                                 {{"op", OpLabel(op)},
                                  {"src_host", write ? host_ : id.remoteHost},
                                  {"src_ibdev", write ? id.ibdev : id.remoteIbdev},
                                  {"src_netdev", write ? id.netdev : id.remoteNetdev},
                                  {"dst_host", write ? id.remoteHost : host_},
                                  {"dst_ibdev", write ? id.remoteIbdev : id.ibdev},
                                  {"dst_netdev", write ? id.remoteNetdev : id.netdev}},
                                 Load(flow->bytes[op])});
    }
    // Once per local NIC.
    if (!localNics.insert(id.ibdev).second) continue;
    auto speed = linkSpeed_.find(id.ibdev);
    if (speed == linkSpeed_.end()) {
      speed = linkSpeed_
                  .emplace(id.ibdev,
                           static_cast<uint64_t>(mori::metrics::IbdevLinkBytesPerSecond(id.ibdev)))
                  .first;
    }
    if (speed->second > 0) {
      out.push_back(MetricSample{names::kIoNicLinkSpeed,
                                 names::kIoNicLinkSpeedHelp,
                                 {{"ibdev", id.ibdev}, {"netdev", id.netdev}},
                                 speed->second,
                                 MetricKind::kGauge});
    }
  }
  return out;
}

IoMetrics::IoMetrics() : host_(mori::metrics::HostId()) {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) == 0) rawHostname_ = buf;
}

std::string IoMetrics::HostLabel(const std::string& hostname) const {
  if (hostname.empty()) return "unknown";
  if (hostname == rawHostname_) return host_;
  return hostname;
}

IoMetrics* IoMetrics::Get() {
  static IoMetrics* table = mori::metrics::Enabled() ? new IoMetrics() : nullptr;
  return table;
}

// A block's identity is written under the table's lock, before its pointer leaves the function,
// and never again: whoever holds the pointer may read it without a lock.

IoFlowCounters* IoMetrics::Flow(IoFlowIdentity id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = flows_.find(id);
  if (it != flows_.end()) return it->second;
  flowBlocks_.emplace_back();
  IoFlowCounters* block = &flowBlocks_.back();
  block->id = id;
  block->samplePeriod = mori::metrics::IoSamplePeriodPtr();
  flows_.emplace(std::move(id), block);
  return block;
}

NicFlowCounters* IoMetrics::NicFlow(NicFlowIdentity id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = nicFlows_.find(id);
  if (it != nicFlows_.end()) return it->second;
  nicFlowBlocks_.emplace_back();
  NicFlowCounters* block = &nicFlowBlocks_.back();
  block->id = id;
  nicFlows_.emplace(std::move(id), block);
  return block;
}

void IoMetrics::ZeroAll() {
  auto zero = [](std::atomic<uint64_t>& a) { a.store(0, std::memory_order_relaxed); };
  for (IoFlowCounters& f : flowBlocks_) {
    for (IoOpCounters& c : f.op) {
      zero(c.submitted);
      zero(c.submittedBytes);
      zero(c.completedOk);
      zero(c.completedFailed);
      zero(c.completedUnobserved);
      zero(c.completedBytes);
      zero(c.latencyNs);
      for (auto& b : c.latencyBuckets) zero(b);
    }
  }
  for (NicFlowCounters& n : nicFlowBlocks_) {
    for (auto& a : n.bytes) zero(a);
  }
}

void IoMetrics::EnsureRegistered() {
  const void* before = registeredWith_.load(std::memory_order_acquire);
  const void* now = &mori::metrics::Registry::Instance();
  if (now == before) return;
  if (!registeredWith_.compare_exchange_strong(before, now, std::memory_order_acq_rel)) return;
  if (before != nullptr) {
    // A forked child: the counters hold the parent's traffic.  Zeroed before the new registry's
    // first sample, without the table's lock: one held at the fork would never be released.
    ZeroAll();
  }
  // Never unregistered: the table is never torn down.
  mori::metrics::Registry::Instance().Register("mori_io", {}, this);
}

}  // namespace internal
}  // namespace io
}  // namespace mori
