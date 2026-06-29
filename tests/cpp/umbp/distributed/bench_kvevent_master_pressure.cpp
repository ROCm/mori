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

// Multi-client master KV-event pressure benchmark (simulates an LLM-agent-style
// put -> LLM-call-gap -> get workload).
//
// PRIMARY GOAL: load the master and quantify whether its per-RPC latency
// (BatchRoutePut / BatchLookup / BatchRouteGet / Heartbeat ...) degrades as QPS
// and client count grow, comparing three kv-event propagation schemes:
//
//   mode=baseline    default heartbeat interval (env-driven, ~5s default)
//   mode=compressed  short heartbeat interval (launch with UMBP_HEARTBEAT_TTL_SEC=1
//                    UMBP_HEARTBEAT_INTERVAL_DIVISOR=10 => 100ms)
//   mode=flush       writer calls MasterClient::FlushHeartbeat() right after each
//                    BatchPut, forcing an immediate (async, coalesced) heartbeat drain
//
// The heartbeat interval is NOT a CLI knob: the master computes it from
// UMBP_HEARTBEAT_TTL_SEC / UMBP_HEARTBEAT_INTERVAL_DIVISOR and echoes it at
// RegisterClient time.  Those env reads are cached in function-local statics, so a
// single process can only exercise one interval -- launch one process per scenario.
//
// SECONDARY GOAL (auxiliary metric): the read-after-write miss rate.  A key is
// globally visible only after its owning peer ships the KvEvent(ADD) on the next
// heartbeat, so a reader querying the master within the LLM-call gap may miss.
// We classify every read into hit / miss / rpc-error (a failed BatchLookup must NOT
// be counted as a "not visible" miss -- BatchExists() folds RPC failure into all
// false, so we call MasterClient::BatchLookup directly and inspect grpc::Status).
//
// Topology: one in-process MasterServer + N PoolClients on loopback (see
// test_cross_node_smoke.cpp for the multi-client setup pattern).  Put placement is
// chosen by the master; clients only own *which key* they produce and which keys
// they read, expressed as a key-pairing / read-fanout schedule (--pattern).
//
// Master RPC latency is read back from the master's own Prometheus endpoint
// (--metrics-port): clients already instrument every RPC via ScopedRpcTimer and
// report it through ReportMetrics; we scrape mori_umbp_master_client_rpc_latency_seconds
// and take a delta between a post-warmup baseline and the end of the run.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/routing/route_put_strategy.h"

using mori::umbp::ClientRegistryConfig;
using mori::umbp::ConfigurableRoutePutStrategy;
using mori::umbp::MasterClient;
using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::PoolClient;
using mori::umbp::PoolClientConfig;
using mori::umbp::TierType;

namespace {

// ---------------------------------------------------------------------------
//  Small infrastructure helpers
// ---------------------------------------------------------------------------

// Ask the OS for an ephemeral TCP port (bind 0 -> getsockname -> close), used for
// each PoolClient's peer-service port.  Mirrors test_cross_node_smoke.cpp.
uint16_t NextPeerServicePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("NextPeerServicePort: socket() failed");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = INADDR_ANY;
  uint16_t port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
      port = ntohs(addr.sin_port);
    }
  }
  ::close(fd);
  if (port == 0) throw std::runtime_error("NextPeerServicePort: no ephemeral port");
  return port;
}

// Raw-socket GET /metrics from a local Prometheus exposition endpoint.  Returns
// the full HTTP response (headers + body) or "" on failure.  Copied from
// test_umbp_client_metrics.cpp::FetchPrometheusMetrics.
std::string FetchPrometheusMetrics(int port) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) return "";
  struct timeval tv{5, 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    return "";
  }
  const char* req = "GET /metrics HTTP/1.0\r\nHost: localhost\r\n\r\n";
  ::send(sock, req, std::strlen(req), 0);
  std::string resp;
  char buf[8192];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) resp.append(buf, static_cast<size_t>(n));
  ::close(sock);
  return resp;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * (static_cast<double>(v.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return v[lo];
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// ---------------------------------------------------------------------------
//  Prometheus histogram parsing (mori_umbp_master_client_rpc_latency_seconds)
// ---------------------------------------------------------------------------
//
// We aggregate per `rpc=` label across the status= dimension (ok+error) -- both
// are cumulative histograms over the same bounds, so per-le cumulative counts add.
// le="+Inf" is stored as +infinity; the count there equals the histogram _count.

struct RpcHist {
  std::map<double, double> cum;  // le (seconds) -> cumulative count
  double Total() const {
    if (cum.empty()) return 0.0;
    return cum.rbegin()->second;  // +Inf bucket
  }
};

// Extract value of label `key="..."` from a metric line; "" if absent.
std::string ExtractLabel(const std::string& line, const std::string& key) {
  const std::string pat = key + "=\"";
  size_t s = line.find(pat);
  if (s == std::string::npos) return "";
  s += pat.size();
  size_t e = line.find('"', s);
  if (e == std::string::npos) return "";
  return line.substr(s, e - s);
}

// Parse the body, grouping `<metric>_bucket` lines by their `rpc` label.
std::map<std::string, RpcHist> ParseRpcLatencyHist(const std::string& body,
                                                   const std::string& metric) {
  const std::string bucket_name = metric + "_bucket";
  std::map<std::string, RpcHist> out;
  size_t pos = 0;
  while (pos < body.size()) {
    size_t nl = body.find('\n', pos);
    std::string line = body.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = (nl == std::string::npos) ? body.size() : nl + 1;
    if (line.empty() || line.front() == '#') continue;
    if (line.find(bucket_name) == std::string::npos) continue;
    const std::string rpc = ExtractLabel(line, "rpc");
    const std::string le = ExtractLabel(line, "le");
    if (rpc.empty() || le.empty()) continue;
    size_t sp = line.rfind(' ');
    if (sp == std::string::npos) continue;
    double count = 0.0;
    try {
      count = std::stod(line.substr(sp + 1));
    } catch (...) {
      continue;
    }
    const double le_val =
        (le == "+Inf") ? std::numeric_limits<double>::infinity() : std::strtod(le.c_str(), nullptr);
    out[rpc].cum[le_val] += count;
  }
  return out;
}

// Quantile from a cumulative-bucket histogram (counts are cumulative by le).
// Linear interpolation inside the chosen bucket; returns seconds.
double HistQuantile(const std::map<double, double>& cum, double q) {
  if (cum.empty()) return 0.0;
  const double total = cum.rbegin()->second;
  if (total <= 0.0) return 0.0;
  const double rank = q * total;
  double prev_bound = 0.0;
  double prev_cum = 0.0;
  for (const auto& [le, c] : cum) {
    if (c >= rank) {
      if (std::isinf(le)) return prev_bound;  // cannot interpolate into +Inf
      if (c <= prev_cum) return le;
      const double frac = (rank - prev_cum) / (c - prev_cum);
      return prev_bound + frac * (le - prev_bound);
    }
    prev_bound = std::isinf(le) ? prev_bound : le;
    prev_cum = c;
  }
  return prev_bound;
}

// final - baseline, per rpc, per le (cumulative stays monotonic under subtraction).
std::map<std::string, RpcHist> DeltaHist(const std::map<std::string, RpcHist>& fin,
                                         const std::map<std::string, RpcHist>& base) {
  std::map<std::string, RpcHist> out;
  for (const auto& [rpc, h] : fin) {
    RpcHist d;
    auto bit = base.find(rpc);
    for (const auto& [le, c] : h.cum) {
      double b = 0.0;
      if (bit != base.end()) {
        auto le_it = bit->second.cum.find(le);
        if (le_it != bit->second.cum.end()) b = le_it->second;
      }
      d.cum[le] = std::max(0.0, c - b);
    }
    out[rpc] = std::move(d);
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Options
// ---------------------------------------------------------------------------

enum class Pattern { kBroadcast, kRotate };
enum class GetMode { kExists, kFetch, kBoth };

struct BenchOpts {
  std::string mode = "baseline";  // baseline | compressed | flush
  Pattern pattern = Pattern::kRotate;
  GetMode get_mode = GetMode::kExists;
  size_t clients = 4;
  size_t rounds = 100;
  size_t warmup_rounds = 5;
  size_t batch = 16;
  size_t page_bytes = 4096;
  double gap_ms = 250.0;
  double gap_jitter_ms = 0.0;
  bool round_barrier = false;
  long read_lag_rounds = 1;
  bool randomize = false;  // mix broadcast/rotate per round
  uint64_t seed = 1234;
  bool with_external_kv = false;
  size_t key_space = 0;         // 0 = unique keys per round; >0 = recycle per producer
  int metrics_port = 0;         // 0 = no Prometheus scrape
  double keep_master_secs = 0;  // keep master alive at the end for Grafana
  double timeout_ms = 5000.0;   // gRPC-side deadline is separate; informational
  // Put placement (injected directly into the master, NOT via env).
  std::string put_algo = "most_available";  // most_available | random
  std::string put_affinity = "local";       // none | same | local
};

void Usage() {
  std::fprintf(stderr,
               "Usage: bench_kvevent_master_pressure [options]\n"
               "  --mode baseline|compressed|flush   (default baseline)\n"
               "  --pattern broadcast|rotate         (default rotate)\n"
               "  --get-mode exists|fetch|both       (default exists)\n"
               "  --clients N        (default 4)\n"
               "  --rounds N         (default 100)\n"
               "  --warmup-rounds N  (default 5)\n"
               "  --batch N          (default 16)\n"
               "  --page-bytes N     (default 4096)\n"
               "  --gap-ms F         (default 250)\n"
               "  --gap-jitter-ms F  (default 0)\n"
               "  --round-barrier    (default off)\n"
               "  --read-lag-rounds N(default 1; 0 needs --round-barrier)\n"
               "  --randomize        (mix broadcast/rotate per round)\n"
               "  --seed N           (default 1234)\n"
               "  --with-external-kv (default off)\n"
               "  --key-space N      (0=unique per round; >0 recycle)\n"
               "  --metrics-port P   (0=off; >0 enable master Prometheus + scrape)\n"
               "  --keep-master-secs F (default 0)\n"
               "  --put-algo most_available|random   (default most_available)\n"
               "  --put-affinity none|same|local     (default local)\n"
               "Heartbeat interval is env-driven (UMBP_HEARTBEAT_TTL_SEC,\n"
               "UMBP_HEARTBEAT_INTERVAL_DIVISOR); launch one process per scenario.\n");
}

bool ParseArgs(int argc, char** argv, BenchOpts* o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--mode") {
      o->mode = need("--mode");
    } else if (a == "--pattern") {
      std::string p = need("--pattern");
      if (p == "broadcast")
        o->pattern = Pattern::kBroadcast;
      else if (p == "rotate")
        o->pattern = Pattern::kRotate;
      else {
        std::fprintf(stderr, "bad --pattern %s\n", p.c_str());
        return false;
      }
    } else if (a == "--get-mode") {
      std::string g = need("--get-mode");
      if (g == "exists")
        o->get_mode = GetMode::kExists;
      else if (g == "fetch")
        o->get_mode = GetMode::kFetch;
      else if (g == "both")
        o->get_mode = GetMode::kBoth;
      else {
        std::fprintf(stderr, "bad --get-mode %s\n", g.c_str());
        return false;
      }
    } else if (a == "--clients") {
      o->clients = std::strtoull(need("--clients"), nullptr, 10);
    } else if (a == "--rounds") {
      o->rounds = std::strtoull(need("--rounds"), nullptr, 10);
    } else if (a == "--warmup-rounds") {
      o->warmup_rounds = std::strtoull(need("--warmup-rounds"), nullptr, 10);
    } else if (a == "--batch") {
      o->batch = std::strtoull(need("--batch"), nullptr, 10);
    } else if (a == "--page-bytes") {
      o->page_bytes = std::strtoull(need("--page-bytes"), nullptr, 10);
    } else if (a == "--gap-ms") {
      o->gap_ms = std::strtod(need("--gap-ms"), nullptr);
    } else if (a == "--gap-jitter-ms") {
      o->gap_jitter_ms = std::strtod(need("--gap-jitter-ms"), nullptr);
    } else if (a == "--round-barrier") {
      o->round_barrier = true;
    } else if (a == "--read-lag-rounds") {
      o->read_lag_rounds = std::strtol(need("--read-lag-rounds"), nullptr, 10);
    } else if (a == "--randomize") {
      o->randomize = true;
    } else if (a == "--seed") {
      o->seed = std::strtoull(need("--seed"), nullptr, 10);
    } else if (a == "--with-external-kv") {
      o->with_external_kv = true;
    } else if (a == "--key-space") {
      o->key_space = std::strtoull(need("--key-space"), nullptr, 10);
    } else if (a == "--metrics-port") {
      o->metrics_port = static_cast<int>(std::strtol(need("--metrics-port"), nullptr, 10));
    } else if (a == "--keep-master-secs") {
      o->keep_master_secs = std::strtod(need("--keep-master-secs"), nullptr);
    } else if (a == "--timeout-ms") {
      o->timeout_ms = std::strtod(need("--timeout-ms"), nullptr);
    } else if (a == "--put-algo") {
      o->put_algo = need("--put-algo");
    } else if (a == "--put-affinity") {
      o->put_affinity = need("--put-affinity");
    } else if (a == "-h" || a == "--help") {
      Usage();
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      Usage();
      return false;
    }
  }
  if (o->mode != "baseline" && o->mode != "compressed" && o->mode != "flush") {
    std::fprintf(stderr, "bad --mode %s\n", o->mode.c_str());
    return false;
  }
  if (o->clients < 1) {
    std::fprintf(stderr, "--clients must be >= 1\n");
    return false;
  }
  if (o->batch < 1) {
    std::fprintf(stderr, "--batch must be >= 1\n");
    return false;
  }
  return true;
}

ConfigurableRoutePutStrategy::SelectAlgo ParseAlgo(const std::string& s) {
  return s == "random" ? ConfigurableRoutePutStrategy::SelectAlgo::kRandom
                       : ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable;
}
ConfigurableRoutePutStrategy::NodeAffinity ParseAffinity(const std::string& s) {
  if (s == "same") return ConfigurableRoutePutStrategy::NodeAffinity::kSame;
  if (s == "local") return ConfigurableRoutePutStrategy::NodeAffinity::kLocal;
  return ConfigurableRoutePutStrategy::NodeAffinity::kNone;
}

// ---------------------------------------------------------------------------
//  Shared run state
// ---------------------------------------------------------------------------

// Reusable N-thread barrier (std::barrier is C++20; keep this self-contained).
class Barrier {
 public:
  explicit Barrier(size_t n) : threshold_(n), count_(n), generation_(0) {}
  void Wait() {
    std::unique_lock<std::mutex> lk(m_);
    const size_t gen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen != generation_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  size_t threshold_;
  size_t count_;
  size_t generation_;
};

struct Metrics {
  std::atomic<uint64_t> put_calls{0};
  std::atomic<uint64_t> put_keys{0};
  std::atomic<uint64_t> put_fail_keys{0};
  std::atomic<uint64_t> lookup_calls{0};
  std::atomic<uint64_t> fetch_calls{0};
  std::atomic<uint64_t> get_keys{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
  std::atomic<uint64_t> rpc_error_keys{0};
  std::atomic<uint64_t> ext_report_calls{0};
  std::atomic<uint64_t> ext_match_calls{0};
  std::atomic<uint64_t> ext_match_hits{0};
  std::mutex lat_mtx;
  std::vector<double> put_ms;
  std::vector<double> lookup_ms;
  std::vector<double> fetch_ms;
  void MergeLatency(const std::vector<double>& p, const std::vector<double>& l,
                    const std::vector<double>& f) {
    std::lock_guard<std::mutex> g(lat_mtx);
    put_ms.insert(put_ms.end(), p.begin(), p.end());
    lookup_ms.insert(lookup_ms.end(), l.begin(), l.end());
    fetch_ms.insert(fetch_ms.end(), f.begin(), f.end());
  }
};

// ---------------------------------------------------------------------------
//  Key schedule / worker
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
inline double DurMs(Clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

// Deterministic per-round pattern so producers and consumers agree across threads.
Pattern PatternForRound(const BenchOpts& o, size_t r) {
  if (!o.randomize) return o.pattern;
  std::mt19937_64 rng(o.seed ^ (r * 0x9E3779B97F4A7C15ULL));
  return (rng() & 1ULL) ? Pattern::kBroadcast : Pattern::kRotate;
}

std::string MakeKey(size_t producer, size_t round, size_t b, const BenchOpts& o) {
  if (o.key_space > 0) {
    const size_t idx = (round * o.batch + b) % o.key_space;
    return "k/" + std::to_string(producer) + "/" + std::to_string(idx);
  }
  return "k/" + std::to_string(producer) + "/" + std::to_string(round) + "/" + std::to_string(b);
}

inline bool IsProducer(Pattern p, size_t id) { return p == Pattern::kRotate || id == 0; }

struct Ctx {
  const BenchOpts* o = nullptr;
  std::vector<std::unique_ptr<PoolClient>>* clients = nullptr;
  std::vector<std::vector<char>>* src_bufs = nullptr;
  std::vector<std::vector<char>>* dst_bufs = nullptr;
  Metrics* m = nullptr;
  Barrier* barrier = nullptr;  // nullptr => free-running
  size_t N = 0;
};

// Run global rounds [round_begin, round_end) for one client.  `measure` gates
// whether samples land in the aggregate Metrics (warmup phase passes false).
void Worker(size_t id, size_t round_begin, size_t round_end, bool measure, Ctx ctx) {
  const BenchOpts& o = *ctx.o;
  PoolClient* cli = (*ctx.clients)[id].get();
  const size_t N = ctx.N;
  const size_t batch = o.batch;
  const size_t page = o.page_bytes;

  char* src = (*ctx.src_bufs)[id].data();
  char* dst = (o.get_mode == GetMode::kExists) ? nullptr : (*ctx.dst_bufs)[id].data();

  std::vector<const void*> srcs(batch);
  std::vector<size_t> sizes(batch, page);
  for (size_t b = 0; b < batch; ++b) srcs[b] = src + b * page;
  std::vector<void*> dsts(batch);
  if (dst)
    for (size_t b = 0; b < batch; ++b) dsts[b] = dst + b * page;

  std::mt19937_64 jitter_rng(o.seed ^ (0xABCDEFULL + id));
  std::uniform_real_distribution<double> jitter(-o.gap_jitter_ms, o.gap_jitter_ms);

  std::vector<double> put_ms, lookup_ms, fetch_ms;

  for (size_t r = round_begin; r < round_end; ++r) {
    const Pattern pat = PatternForRound(o, r);

    // ---- PUT ----
    if (IsProducer(pat, id)) {
      std::vector<std::string> keys(batch);
      for (size_t b = 0; b < batch; ++b) keys[b] = MakeKey(id, r, b, o);
      const auto t0 = Clock::now();
      auto res = cli->BatchPut(keys, srcs, sizes);
      const auto t1 = Clock::now();
      if (measure) {
        put_ms.push_back(DurMs(t1 - t0));
        ctx.m->put_calls.fetch_add(1, std::memory_order_relaxed);
        ctx.m->put_keys.fetch_add(batch, std::memory_order_relaxed);
        size_t fails = 0;
        for (bool ok : res)
          if (!ok) ++fails;
        if (fails) ctx.m->put_fail_keys.fetch_add(fails, std::memory_order_relaxed);
      }
      if (o.mode == "flush") cli->Master().FlushHeartbeat();
      if (o.with_external_kv) {
        const bool ok = cli->ReportExternalKvBlocks(keys, TierType::DRAM);
        if (measure && ok) ctx.m->ext_report_calls.fetch_add(1, std::memory_order_relaxed);
      }
    }

    // ---- BARRIER (all clients finish put before any get) ----
    if (ctx.barrier) ctx.barrier->Wait();

    // ---- LLM-call gap ----
    double gap = o.gap_ms + (o.gap_jitter_ms > 0 ? jitter(jitter_rng) : 0.0);
    if (gap > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(gap * 1000.0)));
    }

    // ---- GET (read keys produced `read_lag_rounds` ago) ----
    const long rr = static_cast<long>(r) - o.read_lag_rounds;
    if (rr < 0) continue;
    const Pattern pat_rr = PatternForRound(o, static_cast<size_t>(rr));
    const size_t producer =
        (pat_rr == Pattern::kBroadcast) ? 0 : (id + static_cast<size_t>(rr)) % N;
    std::vector<std::string> rkeys(batch);
    for (size_t b = 0; b < batch; ++b) rkeys[b] = MakeKey(producer, static_cast<size_t>(rr), b, o);

    if (o.get_mode == GetMode::kExists || o.get_mode == GetMode::kBoth) {
      std::vector<bool> found;
      const auto t0 = Clock::now();
      grpc::Status st = cli->Master().BatchLookup(rkeys, &found);
      const auto t1 = Clock::now();
      if (measure) {
        lookup_ms.push_back(DurMs(t1 - t0));
        ctx.m->lookup_calls.fetch_add(1, std::memory_order_relaxed);
        ctx.m->get_keys.fetch_add(batch, std::memory_order_relaxed);
        if (!st.ok() || found.size() != batch) {
          // [necessary] do NOT fold an RPC failure into the visibility-miss count.
          ctx.m->rpc_error_keys.fetch_add(batch, std::memory_order_relaxed);
        } else {
          size_t h = 0;
          for (bool f : found)
            if (f) ++h;
          ctx.m->hit.fetch_add(h, std::memory_order_relaxed);
          ctx.m->miss.fetch_add(batch - h, std::memory_order_relaxed);
        }
      }
    }
    if (o.get_mode == GetMode::kFetch || o.get_mode == GetMode::kBoth) {
      const auto t0 = Clock::now();
      auto bg = cli->BatchGet(rkeys, dsts, sizes);
      const auto t1 = Clock::now();
      if (measure) {
        fetch_ms.push_back(DurMs(t1 - t0));
        ctx.m->fetch_calls.fetch_add(1, std::memory_order_relaxed);
        if (o.get_mode == GetMode::kFetch) {
          // fetch-only classification is coarse: BatchGet cannot split RPC error from
          // not-found.  BatchRouteGet RPC errors are visible separately in
          // mori_umbp_master_client_rpc_errors_total{rpc="BatchRouteGet"}.
          ctx.m->get_keys.fetch_add(batch, std::memory_order_relaxed);
          size_t h = 0;
          for (bool f : bg)
            if (f) ++h;
          ctx.m->hit.fetch_add(h, std::memory_order_relaxed);
          ctx.m->miss.fetch_add(batch - h, std::memory_order_relaxed);
        }
      }
    }
    if (o.with_external_kv) {
      std::vector<MasterClient::ExternalKvNodeMatch> matches;
      const bool ok = cli->MatchExternalKv(rkeys, &matches, /*count_as_hit=*/false);
      if (measure && ok) {
        ctx.m->ext_match_calls.fetch_add(1, std::memory_order_relaxed);
        size_t matched = 0;
        for (auto& mm : matches) matched += mm.MatchedHashCount();
        ctx.m->ext_match_hits.fetch_add(matched, std::memory_order_relaxed);
      }
    }
  }
  ctx.m->MergeLatency(put_ms, lookup_ms, fetch_ms);
}

// Spawn one thread per client over [begin,end); barrier is sized to clients when on.
void RunPhase(Ctx ctx, size_t begin, size_t end, bool measure) {
  std::vector<std::thread> threads;
  threads.reserve(ctx.N);
  for (size_t id = 0; id < ctx.N; ++id) {
    threads.emplace_back(Worker, id, begin, end, measure, ctx);
  }
  for (auto& t : threads) t.join();
}

constexpr const char* kRpcLatencyMetric = "mori_umbp_master_client_rpc_latency_seconds";

}  // namespace

int main(int argc, char** argv) {
  BenchOpts o;
  if (!ParseArgs(argc, argv, &o)) return 2;
  const size_t N = o.clients;

  // ---- master (put strategy injected directly; NOT via env) ----
  MasterServerConfig mcfg;
  mcfg.listen_address = "0.0.0.0:0";
  mcfg.metrics_port = o.metrics_port;
  mcfg.registry_config = ClientRegistryConfig::FromEnvironment();
  mcfg.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(ParseAlgo(o.put_algo),
                                                                     ParseAffinity(o.put_affinity));
  mcfg.route_put_algo = o.put_algo;
  mcfg.route_put_affinity = o.put_affinity;
  auto master = std::make_unique<MasterServer>(std::move(mcfg));
  std::thread server_thread([&] { master->Run(); });
  for (int i = 0; i < 500 && master->GetBoundPort() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (master->GetBoundPort() == 0) {
    std::fprintf(stderr, "master failed to start\n");
    return 2;
  }
  const std::string master_addr = "localhost:" + std::to_string(master->GetBoundPort());

  // ---- buffer sizing ----
  const size_t total_rounds = o.warmup_rounds + o.rounds;
  const size_t produced_keys_max = (o.key_space > 0) ? o.key_space : total_rounds * o.batch;
  const size_t owned_bytes = produced_keys_max * o.page_bytes;
  const size_t peer_buf_bytes = owned_bytes * 2 + (size_t(32) << 20);  // 2x + slack
  const size_t io_bytes = o.batch * o.page_bytes;                      // src / dst staging

  // ---- clients ----
  std::vector<std::vector<char>> dram_bufs(N);
  std::vector<std::vector<char>> src_bufs(N);
  std::vector<std::vector<char>> dst_bufs(N);
  std::vector<std::unique_ptr<PoolClient>> clients(N);
  const bool need_dst = (o.get_mode != GetMode::kExists);

  for (size_t id = 0; id < N; ++id) {
    dram_bufs[id].assign(peer_buf_bytes, 0);
    src_bufs[id].assign(io_bytes, 0x5a);
    if (need_dst) dst_bufs[id].assign(io_bytes, 0);

    PoolClientConfig cfg;
    cfg.master_config.node_id = "node-" + std::to_string(id);
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_addr;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = o.page_bytes;
    cfg.dram_buffers = {{dram_bufs[id].data(), dram_bufs[id].size()}};
    cfg.tier_capacities = {{TierType::DRAM, {dram_bufs[id].size(), dram_bufs[id].size()}}};
    clients[id] = std::make_unique<PoolClient>(std::move(cfg));
    if (!clients[id]->Init()) {
      std::fprintf(stderr, "client %zu init failed\n", id);
      return 2;
    }
    clients[id]->RegisterMemory(src_bufs[id].data(), src_bufs[id].size());
    if (need_dst) clients[id]->RegisterMemory(dst_bufs[id].data(), dst_bufs[id].size());
  }

  // Let registration + first heartbeat settle so membership is live.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  Metrics m;
  Ctx ctx;
  ctx.o = &o;
  ctx.clients = &clients;
  ctx.src_bufs = &src_bufs;
  ctx.dst_bufs = &dst_bufs;
  ctx.m = &m;
  ctx.N = N;

  // ---- warmup (free-running; not measured; populates owned keys) ----
  if (o.warmup_rounds > 0) {
    ctx.barrier = nullptr;
    RunPhase(ctx, 0, o.warmup_rounds, /*measure=*/false);
  }

  // ---- baseline Prometheus snapshot (after warmup, clients alive) ----
  std::map<std::string, RpcHist> baseline;
  if (o.metrics_port > 0) {
    // Give clients a couple of ReportMetrics flush ticks to ship warmup RPCs.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const std::string body = FetchPrometheusMetrics(o.metrics_port);
    if (body.empty()) std::fprintf(stderr, "warning: baseline /metrics scrape failed\n");
    baseline = ParseRpcLatencyHist(body, kRpcLatencyMetric);
  }

  // ---- measured phase ----
  std::unique_ptr<Barrier> barrier;
  if (o.round_barrier) barrier = std::make_unique<Barrier>(N);
  ctx.barrier = barrier.get();

  const auto t_start = Clock::now();
  RunPhase(ctx, o.warmup_rounds, o.warmup_rounds + o.rounds, /*measure=*/true);
  const auto t_end = Clock::now();
  const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

  const uint64_t put_calls = m.put_calls.load();
  const uint64_t get_calls = m.lookup_calls.load() + m.fetch_calls.load();
  const double put_qps = wall_s > 0 ? put_calls / wall_s : 0.0;
  const double get_qps = wall_s > 0 ? get_calls / wall_s : 0.0;
  const uint64_t classified = m.hit.load() + m.miss.load();
  const double hit_rate = classified > 0 ? double(m.hit.load()) / double(classified) : 0.0;
  const double miss_rate = classified > 0 ? double(m.miss.load()) / double(classified) : 0.0;
  const uint64_t get_total = m.get_keys.load();
  const double rpc_err_rate =
      get_total > 0 ? double(m.rpc_error_keys.load()) / double(get_total) : 0.0;

  // ---- shutdown clients (flushes their buffered metrics to the master) ----
  for (size_t id = 0; id < N; ++id) clients[id]->Shutdown();

  // ---- final Prometheus snapshot + delta ----
  std::map<std::string, RpcHist> rpc_delta;
  if (o.metrics_port > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const std::string body = FetchPrometheusMetrics(o.metrics_port);
    if (body.empty()) std::fprintf(stderr, "warning: final /metrics scrape failed\n");
    const auto final_hist = ParseRpcLatencyHist(body, kRpcLatencyMetric);
    rpc_delta = DeltaHist(final_hist, baseline);
  }

  // ---- segment 1: workload CSV ----
  const char* pat_s = (o.pattern == Pattern::kBroadcast) ? "broadcast" : "rotate";
  if (o.randomize) pat_s = "mixed";
  const char* gm_s = o.get_mode == GetMode::kExists  ? "exists"
                     : o.get_mode == GetMode::kFetch ? "fetch"
                                                     : "both";
  std::printf(
      "mode,pattern,get_mode,clients,rounds,batch,page_bytes,gap_ms,round_barrier,read_lag,"
      "put_affinity,put_algo,wall_s,put_qps,get_qps,put_calls,get_calls,hit,miss,rpc_err_keys,"
      "put_fail_keys,hit_rate,miss_rate,rpc_err_rate,ext_match_hits,"
      "put_ms_p50,put_ms_p95,lookup_ms_p50,lookup_ms_p95,fetch_ms_p50,fetch_ms_p95\n");
  std::printf(
      "%s,%s,%s,%zu,%zu,%zu,%zu,%.1f,%d,%ld,%s,%s,%.3f,%.1f,%.1f,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%.4f,%.4f,%.4f,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
      o.mode.c_str(), pat_s, gm_s, N, o.rounds, o.batch, o.page_bytes, o.gap_ms,
      o.round_barrier ? 1 : 0, o.read_lag_rounds, o.put_affinity.c_str(), o.put_algo.c_str(),
      wall_s, put_qps, get_qps, (unsigned long long)put_calls, (unsigned long long)get_calls,
      (unsigned long long)m.hit.load(), (unsigned long long)m.miss.load(),
      (unsigned long long)m.rpc_error_keys.load(), (unsigned long long)m.put_fail_keys.load(),
      hit_rate, miss_rate, rpc_err_rate, (unsigned long long)m.ext_match_hits.load(),
      Percentile(m.put_ms, 0.50), Percentile(m.put_ms, 0.95), Percentile(m.lookup_ms, 0.50),
      Percentile(m.lookup_ms, 0.95), Percentile(m.fetch_ms, 0.50), Percentile(m.fetch_ms, 0.95));

  // ---- segment 2: master RPC latency CSV (delta over measured window) ----
  if (o.metrics_port > 0) {
    std::printf("\n");
    std::printf("rpc,count,p50_ms,p95_ms,p99_ms\n");
    for (const auto& [rpc, h] : rpc_delta) {
      const double cnt = h.Total();
      if (cnt <= 0.0) continue;
      std::printf("%s,%.0f,%.3f,%.3f,%.3f\n", rpc.c_str(), cnt, HistQuantile(h.cum, 0.50) * 1000.0,
                  HistQuantile(h.cum, 0.95) * 1000.0, HistQuantile(h.cum, 0.99) * 1000.0);
    }
  }
  std::fflush(stdout);

  // ---- keep master up for live Grafana scraping, then tear down ----
  if (o.keep_master_secs > 0) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int64_t>(o.keep_master_secs * 1000.0)));
  }
  master->Shutdown();
  if (server_thread.joinable()) server_thread.join();
  return 0;
}
