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

// Store-level microbenchmark for IMasterMetadataStore — the Phase 1 go/no-go
// signal for the Redis metadata backend. It isolates exactly the hot-path
// methods the design calls out (BatchLookupBlockForRouteGet, BatchExistsBlock,
// ApplyHeartbeat) and measures per-operation latency + throughput, so the cost
// of moving master metadata from an in-process shared_mutex read (nanoseconds)
// to a network + Lua round trip (microseconds) is measured directly, with no
// RDMA data plane or gRPC layer in the way.
//
// Backend is selected by the factory via UMBP_METADATA_BACKEND / UMBP_REDIS_URI,
// exactly as the master picks its store:
//   UMBP_METADATA_BACKEND=inmemory                     ./bench_master_metadata_store ...
//   UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=tcp://127.0.0.1:6379 ./bench... ...
//
// One process runs one workload against one backend; launch it per scenario.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/master_metadata_store_factory.h"
#include "umbp/distributed/types.h"

using namespace mori::umbp;
using Clock = std::chrono::steady_clock;

namespace {

struct Opts {
  std::string workload = "routeget";  // routeget | exists | heartbeat | mixed
  int threads = 8;
  double seconds = 5.0;
  int keys = 50000;
  int batch = 32;
  double warmup_seconds = 1.0;
};

double DurUs(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

double Percentile(std::vector<double>& v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * (static_cast<double>(v.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return v[lo];
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

ClientRegistration MakeReg(int n) {
  ClientRegistration r;
  r.node_id = "node-" + std::to_string(n);
  r.node_address = "127.0.0.1";
  r.peer_address = "127.0.0.1:" + std::to_string(17000 + n);
  r.tier_capacities[TierType::DRAM] = TierCapacity{1ULL << 34, 1ULL << 34};
  r.tags = {"role=bench"};
  return r;
}

std::string SeededKey(int i) { return "k/" + std::to_string(i); }

void Usage() {
  std::fprintf(stderr,
               "Usage: bench_master_metadata_store [options]\n"
               "  --workload routeget|exists|heartbeat|mixed  (default routeget)\n"
               "  --threads N        (default 8)\n"
               "  --seconds F        (default 5)\n"
               "  --warmup-seconds F (default 1)\n"
               "  --keys N           (default 50000)\n"
               "  --batch N          (default 32)\n"
               "Backend is chosen via UMBP_METADATA_BACKEND / UMBP_REDIS_URI.\n");
}

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() -> const char* {
      if (i + 1 >= argc) {
        Usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--workload")
      o.workload = need();
    else if (a == "--threads")
      o.threads = std::atoi(need());
    else if (a == "--seconds")
      o.seconds = std::atof(need());
    else if (a == "--warmup-seconds")
      o.warmup_seconds = std::atof(need());
    else if (a == "--keys")
      o.keys = std::atoi(need());
    else if (a == "--batch")
      o.batch = std::atoi(need());
    else if (a == "-h" || a == "--help") {
      Usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      Usage();
      return 2;
    }
  }
  if (o.threads < 1) o.threads = 1;

  const char* backend = std::getenv("UMBP_METADATA_BACKEND");
  std::string backend_name = backend ? backend : "inmemory";
  // Disambiguate Redis vs Dragonfly vs Valkey (all use UMBP_METADATA_BACKEND=
  // redis): append the target URI so the CSV artifact is self-describing.
  if (backend_name == "redis") {
    const char* uri = std::getenv("UMBP_REDIS_URI");
    backend_name += "[" + std::string(uri ? uri : "tcp://127.0.0.1:6379") + "]";
  }

  std::unique_ptr<IMasterMetadataStore> store;
  try {
    store = MakeMasterMetadataStore();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "failed to build store: %s\n", e.what());
    return 2;
  }

  // One node per thread so the heartbeat workload keeps a private monotonic seq
  // (concurrent heartbeats to one node would seq-gap by design).
  const int nodes = o.threads;
  const auto now = std::chrono::system_clock::now();
  for (int n = 0; n < nodes; ++n) {
    store->RegisterClient(MakeReg(n), now, std::chrono::seconds(120));
  }

  // Seed `keys` block locations, distributed round-robin across nodes, via
  // delta heartbeats (chunked so no single Lua script is enormous). Track the
  // next seq per node for the heartbeat workload.
  std::vector<uint64_t> next_seq(nodes, 1);
  {
    constexpr int kChunk = 256;
    std::vector<std::vector<KvEvent>> pending(nodes);
    auto flush = [&](int n) {
      if (pending[n].empty()) return;
      store->ApplyHeartbeat(MakeReg(n).node_id, next_seq[n]++, now, MakeReg(n).tier_capacities,
                            pending[n], /*is_full_sync=*/false);
      pending[n].clear();
    };
    for (int i = 0; i < o.keys; ++i) {
      const int n = i % nodes;
      pending[n].push_back(KvEvent{KvEvent::Kind::ADD, SeededKey(i), TierType::DRAM, 4096});
      if (static_cast<int>(pending[n].size()) >= kChunk) flush(n);
    }
    for (int n = 0; n < nodes; ++n) flush(n);
  }

  std::atomic<bool> measuring{false};
  std::atomic<bool> stop{false};

  auto worker = [&](int tid, std::vector<double>* lat, uint64_t* ops) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (tid + 1));
    std::uniform_int_distribution<int> key_dist(0, std::max(0, o.keys - 1));
    std::uniform_int_distribution<int> wl_dist(0, 1);
    const std::string node = MakeReg(tid % nodes).node_id;
    const auto caps = MakeReg(tid % nodes).tier_capacities;
    uint64_t hb_seq = next_seq[tid % nodes];
    uint64_t hb_key = 0;
    std::vector<double> local;
    local.reserve(1 << 20);
    uint64_t local_ops = 0;

    std::vector<std::string> keys(o.batch);
    while (!stop.load(std::memory_order_relaxed)) {
      std::string wl = o.workload;
      if (wl == "mixed") wl = wl_dist(rng) ? "routeget" : "heartbeat";

      const auto t0 = Clock::now();
      if (wl == "routeget") {
        for (int b = 0; b < o.batch; ++b) keys[b] = SeededKey(key_dist(rng));
        auto r = store->BatchLookupBlockForRouteGet(keys, {}, std::chrono::system_clock::now(),
                                                    std::chrono::seconds(10));
        (void)r;
      } else if (wl == "exists") {
        for (int b = 0; b < o.batch; ++b) keys[b] = SeededKey(key_dist(rng));
        auto r = store->BatchExistsBlock(keys);
        (void)r;
      } else {  // heartbeat
        std::vector<KvEvent> events;
        events.reserve(o.batch);
        for (int b = 0; b < o.batch; ++b) {
          events.push_back(KvEvent{KvEvent::Kind::ADD,
                                   "hb/" + std::to_string(tid) + "/" + std::to_string(hb_key++),
                                   TierType::DRAM, 4096});
        }
        auto r = store->ApplyHeartbeat(node, hb_seq++, std::chrono::system_clock::now(), caps,
                                       events, /*is_full_sync=*/false);
        (void)r;
      }
      const auto t1 = Clock::now();
      if (measuring.load(std::memory_order_relaxed)) {
        local.push_back(DurUs(t1 - t0));
        ++local_ops;
      }
    }
    *lat = std::move(local);
    *ops = local_ops;
  };

  std::vector<std::vector<double>> lats(o.threads);
  std::vector<uint64_t> ops(o.threads, 0);
  std::vector<std::thread> threads;
  threads.reserve(o.threads);
  for (int t = 0; t < o.threads; ++t) {
    threads.emplace_back(worker, t, &lats[t], &ops[t]);
  }

  std::this_thread::sleep_for(std::chrono::duration<double>(o.warmup_seconds));
  measuring.store(true, std::memory_order_relaxed);
  const auto t_start = Clock::now();
  std::this_thread::sleep_for(std::chrono::duration<double>(o.seconds));
  measuring.store(false, std::memory_order_relaxed);
  const auto t_end = Clock::now();
  stop.store(true, std::memory_order_relaxed);
  for (auto& th : threads) th.join();

  std::vector<double> all;
  uint64_t total_ops = 0;
  for (int t = 0; t < o.threads; ++t) {
    all.insert(all.end(), lats[t].begin(), lats[t].end());
    total_ops += ops[t];
  }
  const double wall_s = std::chrono::duration<double>(t_end - t_start).count();
  const double ops_per_s = wall_s > 0 ? total_ops / wall_s : 0.0;
  const double keys_per_s = ops_per_s * o.batch;

  std::printf(
      "backend,workload,threads,batch,keys,wall_s,ops,ops_per_s,keys_per_s,"
      "lat_us_p50,lat_us_p95,lat_us_p99,lat_us_max\n");
  std::printf("%s,%s,%d,%d,%d,%.3f,%llu,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f\n", backend_name.c_str(),
              o.workload.c_str(), o.threads, o.batch, o.keys, wall_s,
              static_cast<unsigned long long>(total_ops), ops_per_s, keys_per_s,
              Percentile(all, 0.50), Percentile(all, 0.95), Percentile(all, 0.99),
              all.empty() ? 0.0 : *std::max_element(all.begin(), all.end()));
  std::fflush(stdout);
  return 0;
}
