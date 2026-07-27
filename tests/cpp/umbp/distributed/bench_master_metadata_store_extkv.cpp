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

// Store-level microbenchmark for the EXTERNAL-KV hot path of
// IMasterMetadataStore. Sibling of bench_master_metadata_store.cpp (which
// covers RouteGet / Exists / Heartbeat); this one isolates the external-KV
// methods that SGLang's KV-events integration drives, so the per-op cost of
// each external-KV interface can be compared directly across backends
// (in-memory / single Redis / sharded Redis / Redis Cluster / Dragonfly) with
// no gRPC or RDMA layer in the way.
//
// Why these methods (SGLang usage, see sglang umbp_store.py + mori
// master_server.cpp):
//   - MatchExternalKv (count_as_hit=true) is THE hot read: the prefix-lookup /
//     route path asks "which live nodes hold these block hashes?", and each
//     match atomically bumps the per-hash hit counter + last_seen. This is the
//     external-KV analogue of BatchLookupBlockForRouteGet and the single
//     interface most able to move end-to-end latency.
//   - RegisterExternalKvIfAlive backs report_external_kv_blocks, fired on every
//     BlockStored KV-cache event — the dominant external-KV WRITE.
//   - UnregisterExternalKv backs revoke_external_kv_blocks (BlockRemoved).
//   - GetExternalKvHitCounts backs the eviction / admin hit-count read.
//
// Design note that this bench is built to expose: external-KV + hit state all
// hang off ONE control hash tag ({umbp:<ns>}), i.e. a single Redis slot /
// instance. UMBP_REDIS_SHARD_URIS / UMBP_REDIS_BLOCK_SHARDS shard the BLOCK
// index, NOT this path — so sharded-Redis and Cluster are not expected to scale
// the external-KV hot path, while a single multi-threaded Dragonfly instance
// can. Run this bench across topologies to confirm.
//
// Backend is selected by the factory via UMBP_METADATA_BACKEND / UMBP_REDIS_URI,
// exactly as the master picks its store:
//   UMBP_METADATA_BACKEND=inmemory                     ./bench...extkv --workload match ...
//   UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=tcp://127.0.0.1:6379 ./bench...extkv ...
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
  // match | match_nohit | report | revoke | hitcounts | mixed
  std::string workload = "match";
  int threads = 8;
  int nodes = 8;  // registered clients the hashes are spread across
  double seconds = 5.0;
  int keys = 50000;        // external-kv hash keyspace (seeded across nodes)
  int batch = 32;          // hashes per op (real prefix match is tens)
  double hit_ratio = 1.0;  // fraction of queried hashes that actually exist
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

// Seeded external-kv hash present in the store (registered during setup).
std::string SeededHash(int i) { return "h/" + std::to_string(i); }
// Hash guaranteed absent from the store (index beyond the seeded keyspace) —
// used to synthesize misses for --hit-ratio < 1.
std::string MissHash(int i, int keys) { return "h/" + std::to_string(keys + i); }

void Usage() {
  std::fprintf(
      stderr,
      "Usage: bench_master_metadata_store_extkv [options]\n"
      "  --workload match|match_nohit|report|revoke|hitcounts|mixed  (default match)\n"
      "      match       MatchExternalKv(count_as_hit=true)  -- hot read (+hit-count write)\n"
      "      match_nohit MatchExternalKv(count_as_hit=false) -- pure read (isolates hit cost)\n"
      "      report      RegisterExternalKvIfAlive           -- hot write (BlockStored)\n"
      "      revoke      UnregisterExternalKv                -- write (BlockRemoved)\n"
      "      hitcounts   GetExternalKvHitCounts              -- eviction/admin read\n"
      "      mixed       1:1 match(count_as_hit) : report    -- read/write blend\n"
      "  --threads N        (default 8)\n"
      "  --nodes N          registered clients hashes spread across (default 8)\n"
      "  --seconds F        (default 5)\n"
      "  --warmup-seconds F (default 1)\n"
      "  --keys N           external-kv hash keyspace (default 50000)\n"
      "  --batch N          hashes per op (default 32)\n"
      "  --hit-ratio F      fraction of queried hashes that exist (default 1.0)\n"
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
    else if (a == "--nodes")
      o.nodes = std::atoi(need());
    else if (a == "--seconds")
      o.seconds = std::atof(need());
    else if (a == "--warmup-seconds")
      o.warmup_seconds = std::atof(need());
    else if (a == "--keys")
      o.keys = std::atoi(need());
    else if (a == "--batch")
      o.batch = std::atoi(need());
    else if (a == "--hit-ratio")
      o.hit_ratio = std::atof(need());
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
  if (o.nodes < 1) o.nodes = 1;
  if (o.hit_ratio < 0.0) o.hit_ratio = 0.0;
  if (o.hit_ratio > 1.0) o.hit_ratio = 1.0;

  const char* backend = std::getenv("UMBP_METADATA_BACKEND");
  std::string backend_name = backend ? backend : "inmemory";
  // Disambiguate Redis vs Dragonfly vs Valkey / single vs sharded vs cluster
  // (all use UMBP_METADATA_BACKEND=redis): append the target so the CSV is
  // self-describing.
  if (backend_name == "redis") {
    const char* uri = std::getenv("UMBP_REDIS_URI");
    const char* shards = std::getenv("UMBP_REDIS_SHARD_URIS");
    const char* cluster = std::getenv("UMBP_REDIS_CLUSTER");
    std::string tag = uri ? uri : "tcp://127.0.0.1:6379";
    if (cluster && std::string(cluster) != "0") tag = "cluster:" + tag;
    if (shards && shards[0]) tag = "sharded:" + std::string(shards);
    // The CSV is comma-separated; a shard-URI list contains commas — replace
    // them so the artifact stays parseable.
    for (char& c : tag)
      if (c == ',') c = ';';
    backend_name += "[" + tag + "]";
  }

  std::unique_ptr<IMasterMetadataStore> store;
  try {
    store = MakeMasterMetadataStore();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "failed to build store: %s\n", e.what());
    return 2;
  }

  const int nodes = o.nodes;
  const auto now = std::chrono::system_clock::now();
  for (int n = 0; n < nodes; ++n) {
    store->RegisterClient(MakeReg(n), now, std::chrono::seconds(120));
  }

  // Seed `keys` external-kv hashes, round-robin across nodes, at DRAM tier, in
  // chunks so no single Lua script is enormous. Mirrors report_external_kv_blocks
  // fan-in from many nodes.
  {
    constexpr int kChunk = 512;
    std::vector<std::vector<std::string>> pending(nodes);
    auto flush = [&](int n) {
      if (pending[n].empty()) return;
      store->RegisterExternalKvIfAlive(MakeReg(n).node_id, pending[n], TierType::DRAM);
      pending[n].clear();
    };
    for (int i = 0; i < o.keys; ++i) {
      const int n = i % nodes;
      pending[n].push_back(SeededHash(i));
      if (static_cast<int>(pending[n].size()) >= kChunk) flush(n);
    }
    for (int n = 0; n < nodes; ++n) flush(n);
  }

  std::atomic<bool> measuring{false};
  std::atomic<bool> stop{false};

  auto worker = [&](int tid, std::vector<double>* lat, uint64_t* ops) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (tid + 1));
    std::uniform_int_distribution<int> key_dist(0, std::max(0, o.keys - 1));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    const std::string node = MakeReg(tid % nodes).node_id;
    // report/revoke draw from this thread's own node keyspace slice so writes
    // stay attributable to one node (as a real peer's events would be).
    std::uniform_int_distribution<int> node_key_dist(0, std::max(0, o.keys / nodes - 1));
    uint64_t report_seq = 0;  // grows the write keyspace (fresh BlockStored churn)

    std::vector<double> local;
    local.reserve(1 << 20);
    uint64_t local_ops = 0;
    bool mix_toggle = false;

    std::vector<std::string> hashes(o.batch);
    while (!stop.load(std::memory_order_relaxed)) {
      std::string wl = o.workload;
      if (wl == "mixed") {
        wl = mix_toggle ? "report" : "match";
        mix_toggle = !mix_toggle;
      }

      const auto t0 = Clock::now();
      if (wl == "match" || wl == "match_nohit") {
        for (int b = 0; b < o.batch; ++b) {
          if (coin(rng) < o.hit_ratio)
            hashes[b] = SeededHash(key_dist(rng));
          else
            hashes[b] = MissHash(key_dist(rng), o.keys);
        }
        auto r = store->MatchExternalKv(hashes, /*count_as_hit=*/wl == "match",
                                        std::chrono::system_clock::now());
        (void)r;
      } else if (wl == "hitcounts") {
        for (int b = 0; b < o.batch; ++b) hashes[b] = SeededHash(key_dist(rng));
        auto r = store->GetExternalKvHitCounts(hashes);
        (void)r;
      } else if (wl == "report") {
        // Fresh hashes per op in this node's slice => models BlockStored churn.
        for (int b = 0; b < o.batch; ++b) {
          const int base = (tid % nodes) + nodes * node_key_dist(rng);
          hashes[b] = "hw/" + std::to_string(tid) + "/" + std::to_string(report_seq++) + "/" +
                      std::to_string(base);
        }
        auto r = store->RegisterExternalKvIfAlive(node, hashes, TierType::DRAM);
        (void)r;
      } else {  // revoke
        for (int b = 0; b < o.batch; ++b) {
          const int idx = (tid % nodes) + nodes * node_key_dist(rng);
          hashes[b] = SeededHash(idx);
        }
        store->UnregisterExternalKv(node, hashes, TierType::DRAM);
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
      "backend,workload,threads,nodes,batch,keys,hit_ratio,wall_s,ops,ops_per_s,keys_per_s,"
      "lat_us_p50,lat_us_p95,lat_us_p99,lat_us_max\n");
  std::printf("%s,%s,%d,%d,%d,%d,%.2f,%.3f,%llu,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f\n",
              backend_name.c_str(), o.workload.c_str(), o.threads, o.nodes, o.batch, o.keys,
              o.hit_ratio, wall_s, static_cast<unsigned long long>(total_ops), ops_per_s,
              keys_per_s, Percentile(all, 0.50), Percentile(all, 0.95), Percentile(all, 0.99),
              all.empty() ? 0.0 : *std::max_element(all.begin(), all.end()));
  std::fflush(stdout);
  return 0;
}
