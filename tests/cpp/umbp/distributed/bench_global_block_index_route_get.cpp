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

// GlobalBlockIndex::BatchLookupForRouteGet micro-bench (pure logic; no gRPC /
// RDMA).  This is the hot master-side path that BatchRouteGet drives: for a
// batch of keys it does, under a single shared_lock, a hash lookup +
// exclude-node filter + (on a non-empty result) an atomic access-count bump and
// lease grant.  Because the lock is shared, multiple caller threads can run the
// path concurrently; --threads measures that read-scaling.
//
// The index is pre-populated via ApplyEvents (the only real mutator) with
// `index-keys` keys, each replicated across `locs-per-key` of `nodes` peers, so
// the per-key location vector and the exclude filter have realistic length.
//
// Usage:
//   bench_umbp_global_block_index_route_get
//       [--index-keys N] [--batch N] [--nodes N] [--locs-per-key N]
//       [--exclude N] [--hit-pct P] [--threads N] [--iters N] [--lease-ms N]
//       [--sweep batch=1,8,64,512]
//
// CSV (stdout):
//   index_keys,batch,nodes,locs_per_key,exclude,hit_pct,threads,iters,
//   wall_us_per_call,Mlookups_s,locs_per_call

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/types.h"

using mori::umbp::GlobalBlockIndex;
using mori::umbp::KvEvent;
using mori::umbp::Location;
using mori::umbp::TierType;

namespace {

struct BenchOpts {
  size_t index_keys = 100000;  // keys resident in the index
  size_t batch = 64;           // keys per BatchLookupForRouteGet call
  size_t nodes = 8;            // distinct peer node_ids holding locations
  size_t locs_per_key = 2;     // replicas (distinct nodes) per key
  size_t exclude = 0;          // node_ids in the exclude set
  size_t hit_pct = 100;        // % of looked-up keys that exist in the index
  size_t threads = 1;          // concurrent caller threads (shared_lock read scaling)
  size_t iters = 2000;         // measured calls per thread
  size_t lease_ms = 10;        // lease_duration handed to the method
  std::vector<size_t> sweep;   // batch sweep
};

void Usage() {
  std::cerr << "Usage: bench_umbp_global_block_index_route_get\n"
            << "    [--index-keys N] [--batch N] [--nodes N] [--locs-per-key N]\n"
            << "    [--exclude N] [--hit-pct P] [--threads N] [--iters N] [--lease-ms N]\n"
            << "    [--sweep batch=1,8,64,512]\n";
}

bool ParseArgs(int argc, char** argv, BenchOpts* o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    auto u64 = [&](const char* what) { return std::strtoull(next(what), nullptr, 10); };
    if (a == "--index-keys") {
      o->index_keys = u64("--index-keys");
    } else if (a == "--batch") {
      o->batch = u64("--batch");
    } else if (a == "--nodes") {
      o->nodes = u64("--nodes");
    } else if (a == "--locs-per-key") {
      o->locs_per_key = u64("--locs-per-key");
    } else if (a == "--exclude") {
      o->exclude = u64("--exclude");
    } else if (a == "--hit-pct") {
      o->hit_pct = u64("--hit-pct");
    } else if (a == "--threads") {
      o->threads = u64("--threads");
    } else if (a == "--iters") {
      o->iters = u64("--iters");
    } else if (a == "--lease-ms") {
      o->lease_ms = u64("--lease-ms");
    } else if (a == "--sweep") {
      std::string s = next("--sweep");
      auto eq = s.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--sweep expects 'batch=N1,N2,...'\n";
        return false;
      }
      std::stringstream ss(s.substr(eq + 1));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) o->sweep.push_back(std::strtoull(tok.c_str(), nullptr, 10));
      }
    } else if (a == "-h" || a == "--help") {
      Usage();
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      Usage();
      return false;
    }
  }
  if (o->locs_per_key < 1) o->locs_per_key = 1;
  if (o->nodes < o->locs_per_key) o->nodes = o->locs_per_key;
  if (o->threads < 1) o->threads = 1;
  if (o->hit_pct > 100) o->hit_pct = 100;
  return true;
}

std::string KeyName(size_t i) { return "k" + std::to_string(i); }
std::string NodeName(size_t n) { return "node-" + std::to_string(n); }

// Populate the index with `index_keys` keys.  Key i is owned by nodes
// (i + 0 .. i + locs_per_key-1) mod nodes, so replicas are spread evenly and
// every node carries a comparable share.  We batch all ADDs for a given node
// into one ApplyEvents call (ApplyEvents groups by node_id).
void Populate(GlobalBlockIndex* idx, const BenchOpts& o) {
  std::vector<std::vector<KvEvent>> per_node(o.nodes);
  for (size_t i = 0; i < o.index_keys; ++i) {
    for (size_t r = 0; r < o.locs_per_key; ++r) {
      size_t node = (i + r) % o.nodes;
      KvEvent ev;
      ev.kind = KvEvent::Kind::ADD;
      ev.key = KeyName(i);
      ev.tier = TierType::DRAM;
      ev.size = 4096;
      per_node[node].push_back(std::move(ev));
    }
  }
  for (size_t n = 0; n < o.nodes; ++n) {
    idx->ApplyEvents(NodeName(n), per_node[n]);
  }
}

// Pre-build the per-iteration key batches so the measured loop touches only the
// method under test (no string formatting or RNG in the hot path).  `hit_pct`
// of the keys index into the populated range; the rest are guaranteed misses.
std::vector<std::vector<std::string>> BuildBatches(const BenchOpts& o, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> hit_dist(0, o.index_keys ? o.index_keys - 1 : 0);
  std::uniform_int_distribution<size_t> pct(1, 100);
  std::vector<std::vector<std::string>> batches(o.iters);
  for (size_t it = 0; it < o.iters; ++it) {
    auto& b = batches[it];
    b.reserve(o.batch);
    for (size_t j = 0; j < o.batch; ++j) {
      if (pct(rng) <= o.hit_pct && o.index_keys > 0) {
        b.push_back(KeyName(hit_dist(rng)));
      } else {
        b.push_back("miss-" + std::to_string(it) + "-" + std::to_string(j));
      }
    }
  }
  return batches;
}

std::unordered_set<std::string> BuildExclude(const BenchOpts& o) {
  std::unordered_set<std::string> ex;
  for (size_t n = 0; n < o.exclude && n < o.nodes; ++n) ex.insert(NodeName(n));
  return ex;
}

void RunScenario(const BenchOpts& base, size_t batch_override) {
  BenchOpts o = base;
  o.batch = batch_override > 0 ? batch_override : o.batch;

  GlobalBlockIndex idx;
  Populate(&idx, o);
  const auto exclude = BuildExclude(o);
  const auto lease = std::chrono::milliseconds(o.lease_ms);

  // Per-thread inputs (distinct seeds so threads don't all hammer one bucket).
  std::vector<std::vector<std::vector<std::string>>> thread_batches(o.threads);
  for (size_t t = 0; t < o.threads; ++t) {
    thread_batches[t] = BuildBatches(o, 0x9E3779B97F4A7C15ULL ^ (t + 1));
  }

  // Warm-up (not measured): one pass per thread's first batch, single-threaded.
  for (size_t t = 0; t < o.threads; ++t) {
    (void)idx.BatchLookupForRouteGet(thread_batches[t][0], exclude, lease);
  }

  std::atomic<uint64_t> total_locs{0};
  std::atomic<double> max_wall_ms{0.0};

  auto worker = [&](size_t t) {
    const auto& batches = thread_batches[t];
    uint64_t locs = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < o.iters; ++it) {
      auto out = idx.BatchLookupForRouteGet(batches[it], exclude, lease);
      for (const auto& v : out) locs += v.size();
    }
    auto t1 = std::chrono::steady_clock::now();
    total_locs.fetch_add(locs, std::memory_order_relaxed);
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Wall clock is the slowest thread (calls run concurrently under shared_lock).
    double prev = max_wall_ms.load(std::memory_order_relaxed);
    while (ms > prev && !max_wall_ms.compare_exchange_weak(prev, ms)) {
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(o.threads);
  auto wall0 = std::chrono::steady_clock::now();
  for (size_t t = 0; t < o.threads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();
  auto wall1 = std::chrono::steady_clock::now();

  const double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  const uint64_t total_calls = static_cast<uint64_t>(o.iters) * o.threads;
  const uint64_t total_keys = total_calls * o.batch;
  const double us_per_call = total_calls > 0 ? (wall_ms * 1000.0) / total_calls : 0.0;
  const double mlookups_s = wall_ms > 0 ? (total_keys / 1e6) / (wall_ms / 1000.0) : 0.0;
  const double locs_per_call =
      total_calls > 0 ? static_cast<double>(total_locs.load()) / total_calls : 0.0;

  std::printf("%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.4f,%.2f,%.2f\n", o.index_keys, o.batch, o.nodes,
              o.locs_per_key, o.exclude, o.hit_pct, o.threads, o.iters, us_per_call, mlookups_s,
              locs_per_call);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  BenchOpts opts;
  if (!ParseArgs(argc, argv, &opts)) return 2;

  std::printf(
      "index_keys,batch,nodes,locs_per_key,exclude,hit_pct,threads,iters,"
      "wall_us_per_call,Mlookups_s,locs_per_call\n");
  std::fflush(stdout);

  if (!opts.sweep.empty()) {
    for (size_t b : opts.sweep) RunScenario(opts, b);
  } else {
    RunScenario(opts, 0);
  }
  return 0;
}
