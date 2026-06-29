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

// RouteGet strategy micro-bench: per-key Select() loop (the old router hot path)
// vs the batched BatchSelect() the router now calls once per batch.  Pure CPU,
// no master/RDMA — isolates the routing-strategy cost the master pays under
// BatchRouteGet.
//
// The overhead the batch path removes is one virtual dispatch per batch instead
// of one per key.  Logging is suppressed by the default module level (ERROR);
// set MORI_UMBP_LOG_LEVEL=error explicitly if your environment overrides it.
//
// (The RoutePut path is intentionally not benched here: the put strategy exposes
// only the batched SelectBatch — there is no per-key variant left to compare
// against.  See bench_route_put_strategy.cpp for put-side benchmarks.)
//
// Usage:
//   bench_umbp_route_strategy [--keys N] [--iters N] [--replicas R]
//                             [--get-strategy random|tier-priority]
//                             [--sweep keys=1,4,16,64,256,1024]
//
// CSV (stdout):
//   op,keys,iters,total_ms,ns_per_key
// where op is Get / BatchGet (per-key Select() loop vs BatchSelect()).
// total_ms is the duration of ONE iteration (a single batch call over all
// `keys`), averaged across `iters` repetitions; ns_per_key is that divided by
// `keys`.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "umbp/distributed/routing/route_get_strategy.h"
#include "umbp/distributed/types.h"

using mori::umbp::Location;
using mori::umbp::RandomRouteGetStrategy;
using mori::umbp::RouteGetStrategy;
using mori::umbp::TierPriorityRouteGetStrategy;
using mori::umbp::TierType;

namespace {

struct BenchOpts {
  size_t keys = 256;
  size_t iters = 2000;
  size_t replicas = 3;                  // candidate replicas per key on the Get path
  std::string get_strategy = "random";  // random | tier-priority
  std::vector<size_t> sweep;
};

void Usage() {
  std::cerr << "Usage: bench_umbp_route_strategy [--keys N] [--iters N] [--replicas R]\n"
            << "                                 [--get-strategy random|tier-priority]\n"
            << "                                 [--sweep keys=1,4,16,64,256,1024]\n";
}

// Resolve the --get-strategy name to a concrete RouteGetStrategy.
std::unique_ptr<RouteGetStrategy> MakeGetStrategy(const std::string& name) {
  if (name == "random") return std::make_unique<RandomRouteGetStrategy>();
  if (name == "tier-priority") return std::make_unique<TierPriorityRouteGetStrategy>();
  std::cerr << "Unknown --get-strategy: " << name << " (expected random|tier-priority)\n";
  std::exit(2);
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
    if (a == "--keys") {
      o->keys = std::strtoull(next("--keys"), nullptr, 10);
    } else if (a == "--iters") {
      o->iters = std::strtoull(next("--iters"), nullptr, 10);
    } else if (a == "--replicas") {
      o->replicas = std::strtoull(next("--replicas"), nullptr, 10);
    } else if (a == "--get-strategy") {
      o->get_strategy = next("--get-strategy");
    } else if (a == "--sweep") {
      std::string s = next("--sweep");
      auto eq = s.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--sweep expects 'keys=N1,N2,...'\n";
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
  return true;
}

// --- Get-path workload -----------------------------------------------------
// One candidate list per key; replicas spread across tiers/nodes so the
// TierPriority strategy does real work (rank scan + within-tier random pick).
std::vector<std::vector<Location>> BuildGetWorkload(const BenchOpts& o) {
  static const TierType kTiers[] = {TierType::HBM, TierType::DRAM, TierType::SSD};
  std::vector<std::vector<Location>> per_key(o.keys);
  for (size_t k = 0; k < o.keys; ++k) {
    auto& locs = per_key[k];
    locs.reserve(o.replicas);
    for (size_t r = 0; r < o.replicas; ++r) {
      Location loc;
      loc.node_id = "node-" + std::to_string((k + r) % 16);
      loc.size = 4096;
      loc.tier = kTiers[(k + r) % 3];
      locs.push_back(loc);
    }
  }
  return per_key;
}

// Returns total wall time (ns) over all `iters` measured calls (warm-up excluded).
template <typename Fn>
double TotalNs(size_t iters, Fn&& fn) {
  fn();  // warm-up (not measured)
  auto t0 = std::chrono::steady_clock::now();
  for (size_t it = 0; it < iters; ++it) fn();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

void Emit(const char* op, size_t keys, size_t iters, double total_ns) {
  // Per-iteration duration: one batch call over `keys`, averaged across iters.
  const double ns_per_batch = iters ? total_ns / static_cast<double>(iters) : 0.0;
  const double ns_per_key = keys ? ns_per_batch / static_cast<double>(keys) : 0.0;
  std::printf("%s,%zu,%zu,%.5f,%.2f\n", op, keys, iters, ns_per_batch / 1e6, ns_per_key);
  std::fflush(stdout);
}

void RunGet(const BenchOpts& base, size_t keys) {
  BenchOpts o = base;
  o.keys = keys;
  auto per_key = BuildGetWorkload(o);
  auto strat_ptr = MakeGetStrategy(o.get_strategy);
  RouteGetStrategy& strat = *strat_ptr;
  const std::string node_id = "requester";

  // Sink to keep the optimizer from eliding the work.
  volatile uint64_t sink = 0;

  double get_ns = TotalNs(o.iters, [&] {
    for (size_t k = 0; k < per_key.size(); ++k) {
      if (per_key[k].empty()) continue;
      Location sel = strat.Select(per_key[k], node_id);
      sink += sel.size + sel.node_id.size();
    }
  });

  double batch_get_ns = TotalNs(o.iters, [&] {
    auto out = strat.BatchSelect(per_key, node_id);
    for (const auto& sel : out) sink += sel.size + sel.node_id.size();
  });

  (void)sink;
  Emit("Get", o.keys, o.iters, get_ns);
  Emit("BatchGet", o.keys, o.iters, batch_get_ns);
}

}  // namespace

int main(int argc, char** argv) {
  BenchOpts opts;
  if (!ParseArgs(argc, argv, &opts)) return 2;

  std::fprintf(stderr, "[config] replicas/key=%zu iters=%zu get-strategy=%s\n", opts.replicas,
               opts.iters, opts.get_strategy.c_str());
  std::printf("op,keys,iters,total_ms,ns_per_key\n");
  std::fflush(stdout);

  if (!opts.sweep.empty()) {
    for (size_t k : opts.sweep) RunGet(opts, k);
  } else {
    RunGet(opts, opts.keys);
  }
  return 0;
}
