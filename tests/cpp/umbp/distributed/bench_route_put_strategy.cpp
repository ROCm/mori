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

// RoutePut strategy bench: compares batch put / put-then-get performance and
// placement distribution across the orthogonal RoutePut knobs
// (UMBP_ROUTE_PUT_SELECT_ALGO x UMBP_ROUTE_PUT_NODE_AFFINITY) under two capacity
// regimes.  Standalone executable; not a CTest target.
//
// Design notes / known limits (see PR description for full rationale):
//   * Strategy switch is per-combo: setenv() the two env vars then build a fresh
//     in-process MasterServer.  MasterServerConfig::FromEnvironment() reads them
//     live (no static cache) and only these two env vars drive the strategy.
//   * FromEnvironment() does NOT set listen_address (defaults to :50051) — we
//     override to "0.0.0.0:0" and use GetBoundPort() like the other benches.
//   * The master index/capacity only converge on a heartbeat.  We set a 1s
//     heartbeat_ttl and, after each BatchPut, poll BatchExists (read-only, no
//     lease) until every put key is visible before probing placement / timing
//     the get.  This wait is OUTSIDE the timed regions.
//   * Keys are never reclaimed and a remote BatchGet leaves a ~500ms read lease
//     that defers capacity free on Clear().  So every measured iter starts by
//     Clear()-ing all nodes and polling each allocator's TierCapacitiesSnapshot
//     until available==total (lease-deferred frees released by the reaper).
//   * Placement is observed via reader->Master().BatchRouteGet (returns per-key
//     node_id).  This bumps lease/access; the subsequent timed BatchGet RouteGets
//     the same keys again.  Harmless here (eviction is capacity-driven), but it
//     is not a zero-side-effect probe.
//   * BatchPut/BatchGet return only vector<bool>; NO_SPACE is not separable from
//     other failures at that boundary — we report success/fail counts only.
//   * DRAM-only: PoolClientConfig exposes no HBM buffer, so the HBM->DRAM tier
//     order is unit-tested elsewhere; cross-node placement contrast is fully
//     visible with DRAM + multiple nodes.
//   * random goes through the production thread_local RNG (FromEnvironment uses
//     the unseeded ctor), so its node choice is not reproducible run-to-run.
//     That is fine: per-node cold-start is one-time (see RunCombo), so at most
//     `peers` timed iters are cold and the median over >2*peers iters is always
//     a warm sample.  RunCombo enforces that iter floor.
//
// CSV (stdout):
//   select_algo,node_affinity,regime,peers,requester,reader,batch,page_bytes,
//   per_item_pages,iters,put_wall_ms,put_gibps,put_success,put_fail,
//   unique_put_nodes,max_node_share,get_wall_ms,get_gibps,get_success,
//   get_fanout_nodes,local_hit_frac
// (put/get_wall_ms are the MEDIAN per-iter latency and *_gibps is the median-
//  batch throughput over it; success/fail counts are summed over measured iters;
//  distribution metrics are the mean of per-iter values over the routed keys.
//  The measured-iter count is floored at 2*peers+1 (RunCombo) so the median is
//  always a warm sample even for strategies whose target node rotates between
//  iters (e.g. random): per-node cold-start is one-time, so at most `peers`
//  iters are cold and they sit in the median's upper tail.  The `iters` CSV
//  column reports the actual measured count after that floor is applied.)
//
// --dump-placement adds a per-node long table to stderr:
//   select_algo,node_affinity,regime,batch,node,items,bytes

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/pool_client.h"

using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::PoolClient;
using mori::umbp::PoolClientConfig;
using mori::umbp::RouteGetResult;
using mori::umbp::TierCapacity;
using mori::umbp::TierType;

namespace {

// Unique peer-service port per PoolClient so each node registers a peer_address
// and can serve/accept remote AllocateSlot/ResolveKey RPCs.  Monotonic, never
// reused — fine for the realistic number of clusters one bench run builds.
inline uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{56000};
  return next.fetch_add(1);
}

struct BenchOpts {
  std::vector<std::string> algos = {"most_available", "random"};
  std::vector<std::string> affinities = {"none", "same", "local"};
  std::vector<std::string> regimes = {"roomy", "tight"};
  size_t peers = 4;
  size_t requester = 0;
  size_t reader = SIZE_MAX;  // SIZE_MAX => default to requester
  size_t batch = 64;
  size_t page_bytes = 4096;
  size_t per_item_pages = 1;
  size_t iters = 8;
  bool no_get = false;
  bool dump_placement = false;
  std::vector<size_t> sweep;  // empty = single batch
};

void Usage() {
  std::cerr << "Usage: bench_route_put_strategy\n"
            << "  [--algos most_available,random] [--affinities none,same,local]\n"
            << "  [--regimes roomy,tight] [--peers N] [--requester R] [--reader G]\n"
            << "  [--batch N | --sweep batch=1,4,16,64,256]\n"
            << "  [--page-bytes N] [--per-item-pages N] [--iters N]\n"
            << "  [--no-get] [--dump-placement]\n";
}

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(tok);
  }
  return out;
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
    if (a == "--algos") {
      o->algos = SplitCsv(next("--algos"));
    } else if (a == "--affinities") {
      o->affinities = SplitCsv(next("--affinities"));
    } else if (a == "--regimes") {
      o->regimes = SplitCsv(next("--regimes"));
    } else if (a == "--peers") {
      o->peers = std::strtoull(next("--peers"), nullptr, 10);
    } else if (a == "--requester") {
      o->requester = std::strtoull(next("--requester"), nullptr, 10);
    } else if (a == "--reader") {
      o->reader = std::strtoull(next("--reader"), nullptr, 10);
    } else if (a == "--batch") {
      o->batch = std::strtoull(next("--batch"), nullptr, 10);
    } else if (a == "--page-bytes") {
      o->page_bytes = std::strtoull(next("--page-bytes"), nullptr, 10);
    } else if (a == "--per-item-pages") {
      o->per_item_pages = std::strtoull(next("--per-item-pages"), nullptr, 10);
    } else if (a == "--iters") {
      o->iters = std::strtoull(next("--iters"), nullptr, 10);
    } else if (a == "--no-get") {
      o->no_get = true;
    } else if (a == "--dump-placement") {
      o->dump_placement = true;
    } else if (a == "--sweep") {
      std::string s = next("--sweep");
      auto eq = s.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--sweep expects 'batch=N1,N2,...'\n";
        return false;
      }
      for (const auto& tok : SplitCsv(s.substr(eq + 1))) {
        o->sweep.push_back(std::strtoull(tok.c_str(), nullptr, 10));
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
  if (o->reader == SIZE_MAX) o->reader = o->requester;
  if (o->peers < 1) {
    std::cerr << "--peers must be >= 1\n";
    return false;
  }
  if (o->requester >= o->peers || o->reader >= o->peers) {
    std::cerr << "--requester/--reader must be < --peers\n";
    return false;
  }
  return true;
}

uint64_t AlignUp(uint64_t v, uint64_t a) { return ((v + a - 1) / a) * a; }

// Per-node storage capacity for a regime.  roomy: each node holds the whole
// batch with headroom (concentrating strategies fit on one node).  tight: total
// cluster capacity ~1.2x the batch but each node < the batch, forcing spill and
// some NO_SPACE.  Always page-aligned and at least one item.
uint64_t NodeStorageBytes(const std::string& regime, size_t peers, uint64_t batch_bytes,
                          uint64_t item_bytes, uint64_t page_bytes) {
  uint64_t bytes;
  if (regime == "tight") {
    bytes = AlignUp((batch_bytes * 12 / 10 + peers - 1) / peers, page_bytes);
  } else {  // roomy
    const uint64_t k64m = static_cast<uint64_t>(64) << 20;
    bytes = std::max<uint64_t>(k64m, batch_bytes * 4);
    bytes = AlignUp(bytes, page_bytes);
  }
  return std::max<uint64_t>(bytes, item_bytes);
}

// One symmetric node: exportable storage DRAM (receives writes) plus a private
// io buffer used as the put src (when requester) and the get dst (when reader).
struct Node {
  std::vector<char> storage;  // master-managed exportable DRAM
  std::vector<char> io;       // registered src/dst region
  std::unique_ptr<PoolClient> client;
};

class Cluster {
 public:
  Cluster(const std::string& algo, const std::string& affinity, size_t peers, uint64_t page_bytes,
          uint64_t node_storage_bytes, uint64_t io_bytes) {
    setenv("UMBP_ROUTE_PUT_SELECT_ALGO", algo.c_str(), /*overwrite=*/1);
    setenv("UMBP_ROUTE_PUT_NODE_AFFINITY", affinity.c_str(), /*overwrite=*/1);

    MasterServerConfig mcfg = MasterServerConfig::FromEnvironment();
    mcfg.listen_address = "0.0.0.0:0";  // ephemeral; FromEnvironment leaves :50051
    mcfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};  // fast index convergence
    master_ = std::make_unique<MasterServer>(std::move(mcfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 500 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (master_->GetBoundPort() == 0) {
      std::cerr << "master failed to start\n";
      std::exit(2);
    }
    const std::string master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    nodes_.resize(peers);
    for (size_t k = 0; k < peers; ++k) {
      nodes_[k].storage.assign(node_storage_bytes, 0);
      nodes_[k].io.assign(io_bytes, 0);
      PoolClientConfig cc;
      cc.master_config.node_id = "node-" + std::to_string(k);
      cc.master_config.node_address = "127.0.0.1";
      cc.master_config.master_address = master_addr;
      cc.io_engine.host = "0.0.0.0";
      cc.io_engine.port = 0;
      cc.peer_service_port = NextPeerServicePort();
      cc.dram_page_size = page_bytes;
      cc.dram_buffers = {{nodes_[k].storage.data(), nodes_[k].storage.size()}};
      cc.tier_capacities = {{TierType::DRAM, {nodes_[k].storage.size(), nodes_[k].storage.size()}}};
      nodes_[k].client = std::make_unique<PoolClient>(std::move(cc));
      if (!nodes_[k].client->Init()) {
        std::cerr << "node " << k << " init failed\n";
        std::exit(2);
      }
      // Register the io region so put src / get dst go zero-copy (no staging).
      nodes_[k].client->RegisterMemory(nodes_[k].io.data(), nodes_[k].io.size());
    }
  }

  ~Cluster() {
    for (auto& n : nodes_) {
      if (n.client) n.client->Shutdown();
    }
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
  }

  PoolClient* client(size_t k) { return nodes_[k].client.get(); }
  std::vector<char>& io(size_t k) { return nodes_[k].io; }
  size_t num_nodes() const { return nodes_.size(); }

  // Clear every node, then wait until each allocator's DRAM is fully reclaimed
  // (lease-deferred frees released by the reaper).  Returns false on timeout.
  bool ResetAll() {
    for (auto& n : nodes_) n.client->Clear();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (auto& n : nodes_) {
      while (true) {
        auto caps = n.client->DramAllocator()->TierCapacitiesSnapshot();
        auto it = caps.find(TierType::DRAM);
        const bool full = it != caps.end() && it->second.available_bytes == it->second.total_bytes;
        if (full) break;
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    return true;
  }

 private:
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::vector<Node> nodes_;
};

// Poll until every key is visible in the master index (read-only, no lease).
bool WaitVisible(PoolClient* reader, const std::vector<std::string>& keys) {
  if (keys.empty()) return true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    auto present = reader->BatchExists(keys);
    bool all = true;
    for (bool p : present) all = all && p;
    if (all) return true;
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// Median is robust to the one-time cold-start cost (first AllocateSlot / peer
// connection / RDMA registration to a node) that the mean would fold into every
// comparison.  Per-node cold-start happens at most once (connections survive
// Clear()), so across a combo at most `peers` timed iters are cold; with the
// measured-iter count floored at 2*peers+1 (RunCombo) those cold iters are a
// strict minority in the upper tail and the median is always a warm sample —
// even when the strategy's target node rotates between iters (e.g. random).
double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct ComboResult {
  std::vector<double> put_ms, get_ms;  // per-iter wall, for median
  uint64_t put_bytes = 0, get_bytes = 0;
  size_t put_success = 0, put_fail = 0, get_success = 0;
  // Distribution metrics are computed PER measured iter and summed here, then
  // averaged by iters at print time.  Per-iter (not cross-iter-union) keeps
  // intra-batch concentration honest: `same` concentrates each batch on one
  // node even if different iters pick different nodes.
  double sum_unique_nodes = 0, sum_max_share = 0, sum_fanout = 0, sum_local_hit = 0;
  std::map<std::string, uint64_t> node_items;  // routed items per node (aggregated, dump only)
  std::map<std::string, uint64_t> node_bytes;
};

void RunCombo(const BenchOpts& o, const std::string& algo, const std::string& affinity,
              const std::string& regime, size_t batch) {
  const uint64_t item_bytes = static_cast<uint64_t>(o.per_item_pages) * o.page_bytes;
  const uint64_t batch_bytes = static_cast<uint64_t>(batch) * item_bytes;
  const uint64_t node_storage =
      NodeStorageBytes(regime, o.peers, batch_bytes, item_bytes, o.page_bytes);

  Cluster cluster(algo, affinity, o.peers, o.page_bytes, node_storage, batch_bytes);
  PoolClient* requester = cluster.client(o.requester);
  PoolClient* reader = cluster.client(o.reader);
  const std::string reader_node = "node-" + std::to_string(o.reader);

  // Put src / get dst slices into each role's private io buffer.
  std::vector<const void*> srcs(batch);
  std::vector<size_t> sizes(batch);
  for (size_t i = 0; i < batch; ++i) {
    char* slot = cluster.io(o.requester).data() + i * item_bytes;
    std::memset(slot, static_cast<int>(0x10 + (i & 0x7F)), item_bytes);
    srcs[i] = slot;
    sizes[i] = item_bytes;
  }

  // Per-peer cold-start (first peer-service connection + RDMA QP setup to a
  // node) is paid once per node and then stays warm for the rest of the combo
  // (connections survive Clear()).  So the number of cold timed iters is at most
  // `peers` — each node contributes at most one cold first-touch, no matter how
  // the strategy picks (e.g. `random` rotates its single-node anchor across
  // iters).  Reporting the MEDIAN (see Median) over iters > 2*peers therefore
  // guarantees the reported value is a warm sample: the <=peers cold iters are a
  // strict minority and sit in the upper tail.  We bump the measured-iter count
  // up to that floor so the result is steady-state regardless of `peers`.
  const size_t measured_iters = std::max<size_t>(o.iters, 2 * o.peers + 1);
  if (measured_iters != o.iters) {
    std::fprintf(stderr,
                 "[bench] iters bumped %zu -> %zu (>2*peers) so the median excludes per-node "
                 "cold-start (algo=%s aff=%s regime=%s batch=%zu)\n",
                 o.iters, measured_iters, algo.c_str(), affinity.c_str(), regime.c_str(), batch);
  }

  ComboResult r;
  // Iter 0 is an unmeasured warm-up (channels / RDMA registration).
  for (size_t it = 0; it <= measured_iters; ++it) {
    if (!cluster.ResetAll()) {
      std::cerr << "capacity did not recover after Clear (algo=" << algo << " aff=" << affinity
                << " regime=" << regime << " batch=" << batch << ")\n";
      std::exit(2);
    }

    std::vector<std::string> keys(batch);
    for (size_t i = 0; i < batch; ++i) {
      keys[i] = "rp-" + std::to_string(it) + "-" + std::to_string(i);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto put_res = requester->BatchPut(keys, srcs, sizes);
    auto t1 = std::chrono::steady_clock::now();

    std::vector<std::string> ok_keys;
    std::vector<void*> dsts;
    std::vector<size_t> ok_sizes;
    uint64_t put_bytes = 0;
    size_t put_success = 0;
    ok_keys.reserve(batch);
    for (size_t i = 0; i < batch; ++i) {
      if (put_res[i]) {
        void* slot = cluster.io(o.reader).data() + ok_keys.size() * item_bytes;
        ok_keys.push_back(keys[i]);
        dsts.push_back(slot);
        ok_sizes.push_back(item_bytes);
        put_bytes += item_bytes;
        ++put_success;
      }
    }

    if (!WaitVisible(reader, ok_keys)) {
      std::cerr << "put keys not visible within timeout (algo=" << algo << " aff=" << affinity
                << " regime=" << regime << " batch=" << batch << ")\n";
      std::exit(2);
    }

    // Placement probe: realized node per key (also the get fanout).
    std::vector<std::optional<RouteGetResult>> routes;
    reader->Master().BatchRouteGet(ok_keys, {}, &routes);

    // Timed get of the same keys.
    double get_ms = 0;
    uint64_t get_bytes = 0;
    size_t get_success = 0;
    if (!o.no_get && !ok_keys.empty()) {
      auto g0 = std::chrono::steady_clock::now();
      auto get_res = reader->BatchGet(ok_keys, dsts, ok_sizes);
      auto g1 = std::chrono::steady_clock::now();
      get_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
      for (size_t i = 0; i < get_res.size(); ++i) {
        if (get_res[i]) {
          get_bytes += ok_sizes[i];
          ++get_success;
        }
      }
    }

    if (it == 0) continue;  // warm-up, not accumulated

    r.put_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    r.put_bytes += put_bytes;
    r.put_success += put_success;
    r.put_fail += batch - put_success;
    r.get_ms.push_back(get_ms);
    r.get_bytes += get_bytes;
    r.get_success += get_success;

    // Per-iter placement metrics over this iter's routed keys.
    std::map<std::string, uint64_t> iter_items;
    size_t routed = 0;
    for (const auto& route : routes) {
      if (!route) continue;
      ++routed;
      iter_items[route->node_id] += 1;
      r.node_items[route->node_id] += 1;
      r.node_bytes[route->node_id] += item_bytes;
    }
    uint64_t max_items = 0, local_hits = 0;
    std::set<std::string> fanout_nodes;
    for (const auto& [node, items] : iter_items) {
      max_items = std::max<uint64_t>(max_items, items);
      if (node == reader_node) {
        local_hits += items;
      } else {
        fanout_nodes.insert(node);
      }
    }
    if (routed > 0) {
      r.sum_unique_nodes += static_cast<double>(iter_items.size());
      r.sum_max_share += static_cast<double>(max_items) / routed;
      r.sum_fanout += static_cast<double>(fanout_nodes.size());
      r.sum_local_hit += static_cast<double>(local_hits) / routed;
    }
  }

  const double inv_iters = measured_iters > 0 ? 1.0 / measured_iters : 0.0;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  // wall_ms is the median per-iter latency; throughput is the median-batch bytes
  // over that median latency (outlier-robust, see Median()).
  const double put_med_ms = Median(r.put_ms);
  const double get_med_ms = Median(r.get_ms);
  const double avg_put_bytes = r.put_bytes * inv_iters;
  const double avg_get_bytes = r.get_bytes * inv_iters;
  const double put_gibps = put_med_ms > 0 ? (avg_put_bytes / kGiB) / (put_med_ms / 1000.0) : 0.0;
  const double get_gibps = get_med_ms > 0 ? (avg_get_bytes / kGiB) / (get_med_ms / 1000.0) : 0.0;

  std::printf(
      "%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,%zu,%zu,%.2f,%.3f,%.3f,%.3f,%zu,%.2f,%.3f\n",
      algo.c_str(), affinity.c_str(), regime.c_str(), o.peers, o.requester, o.reader, batch,
      o.page_bytes, o.per_item_pages, measured_iters, put_med_ms, put_gibps, r.put_success,
      r.put_fail, r.sum_unique_nodes * inv_iters, r.sum_max_share * inv_iters, get_med_ms,
      get_gibps, r.get_success, r.sum_fanout * inv_iters, r.sum_local_hit * inv_iters);
  std::fflush(stdout);

  if (o.dump_placement) {
    for (const auto& [node, items] : r.node_items) {
      std::fprintf(stderr, "%s,%s,%s,%zu,%s,%zu,%zu\n", algo.c_str(), affinity.c_str(),
                   regime.c_str(), batch, node.c_str(), items, r.node_bytes[node]);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  BenchOpts opts;
  if (!ParseArgs(argc, argv, &opts)) return 2;

  std::printf(
      "select_algo,node_affinity,regime,peers,requester,reader,batch,page_bytes,per_item_pages,"
      "iters,put_wall_ms,put_gibps,put_success,put_fail,unique_put_nodes,max_node_share,"
      "get_wall_ms,get_gibps,get_success,get_fanout_nodes,local_hit_frac\n");
  std::fflush(stdout);
  if (opts.dump_placement) {
    std::fprintf(stderr, "select_algo,node_affinity,regime,batch,node,items,bytes\n");
  }

  std::vector<size_t> batches = opts.sweep.empty() ? std::vector<size_t>{opts.batch} : opts.sweep;
  for (const auto& regime : opts.regimes) {
    for (const auto& algo : opts.algos) {
      for (const auto& affinity : opts.affinities) {
        for (size_t b : batches) {
          if (b == 0) continue;
          RunCombo(opts, algo, affinity, regime, b);
        }
      }
    }
  }
  return 0;
}
