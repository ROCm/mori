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

// Layer-wise KV restore bench for PoolClient's ranged I/O.
//
// WHAT IT MODELS
//
// A KV connector doing layer-wise loading against UMBP topology 3 (one node's
// standalone server whose inner backend is distributed, so one PoolClient is
// shared by every rank on that node).  The shapes below are taken from
// sglang's umbp_tree_connector.py, not invented:
//
//   * A page object holds ALL layers of one page; layer L occupies the byte
//     range [L*layer_bytes, (L+1)*layer_bytes) inside it.
//   * Loading walks the layer stack in groups of UMBP_LAYER_GROUP (default 8).
//     One BatchGetRanges call carries a CHUNK OF PAGES, and each page carries
//     one range per layer in the group -- not one key per call, and not one
//     range per call.  Chunk size is the RANGES_PER_CALL budget divided by the
//     ranges each object actually carries.
//   * Offload does NOT walk groups: one BatchPutRanges call writes every layer
//     of a page at once, because a put's ranges must tile the object exactly.
//   * Per rank there is exactly ONE load thread and ONE offload thread, so the
//     concurrency reaching PoolClient is bounded by the rank count, not by an
//     arbitrary thread count.
//   * Forward runs concurrently with loading and waits per layer group: it can
//     compute layer L while layer L+1 is still in flight.  That overlap is the
//     entire point of layer-wise, so this bench runs a real forward thread that
//     blocks on each group and records how long it stalled.
//
// THE LOCALITY THAT MATTERS
//
// Ideally only the FIRST group touching a page pays a remote fetch, and groups
// 2..N are served by the local medium with no arena and no lock.  How that
// locality gets built is exactly what this bench measures, and it is not free:
// a remote ranged get moves only the bytes asked for, so making the page local
// is a separate asynchronous pull that a later group races against.  Whether
// it wins that race is visible as local_hit_pct and repeat_remote.
//
// A bench that forces every call remote would overstate how often the arena is
// reached, so this one lets the real behavior happen and reports the local-hit
// rate it observes (how that number is obtained, and what it cannot tell you,
// is on AllLocallyResident below).
//
// WHERE "REMOTE" COMES FROM ON A SINGLE HOST
//
// UMBP decides local vs remote by comparing node ids, not machines: a key is
// remote when its route's node_id differs from this client's own
// (PartitionBatchGetTargets).  This bench therefore builds TWO PoolClients in
// one process -- "node-local" and "node-peer" -- each with its own node id,
// peer-service port and io engine, and seeds every page onto the peer only.
// Reading them from the node is then genuinely routed as remote and really
// does go through BatchRouteGet, the MoriIoEngine RDMA path, the scratch arena
// and its mutex.
//
// So the CODE PATH is the real one, but both endpoints sit on one host and the
// bytes cross a local NIC.  Treat the structural results (local-hit rate,
// whether the loader stays ahead of forward, how they scale with ranks) as
// meaningful, and the absolute latency/bandwidth as optimistic: a real
// two-machine deployment pays a longer wire.
//
// BUILDING.  Like the other bench_* targets here it builds with the test tree
// but is not registered with ctest: it takes minutes and its output is numbers
// to read, not an assertion to pass.
//
//   cmake -S . -B build_ranged_bench -GNinja -DBUILD_UMBP=ON -DBUILD_TESTS=ON \
//         -DCMAKE_BUILD_TYPE=Release
//   cmake --build build_ranged_bench --target bench_umbp_pool_client_ranges -j
//
// On a NIC whose RC queues are smaller than mori-io's defaults the peer
// handshake fails with `ibv_create_qp ... Invalid argument`; cap the queues:
//   MORI_IO_QP_MAX_SEND_WR=256 MORI_IO_QP_MAX_RECV_WR=256 MORI_IO_QP_MAX_CQE=1024
//
// Usage:
//   bench_pool_client_ranges [--ranks N] [--layers N] [--layer-group N]
//                            [--layer-bytes N] [--pages N] [--requests N]
//                            [--compute-us-per-layer N] [--offload-ranks N]
//                            [--ranges-per-call N] [--scratch-mib N]
//                            [--page-bytes N] [--no-prefetch]
//                            [--interleave-groups] [--verify]
//
// CSV (stdout): one row per rank count in --ranks
//   ranks,layers,group,pages,object_kib,restore_ms_p50,restore_ms_p95,
//     ttfl_us_p50,ttfl_us_p95,stall_ms_p50,get_gibps,put_gibps,local_hit_pct,
//     remote_calls,remote_keys,repeat_remote,wire_whole_mib,wire_range_mib,
//     local_write_mib
//
// The two wire columns are what the fetch WOULD move whole-object versus what
// a range-granular fetch actually moves, so their ratio is the read
// amplification this bench exists to measure.  repeat_remote counts keys a
// single request had to fetch remotely more than once, which is what falls
// when locality is rebuilt between layer groups.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool_client.h"

using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::PoolClient;
using mori::umbp::PoolClientConfig;

namespace {

inline uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(20000 + (static_cast<unsigned>(::getpid()) * 16u) % 40000u)};
  return next.fetch_add(1);
}

// One KV pool: the layers it holds and how many bytes each occupies in a page.
//
// A model does not have one uniform pool.  DeepSeek-V4 splits its layers across
// six, and their per-layer sizes span more than an order of magnitude
// (37,440 B for the c4 pool, 8,448 for its indexer, 1,728 for c128).  That
// spread is the point: achieved bandwidth on this path is dominated by fragment
// size, measured at 5.4 GB/s at 1,728 B against 37.5 GB/s at 37,440 B, so a
// single --layer-bytes picks one point on a 7x curve and any change that helps
// small fragments specifically is mis-measured by it.
//
// `layers` are logical layer numbers, because pools interleave: DeepSeek-V4's
// c4 and c128 alternate down the stack, so which pools a layer group touches
// depends on their actual positions and not just their counts.
struct Pool {
  std::string name;
  std::vector<size_t> layers;
  size_t layer_bytes = 0;

  size_t object_bytes() const { return layers.size() * layer_bytes; }
  // Position of a logical layer inside the object, or npos.  The object packs
  // the pool's own layers back to back, so this is an index into `layers`, not
  // the logical number -- the twin of DevicePoolEntry.layer_mapping.
  size_t IndexOf(size_t logical_layer) const {
    for (size_t i = 0; i < layers.size(); ++i) {
      if (layers[i] == logical_layer) return i;
    }
    return SIZE_MAX;
  }
};

struct BenchOpts {
  // Concurrency: one load thread + one offload thread per rank, all sharing one
  // PoolClient (topology 3's node-level server).
  std::vector<size_t> rank_sweep;
  size_t offload_ranks = SIZE_MAX;  // default: same as ranks; 0 disables offload

  // Model shape.  Defaults are DeepSeek-V3.2 / TP8 / page_size=64 / bf16, KV
  // pool: 61 layers x 73,728 B per layer per page => a 4.29 MiB page object,
  // read back 8 layers (576 KiB) at a time.  The INDEXER pool of the same model
  // is --layers 61 --layer-bytes 8448.
  size_t layers = 61;
  size_t layer_group = 8;             // UMBP_LAYER_GROUP
  size_t layer_bytes = 73728;         // bytes of one layer of one page
  size_t pages = 64;                  // pages (objects) restored per request
  size_t requests = 3;                // requests per rank
  size_t compute_us_per_layer = 150;  // simulated forward compute

  // Empty unless --pools was given, in which case it replaces the single
  // implied pool above.  Resolve() fills it either way so the rest of the
  // bench only ever reads this.
  std::vector<Pool> pools;

  // Client-side budgets that shape the call pattern.
  size_t ranges_per_call = 8192;  // UMBP_RANGES_PER_CALL
  size_t scratch_mib = 256;       // ranged scratch arena, each direction
  // Medium page size.  0 => one page per object, which is what distributed mode
  // actually runs (master page_size == KV block size, see pool_client.cpp).
  size_t page_bytes = 0;
  // Background whole-object pull after a remote ranged read.  Turning it off is
  // how the wire cost of the duplicate traffic is measured.
  bool locality_prefetch = true;
  // Deal layers to groups round-robin instead of in contiguous blocks, so a
  // group's ranges are strided through the object and cannot be merged into
  // one read.  Same byte count, same call count, same range count -- only
  // adjacency changes.  This is the adversarial case for any implementation
  // that leans on contiguous layer groups.
  bool interleave_groups = false;
  // Check every restored page byte-for-byte after each request.  Off by
  // default because the comparison lands inside the measured window; on, this
  // is the only thing here that proves the numbers above came from correct
  // reads and not from a fast wrong answer.
  bool verify = false;

  // Turn --layers/--layer-bytes into a one-pool list when --pools was not
  // given, so a command line that worked before this option existed still
  // describes exactly the same workload.
  void Resolve() {
    if (pools.empty()) {
      Pool single;
      single.name = "kv";
      single.layer_bytes = layer_bytes;
      single.layers.resize(layers);
      for (size_t l = 0; l < layers; ++l) single.layers[l] = l;
      pools.push_back(std::move(single));
    }
    layers = 0;
    for (const Pool& pool : pools) {
      for (size_t l : pool.layers) layers = std::max(layers, l + 1);
    }
    if (layers == 0) layers = 1;
  }

  // Bytes of one page across every pool -- the whole KV footprint of a token
  // page, which is what a restore has to move.
  size_t object_bytes() const {
    size_t total = 0;
    for (const Pool& pool : pools) total += pool.object_bytes();
    return total;
  }
  size_t group_count() const { return (layers + layer_group - 1) / layer_group; }
  // Bytes one layer group moves, summed over the pools that group touches.
  // With heterogeneous pools this varies by group, so it is the mean.
  size_t group_bytes() const {
    const size_t groups = group_count();
    return groups > 0 ? object_bytes() / groups : object_bytes();
  }
  // Largest single object, which is what the scratch arena must be able to hold.
  size_t max_object_bytes() const {
    size_t most = 0;
    for (const Pool& pool : pools) most = std::max(most, pool.object_bytes());
    return most;
  }
  size_t medium_page_bytes() const { return page_bytes != 0 ? page_bytes : max_object_bytes(); }
};

void Usage() {
  std::cerr << "Usage: bench_pool_client_ranges [--ranks 1,2,4,8]\n"
            << "        [--layers N] [--layer-group N] [--layer-bytes N]\n"
            << "        [--pools SPEC] [--pages N] [--requests N]\n"
            << "        [--compute-us-per-layer N]\n"
            << "        [--offload-ranks N] [--ranges-per-call N] [--scratch-mib N]\n"
            << "        [--page-bytes N] [--no-prefetch] [--interleave-groups] [--verify]\n"
            << "\n"
            << "  --pools replaces --layers/--layer-bytes with a real model's several\n"
            << "  pools, whose per-layer sizes differ by more than 10x and decide the\n"
            << "  achieved bandwidth.  Semicolon separated, each entry:\n"
            << "      name:layer_count:layer_bytes[:first/stride | :l1,l2,l3,...]\n"
            << "  The last field places the pool's layers in the stack (default 0,1,2..),\n"
            << "  which is what decides the layer groups a pool appears in.  Use the\n"
            << "  explicit list when the layers are not an arithmetic progression.\n"
            << "  DeepSeek-V4-Pro, from its config's compress_ratios:\n"
            << "      --pools "
               "'c4:30:37440:2/2;c4_indexer:30:8448:2/"
               "2;c128:31:1728:0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,"
               "47,49,51,53,55,57,59'\n";
}

std::vector<size_t> ParseList(const std::string& s) {
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::strtoull(tok.c_str(), nullptr, 10));
  }
  return out;
}

// "name:count:bytes[:first/stride]" entries separated by ';'.
std::vector<Pool> ParsePools(const std::string& spec) {
  std::vector<Pool> pools;
  std::stringstream entries(spec);
  std::string entry;
  while (std::getline(entries, entry, ';')) {
    if (entry.empty()) continue;
    std::vector<std::string> field;
    std::stringstream fields(entry);
    std::string tok;
    while (std::getline(fields, tok, ':')) field.push_back(tok);
    if (field.size() < 3 || field.size() > 4) {
      std::cerr << "--pools: expected name:count:bytes[:first/stride], got '" << entry << "'\n";
      std::exit(2);
    }
    Pool pool;
    pool.name = field[0];
    const size_t count = std::strtoull(field[1].c_str(), nullptr, 10);
    pool.layer_bytes = std::strtoull(field[2].c_str(), nullptr, 10);
    if (count == 0 || pool.layer_bytes == 0) {
      std::cerr << "--pools: count and bytes must be non-zero in '" << entry << "'\n";
      std::exit(2);
    }
    if (field.size() == 3) {
      pool.layers.reserve(count);
      for (size_t i = 0; i < count; ++i) pool.layers.push_back(i);
    } else if (field[3].find('/') != std::string::npos) {
      const size_t slash = field[3].find('/');
      const size_t first = std::strtoull(field[3].substr(0, slash).c_str(), nullptr, 10);
      const size_t stride = std::strtoull(field[3].substr(slash + 1).c_str(), nullptr, 10);
      if (stride == 0) {
        std::cerr << "--pools: stride must be non-zero in '" << entry << "'\n";
        std::exit(2);
      }
      pool.layers.reserve(count);
      for (size_t i = 0; i < count; ++i) pool.layers.push_back(first + i * stride);
    } else {
      // Explicit list.  Needed because real pools are not always an arithmetic
      // progression: DeepSeek-V4's c128 holds layers 0,1,3,5,...,59 -- the
      // first two adjacent, the rest every other one.
      pool.layers = ParseList(field[3]);
      if (pool.layers.size() != count) {
        std::cerr << "--pools: '" << pool.name << "' declares " << count << " layers but lists "
                  << pool.layers.size() << "\n";
        std::exit(2);
      }
    }
    pools.push_back(std::move(pool));
  }
  if (pools.empty()) {
    std::cerr << "--pools: no pools parsed from '" << spec << "'\n";
    std::exit(2);
  }
  return pools;
}

bool ParseArgs(int argc, char** argv, BenchOpts* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--ranks") {
      o->rank_sweep = ParseList(next("--ranks"));
    } else if (a == "--layers") {
      o->layers = std::strtoull(next("--layers"), nullptr, 10);
    } else if (a == "--layer-group") {
      o->layer_group = std::strtoull(next("--layer-group"), nullptr, 10);
    } else if (a == "--layer-bytes") {
      o->layer_bytes = std::strtoull(next("--layer-bytes"), nullptr, 10);
    } else if (a == "--pools") {
      o->pools = ParsePools(next("--pools"));
    } else if (a == "--pages") {
      o->pages = std::strtoull(next("--pages"), nullptr, 10);
    } else if (a == "--requests") {
      o->requests = std::strtoull(next("--requests"), nullptr, 10);
    } else if (a == "--compute-us-per-layer") {
      o->compute_us_per_layer = std::strtoull(next("--compute-us-per-layer"), nullptr, 10);
    } else if (a == "--offload-ranks") {
      o->offload_ranks = std::strtoull(next("--offload-ranks"), nullptr, 10);
    } else if (a == "--ranges-per-call") {
      o->ranges_per_call = std::strtoull(next("--ranges-per-call"), nullptr, 10);
    } else if (a == "--scratch-mib") {
      o->scratch_mib = std::strtoull(next("--scratch-mib"), nullptr, 10);
    } else if (a == "--page-bytes") {
      o->page_bytes = std::strtoull(next("--page-bytes"), nullptr, 10);
    } else if (a == "--no-prefetch") {
      o->locality_prefetch = false;
    } else if (a == "--interleave-groups") {
      o->interleave_groups = true;
    } else if (a == "--verify") {
      o->verify = true;
    } else if (a == "-h" || a == "--help") {
      Usage();
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      Usage();
      return false;
    }
  }
  if (o->rank_sweep.empty()) o->rank_sweep = {1, 2, 4, 8};
  if (o->layer_group == 0) o->layer_group = 1;
  if (o->layers == 0) o->layers = 1;
  o->Resolve();
  return true;
}

// Layers covered by group `g`, mirroring _layer_groups().  These are logical
// layer numbers spanning the whole stack; which pools they reach is a separate
// question answered by PoolGroupLayers.
//
// With --interleave-groups the same layers are dealt round-robin instead, so
// every group still covers its share of the object but its ranges are strided
// rather than contiguous.  The partition stays exact either way.
std::vector<size_t> GroupLayers(const BenchOpts& o, size_t g) {
  std::vector<size_t> out;
  if (o.interleave_groups) {
    for (size_t l = g; l < o.layers; l += o.group_count()) out.push_back(l);
    return out;
  }
  const size_t start = g * o.layer_group;
  for (size_t l = start; l < std::min(start + o.layer_group, o.layers); ++l) out.push_back(l);
  return out;
}

// Where this pool's own layers sit inside a group: object-relative indices, not
// logical numbers.  Empty when the group misses the pool entirely, which is how
// interleaved pools end up issuing calls on different groups -- the twin of
// _plans_covering() skipping a plan.
std::vector<size_t> PoolGroupLayers(const Pool& pool, const std::vector<size_t>& group) {
  std::vector<size_t> out;
  for (size_t logical : group) {
    const size_t index = pool.IndexOf(logical);
    if (index != SIZE_MAX) out.push_back(index);
  }
  return out;
}

// Objects per RPC, budgeted by the ranges they actually carry -- the C++ twin
// of _entries_per_call().
size_t EntriesPerCall(const BenchOpts& o, size_t ranges_per_object) {
  return std::max<size_t>(1, o.ranges_per_call / std::max<size_t>(1, ranges_per_object));
}

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx = std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5));
  return v[idx];
}

// Busy-wait: sleep_for's granularity is coarse next to a layer group's compute.
void SpinUs(uint64_t us) {
  if (us == 0) return;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
  while (std::chrono::steady_clock::now() < deadline) {
  }
}

// Two PoolClients in one process, distinct node ids: `node` is the one every
// rank drives (the node-level client of topology 3), `peer` exists only to own
// the seeded pages so that the node's first read of a page routes as remote.
// See "WHERE REMOTE COMES FROM ON A SINGLE HOST" at the top of this file.
class Cluster {
 public:
  Cluster(size_t node_capacity, size_t peer_capacity, size_t scratch_bytes, size_t page_bytes,
          bool locality_prefetch)
      : page_bytes_(page_bytes),
        locality_prefetch_(locality_prefetch),
        node_get_scratch_(scratch_bytes),
        node_put_scratch_(scratch_bytes),
        peer_get_scratch_(scratch_bytes),
        peer_put_scratch_(scratch_bytes) {
    MasterServerConfig mcfg;
    mcfg.listen_address = "0.0.0.0:0";
    mcfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};
    master_ = std::make_unique<MasterServer>(std::move(mcfg));
    master_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 300 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (master_->GetBoundPort() == 0) {
      std::cerr << "master failed to start\n";
      std::exit(2);
    }
    const std::string master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    node_ =
        MakeClient("node-local", master_addr, node_capacity, node_get_scratch_, node_put_scratch_);
    peer_ =
        MakeClient("node-peer", master_addr, peer_capacity, peer_get_scratch_, peer_put_scratch_);
  }

  ~Cluster() {
    if (node_) node_->Shutdown();
    if (peer_) peer_->Shutdown();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
  }

  PoolClient* node() { return node_.get(); }
  PoolClient* peer() { return peer_.get(); }

 private:
  std::unique_ptr<PoolClient> MakeClient(const std::string& node_id, const std::string& master_addr,
                                         size_t dram_capacity, std::vector<char>& get_scratch,
                                         std::vector<char>& put_scratch) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_addr;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = page_bytes_;
    cfg.staging_buffer_size = 1024;  // ranged remote must go through the arena
    cfg.dram.buffer_sizes = {dram_capacity};
    cfg.ranged_get_scratch_buffer = get_scratch.data();
    cfg.ranged_get_scratch_size = get_scratch.size();
    cfg.ranged_put_scratch_buffer = put_scratch.data();
    cfg.ranged_put_scratch_size = put_scratch.size();
    cfg.cache_remote_fetches = false;
    cfg.ranged_locality_prefetch = locality_prefetch_;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) {
      std::cerr << node_id << " init failed\n";
      std::exit(2);
    }
    if (!client->RegisterMemory(get_scratch.data(), get_scratch.size()) ||
        !client->RegisterMemory(put_scratch.data(), put_scratch.size())) {
      std::cerr << node_id << " scratch registration failed\n";
      std::exit(2);
    }
    return client;
  }

  size_t page_bytes_;
  bool locality_prefetch_;
  std::vector<char> node_get_scratch_;
  std::vector<char> node_put_scratch_;
  std::vector<char> peer_get_scratch_;
  std::vector<char> peer_put_scratch_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> node_;
  std::unique_ptr<PoolClient> peer_;
};

// How local_hit_pct is obtained, and what it is not.
//
// The bench cannot see which branch BatchGetRanges took -- PoolClient reports
// nothing about it.  What it can do is apply the SAME criterion the call itself
// will apply: phase 1 resolves each key against this node's own medium and
// serves whatever it finds without ever reaching the arena, so a key that
// resolves here is a key that will not go remote.  This runs that identical
// resolve just before the call.
//
// Three limits, so the number is not over-read:
//   * It is a prediction from the same criterion, taken a moment earlier -- not
//     an observation from inside mori.
//   * It races: under concurrency the object can be evicted between this
//     resolve and the call, which would then go remote after being counted
//     local.  Expect the reported rate to be slightly optimistic at high rank
//     counts, which is also where eviction actually starts happening.
//   * It is per CALL, not per key: a chunk counts as local only when every key
//     in it resolves, so a partially-local chunk is counted remote.
//
// Resolved for the whole chunk in one call: this sits inside the timed region,
// so it has to stay cheap.
bool AllLocallyResident(PoolClient* client, const std::vector<std::string>& keys) {
  auto* backend = client->Backends().Get(client->Medium());
  if (backend == nullptr) return false;
  auto resolved = backend->BatchResolve(keys, /*include_descs=*/false);
  return resolved.size() == keys.size() &&
         std::all_of(resolved.begin(), resolved.end(),
                     [](const auto& entry) { return entry.found; });
}

void WaitAllVisible(PoolClient* client, const std::vector<std::string>& keys,
                    std::chrono::milliseconds timeout = std::chrono::seconds{30}) {
  if (keys.empty()) return;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    auto present = client->BatchExists(keys);
    if (std::all_of(present.begin(), present.end(), [](bool b) { return b; })) return;
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "WaitAllVisible: keys not visible before timeout\n";
      std::exit(2);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// Publish whole page objects on the peer so the node must fetch them remotely.
// Byte a page's object should hold at `offset`.  A pure function of the key's
// index so the loader can recompute the expectation without keeping a copy.
inline char SeedByte(size_t key_index, size_t offset) {
  return static_cast<char>((offset * 31 + key_index * 131 + 7) & 0xff);
}

void SeedPeer(PoolClient* peer, const std::vector<std::string>& keys, size_t object_bytes,
              std::vector<char>* storage) {
  storage->assign(keys.size() * object_bytes, 0);
  std::vector<const void*> srcs(keys.size());
  std::vector<size_t> sizes(keys.size(), object_bytes);
  for (size_t i = 0; i < keys.size(); ++i) {
    char* slot = storage->data() + i * object_bytes;
    for (size_t b = 0; b < object_bytes; ++b) slot[b] = SeedByte(i, b);
    srcs[i] = slot;
  }
  peer->RegisterMemory(storage->data(), storage->size());
  auto r = peer->BatchPut(keys, srcs, sizes);
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!r[i]) {
      std::cerr << "seed put failed for " << keys[i] << "\n";
      std::exit(2);
    }
  }
}

// Per-request handoff between the loader thread and the forward thread, the
// role LayerWiseLoadCounter plays in the connector.
struct LayerGate {
  std::mutex mu;
  std::condition_variable cv;
  size_t groups_done = 0;
  bool failed = false;

  void Complete(size_t groups) {
    {
      std::lock_guard<std::mutex> lk(mu);
      groups_done = groups;
    }
    cv.notify_all();
  }
  void Fail() {
    {
      std::lock_guard<std::mutex> lk(mu);
      failed = true;
      groups_done = SIZE_MAX;
    }
    cv.notify_all();
  }
  // Returns microseconds spent stalled waiting for this group.
  double WaitFor(size_t group_index) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return groups_done > group_index; });
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  }
};

struct RankStats {
  std::vector<double> restore_ms;  // per request: first group issued -> last layer computed
  std::vector<double> ttfl_us;     // per request: stall before group 0 is readable
  std::vector<double> stall_ms;    // per request: total forward stall across groups
  size_t get_calls = 0;
  size_t get_calls_local = 0;  // calls whose pages were all already local
  size_t get_bytes = 0;        // bytes the caller asked for
  size_t put_bytes = 0;
  // Wire accounting.  These are PREDICTIONS from the same criterion
  // AllLocallyResident applies -- see its comment for what that does and does
  // not mean.  They are what the bench can know from outside mori.
  //
  // remote_keys is the useful one: it counts (key, call) pairs that had to reach
  // the peer, so the reader can derive both wire figures and compare them
  // directly across a before/after run:
  //   whole-object fetch  ->  remote_keys * object_bytes
  //   range-granular      ->  remote_keys * group_bytes
  size_t remote_calls = 0;
  size_t remote_keys = 0;
  size_t repeat_remote = 0;      // (key, call) pairs that went remote for a key
                                 // this request had already fetched remotely
  size_t local_write_bytes = 0;  // objects that ended up installed on this node
  size_t verify_mismatches = 0;  // pages that came back wrong (--verify only)
  // Per pool, indexed as BenchOpts::pools.  Kept apart from the totals because
  // the totals cannot show the thing heterogeneous pools exist to expose: a
  // pool with a tenth the range size moves its bytes at a fraction of the rate,
  // and only a per-pool split says which pool a change actually helped.
  std::vector<size_t> pool_bytes;
  std::vector<size_t> pool_ranges;
  std::vector<size_t> pool_calls;
  std::vector<size_t> pool_remote_keys;

  void Init(size_t pools) {
    pool_bytes.assign(pools, 0);
    pool_ranges.assign(pools, 0);
    pool_calls.assign(pools, 0);
    pool_remote_keys.assign(pools, 0);
  }
  // Measured window for this rank, so aggregate throughput excludes warm-up.
  std::chrono::steady_clock::time_point window_start{};
  std::chrono::steady_clock::time_point window_end{};
};

// One rank's layer-wise restore: a loader thread walking layer groups while a
// forward thread consumes them.
// `warmup_requests` leading requests run but are not recorded: the first
// restore on a fresh client also pays the peer handshake, which a long-lived
// connector has already paid.
void RunLoadRank(Cluster& cluster, const BenchOpts& o,
                 const std::vector<std::vector<std::vector<std::string>>>& request_keys,
                 size_t warmup_requests, std::atomic<size_t>* warmed_ranks, RankStats* stats) {
  PoolClient* client = cluster.node();

  // Destination for one request's pages, one buffer per pool because a pool's
  // pages are only contiguous among themselves.  Registered once for zero copy.
  std::vector<std::vector<char>> kv(o.pools.size());
  for (size_t p = 0; p < o.pools.size(); ++p) {
    kv[p].assign(o.pages * o.pools[p].object_bytes(), 0);
    client->RegisterMemory(kv[p].data(), kv[p].size());
  }

  for (size_t req = 0; req < request_keys.size(); ++req) {
    const bool measured = req >= warmup_requests;
    if (req == warmup_requests) {
      stats->window_start = std::chrono::steady_clock::now();
      warmed_ranks->fetch_add(1, std::memory_order_relaxed);
    }
    const std::vector<std::vector<std::string>>& pool_keys = request_keys[req];
    LayerGate gate;
    const size_t groups = o.group_count();
    // Keys this request has already pulled from the peer once.  A second remote
    // pull of the same key inside one load means the object did not become
    // locally readable between two layer groups.
    std::unordered_set<std::string> pulled_remote;

    const auto req_start = std::chrono::steady_clock::now();

    std::thread loader([&] {
      for (size_t g = 0; g < groups; ++g) {
        const std::vector<size_t> group = GroupLayers(o, g);
        bool ok = true;
        // Each pool this group reaches issues its own calls, at its own range
        // size -- the loop _run_layer_wise_batch runs over _plans_covering.
        for (size_t pi = 0; pi < o.pools.size() && ok; ++pi) {
          const Pool& pool = o.pools[pi];
          const std::vector<size_t> layers = PoolGroupLayers(pool, group);
          if (layers.empty()) continue;
          const std::vector<std::string>& keys = pool_keys[pi];
          // Every page carries one range per layer of this pool in this group.
          const size_t step = EntriesPerCall(o, layers.size());
          for (size_t start = 0; start < keys.size() && ok; start += step) {
            const size_t end = std::min(start + step, keys.size());
            const size_t count = end - start;

            std::vector<std::string> chunk_keys(keys.begin() + start, keys.begin() + end);
            std::vector<std::vector<void*>> dsts(count);
            std::vector<std::vector<size_t>> sizes(count);
            std::vector<std::vector<size_t>> offsets(count);
            for (size_t i = 0; i < count; ++i) {
              char* page = kv[pi].data() + (start + i) * pool.object_bytes();
              for (size_t l : layers) {
                dsts[i].push_back(page + l * pool.layer_bytes);
                sizes[i].push_back(pool.layer_bytes);
                offsets[i].push_back(l * pool.layer_bytes);
              }
            }

            // Whether this call can be served without the arena at all.
            const bool all_local = AllLocallyResident(client, chunk_keys);

            auto res = client->BatchGetRanges(chunk_keys, dsts, sizes, offsets);
            ok = res.size() == count &&
                 std::all_of(res.begin(), res.end(), [](bool b) { return b; });

            if (measured) {
              stats->get_calls += 1;
              if (all_local) stats->get_calls_local += 1;
              if (ok) {
                const size_t moved = count * layers.size() * pool.layer_bytes;
                stats->get_bytes += moved;
                stats->pool_bytes[pi] += moved;
                stats->pool_ranges[pi] += count * layers.size();
                stats->pool_calls[pi] += 1;
              }
            }
            if (!all_local) {
              for (const auto& k : chunk_keys) {
                const bool again = !pulled_remote.insert(k).second;
                if (measured && again) stats->repeat_remote += 1;
              }
              if (measured) {
                stats->remote_calls += 1;
                stats->remote_keys += count;
                stats->pool_remote_keys[pi] += count;
              }
            }
          }
        }
        if (!ok) {
          gate.Fail();
          return;
        }
        // Only now is every layer in the group readable, so they are released
        // together -- the connector releases a whole group at once too.
        gate.Complete(g + 1);
      }
    });

    // Forward: wait for each group, then "compute" its layers while the loader
    // is already fetching the next one.
    double stall_us = 0.0;
    double ttfl = 0.0;
    for (size_t g = 0; g < groups; ++g) {
      const double waited = gate.WaitFor(g);
      if (g == 0) ttfl = waited;
      stall_us += waited;
      {
        std::lock_guard<std::mutex> lk(gate.mu);
        if (gate.failed) break;
      }
      SpinUs(o.compute_us_per_layer * GroupLayers(o, g).size());
    }
    loader.join();

    const double restore_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - req_start)
            .count();
    if (measured) {
      stats->restore_ms.push_back(restore_ms);
      stats->ttfl_us.push_back(ttfl);
      stats->stall_ms.push_back(stall_us / 1000.0);
      stats->window_end = std::chrono::steady_clock::now();
      // Byte-check the whole restored working set.  Deliberately after the
      // timing is recorded, so per-request latency is unaffected; it does sit
      // inside the throughput window, so --verify runs are for correctness and
      // the timing runs are separate.
      if (o.verify) {
        for (size_t pi = 0; pi < o.pools.size(); ++pi) {
          const size_t object_bytes = o.pools[pi].object_bytes();
          for (size_t p = 0; p < o.pages; ++p) {
            const char* page = kv[pi].data() + p * object_bytes;
            // Seeded per pool, so the index must match SeedPeer's ordering.
            const size_t seed_index = req * o.pages + p;
            for (size_t b = 0; b < object_bytes; ++b) {
              if (page[b] != SeedByte(seed_index, b)) {
                stats->verify_mismatches += 1;
                break;
              }
            }
          }
        }
      }

      // How much of this request's working set the remote path installed here.
      // Counted after the load so an asynchronous install has had the whole
      // request to land; it is still a lower bound if one lands later.
      if (auto* backend = client->Backends().Get(client->Medium())) {
        for (size_t pi = 0; pi < o.pools.size(); ++pi) {
          for (const auto& entry : backend->BatchResolve(pool_keys[pi], /*include_descs=*/false)) {
            if (entry.found) stats->local_write_bytes += o.pools[pi].object_bytes();
          }
        }
      }
    }
  }
}

// One rank's offload stream: whole-object writes that overlap the loads above.
// This is the traffic that used to contend with loading on a single arena.
void RunOffloadRank(Cluster& cluster, const BenchOpts& o, size_t rank,
                    const std::atomic<bool>& stop, const std::atomic<size_t>& warmed_ranks,
                    size_t load_ranks, RankStats* stats) {
  PoolClient* client = cluster.node();
  // One buffer per pool, mirroring the load side.
  std::vector<std::vector<char>> kv(o.pools.size());
  for (size_t pi = 0; pi < o.pools.size(); ++pi) {
    kv[pi].assign(o.pages * o.pools[pi].object_bytes(), 0);
    for (size_t i = 0; i < kv[pi].size(); ++i)
      kv[pi][i] = static_cast<char>((i + rank + pi) & 0xff);
    client->RegisterMemory(kv[pi].data(), kv[pi].size());
  }

  for (size_t round = 0; !stop.load(std::memory_order_relaxed); ++round) {
    for (size_t pi = 0; pi < o.pools.size() && !stop.load(std::memory_order_relaxed); ++pi) {
      const Pool& pool = o.pools[pi];
      const size_t object_bytes = pool.object_bytes();
      // A put names every layer of the page at once: its ranges must tile the
      // object exactly, so offload never walks groups.
      const size_t step = EntriesPerCall(o, pool.layers.size());

      for (size_t start = 0; start < o.pages && !stop.load(std::memory_order_relaxed);
           start += step) {
        const size_t end = std::min(start + step, o.pages);
        const size_t count = end - start;

        std::vector<std::string> chunk_keys(count);
        std::vector<size_t> object_sizes(count, object_bytes);
        std::vector<std::vector<const void*>> srcs(count);
        std::vector<std::vector<size_t>> sizes(count);
        std::vector<std::vector<size_t>> offsets(count);
        for (size_t i = 0; i < count; ++i) {
          // Fresh key every time: a repeat would be short-circuited by
          // RoutePut's dedup and never reach the data path.
          chunk_keys[i] = "off-" + pool.name + "-r" + std::to_string(rank) + "-" +
                          std::to_string(round) + "-" + std::to_string(start + i);
          char* page = kv[pi].data() + (start + i) * object_bytes;
          for (size_t l = 0; l < pool.layers.size(); ++l) {
            srcs[i].push_back(page + l * pool.layer_bytes);
            sizes[i].push_back(pool.layer_bytes);
            offsets[i].push_back(l * pool.layer_bytes);
          }
        }
        auto res = client->BatchPutRanges(chunk_keys, object_sizes, srcs, sizes, offsets);
        // Offload keeps running through warm-up so the arena is loaded from the
        // start, but only bytes inside the measured window are counted.
        const bool in_window = warmed_ranks.load(std::memory_order_relaxed) >= load_ranks;
        for (size_t i = 0; i < res.size(); ++i) {
          if (res[i] && in_window) stats->put_bytes += object_bytes;
        }
      }
    }
  }
}

void RunOne(const BenchOpts& o, size_t ranks) {
  const size_t offload_ranks = (o.offload_ranks == SIZE_MAX) ? ranks : o.offload_ranks;
  const size_t object_bytes = o.object_bytes();

  // The node must be able to hold one restore working set per rank (that is
  // what makes groups 2..N local hits), plus slack for offloads routed here.
  const size_t node_capacity =
      std::max<size_t>(size_t{256} << 20, ranks * o.pages * object_bytes * 3);
  // The peer holds every seeded page for every rank and request (+1 warm-up).
  const size_t peer_capacity =
      std::max<size_t>(size_t{512} << 20,
                       ranks * (o.requests + 1) * o.pages * object_bytes * 2 + (size_t{512} << 20));
  // The arena must hold at least one whole object: the remote path stages
  // through it, and an object that does not fit is unservable.  With several
  // pools the largest one sets the floor.
  const size_t scratch_bytes = std::max<size_t>(o.max_object_bytes(), o.scratch_mib << 20);

  Cluster cluster(node_capacity, peer_capacity, scratch_bytes, o.medium_page_bytes(),
                  o.locality_prefetch);

  // Seed: each (rank, request) gets its own pages, so the first group of every
  // request is a real remote miss.  One extra leading request per rank is a
  // warm-up that pays the peer handshake and is not recorded.
  constexpr size_t kWarmupRequests = 1;
  const size_t total_requests = o.requests + kWarmupRequests;
  // [rank][request][pool] -> that pool's page keys.  A pool's object holds only
  // its own layers, so pools cannot share keys.
  std::vector<std::vector<std::vector<std::vector<std::string>>>> keys(ranks);
  std::vector<std::vector<std::vector<char>>> seed_storage(ranks);
  std::vector<std::string> all_keys;
  for (size_t r = 0; r < ranks; ++r) {
    keys[r].resize(total_requests);
    seed_storage[r].resize(o.pools.size());
    // Seeded per pool so SeedByte's index is (request, page) within the pool,
    // which is what the verify pass recomputes.
    std::vector<std::vector<std::string>> flat(o.pools.size());
    for (size_t q = 0; q < total_requests; ++q) {
      keys[r][q].resize(o.pools.size());
      for (size_t pi = 0; pi < o.pools.size(); ++pi) {
        for (size_t p = 0; p < o.pages; ++p) {
          keys[r][q][pi].push_back(o.pools[pi].name + "-r" + std::to_string(r) + "-q" +
                                   std::to_string(q) + "-p" + std::to_string(p));
          flat[pi].push_back(keys[r][q][pi].back());
        }
      }
    }
    for (size_t pi = 0; pi < o.pools.size(); ++pi) {
      SeedPeer(cluster.peer(), flat[pi], o.pools[pi].object_bytes(), &seed_storage[r][pi]);
      all_keys.insert(all_keys.end(), flat[pi].begin(), flat[pi].end());
    }
  }
  cluster.peer()->Master().FlushHeartbeat();
  cluster.node()->Master().FlushHeartbeat();
  WaitAllVisible(cluster.node(), all_keys);

  std::vector<RankStats> load_stats(ranks);
  std::vector<RankStats> offload_stats(std::max<size_t>(offload_ranks, 1));
  for (auto& s : load_stats) s.Init(o.pools.size());
  for (auto& s : offload_stats) s.Init(o.pools.size());
  std::atomic<bool> stop{false};

  // Offload streams run for the whole measurement window, the way a steady
  // state has finished requests being written back while new ones load.
  std::atomic<size_t> warmed_ranks{0};
  std::vector<std::thread> offloaders;
  for (size_t r = 0; r < offload_ranks; ++r) {
    offloaders.emplace_back(
        [&, r] { RunOffloadRank(cluster, o, r, stop, warmed_ranks, ranks, &offload_stats[r]); });
  }

  std::vector<std::thread> loaders;
  for (size_t r = 0; r < ranks; ++r) {
    loaders.emplace_back([&, r] {
      RunLoadRank(cluster, o, keys[r], kWarmupRequests, &warmed_ranks, &load_stats[r]);
    });
  }
  for (auto& t : loaders) t.join();

  stop.store(true, std::memory_order_relaxed);
  for (auto& t : offloaders) t.join();

  // Measured window: from the first rank leaving warm-up to the last rank
  // finishing, so warm-up traffic is outside the throughput denominator.
  auto window_start = std::chrono::steady_clock::time_point::max();
  auto window_end = std::chrono::steady_clock::time_point::min();
  for (const auto& s : load_stats) {
    if (s.window_start != std::chrono::steady_clock::time_point{}) {
      window_start = std::min(window_start, s.window_start);
      window_end = std::max(window_end, s.window_end);
    }
  }
  const double wall_s = window_end > window_start
                            ? std::chrono::duration<double>(window_end - window_start).count()
                            : 0.0;
  std::vector<double> restore, ttfl, stall;
  size_t get_bytes = 0, put_bytes = 0, calls = 0, calls_local = 0;
  size_t remote_calls = 0, remote_keys = 0, repeat_remote = 0, local_write = 0, mismatches = 0;
  std::vector<size_t> pool_bytes(o.pools.size(), 0), pool_ranges(o.pools.size(), 0);
  std::vector<size_t> pool_calls(o.pools.size(), 0), pool_remote_keys(o.pools.size(), 0);
  for (auto& s : load_stats) {
    restore.insert(restore.end(), s.restore_ms.begin(), s.restore_ms.end());
    ttfl.insert(ttfl.end(), s.ttfl_us.begin(), s.ttfl_us.end());
    stall.insert(stall.end(), s.stall_ms.begin(), s.stall_ms.end());
    get_bytes += s.get_bytes;
    calls += s.get_calls;
    calls_local += s.get_calls_local;
    remote_calls += s.remote_calls;
    remote_keys += s.remote_keys;
    repeat_remote += s.repeat_remote;
    local_write += s.local_write_bytes;
    mismatches += s.verify_mismatches;
    for (size_t pi = 0; pi < o.pools.size(); ++pi) {
      pool_bytes[pi] += s.pool_bytes[pi];
      pool_ranges[pi] += s.pool_ranges[pi];
      pool_calls[pi] += s.pool_calls[pi];
      pool_remote_keys[pi] += s.pool_remote_keys[pi];
    }
  }
  for (auto& s : offload_stats) put_bytes += s.put_bytes;

  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  constexpr double kMiB = 1024.0 * 1024.0;
  // Both wire figures, so a before/after pair can be compared without knowing
  // which fetch granularity the binary under test used.
  const double wire_whole_mib = remote_keys * object_bytes / kMiB;
  const double wire_range_mib = remote_keys * o.group_bytes() / kMiB;
  std::printf(
      "%zu,%zu,%zu,%zu,%zu,%.2f,%.2f,%.1f,%.1f,%.2f,%.4f,%.4f,%.1f,%zu,%zu,%zu,%.1f,%.1f,%.1f\n",
      ranks, o.layers, o.layer_group, o.pages, object_bytes / 1024, Percentile(restore, 0.50),
      Percentile(restore, 0.95), Percentile(ttfl, 0.50), Percentile(ttfl, 0.95),
      Percentile(stall, 0.50), wall_s > 0 ? (get_bytes / kGiB) / wall_s : 0.0,
      wall_s > 0 ? (put_bytes / kGiB) / wall_s : 0.0, calls > 0 ? 100.0 * calls_local / calls : 0.0,
      remote_calls, remote_keys, repeat_remote, wire_whole_mib, wire_range_mib, local_write / kMiB);
  std::fflush(stdout);

  // Per-pool split.  On stderr so the CSV schema above stays stable, and only
  // when there is more than one pool to split.  range_kib is the number that
  // predicts the rate: this path is fragment-size bound, so two pools moving
  // equal bytes at different range sizes will not move them at equal speed.
  if (o.pools.size() > 1) {
    std::fprintf(stderr, "  %-16s %9s %10s %8s %10s %9s %10s\n", "pool", "range_kib", "ranges",
                 "calls", "bytes_mib", "share_pct", "remote_keys");
    for (size_t pi = 0; pi < o.pools.size(); ++pi) {
      std::fprintf(stderr, "  %-16s %9.2f %10zu %8zu %10.1f %9.1f %10zu\n",
                   o.pools[pi].name.c_str(), o.pools[pi].layer_bytes / 1024.0, pool_ranges[pi],
                   pool_calls[pi], pool_bytes[pi] / kMiB,
                   get_bytes > 0 ? 100.0 * pool_bytes[pi] / get_bytes : 0.0, pool_remote_keys[pi]);
    }
    std::fflush(stderr);
  }

  if (o.verify) {
    std::fprintf(stderr, "verify: %zu mismatched pages at %zu ranks\n", mismatches, ranks);
    if (mismatches != 0) std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  BenchOpts opts;
  if (!ParseArgs(argc, argv, &opts)) return 2;

  std::fprintf(stderr,
               "layer-wise restore: %zu layers in groups of %zu (%zu groups/request), "
               "%zu pages/request, page=%zu KiB across %zu pool(s), mean group slice=%zu KiB, "
               "medium page=%zu KiB, arena=%zu MiB, locality_prefetch=%s, layout=%s\n",
               opts.layers, opts.layer_group, opts.group_count(), opts.pages,
               opts.object_bytes() / 1024, opts.pools.size(), opts.group_bytes() / 1024,
               opts.medium_page_bytes() / 1024, opts.scratch_mib,
               opts.locality_prefetch ? "on" : "off",
               opts.interleave_groups ? "interleaved" : "contiguous");
  for (const Pool& pool : opts.pools) {
    std::fprintf(stderr, "  pool %-16s %3zu layers x %8zu B = %8zu KiB/page\n", pool.name.c_str(),
                 pool.layers.size(), pool.layer_bytes, pool.object_bytes() / 1024);
  }

  std::printf(
      "ranks,layers,group,pages,object_kib,restore_ms_p50,restore_ms_p95,ttfl_us_p50,"
      "ttfl_us_p95,stall_ms_p50,get_gibps,put_gibps,local_hit_pct,"
      "remote_calls,remote_keys,repeat_remote,wire_whole_mib,wire_range_mib,local_write_mib\n");
  std::fflush(stdout);

  for (size_t ranks : opts.rank_sweep) RunOne(opts, ranks);
  return 0;
}
