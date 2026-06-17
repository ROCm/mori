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

// PoolClient::BatchGet micro-bench (mirror of bench_pool_client_batch_put.cpp).
// Single implementation path; no --impl switch.  For ablation against an
// older revision, git checkout the previous release commit and re-run the
// same scenario binary; record both CSV rows in the PR description.
//
// Reports wall_ms and GiB/s aggregate throughput for the BatchGet data path.
//
// Usage:
//   bench_pool_client_batch_get [--scenario all_zc|mixed|all_stg|multi_peer]
//                               [--batch N] [--page-bytes N]
//                               [--per-item-pages N] [--iters N]
//                               [--peers N]
//                               [--sweep batch=1,4,16,64,256]
//
// CSV (stdout):
//   scenario,batch,page_bytes,iters,wall_ms,gibps

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::PoolClient;
using mori::umbp::PoolClientConfig;
using mori::umbp::TierType;

namespace {

// Unique peer-service port per PoolClient: required so each node registers a
// peer_address and can serve/accept remote AllocateSlot/ResolveKey RPCs.
// The base is seeded from the pid so rapidly-restarted bench processes (e.g.
// a per-(pip,batch) sweep) don't collide on a port left in TIME_WAIT by a
// prior process.
inline uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(20000 + (static_cast<unsigned>(::getpid()) * 16u) % 40000u)};
  return next.fetch_add(1);
}

struct BenchOpts {
  std::string scenario = "all_zc";
  size_t batch = 64;
  size_t page_bytes = 4096;
  size_t per_item_pages = 1;
  size_t iters = 10;
  size_t peers = 1;
  std::vector<size_t> sweep;
};

void Usage() {
  std::cerr << "Usage: bench_pool_client_batch_get "
               "[--scenario all_zc|mixed|mixed_tier|all_stg|multi_peer]\n"
            << "                                   [--batch N] [--page-bytes N]\n"
            << "                                   [--per-item-pages N] [--iters N]\n"
            << "                                   [--peers N]\n"
            << "                                   [--sweep batch=1,4,16,64,256]\n";
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
    if (a == "--scenario") {
      o->scenario = next("--scenario");
    } else if (a == "--batch") {
      o->batch = std::strtoull(next("--batch"), nullptr, 10);
    } else if (a == "--page-bytes") {
      o->page_bytes = std::strtoull(next("--page-bytes"), nullptr, 10);
    } else if (a == "--per-item-pages") {
      o->per_item_pages = std::strtoull(next("--per-item-pages"), nullptr, 10);
    } else if (a == "--iters") {
      o->iters = std::strtoull(next("--iters"), nullptr, 10);
    } else if (a == "--peers") {
      o->peers = std::strtoull(next("--peers"), nullptr, 10);
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
  return true;
}

// Cluster: one master, one caller (drives BatchGet), N source peers
// (hold the seed data via prior BatchPut).  In all_zc / mixed /
// multi_peer scenarios the caller pre-registers caller_buf_ as dst;
// in all_stg it doesn't, so REMOTE_STG path activates.
struct PeerNode {
  std::vector<char> dram;  // master-managed exportable DRAM (data lives here)
  std::vector<char> seed;  // src for the seed BatchPut (registered for ZC Put)
  std::unique_ptr<PoolClient> client;
};

class Cluster {
 public:
  Cluster(size_t page_bytes, size_t target_dram_bytes, size_t num_peers, size_t caller_buf_bytes,
          size_t caller_local_bytes, size_t seed_bytes_per_peer)
      : page_bytes_(page_bytes), caller_buf_(caller_buf_bytes), caller_local_(caller_local_bytes) {
    MasterServerConfig mcfg;
    mcfg.listen_address = "0.0.0.0:0";
    // The master index is eventually consistent: a committed key becomes
    // visible only after the owning peer ships its ADD event on the next
    // heartbeat (interval = heartbeat_ttl / divisor).  Shorten the TTL so
    // seeded keys converge in ~0.5s instead of the 5s default, keeping the
    // pre-Get visibility barrier (WaitAllVisible) cheap.
    mcfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};
    master_ = std::make_unique<MasterServer>(std::move(mcfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 200 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (master_->GetBoundPort() == 0) {
      std::cerr << "master failed to start\n";
      std::exit(2);
    }
    auto master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    PoolClientConfig cc;
    cc.master_config.node_id = "node-caller";
    cc.master_config.node_address = "127.0.0.1";
    cc.master_config.master_address = master_addr;
    cc.io_engine.host = "0.0.0.0";
    cc.io_engine.port = 0;
    cc.peer_service_port = NextPeerServicePort();
    cc.dram_page_size = page_bytes;
    cc.dram_buffers = {{caller_local_.data(), caller_local_.size()}};
    cc.tier_capacities = {{TierType::DRAM, {caller_local_.size(), caller_local_.size()}}};
    caller_ = std::make_unique<PoolClient>(std::move(cc));
    if (!caller_->Init()) {
      std::cerr << "caller init failed\n";
      std::exit(2);
    }

    peers_.resize(num_peers);
    for (size_t k = 0; k < num_peers; ++k) {
      peers_[k].dram.assign(target_dram_bytes, 0);
      peers_[k].seed.assign(seed_bytes_per_peer, 0);
      PoolClientConfig tc;
      tc.master_config.node_id = "node-target-" + std::to_string(k);
      tc.master_config.node_address = "127.0.0.1";
      tc.master_config.master_address = master_addr;
      tc.io_engine.host = "0.0.0.0";
      tc.io_engine.port = 0;
      tc.peer_service_port = NextPeerServicePort();
      tc.dram_page_size = page_bytes;
      tc.dram_buffers = {{peers_[k].dram.data(), peers_[k].dram.size()}};
      tc.tier_capacities = {{TierType::DRAM, {peers_[k].dram.size(), peers_[k].dram.size()}}};
      peers_[k].client = std::make_unique<PoolClient>(std::move(tc));
      if (!peers_[k].client->Init()) {
        std::cerr << "target " << k << " init failed\n";
        std::exit(2);
      }
      // Peer registers its seed region so its own seed-BatchPut goes ZC.
      peers_[k].client->RegisterMemory(peers_[k].seed.data(), peers_[k].seed.size());
    }
  }

  ~Cluster() {
    if (caller_) caller_->Shutdown();
    for (auto& p : peers_) {
      if (p.client) p.client->Shutdown();
    }
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
  }

  PoolClient* caller() { return caller_.get(); }
  PoolClient* peer(size_t k) { return peers_[k].client.get(); }
  std::vector<char>& peer_seed(size_t k) { return peers_[k].seed; }
  std::vector<char>& caller_buf() { return caller_buf_; }
  size_t num_peers() const { return peers_.size(); }
  size_t page_bytes() const { return page_bytes_; }

 private:
  size_t page_bytes_;
  std::vector<char> caller_buf_;
  std::vector<char> caller_local_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::vector<PeerNode> peers_;
};

// Returns (wall_ms, success_bytes).
std::pair<double, size_t> RunOnce(PoolClient* client, const std::vector<std::string>& keys,
                                  const std::vector<void*>& dsts,
                                  const std::vector<size_t>& sizes) {
  auto t0 = std::chrono::steady_clock::now();
  auto r = client->BatchGet(keys, dsts, sizes);
  auto t1 = std::chrono::steady_clock::now();
  size_t bytes = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    if (r[i]) bytes += sizes[i];
  }
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return {ms, bytes};
}

// Barrier: block until every key is visible to `client` via the master
// index.  Committed keys propagate on the owning peer's next heartbeat, so
// a BatchGet issued immediately after seeding would otherwise route-miss.
// This runs OUTSIDE the timed BatchGet and never counts toward throughput.
void WaitAllVisible(PoolClient* client, const std::vector<std::string>& keys,
                    std::chrono::milliseconds timeout = std::chrono::seconds{10}) {
  if (keys.empty()) return;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    auto present = client->BatchExists(keys);
    bool all = true;
    for (bool p : present) {
      if (!p) {
        all = false;
        break;
      }
    }
    if (all) return;
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "WaitAllVisible: keys not visible before timeout\n";
      std::exit(2);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// Seed `n` keys onto target peer `k` via that peer's BatchPut.  Returns
// the list of seeded keys parallel to caller's dsts/sizes.
void SeedPeer(Cluster& cluster, size_t peer_idx, size_t n, size_t bytes_per_item,
              const std::string& key_prefix, std::vector<std::string>* out_keys) {
  std::vector<const void*> srcs(n);
  std::vector<size_t> sizes(n);
  out_keys->clear();
  out_keys->reserve(n);
  auto& seed = cluster.peer_seed(peer_idx);
  for (size_t i = 0; i < n; ++i) {
    char* slot = seed.data() + i * bytes_per_item;
    std::memset(slot, static_cast<int>(0x10 + (i & 0x7F)), bytes_per_item);
    srcs[i] = slot;
    sizes[i] = bytes_per_item;
    out_keys->push_back(key_prefix + std::to_string(i));
  }
  auto r = cluster.peer(peer_idx)->BatchPut(*out_keys, srcs, sizes);
  for (size_t i = 0; i < n; ++i) {
    if (!r[i]) {
      std::cerr << "seed Put failed for peer " << peer_idx << " key " << (*out_keys)[i] << "\n";
      std::exit(2);
    }
  }
}

// Seed `n` keys onto `client`'s OWN DRAM pool via that client's BatchPut.  With
// the master running UMBP_ROUTE_PUT_NODE_AFFINITY=local (set in mixed_tier),
// these Puts self-route, so the client's later BatchGet of them hits the LOCAL
// DRAM path.  `seed` is a client-owned src buffer (>= n * bytes_per_item).
void SeedClientLocal(PoolClient* client, std::vector<char>& seed, size_t n, size_t bytes_per_item,
                     const std::string& key_prefix, std::vector<std::string>* out_keys) {
  std::vector<const void*> srcs(n);
  std::vector<size_t> sizes(n);
  out_keys->clear();
  out_keys->reserve(n);
  for (size_t i = 0; i < n; ++i) {
    char* slot = seed.data() + i * bytes_per_item;
    std::memset(slot, static_cast<int>(0x40 + (i & 0x3F)), bytes_per_item);
    srcs[i] = slot;
    sizes[i] = bytes_per_item;
    out_keys->push_back(key_prefix + std::to_string(i));
  }
  auto r = client->BatchPut(*out_keys, srcs, sizes);
  for (size_t i = 0; i < n; ++i) {
    if (!r[i]) {
      std::cerr << "local seed Put failed for key " << (*out_keys)[i] << "\n";
      std::exit(2);
    }
  }
}

void RunScenario(const BenchOpts& base, size_t batch_override) {
  BenchOpts o = base;
  o.batch = batch_override > 0 ? batch_override : o.batch;

  const bool mixed_tier = (o.scenario == "mixed_tier");
  // mixed_tier: half the batch is served from the caller's OWN local DRAM and
  // half from a remote peer, so the local memcpy overlaps the remote DRAM RDMA
  // wire under overlap.  Force self-routing of the caller's seed Puts via local
  // node affinity on the (in-process) master; read once at master construction.
  if (mixed_tier) {
    setenv("UMBP_ROUTE_PUT_NODE_AFFINITY", "local", /*overwrite=*/1);
  }
  size_t num_peers = (o.scenario == "multi_peer") ? std::max<size_t>(o.peers, 2) : 1;
  size_t bytes_per_item = o.per_item_pages * o.page_bytes;
  const size_t n_local = mixed_tier ? (o.batch / 2) : 0;
  const size_t n_remote_total = o.batch - n_local;
  size_t per_peer_items = mixed_tier ? n_remote_total : (o.batch + num_peers - 1) / num_peers;
  // Each iter seeds a fresh, unique key set that stays resident in the peer
  // DRAM pool (no eviction between iters), so the target must hold warmup +
  // all measured iters' keys plus slack — otherwise late-iter seed Puts hit
  // ENOSPC.  (+2 = warmup + headroom.)
  size_t target_dram = std::max<size_t>(static_cast<size_t>(64) << 20,
                                        per_peer_items * bytes_per_item * (o.iters + 2));
  size_t seed_bytes = per_peer_items * bytes_per_item;
  size_t caller_buf_bytes = o.batch * bytes_per_item;
  // Caller_local DRAM pool: tiny by default (forces caller self-Puts to remote);
  // for mixed_tier it must hold the caller's local half across warmup + iters.
  size_t caller_local_bytes = mixed_tier
                                  ? std::max<size_t>(static_cast<size_t>(64) << 20,
                                                     n_local * bytes_per_item * (o.iters + 2))
                              : (o.scenario == "mixed") ? caller_buf_bytes
                                                        : o.page_bytes;
  // Caller-owned src buffer for the local seed Puts (mixed_tier only).
  std::vector<char> caller_seed(mixed_tier ? n_local * bytes_per_item : 0, 0);

  Cluster cluster(o.page_bytes, target_dram, num_peers, caller_buf_bytes, caller_local_bytes,
                  seed_bytes);

  // Register caller's dst region for ZC; skip for all_stg.
  if (o.scenario != "all_stg") {
    cluster.caller()->RegisterMemory(cluster.caller_buf().data(), caller_buf_bytes);
  }

  // Seed all peers up-front.  For multi_peer, distribute round-robin
  // across peers.  For mixed, half the keys are seeded on caller (so
  // BatchGet hits LOCAL branch) and half on target.
  // Build the per-iter input vectors.  Keys must change per iter so
  // master allocator doesn't see duplicates from the caller's BatchGet
  // (Lookup itself is idempotent, but seed Puts use unique keys per iter).

  // Warm-up iter (not measured): seed iter 0 + run BatchGet.
  auto build_iter = [&](size_t it, std::vector<std::string>* keys, std::vector<void*>* dsts,
                        std::vector<size_t>* sizes) {
    keys->clear();
    keys->reserve(o.batch);
    dsts->resize(o.batch);
    sizes->resize(o.batch);
    if (o.scenario == "multi_peer") {
      // Seed each peer with its share of keys, distinct prefix per peer
      // and per iter.
      size_t per_peer = (o.batch + num_peers - 1) / num_peers;
      for (size_t p = 0; p < num_peers; ++p) {
        std::vector<std::string> peer_keys;
        SeedPeer(cluster, p, per_peer, bytes_per_item,
                 "g-" + std::to_string(it) + "-p" + std::to_string(p) + "-", &peer_keys);
        for (size_t i = 0; i < per_peer && keys->size() < o.batch; ++i) {
          keys->push_back(peer_keys[i]);
        }
      }
    } else if (mixed_tier) {
      // Half remote (peer 0, REMOTE_ZC) + half local (caller's own DRAM,
      // LOCAL).  Keys ordered remote-then-local; dsts are caller_buf slots.
      std::vector<std::string> remote_keys, local_keys;
      SeedPeer(cluster, 0, n_remote_total, bytes_per_item, "g-" + std::to_string(it) + "-r-",
               &remote_keys);
      SeedClientLocal(cluster.caller(), caller_seed, n_local, bytes_per_item,
                      "g-" + std::to_string(it) + "-l-", &local_keys);
      for (const auto& k : remote_keys) keys->push_back(k);
      for (const auto& k : local_keys) keys->push_back(k);
    } else {
      // Single source peer (peer 0).  For mixed, half the items will
      // be served by the caller's own LOCAL DRAM via a separate seed
      // path... actually the simplest mixed model is: all seeded on
      // peer 0, all Get'd by caller (REMOTE_ZC).  "Mixed" in BatchPut
      // came from caller's own dram_buffers absorbing some Puts; the
      // analogue for Get would be Get'ing keys whose data lives on
      // self-node, which requires seeding via caller_->BatchPut into
      // caller_local_.  We approximate by leaving mixed == all_zc here;
      // the LOCAL fast path is exercised by the cross_node smoke test
      // and by the unit test MixedLocalAndRemoteZC, not the bench.
      std::vector<std::string> peer_keys;
      SeedPeer(cluster, 0, o.batch, bytes_per_item, "g-" + std::to_string(it) + "-", &peer_keys);
      for (size_t i = 0; i < o.batch; ++i) keys->push_back(peer_keys[i]);
    }
    for (size_t i = 0; i < o.batch; ++i) {
      (*dsts)[i] = cluster.caller_buf().data() + i * bytes_per_item;
      (*sizes)[i] = bytes_per_item;
    }
    // Visibility barrier (not timed): ensure the caller can route all keys
    // before the measured BatchGet, otherwise route-miss yields 0 GiB/s.
    WaitAllVisible(cluster.caller(), *keys);
  };

  std::vector<std::string> keys;
  std::vector<void*> dsts;
  std::vector<size_t> sizes;
  build_iter(0, &keys, &dsts, &sizes);
  RunOnce(cluster.caller(), keys, dsts, sizes);

  double total_ms = 0;
  size_t total_bytes = 0;
  for (size_t it = 1; it <= o.iters; ++it) {
    build_iter(it, &keys, &dsts, &sizes);
    auto [ms, bytes] = RunOnce(cluster.caller(), keys, dsts, sizes);
    total_ms += ms;
    total_bytes += bytes;
  }

  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const double mean_gibps = total_ms > 0 ? (total_bytes / kGiB) / (total_ms / 1000.0) : 0.0;

  std::printf("%s,%zu,%zu,%zu,%.3f,%.3f\n", o.scenario.c_str(), o.batch, o.page_bytes, o.iters,
              total_ms / o.iters, mean_gibps);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  BenchOpts opts;
  if (!ParseArgs(argc, argv, &opts)) return 2;

  std::printf("scenario,batch,page_bytes,iters,wall_ms,gibps\n");
  std::fflush(stdout);

  if (!opts.sweep.empty()) {
    for (size_t b : opts.sweep) RunScenario(opts, b);
  } else {
    RunScenario(opts, 0);
  }
  return 0;
}
