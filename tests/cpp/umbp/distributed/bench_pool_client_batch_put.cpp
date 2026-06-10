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

// PoolClient::BatchPut micro-bench.  Standalone executable; reports
// wall_ms and GiB/s aggregate throughput for the BatchPut data path.
//
// Usage:
//   bench_pool_client_batch_put [--scenario all_zc|mixed|all_stg|multi_peer]
//                               [--batch N] [--page-bytes N]
//                               [--per-item-pages N] [--iters N]
//                               [--peers N]
//                               [--sweep batch=1,4,16,64,256]
//
// CSV (stdout):
//   scenario,batch,page_bytes,iters,wall_ms,gibps

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
// peer_address and can serve/accept remote AllocateSlot/CommitSlot RPCs.
inline uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{54000};
  return next.fetch_add(1);
}

struct BenchOpts {
  std::string scenario = "all_zc";
  size_t batch = 64;
  size_t page_bytes = 4096;
  size_t per_item_pages = 1;
  size_t iters = 10;
  size_t peers = 1;
  std::vector<size_t> sweep;  // empty = no sweep
};

void Usage() {
  std::cerr << "Usage: bench_pool_client_batch_put [--scenario all_zc|mixed|all_stg|multi_peer]\n"
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
      // Format: batch=1,4,16,64,256
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

// One peer = one PoolClient with a remote DRAM buffer.
struct PeerNode {
  std::vector<char> buf;
  std::unique_ptr<PoolClient> client;
};

class Cluster {
 public:
  Cluster(size_t page_bytes, size_t target_dram_bytes, size_t num_peers, size_t caller_buf_bytes,
          size_t caller_local_bytes)
      : page_bytes_(page_bytes), caller_buf_(caller_buf_bytes), caller_local_(caller_local_bytes) {
    MasterServerConfig mcfg;
    mcfg.listen_address = "0.0.0.0:0";
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
      peers_[k].buf.assign(target_dram_bytes, 0);
      PoolClientConfig tc;
      tc.master_config.node_id = "node-target-" + std::to_string(k);
      tc.master_config.node_address = "127.0.0.1";
      tc.master_config.master_address = master_addr;
      tc.io_engine.host = "0.0.0.0";
      tc.io_engine.port = 0;
      tc.peer_service_port = NextPeerServicePort();
      tc.dram_page_size = page_bytes;
      tc.dram_buffers = {{peers_[k].buf.data(), peers_[k].buf.size()}};
      tc.tier_capacities = {{TierType::DRAM, {peers_[k].buf.size(), peers_[k].buf.size()}}};
      peers_[k].client = std::make_unique<PoolClient>(std::move(tc));
      if (!peers_[k].client->Init()) {
        std::cerr << "target " << k << " init failed\n";
        std::exit(2);
      }
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
  std::vector<char>& caller_buf() { return caller_buf_; }
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

// Returns (wall_ms, success_bytes).  Caller computes throughput from the
// totals: averaging per-iter GiB/s arithmetically would over-weight fast
// iterations (GiB/s is 1/time, not linear in time).
std::pair<double, size_t> RunOnce(PoolClient* client, const std::vector<std::string>& keys,
                                  const std::vector<const void*>& srcs,
                                  const std::vector<size_t>& sizes) {
  auto t0 = std::chrono::steady_clock::now();
  auto r = client->BatchPut(keys, srcs, sizes);
  auto t1 = std::chrono::steady_clock::now();
  size_t bytes = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    if (r[i]) bytes += sizes[i];
  }
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return {ms, bytes};
}

void RunScenario(const BenchOpts& base, size_t batch_override) {
  BenchOpts o = base;
  o.batch = batch_override > 0 ? batch_override : o.batch;

  // Scenario -> cluster shape + caller setup.
  size_t num_peers = (o.scenario == "multi_peer") ? std::max<size_t>(o.peers, 2) : 1;
  size_t target_dram = std::max<size_t>(static_cast<size_t>(64) << 20,
                                        o.batch * o.per_item_pages * o.page_bytes * 4);  // headroom
  size_t caller_buf_bytes = o.batch * o.per_item_pages * o.page_bytes;
  size_t caller_local_bytes = (o.scenario == "mixed")
                                  ? caller_buf_bytes  // big enough so half routes local
                                  : o.page_bytes;     // tiny, forces remote

  Cluster cluster(o.page_bytes, target_dram, num_peers, caller_buf_bytes, caller_local_bytes);

  // Register source region (skip for all_stg to force staging fallback).
  if (o.scenario != "all_stg") {
    cluster.caller()->RegisterMemory(cluster.caller_buf().data(), caller_buf_bytes);
  }

  // Build batch.
  std::vector<std::string> keys(o.batch);
  std::vector<const void*> srcs(o.batch);
  std::vector<size_t> sizes(o.batch);
  for (size_t i = 0; i < o.batch; ++i) {
    char* slot = cluster.caller_buf().data() + i * o.per_item_pages * o.page_bytes;
    std::memset(slot, static_cast<int>(0x10 + (i & 0x7F)), o.per_item_pages * o.page_bytes);
    keys[i] = "bench-" + std::to_string(i);
    srcs[i] = slot;
    sizes[i] = o.per_item_pages * o.page_bytes;
  }

  // Warm-up (one iter, not measured) to avoid first-call overhead.
  RunOnce(cluster.caller(), keys, srcs, sizes);

  // Build a fresh batch for each iter (master expects unique keys).
  double total_ms = 0;
  size_t total_bytes = 0;
  for (size_t it = 0; it < o.iters; ++it) {
    for (size_t i = 0; i < o.batch; ++i) {
      keys[i] = "bench-" + std::to_string(it) + "-" + std::to_string(i);
    }
    auto [ms, bytes] = RunOnce(cluster.caller(), keys, srcs, sizes);
    total_ms += ms;
    total_bytes += bytes;
  }

  // Aggregate throughput: total_bytes / total_wall_time.  Equivalent to
  // a time-weighted mean of per-iter GiB/s; not biased toward fast iters.
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
