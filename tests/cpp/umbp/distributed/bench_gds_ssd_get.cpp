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

// GDS-vs-staging A/B for SSD reads.  Drives IUMBPClient::BatchGet against an
// SSD-only node and reports the two passes that actually differ:
//
//   cold  -- first touch of each key.  The staging path must do a device read
//            into the arena and then a second copy out to the caller; the GDS
//            path DMAs the range straight into the destination.  This is the
//            only pass where zero-copy can win.
//   warm  -- the same keys again, now holding a read lease.  BatchResolve
//            serves a lease before it ever considers a file ref, so BOTH modes
//            are an arena copy here and the numbers should converge.  Reported
//            because "GDS made no difference" is the expected answer for hot
//            data, and it is worth being able to show that rather than assert
//            it.
//
// The mode is not a flag: run the binary twice, once with UMBP_ENABLE_GDS=1
// and once without, and diff the CSV.  That keeps the two arms honest -- same
// binary, same node, same data, one environment variable apart.
//
// Usage:
//   bench_gds_ssd_get [--dir PATH] [--value-bytes N] [--batch N] [--keys N]
//                     [--capacity-mb N] [--page-bytes N] [--slots N]
//                     [--host-dst] [--io-backend posix|uring] [--no-header]
//
// --dir must name a real block-backed filesystem.  The SSD tier probes for
// O_DIRECT and silently falls back to buffered I/O when the probe fails, and
// a buffered fd is never eligible for the GDS fastpath (RecordLocation::
// direct_io gates it), so pointing this at tmpfs measures nothing.
//
// CSV (stdout):
//   mode,pass,dst,value_bytes,batch,keys,slots,wall_ms,gibps,us_per_key

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/umbp_client.h"

namespace fs = std::filesystem;

using mori::umbp::CreateUMBPClient;
using mori::umbp::IUMBPClient;
using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::UMBPConfig;
using mori::umbp::UMBPDistributedConfig;
using mori::umbp::UMBPIoBackend;
using mori::umbp::UMBPMedium;

namespace {

struct Options {
  std::string dir;
  size_t value_bytes = 1ULL << 20;  // 1 MiB
  size_t batch = 32;
  size_t keys = 512;
  size_t capacity_mb = 8192;
  size_t page_bytes = 64ULL * 1024;
  // Staging pages the SSD backend may hold in flight.  0 = size it from the
  // batch.  The arena is the staging arm's hard concurrency limit: an object of
  // N pages holds N entries until its read lease expires, and a batch that does
  // not fit resolves kBusy and backs off (up to 50 ms a try).  Undersizing it
  // measures the backoff, not the copy -- so size it, and expose the knob so
  // arena pressure can be studied deliberately rather than stumbled into.
  size_t slots = 0;
  bool host_dst = false;
  bool header = true;
  UMBPIoBackend io_backend = UMBPIoBackend::Posix;
};

[[noreturn]] void Die(const std::string& why) {
  std::fprintf(stderr, "bench_gds_ssd_get: %s\n", why.c_str());
  std::exit(2);
}

#define HIP_OK(expr)                                                              \
  do {                                                                            \
    hipError_t _e = (expr);                                                       \
    if (_e != hipSuccess) Die(std::string(#expr) + ": " + hipGetErrorString(_e)); \
  } while (0)

// A device buffer, or a host one when --host-dst asks for the comparison arm.
// Host is worth measuring precisely because it CANNOT use GDS: the file engine
// claims only (file, GPU), so a host destination re-resolves onto staged pages.
class Destination {
 public:
  Destination(size_t bytes, bool host) : bytes_(bytes), host_(host) {
    if (host_) {
      ptr_ = std::malloc(bytes_);
      if (ptr_ == nullptr) Die("malloc failed");
    } else {
      HIP_OK(hipMalloc(&ptr_, bytes_));
    }
  }
  ~Destination() {
    if (ptr_ == nullptr) return;
    if (host_) {
      std::free(ptr_);
    } else {
      (void)hipFree(ptr_);
    }
  }
  Destination(const Destination&) = delete;
  Destination& operator=(const Destination&) = delete;

  void* get() const { return ptr_; }
  bool host() const { return host_; }

 private:
  size_t bytes_ = 0;
  bool host_ = false;
  void* ptr_ = nullptr;
};

double Seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

// One pass over every key, in batches.  Returns wall seconds; fails loudly,
// because a benchmark that silently measures misses is worse than no number.
double RunPass(IUMBPClient* client, const std::vector<std::string>& keys, const Destination& dst,
               const Options& opt) {
  auto* base = static_cast<char*>(dst.get());
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t start = 0; start < keys.size(); start += opt.batch) {
    const size_t n = std::min(opt.batch, keys.size() - start);
    std::vector<std::string> batch_keys(keys.begin() + start, keys.begin() + start + n);
    std::vector<uintptr_t> dsts(n);
    std::vector<size_t> sizes(n, opt.value_bytes);
    for (size_t i = 0; i < n; ++i) {
      dsts[i] = reinterpret_cast<uintptr_t>(base + i * opt.value_bytes);
    }
    auto ok = client->BatchGet(batch_keys, dsts, sizes);
    for (size_t i = 0; i < n; ++i) {
      if (i >= ok.size() || !ok[i]) Die("BatchGet miss for key " + batch_keys[i]);
    }
  }
  return Seconds(t0, std::chrono::steady_clock::now());
}

void Report(const Options& opt, const char* pass, double sec) {
  const double bytes = static_cast<double>(opt.keys) * static_cast<double>(opt.value_bytes);
  const double gibps = bytes / (sec > 0 ? sec : 1e-12) / (1024.0 * 1024.0 * 1024.0);
  const double us_per_key = sec * 1e6 / static_cast<double>(opt.keys);
  const char* mode = std::getenv("UMBP_ENABLE_GDS");
  const bool on = mode != nullptr && (std::string(mode) == "1" || std::string(mode) == "true" ||
                                      std::string(mode) == "on");
  std::printf("%s,%s,%s,%zu,%zu,%zu,%zu,%.3f,%.3f,%.2f\n", on ? "gds" : "staging", pass,
              opt.host_dst ? "host" : "device", opt.value_bytes, opt.batch, opt.keys, opt.slots,
              sec * 1000.0, gibps, us_per_key);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Die("missing value for " + a);
      return argv[++i];
    };
    if (a == "--dir")
      opt.dir = next();
    else if (a == "--value-bytes")
      opt.value_bytes = std::stoull(next());
    else if (a == "--batch")
      opt.batch = std::stoull(next());
    else if (a == "--keys")
      opt.keys = std::stoull(next());
    else if (a == "--capacity-mb")
      opt.capacity_mb = std::stoull(next());
    else if (a == "--page-bytes")
      opt.page_bytes = std::stoull(next());
    else if (a == "--slots")
      opt.slots = std::stoull(next());
    else if (a == "--host-dst")
      opt.host_dst = true;
    else if (a == "--no-header")
      opt.header = false;
    else if (a == "--io-backend") {
      const std::string b = next();
      if (b == "uring" || b == "io_uring")
        opt.io_backend = UMBPIoBackend::IoUring;
      else if (b == "posix")
        opt.io_backend = UMBPIoBackend::Posix;
      else
        Die("--io-backend must be posix or uring");
    } else {
      Die("unknown flag: " + a);
    }
  }
  if (opt.dir.empty()) Die("--dir is required and must be on a real block device");
  if (opt.batch == 0 || opt.keys == 0 || opt.value_bytes == 0) Die("sizes must be non-zero");
  if (opt.page_bytes == 0) Die("--page-bytes must be non-zero");
  const size_t pages_per_value = (opt.value_bytes + opt.page_bytes - 1) / opt.page_bytes;
  // Default: the whole working set.  A staged page is held until its read
  // lease expires, not until the copy finishes, so an arena sized to one batch
  // fills after a couple of batches and every later resolve returns kBusy and
  // sleeps -- which measures the backoff, not the data path.  Pass --slots
  // explicitly to study that regime on purpose; it is a real one, and it is
  // where zero-copy helps most, because a file ref occupies no arena at all.
  if (opt.slots == 0) opt.slots = opt.keys * pages_per_value;

  const fs::path store = fs::path(opt.dir) / ("bench_gds_" + std::to_string(::getpid()));
  fs::remove_all(store);
  fs::create_directories(store);

  MasterServerConfig master_cfg;
  master_cfg.listen_address = "0.0.0.0:0";
  master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds(4);
  auto master = std::make_unique<MasterServer>(std::move(master_cfg));
  std::thread master_thread([&] { master->Run(); });
  for (int i = 0; i < 200 && master->GetBoundPort() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (master->GetBoundPort() == 0) Die("master failed to start");

  UMBPConfig config;
  config.dram.use_hugepages = false;
  config.dram.capacity_bytes = 8 << 20;  // ignored on an SSD node; see medium selection
  config.ssd.storage_dir = store.string();
  config.ssd.capacity_bytes = opt.capacity_mb * 1024ULL * 1024ULL;
  config.ssd.io.backend = opt.io_backend;

  UMBPDistributedConfig distributed;
  distributed.master_config.node_id = "bench-gds-ssd";
  distributed.master_config.node_address = "127.0.0.1";
  distributed.master_config.master_address = "localhost:" + std::to_string(master->GetBoundPort());
  distributed.io_engine.host = "0.0.0.0";
  distributed.io_engine.port = 0;
  distributed.peer_service_port = 0;
  distributed.medium = UMBPMedium::SSD;
  distributed.dram_page_size = opt.page_bytes;
  // ssd_staging_buffer_slots is what PoolClient hands SsdBackend as
  // Config::staging_pages -- the LOCAL arena, in pages, not the remote one.
  distributed.ssd_staging_buffer_slots = static_cast<int>(opt.slots);
  distributed.ssd_staging_buffer_size =
      std::max<size_t>(256ULL << 20, opt.batch * opt.value_bytes * 4);
  config.distributed = std::move(distributed);

  std::string err;
  if (!config.Validate(&err)) Die("config invalid: " + err);
  auto client = CreateUMBPClient(config);
  if (client == nullptr) Die("CreateUMBPClient returned null");

  // ---- populate -------------------------------------------------------------
  std::vector<char> payload(opt.value_bytes, 0x5A);
  std::vector<std::string> keys;
  keys.reserve(opt.keys);
  for (size_t i = 0; i < opt.keys; ++i) keys.push_back("gdsbench/" + std::to_string(i));
  for (size_t i = 0; i < opt.keys; ++i) {
    if (!client->Put(keys[i], reinterpret_cast<uintptr_t>(payload.data()), opt.value_bytes)) {
      Die("Put failed for " + keys[i]);
    }
  }

  Destination dst(opt.batch * opt.value_bytes, opt.host_dst);
  if (!opt.host_dst) {
    client->RegisterMemory(reinterpret_cast<uintptr_t>(dst.get()), opt.batch * opt.value_bytes,
                           mori::io::MemoryLocationType::GPU, 0);
  }

  if (opt.header) {
    std::printf("mode,pass,dst,value_bytes,batch,keys,slots,wall_ms,gibps,us_per_key\n");
  }
  Report(opt, "cold", RunPass(client.get(), keys, dst, opt));
  Report(opt, "warm", RunPass(client.get(), keys, dst, opt));

  // Correctness, after the timed passes so it costs them nothing.  The device
  // arm is the one that matters: it is the only path that can have come from a
  // hipFileRead, and nothing else in the tree checks those bytes end to end
  // through the public client API.
  {
    std::vector<char> back(opt.value_bytes, 0);
    if (opt.host_dst) {
      std::memcpy(back.data(), dst.get(), opt.value_bytes);
    } else {
      HIP_OK(hipMemcpy(back.data(), dst.get(), opt.value_bytes, hipMemcpyDeviceToHost));
    }
    if (std::memcmp(back.data(), payload.data(), opt.value_bytes) != 0) {
      Die("VERIFY FAILED: bytes read back do not match what was written");
    }
    std::fprintf(stderr, "verify ok (%zu bytes)\n", opt.value_bytes);
  }

  client->Close();
  master->Shutdown();
  if (master_thread.joinable()) master_thread.join();
  std::error_code ec;
  fs::remove_all(store, ec);
  return 0;
}
