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

// What a layer-wise restore costs on the standalone-process wire.
//
// The GPU end-to-end harness can only show this diluted: a 256K restore is a
// few hundred milliseconds of which the ranged RPCs are a slice, under a load
// that drifts by more than the slice between runs. This isolates the slice.
//
// The shape is the one that matters. A reader asks about one key set once per
// layer group, changing only which bytes of each object it wants -- so across
// the groups the keys are identical and everything else differs. Keys are 128
// bytes because that is what a real deployment's page hashes are, and the cost
// this measures scales with key length: a thousand of them is over a hundred
// kilobytes of protobuf strings per call, allocated once per key on each side.
//
// Runs entirely on the host over a unix socket. No GPU, no RDMA, no hugepages,
// so it can be run anywhere -- which is the point, since the nodes that have
// those are the scarce thing.
//
// --threads is the dimension that matters most and is easiest to miss: in a
// real deployment every rank on the node reaches one standalone server, so the
// per-call work inside the backend's mutex is paid once per rank, serially. A
// single-threaded measurement cannot see that, and a change that adds work to
// that critical section looks free right up until it is deployed.
//
// Usage: bench_standalone_ranged_wire [--keys N] [--groups N] [--repeats N]
//                                     [--key-bytes N] [--object-bytes N]
//                                     [--threads N]

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"

namespace {

using namespace mori::umbp;

int64_t ArgOr(int argc, char** argv, const char* name, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::strtoll(argv[i + 1], nullptr, 10);
  }
  return fallback;
}

// Hex, like the page hashes a connector actually produces, and distinct in the
// first bytes as well as the last so a comparison cannot shortcut.
std::string MakeKey(size_t index, size_t bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string key(bytes, '0');
  uint64_t mix = index * 0x9e3779b97f4a7c15ULL + 0x1234567ULL;
  for (size_t i = 0; i < bytes; ++i) {
    key[i] = kHex[(mix >> ((i % 16) * 4)) & 0xf];
    if (i % 16 == 15) mix = mix * 6364136223846793005ULL + 1442695040888963407ULL;
  }
  return key;
}

double MedianOf(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const size_t keys_n = static_cast<size_t>(ArgOr(argc, argv, "--keys", 512));
  const size_t groups = static_cast<size_t>(ArgOr(argc, argv, "--groups", 8));
  const size_t repeats = static_cast<size_t>(ArgOr(argc, argv, "--repeats", 20));
  const size_t key_bytes = static_cast<size_t>(ArgOr(argc, argv, "--key-bytes", 128));
  const size_t object_bytes = static_cast<size_t>(ArgOr(argc, argv, "--object-bytes", 8192));
  const size_t threads_n =
      static_cast<size_t>(std::max<int64_t>(1, ArgOr(argc, argv, "--threads", 1)));

  const std::string address =
      "unix:///tmp/umbp_wire_bench_" + std::to_string(getpid()) + ".grpc.sock";
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = static_cast<size_t>(4) << 30;
  cfg.ssd.enabled = false;
  UMBPStandaloneProcessConfig sp;
  sp.address = address;
  sp.startup_timeout_ms = 10000;
  cfg.standalone_process = sp;

  standalone::StandaloneServer server(cfg, address);
  if (!server.Start()) {
    std::fprintf(stderr, "server failed to start\n");
    return 1;
  }
  std::thread server_thread([&] { server.Run(); });

  // One client per thread, each with its own keys and its own slice of one
  // registered region: the shape of N ranks against one standalone server.
  const size_t slice = object_bytes / groups;
  const size_t per_thread_bytes = keys_n * object_bytes * 2;
  const size_t region_bytes = threads_n * per_thread_bytes + (1 << 20);
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = true;
  HostBufferHandle region = allocator.Alloc(region_bytes, opts);
  if (!region.valid()) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }
  auto* bytes = static_cast<unsigned char*>(region.ptr);

  std::vector<std::unique_ptr<IUMBPClient>> clients;
  std::vector<std::vector<std::string>> thread_keys(threads_n);
  for (size_t t = 0; t < threads_n; ++t) {
    UMBPConfig client_cfg = cfg;
    auto client = CreateUMBPClient(client_cfg);
    if (!client->RegisterMemory(reinterpret_cast<uintptr_t>(region.ptr), region.mapped_size)) {
      std::fprintf(stderr, "registration failed for thread %zu\n", t);
      return 1;
    }
    unsigned char* base = bytes + t * per_thread_bytes;
    std::vector<uintptr_t> srcs;
    std::vector<size_t> put_sizes;
    for (size_t k = 0; k < keys_n; ++k) {
      thread_keys[t].push_back(MakeKey(t * keys_n + k, key_bytes));
      unsigned char* src = base + k * object_bytes;
      std::memset(src, static_cast<int>(k & 0xff), object_bytes);
      srcs.push_back(reinterpret_cast<uintptr_t>(src));
      put_sizes.push_back(object_bytes);
    }
    const auto put_ok = client->BatchPut(thread_keys[t], srcs, put_sizes);
    if (std::count(put_ok.begin(), put_ok.end(), true) != static_cast<long>(keys_n)) {
      std::fprintf(stderr, "put failed for thread %zu\n", t);
      return 1;
    }
    clients.push_back(std::move(client));
  }

  // Every thread's restore is timed separately and the medians pooled, so the
  // number reported is what one rank waited, not how long the slowest took.
  std::vector<std::vector<double>> per_thread_ms(threads_n);
  std::atomic<bool> failed{false};
  const auto run_thread = [&](size_t t) {
    unsigned char* read_base = bytes + t * per_thread_bytes + keys_n * object_bytes;
    std::vector<std::vector<uintptr_t>> dsts(keys_n);
    std::vector<std::vector<size_t>> sizes(keys_n, {slice});
    for (size_t k = 0; k < keys_n; ++k) {
      dsts[k] = {reinterpret_cast<uintptr_t>(read_base + k * object_bytes)};
    }
    for (size_t r = 0; r < repeats + 2 && !failed.load(); ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t g = 0; g < groups; ++g) {
        std::vector<std::vector<size_t>> offsets(keys_n, {g * slice});
        const auto ok = clients[t]->BatchGetRanges(thread_keys[t], dsts, sizes, offsets);
        if (std::count(ok.begin(), ok.end(), true) != static_cast<long>(keys_n)) {
          std::fprintf(stderr, "get failed on thread %zu repeat %zu group %zu\n", t, r, g);
          failed.store(true);
          return;
        }
      }
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      // First two are warm-up: the first mints a handle, the second settles.
      if (r >= 2) per_thread_ms[t].push_back(ms);
    }
  };

  std::vector<std::thread> workers;
  for (size_t t = 1; t < threads_n; ++t) workers.emplace_back(run_thread, t);
  run_thread(0);
  for (auto& w : workers) w.join();
  if (failed.load()) return 1;

  std::vector<double> pooled;
  for (const auto& v : per_thread_ms) pooled.insert(pooled.end(), v.begin(), v.end());

  std::printf(
      "keys,threads,groups,key_bytes,object_bytes,key_mib_per_call,restore_ms_p50,call_us_p50\n");
  std::printf("%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,%.1f\n", keys_n, threads_n, groups, key_bytes,
              object_bytes, static_cast<double>(keys_n * key_bytes) / (1024.0 * 1024.0),
              MedianOf(pooled), MedianOf(pooled) * 1000.0 / static_cast<double>(groups));

  for (auto& c : clients) c->Close();
  allocator.Free(region);
  server.Shutdown();
  server_thread.join();
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  return 0;
}
