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
// Three knobs exist because the default shape is NOT the deployed one, and the
// difference is the difference between a mechanism that always works here and
// one that may never work there:
//
//   --chunk   A reader does not put a whole pool's keys in one call; it splits
//             them to fit a per-call range budget and walks the pieces in
//             order, once per layer group. That turns one repeated key set
//             into C of them cycled in a fixed order, which is the access
//             pattern LRU handles worst. Default 0 (one set) reproduces the
//             historical numbers; set it to model the reader.
//   --ranges-per-key
//             A layer group asks for one range per layer, not one per key.
//             Since a handle saves KEY bytes only, and ranges are payload it
//             cannot touch, this sets how large a share of the request the
//             handle can remove at all. Default 1 flatters it the most.
//   --threads Concurrent independent restores, each with its own key sets, as
//             separate ranks reading the same server. Threads are started once
//             and time their own work, so no barrier or thread creation lands
//             inside a measurement.
//
// Runs entirely on the host over a unix socket. No GPU, no RDMA, no hugepages,
// so it can be run anywhere -- which is the point, since the nodes that have
// those are the scarce thing.
//
// Usage: bench_standalone_ranged_wire [--keys N] [--groups N] [--repeats N]
//                                     [--key-bytes N] [--object-bytes N]
//                                     [--chunk N] [--ranges-per-key N]
//                                     [--threads N]

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
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
  const size_t ranges_per_key = static_cast<size_t>(ArgOr(argc, argv, "--ranges-per-key", 1));
  const size_t threads_n = static_cast<size_t>(ArgOr(argc, argv, "--threads", 1));
  // 0 means the whole set in one call, which is what this bench used to do.
  size_t chunk = static_cast<size_t>(ArgOr(argc, argv, "--chunk", 0));
  if (chunk == 0 || chunk > keys_n) chunk = keys_n;
  const size_t chunks_n = (keys_n + chunk - 1) / chunk;

  // Every range is one slice, and a group takes a distinct set of them, so a
  // restore reads each object exactly once across all groups.
  const size_t slices = groups * ranges_per_key;
  if (slices == 0 || object_bytes / slices == 0 || threads_n == 0) {
    std::fprintf(stderr, "need object-bytes >= groups*ranges-per-key and threads >= 1\n");
    return 1;
  }

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

  UMBPConfig client_cfg = cfg;
  auto client = CreateUMBPClient(client_cfg);

  // One registered region holding both the objects written and the buffers read
  // back into, so every range resolves through the registration table the way a
  // worker's KV pool does.
  const size_t slice = object_bytes / slices;
  const size_t region_bytes = threads_n * keys_n * object_bytes * 2 + (1 << 20);
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = true;
  HostBufferHandle region = allocator.Alloc(region_bytes, opts);
  if (!region.valid() ||
      !client->RegisterMemory(reinterpret_cast<uintptr_t>(region.ptr), region.mapped_size)) {
    std::fprintf(stderr, "registration failed\n");
    return 1;
  }
  auto* bytes = static_cast<unsigned char*>(region.ptr);

  // One chunk is one call. Each is built once, because what is being measured
  // is what the call costs on the wire, not what it costs to describe it.
  struct Chunk {
    std::vector<std::string> keys;
    std::vector<std::vector<uintptr_t>> dsts;
    std::vector<std::vector<size_t>> sizes;
    // Indexed by group: the slices that group reads of each object.
    std::vector<std::vector<std::vector<size_t>>> offsets;
  };

  // A worker stands for one rank: its own keys, so concurrent restores are
  // independent the way separate ranks are, and its own buffers.
  std::vector<std::vector<Chunk>> workers(threads_n);
  for (size_t t = 0; t < threads_n; ++t) {
    unsigned char* object_base = bytes + t * keys_n * object_bytes * 2;
    unsigned char* read_base = object_base + keys_n * object_bytes;

    std::vector<std::string> keys;
    std::vector<uintptr_t> srcs;
    std::vector<size_t> put_sizes;
    keys.reserve(keys_n);
    for (size_t k = 0; k < keys_n; ++k) {
      keys.push_back(MakeKey(t * keys_n + k, key_bytes));
      unsigned char* src = object_base + k * object_bytes;
      std::memset(src, static_cast<int>(k & 0xff), object_bytes);
      srcs.push_back(reinterpret_cast<uintptr_t>(src));
      put_sizes.push_back(object_bytes);
    }
    const auto put_ok = client->BatchPut(keys, srcs, put_sizes);
    if (std::count(put_ok.begin(), put_ok.end(), true) != static_cast<long>(keys_n)) {
      std::fprintf(stderr, "put failed\n");
      return 1;
    }

    for (size_t start = 0; start < keys_n; start += chunk) {
      const size_t end = std::min(start + chunk, keys_n);
      Chunk piece;
      piece.keys.assign(keys.begin() + static_cast<long>(start),
                        keys.begin() + static_cast<long>(end));
      piece.sizes.assign(end - start, std::vector<size_t>(ranges_per_key, slice));
      piece.dsts.resize(end - start);
      for (size_t k = start; k < end; ++k) {
        auto& dst = piece.dsts[k - start];
        dst.reserve(ranges_per_key);
        for (size_t j = 0; j < ranges_per_key; ++j) {
          dst.push_back(reinterpret_cast<uintptr_t>(read_base + k * object_bytes + j * slice));
        }
      }
      piece.offsets.resize(groups);
      for (size_t g = 0; g < groups; ++g) {
        std::vector<size_t> per_key(ranges_per_key);
        for (size_t j = 0; j < ranges_per_key; ++j) per_key[j] = (g * ranges_per_key + j) * slice;
        piece.offsets[g].assign(end - start, per_key);
      }
      workers[t].push_back(std::move(piece));
    }
  }

  // Threads are started once and time their own restores: a barrier or a
  // thread creation inside the measurement would be a large share of a
  // millisecond-scale result.
  std::vector<std::vector<double>> thread_ms(threads_n);
  std::atomic<bool> failed{false};
  std::vector<std::thread> workers_running;
  workers_running.reserve(threads_n);
  for (size_t t = 0; t < threads_n; ++t) {
    workers_running.emplace_back([&, t] {
      for (size_t r = 0; r < repeats + 2 && !failed.load(); ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        // The reader's order: every chunk once per group, groups outermost, so
        // a set is revisited only after every other set has been.
        for (size_t g = 0; g < groups; ++g) {
          for (const Chunk& piece : workers[t]) {
            const auto ok =
                client->BatchGetRanges(piece.keys, piece.dsts, piece.sizes, piece.offsets[g]);
            if (std::count(ok.begin(), ok.end(), true) != static_cast<long>(piece.keys.size())) {
              std::fprintf(stderr, "get failed at repeat %zu group %zu\n", r, g);
              failed.store(true);
              return;
            }
          }
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        if (r >= 2) thread_ms[t].push_back(ms);  // first two warm up: one mints, one settles
      }
    });
  }
  for (auto& worker : workers_running) worker.join();
  if (failed.load()) return 1;

  std::vector<double> per_restore_ms;
  std::vector<double> per_call_us;
  for (const auto& samples : thread_ms) {
    for (double ms : samples) {
      per_restore_ms.push_back(ms);
      per_call_us.push_back(ms * 1000.0 / static_cast<double>(groups * chunks_n));
    }
  }

  std::printf(
      "keys,groups,chunk,chunks,ranges_per_key,threads,key_bytes,object_bytes,"
      "key_kib_per_call,restore_ms_p50,call_us_p50\n");
  std::printf("%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.1f,%.3f,%.1f\n", keys_n, groups, chunk, chunks_n,
              ranges_per_key, threads_n, key_bytes, object_bytes,
              static_cast<double>(chunk * key_bytes) / 1024.0, MedianOf(per_restore_ms),
              MedianOf(per_call_us));

  client->Close();
  allocator.Free(region);
  server.Shutdown();
  server_thread.join();
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  return 0;
}
