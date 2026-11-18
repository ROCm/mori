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

// Hardware driver for UMBP over a UALink scale-up fabric: every combination of
// DRAM and HBM between two trays, with the fabric as the only transport.
//
// Run one master, one owner and then a reader or a writer, the owner and the
// other peer on two trays of one vPOD. It needs real hardware and services, so
// it is built but never registered as a CTest.
#include <hip/hip_runtime_api.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::io {
// The fabric allocator (src/io/fabric/backend_impl.cpp).
void* FabricMalloc(size_t size, int device);
void FabricFree(void* ptr);
}  // namespace mori::io

using namespace mori::umbp;
using Clock = std::chrono::steady_clock;

namespace {
volatile std::sig_atomic_t stopRequested = 0;
void Stop(int) { stopRequested = 1; }
void Require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
void Hip(hipError_t value, const char* operation) {
  Require(value == hipSuccess, std::string(operation) + ": " + hipGetErrorString(value));
}

struct Args {
  std::unordered_map<std::string, std::string> options;
  std::string Get(const std::string& name, const std::string& fallback = "") const {
    auto it = options.find(name);
    return it == options.end() ? fallback : it->second;
  }
  size_t Size(const std::string& name, size_t fallback) const {
    const auto text = Get(name);
    if (text.empty()) return fallback;
    size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    Require(consumed == text.size(), "invalid " + name);
    return static_cast<size_t>(value);
  }
  bool Flag(const std::string& name) const { return Size(name, 0) != 0; }
};

void Help() {
  std::cout << "umbp_ualink_direct_driver --role master|owner|reader|writer --master IP:PORT\n"
               "  master: --duration-sec 600\n"
               "  peers:  --advertise IP --node-id NAME --run-id NAME --kernel fabric_copy.hsaco\n"
               "          --device 0 --keys 32 --object-bytes 1048576 --page-bytes 65536\n"
               "  owner:  --medium hbm|dram --prefill 1 --capacity-sets 2 --duration-sec 600\n"
               "          holds the keys; --prefill 0 stores nothing and only receives writes\n"
               "  reader: --owner NAME --medium hbm|dram --memory hbm|fabric-hbm|host\n"
               "          --iterations 20 --ranged 1 --expect-owner-push 0\n"
               "          reads the owner's keys into GPU or registered host memory\n"
               "  writer: --owner NAME --medium hbm|dram --memory hbm|fabric-hbm|host\n"
               "          --iterations 4 --ranged 0 --expect-owner-pull 0\n"
               "          writes objects to the owner, then reads them back\n"
               "fabric-hbm is exportable GPU memory (the fabric allocator); hbm is hipMalloc.\n"
               "Compile the kernel with: hipcc --genco --offload-arch=<arch> -O3 \\\n"
               "    src/io/kernels/fabric_copy.hip -o fabric_copy.hsaco\n";
}

Args Parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      Help();
      std::exit(0);
    }
    Require(key.rfind("--", 0) == 0 && i + 1 < argc, "expected --option value");
    Require(args.options.emplace(key.substr(2), argv[++i]).second, "duplicate option " + key);
  }
  return args;
}

TierType Medium(const Args& args) {
  const auto value = args.Get("medium", "hbm");
  Require(value == "hbm" || value == "dram", "--medium must be hbm or dram");
  return value == "hbm" ? TierType::HBM : TierType::DRAM;
}

// GPU memory: plain hipMalloc, or exportable through the fabric allocator.
struct DeviceBuffer {
  void* pointer = nullptr;
  bool exportable = false;
  DeviceBuffer(size_t bytes, bool fabric, int device) : exportable(fabric) {
    if (exportable) {
      pointer = mori::io::FabricMalloc(bytes, device);
      Require(pointer != nullptr, "exportable GPU allocation");
    } else {
      Hip(hipMalloc(&pointer, bytes), "hipMalloc");
    }
  }
  ~DeviceBuffer() {
    if (pointer == nullptr) return;
    if (exportable)
      mori::io::FabricFree(pointer);
    else
      (void)hipFree(pointer);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Page-aligned anonymous host memory, the shape a caller's pinned buffer has.
struct HostBuffer {
  void* pointer = nullptr;
  size_t bytes = 0;
  explicit HostBuffer(size_t size) : bytes((size + kHuge - 1) / kHuge * kHuge) {
    pointer = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(pointer != MAP_FAILED, "host buffer allocation");
    (void)madvise(pointer, bytes, MADV_HUGEPAGE);
    std::memset(pointer, 0, bytes);
  }
  ~HostBuffer() { (void)munmap(pointer, bytes); }
  static constexpr size_t kHuge = size_t{2} << 20;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
};

// The reader's or writer's working memory, in the place --memory names.
struct PeerMemory {
  PeerMemory(const Args& args, size_t bytes)
      : place(args.Get("memory", "hbm")),
        host(place == "host"),
        device(host ? 4096 : bytes, place == "fabric-hbm",
               static_cast<int>(args.Size("device", 0))),
        memory(host ? bytes : 4096),
        pointer(host ? memory.pointer : device.pointer),
        size(bytes) {
    Require(place == "hbm" || place == "fabric-hbm" || place == "host",
            "--memory must be hbm, fabric-hbm or host");
  }
  void Register(PoolClient* client, int device_id) const {
    Require(client->RegisterMemory(
                pointer, size,
                host ? mori::io::MemoryLocationType::CPU : mori::io::MemoryLocationType::GPU,
                host ? -1 : device_id),
            "memory registration failed");
  }
  void Fill(const void* bytes) const {
    if (host)
      std::memcpy(pointer, bytes, size);
    else
      Hip(hipMemcpy(pointer, bytes, size, hipMemcpyHostToDevice), "fill");
  }
  void Set(int value) const {
    if (host)
      std::memset(pointer, value, size);
    else
      Hip(hipMemset(pointer, value, size), "memset");
  }
  void Read(void* bytes) const {
    if (host)
      std::memcpy(bytes, pointer, size);
    else
      Hip(hipMemcpy(bytes, pointer, size, hipMemcpyDeviceToHost), "download");
  }
  const std::string place;
  const bool host;
  DeviceBuffer device;
  HostBuffer memory;
  void* const pointer;
  const size_t size;
};

unsigned char Byte(uint64_t seed, size_t key, size_t offset) {
  return static_cast<unsigned char>((seed + key * 31 + offset * 13 + (offset >> 8) * 7) % 251);
}
uint64_t Seed(const std::string& run) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : run) hash = (hash ^ byte) * 1099511628211ULL;
  return hash;
}

struct Geometry {
  size_t keys, object, page, total;
  // Three ranges that tile an object, cut off page boundaries.
  std::vector<size_t> cuts;
  explicit Geometry(const Args& args)
      : keys(args.Size("keys", 32)),
        object(args.Size("object-bytes", 1048576)),
        page(args.Size("page-bytes", 65536)),
        total(keys * object) {
    Require(keys > 0 && page >= 256 && object > 4 * page, "object must exceed four pages");
    cuts = {0, page + 48, 3 * page + 16, object};
  }
  std::vector<std::string> KeyNames(const std::string& run) const {
    std::vector<std::string> out;
    for (size_t key = 0; key < keys; ++key)
      out.push_back("ualink/" + run + "/" + std::to_string(key));
    return out;
  }
};

PoolClientConfig PeerConfig(const Args& args, const Geometry& geometry, bool owner) {
  PoolClientConfig config;
  config.master_config.master_address = args.Get("master");
  config.master_config.node_id = args.Get("node-id", owner ? "owner" : "peer");
  config.master_config.node_address = args.Get("advertise", "127.0.0.1");
  config.io_engine.mode = UMBPTransferMode::kFabricStrict;
  config.io_engine.fabric_kernel_path = args.Get("kernel");
  config.io_engine.fabric_host_device = static_cast<int>(args.Size("device", 0));
  // The owner stores in --medium. The other peer keeps a token HBM pool: it
  // holds none of the keys, so every read and write crosses the fabric.
  config.medium = owner ? Medium(args) : TierType::HBM;
  config.hbm.device = static_cast<int>(args.Size("device", 0));
  const size_t pages = geometry.object / geometry.page + (geometry.object % geometry.page != 0);
  const size_t storage = geometry.keys * pages * geometry.page * args.Size("capacity-sets", 2);
  config.hbm.buffer_sizes = {owner ? storage : 2 * geometry.page};
  if (owner && config.medium == TierType::DRAM) config.dram.buffer_sizes = {storage};
  config.dram_page_size = geometry.page;
  config.cache_remote_fetches = false;
  config.ranged_locality_prefetch = false;
  config.prefetch_budget_bytes = 0;
  config.eviction_mode = PeerEvictionMode::kMaster;
  config.auto_peer_service_port = true;
  return config;
}

void Hold(size_t seconds) {
  const auto deadline = Clock::now() + std::chrono::seconds(seconds);
  while (!stopRequested && Clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void Master(const Args& args) {
  MasterServerConfig config;
  config.listen_address = args.Get("master");
  MasterServer master(std::move(config));
  std::thread server([&] { master.Run(); });
  for (size_t i = 0; i < 200 && master.GetBoundPort() == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (!master.GetBoundPort()) {
    master.Shutdown();
    server.join();
    throw std::runtime_error("master failed to bind");
  }
  std::cout << "READY role=master port=" << master.GetBoundPort() << std::endl;
  Hold(args.Size("duration-sec", 600));
  master.Shutdown();
  server.join();
}

// Blocks until the master routes every key to `owner`'s `medium`.
void AwaitPlacement(MasterClient* lookup, const std::vector<std::string>& keys,
                    const std::string& owner, TierType medium, size_t object, size_t seconds) {
  const auto deadline = Clock::now() + std::chrono::seconds(seconds);
  std::vector<std::optional<RouteGetResult>> routes;
  while (true) {
    const auto status = lookup->BatchRouteGet(keys, {}, &routes);
    if (status.ok() && routes.size() == keys.size() &&
        std::all_of(routes.begin(), routes.end(), [](const auto& r) { return r.has_value(); }))
      break;
    Require(!stopRequested && Clock::now() < deadline, "keys did not become visible");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  for (const auto& route : routes)
    Require(route->node_id == owner && route->tier == medium && route->size == object,
            "key placement/size mismatch");
}

void Owner(const Args& args, const Geometry& geometry) {
  auto config = PeerConfig(args, geometry, true);
  PoolClient client(config);
  Require(client.Init(), "owner initialization failed (fabric_strict)");
  const bool prefill = args.Size("prefill", 1) != 0;
  if (prefill) {
    // Stored from host memory through this node's own GPU, like any local put.
    const auto keys = geometry.KeyNames(args.Get("run-id"));
    std::vector<unsigned char> bytes(geometry.total);
    const uint64_t seed = Seed(args.Get("run-id"));
    std::vector<const void*> pointers;
    for (size_t key = 0; key < geometry.keys; ++key) {
      for (size_t i = 0; i < geometry.object; ++i)
        bytes[key * geometry.object + i] = Byte(seed, key, i);
      pointers.push_back(bytes.data() + key * geometry.object);
    }
    const auto stored =
        client.BatchPut(keys, pointers, std::vector<size_t>(geometry.keys, geometry.object));
    Require(std::all_of(stored.begin(), stored.end(), [](bool v) { return v; }),
            "owner BatchPut failed");
    auto* storage = client.Backends().Get(config.medium);
    Require(storage != nullptr, "owner storage backend missing");
    for (const auto& key : keys)
      Require(storage->Contains(key), "owner key absent from its storage tier");
  }
  std::cout << "READY role=owner node=" << client.NodeId()
            << " medium=" << args.Get("medium", "hbm") << " keys=" << (prefill ? geometry.keys : 0)
            << " object_bytes=" << geometry.object << " transport=fabric_strict" << std::endl;
  Hold(args.Size("duration-sec", 600));
  client.Shutdown();
}

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  const size_t rank = static_cast<size_t>(std::ceil(percentile * values.size()));
  return values[std::min(values.size() - 1, rank > 0 ? rank - 1 : size_t{0})];
}

void Summary(const char* op, std::vector<double> seconds, size_t bytes, const std::string& tail) {
  Require(!seconds.empty(), "nothing completed");
  // The first call also pays connection setup.
  if (seconds.size() > 1) seconds.erase(seconds.begin());
  double total = 0;
  for (double value : seconds) total += value;
  std::cout << "SUMMARY op=" << op << " samples=" << seconds.size() << " bytes_per_call=" << bytes
            << " p50_us=" << Percentile(seconds, .50) * 1e6
            << " p95_us=" << Percentile(seconds, .95) * 1e6
            << " GBps=" << static_cast<double>(bytes) * seconds.size() / total / 1e9 << " " << tail
            << " transport=fabric_strict" << std::endl;
}

// Reads the owner's keys into this tray's GPU or host memory.
void Reader(const Args& args, const Geometry& geometry) {
  const auto config = PeerConfig(args, geometry, false);
  const auto keys = geometry.KeyNames(args.Get("run-id"));
  const uint64_t seed = Seed(args.Get("run-id"));
  const bool ranged = args.Size("ranged", 1) != 0;
  const size_t iterations = args.Size("iterations", 20);
  PeerMemory memory(args, geometry.total);
  PoolClient client(config);
  Require(client.Init(), "reader initialization failed (fabric_strict)");
  memory.Register(&client, config.hbm.device);
  MasterClient lookup(config.master_config);
  AwaitPlacement(&lookup, keys, args.Get("owner", "owner"), Medium(args), geometry.object,
                 args.Size("timeout-sec", 60));

  // Ranged: two of an object's three ranges, landed where they sit in the
  // object. The bytes between them must come back untouched.
  std::vector<std::vector<void*>> dsts(geometry.keys);
  std::vector<std::vector<size_t>> lengths(geometry.keys), offsets(geometry.keys);
  std::vector<void*> whole;
  size_t requested = 0;
  for (size_t key = 0; key < geometry.keys; ++key) {
    auto* slot = static_cast<char*>(memory.pointer) + key * geometry.object;
    whole.push_back(slot);
    for (size_t r : {size_t{0}, size_t{2}}) {
      dsts[key].push_back(slot + geometry.cuts[r]);
      lengths[key].push_back(geometry.cuts[r + 1] - geometry.cuts[r]);
      offsets[key].push_back(geometry.cuts[r]);
      requested += lengths[key].back();
    }
  }
  if (!ranged) requested = geometry.total;
  std::vector<unsigned char> actual(geometry.total);
  std::vector<double> latencies;
  for (size_t iteration = 0; iteration < iterations && !stopRequested; ++iteration) {
    memory.Set(0xD3);
    const uint64_t pushed_before = client.OwnerPushBytes();
    const auto begin = Clock::now();
    const auto fetched =
        ranged ? client.BatchGetRanges(keys, dsts, lengths, offsets)
               : client.BatchGet(keys, whole, std::vector<size_t>(geometry.keys, geometry.object));
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    Require(std::all_of(fetched.begin(), fetched.end(), [](bool v) { return v; }),
            "read failed (fabric_strict has no other transport)");
    const uint64_t pushed = client.OwnerPushBytes() - pushed_before;
    if (args.Flag("expect-owner-push"))
      Require(pushed == requested, "read was not pushed by the owner in one copy");
    memory.Read(actual.data());
    for (size_t key = 0; key < geometry.keys; ++key)
      for (size_t i = 0; i < geometry.object; ++i) {
        const bool wanted = !ranged || i < geometry.cuts[1] || i >= geometry.cuts[2];
        const unsigned char expected = wanted ? Byte(seed, key, i) : 0xD3;
        if (actual[key * geometry.object + i] != expected)
          throw std::runtime_error("byte mismatch key=" + std::to_string(key) +
                                   " offset=" + std::to_string(i));
      }
    latencies.push_back(seconds);
    std::cout << "SAMPLE op=get iteration=" << iteration << " bytes=" << requested
              << " latency_us=" << seconds * 1e6 << " owner_push_bytes=" << pushed << std::endl;
  }
  Summary("get", latencies, requested,
          "owner_medium=" + args.Get("medium", "hbm") + " memory=" + memory.place +
              " shape=" + (ranged ? "ranged" : "whole"));
  client.DeregisterMemory(memory.pointer);
  client.Shutdown();
  std::cout << "PASS reader placement, bytes and untouched regions" << std::endl;
}

// Writes objects into the owner's tier from this tray's GPU or host memory,
// then reads them back from the owner and compares every byte.
void Writer(const Args& args, const Geometry& geometry) {
  const auto config = PeerConfig(args, geometry, false);
  const auto owner = args.Get("owner", "owner");
  const uint64_t seed = Seed(args.Get("run-id"));
  const bool ranged = args.Size("ranged", 0) != 0;
  const size_t iterations = args.Size("iterations", 4);
  PeerMemory memory(args, geometry.total);
  DeviceBuffer readback(geometry.total, /*fabric=*/true, config.hbm.device);
  PoolClient client(config);
  Require(client.Init(), "writer initialization failed (fabric_strict)");
  memory.Register(&client, config.hbm.device);
  Require(client.RegisterMemory(readback.pointer, geometry.total, mori::io::MemoryLocationType::GPU,
                                config.hbm.device),
          "read-back registration failed");
  MasterClient lookup(config.master_config);
  std::vector<unsigned char> expected(geometry.total), laid(geometry.total), actual(geometry.total);
  const std::vector<size_t> sizes(geometry.keys, geometry.object);
  std::vector<double> latencies;
  for (size_t iteration = 0; iteration < iterations && !stopRequested; ++iteration) {
    const auto keys = geometry.KeyNames(args.Get("run-id") + "/w" + std::to_string(iteration));
    std::vector<const void*> pointers;
    std::vector<std::vector<const void*>> rangeSources(geometry.keys);
    std::vector<std::vector<size_t>> rangeSizes(geometry.keys), rangeOffsets(geometry.keys);
    for (size_t key = 0; key < geometry.keys; ++key) {
      auto* object = expected.data() + key * geometry.object;
      for (size_t i = 0; i < geometry.object; ++i) object[i] = Byte(seed + iteration, key, i);
      auto* slot = static_cast<char*>(memory.pointer) + key * geometry.object;
      pointers.push_back(slot);
      if (!ranged) {
        std::memcpy(laid.data() + key * geometry.object, object, geometry.object);
        continue;
      }
      // The object's three ranges sit in the source back to front, as a KV
      // pool scatters them.
      size_t at = geometry.object;
      for (size_t r = 0; r + 1 < geometry.cuts.size(); ++r) {
        const size_t length = geometry.cuts[r + 1] - geometry.cuts[r];
        at -= length;
        std::memcpy(laid.data() + key * geometry.object + at, object + geometry.cuts[r], length);
        rangeSources[key].push_back(slot + at);
        rangeSizes[key].push_back(length);
        rangeOffsets[key].push_back(geometry.cuts[r]);
      }
    }
    memory.Fill(laid.data());
    const uint64_t pulled_before = client.OwnerPullBytes();
    // The first write races the owner's registration with the master.
    const auto deadline = Clock::now() + std::chrono::seconds(args.Size("timeout-sec", 60));
    double seconds = 0;
    while (true) {
      const auto begin = Clock::now();
      const auto stored =
          ranged ? client.BatchPutRanges(keys, sizes, rangeSources, rangeSizes, rangeOffsets)
                 : client.BatchPut(keys, pointers, sizes);
      seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      if (std::all_of(stored.begin(), stored.end(), [](bool v) { return v; })) break;
      Require(iteration == 0 && !stopRequested && Clock::now() < deadline,
              "write failed (fabric_strict has no other transport)");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    const uint64_t pulled = client.OwnerPullBytes() - pulled_before;
    if (args.Flag("expect-owner-pull"))
      Require(pulled == geometry.total, "write was not read by the owner in one copy");
    AwaitPlacement(&lookup, keys, owner, Medium(args), geometry.object, 30);
    // Retire the source so the read-back can only come from the owner.
    memory.Set(0xA7);
    Hip(hipMemset(readback.pointer, 0xD3, geometry.total), "read-back sentinel");
    std::vector<void*> dsts;
    for (size_t key = 0; key < geometry.keys; ++key)
      dsts.push_back(static_cast<char*>(readback.pointer) + key * geometry.object);
    const auto fetched = client.BatchGet(keys, dsts, sizes);
    Require(std::all_of(fetched.begin(), fetched.end(), [](bool v) { return v; }),
            "read-back of written keys failed");
    Hip(hipMemcpy(actual.data(), readback.pointer, geometry.total, hipMemcpyDeviceToHost),
        "read-back download");
    for (size_t i = 0; i < geometry.total; ++i)
      if (actual[i] != expected[i])
        throw std::runtime_error("read-back mismatch key=" + std::to_string(i / geometry.object) +
                                 " offset=" + std::to_string(i % geometry.object));
    latencies.push_back(seconds);
    std::cout << "SAMPLE op=put iteration=" << iteration << " bytes=" << geometry.total
              << " latency_us=" << seconds * 1e6 << " owner_pull_bytes=" << pulled << std::endl;
  }
  Summary("put", latencies, geometry.total,
          "owner_medium=" + args.Get("medium", "hbm") + " memory=" + memory.place +
              " shape=" + (ranged ? "ranged" : "whole"));
  client.DeregisterMemory(memory.pointer);
  client.DeregisterMemory(readback.pointer);
  client.Shutdown();
  std::cout << "PASS writer placement and read-back bytes" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = Parse(argc, argv);
    Require(!args.Get("master").empty(), "--master is required");
    std::signal(SIGINT, Stop);
    std::signal(SIGTERM, Stop);
    std::cout << std::setprecision(9);
    const auto role = args.Get("role");
    if (role == "master") {
      Master(args);
      return 0;
    }
    Require(role == "owner" || role == "reader" || role == "writer",
            "--role must be master, owner, reader or writer");
    Require(!args.Get("kernel").empty() && !args.Get("run-id").empty(),
            "--kernel and --run-id are required");
    Hip(hipSetDevice(static_cast<int>(args.Size("device", 0))), "select GPU");
    const Geometry geometry(args);
    if (role == "owner")
      Owner(args, geometry);
    else if (role == "reader")
      Reader(args, geometry);
    else
      Writer(args, geometry);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
