// Standalone hardware driver. Run one master, one producer and one consumer in
// separate processes (normally the producer/consumer are on different trays).
// It deliberately never registers a default CTest that needs these services.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/pool_platform.h"

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
    Require(text.front() != '-', "negative " + name);
    size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    Require(consumed == text.size() && value <= std::numeric_limits<size_t>::max(),
            "invalid " + name);
    return static_cast<size_t>(value);
  }
  bool Boolean(const std::string& name, bool fallback) const {
    const auto value = Get(name, fallback ? "1" : "0");
    Require(value == "0" || value == "1", "expected 0 or 1 for " + name);
    return value == "1";
  }
};

void Help() {
  std::cout << "umbp_ualink_kv_driver --role master|producer|consumer --master IP:PORT\n"
               "  master: --duration-sec 600\n"
               "  peers: --advertise IP --run-id NAME --kernel COPY.hsaco --device 0\n"
               "  common geometry: --keys 32 --object-bytes 1048704 --page-bytes 65536\n"
               "                   --layout unaligned|aligned (default unaligned)\n"
               "  producer: --node-id kv-owner --duration-sec 600\n"
               "  consumer: --node-id kv-reader --expected-owner kv-owner --iterations 100\n"
               "            --cold-every 0 --verify-every 1 --timeout-sec 60 --async 1\n"
               "Integers are decimal bytes/counts. Both peers must use the same geometry/run id.\n";
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

size_t Multiply(size_t left, size_t right) {
  Require(right == 0 || left <= std::numeric_limits<size_t>::max() / right,
          "requested allocation size overflow");
  return left * right;
}
size_t Add(size_t left, size_t right) {
  Require(left <= std::numeric_limits<size_t>::max() - right, "requested allocation size overflow");
  return left + right;
}

int Device(const Args& args) {
  const auto device = args.Size("device", 0);
  Require(device <= static_cast<size_t>(std::numeric_limits<int>::max()), "invalid device ordinal");
  return static_cast<int>(device);
}

struct DeviceBuffer {
  void* pointer = nullptr;
  explicit DeviceBuffer(size_t bytes) { Hip(hipMalloc(&pointer, bytes), "hipMalloc"); }
  ~DeviceBuffer() {
    if (pointer) (void)hipFree(pointer);
  }
  // A failed GPU fence gives no permission to release memory it may still use.
  void Quarantine() { pointer = nullptr; }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct ConsumerStream {
  hipStream_t stream = nullptr;
  ConsumerStream() {
    Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "consumer stream");
  }
  ~ConsumerStream() {
    if (stream) (void)hipStreamDestroy(stream);
  }
  void Quarantine() { stream = nullptr; }
};

// Observability belongs to this hardware driver: production clients keep their
// normal platform, and no test getter is added to their public interface.
class ObservedPlatform final : public PoolPlatform {
 public:
  explicit ObservedPlatform(const UMBPSimConfig& sim) : inner_(MakePoolPlatform(sim)) {}
  Engines CreateTransferEngine(const PoolClientConfig& config,
                               const std::vector<BackendInstanceConfig>& backends) override {
    auto engines = inner_->CreateTransferEngine(config, backends);
    transfer_ = engines.transfer.get();
    return engines;
  }
  std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& backend,
                                               uint64_t pageSize,
                                               const PoolClientConfig& config) override {
    return inner_->CreateBackend(backend, pageSize, config);
  }
  TransferRef LocalUserBytes(void* ptr, size_t size, const TransferRef& ref) const override {
    return inner_->LocalUserBytes(ptr, size, ref);
  }
  HostBufferHandle AllocateHost(size_t size, const HostBufferOptions& options = {}) override {
    return inner_->AllocateHost(size, options);
  }
  void FreeHost(HostBufferHandle& buffer) override { inner_->FreeHost(buffer); }
  void AddHostGatherRegion(void* ptr, size_t size) override {
    inner_->AddHostGatherRegion(ptr, size);
  }
  void ClearHostGatherRegions() override { inner_->ClearHostGatherRegions(); }
  struct Counters {
    uint64_t fabricBytes = 0, otherBytes = 0, fabricPlans = 0;
  };
  Counters ReadCounters() const {
    Require(transfer_ != nullptr, "transport observer unavailable");
    Counters out;
    for (const auto& metric : transfer_->SampleMetrics()) {
      if (metric.name == nullptr) continue;
      std::string backend, direction;
      for (const auto& label : metric.labels) {
        if (label.first == "backend") backend = label.second;
        if (label.first == "direction") direction = label.second;
      }
      if (direction != "pull") continue;
      if (std::string(metric.name) == "mori_umbp_transport_submitted_bytes_total") {
        if (backend == "fabric")
          out.fabricBytes += metric.value;
        else
          out.otherBytes += metric.value;
      } else if (std::string(metric.name) == "mori_umbp_transport_submitted_plans_total" &&
                 backend == "fabric") {
        out.fabricPlans += metric.value;
      }
    }
    return out;
  }

 private:
  std::shared_ptr<PoolPlatform> inner_;
  TransferEngine* transfer_ = nullptr;
};

uint64_t Seed(const std::string& run) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : run) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}
unsigned char Byte(uint64_t seed, size_t key, size_t offset) {
  return static_cast<unsigned char>((seed + key * 31 + offset * 13 + (offset >> 8) * 7) % 251);
}

struct Geometry {
  size_t keys;
  size_t object;
  size_t page;
  size_t total;
  std::string layout;
  std::vector<size_t> sourceOffsets;
  std::vector<size_t> destinationOffsets;
  std::vector<size_t> lengths;
  explicit Geometry(const Args& args)
      : keys(args.Size("keys", 32)),
        object(args.Size("object-bytes", 1048704)),
        page(args.Size("page-bytes", 65536)),
        total(Multiply(keys, object)),
        layout(args.Get("layout", "unaligned")) {
    Require(keys > 0 && page >= 256 && object > Add(Multiply(page, 5), 1024),
            "geometry requires keys>0, page>=256, object>5*page+1024");
    sourceOffsets = {0, Add(Multiply(2, page), 17), object - 513};
    destinationOffsets = {13, Add(Multiply(2, page), 33), object - 498};
    lengths = {Add(page, 37), Add(page, 19), 257};
    Require(layout == "unaligned" || layout == "aligned", "--layout must be unaligned or aligned");
    if (layout == "aligned") {
      Require(page % 16 == 0 && object % 16 == 0,
              "aligned layout requires page/object multiples of 16");
      sourceOffsets = {0, Multiply(2, page), object - 512};
      destinationOffsets = {16, Add(Multiply(2, page), 32), object - 496};
    }
  }
  size_t RequestedBytes() const {
    return Multiply(keys, Add(Add(lengths[0], lengths[1]), lengths[2]));
  }
  std::vector<std::string> KeyNames(const std::string& run) const {
    std::vector<std::string> out;
    for (size_t key = 0; key < keys; ++key)
      out.push_back("ualink/" + run + "/" + std::to_string(key));
    return out;
  }
};

PoolClientConfig PeerConfig(const Args& args, const Geometry& geometry, bool producer) {
  PoolClientConfig config;
  config.master_config.master_address = args.Get("master");
  config.master_config.node_id = args.Get("node-id", producer ? "kv-owner" : "kv-reader");
  config.master_config.node_address = args.Get("advertise", "127.0.0.1");
  config.io_engine.host = "0.0.0.0";
  config.io_engine.mode = UMBPTransferMode::kFabricStrict;
  config.io_engine.fabric_kernel_path = args.Get("kernel");
  config.kv_put_placement = UMBPKvPutPlacement::kOwnerLocal;
  config.kv_ranged_path = UMBPKvRangedPath::kGpuDirect;
  config.kv_max_inflight_reads = 1;
  // Admission accounts for both destination ranges and the logical owner
  // objects whose read pins this group retains until CPU Wait completes.
  config.kv_max_inflight_bytes = Add(geometry.total, geometry.RequestedBytes());
  config.medium = TierType::HBM;
  config.hbm.device = Device(args);
  config.hbm.allocation = UMBPHbmAllocation::kFabricVmm;
  const size_t pages = geometry.object / geometry.page + (geometry.object % geometry.page != 0);
  config.hbm.buffer_sizes = {
      producer ? Multiply(Multiply(geometry.keys, pages), Multiply(2, geometry.page))
               : Multiply(2, geometry.page)};
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
  while (!stopRequested && Clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void Master(const Args& args) {
  MasterServerConfig config;
  config.listen_address = args.Get("master");
  config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
  MasterServer master(std::move(config));
  std::thread server([&] { master.Run(); });
  for (size_t i = 0; i < 200 && master.GetBoundPort() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
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

void Producer(const Args& args, const Geometry& geometry) {
  auto config = PeerConfig(args, geometry, true);
  DeviceBuffer source(geometry.total);
  PoolClient client(config);
  Require(client.Init(), "producer FABRIC strict initialization failed");
  try {
    Require(client.RegisterMemory(source.pointer, geometry.total, mori::io::MemoryLocationType::GPU,
                                  config.hbm.device),
            "producer GPU registration failed");
    const auto keys = geometry.KeyNames(args.Get("run-id"));
    std::vector<const void*> pointers;
    std::vector<size_t> sizes(geometry.keys, geometry.object);
    std::vector<unsigned char> pattern(geometry.object);
    const uint64_t seed = Seed(args.Get("run-id"));
    for (size_t key = 0; key < geometry.keys; ++key) {
      for (size_t i = 0; i < geometry.object; ++i) pattern[i] = Byte(seed, key, i);
      auto* pointer = static_cast<char*>(source.pointer) + key * geometry.object;
      Hip(hipMemcpy(pointer, pattern.data(), geometry.object, hipMemcpyHostToDevice),
          "producer fill");
      pointers.push_back(pointer);
    }
    const auto result = client.BatchPut(keys, pointers, sizes);
    Require(result.size() == keys.size() &&
                std::all_of(result.begin(), result.end(), [](bool v) { return v; }),
            "producer BatchPut failed");
    std::cout << "READY role=producer node=" << client.NodeId() << " keys=" << keys.size()
              << " object_bytes=" << geometry.object << " transport=fabric_strict" << std::endl;
    Hold(args.Size("duration-sec", 600));
    client.DeregisterMemory(source.pointer);
    client.Shutdown();
  } catch (...) {
    // This process exits on failure. Retain the user's allocation even when a
    // synchronous PUT cannot expose whether a GPU error confirmed its drain.
    source.Quarantine();
    throw;
  }
}

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  const size_t rank = static_cast<size_t>(std::ceil(percentile * values.size()));
  return values[std::min(values.size() - 1, rank > 0 ? rank - 1 : size_t{0})];
}

void Consumer(const Args& args, const Geometry& geometry) {
  const auto config = PeerConfig(args, geometry, false);
  const auto keys = geometry.KeyNames(args.Get("run-id"));
  const size_t iterations = args.Size("iterations", 100), coldEvery = args.Size("cold-every", 0);
  const size_t verifyEvery = args.Size("verify-every", 1);
  const bool async = args.Boolean("async", true);
  const uint64_t seed = Seed(args.Get("run-id"));
  Require(iterations > 0 && verifyEvery > 0, "iterations/verify-every must be positive");
  DeviceBuffer destination(geometry.total);
  DeviceBuffer consumed(geometry.total);
  ConsumerStream inference;
  std::shared_ptr<ObservedPlatform> platform;
  std::unique_ptr<PoolClient> client;
  auto initialize = [&] {
    const auto begin = Clock::now();
    platform = std::make_shared<ObservedPlatform>(config.sim);
    client = std::make_unique<PoolClient>(config, platform);
    Require(client->Init(), "consumer FABRIC strict initialization failed");
    Require(client->RegisterMemory(destination.pointer, geometry.total,
                                   mori::io::MemoryLocationType::GPU, config.hbm.device),
            "consumer GPU registration failed");
    std::cout << "INIT elapsed_ms="
              << std::chrono::duration<double, std::milli>(Clock::now() - begin).count()
              << " node=" << client->NodeId() << std::endl;
  };
  try {
    initialize();
    MasterClient lookup(config.master_config);
    std::vector<std::optional<RouteGetResult>> routes;
    const auto deadline = Clock::now() + std::chrono::seconds(args.Size("timeout-sec", 60));
    while (true) {
      const auto status = lookup.BatchRouteGet(keys, {}, &routes);
      if (status.ok() && routes.size() == keys.size() &&
          std::all_of(routes.begin(), routes.end(),
                      [](const auto& route) { return route.has_value(); }))
        break;
      Require(!stopRequested && Clock::now() < deadline, "producer keys did not become visible");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    for (const auto& route : routes) {
      Require(route->node_id == args.Get("expected-owner", "kv-owner") &&
                  route->node_id != client->NodeId() && route->tier == TierType::HBM &&
                  route->size == geometry.object,
              "key placement/size mismatch");
    }
    std::vector<std::vector<void*>> dsts(geometry.keys);
    std::vector<std::vector<size_t>> lengths(geometry.keys, geometry.lengths);
    std::vector<std::vector<size_t>> offsets(geometry.keys, geometry.sourceOffsets);
    for (size_t key = 0; key < geometry.keys; ++key) {
      for (size_t offset : geometry.destinationOffsets) {
        dsts[key].push_back(static_cast<char*>(destination.pointer) + key * geometry.object +
                            offset);
      }
    }
    std::vector<unsigned char> actual(geometry.total);
    std::vector<double> warm, cold;
    for (size_t iteration = 0; iteration < iterations && !stopRequested; ++iteration) {
      const bool isCold = iteration == 0 || (coldEvery && iteration % coldEvery == 0);
      if (isCold && iteration != 0) {
        client->DeregisterMemory(destination.pointer);
        client->Shutdown();
        client.reset();
        initialize();
      }
      for (auto* backend : client->Backends().All())
        for (const auto& key : keys)
          Require(!backend->Contains(key), "reader unexpectedly cached an owner key before fetch");
      Hip(hipMemset(destination.pointer, 0xD3, geometry.total), "destination sentinel fill");
      const auto before = platform->ReadCounters();
      const auto begin = Clock::now();
      auto handle = client->SubmitBatchGetRangesGpuDirect(keys, dsts, lengths, offsets);
      Require(handle != nullptr, "GPU direct submission did not return a handle");
      const bool streamAccepted =
          !async || handle->WaitOnStream(reinterpret_cast<void*>(inference.stream));
      if (async && streamAccepted) {
        // Acts as an inference consumer. Enqueue this before CPU Wait so byte
        // verification proves the GPU dependency, including untouched regions.
        Hip(hipMemcpyAsync(consumed.pointer, destination.pointer, geometry.total,
                           hipMemcpyDeviceToDevice, inference.stream),
            "enqueue consumer GPU copy");
      }
      const auto result = handle->Wait();
      const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      if (handle->ResourcesQuarantined()) {
        destination.Quarantine();
        consumed.Quarantine();
        inference.Quarantine();
        throw std::runtime_error("unconfirmed GPU drain; consumer allocations retained until exit");
      }
      Require(streamAccepted,
              "FABRIC event dependency could not be enqueued on the inference stream");
      Require(result.size() == keys.size() &&
                  std::all_of(result.begin(), result.end(), [](bool v) { return v; }),
              "consumer ranged GET failed (strict mode disallows RDMA fallback)");
      if (!async)
        Hip(hipMemcpyAsync(consumed.pointer, destination.pointer, geometry.total,
                           hipMemcpyDeviceToDevice, inference.stream),
            "enqueue consumer GPU copy");
      Hip(hipStreamSynchronize(inference.stream), "consumer GPU stream completion");
      const double consumerSeconds = std::chrono::duration<double>(Clock::now() - begin).count();
      const auto after = platform->ReadCounters();
      Require(after.fabricBytes >= before.fabricBytes && after.otherBytes >= before.otherBytes &&
                  after.fabricPlans >= before.fabricPlans,
              "transport counters moved backwards");
      const uint64_t fabricBytes = after.fabricBytes - before.fabricBytes;
      const uint64_t fabricPlans = after.fabricPlans - before.fabricPlans;
      Require(fabricBytes == geometry.RequestedBytes() && after.otherBytes == before.otherBytes &&
                  fabricPlans > 0,
              "selected FABRIC submitted bytes differ from requested remote ranges");
      for (auto* backend : client->Backends().All())
        for (const auto& key : keys)
          Require(!backend->Contains(key), "reader unexpectedly cached an owner key after fetch");
      (isCold ? cold : warm).push_back(seconds);
      if (iteration % verifyEvery == 0 || iteration + 1 == iterations) {
        Hip(hipMemcpy(actual.data(), consumed.pointer, geometry.total, hipMemcpyDeviceToHost),
            "verification download");
        for (size_t key = 0; key < geometry.keys; ++key) {
          for (size_t i = 0; i < geometry.object; ++i) {
            unsigned char expected = 0xD3;
            for (size_t range = 0; range < geometry.lengths.size(); ++range) {
              if (i >= geometry.destinationOffsets[range] &&
                  i - geometry.destinationOffsets[range] < geometry.lengths[range]) {
                expected =
                    Byte(seed, key,
                         geometry.sourceOffsets[range] + i - geometry.destinationOffsets[range]);
              }
            }
            Require(actual[key * geometry.object + i] == expected,
                    "byte mismatch key=" + std::to_string(key) + " offset=" + std::to_string(i));
          }
        }
      }
      std::cout << "SAMPLE phase=" << (isCold ? "cold" : "warm") << " iteration=" << iteration
                << " requested_bytes=" << geometry.RequestedBytes()
                << " latency_us=" << seconds * 1e6 << " consumer_ready_us=" << consumerSeconds * 1e6
                << " selected_fabric_submitted_bytes=" << fabricBytes
                << " selected_fabric_submitted_plans=" << fabricPlans
                << " layout=" << geometry.layout
                << " completion=" << (async ? "stream_wait" : "blocking_wait")
                << " requested_GBps=" << geometry.RequestedBytes() / seconds / 1e9 << std::endl;
    }
    for (const auto& phase : {std::make_pair("cold", cold), std::make_pair("warm", warm)}) {
      if (phase.second.empty()) continue;
      double seconds = 0;
      for (double value : phase.second) seconds += value;
      std::cout << "SUMMARY phase=" << phase.first << " samples=" << phase.second.size()
                << " requested_bytes_per_call=" << geometry.RequestedBytes()
                << " p50_us=" << Percentile(phase.second, .50) * 1e6
                << " p95_us=" << Percentile(phase.second, .95) * 1e6
                << " p99_us=" << Percentile(phase.second, .99) * 1e6 << " aggregate_requested_GBps="
                << static_cast<double>(geometry.RequestedBytes()) * phase.second.size() / seconds /
                       1e9
                << " transport=fabric_strict physical_bus_bytes=unmeasured" << std::endl;
    }
    client->DeregisterMemory(destination.pointer);
    client->Shutdown();
    std::cout << "PASS consumer ranged bytes, untouched regions, GPU stream dependency and FABRIC "
                 "counters"
              << std::endl;
  } catch (...) {
    // A hardware validation failure terminates this process. Conservatively
    // avoid GPU cleanup that could race an incomplete consumer or transfer.
    destination.Quarantine();
    consumed.Quarantine();
    inference.Quarantine();
    throw;
  }
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
    if (role == "master")
      Master(args);
    else {
      Require(role == "producer" || role == "consumer",
              "--role must be master, producer or consumer");
      Require(!args.Get("kernel").empty() && !args.Get("run-id").empty(),
              "--kernel and --run-id required");
      Require(!std::getenv("MORI_IO_SIM_ENABLE") ||
                  std::string(std::getenv("MORI_IO_SIM_ENABLE")) == "0",
              "hardware driver must not run with MORI_IO_SIM_ENABLE");
      Hip(hipSetDevice(Device(args)), "select GPU");
      const Geometry geometry(args);
      if (role == "producer")
        Producer(args, geometry);
      else
        Consumer(args, geometry);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
