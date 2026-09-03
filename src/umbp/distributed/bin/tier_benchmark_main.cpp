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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/benchmark/pool_workload_client.h"
#include "umbp/distributed/benchmark/workload_runner.h"
#include "umbp/distributed/benchmark/workload_source.h"
#include "umbp/distributed/benchmark/workload_trace.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool/policy_config.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/routing/route_get_strategy.h"
#include "umbp/distributed/routing/route_put_strategy.h"
namespace {
namespace bench = mori::umbp::benchmark;
using mori::umbp::BackendInstanceConfig;
using mori::umbp::ConfigurableRoutePutStrategy;
using mori::umbp::LocalPreferringRouteGetStrategy;
using mori::umbp::MasterServer;
using mori::umbp::MasterServerConfig;
using mori::umbp::MediumBackend;
using mori::umbp::PoolClient;
using mori::umbp::PoolClientConfig;
using mori::umbp::PoolPlacementPolicy;
using mori::umbp::RandomRouteGetStrategy;
using mori::umbp::TierType;
constexpr uint64_t kDefaultCapacity = 256ULL << 20;
constexpr uint64_t kDefaultPageSize = 2ULL << 20;
constexpr auto kServerStartTimeout = std::chrono::seconds(10);
struct Options {
  std::string command;
  std::string trace_path;
  std::string policy_path;
  // Parsed once by LoadPolicy; the clients lower it per node rather than
  // reading and reparsing the file.
  std::optional<mori::umbp::BackendPolicyConfig> policy;
  // Largest object that can fit one SSD backend's contiguous staging arena.
  uint64_t max_ssd_staging_object_bytes = std::numeric_limits<uint64_t>::max();
  bench::SyntheticWorkloadConfig workload;
  bench::WorkloadRunnerOptions runner;
  std::string profile = "mixed";
  std::string size_distribution = "fixed";
  std::string put_strategy = "most_available";
  std::string affinity = "none";
  std::string get_strategy = "local";
  uint64_t backend_capacity = kDefaultCapacity;
  uint64_t page_size = kDefaultPageSize;
  uint64_t settle_ms = 500;
  bool clients_explicit = false;
  bool scheduling_explicit = false;
  // Set for a trace recorded from production: the workload is whatever the
  // trace contains, so the synthetic generator's knobs describe nothing.
  bool trace_defines_workload = false;
};
[[noreturn]] void UsageError(const std::string& message) {
  throw std::invalid_argument(message + " (use --help for usage)");
}
void PrintHelp() {
  std::cout << R"(UMBP distributed tier-management benchmark
Usage:
  umbp_tier_bench generate --trace PATH [OPTIONS]
  umbp_tier_bench run [OPTIONS]
  umbp_tier_bench replay --trace PATH [OPTIONS]
Workload:
  --profile NAME  --seed N  --operations N  --keys N
  --min-value-bytes SIZE  --max-value-bytes SIZE
  --size-distribution fixed|uniform|log-uniform
  --read-ratio R  --clients N  --batch N  --qps Q
Cluster:
  --config PATH   topology: named backends, logical tiers, weights, watermarks.
                  Without it every peer gets one DRAM backend sized by
                  --backend-capacity.
  --backend-capacity SIZE  --page-size SIZE
  --put-strategy most_available|random  --affinity none|same|local
  --get-strategy local|random
Execution:
  --max-throughput | --open-loop  --time-scale R
  --payload-validation BOOL  --no-payload-validation  --settle-ms N
SIZE may use B, KiB, MiB, or GiB. Options accept --name value or --name=value.
)";
}
template <typename UInt>
UInt ParseUnsigned(std::string_view text, const char* option) {
  static_assert(std::is_unsigned_v<UInt>);
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || text.front() == '-' || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() ||
      value > static_cast<uint64_t>(std::numeric_limits<UInt>::max())) {
    UsageError(std::string(option) + " expects an unsigned integer");
  }
  return static_cast<UInt>(value);
}
double ParseDouble(std::string_view text, const char* option) {
  const std::string owned(text);
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &end);
  if (errno == ERANGE || end == owned.c_str() || *end != '\0' || !std::isfinite(value)) {
    UsageError(std::string(option) + " expects a finite number");
  }
  return value;
}
bool ParseBool(std::string_view text, const char* option) {
  if (text == "true" || text == "1") return true;
  if (text == "false" || text == "0") return false;
  UsageError(std::string(option) + " expects true or false");
}
uint64_t ParseSize(std::string_view text, const char* option) {
  uint64_t scale = 1;
  for (const auto& suffix : {std::pair<std::string_view, uint64_t>{"GiB", 1ULL << 30},
                             {"MiB", 1ULL << 20},
                             {"KiB", 1ULL << 10},
                             {"B", 1}}) {
    if (text.size() >= suffix.first.size() &&
        text.substr(text.size() - suffix.first.size()) == suffix.first) {
      text.remove_suffix(suffix.first.size());
      scale = suffix.second;
      break;
    }
  }
  const uint64_t value = ParseUnsigned<uint64_t>(text, option);
  if (value > std::numeric_limits<uint64_t>::max() / scale) {
    UsageError(std::string(option) + " is too large");
  }
  return value * scale;
}
Options ParseOptions(int argc, char** argv) {
  if (argc < 2) UsageError("missing subcommand");
  Options options;
  options.command = argv[1];
  options.runner.max_throughput = true;
  if (options.command == "help" || options.command == "--help" || options.command == "-h") {
    PrintHelp();
    std::exit(0);
  }
  if (options.command != "run" && options.command != "generate" && options.command != "replay") {
    UsageError("unknown subcommand: " + options.command);
  }
  for (int i = 2; i < argc; ++i) {
    std::string name = argv[i];
    std::string inline_value;
    bool has_inline_value = false;
    if (const size_t equals = name.find('='); equals != std::string::npos) {
      inline_value = name.substr(equals + 1);
      name.resize(equals);
      has_inline_value = true;
    }
    if (name == "--help" || name == "-h") {
      PrintHelp();
      std::exit(0);
    }
    auto value = [&]() {
      if (has_inline_value) {
        if (inline_value.empty()) UsageError(name + " requires a value");
        return inline_value;
      }
      if (i + 1 >= argc) UsageError(name + " requires a value");
      return std::string(argv[++i]);
    };
    if (name == "--max-throughput") {
      if (has_inline_value) UsageError(name + " does not take a value");
      options.runner.max_throughput = true;
      options.scheduling_explicit = true;
    } else if (name == "--open-loop") {
      if (has_inline_value) UsageError(name + " does not take a value");
      options.runner.max_throughput = false;
      options.scheduling_explicit = true;
    } else if (name == "--no-payload-validation") {
      if (has_inline_value) UsageError(name + " does not take a value");
      options.runner.validate_get_payloads = false;
    } else if (name == "--trace") {
      options.trace_path = value();
    } else if (name == "--config") {
      options.policy_path = value();
    } else if (name == "--profile") {
      options.profile = value();
    } else if (name == "--seed") {
      options.workload.seed = ParseUnsigned<uint64_t>(value(), "--seed");
    } else if (name == "--operations") {
      options.workload.operation_count = ParseUnsigned<uint64_t>(value(), "--operations");
    } else if (name == "--keys") {
      options.workload.key_count = ParseUnsigned<uint64_t>(value(), "--keys");
    } else if (name == "--min-value-bytes") {
      options.workload.min_value_size = ParseSize(value(), "--min-value-bytes");
    } else if (name == "--max-value-bytes") {
      options.workload.max_value_size = ParseSize(value(), "--max-value-bytes");
    } else if (name == "--size-distribution") {
      options.size_distribution = value();
    } else if (name == "--read-ratio") {
      options.workload.read_ratio = ParseDouble(value(), "--read-ratio");
    } else if (name == "--clients") {
      options.workload.client_count = ParseUnsigned<uint32_t>(value(), "--clients");
      options.clients_explicit = true;
    } else if (name == "--batch") {
      options.workload.batch_size = ParseUnsigned<uint32_t>(value(), "--batch");
    } else if (name == "--qps") {
      options.workload.qps = ParseDouble(value(), "--qps");
    } else if (name == "--backend-capacity") {
      options.backend_capacity = ParseSize(value(), "--backend-capacity");
    } else if (name == "--page-size") {
      options.page_size = ParseSize(value(), "--page-size");
    } else if (name == "--put-strategy") {
      options.put_strategy = value();
    } else if (name == "--affinity") {
      options.affinity = value();
    } else if (name == "--get-strategy") {
      options.get_strategy = value();
    } else if (name == "--time-scale") {
      options.runner.time_scale = ParseDouble(value(), "--time-scale");
    } else if (name == "--payload-validation") {
      options.runner.validate_get_payloads = ParseBool(value(), "--payload-validation");
    } else if (name == "--settle-ms") {
      options.settle_ms = ParseUnsigned<uint64_t>(value(), "--settle-ms");
    } else {
      UsageError("unknown option: " + name);
    }
  }
  if (options.command == "replay" && !options.scheduling_explicit) {
    options.runner.max_throughput = false;
  }
  return options;
}
bench::SyntheticProfile ParseProfile(const std::string& name) {
  if (name == "sequential") return bench::SyntheticProfile::kSequential;
  if (name == "uniform") return bench::SyntheticProfile::kUniform;
  if (name == "hotset") return bench::SyntheticProfile::kHotsetZipf;
  if (name == "read-after-write") return bench::SyntheticProfile::kReadAfterWrite;
  if (name == "mixed") return bench::SyntheticProfile::kMixed;
  if (name == "capacity-pressure") return bench::SyntheticProfile::kCapacityPressure;
  UsageError("invalid --profile: " + name);
}
void ValidateOptions(Options* options) {
  if (options->page_size == 0) UsageError("--page-size must be positive");
  options->workload.profile = ParseProfile(options->profile);
  if (options->size_distribution == "fixed") {
    options->workload.value_size_distribution = bench::ValueSizeDistribution::kFixed;
  } else if (options->size_distribution == "uniform") {
    options->workload.value_size_distribution = bench::ValueSizeDistribution::kUniform;
  } else if (options->size_distribution == "log-uniform") {
    options->workload.value_size_distribution = bench::ValueSizeDistribution::kLogUniform;
  } else {
    UsageError("--size-distribution must be fixed, uniform, or log-uniform");
  }
  if ((options->command == "generate" || options->command == "replay") &&
      options->trace_path.empty()) {
    UsageError(options->command + " requires --trace");
  }
  if (options->workload.client_count == 0 || options->workload.key_count == 0 ||
      options->workload.batch_size == 0 || options->workload.min_value_size == 0) {
    UsageError("clients, keys, batch, and value sizes must be positive");
  }
  if (options->workload.min_value_size > options->workload.max_value_size) {
    UsageError("--min-value-bytes must not exceed --max-value-bytes");
  }
  if (options->workload.read_ratio < 0 || options->workload.read_ratio > 1 ||
      options->workload.qps < 0 || options->runner.time_scale <= 0) {
    UsageError("read ratio, QPS, or time scale is out of range");
  }
  if (options->put_strategy != "most_available" && options->put_strategy != "random") {
    UsageError("--put-strategy must be most_available or random");
  }
  if (options->affinity != "none" && options->affinity != "same" && options->affinity != "local") {
    UsageError("--affinity must be none, same, or local");
  }
  if (options->get_strategy != "local" && options->get_strategy != "random") {
    UsageError("--get-strategy must be local or random");
  }
  if (options->policy_path.empty() && options->backend_capacity < options->page_size) {
    UsageError("--backend-capacity must be at least --page-size");
  }
  // For a replay the authoritative sizes are in the events, and the trace scan
  // checks them there.
  if (options->command != "replay" &&
      options->workload.max_value_size > options->max_ssd_staging_object_bytes) {
    UsageError("SSD values must not exceed the contiguous staging arena");
  }
}
// Reads --config once. The clients lower it again per node for their own
// instance names; this pass exists to reject a bad topology before anything
// starts, and to tell the rest of the run what media it will be writing to.
void LoadPolicy(Options* options) {
  if (options->policy_path.empty()) return;
  if (options->page_size == 0) UsageError("--page-size must be positive");
  auto loaded = mori::umbp::LoadBackendPolicyFile(options->policy_path);
  if (!loaded.ok()) UsageError("invalid --config: " + loaded.error);
  options->policy = std::move(*loaded.config);

  PoolClientConfig lowered;
  lowered.dram_page_size = options->page_size;
  std::string error;
  if (!mori::umbp::ApplyBackendPolicy(*options->policy, &lowered, &error)) {
    UsageError("invalid --config: " + error);
  }
  if (lowered.backends.size() > mori::umbp::kMaxBackendsPerPeer) {
    UsageError("--config expands beyond the per-peer backend limit");
  }
  for (const auto& backend : lowered.backends) {
    if (backend.tier != TierType::SSD) continue;
    const uint64_t pages = backend.ssd_staging_buffer_slots > 0
                               ? static_cast<uint64_t>(backend.ssd_staging_buffer_slots)
                               : 16;
    if (pages > std::numeric_limits<uint64_t>::max() / options->page_size) {
      UsageError("SSD staging arena size overflows uint64");
    }
    const uint64_t ssd_capacity = backend.ssd.ssd.capacity_bytes;
    options->max_ssd_staging_object_bytes =
        std::min({options->max_ssd_staging_object_bytes, pages * options->page_size, ssd_capacity});
  }
}
void AddTraceMetadata(const Options& options, ::umbp::benchmark::WorkloadTraceHeader* header) {
  auto& metadata = *header->mutable_metadata();
  metadata["profile"] = options.profile;
  metadata["seed"] = std::to_string(options.workload.seed);
  metadata["operations"] = std::to_string(options.workload.operation_count);
  metadata["keys"] = std::to_string(options.workload.key_count);
  metadata["min_value_bytes"] = std::to_string(options.workload.min_value_size);
  metadata["max_value_bytes"] = std::to_string(options.workload.max_value_size);
  metadata["size_distribution"] = options.size_distribution;
  metadata["read_ratio"] = std::to_string(options.workload.read_ratio);
  metadata["clients"] = std::to_string(options.workload.client_count);
  metadata["batch"] = std::to_string(options.workload.batch_size);
  metadata["qps"] = std::to_string(options.workload.qps);
  metadata["hotset_fraction"] = std::to_string(options.workload.hotset_fraction);
  metadata["zipf_exponent"] = std::to_string(options.workload.zipf_exponent);
  metadata["key_prefix"] = options.workload.key_prefix;
}
int GenerateTrace(const Options& options) {
  ::umbp::benchmark::WorkloadTraceHeader header;
  header.set_schema_version(bench::kWorkloadTraceSchemaVersion);
  header.set_time_unit(::umbp::benchmark::WorkloadTraceHeader::TIME_UNIT_NANOSECONDS);
  header.set_seed(options.workload.seed);
  AddTraceMetadata(options, &header);
  bench::SyntheticWorkloadSource source(options.workload);
  bench::TraceWriter writer(options.trace_path, header);
  ::umbp::benchmark::WorkloadEvent event;
  uint64_t events = 0;
  while (source.Next(&event)) {
    writer.Write(event);
    ++events;
  }
  writer.Close();
  std::cout << "trace_path,events,seed,schema_version\n\"" << options.trace_path << "\"," << events
            << ',' << options.workload.seed << ',' << bench::kWorkloadTraceSchemaVersion << '\n';
  return 0;
}
class MasterHarness {
 public:
  explicit MasterHarness(const Options& options) {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.registry_config.default_dram_page_size = options.page_size;
    const auto algorithm = options.put_strategy == "random"
                               ? ConfigurableRoutePutStrategy::SelectAlgo::kRandom
                               : ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable;
    auto affinity = ConfigurableRoutePutStrategy::NodeAffinity::kNone;
    if (options.affinity == "same") {
      affinity = ConfigurableRoutePutStrategy::NodeAffinity::kSame;
    } else if (options.affinity == "local") {
      affinity = ConfigurableRoutePutStrategy::NodeAffinity::kLocal;
    }
    config.put_strategy =
        std::make_unique<ConfigurableRoutePutStrategy>(algorithm, affinity, options.workload.seed);
    if (options.get_strategy == "random") {
      config.get_strategy = std::make_unique<RandomRouteGetStrategy>();
    } else {
      config.get_strategy = std::make_unique<LocalPreferringRouteGetStrategy>();
    }
    server_ = std::make_unique<MasterServer>(std::move(config));
    thread_ = std::thread([this] {
      try {
        server_->Run();
      } catch (...) {
        error_ = std::current_exception();
        failed_.store(true, std::memory_order_release);
      }
    });
    const auto deadline = std::chrono::steady_clock::now() + kServerStartTimeout;
    while (server_->GetBoundPort() == 0 && std::chrono::steady_clock::now() < deadline &&
           !failed_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (server_->GetBoundPort() == 0) {
      Stop();
      if (error_) std::rethrow_exception(error_);
      throw std::runtime_error("timed out starting in-process master");
    }
  }

  ~MasterHarness() { Stop(); }
  std::string Address() const { return "127.0.0.1:" + std::to_string(server_->GetBoundPort()); }

 private:
  void Stop() {
    if (server_) server_->Shutdown();
    if (thread_.joinable()) thread_.join();
  }
  std::unique_ptr<MasterServer> server_;
  std::thread thread_;
  std::exception_ptr error_;
  std::atomic<bool> failed_{false};
};
// Topology for a run without --config: the smallest thing that can serve the
// workload. Anything richer — several backends, other media, logical tiers,
// weights, watermarks — is what the policy file is for.
BackendInstanceConfig DefaultBackend(uint64_t capacity) {
  BackendInstanceConfig backend;
  backend.name = "dram-0";
  backend.tier = TierType::DRAM;
  backend.dram.buffer_sizes = {capacity};
  return backend;
}
std::vector<std::unique_ptr<PoolClient>> StartClients(const Options& options,
                                                      const std::string& master_address) {
  std::vector<std::unique_ptr<PoolClient>> clients;
  const auto& policy = options.policy;
  for (uint32_t node = 0; node < options.workload.client_count; ++node) {
    PoolClientConfig config;
    config.master_config.master_address = master_address;
    config.master_config.node_id = "tier-benchmark-node-" + std::to_string(node);
    config.master_config.node_address = "127.0.0.1";
    config.master_config.auto_heartbeat = true;
    config.auto_peer_service_port = true;
    if (options.workload.client_count > 1) {
      config.io_engine.host = "127.0.0.1";
      config.io_engine.port = 0;
    }
    config.dram_page_size = options.page_size;
    config.cache_remote_fetches = false;
    if (policy.has_value()) {
      std::string error;
      if (!mori::umbp::ApplyBackendPolicy(*policy, &config, &error,
                                          "-tier-benchmark-node-" + std::to_string(node))) {
        throw std::runtime_error("failed to apply backend policy: " + error);
      }
    } else {
      config.placement_policy = PoolPlacementPolicy::SINGLE_BACKEND;
      config.backends.push_back(DefaultBackend(options.backend_capacity));
    }
    auto client = std::make_unique<PoolClient>(std::move(config));
    if (!client->Init()) {
      throw std::runtime_error("failed to initialize PoolClient " + std::to_string(node));
    }
    clients.push_back(std::move(client));
  }
  return clients;
}
void Settle(const std::vector<std::unique_ptr<PoolClient>>& clients, uint64_t ms) {
  for (const auto& client : clients) client->Master().FlushHeartbeat();
  if (ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
std::string Csv(std::string_view value) {
  std::string result = "\"";
  for (char c : value) {
    result += c;
    if (c == '"') result += c;
  }
  return result + '"';
}
void PrintSummary(const Options& options, const bench::WorkloadMetrics& metrics, uint64_t seed,
                  uint64_t wall_ns) {
  const double seconds = static_cast<double>(wall_ns) / 1e9;
  const double ops_per_second = seconds == 0 ? 0 : metrics.total.succeeded / seconds;
  const double gib_per_second =
      seconds == 0 ? 0 : metrics.total.succeeded_bytes / static_cast<double>(1ULL << 30) / seconds;
  // When a recorded trace supplies the workload, the generator's knobs kept
  // their defaults and describe nothing; reporting them invents a shape the run
  // never had. The topology is always reported from the pool itself below.
  const auto generated = [&](std::string_view value) {
    return options.trace_defines_workload ? Csv("") : Csv(value);
  };
  const auto generated_number = [&](auto value) {
    if (options.trace_defines_workload) return std::string{};
    std::ostringstream text;
    text << value;
    return text.str();
  };
  std::cout << "config\n"
            << "command,trace,policy,seed,profile,operations,keys,min_value_bytes,max_value_bytes,"
               "size_distribution,read_ratio,clients,batch,qps,backend_capacity,page_size,"
               "put_strategy,affinity,get_strategy,scheduling,time_scale,payload_validation,"
               "settle_ms\n"
            << Csv(options.command) << ',' << Csv(options.trace_path) << ','
            << Csv(options.policy_path) << ',' << seed << ',' << generated(options.profile) << ','
            << options.workload.operation_count << ','
            << generated_number(options.workload.key_count) << ','
            << generated_number(options.workload.min_value_size) << ','
            << generated_number(options.workload.max_value_size) << ','
            << generated(options.size_distribution) << ','
            << generated_number(options.workload.read_ratio) << ',' << options.workload.client_count
            << ',' << generated_number(options.workload.batch_size) << ','
            << generated_number(options.workload.qps) << ','
            << (options.policy_path.empty() ? std::to_string(options.backend_capacity)
                                            : std::string{})
            << ',' << options.page_size << ',' << Csv(options.put_strategy) << ','
            << Csv(options.affinity) << ',' << Csv(options.get_strategy) << ','
            << Csv(options.runner.max_throughput ? "max-throughput" : "open-loop") << ','
            << options.runner.time_scale << ','
            << (options.runner.validate_get_payloads ? "true" : "false") << ',' << options.settle_ms
            << '\n'
            << "summary\n"
            << "seed,attempted,succeeded,failed,succeeded_bytes,puts,puts_failed,gets,gets_failed,"
               "get_misses,validation_failures,recorded_misses,recorded_failures,"
               "wall_time_ns,ops_per_s,GiB_per_s,latency_p50_ns,"
               "latency_p95_ns,latency_p99_ns,latency_max_ns,schedule_lag_p99_ns\n"
            << seed << ',' << metrics.total.attempted << ',' << metrics.total.succeeded << ','
            << metrics.total.failed << ',' << metrics.total.succeeded_bytes << ','
            << metrics.puts.succeeded << ',' << metrics.puts.failed << ',' << metrics.gets.succeeded
            << ',' << metrics.gets.failed << ',' << metrics.get_misses << ','
            << metrics.get_validation_failures << ',' << metrics.recorded_misses << ','
            << metrics.recorded_failures << ',' << wall_ns << ',' << std::setprecision(10)
            << ops_per_second << ',' << gib_per_second << ',' << metrics.latency.p50_ns << ','
            << metrics.latency.p95_ns << ',' << metrics.latency.p99_ns << ','
            << metrics.latency.max_ns << ',' << metrics.schedule_lag.p99_ns << '\n';
}
void PrintBackendPlacement(const std::vector<std::unique_ptr<PoolClient>>& clients) {
  // logical_tier is the topology as the pool actually built it, which is the
  // only description that stays true when a policy file expands one named
  // backend into several instances.
  std::cout << "backend_placement\n"
            << "node,backend,logical_tier,tier,owned_keys,total_bytes,available_bytes,"
               "max_allocatable_bytes\n";
  for (const auto& client : clients) {
    auto& registry = client->Backends();
    for (MediumBackend* backend : registry.All()) {
      const auto* entry = registry.GetEntry(backend);
      const auto capacity = backend->Capacity();
      std::cout << Csv(client->NodeId()) << ','
                << Csv(entry == nullptr ? backend->Name() : entry->name) << ','
                << Csv(client->LogicalTierForBackend(registry.BackendId(backend))) << ','
                << Csv(mori::umbp::TierTypeName(backend->Tier())) << ',' << backend->OwnedKeyCount()
                << ',' << capacity.total_bytes << ',' << capacity.available_bytes << ','
                << capacity.max_allocatable_bytes << '\n';
    }
  }
  std::cout << "tier_transitions\n"
            << "node,attempted,succeeded,failed,offloaded_bytes,promoted_bytes\n";
  for (const auto& client : clients) {
    const auto metrics = client->TransitionMetrics();
    std::cout << Csv(client->NodeId()) << ',' << metrics.attempted << ',' << metrics.succeeded
              << ',' << metrics.failed << ',' << metrics.offloaded_bytes << ','
              << metrics.promoted_bytes << '\n';
  }
  // Which tier answered the reads: a hierarchy whose hot tier serves few of them
  // is misconfigured even when the throughput number looks acceptable.
  std::cout << "tier_reads\n" << "node,logical_tier,reads_served\n";
  for (const auto& client : clients) {
    for (const auto& [tier, reads] : client->TierReadHits()) {
      std::cout << Csv(client->NodeId()) << ',' << Csv(tier) << ',' << reads << '\n';
    }
  }
}
void InspectReplayTrace(Options* options) {
  bench::TraceReader reader(options->trace_path);
  options->workload.seed = reader.header().seed();
  const auto& metadata = reader.header().metadata();
  auto metadata_value = [&](const char* key) -> const std::string* {
    const auto found = metadata.find(key);
    return found == metadata.end() ? nullptr : &found->second;
  };
  if (const auto* value = metadata_value("profile")) options->profile = *value;
  // A recorded trace carries keys and sizes but not payloads, so the expected
  // bytes cannot be rederived and GET validation would fail on every hit.
  if (const auto* value = metadata_value("payload_mode");
      value != nullptr && *value == "external") {
    options->runner.validate_get_payloads = false;
  }
  if (const auto* value = metadata_value("source"); value != nullptr && *value == "production") {
    options->trace_defines_workload = true;
  }
  if (const auto* value = metadata_value("keys")) {
    options->workload.key_count = ParseUnsigned<uint64_t>(*value, "trace metadata keys");
  }
  if (const auto* value = metadata_value("min_value_bytes")) {
    options->workload.min_value_size =
        ParseUnsigned<size_t>(*value, "trace metadata min_value_bytes");
  }
  if (const auto* value = metadata_value("max_value_bytes")) {
    options->workload.max_value_size =
        ParseUnsigned<size_t>(*value, "trace metadata max_value_bytes");
  }
  if (const auto* value = metadata_value("size_distribution")) {
    options->size_distribution = *value;
  }
  if (const auto* value = metadata_value("read_ratio")) {
    options->workload.read_ratio = ParseDouble(*value, "trace metadata read_ratio");
  }
  if (const auto* value = metadata_value("batch")) {
    options->workload.batch_size = ParseUnsigned<uint32_t>(*value, "trace metadata batch");
  }
  if (const auto* value = metadata_value("qps")) {
    options->workload.qps = ParseDouble(*value, "trace metadata qps");
  }
  if (!options->clients_explicit) {
    const auto found = metadata.find("clients");
    if (found != metadata.end()) {
      options->workload.client_count =
          ParseUnsigned<uint32_t>(found->second, "trace metadata clients");
    }
  }
  bool has_events = false;
  uint32_t max_client = 0;
  uint64_t event_count = 0;
  ::umbp::benchmark::WorkloadEvent event;
  while (reader.ReadNext(&event)) {
    has_events = true;
    ++event_count;
    max_client = std::max(max_client, event.client_id());
    if (event.value_size() > options->max_ssd_staging_object_bytes) {
      UsageError("trace contains an SSD value larger than the contiguous staging arena");
    }
  }
  options->workload.operation_count = event_count;
  if (!options->clients_explicit && metadata.find("clients") == metadata.end() && has_events) {
    if (max_client == std::numeric_limits<uint32_t>::max()) {
      UsageError("trace client id is too large");
    }
    options->workload.client_count = max_client + 1;
  }
  if (has_events && max_client >= options->workload.client_count) {
    UsageError("trace client id exceeds --clients");
  }
}
int RunBenchmark(Options options) {
  if (options.command == "replay") {
    InspectReplayTrace(&options);
    ValidateOptions(&options);
  }
  std::unique_ptr<bench::WorkloadSource> source =
      options.command == "replay"
          ? std::unique_ptr<bench::WorkloadSource>(
                std::make_unique<bench::TraceWorkloadSource>(options.trace_path))
          : std::unique_ptr<bench::WorkloadSource>(
                std::make_unique<bench::SyntheticWorkloadSource>(options.workload));
  MasterHarness master(options);
  auto clients = StartClients(options, master.Address());
  bench::PoolWorkloadClient adapter(&clients);
  Settle(clients, options.settle_ms);
  bench::WorkloadRunner runner(&adapter, options.runner);
  const auto start = std::chrono::steady_clock::now();
  const auto metrics = runner.Run(source.get());
  const uint64_t wall_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count();
  Settle(clients, options.settle_ms);
  PrintSummary(options, metrics, source->seed(), wall_ns);
  PrintBackendPlacement(clients);
  return metrics.total.failed == 0 ? 0 : 2;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Options options = ParseOptions(argc, argv);
    LoadPolicy(&options);
    ValidateOptions(&options);
    return options.command == "generate" ? GenerateTrace(options)
                                         : RunBenchmark(std::move(options));
  } catch (const std::exception& error) {
    std::cerr << "umbp_tier_bench: " << error.what() << '\n';
    return 1;
  }
}
