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
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "umbp/distributed/benchmark/workload_runner.h"
#include "umbp/distributed/benchmark/workload_source.h"
#include "umbp/distributed/benchmark/workload_trace.h"
#include "umbp/distributed/master/master_server.h"
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
constexpr auto kPublicationTimeout = std::chrono::seconds(2);
constexpr auto kServerStartTimeout = std::chrono::seconds(10);
struct Options {
  std::string command;
  std::string trace_path;
  bench::SyntheticWorkloadConfig workload;
  bench::WorkloadRunnerOptions runner;
  std::string profile = "mixed";
  std::string size_distribution = "fixed";
  std::string put_strategy = "most_available";
  std::string affinity = "none";
  std::string get_strategy = "local";
  std::string placement = "single";
  std::string tier = "dram";
  std::string ssd_dir_prefix = "/tmp/umbp-tier-benchmark";
  uint32_t backends_per_peer = 1;
  std::vector<uint32_t> placement_weights;
  uint64_t backend_capacity = kDefaultCapacity;
  uint64_t page_size = kDefaultPageSize;
  uint64_t settle_ms = 500;
  bool clients_explicit = false;
  bool scheduling_explicit = false;
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
  --tier dram|hbm|ssd  --backends-per-peer N  --backend-capacity SIZE
  --page-size SIZE  --placement single|weighted  --placement-weights LIST
  --ssd-dir-prefix PATH  --put-strategy most_available|random
  --affinity none|same|local  --get-strategy local|random
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
  for (const auto& suffix :
       {std::pair<std::string_view, uint64_t>{"GiB", 1ULL << 30},
        {"MiB", 1ULL << 20}, {"KiB", 1ULL << 10}, {"B", 1}}) {
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
std::vector<uint32_t> ParseWeights(std::string_view text) {
  std::vector<uint32_t> weights;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const auto item = text.substr(0, comma);
    const uint32_t weight = ParseUnsigned<uint32_t>(item, "--placement-weights");
    if (weight == 0) UsageError("--placement-weights entries must be positive");
    weights.push_back(weight);
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
    if (text.empty()) UsageError("--placement-weights must not end with a comma");
  }
  if (weights.empty()) UsageError("--placement-weights must not be empty");
  return weights;
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
  if (options.command != "run" && options.command != "generate" &&
      options.command != "replay") {
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
    } else if (name == "--tier") {
      options.tier = value();
    } else if (name == "--backends-per-peer") {
      options.backends_per_peer = ParseUnsigned<uint32_t>(value(), "--backends-per-peer");
    } else if (name == "--backend-capacity") {
      options.backend_capacity = ParseSize(value(), "--backend-capacity");
    } else if (name == "--page-size") {
      options.page_size = ParseSize(value(), "--page-size");
    } else if (name == "--placement") {
      options.placement = value();
    } else if (name == "--placement-weights") {
      options.placement_weights = ParseWeights(value());
    } else if (name == "--ssd-dir-prefix") {
      options.ssd_dir_prefix = value();
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
TierType ParseTier(const std::string& name) {
  if (name == "dram") return TierType::DRAM;
  if (name == "hbm") return TierType::HBM;
  if (name == "ssd") return TierType::SSD;
  UsageError("invalid --tier: " + name);
}
void ValidateOptions(Options* options) {
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
  if (options->affinity != "none" && options->affinity != "same" &&
      options->affinity != "local") {
    UsageError("--affinity must be none, same, or local");
  }
  if (options->get_strategy != "local" && options->get_strategy != "random") {
    UsageError("--get-strategy must be local or random");
  }
  if (options->placement != "single" && options->placement != "weighted") {
    UsageError("--placement must be single or weighted");
  }
  ParseTier(options->tier);
  if (options->backends_per_peer == 0 ||
      options->backends_per_peer > mori::umbp::kMaxBackendsPerPeer) {
    UsageError("--backends-per-peer is out of range");
  }
  if (!options->placement_weights.empty() &&
      options->placement_weights.size() != options->backends_per_peer) {
    UsageError("--placement-weights must have one entry per backend");
  }
  if (options->backend_capacity < options->page_size || options->page_size == 0) {
    UsageError("--backend-capacity must be at least the positive --page-size");
  }
  if (options->tier == "ssd" && options->command != "replay" &&
      options->workload.max_value_size > options->page_size) {
    UsageError("SSD values must not exceed --page-size");
  }
}
void AddTraceMetadata(const Options& options,
                      ::umbp::benchmark::WorkloadTraceHeader* header) {
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
  std::cout << "trace_path,events,seed,schema_version\n\"" << options.trace_path << "\","
            << events << ',' << options.workload.seed << ','
            << bench::kWorkloadTraceSchemaVersion << '\n';
  return 0;
}
class MasterHarness {
 public:
  explicit MasterHarness(const Options& options) {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.registry_config.default_dram_page_size = options.page_size;
    const auto algorithm =
        options.put_strategy == "random"
            ? ConfigurableRoutePutStrategy::SelectAlgo::kRandom
            : ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable;
    auto affinity = ConfigurableRoutePutStrategy::NodeAffinity::kNone;
    if (options.affinity == "same") {
      affinity = ConfigurableRoutePutStrategy::NodeAffinity::kSame;
    } else if (options.affinity == "local") {
      affinity = ConfigurableRoutePutStrategy::NodeAffinity::kLocal;
    }
    config.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(
        algorithm, affinity, options.workload.seed);
    config.route_put_algo = options.put_strategy;
    config.route_put_affinity = options.affinity;
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
  std::string Address() const {
    return "127.0.0.1:" + std::to_string(server_->GetBoundPort());
  }

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
BackendInstanceConfig MakeBackend(const Options& options, uint32_t node,
                                  uint32_t index) {
  BackendInstanceConfig backend;
  backend.name = options.tier + "-" + std::to_string(index);
  backend.tier = ParseTier(options.tier);
  backend.placement_weight =
      options.placement_weights.empty() ? 1 : options.placement_weights[index];
  if (backend.tier == TierType::DRAM) {
    backend.dram.buffer_sizes = {options.backend_capacity};
  } else if (backend.tier == TierType::HBM) {
    backend.hbm.buffer_sizes = {options.backend_capacity};
  } else {
    backend.ssd.enabled = true;
    backend.ssd.ssd.enabled = true;
    backend.ssd.ssd.capacity_bytes = options.backend_capacity;
    backend.ssd.ssd.storage_dir = options.ssd_dir_prefix + "-node-" +
                                  std::to_string(node) + "-backend-" +
                                  std::to_string(index);
    backend.ssd.ssd.ssd_backend = "file";
  }
  return backend;
}
std::vector<std::unique_ptr<PoolClient>> StartClients(
    const Options& options, const std::string& master_address) {
  std::vector<std::unique_ptr<PoolClient>> clients;
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
    config.placement_policy = options.placement == "weighted"
                                  ? PoolPlacementPolicy::WEIGHTED
                                  : PoolPlacementPolicy::SINGLE_BACKEND;
    for (uint32_t i = 0; i < options.backends_per_peer; ++i) {
      config.backends.push_back(MakeBackend(options, node, i));
    }
    auto client = std::make_unique<PoolClient>(std::move(config));
    if (!client->Init()) {
      throw std::runtime_error("failed to initialize PoolClient " + std::to_string(node));
    }
    clients.push_back(std::move(client));
  }
  return clients;
}
class PoolWorkloadClient final : public bench::WorkloadClient {
 public:
  explicit PoolWorkloadClient(std::vector<std::unique_ptr<PoolClient>>* clients)
      : clients_(clients) {}

  bench::ClientResult Put(uint32_t id, const std::string& key, const uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(id);
    if (client == nullptr || !client->Put(key, data, size)) {
      return bench::ClientResult::kFailed;
    }
    Published(key);
    client->Master().FlushHeartbeat();
    return bench::ClientResult::kSuccess;
  }

  bench::ClientResult Get(uint32_t id, const std::string& key, uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(id);
    if (client == nullptr) return bench::ClientResult::kFailed;
    if (client->Get(key, data, size)) {
      Clear(key);
      return bench::ClientResult::kSuccess;
    }
    const auto deadline = Deadline(key);
    while (deadline > std::chrono::steady_clock::now()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (client->Get(key, data, size)) {
        Clear(key);
        return bench::ClientResult::kSuccess;
      }
    }
    Clear(key);
    return client->Exists(key) ? bench::ClientResult::kFailed
                               : bench::ClientResult::kNotFound;
  }

  std::vector<bench::ClientResult> BatchPut(
      uint32_t id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    PoolClient* client = Select(id);
    if (client == nullptr || keys.size() != values.size()) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }
    std::vector<const void*> pointers;
    std::vector<size_t> sizes;
    for (const auto& value : values) {
      pointers.push_back(value.data());
      sizes.push_back(value.size());
    }
    const auto results = client->BatchPut(keys, pointers, sizes);
    std::vector<bench::ClientResult> converted(keys.size(), bench::ClientResult::kFailed);
    for (size_t i = 0; i < std::min(keys.size(), results.size()); ++i) {
      if (results[i]) {
        converted[i] = bench::ClientResult::kSuccess;
        Published(keys[i]);
      }
    }
    client->Master().FlushHeartbeat();
    return converted;
  }

  std::vector<bench::ClientResult> BatchGet(
      uint32_t id, const std::vector<std::string>& keys,
      const std::vector<size_t>& sizes,
      std::vector<std::vector<uint8_t>>* values) override {
    PoolClient* client = Select(id);
    if (client == nullptr || keys.size() != sizes.size() || values == nullptr) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }
    values->resize(keys.size());
    std::vector<void*> pointers;
    for (size_t i = 0; i < keys.size(); ++i) {
      (*values)[i].resize(sizes[i]);
      pointers.push_back((*values)[i].data());
    }
    const auto results = client->BatchGet(keys, pointers, sizes);
    std::vector<bench::ClientResult> converted(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i < results.size() && results[i]) {
        Clear(keys[i]);
        converted[i] = bench::ClientResult::kSuccess;
      } else {
        converted[i] = Get(id, keys[i], (*values)[i].data(), sizes[i]);
      }
    }
    return converted;
  }

 private:
  PoolClient* Select(uint32_t id) const {
    return clients_ != nullptr && id < clients_->size() ? (*clients_)[id].get() : nullptr;
  }
  void Published(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_[key] = std::chrono::steady_clock::now() + kPublicationTimeout;
  }
  std::chrono::steady_clock::time_point Deadline(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = pending_.find(key);
    return found == pending_.end() ? std::chrono::steady_clock::now() : found->second;
  }
  void Clear(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.erase(key);
  }
  std::vector<std::unique_ptr<PoolClient>>* clients_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_;
};
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
std::string Weights(const Options& options) {
  std::string result;
  for (uint32_t i = 0; i < options.backends_per_peer; ++i) {
    if (!result.empty()) result += ':';
    result += std::to_string(options.placement_weights.empty()
                                 ? 1
                                 : options.placement_weights[i]);
  }
  return result;
}
void PrintSummary(const Options& options, const bench::WorkloadMetrics& metrics,
                  uint64_t seed,
                  uint64_t wall_ns) {
  const double seconds = static_cast<double>(wall_ns) / 1e9;
  const double ops_per_second = seconds == 0 ? 0 : metrics.total.succeeded / seconds;
  const double gib_per_second =
      seconds == 0 ? 0 : metrics.total.succeeded_bytes / static_cast<double>(1ULL << 30) /
                                seconds;
  std::cout
      << "config\n"
      << "command,trace,seed,profile,operations,keys,min_value_bytes,max_value_bytes,"
         "size_distribution,read_ratio,clients,batch,qps,tier,"
         "backends_per_peer,backend_capacity,page_size,placement,weights,"
         "put_strategy,affinity,get_strategy,scheduling,time_scale,payload_validation,"
         "settle_ms\n"
      << Csv(options.command) << ',' << Csv(options.trace_path) << ',' << seed << ','
      << Csv(options.profile) << ',' << options.workload.operation_count << ','
      << options.workload.key_count << ',' << options.workload.min_value_size << ','
      << options.workload.max_value_size << ',' << Csv(options.size_distribution) << ','
      << options.workload.read_ratio << ',' << options.workload.client_count << ','
      << options.workload.batch_size << ',' << options.workload.qps << ','
      << Csv(options.tier) << ',' << options.backends_per_peer << ','
      << options.backend_capacity << ',' << options.page_size << ','
      << Csv(options.placement) << ',' << Csv(Weights(options)) << ','
      << Csv(options.put_strategy) << ',' << Csv(options.affinity) << ','
      << Csv(options.get_strategy) << ','
      << Csv(options.runner.max_throughput ? "max-throughput" : "open-loop") << ','
      << options.runner.time_scale << ','
      << (options.runner.validate_get_payloads ? "true" : "false") << ','
      << options.settle_ms << '\n'
      << "summary\n"
      << "seed,attempted,succeeded,failed,succeeded_bytes,puts,gets,get_misses,"
         "validation_failures,wall_time_ns,ops_per_s,GiB_per_s,latency_p50_ns,"
         "latency_p95_ns,latency_p99_ns,latency_max_ns,schedule_lag_p99_ns\n"
      << seed << ',' << metrics.total.attempted << ',' << metrics.total.succeeded << ','
      << metrics.total.failed << ',' << metrics.total.succeeded_bytes << ','
      << metrics.puts.succeeded << ',' << metrics.gets.succeeded << ','
      << metrics.get_misses << ',' << metrics.get_validation_failures << ',' << wall_ns
      << ',' << std::setprecision(10) << ops_per_second << ',' << gib_per_second << ','
      << metrics.latency.p50_ns << ',' << metrics.latency.p95_ns << ','
      << metrics.latency.p99_ns << ',' << metrics.latency.max_ns << ','
      << metrics.schedule_lag.p99_ns << '\n';
}
void PrintBackendPlacement(const std::vector<std::unique_ptr<PoolClient>>& clients) {
  std::cout << "backend_placement\n"
            << "node,backend,tier,owned_keys,total_bytes,available_bytes,"
               "max_allocatable_bytes\n";
  for (const auto& client : clients) {
    auto& registry = client->Backends();
    for (MediumBackend* backend : registry.All()) {
      const auto* entry = registry.GetEntry(backend);
      const auto capacity = backend->Capacity();
      std::cout << Csv(client->NodeId()) << ','
                << Csv(entry == nullptr ? backend->Name() : entry->name) << ','
                << Csv(mori::umbp::TierTypeName(backend->Tier())) << ','
                << backend->OwnedKeyCount() << ',' << capacity.total_bytes << ','
                << capacity.available_bytes << ',' << capacity.max_allocatable_bytes << '\n';
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
    if (options->tier == "ssd" && event.value_size() > options->page_size) {
      UsageError("trace contains an SSD value larger than --page-size");
    }
  }
  options->workload.operation_count = event_count;
  if (!options->clients_explicit &&
      metadata.find("clients") == metadata.end() && has_events) {
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
  PoolWorkloadClient adapter(&clients);
  Settle(clients, options.settle_ms);
  bench::WorkloadRunner runner(&adapter, options.runner);
  const auto start = std::chrono::steady_clock::now();
  const auto metrics = runner.Run(source.get());
  const uint64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
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
    ValidateOptions(&options);
    return options.command == "generate" ? GenerateTrace(options)
                                         : RunBenchmark(std::move(options));
  } catch (const std::exception& error) {
    std::cerr << "umbp_tier_bench: " << error.what() << '\n';
    return 1;
  }
}
