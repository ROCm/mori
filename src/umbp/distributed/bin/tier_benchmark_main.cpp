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
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

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

constexpr uint64_t kDefaultCapacity = 256ULL * 1024 * 1024;
constexpr uint64_t kDefaultPageSize = 2ULL * 1024 * 1024;
constexpr auto kServerStartTimeout = std::chrono::seconds(10);

struct Options {
  std::string command;
  std::string trace_path;

  bench::SyntheticWorkloadConfig workload;
  std::string profile = "mixed";
  std::string size_distribution = "fixed";
  bool clients_explicit = false;
  double duration_sec = 0.0;
  bool duration_explicit = false;

  std::string put_strategy = "most_available";
  std::string affinity = "none";
  std::string get_strategy = "local";
  std::string placement = "single";
  std::string tier = "dram";
  uint32_t backends_per_peer = 1;
  std::vector<uint32_t> placement_weights;
  uint64_t backend_capacity = kDefaultCapacity;
  uint64_t page_size = kDefaultPageSize;
  std::string ssd_dir_prefix = "/tmp/umbp-tier-benchmark";

  bench::WorkloadRunnerOptions runner;
  bool scheduling_explicit = false;
  uint64_t warmup_operations = 0;
  uint64_t settle_ms = 500;
  uint16_t metrics_port = 0;
  uint64_t publication_timeout_ms = 2000;
  double eviction_high_watermark = 0.9;
  double eviction_low_watermark = 0.7;
  uint32_t eviction_check_sec = 1;
  uint32_t eviction_lease_sec = 2;
  size_t eviction_batch_size = 32;
};

[[noreturn]] void UsageError(const std::string& message) {
  throw std::invalid_argument(message + " (use --help for usage)");
}

void PrintHelp(std::ostream& out) {
  out << R"(UMBP distributed tier-management benchmark

Usage:
  umbp_tier_bench generate --trace PATH [workload options]
  umbp_tier_bench run [workload, cluster, and scheduling options]
  umbp_tier_bench replay --trace PATH [cluster and scheduling options]

Workload options:
  --profile NAME              sequential|uniform|hotset|read-after-write|mixed|
                              capacity-pressure (default: mixed)
  --seed N                    Generator/payload seed (default: 1)
  --operations N              Measured/generated operations (default: 1000)
  --duration-sec R            Derive operations as ceil(duration*QPS)
  --keys N                    Key count (default: 100)
  --min-value-bytes SIZE      Minimum value size (default: 4096)
  --max-value-bytes SIZE      Maximum value size (default: 4096)
  --size-distribution NAME    fixed|uniform|log-uniform (default: fixed)
  --read-ratio R              Fraction in [0,1] (default: 0.5)
  --clients N                 In-process PoolClients (default: 1)
  --batch N                   Operations per client batch (default: 1)
  --qps Q                     Aggregate open-loop generator QPS; 0 means timestamps 0
  --hotset-fraction R         Fraction in (0,1] (default: 0.1)
  --zipf-exponent E           Positive exponent (default: 1.1)

Cluster options (run/replay):
  --put-strategy NAME         most_available|random (default: most_available)
  --affinity NAME             none|same|local (default: none)
  --get-strategy NAME         local|random (default: local)
  --placement NAME            single|weighted (default: single)
  --backends-per-peer N       Same-tier backend instances per peer (default: 1)
  --placement-weights LIST    Comma-separated positive weights; one per backend
  --tier NAME                 dram|hbm|ssd (default: dram)
  --backend-capacity SIZE     Capacity of each backend (default: 256MiB)
  --page-size SIZE            Backend page size (default: 2MiB)
  --ssd-dir-prefix PATH       Per-node/backend SSD directory prefix

Scheduling options (run/replay):
  --max-throughput            Ignore event timestamps (default for run)
  --open-loop                 Honor event timestamps (default for replay)
  --time-scale R              Positive timestamp multiplier (default: 1)
  --payload-validation BOOL   Validate successful GET payloads (default: true)
  --no-payload-validation     Disable GET payload validation
  --warmup-operations N       Separate unmeasured synthetic warmup (default: 0)
  --settle-ms N               Registration/post-run heartbeat settle (default: 500)
  --metrics-port N            Expose/scrape Master Prometheus metrics; 0 disables
  --publication-timeout-ms N  Await visibility for keys put by this run (default: 2000)
  --window-ms N               Emit per-window CSV counters; 0 disables
  --eviction-high-watermark R Master LRU trigger ratio (default: 0.9)
  --eviction-low-watermark R  Master LRU target ratio (default: 0.7)
  --eviction-check-sec N       Master eviction interval (default: 1)
  --eviction-lease-sec N       Read lease duration (default: 2)
  --eviction-batch-size N      Victims selected per round (default: 32)

SIZE accepts an optional B, KiB, MiB, or GiB suffix. Both --name=value and
--name value forms are accepted.
)";
}

template <typename UInt>
UInt ParseUnsigned(std::string_view text, const char* option) {
  static_assert(std::is_unsigned_v<UInt>);
  if (text.empty() || text.front() == '-') UsageError(std::string(option) + " expects an integer");
  uint64_t value = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      value > static_cast<uint64_t>(std::numeric_limits<UInt>::max())) {
    UsageError(std::string(option) + " has an invalid integer: " + std::string(text));
  }
  return static_cast<UInt>(value);
}

double ParseDouble(std::string_view text, const char* option) {
  std::string owned(text);
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &end);
  if (errno == ERANGE || end == owned.c_str() || *end != '\0' || !std::isfinite(value)) {
    UsageError(std::string(option) + " has an invalid number: " + owned);
  }
  return value;
}

bool ParseBool(std::string_view text, const char* option) {
  if (text == "true" || text == "1" || text == "yes" || text == "on") return true;
  if (text == "false" || text == "0" || text == "no" || text == "off") return false;
  UsageError(std::string(option) + " expects true or false");
}

uint64_t ParseSize(std::string_view text, const char* option) {
  uint64_t multiplier = 1;
  auto strip_suffix = [&](std::string_view suffix, uint64_t scale) {
    if (text.size() >= suffix.size() &&
        text.substr(text.size() - suffix.size()) == suffix) {
      text.remove_suffix(suffix.size());
      multiplier = scale;
      return true;
    }
    return false;
  };
  if (!strip_suffix("GiB", 1ULL << 30) && !strip_suffix("MiB", 1ULL << 20) &&
      !strip_suffix("KiB", 1ULL << 10)) {
    strip_suffix("B", 1);
  }
  const uint64_t value = ParseUnsigned<uint64_t>(text, option);
  if (value > std::numeric_limits<uint64_t>::max() / multiplier) {
    UsageError(std::string(option) + " is too large");
  }
  return value * multiplier;
}

std::vector<uint32_t> ParseWeights(std::string_view text) {
  if (text.empty()) UsageError("--placement-weights must not be empty");
  std::vector<uint32_t> result;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t comma = text.find(',', begin);
    const auto item = text.substr(begin, comma == std::string_view::npos ? text.size() - begin
                                                                        : comma - begin);
    const uint32_t weight = ParseUnsigned<uint32_t>(item, "--placement-weights");
    if (weight == 0) UsageError("--placement-weights entries must be positive");
    result.push_back(weight);
    if (comma == std::string_view::npos) break;
    begin = comma + 1;
  }
  return result;
}

std::pair<std::string, std::string> SplitOption(const std::string& argument) {
  const size_t equals = argument.find('=');
  if (equals == std::string::npos) return {argument, {}};
  return {argument.substr(0, equals), argument.substr(equals + 1)};
}

Options ParseOptions(int argc, char** argv) {
  if (argc < 2) UsageError("missing subcommand");
  Options options;
  options.runner.max_throughput = true;
  options.command = argv[1];
  if (options.command == "--help" || options.command == "-h" || options.command == "help") {
    PrintHelp(std::cout);
    std::exit(0);
  }
  if (options.command != "run" && options.command != "generate" &&
      options.command != "replay") {
    UsageError("unknown subcommand: " + options.command);
  }

  for (int i = 2; i < argc; ++i) {
    auto [name, inline_value] = SplitOption(argv[i]);
    if (name == "--help" || name == "-h") {
      PrintHelp(std::cout);
      std::exit(0);
    }
    auto value = [&]() -> std::string {
      if (argv[i][name.size()] == '=') {
        if (inline_value.empty()) UsageError(name + " requires a value");
        return inline_value;
      }
      if (i + 1 >= argc) UsageError(name + " requires a value");
      return argv[++i];
    };

    if (name == "--max-throughput") {
      options.runner.max_throughput = true;
      options.scheduling_explicit = true;
    } else if (name == "--open-loop") {
      options.runner.max_throughput = false;
      options.scheduling_explicit = true;
    } else if (name == "--no-payload-validation") {
      options.runner.validate_get_payloads = false;
    } else if (name == "--trace") {
      options.trace_path = value();
    } else if (name == "--profile") {
      options.profile = value();
    } else if (name == "--seed") {
      options.workload.seed = ParseUnsigned<uint64_t>(value(), "--seed");
    } else if (name == "--operations") {
      options.workload.operation_count = ParseUnsigned<uint64_t>(value(), "--operations");
    } else if (name == "--duration-sec") {
      options.duration_sec = ParseDouble(value(), "--duration-sec");
      options.duration_explicit = true;
    } else if (name == "--keys") {
      options.workload.key_count = ParseUnsigned<uint64_t>(value(), "--keys");
    } else if (name == "--min-value-bytes") {
      options.workload.min_value_size =
          static_cast<size_t>(ParseSize(value(), "--min-value-bytes"));
    } else if (name == "--max-value-bytes") {
      options.workload.max_value_size =
          static_cast<size_t>(ParseSize(value(), "--max-value-bytes"));
    } else if (name == "--size-distribution" || name == "--value-size-distribution") {
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
    } else if (name == "--hotset-fraction") {
      options.workload.hotset_fraction = ParseDouble(value(), "--hotset-fraction");
    } else if (name == "--zipf-exponent") {
      options.workload.zipf_exponent = ParseDouble(value(), "--zipf-exponent");
    } else if (name == "--put-strategy") {
      options.put_strategy = value();
    } else if (name == "--affinity") {
      options.affinity = value();
    } else if (name == "--get-strategy") {
      options.get_strategy = value();
    } else if (name == "--placement") {
      options.placement = value();
    } else if (name == "--backends-per-peer") {
      options.backends_per_peer =
          ParseUnsigned<uint32_t>(value(), "--backends-per-peer");
    } else if (name == "--placement-weights") {
      options.placement_weights = ParseWeights(value());
    } else if (name == "--tier") {
      options.tier = value();
    } else if (name == "--backend-capacity" || name == "--backend-capacity-bytes") {
      options.backend_capacity = ParseSize(value(), "--backend-capacity");
    } else if (name == "--page-size") {
      options.page_size = ParseSize(value(), "--page-size");
    } else if (name == "--ssd-dir-prefix") {
      options.ssd_dir_prefix = value();
    } else if (name == "--time-scale") {
      options.runner.time_scale = ParseDouble(value(), "--time-scale");
    } else if (name == "--payload-validation" || name == "--validate-payloads") {
      options.runner.validate_get_payloads = ParseBool(value(), name.c_str());
    } else if (name == "--warmup-operations") {
      options.warmup_operations = ParseUnsigned<uint64_t>(value(), "--warmup-operations");
    } else if (name == "--settle-ms") {
      options.settle_ms = ParseUnsigned<uint64_t>(value(), "--settle-ms");
    } else if (name == "--metrics-port") {
      options.metrics_port = ParseUnsigned<uint16_t>(value(), "--metrics-port");
    } else if (name == "--publication-timeout-ms") {
      options.publication_timeout_ms =
          ParseUnsigned<uint64_t>(value(), "--publication-timeout-ms");
    } else if (name == "--window-ms") {
      const uint64_t milliseconds = ParseUnsigned<uint64_t>(value(), "--window-ms");
      if (milliseconds > std::numeric_limits<uint64_t>::max() / 1000000ULL) {
        UsageError("--window-ms is too large");
      }
      options.runner.window_ns = milliseconds * 1000000ULL;
    } else if (name == "--eviction-high-watermark") {
      options.eviction_high_watermark =
          ParseDouble(value(), "--eviction-high-watermark");
    } else if (name == "--eviction-low-watermark") {
      options.eviction_low_watermark =
          ParseDouble(value(), "--eviction-low-watermark");
    } else if (name == "--eviction-check-sec") {
      options.eviction_check_sec =
          ParseUnsigned<uint32_t>(value(), "--eviction-check-sec");
    } else if (name == "--eviction-lease-sec") {
      options.eviction_lease_sec =
          ParseUnsigned<uint32_t>(value(), "--eviction-lease-sec");
    } else if (name == "--eviction-batch-size") {
      options.eviction_batch_size =
          ParseUnsigned<size_t>(value(), "--eviction-batch-size");
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

bench::ValueSizeDistribution ParseSizeDistribution(const std::string& name) {
  if (name == "fixed") return bench::ValueSizeDistribution::kFixed;
  if (name == "uniform") return bench::ValueSizeDistribution::kUniform;
  if (name == "log-uniform") return bench::ValueSizeDistribution::kLogUniform;
  UsageError("invalid --size-distribution: " + name);
}

TierType ParseTier(const std::string& name) {
  if (name == "dram") return TierType::DRAM;
  if (name == "hbm") return TierType::HBM;
  if (name == "ssd") return TierType::SSD;
  UsageError("invalid --tier: " + name);
}

void ValidateOptions(Options* options) {
  options->workload.profile = ParseProfile(options->profile);
  options->workload.value_size_distribution =
      ParseSizeDistribution(options->size_distribution);

  if ((options->command == "generate" || options->command == "replay") &&
      options->trace_path.empty()) {
    UsageError(options->command + " requires --trace");
  }
  if (options->workload.client_count == 0) UsageError("--clients must be positive");
  if (options->workload.key_count == 0) UsageError("--keys must be positive");
  if (options->workload.batch_size == 0) UsageError("--batch must be positive");
  if (options->workload.min_value_size == 0) {
    UsageError("--min-value-bytes must be positive");
  }
  if (options->workload.min_value_size > options->workload.max_value_size) {
    UsageError("--min-value-bytes must not exceed --max-value-bytes");
  }
  if (options->workload.read_ratio < 0.0 || options->workload.read_ratio > 1.0) {
    UsageError("--read-ratio must be in [0,1]");
  }
  if (options->workload.qps < 0.0) UsageError("--qps must be non-negative");
  if (options->duration_sec < 0.0) UsageError("--duration-sec must be non-negative");
  if (options->command == "replay" && options->duration_explicit) {
    UsageError("--duration-sec is only valid for run/generate");
  }
  if (options->command != "replay" && options->duration_sec > 0.0) {
    if (options->workload.qps <= 0.0) {
      UsageError("--duration-sec requires a positive --qps");
    }
    const long double operations =
        std::ceil(static_cast<long double>(options->duration_sec) * options->workload.qps);
    if (operations > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
      UsageError("--duration-sec and --qps produce too many operations");
    }
    options->workload.operation_count = static_cast<uint64_t>(operations);
  }
  if (options->workload.hotset_fraction <= 0.0 ||
      options->workload.hotset_fraction > 1.0) {
    UsageError("--hotset-fraction must be in (0,1]");
  }
  if (options->workload.zipf_exponent <= 0.0) {
    UsageError("--zipf-exponent must be positive");
  }
  if (options->runner.time_scale <= 0.0) UsageError("--time-scale must be positive");
  if (!(options->eviction_low_watermark > 0.0 &&
        options->eviction_low_watermark < options->eviction_high_watermark &&
        options->eviction_high_watermark <= 1.0)) {
    UsageError("eviction watermarks must satisfy 0 < low < high <= 1");
  }
  if (options->eviction_check_sec == 0) UsageError("--eviction-check-sec must be positive");
  if (options->eviction_batch_size == 0) UsageError("--eviction-batch-size must be positive");

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
    UsageError("--backends-per-peer must be in [1, " +
               std::to_string(mori::umbp::kMaxBackendsPerPeer) + "]");
  }
  if (!options->placement_weights.empty() &&
      options->placement_weights.size() != options->backends_per_peer) {
    UsageError("--placement-weights must contain exactly --backends-per-peer entries");
  }
  if (options->backend_capacity == 0) UsageError("--backend-capacity must be positive");
  if (options->page_size == 0) UsageError("--page-size must be positive");
  if (options->backend_capacity < options->page_size) {
    UsageError("--backend-capacity must be at least --page-size");
  }
  if (options->tier == "ssd" && options->workload.max_value_size > options->page_size &&
      options->command != "replay") {
    UsageError("SSD values must not exceed --page-size");
  }
  if (options->tier == "ssd" && options->ssd_dir_prefix.empty()) {
    UsageError("--ssd-dir-prefix must not be empty");
  }
}

std::string BoolString(bool value) { return value ? "true" : "false"; }

std::string WeightsString(const Options& options) {
  std::ostringstream out;
  for (uint32_t i = 0; i < options.backends_per_peer; ++i) {
    if (i != 0) out << ',';
    out << (options.placement_weights.empty() ? 1 : options.placement_weights[i]);
  }
  return out.str();
}

void AddGenerationMetadata(const Options& options,
                           ::umbp::benchmark::WorkloadTraceHeader* header) {
  auto& metadata = *header->mutable_metadata();
  metadata["profile"] = options.profile;
  metadata["seed"] = std::to_string(options.workload.seed);
  metadata["operations"] = std::to_string(options.workload.operation_count);
  metadata["duration_sec"] = std::to_string(options.duration_sec);
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
  header.set_time_unit(
      ::umbp::benchmark::WorkloadTraceHeader::TIME_UNIT_NANOSECONDS);
  header.set_seed(options.workload.seed);
  AddGenerationMetadata(options, &header);

  bench::SyntheticWorkloadSource source(options.workload);
  bench::TraceWriter writer(options.trace_path, header);
  ::umbp::benchmark::WorkloadEvent event;
  uint64_t count = 0;
  while (source.Next(&event)) {
    writer.Write(event);
    ++count;
  }
  writer.Close();
  std::cout << "trace_path,events,seed,schema_version\n"
            << '"' << options.trace_path << "\"," << count << ',' << options.workload.seed << ','
            << bench::kWorkloadTraceSchemaVersion << '\n';
  return 0;
}

ConfigurableRoutePutStrategy::SelectAlgo PutAlgo(const Options& options) {
  return options.put_strategy == "random"
             ? ConfigurableRoutePutStrategy::SelectAlgo::kRandom
             : ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable;
}

ConfigurableRoutePutStrategy::NodeAffinity PutAffinity(const Options& options) {
  if (options.affinity == "same") return ConfigurableRoutePutStrategy::NodeAffinity::kSame;
  if (options.affinity == "local") return ConfigurableRoutePutStrategy::NodeAffinity::kLocal;
  return ConfigurableRoutePutStrategy::NodeAffinity::kNone;
}

class MasterHarness {
 public:
  explicit MasterHarness(const Options& options) {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.metrics_port = options.metrics_port;
    config.registry_config.default_dram_page_size = options.page_size;
    config.eviction_config.high_watermark = options.eviction_high_watermark;
    config.eviction_config.low_watermark = options.eviction_low_watermark;
    config.eviction_config.check_interval =
        std::chrono::seconds(options.eviction_check_sec);
    config.eviction_config.lease_duration =
        std::chrono::seconds(options.eviction_lease_sec);
    config.eviction_config.evict_batch_size = options.eviction_batch_size;
    config.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(
        PutAlgo(options), PutAffinity(options), options.workload.seed);
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
  MasterHarness(const MasterHarness&) = delete;
  MasterHarness& operator=(const MasterHarness&) = delete;

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

BackendInstanceConfig MakeBackend(const Options& options, uint32_t node_index,
                                  uint32_t backend_index) {
  BackendInstanceConfig backend;
  backend.name = options.tier + "-" + std::to_string(backend_index);
  backend.tier = ParseTier(options.tier);
  backend.placement_weight =
      options.placement_weights.empty() ? 1 : options.placement_weights[backend_index];
  if (backend.tier == TierType::DRAM) {
    backend.dram.buffer_sizes = {options.backend_capacity};
  } else if (backend.tier == TierType::HBM) {
    backend.hbm.buffer_sizes = {options.backend_capacity};
  } else {
    backend.ssd.enabled = true;
    backend.ssd.ssd.enabled = true;
    backend.ssd.ssd.capacity_bytes = options.backend_capacity;
    backend.ssd.ssd.storage_dir = options.ssd_dir_prefix + "-node-" +
                                  std::to_string(node_index) + "-backend-" +
                                  std::to_string(backend_index);
    backend.ssd.ssd.ssd_backend = "file";
  }
  return backend;
}

std::vector<std::unique_ptr<PoolClient>> StartClients(const Options& options,
                                                       const std::string& master_address) {
  std::vector<std::unique_ptr<PoolClient>> clients;
  clients.reserve(options.workload.client_count);
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
    for (uint32_t backend = 0; backend < options.backends_per_peer; ++backend) {
      config.backends.push_back(MakeBackend(options, node, backend));
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
  PoolWorkloadClient(std::vector<std::unique_ptr<PoolClient>>* clients,
                     uint64_t publication_timeout_ms)
      : clients_(clients), publication_timeout_ms_(publication_timeout_ms) {}

  bench::ClientResult Put(uint32_t client_id, const std::string& key, const uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr) return bench::ClientResult::kFailed;
    if (!client->Put(key, data, size)) return bench::ClientResult::kFailed;
    MarkPublicationPending(key);
    client->Master().FlushHeartbeat();
    return bench::ClientResult::kSuccess;
  }

  bench::ClientResult Get(uint32_t client_id, const std::string& key, uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr) return bench::ClientResult::kFailed;
    if (client->Get(key, data, size)) {
      ClearPublicationPending(key);
      return bench::ClientResult::kSuccess;
    }
    if (IsPublicationPending(key)) {
      const auto deadline = PublicationDeadline(key);
      while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (client->Get(key, data, size)) {
          ClearPublicationPending(key);
          return bench::ClientResult::kSuccess;
        }
      }
      ClearPublicationPending(key);
    }
    return client->Exists(key) ? bench::ClientResult::kFailed
                               : bench::ClientResult::kNotFound;
  }

  std::vector<bench::ClientResult> BatchPut(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr || keys.size() != values.size()) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }
    std::vector<const void*> pointers;
    std::vector<size_t> sizes;
    pointers.reserve(values.size());
    sizes.reserve(values.size());
    for (const auto& value : values) {
      pointers.push_back(value.data());
      sizes.push_back(value.size());
    }
    const auto results = client->BatchPut(keys, pointers, sizes);
    std::vector<bench::ClientResult> converted(keys.size(), bench::ClientResult::kFailed);
    for (size_t i = 0; i < std::min(results.size(), converted.size()); ++i) {
      if (results[i]) {
        converted[i] = bench::ClientResult::kSuccess;
        MarkPublicationPending(keys[i]);
      }
    }
    client->Master().FlushHeartbeat();
    return converted;
  }

  std::vector<bench::ClientResult> BatchGet(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<size_t>& sizes,
      std::vector<std::vector<uint8_t>>* values) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr || keys.size() != sizes.size() || values == nullptr) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }
    values->clear();
    values->resize(keys.size());
    std::vector<void*> pointers;
    pointers.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      (*values)[i].resize(sizes[i]);
      pointers.push_back((*values)[i].data());
    }
    auto results = client->BatchGet(keys, pointers, sizes);
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i < results.size() && results[i]) {
        ClearPublicationPending(keys[i]);
        continue;
      }
      if (IsPublicationPending(keys[i])) {
        const auto deadline = PublicationDeadline(keys[i]);
        while (std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          if (client->Get(keys[i], (*values)[i].data(), sizes[i])) {
            if (i >= results.size()) results.resize(keys.size(), false);
            results[i] = true;
            break;
          }
        }
        ClearPublicationPending(keys[i]);
      }
    }
    std::vector<bench::ClientResult> converted(keys.size(), bench::ClientResult::kFailed);
    std::vector<std::string> failed_keys;
    std::vector<size_t> failed_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i < results.size() && results[i]) {
        converted[i] = bench::ClientResult::kSuccess;
      } else {
        failed_keys.push_back(keys[i]);
        failed_indices.push_back(i);
      }
    }
    if (!failed_keys.empty()) {
      const auto exists = client->BatchExists(failed_keys);
      for (size_t i = 0; i < failed_indices.size(); ++i) {
        if (i >= exists.size() || !exists[i]) {
          converted[failed_indices[i]] = bench::ClientResult::kNotFound;
        }
      }
    }
    return converted;
  }

 private:
  void MarkPublicationPending(const std::string& key) {
    std::lock_guard<std::mutex> lock(put_keys_mutex_);
    pending_publications_[key] =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(publication_timeout_ms_);
  }

  bool IsPublicationPending(const std::string& key) {
    std::lock_guard<std::mutex> lock(put_keys_mutex_);
    const auto found = pending_publications_.find(key);
    if (found == pending_publications_.end()) return false;
    if (std::chrono::steady_clock::now() >= found->second) {
      pending_publications_.erase(found);
      return false;
    }
    return true;
  }

  std::chrono::steady_clock::time_point PublicationDeadline(
      const std::string& key) const {
    std::lock_guard<std::mutex> lock(put_keys_mutex_);
    const auto found = pending_publications_.find(key);
    return found == pending_publications_.end() ? std::chrono::steady_clock::now()
                                                : found->second;
  }

  void ClearPublicationPending(const std::string& key) {
    std::lock_guard<std::mutex> lock(put_keys_mutex_);
    pending_publications_.erase(key);
  }

  PoolClient* Select(uint32_t client_id) const {
    if (clients_ == nullptr || client_id >= clients_->size()) return nullptr;
    return (*clients_)[client_id].get();
  }

  std::vector<std::unique_ptr<PoolClient>>* clients_;
  uint64_t publication_timeout_ms_;
  mutable std::mutex put_keys_mutex_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      pending_publications_;
};

void FlushAndSettle(const std::vector<std::unique_ptr<PoolClient>>& clients, uint64_t settle_ms) {
  for (const auto& client : clients) client->Master().FlushHeartbeat();
  if (settle_ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
}

std::string Csv(std::string_view value) {
  std::string escaped = "\"";
  for (char c : value) {
    escaped.push_back(c);
    if (c == '"') escaped.push_back('"');
  }
  escaped.push_back('"');
  return escaped;
}

using MetricSnapshot = std::map<std::string, double>;

MetricSnapshot ScrapeMasterMetrics(uint16_t port) {
  MetricSnapshot metrics;
  if (port == 0) return metrics;

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return metrics;
  timeval timeout{5, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    return metrics;
  }
  constexpr char request[] = "GET /metrics HTTP/1.0\r\nHost: localhost\r\n\r\n";
  if (::send(fd, request, sizeof(request) - 1, 0) < 0) {
    ::close(fd);
    return metrics;
  }
  std::string response;
  char buffer[8192];
  ssize_t received = 0;
  while ((received = ::recv(fd, buffer, sizeof(buffer), 0)) > 0) {
    response.append(buffer, static_cast<size_t>(received));
  }
  ::close(fd);

  size_t begin = 0;
  while (begin < response.size()) {
    const size_t end = response.find('\n', begin);
    std::string_view line(response.data() + begin,
                          (end == std::string::npos ? response.size() : end) - begin);
    begin = end == std::string::npos ? response.size() : end + 1;
    if (line.rfind("mori_umbp_", 0) != 0) continue;
    const size_t separator = line.find_last_of(" \t");
    if (separator == std::string_view::npos || separator + 1 == line.size()) continue;
    try {
      metrics.emplace(std::string(line.substr(0, separator)),
                      std::stod(std::string(line.substr(separator + 1))));
    } catch (const std::exception&) {
      // Ignore non-numeric exposition lines; malformed relevant metrics are
      // visible as missing rows instead of aborting a completed benchmark.
    }
  }
  return metrics;
}

void PrintMasterMetrics(const MetricSnapshot& baseline, const MetricSnapshot& final) {
  if (baseline.empty() && final.empty()) return;
  std::cout << "master_metrics\n"
            << "series,baseline,final,delta\n";
  std::map<std::string, bool> names;
  for (const auto& [name, value] : baseline) {
    (void)value;
    names[name] = true;
  }
  for (const auto& [name, value] : final) {
    (void)value;
    names[name] = true;
  }
  for (const auto& [name, ignored] : names) {
    (void)ignored;
    const auto before = baseline.find(name);
    const auto after = final.find(name);
    const double before_value = before == baseline.end() ? 0.0 : before->second;
    const double after_value = after == final.end() ? 0.0 : after->second;
    std::cout << Csv(name) << ',' << before_value << ',' << after_value << ','
              << (after_value - before_value) << '\n';
  }
}

void PrintSummary(const Options& options, uint64_t source_seed,
                  const bench::WorkloadMetrics& metrics, uint64_t wall_ns) {
  const double seconds = static_cast<double>(wall_ns) / 1e9;
  const double ops_per_second = seconds > 0.0 ? metrics.total.succeeded / seconds : 0.0;
  const double gib_per_second =
      seconds > 0.0 ? (static_cast<double>(metrics.total.succeeded_bytes) / (1ULL << 30)) / seconds
                    : 0.0;
  std::cout
      << "summary\n"
      << "command,trace,seed,profile,operations,duration_sec,keys,min_value_bytes,max_value_bytes,"
         "size_distribution,read_ratio,clients,batch,qps,hotset_fraction,zipf_exponent,"
         "tier,backends_per_peer,backend_capacity,page_size,placement,placement_weights,"
         "ssd_dir_prefix,put_strategy,affinity,get_strategy,eviction_high_watermark,"
         "eviction_low_watermark,eviction_check_sec,eviction_lease_sec,eviction_batch_size,"
         "scheduling,time_scale,payload_validation,"
         "warmup_operations,settle_ms,metrics_port,publication_timeout_ms,window_ms,"
         "attempted,succeeded,failed,attempted_bytes,"
         "succeeded_bytes,puts_attempted,puts_succeeded,puts_failed,puts_attempted_bytes,"
         "puts_succeeded_bytes,gets_attempted,gets_succeeded,gets_failed,gets_attempted_bytes,"
         "gets_succeeded_bytes,get_misses,validation_failures,wall_time_ns,ops_per_s,"
         "GiB_per_s,latency_p50_ns,latency_p95_ns,latency_p99_ns,latency_max_ns,"
         "schedule_lag_p50_ns,schedule_lag_p95_ns,schedule_lag_p99_ns,schedule_lag_max_ns\n";
  std::cout << Csv(options.command) << ',' << Csv(options.trace_path) << ',' << source_seed << ','
            << Csv(options.profile) << ',' << options.workload.operation_count << ','
            << options.duration_sec << ',' << options.workload.key_count << ','
            << options.workload.min_value_size << ','
            << options.workload.max_value_size << ',' << Csv(options.size_distribution) << ','
            << options.workload.read_ratio << ',' << options.workload.client_count << ','
            << options.workload.batch_size << ',' << options.workload.qps << ','
            << options.workload.hotset_fraction << ',' << options.workload.zipf_exponent << ','
            << Csv(options.tier) << ',' << options.backends_per_peer << ','
            << options.backend_capacity << ',' << options.page_size << ','
            << Csv(options.placement) << ',' << Csv(WeightsString(options)) << ','
            << Csv(options.ssd_dir_prefix) << ',' << Csv(options.put_strategy) << ','
            << Csv(options.affinity) << ',' << Csv(options.get_strategy) << ','
            << options.eviction_high_watermark << ',' << options.eviction_low_watermark << ','
            << options.eviction_check_sec << ',' << options.eviction_lease_sec << ','
            << options.eviction_batch_size << ','
            << Csv(options.runner.max_throughput ? "max-throughput" : "open-loop") << ','
            << options.runner.time_scale << ',' << BoolString(options.runner.validate_get_payloads)
            << ',' << options.warmup_operations << ',' << options.settle_ms << ','
            << options.metrics_port << ',' << options.publication_timeout_ms << ','
            << options.runner.window_ns / 1000000ULL << ',' << metrics.total.attempted << ','
            << metrics.total.succeeded << ','
            << metrics.total.failed << ',' << metrics.total.attempted_bytes << ','
            << metrics.total.succeeded_bytes << ',' << metrics.puts.attempted << ','
            << metrics.puts.succeeded << ',' << metrics.puts.failed << ','
            << metrics.puts.attempted_bytes << ',' << metrics.puts.succeeded_bytes << ','
            << metrics.gets.attempted << ',' << metrics.gets.succeeded << ','
            << metrics.gets.failed << ',' << metrics.gets.attempted_bytes << ','
            << metrics.gets.succeeded_bytes << ',' << metrics.get_misses << ','
            << metrics.get_validation_failures << ',' << wall_ns << ',' << std::setprecision(10)
            << ops_per_second << ',' << gib_per_second << ',' << metrics.latency.p50_ns << ','
            << metrics.latency.p95_ns << ',' << metrics.latency.p99_ns << ','
            << metrics.latency.max_ns << ',' << metrics.schedule_lag.p50_ns << ','
            << metrics.schedule_lag.p95_ns << ',' << metrics.schedule_lag.p99_ns << ','
            << metrics.schedule_lag.max_ns << '\n';
}

void PrintBackendPlacement(const std::vector<std::unique_ptr<PoolClient>>& clients) {
  std::cout << "backend_placement\n"
            << "node,backend,tier,owned_keys,total_bytes,available_bytes,max_allocatable_bytes\n";
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

void PrintWindows(const bench::WorkloadMetrics& metrics) {
  if (metrics.windows.empty()) return;
  std::cout << "time_windows\n"
            << "window_index,start_ns,end_ns,attempted,succeeded,failed,"
               "attempted_bytes,succeeded_bytes,ops_per_s,GiB_per_s\n";
  for (const auto& window : metrics.windows) {
    const double seconds = static_cast<double>(window.end_ns - window.start_ns) / 1e9;
    const double ops_per_second =
        seconds > 0.0 ? static_cast<double>(window.total.succeeded) / seconds : 0.0;
    const double gib_per_second =
        seconds > 0.0
            ? (static_cast<double>(window.total.succeeded_bytes) / (1ULL << 30)) / seconds
            : 0.0;
    std::cout << window.index << ',' << window.start_ns << ',' << window.end_ns << ','
              << window.total.attempted << ',' << window.total.succeeded << ','
              << window.total.failed << ',' << window.total.attempted_bytes << ','
              << window.total.succeeded_bytes << ',' << ops_per_second << ','
              << gib_per_second << '\n';
  }
}

void InspectReplayTrace(Options* options) {
  bench::TraceReader reader(options->trace_path);
  const auto& header = reader.header();
  options->workload.seed = header.seed();
  const auto& metadata = header.metadata();
  auto text = [&](const char* key) -> const std::string* {
    const auto found = metadata.find(key);
    return found == metadata.end() ? nullptr : &found->second;
  };
  if (const auto* value = text("profile")) options->profile = *value;
  if (const auto* value = text("duration_sec")) {
    options->duration_sec = ParseDouble(*value, "trace metadata duration_sec");
  }
  if (const auto* value = text("keys")) {
    options->workload.key_count = ParseUnsigned<uint64_t>(*value, "trace metadata keys");
  }
  if (const auto* value = text("min_value_bytes")) {
    options->workload.min_value_size =
        ParseUnsigned<size_t>(*value, "trace metadata min_value_bytes");
  }
  if (const auto* value = text("max_value_bytes")) {
    options->workload.max_value_size =
        ParseUnsigned<size_t>(*value, "trace metadata max_value_bytes");
  }
  if (const auto* value = text("size_distribution")) options->size_distribution = *value;
  if (const auto* value = text("read_ratio")) {
    options->workload.read_ratio = ParseDouble(*value, "trace metadata read_ratio");
  }
  if (const auto* value = text("batch")) {
    options->workload.batch_size = ParseUnsigned<uint32_t>(*value, "trace metadata batch");
  }
  if (const auto* value = text("qps")) {
    options->workload.qps = ParseDouble(*value, "trace metadata qps");
  }
  if (const auto* value = text("hotset_fraction")) {
    options->workload.hotset_fraction =
        ParseDouble(*value, "trace metadata hotset_fraction");
  }
  if (const auto* value = text("zipf_exponent")) {
    options->workload.zipf_exponent =
        ParseDouble(*value, "trace metadata zipf_exponent");
  }

  bool has_events = false;
  uint32_t max_client_id = 0;
  uint64_t event_count = 0;
  ::umbp::benchmark::WorkloadEvent event;
  while (reader.ReadNext(&event)) {
    has_events = true;
    max_client_id = std::max(max_client_id, event.client_id());
    if (options->tier == "ssd" && event.value_size() > options->page_size) {
      throw std::invalid_argument("trace contains an SSD value larger than --page-size");
    }
    ++event_count;
  }
  options->workload.operation_count = event_count;

  if (!options->clients_explicit) {
    const auto configured = metadata.find("clients");
    if (configured != metadata.end()) {
      options->workload.client_count =
          ParseUnsigned<uint32_t>(configured->second, "trace metadata clients");
      if (options->workload.client_count == 0) {
        throw std::invalid_argument("trace metadata clients must be positive");
      }
    } else if (has_events) {
      if (max_client_id == std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("trace client_id cannot be represented as a client count");
      }
      options->workload.client_count = max_client_id + 1;
    }
  }
  if (has_events && max_client_id >= options->workload.client_count) {
    throw std::invalid_argument("trace client_id " + std::to_string(max_client_id) +
                                " is outside configured --clients=" +
                                std::to_string(options->workload.client_count));
  }
}

void RunWarmup(const Options& options, PoolWorkloadClient* adapter) {
  if (options.warmup_operations == 0) return;
  auto warmup = options.workload;
  warmup.operation_count = options.warmup_operations;
  warmup.key_prefix = "umbp-bench-warmup-";
  bench::SyntheticWorkloadSource source(std::move(warmup));
  auto runner_options = options.runner;
  runner_options.max_throughput = true;
  bench::WorkloadRunner runner(adapter, runner_options);
  const auto metrics = runner.Run(&source);
  if (metrics.total.failed != 0) {
    throw std::runtime_error("warmup failed " + std::to_string(metrics.total.failed) +
                             " operations");
  }
}

int RunBenchmark(Options options) {
  std::unique_ptr<bench::WorkloadSource> source;
  if (options.command == "replay") {
    InspectReplayTrace(&options);
    ValidateOptions(&options);
    source = std::make_unique<bench::TraceWorkloadSource>(options.trace_path);
  } else {
    source = std::make_unique<bench::SyntheticWorkloadSource>(options.workload);
  }

  MasterHarness master(options);
  auto clients = StartClients(options, master.Address());
  PoolWorkloadClient adapter(&clients, options.publication_timeout_ms);

  // Registration is synchronous, then this pause gives all peer services and
  // heartbeat state a quiet boundary before any unmeasured or measured work.
  FlushAndSettle(clients, options.settle_ms);
  RunWarmup(options, &adapter);
  if (options.warmup_operations != 0) {
    for (const auto& client : clients) {
      if (!client->Clear()) throw std::runtime_error("failed to clear warmup state");
    }
    FlushAndSettle(clients, options.settle_ms);
  }
  const MetricSnapshot metric_baseline = ScrapeMasterMetrics(options.metrics_port);

  bench::WorkloadRunner runner(&adapter, options.runner);
  const auto start = std::chrono::steady_clock::now();
  const bench::WorkloadMetrics metrics = runner.Run(source.get());
  const auto stop = std::chrono::steady_clock::now();
  const uint64_t wall_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

  // Placement inspection is intentionally after the measured interval and
  // after a heartbeat flush; it directly reads each backend and issues no probes.
  FlushAndSettle(clients, options.settle_ms);
  const MetricSnapshot metric_final = ScrapeMasterMetrics(options.metrics_port);
  PrintSummary(options, source->seed(), metrics, wall_ns);
  PrintWindows(metrics);
  PrintBackendPlacement(clients);
  PrintMasterMetrics(metric_baseline, metric_final);
  return metrics.total.failed == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = ParseOptions(argc, argv);
    ValidateOptions(&options);
    if (options.command == "generate") return GenerateTrace(options);
    return RunBenchmark(std::move(options));
  } catch (const std::exception& error) {
    std::cerr << "umbp_tier_bench: " << error.what() << '\n';
    return 1;
  }
}
