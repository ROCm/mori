#include <spdlog/spdlog.h>

#include <csignal>
#include <cstring>
#include <filesystem>
#include <thread>

#include "umbp/pool_client.h"

static volatile std::sig_atomic_t g_running = 1;
static void SignalHandler(int) { g_running = 0; }
static bool IsRunning() { return g_running != 0; }

static bool SleepInterruptible(std::chrono::seconds total) {
  constexpr auto kStep = std::chrono::milliseconds(100);
  auto elapsed = std::chrono::milliseconds(0);
  while (IsRunning() && elapsed < total) {
    std::this_thread::sleep_for(kStep);
    elapsed += kStep;
  }
  return IsRunning();
}

static void PrintUsage(const char* prog) {
  fprintf(stderr,
    "Usage: %s <master_addr> <node_id> <node_addr> [options]\n"
    "\n"
    "Options:\n"
    "  --provider          Register DRAM + SSD storage (default)\n"
    "  --consumer          No local storage, all Put/Get goes to remote nodes\n"
    "  --io-host <host>    IO Engine OOB host (for RDMA, e.g. 10.67.77.61)\n"
    "  --io-port <port>    IO Engine OOB port\n"
    "  --peer-port <port>  PeerService listen port (provider mode only)\n"
    "  --dram-mb <MB>      DRAM buffer size in MB (default: 64)\n"
    "  --ssd-mb <MB>       SSD capacity in MB (default: 256)\n"
    "  --ssd-dir <path>   SSD storage directory (default: /tmp/umbp_pool_ssd_<node_id>)\n"
    "  --tier <dram|ssd>  Provider only registers specified tier (default: both)\n"
    "\n"
    "Examples:\n"
    "  # Provider: registers 64MB DRAM + 256MB SSD, serves remote requests\n"
    "  %s localhost:50051 node-1 localhost:8080 --provider \\\n"
    "    --io-host 10.67.77.61 --io-port 18080 --peer-port 19080\n"
    "\n"
    "  # Consumer: no local storage, Put/Get routes to providers via RDMA\n"
    "  %s localhost:50051 node-2 localhost:8081 --consumer \\\n"
    "    --io-host 10.67.77.61 --io-port 18081\n",
    prog, prog, prog);
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  if (argc < 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string master_addr = argv[1];
  std::string node_id = argv[2];
  std::string node_addr = argv[3];

  bool is_provider = true;
  std::string io_host;
  uint16_t io_port = 0;
  uint16_t peer_port = 0;
  size_t dram_mb = 64;
  size_t ssd_mb = 256;
  std::string ssd_dir_override;
  std::string tier_filter;  // "", "dram", or "ssd"

  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--provider") { is_provider = true; }
    else if (arg == "--consumer") { is_provider = false; }
    else if (arg == "--io-host" && i + 1 < argc) { io_host = argv[++i]; }
    else if (arg == "--io-port" && i + 1 < argc) { io_port = static_cast<uint16_t>(std::stoi(argv[++i])); }
    else if (arg == "--peer-port" && i + 1 < argc) { peer_port = static_cast<uint16_t>(std::stoi(argv[++i])); }
    else if (arg == "--dram-mb" && i + 1 < argc) { dram_mb = std::stoull(argv[++i]); }
    else if (arg == "--ssd-mb" && i + 1 < argc) { ssd_mb = std::stoull(argv[++i]); }
    else if (arg == "--ssd-dir" && i + 1 < argc) { ssd_dir_override = argv[++i]; }
    else if (arg == "--tier" && i + 1 < argc) { tier_filter = argv[++i]; }
    else {
      spdlog::error("Unknown option: {}", arg);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  size_t dram_size = dram_mb * 1024 * 1024;
  size_t ssd_capacity = ssd_mb * 1024 * 1024;
  std::string ssd_dir = ssd_dir_override.empty()
      ? "/tmp/umbp_pool_ssd_" + node_id
      : ssd_dir_override;

  std::unique_ptr<char[]> dram_buffer;
  if (is_provider) {
    dram_buffer = std::make_unique<char[]>(dram_size);
    std::memset(dram_buffer.get(), 0, dram_size);
    std::filesystem::create_directories(ssd_dir);
  }

  mori::umbp::PoolClientConfig config;
  config.master_config.master_address = master_addr;
  config.master_config.node_id = node_id;
  config.master_config.node_address = node_addr;
  config.master_config.auto_heartbeat = true;

  if (is_provider) {
    bool reg_dram = (tier_filter.empty() || tier_filter == "dram");
    bool reg_ssd  = (tier_filter.empty() || tier_filter == "ssd");
    if (reg_dram) {
      config.exportable_dram_buffer = dram_buffer.get();
      config.exportable_dram_buffer_size = dram_size;
      config.tier_capacities[mori::umbp::TierType::DRAM] = {dram_size, dram_size};
    }
    if (reg_ssd) {
      config.exportable_ssd_dir = ssd_dir;
      config.exportable_ssd_capacity = ssd_capacity;
      config.tier_capacities[mori::umbp::TierType::SSD] = {ssd_capacity, ssd_capacity};
    }
    config.peer_service_port = peer_port;
  }

  config.io_engine_host = io_host;
  config.io_engine_port = io_port;

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  mori::umbp::PoolClient client(std::move(config));

  if (!client.Init()) {
    spdlog::error("[Demo] Init failed");
    return 1;
  }

  std::string tier_info = "none";
  if (is_provider) {
    bool reg_dram = (tier_filter.empty() || tier_filter == "dram");
    bool reg_ssd  = (tier_filter.empty() || tier_filter == "ssd");
    tier_info = "";
    if (reg_dram) tier_info += "DRAM=" + std::to_string(dram_mb) + "MB ";
    if (reg_ssd)  tier_info += "SSD=" + std::to_string(ssd_mb) + "MB dir=" + ssd_dir;
  }
  spdlog::info("[Demo] '{}' running as {} | {} | io={}:{} peer_port={}",
               node_id, is_provider ? "PROVIDER" : "CONSUMER", tier_info,
               io_host.empty() ? "(none)" : io_host, io_port, peer_port);

  // Register a data buffer for zero-copy RDMA
  constexpr size_t kDataBufSize = 4096;
  auto data_buffer = std::make_unique<char[]>(kDataBufSize);
  bool zero_copy_enabled = false;
  if (io_port > 0) {
    zero_copy_enabled = client.RegisterMemory(data_buffer.get(), kDataBufSize);
    if (zero_copy_enabled) {
      spdlog::info("[Demo] Zero-copy buffer registered ({} bytes)", kDataBufSize);
    }
  }

  if (!is_provider) {
    spdlog::info("[Demo] Consumer mode: waiting 3s for providers to register...");
    if (!SleepInterruptible(std::chrono::seconds(3))) { client.Shutdown(); return 0; }
  }

  uint64_t iteration = 0;
  uint64_t pass = 0, fail = 0;

  while (IsRunning()) {
    ++iteration;
    const std::string key = node_id + "-blk-" + std::to_string(iteration);
    const std::string data = "data-from-" + node_id + "-iter-" + std::to_string(iteration);

    // Put (zero-copy when buffer is registered)
    std::memcpy(data_buffer.get(), data.data(), data.size());
    bool put_ok = client.Put(key, data_buffer.get(), data.size(), zero_copy_enabled);
    if (!put_ok) {
      spdlog::warn("[Demo] #{} Put({}) FAILED", iteration, key);
      ++fail;
      if (!SleepInterruptible(std::chrono::seconds(3))) break;
      continue;
    }
    spdlog::info("[Demo] #{} Put({}, {} bytes) OK (zero_copy={})", iteration,
                 key, data.size(), zero_copy_enabled);

    if (!SleepInterruptible(std::chrono::seconds(1))) break;

    // Get (zero-copy when buffer is registered)
    std::memset(data_buffer.get(), 0, data.size());
    bool get_ok = client.Get(key, data_buffer.get(), data.size(), zero_copy_enabled);
    if (!get_ok) {
      spdlog::warn("[Demo] #{} Get({}) FAILED", iteration, key);
      ++fail;
      if (!SleepInterruptible(std::chrono::seconds(3))) break;
      continue;
    }

    std::string got(data_buffer.get(), data_buffer.get() + data.size());
    if (got == data) {
      spdlog::info("[Demo] #{} Get({}) OK - data verified (zero_copy={})",
                   iteration, key, zero_copy_enabled);
      ++pass;
    } else {
      spdlog::error("[Demo] #{} Get({}) DATA MISMATCH: expected='{}' got='{}'",
                    iteration, key, data, got);
      ++fail;
    }

    // Remove
    if (client.Remove(key)) {
      spdlog::info("[Demo] #{} Remove({}) OK", iteration, key);
    } else {
      spdlog::warn("[Demo] #{} Remove({}) FAILED", iteration, key);
    }

    if (!SleepInterruptible(std::chrono::seconds(3))) break;
  }

  if (zero_copy_enabled) {
    client.DeregisterMemory(data_buffer.get());
  }
  client.Shutdown();
  spdlog::info("[Demo] Finished. {} passed, {} failed out of {} iterations",
               pass, fail, iteration);
  return 0;
}
