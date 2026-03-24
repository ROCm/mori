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
#include <csignal>
#include <cstring>
#include <filesystem>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/pool_client.h"

static volatile std::sig_atomic_t g_running = 1;
static void SignalHandler(int) { g_running = 0; }
static bool IsRunning() { return g_running != 0; }

static std::vector<std::string> SplitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::string::size_type start = 0;
  while (start < s.size()) {
    auto pos = s.find(',', start);
    if (pos == std::string::npos) {
      parts.push_back(s.substr(start));
      break;
    }
    parts.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

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
          "  --dram-mb <MB>      DRAM buffer size(s) in MB, comma-separated (default: 64)\n"
          "  --ssd-mb <MB>       SSD capacity(s) in MB, comma-separated (default: 256)\n"
          "  --ssd-dir <path>    SSD storage dir(s), comma-separated (default: "
          "/tmp/umbp_pool_ssd_<node_id>)\n"
          "  --tier <dram|ssd>   Provider only registers specified tier (default: both)\n"
          "\n"
          "Examples:\n"
          "  # Provider: registers 2x64MB DRAM + 256MB SSD, serves remote requests\n"
          "  %s localhost:50051 node-1 localhost:8080 --provider \\\n"
          "    --io-host 10.67.77.61 --io-port 18080 --peer-port 19080 --dram-mb 64,64\n"
          "\n"
          "  # Consumer: no local storage, Put/Get routes to providers via RDMA\n"
          "  %s localhost:50051 node-2 localhost:8081 --consumer \\\n"
          "    --io-host 10.67.77.61 --io-port 18081\n",
          prog, prog, prog);
}

int main(int argc, char** argv) {
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
  std::string dram_mb_str = "64";
  std::string ssd_mb_str = "256";
  std::string ssd_dir_override;
  std::string tier_filter;

  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--provider") {
      is_provider = true;
    } else if (arg == "--consumer") {
      is_provider = false;
    } else if (arg == "--io-host" && i + 1 < argc) {
      io_host = argv[++i];
    } else if (arg == "--io-port" && i + 1 < argc) {
      io_port = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--peer-port" && i + 1 < argc) {
      peer_port = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--dram-mb" && i + 1 < argc) {
      dram_mb_str = argv[++i];
    } else if (arg == "--ssd-mb" && i + 1 < argc) {
      ssd_mb_str = argv[++i];
    } else if (arg == "--ssd-dir" && i + 1 < argc) {
      ssd_dir_override = argv[++i];
    } else if (arg == "--tier" && i + 1 < argc) {
      tier_filter = argv[++i];
    } else {
      MORI_UMBP_ERROR("Unknown option: {}", arg);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  // Parse comma-separated DRAM sizes
  std::vector<size_t> dram_sizes;
  for (const auto& s : SplitComma(dram_mb_str)) {
    dram_sizes.push_back(std::stoull(s) * 1024 * 1024);
  }

  // Parse comma-separated SSD sizes
  std::vector<size_t> ssd_capacities;
  for (const auto& s : SplitComma(ssd_mb_str)) {
    ssd_capacities.push_back(std::stoull(s) * 1024 * 1024);
  }

  // Parse comma-separated SSD dirs
  std::vector<std::string> ssd_dirs;
  if (!ssd_dir_override.empty()) {
    ssd_dirs = SplitComma(ssd_dir_override);
  } else {
    for (size_t i = 0; i < ssd_capacities.size(); ++i) {
      ssd_dirs.push_back("/tmp/umbp_pool_ssd_" + node_id +
                         (ssd_capacities.size() > 1 ? "_" + std::to_string(i) : ""));
    }
  }

  // Allocate DRAM buffers
  std::vector<std::unique_ptr<char[]>> dram_allocs;
  if (is_provider) {
    for (size_t sz : dram_sizes) {
      auto buf = std::make_unique<char[]>(sz);
      std::memset(buf.get(), 0, sz);
      dram_allocs.push_back(std::move(buf));
    }
    for (const auto& dir : ssd_dirs) {
      std::filesystem::create_directories(dir);
    }
  }

  mori::umbp::PoolClientConfig config;
  config.master_config.master_address = master_addr;
  config.master_config.node_id = node_id;
  config.master_config.node_address = node_addr;
  config.master_config.auto_heartbeat = true;

  if (is_provider) {
    bool reg_dram = (tier_filter.empty() || tier_filter == "dram");
    bool reg_ssd = (tier_filter.empty() || tier_filter == "ssd");
    if (reg_dram) {
      size_t total_dram = 0;
      for (size_t i = 0; i < dram_allocs.size(); ++i) {
        config.dram_buffers.push_back({dram_allocs[i].get(), dram_sizes[i]});
        total_dram += dram_sizes[i];
      }
      config.tier_capacities[mori::umbp::TierType::DRAM] = {total_dram, total_dram};
    }
    if (reg_ssd) {
      size_t total_ssd = 0;
      for (size_t i = 0; i < ssd_dirs.size(); ++i) {
        size_t cap = (i < ssd_capacities.size()) ? ssd_capacities[i] : ssd_capacities.back();
        config.ssd_stores.push_back({ssd_dirs[i], cap});
        total_ssd += cap;
      }
      config.tier_capacities[mori::umbp::TierType::SSD] = {total_ssd, total_ssd};
    }
    config.peer_service_port = peer_port;
  }

  config.io_engine_host = io_host;
  config.io_engine_port = io_port;

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  // Capture info before move
  size_t num_dram = config.dram_buffers.size();
  size_t num_ssd = config.ssd_stores.size();

  mori::umbp::PoolClient client(std::move(config));

  if (!client.Init()) {
    MORI_UMBP_ERROR("[Demo] Init failed");
    return 1;
  }

  std::string tier_info = "none";
  if (is_provider) {
    tier_info = "";
    if (num_dram > 0) {
      tier_info += "DRAM=" + std::to_string(num_dram) + "bufs ";
    }
    if (num_ssd > 0) {
      tier_info += "SSD=" + std::to_string(num_ssd) + "dirs ";
    }
  }
  MORI_UMBP_INFO("[Demo] '{}' running as {} | {} | io={}:{} peer_port={}", node_id,
                 is_provider ? "PROVIDER" : "CONSUMER", tier_info,
                 io_host.empty() ? "(none)" : io_host, io_port, peer_port);

  // Register a data buffer for zero-copy RDMA
  constexpr size_t kDataBufSize = 4096;
  auto data_buffer = std::make_unique<char[]>(kDataBufSize);
  bool zero_copy_enabled = false;
  if (io_port > 0) {
    zero_copy_enabled = client.RegisterMemory(data_buffer.get(), kDataBufSize);
    if (zero_copy_enabled) {
      MORI_UMBP_INFO("[Demo] Zero-copy buffer registered ({} bytes)", kDataBufSize);
    }
  }

  if (!is_provider) {
    MORI_UMBP_INFO("[Demo] Consumer mode: waiting 3s for providers to register...");
    if (!SleepInterruptible(std::chrono::seconds(3))) {
      client.Shutdown();
      return 0;
    }
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
      MORI_UMBP_WARN("[Demo] #{} Put({}) FAILED", iteration, key);
      ++fail;
      if (!SleepInterruptible(std::chrono::seconds(3))) break;
      continue;
    }
    MORI_UMBP_INFO("[Demo] #{} Put({}, {} bytes) OK (zero_copy={})", iteration, key, data.size(),
                   zero_copy_enabled);

    if (!SleepInterruptible(std::chrono::seconds(1))) break;

    // Get (zero-copy when buffer is registered)
    std::memset(data_buffer.get(), 0, data.size());
    bool get_ok = client.Get(key, data_buffer.get(), data.size(), zero_copy_enabled);
    if (!get_ok) {
      MORI_UMBP_WARN("[Demo] #{} Get({}) FAILED", iteration, key);
      ++fail;
      if (!SleepInterruptible(std::chrono::seconds(3))) break;
      continue;
    }

    std::string got(data_buffer.get(), data_buffer.get() + data.size());
    if (got == data) {
      MORI_UMBP_INFO("[Demo] #{} Get({}) OK - data verified (zero_copy={})", iteration, key,
                     zero_copy_enabled);
      ++pass;
    } else {
      MORI_UMBP_ERROR("[Demo] #{} Get({}) DATA MISMATCH: expected='{}' got='{}'", iteration, key,
                      data, got);
      ++fail;
    }

    // Remove
    if (client.Remove(key)) {
      MORI_UMBP_INFO("[Demo] #{} Remove({}) OK", iteration, key);
    } else {
      MORI_UMBP_WARN("[Demo] #{} Remove({}) FAILED", iteration, key);
    }

    if (!SleepInterruptible(std::chrono::seconds(3))) break;
  }

  if (zero_copy_enabled) {
    client.DeregisterMemory(data_buffer.get());
  }
  client.Shutdown();
  MORI_UMBP_INFO("[Demo] Finished. {} passed, {} failed out of {} iterations", pass, fail,
                 iteration);
  return 0;
}
