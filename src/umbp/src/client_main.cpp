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
#include <spdlog/spdlog.h>

#include <csignal>
#include <thread>

#include "umbp/client.h"

static volatile std::sig_atomic_t g_running = 1;

static void SignalHandler(int /*signum*/) { g_running = 0; }

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

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  std::string master_addr = "localhost:50051";
  std::string node_id = "node-1";
  std::string node_addr = "localhost:8080";

  if (argc > 1) master_addr = argv[1];
  if (argc > 2) node_id = argv[2];
  if (argc > 3) node_addr = argv[3];

  mori::umbp::UMBPClientConfig config;
  config.master_address = master_addr;
  config.node_id = node_id;
  config.node_address = node_addr;
  config.auto_heartbeat = true;

  mori::umbp::UMBPClient client(config);

  // Report some example capacities
  std::map<mori::umbp::TierType, mori::umbp::TierCapacity> caps;
  caps[mori::umbp::TierType::HBM] = {80ULL * 1024 * 1024 * 1024, 80ULL * 1024 * 1024 * 1024};
  caps[mori::umbp::TierType::DRAM] = {512ULL * 1024 * 1024 * 1024, 512ULL * 1024 * 1024 * 1024};

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  auto register_self_status = client.RegisterSelf(caps);
  if (!register_self_status.ok()) {
    spdlog::error("[Client] RegisterSelf failed: code={}, message={}",
                  static_cast<int>(register_self_status.error_code()),
                  register_self_status.error_message());
    return 1;
  }

  const std::string key = "demo-block-key";
  mori::umbp::Location location;
  location.node_id = node_id;
  location.location_id = "demo-location-0";
  location.size = 4ULL * 1024 * 1024;
  location.tier = mori::umbp::TierType::HBM;

  constexpr auto kOperationInterval = std::chrono::seconds(3);
  uint64_t iteration = 0;
  bool key_registered = false;

  spdlog::info("[Client] Simulating block index Register/Unregister as '{}'. Press Ctrl+C to stop.",
               node_id);

  while (IsRunning()) {
    ++iteration;

    auto register_status = client.Register(key, location);
    if (!register_status.ok()) {
      spdlog::warn("[Client] Iteration {} Register(key={}) failed: code={}, message={}", iteration,
                   key, static_cast<int>(register_status.error_code()),
                   register_status.error_message());
    } else {
      key_registered = true;
      spdlog::info("[Client] Iteration {} Register(key={}) succeeded", iteration, key);
    }

    if (!SleepInterruptible(kOperationInterval)) {
      break;
    }

    if (key_registered) {
      uint32_t removed = 0;
      auto unregister_status = client.Unregister(key, location, &removed);
      if (!unregister_status.ok()) {
        spdlog::warn("[Client] Iteration {} Unregister(key={}) failed: code={}, message={}",
                     iteration, key, static_cast<int>(unregister_status.error_code()),
                     unregister_status.error_message());
      } else {
        key_registered = false;
        spdlog::info("[Client] Iteration {} Unregister(key={}) succeeded (removed={})", iteration,
                     key, removed);
      }
    }

    if (!SleepInterruptible(kOperationInterval)) {
      break;
    }
  }

  if (client.IsRegistered() && key_registered) {
    uint32_t removed = 0;
    auto unregister_status = client.Unregister(key, location, &removed);
    if (!unregister_status.ok()) {
      spdlog::error("[Client] Final key unregister failed: code={}, message={}",
                    static_cast<int>(unregister_status.error_code()),
                    unregister_status.error_message());
    }
  }

  if (client.IsRegistered()) {
    auto unregister_status = client.UnregisterSelf();
    if (!unregister_status.ok()) {
      spdlog::error("[Client] Final UnregisterSelf failed: code={}, message={}",
                    static_cast<int>(unregister_status.error_code()),
                    unregister_status.error_message());
      return 1;
    }
  }

  spdlog::info("[Client] Exited cleanly");
  return 0;
}
