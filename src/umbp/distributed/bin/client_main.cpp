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
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/master_client.h"

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
  std::string master_addr = "localhost:50051";
  std::string node_id = "node-1";
  std::string node_addr = "localhost:8080";

  if (argc > 1) master_addr = argv[1];
  if (argc > 2) node_id = argv[2];
  if (argc > 3) node_addr = argv[3];

  mori::umbp::MasterClientConfig config;
  config.master_address = master_addr;
  config.node_id = node_id;
  config.node_address = node_addr;
  config.auto_heartbeat = true;

  mori::umbp::MasterClient client(config);

  // Report some example capacities
  std::map<mori::umbp::TierType, mori::umbp::TierCapacity> caps;
  caps[mori::umbp::TierType::HBM] = {80ULL * 1024 * 1024 * 1024, 80ULL * 1024 * 1024 * 1024};
  caps[mori::umbp::TierType::DRAM] = {512ULL * 1024 * 1024 * 1024, 512ULL * 1024 * 1024 * 1024};

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  auto register_self_status = client.RegisterSelf(caps);
  if (!register_self_status.ok()) {
    MORI_UMBP_ERROR("[Client] RegisterSelf failed: code={}, message={}",
                    static_cast<int>(register_self_status.error_code()),
                    register_self_status.error_message());
    return 1;
  }

  constexpr auto kOperationInterval = std::chrono::seconds(3);
  uint64_t iteration = 0;

  MORI_UMBP_INFO("[Client] Starting RoutePut -> Register -> RouteGet demo as '{}'. Ctrl+C to stop.",
                 node_id);

  while (IsRunning()) {
    ++iteration;
    const std::string key = "demo-block-iter-" + std::to_string(iteration);

    // ---- Step 1: RoutePut — ask master where to write ----
    std::optional<mori::umbp::RoutePutResult> put_target;
    auto route_put_status = client.RoutePut(key, 4ULL * 1024 * 1024, &put_target);
    if (!route_put_status.ok()) {
      MORI_UMBP_WARN("[Client] Iteration {} RoutePut(key={}) RPC failed: {}", iteration, key,
                     route_put_status.error_message());
      if (!SleepInterruptible(kOperationInterval)) break;
      continue;
    }

    if (!put_target.has_value()) {
      MORI_UMBP_WARN("[Client] Iteration {} RoutePut(key={}): no suitable target node", iteration,
                     key);
      if (!SleepInterruptible(kOperationInterval)) break;
      continue;
    }

    MORI_UMBP_INFO("[Client] Iteration {} RoutePut(key={}): target_node={}, addr={}, tier={}",
                   iteration, key, put_target->node_id, put_target->node_address,
                   mori::umbp::TierTypeName(put_target->tier));

    // ---- Step 2: Simulate MORI-IO write (would be real RDMA in production) ----
    std::string simulated_location_id = "sim-loc-" + std::to_string(iteration);
    MORI_UMBP_INFO("[Client] Iteration {} Simulating MORI-IO write to {} -> location_id='{}'",
                   iteration, put_target->node_id, simulated_location_id);

    // ---- Step 3: Register — tell master where the block landed ----
    mori::umbp::Location location;
    location.node_id = put_target->node_id;
    location.location_id = simulated_location_id;
    location.size = 4ULL * 1024 * 1024;
    location.tier = put_target->tier;

    auto register_status = client.Register(key, location);
    if (!register_status.ok()) {
      MORI_UMBP_WARN("[Client] Iteration {} Register(key={}) failed: {}", iteration, key,
                     register_status.error_message());
      if (!SleepInterruptible(kOperationInterval)) break;
      continue;
    }
    MORI_UMBP_INFO("[Client] Iteration {} Register(key={}) succeeded", iteration, key);

    if (!SleepInterruptible(std::chrono::seconds(1))) break;

    // ---- Step 4: RouteGet — ask master where to read the block back ----
    std::optional<mori::umbp::RouteGetResult> get_result;
    auto route_get_status = client.RouteGet(key, &get_result);
    if (!route_get_status.ok()) {
      MORI_UMBP_WARN("[Client] Iteration {} RouteGet(key={}) RPC failed: {}", iteration, key,
                     route_get_status.error_message());
      if (!SleepInterruptible(kOperationInterval)) break;
      continue;
    }

    if (get_result.has_value()) {
      MORI_UMBP_INFO(
          "[Client] Iteration {} RouteGet(key={}): read from node={}, location={}, tier={}",
          iteration, key, get_result->location.node_id, get_result->location.location_id,
          mori::umbp::TierTypeName(get_result->location.tier));
    } else {
      MORI_UMBP_WARN("[Client] Iteration {} RouteGet(key={}): not found (unexpected)", iteration,
                     key);
    }

    // ---- Step 5: Cleanup — unregister the block ----
    uint32_t removed = 0;
    auto unregister_status = client.Unregister(key, location, &removed);
    if (!unregister_status.ok()) {
      MORI_UMBP_WARN("[Client] Iteration {} Unregister(key={}) failed: {}", iteration, key,
                     unregister_status.error_message());
    } else {
      MORI_UMBP_INFO("[Client] Iteration {} Unregister(key={}) removed={}", iteration, key,
                     removed);
    }

    if (!SleepInterruptible(kOperationInterval)) break;
  }

  if (client.IsRegistered()) {
    auto unregister_status = client.UnregisterSelf();
    if (!unregister_status.ok()) {
      MORI_UMBP_ERROR("[Client] Final UnregisterSelf failed: code={}, message={}",
                      static_cast<int>(unregister_status.error_code()),
                      unregister_status.error_message());
      return 1;
    }
  }

  MORI_UMBP_INFO("[Client] Exited cleanly");
  return 0;
}
