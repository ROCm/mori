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
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/master_client.h"

static volatile std::sig_atomic_t g_running = 1;

static void SignalHandler(int /*signum*/) { g_running = 0; }

static bool IsRunning() { return g_running != 0; }

// Generate a batch of synthetic KV hashes for simulation purposes.
// Each hash is a hex-formatted 16-character string derived from the node id,
// iteration number, and index within the batch.
static std::vector<std::string> MakeHashes(const std::string& node_id, uint64_t iteration,
                                           int count) {
  std::vector<std::string> hashes;
  hashes.reserve(count);
  for (int i = 0; i < count; ++i) {
    char buf[32];
    // Simple deterministic hash: mix node_id length, iteration, and index.
    uint64_t val = (static_cast<uint64_t>(node_id.size()) * 1000003ULL) ^
                   (iteration * 6364136223846793005ULL) ^ static_cast<uint64_t>(i);
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(val));
    hashes.emplace_back(buf);
  }
  return hashes;
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

int main(int argc, char** argv) {
  std::string master_addr = "localhost:50051";
  std::string node_id = "node-1";
  std::string node_addr = "localhost:8080";

  if (argc > 1) master_addr = argv[1];
  if (argc > 2) node_id = argv[2];
  if (argc > 3) node_addr = argv[3];

  mori::umbp::UMBPMasterClientConfig config;
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

  MORI_UMBP_INFO(
      "[Client] Starting RoutePut -> Register -> RouteGet + External KV simulation as '{}'. "
      "Ctrl+C to stop.",
      node_id);

  // Tracks the hashes currently reported as live external KV blocks so we can
  // revoke them on the next cycle, simulating KV cache eviction.
  std::vector<std::string> live_ext_kv_hashes;

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

    // ---- Step 6: External KV simulation ----
    // Revoke the previous batch (simulates KV cache eviction), then report a
    // fresh batch (simulates new KV cache tokens being prefilled).
    constexpr int kExtKvBatchSize = 10;

    if (!live_ext_kv_hashes.empty()) {
      auto revoke_status = client.RevokeExternalKvBlocks(node_id, live_ext_kv_hashes);
      if (revoke_status.ok()) {
        MORI_UMBP_INFO("[Client] Iteration {} RevokeExternalKvBlocks: revoked {} hashes", iteration,
                       live_ext_kv_hashes.size());
      } else {
        MORI_UMBP_WARN("[Client] Iteration {} RevokeExternalKvBlocks failed: {}", iteration,
                       revoke_status.error_message());
      }
    }

    live_ext_kv_hashes = MakeHashes(node_id, iteration, kExtKvBatchSize);
    auto report_status =
        client.ReportExternalKvBlocks(node_id, live_ext_kv_hashes, mori::umbp::TierType::HBM);
    if (report_status.ok()) {
      MORI_UMBP_INFO("[Client] Iteration {} ReportExternalKvBlocks: reported {} hashes", iteration,
                     live_ext_kv_hashes.size());
    } else {
      MORI_UMBP_WARN("[Client] Iteration {} ReportExternalKvBlocks failed: {}", iteration,
                     report_status.error_message());
      live_ext_kv_hashes.clear();
    }

    // Query back the hashes we just reported to exercise MatchExternalKv.
    std::vector<mori::umbp::MasterClient::ExternalKvNodeMatch> matches;
    auto match_status = client.MatchExternalKv(live_ext_kv_hashes, &matches);
    if (match_status.ok()) {
      size_t total_matched = 0;
      for (const auto& m : matches) total_matched += m.matched_hashes.size();
      MORI_UMBP_INFO(
          "[Client] Iteration {} MatchExternalKv: queried={}, matched_nodes={}, "
          "total_matched_hashes={}",
          iteration, live_ext_kv_hashes.size(), matches.size(), total_matched);
    } else {
      MORI_UMBP_WARN("[Client] Iteration {} MatchExternalKv failed: {}", iteration,
                     match_status.error_message());
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
