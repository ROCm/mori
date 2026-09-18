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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kTpSize = 8;
constexpr size_t kPageSize = 64 * 1024;
constexpr size_t kPoolSize = 64 * 1024 * 1024;
constexpr auto kCoordinationTimeout = std::chrono::seconds(60);

std::string RequiredEnv(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

uint16_t RequiredPort(const char* name) {
  const std::string raw = RequiredEnv(name);
  if (raw.empty()) return 0;
  const unsigned long parsed = std::stoul(raw);
  return parsed <= 65535 ? static_cast<uint16_t>(parsed) : 0;
}

std::string Marker(const std::string& sync_dir, const char* name) {
  return sync_dir + "/" + name;
}

bool WriteMarker(const std::string& path) {
  std::ofstream out(path, std::ios::trunc);
  out << "ready\n";
  return out.good();
}

bool WaitForMarker(const std::string& path,
                   std::chrono::steady_clock::duration timeout = kCoordinationTimeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream in(path);
    if (in.good()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return std::ifstream(path).good();
}

std::vector<std::string> TpKeys() {
  std::vector<std::string> keys;
  keys.reserve(kTpSize);
  for (size_t rank = 0; rank < kTpSize; ++rank) {
    keys.push_back("cross-host-tp8-prefetch-rank-" + std::to_string(rank));
  }
  return keys;
}

std::vector<std::vector<char>> TpPayloads() {
  std::vector<std::vector<char>> payloads(kTpSize, std::vector<char>(kPageSize));
  for (size_t rank = 0; rank < kTpSize; ++rank) {
    for (size_t i = 0; i < kPageSize; ++i) {
      payloads[rank][i] = static_cast<char>((rank * 41 + i * 13) & 0xff);
    }
  }
  return payloads;
}

std::unique_ptr<PoolClient> MakeClient(const std::string& role) {
  const std::string master_address = RequiredEnv("UMBP_PREFETCH_TEST_MASTER_ADDRESS");
  const std::string node_address = RequiredEnv("UMBP_PREFETCH_TEST_NODE_ADDRESS");
  const uint16_t peer_port = RequiredPort("UMBP_PREFETCH_TEST_PEER_PORT");
  const uint16_t io_port = RequiredPort("UMBP_PREFETCH_TEST_IO_PORT");
  if (master_address.empty() || node_address.empty() || peer_port == 0 || io_port == 0) return {};

  PoolClientConfig config;
  config.master_config.master_address = master_address;
  config.master_config.node_id = "cross-host-tp8-" + role;
  config.master_config.node_address = node_address;
  config.io_engine.host = node_address;
  config.io_engine.port = io_port;
  config.peer_service_port = peer_port;
  config.dram_page_size = kPageSize;
  config.dram.buffer_sizes = {kPoolSize};
  config.staging_buffer_size = 4 * kPageSize;
  config.cache_remote_fetches = false;
  config.ranged_locality_prefetch = false;
  config.local_first = true;

  auto client = std::make_unique<PoolClient>(std::move(config));
  if (!client->Init()) return {};
  return client;
}

bool WaitForClusterKey(PoolClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    if (client->Exists(key)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return client->Exists(key);
}

TEST(CrossHostPrefetch, Tp8BatchPrefetch) {
  const std::string role = RequiredEnv("UMBP_PREFETCH_TEST_ROLE");
  const std::string sync_dir = RequiredEnv("UMBP_PREFETCH_TEST_SYNC_DIR");
  ASSERT_TRUE(role == "source" || role == "target")
      << "set UMBP_PREFETCH_TEST_ROLE=source|target";
  ASSERT_FALSE(sync_dir.empty()) << "set UMBP_PREFETCH_TEST_SYNC_DIR to shared storage";

  const auto keys = TpKeys();
  const auto payloads = TpPayloads();

  if (role == "source") {
    auto client = MakeClient(role);
    ASSERT_NE(client, nullptr);

    std::vector<const void*> srcs;
    srcs.reserve(kTpSize);
    for (const auto& payload : payloads) srcs.push_back(payload.data());
    ASSERT_EQ(client->BatchPut(keys, srcs, std::vector<size_t>(kTpSize, kPageSize)),
              std::vector<bool>(kTpSize, true));
    client->Master().FlushHeartbeat();
    ASSERT_TRUE(WriteMarker(Marker(sync_dir, "source.ready")));

    ASSERT_TRUE(WaitForMarker(Marker(sync_dir, "prefetch.done")));
    client->Shutdown();
    ASSERT_TRUE(WriteMarker(Marker(sync_dir, "source.stopped")));
    return;
  }

  ASSERT_TRUE(WaitForMarker(Marker(sync_dir, "source.ready")));
  auto client = MakeClient(role);
  ASSERT_NE(client, nullptr);
  for (const auto& key : keys) ASSERT_TRUE(WaitForClusterKey(client.get(), key)) << key;

  const auto prefetch_started_at = std::chrono::steady_clock::now();
  const auto prefetch_results =
      client->BatchPrefetch(keys, std::chrono::seconds(10),
                            prefetch_started_at + std::chrono::seconds(30));
  const auto prefetch_latency_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                            prefetch_started_at)
          .count();
  std::cout << "[ TP8 PREFETCH LATENCY ] total_latency_us=" << prefetch_latency_us << '\n';
  ASSERT_GT(prefetch_latency_us, 0);
  ASSERT_EQ(prefetch_results, std::vector<bool>(kTpSize, true));
  auto* dram = client->Backends().Get(TierType::DRAM);
  ASSERT_NE(dram, nullptr);
  for (const auto& key : keys) ASSERT_TRUE(dram->Contains(key)) << key;
  ASSERT_TRUE(WriteMarker(Marker(sync_dir, "prefetch.done")));

  // The source exits before Get. Success now proves every TP8 shard is served
  // from the target host's local DRAM rather than through a remote fallback.
  ASSERT_TRUE(WaitForMarker(Marker(sync_dir, "source.stopped")));
  std::vector<std::vector<char>> restored(kTpSize, std::vector<char>(kPageSize, 0));
  std::vector<void*> dsts;
  dsts.reserve(kTpSize);
  for (auto& payload : restored) dsts.push_back(payload.data());
  ASSERT_EQ(client->BatchGet(keys, dsts, std::vector<size_t>(kTpSize, kPageSize)),
            std::vector<bool>(kTpSize, true));
  EXPECT_EQ(restored, payloads);
  client->Shutdown();
}

}  // namespace
}  // namespace mori::umbp
