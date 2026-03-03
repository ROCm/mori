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

static std::atomic<bool> g_running{true};

static void SignalHandler(int /*signum*/) { g_running = false; }

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  std::string master_addr = "localhost:50051";
  std::string client_id = "client-1";
  std::string node_addr = "localhost:8080";

  if (argc > 1) master_addr = argv[1];
  if (argc > 2) client_id = argv[2];
  if (argc > 3) node_addr = argv[3];

  mori::umbp::UMBPClientConfig config;
  config.master_address = master_addr;
  config.client_id = client_id;
  config.node_address = node_addr;
  config.auto_heartbeat = true;

  mori::umbp::UMBPClient client(config);

  // Report some example capacities
  std::map<mori::umbp::TierType, mori::umbp::TierCapacity> caps;
  caps[mori::umbp::TierType::HBM] = {80ULL * 1024 * 1024 * 1024, 80ULL * 1024 * 1024 * 1024};
  caps[mori::umbp::TierType::DRAM] = {512ULL * 1024 * 1024 * 1024, 512ULL * 1024 * 1024 * 1024};

  client.RegisterSelf(caps);

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  spdlog::info("[Client] Running as '{}'. Press Ctrl+C to stop.", client_id);
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  client.UnregisterSelf();
  spdlog::info("[Client] Exited cleanly");
  return 0;
}
