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
#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <ctime>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/master_server.h"

int main(int argc, char** argv) {
  std::string address = "0.0.0.0:50051";
  if (argc > 1) {
    address = argv[1];
  }

  mori::umbp::MasterServerConfig config;
  config.listen_address = address;
  // Defaults: heartbeat_ttl=10s, reaper_interval=5s, max_missed=3

  mori::umbp::MasterServer server(std::move(config));

  sigset_t signal_set;
  sigemptyset(&signal_set);
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);

  // Block signals in this thread so the waiter thread can synchronously
  // receive them via sigtimedwait.
  const int block_rc = pthread_sigmask(SIG_BLOCK, &signal_set, nullptr);
  if (block_rc != 0) {
    MORI_UMBP_ERROR("[Master] Failed to block signals: {}", std::strerror(block_rc));
    return 1;
  }

  std::atomic<bool> stop_signal_waiter{false};
  std::thread signal_waiter([&server, &signal_set, &stop_signal_waiter]() {
    while (!stop_signal_waiter.load()) {
      timespec timeout{};
      timeout.tv_sec = 1;
      timeout.tv_nsec = 0;

      const int signum = sigtimedwait(&signal_set, nullptr, &timeout);
      if (signum == SIGINT || signum == SIGTERM) {
        MORI_UMBP_INFO("[Master] Caught signal {}, shutting down", signum);
        server.Shutdown();
        return;
      }
      if (signum == -1 && errno != EAGAIN && errno != EINTR) {
        MORI_UMBP_ERROR("[Master] sigtimedwait failed: {}", std::strerror(errno));
        return;
      }
    }
  });

  MORI_UMBP_INFO("[Master] Starting UMBP master on {}", address);
  server.Run();  // blocks until Shutdown

  stop_signal_waiter = true;
  signal_waiter.join();

  MORI_UMBP_INFO("[Master] Exited cleanly");
  return 0;
}
