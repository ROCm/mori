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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <ctime>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"

int main(int argc, char** argv) {
  mori::umbp::UMBPConfig config = mori::umbp::UMBPConfig::FromEnvironment();
  config.role = mori::umbp::UMBPRole::Standalone;
  config.follower_mode = false;
  config.force_ssd_copy_on_write = false;
  config.distributed.reset();

  std::string address;
  if (argc > 1 && argv[1] && argv[1][0] != '\0') {
    address = argv[1];
  } else if (const char* env = std::getenv("UMBP_STANDALONE_ADDRESS")) {
    address = env;
  } else {
    address = mori::umbp::standalone::DefaultStandaloneAddress();
  }

  mori::umbp::UMBPStandaloneProcessConfig standalone_cfg;
  standalone_cfg.address = address;
  config.standalone_process = standalone_cfg;

  mori::umbp::standalone::StandaloneServer server(config, address);

  sigset_t signal_set;
  sigemptyset(&signal_set);
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);

  const int block_rc = pthread_sigmask(SIG_BLOCK, &signal_set, nullptr);
  if (block_rc != 0) {
    MORI_UMBP_ERROR("[StandaloneServer] failed to block signals: {}", std::strerror(block_rc));
    return 1;
  }

  if (!server.Start()) {
    MORI_UMBP_ERROR("[StandaloneServer] failed to start on {}", address);
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
        MORI_UMBP_INFO("[StandaloneServer] caught signal {}, shutting down", signum);
        server.Shutdown();
        return;
      }
      if (signum == -1 && errno != EAGAIN && errno != EINTR) {
        MORI_UMBP_ERROR("[StandaloneServer] sigtimedwait failed: {}", std::strerror(errno));
        return;
      }
    }
  });

  MORI_UMBP_INFO("[StandaloneServer] running on {}", address);
  server.Run();

  stop_signal_waiter = true;
  signal_waiter.join();
  MORI_UMBP_INFO("[StandaloneServer] exited cleanly");
  return 0;
}
