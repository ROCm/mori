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
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <ctime>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"

namespace {

std::optional<std::string> EnvString(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') return std::nullopt;
  return std::string(value);
}

bool ParseSizeEnv(const char* name, size_t* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    *out = static_cast<size_t>(std::stoull(*raw));
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

bool ParseUint16Env(const char* name, uint16_t* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    unsigned long value = std::stoul(*raw);
    if (value > 65535) {
      *error = std::string(name) + " must be <= 65535";
      return false;
    }
    *out = static_cast<uint16_t>(value);
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

bool ParseIntEnv(const char* name, int* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    *out = std::stoi(*raw);
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

bool ParseBoolEnv(const char* name, bool* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  std::string value = *raw;
  for (char& ch : value) ch = static_cast<char>(std::tolower(ch));
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    *out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    *out = false;
    return true;
  }
  *error = std::string("invalid boolean ") + name + "=" + *raw;
  return false;
}

bool AnyEnv(const std::vector<const char*>& names) {
  for (const char* name : names) {
    if (EnvString(name).has_value()) return true;
  }
  return false;
}

bool ApplyDistributedBackendConfigFromEnv(mori::umbp::UMBPConfig* config,
                                          bool* distributed_requested, std::string* error) {
  static const std::vector<const char*> kRequiredDistributedEnv = {
      "UMBP_MASTER_ADDRESS",
      "UMBP_NODE_ADDRESS",
      "UMBP_NODE_ID",
      "UMBP_IO_ENGINE_HOST",
  };

  *distributed_requested = AnyEnv(kRequiredDistributedEnv);
  if (!*distributed_requested) {
    config->distributed.reset();
    return true;
  }

  auto master_address = EnvString("UMBP_MASTER_ADDRESS");
  auto node_address = EnvString("UMBP_NODE_ADDRESS");
  auto node_id = EnvString("UMBP_NODE_ID");
  auto io_engine_host = EnvString("UMBP_IO_ENGINE_HOST");
  std::vector<const char*> missing;
  if (!master_address.has_value()) missing.push_back("UMBP_MASTER_ADDRESS");
  if (!node_address.has_value()) missing.push_back("UMBP_NODE_ADDRESS");
  if (!node_id.has_value()) missing.push_back("UMBP_NODE_ID");
  if (!io_engine_host.has_value()) missing.push_back("UMBP_IO_ENGINE_HOST");
  if (!missing.empty()) {
    std::ostringstream oss;
    oss << "distributed-backed standalone server requested but missing required env:";
    for (const char* name : missing) oss << " " << name;
    *error = oss.str();
    return false;
  }

  mori::umbp::UMBPDistributedConfig dist;
  dist.master_config.master_address = *master_address;
  dist.master_config.node_address = *node_address;
  dist.master_config.node_id = *node_id;
  dist.io_engine.host = *io_engine_host;

  if (!ParseUint16Env("UMBP_IO_ENGINE_PORT", &dist.io_engine.port, error)) return false;
  if (!ParseUint16Env("UMBP_PEER_SERVICE_PORT", &dist.peer_service_port, error)) return false;
  if (!ParseSizeEnv("UMBP_DISTRIBUTED_STAGING_BUFFER_SIZE", &dist.staging_buffer_size, error))
    return false;
  if (!ParseSizeEnv("UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SIZE", &dist.ssd_staging_buffer_size,
                    error)) {
    return false;
  }
  if (!ParseIntEnv("UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS", &dist.ssd_staging_buffer_slots,
                   error)) {
    return false;
  }
  if (dist.ssd_staging_buffer_slots <= 0) {
    *error = "UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS must be > 0";
    return false;
  }
  if (!ParseBoolEnv("UMBP_DISTRIBUTED_CACHE_REMOTE_FETCHES", &dist.cache_remote_fetches, error))
    return false;
  size_t dram_page_size = static_cast<size_t>(dist.dram_page_size);
  if (!ParseSizeEnv("UMBP_DISTRIBUTED_DRAM_PAGE_SIZE", &dram_page_size, error)) return false;
  dist.dram_page_size = static_cast<uint64_t>(dram_page_size);

  config->distributed = dist;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  mori::umbp::UMBPConfig config = mori::umbp::UMBPConfig::FromEnvironment();
  config.role = mori::umbp::UMBPRole::Standalone;
  config.follower_mode = false;
  config.force_ssd_copy_on_write = false;
  config.standalone_process.reset();

  bool distributed_requested = false;
  std::string config_error;
  if (!ApplyDistributedBackendConfigFromEnv(&config, &distributed_requested, &config_error)) {
    MORI_UMBP_ERROR("[StandaloneServer] {}", config_error);
    return 1;
  }
  if (!config.Validate(&config_error)) {
    MORI_UMBP_ERROR("[StandaloneServer] invalid backend config: {}", config_error);
    return 1;
  }

  std::string address;
  if (argc > 1 && argv[1] && argv[1][0] != '\0') {
    address = argv[1];
  } else if (const char* env = std::getenv("UMBP_STANDALONE_ADDRESS")) {
    address = env;
  } else {
    address = mori::umbp::standalone::DefaultStandaloneAddress();
  }

  std::unique_ptr<mori::umbp::standalone::StandaloneServer> server;
  try {
    server = std::make_unique<mori::umbp::standalone::StandaloneServer>(config, address);
  } catch (const std::exception& exc) {
    MORI_UMBP_ERROR("[StandaloneServer] backend initialization failed: {}", exc.what());
    return 1;
  }

  sigset_t signal_set;
  sigemptyset(&signal_set);
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);

  const int block_rc = pthread_sigmask(SIG_BLOCK, &signal_set, nullptr);
  if (block_rc != 0) {
    MORI_UMBP_ERROR("[StandaloneServer] failed to block signals: {}", std::strerror(block_rc));
    return 1;
  }

  if (!server->Start()) {
    MORI_UMBP_ERROR("[StandaloneServer] failed to start on {}", address);
    return 1;
  }

  std::atomic<bool> stop_signal_waiter{false};
  std::thread signal_waiter([server = server.get(), &signal_set, &stop_signal_waiter]() {
    while (!stop_signal_waiter.load()) {
      timespec timeout{};
      timeout.tv_sec = 1;
      timeout.tv_nsec = 0;
      const int signum = sigtimedwait(&signal_set, nullptr, &timeout);
      if (signum == SIGINT || signum == SIGTERM) {
        MORI_UMBP_INFO("[StandaloneServer] caught signal {}, shutting down", signum);
        server->Shutdown();
        return;
      }
      if (signum == -1 && errno != EAGAIN && errno != EINTR) {
        MORI_UMBP_ERROR("[StandaloneServer] sigtimedwait failed: {}", std::strerror(errno));
        return;
      }
    }
  });

  MORI_UMBP_INFO("[StandaloneServer] running on {} backend={}", address,
                 distributed_requested ? "distributed" : "local");
  server->Run();

  stop_signal_waiter = true;
  signal_waiter.join();
  MORI_UMBP_INFO("[StandaloneServer] exited cleanly");
  return 0;
}
