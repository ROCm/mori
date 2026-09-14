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
#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/io/rdma/common.hpp"

namespace mori {
namespace io {

// Consumes verbs async events from each unique ibv_context owned by MORI-IO's
// persistent RdmaContext and reports them through the cached IO logger. One
// epoll thread drains all async fds plus an eventfd used for shutdown. The
// monitor holds non-owning ibv_context* references and must be destroyed before
// ibv_close_device().
//
// Events that make outstanding work uncompletable are also forwarded to the
// optional QpErrorHandler. This matters most for CQ_ERR / DEVICE_FATAL: once the
// CQ is dead no CQE can ever be reaped, so the flush cascade the CQ poller relies
// on never arrives and in-flight transfers would otherwise stay IN_PROGRESS
// forever.
class RdmaAsyncEventMonitor {
 public:
  static std::unique_ptr<RdmaAsyncEventMonitor> Create(const application::RdmaDeviceList& devices,
                                                       std::shared_ptr<spdlog::logger> logger,
                                                       QpErrorHandler errorHandler = nullptr);
  ~RdmaAsyncEventMonitor();

  RdmaAsyncEventMonitor(const RdmaAsyncEventMonitor&) = delete;
  RdmaAsyncEventMonitor& operator=(const RdmaAsyncEventMonitor&) = delete;

 private:
  struct Watch {
    std::string deviceName;
    uint32_t physicalPortCount;
    ibv_context* context;
    int asyncFd;
    int originalFdFlags;
    bool registered;
    bool changedFdFlags;
  };

  // POD copied out of ibv_async_event before it is acked; the union members are
  // invalid to dereference once ibv_ack_async_event() returns.
  struct EventInfo {
    ibv_event_type type;
    const void* objPtr;
    uint32_t qpNum;
    int cqDepth;
    int portNum;
  };

  enum class GetResult { kEvent, kDrained, kError };

  RdmaAsyncEventMonitor(std::shared_ptr<spdlog::logger> logger, QpErrorHandler errorHandler);

  bool Start(const application::RdmaDeviceList& devices);
  void MainLoop() noexcept;
  GetResult ProcessOneEvent(Watch& watch) noexcept;
  void DescribeAndLog(const Watch& watch, const EventInfo& info) noexcept;
  void ReportUncompletableIfNeeded(const Watch& watch, const EventInfo& info) noexcept;
  void RemoveWatch(Watch& watch) noexcept;
  void RestoreWatchFd(Watch& watch) noexcept;
  void DrainWake() noexcept;
  void RestoreFdFlags() noexcept;
  void Shutdown() noexcept;

  // Logging on the monitor thread runs inside noexcept boundaries; swallow any
  // spdlog exception so a formatting/sink failure can never call std::terminate.
  template <typename FormatString, typename... Args>
  void SafeLog(spdlog::level::level_enum level, const FormatString& fmt,
               const Args&... args) noexcept {
    if (!logger_) return;
    try {
      logger_->log(level, fmt, args...);
    } catch (...) {
    }
  }

  std::shared_ptr<spdlog::logger> logger_;
  QpErrorHandler errorHandler_;
  std::vector<Watch> watches_;
  int epollFd_{-1};
  int wakeFd_{-1};
  std::atomic<bool> stopRequested_{false};
  std::thread thread_;
};

}  // namespace io
}  // namespace mori
