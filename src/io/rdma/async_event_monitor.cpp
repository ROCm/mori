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
#include "src/io/rdma/async_event_monitor.hpp"

#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace mori {
namespace io {

namespace {

// Each async fd is drained up to this many events per readiness before yielding
// back to epoll, so a single busy context can neither starve the others nor the
// shutdown wake fd. Level-triggered epoll re-reports any remainder.
constexpr int kFairnessCap = 32;

enum class Severity { kInfo, kWarn, kError };
enum class Category { kPort, kQp, kCq, kSrq, kDevice, kOther };

struct EventDescriptor {
  const char* name;  // nullptr for events not in the explicit table
  Severity severity;
  Category category;
  bool queryPort;
};

// Single source of truth for event name, severity, which union member is valid
// and whether a live ibv_query_port() is warranted. WQ_FATAL / speed-change and
// any newer enum fall through to the numeric default without touching the union.
EventDescriptor DescribeAsyncEvent(ibv_event_type type) {
  switch (type) {
    case IBV_EVENT_PORT_ERR:
      return {"IBV_EVENT_PORT_ERR", Severity::kError, Category::kPort, true};
    case IBV_EVENT_PORT_ACTIVE:
      return {"IBV_EVENT_PORT_ACTIVE", Severity::kError, Category::kPort, true};
    case IBV_EVENT_DEVICE_FATAL:
      return {"IBV_EVENT_DEVICE_FATAL", Severity::kError, Category::kDevice, false};
    case IBV_EVENT_CQ_ERR:
      return {"IBV_EVENT_CQ_ERR", Severity::kError, Category::kCq, false};
    case IBV_EVENT_QP_FATAL:
      return {"IBV_EVENT_QP_FATAL", Severity::kError, Category::kQp, false};
    case IBV_EVENT_QP_REQ_ERR:
      return {"IBV_EVENT_QP_REQ_ERR", Severity::kError, Category::kQp, false};
    case IBV_EVENT_QP_ACCESS_ERR:
      return {"IBV_EVENT_QP_ACCESS_ERR", Severity::kError, Category::kQp, false};
    case IBV_EVENT_SRQ_ERR:
      return {"IBV_EVENT_SRQ_ERR", Severity::kError, Category::kSrq, false};
    case IBV_EVENT_PATH_MIG_ERR:
      return {"IBV_EVENT_PATH_MIG_ERR", Severity::kWarn, Category::kQp, false};
    case IBV_EVENT_PATH_MIG:
      return {"IBV_EVENT_PATH_MIG", Severity::kInfo, Category::kQp, false};
    case IBV_EVENT_COMM_EST:
      return {"IBV_EVENT_COMM_EST", Severity::kInfo, Category::kQp, false};
    case IBV_EVENT_SQ_DRAINED:
      return {"IBV_EVENT_SQ_DRAINED", Severity::kInfo, Category::kQp, false};
    case IBV_EVENT_QP_LAST_WQE_REACHED:
      return {"IBV_EVENT_QP_LAST_WQE_REACHED", Severity::kInfo, Category::kQp, false};
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      return {"IBV_EVENT_SRQ_LIMIT_REACHED", Severity::kInfo, Category::kSrq, false};
    case IBV_EVENT_LID_CHANGE:
      return {"IBV_EVENT_LID_CHANGE", Severity::kInfo, Category::kPort, false};
    case IBV_EVENT_PKEY_CHANGE:
      return {"IBV_EVENT_PKEY_CHANGE", Severity::kInfo, Category::kPort, false};
    case IBV_EVENT_SM_CHANGE:
      return {"IBV_EVENT_SM_CHANGE", Severity::kInfo, Category::kPort, false};
    case IBV_EVENT_GID_CHANGE:
      return {"IBV_EVENT_GID_CHANGE", Severity::kInfo, Category::kPort, false};
    case IBV_EVENT_CLIENT_REREGISTER:
      return {"IBV_EVENT_CLIENT_REREGISTER", Severity::kInfo, Category::kPort, false};
    default:
      return {nullptr, Severity::kWarn, Category::kOther, false};
  }
}

class AsyncEventAckGuard {
 public:
  explicit AsyncEventAckGuard(ibv_async_event* event) noexcept : event_(event) {}
  ~AsyncEventAckGuard() { ibv_ack_async_event(event_); }

  AsyncEventAckGuard(const AsyncEventAckGuard&) = delete;
  AsyncEventAckGuard& operator=(const AsyncEventAckGuard&) = delete;

 private:
  ibv_async_event* event_;
};

}  // namespace

std::unique_ptr<RdmaAsyncEventMonitor> RdmaAsyncEventMonitor::Create(
    const application::RdmaDeviceList& devices, std::shared_ptr<spdlog::logger> logger) {
  // Thread creation and allocations can throw; a failed monitor must degrade to
  // "no observability", never abort RDMA backend construction. On any throw the
  // in-scope unique_ptr unwinds through Shutdown() and releases monitor fds.
  try {
    std::unique_ptr<RdmaAsyncEventMonitor> monitor(new RdmaAsyncEventMonitor(std::move(logger)));
    if (!monitor->Start(devices)) return nullptr;
    return monitor;
  } catch (...) {
    return nullptr;
  }
}

RdmaAsyncEventMonitor::RdmaAsyncEventMonitor(std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {}

RdmaAsyncEventMonitor::~RdmaAsyncEventMonitor() { Shutdown(); }

bool RdmaAsyncEventMonitor::Start(const application::RdmaDeviceList& devices) {
  std::unordered_set<ibv_context*> seen;
  for (application::RdmaDevice* device : devices) {
    ibv_context* context = device->GetIbvContext();
    if (context == nullptr || !seen.insert(context).second) continue;
    watches_.push_back(Watch{device->Name(), static_cast<uint32_t>(device->GetDevicePortNum()),
                             context, context->async_fd, 0, false, false});
  }
  if (watches_.empty()) return false;

  for (Watch& watch : watches_) {
    int flags = fcntl(watch.asyncFd, F_GETFL);
    if (flags < 0) {
      SafeLog(spdlog::level::err, "RDMA async monitor: F_GETFL failed on {}: {}", watch.deviceName,
              std::strerror(errno));
      return false;
    }
    watch.originalFdFlags = flags;
    if (!(flags & O_NONBLOCK) && fcntl(watch.asyncFd, F_SETFL, flags | O_NONBLOCK) < 0) {
      SafeLog(spdlog::level::err, "RDMA async monitor: F_SETFL O_NONBLOCK failed on {}: {}",
              watch.deviceName, std::strerror(errno));
      return false;
    }
    watch.changedFdFlags = !(flags & O_NONBLOCK);
  }

  epollFd_ = epoll_create1(EPOLL_CLOEXEC);
  if (epollFd_ < 0) {
    SafeLog(spdlog::level::err, "RDMA async monitor: epoll_create1 failed: {}",
            std::strerror(errno));
    return false;
  }

  wakeFd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (wakeFd_ < 0) {
    SafeLog(spdlog::level::err, "RDMA async monitor: eventfd failed: {}", std::strerror(errno));
    return false;
  }

  epoll_event wake{};
  wake.events = EPOLLIN;
  wake.data.u64 = 0;
  if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, wakeFd_, &wake) < 0) {
    SafeLog(spdlog::level::err, "RDMA async monitor: epoll_ctl(wake) failed: {}",
            std::strerror(errno));
    return false;
  }

  for (size_t i = 0; i < watches_.size(); ++i) {
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = i + 1;
    if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, watches_[i].asyncFd, &ev) < 0) {
      SafeLog(spdlog::level::err, "RDMA async monitor: epoll_ctl({}) failed: {}",
              watches_[i].deviceName, std::strerror(errno));
      return false;
    }
    watches_[i].registered = true;
  }

  thread_ = std::thread([this] { MainLoop(); });
  return true;
}

void RdmaAsyncEventMonitor::MainLoop() noexcept {
  constexpr int kMaxEvents = 16;
  epoll_event events[kMaxEvents];
  while (!stopRequested_.load(std::memory_order_acquire)) {
    int n = epoll_wait(epollFd_, events, kMaxEvents, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      SafeLog(spdlog::level::err, "RDMA async monitor: epoll_wait failed: {}; stopping",
              std::strerror(errno));
      return;
    }

    for (int i = 0; i < n; ++i) {
      uint64_t id = events[i].data.u64;
      if (id == 0) {
        DrainWake();
        continue;
      }

      Watch& watch = watches_[id - 1];
      if (!watch.registered) continue;

      bool terminal = (events[i].events & (EPOLLERR | EPOLLHUP)) != 0;
      bool drained = false;
      for (int c = 0; c < kFairnessCap; ++c) {
        GetResult result = ProcessOneEvent(watch);
        if (result == GetResult::kDrained) {
          drained = true;
          break;
        }
        if (result == GetResult::kError) {
          RemoveWatch(watch);
          break;
        }
      }
      if (terminal && watch.registered && drained) RemoveWatch(watch);
    }

    bool anyRegistered = false;
    for (const Watch& watch : watches_) {
      if (watch.registered) {
        anyRegistered = true;
        break;
      }
    }
    if (!anyRegistered) {
      SafeLog(spdlog::level::err, "RDMA async monitor: all async fds removed; stopping");
      return;
    }
  }
}

RdmaAsyncEventMonitor::GetResult RdmaAsyncEventMonitor::ProcessOneEvent(Watch& watch) noexcept {
  ibv_async_event event{};
  if (ibv_get_async_event(watch.context, &event) != 0) {
    if (errno == EINTR) return GetResult::kEvent;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return GetResult::kDrained;
    SafeLog(spdlog::level::err, "RDMA async monitor: ibv_get_async_event failed on {}: {}",
            watch.deviceName, std::strerror(errno));
    return GetResult::kError;
  }

  EventInfo info{event.event_type, nullptr, 0, 0, -1};
  {
    // Snapshot the union while the guard keeps the event alive, then ack before
    // the (slower) query and logging so ibv_destroy_* waits on us as little as
    // possible. The union members must not be touched once ack returns.
    AsyncEventAckGuard ack(&event);
    switch (DescribeAsyncEvent(event.event_type).category) {
      case Category::kCq:
        if (event.element.cq != nullptr) {
          info.objPtr = event.element.cq;
          info.cqDepth = event.element.cq->cqe;
        }
        break;
      case Category::kQp:
        if (event.element.qp != nullptr) {
          info.objPtr = event.element.qp;
          info.qpNum = event.element.qp->qp_num;
        }
        break;
      case Category::kSrq:
        info.objPtr = event.element.srq;
        break;
      case Category::kPort:
        info.portNum = event.element.port_num;
        break;
      default:
        break;
    }
  }
  DescribeAndLog(watch, info);
  return GetResult::kEvent;
}

void RdmaAsyncEventMonitor::DescribeAndLog(const Watch& watch, const EventInfo& info) noexcept {
  EventDescriptor desc = DescribeAsyncEvent(info.type);
  const char* name = desc.name != nullptr ? desc.name : "IBV_EVENT_UNKNOWN";

  char detail[256] = "";
  switch (desc.category) {
    case Category::kPort: {
      if (info.portNum >= 1 && static_cast<uint32_t>(info.portNum) <= watch.physicalPortCount) {
        int off = std::snprintf(detail, sizeof(detail), " port=%d", info.portNum);
        if (desc.queryPort && off > 0) {
          ibv_port_attr attr{};
          int qret = ibv_query_port(watch.context, static_cast<uint8_t>(info.portNum), &attr);
          if (qret == 0) {
            std::snprintf(detail + off, sizeof(detail) - off,
                          " state=%d phys_state=%d link_layer=%d lid=%u active_speed=%u"
                          " active_width=%u",
                          static_cast<int>(attr.state), static_cast<int>(attr.phys_state),
                          static_cast<int>(attr.link_layer), static_cast<unsigned>(attr.lid),
                          static_cast<unsigned>(attr.active_speed),
                          static_cast<unsigned>(attr.active_width));
          } else {
            std::snprintf(detail + off, sizeof(detail) - off, " query_port_failed ret=%d errno=%d",
                          qret, errno);
          }
        }
      } else {
        std::snprintf(detail, sizeof(detail), " port=invalid(%d)", info.portNum);
      }
      break;
    }
    case Category::kQp:
      std::snprintf(detail, sizeof(detail), " qp=%p qpn=%u", info.objPtr, info.qpNum);
      break;
    case Category::kCq:
      std::snprintf(detail, sizeof(detail), " cq=%p cqe=%d", info.objPtr, info.cqDepth);
      break;
    case Category::kSrq:
      std::snprintf(detail, sizeof(detail), " srq=%p", info.objPtr);
      break;
    default:
      break;
  }

  const char* recovery = info.type == IBV_EVENT_PORT_ACTIVE ? " (recovery)" : "";
  spdlog::level::level_enum level = desc.severity == Severity::kError  ? spdlog::level::err
                                    : desc.severity == Severity::kWarn ? spdlog::level::warn
                                                                       : spdlog::level::info;
  SafeLog(level, "RDMA async event {}{} (type={}) dev={} ctx={} async_fd={}{}", name, recovery,
          static_cast<int>(info.type), watch.deviceName, static_cast<const void*>(watch.context),
          watch.asyncFd, detail);
}

void RdmaAsyncEventMonitor::RestoreWatchFd(Watch& watch) noexcept {
  if (!watch.changedFdFlags) return;
  int ret;
  do {
    ret = fcntl(watch.asyncFd, F_SETFL, watch.originalFdFlags);
  } while (ret < 0 && errno == EINTR);
  if (ret < 0) {
    SafeLog(spdlog::level::warn, "RDMA async monitor: failed to restore fd flags on {}: {}",
            watch.deviceName, std::strerror(errno));
    return;  // leave changedFdFlags set so a later restore attempt retries
  }
  watch.changedFdFlags = false;
}

void RdmaAsyncEventMonitor::RemoveWatch(Watch& watch) noexcept {
  if (!watch.registered) return;
  epoll_ctl(epollFd_, EPOLL_CTL_DEL, watch.asyncFd, nullptr);
  watch.registered = false;
  RestoreWatchFd(watch);
}

void RdmaAsyncEventMonitor::DrainWake() noexcept {
  uint64_t value;
  while (read(wakeFd_, &value, sizeof(value)) == sizeof(value)) {
  }
}

void RdmaAsyncEventMonitor::RestoreFdFlags() noexcept {
  for (Watch& watch : watches_) RestoreWatchFd(watch);
}

void RdmaAsyncEventMonitor::Shutdown() noexcept {
  if (thread_.joinable()) {
    stopRequested_.store(true, std::memory_order_release);
    if (wakeFd_ >= 0) {
      uint64_t one = 1;
      while (write(wakeFd_, &one, sizeof(one)) < 0 && errno == EINTR) {
      }
    }
    thread_.join();
  }
  RestoreFdFlags();
  if (epollFd_ >= 0) {
    close(epollFd_);
    epollFd_ = -1;
  }
  if (wakeFd_ >= 0) {
    close(wakeFd_);
    wakeFd_ = -1;
  }
}

}  // namespace io
}  // namespace mori
