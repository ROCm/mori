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
#include "src/io/rdma/fault_injector.hpp"

#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mori/io/common.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

namespace {

struct NamedValue {
  const char* name;
  int value;
};

constexpr NamedValue kKinds[] = {
    {"post_send_fail", static_cast<int>(FaultKind::PostSendFail)},
    {"post_recv_fail", static_cast<int>(FaultKind::PostRecvFail)},
    {"qp_error", static_cast<int>(FaultKind::QpError)},
    {"cqe_error", static_cast<int>(FaultKind::CqeError)},
    {"cqe_drop", static_cast<int>(FaultKind::CqeDrop)},
    {"async_event", static_cast<int>(FaultKind::AsyncEvent)},
    {"port_down", static_cast<int>(FaultKind::PortDown)},
    {"port_flap", static_cast<int>(FaultKind::PortFlap)},
};

constexpr NamedValue kOpcodes[] = {
    {"write", IBV_WC_RDMA_WRITE},
    {"read", IBV_WC_RDMA_READ},
    {"send", IBV_WC_SEND},
    {"recv", IBV_WC_RECV},
};

constexpr NamedValue kWcStatuses[] = {
    {"retry_exc", IBV_WC_RETRY_EXC_ERR},   {"rnr_retry_exc", IBV_WC_RNR_RETRY_EXC_ERR},
    {"rem_access", IBV_WC_REM_ACCESS_ERR}, {"rem_op", IBV_WC_REM_OP_ERR},
    {"loc_prot", IBV_WC_LOC_PROT_ERR},     {"flush", IBV_WC_WR_FLUSH_ERR},
    {"fatal", IBV_WC_FATAL_ERR},
};

constexpr NamedValue kAsyncEvents[] = {
    {"qp_fatal", IBV_EVENT_QP_FATAL},
    {"qp_req_err", IBV_EVENT_QP_REQ_ERR},
    {"qp_access_err", IBV_EVENT_QP_ACCESS_ERR},
    {"cq_err", IBV_EVENT_CQ_ERR},
    {"device_fatal", IBV_EVENT_DEVICE_FATAL},
    {"port_err", IBV_EVENT_PORT_ERR},
    {"port_active", IBV_EVENT_PORT_ACTIVE},
    {"gid_change", IBV_EVENT_GID_CHANGE},
    {"lid_change", IBV_EVENT_LID_CHANGE},
    {"pkey_change", IBV_EVENT_PKEY_CHANGE},
    {"sm_change", IBV_EVENT_SM_CHANGE},
    {"client_rereg", IBV_EVENT_CLIENT_REREGISTER},
    {"path_mig", IBV_EVENT_PATH_MIG},
    {"path_mig_err", IBV_EVENT_PATH_MIG_ERR},
    {"comm_est", IBV_EVENT_COMM_EST},
    {"sq_drained", IBV_EVENT_SQ_DRAINED},
    {"last_wqe", IBV_EVENT_QP_LAST_WQE_REACHED},
};

thread_local ibv_async_event* tlsSynthetic = nullptr;

bool IsAsyncKind(FaultKind kind) {
  return kind == FaultKind::AsyncEvent || kind == FaultKind::PortDown ||
         kind == FaultKind::PortFlap;
}

int SendOpcode(const ibv_send_wr* wr) {
  switch (wr->opcode) {
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      return IBV_WC_RDMA_WRITE;
    case IBV_WR_RDMA_READ:
      return IBV_WC_RDMA_READ;
    case IBV_WR_SEND:
    case IBV_WR_SEND_WITH_IMM:
      return IBV_WC_SEND;
    default:
      return -1;
  }
}

// Called under mu_: the counter is either zero or one, mirroring queue emptiness.
bool SetSourceReadable(int fd, bool readable) noexcept {
  uint64_t one = 1;
  ssize_t ret;
  do {
    ret = readable ? write(fd, &one, sizeof(one)) : read(fd, &one, sizeof(one));
  } while (ret < 0 && errno == EINTR);
  return ret == sizeof(one);
}

template <size_t N>
int Lookup(const NamedValue (&table)[N], const std::string& token, bool allowNumber) {
  for (const NamedValue& nv : table) {
    if (token == nv.name) return nv.value;
  }
  if (allowNumber) {
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(token.c_str(), &end, 0);
    if (end != token.c_str() && *end == '\0' && errno != ERANGE &&
        v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max()) {
      return static_cast<int>(v);
    }
  }
  throw std::invalid_argument("MORI_IO_FAULT: unknown token '" + token + "'");
}

uint64_t ParseU64(const std::string& key, const std::string& token) {
  char* end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(token.c_str(), &end, 0);
  if (end == token.c_str() || *end != '\0' || token[0] == '-' || errno == ERANGE) {
    throw std::invalid_argument("MORI_IO_FAULT: bad number for " + key + ": '" + token + "'");
  }
  return v;
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t pos = s.find(sep, start);
    out.push_back(s.substr(start, pos - start));
    if (pos == std::string::npos) return out;
    start = pos + 1;
  }
}

}  // namespace

const char* FaultKindName(FaultKind kind) {
  for (const NamedValue& nv : kKinds) {
    if (nv.value == static_cast<int>(kind)) return nv.name;
  }
  return "unknown";
}

FaultRule ParseFaultRule(const std::string& spec) {
  std::vector<std::string> parts = Split(spec, ':');
  FaultRule rule;
  rule.kind = static_cast<FaultKind>(Lookup(kKinds, parts[0], false));
  if (rule.kind == FaultKind::PostSendFail || rule.kind == FaultKind::PostRecvFail) {
    rule.value = EIO;
  } else if (rule.kind == FaultKind::CqeError) {
    rule.value = IBV_WC_RETRY_EXC_ERR;
  }

  bool hasValue = false;
  if (rule.kind == FaultKind::PortFlap) rule.value = 200;
  for (size_t i = 1; i < parts.size(); ++i) {
    size_t eq = parts[i].find('=');
    if (eq == std::string::npos) {
      throw std::invalid_argument("MORI_IO_FAULT: expected key=value, got '" + parts[i] + "'");
    }
    std::string key = parts[i].substr(0, eq);
    std::string val = parts[i].substr(eq + 1);
    if (key == "skip") {
      rule.skip = ParseU64(key, val);
    } else if (key == "count") {
      rule.count = ParseU64(key, val);
    } else if (key == "qpn") {
      uint64_t qpn = ParseU64(key, val);
      if (qpn > UINT32_MAX) throw std::invalid_argument("MORI_IO_FAULT: qpn out of range");
      rule.qpn = static_cast<uint32_t>(qpn);
    } else if (key == "op") {
      rule.wcOpcode = Lookup(kOpcodes, val, true);
    } else if (key == "value") {
      hasValue = true;
      if (rule.kind == FaultKind::AsyncEvent) {
        rule.value = Lookup(kAsyncEvents, val, true);
        bool supported = false;
        for (const auto& event : kAsyncEvents) supported |= event.value == rule.value;
        if (!supported) throw std::invalid_argument("MORI_IO_FAULT: unsupported async event");
      } else if (rule.kind == FaultKind::CqeError) {
        rule.value = Lookup(kWcStatuses, val, true);
      } else {
        uint64_t value = ParseU64(key, val);
        if (value > INT32_MAX) throw std::invalid_argument("MORI_IO_FAULT: value out of range");
        rule.value = static_cast<int>(value);
      }
    } else {
      throw std::invalid_argument("MORI_IO_FAULT: unknown key '" + key + "'");
    }
  }
  if (rule.kind == FaultKind::AsyncEvent && !hasValue) {
    throw std::invalid_argument("MORI_IO_FAULT: async_event requires value");
  }
  if (rule.kind == FaultKind::PortDown || rule.kind == FaultKind::PortFlap) {
    if (rule.count != 1) throw std::invalid_argument("MORI_IO_FAULT: port kinds require count=1");
    if (rule.kind == FaultKind::PortDown && hasValue) {
      throw std::invalid_argument("MORI_IO_FAULT: port_down does not take value");
    }
    if (rule.kind == FaultKind::PortFlap && rule.value <= 0) {
      throw std::invalid_argument("MORI_IO_FAULT: port_flap duration must be positive");
    }
  }
  return rule;
}

FaultInjector& FaultInjector::Instance() {
  // Verbs() can still be called by an engine's monitor during static teardown.
  // Keep the test-build singleton alive until the OS reclaims the process.
  static auto* injector = new FaultInjector;
  return *injector;
}

FaultInjector::FaultInjector() {
  const char* spec = std::getenv("MORI_IO_FAULT");
  if (spec != nullptr && spec[0] != '\0') {
    try {
      Arm(ParseFaultRule(spec));
    } catch (const std::exception& error) {
      // A typo must never turn into a silently fault-free run, and this runs
      // inside the first Verbs() call, where an exception may not be caught.
      std::fprintf(stderr, "MORI_IO_FAULT='%s' is invalid: %s\n", spec, error.what());
      std::abort();
    }
  }
}

void FaultInjector::Arm(const FaultRule& rule) {
  Disarm();
  {
    std::lock_guard<std::mutex> lock(mu_);
    rule_ = rule;
    matched_ = 0;
    triggered_ = 0;
    failed_.store(false, std::memory_order_relaxed);
    qps_.clear();
  }
  fired_.store(0, std::memory_order_relaxed);
  armed_.store(true, std::memory_order_release);
  MORI_IO_WARN("Fault injection armed: kind={} skip={} count={} value={} op={} qpn={}",
               FaultKindName(rule.kind), rule.skip, rule.count, rule.value, rule.wcOpcode,
               rule.qpn);
}

void FaultInjector::Disarm() noexcept try {
  std::lock_guard<std::mutex> lock(mu_);
  armed_.store(false, std::memory_order_release);
  // Rule exhaustion already stopped new triggers. Only port_down uses explicit
  // Disarm as recovery; a flap must finish its timed interval independently.
  for (auto it = downPorts_.begin(); it != downPorts_.end();) {
    if (it->second.durationMs == 0) {
      auto key = (it++)->first;
      RestorePortLocked(key.first, key.second);
    } else {
      ++it;
    }
  }
  timerCv_.notify_all();
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("disarming injection");
}

bool FaultInjector::ShouldFire(FaultKind kind, uint32_t qpn, int wcOpcode, int* value) {
  std::lock_guard<std::mutex> lock(mu_);
  return ShouldFireLocked(kind, qpn, wcOpcode, value);
}

bool FaultInjector::ShouldFireLocked(FaultKind kind, uint32_t qpn, int wcOpcode, int* value) {
  if (!armed_.load(std::memory_order_acquire) || rule_.kind != kind) return false;
  if (rule_.qpn != 0 && rule_.qpn != qpn) return false;
  if (rule_.wcOpcode >= 0 && rule_.wcOpcode != wcOpcode) return false;
  if (matched_++ < rule_.skip) return false;
  ++triggered_;
  if (rule_.count != 0 && triggered_ >= rule_.count) armed_.store(false, std::memory_order_release);
  *value = rule_.value;
  return true;
}

void FaultInjector::RecordInjection(FaultKind kind, uint32_t qpn, int wcOpcode,
                                    int value) noexcept {
  uint64_t fired = fired_.fetch_add(1, std::memory_order_relaxed) + 1;
  try {
    MORI_IO_WARN("Fault injected: kind={} qpn={} opcode={} value={} (#{})", FaultKindName(kind),
                 qpn, wcOpcode, value, fired);
  } catch (...) {
  }
}

void FaultInjector::FailInjection(const char* operation, int error) noexcept {
  failed_.store(true, std::memory_order_relaxed);
  armed_.store(false, std::memory_order_release);
  try {
    MORI_IO_ERROR("Fault injection failed: {} (error={}); stopping new triggers", operation, error);
  } catch (...) {
  }
}

void FaultInjector::RememberQp(ibv_qp* qp) {
  std::lock_guard<std::mutex> lock(mu_);
  qps_[qp->context][qp->qp_num] = qp;
}

bool FaultInjector::MoveQpToError(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_ERR;
  int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (ret != 0) FailInjection("ibv_modify_qp(ERR)", ret);
  return ret == 0;
}

int FaultInjector::PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) {
  ibv_async_event event{};
  bool inject = false;
  try {
    if (hasDeadResources_.load(std::memory_order_relaxed) && DeviceDead(qp->context)) {
      *bad = wr;
      return EIO;
    }
    int value = 0;
    if (armed_.load(std::memory_order_relaxed)) {
      RememberQp(qp);
      if (ShouldFire(FaultKind::PostSendFail, qp->qp_num, -1, &value)) {
        RecordInjection(FaultKind::PostSendFail, qp->qp_num, -1, value);
        *bad = wr;
        return value;
      }
      if (ShouldFire(FaultKind::QpError, qp->qp_num, -1, &value) && MoveQpToError(qp)) {
        RecordInjection(FaultKind::QpError, qp->qp_num, -1, value);
      }
    }
    inject =
        armed_.load(std::memory_order_relaxed) && PrepareAsyncEvent(qp, SendOpcode(wr), &event);
  } catch (const std::exception& error) {
    FailInjection(error.what());
  } catch (...) {
    FailInjection("preparing send injection");
  }
  int ret = real_.PostSend(qp, wr, bad);
  if (inject) PublishAsyncEvent(qp, SendOpcode(wr), event);
  return ret;
}

int FaultInjector::PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) {
  ibv_async_event event{};
  bool inject = false;
  try {
    if (hasDeadResources_.load(std::memory_order_relaxed) && DeviceDead(qp->context)) {
      *bad = wr;
      return EIO;
    }
    int value = 0;
    if (armed_.load(std::memory_order_relaxed)) {
      RememberQp(qp);
      if (ShouldFire(FaultKind::PostRecvFail, qp->qp_num, -1, &value)) {
        RecordInjection(FaultKind::PostRecvFail, qp->qp_num, -1, value);
        *bad = wr;
        return value;
      }
    }
    inject = armed_.load(std::memory_order_relaxed) && PrepareAsyncEvent(qp, IBV_WC_RECV, &event);
  } catch (const std::exception& error) {
    FailInjection(error.what());
  } catch (...) {
    FailInjection("preparing recv injection");
  }
  int ret = real_.PostRecv(qp, wr, bad);
  if (inject) PublishAsyncEvent(qp, IBV_WC_RECV, event);
  return ret;
}

int FaultInjector::PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) try {
  if (hasDeadResources_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(mu_);
    if (deadCqs_.count(cq) || deadDevices_.count(cq->context)) return 0;
  }
  int n = real_.PollCq(cq, numEntries, wc);
  // A fatal injection can race a poll already inside the provider.
  if (hasDeadResources_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(mu_);
    if (deadCqs_.count(cq) || deadDevices_.count(cq->context)) return 0;
  }
  if (n <= 0 || !armed_.load(std::memory_order_relaxed)) return n;

  // Only successful CQEs carry a valid opcode, and only they can be "lost" or
  // turned into errors; real error CQEs pass through untouched.
  int kept = 0;
  for (int i = 0; i < n; ++i) {
    int value = 0;
    if (wc[i].status == IBV_WC_SUCCESS) {
      if (ShouldFire(FaultKind::CqeDrop, wc[i].qp_num, wc[i].opcode, &value)) {
        RecordInjection(FaultKind::CqeDrop, wc[i].qp_num, wc[i].opcode, value);
        continue;
      }
      if (ShouldFire(FaultKind::CqeError, wc[i].qp_num, wc[i].opcode, &value)) {
        // Like the NIC: after an error completion the QP is in ERR and every
        // WR still queued on it flushes.
        ibv_qp* qp = nullptr;
        {
          std::lock_guard<std::mutex> lock(mu_);
          auto ctx = qps_.find(cq->context);
          if (ctx != qps_.end()) {
            auto it = ctx->second.find(wc[i].qp_num);
            if (it != ctx->second.end()) qp = it->second;
          }
        }
        if (qp != nullptr) {
          if (MoveQpToError(qp)) {
            wc[i].status = static_cast<ibv_wc_status>(value);
            wc[i].vendor_err = 0;
            RecordInjection(FaultKind::CqeError, wc[i].qp_num, wc[i].opcode, value);
          }
        } else {
          FailInjection("CQE QP not recorded");
        }
      }
    }
    wc[kept++] = wc[i];
  }
  return kept;
} catch (const std::exception& error) {
  FailInjection(error.what());
  return 0;
} catch (...) {
  FailInjection("polling injected CQ");
  return 0;
}

bool FaultInjector::DeviceDead(ibv_context* ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  return deadDevices_.count(ctx) != 0;
}

bool FaultInjector::PrepareAsyncEvent(ibv_qp* qp, int wcOpcode, ibv_async_event* event) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!IsAsyncKind(rule_.kind)) return false;
  if (rule_.kind == FaultKind::AsyncEvent) {
    bool supported = false;
    for (const auto& event : kAsyncEvents) supported |= event.value == rule_.value;
    if (!supported) {
      FailInjection("unsupported async event");
      return false;
    }
  }
  int value = 0;
  if (!ShouldFireLocked(rule_.kind, qp->qp_num, wcOpcode, &value)) return false;
  event->event_type =
      rule_.kind == FaultKind::AsyncEvent ? static_cast<ibv_event_type>(value) : IBV_EVENT_PORT_ERR;
  const char* state = "unchanged";
  bool portEvent = false;
  switch (event->event_type) {
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      event->element.qp = qp;
      if (!MoveQpToError(qp)) return false;
      state = "qp->ERR";
      break;
    case IBV_EVENT_CQ_ERR:
    case IBV_EVENT_DEVICE_FATAL: {
      const bool device = event->event_type == IBV_EVENT_DEVICE_FATAL;
      if (device)
        deadDevices_.insert(qp->context);
      else {
        event->element.cq = qp->send_cq;
        deadCqs_[qp->send_cq] = qp->context;
      }
      hasDeadResources_.store(true, std::memory_order_relaxed);
      for (const auto& [qpn, known] : qps_[qp->context]) {
        if (device || known->send_cq == qp->send_cq || known->recv_cq == qp->send_cq) {
          if (!MoveQpToError(known)) return false;
        }
      }
      state = device ? "device dead, known QPs->ERR" : "cq dead, known QPs->ERR";
      break;
    }
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_COMM_EST:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
      event->element.qp = qp;
      break;
    default: {
      portEvent = true;
      ibv_qp_attr attr{};
      ibv_qp_init_attr init{};
      int ret = ibv_query_qp(qp, &attr, IBV_QP_PORT, &init);
      if (ret != 0) {
        FailInjection("ibv_query_qp(PORT)", ret);
        return false;
      }
      // Some providers (ionic) report port_num 0 for a QP past INIT. A
      // single-port device leaves only one answer; on a multi-port one, refuse
      // to guess rather than raise an event the monitor cannot place.
      int port = attr.port_num;
      if (port == 0) {
        ibv_device_attr dev{};
        ret = ibv_query_device(qp->context, &dev);
        if (ret != 0 || dev.phys_port_cnt != 1) {
          FailInjection("QP port unknown on a multi-port device", ret);
          return false;
        }
        port = 1;
      }
      event->element.port_num = port;
      if (rule_.kind == FaultKind::PortDown || rule_.kind == FaultKind::PortFlap) {
        downPorts_[{qp->context, port}] =
            DownPort{Clock::time_point::max(), rule_.kind == FaultKind::PortFlap ? value : 0};
        state =
            rule_.kind == FaultKind::PortFlap ? "port down until timer" : "port down until disarm";
      }
      break;
    }
  }
  try {
    MORI_IO_WARN("Fault async event prepared: type={} qpn={} ctx={} state={}",
                 static_cast<int>(event->event_type), qp->qp_num, static_cast<void*>(qp->context),
                 state);
  } catch (...) {
  }
  if (portEvent) {
    // Port events do not call FailAll. Publish them with the override under the
    // same lock, before posting, so Disarm can only queue ACTIVE after ERR.
    PublishAsyncEventLocked(qp, wcOpcode, *event);
    return false;  // already published; the caller still performs the real post
  }
  return true;
}

void FaultInjector::PublishAsyncEvent(ibv_qp* qp, int wcOpcode,
                                      const ibv_async_event& event) noexcept try {
  std::lock_guard<std::mutex> lock(mu_);
  PublishAsyncEventLocked(qp, wcOpcode, event);
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("publishing async event");
}

void FaultInjector::PublishAsyncEventLocked(ibv_qp* qp, int wcOpcode,
                                            const ibv_async_event& event) {
  ibv_context* ctx = qp->context;
  if (!QueueEventLocked(ctx, event)) return;
  if (event.event_type == IBV_EVENT_PORT_ERR) {
    auto port = downPorts_.find({ctx, event.element.port_num});
    if (port != downPorts_.end() && port->second.durationMs > 0) {
      // Start the interval after publishing DOWN, so ACTIVE cannot overtake it.
      port->second.restoreAt = Clock::now() + std::chrono::milliseconds(port->second.durationMs);
      if (!timer_.joinable()) timer_ = std::thread([this] { PortTimerLoop(); });
      timerCv_.notify_all();
    }
  }
  RecordInjection(rule_.kind, qp->qp_num, wcOpcode, static_cast<int>(event.event_type));
}

int FaultInjector::WatchAsyncContext(ibv_context* ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = sources_.find(ctx);
  if (it != sources_.end()) return it->second.fd;
  int fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) {
    FailInjection("eventfd creation", errno);
    return -1;
  }
  try {
    sources_.emplace(ctx, AsyncSource{fd, {}});
  } catch (...) {
    close(fd);
    FailInjection("allocating async source");
    return -1;
  }
  return fd;
}

void FaultInjector::UnwatchAsyncContext(ibv_context* ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = sources_.find(ctx);
  if (it != sources_.end()) {
    close(it->second.fd);
    sources_.erase(it);
  }
  for (auto port = downPorts_.begin(); port != downPorts_.end();) {
    if (port->first.first == ctx)
      port = downPorts_.erase(port);
    else
      ++port;
  }
  // Removing a watch does not retire its live QPs/CQs. Their fault markers
  // are cleared by the owning endpoint/context's lifecycle callbacks.
  timerCv_.notify_all();
}

void FaultInjector::ReleaseAsyncContext(ibv_context* ctx) noexcept try {
  UnwatchAsyncContext(ctx);
  std::lock_guard<std::mutex> lock(mu_);
  qps_.erase(ctx);
  deadDevices_.erase(ctx);
  for (auto it = deadCqs_.begin(); it != deadCqs_.end();) {
    if (it->second == ctx)
      it = deadCqs_.erase(it);
    else
      ++it;
  }
  hasDeadResources_.store(!deadCqs_.empty() || !deadDevices_.empty(), std::memory_order_relaxed);
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("releasing async context");
}

void FaultInjector::ForgetAsyncEndpoint(ibv_context* ctx, uint32_t qpn, ibv_cq* cq) noexcept try {
  std::lock_guard<std::mutex> lock(mu_);
  auto qps = qps_.find(ctx);
  if (qps != qps_.end()) qps->second.erase(qpn);
  deadCqs_.erase(cq);
  hasDeadResources_.store(!deadCqs_.empty() || !deadDevices_.empty(), std::memory_order_relaxed);
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("forgetting async endpoint");
}

void FaultInjector::BeforeNotify(ibv_qp* qp, TransferStatus* status) noexcept try {
  if (!armed_.load(std::memory_order_relaxed)) return;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!armed_.load(std::memory_order_relaxed) || rule_.kind != FaultKind::AsyncEvent ||
        rule_.wcOpcode != IBV_WC_SEND || (rule_.qpn != 0 && rule_.qpn != qp->qp_num))
      return;
    switch (rule_.value) {
      case IBV_EVENT_QP_FATAL:
      case IBV_EVENT_QP_REQ_ERR:
      case IBV_EVENT_QP_ACCESS_ERR:
      case IBV_EVENT_CQ_ERR:
      case IBV_EVENT_DEVICE_FATAL:
        break;
      default:
        return;
    }
  }
  // The normal CQ consumer updates status after retiring the data record.
  // Wait outside mu_; no post guard remains when RdmaNotifyTransfer is entered.
  if (status == nullptr || status->WaitFor(5000) != StatusCode::SUCCESS) {
    FailInjection("data did not complete before notification fault");
  }
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("staging notification fault");
}

bool FaultInjector::QueueEventLocked(ibv_context* ctx, const ibv_async_event& event) {
  auto it = sources_.find(ctx);
  if (it == sources_.end()) {
    try {
      MORI_IO_WARN("Fault injection: no async monitor for context {}; synthetic event dropped",
                   static_cast<void*>(ctx));
    } catch (...) {
    }
    return true;  // monitor-off tests still exercise the associated state change
  }
  auto& source = it->second;
  bool wasEmpty = source.queue.empty();
  source.queue.push_back(event);
  if (wasEmpty && !SetSourceReadable(source.fd, true)) {
    source.queue.pop_back();
    FailInjection("eventfd write", errno);
    return false;
  }
  return true;
}

int FaultInjector::GetAsyncEvent(ibv_context* ctx, ibv_async_event* event) noexcept try {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sources_.find(ctx);
    if (it != sources_.end() && !it->second.queue.empty()) {
      *event = it->second.queue.front();
      it->second.queue.pop_front();
      if (it->second.queue.empty() && !SetSourceReadable(it->second.fd, false)) {
        FailInjection("eventfd read", errno);
        errno = EIO;
        return -1;
      }
      tlsSynthetic = event;
      return 0;
    }
  }
  return real_.GetAsyncEvent(ctx, event);
} catch (const std::exception& error) {
  FailInjection(error.what());
  errno = EIO;
  return -1;
} catch (...) {
  FailInjection("getting async event");
  errno = EIO;
  return -1;
}

void FaultInjector::AckAsyncEvent(ibv_async_event* event) {
  if (event == tlsSynthetic)
    tlsSynthetic = nullptr;
  else
    real_.AckAsyncEvent(event);
}

int FaultInjector::QueryPort(ibv_context* ctx, uint8_t port, ibv_port_attr* attr) noexcept try {
  int ret = real_.QueryPort(ctx, port, attr);
  if (ret == 0) {
    std::lock_guard<std::mutex> lock(mu_);
    if (downPorts_.count({ctx, port})) attr->state = IBV_PORT_DOWN;
  }
  return ret;
} catch (const std::exception& error) {
  FailInjection(error.what());
  errno = EIO;
  return -1;
} catch (...) {
  FailInjection("querying injected port");
  errno = EIO;
  return -1;
}

void FaultInjector::RestorePortLocked(ibv_context* ctx, int port) {
  downPorts_.erase({ctx, port});
  ibv_async_event event{};
  event.event_type = IBV_EVENT_PORT_ACTIVE;
  event.element.port_num = port;
  QueueEventLocked(ctx, event);
}

void FaultInjector::PortTimerLoop() try {
  std::unique_lock<std::mutex> lock(mu_);
  while (true) {
    auto next = downPorts_.end();
    for (auto it = downPorts_.begin(); it != downPorts_.end(); ++it) {
      if (it->second.restoreAt != Clock::time_point::max() &&
          (next == downPorts_.end() || it->second.restoreAt < next->second.restoreAt))
        next = it;
    }
    if (next == downPorts_.end()) {
      timerCv_.wait(lock);
    } else if (next->second.restoreAt <= Clock::now()) {
      auto key = next->first;
      RestorePortLocked(key.first, key.second);
    } else {
      const auto deadline = next->second.restoreAt;
      timerCv_.wait_until(lock, deadline);
    }
  }
} catch (const std::exception& error) {
  FailInjection(error.what());
} catch (...) {
  FailInjection("port timer");
}

VerbsOps& Verbs() { return FaultInjector::Instance(); }

}  // namespace io
}  // namespace mori
