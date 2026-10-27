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

#ifdef MORI_IO_FAULT_INJECTION

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "src/io/rdma/verbs_ops.hpp"

namespace mori {
namespace io {

// Each kind is one verbs-level symptom; real-world causes (link down, peer
// frozen, NIC reset, ...) are expressed as one of these.
enum class FaultKind : uint8_t {
  PostSendFail,  // ibv_post_send returns `value` (errno), nothing posted
  PostRecvFail,  // ibv_post_recv returns `value` (errno)
  QpError,       // move the QP to IBV_QPS_ERR right before a send: real flush cascade
  CqeError,      // rewrite a successful CQE's status to `value` (ibv_wc_status) and move
                 // its QP to IBV_QPS_ERR, as the NIC does after any error completion
  CqeDrop,       // swallow a successful CQE: the completion never arrives
  AsyncEvent,    // synthesize a verbs async event after posting
  PortDown,      // PORT_ERR until explicit Disarm(), then PORT_ACTIVE
  PortFlap,      // PORT_ERR followed by PORT_ACTIVE after value milliseconds
};

struct FaultRule {
  FaultKind kind{FaultKind::CqeDrop};
  uint64_t skip{0};   // let this many matching ops through first
  uint64_t count{1};  // then fire this many times; 0 = forever
  int value{0};       // errno, WC status, async event type, or flap duration (ms)
  int wcOpcode{-1};   // only match this ibv_wc_opcode; -1 = any
  uint32_t qpn{0};    // only match this QP number; 0 = any
};

// Parses "kind[:key=value]...", e.g. "cqe_drop:op=write:skip=10:count=1" or
// "cqe_error:value=retry_exc". Kinds: post_send_fail, post_recv_fail, qp_error,
// cqe_error, cqe_drop, async_event, port_down, port_flap. Keys: skip, count,
// value, op (write|read|send|recv|<int>),
// qpn. Throws std::invalid_argument on a malformed spec.
FaultRule ParseFaultRule(const std::string& spec);
const char* FaultKindName(FaultKind kind);

// Decorates RealVerbs with a single armed FaultRule. Without active faults the
// data path checks relaxed atomics only. Arms at first use from MORI_IO_FAULT.
class FaultInjector final : public VerbsOps {
 public:
  static FaultInjector& Instance();

  void Arm(const FaultRule& rule);
  void Disarm() noexcept;
  uint64_t Fired() const { return fired_.load(std::memory_order_relaxed); }
  bool Failed() const { return failed_.load(std::memory_order_relaxed); }

  int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) override;
  int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) override;
  int PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) override;
  int GetCqEvent(ibv_comp_channel* ch, ibv_cq** cq, void** cqCtx) override {
    return real_.GetCqEvent(ch, cq, cqCtx);
  }
  int GetAsyncEvent(ibv_context* ctx, ibv_async_event* event) noexcept override;
  void AckAsyncEvent(ibv_async_event* event) override;
  int QueryPort(ibv_context* ctx, uint8_t port, ibv_port_attr* attr) noexcept override;
  int WatchAsyncContext(ibv_context* ctx) override;
  void UnwatchAsyncContext(ibv_context* ctx) override;
  void ReleaseAsyncContext(ibv_context* ctx) noexcept override;
  void ForgetAsyncEndpoint(ibv_context* ctx, uint32_t qpn, ibv_cq* cq) noexcept override;
  void BeforeNotify(ibv_qp* qp, TransferStatus* status) noexcept override;

 private:
  FaultInjector();
  // True if the armed rule matches and is due; `value` gets the rule's value.
  bool ShouldFire(FaultKind kind, uint32_t qpn, int wcOpcode, int* value);
  bool ShouldFireLocked(FaultKind kind, uint32_t qpn, int wcOpcode, int* value);
  void RecordInjection(FaultKind kind, uint32_t qpn, int wcOpcode, int value) noexcept;
  void FailInjection(const char* operation, int error = 0) noexcept;
  // CQEs carry only a QP number; remember QPs seen at post time so an injected
  // error CQE can move its QP to ERR. Only recorded while armed.
  void RememberQp(ibv_qp* qp);
  bool MoveQpToError(ibv_qp* qp);
  bool PrepareAsyncEvent(ibv_qp* qp, int wcOpcode, ibv_async_event* event);
  bool DeviceDead(ibv_context* ctx);
  void PublishAsyncEvent(ibv_qp* qp, int wcOpcode, const ibv_async_event& event) noexcept;
  void PublishAsyncEventLocked(ibv_qp* qp, int wcOpcode, const ibv_async_event& event);
  bool QueueEventLocked(ibv_context* ctx, const ibv_async_event& event);
  void RestorePortLocked(ibv_context* ctx, int port);
  void PortTimerLoop();

  struct AsyncSource {
    int fd;
    std::deque<ibv_async_event> queue;
  };
  using Clock = std::chrono::steady_clock;
  using PortKey = std::pair<ibv_context*, int>;
  struct DownPort {
    Clock::time_point restoreAt;
    int durationMs;  // zero = port_down; positive = flap, even before publication
  };

  RealVerbs real_;
  std::atomic<bool> armed_{false};
  std::atomic<uint64_t> fired_{0};
  std::atomic<bool> failed_{false};
  std::mutex mu_;
  FaultRule rule_;
  uint64_t matched_{0};
  uint64_t triggered_{0};  // rule count includes attempted injections, even if they fail
  // QP numbers are unique only within one context/device.
  std::unordered_map<ibv_context*, std::unordered_map<uint32_t, ibv_qp*>> qps_;
  std::unordered_map<ibv_context*, AsyncSource> sources_;
  std::unordered_map<ibv_cq*, ibv_context*> deadCqs_;
  std::unordered_set<ibv_context*> deadDevices_;
  std::atomic<bool> hasDeadResources_{false};
  std::map<PortKey, DownPort> downPorts_;
  std::condition_variable timerCv_;
  std::thread timer_;
};

}  // namespace io
}  // namespace mori

#endif  // MORI_IO_FAULT_INJECTION
