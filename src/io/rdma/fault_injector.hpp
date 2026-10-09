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
#include <cstdint>
#include <mutex>
#include <string>

#include "src/io/rdma/verbs_ops.hpp"

namespace mori {
namespace io {

// Each kind is one verbs-level symptom; real-world causes (link down, peer
// frozen, NIC reset, ...) are expressed as one of these.
enum class FaultKind : uint8_t {
  PostSendFail,  // ibv_post_send returns `value` (errno), nothing posted
  PostRecvFail,  // ibv_post_recv returns `value` (errno)
  QpError,       // move the QP to IBV_QPS_ERR right before a send: real flush cascade
  CqeError,      // rewrite a successful CQE's status to `value` (ibv_wc_status)
  CqeDrop,       // swallow a successful CQE: the completion never arrives
};

struct FaultRule {
  FaultKind kind{FaultKind::CqeDrop};
  uint64_t skip{0};   // let this many matching ops through first
  uint64_t count{1};  // then fire this many times; 0 = forever
  int value{0};       // errno for *Fail kinds, ibv_wc_status for CqeError
  int wcOpcode{-1};   // CQE kinds: only match this ibv_wc_opcode; -1 = any
  uint32_t qpn{0};    // only match this QP number; 0 = any
};

// Parses "kind[:key=value]...", e.g. "cqe_drop:op=write:skip=10:count=1" or
// "cqe_error:value=retry_exc". Kinds: post_send_fail, post_recv_fail, qp_error,
// cqe_error, cqe_drop. Keys: skip, count, value, op (write|read|send|recv|<int>),
// qpn. Throws std::invalid_argument on a malformed spec.
FaultRule ParseFaultRule(const std::string& spec);
const char* FaultKindName(FaultKind kind);

// Decorates DirectVerbs with a single armed FaultRule. Disarmed, each hook costs
// one relaxed atomic load. Arms itself at first use from MORI_IO_FAULT if set.
class FaultInjector final : public VerbsOps {
 public:
  static FaultInjector& Instance();

  void Arm(const FaultRule& rule);
  void Disarm();
  uint64_t Fired() const { return fired_.load(std::memory_order_relaxed); }

  int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) override;
  int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) override;
  int PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) override;
  int GetCqEvent(ibv_comp_channel* ch, ibv_cq** cq, void** cqCtx) override {
    return direct_.GetCqEvent(ch, cq, cqCtx);
  }
  int GetAsyncEvent(ibv_context* ctx, ibv_async_event* event) override {
    return direct_.GetAsyncEvent(ctx, event);
  }

 private:
  FaultInjector();
  // True if the armed rule matches and is due; `value` gets the rule's value.
  bool ShouldFire(FaultKind kind, uint32_t qpn, int wcOpcode, int* value);

  DirectVerbs direct_;
  std::atomic<bool> armed_{false};
  std::atomic<uint64_t> fired_{0};
  std::mutex mu_;
  FaultRule rule_;
  uint64_t matched_{0};
};

}  // namespace io
}  // namespace mori

#endif  // MORI_IO_FAULT_INJECTION
