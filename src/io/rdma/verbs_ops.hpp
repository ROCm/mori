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

namespace mori {
namespace io {

// Seam over the ibverbs calls on MORI-IO's host data path. Every such call site
// goes through Verbs() so a build with MORI_IO_FAULT_INJECTION can swap in the
// FaultInjector decorator (src/io/rdma/fault_injector.hpp) without touching the
// call sites again. Without the macro, Verbs() is a final, stateless
// RealVerbs, so calls devirtualize and inline to plain ibv_* as before.
class VerbsOps {
 public:
  virtual ~VerbsOps() = default;
  virtual int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) = 0;
  virtual int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) = 0;
  virtual int PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) = 0;
  virtual int GetCqEvent(ibv_comp_channel* ch, ibv_cq** cq, void** cqCtx) = 0;
  virtual int GetAsyncEvent(ibv_context* ctx, ibv_async_event* event) = 0;
};

// The real calls: standard libibverbs ibv_*, not a vendor direct-verbs API.
class RealVerbs final : public VerbsOps {
 public:
  int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) override {
    return ibv_post_send(qp, wr, bad);
  }
  int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) override {
    return ibv_post_recv(qp, wr, bad);
  }
  int PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) override {
    return ibv_poll_cq(cq, numEntries, wc);
  }
  int GetCqEvent(ibv_comp_channel* ch, ibv_cq** cq, void** cqCtx) override {
    return ibv_get_cq_event(ch, cq, cqCtx);
  }
  int GetAsyncEvent(ibv_context* ctx, ibv_async_event* event) override {
    return ibv_get_async_event(ctx, event);
  }
};

#ifdef MORI_IO_FAULT_INJECTION
VerbsOps& Verbs();
#else
inline RealVerbs kRealVerbs;
inline RealVerbs& Verbs() { return kRealVerbs; }
#endif

}  // namespace io
}  // namespace mori
