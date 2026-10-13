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
//
// ---------------------------------------------------------------------------
// Host side of the AsyncLL kernels.
//
// Everything about the Cfg, the Request and the Args surface is borrowed from
// ep_internode_spec.hpp unchanged -- AsyncLL is specialised on the same shape and
// launched against the same argument struct, so giving it a parallel set would
// mean two things to keep in step for no behavioural difference. What is new is
// the kernel enum, the entry names, which *_body each TU instantiates and the
// per-pass geometry.
//
// Host-only; the device TU must never include it.
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

#include "mori/jit/v2/spec.hpp"
#include "mori/ops/dispatch_combine_v2/ep_internode_spec.hpp"

namespace mori {
namespace ops {
namespace v2 {

// The nine AsyncLL entry points, in launch order within each phase.
//
// The split is the feature: DispatchSlotAssign/SendCopy/SendTransfer are what
// `dispatch_send` launches and they return without waiting on anything, while
// DispatchRecvTransfer/RecvCopy are what `dispatch_recv` launches. Whatever the
// caller runs between the two overlaps the transfers.
enum class EpAsyncLLKernel {
  DispatchSlotAssign,
  DispatchSendCopy,
  DispatchSendTransfer,
  DispatchRecvTransfer,
  DispatchRecvCopy,
  CombineSendCopy,
  CombineSendTransfer,
  CombineRecvTransfer,
  CombineRecvCopy,
  // Transfer over the flat LSA VA instead of the copy engine. Separate entries
  // rather than a runtime branch, for the reason v2 and v2_ll are separate: the
  // choice is fixed for an op, and a branch would compile both bodies into every
  // module.
  // SendCopy and SendTransferLsa fused: the gather lands straight in the peer,
  // so the staging hop the engine needed is not there to copy out of. Replaces
  // BOTH of them in the LSA sequence; there is no SDMA counterpart because that
  // transport does need the local contiguous source.
  DispatchSendCopyLsa,
  DispatchSendTransferLsa,
  DispatchRecvTransferLsa,
  CombineSendTransferLsa,
  CombineRecvTransferLsa,
};

// True for the passes that take the device communicator, i.e. the ones that
// touch an engine. The other five are local copies and need no comm.
bool EpAsyncLLTakesComm(EpAsyncLLKernel kind);

std::string EpAsyncLLEntryName(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind);
std::string EpAsyncLLRenderSource(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind);
mori::jit::v2::LaunchGeometry EpAsyncLLGeometry(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind);

// Grids that divide the work per destination rank must be an exact multiple of
// worldSize: DispatchRecvCopy derives `blocksPerPe = gridDim.x / worldSize` and
// then `destPe = blockIdx.x / blocksPerPe`, so a remainder leaves the tail blocks
// decoding to a PE that does not exist. v1 aligns the same way at its launch
// site (`mp - mp % worldSize`).
constexpr int EpAsyncLLAlignGrid(int grid, int worldSize) {
  const int aligned = grid - (grid % worldSize);
  return aligned > 0 ? aligned : worldSize;
}

#define MORI_EP_ASYNCLL_DECLARE_SPEC(ClassName, planName)                         \
  class ClassName : public mori::jit::v2::KernelSpec<ClassName, EpInterNodeCfg> { \
   public:                                                                        \
    using Args = EpInterNodeCcoArgs;                                              \
    static constexpr const char* kName = planName;                                \
    static std::string EntryName(const Cfg& cfg);                                 \
    static std::string RenderSource(const Cfg& cfg);                              \
    static mori::jit::v2::LaunchGeometry Geometry(const Cfg& cfg);                \
    static const std::vector<std::string>& SourceDeps();                          \
  }

MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchSlotAssignSpec, "ep_asyncll_dispatch_slotassign");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchSendCopySpec, "ep_asyncll_dispatch_sendcopy");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchSendTransferSpec, "ep_asyncll_dispatch_sendtransfer");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchRecvTransferSpec, "ep_asyncll_dispatch_recvtransfer");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchRecvCopySpec, "ep_asyncll_dispatch_recvcopy");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineSendCopySpec, "ep_asyncll_combine_sendcopy");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineSendTransferSpec, "ep_asyncll_combine_sendtransfer");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineRecvTransferSpec, "ep_asyncll_combine_recvtransfer");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineRecvCopySpec, "ep_asyncll_combine_recvcopy");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchSendCopyLsaSpec,
                             "ep_asyncll_dispatch_sendcopy_lsa");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchSendTransferLsaSpec,
                             "ep_asyncll_dispatch_sendtransfer_lsa");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLDispatchRecvTransferLsaSpec,
                             "ep_asyncll_dispatch_recvtransfer_lsa");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineSendTransferLsaSpec,
                             "ep_asyncll_combine_sendtransfer_lsa");
MORI_EP_ASYNCLL_DECLARE_SPEC(EpAsyncLLCombineRecvTransferLsaSpec,
                             "ep_asyncll_combine_recvtransfer_lsa");

#undef MORI_EP_ASYNCLL_DECLARE_SPEC

}  // namespace v2
}  // namespace ops
}  // namespace mori
