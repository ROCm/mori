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
// Host-only. Never includes the device kernel header -- that is what keeps this
// target compilable without hipcc.

#include "mori/ops/dispatch_combine_v2/ep_asyncll_spec.hpp"

#include <stdexcept>
#include <string>

#include "mori/jit/v2/toolchain.hpp"

namespace mori {
namespace ops {
namespace v2 {

namespace {

struct AsyncLLDescriptor {
  const char* tag;   // goes in the entry name
  const char* body;  // the *_body function in ep_asyncll_kernel.hpp
};

AsyncLLDescriptor DescriptorFor(EpAsyncLLKernel kind) {
  switch (kind) {
    case EpAsyncLLKernel::DispatchSlotAssign:
      return {"dispatch_slotassign", "EpDispatchAsyncLLSlotAssign_body"};
    case EpAsyncLLKernel::DispatchSendCopy:
      return {"dispatch_sendcopy", "EpDispatchAsyncLLSendCopy_body"};
    case EpAsyncLLKernel::DispatchSendTransfer:
      return {"dispatch_sendtransfer", "EpDispatchAsyncLLSendTransfer_body"};
    case EpAsyncLLKernel::DispatchRecvTransfer:
      return {"dispatch_recvtransfer", "EpDispatchAsyncLLRecvTransfer_body"};
    case EpAsyncLLKernel::DispatchRecvCopy:
      return {"dispatch_recvcopy", "EpDispatchAsyncLLRecvCopy_body"};
    case EpAsyncLLKernel::CombineSendCopy:
      return {"combine_sendcopy", "EpCombineAsyncLLSendCopy_body"};
    case EpAsyncLLKernel::CombineSendTransfer:
      return {"combine_sendtransfer", "EpCombineAsyncLLSendTransfer_body"};
    case EpAsyncLLKernel::CombineRecvTransfer:
      return {"combine_recvtransfer", "EpCombineAsyncLLRecvTransfer_body"};
    case EpAsyncLLKernel::CombineRecvCopy:
      return {"combine_recvcopy", "EpCombineAsyncLLRecvCopy_body"};
    case EpAsyncLLKernel::DispatchSendCopyLsa:
      return {"dispatch_sendcopy_lsa", "EpDispatchAsyncLLSendCopyLsa_body"};
    case EpAsyncLLKernel::DispatchSendTransferLsa:
      return {"dispatch_sendtransfer_lsa", "EpDispatchAsyncLLSendTransferLsa_body"};
    case EpAsyncLLKernel::DispatchRecvTransferLsa:
      return {"dispatch_recvtransfer_lsa", "EpDispatchAsyncLLRecvTransferLsa_body"};
    case EpAsyncLLKernel::CombineSendTransferLsa:
      return {"combine_sendtransfer_lsa", "EpCombineAsyncLLSendTransferLsa_body"};
    case EpAsyncLLKernel::CombineRecvTransferLsa:
      return {"combine_recvtransfer_lsa", "EpCombineAsyncLLRecvTransferLsa_body"};
  }
  throw std::runtime_error("mori ep asyncll jit: unknown kernel");
}

// Same subtrees the internode family invalidates on: the AsyncLL header includes
// ep_internode_kernel.hpp, so its dependency set is a superset of nothing new.
const std::vector<std::string>& EpAsyncLLSourceDeps() {
  static const std::vector<std::string> deps{"include/mori", "src/ops/dispatch_combine_v2",
                                             "src/cco"};
  return deps;
}

}  // namespace

bool EpAsyncLLTakesComm(EpAsyncLLKernel kind) {
  switch (kind) {
    case EpAsyncLLKernel::DispatchSendTransfer:
    case EpAsyncLLKernel::DispatchRecvTransfer:
    case EpAsyncLLKernel::CombineSendTransfer:
    case EpAsyncLLKernel::CombineRecvTransfer:
      return true;
    // The LSA variants cross over the flat VA, so they take no communicator --
    // the window handle they already have is the whole addressing mechanism.
    default:
      return false;
  }
}

std::string EpAsyncLLEntryName(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind) {
  std::string name = "mori_ep_asyncll_";
  if (cfg.asyncLlMode == 2)
    name += "fused_";
  else if (cfg.asyncLlMode == 1)
    name += "early_";
  name += DescriptorFor(kind).tag;
  name += '_';
  name += EpInterNodeDTypeTag(cfg.dtype);
  return name;
}

std::string EpAsyncLLRenderSource(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind) {
  // A zero divisor becomes a division by a literal zero once the cfg is an NTTP,
  // diagnosed (if at all) against generated source no one can trace back. The
  // Request path already rejects it; this is the funnel every compile goes
  // through, including a hand-initialised Cfg.
  if (!EpInterNodeKernelCfgIsValid(cfg.kernelCfg)) {
    throw std::runtime_error("mori ep asyncll jit: unusable config " + Render(cfg.kernelCfg));
  }

  // DispatchRecvCopy derives every PE's output offset from a prefix sum over one
  // wavefront -- lane i holds PE i's count. Past the end __shfl returns the
  // issuing lane's own value instead of faulting, so a worldSize wider than the
  // wave folds several PEs onto one offset and the ranks disagree about where
  // their tokens went. The kernel can only static_assert the absolute bound,
  // because waveSize is host-side state and not part of the rendered cfg; this
  // is the funnel that knows the real one.
  if (cfg.waveSize > 0 && cfg.kernelCfg.worldSize > cfg.waveSize) {
    throw std::runtime_error(
        "mori ep asyncll jit: worldSize " + std::to_string(cfg.kernelCfg.worldSize) +
        " exceeds the wavefront (" + std::to_string(cfg.waveSize) +
        " lanes on this arch); the per-PE offset prefix sum is one wavefront wide");
  }

  if (cfg.asyncLlMode < 0 || cfg.asyncLlMode > 2)
    throw std::runtime_error("AsyncLL mode must be 0 (legacy), 1 (early), or 2 (fused CTA)");
  const bool early = cfg.asyncLlMode != 0;
  const bool fused = cfg.asyncLlMode == 2;
  if (early && cfg.kernelCfg.worldSize != cfg.kernelCfg.gpuPerNode)
    throw std::runtime_error("AsyncLL early notification requires a single node");
  if (early && (kind == EpAsyncLLKernel::DispatchSendCopyLsa ||
                kind == EpAsyncLLKernel::DispatchSendTransferLsa ||
                kind == EpAsyncLLKernel::DispatchRecvTransferLsa ||
                kind == EpAsyncLLKernel::CombineSendTransferLsa ||
                kind == EpAsyncLLKernel::CombineRecvTransferLsa))
    throw std::runtime_error("AsyncLL early notification requires the SDMA transport");
  const AsyncLLDescriptor descriptor = DescriptorFor(kind);
  const std::string entry = EpAsyncLLEntryName(cfg, kind);

  std::string source = "// mori jit — generated, do not edit.\n";
  source += "#define MORI_ASYNCLL_EARLY_NOTIFY " + std::to_string(early ? 1 : 0) + "\n";
  source += "#define MORI_ASYNCLL_FUSED_RECV " + std::to_string(fused ? 1 : 0) + "\n";
  source +=
      "#include \"src/ops/dispatch_combine_v2/ep_asyncll_kernel.hpp\"\n"
      "constexpr ::mori::ops::v2::EpInterNodeKernelCfg kConfig = ";
  source += Render(cfg.kernelCfg);
  source += ";\nusing TokT = ";
  source += EpInterNodeDTypeName(cfg.dtype);
  source += ";\n";
  // The entry macros come from ep_internode_kernel.hpp, which the AsyncLL header
  // includes; the _LOCAL form is the one whose body takes no communicator.
  const bool fusedCopy = fused && (kind == EpAsyncLLKernel::DispatchRecvCopy ||
                                   kind == EpAsyncLLKernel::CombineRecvCopy);
  source += (EpAsyncLLTakesComm(kind) || fusedCopy) ? "MORI_EP_INTERNODE_CCO_ENTRY("
                                                    : "MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(";
  source += entry;
  source += ", ";
  if (fusedCopy) {
    source += kind == EpAsyncLLKernel::DispatchRecvCopy ? "EpDispatchAsyncLLRecvCopyFused_body"
                                                        : "EpCombineAsyncLLRecvCopyFused_body";
  } else {
    source += descriptor.body;
  }
  source += ")\n";
  return source;
}

mori::jit::v2::LaunchGeometry EpAsyncLLGeometry(const EpInterNodeCfg& cfg, EpAsyncLLKernel kind) {
  mori::jit::v2::LaunchGeometry geometry;
  geometry.blockX = static_cast<unsigned>(EpInterNodeBlockThreads(cfg));
  const int worldSize = cfg.kernelCfg.worldSize;

  switch (kind) {
    // The copy passes fan out over the whole device. Aligned to worldSize because
    // DispatchRecvCopy partitions its grid per source rank; the other two do not
    // need it but share the number so a profile lines the three up.
    case EpAsyncLLKernel::DispatchSlotAssign:
    case EpAsyncLLKernel::DispatchSendCopy:
    // The fused LSA copy belongs here and not with the transfers: it walks token
    // ENTRIES like the other copy passes, and reaches the peer by pointer rather
    // than by owning a slice of the grid.
    case EpAsyncLLKernel::DispatchSendCopyLsa:
    case EpAsyncLLKernel::DispatchRecvCopy:
    case EpAsyncLLKernel::CombineSendCopy:
      geometry.gridX = static_cast<unsigned>(EpAsyncLLAlignGrid(cfg.mpCount, worldSize));
      geometry.sharedBytes = 0;
      break;

    // One block per destination rank, one warp in it. More would not make an SDMA
    // copy faster -- a peer is one xGMI link -- and several warps issuing on one
    // queue serialise on its in-order commit chain, so the narrow block is the
    // point rather than a saving.
    case EpAsyncLLKernel::DispatchSendTransfer:
    case EpAsyncLLKernel::DispatchRecvTransfer:
    case EpAsyncLLKernel::CombineSendTransfer:
    case EpAsyncLLKernel::CombineRecvTransfer:
      geometry.gridX = static_cast<unsigned>(worldSize);
      geometry.blockX = static_cast<unsigned>(cfg.waveSize);
      geometry.sharedBytes = 0;
      break;

    // The opposite shape: a peer store has no engine to hand the bytes to, so the
    // transfer is only as wide as the grid. Aligned to worldSize because the
    // bodies derive `destPe = globalWarpId / (globalWarpNum / worldSize)` and a
    // remainder would leave the tail warps addressing a PE that does not exist.
    case EpAsyncLLKernel::DispatchSendTransferLsa:
    case EpAsyncLLKernel::DispatchRecvTransferLsa:
    case EpAsyncLLKernel::CombineSendTransferLsa:
    case EpAsyncLLKernel::CombineRecvTransferLsa:
      geometry.gridX = static_cast<unsigned>(EpAsyncLLAlignGrid(cfg.mpCount, worldSize));
      geometry.sharedBytes = 0;
      break;

    case EpAsyncLLKernel::CombineRecvCopy:
      geometry.gridX = static_cast<unsigned>(EpAsyncLLAlignGrid(cfg.mpCount, worldSize));
      // Reuses the internode formula, which sizes two pointer arrays per warp.
      // AsyncLL's combine uses only the first (it does not fold weights), so this
      // over-reserves by half -- cheap, and one formula is one thing to get wrong
      // instead of two.
      geometry.sharedBytes = static_cast<unsigned>(EpInterNodeCombineSharedBytes(cfg));
      break;
  }
  return geometry;
}

#define MORI_EP_ASYNCLL_DEFINE_SPEC(ClassName, KIND)                  \
  std::string ClassName::EntryName(const Cfg& cfg) {                  \
    return EpAsyncLLEntryName(cfg, EpAsyncLLKernel::KIND);            \
  }                                                                   \
  std::string ClassName::RenderSource(const Cfg& cfg) {               \
    return EpAsyncLLRenderSource(cfg, EpAsyncLLKernel::KIND);         \
  }                                                                   \
  mori::jit::v2::LaunchGeometry ClassName::Geometry(const Cfg& cfg) { \
    return EpAsyncLLGeometry(cfg, EpAsyncLLKernel::KIND);             \
  }                                                                   \
  const std::vector<std::string>& ClassName::SourceDeps() { return EpAsyncLLSourceDeps(); }

MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchSlotAssignSpec, DispatchSlotAssign)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchSendCopySpec, DispatchSendCopy)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchSendTransferSpec, DispatchSendTransfer)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchRecvTransferSpec, DispatchRecvTransfer)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchRecvCopySpec, DispatchRecvCopy)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineSendCopySpec, CombineSendCopy)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineSendTransferSpec, CombineSendTransfer)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineRecvTransferSpec, CombineRecvTransfer)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineRecvCopySpec, CombineRecvCopy)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchSendCopyLsaSpec, DispatchSendCopyLsa)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchSendTransferLsaSpec, DispatchSendTransferLsa)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLDispatchRecvTransferLsaSpec, DispatchRecvTransferLsa)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineSendTransferLsaSpec, CombineSendTransferLsa)
MORI_EP_ASYNCLL_DEFINE_SPEC(EpAsyncLLCombineRecvTransferLsaSpec, CombineRecvTransferLsa)

#undef MORI_EP_ASYNCLL_DEFINE_SPEC

}  // namespace v2
}  // namespace ops
}  // namespace mori

// ===========================================================================
// Plan registration. The Request, the Cfg builder and the Args schema are the
// internode family's -- only the Spec and the kernel enum differ per plan.
// ===========================================================================

#include "mori/jit/v2/plan_api.hpp"

namespace {

// AsyncLL is specialised on the same shape as the internode family, so it reuses
// MakeEpInterNodeCfg. That function also takes an EpInterNodeKernel to decide
// geometry, but AsyncLL supplies its own in EpAsyncLLGeometry and never reads
// what it picked -- Dispatch is passed purely because the parameter is not
// optional.
mori::ops::v2::EpInterNodeCfg EpAsyncLLCfgFromFields(const mori::jit::v2::FieldBag& fields) {
  using namespace mori::ops::v2;
  EpInterNodeRequest request;
  const EpInterNodeRequest defaults{};
  VisitFields(request, defaults, [&fields](const char* name, auto& slot, const auto&) {
    using mori::jit::v2::WireAssign;
    if (fields.Has(name)) WireAssign(slot, fields.Get(name, 0));
  });
  return MakeEpInterNodeCfg(mori::jit::v2::GetToolchain().arch, request,
                            EpInterNodeKernel::Dispatch, EpInterNodeFamily::AsyncLL);
}

int EpAsyncLLNoPrecompile(const std::string&) { return 0; }

}  // namespace

#define MORI_EP_ASYNCLL_DEFINE_PLAN(planName, ClassName)                                 \
  MORI_JIT_DEFINE_PLAN(planName, mori::ops::v2::ClassName, EpAsyncLLCfgFromFields,       \
                       mori::ops::v2::EpInterNodeRequestSchema, mori::ops::v2::Describe, \
                       EpAsyncLLNoPrecompile, mori::ops::v2::EpInterNodeCcoArgs,         \
                       mori::ops::v2::EpInterNodeArgsSchema())

MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_slotassign, EpAsyncLLDispatchSlotAssignSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_sendcopy, EpAsyncLLDispatchSendCopySpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_sendtransfer, EpAsyncLLDispatchSendTransferSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_recvtransfer, EpAsyncLLDispatchRecvTransferSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_recvcopy, EpAsyncLLDispatchRecvCopySpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_sendcopy, EpAsyncLLCombineSendCopySpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_sendtransfer, EpAsyncLLCombineSendTransferSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_recvtransfer, EpAsyncLLCombineRecvTransferSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_recvcopy, EpAsyncLLCombineRecvCopySpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_sendcopy_lsa, EpAsyncLLDispatchSendCopyLsaSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_sendtransfer_lsa,
                            EpAsyncLLDispatchSendTransferLsaSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_dispatch_recvtransfer_lsa,
                            EpAsyncLLDispatchRecvTransferLsaSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_sendtransfer_lsa,
                            EpAsyncLLCombineSendTransferLsaSpec)
MORI_EP_ASYNCLL_DEFINE_PLAN(ep_asyncll_combine_recvtransfer_lsa,
                            EpAsyncLLCombineRecvTransferLsaSpec)
