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

// ===========================================================================
// collectives_facade_impl.hpp
//
// Out-of-line definitions for CollectivesFacade (the XLA push/pull collectives):
// the Run* enum entry points, the templated reduceScatterImpl/allReduceImpl, and
// the host-side resolveMode/resolveLogS helpers. These contain kernel-launch
// syntax and pull in the __global__ kernel headers, so they are compiled only by
// the single translation unit that defines MORI_KERNELS_IMPL (a HIP compiler) --
// e.g. the XLA device TU mori_kernels.cu.cc, or each standalone MORI example
// binary. Every other TU sees only the declarations in collectives_facade.hpp
// and links against this TU's symbols.
//
// NOTE: do not include this header directly. It is included from the bottom of
// collectives_facade.hpp, guarded by `#if defined(MORI_KERNELS_IMPL)`, and
// depends on everything that header declares (CollectivesFacade, RsMode,
// FACADE_PRINTF, the detail:: reduce helpers, VecBytes, ...).
// ===========================================================================
#pragma once

// Device compilation is restricted to gfx942 / gfx950 / gfx1250: the host pass
// and those three arches get the real kernel-launch definitions (and pull in the
// reduction-op templates SumOp/MaxOp/... from the kernel headers below). Every
// other offload arch skips the kernel headers and gets the Run* stubs in the
// #else branch, so fatbin builds do not pull SDMA/b128 device code.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx1250__)

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include "mori/collective/XLA/all_gather_kernels.hpp"
#include "mori/collective/XLA/all_reduce_kernels.hpp"
#include "mori/collective/XLA/all_to_all_kernels.hpp"
#include "mori/collective/XLA/collective_permute_kernels.hpp"
#include "mori/collective/XLA/reduce_scatter_kernels.hpp"

// ---------------------------------------------------------------------------
// Real Run* definitions: the host pass and gfx942/gfx950/gfx1250 device compile
// land here and get the actual kernel launches. (The #else branch below stubs
// every Run* to hipErrorNotSupported for the other device arches.)
// ---------------------------------------------------------------------------

namespace mori {
namespace collective {
namespace detail {
// Device-wide barrier: dummy no-op. The old shmem device barrier
// (ShmemBarrierAllThread) assumed shmem init, which is incompatible with the
// CCO-backed facade. Callers currently barrier host-side via ccoBarrierAll; a
// device-side CCO barrier can be wired here later if RunBarrier is needed.
__global__ void BarrierKernel() {}

// The pull kernels take the peer count as a template argument so their
// per-peer register tile (M positions x NPES peers) stays in VGPRs, so the
// runtime npes has to be turned into a compile-time constant. Only the team
// sizes that actually ship are instantiated; any other count is REJECTED
// rather than rounded up to the nearest instantiation, which would silently
// read past the end of the team.
template <class F>
hipError_t DispatchNpes(int npes, F&& f) {
#define FACADE_NPES_CASE(n) \
  case n:                   \
    return std::forward<F>(f)(std::integral_constant<int, n>{});
  switch (npes) {
    FACADE_NPES_CASE(2)
    FACADE_NPES_CASE(4)
    FACADE_NPES_CASE(8)
    default:
      FACADE_PRINTF("pull mode supports npes 2, 4 or 8, got %d", npes);
      return hipErrorInvalidConfiguration;
  }
#undef FACADE_NPES_CASE
}

}  // namespace detail

auto CollectivesFacade::resolveMode(size_t totalBytes, bool allReduce) const -> RsMode
{
  if (mode_ != RsMode::kAuto) return mode_;
  if (nPes_ != 2 && nPes_ != 4 && nPes_ != 8) return RsMode::kPush;
  size_t pullMax = kAutoPullMaxBytes;
  if (allReduce && nPes_ == 4) pullMax = kAutoPullMaxBytesAr4;
  if (!allReduce && nPes_ == 8) pullMax = kAutoPullMaxBytesRs;
  return totalBytes < pullMax ? RsMode::kPull : RsMode::kPush;
}

int CollectivesFacade::resolveLogS(size_t chunkBytes, bool allReduce) const {
  if (logS_ >= 0) return logS_;
  if (allReduce) return kAutoAllReduceLogS;
  int logS = 0;
  while (logS < kAutoMaxLogS && (chunkBytes >> (logS + 1)) >= kAutoMinSliceBytes) logS++;
  return logS;
}

template <class T, class Op>
hipError_t CollectivesFacade::reduceScatterImpl(const void* input_v, void* output_v, size_t count,
                                                hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename detail::ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = count;
  const size_t chunkBytes = chunkElems * sizeof(T);
  const RsMode mode = resolveMode(chunkBytes * nPes_, /*allReduce=*/false);
  // Staging is only needed by the push path; pull reads peers directly (no staging).
  if (mode != RsMode::kPull && (nPes_ - 1) * chunkBytes > state_.stagingBytes) {
    FACADE_PRINTF("ReduceScatter: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }

  // The reduce runs on a compute type derived from T (float reduces as float,
  // bf16 reduces as a packed pair). Data buffers stay physically T; counts
  // passed to the kernels are in packs (kPack = sizeof(ComputeT)/sizeof(T)).
  constexpr size_t kPack = sizeof(ComputeT) / sizeof(T);
  const size_t chunkElemsC = chunkElems / kPack;
  constexpr int NumPushVecs = 8, NumPullVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ComputeT);
  constexpr int kThreads = 256;
  size_t totalVecs = chunkElemsC / (VecSize * NumPushVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, state_.MP_count));
  // MORI_CCL_MAX_NBLOCKS tightens (never raises) the grid; it stays co-resident.
  if (state_.maxNBlocks > 0) blocks = std::min(blocks, state_.maxNBlocks);

  int logS = resolveLogS(chunkBytes, /*allReduce=*/false);
  const size_t maxSlicesByData = std::max<size_t>(1, chunkElemsC / VecSize);
  while (logS > 0 && (1ULL << logS) > maxSlicesByData) logS--;

  if (mode == RsMode::kPull) {
    // The kernel resolves peer pe's copy of `input` device-side by the same
    // flat-VA rank delta the push path uses: input + (pe - myPe)*stride4G<<32,
    // with stride4G read from heapWin_.
    return detail::DispatchNpes(nPes_, [&](auto NPES_c) {
      ReduceScatterPullKernel<NumPullVecs, decltype(NPES_c)::value, ReduceOp>
          <<<blocks, kThreads, 0, stream>>>(myPe_, state_.heapWin,
                                  reinterpret_cast<const ComputeT*>(input),
                                  reinterpret_cast<ComputeT*>(output), chunkElemsC,
                                  state_.groupCounters, state_.barrierCtr);
      return hipGetLastError();
    });
  } else {
    // Every block loops over all S slices (no block-to-slice mapping), so the
    // grid needs no rounding to a multiple of S.
    ReduceScatterPushKernel<NumPushVecs, ReduceOp><<<blocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<ComputeT*>(output),
        state_.groupCounters, chunkElemsC, state_.devComm, state_.heapWin,
        reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(state_.staging), state_.barrierCtr);
  }
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to reduceScatterImpl for F32/BF16/F16/S32/S64
// and all four reduction ops.
hipError_t CollectivesFacade::RunReduceScatter(const void* input, void* output, size_t count,
                                               DataType dt, ReduceOpKind op, hipStream_t stream) {
  auto go = [=](auto typeTag) {
    using T = decltype(typeTag);
    return detail::DispatchReduceOp<T>(op, [=](auto reduceOp) {
      using Op = decltype(reduceOp);
      return this->reduceScatterImpl<T, Op>(input, output, count, stream);
    });
  };
  return detail::DispatchReduceType(dt, go);
}

template <class T, class Op>
hipError_t CollectivesFacade::allReduceImpl(const void* input_v, void* output_v, size_t numElems,
                                            hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename detail::ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = numElems / nPes_;
  const size_t chunkBytes = chunkElems * sizeof(T);
  const RsMode mode = resolveMode(chunkBytes * nPes_, /*allReduce=*/true);
  // Only the push path stages through peer-writable scratch; pull reads peers
  // directly and gathers straight into output.
  if (mode != RsMode::kPull && (nPes_ - 1) * chunkBytes > state_.stagingBytes) {
    FACADE_PRINTF("AllReduce: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }

  constexpr size_t kPack = sizeof(ComputeT) / sizeof(T);
  const size_t chunkElemsC = chunkElems / kPack;
  constexpr int NumPushVecs = 8, NumPullVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ComputeT);
  constexpr int kThreads = 256;
  size_t totalVecs = chunkElemsC / (VecSize * NumPushVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  // Cap the grid to the SM count so all blocks are co-resident: push needs it
  // for the multi-producer broadcast submitPacket ordering, pull for the
  // entry go-flag spin.
  int blocks = std::min(wantBlocks, std::max(1, state_.MP_count));
  // MORI_CCL_MAX_NBLOCKS tightens (never raises) the grid; it stays co-resident.
  if (state_.maxNBlocks > 0) blocks = std::min(blocks, state_.maxNBlocks);

  int logS = resolveLogS(chunkBytes, /*allReduce=*/true);
  const size_t maxSlicesByData = std::max<size_t>(1, chunkElemsC / VecSize);
  while (logS > 0 && (1ULL << logS) > maxSlicesByData) logS--;

  if (mode == RsMode::kPull) {
    // Fused pull: reduce this PE's shard straight off the peers' inputs and fan
    // the result out to every peer from registers, in one pass. No SDMA and no
    // staging; entry and exit PushEntryBarrier rounds on barrierCtr_.
    return detail::DispatchNpes(nPes_, [&](auto NPES_c) {
      AllReducePullKernel<NumPullVecs, decltype(NPES_c)::value, ReduceOp>
          <<<blocks, kThreads, 0, stream>>>(myPe_, state_.heapWin,
                                            reinterpret_cast<const ComputeT*>(input),
                                            reinterpret_cast<ComputeT*>(output), chunkElemsC,
                                            state_.groupCounters, state_.barrierCtr);
      return hipGetLastError();
    });
  }

  // Sliced push reduce-scatter + pipelined per-slice broadcast, all in one kernel.
  // Every block loops over all S slices (no block-to-slice mapping), so the grid
  // needs no rounding to a multiple of S.
  // input/output/staging all live in the static heap (heapWin_); the kernel
  // derives each intra-heap offset from the raw pointer via ccoGetLocalPtr.
  AllReducePushKernel<NumPushVecs, ReduceOp><<<blocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(output), state_.groupCounters, chunkElemsC, state_.devComm,
        reinterpret_cast<ComputeT*>(state_.staging), state_.heapWin, state_.barrierCtr);
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to allReduceImpl for F32/BF16/F16/S32/S64
// and all four reduction ops.
hipError_t CollectivesFacade::RunAllReduce(const void* input, void* output, size_t numElems,
                                           DataType dt, ReduceOpKind op, hipStream_t stream) {
  auto go = [&](auto typeTag) {
    using T = decltype(typeTag);
    return detail::DispatchReduceOp<T>(op, [&](auto reduceOp) {
      using Op = decltype(reduceOp);
      return this->allReduceImpl<T, Op>(input, output, numElems, stream);
    });
  };
  return detail::DispatchReduceType(dt, go);
}

hipError_t CollectivesFacade::RunAllGather(const void* input, void* output, size_t chunkBytes,
                                           hipStream_t stream) {
  constexpr int kThreads = 64;
  // Single block: SDMA pushes my shard to every peer's output[myPe] slot (+ self).
  AllGatherPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, input, output, chunkBytes,
                                     state_.devComm, state_.heapWin, state_.barrierCtr);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunAllToAll(const AddressVector& addrs, size_t chunkBytes,
                                          hipStream_t stream) {
  assert(static_cast<int>(addrs.size()) == nPes_ && "addrs size must equal npes");
  // Fill the preallocated pinned (host-visible, device-readable) AddressPair
  // buffer with a plain host write, then launch -- no per-call device copy.
  for (int p = 0; p < nPes_; ++p) {
    state_.pinnedPairs[p].source = addrs[p].first;
    state_.pinnedPairs[p].dest = addrs[p].second;
  }
  constexpr int kThreads = 64;
  // Single block: SDMA pushes each send slot p to peer p's recv slot (+ self).
  AllToAllPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, 
                    state_.pinnedPairs, chunkBytes, state_.devComm,
                    state_.heapWin, state_.barrierCtr);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunBarrier(hipStream_t stream) {
  detail::BarrierKernel<<<1, 1, 0, stream>>>();
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunSend(const void* /*sendBuf*/, size_t /*numBytes*/,
                                      int /*peer*/, hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real P2P send push kernel goes here.
  return hipErrorNotSupported;
}

hipError_t CollectivesFacade::RunRecv(void* /*recvBuf*/, size_t /*numBytes*/,
                                      int /*peer*/, hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real P2P recv kernel goes here.
  return hipErrorNotSupported;
}

hipError_t CollectivesFacade::RunCollectivePermute(const void* sendBuf, void* recvBuf,
                                                   size_t numBytes, int srcPe,
                                                   const std::vector<int>& dstPes,
                                                   hipStream_t stream) {
  if (dstPes.size() > 1) {
    FACADE_PRINTF("CollectivePermute: unsupported dstPes.size()=%zu", dstPes.size());
    return hipErrorNotSupported;
  }
  // Empty dstPes: receive-only rank (nobody receives from us).
  const int dstPe = dstPes.empty() ? -1 : dstPes[0];
  if (dstPe < -1 || dstPe >= nPes_ || srcPe < -1 || srcPe >= nPes_) {
    FACADE_PRINTF("CollectivePermute: dstPe=%d srcPe=%d out of range (nPes=%d)", dstPe, srcPe,
                  nPes_);
    return hipErrorInvalidValue;
  }
  if (dstPe < 0 && srcPe < 0) {
    return hipSuccess;
  }
  constexpr int kThreads = 64;
  CollectivePermutePushKernel<<<1, kThreads, 0, stream>>>(
                nPes_, dstPe, srcPe, sendBuf, recvBuf,
                numBytes, state_.devComm, state_.heapWin, state_.permuteReady);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunQuiet(hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real quiet/drain goes here.
  return hipSuccess;
}

hipError_t CollectivesFacade::RunFence() {
  // TODO: dummy placeholder; real device fence goes here.
  return hipSuccess;
}

}  // namespace collective
}  // namespace mori

#else // !__HIP_DEVICE_COMPILE__ || __gfx942__ || __gfx950__ || __gfx1250__

namespace mori {
namespace collective {

hipError_t CollectivesFacade::RunReduceScatter(const void*, void*, size_t, DataType, ReduceOpKind,
                                               hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllReduce(const void*, void*, size_t, DataType, ReduceOpKind,
                                           hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllGather(const void*, void*, size_t, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllToAll(const AddressVector&, size_t, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunBarrier(hipStream_t) { return hipErrorNotSupported; }
hipError_t CollectivesFacade::RunSend(const void*, size_t, int, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunRecv(void*, size_t, int, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunCollectivePermute(const void*, void*, size_t, int,
                                                   const std::vector<int>&, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunQuiet(hipStream_t) { return hipErrorNotSupported; }
hipError_t CollectivesFacade::RunFence() { return hipErrorNotSupported; }

}  // namespace collective
}  // namespace mori

#endif // !__HIP_DEVICE_COMPILE__ || __gfx942__ || __gfx950__ || __gfx1250__
