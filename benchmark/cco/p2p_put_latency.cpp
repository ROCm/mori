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

// CCO p2p put latency — unidirectional, PE 0 → PE 1, one op per iteration.
//
//   -t lsa   : single flat-VA store of the whole buffer + system fence.
//   -t sdma  : single copy-engine put + completion wait (-C quiet | signal).
//   -t ibgda : single RDMA write + flush per iteration.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "util.hpp"

namespace mori::cco::benchmark {

// LSA: one block stores the whole buffer to the peer, fences, each iteration.
__global__ void lsa_put_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                            int peerLsa, int iter) {
  if (blockIdx.x != 0) return;
  const int tid = linear_tid();
  const int lanes = blockDim.x * blockDim.y * blockDim.z;

  double* dst = reinterpret_cast<double*>(ccoGetLsaPeerPtr(recvWin, peerLsa, 0));
  const double* src = reinterpret_cast<const double*>(ccoGetLocalPtr(sendWin, 0));

  for (int i = 0; i < iter; i++) {
    lsa_copy_strided(dst, src, len_doubles, tid, lanes);
    __syncthreads();
    if (tid == 0) __threadfence_system();
    __syncthreads();
  }
}

// IBGDA: one block issues a single RDMA write of the whole buffer + flush per
// iteration (mirrors shmem's lat_block put_nbi + quiet). flush drains the local
// CQ each iteration, so on real hardware the per-op time tracks wire latency.
template <core::ProviderType PrvdType>
__global__ void ibgda_put_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                              size_t len_doubles, ccoDevComm devComm, int iter) {
  if (blockIdx.x != 0) return;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  const int peer = !devComm.rank;
  const size_t bytes = len_doubles * sizeof(double);

  for (int i = 0; i < iter; i++) {
    gda.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), 0, reinterpret_cast<ccoWindow_t>(sendWin),
            0, bytes, ccoGda_NoSignal{}, ccoCoopBlock{});
    gda.flush(ccoCoopWarp{});
  }
}

// SDMA: a single whole-buffer put on queue 0 + quiet per iteration, so the
// per-op time tracks the SDMA dispatch + completion round trip. Coop selects the
// issue granularity only — the copy is never split, so all three scopes move the
// same bytes over the same queue and the delta is pure issue overhead:
//   thread — the calling thread fills the WQE and rings the doorbell.
//   warp / block — the group's leader does it (leader-only single writer).
// SignalComp picks the local-completion mechanism (both are sender-side only):
//   false — quietQueue(): wait for the engine's read pointer to pass our wptr.
//   true  — the put carries a trailing ATOMIC into our own signalBuf slot and we
//           waitSignal() on it. The slot is never reset, so we snapshot it on
//           entry and count up from there.
// Completion waits on queue 0 only. quiet<Coop>() would drain every queue, and
// the extra uncached rptr read on an idle queue costs ~1.3us here — real, but an
// artifact of quiet's all-queue semantics rather than of the issue scope.
template <typename Coop, bool SignalComp>
__global__ void sdma_put_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                             ccoDevComm devComm, int peerLsa, int iter) {
  if (blockIdx.x != 0) return;
  ccoSdma sdma{devComm};
  const size_t bytes = len_doubles * sizeof(double);

  // Local signals land in signalBuf[myLsaRank*n + q] — indexed by us as the
  // sender, not by the peer — and persist across launches.
  uint64_t expected = 0;
  if constexpr (SignalComp) {
    const uint32_t nq = devComm.sdma.sdmaNumQueue;
    expected = devComm.sdma.signalBuf[static_cast<uint32_t>(devComm.lsaRank) * nq];
  }

  for (int i = 0; i < iter; i++) {
    sdma.put<Coop, SignalComp, /*remoteSignal=*/false>(
        peerLsa, reinterpret_cast<ccoWindow_t>(recvWin), 0, reinterpret_cast<ccoWindow_t>(sendWin),
        0, bytes, 0);
    if constexpr (SignalComp) {
      sdma.waitSignal(devComm.lsaRank, 0, ++expected);
      Coop{}.sync();  // pin the iteration boundary for the whole group
    } else {
      // Scope-aware: leader-only drain + sync. Letting every lane poll the
      // uncached rptr costs ~0.4us at block scope and drains no sooner.
      sdma.quietQueue<Coop>(peerLsa, 0);  // syncs internally
    }
  }
}

template <typename Coop>
static void launch_sdma_lat_scope(SdmaComp comp, dim3 block, ccoWindowDevice* sendWin,
                                  ccoWindowDevice* recvWin, size_t len_doubles, ccoDevComm devComm,
                                  int peerLsa, int iter) {
  if (comp == SdmaComp::kSignal) {
    hipLaunchKernelGGL((sdma_put_lat<Coop, true>), dim3(1), block, 0, 0, sendWin, recvWin,
                       len_doubles, devComm, peerLsa, iter);
  } else {
    hipLaunchKernelGGL((sdma_put_lat<Coop, false>), dim3(1), block, 0, 0, sendWin, recvWin,
                       len_doubles, devComm, peerLsa, iter);
  }
}

static void launch_sdma_lat(PutScope scope, SdmaComp comp, ccoWindowDevice* sendWin,
                            ccoWindowDevice* recvWin, size_t len_doubles, ccoDevComm devComm,
                            int peerLsa, int iter, int block_threads) {
  if (scope == PutScope::kWarp) {
    launch_sdma_lat_scope<ccoCoopWarp>(comp, dim3(block_threads), sendWin, recvWin, len_doubles,
                                       devComm, peerLsa, iter);
  } else if (scope == PutScope::kBlock) {
    launch_sdma_lat_scope<ccoCoopBlock>(comp, dim3(block_threads), sendWin, recvWin, len_doubles,
                                        devComm, peerLsa, iter);
  } else {
    // thread / thread_agg: SDMA has no ThreadAggregate mode, so both run one thread.
    launch_sdma_lat_scope<ccoCoopThread>(comp, dim3(1), sendWin, recvWin, len_doubles, devComm,
                                         peerLsa, iter);
  }
}

}  // namespace mori::cco::benchmark

int main(int argc, char** argv) {
  using namespace mori::cco;
  using namespace mori::cco::benchmark;

  PerfContext ctx{};
  const int init_rc = PerfInit(argc, argv, &ctx);
  if (init_rc != 0) {
    return init_rc == 2 ? 0 : 1;
  }

  PerfArgs& args = ctx.args;
  const int my_pe = ctx.my_pe;
  const bool run_kernels = (my_pe == 0);

  // SDMA keeps its historical thread-scope default; -s selects warp/block.
  if (args.transport == Transport::kSdma && !args.put_scope_explicit) {
    args.put_scope = PutScope::kThread;
  }

  const int block_threads =
      LatencyBlockThreads(args.put_scope, args.threads_per_block, ctx.device_warp_size);
  const dim3 grid(1, 1, 1);
  const dim3 block(block_threads, 1, 1);

  PerfRes res;
  if (run_kernels) {
    PerfResAlloc(&res);
  }

  std::vector<PerfTableRow> table;
  if (my_pe == 0) {
    table.reserve(64);
  }

  for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
       size_bytes *= args.step_factor) {
    if (size_bytes % sizeof(double) != 0) continue;
    const size_t len_doubles = size_bytes / sizeof(double);

    if (!latency_size_ok(len_doubles)) {
      if (my_pe == 0) table.push_back(PerfTableRow{size_bytes, true, 0.0});
      ccoBarrierAll(ctx.comm);
      continue;
    }

    if (run_kernels) {
      const float ms = RunWarmupAndTimed(res, args.warmup, args.iters, [&](int count) {
        if (args.transport == Transport::kSdma) {
          launch_sdma_lat(args.put_scope, args.sdma_comp, ctx.send_win, ctx.recv_win, len_doubles,
                          ctx.devComm, ctx.peer_lsa_rank, count, block_threads);
        } else if (args.transport == Transport::kLsa) {
          hipLaunchKernelGGL(lsa_put_lat, grid, block, 0, 0, ctx.send_win, ctx.recv_win,
                             len_doubles, ctx.peer_lsa_rank, count);
        } else {
          CCO_GDA_DISPATCH(hipLaunchKernelGGL((ibgda_put_lat<P>), grid, block, 0, 0, ctx.send_win,
                                              ctx.recv_win, len_doubles, ctx.devComm, count));
        }
        HIP_RUNTIME_CHECK(hipGetLastError());
      });

      const double latency_us = (static_cast<double>(ms) * static_cast<double>(kMsToUs)) /
                                static_cast<double>(args.iters);
      table.push_back(PerfTableRow{size_bytes, false, latency_us});
    }

    ccoBarrierAll(ctx.comm);
  }

  ccoBarrierAll(ctx.comm);
  if (my_pe == 0) {
    // SDMA latency always uses a single queue; thread_agg has no SDMA analogue.
    int print_block = block_threads;
    const char* print_scope = ScopeToChar(args.put_scope);
    if (args.transport == Transport::kSdma &&
        (args.put_scope == PutScope::kThread || args.put_scope == PutScope::kThreadAgg)) {
      print_block = 1;
      print_scope = "thread";
    }
    if (args.transport == Transport::kSdma) {
      std::printf("# sdma completion = %s\n", SdmaCompToChar(args.sdma_comp));
    }
    PrintPerfTable("p2p_put_latency unidirection", TransportToChar(args.transport), print_scope, 1,
                   print_block, ctx.device_warp_size, args.iters, args.warmup,
                   PerfTableMetric::kLatencyUs, table);
  }

  if (run_kernels) {
    PerfResFree(&res);
  }
  PerfFinalize(&ctx);
  return 0;
}
