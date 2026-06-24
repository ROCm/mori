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

// CCO p2p put bandwidth — unidirectional, PE 0 → PE 1.
//
//   -T lsa   : intra-node flat-VA store loop (no NIC).
//   -T ibgda : cross-node one-sided RDMA write via ccoGda<PrvdType>.
//
// The buffer is split into `nblocks` chunks (one per block); scope controls
// the per-chunk cooperation granularity (block/warp/thread).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "util.hpp"

namespace mori::cco::benchmark {

// LSA: flat-VA store loop. scope_size = copy cooperation granularity (block /
// warp / single thread); all block threads always participate (matches shmem).
__global__ void lsa_put_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                           volatile unsigned int* counter_d, size_t len_doubles, int peerLsa,
                           int iter, int scope_size) {
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int tid = linear_tid();
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;

  const size_t chunk = len_doubles / static_cast<size_t>(nblocks);
  const int nunits = nthreads / scope_size;
  const int unit = tid / scope_size;
  const int lane = tid % scope_size;
  const size_t per_unit = chunk / static_cast<size_t>(nunits);

  const size_t off_bytes =
      (static_cast<size_t>(bid) * chunk + static_cast<size_t>(unit) * per_unit) * sizeof(double);
  double* dst = reinterpret_cast<double*>(ccoGetLsaPeerPtr(recvWin, peerLsa, off_bytes));
  const double* src = reinterpret_cast<const double*>(ccoGetLocalPtr(sendWin, off_bytes));

  for (int i = 0; i < iter; i++) {
    lsa_copy_strided(dst, src, per_unit, lane, scope_size);
    // System fence forces each round's cross-GPU stores out — otherwise the
    // repeated same-region traffic is cache-absorbed and we'd time cache BW.
    __threadfence_system();
    // Per-round all-block barrier mirrors shmem so the timed window matches.
    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }
}

// IBGDA: one QP per block (ginContext=blockIdx); each block pipelines its chunk
// then flushes its own QP. block scope = one bulk write (== shmem
// ShmemPutMemNbiBlock); warp/thread subdivide.
template <core::ProviderType PrvdType, typename Coop>
__global__ void ibgda_put_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                             ccoDevComm devComm, int iter) {
  Coop coop;
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/bid};  // one QP context per block
  const int peer = !devComm.rank;
  const size_t chunk = len_doubles / static_cast<size_t>(nblocks);

  const int tid = linear_tid();
  const int unit = tid / coop.size();
  const int nunits = (blockDim.x * blockDim.y * blockDim.z) / coop.size();
  const size_t per_unit = chunk / static_cast<size_t>(nunits);
  const size_t base = static_cast<size_t>(bid) * chunk + static_cast<size_t>(unit) * per_unit;
  const size_t off_bytes = base * sizeof(double);
  const size_t bytes = per_unit * sizeof(double);

  // Per-op doorbell: each put rings its own (grouped-per-peer) doorbell, so the
  // SQ-space flow control inside put drains completions (quietUntil) as the queue
  // fills. The trailing flush waits for the last ops to complete before timing.
  for (int i = 0; i < iter; i++) {
    gda.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), off_bytes,
            reinterpret_cast<ccoWindow_t>(sendWin), off_bytes, bytes, ccoGda_NoSignal{}, coop);
  }
  gda.flush(ccoCoopBlock{});
}

static void launch_lsa(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                       ccoWindow_t recvWin, unsigned int* counter_d, size_t len_doubles,
                       int peerLsa, int count, int warp_size) {
  // scope_size = cooperation granularity of the copy (block: whole block,
  // warp: one wavefront, thread: a single thread). All threads always run.
  int scope_size = block.x;
  if (scope == PutScope::kWarp) scope_size = warp_size;
  if (scope == PutScope::kThread) scope_size = 1;
  hipLaunchKernelGGL(lsa_put_bw, grid, block, 0, 0, sendWin, recvWin, counter_d, len_doubles,
                     peerLsa, count, scope_size);
}

template <core::ProviderType PrvdType>
static void launch_ibgda(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                         ccoWindow_t recvWin, size_t len_doubles, ccoDevComm devComm, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL((ibgda_put_bw<PrvdType, ccoCoopBlock>), grid, block, 0, 0, sendWin,
                         recvWin, len_doubles, devComm, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL((ibgda_put_bw<PrvdType, ccoCoopWarp>), grid, block, 0, 0, sendWin, recvWin,
                         len_doubles, devComm, count);
      break;
    case PutScope::kThread:
      hipLaunchKernelGGL((ibgda_put_bw<PrvdType, ccoCoopThread>), grid, block, 0, 0, sendWin,
                         recvWin, len_doubles, devComm, count);
      break;
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
  const bool run_kernels = (my_pe == 0);  // unidirectional: PE 0 issues

  const dim3 grid(args.nblocks, 1, 1);
  const dim3 block(args.threads_per_block, 1, 1);

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

    if (!size_ok(args.put_scope, size_bytes, args.nblocks, args.threads_per_block,
                 ctx.device_warp_size)) {
      if (my_pe == 0) table.push_back(PerfTableRow{size_bytes, true, 0.0});
      ccoBarrierAll(ctx.comm);
      continue;
    }

    if (run_kernels) {
      const float ms = RunWarmupAndTimed(res, args.warmup, args.iters, [&](int count) {
        if (args.transport == Transport::kLsa) {
          launch_lsa(args.put_scope, grid, block, ctx.send_win, ctx.recv_win, res.counter_d,
                     len_doubles, ctx.peer_lsa_rank, count, ctx.device_warp_size);
        } else {
          CCO_GDA_DISPATCH(launch_ibgda<P>(args.put_scope, grid, block, ctx.send_win, ctx.recv_win,
                                           len_doubles, ctx.devComm, count));
        }
        HIP_RUNTIME_CHECK(hipGetLastError());
      });

      const double gbps = static_cast<double>(size_bytes) /
                          (static_cast<double>(ms) * (kBToGb / (args.iters * kMsToS)));
      table.push_back(PerfTableRow{size_bytes, false, gbps});
    }

    ccoBarrierAll(ctx.comm);
  }

  ccoBarrierAll(ctx.comm);
  if (my_pe == 0) {
    PrintPerfTable("p2p_put_bw unidirection", TransportToChar(args.transport),
                   ScopeToChar(args.put_scope), args.nblocks, args.threads_per_block,
                   ctx.device_warp_size, args.iters, args.warmup, PerfTableMetric::kBandwidthGbps,
                   table);
  }

  if (run_kernels) {
    PerfResFree(&res);
  }
  PerfFinalize(&ctx);
  return 0;
}
