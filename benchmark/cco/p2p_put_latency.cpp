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
//   -T lsa   : single flat-VA store of the whole buffer + system fence.
//   -T ibgda : single RDMA write + flush per iteration.

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
        if (args.transport == Transport::kLsa) {
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
    PrintPerfTable("p2p_put_latency unidirection", TransportToChar(args.transport),
                   ScopeToChar(args.put_scope), 1, block_threads, ctx.device_warp_size, args.iters,
                   args.warmup, PerfTableMetric::kLatencyUs, table);
  }

  if (run_kernels) {
    PerfResFree(&res);
  }
  PerfFinalize(&ctx);
  return 0;
}
