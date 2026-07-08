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

// CCO p2p get latency — PE 0 pulls from PE 1, one op per iteration.
//
//   -T lsa   : single flat-VA load of the whole buffer per iteration.
//   -T ibgda : single RDMA read + flush per iteration.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "util.hpp"

namespace mori::cco::benchmark {

// LSA: one block reads the whole buffer from the peer's send window.
__global__ void lsa_get_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                            int peerLsa, int iter) {
  if (blockIdx.x != 0) return;
  const int tid = linear_tid();
  const int lanes = blockDim.x * blockDim.y * blockDim.z;

  const double* src = reinterpret_cast<const double*>(ccoGetLsaPeerPtr(sendWin, peerLsa, 0));
  double* dst = reinterpret_cast<double*>(ccoGetLocalPtr(recvWin, 0));

  for (int i = 0; i < iter; i++) {
    lsa_copy_strided(dst, src, len_doubles, tid, lanes);
    __syncthreads();
  }
}

// IBGDA: one block issues a single RDMA read of the whole buffer + flush per
// iteration.
template <core::ProviderType PrvdType>
__global__ void ibgda_get_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                              size_t len_doubles, ccoDevComm devComm, int iter) {
  if (blockIdx.x != 0) return;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  const int peer = !devComm.rank;
  const size_t bytes = len_doubles * sizeof(double);

  for (int i = 0; i < iter; i++) {
    gda.get(peer, reinterpret_cast<ccoWindow_t>(sendWin), 0, reinterpret_cast<ccoWindow_t>(recvWin),
            0, bytes, ccoCoopBlock{});
    gda.flush(ccoCoopWarp{});
  }
}

// SDMA: one thread issues a single whole-buffer get on queue 0 + quiet per
// iteration, so the per-op time tracks the SDMA dispatch + completion round trip.
__global__ void sdma_get_lat(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                             ccoDevComm devComm, int peerLsa, int iter) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  ccoSdma sdma{devComm};
  const size_t bytes = len_doubles * sizeof(double);
  for (int i = 0; i < iter; i++) {
    sdma.get(peerLsa, reinterpret_cast<ccoWindow_t>(recvWin), 0,
             reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes, 0);
    sdma.quietQueue(peerLsa, 0);
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
        if (args.transport == Transport::kSdma) {
          hipLaunchKernelGGL(sdma_get_lat, dim3(1), dim3(1), 0, 0, ctx.send_win, ctx.recv_win,
                             len_doubles, ctx.devComm, ctx.peer_lsa_rank, count);
        } else if (args.transport == Transport::kLsa) {
          hipLaunchKernelGGL(lsa_get_lat, grid, block, 0, 0, ctx.send_win, ctx.recv_win,
                             len_doubles, ctx.peer_lsa_rank, count);
        } else {
          CCO_GDA_DISPATCH(hipLaunchKernelGGL((ibgda_get_lat<P>), grid, block, 0, 0, ctx.send_win,
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
    // SDMA latency uses a single queue / single thread.
    int print_block = block_threads;
    const char* print_scope = ScopeToChar(args.put_scope);
    if (args.transport == Transport::kSdma) {
      print_block = 1;
      print_scope = "thread";
    }
    PrintPerfTable("p2p_get_latency unidirection", TransportToChar(args.transport), print_scope, 1,
                   print_block, ctx.device_warp_size, args.iters, args.warmup,
                   PerfTableMetric::kLatencyUs, table);
  }

  if (run_kernels) {
    PerfResFree(&res);
  }
  PerfFinalize(&ctx);
  return 0;
}
