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

// CCO p2p get bandwidth — PE 0 pulls from PE 1's send window into its recv
// window.
//
//   -T lsa   : intra-node flat-VA load loop (read peer slot → local).
//   -T igbda : cross-node one-sided RDMA read via ccoGda<PrvdType>.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "util.hpp"

namespace mori::cco::benchmark {

// LSA: each block reads its chunk from the peer's send window into the local
// recv window.
__global__ void lsa_get_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                           int peerLsa, int iter, int lanes) {
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int tid = linear_tid();
  if (tid >= lanes) return;

  const size_t chunk = len_doubles / static_cast<size_t>(nblocks);
  const size_t off_bytes = static_cast<size_t>(bid) * chunk * sizeof(double);

  const double* src =
      reinterpret_cast<const double*>(ccoGetLsaPeerPtr(sendWin, peerLsa, off_bytes));
  double* dst = reinterpret_cast<double*>(ccoGetLocalPtr(recvWin, off_bytes));

  for (int i = 0; i < iter; i++) {
    lsa_copy_strided(dst, src, chunk, tid, lanes);
    // Per-iteration system fence — see p2p_put_bw for the cache-absorption
    // rationale (repeated same-region traffic otherwise measures cache BW).
    __threadfence_system();
  }
}

// IGBDA: one RDMA read per coop unit of its chunk.
template <core::ProviderType PrvdType, typename Coop>
__global__ void igbda_get_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
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

  // RDMA read completions are self-confirming (data has landed locally when the
  // CQE is reaped). Pipeline the reads, one end flush drains the whole batch.
  for (int i = 0; i < iter; i++) {
    gda.get(peer, reinterpret_cast<ccoWindow_t>(sendWin), off_bytes,
            reinterpret_cast<ccoWindow_t>(recvWin), off_bytes, bytes, coop);
  }
  gda.flush(ccoCoopBlock{});
}

static void launch_lsa(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                       ccoWindow_t recvWin, size_t len_doubles, int peerLsa, int count,
                       int warp_size) {
  int lanes = block.x;
  if (scope == PutScope::kWarp) lanes = warp_size;
  if (scope == PutScope::kThread) lanes = 1;
  hipLaunchKernelGGL(lsa_get_bw, grid, block, 0, 0, sendWin, recvWin, len_doubles, peerLsa, count,
                     lanes);
}

template <core::ProviderType PrvdType>
static void launch_igbda(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                         ccoWindow_t recvWin, size_t len_doubles, ccoDevComm devComm, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL((igbda_get_bw<PrvdType, ccoCoopBlock>), grid, block, 0, 0, sendWin,
                         recvWin, len_doubles, devComm, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL((igbda_get_bw<PrvdType, ccoCoopWarp>), grid, block, 0, 0, sendWin, recvWin,
                         len_doubles, devComm, count);
      break;
    case PutScope::kThread:
      hipLaunchKernelGGL((igbda_get_bw<PrvdType, ccoCoopThread>), grid, block, 0, 0, sendWin,
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
  const bool run_kernels = (my_pe == 0);  // unidirectional: PE 0 pulls

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
          launch_lsa(args.put_scope, grid, block, ctx.send_win, ctx.recv_win, len_doubles,
                     ctx.peer_lsa_rank, count, ctx.device_warp_size);
        } else {
          CCO_GDA_DISPATCH(launch_igbda<P>(args.put_scope, grid, block, ctx.send_win, ctx.recv_win,
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
    PrintPerfTable("p2p_get_bw unidirection", TransportToChar(args.transport),
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
