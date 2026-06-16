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
//   -T igbda : cross-node one-sided RDMA write via ccoGda<PrvdType>.
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

// ── LSA: flat-VA store loop. Each block copies its chunk into the peer's recv
// window. `lanes`/`lane0` select how many threads of the block participate
// (block: all, warp: first warp, thread: thread 0). ──
__global__ void lsa_put_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                           int peerLsa, int iter, int lanes) {
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int tid = linear_tid();
  if (tid >= lanes) return;

  const size_t chunk = len_doubles / static_cast<size_t>(nblocks);
  const size_t off_bytes = static_cast<size_t>(bid) * chunk * sizeof(double);

  double* dst = reinterpret_cast<double*>(ccoGetLsaPeerPtr(recvWin, peerLsa, off_bytes));
  const double* src = reinterpret_cast<const double*>(ccoGetLocalPtr(sendWin, off_bytes));

  for (int i = 0; i < iter; i++) {
    lsa_copy_strided(dst, src, chunk, tid, lanes);
    // Per-iteration system fence: every iteration writes the SAME src->dst, so
    // without forcing each round's stores out to the peer they get absorbed by
    // L2/Infinity cache and the timer measures cache bandwidth, not real P2P
    // (symptom: mid-size BW spikes far above the NIC/xGMI limit, then drops to
    // the true rate once the buffer exceeds cache). The fence makes each round's
    // writes visible to the peer before the next round reuses the same region.
    __threadfence_system();
  }
}

// ── IGBDA: perftest ib_write_bw model. Each block owns a QP context and
// pipelines `iter` RDMA writes of its chunk back-to-back (no per-op flush), then
// the final write carries a SignalInc — an RDMA atomic whose completion only
// fires after the remote read-modify-write round-trip. RC same-QP ordering means
// that completion also guarantees every preceding write landed remotely. One
// end flush waits the whole pipeline. This is what makes the host timer span
// real wire throughput; a write-only flush returns on the local send-CQE before
// data lands (intra-node loopback), inflating bandwidth to bogus TB/s. ──
template <core::ProviderType PrvdType, typename Coop>
__global__ void igbda_put_bw(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t len_doubles,
                             ccoDevComm devComm, int iter) {
  Coop coop;
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/bid};  // one QP context per block
  const int peer = !devComm.rank;
  const size_t chunk = len_doubles / static_cast<size_t>(nblocks);

  // Sub-divide the block's chunk across coop units (warps for warp scope, the
  // whole block for block scope, the lane for thread scope).
  const int tid = linear_tid();
  const int unit = tid / coop.size();
  const int nunits = (blockDim.x * blockDim.y * blockDim.z) / coop.size();
  const size_t per_unit = chunk / static_cast<size_t>(nunits);
  const size_t base = static_cast<size_t>(bid) * chunk + static_cast<size_t>(unit) * per_unit;
  const size_t off_bytes = base * sizeof(double);
  const size_t bytes = per_unit * sizeof(double);

  const auto sig = ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(bid)};
  for (int i = 0; i < iter; i++) {
    if (i == iter - 1) {
      // Last op carries the signal; its atomic round-trip confirms landing.
      gda.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), off_bytes,
              reinterpret_cast<ccoWindow_t>(sendWin), off_bytes, bytes, sig, coop);
    } else {
      gda.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), off_bytes,
              reinterpret_cast<ccoWindow_t>(sendWin), off_bytes, bytes, ccoGda_NoSignal{}, coop);
    }
  }
  gda.flush(ccoCoopBlock{});
}

static void launch_lsa(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                       ccoWindow_t recvWin, size_t len_doubles, int peerLsa, int count,
                       int warp_size) {
  int lanes = block.x;
  if (scope == PutScope::kWarp) lanes = warp_size;
  if (scope == PutScope::kThread) lanes = 1;
  hipLaunchKernelGGL(lsa_put_bw, grid, block, 0, 0, sendWin, recvWin, len_doubles, peerLsa, count,
                     lanes);
}

template <core::ProviderType PrvdType>
static void launch_igbda(PutScope scope, dim3 grid, dim3 block, ccoWindow_t sendWin,
                         ccoWindow_t recvWin, size_t len_doubles, ccoDevComm devComm, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL((igbda_put_bw<PrvdType, ccoCoopBlock>), grid, block, 0, 0, sendWin,
                         recvWin, len_doubles, devComm, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL((igbda_put_bw<PrvdType, ccoCoopWarp>), grid, block, 0, 0, sendWin, recvWin,
                         len_doubles, devComm, count);
      break;
    case PutScope::kThread:
      hipLaunchKernelGGL((igbda_put_bw<PrvdType, ccoCoopThread>), grid, block, 0, 0, sendWin,
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
