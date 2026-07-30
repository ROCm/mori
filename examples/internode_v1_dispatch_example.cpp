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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// InterNodeV1 dispatch/combine example — 2 nodes, 1 GPU per node, RDMA path.
//
// Topology:
//   npes=2, gpuPerNode=1, nNodes=2, numQpPerPe=1
//   numExpertPerRank=1, numExpertPerToken=1 (top-1)
//   2 experts total: expert 0 → PE0(Node0) | expert 1 → PE1(Node1)
//   numTokens=1, hiddenDim=4
//
// Token routing:
//   Both ranks send their single token to the remote node's expert.
//   PE0: token0 → expert1 (Node1)
//   PE1: token0 → expert0 (Node0)
//   → 完全跨节点，所有 token 走 RDMA 路径
//
// 本用例通过公共入口 LaunchDispatch / LaunchCombine 驱动，不手工逐个拉起
// sub-kernel：grid/block/dynamic-shared-mem 的取值以及 sub-kernel 的先后顺序
// 都由 launch.cpp 统一负责，手写容易与库不一致（例如 dispatch 阶段需要
// dispatch_shared_mem() 字节的动态共享内存，漏传会直接踩内存）。
//
// LaunchDispatch(InterNodeV1) 内部依次拉起:
//   1. EpDispatchCopyToStaging      — 把 token 打包进 dispatchStaging
//   2. EpDispatchInterNodeV1Kernel  — RDMA send + recv + sync
// LaunchCombine(InterNodeV1) 内部依次拉起:
//   1. EpCombineSync                — FFN 输出拷进 combineInp，权重拷进 shmemInpWeights
//   2. EpCombineSyncBarrier         — 跨节点 barrier，等 combineInp 可见
//   3. EpCombineInterNodeV1Kernel   — WarpAccum + RDMA 回送 / 本地 XGMI 聚合 → staging
//   4. EpCombineAll                 — 从各节点 staging 读回做最终加权求和 → combineOut
//
// Launch:
//   mpirun -np 2 ./internode_v1_dispatch_example
//

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::moe;
using namespace mori::shmem;

#define HIP_CHECK(cmd)                                                                      \
  do {                                                                                      \
    hipError_t e = (cmd);                                                                   \
    if (e != hipSuccess) {                                                                  \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

static void* gpu_alloc_zero(size_t bytes) {
  void* p = nullptr;
  HIP_CHECK(hipMalloc(&p, bytes));
  HIP_CHECK(hipMemset(p, 0, bytes));
  return p;
}

// 只打印前 kDumpWidth 个元素，hiddenDim 动辄几百上千，全打会把日志冲掉。
static constexpr int kDumpWidth = 8;

static void dump_bf16(int rank, const char* tag, void* d_ptr, int ntok, int dim) {
  std::vector<__hip_bfloat16> h(static_cast<size_t>(ntok) * dim);
  HIP_CHECK(hipMemcpy(h.data(), d_ptr, h.size() * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
  int shown = std::min(dim, kDumpWidth);
  printf("[PE%d]   %-34s (%d tok x dim=%d):\n", rank, tag, ntok, dim);
  for (int t = 0; t < ntok; ++t) {
    printf("[PE%d]     tok%d: [", rank, t);
    for (int d = 0; d < shown; ++d)
      printf("%s%.1f", d ? "," : "", __bfloat162float(h[static_cast<size_t>(t) * dim + d]));
    printf("%s]\n", (shown < dim) ? ",..." : "");
  }
}

static void dump_f32(int rank, const char* tag, void* d_ptr, int ntok, int slots) {
  std::vector<float> h(static_cast<size_t>(ntok) * slots);
  HIP_CHECK(hipMemcpy(h.data(), d_ptr, h.size() * sizeof(float), hipMemcpyDeviceToHost));
  printf("[PE%d]   %-34s (%d tok x %d slots):\n", rank, tag, ntok, slots);
  for (int t = 0; t < ntok; ++t) {
    printf("[PE%d]     tok%d: [", rank, t);
    for (int s = 0; s < slots; ++s)
      printf("%s%.2f", s ? "," : "", h[static_cast<size_t>(t) * slots + s]);
    printf("]\n");
  }
}

// ---------------------------------------------------------------------------
// run_internode_v1
// ---------------------------------------------------------------------------
static bool run_internode_v1(int rank, int world, hipStream_t stream) {
  printf("\n[PE%d] ===== InterNodeV1 dispatch/combine =====\n", rank);

  const int numTokens = 1;
  const int hiddenDim = 512;       // bf16 → hiddenBytes = 512 * 2 = 1024 B
  const int numEpt = 1;            // top-1
  const int numExpertPerRank = 1;  // 1 expert per GPU
  // expert 0 → PE0, expert 1 → PE1
  // PE0's token routes to expert 1 (remote), PE1's token routes to expert 0 (remote)
  const int destExpert = (rank == 0) ? 1 : 0;

  // ── input token: PE0 fills [10,10,10,10], PE1 fills [20,20,20,20] ────────
  const size_t hBytes = numTokens * hiddenDim * sizeof(__hip_bfloat16);
  std::vector<__hip_bfloat16> h_input(numTokens * hiddenDim);
  float val = (rank == 0) ? 10.0f : 20.0f;
  for (int d = 0; d < hiddenDim; ++d) h_input[d] = __float2bfloat16(val);

  void* d_input = gpu_alloc_zero(hBytes);
  HIP_CHECK(hipMemcpy(d_input, h_input.data(), hBytes, hipMemcpyHostToDevice));

  // indices: single expert choice per token
  int32_t h_idx[1] = {destExpert};
  void* d_indices = gpu_alloc_zero(numTokens * numEpt * sizeof(int32_t));
  HIP_CHECK(hipMemcpy(d_indices, h_idx, sizeof(h_idx), hipMemcpyHostToDevice));

  // weights: top-1, weight = 1.0
  float h_wgt[1] = {1.0f};
  void* d_weights = gpu_alloc_zero(numTokens * numEpt * sizeof(float));
  HIP_CHECK(hipMemcpy(d_weights, h_wgt, sizeof(h_wgt), hipMemcpyHostToDevice));

  printf("[PE%d] input token: [%.0f,%.0f,%.0f,%.0f] → expert%d (PE%d)\n", rank, val, val, val, val,
         destExpert, destExpert / numExpertPerRank);

  // ── config ────────────────────────────────────────────────────────────────
  EpDispatchCombineConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = world;  // 2
  cfg.hiddenDim = hiddenDim;
  cfg.numExpertPerRank = numExpertPerRank;  // 1
  cfg.numExpertPerToken = numEpt;           // 1
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe = 1;
  cfg.gpuPerNode = 1;  // 1 GPU per node → nNodes = world / gpuPerNode = 2
  cfg.kernelType = KernelType::InterNodeV1;
  cfg.warpNumPerBlock = 4;
  cfg.useExternalInpBuffer = true;
  cfg.quantType = QuantType::None;
  cfg.enableSdma = false;

  EpDispatchCombineHandle handle(cfg);

  // rdmaBlockNum: 前 rdmaBlockNum 个 block 跑 RDMA send/recv，其余跑 intra-node
  // XGMI，因此必须满足 blockNum > rdmaBlockNum。
  handle.config.rdmaBlockNum = std::max(1, handle.multiProcessorCount / 4);
  handle.config.blockNum = handle.config.rdmaBlockNum * 2;
  printf("[PE%d] CUs=%d  blockNum=%d  rdmaBlockNum=%d  warpNumPerBlock=%d\n", rank,
         handle.multiProcessorCount, handle.config.blockNum, handle.config.rdmaBlockNum,
         cfg.warpNumPerBlock);

  // ── dispatch ─────────────────────────────────────────────────────────────
  LaunchDispatch(handle, d_input, d_weights, /*scales=*/nullptr, d_indices, numTokens, HIP_R_16BF,
                 /*block_num=*/-1, /*rdma_block_num=*/-1, /*warp_per_block=*/-1, stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // ── verify dispatchOut ────────────────────────────────────────────────────
  index_t h_total = 0;
  HIP_CHECK(hipMemcpy(&h_total, handle.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));
  void* d_disp_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
  printf("[PE%d][Dispatch] received %lld token(s) from remote:\n", rank, (long long)h_total);
  if (h_total > 0) dump_bf16(rank, "dispatchOut", d_disp_out, (int)h_total, hiddenDim);

  // ── simulate FFN: multiply by 2.0 ────────────────────────────────────────
  std::vector<__hip_bfloat16> h_disp_out(static_cast<size_t>(h_total) * hiddenDim);
  HIP_CHECK(hipMemcpy(h_disp_out.data(), d_disp_out, h_disp_out.size() * sizeof(__hip_bfloat16),
                      hipMemcpyDeviceToHost));
  std::vector<__hip_bfloat16> h_ffn_out(h_disp_out.size());
  for (size_t i = 0; i < h_disp_out.size(); ++i)
    h_ffn_out[i] = __float2bfloat16(__bfloat162float(h_disp_out[i]) * 2.0f);

  void* d_ffn_out = gpu_alloc_zero(std::max<size_t>(h_ffn_out.size(), 1) * sizeof(__hip_bfloat16));
  HIP_CHECK(hipMemcpy(d_ffn_out, h_ffn_out.data(), h_ffn_out.size() * sizeof(__hip_bfloat16),
                      hipMemcpyHostToDevice));

  // ── combine ───────────────────────────────────────────────────────────────
  // Combine weights come from the forwarded-during-dispatch buffer, indexed by
  // received token order — NOT by the local token order.
  float* d_recv_weights =
      reinterpret_cast<float*>(handle.shmemDispatchOutWeightsMemObj.cpu->localPtr);

  printf("\n[PE%d] ──── COMBINE 前状态 ────────────────────────────────────────────\n", rank);
  if (h_total > 0) {
    dump_bf16(rank, "d_ffn_out (expert out, recv-order)", d_ffn_out, (int)h_total, hiddenDim);
    dump_f32(rank, "d_recv_weights", d_recv_weights, (int)h_total, numEpt);
  }

  LaunchCombine(handle, d_ffn_out, d_recv_weights, d_indices, numTokens, HIP_R_16BF,
                /*block_num=*/-1, /*rdma_block_num=*/-1, /*warp_per_block=*/-1,
                /*use_external_inp_buf=*/-1, stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // ── verify combineOut ─────────────────────────────────────────────────────
  // Expected: FFN output * weight = (input * 2.0) * 1.0
  // PE0 sent 10s → PE1 ran FFN → 20s → sent back → combineOut = 20s
  // PE1 sent 20s → PE0 ran FFN → 40s → sent back → combineOut = 40s
  void* d_comb_out = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(numTokens * hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb_out, numTokens * hBytes, hipMemcpyDeviceToHost));

  float expected = val * 2.0f;  // input * FFN-scale
  printf("\n[PE%d][Combine] combineOut (%d token):\n", rank, numTokens);
  bool ok = true;
  // 校验覆盖全部 hiddenDim，但只打印前 kDumpWidth 个。
  for (int t = 0; t < numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int d = 0; d < hiddenDim; ++d) {
      float got = __bfloat162float(h_comb[t * hiddenDim + d]);
      if (d < kDumpWidth) printf("%s%.1f", d ? "," : "", got);
      if (std::abs(got - expected) > 1.0f) ok = false;
    }
    printf("%s]  (expected %.1f for all %d dims)\n", (hiddenDim > kDumpWidth) ? ",..." : "",
           expected, hiddenDim);
  }
  printf("[PE%d] result: %s\n", rank, ok ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_indices));
  HIP_CHECK(hipFree(d_weights));
  HIP_CHECK(hipFree(d_ffn_out));
  return ok;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  HIP_CHECK(hipSetDevice(mpi_rank));

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  const int rank = ShmemMyPe();
  const int world = ShmemNPes();
  assert(world == 2 && "launch with: mpirun -np 2 ./internode_v1_dispatch_example");

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  bool ok = run_internode_v1(rank, world, stream);
  MPI_Barrier(MPI_COMM_WORLD);

  HIP_CHECK(hipStreamDestroy(stream));
  // ShmemFinalize() 里的 MPI bootstrap 会调用 MPI_Finalize，这里不能再调一次，
  // 否则会触发 "MPI_Finalize() called after MPI_FINALIZE was invoked" 而 abort。
  ShmemFinalize();
  return ok ? 0 : 1;
}
