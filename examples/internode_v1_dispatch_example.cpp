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
//   numTokens=512, hiddenDim=512
//
// Token routing:
//   两个 rank 各发 512 个 token，全部投给对端节点的 expert。
//   PE0: token0..511 → expert1 (Node1)
//   PE1: token0..511 → expert0 (Node0)
//   → 完全跨节点，所有 token 走 RDMA 路径
//
//   chunk 的粒度是 warpSize(64) 个 token，512 token = 正好 8 个满 chunk，
//   因此这个用例会完整走一遍 chunk 的 "凑齐 numRecvBlock*warpNum 个 warp
//   才回传" 那条路径，且覆盖 k=0..7 全部 8 个 chunk 索引。
//
//   每个 token 填一个互不相同的值（PE0 落在 [1,16)，PE1 落在 [32,512)），
//   而不是整批同值：同值的话 token 丢失、错位、被别的 token 覆盖，结果看
//   上去都一样。取值方案见 token_value 处的说明。
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
// 同理，token 数上到 64 之后逐个打也会刷屏，只打印前几个做眼检，
// 正确性由后面覆盖全部 token 的校验循环负责。
static constexpr int kDumpTokens = 4;

static void dump_bf16(int rank, const char* tag, void* d_ptr, int ntok, int dim) {
  std::vector<__hip_bfloat16> h(static_cast<size_t>(ntok) * dim);
  HIP_CHECK(hipMemcpy(h.data(), d_ptr, h.size() * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
  int shown = std::min(dim, kDumpWidth);
  int shownTok = std::min(ntok, kDumpTokens);
  printf("[PE%d]   %-34s (%d tok x dim=%d):\n", rank, tag, ntok, dim);
  for (int t = 0; t < shownTok; ++t) {
    printf("[PE%d]     tok%d: [", rank, t);
    for (int d = 0; d < shown; ++d)
      printf("%s%g", d ? "," : "", __bfloat162float(h[static_cast<size_t>(t) * dim + d]));
    printf("%s]\n", (shown < dim) ? ",..." : "");
  }
  if (shownTok < ntok) printf("[PE%d]     ... (%d more token(s))\n", rank, ntok - shownTok);
}

static void dump_f32(int rank, const char* tag, void* d_ptr, int ntok, int slots) {
  std::vector<float> h(static_cast<size_t>(ntok) * slots);
  HIP_CHECK(hipMemcpy(h.data(), d_ptr, h.size() * sizeof(float), hipMemcpyDeviceToHost));
  int shownTok = std::min(ntok, kDumpTokens);
  printf("[PE%d]   %-34s (%d tok x %d slots):\n", rank, tag, ntok, slots);
  for (int t = 0; t < shownTok; ++t) {
    printf("[PE%d]     tok%d: [", rank, t);
    for (int s = 0; s < slots; ++s)
      printf("%s%.2f", s ? "," : "", h[static_cast<size_t>(t) * slots + s]);
    printf("]\n");
  }
  if (shownTok < ntok) printf("[PE%d]     ... (%d more token(s))\n", rank, ntok - shownTok);
}

// ---------------------------------------------------------------------------
// run_internode_v1
// ---------------------------------------------------------------------------
static bool run_internode_v1(int rank, int world, hipStream_t stream) {
  printf("\n[PE%d] ===== InterNodeV1 dispatch/combine =====\n", rank);

  const int numTokens = 512;       // 每个 PE 发 512 个 token = 8 个 chunk (chunk = warpSize)
  const int hiddenDim = 512;       // bf16 → hiddenBytes = 512 * 2 = 1024 B
  const int numEpt = 1;            // top-1
  const int numExpertPerRank = 1;  // 1 expert per GPU
  // expert 0 → PE0, expert 1 → PE1
  // PE0's tokens route to expert 1 (remote), PE1's tokens route to expert 0 (remote)
  const int destExpert = (rank == 0) ? 1 : 0;

  // ── input tokens ─────────────────────────────────────────────────────────
  // 每个 token 一个互不相同的值，而不是整批填同一个数：否则 token 被丢弃、
  // 被别的 token 覆盖、或者顺序错位，结果看上去都一样，校验不出来。
  //
  // 取值必须在 bf16 下精确，否则只能用容差比较，而容差会把"收到隔壁 token"
  // 这类差一档的错误放过去。bf16 只有 8 位有效位（1 隐含 + 7 显式），因此
  // 连续整数只在 [1,256] 内精确，512 个 token 用不了 "基址 + t"。
  //
  // 改为按 binade 取值：每个 binade [2^b, 2^(b+1)) 内 bf16 恰好有 128 个等距
  // 可精确表示的值 2^b * (1 + j/128), j=0..127。512 = 128 * 4，故各用 4 个 binade：
  //   PE0: b = 0 + t/128  → 输入 [1, 16)     ×2 后 [2, 32)
  //   PE1: b = 5 + t/128  → 输入 [32, 512)   ×2 后 [64, 1024)
  // 乘 2 只改指数不动尾数，故 ×2 后仍精确。PE1 基址取 5 而非 4，是为了让
  // PE0 的 ×2 区间 [2,32) 与 PE1 的输入区间 [32,512) 也不相交 —— 否则一个
  // 迷路的 PE1 原始 token 会伪装成合法的 PE0 结果。
  // 另外所有取值都 >= 1，而被丢弃的 token 读出来是 0，因此丢失总能被发现。
  auto token_value = [](int r, int t) {
    int j = t % 128;  // binade 内的第 j 档，j/128 = j * 2^-7，7 bit 尾数正好装下
    int b = t / 128 + (r ? 5 : 0);
    return (1.0f + j / 128.0f) * static_cast<float>(1 << b);
  };

  const size_t hBytes = static_cast<size_t>(numTokens) * hiddenDim * sizeof(__hip_bfloat16);
  std::vector<__hip_bfloat16> h_input(static_cast<size_t>(numTokens) * hiddenDim);
  for (int t = 0; t < numTokens; ++t) {
    __hip_bfloat16 v = __float2bfloat16(token_value(rank, t));
    for (int d = 0; d < hiddenDim; ++d) h_input[static_cast<size_t>(t) * hiddenDim + d] = v;
  }

  void* d_input = gpu_alloc_zero(hBytes);
  HIP_CHECK(hipMemcpy(d_input, h_input.data(), hBytes, hipMemcpyHostToDevice));

  // indices: 每个 token 都是 top-1，且都指向对端节点的 expert
  std::vector<int32_t> h_idx(static_cast<size_t>(numTokens) * numEpt, destExpert);
  void* d_indices = gpu_alloc_zero(h_idx.size() * sizeof(int32_t));
  HIP_CHECK(
      hipMemcpy(d_indices, h_idx.data(), h_idx.size() * sizeof(int32_t), hipMemcpyHostToDevice));

  // weights: top-1, weight = 1.0
  std::vector<float> h_wgt(static_cast<size_t>(numTokens) * numEpt, 1.0f);
  void* d_weights = gpu_alloc_zero(h_wgt.size() * sizeof(float));
  HIP_CHECK(
      hipMemcpy(d_weights, h_wgt.data(), h_wgt.size() * sizeof(float), hipMemcpyHostToDevice));

  printf("[PE%d] %d input tokens, values %g..%g (all distinct, bf16-exact) → expert%d (PE%d)\n",
         rank, numTokens, token_value(rank, 0), token_value(rank, numTokens - 1), destExpert,
         destExpert / numExpertPerRank);

  // ── config ────────────────────────────────────────────────────────────────
  EpDispatchCombineConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = world;  // 2
  cfg.hiddenDim = hiddenDim;
  cfg.numExpertPerRank = numExpertPerRank;  // 1
  cfg.numExpertPerToken = numEpt;           // 1
  // 决定每 PE 的 staging / chunk-flag 容量，必须 >= numTokens。
  // 同时 maxChunkNum = ceil(maxNumInpTokenPerRank / warpSize) = 512/64 = 8，
  // 即 combine 侧的 bid 空间 = numRecvBlock(8) * maxChunkNum(8) * (nNodes-1)(1) = 64。
  cfg.maxNumInpTokenPerRank = 512;
  assert(numTokens <= cfg.maxNumInpTokenPerRank &&
         "numTokens exceeds maxNumInpTokenPerRank: staging buffer would overflow");
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
  //
  // 固定成 16（而不是按 CU 数取 multiProcessorCount/4）：combine 侧的 bid 空间是
  // numRecvBlock(8) * maxChunkNum(8) * (nNodes-1)(1) = 64，取 16 让每个 block
  // 需要循环 64/16 = 4 轮才能走完自己那份 bid，于是 bid != blockId —— 而按 CU 取
  // 值时（MI355X 上是 64）恰好一人一个 bid，bid 恒等于 blockId，那条多轮路径
  // 一次都不会走到。
  handle.config.rdmaBlockNum = 16;
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
  printf("[PE%d][Dispatch] received %lld token(s) from remote (expect %d):\n", rank,
         (long long)h_total, numTokens);
  if (h_total > 0) dump_bf16(rank, "dispatchOut", d_disp_out, (int)h_total, hiddenDim);

  // ── simulate FFN: multiply by 2.0 ────────────────────────────────────────
  std::vector<__hip_bfloat16> h_disp_out(static_cast<size_t>(h_total) * hiddenDim);
  HIP_CHECK(hipMemcpy(h_disp_out.data(), d_disp_out, h_disp_out.size() * sizeof(__hip_bfloat16),
                      hipMemcpyDeviceToHost));

  // dispatch 侧独立校验：收到的 token 应当恰好是对端那 64 个值的一个排列。
  // dispatch 的接收顺序由 atomicAdd 抢占决定，不保证和发送端一致，所以按多重集比。
  // 这一步能把 dispatch 阶段的丢失/污染和 combine 阶段的问题区分开。
  bool dispatchOk = (h_total == numTokens);
  if (!dispatchOk)
    printf("[PE%d][Dispatch] MISMATCH: token count %lld != %d\n", rank, (long long)h_total,
           numTokens);
  {
    const int peer = 1 - rank;
    std::vector<float> got, want;
    for (int t = 0; t < static_cast<int>(h_total); ++t) {
      // 每个 token 的 hiddenDim 个元素本应全等，不等说明这个 token 被写坏了。
      float v0 = __bfloat162float(h_disp_out[static_cast<size_t>(t) * hiddenDim]);
      for (int d = 1; d < hiddenDim; ++d) {
        if (__bfloat162float(h_disp_out[static_cast<size_t>(t) * hiddenDim + d]) != v0) {
          printf("[PE%d][Dispatch] token %d not uniform across hiddenDim (d=%d)\n", rank, t, d);
          dispatchOk = false;
          break;
        }
      }
      got.push_back(v0);
    }
    for (int t = 0; t < numTokens; ++t) want.push_back(token_value(peer, t));
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got != want) {
      dispatchOk = false;
      printf("[PE%d][Dispatch] MISMATCH: received token values are not a permutation of PE%d's\n",
             rank, peer);
      for (int t = 0; t < static_cast<int>(got.size()) && t < 16; ++t)
        printf("[PE%d][Dispatch]   got[%d]=%g want[%d]=%g\n", rank, t, got[t], t,
               (t < static_cast<int>(want.size())) ? want[t] : -1.0f);
    }
  }
  printf("[PE%d][Dispatch] %s\n", rank, dispatchOk ? "PASS" : "FAIL");
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
  // combineOut 按**本地 token 顺序**索引（不是接收顺序），所以每个 token 的
  // 期望值是它自己的输入 ×2，逐 token 各不相同 —— 这正是能抓出错位/丢失的地方。
  //   PE0: tok t 输入 1+t   → combineOut = 2+2t     (2..130)
  //   PE1: tok t 输入 101+t → combineOut = 202+2t   (202..328)
  void* d_comb_out = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(static_cast<size_t>(numTokens) * hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb_out, hBytes, hipMemcpyDeviceToHost));

  printf("\n[PE%d][Combine] combineOut (%d tokens, expect %g..%g):\n", rank, numTokens,
         token_value(rank, 0) * 2.0f, token_value(rank, numTokens - 1) * 2.0f);
  bool ok = true;
  int badTokens = 0;
  // 校验覆盖全部 token × 全部 hiddenDim，但只打印前 kDumpTokens 个 token
  // 的前 kDumpWidth 个元素，外加所有出错的 token。
  for (int t = 0; t < numTokens; ++t) {
    float expected = token_value(rank, t) * 2.0f;  // input * FFN-scale
    bool tokOk = true;
    float firstBad = 0.0f;
    int firstBadDim = -1;
    for (int d = 0; d < hiddenDim; ++d) {
      float got = __bfloat162float(h_comb[static_cast<size_t>(t) * hiddenDim + d]);
      // 取值全程无舍入（见 token_value 处说明），所以用严格相等而非容差：
      // 容差会把"收到隔壁 token"这类相差 2 的错误放过去。
      if (got != expected && tokOk) {
        tokOk = false;
        firstBad = got;
        firstBadDim = d;
      }
    }
    if (!tokOk) {
      ok = false;
      badTokens++;
      if (badTokens <= 16)
        printf("[PE%d]   tok%-3d MISMATCH: expected %g, got %g (first at dim %d)\n", rank, t,
               expected, firstBad, firstBadDim);
    } else if (t < kDumpTokens) {
      printf("[PE%d]   tok%-3d [", rank, t);
      for (int d = 0; d < std::min(hiddenDim, kDumpWidth); ++d)
        printf("%s%g", d ? "," : "",
               __bfloat162float(h_comb[static_cast<size_t>(t) * hiddenDim + d]));
      printf("%s]  OK (expected %g)\n", (hiddenDim > kDumpWidth) ? ",..." : "", expected);
    }
  }
  if (badTokens > 16) printf("[PE%d]   ... (%d more bad token(s))\n", rank, badTokens - 16);
  printf("[PE%d][Combine] %d/%d tokens correct\n", rank, numTokens - badTokens, numTokens);
  ok = ok && dispatchOk;
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
