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
// IntraNode dispatch/combine example — single-kernel dispatch + single-kernel
// combine, XGMI/P2P only (no RDMA). run_intranode() is called twice: once with
// the combine FP8 direct-cast path disabled and once with it enabled.
//
// Shared topology:
//   npes=2, numQpPerPe=2, numEpt=2, numTokens=3, hiddenDim=4
//   4 experts total: expert 0,1 → PE0 | expert 2,3 → PE1
//
// Token routing (each rank sends numEpt=2 experts per token):
//   token0 → expert[0(PE0), 2(PE1)]  ← cross-PE
//   token1 → expert[1(PE0), 1(PE0)]  ← same PE, dedup → 只发1次
//   token2 → expert[2(PE1), 3(PE1)]  ← cross-PE
//
// ──────────────────────────────────────────────────────────────────────────
// IntraNode kernel flow (single-kernel dispatch + single-kernel combine,
// XGMI/P2P only — no RDMA):
//
//  dispatch (1 kernel):
//   EpDispatchIntraNodeKernel — 每个 warp 处理一个 (token,expert) 对: dedup 同 PE 的重复 expert,
//                               atomicAdd 分配 dest slot, 然后 XGMI 直写对端 dispatchOut/
//                               indices/weights, 最后 grid barrier + 通知/等待对端 token 数
//
//  combine (1 kernel, 由 quantType 决定 kernel 变体; useExternalInpBuffer=true → "_nop2p"
//  = push 模型: Stage1 每个 PE 把自己收到 token 的 FFN 输出 P2P 直写到各目的 PE 的 combineInp
//  暂存区, barrier 后 Stage2 只需本地读 combineInp 做加权求和, 不需要在累加阶段发起 P2P 读):
//   EpCombineIntraNodeKernel_bf16_nop2p         — FP8 disabled: Stage1 原样拷 bf16
//   EpCombineIntraNodeKernel_bf16_nop2p_fp8cast — FP8 enabled:  Stage1 顺便把数据转成
//                                                 内部 8bit 表示(无 blockwise scale), 减少
//                                                 P2P 写带宽, Stage2 本地读回再 dequant 累加
// ──────────────────────────────────────────────────────────────────────────
//
// Launch: mpirun -np 2 ./async_ll_dispatch_example
// ──────────────────────────────────────────────────────────────────────────

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::moe;
using namespace mori::shmem;

// 4 experts total, 2 experts per rank: expert 0,1 → PE0 | expert 2,3 → PE1
constexpr int kNumExpertPerRank = 2;

// ---------------------------------------------------------------------------
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

static void load_kernels() {
  KernelRegistry::Instance().AutoLoad();
  if (!KernelRegistry::Instance().IsLoaded()) {
    char buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
      std::string exe(buf, n);
      std::string exe_dir = exe.substr(0, exe.rfind('/'));
      KernelRegistry::Instance().AutoLoad(exe_dir + "/../lib");
    }
  }
}

// Pretty-print each local token's value together with which PE(s) it routes to.
// Dedup rule mirrors EpDispatchIntraNodeKernel's own logic (intranode.hpp:142-154):
// a slot is skipped ("重复,去重跳过") if an earlier slot of the same token already
// targets the same destination PE.
static void print_routing_table(int rank, int numTokens, int hiddenDim, int numEpt,
                                const std::vector<__hip_bfloat16>& h_input, const int32_t* h_idx) {
  printf("[PE%d] +------------------- local tokens & routing (PE%d) -------------------+\n", rank,
         rank);
  for (int t = 0; t < numTokens; ++t) {
    printf("[PE%d] | tok%d val=[", rank, t);
    for (int d = 0; d < hiddenDim; ++d)
      printf("%s%.0f", d ? "," : "", __bfloat162float(h_input[t * hiddenDim + d]));
    printf("]  ->");
    for (int k = 0; k < numEpt; ++k) {
      int expert = h_idx[t * numEpt + k];
      int destPe = expert / kNumExpertPerRank;
      bool dup = false;
      for (int k2 = 0; k2 < k; ++k2) {
        if (h_idx[t * numEpt + k2] / kNumExpertPerRank == destPe) {
          dup = true;
          break;
        }
      }
      printf("  expert%d->PE%d%s", expert, destPe, dup ? "(重复,去重跳过)" : "");
    }
    printf("\n");
  }
  printf("[PE%d] +----------------------------------------------------------------------+\n", rank);
}

// ---------------------------------------------------------------------------
// Shared test data: 3 tokens, hiddenDim=4, numEpt=2
//   token0 → expert[0(PE0), 2(PE1)]
//   token1 → expert[1(PE0), 1(PE0)]  ← dedup
//   token2 → expert[2(PE1), 3(PE1)]
// ---------------------------------------------------------------------------
struct TestData {
  int numTokens = 3;
  int hiddenDim = 4;
  int numEpt = 2;
  int rank;

  void* d_input = nullptr;
  void* d_indices = nullptr;
  void* d_weights = nullptr;
  std::vector<int32_t> h_idx;  // kept around for later pretty-printing (e.g. dispDestTokIdMap)

  void alloc_and_fill(int rank_) {
    rank = rank_;
    const size_t hBytes = numTokens * hiddenDim * sizeof(__hip_bfloat16);

    std::vector<__hip_bfloat16> h_input(numTokens * hiddenDim);
    for (int t = 0; t < numTokens; ++t) {
      float val = rank * 10.0f + t;
      for (int d = 0; d < hiddenDim; ++d) h_input[t * hiddenDim + d] = __float2bfloat16(val);
    }

    h_idx = {0, 2, 1, 1, 2, 3};
    float h_wgt[6] = {0.6f, 0.4f, 0.5f, 0.5f, 0.7f, 0.3f};
    print_routing_table(rank, numTokens, hiddenDim, numEpt, h_input, h_idx.data());

    HIP_CHECK(hipMalloc(&d_input, hBytes));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), hBytes, hipMemcpyHostToDevice));

    d_indices = gpu_alloc_zero(numTokens * numEpt * sizeof(int32_t));
    d_weights = gpu_alloc_zero(numTokens * numEpt * sizeof(float));
    HIP_CHECK(
        hipMemcpy(d_indices, h_idx.data(), h_idx.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_wgt, sizeof(h_wgt), hipMemcpyHostToDevice));
  }

  void free_all() {
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
  }
};

// ===========================================================================
// run_intranode: IntraNode — 1 dispatch + 1 combine kernel, XGMI/P2P only
//
//   enableFp8 == false → EpCombineIntraNodeKernel_bf16_nop2p         (plain bf16 combine)
//   enableFp8 == true  → EpCombineIntraNodeKernel_bf16_nop2p_fp8cast (FP8 direct-cast combine)
//
// Dispatch is unchanged between the two variants: only the combine phase's
// intra-node payload is quantized, to cut P2P read bandwidth on the combine
// all-to-all. quantType only takes effect on combine — see
// EpDispatchCombineHandle::LaunchCombine()'s IntraNode branch in launch.cpp.
// ===========================================================================
static void run_intranode(int rank, int world, hipStream_t stream, bool enableFp8) {
  printf("\n[PE%d] ===== IntraNode (FP8 %s) =====\n", rank, enableFp8 ? "enabled" : "disabled");

  const int kWarpSize = 64;
  TestData td;
  td.alloc_and_fill(rank);

  EpDispatchCombineConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = world;
  cfg.hiddenDim = td.hiddenDim;
  cfg.numExpertPerRank = 2;
  cfg.numExpertPerToken = td.numEpt;
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe = 2;
  cfg.gpuPerNode = 2;
  cfg.kernelType = KernelType::IntraNode;
  cfg.warpNumPerBlock = 4;
  // useExternalInpBuffer=true: our input/FFN-output buffers are plain hipMalloc
  // allocations (not shmem-registered), so combine must copy them into the
  // registered staging buffer before other PEs can read them (the "_nop2p" path).
  cfg.useExternalInpBuffer = true;
  // FP8 direct-cast only affects the combine kernel variant selected below; it
  // requires bf16 input/output and useExternalInpBuffer=true (both already set).
  cfg.quantType = enableFp8 ? QuantType::Fp8DirectCast : QuantType::None;
  cfg.enableSdma = false;

  EpDispatchCombineHandle handle(cfg);

  int mp = handle.multiProcessorCount;
  int mp_aligned = (mp / world) * world;
  cfg.blockNum = mp_aligned;
  const std::string sfx = "bf16";

  // ── dispatch: single kernel ─────────────────────────────────────────────
  handle.PrepareInference(HIP_R_16BF, td.d_input, nullptr, reinterpret_cast<float*>(td.d_weights),
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, 0);
  args.config.hiddenDim = td.hiddenDim;
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);

  // D1: EpDispatchIntraNodeKernel — dedup + slot assign + XGMI 直写对端 dispatchOut
  printf("[PE%d][IntraNode] D1: EpDispatchIntraNodeKernel  grid=%d block=%d\n", rank, mp_aligned,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpDispatchIntraNodeKernel_" + sfx, mp_aligned,
                                    kWarpSize * cfg.warpNumPerBlock, 0, stream, &args, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify dispatchOut ────────────────────────────────────────────────────
  index_t h_total = 0;
  HIP_CHECK(hipMemcpy(&h_total, args.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));
  void* d_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
  const size_t hBytes = td.hiddenDim * sizeof(__hip_bfloat16);
  std::vector<__hip_bfloat16> h_out(h_total * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_total * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][IntraNode] dispatchOut (%lld tokens):\n", rank, (long long)h_total);
  for (index_t i = 0; i < h_total; ++i) {
    printf("[PE%d]   tok%lld: [", rank, (long long)i);
    for (int d = 0; d < td.hiddenDim; ++d)
      printf("%s%.0f", d ? "," : "", __bfloat162float(h_out[i * td.hiddenDim + d]));
    printf("]\n");
  }

  // ── simulate FFN: scale by 2.0 ───────────────────────────────────────────
  std::vector<__hip_bfloat16> h_ffn_out(h_total * td.hiddenDim);
  for (index_t i = 0; i < h_total * td.hiddenDim; ++i)
    h_ffn_out[i] = __float2bfloat16(__bfloat162float(h_out[i]) * 2.0f);
  void* d_ffn_out = nullptr;
  HIP_CHECK(hipMalloc(&d_ffn_out, h_total * hBytes));
  HIP_CHECK(hipMemcpy(d_ffn_out, h_ffn_out.data(), h_total * hBytes, hipMemcpyHostToDevice));

  // ── combine: single kernel, variant picked by quantType ──────────────────
  // IMPORTANT: the weights buffer combine expects is indexed by *received* token
  // order (0..h_total), matching d_ffn_out — NOT by this rank's local token order
  // (0..numTokens). The weights that travel alongside each received token were
  // already forwarded during dispatch into shmemDispatchOutWeightsMemObj (mirrors
  // GetShmemDispatchOutTokMemObj() for the data), so reuse that buffer directly
  // instead of td.d_weights (wrong size/order — would read local-token slots and
  // even go out of bounds for remote-received tokens).
  float* d_recv_weights =
      reinterpret_cast<float*>(handle.shmemDispatchOutWeightsMemObj.cpu->localPtr);
  handle.PrepareInference(HIP_R_16BF, d_ffn_out, nullptr, d_recv_weights,
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);
  EpDispatchCombineArgsRaw cargs = GetEpDispatchCombineArgsRaw(handle, 0);
  cargs.config.hiddenDim = td.hiddenDim;
  int combine_smem = cfg.warpNumPerBlock * cfg.numExpertPerToken * 16;

  const std::string combine_kernel = enableFp8 ? "EpCombineIntraNodeKernel_bf16_nop2p_fp8cast"
                                               : "EpCombineIntraNodeKernel_bf16_nop2p";
  printf("[PE%d][IntraNode] C1: %s  grid=%d block=%d\n", rank, combine_kernel.c_str(), mp_aligned,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch(combine_kernel, mp_aligned, kWarpSize * cfg.warpNumPerBlock,
                                    combine_smem, stream, &cargs, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify combineOut ─────────────────────────────────────────────────────
  void* d_comb = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(td.numTokens * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb, td.numTokens * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][IntraNode] combineOut (%d tokens):\n", rank, td.numTokens);
  for (int t = 0; t < td.numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int d = 0; d < td.hiddenDim; ++d)
      printf("%s%.1f", d ? "," : "", __bfloat162float(h_comb[t * td.hiddenDim + d]));
    printf("]\n");
  }

  // ── verify combineOutWeights (sum of each contributor's forwarded weight) ─
  void* d_comb_w = handle.shmemCombineOutWeightsMemObj.cpu->localPtr;
  std::vector<float> h_comb_w(td.numTokens * td.numEpt);
  HIP_CHECK(
      hipMemcpy(h_comb_w.data(), d_comb_w, h_comb_w.size() * sizeof(float), hipMemcpyDeviceToHost));
  printf("[PE%d][IntraNode] combineOutWeights (%d tokens x %d slots):\n", rank, td.numTokens,
         td.numEpt);
  for (int t = 0; t < td.numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int k = 0; k < td.numEpt; ++k) printf("%s%.2f", k ? "," : "", h_comb_w[t * td.numEpt + k]);
    printf("]\n");
  }
  printf("[PE%d][IntraNode] done (FP8 %s)\n", rank, enableFp8 ? "enabled" : "disabled");

  HIP_CHECK(hipFree(d_ffn_out));
  td.free_all();
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
  assert(world == 2);

  load_kernels();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Run IntraNode with FP8 combine disabled
  run_intranode(rank, world, stream, /*enableFp8=*/false);
  MPI_Barrier(MPI_COMM_WORLD);

  // Run IntraNode with FP8 combine enabled
  // run_intranode(rank, world, stream, /*enableFp8=*/true);
  // MPI_Barrier(MPI_COMM_WORLD);

  HIP_CHECK(hipStreamDestroy(stream));

  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
  return 0;
}
