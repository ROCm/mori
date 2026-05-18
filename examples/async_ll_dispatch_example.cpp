// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Two dispatch/combine examples in one file:
//
//  run_async_ll()  — AsyncLL:  5 dispatch kernels + 4 combine kernels, fully explicit
//  run_v1ll()      — V1LL:     2 dispatch kernels + 4 combine kernels, fused send+recv
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
// V1LL kernel flow:
//
//  dispatch (2 kernels):
//   D1: EpDispatchCopyToStaging          → 把 input 打包进 staging (hidden+idx+wgt+srcId)
//   D2: EpDispatchInterNodeV1KernelLowLatency
//         rdma blocks: DispatchInterNodeLLSend  → 逐 chunk RDMA PUT + chunkFlag signal
//                      DispatchInterNodeLLRecv  → spin-wait chunkFlag, WarpCopy 到 dispatchOut
//         xgmi blocks: DispatchIntraNode        → XGMI 直写 dispatchOut
//         all blocks:  DispatchSync             → grid barrier → 写 recvTokenNum → crossDevBarrier++
//
//  combine (4 kernels):
//   C1: EpCombineSync        → 把 FFN 输出拷到 combineInp + weights 到 shmem
//   C2: EpCombineSyncBarrier → 等 crossDeviceBarrierFlag (dispatch 已完)
//   C3: EpCombineInterNodeV1KernelLowLatency
//         rdma blocks: CombineInterNodeLL → WarpAccum → staging → RDMA PUT 回 srcPe
//         xgmi blocks: CombineIntraNodeLL → WarpAccum → staging
//   C4: EpCombineAll         → 汇聚各 node 的 staging 结果 → combineOut
//
// Launch: mpirun -np 2 ./async_ll_dispatch_example
// ──────────────────────────────────────────────────────────────────────────

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

// ---------------------------------------------------------------------------
// Shared test data: 3 tokens, hiddenDim=4, numEpt=2
//   token0 → expert[0(PE0), 2(PE1)]
//   token1 → expert[1(PE0), 1(PE0)]  ← dedup
//   token2 → expert[2(PE1), 3(PE1)]
// ---------------------------------------------------------------------------
struct TestData {
  int numTokens = 3;
  int hiddenDim = 4;
  int numEpt    = 2;
  int rank;

  void* d_input   = nullptr;
  void* d_indices = nullptr;
  void* d_weights = nullptr;

  void alloc_and_fill(int rank_) {
    rank = rank_;
    const size_t hBytes = numTokens * hiddenDim * sizeof(__hip_bfloat16);

    std::vector<__hip_bfloat16> h_input(numTokens * hiddenDim);
    for (int t = 0; t < numTokens; ++t) {
      float val = rank * 10.0f + t;
      for (int d = 0; d < hiddenDim; ++d) h_input[t * hiddenDim + d] = __float2bfloat16(val);
    }
    printf("[PE%d] input (%d tokens):\n", rank, numTokens);
    for (int t = 0; t < numTokens; ++t) {
      printf("[PE%d]   tok%d: [", rank, t);
      for (int d = 0; d < hiddenDim; ++d)
        printf("%s%.0f", d ? "," : "", __bfloat162float(h_input[t * hiddenDim + d]));
      printf("]\n");
    }

    HIP_CHECK(hipMalloc(&d_input, hBytes));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), hBytes, hipMemcpyHostToDevice));

    int32_t h_idx[6] = {0, 2, 1, 1, 2, 3};
    float   h_wgt[6] = {0.6f, 0.4f, 0.5f, 0.5f, 0.7f, 0.3f};
    d_indices = gpu_alloc_zero(numTokens * numEpt * sizeof(int32_t));
    d_weights = gpu_alloc_zero(numTokens * numEpt * sizeof(float));
    HIP_CHECK(hipMemcpy(d_indices, h_idx, sizeof(h_idx), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_wgt, sizeof(h_wgt), hipMemcpyHostToDevice));
  }

  void free_all() {
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
  }
};

// ===========================================================================
// run_async_ll: AsyncLL — 5 dispatch + 4 combine kernels
// ===========================================================================
static void run_async_ll(int rank, int world, hipStream_t stream) {
  printf("\n[PE%d] ===== AsyncLL =====\n", rank);

  const int kWarpSize = 64;
  TestData td;
  td.alloc_and_fill(rank);

  EpDispatchCombineConfig cfg;
  cfg.rank               = rank;
  cfg.worldSize          = world;
  cfg.hiddenDim          = td.hiddenDim;
  cfg.numExpertPerRank   = 2;
  cfg.numExpertPerToken  = td.numEpt;
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe         = 2;
  cfg.gpuPerNode         = 2;
  cfg.kernelType         = KernelType::AsyncLL;
  cfg.warpNumPerBlock    = 4;
  cfg.blockNum           = world;
  cfg.enableSdma         = false;

  EpDispatchCombineHandle handle(cfg);

  handle.PrepareInference(HIP_R_16BF, td.d_input, nullptr,
                          reinterpret_cast<float*>(td.d_weights),
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, 0);
  args.config.hiddenDim = td.hiddenDim;
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);

  int mp         = handle.multiProcessorCount;
  int mp_aligned = (mp / world) * world;
  const std::string sfx = "bf16";

  // ── dispatch_send: K1 + K2 + K3 ──────────────────────────────────────────

  // K1: SlotAssign — dedup + atomicAdd 分配 staging slot
  printf("[PE%d][AsyncLL] K1: SlotAssign  grid=%d block=%d\n", rank, mp_aligned, kWarpSize * 16);
  KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendCopySlotAssign_" + sfx,
                                    mp_aligned, kWarpSize * 16, 0, stream, &args, args_size);

  // K2: SendCopyMultiBlock — 多 warp 协作把 hidden+meta 拷到 staging
  printf("[PE%d][AsyncLL] K2: SendCopyMultiBlock  grid=%d block=%d\n", rank, mp_aligned,
         kWarpSize * 16);
  KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendCopyMultiBlock_" + sfx,
                                    mp_aligned, kWarpSize * 16, 0, stream, &args, args_size);

  // K3: SendTransfer — 按 destPe/qpId 发 RDMA PUT (non-blocking)
  printf("[PE%d][AsyncLL] K3: SendTransfer  grid=%d block=%d\n", rank, world,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendTransfer_" + sfx,
                                    world, kWarpSize * cfg.warpNumPerBlock, 0, stream, &args,
                                    args_size);

  printf("[PE%d][AsyncLL] (RDMA in-flight, local FFN overlap here)\n", rank);

  // ── dispatch_recv: K4 + K5 ───────────────────────────────────────────────

  // K4: RecvTransfer — quiet → signal → poll 对端信号
  printf("[PE%d][AsyncLL] K4: RecvTransfer  grid=%d block=%d\n", rank, world,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncRecvTransfer_" + sfx,
                                    world, kWarpSize * cfg.warpNumPerBlock, 0, stream, &args,
                                    args_size);

  // K5: RecvCopyMultiBlock — prefix sum 算 offset + 拷到 dispatchOut
  printf("[PE%d][AsyncLL] K5: RecvCopyMultiBlock  grid=%d block=%d\n", rank, mp_aligned,
         kWarpSize * 16);
  KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncRecvCopyMultiBlock_" + sfx,
                                    mp_aligned, kWarpSize * 16, 0, stream, &args, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify dispatchOut ────────────────────────────────────────────────────
  index_t h_total = 0;
  HIP_CHECK(hipMemcpy(&h_total, args.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));
  void* d_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
  const size_t hBytes = td.hiddenDim * sizeof(__hip_bfloat16);
  std::vector<__hip_bfloat16> h_out(h_total * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_total * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][AsyncLL] dispatchOut (%lld tokens):\n", rank, (long long)h_total);
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

  // ── combine_send: C1 + C2 ────────────────────────────────────────────────
  handle.PrepareInference(HIP_R_16BF, d_ffn_out, nullptr,
                          reinterpret_cast<float*>(td.d_weights),
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);
  EpDispatchCombineArgsRaw cargs = GetEpDispatchCombineArgsRaw(handle, 0);
  cargs.config.hiddenDim = td.hiddenDim;
  int combine_smem = cfg.warpNumPerBlock * cfg.numExpertPerToken * 16;

  // C1: CombineSendCopy — 按 dispReceiverIdxMap 把 FFN 结果拷回 staging slot
  printf("[PE%d][AsyncLL] C1: CombineSendCopy  grid=%d block=%d\n", rank, mp_aligned,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncSendCopy_" + sfx,
                                    mp_aligned, kWarpSize * cfg.warpNumPerBlock, 0, stream,
                                    &cargs, args_size);

  // C2: CombineSendTransfer — RDMA PUT 回 srcPe 的 combineInp
  printf("[PE%d][AsyncLL] C2: CombineSendTransfer  grid=%d block=%d\n", rank, world,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncSendTransfer_" + sfx,
                                    world, kWarpSize * cfg.warpNumPerBlock, 0, stream,
                                    &cargs, args_size);

  // ── combine_recv: C3 + C4 ────────────────────────────────────────────────
  EpDispatchCombineArgsRaw crargs = GetEpDispatchCombineArgsRaw(handle, 0);
  crargs.config.hiddenDim = td.hiddenDim;

  // C3: CombineRecvTransfer — quiet → 发 barrier flag → poll 所有 PE 的 barrier
  printf("[PE%d][AsyncLL] C3: CombineRecvTransfer  grid=%d block=%d\n", rank, world,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncRecvTransfer_" + sfx,
                                    world, kWarpSize * cfg.warpNumPerBlock, 0, stream,
                                    &crargs, args_size);

  // C4: CombineRecvCopy — 从 dispDestTokIdMap 找 staging slot，WarpAccum 写 combineOut
  printf("[PE%d][AsyncLL] C4: CombineRecvCopy  grid=%d block=%d\n", rank, mp_aligned,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncRecvCopy_" + sfx,
                                    mp_aligned, kWarpSize * cfg.warpNumPerBlock, combine_smem,
                                    stream, &crargs, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify combineOut ─────────────────────────────────────────────────────
  void* d_comb = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(td.numTokens * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb, td.numTokens * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][AsyncLL] combineOut (%d tokens):\n", rank, td.numTokens);
  for (int t = 0; t < td.numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int d = 0; d < td.hiddenDim; ++d)
      printf("%s%.1f", d ? "," : "", __bfloat162float(h_comb[t * td.hiddenDim + d]));
    printf("]\n");
  }
  printf("[PE%d][AsyncLL] done\n", rank);

  HIP_CHECK(hipFree(d_ffn_out));
  td.free_all();
}

// ===========================================================================
// run_v1ll: V1LL — 2 dispatch + 4 combine kernels
//
// Dispatch 内部分工（由 rdmaBlockNum 控制）：
//   blockId < rdmaBlockNum → DispatchInterNodeLLSend  (RDMA PUT + chunkFlag signal)
//                          → DispatchInterNodeLLRecv  (spin-wait + XGMI WarpCopy)
//   blockId >= rdmaBlockNum → DispatchIntraNode        (节点内 XGMI 直写)
//   所有 block:             → DispatchSync             (grid barrier + crossDevBarrier++)
// ===========================================================================
static void run_v1ll(int rank, int world, hipStream_t stream) {
  printf("\n[PE%d] ===== V1LL =====\n", rank);

  const int kWarpSize = 64;
  TestData td;
  td.alloc_and_fill(rank);

  // nNodes = worldSize / gpuPerNode = 2/2 = 1 (单节点双 GPU)
  // 单节点场景没有跨节点 RDMA: rdmaBlockNum=0, 全部走 DispatchIntraNode
  // 若 nNodes>1 则 rdmaBlockNum>0, RDMA block 负责跨节点, 其余 block 负责节点内
  const int nNodes      = world / 2;  // gpuPerNode=2
  const int rdmaBlockNum = (nNodes > 1) ? 2 : 0;

  EpDispatchCombineConfig cfg;
  cfg.rank               = rank;
  cfg.worldSize          = world;
  cfg.hiddenDim          = td.hiddenDim;
  cfg.numExpertPerRank   = 2;
  cfg.numExpertPerToken  = td.numEpt;
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe         = 2;
  cfg.gpuPerNode         = 2;
  cfg.kernelType         = KernelType::InterNodeV1LL;
  cfg.warpNumPerBlock    = 4;
  cfg.rdmaBlockNum       = rdmaBlockNum;
  // 总 block 数 = rdmaBlockNum (RDMA) + xgmi blocks
  // xgmi blocks 处理节点内 token, 至少需要 1
  cfg.blockNum           = rdmaBlockNum + world;
  cfg.enableSdma         = false;

  EpDispatchCombineHandle handle(cfg);

  int mp         = handle.multiProcessorCount;
  int mp_aligned = (mp / world) * world;
  const std::string sfx = "bf16";
  int dispatch_smem = (cfg.worldSize * cfg.warpNumPerBlock +
                       cfg.numExpertPerRank * cfg.warpNumPerBlock +
                       cfg.numExpertPerRank) * static_cast<int>(sizeof(index_t));
  int combine_smem  = cfg.warpNumPerBlock * cfg.numExpertPerToken * 16;
  size_t args_size  = sizeof(EpDispatchCombineArgsRaw);

  // ── dispatch: D1 + D2 ────────────────────────────────────────────────────

  handle.PrepareInference(HIP_R_16BF, td.d_input, nullptr,
                          reinterpret_cast<float*>(td.d_weights),
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  args.config.hiddenDim = td.hiddenDim;

  // D1: CopyToStaging
  //   全量 SM 运行，把每个 token 打包成 [hidden | indices | weights | scales | srcTokId]
  //   写入 staging buffer（RDMA 可见的 pinned 内存）
  printf("[PE%d][V1LL] D1: CopyToStaging  grid=%d block=%d\n", rank, mp, kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpDispatchCopyToStaging_" + sfx,
                                    /*grid_x=*/mp,
                                    /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                    /*smem=*/0, stream, &args, args_size);

  // D2: DispatchInterNodeV1KernelLowLatency
  //   rdma blocks (blockId < rdmaBlockNum):
  //     DispatchInterNodeLLSend: 按 warpSize chunk 循环 nNodes, laneId==0 发 RDMA PUT+signal
  //     DispatchInterNodeLLRecv: 展开 (expert × token × node), spin-wait chunkFlag,
  //                              WarpCopy staging → dispatchOut + shmem indices/weights
  //   xgmi blocks (blockId >= rdmaBlockNum):
  //     DispatchIntraNode: destNode==myNode 的 token, XGMI 直写 dispatchOut
  //   all blocks:
  //     DispatchSync: grid barrier → 向对端写 recvTokenNum → crossDeviceBarrierFlag++
  printf("[PE%d][V1LL] D2: DispatchV1LL  grid=%d block=%d smem=%d (rdmaBlocks=%d)\n",
         rank, cfg.blockNum, kWarpSize * cfg.warpNumPerBlock, dispatch_smem, rdmaBlockNum);
  KernelRegistry::Instance().Launch("EpDispatchInterNodeV1KernelLowLatency_" + sfx,
                                    /*grid_x=*/cfg.blockNum,
                                    /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                    /*smem=*/dispatch_smem, stream, &args, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify dispatchOut ────────────────────────────────────────────────────
  index_t h_total = 0;
  HIP_CHECK(hipMemcpy(&h_total, args.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));
  void* d_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
  const size_t hBytes = td.hiddenDim * sizeof(__hip_bfloat16);
  std::vector<__hip_bfloat16> h_out(h_total * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_total * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][V1LL] dispatchOut (%lld tokens):\n", rank, (long long)h_total);
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

  // ── combine: C1 + C2 + C3 + C4 ───────────────────────────────────────────
  handle.PrepareInference(HIP_R_16BF, d_ffn_out, nullptr,
                          reinterpret_cast<float*>(td.d_weights),
                          reinterpret_cast<int32_t*>(td.d_indices), td.numTokens);

  EpDispatchCombineArgsRaw cargs = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  cargs.config.hiddenDim = td.hiddenDim;

  // C1: CombineSync
  //   全量 SM 运行，把 FFN 输出从 inpTokenBuf 拷到 combineInp (RDMA 可见)
  //   同时把 weights 写到 shmemInpWeightsMemObj
  printf("[PE%d][V1LL] C1: CombineSync  grid=%d block=%d\n", rank, mp,
         kWarpSize * cfg.warpNumPerBlock);
  KernelRegistry::Instance().Launch("EpCombineSync_" + sfx,
                                    /*grid_x=*/mp,
                                    /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                    /*smem=*/0, stream, &cargs, args_size);

  // C2: CombineSyncBarrier
  //   1 block, 1 warp: 轮询 crossDeviceBarrierMemObj 等于 crossDeviceBarrierFlag
  //   确认所有 rank 的 dispatch 都已完成才允许 combine 跨节点操作
  printf("[PE%d][V1LL] C2: CombineSyncBarrier  grid=1 block=%d\n", rank, kWarpSize);
  KernelRegistry::Instance().Launch("EpCombineSyncBarrier_" + sfx,
                                    /*grid_x=*/1,
                                    /*block_x=*/kWarpSize,
                                    /*smem=*/0, stream, &cargs, args_size);

  // C3: CombineInterNodeV1KernelLowLatency
  //   rdma blocks: CombineInterNodeLL
  //     固定 warpsPerToken=4 分片，从 interNodeDispDestTokIdMap 找 combineInp 位置
  //     WarpAccum 加权累加写 staging, chunk 完成后 RDMA PUT 回 srcPe + 清 chunkFlag
  //     最后跨节点 barrier (crossDeviceBarrierMemObj AMO + spin-wait)
  //   xgmi blocks: CombineIntraNodeLL
  //     MultiWarpIter 按 hiddenDim 分片, 从 dispDestTokIdMap 找本节点 source
  //     WarpAccum 累加写 staging
  printf("[PE%d][V1LL] C3: CombineV1LL  grid=%d block=%d smem=%d (rdmaBlocks=%d)\n",
         rank, cfg.blockNum, kWarpSize * cfg.warpNumPerBlock, combine_smem, rdmaBlockNum);
  KernelRegistry::Instance().Launch("EpCombineInterNodeV1KernelLowLatency_" + sfx,
                                    /*grid_x=*/cfg.blockNum,
                                    /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                    /*smem=*/combine_smem, stream, &cargs, args_size);

  // C4: CombineAll
  //   全量 SM 运行，此时各节点结果已 PUT 回本 PE 的 staging
  //   按 interNodeDispSendMap 找各 node 对应 staging slot
  //   WarpAccum 汇聚所有节点结果写 combineOut
  printf("[PE%d][V1LL] C4: CombineAll  grid=%d block=%d smem=%d\n", rank, mp_aligned,
         kWarpSize * cfg.warpNumPerBlock, combine_smem);
  KernelRegistry::Instance().Launch("EpCombineAll_" + sfx,
                                    /*grid_x=*/mp_aligned,
                                    /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                    /*smem=*/combine_smem, stream, &cargs, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));

  // ── verify combineOut ─────────────────────────────────────────────────────
  void* d_comb = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(td.numTokens * td.hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb, td.numTokens * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][V1LL] combineOut (%d tokens):\n", rank, td.numTokens);
  for (int t = 0; t < td.numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int d = 0; d < td.hiddenDim; ++d)
      printf("%s%.1f", d ? "," : "", __bfloat162float(h_comb[t * td.hiddenDim + d]));
    printf("]\n");
  }
  printf("[PE%d][V1LL] done\n", rank);

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

  const int rank  = ShmemMyPe();
  const int world = ShmemNPes();
  assert(world == 2);

  load_kernels();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Run AsyncLL
  run_async_ll(rank, world, stream);
  MPI_Barrier(MPI_COMM_WORLD);

  // Run V1LL
  run_v1ll(rank, world, stream);
  MPI_Barrier(MPI_COMM_WORLD);

  HIP_CHECK(hipStreamDestroy(stream));

  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
  return 0;
}
