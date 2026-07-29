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
// Kernel sequence (InterNodeV1):
//   Dispatch:
//     1. EpDispatchCopyToStaging  — 把 token 打包进 staging buffer
//     2. EpDispatchInterNodeV1Kernel — RDMA send + recv + sync
//   Combine:
//     1. EpCombineSync            — 把 FFN 输出拷进 combine staging
//     2. EpCombineSyncBarrier     — 跨节点 barrier
//     3. EpCombineInterNodeV1Kernel — combine internode (WarpAccum + RDMA 回送)
//        + CombineIntraNode (本地 XGMI 聚合)
//     4. EpCombineAll             — 从各节点 staging 读回做最终加权求和
//
// Launch:
//   mpirun -np 2 ./internode_v1_dispatch_example
//

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
// run_internode_v1
// ---------------------------------------------------------------------------
static void run_internode_v1(int rank, int world, hipStream_t stream) {
  printf("\n[PE%d] ===== InterNodeV1 dispatch/combine =====\n", rank);

  const int kWarpSize = 64;
  const int numTokens = 1;
  const int hiddenDim = 4;
  const int numEpt = 1;           // top-1
  const int numExpertPerRank = 1; // 1 expert per GPU
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

  printf("[PE%d] input token: [%.0f,%.0f,%.0f,%.0f] → expert%d (PE%d)\n", rank, val, val, val,
         val, destExpert, destExpert / numExpertPerRank);

  // ── config ────────────────────────────────────────────────────────────────
  EpDispatchCombineConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = world;           // 2
  cfg.hiddenDim = hiddenDim;
  cfg.numExpertPerRank = numExpertPerRank;  // 1
  cfg.numExpertPerToken = numEpt;           // 1
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe = 1;
  cfg.gpuPerNode = 1;              // 1 GPU per node → nNodes = world / gpuPerNode = 2
  cfg.kernelType = KernelType::InterNodeV1;
  cfg.warpNumPerBlock = 4;
  cfg.useExternalInpBuffer = true;
  cfg.quantType = QuantType::None;
  cfg.enableSdma = false;

  EpDispatchCombineHandle handle(cfg);

  int mp = handle.multiProcessorCount;
  // rdmaBlockNum: number of CUs dedicated to RDMA send/recv, rest handle intra-node XGMI
  // For InterNodeV1, half the blocks do RDMA, half do XGMI. Use a small fixed number.
  int rdmaBlockNum = std::max(1, mp / 4);
  // blockNum must be > rdmaBlockNum (the remainder is the XGMI portion)
  int bn = rdmaBlockNum * 2;

  const size_t args_size = sizeof(EpDispatchCombineArgsRaw);
  const unsigned int block_x = kWarpSize * cfg.warpNumPerBlock;

  // ── dispatch ─────────────────────────────────────────────────────────────
  // Step 1: pack token into staging buffer (runs on all CUs)
  // Step 2: EpDispatchInterNodeV1Kernel
  //   - blockId < rdmaBlockNum  → DispatchInterNodeSend + DispatchInterNodeRecv
  //   - blockId >= rdmaBlockNum → DispatchIntraNode (noop here, no same-node tokens)
  //   - all blocks              → DispatchSync

  handle.PrepareInference(HIP_R_16BF, d_input, nullptr,
                          reinterpret_cast<float*>(d_weights), nullptr,
                          reinterpret_cast<int32_t*>(d_indices), numTokens);

  EpDispatchCombineArgsRaw dargs = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  dargs.config.hiddenDim = hiddenDim;

  printf("[PE%d][Dispatch] D1: EpDispatchCopyToStaging  grid=%d block=%d\n", rank, mp, block_x);
  KernelRegistry::Instance().Launch("EpDispatchCopyToStaging_bf16", mp, block_x, 0, stream,
                                    &dargs, args_size);

  printf("[PE%d][Dispatch] D2: EpDispatchInterNodeV1Kernel  grid=%d rdmaBlocks=%d\n", rank, bn,
         rdmaBlockNum);
  int disp_smem = 0;
  KernelRegistry::Instance().Launch("EpDispatchInterNodeV1Kernel_bf16", bn, block_x, disp_smem,
                                    stream, &dargs, args_size);

  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // ── verify dispatchOut ────────────────────────────────────────────────────
  index_t h_total = 0;
  HIP_CHECK(hipMemcpy(&h_total, dargs.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));
  void* d_disp_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_disp_out(h_total * hiddenDim);
  HIP_CHECK(hipMemcpy(h_disp_out.data(), d_disp_out, h_total * hBytes, hipMemcpyDeviceToHost));
  printf("[PE%d][Dispatch] received %lld token(s) from remote:\n", rank, (long long)h_total);
  for (index_t i = 0; i < h_total; ++i) {
    printf("[PE%d]   tok%lld: [", rank, (long long)i);
    for (int d = 0; d < hiddenDim; ++d)
      printf("%s%.0f", d ? "," : "", __bfloat162float(h_disp_out[i * hiddenDim + d]));
    printf("]\n");
  }

  // ── simulate FFN: multiply by 2.0 ────────────────────────────────────────
  std::vector<__hip_bfloat16> h_ffn_out(h_total * hiddenDim);
  for (index_t i = 0; i < h_total * hiddenDim; ++i)
    h_ffn_out[i] = __float2bfloat16(__bfloat162float(h_disp_out[i]) * 2.0f);

  void* d_ffn_out = gpu_alloc_zero(h_total * hBytes);
  HIP_CHECK(hipMemcpy(d_ffn_out, h_ffn_out.data(), h_total * hBytes, hipMemcpyHostToDevice));

  // ── combine ───────────────────────────────────────────────────────────────
  // Combine weights come from the forwarded-during-dispatch buffer, indexed by
  // received token order — NOT by the local token order.
  float* d_recv_weights =
      reinterpret_cast<float*>(handle.shmemDispatchOutWeightsMemObj.cpu->localPtr);

  handle.PrepareInference(HIP_R_16BF, d_ffn_out, nullptr, d_recv_weights, nullptr,
                          reinterpret_cast<int32_t*>(d_indices), numTokens);

  EpDispatchCombineArgsRaw cargs = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  cargs.config.hiddenDim = hiddenDim;

  // shared mem: warpNum * numExpertPerToken * 2 pointers (TokT* + float*)
  int comb_smem = cfg.warpNumPerBlock * cfg.numExpertPerToken * 2 * sizeof(void*);

  // ── helper: dump a bf16 buffer on host ────────────────────────────────────
  auto dump_bf16 = [&](const char* tag, void* d_ptr, int ntok, int dim) {
    std::vector<__hip_bfloat16> h(ntok * dim);
    HIP_CHECK(hipMemcpy(h.data(), d_ptr, ntok * dim * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
    printf("[PE%d]   %-36s (%d tok x dim=%d):\n", rank, tag, ntok, dim);
    for (int t = 0; t < ntok; ++t) {
      printf("[PE%d]     tok%d: [", rank, t);
      for (int d = 0; d < dim; ++d)
        printf("%s%.1f", d ? "," : "", __bfloat162float(h[t * dim + d]));
      printf("]\n");
    }
  };
  auto dump_f32 = [&](const char* tag, void* d_ptr, int ntok, int slots) {
    std::vector<float> h(ntok * slots);
    HIP_CHECK(hipMemcpy(h.data(), d_ptr, ntok * slots * sizeof(float), hipMemcpyDeviceToHost));
    printf("[PE%d]   %-36s (%d tok x %d slots):\n", rank, tag, ntok, slots);
    for (int t = 0; t < ntok; ++t) {
      printf("[PE%d]     tok%d: [", rank, t);
      for (int s = 0; s < slots; ++s) printf("%s%.2f", s ? "," : "", h[t * slots + s]);
      printf("]\n");
    }
  };

  // ── PRE-COMBINE state ─────────────────────────────────────────────────────
  // At this point:
  //   d_ffn_out              = expert 计算完的输出（按 recv 顺序排列，共 h_total 个 token）
  //   d_recv_weights         = dispatch 阶段随 token 一起传过来的权重
  //   combineInp (shmem buf) = 尚未写入，等 EpCombineSync 填充
  printf("\n[PE%d] ──── COMBINE 前状态 ────────────────────────────────────────────\n", rank);
  printf("[PE%d]   inpTokenBuf (FFN输出, 将作为 combineInp 源):  d_ffn_out  %lld tok\n",
         rank, (long long)h_total);
  dump_bf16("d_ffn_out (expert output, recv-order)", d_ffn_out, (int)h_total, hiddenDim);
  printf("[PE%d]   recv weights (dispatch 时随 token 转发的权重):\n", rank);
  dump_f32("d_recv_weights", d_recv_weights, (int)h_total, numEpt);

  // ─────────────────────────────────────────────────────────────────────────
  // C1: EpCombineSync
  //   作用: 把 inpTokenBuf(d_ffn_out) 按 totalRecvTokenNum 个 token
  //         复制进 interNodeV1TokBufs.combineInp（shmem 注册内存），
  //         同时把权重复制进 shmemInpWeightsMemObj。
  //         CombineInterNode 和 CombineIntraNode 都从 combineInp 读数据。
  // ─────────────────────────────────────────────────────────────────────────
  printf("\n[PE%d] ──── C1: EpCombineSync  grid=%d block=%d ────────────────────────\n",
         rank, mp, block_x);
  printf("[PE%d]   将 d_ffn_out → combineInp (shmem), 权重 → shmemInpWeights\n", rank);
  KernelRegistry::Instance().Launch("EpCombineSync_bf16", mp, block_x, 0, stream, &cargs,
                                    args_size);
  HIP_CHECK(hipStreamSynchronize(stream));

  // combineInp 是 shmem 对象，localPtr 可直接读
  void* d_comb_inp = handle.GetShmemCombineInpTokMemObj().cpu->localPtr;
  printf("[PE%d]   [after C1] combineInp (expert输出已拷入 shmem):\n", rank);
  dump_bf16("combineInp", d_comb_inp, (int)h_total, hiddenDim);
  void* d_inp_w = handle.shmemInpWeightsMemObj.cpu->localPtr;
  dump_f32("shmemInpWeights", d_inp_w, (int)h_total, numEpt);

  // ─────────────────────────────────────────────────────────────────────────
  // C2: EpCombineSyncBarrier
  //   作用: 1 个 block，1 个 warp。
  //         把本节点的 crossDeviceBarrierMem 槽写为当前 barrierFlag，
  //         然后自旋等待所有其他节点的对应槽也到达相同值。
  //         确保 combineInp 对所有节点可见后再开始 C3。
  // ─────────────────────────────────────────────────────────────────────────
  printf("\n[PE%d] ──── C2: EpCombineSyncBarrier  grid=1 block=%d ────────────────────\n",
         rank, kWarpSize);
  printf("[PE%d]   跨节点 barrier：等所有节点 combineInp 写完可见\n", rank);
  KernelRegistry::Instance().Launch("EpCombineSyncBarrier_bf16", 1, kWarpSize, 0, stream, &cargs,
                                    args_size);
  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  printf("[PE%d]   [after C2] barrier 完成，所有节点 combineInp 就绪\n", rank);

  // ─────────────────────────────────────────────────────────────────────────
  // C3: EpCombineInterNodeV1Kernel
  //   blockId < rdmaBlockNum  → CombineInterNode:
  //     轮询 chunkFlag（dispatch 阶段写的 RDMA 到达信号）
  //     对每个到达的远端 token，查 interNodeDispDestTokIdMap 找 expert 输出位置
  //     WarpAccum(combineInp[destPe][destLocalTokId], weight) → staging[tokIdx]
  //     最后一个 warp 完成后：清零 chunkFlag，RDMA PUT → 源节点 staging
  //   blockId >= rdmaBlockNum → CombineIntraNode:
  //     对本节点 token 的 intra-node expert slot，查 dispDestTokIdMap
  //     WarpAccum → 本地 staging（nNodes+myNode 区段）
  // ─────────────────────────────────────────────────────────────────────────
  printf("\n[PE%d] ──── C3: EpCombineInterNodeV1Kernel  grid=%d rdmaBlocks=%d smem=%d ────\n",
         rank, bn, rdmaBlockNum, comb_smem);
  printf("[PE%d]   RDMA blocks: 轮询 chunkFlag → WarpAccum → staging → RDMA PUT 回源\n", rank);
  printf("[PE%d]   XGMI blocks: 本地 expert slot → WarpAccum → 本地 staging\n", rank);
  KernelRegistry::Instance().Launch("EpCombineInterNodeV1Kernel_bf16", bn, block_x, comb_smem,
                                    stream, &cargs, args_size);
  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // staging 里现在有两部分:
  //   staging[SendBufSlotOffset(remoteNode, j)]  = 远端 token 的 WarpAccum 结果（已 RDMA 发走）
  //   staging[SendBufSlotOffset(myNode+nNodes,j)] = 本节点 intra expert 的聚合结果 &
  //                                                 来自其他节点 RDMA PUT 回来的聚合结果
  // EpCombineAll 会从所有节点的 staging 区段读取后做最终求和
  void* d_staging = handle.interNodeV1TokBufs.staging->cpu->localPtr;
  // staging 里本节点的"combine 回程接收区"：SendBufSlotOffset(myNode+nNodes, 0)
  // = (myNode + nNodes) * MaxNumTokensToSendPerRank * tokCombXferBytes
  // 这里 tokCombXferBytes = hiddenBytes（无 weights buf 时）
  // 用 maxNumInpTokenPerRank 做 stride
  int maxSend = cfg.maxNumInpTokenPerRank;
  int nNodes_val = 2;  // worldSize / gpuPerNode
  size_t comb_xfer = hiddenDim * sizeof(__hip_bfloat16);  // tokCombXferBytes (no weights here)
  // slot for myNode's combine-recv area (where remote sends back results):
  size_t recv_area_offset = (size_t)(rank + nNodes_val) * maxSend * comb_xfer;
  printf("[PE%d]   [after C3] staging combine-recv 区 (index=myNode+nNodes=%d, offset=%zuB):\n",
         rank, rank + nNodes_val, recv_area_offset);
  // dump the slot for our single local token (slot 0 in the recv area)
  std::vector<__hip_bfloat16> h_staging_recv(hiddenDim);
  HIP_CHECK(hipMemcpy(h_staging_recv.data(),
                      reinterpret_cast<uint8_t*>(d_staging) + recv_area_offset,
                      hiddenDim * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
  printf("[PE%d]     slot0: [", rank);
  for (int d = 0; d < hiddenDim; ++d)
    printf("%s%.1f", d ? "," : "", __bfloat162float(h_staging_recv[d]));
  printf("]\n");
  printf("[PE%d]   (这是本节点 token 经远端 expert 计算后，通过 RDMA 送回来的聚合结果)\n", rank);

  // ─────────────────────────────────────────────────────────────────────────
  // C4: EpCombineAll
  //   对本节点每个 local token，遍历所有节点的 staging 区段：
  //     node == myNode → staging[myNode]（intra expert 贡献）
  //     node != myNode → staging[myNode+nNodes]（RDMA 回程区，远端 expert 贡献）
  //   WarpAccum 最终加权求和 → combineOut
  // ─────────────────────────────────────────────────────────────────────────
  printf("\n[PE%d] ──── C4: EpCombineAll  grid=%d block=%d smem=%d ────────────────────\n",
         rank, mp, block_x, comb_smem);
  printf("[PE%d]   从各节点 staging 读聚合结果，WarpAccum → combineOut\n", rank);
  KernelRegistry::Instance().Launch("EpCombineAll_bf16", mp, block_x, comb_smem, stream, &cargs,
                                    args_size);
  HIP_CHECK(hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // ── verify combineOut ─────────────────────────────────────────────────────
  // Expected: FFN output * weight = (input * 2.0) * 1.0
  // PE0 sent [10,10,10,10] → PE1 ran FFN → [20,20,20,20] → sent back → combineOut=[20,20,20,20]
  // PE1 sent [20,20,20,20] → PE0 ran FFN → [40,40,40,40] → sent back → combineOut=[40,40,40,40]
  void* d_comb_out = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
  std::vector<__hip_bfloat16> h_comb(numTokens * hiddenDim);
  HIP_CHECK(hipMemcpy(h_comb.data(), d_comb_out, numTokens * hBytes, hipMemcpyDeviceToHost));

  float expected = val * 2.0f;  // input * FFN-scale
  printf("[PE%d][Combine] combineOut (%d token):\n", rank, numTokens);
  bool ok = true;
  for (int t = 0; t < numTokens; ++t) {
    printf("[PE%d]   tok%d: [", rank, t);
    for (int d = 0; d < hiddenDim; ++d) {
      float got = __bfloat162float(h_comb[t * hiddenDim + d]);
      printf("%s%.1f", d ? "," : "", got);
      if (std::abs(got - expected) > 1.0f) ok = false;
    }
    printf("]  (expected %.1f)\n", expected);
  }
  printf("[PE%d] result: %s\n", rank, ok ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_indices));
  HIP_CHECK(hipFree(d_weights));
  HIP_CHECK(hipFree(d_ffn_out));
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

  load_kernels();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  run_internode_v1(rank, world, stream);
  MPI_Barrier(MPI_COMM_WORLD);

  HIP_CHECK(hipStreamDestroy(stream));
  ShmemFinalize();
  MPI_Finalize();
  return 0;
}
