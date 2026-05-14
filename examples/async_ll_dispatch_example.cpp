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
// AsyncLL dispatch example — 5 kernel launches 显式展示
//
// Topology:
//   npes=2, numQpPerPe=2, numEpt=2, numTokens=3, hiddenDim=64
//   4 experts total: expert 0,1 → PE0 | expert 2,3 → PE1
//
// dispatch_send (3 kernels):
//   K1: EpDispatchLowLatencyAsyncSendCopySlotAssign   → slot 分配 + dedup
//   K2: EpDispatchLowLatencyAsyncSendCopyMultiBlock   → staging 数据拷贝
//   K3: EpDispatchLowLatencyAsyncSendTransfer          → IBGDA non-blocking put
//
// dispatch_recv (2 kernels):
//   K4: EpDispatchLowLatencyAsyncRecvTransfer          → quiet + 信号 + 轮询
//   K5: EpDispatchLowLatencyAsyncRecvCopyMultiBlock    → prefix sum + 拷到 dispatchOut
//
// combine_send (2 kernels):
//   C1: EpCombineLowLatencyAsyncSendCopy              → FFN output 拷到 staging (用
//   dispReceiverIdxMap 还原 slot) C2: EpCombineLowLatencyAsyncSendTransfer          → IBGDA
//   non-blocking put 回 src PE
//
// combine_recv (2 kernels):
//   C3: EpCombineLowLatencyAsyncRecvTransfer          → quiet + poll 信号
//   C4: EpCombineLowLatencyAsyncRecvCopy              → 加权求和写 combineOut
//
// Token routing (each rank sends numEpt=2 experts per token):
//   token0 → expert[0(PE0), 2(PE1)]  ← cross-PE
//   token1 → expert[1(PE0), 1(PE0)]  ← same PE, dedup → 只发1次
//   token2 → expert[2(PE1), 3(PE1)]  ← cross-PE
//
// Launch: mpirun -np 2 ./async_ll_dispatch_example

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

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // ── 0. MPI + shmem init ────────────────────────────────────────────────
  MPI_Init(&argc, &argv);

  // Set device BEFORE ShmemMpiInit: the init path calls hipGetDevice()
  // internally for NIC-GPU topology matching; all ranks must already be on
  // their own GPU or the matcher returns the same NIC for everyone.
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  HIP_CHECK(hipSetDevice(mpi_rank));

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  const int rank = ShmemMyPe();
  const int world = ShmemNPes();
  assert(world == 2);

  // ── 1. Config ──────────────────────────────────────────────────────────
  const int numEpt = 2;  // experts per token
  const int hiddenDim = 4;
  const int numTokens = 3;   // this rank's tokens
  const int kWarpSize = 64;  // AMD GPU

  EpDispatchCombineConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = world;  // 2
  cfg.hiddenDim = hiddenDim;
  cfg.numExpertPerRank = 2;  // 4 experts / 2 PE
  cfg.numExpertPerToken = numEpt;
  cfg.maxNumInpTokenPerRank = 128;
  cfg.numQpPerPe = 2;
  cfg.gpuPerNode = 2;
  cfg.kernelType = KernelType::AsyncLL;
  cfg.warpNumPerBlock = 4;
  cfg.blockNum = world;
  cfg.enableSdma = false;

  // ── 2. Handle (allocs symmetric buffers: staging/dispatchInp/dispatchOut)
  // Scoped so handle destructs (calls ShmemFree) before ShmemFinalize below.
  {
    EpDispatchCombineHandle handle(cfg);

    // ── 3. Device tensors ──────────────────────────────────────────────────
    // expert → PE: expert/numExpertPerRank
    //   expert 0,1 → PE0 | expert 2,3 → PE1
    //
    // token0 → expert[0(PE0), 2(PE1)]  ← cross-PE
    // token1 → expert[1(PE0), 1(PE0)]  ← 同一PE，dedup，只发1次
    // token2 → expert[2(PE1), 3(PE1)]  ← 均去PE1
    const size_t hBytes = hiddenDim * sizeof(__hip_bfloat16);

    // Fill each token with a fingerprint: token t on PE r → all elements = r*10+t
    // e.g. PE1 token2 → 12.0, so we can verify which PE/token arrived after dispatch.
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

    void* d_input = nullptr;
    HIP_CHECK(hipMalloc(&d_input, numTokens * hBytes));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), numTokens * hBytes, hipMemcpyHostToDevice));

    // token buff layout: |token bytes = dtype * hidden|indices|weights|sclaes|srcTokenId|
    int32_t h_idx[6] = {0, 2, 1, 1, 2, 3};
    float h_wgt[6] = {0.6f, 0.4f, 0.5f, 0.5f, 0.7f, 0.3f};
    void* d_indices = gpu_alloc_zero(numTokens * numEpt * sizeof(int32_t));
    void* d_weights = gpu_alloc_zero(numTokens * numEpt * sizeof(float));
    HIP_CHECK(hipMemcpy(d_indices, h_idx, sizeof(h_idx), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_wgt, sizeof(h_wgt), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ── 4. Load .hsaco (AOT kernels) ──────────────────────────────────────
    // AutoLoad()'s get_self_lib_dir() can fail in containers (dladdr returns
    // empty). Fall back: resolve exe path via /proc/self/exe and look for
    // kernels in <exe_dir>/../lib/ (build layout: examples/ → lib/).
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

    // ── 5. Prepare args (告诉 handle 本轮的 input/indices/token数) ─────────
    handle.PrepareInference(HIP_R_16BF, d_input,
                            /*output=*/nullptr, reinterpret_cast<float*>(d_weights),
                            reinterpret_cast<int32_t*>(d_indices), numTokens);

    EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, 0);
    args.config.hiddenDim = hiddenDim;
    size_t args_size = sizeof(EpDispatchCombineArgsRaw);

    // GPU 的 SM 数量决定 grid size
    int mp = handle.multiProcessorCount;
    int mp_aligned = (mp / world) * world;  // 对齐到 world_size

    const std::string sfx = "bf16";  // 对应 dtype suffix

    // ──────────────────────────────────────────────────────────────────────
    //  dispatch_send : K1 + K2 + K3
    // ──────────────────────────────────────────────────────────────────────

    // K1: SlotAssign
    //   grid  = mp_aligned    (尽量多 block，充分利用 SM)
    //   block = 64*16 = 1024  (16 warp/block)
    //   做什么: 每个 warp 处理 warpSize/numEpt = 32 个 token
    //           用 __shfl dedup 检查同一 token 的多个 expert 是否去同一 PE
    //           不重复: atomicAdd(destPeTokenCounter[destPe]) 分配 slot
    //           重复:   dispDestTokIdMap[i] = NullSendBufSlotOffset
    printf("[PE%d] K1: SlotAssign  grid=%d block=%d\n", rank, mp_aligned, kWarpSize * 16);
    KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendCopySlotAssign_" + sfx,
                                      /*grid_x=*/mp_aligned,
                                      /*block_x=*/kWarpSize * 16,
                                      /*smem=*/0, stream, &args, args_size);

    // K2: SendCopyMultiBlock
    //   grid  = mp_aligned
    //   block = 1024
    //   做什么: 读 dispDestTokIdMap 拿 pre-computed slot
    //           跳过 NullSendBufSlotOffset (dedup 掉的)
    //           warpsPerToken = ceil(globalWarpNum / totalEntries) 个 warp 合作搬一个 token
    //           hidden bytes 分段拷: WarpCopy(staging+slot*xferBytes, inpTokenBuf+tok*hiddenBytes)
    //           inTokenPartId==0 的 warp 额外拷 indices/weights/srcTokId
    printf("[PE%d] K2: SendCopyMultiBlock  grid=%d block=%d\n", rank, mp_aligned, kWarpSize * 16);
    KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendCopyMultiBlock_" + sfx,
                                      /*grid_x=*/mp_aligned,
                                      /*block_x=*/kWarpSize * 16,
                                      /*smem=*/0, stream, &args, args_size);

    // K3: SendTransfer
    //   grid  = world_size = 2  (一个 block 负责一个 destPe)
    //   block = kWarpSize * warpNumPerBlock = 64*4 = 256
    //   做什么: blockId → destPe, warpId → qpId, laneId==0 执行
    //           读 destPeTokenCounter[destPe] 知道发多少 token
    //           按 QP 均分: tokenChunkNum = ceil(tokenNum / numQpPerPe)
    //           ShmemPutMemNbiThread(peer.dispatchInp, staging, size, destPe, qpId)
    //           Non-blocking! 数据可能仍在传输中
    //           destPe==myPe 跳过 (自己的数据留在 staging，RecvCopy 直接读)
    printf("[PE%d] K3: SendTransfer  grid=%d block=%d\n", rank, world,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncSendTransfer_" + sfx,
                                      /*grid_x=*/world,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/0, stream, &args, args_size);

    // ── 6. 计算通信 overlap ────────────────────────────────────────────────
    // K3 提交后 RDMA put 在后台传输，这里可以跑本地 FFN
    // e.g. local_ffn<<<grid, block, 0, stream>>>(local_input, ...);
    printf("[PE%d] (RDMA in-flight, local FFN overlap here)\n", rank);

    // ──────────────────────────────────────────────────────────────────────
    //  dispatch_recv : K4 + K5
    // ──────────────────────────────────────────────────────────────────────

    // K4: RecvTransfer
    //   grid  = world_size = 2
    //   block = 64*4 = 256
    //   做什么: blockId → srcPe, warpId → qpId, laneId==0 执行
    //           ShmemQuietThread(srcPe, qpId)    ← drain QP，确保 put 落地
    //           ShmemPutUint64Imm(recvTokenNumMemObj,
    //                             myPe*numQpPerPe+qpId,
    //                             tokenNum+1, srcPe, qpId)  ← 发信号
    //           然后轮询: ShmemUint64WaitUntilGreaterThan(
    //                         recvTokenNums[srcPe*numQpPerPe+laneId], 0)
    //           所有 QP 信号到达后退出
    printf("[PE%d] K4: RecvTransfer  grid=%d block=%d\n", rank, world,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncRecvTransfer_" + sfx,
                                      /*grid_x=*/world,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/0, stream, &args, args_size);

    // K5: RecvCopyMultiBlock
    //   grid  = mp_aligned  (blocks 静态分给各 PE: blocksPerPe = mp_aligned/world)
    //   block = 1024
    //   做什么:
    //   Step A: warp-shuffle prefix sum (log2(world)=1 轮)
    //     lane i 读 recvTokenNums[i*numQpPerPe] - 1 → 各 PE 收到的 token 数
    //     inclusive scan → 累加和
    //     exclusive → 各 PE 在 dispatchOut 里的起始 offset
    //   Step B: __shfl 取本 block 对应 PE 的 (peOffset, recvTokenNum, totalTokens)
    //   Step C: 多 warp 协作拷贝
    //     destTokId = peOffset + tokenId
    //     WarpCopy(dispatchOut[destTokId], dispatchInp/staging[tokenId])
    //     inTokenPartId==0: 拷 indices/weights 到 shmemOutIndices/Weights
    //     laneId==0:        写 dispReceiverIdxMap[destTokId] (combine 阶段用)
    //   Step D: globalWarpId==0 写 totalRecvTokenNum, 清 counter, 更新 barrier flag
    printf("[PE%d] K5: RecvCopyMultiBlock  grid=%d block=%d\n", rank, mp_aligned, kWarpSize * 16);
    KernelRegistry::Instance().Launch("EpDispatchLowLatencyAsyncRecvCopyMultiBlock_" + sfx,
                                      /*grid_x=*/mp_aligned,
                                      /*block_x=*/kWarpSize * 16,
                                      /*smem=*/0, stream, &args, args_size);

    // ── 7. Sync ────────────────────────────────────────────────────────────
    HIP_CHECK(hipStreamSynchronize(stream));

    // ── 8. Verify: dump dispatchOut ────────────────────────────────────────
    // dispatchOut layout: [PE0 tokens | PE1 tokens]
    // Each token's first element encodes the sender: r*10+t (e.g. 12 = PE1 token2)
    //
    // Expected per PE (2 sender PEs × 3 tokens each, fingerprint = senderRank*10+tokId):
    //   PE0 receives:
    //     expert0 ← token0 from PE0,PE1           → fingerprints 0, 10
    //     expert1 ← token1 dedup from PE0,PE1     → fingerprints 1, 11
    //   PE1 receives:
    //     expert2 ← token0+token2 from PE0,PE1    → fingerprints 0,10,2,12
    //     expert3 ← token2 from PE0,PE1           → fingerprints 2, 12

    // Read totalRecvTokenNum from device (written by K5 globalWarpId==0)
    index_t h_total = 0;
    HIP_CHECK(hipMemcpy(&h_total, args.totalRecvTokenNum, sizeof(index_t), hipMemcpyDeviceToHost));

    // Copy dispatchOut back to host (device ptr = dispatchOut.cpu->localPtr)
    void* d_out = handle.GetShmemDispatchOutTokMemObj().cpu->localPtr;
    size_t outBytes = h_total * hBytes;
    std::vector<__hip_bfloat16> h_out(h_total * hiddenDim);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, outBytes, hipMemcpyDeviceToHost));

    printf("[PE%d] dispatchOut (%lld tokens):\n", rank, (long long)h_total);
    for (index_t i = 0; i < h_total; ++i) {
      printf("[PE%d]   tok%lld: [", rank, (long long)i);
      for (int d = 0; d < hiddenDim; ++d)
        printf("%s%.0f", d ? "," : "", __bfloat162float(h_out[i * hiddenDim + d]));
      printf("]\n");
    }
    printf("[PE%d] dispatch done\n", rank);

    // ──────────────────────────────────────────────────────────────────────
    //  simulate FFN: scale each received token by 2.0 (in-place on dispatchOut)
    //  In a real MoE, FFN would write results to a separate buffer; here we
    //  reuse dispatchOut as the "FFN output" to keep the example self-contained.
    // ──────────────────────────────────────────────────────────────────────
    std::vector<__hip_bfloat16> h_ffn_out(h_total * hiddenDim);
    for (index_t i = 0; i < h_total * hiddenDim; ++i)
      h_ffn_out[i] = __float2bfloat16(__bfloat162float(h_out[i]) * 2.0f);
    void* d_ffn_out = nullptr;
    HIP_CHECK(hipMalloc(&d_ffn_out, h_total * hBytes));
    HIP_CHECK(hipMemcpy(d_ffn_out, h_ffn_out.data(), h_total * hBytes, hipMemcpyHostToDevice));

    // ── 9. combine_send : C1 + C2 ─────────────────────────────────────────
    // combineOut lives in shmem (interNodeTokBufs.combineOut), allocated by handle.
    // PrepareInference: input = FFN output; output = nullptr (kernel writes shmem directly)
    handle.PrepareInference(HIP_R_16BF, d_ffn_out,
                            /*output=*/nullptr, reinterpret_cast<float*>(d_weights),
                            reinterpret_cast<int32_t*>(d_indices), numTokens);

    // Rebuild args (inpTokenBuf now points to d_ffn_out)
    EpDispatchCombineArgsRaw cargs = GetEpDispatchCombineArgsRaw(handle, 0);
    cargs.config.hiddenDim = hiddenDim;

    // combine shared mem: warpNumPerBlock * numExpertPerToken * (8+8)
    int combine_smem = cfg.warpNumPerBlock * cfg.numExpertPerToken * 16;

    // C1: CombineSendCopy
    //   grid  = mp_aligned
    //   block = kWarpSize * warpNumPerBlock
    //   做什么: 对每个收到的token(totalRecvTokenNum个)
    //           dispReceiverIdxMap[tokenId] → staging slot offset
    //           WarpCopy(staging[slot], inpTokenBuf[tokenId])
    printf("[PE%d] C1: CombineSendCopy  grid=%d block=%d\n", rank, mp_aligned,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncSendCopy_" + sfx,
                                      /*grid_x=*/mp_aligned,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/0, stream, &cargs, args_size);

    // C2: CombineSendTransfer
    //   grid  = world_size
    //   block = kWarpSize * warpNumPerBlock
    //   做什么: 读 recvTokenNumMemObj 知道每个destPe/qpId发了多少token
    //           ShmemPutMemNbiThread(peer.combineInp, staging, size, destPe, qpId)
    //           发回 src PE 的 combineInp symmetric buffer
    printf("[PE%d] C2: CombineSendTransfer  grid=%d block=%d\n", rank, world,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncSendTransfer_" + sfx,
                                      /*grid_x=*/world,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/0, stream, &cargs, args_size);

    // ── 10. combine_recv : C3 + C4 ────────────────────────────────────────
    // Rebuild args for recv (same handle state, just a fresh raw args snapshot)
    EpDispatchCombineArgsRaw crargs = GetEpDispatchCombineArgsRaw(handle, 0);
    crargs.config.hiddenDim = hiddenDim;

    // C3: CombineRecvTransfer
    //   grid  = world_size
    //   block = kWarpSize * warpNumPerBlock
    //   做什么: 对每个 srcPe/qpId: quiet → 发信号 → poll 对端信号
    printf("[PE%d] C3: CombineRecvTransfer  grid=%d block=%d\n", rank, world,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncRecvTransfer_" + sfx,
                                      /*grid_x=*/world,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/0, stream, &crargs, args_size);

    // C4: CombineRecvCopy
    //   grid  = mp_aligned
    //   block = kWarpSize * warpNumPerBlock
    //   做什么: 对每个原始token(curRankNumToken个)
    //           遍历numExpertPerToken个expert entry
    //           dispDestTokIdMap[entryId] → staging slot (跳过 null/dedup)
    //           weighted sum: combineOut[tokId] += weight * staging[slot]
    printf("[PE%d] C4: CombineRecvCopy  grid=%d block=%d\n", rank, mp_aligned,
           kWarpSize * cfg.warpNumPerBlock);
    KernelRegistry::Instance().Launch("EpCombineLowLatencyAsyncRecvCopy_" + sfx,
                                      /*grid_x=*/mp_aligned,
                                      /*block_x=*/kWarpSize * cfg.warpNumPerBlock,
                                      /*smem=*/combine_smem, stream, &crargs, args_size);

    HIP_CHECK(hipStreamSynchronize(stream));

    // ── 11. Verify combineOut ──────────────────────────────────────────────
    // combineOut lives in handle's shmem buffer (interNodeTokBufs.combineOut).
    // Expected: combineOut[tok][0] = sum_k(weight_k * FFN(dispatch_result))
    //   FFN doubles value, so = sum_k(weight_k * 2 * (rank*10+tok))
    //   token0: (0.6+0.4)*2*(rank*10+0) = 2*(rank*10)
    //   token1: 0.5*2*(rank*10+1)  [dedup: only one expert slot valid]
    //   token2: (0.7+0.3)*2*(rank*10+2) = 2*(rank*10+2)
    void* d_comb_shmem = handle.GetShmemCombineOutTokMemObj().cpu->localPtr;
    std::vector<__hip_bfloat16> h_comb(numTokens * hiddenDim);
    HIP_CHECK(hipMemcpy(h_comb.data(), d_comb_shmem, numTokens * hBytes, hipMemcpyDeviceToHost));

    printf("[PE%d] combineOut (%d tokens):\n", rank, numTokens);
    for (int t = 0; t < numTokens; ++t) {
      printf("[PE%d]   tok%d: [", rank, t);
      for (int d = 0; d < hiddenDim; ++d)
        printf("%s%.1f", d ? "," : "", __bfloat162float(h_comb[t * hiddenDim + d]));
      printf("]\n");
    }
    printf("[PE%d] combine done\n", rank);

    // ── 12. Cleanup ────────────────────────────────────────────────────────
    HIP_CHECK(hipFree(d_ffn_out));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipStreamDestroy(stream));

  }  // handle destructs here → ShmemFree called before ShmemFinalize

  // Barrier before finalize: ensures all ranks finish before any rank starts
  // tearing down IPC handles, avoiding hipIpcCloseMemHandle invalid-argument hang.
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();  // internally calls MPI_Finalize via MpiBootstrapNetwork::Finalize
  return 0;
}
