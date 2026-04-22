// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License — see other MORI examples for the full license text.
//
// ============================================================================
// CU XGMI Load Micro-Benchmark
// ============================================================================
// Purpose: measure the actual bandwidth at which a compute kernel can read
// data from 7 peer GPUs' HBM via P2P/XGMI pointers simultaneously.
//
// This decides whether the "CU-based AG" phase (plan A or plan B of
// hybrid SDMA-scatter + CU-AR AR kernels) is viable:
//   - Plan A (SDMA scatter + CU reduce + CU XGMI-load AG + direct write):
//       CU phase 2 reads 224 MB from 7 peers, writes 256 MB local.
//       Target: ≥ 400 GB/s XGMI read. If < 200 GB/s → plan A dead.
//   - Plan B (full CU reduce+AG via XGMI, SDMA copy at end):
//       CU phase 1 reads same 224 MB from 7 peers.
//
// Design: each compute block is statically assigned 1 peer. The block
// reads a chunk of peer_obj's memory via the P2P pointer p2pPeerPtrs[peer]
// and writes to a local destination buffer. Vectorized via ulonglong2
// (16 B loads).
// ============================================================================

#include <mpi.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::shmem;
using namespace mori::application;

#define CHECK_HIP(call)                                                     \
  do {                                                                      \
    hipError_t err = (call);                                                \
    if (err != hipSuccess) {                                                \
      fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err));                                      \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

// ============================================================================
// Kernel: each block is bound to 1 peer. All blocks per peer split the
// peer's shard and concurrently read from p2pPeerPtrs[peer]. Writes go to
// local destBuf[peer * bytesPerPeer ..].
// ============================================================================
__global__ void CuXgmiLoadKernel(const SymmMemObjPtr srcObj, void* dstBuf,
                                 size_t bytesPerPeer, int myPe, int npes,
                                 int blocksPerPeer) {
  const int blockId = static_cast<int>(blockIdx.x);
  const int peer = blockId / blocksPerPeer;
  const int posInPeer = blockId % blocksPerPeer;

  if (peer >= npes || peer == myPe) return;

  // src = peer's base pointer, obtained via the P2P route
  const uint8_t* srcBase =
      reinterpret_cast<const uint8_t*>(srcObj->peerPtrs[peer]);
  uint8_t* dstBase = reinterpret_cast<uint8_t*>(dstBuf)
                   + static_cast<size_t>(peer) * bytesPerPeer;

  using vec_t = ulonglong2;  // 16 B
  const size_t totalVecs = bytesPerPeer / sizeof(vec_t);
  const size_t vecsPerBlock =
      (totalVecs + blocksPerPeer - 1) / blocksPerPeer;
  const size_t vStart = static_cast<size_t>(posInPeer) * vecsPerBlock;
  const size_t vEnd =
      (vStart + vecsPerBlock > totalVecs) ? totalVecs : (vStart + vecsPerBlock);

  const vec_t* src = reinterpret_cast<const vec_t*>(srcBase);
  vec_t* dst = reinterpret_cast<vec_t*>(dstBase);

  for (size_t v = vStart + threadIdx.x; v < vEnd; v += blockDim.x) {
    dst[v] = src[v];
  }
}

static void runOneSize(size_t bytesPerPeer, int myPe, int npes, int warmup,
                       int iterations, hipStream_t stream,
                       int blocksPerPeer) {
  // Allocate peer-visible source buffer (each PE puts some data here, other
  // PEs will read from it). Each PE's srcBuf is npes * bytesPerPeer total,
  // but we only care about one shard per peer for this bench.
  const size_t srcBufBytes = static_cast<size_t>(npes) * bytesPerPeer;
  void* srcBuf = ShmemMalloc(srcBufBytes);
  assert(srcBuf);
  CHECK_HIP(hipMemset(srcBuf, 0xAB, srcBufBytes));
  SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuf);
  assert(srcObj.IsValid());

  // Local destination buffer (where we accumulate peers' reads).
  void* dstBuf = nullptr;
  CHECK_HIP(hipMalloc(&dstBuf, srcBufBytes));
  CHECK_HIP(hipMemset(dstBuf, 0, srcBufBytes));
  CHECK_HIP(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  const int threads = 256;
  const int blocks = npes * blocksPerPeer;

  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    CuXgmiLoadKernel<<<blocks, threads, 0, stream>>>(
        srcObj, dstBuf, bytesPerPeer, myPe, npes, blocksPerPeer);
    CHECK_HIP(hipStreamSynchronize(stream));
    double t1 = MPI_Wtime();
    if (i >= warmup) times.push_back(t1 - t0);
  }

  double avg = 0;
  for (double t : times) avg += t;
  avg /= times.size();

  // Effective XGMI read bytes per PE: 7 (peers) * bytesPerPeer
  const double readBytes =
      static_cast<double>(npes - 1) * static_cast<double>(bytesPerPeer);
  const double gbs = readBytes / avg / (1024.0 * 1024.0 * 1024.0);

  if (myPe == 0) {
    printf("%10.1f  %8d  %12.2f  %12.3f\n",
           bytesPerPeer / (1024.0 * 1024.0),
           blocksPerPeer,
           gbs,
           avg * 1000.0);
  }

  CHECK_HIP(hipFree(dstBuf));
  ShmemFree(srcBuf);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &localComm);
  int localRank;
  MPI_Comm_rank(localComm, &localRank);
  int deviceCount;
  CHECK_HIP(hipGetDeviceCount(&deviceCount));
  CHECK_HIP(hipSetDevice(localRank % deviceCount));

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  if (myPe == 0) {
    printf("=== CU XGMI Load Bench (each block reads from 1 peer) ===\n");
    printf("Peers (npes): %d (reading from npes-1 = %d peers concurrently)\n",
           npes, npes - 1);
    printf("\n");
    printf(" BytesPerPeer  Blocks/  XGMI_Read_BW     avg_time\n");
    printf("     (MB)        peer    (GB/s agg)      (ms)\n");
    printf("%s\n", std::string(55, '-').c_str());
  }

  // Cover the sizes AR uses (per-peer shard ~32 MB at 256 MB total).
  std::vector<size_t> sizes = {
      1  * 1024 * 1024,
      4  * 1024 * 1024,
      16 * 1024 * 1024,
      32 * 1024 * 1024,   // ← AR[N] per-peer shard at 256 MB total
      64 * 1024 * 1024,
  };

  // Sweep number of blocks per peer to find saturation point
  std::vector<int> blocksPerPeerList = {1, 4, 8, 16, 32};

  const int warmup = 5;
  const int iterations = 20;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  for (size_t bytesPerPeer : sizes) {
    if (myPe == 0) {
      printf("\n-- per-peer bytes = %.0f MB --\n",
             bytesPerPeer / (1024.0 * 1024.0));
    }
    for (int bpp : blocksPerPeerList) {
      runOneSize(bytesPerPeer, myPe, npes, warmup, iterations, stream, bpp);
    }
  }

  CHECK_HIP(hipStreamDestroy(stream));
  MPI_Comm_free(&localComm);
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
  MPI_Finalize();
  return 0;
}
