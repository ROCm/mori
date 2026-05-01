// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License — see other MORI examples for the full license text.
//
// ============================================================================
// Multi-ring XGMI lane micro-benchmark
// ============================================================================
// Measures aggregate peer-read bandwidth for multiple logical ring lanes.
// Each lane reads a disjoint slice from one peer and writes it locally. Lanes can
// be split between forward and reverse directions to estimate whether a planned
// multi-ring allreduce can fill bidirectional XGMI bandwidth.

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::application;
using namespace mori::shmem;

#define CHECK_HIP(call)                                                     \
  do {                                                                      \
    hipError_t err = (call);                                                \
    if (err != hipSuccess) {                                                \
      fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err));                                      \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

struct LaneSpec {
  int forward;
  int reverse;
};

static LaneSpec ParseLaneSpec(const char* s) {
  LaneSpec spec{0, 0};
  if (s == nullptr || s[0] == '\0') {
    spec.forward = 3;
    spec.reverse = 3;
    return spec;
  }
  int f = 0;
  int r = 0;
  if (std::sscanf(s, "%dF%dR", &f, &r) == 2 ||
      std::sscanf(s, "%df%dr", &f, &r) == 2) {
    spec.forward = f;
    spec.reverse = r;
    return spec;
  }
  if (std::sscanf(s, "%d", &f) == 1) {
    spec.forward = f;
    spec.reverse = 0;
    return spec;
  }
  spec.forward = 3;
  spec.reverse = 3;
  return spec;
}

__global__ void MultiRingXgmiLaneKernel(const SymmMemObjPtr srcObj, void* dstBuf,
                                        size_t bytesPerLane, int myPe, int npes,
                                        int forwardLanes, int reverseLanes,
                                        int blocksPerLane) {
  const int laneCount = forwardLanes + reverseLanes;
  const int blockId = static_cast<int>(blockIdx.x);
  const int lane = blockId / blocksPerLane;
  const int posInLane = blockId - lane * blocksPerLane;
  if (lane >= laneCount || bytesPerLane == 0) return;

  int peer = -1;
  if (lane < forwardLanes) {
    const int hop = 1 + (lane % (npes - 1));
    peer = (myPe + hop) % npes;
  } else {
    const int reverseLane = lane - forwardLanes;
    const int hop = 1 + (reverseLane % (npes - 1));
    peer = (myPe - hop + npes) % npes;
  }
  if (peer == myPe) return;

  using vec_t = ulonglong2;
  const size_t totalVecs = bytesPerLane / sizeof(vec_t);
  const size_t vecsPerBlock = (totalVecs + blocksPerLane - 1) / blocksPerLane;
  const size_t vStart = static_cast<size_t>(posInLane) * vecsPerBlock;
  const size_t vEnd =
      (vStart + vecsPerBlock > totalVecs) ? totalVecs : (vStart + vecsPerBlock);

  const uint8_t* srcBase =
      reinterpret_cast<const uint8_t*>(srcObj->peerPtrs[peer])
      + static_cast<size_t>(lane) * bytesPerLane;
  uint8_t* dstBase =
      reinterpret_cast<uint8_t*>(dstBuf) + static_cast<size_t>(lane) * bytesPerLane;
  const vec_t* src = reinterpret_cast<const vec_t*>(srcBase);
  vec_t* dst = reinterpret_cast<vec_t*>(dstBase);

  for (size_t v = vStart + threadIdx.x; v < vEnd; v += blockDim.x) {
    dst[v] = src[v];
  }
}

static void RunOne(size_t bytesPerLane, LaneSpec spec, int blocksPerLane,
                   int myPe, int npes, int warmup, int iterations,
                   hipStream_t stream) {
  const int laneCount = spec.forward + spec.reverse;
  const size_t totalBytes = bytesPerLane * static_cast<size_t>(laneCount);
  void* srcBuf = ShmemMalloc(totalBytes);
  assert(srcBuf);
  CHECK_HIP(hipMemset(srcBuf, 0xCD, totalBytes));
  SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuf);
  assert(srcObj.IsValid());

  void* dstBuf = nullptr;
  CHECK_HIP(hipMalloc(&dstBuf, totalBytes));
  CHECK_HIP(hipMemset(dstBuf, 0, totalBytes));
  CHECK_HIP(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  const int threads = 256;
  const int blocks = laneCount * blocksPerLane;
  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    MultiRingXgmiLaneKernel<<<blocks, threads, 0, stream>>>(
        srcObj, dstBuf, bytesPerLane, myPe, npes, spec.forward, spec.reverse,
        blocksPerLane);
    CHECK_HIP(hipStreamSynchronize(stream));
    const double t1 = MPI_Wtime();
    if (i >= warmup) times.push_back(t1 - t0);
  }

  double avg = 0.0;
  for (double t : times) avg += t;
  avg /= static_cast<double>(times.size());

  const double readBytes = static_cast<double>(totalBytes);
  const double gbs = readBytes / avg / (1024.0 * 1024.0 * 1024.0);
  if (myPe == 0) {
    printf("%8zu %4dF/%-4dR %8d %12.2f %12.3f\n",
           bytesPerLane / (1024 * 1024), spec.forward, spec.reverse,
           blocksPerLane, gbs, avg * 1000.0);
  }

  CHECK_HIP(hipFree(dstBuf));
  ShmemFree(srcBuf);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &localComm);
  int localRank = 0;
  MPI_Comm_rank(localComm, &localRank);
  int deviceCount = 0;
  CHECK_HIP(hipGetDeviceCount(&deviceCount));
  CHECK_HIP(hipSetDevice(localRank % deviceCount));
  MPI_Comm_free(&localComm);

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(status == 0);
  const int myPe = ShmemMyPe();
  const int npes = ShmemNPes();

  const int warmup = std::getenv("MORI_MR_WARMUP") ? std::atoi(std::getenv("MORI_MR_WARMUP")) : 5;
  const int iterations = std::getenv("MORI_MR_ITERS") ? std::atoi(std::getenv("MORI_MR_ITERS")) : 20;
  const int blocksPerLane =
      std::getenv("MORI_MR_BLOCKS_PER_LANE") ? std::atoi(std::getenv("MORI_MR_BLOCKS_PER_LANE")) : 8;
  const size_t bytesPerLane =
      (std::getenv("MORI_MR_MB_PER_LANE") ? std::strtoull(std::getenv("MORI_MR_MB_PER_LANE"), nullptr, 10) : 16ULL)
      * 1024ULL * 1024ULL;

  std::vector<std::string> specs = {"1F0R", "3F0R", "3F3R", "4F4R", "6F6R"};
  if (const char* env = std::getenv("MORI_MR_LANES")) {
    specs.clear();
    std::string s(env);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t next = s.find(',', pos);
      specs.push_back(s.substr(pos, next == std::string::npos ? std::string::npos : next - pos));
      if (next == std::string::npos) break;
      pos = next + 1;
    }
  }

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  if (myPe == 0) {
    printf("=== Multi-ring XGMI lane bench ===\n");
    printf("npes=%d MB/lane=%zu blocks/lane=%d warmup=%d iters=%d\n",
           npes, bytesPerLane / (1024 * 1024), blocksPerLane, warmup, iterations);
    printf("%8s %10s %8s %12s %12s\n",
           "MB/lane", "lanes", "blk/lane", "GB/s", "avg_ms");
  }
  for (const auto& s : specs) {
    RunOne(bytesPerLane, ParseLaneSpec(s.c_str()), blocksPerLane, myPe, npes,
           warmup, iterations, stream);
  }

  CHECK_HIP(hipStreamDestroy(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
  return 0;
}
