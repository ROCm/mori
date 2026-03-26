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

// Straightforward all-to-all test using ShmemPutMemNbiThread with SDMA and
// socket bootstrap (no MPI required).
//
// Each of N PEs owns N chunks of data (one chunk destined for each peer).
// After the all-to-all every PE holds N chunks in a single contiguous output
// buffer, where chunk[srcPe] contains the data that srcPe sent to this PE:
//
//   output[ srcPe * chunkBytes .. (srcPe+1) * chunkBytes ) == srcPe's chunk for myPe
//
// Fill pattern (element-wise, flat over the whole input buffer):
//   input[j] = seed ^ j,  where seed = (myPe + 1) and j is the flat index
//
// After all-to-all, PE myPe expects at output[srcPe * chunkElems + i]:
//   (srcPe + 1) ^ (myPe * chunkElems + i)
// because srcPe sent input[myPe * chunkElems + i] = (srcPe+1) ^ (myPe*chunkElems+i).
//
// Algorithm:
//   - Each PE fills its npes-chunk input buffer with the pattern above.
//   - A GPU kernel launches npes*numQ threads. Thread (destPe*numQ + qpId)
//     puts input[destPe*chunkSz + qpOfs .. +qpChunkSz) to PE destPe,
//     landing at output[myPe*chunkSz + qpOfs].
//   - After a global barrier every PE verifies the full output buffer.
//
// Bootstrap coordination via NFS: rank 0 generates a UniqueId and writes it
// to a shared file; other ranks poll until the file appears and read it.
//
// Usage: ./all2all_test <rank> <world_size> <num_gpus_per_process> [chunk_bytes]
//   chunk_bytes defaults to 1 MB and must be a multiple of 4.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr size_t DEFAULT_CHUNK_BYTES = 54 * 2 * 1 * 256 * 48 * 128 * sizeof(uint16_t);
constexpr const char* NFS_DIR = "/data/mori";
constexpr const char* UID_FILENAME = "all2all_test_uid.bin";
constexpr int UID_POLL_INTERVAL_MS = 100;
constexpr int UID_POLL_TIMEOUT_S = 120;

#define USE_MORI_REGISTERED_BUFFER 1

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define XHERE XPUT("ZZZZ %d", __LINE__)

// ---------------------------------------------------------------------------
// UniqueId file I/O helpers
// ---------------------------------------------------------------------------
static std::string GetUidFilePath() {
  return std::string(NFS_DIR) + "/" + UID_FILENAME;
}

static void WriteUniqueIdFile(const mori_shmem_uniqueid_t& uid) {
  std::string path = GetUidFilePath();
  std::string tmpPath = path + ".tmp";

  FILE* f = fopen(tmpPath.c_str(), "wb");
  if (!f) {
    XPUT("ERROR: cannot open %s for writing: %s", tmpPath.c_str(), strerror(errno));
    exit(1);
  }
  if (fwrite(uid.data(), 1, uid.size(), f) != uid.size()) {
    XPUT("ERROR: short write to %s", tmpPath.c_str());
    fclose(f);
    exit(1);
  }
  fclose(f);

  if (rename(tmpPath.c_str(), path.c_str()) != 0) {
    XPUT("ERROR: rename %s -> %s failed: %s", tmpPath.c_str(), path.c_str(), strerror(errno));
    exit(1);
  }
  XPUT("Rank 0: wrote UniqueId to %s", path.c_str());
}

static void ReadUniqueIdFile(mori_shmem_uniqueid_t& uid) {
  std::string path = GetUidFilePath();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(UID_POLL_TIMEOUT_S);

  while (true) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) {
      size_t n = fread(uid.data(), 1, uid.size(), f);
      fclose(f);
      if (n == uid.size()) {
        XPUT("Read UniqueId from %s", path.c_str());
        return;
      }
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      XPUT("ERROR: timed out waiting for %s after %d seconds", path.c_str(), UID_POLL_TIMEOUT_S);
      exit(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(UID_POLL_INTERVAL_MS));
  }
}

static void RemoveUidFile() {
  std::string path = GetUidFilePath();
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Fill the flat input buffer: buf[j] = seed ^ j
__global__ void FillPatternKernel(uint32_t* buf, size_t numElements, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = seed ^ static_cast<uint32_t>(i);
  }
}

// All-to-all kernel using SDMA puts.
//
// Each PE sends npes chunks: chunk destPe lives at inBuf[inOfs + destPe*chunkSz].
// It is PUT to outBuf[outOfs + myPe*chunkSz] on PE destPe.
//
// Thread layout: one thread per (destPe, qpId) pair, packed as
//   tid = destPe * numQ + qpId
__global__ void All2allKernel(int myPe, int npes, int numQ, SymmMemObjPtr inBuf, size_t inOfs,
                              SymmMemObjPtr outBuf, size_t outOfs, size_t chunkSz) {
  const int tid = threadIdx.x;

  if (tid < npes * numQ) {
    int destPe = tid / numQ;
    int qpId = tid - destPe * numQ;

    size_t qpChunkSz = chunkSz / numQ;
    size_t qpOfs = static_cast<size_t>(qpId) * qpChunkSz;
    if (qpId == numQ - 1) {
      qpChunkSz = chunkSz - qpOfs;
    }

    // Source: my input slice meant for destPe
    size_t srcOfs = static_cast<size_t>(destPe) * chunkSz + qpOfs;
    // Destination: on destPe's output buffer, in the slot reserved for myPe
    size_t dstOfs = static_cast<size_t>(myPe) * chunkSz + qpOfs;

    ShmemPutMemNbiThread(outBuf, outOfs + dstOfs, inBuf, inOfs + srcOfs, qpChunkSz, destPe, qpId);

    auto ttype = GetGlobalGpuStatesPtr()->transportTypes[destPe];
    if (ttype == mori::application::SDMA) {
      ShmemQuietThread(destPe, outBuf);
    }
  }

  if (tid == 0) {
    ShmemBarrierAllThread();
  }
}

// Verify the output buffer.
//
// After all-to-all, output[srcPe * chunkElems + i] on PE myPe should equal:
//   (srcPe + 1) ^ (myPe * chunkElems + i)
// because srcPe filled input[myPe * chunkElems + i] = (srcPe+1) ^ (myPe*chunkElems+i).
__global__ void VerifyKernel(const uint32_t* outBuf, size_t chunkElems, int npes, int myPe,
                             uint32_t* errorCount) {
  size_t totalElements = static_cast<size_t>(npes) * chunkElems;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElements;
       i += (size_t)gridDim.x * blockDim.x) {
    int srcPe = static_cast<int>(i / chunkElems);
    size_t localIdx = i % chunkElems;
    // srcPe seeded with (srcPe + 1), and filled at flat index (myPe * chunkElems + localIdx)
    uint32_t expected =
        static_cast<uint32_t>(srcPe + 1) ^
        static_cast<uint32_t>(static_cast<size_t>(myPe) * chunkElems + localIdx);
    if (outBuf[i] != expected) {
      atomicAdd(errorCount, 1u);
    }
  }
}

struct ThreadInfo {
  int rank{-1};
  int worldSize{-1};
  int deviceId{-1};
  int ret_code{-1};
  std::thread::id thread_id;
};

// ---------------------------------------------------------------------------
// Test body (runs after ShmemInit)
// ---------------------------------------------------------------------------
static void RunAll2allThreadedTest(size_t chunkBytes, const UniqueId& uid, ThreadInfo& info) {
  HIP_RUNTIME_CHECK(hipSetDevice(info.deviceId));
  XPUT("---------------------- Global rank %d deviceID %d worldSize %d", info.rank, info.deviceId,
       info.worldSize);

  // --- Bootstrap and init ---
  auto* bootstrap = new SocketBootstrapNetwork(uid, info.rank, info.worldSize);
  int status = ShmemInit(bootstrap);
  if (status != 0) {
    XPUT("ERROR: ShmemInit failed (ret=%d)", status);
    info.ret_code = status;
    return;
  }

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Input:  npes chunks, one per destination PE
  // Output: npes chunks, one per source PE
  size_t inTotalBytes = static_cast<size_t>(npes) * chunkBytes;
  size_t outTotalBytes = inTotalBytes;
  size_t chunkElems = chunkBytes / sizeof(uint32_t);

  if (info.deviceId == 0) {
    XPUT("all2all_test: %d PEs rank: %d, %zu bytes/chunk (%zu KB), %zu bytes total per PE", npes,
         info.rank, chunkBytes, chunkBytes / 1024, inTotalBytes);
  }
  ShmemBarrierAll();

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Layout: [input (npes*chunkBytes)] [output (npes*chunkBytes)]
  size_t inOfs = 0;
  size_t outOfs = inTotalBytes;
  size_t baseTotalBytes = inTotalBytes + outTotalBytes;

  void* baseBuf = nullptr;
  SymmMemObjPtr baseBufObj{};
  if (USE_MORI_REGISTERED_BUFFER) {
    HIP_RUNTIME_CHECK(hipMalloc(&baseBuf, baseTotalBytes));
    baseBufObj = ShmemSymmetricRegister(baseBuf, baseTotalBytes);
  } else {
    baseBuf = ShmemMalloc(baseTotalBytes);
    baseBufObj = ShmemQueryMemObjPtr(baseBuf);
  }

  XPUT("%d: baseBuf: %p baseBufObj: %p", myPe, baseBuf, baseBufObj->localPtr);

  void* inBuf = static_cast<char*>(baseBuf) + inOfs;
  void* outBuf = static_cast<char*>(baseBuf) + outOfs;
  HIP_RUNTIME_CHECK(hipMemsetAsync(baseBuf, 0, baseTotalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  uint32_t seed = static_cast<uint32_t>(myPe + 1);
  size_t inTotalElems = static_cast<size_t>(npes) * chunkElems;
  constexpr int kThreads = 256;
  int fillBlocks =
      static_cast<int>(std::min<size_t>(1024, (inTotalElems + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(reinterpret_cast<uint32_t*>(inBuf),
                                                         inTotalElems, seed);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  ShmemBarrierAll();
  const uint32_t numQ = std::min(baseBufObj->sdmaNumQueue, 4u);

  // --- All-to-all benchmark ---
  constexpr int nWarmup = 5;
  constexpr int nRuns = 20;

  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));

  float totalMs = 0;
  float minMs = 1e9f, maxMs = 0;

  for (int iter = 0; iter < nWarmup + nRuns; iter++) {
    HIP_RUNTIME_CHECK(hipMemsetAsync(outBuf, 0, outTotalBytes, stream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

    ShmemBarrierAll();

    HIP_RUNTIME_CHECK(hipEventRecord(tStart, stream));
    All2allKernel<<<1, kThreads, 0, stream>>>(myPe, npes, numQ, baseBufObj, inOfs, baseBufObj,
                                              outOfs, chunkBytes);
    HIP_RUNTIME_CHECK(hipEventRecord(tStop, stream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

    float iterMs = 0;
    HIP_RUNTIME_CHECK(hipEventElapsedTime(&iterMs, tStart, tStop));

    if (iter >= nWarmup) {
      totalMs += iterMs;
      minMs = std::min(minMs, iterMs);
      maxMs = std::max(maxMs, iterMs);
    }

    ShmemBarrierAll();
  }

  float avgMs = totalMs / nRuns;
  // All-to-all transfers (npes-1)/npes * totalBytes per PE; report total moved
  double avgBw = (outTotalBytes / 1e9) / (avgMs / 1e3);
  double maxBw = (outTotalBytes / 1e9) / (minMs / 1e3);

  // --- Verify (last iteration's result) ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));

  size_t outTotalElems = static_cast<size_t>(npes) * chunkElems;
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (outTotalElems + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(reinterpret_cast<const uint32_t*>(outBuf),
                                                 chunkElems, npes, myPe, dErrors);

  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(
      hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  XPUT(
      "Rank %d: %s (%u errors) | %d warmup + %d runs | avg %.2f ms (%.3f GB/s) "
      "min %.2f ms (%.3f GB/s) max %.2f ms",
      myPe, hErrors == 0 ? "PASS" : "FAIL", hErrors, nWarmup, nRuns, avgMs, avgBw, minMs, maxBw,
      maxMs);

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  if (USE_MORI_REGISTERED_BUFFER) {
    ShmemSymmetricDeregister(baseBuf, baseTotalBytes);
    HIP_RUNTIME_CHECK(hipFree(baseBuf));
  } else {
    ShmemFree(baseBuf);
  }
  ShmemFinalize();
  info.ret_code = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <rank> <world_size> <num_gpus_per_process> [chunk_bytes]\n",
            argv[0]);
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int worldSize = std::atoi(argv[2]);
  int gpusPerProcess = std::atoi(argv[3]);
  assert(rank >= 0 && rank < worldSize && worldSize >= 1);

  size_t chunkBytes = DEFAULT_CHUNK_BYTES;
  if (argc > 4) chunkBytes = std::atol(argv[4]);
  assert(chunkBytes >= 4 && (chunkBytes % 4) == 0);

  XPUT("Rank %d / %d starting (chunk_bytes=%zu)", rank, worldSize, chunkBytes);

  // --- Obtain UniqueId ---
  mori_shmem_uniqueid_t uid_bytes{};

  if (rank == 0) {
    RemoveUidFile();
    int ret = ShmemGetUniqueId(&uid_bytes);
    if (ret != 0) {
      XPUT("ERROR: ShmemGetUniqueId failed (ret=%d)", ret);
      return 1;
    }
    WriteUniqueIdFile(uid_bytes);
  } else {
    ReadUniqueIdFile(uid_bytes);
  }
  UniqueId uid;
  std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

  int deviceCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));

  // NOTE: this only works for processes on the same node
  int startDeviceId = rank * gpusPerProcess;
  if (startDeviceId + gpusPerProcess > deviceCount) {
    XPUT("ERROR: startDeviceId + gpusPerProcess > deviceCount");
    return 1;
  }

  std::vector<std::thread> threads;
  std::vector<ThreadInfo> infos(gpusPerProcess);
  threads.reserve(gpusPerProcess);
  for (int i = 0, deviceId = startDeviceId; i < gpusPerProcess; i++, deviceId++) {
    infos[i].rank = deviceId;  // HACK HACK HACK
    infos[i].worldSize = worldSize * gpusPerProcess;
    infos[i].deviceId = deviceId;
    threads.emplace_back(RunAll2allThreadedTest, chunkBytes, uid, std::ref(infos[i]));
  }
  for (auto& t : threads) {
    t.join();
  }

  for (const auto& info : infos) {
    if (info.ret_code != 0) {
      XPUT("ERROR: Rank %d returned non-zero ret_code %d", info.rank, info.ret_code);
      return 1;
    }
  }

  // Rank 0 cleans up the uid file after everyone is done
  if (rank == 0) {
    RemoveUidFile();
  }
  return 0;
}
