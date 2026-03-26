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

// Straightforward allgather test using ShmemPutMemNbiBlock with socket
// bootstrap (no MPI required).
//
// Each of N PEs owns a chunk of data. After the allgather every PE holds
// all N chunks in a single contiguous output buffer:
//
//   output[ pe * chunkBytes .. (pe+1) * chunkBytes ) == PE pe's data
//
// Algorithm:
//   - Each PE fills its own slot in the symmetric buffer with a pattern.
//   - A GPU kernel launches N blocks. Block i puts myPe's chunk to PE i.
//   - After a global barrier every PE verifies the full buffer.
//
// Bootstrap coordination via NFS: rank 0 generates a UniqueId and writes it
// to a shared file; other ranks poll until the file appears and read it.
//
// Usage: ./allgather_test <rank> <world_size> [chunk_bytes]
//   chunk_bytes defaults to 1 MB and must be a multiple of 4.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr size_t DEFAULT_CHUNK_BYTES = 1 * 1024 * 1024;
constexpr const char* NFS_DIR = "/data/mori";
constexpr const char* UID_FILENAME = "allgather_test_uid.bin";
constexpr int UID_POLL_INTERVAL_MS = 100;
constexpr int UID_POLL_TIMEOUT_S = 120;

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

  // Atomic rename so readers never see a partial file
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

__global__ void FillPatternKernel(uint32_t* buf, size_t numElements, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = seed ^ static_cast<uint32_t>(i);
  }
}

__global__ void AllGatherKernel(int myPe, int npes, const SymmMemObjPtr buf, size_t chunkBytes) {
  int destPe = blockIdx.x;
  if (destPe >= npes || destPe == myPe) return;

  size_t myOffset = static_cast<size_t>(myPe) * chunkBytes;

  // NYI for SDMA path
  //ShmemPutMemNbiBlock(buf, myOffset, buf, myOffset, chunkBytes, destPe);
  // ShmemPutMemNbiWarp(buf, myOffset, buf, myOffset, chunkBytes, destPe, /*qpId=*/0);

  if (threadIdx.x == 0) {
    // printf(" rank=%d at %d\n", myPe, __LINE__);
    ShmemPutMemNbiThread(buf, myOffset, buf, myOffset, chunkBytes, destPe, 0);
    // //printf("rank=%d at %d\n", myPe, __LINE__);
    // // //    enum TransportType { RDMA = 0, P2P = 1, SDMA = 2 };
    auto ttype = GetGlobalGpuStatesPtr()->transportTypes[destPe];
    printf("transport type: %d numQ: %d\n", ttype, buf->sdmaNumQueue);
    if (ttype == mori::application::SDMA) {
       //ShmemQuietThread(destPe, buf);
      ShmemQuietThreadKernel<mori::application::SDMA>(destPe, buf);
    }
  }
}

__global__ void BarrierKernel() {
  ShmemBarrierAllThread();
}

__global__ void VerifyKernel(const uint32_t* buf, size_t elementsPerChunk, int npes,
                             uint32_t* errorCount) {
  size_t totalElements = static_cast<size_t>(npes) * elementsPerChunk;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElements;
       i += (size_t)gridDim.x * blockDim.x) {
    int rank = static_cast<int>(i / elementsPerChunk);
    size_t localIdx = i % elementsPerChunk;
    uint32_t expected = static_cast<uint32_t>(rank + 1) ^ static_cast<uint32_t>(localIdx);
    if (buf[i] != expected) {
      atomicAdd(errorCount, 1);
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
static void RunAllgatherThreadedTest(size_t chunkBytes, const UniqueId& uid, 
              ThreadInfo& info) {

// 2 processes each 4 GPUs
// rank 0,1,2,3 in process 0
// rank 4,5,6,7 in process 1

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

  auto* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  auto* bootnet = states->bootStates->bootNet;
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  size_t totalBytes = static_cast<size_t>(npes) * chunkBytes;
  size_t elementsPerChunk = chunkBytes / sizeof(uint32_t);

  if (info.deviceId == 0) {
    XPUT("allgather_test: %d PEs rank: %d, %zu bytes/PE (%zu KB), %zu bytes total",
         npes, info.rank, chunkBytes, chunkBytes / 1024, totalBytes);
  }
  bootnet->Barrier();

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  void* buf = ShmemMalloc(totalBytes);
  HIP_RUNTIME_CHECK(hipMemsetAsync(buf, 0, totalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  uint32_t* myChunk = reinterpret_cast<uint32_t*>(buf) + myPe * elementsPerChunk;
  uint32_t seed = static_cast<uint32_t>(myPe + 1);
  constexpr int kThreads = 64;
  int fillBlocks =
      static_cast<int>(std::min<size_t>(1024, (elementsPerChunk + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(myChunk, elementsPerChunk, seed);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  bootnet->Barrier();

  SymmMemObjPtr bufObj = ShmemQueryMemObjPtr(buf);
  assert(bufObj.IsValid());

  // --- Allgather ---
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));
  HIP_RUNTIME_CHECK(hipEventRecord(tStart));

  bootnet->Barrier();

  AllGatherKernel<<<npes, kThreads, 0, stream>>>(myPe, npes, bufObj, chunkBytes);
  BarrierKernel<<<1, 1, 0, stream>>>();
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  HIP_RUNTIME_CHECK(hipEventRecord(tStop));
  HIP_RUNTIME_CHECK(hipEventSynchronize(tStop));
  float elapsedMs = 0;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&elapsedMs, tStart, tStop));

  bootnet->Barrier();

  // std::this_thread::sleep_for(std::chrono::seconds(5));

  // --- Verify ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));

  size_t totalElements = static_cast<size_t>(npes) * elementsPerChunk;
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (totalElements + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(reinterpret_cast<const uint32_t*>(buf), elementsPerChunk,
                                      npes, dErrors);

  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  double bw = (totalBytes / 1e9) / (elapsedMs / 1e3);
  XPUT("Rank %d: %s (%u errors), %.2f ms, %.3f GB/s",
       myPe, hErrors == 0 ? "PASS" : "FAIL", hErrors, elapsedMs, bw);

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  ShmemFree(buf);
  ShmemFinalize();
  info.ret_code = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <rank> <world_size> <num_gpus_per_process> [chunk_bytes]\n", argv[0]);
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
  
  // NOTE this only works for processes on the same node
  int startDeviceId = rank * gpusPerProcess;
  if (startDeviceId + gpusPerProcess > deviceCount) {
    XPUT("ERROR: startDeviceId + gpusPerProcess > deviceCount");
    return 1;
  }

  std::vector<std::thread> threads;
  std::vector<ThreadInfo> infos(gpusPerProcess);
  threads.reserve(gpusPerProcess);
  for (int i = 0, deviceId = startDeviceId; i < gpusPerProcess; i++, deviceId++) {
    infos[i].rank = deviceId; // HACK HACK HACK
    infos[i].worldSize = worldSize * gpusPerProcess;
    infos[i].deviceId = deviceId;
    threads.emplace_back(RunAllgatherThreadedTest, 
             chunkBytes, uid, std::ref(infos[i]));
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
