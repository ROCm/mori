// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// A simple 2-GPU shmem test without MPI:
//   PE 0 (writer): directly writes to PE 1's buffer via P2P pointer
//   PE 1 (reader): spins until the expected value arrives, then exits
//
// Usage (two terminals):
//   Terminal 1: ./simple_shm_test 0
//   Terminal 2: ./simple_shm_test 1

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "mori/shmem/shmem.hpp"

using namespace mori::shmem;

constexpr int kNumPes = 2;
constexpr uint32_t kExpectedVal = 0xDEAD;
constexpr const char* kUidFile = "/tmp/mori_simple_shm_test.uid";

// Writer stores directly via P2P pointer; reader polls local buffer.
__global__ void SimplePutKernel(uint32_t* writePtr, uint32_t* readPtr, int myPe) {
  if (myPe == 0) {
    // atomicExch(writePtr, kExpectedVal);
    *writePtr = kExpectedVal;
    __threadfence_system();
  } else {
    while (atomicAdd(readPtr, 0) != kExpectedVal) {
    }
  }
}

// 辅助函数：打印当前显存使用
size_t PrintGpuMemUsage(const char* label, int rank) {
  size_t free_bytes = 0, total_bytes = 0;
  hipMemGetInfo(&free_bytes, &total_bytes);
  printf("[PE %d] %s: used = %zu MB, free = %zu MB, total = %zu MB\n", rank, label,
         (total_bytes - free_bytes) / (1024 * 1024), free_bytes / (1024 * 1024),
         total_bytes / (1024 * 1024));

  return total_bytes - free_bytes;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <rank>  (rank = 0 or 1)\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  assert(rank == 0 || rank == 1);
  hipSetDevice(rank);
  printf("[PE %d] Using GPU %d\n", rank, rank);

  // Bootstrap: PE 0 generates unique ID and writes to file; PE 1 waits and reads it.
  mori_shmem_uniqueid_t uid{};
  if (rank == 0) {
    assert(ShmemGetUniqueId(&uid) == 0);
    std::ofstream f(kUidFile, std::ios::binary);
    f.write(reinterpret_cast<const char*>(uid.data()), uid.size());
    printf("[PE 0] Unique ID written to %s — start PE 1 now\n", kUidFile);
  } else {
    printf("[PE 1] Waiting for %s ...\n", kUidFile);
    std::ifstream f;
    while (!f.is_open()) {
      f.open(kUidFile, std::ios::binary);
      if (!f.is_open()) usleep(100000);
    }
    f.read(reinterpret_cast<char*>(uid.data()), uid.size());
  }

  mori_shmem_init_attr_t attr{};
  assert(ShmemSetAttrUniqueIdArgs(rank, kNumPes, &uid, &attr) == 0);

  size_t used1, used2;
  used1 =  PrintGpuMemUsage("Before shmem init", rank);
  assert(ShmemInitAttr(MORI_SHMEM_INIT_WITH_UNIQUEID, &attr) == 0);
  used2 = PrintGpuMemUsage("After shmem init", rank);

  printf("%u GB gpu mem has been used after mori startup init", (used2 - used1) >> 30);

  printf("[PE %d] ShmemInitAttr OK\n", rank);

  auto* buf = reinterpret_cast<uint32_t*>(ShmemMalloc(sizeof(uint32_t)));
  assert(buf);
  hipMemset(buf, 0, sizeof(uint32_t));
  hipDeviceSynchronize();

  ShmemBarrierAll();

  // Host-side P2P address translation
  uint32_t* remotePtr = nullptr;
  const char* mode = std::getenv("MORI_SHMEM_MODE");
  bool isIsolation = mode && std::strcmp(mode, "isolation") == 0;

  if (isIsolation) {
    // Isolation mode: each allocation has its own SymmMemObj, use it directly
    auto memObj = ShmemQueryMemObjPtr(buf);
    assert(memObj.IsValid());
    remotePtr = reinterpret_cast<uint32_t*>(memObj.cpu->p2pPeerPtrs[rank ^ 1]);
    printf("[PE %d] Isolation mode: remote P2P ptr = %p\n", rank, remotePtr);
  } else {
    // Heap mode: use heap-based offset calculation
    remotePtr =
        reinterpret_cast<uint32_t*>(ShmemPtrP2p(reinterpret_cast<uint64_t>(buf), rank, rank ^ 1));
  }

  printf("[PE %d] Launching kernel...\n", rank);
  SimplePutKernel<<<1, 1>>>(remotePtr, buf, rank);
  hipDeviceSynchronize();

  if (rank == 1) {
    uint32_t val = 0;
    hipMemcpy(&val, buf, sizeof(val), hipMemcpyDeviceToHost);
    printf("[PE %d] %s: got 0x%X\n", rank, val == kExpectedVal ? "PASSED" : "FAILED", val);
  } else {
    printf("[PE %d] Write complete\n", rank);
    std::remove(kUidFile);
  }

  ShmemFree(buf);
  ShmemFinalize();
  return 0;
}
