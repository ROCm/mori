// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Two-terminal RDMA put test (no MPI required):
//   PE 0 (sender): puts data to PE 1 via IBGDA RDMA
//   PE 1 (receiver): spins until data arrives, then verifies
//
// Usage (two terminals):
//   Terminal 1: ./shm_put_rdma_test 0
//   Terminal 2: ./shm_put_rdma_test 1

#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::shmem;
using namespace mori::application;

constexpr int kNumPes = 2;
constexpr const char* kUidFile = "/tmp/mori_shm_put_rdma_test.uid";
constexpr int kThreadNum = 64;
constexpr int kBlockNum = 16;

__global__ void PutThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = tid * sizeof(uint32_t);

  if (myPe == 0) {
    ShmemPutMemNbiThread(memObj, offset, memObj, offset, sizeof(uint32_t), /*pe=*/1, /*qpId=*/1);
    ShmemFenceThread();  // internal call ThreadQuiet, complete all RDMA QP of current PE
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + tid, 0) != 0) {
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <rank>  (rank = 0 or 1)\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  assert(rank == 0 || rank == 1);
  HIP_RUNTIME_CHECK(hipSetDevice(rank));
  printf("[PE %d] Using GPU %d\n", rank, rank);

  // Bootstrap via unique ID file exchange
  mori_shmem_uniqueid_t uid{};
  if (rank == 0) {
    int ret = ShmemGetUniqueId(&uid);
    assert(ret == 0);
    std::ofstream f(kUidFile, std::ios::binary);
    f.write(reinterpret_cast<const char*>(uid.data()), uid.size());
    printf("[PE 0] Unique ID written -- start PE 1 now\n");
  } else {
    printf("[PE 1] Waiting for unique ID ...\n");
    std::ifstream f;
    while (!f.is_open()) {
      f.open(kUidFile, std::ios::binary);
      if (!f.is_open()) usleep(100000);
    }
    f.read(reinterpret_cast<char*>(uid.data()), uid.size());
  }

  mori_shmem_init_attr_t attr{};
  int ret = ShmemSetAttrUniqueIdArgs(rank, kNumPes, &uid, &attr);
  assert(ret == 0);
  ret = ShmemInitAttr(MORI_SHMEM_INIT_WITH_UNIQUEID, &attr);
  assert(ret == 0);

  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  printf("[PE %d] SHMEM init OK\n", rank);

  // Allocate symmetric buffer, PE 0 fills with 0, PE 1 fills with 1
  int numEle = kThreadNum * kBlockNum;
  int buffSize = numEle * sizeof(uint32_t);
  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), rank, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run: PE 0 puts its data to PE 1, PE 1 spins until arrival
  ShmemBarrierAll();
  PutThreadKernel<<<kBlockNum, kThreadNum>>>(rank, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  ShmemBarrierAll();

  // PE 1 verifies received data
  if (rank == 1) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), buff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 0) {
        printf("Error at index %d: expected 0, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    printf("%s: %d elements verified.\n", success ? "PASSED" : "FAILED", numEle);
  }

  ShmemFree(buff);
  if (rank == 0) std::remove(kUidFile);
  ShmemFinalize();
  return 0;
}
