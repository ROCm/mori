#include <hip/hip_runtime.h>

#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/core.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core;

#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

#define MAX_INLINE_DATA_SIZE 12

__device__ void SendThreadKernel(RdmaEndpoint& epSend, MemoryRegion mr) {
  uint32_t postIdx = 0;
  uint8_t vals[MAX_INLINE_DATA_SIZE];
  uintptr_t raddr = mr.addr;

  for (int i = 1; i <= MAX_INLINE_DATA_SIZE; i++) {
    uint8_t sendVal = i;
    for (int j = 0; j < i; j++) {
      vals[j] = sendVal;
    }

    uint64_t dbr_val =
        PostWriteInline<ProviderType::MLX5>(epSend.wqHandle.sqAddr, epSend.wqHandle.sqWqeNum,
                                            &postIdx, postIdx, epSend.handle.qpn, vals, raddr, mr.rkey, i);
    UpdateSendDbrRecord<ProviderType::MLX5>(epSend.wqHandle.dbrRecAddr, postIdx);
    __threadfence_system();
    RingDoorbell<ProviderType::MLX5>(epSend.wqHandle.dbrAddr, dbr_val);
    __threadfence_system();

    int opcode = PollCq<ProviderType::MLX5>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum,
                                            &epSend.cqHandle.consIdx);
    UpdateCqDbrRecord<ProviderType::MLX5>(epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx);
    // printf("round %d snd_opcode %d\n", i, opcode);

    raddr += i;
  }
}

__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, MemoryRegion mr) {
  uint32_t postIdx = 0;
  uint8_t* addr = reinterpret_cast<uint8_t*>(mr.addr);

  for (int i = 1; i <= MAX_INLINE_DATA_SIZE; i++) {
    uint8_t sendVal = i;
    for (int j = 0; j < i; j++) {
      while (core::AtomicLoadSeqCst(addr + j) != sendVal) {
      }
      //   printf("%d %d %d\n", i, j, core::AtomicLoadSeqCst(addr + j));
    }
    printf("round %d pass\n", i);
    addr += i;
  }
}

__global__ void SendRecvOnGpu(RdmaEndpoint& epSend, RdmaEndpoint& epRecv, MemoryRegion mrRecv) {
  assert(gridDim.x == 2);
  int tid = blockIdx.x;
  printf("tid %d start \n", tid);
  if (tid == 0) {
    printf("tid %d send\n", tid);
    SendThreadKernel(epSend, mrRecv);
  } else if (tid == 1) {
    printf("tid %d recv\n", tid);
    RecvThreadKernel(epRecv, mrRecv);
  }
  __syncthreads();
}

void LocalRdmaOps() {
  int msgSize = 78;

  // RDMA initialization
  // 1 Create device
  RdmaContext rdmaContext;
  RdmaDeviceList rdmaDevices = rdmaContext.GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdmaDevices);
  assert(!activeDevicePortList.empty());

  ActiveDevicePort devicePort = activeDevicePortList[0];

  RdmaDevice* device = devicePort.first;
  RdmaDeviceContext* deviceContextSend = device->CreateRdmaDeviceContext();
  RdmaDeviceContext* deviceContextRecv = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = devicePort.second;
  config.gidIdx = 3;
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  config.onGpu = true;
  RdmaEndpoint epSend = deviceContextSend->CreateRdmaEndpoint(config);
  RdmaEndpoint epRecv = deviceContextRecv->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  deviceContextSend->ConnectEndpoint(epSend.handle, epRecv.handle);
  deviceContextRecv->ConnectEndpoint(epRecv.handle, epSend.handle);
  printf("ep1 qpn %d ep2 qpn %d\n", epSend.handle.qpn, epRecv.handle.qpn);

  // 4 Register buffer
  RdmaEndpoint* devEpSend;
  HIP_RUNTIME_CHECK(hipMalloc(&devEpSend, sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(hipMemcpy(devEpSend, &epSend, sizeof(RdmaEndpoint), hipMemcpyHostToDevice));
  RdmaEndpoint* devEpRecv;
  HIP_RUNTIME_CHECK(hipMalloc(&devEpRecv, sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(hipMemcpy(devEpRecv, &epRecv, sizeof(RdmaEndpoint), hipMemcpyHostToDevice));
  void* recvBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&recvBuf, msgSize));
  HIP_RUNTIME_CHECK(hipMemset(recvBuf, 99, msgSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MemoryRegion mrRecv = deviceContextRecv->RegisterMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(*devEpSend, *devEpRecv, mrRecv);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  deviceContextRecv->DeRegisterMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
}

int main() { LocalRdmaOps(); }