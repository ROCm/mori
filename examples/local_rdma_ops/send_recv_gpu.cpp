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

__device__ void SendThreadKernel(RdmaEndpoint& epSend, MemoryRegion mr, int msgSize, int msgNum) {
  uint32_t postIdx = 0;

  for (int i = 0; i < msgNum; i++) {
    uint8_t sendVal = i;
    for (int j = 0; j < msgSize; j++) {
      reinterpret_cast<char*>(mr.addr)[j] = sendVal;
    }

    __threadfence_system();
    uint64_t dbr_val =
        PostSend<ProviderType::MLX5>(epSend.wqHandle.sqAddr, &postIdx, epSend.wqHandle.sqWqeNum,
                                     epSend.handle.qpn, mr.addr, mr.lkey, msgSize);
    __threadfence_system();
    UpdateSendDbrRecord<ProviderType::MLX5>(epSend.wqHandle.dbrRecAddr, postIdx);
    __threadfence_system();
    RingDoorbell<ProviderType::MLX5>(epSend.wqHandle.dbrAddr, dbr_val);
    __threadfence_system();

    int snd_opcode = PollCq<ProviderType::MLX5>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeSize,
                                                epSend.cqHandle.cqeNum, &epSend.cqHandle.consIdx);
    UpdateCqDbrRecord<ProviderType::MLX5>(epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx);
    // printf("snd_opcode %d val %d\n", snd_opcode, reinterpret_cast<char*>(mrSend.addr)[0]);
  }
}

__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, MemoryRegion mr, int msgSize, int msgNum) {
  uint32_t postIdx = 0;

  for (int i = 0; i < msgNum; i++) {
    uint8_t sendVal = i;

    __threadfence_system();
    PostRecv<ProviderType::MLX5>(epRecv.wqHandle.rqAddr, epRecv.wqHandle.rqWqeNum, &postIdx,
                                 mr.addr, mr.lkey, msgSize);
    __threadfence_system();
    UpdateRecvDbrRecord<ProviderType::MLX5>(epRecv.wqHandle.dbrRecAddr, postIdx);
    __threadfence_system();

    int rcv_opcode = PollCq<ProviderType::MLX5>(epRecv.cqHandle.cqAddr, epRecv.cqHandle.cqeSize,
                                                epRecv.cqHandle.cqeNum, &epRecv.cqHandle.consIdx);
    UpdateCqDbrRecord<ProviderType::MLX5>(epRecv.cqHandle.dbrRecAddr, epRecv.cqHandle.consIdx);

    for (int j = 0; j < msgSize; j++) {
      uint8_t recvVal = reinterpret_cast<char*>(mr.addr)[j];
      if (recvVal != sendVal) {
        printf("round %d expected %d got %d\n", i, sendVal, recvVal);
        assert(false);
      }
    }
    printf("round %d expected %d got %d pass\n", i, sendVal,
           reinterpret_cast<uint8_t*>(mr.addr)[768]);
  }
}

__global__ void SendRecvOnGpu(RdmaEndpoint epSend, RdmaEndpoint epRecv, MemoryRegion mrSend,
                              MemoryRegion mrRecv, int msgSize, int msgNum) {
  assert(gridDim.x == 2);
  int tid = blockIdx.x;
  printf("tid %d start \n", tid);
  if (tid == 0) {
    printf("tid %d send\n", tid);
    SendThreadKernel(epSend, mrSend, msgSize, msgNum);
  } else if (tid == 1) {
    printf("tid %d recv\n", tid);
    RecvThreadKernel(epRecv, mrRecv, msgSize, msgNum);
  }
}

void LocalRdmaOps() {
  int msgSize = 1024;
  int msgNum = 1000;

  // RDMA initialization
  // 1 Create device
  RdmaContext rdmaContext(RdmaBackendType::DirectVerbs);
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
  config.gidIdx = 1;
  config.maxMsgsNum = 1000;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = true;
  RdmaEndpoint epSend = deviceContextSend->CreateRdmaEndpoint(config);
  RdmaEndpoint epRecv = deviceContextRecv->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  deviceContextSend->ConnectEndpoint(epSend.handle, epRecv.handle);
  deviceContextRecv->ConnectEndpoint(epRecv.handle, epSend.handle);
  printf("ep1 qpn %d ep2 qpn %d\n", epSend.handle.qpn, epRecv.handle.qpn);

  // 4 Register buffer
  void* sendBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&sendBuf, msgSize));
  MemoryRegion mrSend = deviceContextSend->RegisterMemoryRegion(sendBuf, msgSize, MR_ACCESS_FLAG);

  void* recvBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&recvBuf, msgSize));
  MemoryRegion mrRecv = deviceContextRecv->RegisterMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(epSend, epRecv, mrSend, mrRecv, msgSize, msgNum);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  deviceContextSend->DeRegisterMemoryRegion(sendBuf);
  deviceContextRecv->DeRegisterMemoryRegion(recvBuf);
}

int main() { LocalRdmaOps(); }