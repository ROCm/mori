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

__device__ void SendThreadKernel(RdmaEndpoint& epSend, MemoryRegion sendMr, MemoryRegion recvMr) {
  atomicType amoOp = AMO_SET;
  uint32_t value = 2;

  uint64_t dbr_val = PostAtomic<ProviderType::MLX5, uint32_t>(
      epSend.wqHandle.sqAddr, epSend.wqHandle.sqWqeNum, &epSend.wqHandle.postIdx, epSend.wqHandle.postIdx, epSend.handle.qpn, sendMr.addr,
      sendMr.lkey, recvMr.addr, recvMr.rkey, value, value, amoOp);
  UpdateSendDbrRecord<ProviderType::MLX5>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::MLX5>(epSend.wqHandle.dbrAddr, dbr_val);
  __threadfence_system();

  uint16_t wqeCounter;
  int opcode = PollCq<ProviderType::MLX5>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum,
                                          &epSend.cqHandle.consIdx, &wqeCounter);
  UpdateCqDbrRecord<ProviderType::MLX5>(epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx);
  // printf("send block is done, opcode is %d postIdx %u consIdx %u\n", opcode, epSend.wqHandle.postIdx, epSend.cqHandle.consIdx);
}

__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, MemoryRegion mr) {
  uint32_t postIdx = 0;
  uint32_t* addr = reinterpret_cast<uint32_t*>(mr.addr);
  uint32_t val = core::AtomicLoadSeqCst(addr);
  printf("val = %u\n",val);
  // while (val != 1) {
  //   val = core::AtomicLoadSeqCst(addr);
  //   printf("val = %u\n",val);
  // }
}

__global__ void SendRecvOnGpu(RdmaEndpoint& epSend, RdmaEndpoint& epRecv, MemoryRegion mrSend, MemoryRegion mrRecv) {
  assert(gridDim.x == 2);
  int tid = blockIdx.x;
  printf("tid %d start \n", tid);
  if (tid == 0) {
    printf("tid %d send\n", tid);
    SendThreadKernel(epSend, mrSend, mrRecv);
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
  HIP_RUNTIME_CHECK(hipMemset(recvBuf, 0, msgSize));
  void* sendBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&sendBuf, msgSize));
  HIP_RUNTIME_CHECK(hipMemset(sendBuf, 1, msgSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MemoryRegion mrSend = deviceContextSend->RegisterMemoryRegion(sendBuf, msgSize, MR_ACCESS_FLAG);
  MemoryRegion mrRecv = deviceContextRecv->RegisterMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(*devEpSend, *devEpRecv, mrSend, mrRecv);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  uint32_t value; 
  HIP_RUNTIME_CHECK(hipMemcpy(&value, recvBuf, sizeof(uint32_t), hipMemcpyDeviceToHost));
  std::cout << "After atomic op value = " << value << std::endl;  

  // 8 Finalize
  deviceContextSend->DeRegisterMemoryRegion(sendBuf);
  deviceContextRecv->DeRegisterMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(sendBuf));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
}

int main() { LocalRdmaOps(); }