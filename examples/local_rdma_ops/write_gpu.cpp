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

template <ProviderType P>
__device__ void SendThreadKernel(RdmaEndpoint& epSend, RdmaMemoryRegion sendMr, RdmaEndpoint& epRecv,
                                 RdmaMemoryRegion recvMr, int msgSize, int msgNum) {
    uint8_t sendVal = msgNum;
    for (int j = 0; j < msgSize; j++) {
      reinterpret_cast<char*>(sendMr.addr)[j] = sendVal;
    }

    __threadfence_system();
    uint64_t dbr_val = PostWrite<P>(epSend.wqHandle, epSend.handle.qpn, sendMr.addr, sendMr.lkey, recvMr.addr, recvMr.rkey, msgSize);
    printf("PostWrite is done\n");
    __threadfence_system();
    UpdateSendDbrRecord<P>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);
    printf("UpdateSendDbrRecord is done\n");
    __threadfence_system();
    RingDoorbell<P>(epSend.wqHandle.dbrAddr, dbr_val);
    printf("RingDoorbell is done\n");
    __threadfence_system();
    uint16_t wqeIdx;
    int snd_opcode = PollCq<P>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum,
                               &epSend.cqHandle.consIdx, &wqeIdx);
    epSend.cqHandle.consIdx += 1;
    printf("send PollCq is done, wqeIdx: %hu\n", wqeIdx);
    UpdateCqDbrRecord<P>(epSend.cqHandle, epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx,
                         epSend.cqHandle.cqeNum);
    printf("send UpdateCqDbrRecord is done\n");
    // printf("snd_opcode %d val %d\n", snd_opcode, reinterpret_cast<char*>(mrSend.addr)[0]);

}

template <ProviderType P>
__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, RdmaMemoryRegion mr, int msgSize, int msgNum) {
    printf("round %d recv verify for result\n", msgNum);
    uint8_t sendVal = msgNum;

    for (int j = 0; j < msgSize; j++) {
      uint8_t recvVal = reinterpret_cast<char*>(mr.addr)[j];
      if (recvVal != sendVal) {
        printf("round %d expected %d got %d\n", msgNum, sendVal, recvVal);
        assert(false);
      }
    }
    printf("round %d expected %d got %d pass\n", msgNum, sendVal,
           reinterpret_cast<uint8_t*>(mr.addr)[768]);
}

__global__ void SendRecvOnGpu(RdmaEndpoint& epSend, RdmaEndpoint& epRecv, RdmaMemoryRegion mrSend,
                              RdmaMemoryRegion mrRecv, int msgSize, int msgNum) {
  assert(gridDim.x == 1);
  int bid = blockIdx.x;
  for (int i = 0; i < msgNum; i++) {
    printf("bid %d start \n", bid);
    printf("bid %d send\n", bid);
    switch (epSend.GetProviderType()) {
      case ProviderType::MLX5:
        SendThreadKernel<ProviderType::MLX5>(epSend, mrSend, epRecv, mrRecv, msgSize, i);
        break;
#ifdef ENABLE_BNXT        
      case ProviderType::BNXT:
        SendThreadKernel<ProviderType::BNXT>(epSend, mrSend, epRecv, mrRecv, msgSize, i);
        break;
#endif // ENABLE_BNXT        
      case ProviderType::PSD:
        SendThreadKernel<ProviderType::PSD>(epSend, mrSend, epRecv, mrRecv, msgSize, i);
        break;
      default:
        // unsupported provider
        break;
    }

    printf("bid %d recv\n", bid);
    switch (epRecv.GetProviderType()) {
      case ProviderType::MLX5:
        RecvThreadKernel<ProviderType::MLX5>(epRecv, mrRecv, msgSize, i);
        break;
#ifdef ENABLE_BNXT
      case ProviderType::BNXT:
        RecvThreadKernel<ProviderType::BNXT>(epRecv, mrRecv, msgSize, i);
        break;
#endif // ENABLE_BNXT
      case ProviderType::PSD:
        RecvThreadKernel<ProviderType::PSD>(epRecv, mrRecv, msgSize, i);
        break;      
      default:
        // unsupported provider
        break;
    }
  }
}

void LocalRdmaOps() {
  int msgSize = 1024;
  int msgNum = 128;

  // RDMA initialization
  // 1 Create device
  RdmaContext rdmaContext(RdmaBackendType::DirectVerbs);
  printf("nums_device: %d\n", rdmaContext.nums_device);
  RdmaDeviceList rdmaDevices = rdmaContext.GetRdmaDeviceList();
  assert(!rdmaDevices.empty());
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdmaDevices);
  assert(!activeDevicePortList.empty());
  printf("GetActiveDevicePortList done: %zu\n", activeDevicePortList.size());
  ActiveDevicePort devicePort = activeDevicePortList[0];

  RdmaDevice* device = devicePort.first;
  RdmaDeviceContext* deviceContextSend = device->CreateRdmaDeviceContext();
  RdmaDeviceContext* deviceContextRecv = device->CreateRdmaDeviceContext();
  printf("CreateRdmaDeviceContext done\n");

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = devicePort.second;
  config.gidIdx = 3;
  config.maxMsgsNum = 256;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = true;
  RdmaEndpoint epSend = deviceContextSend->CreateRdmaEndpoint(config);
  RdmaEndpoint epRecv = deviceContextRecv->CreateRdmaEndpoint(config);
  printf("CreateRdmaEndpoint done\n");

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
  void* sendBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&sendBuf, msgSize));
  RdmaMemoryRegion mrSend = deviceContextSend->RegisterRdmaMemoryRegion(sendBuf, msgSize, MR_ACCESS_FLAG);

  void* recvBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&recvBuf, msgSize));
  RdmaMemoryRegion mrRecv = deviceContextRecv->RegisterRdmaMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<1, 1>>>(*devEpSend, *devEpRecv, mrSend, mrRecv, msgSize, msgNum);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  deviceContextSend->DeregisterRdmaMemoryRegion(sendBuf);
  deviceContextRecv->DeregisterRdmaMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(sendBuf));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
}

int main() { LocalRdmaOps(); }