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
__device__ void SendThreadKernel(RdmaEndpoint& epSend, RdmaMemoryRegion sendMr,
                                 RdmaMemoryRegion recvMr) {
  atomicType amoOp = AMO_COMPARE_SWAP;
  uint32_t value = 2;

  uint64_t dbr_val =
      PostAtomic<P, uint32_t>(epSend.wqHandle, epSend.handle.qpn, sendMr.addr, sendMr.lkey,
                              recvMr.addr, recvMr.rkey, value, 0, amoOp);
  UpdateSendDbrRecord<P>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);
  __threadfence_system();
  RingDoorbell<P>(epSend.wqHandle.dbrAddr, dbr_val);
  __threadfence_system();

  uint16_t wqeCounter;
  int opcode = PollCq<P>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum, &epSend.cqHandle.consIdx,
                         &wqeCounter);
  epSend.cqHandle.consIdx += 1;
  UpdateCqDbrRecord<P>(epSend.cqHandle, epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx, epSend.cqHandle.cqeNum);
  printf("wqeCounter = %hu\n", wqeCounter);
  // printf("send block is done, opcode is %d postIdx %u consIdx %u\n", opcode,
  // epSend.wqHandle.postIdx, epSend.cqHandle.consIdx);
  amoOp = AMO_FETCH_ADD;
  dbr_val = PostAtomic<P, uint32_t>(epSend.wqHandle, epSend.handle.qpn, sendMr.addr, sendMr.lkey,
                                    recvMr.addr, recvMr.rkey, value, 0, amoOp);
  UpdateSendDbrRecord<P>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);
  __threadfence_system();
  RingDoorbell<P>(epSend.wqHandle.dbrAddr, dbr_val);
  __threadfence_system();
  opcode = PollCq<P>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum, &epSend.cqHandle.consIdx,
                     &wqeCounter);
  epSend.cqHandle.consIdx += 1;
  UpdateCqDbrRecord<P>(epSend.cqHandle, epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx, epSend.cqHandle.cqeNum);
  printf("wqeCounter = %hu\n", wqeCounter);
  printf("send block is done, opcode is %d postIdx %u consIdx %u\n", opcode,
         epSend.wqHandle.postIdx, epSend.cqHandle.consIdx);
}

__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, RdmaMemoryRegion mr) {
  uint32_t postIdx = 0;
  uint32_t* addr = reinterpret_cast<uint32_t*>(mr.addr);
  uint32_t val = core::AtomicLoadSeqCst(addr);
  printf("val = %u\n", val);
  while (val != 2) {
    val = core::AtomicLoadSeqCst(addr);
    printf("after compare and swap val = %u\n", val);
  }
  while (val != 4) {
    val = core::AtomicLoadSeqCst(addr);
    printf("after fetch add val = %u\n", val);
  }
}

__global__ void SendRecvOnGpu(RdmaEndpoint& epSend, RdmaEndpoint& epRecv, RdmaMemoryRegion mrSend,
                              RdmaMemoryRegion mrRecv) {
  assert(gridDim.x == 2);
  int bid = blockIdx.x;
  printf("bid %d start \n", bid);
  if (bid == 0) {
    printf("bid %d send\n", bid);
    switch (epSend.GetProviderType()) {
      case ProviderType::MLX5:
        SendThreadKernel<ProviderType::MLX5>(epSend, mrSend, mrRecv);
        break;
#ifdef ENABLE_BNXT
      case ProviderType::BNXT:
        SendThreadKernel<ProviderType::BNXT>(epSend, mrSend, mrRecv);
        break;
#endif  // ENABLE_BNXT
      case ProviderType::PSD:
        SendThreadKernel<ProviderType::PSD>(epSend, mrSend, mrRecv);
        break;
      default:
        // unsupported provider
        break;
    }
  } else if (bid == 1) {
    printf("bid %d recv\n", bid);
    RecvThreadKernel(epRecv, mrRecv);
  }
  __syncthreads();
}

void LocalRdmaOps() {
  int msgSize = 78;
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
  RdmaMemoryRegion mrSend =
      deviceContextSend->RegisterRdmaMemoryRegion(sendBuf, msgSize, MR_ACCESS_FLAG);
  RdmaMemoryRegion mrRecv =
      deviceContextRecv->RegisterRdmaMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(*devEpSend, *devEpRecv, mrSend, mrRecv);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  uint32_t valueSend, valueRecv;
  HIP_RUNTIME_CHECK(hipMemcpy(&valueRecv, recvBuf, sizeof(uint32_t), hipMemcpyDeviceToHost));
  std::cout << "After atomic op recv value = " << valueRecv << std::endl;
  HIP_RUNTIME_CHECK(hipMemcpy(&valueSend, sendBuf, sizeof(uint32_t), hipMemcpyDeviceToHost));
  std::cout << "After atomic op send value = " << valueSend << std::endl;

  // 8 Finalize
  deviceContextSend->DeregisterRdmaMemoryRegion(sendBuf);
  deviceContextRecv->DeregisterRdmaMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(sendBuf));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
}

int main() { LocalRdmaOps(); }
