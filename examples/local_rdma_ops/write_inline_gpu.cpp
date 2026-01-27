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

#define MAX_INLINE_DATA_SIZE 12

template <ProviderType P>
__device__ void SendThreadKernel(RdmaEndpoint& epSend, RdmaMemoryRegion mr) {
  uint8_t vals[MAX_INLINE_DATA_SIZE];
  uintptr_t raddr = mr.addr;

  for (int i = 1; i <= MAX_INLINE_DATA_SIZE; i++) {
    uint8_t sendVal = i;
    for (int j = 0; j < i; j++) {
      vals[j] = sendVal;
    }

    uint64_t dbr_val =
        PostWriteInline<P>(epSend.wqHandle, epSend.handle.qpn, vals, raddr, mr.rkey, i);
    UpdateSendDbrRecord<P>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);
    __threadfence_system();
    RingDoorbell<P>(epSend.wqHandle.dbrAddr, dbr_val);
    __threadfence_system();

    int opcode =
        PollCq<P>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum, &epSend.cqHandle.consIdx);
    __threadfence_system();
    UpdateCqDbrRecord<P>(epSend.cqHandle, epSend.cqHandle.consIdx);
    // printf("round %d snd_opcode %d\n", i, opcode);

    raddr += i;
  }
}

__device__ void RecvThreadKernel(RdmaEndpoint& epRecv, RdmaMemoryRegion mr) {
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

__global__ void SendRecvOnGpu(RdmaEndpoint& epSend, RdmaEndpoint& epRecv, RdmaMemoryRegion mrRecv) {
  assert(gridDim.x == 2);
  int bid = blockIdx.x;
  printf("bid %d start \n", bid);
  if (bid == 0) {
    printf("bid %d send\n", bid);
    switch (epSend.GetProviderType()) {
      case ProviderType::MLX5:
        SendThreadKernel<ProviderType::MLX5>(epSend, mrRecv);
        break;
#ifdef ENABLE_BNXT
      case ProviderType::BNXT:
        SendThreadKernel<ProviderType::BNXT>(epSend, mrRecv);
        break;
#endif
#ifdef ENABLE_IONIC
      case ProviderType::PSD:
        SendThreadKernel<ProviderType::PSD>(epSend, mrRecv);
        break;
#endif
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
  // config.gidIdx = 3;
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
  RdmaMemoryRegion mrRecv =
      deviceContextRecv->RegisterRdmaMemoryRegion(recvBuf, msgSize, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(*devEpSend, *devEpRecv, mrRecv);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  deviceContextRecv->DeregisterRdmaMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
}

int main() { LocalRdmaOps(); }
