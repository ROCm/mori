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

__device__ void SendThreadKernel(RdmaEndpoint& endpoint_1, MemoryRegion mr, int msg_size,
                                 int msg_num) {
  uint32_t postIdx = 0;

  for (int i = 0; i < msg_num; i++) {
    uint8_t send_val = i;
    for (int j = 0; j < msg_size; j++) {
      reinterpret_cast<char*>(mr.addr)[j] = send_val;
    }

    __threadfence_system();
    uint64_t dbr_val = PostSend<ProviderType::MLX5>(
        endpoint_1.wqHandle.sqAddr, postIdx, endpoint_1.wqHandle.sqWqeNum, endpoint_1.handle.qpn,
        mr.addr, mr.lkey, msg_size);
    __threadfence_system();
    UpdateSendDbrRecord<ProviderType::MLX5>(endpoint_1.wqHandle.dbrRecAddr, postIdx);
    __threadfence_system();
    RingDoorbell<ProviderType::MLX5>(endpoint_1.wqHandle.dbrAddr, dbr_val);
    __threadfence_system();

    int snd_opcode = PoolCq<ProviderType::MLX5>(endpoint_1.cqHandle);
    endpoint_1.cqHandle.consIdx += 1;
    UpdateCqDbrRecord<ProviderType::MLX5>(endpoint_1.cqHandle.dbrRecAddr,
                                          endpoint_1.cqHandle.consIdx);
    // printf("snd_opcode %d val %d\n", snd_opcode, reinterpret_cast<char*>(mr_handle_1.addr)[0]);
  }
}

__device__ void RecvThreadKernel(RdmaEndpoint& endpoint_2, MemoryRegion mr, int msg_size,
                                 int msg_num) {
  uint32_t postIdx = 0;

  for (int i = 0; i < msg_num; i++) {
    uint8_t send_val = i;

    __threadfence_system();
    PostRecv<ProviderType::MLX5>(endpoint_2.wqHandle.rqAddr, endpoint_2.wqHandle.rqWqeNum, postIdx,
                                 mr.addr, mr.lkey, msg_size);
    __threadfence_system();
    UpdateRecvDbrRecord<ProviderType::MLX5>(endpoint_2.wqHandle.dbrRecAddr, postIdx);
    __threadfence_system();

    int rcv_opcode = PoolCq<ProviderType::MLX5>(endpoint_2.cqHandle);
    endpoint_2.cqHandle.consIdx += 1;
    UpdateCqDbrRecord<ProviderType::MLX5>(endpoint_2.cqHandle.dbrRecAddr,
                                          endpoint_2.cqHandle.consIdx);

    for (int j = 0; j < msg_size; j++) {
      uint8_t recv_val = reinterpret_cast<char*>(mr.addr)[j];
      if (recv_val != send_val) {
        printf("round %d expected %d got %d\n", i, send_val, recv_val);
        assert(false);
      }
    }
    printf("round %d expected %d got %d pass\n", i, send_val,
           reinterpret_cast<char*>(mr.addr)[768]);
  }
}

__global__ void SendRecvOnGpu(RdmaEndpoint endpoint_1, RdmaEndpoint endpoint_2,
                              MemoryRegion mr_handle_1, MemoryRegion mr_handle_2, int msg_size,
                              int msg_num) {
  assert(gridDim.x == 2);
  int tid = blockIdx.x;
  printf("tid %d start \n", tid);
  if (tid == 0) {
    // SendRecvKernel(endpoint_1, endpoint_2, mr_handle_1, mr_handle_2, msg_size, msg_num);
    printf("tid %d send\n", tid);
    SendThreadKernel(endpoint_1, mr_handle_1, msg_size, msg_num);
  } else if (tid == 1) {
    printf("tid %d recv\n", tid);
    RecvThreadKernel(endpoint_2, mr_handle_2, msg_size, msg_num);
  }
}

void LocalRdmaOps() {
  //   bool onGpu = false;
  int msg_size = 1024;
  int msg_num = 1000;

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context;
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  RdmaDevice* device = rdma_devices[1];
  RdmaDeviceContext* device_context_1 = device->CreateRdmaDeviceContext();
  RdmaDeviceContext* device_context_2 = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = 1;
  config.gidIdx = 1;
  config.maxMsgsNum = 1000;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = true;
  RdmaEndpoint endpoint_1 = device_context_1->CreateRdmaEndpoint(config);
  RdmaEndpoint endpoint_2 = device_context_2->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  device_context_1->ConnectEndpoint(endpoint_1.handle, endpoint_2.handle);
  device_context_2->ConnectEndpoint(endpoint_2.handle, endpoint_1.handle);
  printf("ep1 qpn %d ep2 qpn %d\n", endpoint_1.handle.qpn, endpoint_2.handle.qpn);

  // 4 Register buffer
  void* send_buff;
  HIP_RUNTIME_CHECK(hipMalloc(&send_buff, msg_size));
  MemoryRegion mr_handle_1 =
      device_context_1->RegisterMemoryRegion(send_buff, msg_size, MR_ACCESS_FLAG);

  void* recv_buff;
  HIP_RUNTIME_CHECK(hipMalloc(&recv_buff, msg_size));
  MemoryRegion mr_handle_2 =
      device_context_2->RegisterMemoryRegion(recv_buff, msg_size, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(endpoint_1, endpoint_2, mr_handle_1, mr_handle_2, msg_size, msg_num);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  device_context_1->DeRegisterMemoryRegion(send_buff);
  device_context_2->DeRegisterMemoryRegion(recv_buff);
}

int main() { LocalRdmaOps(); }