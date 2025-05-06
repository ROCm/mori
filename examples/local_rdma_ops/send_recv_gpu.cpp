#include <hip/hip_runtime.h>

#include "examples/local_rdma_ops/utils.hpp"
#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/transport/ibgda/ibgda.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core::transport;

#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

__device__ void SendThreadKernel(transport::rdma::RdmaEndpoint& endpoint_1,
                                 ibgda::MemoryRegion mr_handle_1, int msg_size, int msg_num) {
  ibgda::IbgdaReadWriteReq send_req;
  send_req.qp_handle.qpn = endpoint_1.handle.qpn;
  send_req.qp_handle.post_idx = 0;
  send_req.qp_handle.queue_buff_addr = endpoint_1.wq_handle.sq_addr;
  send_req.qp_handle.dbr_rec_addr = endpoint_1.wq_handle.dbr_rec_addr;
  send_req.qp_handle.dbr_addr = endpoint_1.wq_handle.dbr_addr;
  send_req.qp_handle.wqe_num = endpoint_1.wq_handle.sq_wqe_num;
  send_req.local_mr = mr_handle_1;
  send_req.bytes_count = msg_size;

  for (int i = 0; i < msg_num; i++) {
    uint8_t send_val = i;
    for (int j = 0; j < msg_size; j++) {
      reinterpret_cast<char*>(send_req.local_mr.addr)[j] = send_val;
    }

    __threadfence_system();
    uint64_t dbr_val = ibgda::PostSend<ibgda::ProviderType::MLX5>(send_req);
    __threadfence_system();
    ibgda::UpdateSendDbrRecord<ibgda::ProviderType::MLX5>(endpoint_1.wq_handle.dbr_rec_addr,
                                                          send_req.qp_handle.post_idx);
    __threadfence_system();
    ibgda::RingDoorbell<ibgda::ProviderType::MLX5>(endpoint_1.wq_handle.dbr_addr, dbr_val);
    __threadfence_system();

    int snd_opcode = ibgda::PoolCq<ibgda::ProviderType::MLX5>(endpoint_1.cq_handle);
    endpoint_1.cq_handle.consumer_idx += 1;
    ibgda::UpdateCqDbrRecord<ibgda::ProviderType::MLX5>(endpoint_1.cq_handle.dbr_rec_addr,
                                                        endpoint_1.cq_handle.consumer_idx);
    // printf("snd_opcode %d val %d\n", snd_opcode, reinterpret_cast<char*>(mr_handle_1.addr)[0]);
  }
}

__device__ void RecvThreadKernel(transport::rdma::RdmaEndpoint& endpoint_2,
                                 ibgda::MemoryRegion mr_handle_2, int msg_size, int msg_num) {
  ibgda::IbgdaReadWriteReq recv_req;
  recv_req.qp_handle.qpn = endpoint_2.handle.qpn;
  recv_req.qp_handle.post_idx = 0;
  recv_req.qp_handle.queue_buff_addr = endpoint_2.wq_handle.rq_addr;
  recv_req.qp_handle.dbr_rec_addr = endpoint_2.wq_handle.dbr_rec_addr;
  recv_req.qp_handle.dbr_addr = endpoint_2.wq_handle.dbr_addr;
  recv_req.qp_handle.wqe_num = endpoint_2.wq_handle.rq_wqe_num;
  recv_req.local_mr = mr_handle_2;
  recv_req.bytes_count = msg_size;

  for (int i = 0; i < msg_num; i++) {
    uint8_t send_val = i;

    __threadfence_system();
    ibgda::PostRecv<ibgda::ProviderType::MLX5>(recv_req);
    __threadfence_system();
    ibgda::UpdateRecvDbrRecord<ibgda::ProviderType::MLX5>(endpoint_2.wq_handle.dbr_rec_addr,
                                                          recv_req.qp_handle.post_idx);
    __threadfence_system();

    int rcv_opcode = ibgda::PoolCq<ibgda::ProviderType::MLX5>(endpoint_2.cq_handle);
    endpoint_2.cq_handle.consumer_idx += 1;
    ibgda::UpdateCqDbrRecord<ibgda::ProviderType::MLX5>(endpoint_2.cq_handle.dbr_rec_addr,
                                                        endpoint_2.cq_handle.consumer_idx);

    for (int j = 0; j < msg_size; j++) {
      uint8_t recv_val = reinterpret_cast<char*>(recv_req.local_mr.addr)[j];
      if (recv_val != send_val) {
        printf("round %d expected %d got %d\n", i, send_val, recv_val);
        assert(false);
      }
    }
    printf("round %d expected %d got %d pass\n", i, send_val,
           reinterpret_cast<char*>(recv_req.local_mr.addr)[768]);
  }
}

__global__ void SendRecvOnGpu(transport::rdma::RdmaEndpoint endpoint_1,
                              transport::rdma::RdmaEndpoint endpoint_2,
                              ibgda::MemoryRegion mr_handle_1, ibgda::MemoryRegion mr_handle_2,
                              int msg_size, int msg_num) {
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
  //   bool on_gpu = false;
  int msg_size = 1024;
  int msg_num = 1000;

  // RDMA initialization
  // 1 Create device
  transport::rdma::RdmaContext rdma_context;
  transport::rdma::RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  transport::rdma::RdmaDevice* device = rdma_devices[1];
  transport::rdma::RdmaDeviceContext* device_context_1 = device->CreateRdmaDeviceContext();
  transport::rdma::RdmaDeviceContext* device_context_2 = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  transport::rdma::RdmaEndpointConfig config;
  config.port_id = 1;
  config.gid_index = 1;
  config.max_msgs_num = 1000;
  config.max_cqe_num = 256;
  config.alignment = 4096;
  config.on_gpu = true;
  transport::rdma::RdmaEndpoint endpoint_1 = device_context_1->CreateRdmaEndpoint(config);
  transport::rdma::RdmaEndpoint endpoint_2 = device_context_2->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  device_context_1->ConnectEndpoint(endpoint_1.handle, endpoint_2.handle);
  device_context_2->ConnectEndpoint(endpoint_2.handle, endpoint_1.handle);
  printf("ep1 qpn %d ep2 qpn %d\n", endpoint_1.handle.qpn, endpoint_2.handle.qpn);

  // 4 Register buffer
  void* send_buff;
  HIP_RUNTIME_CHECK(hipMalloc(&send_buff, msg_size));
  ibgda::MemoryRegion mr_handle_1 =
      device_context_1->RegisterMemoryRegion(send_buff, msg_size, MR_ACCESS_FLAG);

  void* recv_buff;
  HIP_RUNTIME_CHECK(hipMalloc(&recv_buff, msg_size));
  ibgda::MemoryRegion mr_handle_2 =
      device_context_2->RegisterMemoryRegion(recv_buff, msg_size, MR_ACCESS_FLAG);

  SendRecvOnGpu<<<2, 1>>>(endpoint_1, endpoint_2, mr_handle_1, mr_handle_2, msg_size, msg_num);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // 8 Finalize
  device_context_1->DeRegisterMemoryRegion(send_buff);
  device_context_2->DeRegisterMemoryRegion(recv_buff);
}

int main() { LocalRdmaOps(); }