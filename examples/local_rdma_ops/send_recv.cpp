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

void SendRecvOnCpu(IbgdaReadWriteReq& send_req, IbgdaReadWriteReq& recv_req,
                   RdmaEndpoint& endpoint_1, RdmaEndpoint& endpoint_2) {
  // Recv
  udma_to_device_barrier();
  PostRecv<ProviderType::MLX5>(recv_req);
  udma_to_device_barrier();
  UpdateRecvDbrRecord<ProviderType::MLX5>(endpoint_2.wq_handle.dbr_rec_addr,
                                          recv_req.qp_handle.post_idx);

  // Send
  uint64_t dbr_val = PostSend<ProviderType::MLX5>(send_req);
  udma_to_device_barrier();
  UpdateSendDbrRecord<ProviderType::MLX5>(endpoint_1.wq_handle.dbr_rec_addr,
                                          send_req.qp_handle.post_idx);
  udma_to_device_barrier();
  RingDoorbell<ProviderType::MLX5>(endpoint_1.wq_handle.dbr_addr, dbr_val);
  udma_to_device_barrier();

  // Poll CQ
  int snd_opcode = PollCq<ProviderType::MLX5>(endpoint_1.cq_handle);
  int rcv_opcode = PollCq<ProviderType::MLX5>(endpoint_2.cq_handle);
  udma_from_device_barrier();

  endpoint_1.cq_handle.consumer_idx += 1;
  endpoint_2.cq_handle.consumer_idx += 1;
  UpdateCqDbrRecord<ProviderType::MLX5>(endpoint_1.cq_handle.dbr_rec_addr,
                                        endpoint_1.cq_handle.consumer_idx);
  UpdateCqDbrRecord<ProviderType::MLX5>(endpoint_2.cq_handle.dbr_rec_addr,
                                        endpoint_2.cq_handle.consumer_idx);
  udma_to_device_barrier();
}

void LocalRdmaOps() {
  bool on_gpu = false;
  int msg_size = 1024;
  int msg_num = 1000;

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context(RdmaBackendType::DirectVerbs);
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  RdmaDevice* device = rdma_devices[0];
  RdmaDeviceContext* device_context_1 = device->CreateRdmaDeviceContext();
  RdmaDeviceContext* device_context_2 = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.port_id = 1;
  config.gid_index = 1;
  config.max_msgs_num = 1000;
  config.max_cqe_num = 256;
  config.alignment = 4096;
  config.on_gpu = on_gpu;
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

  IbgdaReadWriteReq send_req;
  send_req.qp_handle.qpn = endpoint_1.handle.qpn;
  send_req.qp_handle.post_idx = 0;
  send_req.qp_handle.queue_buff_addr = endpoint_1.wq_handle.sq_addr;
  send_req.qp_handle.dbr_rec_addr = endpoint_1.wq_handle.dbr_rec_addr;
  send_req.qp_handle.dbr_addr = endpoint_1.wq_handle.dbr_addr;
  send_req.qp_handle.wqe_num = endpoint_1.wq_handle.sq_wqe_num;
  send_req.local_mr = mr_handle_1;
  send_req.remote_mr = mr_handle_2;
  send_req.bytes_count = msg_size;

  IbgdaReadWriteReq recv_req;
  recv_req.qp_handle.qpn = endpoint_2.handle.qpn;
  recv_req.qp_handle.post_idx = 0;
  recv_req.qp_handle.queue_buff_addr = endpoint_2.wq_handle.rq_addr;
  recv_req.qp_handle.dbr_rec_addr = endpoint_2.wq_handle.dbr_rec_addr;
  recv_req.qp_handle.dbr_addr = endpoint_2.wq_handle.dbr_addr;
  recv_req.qp_handle.wqe_num = endpoint_2.wq_handle.rq_wqe_num;
  recv_req.local_mr = mr_handle_2;
  recv_req.remote_mr = mr_handle_1;
  recv_req.bytes_count = msg_size;

  for (int i = 1; i < msg_num; i++) {
    uint8_t send_val = i;

    // TODO: figure out why without the sync, this memset has no effect, the behavior is that
    // the value wrote to recv_buff is always value of send_buff of the first round (1)
    HIP_RUNTIME_CHECK(hipMemset(send_buff, send_val, msg_size));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    SendRecvOnCpu(send_req, recv_req, endpoint_1, endpoint_2);

    // Chekc results
    for (int j = 0; j < msg_size; j++) {
      uint8_t val = reinterpret_cast<uint8_t*>(recv_buff)[j];
      if (val != send_val) {
        printf("round %d at pos %d expected %d got %d send_buff %d\n", i, j, send_val, val,
               reinterpret_cast<char*>(send_buff)[256]);
        assert(false);
      }
    }
    printf("round %d expected %d got %d pass\n", i, send_val,
           reinterpret_cast<uint8_t*>(recv_buff)[25]);
  }

  // 8 Finalize
  device_context_1->DeRegisterMemoryRegion(send_buff);
  device_context_2->DeRegisterMemoryRegion(recv_buff);
}

int main() { LocalRdmaOps(); }