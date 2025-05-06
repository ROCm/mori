#include <hip/hip_runtime.h>

#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/transport/ibgda/ibgda.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core::transport;

#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

struct AllReduceParams {
  size_t allreduce_size;
  int local_rank{0};
  int world_size{0};
  ibgda::QueuePairHandle qp;
  ibgda::MemoryRegion* mrs{nullptr};
  ibgda::CompletionQueueHandle cq;
};

__global__ void AllReduceOverRdmaKernel(AllReduceParams params) {
  int local_rank = params.local_rank;

  for (int i = 0; i < params.world_size; i++) {
    if (i == local_rank) continue;
    ibgda::IbgdaReadWriteReq wreq;
    wreq.qp_handle = params.qp;
    wreq.local_mr = params.mrs[local_rank];
    wreq.remote_mr = params.mrs[i];
    wreq.bytes_count = params.allreduce_size;

    uint64_t dbr_val = ibgda::PostRead<ibgda::ProviderType::MLX5>(wreq);
    ibgda::UpdateSendDbrRecord<ibgda::ProviderType::MLX5>(params.qp.dbr_rec_addr,
                                                          params.qp.post_idx);
    ibgda::RingDoorbell<ibgda::ProviderType::MLX5>(params.qp.dbr_addr, dbr_val);
    int opcode = ibgda::PoolCq<ibgda::ProviderType::MLX5>(params.cq);
  }
}

void AllReduceOverRdmaCpu(AllReduceParams params) {
  int local_rank = params.local_rank;
  int allreduce_size_rank = params.allreduce_size / params.world_size;

  for (int i = 0; i < params.world_size; i++) {
    if (i == local_rank) continue;
    ibgda::IbgdaReadWriteReq wreq;
    wreq.qp_handle = params.qp;
    wreq.local_mr = params.mrs[local_rank];
    // wreq.local_mr.addr += allreduce_size_rank*i;
    wreq.remote_mr = params.mrs[i];
    // wreq.remote_mr.addr += allreduce_size_rank*i;
    wreq.bytes_count = params.allreduce_size;

    printf("Local rank %d local buffer %p remote buffer %p\n", local_rank, wreq.local_mr.addr,
           wreq.remote_mr.addr);

    uint64_t dbr_val = ibgda::PostRead<ibgda::ProviderType::MLX5>(wreq);
    // params.qp.post_idx += 1;
    udma_to_device_barrier();
    ibgda::UpdateSendDbrRecord<ibgda::ProviderType::MLX5>(params.qp.dbr_rec_addr,
                                                          params.qp.post_idx);
    udma_to_device_barrier();
    ibgda::RingDoorbell<ibgda::ProviderType::MLX5>(params.qp.dbr_addr, dbr_val);
    udma_to_device_barrier();

    params.cq.consumer_idx = 0;
    int opcode = ibgda::PoolCq<ibgda::ProviderType::MLX5>(params.cq);

    udma_from_device_barrier();

    printf("local rank %d remote rank %d opcode %d\n", local_rank, i, opcode);
  }
}

void AllReduceOverRdma() {
  MpiBootstrapNetwork bootstrap_net;
  bootstrap_net.Initialize();

  bool on_gpu = false;
  int allreduce_size = 1024;
  int local_rank = bootstrap_net.GetLocalRank();
  int world_size = bootstrap_net.GetWorldSize();

  // RDMA initialization
  // 1 Create device
  transport::rdma::RdmaContext rdma_context;
  transport::rdma::RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  transport::rdma::RdmaDevice* device_0 = rdma_devices[0];
  transport::rdma::RdmaDeviceContext* device_0_context = device_0->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  transport::rdma::RdmaEndpointConfig config;
  config.port_id = 1;
  config.gid_index = 1;
  config.max_msgs_num = 10;
  config.max_cqe_num = 256;
  config.alignment = 4096;
  config.on_gpu = on_gpu;
  transport::rdma::RdmaEndpoint local_endpoint = device_0_context->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  transport::rdma::RdmaEndpointHandle global_rdma_ep_handles[world_size];
  bootstrap_net.Allgather(&local_endpoint.handle, global_rdma_ep_handles,
                          sizeof(local_endpoint.handle));

  std::cout << "Local rank " << local_rank << " " << local_endpoint.handle << std::endl;

  for (int i = 0; i < world_size; i++) {
    if (i == local_rank) continue;
    device_0_context->ConnectEndpoint(local_endpoint.handle, global_rdma_ep_handles[i]);
    std::cout << "Local rank " << local_rank << " received " << global_rdma_ep_handles[i]
              << std::endl;
  }

  // 4 Register buffer
  void* buffer;
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, allreduce_size));
  HIP_RUNTIME_CHECK(hipMemset(buffer, local_rank, allreduce_size));
  ibgda::MemoryRegion local_mr_handle =
      device_0_context->RegisterMemoryRegion(buffer, allreduce_size, MR_ACCESS_FLAG);

  ibgda::MemoryRegion global_mr_handles[world_size];
  bootstrap_net.Allgather(&local_mr_handle, global_mr_handles, sizeof(local_mr_handle));
  global_mr_handles[local_rank] = local_mr_handle;

  // 5 Prepare kernel argument
  AllReduceParams params;
  params.local_rank = local_rank;
  params.world_size = world_size;
  params.qp.qpn = local_endpoint.handle.qpn;
  params.qp.post_idx = 0;
  params.qp.queue_buff_addr = local_endpoint.wq_handle.sq_addr;
  params.qp.dbr_rec_addr = local_endpoint.wq_handle.dbr_rec_addr;
  params.qp.dbr_addr = local_endpoint.wq_handle.dbr_addr;
  HIP_RUNTIME_CHECK(hipMalloc(&params.mrs, sizeof(global_mr_handles)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(params.mrs, global_mr_handles, sizeof(global_mr_handles), hipMemcpyHostToDevice));
  params.cq = local_endpoint.cq_handle;

  // 6 Launch kernel
  if (on_gpu) {
    AllReduceOverRdmaKernel<<<1, 1>>>(params);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  } else {
    AllReduceOverRdmaCpu(params);
  }

  // 7 Check correctness
  printf("Local Rank %d 0th %d 512th %d\n", local_rank, ((char*)buffer)[0], ((char*)buffer)[768]);

  // 8 Finalize
  device_0_context->DeRegisterMemoryRegion(buffer);
  bootstrap_net.Finalize();
}

int main() { AllReduceOverRdma(); }