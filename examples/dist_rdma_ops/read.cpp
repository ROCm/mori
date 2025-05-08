#include <hip/hip_runtime.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/core.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core;

#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

void LocalRdmaOps() {
  MpiBootstrapNetwork bootstrap_net(MPI_COMM_WORLD);
  bootstrap_net.Initialize();

  bool on_gpu = false;
  int allreduce_size = 1024;
  int local_rank = bootstrap_net.GetLocalRank();
  int world_size = bootstrap_net.GetWorldSize();

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context;
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  RdmaDevice* device = rdma_devices[1];
  RdmaDeviceContext* device_context = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.port_id = 1;
  config.gid_index = 1;
  config.max_msgs_num = 10;
  config.max_cqe_num = 256;
  config.alignment = 4096;
  config.on_gpu = on_gpu;
  RdmaEndpoint endpoint = device_context->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  RdmaEndpointHandle global_rdma_ep_handles[world_size];
  bootstrap_net.Allgather(&endpoint.handle, global_rdma_ep_handles, sizeof(endpoint.handle));

  std::cout << "Local rank " << local_rank << " " << endpoint.handle << std::endl;

  for (int i = 0; i < world_size; i++) {
    if (i == local_rank) continue;
    device_context->ConnectEndpoint(endpoint.handle, global_rdma_ep_handles[i]);
    std::cout << "Local rank " << local_rank << " received " << global_rdma_ep_handles[i]
              << std::endl;
  }

  // 4 Register buffer
  void* buffer;
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, allreduce_size));
  HIP_RUNTIME_CHECK(hipMemset(buffer, local_rank, allreduce_size));
  // assert(!posix_memalign(&buffer_1, 4096, allreduce_size));
  // memset(buffer_1, 1, allreduce_size);
  MemoryRegion mr_handle =
      device_context->RegisterMemoryRegion(buffer, allreduce_size, MR_ACCESS_FLAG);
  MemoryRegion global_mr_handles[world_size];
  bootstrap_net.Allgather(&mr_handle, global_mr_handles, sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;
  //   printf("Before Buffer 2 0th %d 512th %d\n", ((char*)buffer_2)[0], ((char*)buffer_2)[512]);

  // 5 Prepare kernel argument
  printf("Before: Local rank %d val %d\n", local_rank, ((char*)buffer)[256]);

  if (local_rank == 0) {
    IbgdaReadWriteReq rreq;
    rreq.qp_handle.qpn = endpoint.handle.qpn;
    rreq.qp_handle.post_idx = 0;
    rreq.qp_handle.queue_buff_addr = endpoint.wq_handle.sq_addr;
    rreq.qp_handle.dbr_rec_addr = endpoint.wq_handle.dbr_rec_addr;
    rreq.qp_handle.dbr_addr = endpoint.wq_handle.dbr_addr;
    rreq.local_mr = mr_handle;
    rreq.remote_mr = global_mr_handles[1];
    rreq.bytes_count = allreduce_size;

    uint64_t dbr_val = PostRead<ProviderType::MLX5>(rreq);
    // rreq.qp_handle.post_idx = 1;
    udma_to_device_barrier();
    UpdateSendDbrRecord<ProviderType::MLX5>(endpoint.wq_handle.dbr_rec_addr,
                                            rreq.qp_handle.post_idx);
    udma_to_device_barrier();
    RingDoorbell<ProviderType::MLX5>(endpoint.wq_handle.dbr_addr, dbr_val);
    udma_to_device_barrier();

    endpoint.cq_handle.consumer_idx = 0;
    int snd_opcode = PoolCq<ProviderType::MLX5>(endpoint.cq_handle);
    printf("snd opcode %d\n", snd_opcode);
  }

  printf("After: Local rank %d val %d\n", local_rank, ((char*)buffer)[256]);

  bootstrap_net.Barrier();
  bootstrap_net.Finalize();
}

int main() { LocalRdmaOps(); }