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

__global__ void Write(RdmaEndpoint endpoint, MemoryRegion localMr, MemoryRegion remoteMr,
                      int msg_size) {
  uint32_t postIdx = 0;
  printf("in kernel %p\n", endpoint.wqHandle.sqAddr);
  uint64_t dbr_val = PostWrite<ProviderType::MLX5>(
      endpoint.wqHandle.sqAddr, endpoint.wqHandle.sqWqeNum, &postIdx, endpoint.handle.qpn,
      localMr.addr, localMr.lkey, remoteMr.addr, remoteMr.rkey, msg_size);
  __threadfence_system();
  UpdateSendDbrRecord<ProviderType::MLX5>(endpoint.wqHandle.dbrRecAddr, postIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::MLX5>(endpoint.wqHandle.dbrAddr, dbr_val);
  __threadfence_system();

  uint32_t consIdx = 0;
  int snd_opcode = PollCq<ProviderType::MLX5>(endpoint.cqHandle.cqAddr, endpoint.cqHandle.cqeSize,
                                              endpoint.cqHandle.cqeNum, &consIdx);
}

void LocalRdmaOps() {
  MpiBootstrapNetwork bootNet(MPI_COMM_WORLD);
  bootNet.Initialize();

  bool on_gpu = true;
  int allreduce_size = 1024;
  int local_rank = bootNet.GetLocalRank();
  int world_size = bootNet.GetWorldSize();
  HIP_RUNTIME_CHECK(hipSetDevice(local_rank));

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context;
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  RdmaDevice* device = rdma_devices[1];
  RdmaDeviceContext* device_context = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = 1;
  config.gidIdx = 1;
  config.maxMsgsNum = 10;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = on_gpu;
  RdmaEndpoint endpoint = device_context->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  RdmaEndpointHandle global_rdma_ep_handles[world_size];
  bootNet.Allgather(&endpoint.handle, global_rdma_ep_handles, sizeof(RdmaEndpointHandle));

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
  bootNet.Allgather(&mr_handle, global_mr_handles, sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;
  //   printf("Before Buffer 2 0th %d 512th %d\n", ((char*)buffer_2)[0], ((char*)buffer_2)[512]);

  // 5 Prepare kernel argument
  printf("Before: Local rank %d val %d\n", local_rank, ((char*)buffer)[256]);

  if (local_rank == 0) {
    printf("out kernel %p\n", endpoint.wqHandle.sqAddr);
    Write<<<1, 1>>>(endpoint, global_mr_handles[0], global_mr_handles[1], allreduce_size);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }
  bootNet.Barrier();

  printf("After: Local rank %d val %d\n", local_rank, ((char*)buffer)[256]);
  bootNet.Finalize();

  MPI_Finalize();
}

int main() { LocalRdmaOps(); }