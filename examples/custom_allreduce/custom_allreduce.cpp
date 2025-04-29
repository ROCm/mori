#include "mori/application/application.hpp"

using namespace mori;
using namespace mori::application;

__global__ void AllReduceOverRdmaKernel() {}

void AllReduceOverRdma() {
  MpiBootstrapNetwork bootstrap_net;
  bootstrap_net.Initialize();

  int local_rank = bootstrap_net.GetLocalRank();
  int world_size = bootstrap_net.GetWorldSize();
  std::cout << "Local Rank: " << local_rank << std::endl;
  std::cout << "World Size: " << world_size << std::endl;

  // RDMA initialization
  // 1 Create device
  transport::rdma::RdmaContext rdma_context;
  transport::rdma::RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  transport::rdma::RdmaDevice* device_0 = rdma_devices[0];

  // 2 Register buffer
  void* mem = malloc(1024);
  transport::rdma::RdmaDeviceContext* device_0_context = device_0->CreateRdmaDeviceContext();
  device_0_context->RegisterMemoryRegion(mem, 1024,
                                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                             IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  // 3 Create an endpoint
  transport::rdma::RdmaEndpointConfig config;
  config.port_id = 1;
  config.gid_index = 1;
  config.max_msgs_num = 16;
  config.max_recv_sge = 16;
  config.max_cqe_num = 16;
  config.alignment = 4096;
  transport::rdma::RdmaEndpoint endpoint = device_0_context->CreateRdmaEndpoint(config);

  // 4 Allgather global endpoint
  transport::rdma::RdmaEndpointHandle global_rdma_ep_handles[world_size];
  bootstrap_net.Allgather(&endpoint.handle, global_rdma_ep_handles, sizeof(endpoint.handle));

  printf("Rank %d ready to connect, qpn %d\n", local_rank, endpoint.handle.qpn);
  for (int i = 0; i < world_size; i++) {
    if (i == local_rank) continue;
    device_0_context->ConnectEndpoint(endpoint.handle, global_rdma_ep_handles[i]);
  }

  device_0_context->DeRegisterMemoryRegion(mem);

  // 5 Prepare kernel argument

  bootstrap_net.Finalize();
}

int main() { AllReduceOverRdma(); }