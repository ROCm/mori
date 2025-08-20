#include <hip/hip_runtime.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core;

int main(int argc, char* argv[]) {
  MpiBootstrapNetwork bootNet(MPI_COMM_WORLD);
  bootNet.Initialize();
  int local_rank = bootNet.GetLocalRank();
  int world_size = bootNet.GetWorldSize();

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context(RdmaBackendType::IBVerbs);
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdma_devices);
  RdmaDevice* device = activeDevicePortList[0].first;

  RdmaDeviceContext* device_context = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = activeDevicePortList[0].second;
  config.gidIdx = 1;
  config.maxMsgsNum = 200;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  config.onGpu = false;
  RdmaEndpoint endpoint = device_context->CreateRdmaEndpoint(config);

  // 3 Allgather global endpoint and connect
  std::vector<RdmaEndpointHandle> global_rdma_ep_handles(world_size);
  bootNet.Allgather(&endpoint.handle, global_rdma_ep_handles.data(), sizeof(RdmaEndpointHandle));

  std::cout << "Local rank " << local_rank << " " << endpoint.handle << std::endl;

  for (int i = 0; i < world_size; i++) {
    if (i == local_rank) continue;
    device_context->ConnectEndpoint(endpoint.handle, global_rdma_ep_handles[i]);
    std::cout << "Local rank " << local_rank << " received " << global_rdma_ep_handles[i]
              << std::endl;
  }
}
