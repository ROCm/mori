#include "mori/application/application.hpp"

using namespace mori;
using namespace mori::application;

template <int GpuPerNode, int NodeNum>
void HierarchicalRingAllReduce() {
  MpiBootstrapNetwork bootstrap_net;
  bootstrap_net.Initialize();

  int local_rank = bootstrap_net.GetLocalRank();
  int world_size = bootstrap_net.GetWorldSize();

  int node_id = local_rank / GpuPerNode;
  int gpu_id = local_rank % GpuPerNode;

  // TODO P2P initialization

  // RDMA initialization
  transport::rdma::RdmaContext rdma_context;

  transport::rdma::RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  transport::rdma::RdmaDevice* device_0 = rdma_devices[0];
  std::cout << "Get device num " << rdma_devices.size() << std::endl;
  std::cout << "Get device 0 port num " << device_0->GetDevicePortNum() << std::endl;

  void* mem = malloc(1024);

  transport::rdma::RdmaDeviceContext* device_0_context = device_0->CreateRdmaDeviceContext();
  device_0_context->RegisterMemoryRegion(mem, 1024,
                                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                             IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  device_0_context->DeRegisterMemoryRegion(mem);

  transport::rdma::RdmaEndpointConfig config;
  config.max_cqe_num = 16;
  config.sq_max_wqe_num = 16;
  config.rq_max_wqe_num = 16;
  config.alignment = 4096;
  device_0_context->CreateRdmaEndpoint(config);
}

int main() { HierarchicalRingAllReduce<1, 1>(); }