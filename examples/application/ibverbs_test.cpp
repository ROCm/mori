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
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
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
