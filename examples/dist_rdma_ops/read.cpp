#include <hip/hip_runtime.h>
#include <mpi.h>

#include "args_parser.hpp"
#include "mori/application/application.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/core.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core;

#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

__global__ void CheckBufferKernel(const uint32_t* buffer, size_t numElems, uint32_t expected) {  
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < numElems) {  
        uint32_t val = buffer[idx];  
        if (val != expected) {  
            printf("Mismatch at index %zu: expected=%u, got=%u\n", idx, expected, val);  
        }  
    }  
}  
   
void VerifyBuffer(void* buffer, size_t maxSize, uint32_t expected) {  
    size_t numElems = maxSize / sizeof(uint32_t);  
  
    int threadsPerBlock = 256;  
    int blocks = (static_cast<int>(numElems) + threadsPerBlock - 1) / threadsPerBlock;  
  
    CheckBufferKernel<<<blocks, threadsPerBlock>>>(  
        reinterpret_cast<uint32_t*>(buffer),  
        numElems,  
        expected  
    );  
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());  
} 

__global__ void Write(RdmaEndpoint endpoint, MemoryRegion localMr, MemoryRegion remoteMr,
                      size_t msg_size, int iters) {
  // TODO: check Whether pass by value not by pointer
  printf("in kernel iter:%d msg_size:%zu localMr.addr: %lu remoteMr.addr:%lu\n", iters, msg_size,
         localMr.addr, remoteMr.addr);
  for (int i = 0; i < iters; i++) {
    uint64_t dbr_val = PostWrite<ProviderType::MLX5>(
        endpoint.wqHandle.sqAddr, endpoint.wqHandle.sqWqeNum, &endpoint.wqHandle.postIdx,
        endpoint.handle.qpn, localMr.addr, localMr.lkey, remoteMr.addr, remoteMr.rkey, msg_size);
    __threadfence_system();
    UpdateDbrAndRingDbSend<ProviderType::MLX5>(endpoint.wqHandle.dbrRecAddr,
                                               endpoint.wqHandle.postIdx, endpoint.wqHandle.dbrAddr,
                                               dbr_val, &endpoint.wqHandle.postSendLock);
    __threadfence_system();
    PollCqAndUpdateDbr<ProviderType::MLX5>(
        endpoint.cqHandle.cqAddr, endpoint.cqHandle.cqeSize, endpoint.cqHandle.cqeNum,
        &endpoint.cqHandle.consIdx, endpoint.cqHandle.dbrRecAddr, &endpoint.cqHandle.pollCqLock);

    // printf("postIdx: %d, consIdx: %d\n", endpoint.wqHandle.postIdx, endpoint.cqHandle.consIdx);
  }
}

void distRdmaOps(int argc, char* argv[]) {
  BenchmarkConfig args;
  args.readArgs(argc, argv);

  MpiBootstrapNetwork bootNet(MPI_COMM_WORLD);
  bootNet.Initialize();

  bool on_gpu = true;
  size_t minSize = args.getMinSize();
  size_t maxSize = args.getMaxSize();
  size_t stepFactor = args.getStepFactor();
  size_t maxSizeLog = args.getMaxSizeLog();
  int validSizeLog = 0;
  size_t warmupIters = args.getWarmupIters();
  size_t iters = args.getIters();
  float milliseconds;
  int local_rank = bootNet.GetLocalRank();
  int world_size = bootNet.GetWorldSize();
  HIP_RUNTIME_CHECK(hipSetDevice(local_rank + 2));
  hipEvent_t start, end;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&end));

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
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, maxSize));
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buffer), local_rank, maxSize/sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  // assert(!posix_memalign(&buffer_1, 4096, allreduce_size));
  // memset(buffer_1, 1, allreduce_size);
  MemoryRegion mr_handle = device_context->RegisterMemoryRegion(buffer, maxSize, MR_ACCESS_FLAG);
  MemoryRegion global_mr_handles[world_size];
  bootNet.Allgather(&mr_handle, global_mr_handles, sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;
  //   printf("Before Buffer 2 0th %d 512th %d\n", ((char*)buffer_2)[0], ((char*)buffer_2)[512]);

  double* bwTable;
  uint64_t* sizeTable;
  float* times;
  HIP_RUNTIME_CHECK(hipHostAlloc(&bwTable, maxSizeLog * sizeof(double), hipHostAllocMapped));
  memset(bwTable, 0, maxSizeLog * sizeof(double));
  HIP_RUNTIME_CHECK(hipHostAlloc(&sizeTable, maxSizeLog * sizeof(uint64_t), hipHostAllocMapped));
  memset(sizeTable, 0, maxSizeLog * sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipHostAlloc(&times, maxSizeLog * sizeof(float), hipHostAllocMapped));
  memset(times, 0, maxSizeLog * sizeof(float));
  // 5 Prepare kernel argument
  printf("Before: Local rank %d val %d\n", local_rank, ((char*)buffer)[0]);

  for (size_t size = minSize; size <= maxSize; size *= stepFactor) {
    if (local_rank == 0)
    {
      Write<<<1, 1>>>(endpoint, global_mr_handles[0], global_mr_handles[1], size, 1);
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    }
    bootNet.Barrier();
    if (local_rank == 1)
    {
      VerifyBuffer(reinterpret_cast<uint32_t*>(buffer), size, 0);
    }
    bootNet.Barrier();
  }

  if (local_rank == 0) {
    printf("out kernel %p\n", endpoint.wqHandle.sqAddr);
    for (size_t size = minSize; size <= maxSize; size *= stepFactor) {
      // warmup
      Write<<<1, 1>>>(endpoint, global_mr_handles[0], global_mr_handles[1], size, warmupIters);
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());

      // test and record
      HIP_RUNTIME_CHECK(hipEventRecord(start));
      Write<<<1, 1>>>(endpoint, global_mr_handles[0], global_mr_handles[1], size, iters);
      HIP_RUNTIME_CHECK(hipEventRecord(end));
      HIP_RUNTIME_CHECK(hipEventSynchronize(end));
      // HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&milliseconds, start, end));
      times[validSizeLog] = milliseconds;
      sizeTable[validSizeLog] = size;
      bwTable[validSizeLog] = size / (milliseconds * (B_TO_GB / (iters * MS_TO_S)));
      validSizeLog++;
    }
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }
  bootNet.Barrier();

  printf("After: Local rank %d val %d %d\n", local_rank, ((uint32_t*)buffer)[0],((uint32_t*)buffer)[maxSize/sizeof(uint32_t)-1]);
  if (local_rank == 0) {
    printf("Index\tsize\t\tbw\ttimes\n");
    for (size_t i = 0; i < validSizeLog; ++i) {
      printf("%zu\t%lu\t%f\t%f\n", i, sizeTable[i], bwTable[i], times[i]);
    }
  }

  bootNet.Finalize();
  HIP_RUNTIME_CHECK(hipFree(buffer));
  HIP_RUNTIME_CHECK(hipHostFree(bwTable));
  HIP_RUNTIME_CHECK(hipHostFree(sizeTable));
  HIP_RUNTIME_CHECK(hipHostFree(times));
}

int main(int argc, char* argv[]) { distRdmaOps(argc, argv); }