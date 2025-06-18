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

template <ProviderType PrvdType, typename T>
__global__ void Atomic(RdmaEndpoint endpoint, MemoryRegion localMr, MemoryRegion remoteMr,
                       atomicType amoOp, int iters) {
  T value = 1;
  for (int i = 0; i < iters; i++) {
    uint64_t dbr_val =
        PostAtomic<PrvdType, T>(endpoint.wqHandle.sqAddr, endpoint.wqHandle.sqWqeNum,
                                &endpoint.wqHandle.postIdx, endpoint.handle.qpn, localMr.addr,
                                localMr.lkey, remoteMr.addr, remoteMr.rkey, value, value, amoOp);
    __threadfence_system();
    UpdateDbrAndRingDbSend<PrvdType>(endpoint.wqHandle.dbrRecAddr, endpoint.wqHandle.postIdx,
                                     endpoint.wqHandle.dbrAddr, dbr_val,
                                     &endpoint.wqHandle.postSendLock);
    __threadfence_system();
    int snd_opcode = PollCqAndUpdateDbr<ProviderType::MLX5>(
        endpoint.cqHandle.cqAddr, endpoint.cqHandle.cqeSize, endpoint.cqHandle.cqeNum,
        &endpoint.cqHandle.consIdx, endpoint.cqHandle.dbrRecAddr, &endpoint.cqHandle.pollCqLock);

    printf("postIdx: %d, consIdx: %d\n", endpoint.wqHandle.postIdx, endpoint.cqHandle.consIdx);
  }
}

void launchAtomicKernel(const Datatype& dt, ProviderType pType, hipStream_t stream,
                        RdmaEndpoint endpoint, MemoryRegion localMr, MemoryRegion remoteMr,
                        atomicType amoOp, int iters) {
  dim3 block(1);
  dim3 thread(1);
  switch (pType) {  
    case ProviderType::MLX5:  
      switch (dt.type) {  
        case MORIDataType::MORI_INT32:  
          hipLaunchKernelGGL((Atomic<ProviderType::MLX5, int32_t>),  
                             block, thread, 0, stream,  
                             endpoint, localMr, remoteMr, amoOp, iters);  
          break;  
        case MORIDataType::MORI_INT64:  
          hipLaunchKernelGGL((Atomic<ProviderType::MLX5, int64_t>),  
                             block, thread, 0, stream,  
                             endpoint, localMr, remoteMr, amoOp, iters);  
          break;  
        case MORIDataType::MORI_UINT32:  
          hipLaunchKernelGGL((Atomic<ProviderType::MLX5, uint32_t>),  
                             block, thread, 0, stream,  
                             endpoint, localMr, remoteMr, amoOp, iters);  
          break;  
        case MORIDataType::MORI_UINT64:  
          hipLaunchKernelGGL((Atomic<ProviderType::MLX5, uint64_t>),  
                             block, thread, 0, stream,  
                             endpoint, localMr, remoteMr, amoOp, iters);  
          break;  
        default:  
          std::cerr << "Unknown MORIDataType, cannot launch Atomic kernel!\n";  
          return;  
      }  
      break;  
    default:  
      std::cerr << "Unsupported ProviderType in launchAtomicKernel\n";  
      return;  
  }  
}

void distRdmaOps(int argc, char* argv[]) {
  BenchmarkConfig args;
  args.readArgs(argc, argv);

  MpiBootstrapNetwork bootNet(MPI_COMM_WORLD);
  bootNet.Initialize();

  bool on_gpu = true;
  size_t warmupIters = args.getWarmupIters();
  size_t iters = args.getIters();
  Datatype dt = args.getDatatype();
  AMO amo = args.getTestAMO();
  float milliseconds;
  double bw;
  float time;
  int local_rank = bootNet.GetLocalRank();
  int world_size = bootNet.GetWorldSize();

  int hipDevCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&hipDevCount));
  HIP_RUNTIME_CHECK(hipSetDevice((local_rank + 2) % hipDevCount));

  hipEvent_t start, end;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&end));

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context;
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdma_devices);
  assert(!activeDevicePortList.empty());
  ActiveDevicePort devicePort = activeDevicePortList[0];

  RdmaDevice* device = devicePort.first;
  RdmaDeviceContext* device_context = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = 1;
  config.gidIdx = 3;
  config.maxMsgsNum = 10;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = on_gpu;
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

  // 4 Register buffer
  void* buffer;
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, 8));
  HIP_RUNTIME_CHECK(hipMemset(buffer, 0, 8));

  MemoryRegion mr_handle = device_context->RegisterMemoryRegion(buffer, 8, MR_ACCESS_FLAG);
  std::vector<MemoryRegion> global_mr_handles(world_size);
  bootNet.Allgather(&mr_handle, global_mr_handles.data(), sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;

  // 5 Prepare kernel argument
  printf("Before: Local rank %d val %lu\n", local_rank, ((uint64_t*)buffer)[0]);
  hipStream_t myStream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&myStream));

  if (local_rank == 0) {
    // warmup
    launchAtomicKernel(dt, ProviderType::MLX5, myStream, endpoint, global_mr_handles[0],
                       global_mr_handles[1], amo.type, warmupIters);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    // test and record
    HIP_RUNTIME_CHECK(hipEventRecord(start, myStream));
    launchAtomicKernel(dt, ProviderType::MLX5, myStream, endpoint, global_mr_handles[0],
                       global_mr_handles[1], amo.type, iters);
    HIP_RUNTIME_CHECK(hipEventRecord(end, myStream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(myStream));
    HIP_RUNTIME_CHECK(hipEventElapsedTime(&milliseconds, start, end));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    time = milliseconds / iters;
    bw = dt.size / (milliseconds * (B_TO_GB / (iters * MS_TO_S)));
  }
  bootNet.Barrier();
  printf("After: Local rank %d val %lu\n", local_rank, ((uint64_t*)buffer)[0]);

  if (local_rank == 0) {
    printf("\nIters\tsize\t\tbw\ttimes\n");
    printf("%zu\t%lu\t%f\t%f\n", iters, dt.size, bw, time);
  }

  bootNet.Finalize();
  HIP_RUNTIME_CHECK(hipFree(buffer));
}

int main(int argc, char* argv[]) { distRdmaOps(argc, argv); }