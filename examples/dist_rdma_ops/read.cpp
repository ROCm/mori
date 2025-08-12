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

__global__ void CheckBufferKernel(const char* buffer, size_t numElems, char expected) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElems) {
    char val = buffer[idx];
    if (val != expected) {
      // printf("Mismatch at index %zu: expected=%d, got=%d\n", idx, expected, val);
      assert(false && "Buffer mismatch detected!");
    }
  }
}

void VerifyBuffer(void* buffer, size_t maxSize, char expected) {
  size_t numElems = maxSize / sizeof(char);

  int threadsPerBlock = 256;
  int blocks = (static_cast<int>(numElems) + threadsPerBlock - 1) / threadsPerBlock;

  CheckBufferKernel<<<blocks, threadsPerBlock>>>(reinterpret_cast<char*>(buffer), numElems,
                                                 expected);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
}

__device__ void Quite(RdmaEndpoint* endpoint) {
  constexpr size_t BROADCAST_SIZE = 1024 / warpSize;
  __shared__ uint64_t wqe_broadcast[BROADCAST_SIZE];
  uint8_t warp_id = FlatBlockThreadId() / warpSize;
  wqe_broadcast[warp_id] = 0;

  uint64_t activemask = GetActiveLaneMask();
  uint8_t num_active_lanes = GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = GetFirstActiveLaneID(activemask);
  CompletionQueueHandle* cqHandle = &endpoint->cqHandle;

  while (true) {
    bool done{false};
    uint32_t quiet_amount{0};
    uint32_t warp_cq_consumer{0};
    // printf("quiet_amount =  %u\n", quiet_amount);
    while (!done) {
      // printf("is_leader %d\n",is_leader);
      uint32_t posted =
          __hip_atomic_load(&cqHandle->needConsIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t active =
          __hip_atomic_load(&cqHandle->activeIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t completed =
          __hip_atomic_load(&cqHandle->consIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      if (!(posted - completed)) {
        return;
      }
      uint32_t quiet_val = posted - active;

      if (!quiet_val) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);
      if (is_leader) {
        // printf("posted %u, active %u, completed %u\n", posted, active, completed);
        done = __hip_atomic_compare_exchange_strong(&cqHandle->activeIdx, &active,
                                                    active + quiet_amount, __ATOMIC_SEQ_CST,
                                                    __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
        if (done) {
          warp_cq_consumer = active;
        }
      }
      done = __shfl(done, leader_phys_lane_id);
    }
    warp_cq_consumer = __shfl(warp_cq_consumer, leader_phys_lane_id);
    uint32_t my_cq_consumer = warp_cq_consumer + my_logical_lane_id;
    uint32_t my_cq_index = my_cq_consumer % cqHandle->cqeNum;

    if (my_logical_lane_id < quiet_amount) {
      uint16_t wqe_counter;
      PollCq<ProviderType::MLX5>(cqHandle->cqAddr, cqHandle->cqeNum, &my_cq_consumer, &wqe_counter);
      __threadfence_system();
      uint64_t wqe_id = endpoint->wqHandle.outstandingWqe[wqe_counter];
      __hip_atomic_fetch_max(&wqe_broadcast[warp_id], wqe_id, __ATOMIC_SEQ_CST,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
    }
    if (is_leader) {
      uint64_t completed{0};
      do {
        completed =
            __hip_atomic_load(&cqHandle->consIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      } while (completed != warp_cq_consumer);
      UpdateCqDbrRecord<core::ProviderType::MLX5>(cqHandle->dbrRecAddr,
                                                  (uint32_t)(warp_cq_consumer + quiet_amount));
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      uint64_t doneIdx = wqe_broadcast[warp_id];
      __hip_atomic_store(&endpoint->wqHandle.doneIdx, doneIdx, __ATOMIC_SEQ_CST,
                         __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&cqHandle->consIdx, quiet_amount, __ATOMIC_SEQ_CST,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

__global__ void Write(RdmaEndpoint* endpoint, RdmaMemoryRegion localMr, RdmaMemoryRegion remoteMr,
                      size_t msg_size, int iters) {
  for (int i = 0; i < iters; i++) {
    uint64_t activemask = GetActiveLaneMask();
    uint8_t num_active_lanes = GetActiveLaneCount(activemask);
    uint8_t my_logical_lane_id = GetActiveLaneNum(activemask);
    bool is_leader{my_logical_lane_id == 0};
    const uint64_t leader_phys_lane_id = GetFirstActiveLaneID(activemask);
    uint8_t num_wqes{num_active_lanes};
    uint64_t warp_sq_counter{0};
    WorkQueueHandle* wqHandle = &endpoint->wqHandle;
    CompletionQueueHandle* cqHandle = &endpoint->cqHandle;

    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wqHandle->postIdx, num_wqes, __ATOMIC_SEQ_CST,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id;
    uint64_t my_sq_index = my_sq_counter % wqHandle->sqWqeNum;
    while (true) {
      uint64_t db_touched =
          __hip_atomic_load(&wqHandle->dbTouchIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t db_done =
          __hip_atomic_load(&wqHandle->doneIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t num_active_sq_entries = db_touched - db_done;
      uint64_t num_free_entries = min(wqHandle->sqWqeNum, cqHandle->cqeNum) - num_active_sq_entries;
      uint64_t num_entries_until_wave_last_entry = warp_sq_counter + num_active_lanes - db_touched;
      if (num_free_entries > num_entries_until_wave_last_entry) {
        break;
      }
      Quite(endpoint);
    }
    wqHandle->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    uintptr_t srcAddr = localMr.addr + FlatThreadId() * msg_size;
    uintptr_t dstAddr = remoteMr.addr + FlatThreadId() * msg_size;
    uint64_t dbr_val = PostWrite<ProviderType::MLX5>(
        wqHandle->sqAddr, wqHandle->sqWqeNum, nullptr, my_sq_counter, endpoint->handle.qpn, srcAddr,
        localMr.lkey, dstAddr, remoteMr.rkey, msg_size);

    if (is_leader) {
      uint64_t db_touched{0};
      do {
        db_touched =
            __hip_atomic_load(&wqHandle->dbTouchIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != warp_sq_counter);

      uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wqHandle->sqAddr);
      uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
          &base_ptr[64 * ((warp_sq_counter + num_wqes - 1) % wqHandle->sqWqeNum)]);
      UpdateSendDbrRecord<ProviderType::MLX5>(wqHandle->dbrRecAddr, warp_sq_counter + num_wqes);
      // __threadfence_system();
      RingDoorbell<ProviderType::MLX5>(wqHandle->dbrAddr, *ctrl_wqe_8B_for_db);

      __hip_atomic_fetch_add(&cqHandle->needConsIdx, num_wqes, __ATOMIC_SEQ_CST,
                             __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&wqHandle->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_SEQ_CST,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    Quite(endpoint);
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
  size_t blocks = args.getNumBlocks();
  size_t threads = args.getThreadsPerBlock();
  int validSizeLog = 0;
  size_t warmupIters = args.getWarmupIters();
  size_t iters = args.getIters();
  float milliseconds;
  int local_rank = bootNet.GetLocalRank();
  int world_size = bootNet.GetWorldSize();
  HIP_RUNTIME_CHECK(hipSetDevice(local_rank));
  hipEvent_t start, end;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&end));

  // RDMA initialization
  // 1 Create device
  RdmaContext rdma_context(RdmaBackendType::DirectVerbs);
  RdmaDeviceList rdma_devices = rdma_context.GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdma_devices);
  RdmaDevice* device = activeDevicePortList[local_rank % activeDevicePortList.size()].first;
  std::cout << "localRank " << local_rank << " select device " << device->Name() << std::endl;

  RdmaDeviceContext* device_context = device->CreateRdmaDeviceContext();

  // 2 Create an endpoint
  RdmaEndpointConfig config;
  config.portId = activeDevicePortList[local_rank % activeDevicePortList.size()].second;
  config.gidIdx = 3;
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 4096;
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
  size_t totalSize = maxSize * blocks * threads;
  assert(totalSize <= 0x800000000ULL && "Error: totalSize cannot exceed 32GB!");
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, totalSize));
  HIP_RUNTIME_CHECK(hipMemset(buffer, local_rank, totalSize));

  // assert(!posix_memalign(&buffer_1, 4096, allreduce_size));
  // memset(buffer_1, 1, allreduce_size);
  RdmaMemoryRegion mr_handle =
      device_context->RegisterRdmaMemoryRegion(buffer, totalSize, MR_ACCESS_FLAG);
  std::vector<RdmaMemoryRegion> global_mr_handles(world_size);
  bootNet.Allgather(&mr_handle, global_mr_handles.data(), sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;
  RdmaEndpoint* devEndpoint;
  HIP_RUNTIME_CHECK(hipMalloc(&devEndpoint, sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(hipMemcpy(devEndpoint, &endpoint, sizeof(RdmaEndpoint), hipMemcpyHostToDevice));

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
  // printf("Before: Local rank %d val %d\n", local_rank, ((char*)buffer)[0]);

  for (size_t size = minSize; size <= maxSize; size *= stepFactor) {
    if (local_rank == 0) {
      Write<<<blocks, threads>>>(devEndpoint, global_mr_handles[0], global_mr_handles[1], size, 1);
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    }
    bootNet.Barrier();
    if (local_rank == 1) {
      VerifyBuffer(reinterpret_cast<char*>(buffer), size, 0);
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    }
    bootNet.Barrier();
  }
  printf("rank %d data verify is done\n", local_rank);

  if (local_rank == 0) {
    for (size_t size = minSize; size <= maxSize; size *= stepFactor) {
      // warmup
      Write<<<blocks, threads>>>(devEndpoint, global_mr_handles[0], global_mr_handles[1], size,
                                 warmupIters);
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());

      // test and record
      HIP_RUNTIME_CHECK(hipEventRecord(start));
      Write<<<blocks, threads>>>(devEndpoint, global_mr_handles[0], global_mr_handles[1], size,
                                 iters);
      HIP_RUNTIME_CHECK(hipEventRecord(end));
      HIP_RUNTIME_CHECK(hipEventSynchronize(end));
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&milliseconds, start, end));
      times[validSizeLog] = milliseconds;
      sizeTable[validSizeLog] = size;
      bwTable[validSizeLog] =
          size * blocks * threads / (milliseconds * (B_TO_GB / (iters * MS_TO_S)));
      validSizeLog++;
    }
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }
  bootNet.Barrier();
  // printf("After: Local rank %d val %d %d\n", local_rank,
  // ((char*)buffer)[0],((char*)buffer)[maxSize/sizeof(char)-1]);

  if (local_rank == 0) {
    printf("\nIBGDA Wite benchmark:\n");
    printf("%-8s %-12s %-12s %-12s\n", "Index", "Size(B)", "bw(GB)", "Time(ms)");

    for (size_t i = 0; i < validSizeLog; ++i) {
      printf("%-8zu %-12lu %-12.4f %-12.4f\n", i + 1, sizeTable[i], bwTable[i], times[i]);
    }
  }

  bootNet.Finalize();
  HIP_RUNTIME_CHECK(hipFree(buffer));
  HIP_RUNTIME_CHECK(hipFree(devEndpoint));
  HIP_RUNTIME_CHECK(hipHostFree(bwTable));
  HIP_RUNTIME_CHECK(hipHostFree(sizeTable));
  HIP_RUNTIME_CHECK(hipHostFree(times));
}

int main(int argc, char* argv[]) { distRdmaOps(argc, argv); }