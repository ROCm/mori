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

template <ProviderType P>
inline __device__ void QuiteSerial(RdmaEndpoint* endpoint) {
  if (GetActiveLaneNum() != 0) return;
  CompletionQueueHandle& cq = endpoint->cqHandle;
  WorkQueueHandle& wq = endpoint->wqHandle;
  if (!AcquireLockOnce(&cq.pollCqLock)) return;
  while (true) {
    bool done{false};
    uint32_t quiet_amount{0};
    uint32_t my_cq_consumer{0};

    uint32_t dbTouchIdx =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t doneIdx = __hip_atomic_load(&wq.doneIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    // printf("dbTouchIdx: %u, doneIdx: %u\n", dbTouchIdx, doneIdx);
    if (dbTouchIdx == doneIdx) {
      ReleaseLock(&cq.pollCqLock);
      return;
    }

    my_cq_consumer =
        __hip_atomic_fetch_add(&cq.cq_consumer, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint16_t wqe_counter;
    uint64_t wqe_id;
    int opcode = core::PollCq<P>(cq.cqAddr, cq.cqeNum, &my_cq_consumer, &wqe_counter);
    // printf("wqe_counter: %u\n",wqe_counter);
    if constexpr (P == core::ProviderType::MLX5) {
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;
        core::DumpMlx5Wqe(wq.sqAddr, my_cq_index);
        assert(false);
      }
      wqe_id = wq.outstandingWqe[wqe_counter];
    } else if constexpr (P == core::ProviderType::BNXT) {
      if (opcode != BNXT_RE_REQ_ST_OK) {
        uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;
        assert(false);
      }
      wqe_counter = (wqe_counter + wq.sqWqeNum - 1) % wq.sqWqeNum;
      wqe_id = wq.outstandingWqe[wqe_counter] + 1;
    }

    // core::UpdateCqDbrRecord<P>(cq.dbrRecAddr, (uint32_t)(my_cq_consumer + 1), cq.cqeNum);

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_fetch_max(&wq.doneIdx, wqe_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  ReleaseLock(&cq.pollCqLock);
}

template <ProviderType P>
__device__ void Quite(RdmaEndpoint* endpoint) {
  constexpr size_t BROADCAST_SIZE = 1024 / warpSize;
  __shared__ uint64_t wqe_broadcast[BROADCAST_SIZE];
  uint8_t warp_id = FlatBlockThreadId() / warpSize;
  wqe_broadcast[warp_id] = 0;

  uint64_t activemask = GetActiveLaneMask();
  uint8_t num_active_lanes = GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == 0};
  bool is_last{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = GetFirstActiveLaneID(activemask);
  CompletionQueueHandle* cqHandle = &endpoint->cqHandle;
  uint8_t num_slot_per_wqe;
  if constexpr (P == ProviderType::MLX5) {
    num_slot_per_wqe = 1;
  } else if constexpr (P == ProviderType::BNXT) {
    num_slot_per_wqe = 3;
  }

  while (true) {
    bool done{false};
    uint32_t quiet_amount{0};
    uint32_t warp_cq_consumer{0};
    uint32_t my_cq_consumer{0};
    while (!done) {
      uint32_t posted =
          __hip_atomic_load(&cqHandle->needConsIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t active =
          __hip_atomic_load(&cqHandle->activeIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t completed =
          __hip_atomic_load(&cqHandle->consIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (!(posted - completed)) {
        return;
      }
      uint32_t quiet_val = posted - active;

      if (!quiet_val) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);
      if (is_leader) {
        done = __hip_atomic_compare_exchange_strong(&cqHandle->activeIdx, &active,
                                                    active + quiet_amount, __ATOMIC_RELAXED,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (done) {
          warp_cq_consumer = active;
        }
      }
      done = __shfl(done, leader_phys_lane_id);
      warp_cq_consumer = __shfl(warp_cq_consumer, leader_phys_lane_id);
      my_cq_consumer = warp_cq_consumer + my_logical_lane_id;
    }
    uint32_t my_cq_index = my_cq_consumer % cqHandle->cqeNum;
    uint64_t wqe_id;
    if (my_logical_lane_id < quiet_amount) {
      uint16_t wqe_counter;
      PollCq<P>(cqHandle->cqAddr, cqHandle->cqeNum, &my_cq_consumer, &wqe_counter);
      if constexpr (P == ProviderType::BNXT) {
        wqe_counter = (num_slot_per_wqe * (wqe_counter + endpoint->wqHandle.sqWqeNum - 1) %
                       endpoint->wqHandle.sqWqeNum);
      }
      __threadfence_system();
      wqe_id = endpoint->wqHandle.outstandingWqe[wqe_counter];
      __hip_atomic_fetch_max(&wqe_broadcast[warp_id], wqe_id, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    if (is_leader) {
      uint64_t completed{0};
      do {
        completed =
            __hip_atomic_load(&cqHandle->consIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (completed != warp_cq_consumer);
      UpdateCqDbrRecord<P>(cqHandle->dbrRecAddr, (uint32_t)(warp_cq_consumer + quiet_amount),
                           cqHandle->cqeNum);

      uint64_t doneIdx = wqe_broadcast[warp_id];
      __hip_atomic_fetch_max(&endpoint->wqHandle.doneIdx, doneIdx, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&cqHandle->consIdx, quiet_amount, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

template <ProviderType P>
__device__ void Write(RdmaEndpoint* endpoint, RdmaMemoryRegion localMr, RdmaMemoryRegion remoteMr,
                      size_t msg_size) {
  uint64_t activemask = GetActiveLaneMask();
  uint8_t num_active_lanes = GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = GetLastActiveLaneID(activemask);

  uint8_t num_wqes = num_active_lanes;

  uint64_t warp_sq_counter{0};
  uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
  WorkQueueHandle* wqHandle = &endpoint->wqHandle;
  CompletionQueueHandle* cqHandle = &endpoint->cqHandle;
  uint32_t psnCnt = (msg_size + wqHandle->mtuSize - 1) / wqHandle->mtuSize;
  if (is_leader) {
    core::atomic_add_packed_msn_and_psn(&wqHandle->msnPack, num_wqes, psnCnt * num_wqes,
                                        &warp_msntbl_counter, &warp_psn_counter);
    // TODO: if warp_msntbl_counter overflow 32bit, sq_slot's caculation will be wrong
    warp_sq_counter = warp_msntbl_counter;
    __hip_atomic_fetch_max(&wqHandle->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
  }
  warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
  warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
  warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
  uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id;
  uint64_t my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
  uint64_t my_psn_counter = warp_psn_counter + my_logical_lane_id * psnCnt;
  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wqHandle->dbTouchIdx, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done =
        __hip_atomic_load(&wqHandle->doneIdx, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wqHandle->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) {
      break;
    }
    if constexpr (P == ProviderType::MLX5) {
      Quite<P>(endpoint);
    } else if constexpr (P == ProviderType::BNXT) {
      QuiteSerial<P>(endpoint);
    }
  }
  if constexpr (P == ProviderType::MLX5) {
    wqHandle->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  } else if constexpr (P == ProviderType::BNXT) {
    wqHandle->outstandingWqe[my_sq_counter % wqHandle->sqWqeNum] = my_sq_counter;
  }
  uintptr_t srcAddr = localMr.addr + FlatThreadId() * msg_size;
  uintptr_t dstAddr = remoteMr.addr + FlatThreadId() * msg_size;
  uint64_t dbr_val =
      PostWrite<P>(*wqHandle, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                   endpoint->handle.qpn, srcAddr, localMr.lkey, dstAddr, remoteMr.rkey, msg_size);

  if (is_leader) {
    uint64_t db_touched{0};
    do {
      db_touched =
          __hip_atomic_load(&wqHandle->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    // uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wqHandle->sqAddr);
    // uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
    //     &base_ptr[64 * ((warp_sq_counter + num_wqes - 1) % wqHandle->sqWqeNum)]);
    UpdateSendDbrRecord<P>(wqHandle->dbrRecAddr, warp_sq_counter + num_wqes);
    // __threadfence_system();
    RingDoorbell<P>(wqHandle->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cqHandle->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wqHandle->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
}

template <ProviderType P>
__global__ void MultiQpWrite(RdmaEndpoint* endpoints, RdmaMemoryRegion localMr,
                             RdmaMemoryRegion remoteMr, size_t msg_size, int iters,
                             uint32_t* blockSync, int num_qp) {
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = (blockDim.x + warpSize - 1) / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int qp_id = globalWarpId % num_qp;
  for (int i = 0; i < iters; i++) {
    for (int qp_offset = qp_id; qp_offset < num_qp; qp_offset += globalWarpNum) {
      Write<P>(endpoints + qp_id, localMr, remoteMr, msg_size);
    }
    if constexpr (P == ProviderType::MLX5) {
      Quite<P>(endpoints + qp_id);
    } else if constexpr (P == ProviderType::BNXT) {
      for (int t = globalWarpId; t < num_qp; t += globalWarpNum) {
        // printf("qp_offset:%d\n",qp_offset);
        QuiteSerial<P>(endpoints + t);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(blockSync + i, 1);
    }
    while (atomicAdd(blockSync + i, 0) < gridDim.x) {
      ;
    }
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
  int num_qp = args.getNumQp();

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
  config.maxMsgsNum = 10000;
  config.maxCqeNum = 1;
  config.alignment = 4096;
  config.onGpu = on_gpu;
  std::vector<RdmaEndpoint> endpoints;
  for (int i = 0; i < num_qp; ++i) {
    endpoints.push_back(device_context->CreateRdmaEndpoint(config));
  }

  // 3 Allgather global endpoint and connect
  std::vector<RdmaEndpointHandle> global_rdma_ep_handles(world_size * num_qp);
  for (int i = 0; i < num_qp; ++i) {
    bootNet.Allgather(&endpoints[i].handle, global_rdma_ep_handles.data() + i * world_size,
                      sizeof(RdmaEndpointHandle));
  }

  std::cout << "Local rank " << local_rank << " " << endpoints[0].handle << std::endl;

  for (int i = 0; i < world_size; i++) {
    if (i == local_rank) continue;
    for (int qp = 0; qp < num_qp; ++qp) {
      device_context->ConnectEndpoint(endpoints[qp].handle,
                                      global_rdma_ep_handles[i + qp * world_size], qp);
      std::cout << "Local rank " << local_rank << " connected to rank " << i << " qp " << qp
                << " with handle " << global_rdma_ep_handles[i + qp * world_size] << std::endl;
    }
  }

  // 4 Register buffer and block sync memory
  void* buffer;
  size_t totalSize = maxSize * blocks * threads;
  assert(totalSize <= 0x1000000000ULL && "Error: totalSize cannot exceed 64GB!");
  HIP_RUNTIME_CHECK(hipMalloc(&buffer, totalSize));
  HIP_RUNTIME_CHECK(hipMemset(buffer, local_rank, totalSize));
  uint32_t* blockSync;
  HIP_RUNTIME_CHECK(hipMalloc(&blockSync, (warmupIters + iters + 1) * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(blockSync, 0, (warmupIters + iters + 1) * sizeof(uint32_t)));

  // assert(!posix_memalign(&buffer_1, 4096, allreduce_size));
  // memset(buffer_1, 1, allreduce_size);
  RdmaMemoryRegion mr_handle =
      device_context->RegisterRdmaMemoryRegion(buffer, totalSize, MR_ACCESS_FLAG);
  std::vector<RdmaMemoryRegion> global_mr_handles(world_size);
  bootNet.Allgather(&mr_handle, global_mr_handles.data(), sizeof(mr_handle));
  global_mr_handles[local_rank] = mr_handle;
  RdmaEndpoint* devEndpoints;
  HIP_RUNTIME_CHECK(hipMalloc(&devEndpoints, num_qp * sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(hipMemcpy(devEndpoints, endpoints.data(), num_qp * sizeof(RdmaEndpoint),
                              hipMemcpyHostToDevice));

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
      switch (endpoints[0].GetProviderType()) {
        case ProviderType::MLX5:
          MultiQpWrite<ProviderType::MLX5><<<blocks, threads>>>(
              devEndpoints, global_mr_handles[0], global_mr_handles[1], size, 1, blockSync, num_qp);
          break;
#ifdef ENABLE_BNXT
        case ProviderType::BNXT:
          MultiQpWrite<ProviderType::BNXT><<<blocks, threads>>>(
              devEndpoints, global_mr_handles[0], global_mr_handles[1], size, 1, blockSync, num_qp);
          break;
#endif
        default:
          break;
      }
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
      switch (endpoints[0].GetProviderType()) {
        case ProviderType::MLX5:
          MultiQpWrite<ProviderType::MLX5><<<blocks, threads>>>(devEndpoints, global_mr_handles[0],
                                                                global_mr_handles[1], size,
                                                                warmupIters, blockSync + 1, num_qp);
          break;
#ifdef ENABLE_BNXT
        case ProviderType::BNXT:
          MultiQpWrite<ProviderType::BNXT><<<blocks, threads>>>(devEndpoints, global_mr_handles[0],
                                                                global_mr_handles[1], size,
                                                                warmupIters, blockSync + 1, num_qp);
          break;
#endif
        default:
          break;
      }
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());

      // test and record
      HIP_RUNTIME_CHECK(hipEventRecord(start));
      switch (endpoints[0].GetProviderType()) {
        case ProviderType::MLX5:
          MultiQpWrite<ProviderType::MLX5>
              <<<blocks, threads>>>(devEndpoints, global_mr_handles[0], global_mr_handles[1], size,
                                    iters, blockSync + 1 + warmupIters, num_qp);
          break;
#ifdef ENABLE_BNXT
        case ProviderType::BNXT:
          MultiQpWrite<ProviderType::BNXT>
              <<<blocks, threads>>>(devEndpoints, global_mr_handles[0], global_mr_handles[1], size,
                                    iters, blockSync + 1 + warmupIters, num_qp);
          break;
#endif
        default:
          break;
      }
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
    printf("\nIBGDA White benchmark:\n");
    printf("Blocks: %zu, Threads: %zu, Iterations: %zu, QPs:%d \n", blocks, threads, iters, num_qp);
    printf("%-8s %-12s %-12s %-12s %-12s\n", "Index", "Size(B)", "bw(GB)", "Time(ms)", "Rate(Mpps)");

    for (size_t i = 0; i < validSizeLog; ++i) {
      double rate_mpps = (blocks * threads * iters) / (times[i] / MS_TO_S) / 1000000.0;
      printf("%-8zu %-12lu %-12.4f %-12.4f %-12.4f\n", i + 1, sizeTable[i], bwTable[i], times[i],
             rate_mpps);
    }
  }

  bootNet.Finalize();
  HIP_RUNTIME_CHECK(hipFree(buffer));
  HIP_RUNTIME_CHECK(hipFree(devEndpoints));
  HIP_RUNTIME_CHECK(hipHostFree(bwTable));
  HIP_RUNTIME_CHECK(hipHostFree(sizeTable));
  HIP_RUNTIME_CHECK(hipHostFree(times));
}

int main(int argc, char* argv[]) { distRdmaOps(argc, argv); }
