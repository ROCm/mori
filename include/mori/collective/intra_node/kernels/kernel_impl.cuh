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
#pragma once
/*
 * impl kernel function
 * both global function and device function
 * included by kernel_entity.cuh
 * */
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#include "vec_type.cuh"

constexpr int kMaxBlocks = 80;

struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][8];
  alignas(128) uint32_t end[kMaxBlocks][8];
  alignas(128) uint32_t _flag[kMaxBlocks];
};

// data pointer array
struct __align__(16) RankData {
  const void* ptrs[8];
};

// Signal pointer array
struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <int ngpus>
DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;

  if (threadIdx.x < ngpus) {
    // cross device store
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],  // dst addr
                            flag,                                               // src data
                            __ATOMIC_RELAXED,                                   // atomic level
                            __MEMORY_SCOPE_SYSTEM                               // memory scope
    );

    // local device load
    // wait other ranks cross device store finish
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    self_sg->_flag[blockIdx.x] = flag;
  }
}

template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();

  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank], flag,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                                  final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    self_sg->_flag[blockIdx.x] = flag;
  }
}

// get temp buffer for 2-stage allreduce
template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// 1-stage allreduce kernel (naive version)
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage_naive(RankData* _dp, RankSignals sg, Signal* self_sg,
                                     T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  auto dp = *_dp;
  start_sync<ngpus>(sg, self_sg, rank);

  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }

  end_sync<ngpus, true>(sg, self_sg, rank);
}

// 2-stage allreduce kernel (naive version)
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage_naive(RankData* _dp, RankSignals sg, Signal* self_sg,
                                     T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;

  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];

  start_sync<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }

  end_sync<ngpus>(sg, self_sg, rank);

  // stage 2: allgather
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// 1-stage allreduce kernel (optimized version)
#define THREAD_NUM 512

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = packed_t<T>::P::size;
  constexpr int tnum_gpu = THREAD_NUM / ngpus;
  __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];

  auto dp = *_dp;
  int warp_id = threadIdx.x / tnum_gpu;
  int lane_id = threadIdx.x % tnum_gpu;

  start_sync<ngpus>(sg, self_sg, rank);

  // do the actual reduction
  for (int idx = blockIdx.x * tnum_gpu + lane_id; idx < size; idx += gridDim.x * tnum_gpu) {
    *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = ((const P**)&dp.ptrs[0])[warp_id][idx];
    __syncthreads();

    if (warp_id == 0) {
      A add_reg = upcast_v<typename P::type, pack_size>(
          *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x));

      constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
      for (int i = 1; i < ngpus; ++i) {
        P tmp = *(reinterpret_cast<P*>(&tmp_smem[smem_gpu_loop_stride * i]) + threadIdx.x);
        packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(tmp));
      }

      ((P*)result)[idx] = downcast_v<typename P::type, pack_size>(add_reg);
    }
    __syncthreads();
  }
}

// 2-stage allreduce kernel (optimized version)
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  constexpr int pack_size = packed_t<T>::P::size;
  constexpr int tnum_gpu = THREAD_NUM / ngpus;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];

  int warp_id = threadIdx.x / tnum_gpu;
  int lane_id = threadIdx.x % tnum_gpu;
  int tid = blockIdx.x * tnum_gpu + lane_id;
  int stride = gridDim.x * tnum_gpu;

  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;

  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];

  start_sync<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = ptrs[warp_id][idx];
    __syncthreads();

    if (warp_id == 0) {
      A add_reg = upcast_v<typename P::type, pack_size>(
          *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x));

      constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
      for (int i = 1; i < ngpus; ++i) {
        P tmp = *(reinterpret_cast<P*>(&tmp_smem[i * smem_gpu_loop_stride]) + threadIdx.x);
        packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(tmp));
      }

      tmp_out[idx - start] = downcast_v<typename P::type, pack_size>(add_reg);
    }
    __syncthreads();
  }

  end_sync<ngpus>(sg, self_sg, rank);

  // stage 2: allgather
  for (int idx = tid; idx < largest_part; idx += stride) {
    int dst_idx = (warp_id + rank) % ngpus * part + idx;
    ((P*)result)[dst_idx] = tmps[warp_id][idx];
  }
}
