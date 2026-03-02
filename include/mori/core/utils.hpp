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

#ifndef warpSize
#if defined(__GFX8__) || defined(__GFX9__)
  #define warpSize 64
#else
  #define warpSize 32
#endif
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                         Debug Printf                                           */
/* ---------------------------------------------------------------------------------------------- */
#ifdef MORI_ENABLE_DEBUG_PRINTF
  #define MORI_PRINTF(...) printf(__VA_ARGS__)
#else
  #define MORI_PRINTF(...) ((void)0)
#endif

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                             Thread                                             */
/* ---------------------------------------------------------------------------------------------- */

inline __device__ int DeviceWarpSize() { return warpSize; }

inline __device__ int FlatBlockSize() { return blockDim.z * blockDim.y * blockDim.x; }

inline __device__ int FlatBlockThreadId() {
  return (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

inline __device__ int FlatThreadId() {
  return FlatBlockThreadId() +
         (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
             FlatBlockSize();
}

inline __device__ int FlatBlockWarpNum() { return FlatBlockSize() / DeviceWarpSize(); }

inline __device__ int FlatBlockWarpId() { return FlatBlockThreadId() / DeviceWarpSize(); }

inline __device__ int WarpLaneId() { return FlatBlockThreadId() & (DeviceWarpSize() - 1); }

inline __device__ int WarpLaneId1D() { return threadIdx.x & (warpSize - 1); }

inline __device__ bool IsThreadZeroInBlock() {
  return (FlatBlockThreadId() % DeviceWarpSize()) == 0;
}

inline __device__ uint64_t GetActiveLaneMask() { return __ballot(true); }

inline __device__ unsigned int GetActiveLaneCount(uint64_t activeLaneMask) {
  return __popcll(activeLaneMask);
}

inline __device__ unsigned int GetActiveLaneCount() {
  return GetActiveLaneCount(GetActiveLaneMask());
}

inline __device__ unsigned int GetActiveLaneNum(uint64_t activeLaneMask) {
  return __popcll(activeLaneMask & __lanemask_lt());
}

inline __device__ unsigned int GetActiveLaneNum() { return GetActiveLaneNum(GetActiveLaneMask()); }

inline __device__ int GetFirstActiveLaneID(uint64_t activeLaneMask) {
  return activeLaneMask ? __ffsll((unsigned long long int)activeLaneMask) - 1 : -1;
}

inline __device__ int GetFirstActiveLaneID() { return GetFirstActiveLaneID(GetActiveLaneMask()); }

inline __device__ int GetLastActiveLaneID(uint64_t activeLaneMask) {
  return activeLaneMask ? 63 - __clzll(activeLaneMask) : -1;
}

inline __device__ int GetLastActiveLaneID() { return GetLastActiveLaneID(GetActiveLaneMask()); }

inline __device__ bool IsFirstActiveLane(uint64_t activeLaneMask) {
  return GetActiveLaneNum(activeLaneMask) == 0;
}

inline __device__ bool IsFirstActiveLane() { return IsFirstActiveLane(GetActiveLaneMask()); }

inline __device__ bool IsLastActiveLane(uint64_t activeLaneMask) {
  return GetActiveLaneNum(activeLaneMask) == GetActiveLaneCount(activeLaneMask) - 1;
}

inline __device__ bool IsLastActiveLane() { return IsLastActiveLane(GetActiveLaneMask()); }

/* ---------------------------------------------------------------------------------------------- */
/*                                        Atomic Operations                                       */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ T AtomicLoadSeqCst(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicLoadSeqCstSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ T AtomicLoadRelaxed(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicLoadRelaxedSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ void AtomicStoreRelaxed(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ void AtomicStoreRelaxedSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ void AtomicStoreReleaseSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ T AtomicAddReleaseSystem(T* ptr, T val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ void AtomicStoreSeqCst(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ void AtomicStoreSeqCstSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ T AtomicAddSeqCst(T* ptr, T val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicAddSeqCstSystem(T* ptr, T val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ T AtomicAddRelaxed(T* ptr, T val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicAddRelaxedSystem(T* ptr, T val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ T AtomicCompareExchange(T* address, T* compare, T val) {
  __hip_atomic_compare_exchange_strong(address, compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return *compare;
}

template <typename T>
__device__ T AtomicCompareExchangeSystem(T* address, T* compare, T val) {
  __hip_atomic_compare_exchange_strong(address, compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return *compare;
}

/* -------------------------------------------------------------------------- */
/*                                    Match                                   */
/* -------------------------------------------------------------------------- */
template <typename T>
constexpr inline __device__ __host__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
constexpr inline __device__ __host__ T IsPowerOf2(T x) {
  return (x > 0) && ((x & (x - 1)) == 0);
}

/* -------------------------------------------------------------------------- */
/*                                    Lock                                    */
/* -------------------------------------------------------------------------- */
// TODO: Whether to use GPU lock in lock.hpp
__device__ inline void AcquireLock(uint32_t* lockVar) {
  while (atomicCAS(lockVar, 0, 1) != 0) {
  }
}

__device__ inline bool AcquireLockOnce(uint32_t* lockVar) {
  return atomicCAS(lockVar, 0, 1) == 0;
}


__device__ inline void ReleaseLock(uint32_t* lockVar) { atomicExch(lockVar, 0); }


/* Device-side internal functions */
__device__ __forceinline__ uint32_t lowerID() {
  return __ffsll(__ballot(1)) - 1;
}

__device__ __forceinline__ int wave_SZ() { return __popcll(__ballot(1)); }

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
__device__ __forceinline__ bool is_thread_zero_in_block() {
  return hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0;
}

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
__device__ __forceinline__ bool is_block_zero_in_grid() {
  return hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0;
}

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
__device__ __forceinline__ int get_flat_block_size() {
  return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
}

/*
 * Returns the number of threads in the caller's flattened grid.
 */
__device__ __forceinline__ int get_flat_grid_size() {
  return get_flat_block_size() * hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ __forceinline__ int get_flat_block_id() {
  return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
         hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

/*
 * Returns the number of blocks in the caller's flattened grid.
 */
__device__ __forceinline__ int get_grid_num_blocks() {
  return hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ __forceinline__ int get_flat_grid_id() {
  return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
         hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

/*
 * Returns the flattened thread index of the calling thread within the grid.
 */
__device__ __forceinline__ int get_flat_id() {
  return get_flat_grid_id() * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z) + get_flat_block_id();
}

/*
 * Returns true if the caller's thread flad_id is 0 in its wave.
 */
__device__ __forceinline__ bool is_thread_zero_in_wave() {
  return (get_flat_block_id() % warpSize) == 0;
}

__device__ __forceinline__ uint64_t get_active_lane_mask() {
  return __ballot(true);
}

__device__ __forceinline__ unsigned int get_active_lane_count(uint64_t active_lane_mask) {
  return __popcll(active_lane_mask);
}

__device__ __forceinline__ unsigned int get_active_lane_count() {
  return get_active_lane_count(get_active_lane_mask());
}

__device__ __forceinline__ unsigned int get_active_lane_num(uint64_t active_lane_mask) {
  return __popcll(active_lane_mask & __lanemask_lt());
}

__device__ __forceinline__ unsigned int get_active_lane_num() {
  return get_active_lane_num(get_active_lane_mask());
}

__device__ __forceinline__ int get_first_active_lane_id(uint64_t active_lane_mask) {
  return __ffsll((unsigned long long int)active_lane_mask) - 1;
}

__device__ __forceinline__ int get_first_active_lane_id() {
  return get_first_active_lane_id(get_active_lane_mask());
}

__device__ __forceinline__ bool is_first_active_lane(uint64_t active_lane_mask) {
  return get_active_lane_num(active_lane_mask) == 0;
}

__device__ __forceinline__ bool is_first_active_lane() {
  return is_first_active_lane(get_active_lane_mask());
}

__device__ __forceinline__ bool is_last_active_lane(uint64_t active_lane_mask) {
  return get_active_lane_num(active_lane_mask) == get_active_lane_count(active_lane_mask) - 1;
}

__device__ __forceinline__ bool is_last_active_lane() {
  return is_last_active_lane(get_active_lane_mask());
}

#define SPIN_LOCK_INVALID  0xdead
#define SPIN_LOCK_UNLOCKED 0x0
#define SPIN_LOCK_LOCKED   0xabcd
	
/*
 * Each thread in wave tries to acquire a different lock.
 */
__device__ __forceinline__ bool spin_lock_try_acquire_unique(uint32_t *lock) {
  uint32_t lock_val = SPIN_LOCK_UNLOCKED;

  __hip_atomic_compare_exchange_strong(lock, &lock_val, SPIN_LOCK_LOCKED,
									   __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
									   __HIP_MEMORY_SCOPE_AGENT);

  return lock_val == SPIN_LOCK_UNLOCKED;
}

/*
 * Each thread in wave acquires a different lock.
 * (deadlock if locks are not different)
 */
__device__ __forceinline__ void spin_lock_acquire_unique(uint32_t *lock) {
  while (!spin_lock_try_acquire_unique(lock)) {
	// spin
  }
}

/*
 * Each thread in wave releases a different lock.
 */
__device__ __forceinline__ void spin_lock_release_unique(uint32_t *lock) {
  __hip_atomic_store(lock, SPIN_LOCK_UNLOCKED, __ATOMIC_RELEASE,
					 __HIP_MEMORY_SCOPE_AGENT);
}

/*
 * Threads in activemask together try to acquire the same lock.
 */
__device__ __forceinline__ bool spin_lock_try_acquire_shared(uint32_t *lock, uint64_t activemask) {
  uint32_t lock_val = SPIN_LOCK_INVALID;

  if (is_first_active_lane(activemask)) {
	lock_val = SPIN_LOCK_UNLOCKED;
	__hip_atomic_compare_exchange_strong(lock, &lock_val, SPIN_LOCK_LOCKED,
										 __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
										 __HIP_MEMORY_SCOPE_AGENT);
  }
  lock_val = __shfl(lock_val, get_first_active_lane_id(activemask));

  return lock_val == SPIN_LOCK_UNLOCKED;
}

/*
 * Threads in activemask together acquire the same lock.
 */
__device__ __forceinline__ void spin_lock_acquire_shared(uint32_t *lock, uint64_t activemask) {
  while (!spin_lock_try_acquire_shared(lock, activemask)) {
	// spin
  }
}

/*
 * Threads in activemask together release the same lock.
 */
__device__ __forceinline__ void spin_lock_release_shared(uint32_t *lock, uint64_t activemask) {
  if (is_first_active_lane(activemask)) {
	__hip_atomic_store(lock, SPIN_LOCK_UNLOCKED, __ATOMIC_RELEASE,
					   __HIP_MEMORY_SCOPE_AGENT);
  }
}

}  // namespace core
}  // namespace mori
