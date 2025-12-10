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

#include <stddef.h>
#include <stdint.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"

using atomicType = mori::core::atomicType;

extern "C" {
// ============================================================================
// Synchronization APIs
// ============================================================================
__device__ __attribute__((visibility("default"))) void shmem_quiet_thread();
__device__ __attribute__((visibility("default"))) void shmem_quiet_thread_pe(int pe);
__device__ __attribute__((visibility("default"))) void shmem_quiet_thread_pe_qp(int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_fence_thread();
__device__ __attribute__((visibility("default"))) void shmem_fence_thread_pe(int pe);
__device__ __attribute__((visibility("default"))) void shmem_fence_thread_pe_qp(int pe, int qpId);

// ============================================================================
// Point-to-Point APIs
// ============================================================================
__device__ __attribute__((visibility("default"))) uint64_t shmem_ptr_p2p(const uint64_t destPtr,
                                                                         const int myPe,
                                                                         int destPe);

// ============================================================================
// PutNbi APIs - Thread Scope (Address-based only)
// ============================================================================
__device__ __attribute__((visibility("default"))) void shmem_putmem_nbi_thread(void* dest,
                                                                               const void* source,
                                                                               size_t bytes, int pe,
                                                                               int qpId);

__device__ __attribute__((visibility("default"))) void shmem_put_uint32_nbi_thread(
    uint32_t* dest, const uint32_t* source, size_t nelems, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_put_uint64_nbi_thread(
    uint64_t* dest, const uint64_t* source, size_t nelems, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_put_float_nbi_thread(
    float* dest, const float* source, size_t nelems, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_put_double_nbi_thread(
    double* dest, const double* source, size_t nelems, int pe, int qpId);

// ============================================================================
// PutNbi Immediate APIs - Thread Scope
// ============================================================================
__device__ __attribute__((visibility("default"))) void shmem_put_size_imm_nbi_thread(
    void* dest, void* val, size_t bytes, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_int_p(int* dest, int val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_long_p(long* dest, long val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_longlong_p(long long* dest, long long val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_float_p(float* dest, float val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_double_p(double* dest, double val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_char_p(char* dest, char val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_short_p(short* dest, short val, int pe);

__device__ __attribute__((visibility("default"))) void shmem_uint_p(unsigned int* dest, unsigned int val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_ulong_p(unsigned long* dest, unsigned long val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_ulonglong_p(unsigned long long* dest, unsigned long long val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_uchar_p(unsigned char* dest, unsigned char val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_ushort_p(unsigned short* dest, unsigned short val, int pe);

__device__ __attribute__((visibility("default"))) void shmem_int32_p(int32_t* dest, int32_t val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_int64_p(int64_t* dest, int64_t val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_uint32_p(uint32_t* dest, uint32_t val, int pe);
__device__ __attribute__((visibility("default"))) void shmem_uint64_p(uint64_t* dest, uint64_t val, int pe);

// ============================================================================
// Atomic NonFetch APIs - Thread Scope
// ============================================================================
__device__ __attribute__((visibility("default"))) void shmem_atomic_size_nonfetch_thread(
    void* dest, void* val, size_t bytes, atomicType amoType, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_atomic_uint32_nonfetch_thread(
    uint32_t* dest, uint32_t val, atomicType amoType, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_atomic_uint64_nonfetch_thread(
    uint64_t* dest, uint64_t val, atomicType amoType, int pe, int qpId);

// ============================================================================
// Atomic Fetch APIs - Thread Scope
// ============================================================================
__device__ __attribute__((visibility("default"))) uint32_t shmem_atomic_uint32_fetch_thread(
    uint32_t* dest, uint32_t val, uint32_t compare, atomicType amoType, int pe, int qpId);

__device__ __attribute__((visibility("default"))) uint64_t shmem_atomic_uint64_fetch_thread(
    uint64_t* dest, uint64_t val, uint64_t compare, atomicType amoType, int pe, int qpId);

// ============================================================================
// Atomic Add Convenience APIs - Thread Scope
// ============================================================================
__device__ __attribute__((visibility("default"))) void shmem_uint32_atomic_add_thread(
    uint32_t* dest, uint32_t val, int pe, int qpId);

__device__ __attribute__((visibility("default"))) uint32_t
shmem_uint32_atomic_fetch_add_thread(uint32_t* dest, uint32_t val, int pe, int qpId);

__device__ __attribute__((visibility("default"))) void shmem_uint64_atomic_add_thread(
    uint64_t* dest, uint64_t val, int pe, int qpId);

__device__ __attribute__((visibility("default"))) uint64_t
shmem_uint64_atomic_fetch_add_thread(uint64_t* dest, uint64_t val, int pe, int qpId);

// ============================================================================
// Wait APIs
// ============================================================================
__device__ __attribute__((visibility("default"))) uint32_t
shmem_uint32_wait_until_greater_than(uint32_t* addr, uint32_t val);
__device__ __attribute__((visibility("default"))) void shmem_uint32_wait_until_equals(
    uint32_t* addr, uint32_t val);

__device__ __attribute__((visibility("default"))) uint64_t
shmem_uint64_wait_until_greater_than(uint64_t* addr, uint64_t val);
__device__ __attribute__((visibility("default"))) void shmem_uint64_wait_until_equals(
    uint64_t* addr, uint64_t val);

__device__ __attribute__((visibility("default"))) int32_t
shmem_int32_wait_until_greater_than(int32_t* addr, int32_t val);
__device__ __attribute__((visibility("default"))) void shmem_int32_wait_until_equals(int32_t* addr,
                                                                                     int32_t val);

__device__ __attribute__((visibility("default"))) int64_t
shmem_int64_wait_until_greater_than(int64_t* addr, int64_t val);
__device__ __attribute__((visibility("default"))) void shmem_int64_wait_until_equals(int64_t* addr,
                                                                                     int64_t val);

// ============================================================================
// Query APIs
// ============================================================================
__device__ __attribute__((visibility("default"))) int shmem_my_pe();
__device__ __attribute__((visibility("default"))) int shmem_n_pes();

}  // extern "C"
