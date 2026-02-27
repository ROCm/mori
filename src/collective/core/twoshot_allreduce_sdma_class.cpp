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

#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// Delegating constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t transit_buffer_size,
                                bool copy_output_to_user, bool /*use_graph_mode*/)
    : AllreduceSdma(myPe, npes, 0, transit_buffer_size,
                    copy_output_to_user, false) {
}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes,
                                size_t /*input_buffer_size*/,
                                size_t output_buffer_size,
                                bool copy_output_to_user,
                                bool /*use_graph_mode*/)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      max_blocks_(getDeviceMaxBlocks()),
      flags_(nullptr, ShmemDeleter()),
      barrierPtr_(nullptr),
      barrierMem_(nullptr, ShmemDeleter()),
      output_transit_buffer_(nullptr),
      output_transit_buffer_size_(output_buffer_size),
      output_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      copy_output_to_user_(copy_output_to_user) {

    // 1. Allocate SDMA completion flags
    size_t flagsSize = npes_ * sizeof(uint64_t);
    void* flags = shmem::ShmemMalloc(flagsSize);
    if (!flags) throw std::runtime_error("Failed to allocate flags memory");
    flags_.reset(static_cast<uint64_t*>(flags));
    memset(flags_.get(), 0, flagsSize);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_.get());
    if (!flagsObj_.IsValid())
        throw std::runtime_error("Failed to get valid flags memory object");

    // 2. Allocate CrossPeBarrier (device-scope broadcast flag, ~128 bytes)
    size_t barrierSize = sizeof(CrossPeBarrier);
    void* bMem = shmem::ShmemMalloc(barrierSize);
    if (!bMem) throw std::runtime_error("Failed to allocate barrier memory");
    barrierMem_.reset(bMem);
    barrierPtr_ = reinterpret_cast<CrossPeBarrier*>(bMem);
    hipError_t me = hipMemset(bMem, 0, barrierSize);
    if (me != hipSuccess)
        throw std::runtime_error("Failed to zero-init barrier memory");

    // 3. Allocate output transit buffer (gather + reduce + allgather)
    output_transit_buffer_ = shmem::ShmemMalloc(output_transit_buffer_size_);
    if (!output_transit_buffer_)
        throw std::runtime_error("Failed to allocate output transit buffer");
    output_transit_buffer_ptr_.reset(output_transit_buffer_);

    output_transit_buffer_obj_ =
        shmem::ShmemSymmetricRegister(output_transit_buffer_,
                                      output_transit_buffer_size_);
    if (!output_transit_buffer_obj_.IsValid())
        throw std::runtime_error("Failed to register output transit buffer");

    printf("AllreduceSdma(SDMA) initialized: PE %d of %d, max_blocks=%d\n",
           myPe_, npes_, max_blocks_);
    printf("  Flags: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Barrier: %zu bytes at %p\n", barrierSize, bMem);
    printf("  Output transit buffer: %.2f MB at %p\n",
           output_transit_buffer_size_ / (1024.0 * 1024.0),
           output_transit_buffer_);
}

// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
    if (flags_) {
        printf("AllreduceSdma destroyed: PE %d\n", myPe_);
    }
}

// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::copy_output_to_user(T* output, size_t total_count,
                                            hipStream_t stream) {
    size_t bytes = total_count * dtype_size_;
    if (!output)  throw std::runtime_error("Output pointer is null");
    if (!output_transit_buffer_)
        throw std::runtime_error("Output transit buffer is null");

    hipError_t err = stream
        ? hipMemcpyAsync(output, output_transit_buffer_, bytes,
                         hipMemcpyDeviceToDevice, stream)
        : hipMemcpy(output, output_transit_buffer_, bytes,
                    hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: copy_output_to_user failed: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Output copy failed");
    }
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::operator()(T* input, T* output, size_t total_count,
                                  hipStream_t stream) {
    try {
        // Step 1: SdmaReduceScatter — SDMA scatter + local reduce
        constexpr int pack_size = packed_t<T>::P::size;
        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;

        SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: SdmaReduceScatter launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 2: AllGather via SDMA
        AllGatherSdmaKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_, total_count);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: AllGather launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 3: Copy result to user buffer
        if (copy_output_to_user_) {
            copy_output_to_user(output, total_count, stream);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: AllReduce failed: %s\n", myPe_, e.what());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::allreduce_inplace(T* data, size_t total_count,
                                          hipStream_t stream) {
    bool saved = copy_output_to_user_;
    copy_output_to_user_ = true;
    bool ok = (*this)(data, data, total_count, stream);
    copy_output_to_user_ = saved;
    return ok;
}

// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::resetFlags() {
    if (flags_) {
        memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class AllreduceSdma<uint32_t>;
template class AllreduceSdma<uint64_t>;
template class AllreduceSdma<int32_t>;
template class AllreduceSdma<int64_t>;
template class AllreduceSdma<float>;
template class AllreduceSdma<double>;
template class AllreduceSdma<half>;
template class AllreduceSdma<__hip_bfloat16>;

} // namespace collective
} // namespace mori
