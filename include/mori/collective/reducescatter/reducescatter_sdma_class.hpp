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

#ifndef REDUCESCATTER_SDMA_CLASS_HPP
#define REDUCESCATTER_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <memory>
#include <cstdint>
#include <atomic>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/collective/collective_pub.hpp"

namespace mori {
namespace collective {

struct CrossPeBarrier;

template <typename T>
class ReduceScatterSdma {
private:
    int myPe_;
    int npes_;
    size_t dtype_size_;
    int max_blocks_;

    // SDMA completion flags
    application::SymmMemObjPtr flagsObj_;
    std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

    // Device-scope barrier for block-0-to-all broadcast
    CrossPeBarrier* barrierPtr_;
    std::unique_ptr<void, ShmemDeleter> barrierMem_;

    // Transit buffer (gather buffer): npes * chunkSize slots for SDMA scatter
    void* transit_buffer_;
    size_t transit_buffer_size_;
    application::SymmMemObjPtr transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> transit_buffer_ptr_;

    // Async state
    std::atomic<bool> async_in_progress_;
    T* async_input_;
    T* async_output_;
    size_t async_total_count_;
    hipStream_t async_stream_;
    double async_start_time_;

    bool copy_output_to_user_;

    ReduceScatterSdma(const ReduceScatterSdma&) = delete;
    ReduceScatterSdma& operator=(const ReduceScatterSdma&) = delete;

    bool ensure_buffer_size(void*& buffer,
                           std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                           size_t& current_size,
                           application::SymmMemObjPtr& buffer_obj,
                           size_t required_size,
                           const char* buffer_name);

    void copy_result_to_user(T* output, size_t total_count, hipStream_t stream);

public:
    /**
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param transit_buffer_size Transit buffer size in bytes (default 512MB)
     * @param copy_output_to_user If true, copy reduced shard to user output buffer
     */
    ReduceScatterSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024,
                      bool copy_output_to_user = true);

    ReduceScatterSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size,
                      bool copy_output_to_user = true);

    ~ReduceScatterSdma();

    /**
     * @brief Synchronous ReduceScatter via SDMA
     * @param input  Input data — total_count elements per rank
     * @param output Output data — total_count/npes reduced elements per rank
     * @param total_count Number of input elements per PE
     * @param stream HIP stream
     */
    bool operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

    bool start_async(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);
    double wait_async(hipStream_t stream = nullptr);
    bool is_async_in_progress() const { return async_in_progress_; }
    void cancel_async();

    application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }
    void* getTransitBuffer() const { return transit_buffer_; }
    size_t getTransitBufferSize() const { return transit_buffer_size_; }
    application::SymmMemObjPtr getTransitBufferObj() const { return transit_buffer_obj_; }

    void resetFlags();
};

} // namespace collective
} // namespace mori

#endif // REDUCESCATTER_SDMA_CLASS_HPP
