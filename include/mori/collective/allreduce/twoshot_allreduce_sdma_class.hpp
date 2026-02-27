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

#ifndef TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
#define TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <memory>
#include <cstdint>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/collective/collective_pub.hpp"

namespace mori {
namespace collective {

struct CrossPeBarrier;

template <typename T>
class AllreduceSdma {
private:
    int myPe_;
    int npes_;
    size_t dtype_size_;
    int max_blocks_;

    // SDMA completion flags (shared by SdmaReduceScatter and AllGather phases;
    // each phase resets flags before handing off to the next).
    application::SymmMemObjPtr flagsObj_;
    std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

    // Device-scope barrier for block-0-to-all broadcast inside
    // SdmaReduceScatterKernel (generation counter, ~128 bytes).
    CrossPeBarrier* barrierPtr_;
    std::unique_ptr<void, ShmemDeleter> barrierMem_;

    // Output transit buffer — serves as:
    //   1. SDMA scatter destination (gather buffer, npes * chunkSize)
    //   2. Local reduce output (myPe's slot)
    //   3. AllGather source / final result
    void* output_transit_buffer_;
    size_t output_transit_buffer_size_;
    application::SymmMemObjPtr output_transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> output_transit_buffer_ptr_;

    bool copy_output_to_user_;

    AllreduceSdma(const AllreduceSdma&) = delete;
    AllreduceSdma& operator=(const AllreduceSdma&) = delete;

    void copy_output_to_user(T* output, size_t total_count, hipStream_t stream);

public:
    /**
     * @brief Constructor
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param transit_buffer_size Output transit buffer size in bytes (default 512MB)
     * @param copy_output_to_user If true, copy result to user output buffer
     * @param use_graph_mode Kept for API compat — ignored (SDMA always reads
     *        input directly, no IPC registration needed).
     */
    AllreduceSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024,
                  bool copy_output_to_user = true, bool use_graph_mode = false);

    AllreduceSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size,
                  bool copy_output_to_user = true, bool use_graph_mode = false);

    ~AllreduceSdma();

    bool operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);
    bool allreduce_inplace(T* data, size_t total_count, hipStream_t stream = nullptr);

    application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }
    void* getOutputTransitBuffer() const { return output_transit_buffer_; }
    size_t getOutputTransitBufferSize() const { return output_transit_buffer_size_; }
    application::SymmMemObjPtr getOutputTransitBufferObj() const { return output_transit_buffer_obj_; }

    void resetFlags();
};

} // namespace collective
} // namespace mori

#endif // TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
