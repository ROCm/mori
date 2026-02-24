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
#include <atomic>

// Include necessary headers
#include "mori/application/application.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/collective/collective_pub.hpp"

namespace mori {
namespace collective {

template <typename T>
class AllreduceSdma {
private:
    int myPe_;
    int npes_;
    size_t dtype_size_;

    // Flag memory
    application::SymmMemObjPtr flagsObj_;
    std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

    // Input transit buffer (srcMemObj — registered wrapper around user input)
    void* input_transit_buffer_;
    size_t input_transit_buffer_size_;
    application::SymmMemObjPtr input_transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> input_transit_buffer_ptr_;

    // Output transit buffer (dstMemObj — intermediate gather/reduce/allgather buffer)
    void* output_transit_buffer_;
    size_t output_transit_buffer_size_;
    application::SymmMemObjPtr output_transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> output_transit_buffer_ptr_;

    // Async state variables
    std::atomic<bool> async_in_progress_;
    T* async_input_;
    T* async_output_;
    size_t async_total_count_;
    hipStream_t async_stream_;
    double async_start_time_;

    // Copy mode flag: if true, copy output_transit_buffer to user output buffer
    // if false, user should directly use output_transit_buffer
    bool copy_output_to_user_;

    // Disable copy constructor and assignment operator
    AllreduceSdma(const AllreduceSdma&) = delete;
    AllreduceSdma& operator=(const AllreduceSdma&) = delete;

    // Internal methods
    bool ensure_buffer_size(void*& buffer,
                           std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                           size_t& current_size,
                           application::SymmMemObjPtr& buffer_obj,
                           size_t required_size,
                           const char* buffer_name);

    void copy_input_to_transit(T* input, size_t total_count, hipStream_t stream);
    void copy_output_to_user(T* output, size_t total_count, hipStream_t stream);

public:
    /**
     * @brief Constructor, initializes AllreduceSdma class
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param transit_buffer_size Transit buffer size in bytes (default 512MB), half for input and half for output
     * @param copy_output_to_user If true, copy output_transit_buffer to user output buffer (default true)
     */
    AllreduceSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024, bool copy_output_to_user = true);

    /**
     * @brief Constructor, specifying input and output transit buffer sizes separately
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param input_buffer_size Input transit buffer size in bytes
     * @param output_buffer_size Output transit buffer size in bytes
     * @param copy_output_to_user If true, copy output_transit_buffer to user output buffer (default true)
     */
    AllreduceSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size, bool copy_output_to_user = true);

    /**
     * @brief Destructor, cleans up resources
     */
    ~AllreduceSdma();

    /**
     * @brief Executes synchronous AllReduce SDMA operation
     * @param input Input data pointer (elementCount elements on each rank)
     * @param output Output data pointer (elementCount elements — the reduced result)
     * @param total_count Number of data elements per PE
     * @param stream HIP stream
     * @return true if successful, false if failed
     * @note Synchronization must be handled by the caller
     */
    bool operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

    /**
     * @brief Start asynchronous AllReduce operation (AllGather PUT phase)
     * @param input Input data pointer
     * @param output Output data pointer
     * @param total_count Number of data elements per PE
     * @param stream HIP stream
     * @return true if successful, false otherwise
     */
    bool start_async(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

    /**
     * @brief Wait for asynchronous AllReduce operation to complete
     *        (WAIT phase + local reduce + optional copy)
     * @param stream HIP stream (optional, can be different from start_async stream)
     * @return Execution time in seconds, -1.0 if failed
     */
    double wait_async(hipStream_t stream = nullptr);

    /**
     * @brief Check if async operation is in progress
     * @return true if async operation is active
     */
    bool is_async_in_progress() const { return async_in_progress_; }

    /**
     * @brief Cancel ongoing async operation
     */
    void cancel_async();

    /**
     * @brief Gets flag symmetric memory object
     */
    application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }

    /**
     * @brief Gets input transit buffer pointer
     */
    void* getInputTransitBuffer() const { return input_transit_buffer_; }

    /**
     * @brief Gets input transit buffer size in bytes
     */
    size_t getInputTransitBufferSize() const { return input_transit_buffer_size_; }

    /**
     * @brief Gets input transit buffer symmetric memory object
     */
    application::SymmMemObjPtr getInputTransitBufferObj() const { return input_transit_buffer_obj_; }

    /**
     * @brief Gets output transit buffer pointer
     */
    void* getOutputTransitBuffer() const { return output_transit_buffer_; }

    /**
     * @brief Gets output transit buffer size in bytes
     */
    size_t getOutputTransitBufferSize() const { return output_transit_buffer_size_; }

    /**
     * @brief Gets output transit buffer symmetric memory object
     */
    application::SymmMemObjPtr getOutputTransitBufferObj() const { return output_transit_buffer_obj_; }

    /**
     * @brief Resets flags (sets to 0)
     */
    void resetFlags();
};

} // namespace collective
} // namespace mori

#endif // TWOSHOT_ALLREDUCE_SDMA_CLASS_HPP
