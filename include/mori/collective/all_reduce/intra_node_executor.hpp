#pragma once

#include "mori/collective/all_reduce/allreduce_config.hpp"
#include "mori/collective/all_reduce/allreduce_executor.hpp"
#include "mori/collective/all_reduce/intra_node_impl.cuh"
#include <memory>
#include <mpi.h>

namespace mori {
namespace collective {

/**
 * IntraNodeAllReduceExecutor - intra node AllReduce high level wrapper
 * 
 * usage:
 * 1. one initialization: AllReduceExecutor<T>* executor = new IntraNodeAllReduceExecutor<T>(num_ranks, rank, mpi_comm);
 * 2. multiple calls: executor->Execute(d_input, d_output, count, stream);
 * 
 * internally call private interfaces (initializeSignals, initializeIPC, cleanupIPC)
 * to call member functions of intra_node_ to implement high level wrapper
 */
template <typename T>
class IntraNodeAllReduceExecutor : public AllReduceExecutor<T> {
public:
  /**
   * constructor - initialize executor
   * 
   * @param num_ranks number of GPUs (must be 2, 4, 6, 8)
   * @param rank rank of current GPU (0 to num_ranks-1)
   * @param mpi_comm MPI communicator (for IPC handle exchange)
   * @param max_size maximum buffer size (default 64MB)
   * @param config AllReduce configuration (optional)
   */
  IntraNodeAllReduceExecutor(
      int num_ranks,
      int rank,
      MPI_Comm mpi_comm,
      size_t max_size = 8192 * 1024 * 8,
      const AllReduceConfig& config = AllReduceConfig()
  );

  ~IntraNodeAllReduceExecutor() override;

  /**
   * execute AllReduce operation
   * 
   * @param input input data pointer (device memory)
   * @param output output data pointer (device memory, can be same as input)
   * @param count number of elements
   * @param stream HIP stream (asynchronous execution)
   * @return 0 on success, error code otherwise
   */
  int Execute(T* input, T* output, size_t count, hipStream_t stream) override;

private:
  std::unique_ptr<IntraNode> intra_node_;  // core implementation
  AllReduceConfig config_;                  // configuration
  MPI_Comm mpi_comm_;                       // MPI communicator
  bool initialized_;                        // initialization flag

  // ===== private interfaces =====
  
  /**
   * initialize signal buffer
   * call intra_node_->allocate_buffers()
   */
  void initializeSignals();

  /**
   * initialize IPC communication
   * 1. get local IPC handle: intra_node_->get_meta_buffer_ipc_handle()
   * 2. use MPI_Allgather to collect IPC handles from all ranks
   * 3. call intra_node_->init_custom_ar() to create kernel entity
   * 4. get local pre-registered buffer IPC handle: intra_node_->get_buffer_ipc_handle()
   * 5. use MPI_Allgather to collect IPC handles from all ranks
   * 6. register pre-registered buffer (similar to aiter's self.register_buffer(self.buffer))
   */
  void initializeIPC();

  /**
   * clean up IPC resources
   * call intra_node_->dispose()
   */
  void cleanupIPC();
};

// ===== Template Implementation =====

template <typename T>
IntraNodeAllReduceExecutor<T>::IntraNodeAllReduceExecutor(
    int num_ranks,
    int rank,
    MPI_Comm mpi_comm,
    size_t max_size,
    const AllReduceConfig& config)
    : config_(config),
      mpi_comm_(mpi_comm),
      initialized_(false) {
  
  // create IntraNode instance
  intra_node_ = std::make_unique<IntraNode>(num_ranks, rank, max_size, true);
  
  try {
    // call private interfaces in order to initialize
    initializeSignals();
    initializeIPC();
    initialized_ = true;
  } catch (const std::exception& e) {
    cleanupIPC();
    throw;
  }
}

template <typename T>
IntraNodeAllReduceExecutor<T>::~IntraNodeAllReduceExecutor() {
  cleanupIPC();
}

template <typename T>
int IntraNodeAllReduceExecutor<T>::Execute(T* input, T* output, size_t count, 
                                            hipStream_t stream) {
  if (!initialized_ || !intra_node_) {
    return -1;  // Not initialized
  }
  
  try {
    // call all_reduce method of intra_node_
    // registered=false: Eager mode, will copy to pre-registered buffer first
    // TODO: graph mode will be supported later
    bool use_new = true;
    bool registered = false;  // default use eager mode
    intra_node_->all_reduce<T>(stream, input, output, count, use_new, registered);
    return 0;
  } catch (const std::exception& e) {
    // TODO: add error logging
    return -2;  // Execution failed
  }
}

// ===== private interfaces implementation =====

template <typename T>
void IntraNodeAllReduceExecutor<T>::initializeSignals() {
  // call allocate_buffers() of intra_node_ to allocate signal buffer
  intra_node_->allocate_buffers();
}

template <typename T>
void IntraNodeAllReduceExecutor<T>::initializeIPC() {
  int num_ranks = intra_node_->get_num_ranks();
  int rank = intra_node_->get_rank();
  
  // 1. get IPC handle of local meta buffer
  hipIpcMemHandle_t my_meta_handle = intra_node_->get_meta_buffer_ipc_handle();
  int64_t my_meta_offset = 0;  // meta buffer starts from the beginning
  
  // 2. use MPI_Allgather to collect IPC handles of meta buffer from all ranks
  std::vector<hipIpcMemHandle_t> all_meta_handles(num_ranks);
  std::vector<int64_t> all_meta_offsets(num_ranks);
  
  MPI_Allgather(&my_meta_handle, sizeof(hipIpcMemHandle_t), MPI_BYTE,
                all_meta_handles.data(), sizeof(hipIpcMemHandle_t), MPI_BYTE,
                mpi_comm_);
  
  MPI_Allgather(&my_meta_offset, 1, MPI_INT64_T,
                all_meta_offsets.data(), 1, MPI_INT64_T,
                mpi_comm_);
  
  // 3. call init_custom_ar() of intra_node_ to create kernel entity
  intra_node_->init_custom_ar(all_meta_handles, all_meta_offsets);
  
  // 4. get IPC handle of local pre-registered buffer
  hipIpcMemHandle_t my_buffer_handle = intra_node_->get_buffer_ipc_handle();
  int64_t my_buffer_offset = 0;  // buffer starts from the beginning
  
  // 5. use MPI_Allgather to collect IPC handles of buffer from all ranks
  std::vector<hipIpcMemHandle_t> all_buffer_handles(num_ranks);
  std::vector<int64_t> all_buffer_offsets(num_ranks);
  
  MPI_Allgather(&my_buffer_handle, sizeof(hipIpcMemHandle_t), MPI_BYTE,
                all_buffer_handles.data(), sizeof(hipIpcMemHandle_t), MPI_BYTE,
                mpi_comm_);
  
  MPI_Allgather(&my_buffer_offset, 1, MPI_INT64_T,
                all_buffer_offsets.data(), 1, MPI_INT64_T,
                mpi_comm_);
  
  // 6. register pre-registered buffer (similar to aiter's self.register_buffer(self.buffer))
  intra_node_->register_pre_allocated_buffer(all_buffer_handles, all_buffer_offsets);
}

template <typename T>
void IntraNodeAllReduceExecutor<T>::cleanupIPC() {
  if (intra_node_) {
    // call dispose() of intra_node_ to clean up resources
    intra_node_->dispose();
  }
}

} // namespace collective
} // namespace mori

