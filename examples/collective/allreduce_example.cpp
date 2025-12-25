// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// Example: Using IntraNodeAllReduceExecutor with MPI
//
// This example demonstrates AllReduce using MPI for IPC handle exchange

#include "mori/collective/all_reduce/intra_node_executor.hpp"
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <memory>

// Helper function to initialize data
void initializeData(float* data, size_t count, int rank) {
  for (size_t i = 0; i < count; i++) {
    data[i] = static_cast<float>(rank + 1);
  }
}

// Helper function to verify results
bool verifyResults(const float* data, size_t count, int num_ranks) {
  float expected = 0.0f;
  for (int r = 0; r < num_ranks; r++) {
    expected += static_cast<float>(r + 1);
  }
  
  bool success = true;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(data[i] - expected) > 1e-5) {
      std::cerr << "Mismatch at index " << i 
                << ": expected " << expected 
                << ", got " << data[i] << std::endl;
      success = false;
      break;
    }
  }
  return success;
}

int main(int argc, char** argv) {
  // ===== init MPI =====
  MPI_Init(&argc, &argv);
  
  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  
  const size_t count = 1024 * 1024;  // Number of elements
  
  if (rank == 0) {
    std::cout << "==================================" << std::endl;
    std::cout << "IntraNode AllReduce Example (MPI)" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Num Ranks: " << num_ranks << std::endl;
    std::cout << "Count: " << count << std::endl;
    std::cout << "Data Type: float" << std::endl;
    std::cout << std::endl;
  }
  
  try {
    // set device
    hipSetDevice(rank);
    
    // ===== step 1: initialize executor (using polymorphism with smart pointer) =====
    if (rank == 0) {
      std::cout << "[Step 1] initialize executor..." << std::endl;
    }
    
    // Use base class smart pointer to hold derived class object (polymorphism)
    std::unique_ptr<mori::collective::AllReduceExecutor<float>> executor(
        new mori::collective::IntraNodeAllReduceExecutor<float>(
            num_ranks,           // number of GPUs
            rank,                // current rank
            MPI_COMM_WORLD       // MPI communicator
        )
    );
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "  ✓ initialization completed (IPC handles exchanged via MPI)" << std::endl;
      std::cout << std::endl;
    }
    
    // allocate device memory
    float* d_data;
    hipMalloc(&d_data, count * sizeof(float));
    
    // initialize input data
    std::vector<float> h_data(count);
    initializeData(h_data.data(), count, rank);
    hipMemcpy(d_data, h_data.data(), count * sizeof(float), hipMemcpyHostToDevice);
    
    // create stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    // ===== step 2: execute AllReduce (can be called multiple times) =====
    if (rank == 0) {
      std::cout << "[Step 2] execute AllReduce..." << std::endl;
    }
    
    // first call
    int result = executor->Execute(
        d_data,      // input
        d_data,      // output (in-place)
        count,       // number of elements
        stream       // HIP stream
    );
    
    if (result != 0) {
      std::cerr << "  Rank " << rank << ": ✗ AllReduce execution failed, error code: " << result << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    hipStreamSynchronize(stream);
    
    // verify results
    hipMemcpy(h_data.data(), d_data, count * sizeof(float), hipMemcpyDeviceToHost);
    bool success = verifyResults(h_data.data(), count, num_ranks);
    int success_int = success ? 1 : 0;
    
    int global_success;
    MPI_Allreduce(&success_int, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (rank == 0) {
      if (global_success) {
        std::cout << "  ✓ first AllReduce completed, results verified" << std::endl;
        std::cout << std::endl;
      } else {
        std::cout << "  ✗ results verification failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    
    // ===== step 3: multiple calls demonstration =====
    if (rank == 0) {
      std::cout << "[Step 3] multiple calls demonstration..." << std::endl;
    }
    
    for (int iter = 0; iter < 5; iter++) {
      // reinitialize data
      initializeData(h_data.data(), count, rank);
      hipMemcpy(d_data, h_data.data(), count * sizeof(float), hipMemcpyHostToDevice);
      
      // execute AllReduce (completely same call)
      result = executor->Execute(d_data, d_data, count, stream);
      hipStreamSynchronize(stream);
      
      if (result != 0) {
        std::cerr << "  Rank " << rank << ": ✗ call " << iter + 1 << " failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      if (rank == 0) {
        std::cout << "  ✓ call " << iter + 1 << " succeeded" << std::endl;
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << std::endl;
      std::cout << "==================================" << std::endl;
      std::cout << "✓ all tests passed!" << std::endl;
      std::cout << "==================================" << std::endl;
    }
    
    // clean up
    hipFree(d_data);
    hipStreamDestroy(stream);
    
    MPI_Finalize();
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << rank << ": ✗ Exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

/*
 * how to run:
 * 
 * run (8 processes):
 * mpirun --allow-run-as-root -np 8 ./allreduce_example
 * 
 * run (4 processes):
 * mpirun --allow-run-as-root -np 4 ./allreduce_example
 * 
 * usage:
 * 
 * 1. initialize (once, using polymorphism with smart pointer):
 *    std::unique_ptr<AllReduceExecutor<float>> executor(
 *        new IntraNodeAllReduceExecutor<float>(num_ranks, rank, MPI_COMM_WORLD));
 *    
 * 2. use (multiple times):
 *    executor->Execute(d_input, d_output, count, stream);
 *    
 * key points:
 * - uses template-based polymorphism with smart pointer for automatic memory management
 * - data type is specified via template parameter (e.g., float, half)
 * - no need for dtype_size parameter anymore
 * - no need for manual delete - unique_ptr handles cleanup automatically
 * - IPC handles are exchanged automatically via MPI_Allgather
 * - each process runs on an independent GPU (via hipSetDevice(rank))
 * - supports 2/4/6/8 processes (GPUs)
 */
