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

/**
 * @file test_vmm_mpi_fd_direct.cpp
 * @brief Test VMM with direct MPI FD passing (no Unix socket)
 * 
 * Usage: mpirun --allow-run-as-root -np 2 ./test_vmm_mpi_fd_direct
 * 
 * This test directly passes the file descriptor as an integer through MPI
 * to see if it works (spoiler: it probably won't, but let's verify).
 * 
 * Test scenario:
 * - Rank 0: Allocate VMM memory, export FD, send FD via MPI_Send as int
 * - Rank 1: Receive FD via MPI_Recv, try to import and use it
 */

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "[Rank " << rank << "] HIP Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  " << #call << std::endl; \
            std::cerr << "  Error: " << hipGetErrorString(err) << " (" << err << ")" << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(err, error_string, &length); \
            std::cerr << "MPI Error: " << error_string << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Kernel to write data
__global__ void WriteDataKernel(int* data, size_t numElements, int baseValue) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        data[idx] = baseValue + static_cast<int>(idx);
    }
}

// Kernel to verify data
__global__ void VerifyDataKernel(int* data, size_t numElements, int expectedBase, int* errorCount) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        if (data[idx] != expectedBase + static_cast<int>(idx)) {
            atomicAdd(errorCount, 1);
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    
    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Error: This test requires exactly 2 MPI processes\n";
            std::cerr << "Usage: mpirun --allow-run-as-root -np 2 ./test_vmm_mpi_fd_direct\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Set device based on rank
    int deviceId = rank;
    HIP_CHECK(hipSetDevice(deviceId));
    
    if (rank == 0) {
        std::cout << "================================================================\n";
        std::cout << "VMM Test: Direct MPI FD Passing (No Unix Socket)\n";
        std::cout << "================================================================\n";
        std::cout << "This test directly passes FD as integer through MPI_Send/Recv\n";
        std::cout << "Expected: Likely to FAIL because FDs are process-local\n";
        std::cout << "================================================================\n\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Check VMM support
    int vmmSupported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&vmmSupported, 
                                   hipDeviceAttributeVirtualMemoryManagementSupported, 
                                   deviceId));
    
    if (!vmmSupported) {
        std::cerr << "[Rank " << rank << "] Error: GPU " << deviceId 
                  << " does not support VMM\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, deviceId));
    std::cout << "[Rank " << rank << "] Using GPU " << deviceId << ": " 
              << prop.name << " (VMM Supported)\n";
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test parameters
    const size_t memorySize = 64 * 1024 * 1024;  // 64 MB
    const size_t numElements = memorySize / sizeof(int);
    
    void* virtualPtr = nullptr;
    hipMemGenericAllocationHandle_t memHandle;
    int shareableFd = -1;
    
    // ================================================================
    // Rank 0: Allocate memory, export FD, send via MPI
    // ================================================================
    if (rank == 0) {
        std::cout << "\n[Rank 0] ===== PHASE 1: Allocate and Export =====\n";
        
        // Reserve virtual address space
        std::cout << "[Rank 0] Reserving virtual address space (" 
                  << (memorySize / (1024*1024)) << " MB)...\n";
        HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
        std::cout << "[Rank 0] ✅ Virtual address: " << virtualPtr << "\n";
        
        // Create physical memory allocation
        std::cout << "[Rank 0] Creating physical memory allocation...\n";
        hipMemAllocationProp allocProp = {};
        allocProp.type = hipMemAllocationTypePinned;
        allocProp.location.type = hipMemLocationTypeDevice;
        allocProp.location.id = deviceId;
        allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
        
        HIP_CHECK(hipMemCreate(&memHandle, memorySize, &allocProp, 0));
        std::cout << "[Rank 0] ✅ Physical memory created\n";
        
        // Map physical memory to virtual address
        std::cout << "[Rank 0] Mapping physical memory to virtual address...\n";
        HIP_CHECK(hipMemMap(virtualPtr, memorySize, 0, memHandle, 0));
        std::cout << "[Rank 0] ✅ Memory mapped\n";
        
        // Set access permissions
        std::cout << "[Rank 0] Setting access permissions...\n";
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = deviceId;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
        std::cout << "[Rank 0] ✅ Access permissions set\n";
        
        // Export shareable handle
        std::cout << "[Rank 0] Exporting shareable handle...\n";
        HIP_CHECK(hipMemExportToShareableHandle(
            (void*)&shareableFd,
            memHandle,
            hipMemHandleTypePosixFileDescriptor,
            0));
        std::cout << "[Rank 0] ✅ Shareable handle exported (FD: " << shareableFd << ")\n";
        
        // Check FD validity
        int fd_flags = fcntl(shareableFd, F_GETFD);
        if (fd_flags == -1) {
            std::cerr << "[Rank 0] ⚠️  FD " << shareableFd << " is INVALID (fcntl failed)\n";
            perror("fcntl");
        } else {
            std::cout << "[Rank 0] ✅ FD " << shareableFd << " is valid (flags: " << fd_flags << ")\n";
        }
        
        // Send FD directly through MPI as integer
        std::cout << "[Rank 0] Sending FD directly via MPI_Send as integer...\n";
        MPI_CHECK(MPI_Send(&shareableFd, 1, MPI_INT, 1, 0, MPI_COMM_WORLD));
        std::cout << "[Rank 0] ✅ FD sent: " << shareableFd << "\n";
        std::cout << "[Rank 0] ⚠️  WARNING: This FD is only valid in Rank 0's process!\n";
        
        // Write data
        std::cout << "\n[Rank 0] ===== PHASE 2: Write Data =====\n";
        std::cout << "[Rank 0] Launching write kernel...\n";
        
        int* dataPtr = static_cast<int*>(virtualPtr);
        dim3 blockSize(256);
        dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
        
        hipLaunchKernelGGL(WriteDataKernel, gridSize, blockSize, 0, 0,
                          dataPtr, numElements, 1000000);
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "[Rank 0] ✅ Wrote data (base value: 1000000)\n";
        
        // Verify on CPU
        std::cout << "[Rank 0] Verifying data on CPU...\n";
        std::vector<int> hostData(std::min(size_t(1000), numElements));
        HIP_CHECK(hipMemcpy(hostData.data(), dataPtr, 
                           hostData.size() * sizeof(int), hipMemcpyDeviceToHost));
        
        bool ok = true;
        for (size_t i = 0; i < hostData.size(); i++) {
            if (hostData[i] != 1000000 + (int)i) {
                std::cout << "[Rank 0] ❌ Mismatch at [" << i << "]: expected " 
                         << (1000000 + i) << ", got " << hostData[i] << "\n";
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Rank 0] ✅ Data verified: [0]=" << hostData[0] 
                     << ", [999]=" << hostData[999] << "\n";
        }
        
        // Signal Rank 1 that data is ready
        std::cout << "[Rank 0] Signaling Rank 1 that data is ready...\n";
        int ready = 1;
        MPI_CHECK(MPI_Send(&ready, 1, MPI_INT, 1, 1, MPI_COMM_WORLD));
        
        // Wait for Rank 1 to finish (it might fail early or complete)
        std::cout << "[Rank 0] Waiting for Rank 1 result...\n";
        int result = 0;
        MPI_Status status;
        MPI_CHECK(MPI_Recv(&result, 1, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
        
        std::cout << "[Rank 0] Received result from Rank 1 (tag=" << status.MPI_TAG << ", code=" << result << ")\n";
        
        if (result == 0) {
            std::cout << "[Rank 0] ✅ Rank 1 reported SUCCESS\n";
        } else if (result == 1) {
            std::cout << "[Rank 0] ✅ TEST SUCCESSFUL: Rank 1 confirmed FD is invalid!\n";
            std::cout << "[Rank 0] This proves FDs cannot be passed as integers via MPI\n";
        } else {
            std::cout << "[Rank 0] ⚠️  Rank 1 reported error code: " << result << "\n";
        }
    }
    // ================================================================
    // Rank 1: Receive FD via MPI, try to import
    // ================================================================
    else if (rank == 1) {
        std::cout << "\n[Rank 1] ===== PHASE 1: Receive and Import FD =====\n";
        
        // Receive FD directly through MPI as integer
        std::cout << "[Rank 1] Receiving FD via MPI_Recv...\n";
        MPI_CHECK(MPI_Recv(&shareableFd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::cout << "[Rank 1] ✅ Received FD: " << shareableFd << "\n";
        
        // Check if this FD is valid in Rank 1's process
        std::cout << "[Rank 1] Checking if FD is valid in this process...\n";
        int fd_flags = fcntl(shareableFd, F_GETFD);
        if (fd_flags == -1) {
            std::cerr << "[Rank 1] ❌ FD " << shareableFd << " is INVALID in Rank 1's process!\n";
            perror("[Rank 1] fcntl");
            std::cerr << "[Rank 1] This confirms: FDs cannot be passed as integers between processes\n";
            std::cerr << "[Rank 1] \n";
            std::cerr << "[Rank 1] ===== CONCLUSION =====\n";
            std::cerr << "[Rank 1] ✅ TEST SUCCESSFUL: Proved FD passing via MPI doesn't work!\n";
            std::cerr << "[Rank 1] Reason: File descriptors are process-local identifiers\n";
            std::cerr << "[Rank 1] The integer value '37' was transmitted, but it's meaningless in Rank 1\n";
            std::cerr << "[Rank 1] Solution: Must use Unix domain socket + SCM_RIGHTS\n";
            std::cerr << "[Rank 1] ========================\n";
            
            // Signal Rank 0 immediately with failure code
            std::cerr << "[Rank 1] Signaling Rank 0 about failure...\n";
            int result = 1;  // FD invalid
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 99, MPI_COMM_WORLD));
            
            std::cerr << "[Rank 1] Skipping to cleanup and barrier...\n";
            // Jump to cleanup section - don't exit early, need to reach MPI_Barrier
            goto cleanup_rank1;
        } else {
            std::cout << "[Rank 1] ⚠️  FD " << shareableFd << " appears valid (flags: " << fd_flags << ")\n";
            std::cout << "[Rank 1] BUT: This might be a coincidence (same FD number exists in both processes)\n";
        }
        
        // Try to reserve virtual address space
        std::cout << "[Rank 1] Reserving virtual address space...\n";
        HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
        std::cout << "[Rank 1] ✅ Virtual address: " << virtualPtr << "\n";
        
        // Try to import the shareable handle
        std::cout << "[Rank 1] Attempting to import shareable handle...\n";
        std::cout << "[Rank 1] This is likely to FAIL with 'invalid argument'\n";
        
        hipError_t importErr = hipMemImportFromShareableHandle(
            &memHandle,
            (void*)&shareableFd,
            hipMemHandleTypePosixFileDescriptor);
        
        if (importErr != hipSuccess) {
            std::cerr << "[Rank 1] ❌ hipMemImportFromShareableHandle FAILED!\n";
            std::cerr << "[Rank 1] Error: " << hipGetErrorString(importErr) << " (" << importErr << ")\n";
            std::cerr << "[Rank 1] \n";
            std::cerr << "[Rank 1] ===== CONCLUSION =====\n";
            std::cerr << "[Rank 1] FD passing via MPI_Send/Recv does NOT work!\n";
            std::cerr << "[Rank 1] Reason: File descriptors are process-local identifiers\n";
            std::cerr << "[Rank 1] Solution: Must use Unix domain socket + SCM_RIGHTS\n";
            std::cerr << "[Rank 1] ========================\n";
            
            // Clean up
            HIP_CHECK(hipMemAddressFree(virtualPtr, memorySize));
            
            // Report failure to Rank 0
            int ready = 0;
            MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            int result = 2;  // Import failed
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
            
            MPI_Finalize();
            return 1;
        }
        
        std::cout << "[Rank 1] ✅ Handle imported successfully!\n";
        std::cout << "[Rank 1] ⚠️  Unexpected! Let's continue and see if it actually works...\n";
        
        // Try to map the imported memory
        std::cout << "[Rank 1] Mapping imported physical memory...\n";
        hipError_t mapErr = hipMemMap(virtualPtr, memorySize, 0, memHandle, 0);
        if (mapErr != hipSuccess) {
            std::cerr << "[Rank 1] ❌ hipMemMap FAILED: " << hipGetErrorString(mapErr) << "\n";
            
            HIP_CHECK(hipMemRelease(memHandle));
            HIP_CHECK(hipMemAddressFree(virtualPtr, memorySize));
            
            int ready = 0;
            MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            int result = 3;  // Map failed
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
            
            MPI_Finalize();
            return 1;
        }
        std::cout << "[Rank 1] ✅ Memory mapped\n";
        
        // Set access permissions
        std::cout << "[Rank 1] Setting access permissions...\n";
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = deviceId;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
        std::cout << "[Rank 1] ✅ Access permissions set\n";
        
        // Wait for Rank 0 to write data
        std::cout << "[Rank 1] Waiting for Rank 0 to write data...\n";
        int ready = 0;
        MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::cout << "[Rank 1] ✅ Rank 0 finished writing\n";
        
        // Try to verify data
        std::cout << "\n[Rank 1] ===== PHASE 2: Verify Data =====\n";
        std::cout << "[Rank 1] Attempting to verify data using GPU kernel...\n";
        
        int* dataPtr = static_cast<int*>(virtualPtr);
        
        // Allocate error counter on GPU
        int* d_errorCount;
        HIP_CHECK(hipMalloc(&d_errorCount, sizeof(int)));
        HIP_CHECK(hipMemset(d_errorCount, 0, sizeof(int)));
        
        // Launch verification kernel
        dim3 blockSize(256);
        dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
        
        std::cout << "[Rank 1] Launching verification kernel...\n";
        hipLaunchKernelGGL(VerifyDataKernel, gridSize, blockSize, 0, 0,
                          dataPtr, numElements, 1000000, d_errorCount);
        
        hipError_t syncErr = hipDeviceSynchronize();
        if (syncErr != hipSuccess) {
            std::cerr << "[Rank 1] ❌ Kernel execution FAILED: " 
                     << hipGetErrorString(syncErr) << "\n";
            std::cerr << "[Rank 1] This is expected: memory access fault\n";
            
            HIP_CHECK(hipFree(d_errorCount));
            HIP_CHECK(hipMemUnmap(virtualPtr, memorySize));
            HIP_CHECK(hipMemAddressFree(virtualPtr, memorySize));
            HIP_CHECK(hipMemRelease(memHandle));
            
            int result = 4;  // Kernel execution failed
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
            
            MPI_Finalize();
            return 1;
        }
        
        std::cout << "[Rank 1] ✅ Kernel completed without errors\n";
        
        // Get error count
        int h_errorCount = 0;
        HIP_CHECK(hipMemcpy(&h_errorCount, d_errorCount, sizeof(int), hipMemcpyDeviceToHost));
        
        if (h_errorCount == 0) {
            std::cout << "[Rank 1] ✅ GPU kernel verification successful!\n";
            std::cout << "[Rank 1] All " << numElements << " elements verified\n";
            
            // Sample values
            std::vector<int> hostData(3);
            HIP_CHECK(hipMemcpy(&hostData[0], &dataPtr[0], sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&hostData[1], &dataPtr[999], sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&hostData[2], &dataPtr[numElements-1], sizeof(int), hipMemcpyDeviceToHost));
            
            std::cout << "[Rank 1] Sample values: [0]=" << hostData[0] 
                     << ", [999]=" << hostData[1]
                     << ", [last]=" << hostData[2] << "\n";
            
            int result = 0;  // Success
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
        } else {
            std::cout << "[Rank 1] ❌ GPU kernel found " << h_errorCount << " errors\n";
            
            int result = 5;  // Verification failed
            MPI_CHECK(MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
        }
        
        // Cleanup
        HIP_CHECK(hipFree(d_errorCount));
    }
    
cleanup_rank1:  // Label for early exit from Rank 1
    // ================================================================
    // Cleanup
    // ================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n================================================================\n";
        std::cout << "                    TEST SUMMARY\n";
        std::cout << "================================================================\n";
        std::cout << "This test demonstrated:\n";
        std::cout << "- FD passing via MPI_Send/Recv as integer\n";
        std::cout << "- Expected to fail at hipMemImportFromShareableHandle\n";
        std::cout << "- Reason: FDs are process-local, not global identifiers\n";
        std::cout << "- Solution: Use Unix socket + SCM_RIGHTS for proper FD passing\n";
        std::cout << "================================================================\n";
    }
    
    std::cout << "[Rank " << rank << "] Cleaning up...\n";
    
    if (virtualPtr) {
        HIP_CHECK(hipMemUnmap(virtualPtr, memorySize));
        HIP_CHECK(hipMemAddressFree(virtualPtr, memorySize));
    }
    
    if (memHandle) {
        HIP_CHECK(hipMemRelease(memHandle));
    }
    
    if (shareableFd != -1 && rank == 0) {
        close(shareableFd);
    }
    
    MPI_Finalize();
    
    return 0;
}
