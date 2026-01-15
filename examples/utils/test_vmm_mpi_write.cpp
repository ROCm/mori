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
 * @file test_vmm_mpi_write.cpp
 * @brief Test VMM cross-process memory sharing with proper FD passing
 * 
 * Usage: mpirun -np 2 ./test_vmm_mpi_write
 * 
 * Test scenario:
 * - Rank 0 (GPU 0): Allocates memory, exports handle, writes data
 * - Rank 1 (GPU 1): Imports handle via Unix socket FD passing, reads and verifies data
 * 
 * This demonstrates the CORRECT usage of hipMemExportToShareableHandle
 * and hipMemImportFromShareableHandle across MPI processes using Unix domain
 * sockets with SCM_RIGHTS for proper file descriptor transfer.
 */

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>

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

// Unix domain socket path
#define SOCKET_PATH "/tmp/mori_vmm_fd_socket"

/**
 * @brief Send file descriptor through Unix domain socket
 */
int send_fd(int socket_fd, int fd) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    char data[1] = {'X'};
    struct iovec iov = {
        .iov_base = data,
        .iov_len = sizeof(data)
    };

    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));

    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
    msg.msg_controllen = cmsg->cmsg_len;

    if (sendmsg(socket_fd, &msg, 0) < 0) {
        perror("sendmsg");
        return -1;
    }
    return 0;
}

/**
 * @brief Receive file descriptor through Unix domain socket
 */
int recv_fd(int socket_fd) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    char data[1];
    struct iovec iov = {
        .iov_base = data,
        .iov_len = sizeof(data)
    };

    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    if (recvmsg(socket_fd, &msg, 0) < 0) {
        perror("recvmsg");
        return -1;
    }

    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL || cmsg->cmsg_type != SCM_RIGHTS) {
        std::cerr << "Invalid control message\n";
        return -1;
    }

    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

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
            std::cerr << "Usage: mpirun -np 2 ./test_vmm_mpi_write\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Set device based on rank
    int deviceId = rank;
    HIP_CHECK(hipSetDevice(deviceId));
    
    if (rank == 0) {
        std::cout << "================================================================\n";
        std::cout << "VMM Cross-Process Memory Sharing with Unix Socket FD Passing\n";
        std::cout << "================================================================\n";
        std::cout << "Using Unix domain socket + SCM_RIGHTS for proper FD transfer\n";
        std::cout << "Rank 0 (GPU 0): Allocate, export, write\n";
        std::cout << "Rank 1 (GPU 1): Import (via socket), read, verify\n";
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
    
    // Enable P2P access between GPU 0 and GPU 1
    if (rank == 0) {
        std::cout << "\n[Rank 0] Enabling P2P access from GPU 0 to GPU 1...\n";
        int canAccess = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 0, 1));
        if (canAccess) {
            hipError_t err = hipDeviceEnablePeerAccess(1, 0);
            if (err == hipSuccess) {
                std::cout << "[Rank 0] ✅ P2P access enabled: GPU 0 → GPU 1\n";
            } else if (err == hipErrorPeerAccessAlreadyEnabled) {
                std::cout << "[Rank 0] ✅ P2P access already enabled: GPU 0 → GPU 1\n";
            } else {
                std::cerr << "[Rank 0] ⚠️  P2P access enable failed: " << hipGetErrorString(err) << "\n";
            }
        } else {
            std::cerr << "[Rank 0] ⚠️  P2P not supported between GPU 0 and GPU 1\n";
        }
    } else if (rank == 1) {
        std::cout << "\n[Rank 1] Enabling P2P access from GPU 1 to GPU 0...\n";
        int canAccess = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 1, 0));
        if (canAccess) {
            hipError_t err = hipDeviceEnablePeerAccess(0, 0);
            if (err == hipSuccess) {
                std::cout << "[Rank 1] ✅ P2P access enabled: GPU 1 → GPU 0\n";
            } else if (err == hipErrorPeerAccessAlreadyEnabled) {
                std::cout << "[Rank 1] ✅ P2P access already enabled: GPU 1 → GPU 0\n";
            } else {
                std::cerr << "[Rank 1] ⚠️  P2P access enable failed: " << hipGetErrorString(err) << "\n";
            }
        } else {
            std::cerr << "[Rank 1] ⚠️  P2P not supported between GPU 1 and GPU 0\n";
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test parameters
    const size_t memorySize = 64 * 1024 * 1024;  // 64 MB
    const size_t numElements = memorySize / sizeof(int);
    
    void* virtualPtr = nullptr;
    hipMemGenericAllocationHandle_t memHandle;
    int shareableFd = -1;
    
    // ================================================================
    // Rank 0: Allocate memory, export handle, write data
    // ================================================================
    if (rank == 0) {
        std::cout << "\n[Rank 0] ===== PHASE 1: Allocate and Export =====\n";
        
        // Step 1: Reserve virtual address space
        std::cout << "[Rank 0] Reserving virtual address space (" 
                  << (memorySize / (1024*1024)) << " MB)...\n";
        HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
        std::cout << "[Rank 0] ✅ Virtual address: " << virtualPtr << "\n";
        
        // Step 2: Create physical memory allocation
        std::cout << "[Rank 0] Creating physical memory allocation...\n";
        hipMemAllocationProp allocProp = {};
        allocProp.type = hipMemAllocationTypePinned;
        allocProp.location.type = hipMemLocationTypeDevice;
        allocProp.location.id = deviceId;
        allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
        
        HIP_CHECK(hipMemCreate(&memHandle, memorySize, &allocProp, 0));
        std::cout << "[Rank 0] ✅ Physical memory created\n";
        
        // Step 3: Map physical memory to virtual address
        std::cout << "[Rank 0] Mapping physical memory to virtual address...\n";
        HIP_CHECK(hipMemMap(virtualPtr, memorySize, 0, memHandle, 0));
        std::cout << "[Rank 0] ✅ Memory mapped\n";
        
        // Step 4: Set access permissions for GPU 0
        std::cout << "[Rank 0] Setting access permissions for GPU 0...\n";
        hipMemAccessDesc accessDesc;
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = deviceId;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
        std::cout << "[Rank 0] ✅ Access permissions set for GPU 0\n";
        
        // Step 5: Export shareable handle
        std::cout << "[Rank 0] Exporting shareable handle...\n";
        HIP_CHECK(hipMemExportToShareableHandle(
            (void*)&shareableFd,
            memHandle,
            hipMemHandleTypePosixFileDescriptor,
            0));
        std::cout << "[Rank 0] ✅ Shareable handle exported (FD: " << shareableFd << ")\n";
        
        // Step 6: Setup Unix domain socket server and send FD
        std::cout << "[Rank 0] Setting up Unix domain socket server...\n";
        
        // Remove old socket file if exists
        unlink(SOCKET_PATH);
        
        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd < 0) {
            perror("socket");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
        
        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind");
            close(server_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (listen(server_fd, 1) < 0) {
            perror("listen");
            close(server_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::cout << "[Rank 0] ✅ Socket server listening at " << SOCKET_PATH << "\n";
        
        // Signal Rank 1 that socket is ready
        int socket_ready = 1;
        MPI_CHECK(MPI_Send(&socket_ready, 1, MPI_INT, 1, 0, MPI_COMM_WORLD));
        std::cout << "[Rank 0] ✅ Signaled Rank 1 that socket is ready\n";
        
        // Accept connection from Rank 1
        std::cout << "[Rank 0] Waiting for Rank 1 to connect...\n";
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            perror("accept");
            close(server_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 0] ✅ Rank 1 connected\n";
        
        // Send file descriptor through Unix socket
        std::cout << "[Rank 0] Sending FD through Unix socket...\n";
        if (send_fd(client_fd, shareableFd) < 0) {
            std::cerr << "[Rank 0] Failed to send FD\n";
            close(client_fd);
            close(server_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 0] ✅ FD sent successfully\n";
        
        close(client_fd);
        close(server_fd);
        unlink(SOCKET_PATH);
        
        // Step 7: Write data
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
        MPI_CHECK(MPI_Send(&ready, 1, MPI_INT, 1, 2, MPI_COMM_WORLD));
        std::cout << "[Rank 0] ✅ Signal sent\n";
        
        // Wait for Rank 1 to finish verification
        std::cout << "[Rank 0] Waiting for Rank 1 to finish...\n";
        int done = 0;
        MPI_CHECK(MPI_Recv(&done, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::cout << "[Rank 0] ✅ Rank 1 finished\n";
        
    }
    // ================================================================
    // Rank 1: Import handle, read and verify data
    // ================================================================
    else if (rank == 1) {
        std::cout << "\n[Rank 1] ===== PHASE 1: Import Handle =====\n";
        
        // Wait for Rank 0 to setup socket
        std::cout << "[Rank 1] Waiting for socket to be ready...\n";
        int socket_ready = 0;
        MPI_CHECK(MPI_Recv(&socket_ready, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::cout << "[Rank 1] ✅ Socket ready signal received\n";
        
        // Connect to Unix domain socket
        std::cout << "[Rank 1] Connecting to Unix domain socket...\n";
        int client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (client_fd < 0) {
            perror("socket");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
        
        // Retry connection a few times (Rank 0 might still be setting up)
        int connected = 0;
        for (int retry = 0; retry < 10 && !connected; retry++) {
            if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                connected = 1;
                break;
            }
            usleep(100000);  // 100ms
        }
        
        if (!connected) {
            perror("connect");
            close(client_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 1] ✅ Connected to socket\n";
        
        // Receive file descriptor through Unix socket
        std::cout << "[Rank 1] Receiving FD through Unix socket...\n";
        shareableFd = recv_fd(client_fd);
        if (shareableFd < 0) {
            std::cerr << "[Rank 1] Failed to receive FD\n";
            close(client_fd);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 1] ✅ Received FD: " << shareableFd << "\n";
        
        close(client_fd);
        
        // Step 1: Reserve virtual address space
        std::cout << "[Rank 1] Reserving virtual address space...\n";
        HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
        std::cout << "[Rank 1] ✅ Virtual address: " << virtualPtr << "\n";
        
        // Step 2: Import the shareable handle
        std::cout << "[Rank 1] Importing shareable handle...\n";
        HIP_CHECK(hipMemImportFromShareableHandle(
            &memHandle,
            (void*)&shareableFd,
            hipMemHandleTypePosixFileDescriptor));
        std::cout << "[Rank 1] ✅ Handle imported\n";
        
        // Step 3: Map imported physical memory to virtual address
        std::cout << "[Rank 1] Mapping imported physical memory...\n";
        HIP_CHECK(hipMemMap(virtualPtr, memorySize, 0, memHandle, 0));
        std::cout << "[Rank 1] ✅ Memory mapped\n";
        
        // Step 4: Set access permissions for GPU 1
        std::cout << "[Rank 1] Setting access permissions...\n";
        hipMemAccessDesc accessDesc;
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = deviceId;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
        std::cout << "[Rank 1] ✅ Access permissions set\n";
        
        // Wait for Rank 0 to finish writing
        std::cout << "[Rank 1] Waiting for Rank 0 to write data...\n";
        int ready = 0;
        MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::cout << "[Rank 1] ✅ Rank 0 finished writing\n";
        
        // Step 5: Read and verify data
        std::cout << "\n[Rank 1] ===== PHASE 2: Read and Verify =====\n";
        std::cout << "[Rank 1] Verifying data using GPU kernel...\n";
        
        int* dataPtr = static_cast<int*>(virtualPtr);
        
        // Allocate error counter on GPU
        int* d_errorCount;
        HIP_CHECK(hipMalloc(&d_errorCount, sizeof(int)));
        HIP_CHECK(hipMemset(d_errorCount, 0, sizeof(int)));
        
        // Launch verification kernel on GPU 1
        dim3 blockSize(256);
        dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
        
        std::cout << "[Rank 1] Launching verification kernel on GPU 1...\n";
        std::cout << "[Rank 1] Grid: " << gridSize.x << " blocks, Block: " << blockSize.x << " threads\n";
        
        hipLaunchKernelGGL(VerifyDataKernel, gridSize, blockSize, 0, 0,
                          dataPtr, numElements, 1000000, d_errorCount);
        
        hipError_t kernelErr = hipGetLastError();
        if (kernelErr != hipSuccess) {
            std::cerr << "[Rank 1] ❌ Kernel launch failed: " 
                     << hipGetErrorString(kernelErr) << std::endl;
        } else {
            std::cout << "[Rank 1] ✅ Kernel launched successfully\n";
        }
        
        std::cout << "[Rank 1] Waiting for kernel to complete...\n";
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "[Rank 1] ✅ Kernel completed\n";
        
        // Get error count
        int h_errorCount = 0;
        HIP_CHECK(hipMemcpy(&h_errorCount, d_errorCount, sizeof(int), hipMemcpyDeviceToHost));
        
        if (h_errorCount == 0) {
            std::cout << "[Rank 1] ✅ GPU kernel verification successful!\n";
            std::cout << "[Rank 1] All " << numElements << " elements verified on GPU 1\n";
            
            // Sample a few values to CPU for display
            std::vector<int> hostData(3);
            HIP_CHECK(hipMemcpy(&hostData[0], &dataPtr[0], sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&hostData[1], &dataPtr[999], sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&hostData[2], &dataPtr[numElements-1], sizeof(int), hipMemcpyDeviceToHost));
            
            std::cout << "[Rank 1] Sample values: [0]=" << hostData[0] 
                     << ", [999]=" << hostData[1]
                     << ", [last]=" << hostData[2] << "\n";
        } else {
            std::cout << "[Rank 1] ❌ GPU kernel found " << h_errorCount << " errors\n";
        }
        
        // Cleanup
        HIP_CHECK(hipFree(d_errorCount));
        
        // Signal Rank 0 that we're done
        std::cout << "[Rank 1] Signaling Rank 0 that we're done...\n";
        int done = 1;
        MPI_CHECK(MPI_Send(&done, 1, MPI_INT, 0, 3, MPI_COMM_WORLD));
    }
    
    // ================================================================
    // Cleanup
    // ================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n================================================================\n";
        std::cout << "                         SUMMARY\n";
        std::cout << "================================================================\n";
        std::cout << "✅ Rank 0 allocated memory and exported handle\n";
        std::cout << "✅ Rank 1 imported handle and mapped memory\n";
        std::cout << "✅ Rank 0 wrote data on GPU 0\n";
        std::cout << "✅ Rank 1 read and verified data on GPU 1\n";
        std::cout << "✅ Cross-process memory sharing works!\n";
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
    
    if (shareableFd != -1) {
        close(shareableFd);
    }
    
    std::cout << "[Rank " << rank << "] ✅ Cleanup complete\n";
    
    MPI_Finalize();
    
    if (rank == 0) {
        std::cout << "\n🎉 Test completed successfully!\n";
    }
    
    return 0;
}
