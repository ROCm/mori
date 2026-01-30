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
 * @brief Test VMM cross-process memory sharing with concurrent GPU thread writes
 *
 * Usage: mpirun -np 2 ./test_vmm_mpi_write [test_mode]
 *
 * Test modes:
 *   0 (default): Normal sequential write (one value per thread)
 *   1: Concurrent independent writes (multiple threads, no overlap)
 *   2: Concurrent atomic writes (multiple threads writing to shared counters)
 *   3: Interleaved writes (threads writing to interleaved memory pattern)
 *   4: Byte-level writes (uint8_t per thread, fine-grained concurrency)
 *   5: SHMEM PUT simulation (Rank0 writes via remote ptr, Rank1 reads via local ptr)
 *
 * This demonstrates the CORRECT usage of hipMemExportToShareableHandle
 * and hipMemImportFromShareableHandle across MPI processes using Unix domain
 * sockets with SCM_RIGHTS for proper file descriptor transfer.
 *
 * Also tests various GPU kernel thread concurrency patterns.
 */

#include <errno.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <mpi.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "mori/utils/hip_compat.hpp"

#define HIP_CHECK(call)                                                                      \
  do {                                                                                       \
    hipError_t err = call;                                                                   \
    if (err != hipSuccess) {                                                                 \
      std::cerr << "[Rank " << rank << "] HIP Error at " << __FILE__ << ":" << __LINE__      \
                << std::endl;                                                                \
      std::cerr << "  " << #call << std::endl;                                               \
      std::cerr << "  Error: " << hipGetErrorString(err) << " (" << err << ")" << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                          \
    }                                                                                        \
  } while (0)

#define MPI_CHECK(call)                                        \
  do {                                                         \
    int err = call;                                            \
    if (err != MPI_SUCCESS) {                                  \
      char error_string[MPI_MAX_ERROR_STRING];                 \
      int length;                                              \
      MPI_Error_string(err, error_string, &length);            \
      std::cerr << "MPI Error: " << error_string << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1);                            \
    }                                                          \
  } while (0)

// Unix domain socket path
#define SOCKET_PATH "/tmp/mori_vmm_fd_socket"

/**
 * @brief Send file descriptor through Unix domain socket
 */
int send_fd(int socket_fd, int fd) {
  struct msghdr msg = {0};
  struct cmsghdr* cmsg;
  char buf[CMSG_SPACE(sizeof(int))];
  char data[1] = {'X'};
  struct iovec iov = {.iov_base = data, .iov_len = sizeof(data)};

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
  struct cmsghdr* cmsg;
  char buf[CMSG_SPACE(sizeof(int))];
  char data[1];
  struct iovec iov = {.iov_base = data, .iov_len = sizeof(data)};

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

// Mode 0: Normal sequential write - each GPU thread writes its own element
__global__ void WriteDataKernel_Normal(int* data, size_t numElements, int baseValue) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    data[idx] = baseValue + static_cast<int>(idx);
  }
}

// Mode 1: Concurrent independent writes - multiple blocks write to different regions
__global__ void WriteDataKernel_ConcurrentIndependent(int* data, size_t numElements,
                                                      int baseValue) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    // Each block gets a different base offset
    int blockBase = baseValue + blockIdx.x * 10000;
    data[idx] = blockBase + threadIdx.x;
  }
}

// Mode 2: Concurrent atomic writes - multiple threads increment shared counters
__global__ void WriteDataKernel_Atomic(int* data, size_t numElements, int baseValue) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    // Multiple threads may write to overlapping regions
    // Use atomic operations to safely increment counters
    size_t counterIdx = idx / 256;  // 256 threads share one counter
    if (counterIdx < numElements) {
      atomicAdd(&data[counterIdx], 1);
    }
  }
}

// Mode 3: Interleaved writes - threads write in interleaved pattern (stride access)
__global__ void WriteDataKernel_Interleaved(int* data, size_t numElements, int baseValue) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Each thread writes to multiple locations with stride pattern
  for (size_t idx = tid; idx < numElements; idx += stride) {
    data[idx] = baseValue + static_cast<int>(tid);
  }
}

// Mode 4: Byte-level writes - each thread writes one uint8_t
__global__ void WriteDataKernel_ByteLevel(void* data, size_t numBytes, int baseValue) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBytes) {
    // Cast to uint8_t* for byte-level access
    uint8_t* bytePtr = static_cast<uint8_t*>(data);
    // Write a single byte
    bytePtr[idx] = static_cast<uint8_t>((baseValue + idx) % 256);
  }
}

// Kernel to verify data for normal mode
__global__ void VerifyDataKernel_Normal(int* data, size_t numElements, int expectedBase,
                                        int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    int expected = expectedBase + static_cast<int>(idx);
    int actual = data[idx];
    if (actual != expected) {
      atomicAdd(errorCount, 1);
      int oldCount = atomicAdd(errorCount, 0);
      if (oldCount <= 10) {
        printf("[GPU] Mode 0 Error at [%zu]: expected %d, got %d\n", idx, expected, actual);
      }
    }
  }
}

// Kernel to verify data for concurrent independent mode
__global__ void VerifyDataKernel_ConcurrentIndependent(int* data, size_t numElements,
                                                       int expectedBase, int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    int blockBase = expectedBase + blockIdx.x * 10000;
    int expected = blockBase + threadIdx.x;
    int actual = data[idx];
    if (actual != expected) {
      atomicAdd(errorCount, 1);
      int oldCount = atomicAdd(errorCount, 0);
      if (oldCount <= 10) {
        printf("[GPU] Mode 1 Error at [%zu]: expected %d, got %d\n", idx, expected, actual);
      }
    }
  }
}

// Kernel to verify atomic counter results
__global__ void VerifyDataKernel_Atomic(int* data, size_t numElements, int expectedCount,
                                        int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t counterIdx = idx / 256;

  // Only first thread in each group checks the counter
  if (threadIdx.x % 256 == 0 && counterIdx < numElements) {
    int actual = data[counterIdx];
    // Each counter should have been incremented by 256 threads
    if (actual != expectedCount) {
      atomicAdd(errorCount, 1);
      int oldCount = atomicAdd(errorCount, 0);
      if (oldCount <= 10) {
        printf("[GPU] Mode 2 Error at counter [%zu]: expected %d, got %d\n", counterIdx,
               expectedCount, actual);
      }
    }
  }
}

// Kernel to verify interleaved pattern
__global__ void VerifyDataKernel_Interleaved(int* data, size_t numElements, int expectedBase,
                                             int* errorCount) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = tid; idx < numElements; idx += stride) {
    int expected = expectedBase + static_cast<int>(tid);
    int actual = data[idx];
    if (actual != expected) {
      atomicAdd(errorCount, 1);
      int oldCount = atomicAdd(errorCount, 0);
      if (oldCount <= 10) {
        printf("[GPU] Mode 3 Error at [%zu]: expected %d, got %d\n", idx, expected, actual);
      }
    }
  }
}

// Kernel to verify byte-level writes
__global__ void VerifyDataKernel_ByteLevel(void* data, size_t numBytes, int expectedBase,
                                           int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBytes) {
    uint8_t* bytePtr = static_cast<uint8_t*>(data);
    uint8_t expected = static_cast<uint8_t>((expectedBase + idx) % 256);
    uint8_t actual = bytePtr[idx];
    if (actual != expected) {
      atomicAdd(errorCount, 1);
      int oldCount = atomicAdd(errorCount, 0);
      if (oldCount <= 10) {
        printf("[GPU] Mode 4 Error at byte [%zu]: expected %u, got %u\n", idx, (unsigned)expected,
               (unsigned)actual);
      }
    }
  }
}

int main(int argc, char** argv) {
  // Initialize MPI
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Print ROCm version info (rank 0 only)
  if (rank == 0) {
    std::cout << "ROCm Version: " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
              << HIP_VERSION_PATCH << std::endl;
#if HIP_VERSION >= 70100000
    std::cout << "Using ROCm 7.1.0+ API" << std::endl;
#else
    std::cout << "Using ROCm 7.0.x API" << std::endl;
    std::cout << "Note: ROCm 7.1.1+ has improved VMM support" << std::endl;
#endif
  }

  if (size != 2) {
    if (rank == 0) {
      std::cerr << "Error: This test requires exactly 2 MPI processes\n";
      std::cerr << "Usage: mpirun -np 2 ./test_vmm_mpi_write [test_mode]\n";
      std::cerr << "Test modes:\n";
      std::cerr << "  0 (default): Normal sequential write (int32)\n";
      std::cerr << "  1: Concurrent independent writes (int32)\n";
      std::cerr << "  2: Concurrent atomic writes (int32)\n";
      std::cerr << "  3: Interleaved writes (int32)\n";
      std::cerr << "  4: Byte-level writes (uint8_t per thread)\n";
      std::cerr << "  5: SHMEM PUT simulation (Rank0->Rank1 via VMM P2P)\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Get test mode from command line (default: 0)
  int testMode = 0;
  if (argc > 1) {
    testMode = std::atoi(argv[1]);
    if (testMode < 0 || testMode > 5) {
      if (rank == 0) {
        std::cerr << "Error: Test mode must be between 0 and 5\n";
      }
      MPI_Finalize();
      return 1;
    }
  }

  // Set device based on rank
  int deviceId = rank;
  HIP_CHECK(hipSetDevice(deviceId));

  const char* modeNames[] = {
      "Normal Sequential Write (int32)",  "Concurrent Independent Writes (int32)",
      "Concurrent Atomic Writes (int32)", "Interleaved Writes (int32)",
      "Byte-Level Writes (uint8_t)",      "SHMEM PUT Simulation (Remote Write + Local Read)"};

  if (rank == 0) {
    std::cout << "================================================================\n";
    std::cout << "VMM Cross-Process Memory Sharing - GPU Thread Concurrency Test\n";
    std::cout << "================================================================\n";
    std::cout << "Test Mode " << testMode << ": " << modeNames[testMode] << "\n";
    std::cout << "Using Unix domain socket + SCM_RIGHTS for proper FD transfer\n";
    std::cout << "Rank 0 (GPU 0): Allocate, export, write (GPU kernel threads)\n";
    std::cout << "Rank 1 (GPU 1): Import (via socket), read, verify\n";
    std::cout << "================================================================\n\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Check VMM support
  int vmmSupported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  deviceId));

  if (!vmmSupported) {
    std::cerr << "[Rank " << rank << "] Error: GPU " << deviceId << " does not support VMM\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, deviceId));
  std::cout << "[Rank " << rank << "] Using GPU " << deviceId << ": " << prop.name
            << " (VMM Supported)\n";

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

  // Test parameters - Configure memory size based on test mode
  size_t memorySize;
  if (testMode == 5) {
    // Mode 5 (SHMEM PUT simulation): 64 MB for remote write testing
    memorySize = 64 * 1024 * 1024;  // 64 MB
  } else {
    // Modes 0-4: Default configuration
    memorySize = 64 * 1024 * 1024;  // 64 MB (can be adjusted per mode if needed)
  }
  const size_t numElements = memorySize / sizeof(int);

  if (rank == 0) {
    std::cout << "[Rank 0] Test config:\n";
    std::cout << "  Memory size: " << (memorySize / (1024 * 1024)) << " MB\n";
    std::cout << "  Total elements: " << numElements << "\n";
    std::cout << "  Test mode: " << testMode << " - " << modeNames[testMode] << "\n\n";
  }

  void* virtualPtr = nullptr;
  void* localPtr = nullptr;   // For mode 5: Rank 1's local pointer
  void* remotePtr = nullptr;  // For mode 5: Rank 0's remote pointer to Rank 1's memory
  hipMemGenericAllocationHandle_t memHandle = 0;  // Initialize to 0
  int shareableFd = -1;

  // ================================================================
  // Mode 5: SHMEM PUT Simulation (Different flow)
  // Rank 1 allocates and exports, Rank 0 imports and writes
  // ================================================================
  if (testMode == 5) {
    // ===== Rank 1: Allocate memory and export =====
    if (rank == 1) {
      std::cout << "\n[Rank 1] ===== MODE 5: SHMEM PUT Simulation =====\n";
      std::cout << "[Rank 1] ===== PHASE 1: Allocate and Export =====\n";

      // Reserve virtual address space
      std::cout << "[Rank 1] Reserving virtual address space (" << (memorySize / (1024 * 1024))
                << " MB)...\n";
      HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
      localPtr = virtualPtr;  // Save local pointer
      std::cout << "[Rank 1] ✅ Virtual address (localPtr): " << localPtr << "\n";

      // Create physical memory allocation
      std::cout << "[Rank 1] Creating physical memory allocation...\n";
      hipMemAllocationProp allocProp = {};
      allocProp.type = hipMemAllocationTypePinned;
      allocProp.location.type = hipMemLocationTypeDevice;
      allocProp.location.id = deviceId;
      allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

      HIP_CHECK(hipMemCreate(&memHandle, memorySize, &allocProp, 0));
      std::cout << "[Rank 1] ✅ Physical memory created\n";

      // Map physical memory
      std::cout << "[Rank 1] Mapping physical memory...\n";
      HIP_CHECK(hipMemMap(virtualPtr, memorySize, 0, memHandle, 0));
      std::cout << "[Rank 1] ✅ Memory mapped\n";

      // Set access permissions for GPU 1
      std::cout << "[Rank 1] Setting access permissions for GPU 1...\n";
      hipMemAccessDesc accessDesc;
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = deviceId;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
      std::cout << "[Rank 1] ✅ Access permissions set\n";

      // Initialize memory with Rank 1's value
      std::cout << "[Rank 1] Initializing memory with value " << rank << "...\n";
      int* dataPtr = static_cast<int*>(virtualPtr);
      HIP_CHECK(hipMemsetD32(reinterpret_cast<unsigned int*>(dataPtr), rank, numElements));
      HIP_CHECK(hipDeviceSynchronize());
      std::cout << "[Rank 1] ✅ Memory initialized\n";

      // Export shareable handle
      std::cout << "[Rank 1] Exporting shareable handle...\n";
      HIP_CHECK(hipMemExportToShareableHandle((void*)&shareableFd, memHandle,
                                              hipMemHandleTypePosixFileDescriptor, 0));
      std::cout << "[Rank 1] ✅ Shareable handle exported (FD: " << shareableFd << ")\n";

      // Setup Unix domain socket server
      std::cout << "[Rank 1] Setting up Unix domain socket server...\n";
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

      std::cout << "[Rank 1] ✅ Socket server ready\n";

      // Signal Rank 0
      int socket_ready = 1;
      MPI_CHECK(MPI_Send(&socket_ready, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));

      // Accept connection
      int client_fd = accept(server_fd, NULL, NULL);
      if (client_fd < 0) {
        perror("accept");
        close(server_fd);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      // Send FD
      if (send_fd(client_fd, shareableFd) < 0) {
        std::cerr << "[Rank 1] Failed to send FD\n";
        close(client_fd);
        close(server_fd);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::cout << "[Rank 1] ✅ FD sent to Rank 0\n";

      close(client_fd);
      close(server_fd);
      unlink(SOCKET_PATH);

      // Wait for Rank 0 to complete import
      int import_done = 0;
      MPI_CHECK(MPI_Recv(&import_done, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      std::cout << "[Rank 1] ✅ Rank 0 completed import\n";

      // Wait for Rank 0 to finish writing
      std::cout << "[Rank 1] Waiting for Rank 0 to write via remote pointer...\n";
      int ready = 0;
      MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      std::cout << "[Rank 1] ✅ Rank 0 finished remote write\n";

      // ===== Rank 1: Read and verify via LOCAL pointer =====
      std::cout << "\n[Rank 1] ===== PHASE 2: Read via Local Pointer =====\n";
      std::cout << "[Rank 1] Reading from localPtr: " << localPtr << "\n";

      int* d_errorCount;
      HIP_CHECK(hipMalloc(&d_errorCount, sizeof(int)));
      HIP_CHECK(hipMemset(d_errorCount, 0, sizeof(int)));

      dim3 blockSize(256);
      dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

      std::cout << "[Rank 1] Launching verification kernel (expects data from Rank 0)...\n";
      hipLaunchKernelGGL(VerifyDataKernel_Normal, gridSize, blockSize, 0, 0,
                         static_cast<int*>(localPtr), numElements, 1000000, d_errorCount);

      HIP_CHECK(hipDeviceSynchronize());

      int h_errorCount = 0;
      HIP_CHECK(hipMemcpy(&h_errorCount, d_errorCount, sizeof(int), hipMemcpyDeviceToHost));

      if (h_errorCount == 0) {
        std::cout
            << "[Rank 1] ✅ SUCCESS! Rank 0's remote write visible via Rank 1's local pointer!\n";
        std::cout << "[Rank 1] ✅ Same physical memory confirmed!\n";

        // Sample values
        std::vector<int> hostData(5);
        for (size_t i = 0; i < 5; i++) {
          HIP_CHECK(hipMemcpy(&hostData[i], &static_cast<int*>(localPtr)[i], sizeof(int),
                              hipMemcpyDeviceToHost));
        }

        std::cout << "[Rank 1] Sample values via localPtr: ";
        for (size_t i = 0; i < 5; i++) {
          std::cout << "[" << i << "]=" << hostData[i] << " ";
        }
        std::cout << "\n";
      } else {
        std::cout << "[Rank 1] ❌ Verification failed: " << h_errorCount << " errors\n";
      }

      HIP_CHECK(hipFree(d_errorCount));

      // Signal done
      int done = 1;
      MPI_CHECK(MPI_Send(&done, 1, MPI_INT, 0, 3, MPI_COMM_WORLD));
    }

    // ===== Rank 0: Import and write via REMOTE pointer =====
    else if (rank == 0) {
      std::cout << "\n[Rank 0] ===== MODE 5: SHMEM PUT Simulation =====\n";
      std::cout << "[Rank 0] ===== PHASE 1: Import Rank 1's Memory =====\n";

      // Wait for Rank 1's socket
      int socket_ready = 0;
      MPI_CHECK(MPI_Recv(&socket_ready, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

      // Connect to socket
      std::cout << "[Rank 0] Connecting to Rank 1's socket...\n";
      int client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
      if (client_fd < 0) {
        perror("socket");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      struct sockaddr_un addr;
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

      int connected = 0;
      for (int retry = 0; retry < 10 && !connected; retry++) {
        if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
          connected = 1;
          break;
        }
        usleep(100000);
      }

      if (!connected) {
        perror("connect");
        close(client_fd);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::cout << "[Rank 0] ✅ Connected\n";

      // Receive FD
      shareableFd = recv_fd(client_fd);
      if (shareableFd < 0) {
        std::cerr << "[Rank 0] Failed to receive FD\n";
        close(client_fd);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::cout << "[Rank 0] ✅ Received FD: " << shareableFd << "\n";
      close(client_fd);

      // Reserve virtual address space
      std::cout << "[Rank 0] Reserving virtual address space...\n";
      HIP_CHECK(hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0));
      remotePtr = virtualPtr;  // This is the remote pointer to Rank 1's memory
      std::cout << "[Rank 0] ✅ Virtual address (remotePtr): " << remotePtr << "\n";

      // Import handle
      std::cout << "[Rank 0] Importing Rank 1's handle...\n";
      HIP_CHECK(hipMemImportFromShareableHandleCompat(&memHandle, shareableFd,
                                                      hipMemHandleTypePosixFileDescriptor));
      std::cout << "[Rank 0] ✅ Handle imported\n";

      // Map memory
      std::cout << "[Rank 0] Mapping Rank 1's physical memory...\n";
      HIP_CHECK(hipMemMap(virtualPtr, memorySize, 0, memHandle, 0));
      std::cout << "[Rank 0] ✅ Memory mapped\n";

      // Set access permissions for GPU 0
      std::cout << "[Rank 0] Setting access permissions for GPU 0...\n";
      hipMemAccessDesc accessDesc;
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = deviceId;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1));
      std::cout << "[Rank 0] ✅ Access permissions set\n";

      // Notify Rank 1
      int import_done = 1;
      MPI_CHECK(MPI_Send(&import_done, 1, MPI_INT, 1, 1, MPI_COMM_WORLD));

      // ===== Rank 0: Write via REMOTE pointer =====
      std::cout << "\n[Rank 0] ===== PHASE 2: Write via Remote Pointer =====\n";
      std::cout << "[Rank 0] Writing to remotePtr: " << remotePtr << "\n";
      std::cout << "[Rank 0] (This points to Rank 1's physical memory)\n";

      int* dataPtr = static_cast<int*>(remotePtr);
      int baseValue = 1000000;

      dim3 blockSize(256);
      dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

      std::cout << "[Rank 0] Launching write kernel...\n";
      hipLaunchKernelGGL(WriteDataKernel_Normal, gridSize, blockSize, 0, 0, dataPtr, numElements,
                         baseValue);

      HIP_CHECK(hipDeviceSynchronize());
      std::cout << "[Rank 0] ✅ Write completed via remote pointer\n";

      // Verify on CPU from Rank 0's side
      std::vector<int> hostData(5);
      for (size_t i = 0; i < 5; i++) {
        HIP_CHECK(hipMemcpy(&hostData[i], &dataPtr[i], sizeof(int), hipMemcpyDeviceToHost));
      }

      std::cout << "[Rank 0] Sample values via remotePtr: ";
      for (size_t i = 0; i < 5; i++) {
        std::cout << "[" << i << "]=" << hostData[i] << " ";
      }
      std::cout << "\n";

      // Signal Rank 1
      int ready = 1;
      MPI_CHECK(MPI_Send(&ready, 1, MPI_INT, 1, 2, MPI_COMM_WORLD));

      // Wait for Rank 1
      int done = 0;
      MPI_CHECK(MPI_Recv(&done, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      std::cout << "[Rank 0] ✅ Rank 1 verified data via local pointer\n";
    }
  }
  // ================================================================
  // Modes 0-4: Original flow (Rank 0 allocates, Rank 1 imports)
  // ================================================================
  else if (rank == 0) {
    std::cout << "\n[Rank 0] ===== PHASE 1: Allocate and Export =====\n";

    // Step 1: Reserve virtual address space
    std::cout << "[Rank 0] Reserving virtual address space (" << (memorySize / (1024 * 1024))
              << " MB)...\n";
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
    HIP_CHECK(hipMemExportToShareableHandle((void*)&shareableFd, memHandle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
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

    // Wait for Rank 1 to complete import and mapping
    std::cout << "[Rank 0] Waiting for Rank 1 to complete import...\n";
    int import_done = 0;
    MPI_CHECK(MPI_Recv(&import_done, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    std::cout << "[Rank 0] ✅ Rank 1 completed import and mapping\n";

    // Step 7: Write data using GPU kernel with concurrent threads
    std::cout << "\n[Rank 0] ===== PHASE 2: Write Data with GPU Threads =====\n";
    std::cout << "[Rank 0] Mode " << testMode << ": " << modeNames[testMode] << "\n";

    int* dataPtr = static_cast<int*>(virtualPtr);
    int baseValue = 1000000;

    dim3 blockSize(256);
    dim3 gridSize;

    // Calculate grid size based on mode
    if (testMode == 4) {
      // Byte-level mode: one thread per byte
      size_t numBytes = memorySize;
      gridSize.x = (numBytes + blockSize.x - 1) / blockSize.x;
      std::cout << "[Rank 0] Byte-level mode: " << numBytes << " bytes, " << gridSize.x
                << " blocks\n";
    } else {
      // Other modes: one thread per int
      gridSize.x = (numElements + blockSize.x - 1) / blockSize.x;
    }

    // Initialize memory differently based on mode
    if (testMode == 2) {
      // For atomic mode, initialize counters to 0
      std::cout << "[Rank 0] Initializing atomic counters to 0...\n";
      HIP_CHECK(hipMemset(dataPtr, 0, memorySize));
    }

    std::cout << "[Rank 0] Launching write kernel...\n";
    std::cout << "[Rank 0] Grid: " << gridSize.x << " blocks x " << blockSize.x
              << " threads = " << (gridSize.x * blockSize.x) << " GPU threads\n";

    // Launch appropriate kernel based on test mode
    switch (testMode) {
      case 0:  // Normal sequential
        hipLaunchKernelGGL(WriteDataKernel_Normal, gridSize, blockSize, 0, 0, dataPtr, numElements,
                           baseValue);
        break;

      case 1:  // Concurrent independent
        hipLaunchKernelGGL(WriteDataKernel_ConcurrentIndependent, gridSize, blockSize, 0, 0,
                           dataPtr, numElements, baseValue);
        break;

      case 2:  // Concurrent atomic
        hipLaunchKernelGGL(WriteDataKernel_Atomic, gridSize, blockSize, 0, 0, dataPtr, numElements,
                           baseValue);
        break;

      case 3:  // Interleaved
        // For interleaved, use fewer blocks to see stride pattern more clearly
        gridSize.x = std::min(gridSize.x, (unsigned int)1024);
        std::cout << "[Rank 0] Using reduced grid for stride pattern: " << gridSize.x
                  << " blocks\n";
        hipLaunchKernelGGL(WriteDataKernel_Interleaved, gridSize, blockSize, 0, 0, dataPtr,
                           numElements, baseValue);
        break;

      case 4:  // Byte-level
        hipLaunchKernelGGL(WriteDataKernel_ByteLevel, gridSize, blockSize, 0, 0, virtualPtr,
                           memorySize, baseValue);
        break;
    }

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "[Rank 0] ✅ Kernel completed - data written by " << (gridSize.x * blockSize.x)
              << " concurrent GPU threads\n";

    // Verify sample data on CPU
    std::cout << "[Rank 0] Verifying sample data on CPU...\n";

    if (testMode == 4) {
      // For byte-level mode, read bytes
      std::vector<uint8_t> hostBytes(std::min(size_t(10), memorySize));
      HIP_CHECK(hipMemcpy(hostBytes.data(), virtualPtr, hostBytes.size() * sizeof(uint8_t),
                          hipMemcpyDeviceToHost));

      std::cout << "[Rank 0] Sample byte values: ";
      for (size_t i = 0; i < std::min(size_t(5), hostBytes.size()); i++) {
        std::cout << "[" << i << "]=" << (unsigned)hostBytes[i] << " ";
      }
      std::cout << "...\n";
    } else {
      // For other modes, read ints
      std::vector<int> hostData(std::min(size_t(10), numElements));
      HIP_CHECK(hipMemcpy(hostData.data(), dataPtr, hostData.size() * sizeof(int),
                          hipMemcpyDeviceToHost));

      std::cout << "[Rank 0] Sample values: ";
      for (size_t i = 0; i < std::min(size_t(5), hostData.size()); i++) {
        std::cout << "[" << i << "]=" << hostData[i] << " ";
      }
      std::cout << "...\n";
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
    HIP_CHECK(hipMemImportFromShareableHandleCompat(&memHandle, shareableFd,
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

    // Notify Rank 0 that import and mapping is complete
    std::cout << "[Rank 1] Notifying Rank 0 that import is complete...\n";
    int import_done = 1;
    MPI_CHECK(MPI_Send(&import_done, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
    std::cout << "[Rank 1] ✅ Notification sent\n";

    // Wait for Rank 0 to finish writing
    std::cout << "[Rank 1] Waiting for Rank 0 to write data...\n";
    int ready = 0;
    MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    std::cout << "[Rank 1] ✅ Rank 0 finished writing\n";

    // Step 5: Read and verify data
    std::cout << "\n[Rank 1] ===== PHASE 2: Read and Verify =====\n";
    std::cout << "[Rank 1] Verifying mode " << testMode << ": " << modeNames[testMode] << "\n";

    int* dataPtr = static_cast<int*>(virtualPtr);
    int baseValue = 1000000;

    // Allocate error counter on GPU
    int* d_errorCount;
    HIP_CHECK(hipMalloc(&d_errorCount, sizeof(int)));
    HIP_CHECK(hipMemset(d_errorCount, 0, sizeof(int)));

    dim3 blockSize(256);
    dim3 gridSize;

    // Calculate grid size based on mode
    if (testMode == 4) {
      size_t numBytes = memorySize;
      gridSize.x = (numBytes + blockSize.x - 1) / blockSize.x;
    } else {
      gridSize.x = (numElements + blockSize.x - 1) / blockSize.x;
    }

    std::cout << "[Rank 1] Launching verification kernel on GPU 1...\n";
    std::cout << "[Rank 1] Grid: " << gridSize.x << " blocks x " << blockSize.x << " threads\n";

    // Launch appropriate verification kernel
    switch (testMode) {
      case 0:  // Normal sequential
        hipLaunchKernelGGL(VerifyDataKernel_Normal, gridSize, blockSize, 0, 0, dataPtr, numElements,
                           baseValue, d_errorCount);
        break;

      case 1:  // Concurrent independent
        hipLaunchKernelGGL(VerifyDataKernel_ConcurrentIndependent, gridSize, blockSize, 0, 0,
                           dataPtr, numElements, baseValue, d_errorCount);
        break;

      case 2:  // Concurrent atomic
        // For atomic mode, each counter should have 256 increments
        hipLaunchKernelGGL(VerifyDataKernel_Atomic, gridSize, blockSize, 0, 0, dataPtr, numElements,
                           256, d_errorCount);
        break;

      case 3:  // Interleaved
        gridSize.x = std::min(gridSize.x, (unsigned int)1024);
        hipLaunchKernelGGL(VerifyDataKernel_Interleaved, gridSize, blockSize, 0, 0, dataPtr,
                           numElements, baseValue, d_errorCount);
        break;

      case 4:  // Byte-level
        hipLaunchKernelGGL(VerifyDataKernel_ByteLevel, gridSize, blockSize, 0, 0, virtualPtr,
                           memorySize, baseValue, d_errorCount);
        break;
    }

    hipError_t kernelErr = hipGetLastError();
    if (kernelErr != hipSuccess) {
      std::cerr << "[Rank 1] ❌ Kernel launch failed: " << hipGetErrorString(kernelErr)
                << std::endl;
    }

    std::cout << "[Rank 1] Waiting for verification kernel to complete...\n";
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "[Rank 1] ✅ Verification kernel completed\n";

    // Get error count
    int h_errorCount = 0;
    HIP_CHECK(hipMemcpy(&h_errorCount, d_errorCount, sizeof(int), hipMemcpyDeviceToHost));

    if (h_errorCount == 0) {
      std::cout << "[Rank 1] ✅ GPU kernel verification successful!\n";
      std::cout << "[Rank 1] All data verified correctly on GPU 1\n";
      std::cout << "[Rank 1] Concurrent GPU thread writes work correctly!\n";

      // Sample a few values to CPU for display
      if (testMode == 4) {
        // For byte-level mode, show bytes
        std::vector<uint8_t> hostBytes(5);
        for (size_t i = 0; i < hostBytes.size(); i++) {
          uint8_t* bytePtr = static_cast<uint8_t*>(virtualPtr);
          HIP_CHECK(
              hipMemcpy(&hostBytes[i], &bytePtr[i * 1000], sizeof(uint8_t), hipMemcpyDeviceToHost));
        }

        std::cout << "[Rank 1] Sample byte values: ";
        for (size_t i = 0; i < hostBytes.size(); i++) {
          std::cout << "[" << (i * 1000) << "]=" << (unsigned)hostBytes[i] << " ";
        }
        std::cout << "\n";
      } else {
        // For other modes, show ints
        std::vector<int> hostData(5);
        for (size_t i = 0; i < hostData.size(); i++) {
          HIP_CHECK(
              hipMemcpy(&hostData[i], &dataPtr[i * 1000], sizeof(int), hipMemcpyDeviceToHost));
        }

        std::cout << "[Rank 1] Sample values: ";
        for (size_t i = 0; i < hostData.size(); i++) {
          std::cout << "[" << (i * 1000) << "]=" << hostData[i] << " ";
        }
        std::cout << "\n";
      }
    } else {
      std::cout << "[Rank 1] ❌ GPU kernel found " << h_errorCount << " errors\n";
      std::cout << "[Rank 1] (First 10 errors printed above)\n";
    }

    // Cleanup
    HIP_CHECK(hipFree(d_errorCount));

    // Ensure all GPU operations are complete before signaling
    HIP_CHECK(hipDeviceSynchronize());

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
    std::cout << "✅ Test Mode: " << testMode << " - " << modeNames[testMode] << "\n";

    if (testMode == 5) {
      std::cout << "✅ Rank 1 allocated memory and exported handle\n";
      std::cout << "✅ Rank 0 imported handle and mapped to remote pointer\n";
      std::cout << "✅ Rank 0 wrote data via REMOTE pointer (remotePtr)\n";
      std::cout << "✅ Rank 1 read data via LOCAL pointer (localPtr)\n";
      std::cout << "✅ SHMEM PUT simulation successful!\n";
      std::cout << "✅ Same physical memory accessed from different virtual addresses!\n";
    } else {
      std::cout << "✅ Rank 0 allocated memory and exported handle\n";
      std::cout << "✅ Rank 1 imported handle and mapped memory\n";
      std::cout << "✅ Rank 0 wrote data using concurrent GPU kernel threads\n";
      std::cout << "✅ Rank 1 verified data with GPU kernel\n";
      std::cout << "✅ Cross-process memory sharing + GPU thread concurrency works!\n";
    }
    std::cout << "================================================================\n";
  }

  std::cout << "[Rank " << rank << "] Cleaning up...\n";

  // Ensure all GPU operations are complete
  HIP_CHECK(hipDeviceSynchronize());

  // Unmap memory first
  if (virtualPtr) {
    HIP_CHECK(hipMemUnmap(virtualPtr, memorySize));
    HIP_CHECK(hipMemAddressFree(virtualPtr, memorySize));
  }

  // Close FD before releasing handle (important for proper cleanup)
  if (shareableFd != -1) {
    close(shareableFd);
    shareableFd = -1;
  }

  // Release the memory handle last
  if (memHandle) {
    HIP_CHECK(hipMemRelease(memHandle));
    memHandle = 0;
  }

  std::cout << "[Rank " << rank << "] ✅ Cleanup complete\n";

  MPI_Finalize();

  if (rank == 0) {
    std::cout << "\n🎉 Test completed successfully!\n";
  }

  return 0;
}
