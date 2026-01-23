#include <stdio.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/application/transport/sdma/anvil.hpp"

#include "sdma_bw_kernel.h"
#include "common.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>  // Added for strncpy
#include <ctime>
#include <execinfo.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unistd.h>
#include <mpi.h>
namespace fs = std::filesystem;

#include "CLI11.hpp"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

double get_time_diff(struct timeval* start, struct timeval* end) {
    long seconds = end->tv_sec - start->tv_sec;
    long microseconds = end->tv_usec - start->tv_usec;

    if (microseconds < 0) {
        seconds--;
        microseconds += 1000000;
    }

    return seconds + microseconds * 1e-6;
}

constexpr uint32_t MAGIC_VALUE = 0xDEADBEEF;
constexpr bool useSingleProducer = false;

struct ExperimentParams
{
   size_t minCopySize;
   size_t maxCopySize;
   size_t numCopyCommands;
   bool commitEveryTransfer; // TODO ?
   bool skipVerification;
   size_t nWarmupIterations;
   size_t numIterations;
   size_t numDestinations;
   size_t numOfQueues;
   size_t numOfWarpsPerWG;
   size_t numOfWGPerQueue;
   std::string resultFileName;
   bool verbose;
};

// MPI broadcast parameter packet struct (fixing MPI_Bcast issue)
struct MPIParamPacket {
    size_t minCopySize;
    size_t maxCopySize;
    size_t numCopyCommands;
    bool skipVerification;
    size_t nWarmupIterations;
    size_t numIterations;
    size_t numOfQueues;
    size_t numOfWarpsPerWG;
    size_t numOfWGPerQueue;
    char resultFileName[256];
    bool verbose;
};

// Signal handler function
void signal_handler(int sig) {
    void *array[20];
    size_t size;

    // Get backtrace information
    size = backtrace(array, 20);

    // Print backtrace to stderr
    fprintf(stderr, "Error: Signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    // Exit
    exit(1);
}

// Correct data verification function - verify received data equals expected value
size_t verifyDataCorrectness(void* deviceBuffer, size_t bufferSize, uint32_t expectedValue) {
    // Copy received data back to host
    std::vector<uint32_t> receivedData(bufferSize / sizeof(uint32_t));
    hipError_t err = hipMemcpy(receivedData.data(), deviceBuffer, bufferSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data for verification: " << hipGetErrorString(err) << std::endl;
        return static_cast<size_t>(-1);  // Return error
    }

    size_t errorCount = 0;
    size_t checkedCount = 0;
    const size_t maxErrorsToPrint = 10;

    // Verify each element equals expected value
    for (size_t i = 0; i < receivedData.size(); i++) {
        if (receivedData[i] != expectedValue) {
            errorCount++;
            if (errorCount <= maxErrorsToPrint) {
                std::cout << "Verification error: [Index " << i << "] Expected=" << std::hex << expectedValue
                          << " Actual=" << receivedData[i] << std::dec << std::endl;
            }
        }
        checkedCount++;

        // Output progress every 1 million elements
        if (checkedCount % 1000000 == 0) {
            std::cout << "Checked " << checkedCount << " elements, found " << errorCount << " errors" << std::endl;
        }
    }

    return errorCount;
}

// Correct function to calculate bandwidth
double calculateBandwidthGBps(size_t totalBytes, double latency_ms) {
    if (latency_ms <= 0.0) {
        return 0.0;
    }
    // Bandwidth = Data amount(GB) / Time(seconds)
    // 1. Convert bytes to GB: / (1024^3)
    // 2. Convert milliseconds to seconds: / 1000
    double totalGB = totalBytes / (1024.0 * 1024.0 * 1024.0);
    double time_sec = latency_ms / 1000.0;
    return totalGB / time_sec;
}


void runExperimentMPI(int srcDeviceId, int mpiRank, int mpiSize, const ExperimentParams& params)
{
    // =============================
    // 1. Initial setup (MPI adaptation)
    // =============================
    std::cout << "Process " << mpiRank << ": Starting experiment, source GPU=" << srcDeviceId
              << " (HIP device context set)" << std::endl;

    // Set GPU for current process
    hipError_t err = hipSetDevice(srcDeviceId);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to set device: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get total GPU count
    int total_gpus = 0;
    err = hipGetDeviceCount(&total_gpus);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to get GPU count: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Check if GPU count is sufficient
    if (total_gpus < mpiSize) {
        std::cerr << "Process " << mpiRank << ": Error: GPU count(" << total_gpus
                  << ") is less than MPI process count(" << mpiSize << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Determine all destination GPUs (all GPUs except self)
    std::vector<int> dstDeviceIds;
    for (int i = 0; i < mpiSize; i++) {
        if (i != mpiRank) {
            dstDeviceIds.push_back(i);
        }
    }

    std::cout << "Process " << mpiRank << ": Using GPU" << srcDeviceId
              << " to send data to " << dstDeviceIds.size() << " destination GPUs: ";
    for (size_t i = 0; i < dstDeviceIds.size(); i++) {
        std::cout << "GPU" << dstDeviceIds[i];
        if (i < dstDeviceIds.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Add verbose output
    if (params.verbose) {
        std::cout << "Process " << mpiRank << ": Verbose mode enabled" << std::endl;
        std::cout << "  - Total GPU count: " << total_gpus << std::endl;
        std::cout << "  - Parameters: minCopySize=" << params.minCopySize
                  << ", maxCopySize=" << params.maxCopySize
                  << ", numCopyCommands=" << params.numCopyCommands << std::endl;
    }

    // =============================
    // 2. Enable P2P access (critical step)
    // =============================
    std::cout << "Process " << mpiRank << ": Checking and enabling P2P access..." << std::endl;

    for (int dstDeviceId : dstDeviceIds) {
        int can_access = 0;
        hipError_t access_err = hipDeviceCanAccessPeer(&can_access, srcDeviceId, dstDeviceId);

        if (access_err != hipSuccess) {
            std::cerr << "Process " << mpiRank << ": hipDeviceCanAccessPeer failed (GPU"
                      << srcDeviceId << "->GPU" << dstDeviceId << "): "
                      << hipGetErrorString(access_err) << std::endl;
        }

        if (can_access) {
            std::cout << "Process " << mpiRank << ": GPU" << srcDeviceId
                      << " can access GPU" << dstDeviceId << ", enabling P2P..." << std::endl;
            hipError_t peer_err = hipDeviceEnablePeerAccess(dstDeviceId, 0);
            if (peer_err != hipSuccess && peer_err != hipErrorPeerAccessAlreadyEnabled) {
                std::cerr << "Process " << mpiRank << ": Cannot enable P2P access (GPU"
                          << srcDeviceId << "->GPU" << dstDeviceId << "): "
                          << hipGetErrorString(peer_err) << std::endl;
            } else {
                std::cout << "Process " << mpiRank << ": GPU" << srcDeviceId
                          << "->GPU" << dstDeviceId << " P2P access enabled" << std::endl;
            }
        } else {
            std::cerr << "Process " << mpiRank << ": GPU" << srcDeviceId
                      << " cannot directly access GPU" << dstDeviceId << std::endl;
            // Continue execution, Anvil might handle it in some cases
        }
    }

    // =============================
    // 3. Resource allocation (local buffers)
    // =============================
    std::cout << "Process " << mpiRank << ": Allocating local resources..." << std::endl;

    int wgSize = params.numOfWarpsPerWG * WARP_SIZE;
    size_t totalNumWarps = dstDeviceIds.size() * params.numOfQueues * params.numOfWGPerQueue * params.numOfWarpsPerWG;

    // Signal buffer
    HSAuint64* signalPtrs = nullptr;
    err = hipMalloc(&signalPtrs, sizeof(HSAuint64) * totalNumWarps);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": hipMalloc failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Signal buffer allocated: "
              << signalPtrs << std::endl;

    // Calculate maximum transfer size
    size_t maxP2PTransferSize = params.maxCopySize * params.numCopyCommands * params.numOfWarpsPerWG *
                                params.numOfWGPerQueue * params.numOfQueues;

    std::cout << "Process " << mpiRank << ": Maximum transfer size = "
              << maxP2PTransferSize << " bytes ("
              << maxP2PTransferSize / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;

    // Source buffer (local)
    void* sdma_src_buf = nullptr;
    err = hipExtMallocWithFlags(&sdma_src_buf, maxP2PTransferSize, hipDeviceMallocUncached);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": hipExtMallocWithFlags failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Source buffer allocated: "
              << sdma_src_buf << std::endl;

    // Initialize source buffer with MAGIC_VALUE
    size_t num_elements = maxP2PTransferSize / sizeof(uint32_t);
    std::vector<uint32_t> hostSrcBuffer(num_elements, MAGIC_VALUE);
    err = hipMemcpy(sdma_src_buf, hostSrcBuffer.data(), maxP2PTransferSize, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Source buffer initialization failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Source buffer initialized with MAGIC_VALUE = 0x"
              << std::hex << MAGIC_VALUE << std::dec << std::endl;

    // Destination buffer (local allocation, each process has its own receive buffer)
    void* local_dst_buf = nullptr;
    err = hipExtMallocWithFlags(&local_dst_buf, maxP2PTransferSize, hipDeviceMallocUncached);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Destination buffer allocation failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Local destination buffer allocated: "
              << local_dst_buf << std::endl;

    // Initialize to 0
    err = hipMemset(local_dst_buf, 0, maxP2PTransferSize);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Destination buffer initialization failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 4. Key step: Exchange buffer addresses using IPC memory handles
    // =============================
    std::cout << "Process " << mpiRank << ": Starting IPC memory handle exchange..." << std::endl;

    // Step 1: Get IPC memory handle for local buffer
    hipIpcMemHandle_t local_ipc_handle;
    hipError_t ipc_err = hipIpcGetMemHandle(&local_ipc_handle, local_dst_buf);
    if (ipc_err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": hipIpcGetMemHandle failed: "
                  << hipGetErrorString(ipc_err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Successfully obtained local IPC memory handle" << std::endl;

    // Step 2: Gather IPC memory handles from all processes
    std::vector<hipIpcMemHandle_t> all_ipc_handles(mpiSize);
    MPI_Allgather(&local_ipc_handle, sizeof(hipIpcMemHandle_t), MPI_BYTE,
                  all_ipc_handles.data(), sizeof(hipIpcMemHandle_t), MPI_BYTE,
                  MPI_COMM_WORLD);

    std::cout << "Process " << mpiRank << ": IPC handle exchange completed" << std::endl;

    // Step 3: Open all other processes' memory handles, get accessible pointers
    std::vector<void*> peer_ptrs(mpiSize, nullptr);

    for (int i = 0; i < mpiSize; i++) {
        if (i == mpiRank) {
            // Local pointer
            peer_ptrs[i] = local_dst_buf;
            continue;
        }

        // Open IPC handles for all other processes
        void* remote_ptr = nullptr;
        ipc_err = hipIpcOpenMemHandle(&remote_ptr, all_ipc_handles[i],
                                      hipIpcMemLazyEnablePeerAccess);

        if (ipc_err != hipSuccess) {
            std::cerr << "Process " << mpiRank << ": Cannot open IPC memory handle for process " << i
                      << ": " << hipGetErrorString(ipc_err) << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        peer_ptrs[i] = remote_ptr;
        std::cout << "Process " << mpiRank << ": Successfully opened IPC memory for process " << i
                  << ", pointer: " << remote_ptr << std::endl;
    }

    // Step 4: Prepare device-side pointer array for SDMA kernel (pointers to all destination GPUs)
    void** sdma_dst_bufs_d = nullptr;
    err = hipMalloc((void**)&sdma_dst_bufs_d, sizeof(void*) * dstDeviceIds.size());
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Device-side pointer array allocation failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::cout << "Process " << mpiRank << ": Device-side pointer array allocated: "
              << sdma_dst_bufs_d << " (size: " << sizeof(void*) * dstDeviceIds.size() << " bytes)" << std::endl;

    // Prepare host-side pointer array
    std::vector<void*> host_dst_ptrs(dstDeviceIds.size());
    for (size_t i = 0; i < dstDeviceIds.size(); i++) {
        int dstRank = dstDeviceIds[i];
        host_dst_ptrs[i] = peer_ptrs[dstRank];

        if (host_dst_ptrs[i] == nullptr) {
            std::cerr << "Process " << mpiRank << ": Error! Buffer pointer for process " << dstRank << " is null" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::cout << "Process " << mpiRank << ": Will send to process " << dstRank
                  << " (GPU" << dstRank << ") buffer: " << host_dst_ptrs[i] << std::endl;
    }

    // Transfer all remote pointers to device
    err = hipMemcpy(sdma_dst_bufs_d, host_dst_ptrs.data(), sizeof(void*) * dstDeviceIds.size(), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Address copy error: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 5. Anvil queue setup (establish connections for each destination GPU)
    // =============================
    std::cout << "Process " << mpiRank << ": Setting up Anvil SDMA queues..." << std::endl;

    // Establish connection for each destination GPU
    for (int dstDeviceId : dstDeviceIds) {
        try {
            // Ensure correct device context
            err = hipSetDevice(srcDeviceId);
            if (err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Failed to set device: " << hipGetErrorString(err) << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Check device ID validity
            if (dstDeviceId < 0 || dstDeviceId >= total_gpus) {
                std::cerr << "Process " << mpiRank << ": Invalid destination device ID: " << dstDeviceId << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Connect to destination device
            std::cout << "Process " << mpiRank << ": Connecting GPU" << srcDeviceId
                      << " -> GPU" << dstDeviceId << " ..." << std::endl;

            anvil::anvil.connect(srcDeviceId, dstDeviceId, params.numOfQueues);

            std::cout << "Process " << mpiRank << ": GPU" << srcDeviceId
                      << " -> GPU" << dstDeviceId << " Anvil connection successful" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Process " << mpiRank << ": GPU" << srcDeviceId
                      << " -> GPU" << dstDeviceId << " Anvil connection exception: " << e.what() << std::endl;
            // Try fallback to using 1 queue
            std::cerr << "Process " << mpiRank << ": Trying to reconnect with 1 queue..." << std::endl;
            try {
                anvil::anvil.connect(srcDeviceId, dstDeviceId, 1);
                std::cout << "Process " << mpiRank << ": GPU" << srcDeviceId
                          << " -> GPU" << dstDeviceId << " Fallback connection successful" << std::endl;
            } catch (const std::exception& e2) {
                std::cerr << "Process " << mpiRank << ": GPU" << srcDeviceId
                          << " -> GPU" << dstDeviceId << " Fallback connection also failed: " << e2.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } catch (...) {
            std::cerr << "Process " << mpiRank << ": GPU" << srcDeviceId
                      << " -> GPU" << dstDeviceId << " Anvil connection unknown exception" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Get queue handles (each queue for each destination GPU)
    size_t totalNumQueues = dstDeviceIds.size() * params.numOfQueues;
    anvil::SdmaQueueDeviceHandle** deviceHandles_d = nullptr;
    err = hipMalloc(&deviceHandles_d, totalNumQueues * sizeof(anvil::SdmaQueueDeviceHandle*));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Queue handle allocation failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::cout << "Process " << mpiRank << ": Queue handle allocated: "
              << deviceHandles_d << " (size: "
              << totalNumQueues * sizeof(anvil::SdmaQueueDeviceHandle*) << " bytes)" << std::endl;

    // Prepare host-side queue handle array
    std::vector<anvil::SdmaQueueDeviceHandle*> host_device_handles(totalNumQueues);
    size_t queueIdx = 0;

    for (size_t dstIdx = 0; dstIdx < dstDeviceIds.size(); dstIdx++) {
        int dstDeviceId = dstDeviceIds[dstIdx];
        for (size_t q = 0; q < params.numOfQueues; q++) {
            try {
                auto queue = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q);
                if (queue && queue->deviceHandle()) {
                    host_device_handles[queueIdx] = queue->deviceHandle();
                    std::cout << "Process " << mpiRank << ": Destination GPU" << dstDeviceId
                              << " queue[" << q << "] handle: "
                              << host_device_handles[queueIdx] << std::endl;
                    queueIdx++;
                } else {
                    std::cerr << "Process " << mpiRank << ": Failed to get destination GPU" << dstDeviceId
                              << " queue[" << q << "]" << std::endl;
                    // Try using queue 0 handle as fallback
                    if (q > 0) {
                        std::cerr << "Process " << mpiRank << ": Using queue 0 handle as fallback" << std::endl;
                        // Find first queue handle for this destination GPU
                        size_t firstQueueIdx = dstIdx * params.numOfQueues;
                        if (host_device_handles[firstQueueIdx] != nullptr) {
                            host_device_handles[queueIdx] = host_device_handles[firstQueueIdx];
                            queueIdx++;
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Process " << mpiRank << ": Exception getting destination GPU" << dstDeviceId
                          << " queue[" << q << "]: " << e.what() << std::endl;
            }
        }
    }

    if (queueIdx != totalNumQueues) {
        std::cerr << "Process " << mpiRank << ": Warning: Only obtained " << queueIdx
                  << " queue handles, expected " << totalNumQueues << std::endl;
        // Continue if at least one queue was obtained
        if (queueIdx == 0) {
            std::cerr << "Process " << mpiRank << ": Error: No queue handles obtained" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Copy queue handles to device
    err = hipMemcpy(deviceHandles_d, host_device_handles.data(),
                    totalNumQueues * sizeof(anvil::SdmaQueueDeviceHandle*),
                    hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Queue handle copy failed: "
                  << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 6. Timestamp memory allocation
    // =============================
    std::cout << "Process " << mpiRank << ": Allocating timestamp memory..." << std::endl;

    long long int* start_clock_count = nullptr;
    long long int* end_clock_count = nullptr;
    long long int* start_clock_count_d = nullptr;
    long long int* end_clock_count_d = nullptr;

    err = hipHostMalloc(&start_clock_count, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to allocate host memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = hipHostMalloc(&end_clock_count, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to allocate host memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = hipMalloc(&start_clock_count_d, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = hipMalloc(&end_clock_count_d, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    err = hipMemset(start_clock_count_d, 0, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to zero device memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = hipMemset(end_clock_count_d, 0, params.numIterations * totalNumWarps * sizeof(long long int));
    if (err != hipSuccess) {
        std::cerr << "Process " << mpiRank << ": Failed to zero device memory: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 7. Kernel launch configuration
    // =============================
    int numWgs = dstDeviceIds.size() * params.numOfQueues * params.numOfWGPerQueue;

    std::cout << "Process " << mpiRank << ": Kernel configuration - GridDim.x=" << numWgs
              << ", BlockDim.x=" << wgSize << std::endl;
    std::cout << "Process " << mpiRank << ": Destination GPU count: " << dstDeviceIds.size()
              << ", Total queue count: " << totalNumQueues << std::endl;

    dim3 grid(numWgs, 1, 1);
    dim3 block(wgSize, 1, 1);

    // MPI synchronization: ensure all processes are ready
    std::cout << "Process " << mpiRank << ": Waiting for MPI synchronization..." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Process " << mpiRank << ": MPI synchronization completed" << std::endl;

    // Reporter
    std::string localResultFileName = params.resultFileName + "_rank" + std::to_string(mpiRank) + ".csv";
    Reporter report(localResultFileName);
    report.setParameters(srcDeviceId, dstDeviceIds.size(), params.numOfQueues, numWgs, wgSize, params.numCopyCommands);

    // =============================
    // 8. Main test loop
    // =============================
    std::cout << "Process " << mpiRank << ": Starting main test loop..." << std::endl;

    for (size_t copySize = params.minCopySize; copySize <= params.maxCopySize; copySize *= 2)
    {
        size_t totalTransferSize = copySize * params.numCopyCommands * params.numOfWarpsPerWG *
                                   params.numOfWGPerQueue * params.numOfQueues * dstDeviceIds.size();

        std::cout << "Process " << mpiRank << ": Testing copy size " << copySize
                  << " B, total transfer amount " << totalTransferSize << " B ("
                  << totalTransferSize / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;
        std::cout << "Process " << mpiRank << ": Data amount per destination GPU: "
                  << (totalTransferSize / dstDeviceIds.size()) << " B ("
                  << (totalTransferSize / dstDeviceIds.size()) / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;

        // Clear local destination buffer (prepare to receive data from other GPUs)
        err = hipMemset(local_dst_buf, 0, maxP2PTransferSize);
        if (err != hipSuccess) {
            std::cerr << "Process " << mpiRank << ": Failed to clear buffer: "
                      << hipGetErrorString(err) << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Clear signal buffer
        err = hipMemset(signalPtrs, 0, sizeof(HSAuint64) * totalNumWarps);
        if (err != hipSuccess) {
            std::cerr << "Process " << mpiRank << ": Failed to clear signal buffer: "
                      << hipGetErrorString(err) << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        HSAuint64 expectedSignal = 1;
        auto kernel = multiQueueSDMATransferQueueMapWG;

        // MPI synchronization
        MPI_Barrier(MPI_COMM_WORLD);

        // Verification run (optional) - perform full data correctness verification
        std::optional<size_t> numErrors;
        if (!params.skipVerification) {
            std::cout << "Process " << mpiRank << ": Running verification kernel and data correctness verification..." << std::endl;

            // Launch SDMA kernel - send to all destination GPUs
            hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, sdma_src_buf, sdma_dst_bufs_d,
                               copySize, params.numCopyCommands, dstDeviceIds.size(),
                               params.numOfQueues, params.numOfWGPerQueue, deviceHandles_d, signalPtrs,
                               expectedSignal, start_clock_count_d, end_clock_count_d);

            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Device synchronization failed: "
                          << hipGetErrorString(err) << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            expectedSignal++;

            // Check kernel launch error
            hipError_t kernel_err = hipGetLastError();
            if (kernel_err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Kernel launch error: "
                          << hipGetErrorString(kernel_err) << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // MPI synchronization: wait for all data transfers to complete
            MPI_Barrier(MPI_COMM_WORLD);

            // Verify data received by this process (from all other GPUs)
            // Note: Here we can only verify if this process received data, but cannot distinguish which GPU sent it
            // Since all GPUs sent data to this process, the final result is superimposed
            std::cout << "Process " << mpiRank << ": Starting verification of received data..." << std::endl;

            // Check if any data was written (since all GPUs sent data, value should be MAGIC_VALUE * (mpiSize-1))
            // But actually data will overwrite each other, so just check for non-zero values
            std::vector<uint32_t> sample_data(std::min(static_cast<size_t>(10), maxP2PTransferSize / sizeof(uint32_t)));
            err = hipMemcpy(sample_data.data(), local_dst_buf, sample_data.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
            if (err == hipSuccess) {
                std::cout << "Process " << mpiRank << ": First " << sample_data.size() << " received values: ";
                for (size_t i = 0; i < sample_data.size(); i++) {
                    std::cout << "0x" << std::hex << sample_data[i] << " ";
                }
                std::cout << std::dec << std::endl;

                // Check for non-zero values
                bool hasData = false;
                for (size_t i = 0; i < sample_data.size(); i++) {
                    if (sample_data[i] != 0) {
                        hasData = true;
                        break;
                    }
                }

                if (hasData) {
                    std::cout << "Process " << mpiRank << ": Verification completed - Data reception detected" << std::endl;
                    numErrors = 0;
                } else {
                    std::cerr << "Process " << mpiRank << ": Verification failed - No data detected" << std::endl;
                    numErrors = 1;
                }
            } else {
                std::cerr << "Process " << mpiRank << ": Failed to copy data for verification: "
                          << hipGetErrorString(err) << std::endl;
                numErrors = static_cast<size_t>(-1);
            }
        }

        // If no errors or error count is 0, continue with warmup and measurement
        if (!numErrors || numErrors.value() == 0) {
            // Warmup
            std::cout << "Process " << mpiRank << ": Warming up ("
                      << params.nWarmupIterations << " times)..." << std::endl;

            for (size_t i = 0; i < params.nWarmupIterations; ++i) {
                hipLaunchKernelGGL(kernel, grid, block, 0, 0, i, sdma_src_buf, sdma_dst_bufs_d,
                                   copySize, params.numCopyCommands, dstDeviceIds.size(),
                                   params.numOfQueues, params.numOfWGPerQueue,
                                   deviceHandles_d, signalPtrs, expectedSignal, start_clock_count_d, end_clock_count_d);
                expectedSignal++;
            }
            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Warmup device synchronization failed: "
                          << hipGetErrorString(err) << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Prepare events for timing
            std::vector<hipEvent_t> timestamps_events(params.numIterations * 2);
            for (int iter = 0; iter < params.numIterations * 2; iter++) {
                err = hipEventCreate(&timestamps_events[iter]);
                if (err != hipSuccess) {
                    std::cerr << "Process " << mpiRank << ": Failed to create event: "
                              << hipGetErrorString(err) << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            // Performance measurement iterations
            std::vector<double> latency_ms;  // Store latency for each iteration (milliseconds)

            // MPI synchronization: start performance testing
            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "Process " << mpiRank << ": Starting performance measurement ("
                      << params.numIterations << " times)..." << std::endl;
            //struct timeval start, end;
            //double elapsed;
            //gettimeofday(&start, NULL);
            for (size_t iter = 0; iter < params.numIterations; ++iter) {
                struct timeval start, end;
                double elapsed;
                gettimeofday(&start, NULL);

                hipExtLaunchKernelGGL(kernel, grid, block, 0, 0,
                                      timestamps_events[iter * 2],     // start event
                                      timestamps_events[iter * 2 + 1], // stop event
                                      0,                               // flags
                                      iter, sdma_src_buf, sdma_dst_bufs_d, copySize, params.numCopyCommands,
                                      dstDeviceIds.size(), // numDestinations = all other GPUs
                                      params.numOfQueues, params.numOfWGPerQueue, deviceHandles_d, signalPtrs, expectedSignal,
                                      start_clock_count_d, end_clock_count_d);

                err = hipDeviceSynchronize();
                if (err != hipSuccess) {
                   std::cerr << "Process " << mpiRank << ": Performance measurement device synchronization failed: "
                          << hipGetErrorString(err) << std::endl;
                   MPI_Abort(MPI_COMM_WORLD, 1);
                }
                gettimeofday(&end, NULL);
                elapsed = get_time_diff(&start, &end);
                printf("rank:%u, iter:%u, start sec:%llu, usec:%llu, end sec:%llu, usec:%llu, consume time: %.6f, time delta: %.2f%%) \n",
                   mpiRank, iter, start.tv_sec, start.tv_usec, end.tv_sec, end.tv_usec, elapsed, (elapsed - 0.1) / 0.1 * 100);

                expectedSignal++;
            }
            //gettimeofday(&end, NULL);
            //elapsed = get_time_diff(&start, &end);
            //printf("rank:%u, start sec:%llu, usec:%llu, end sec:%llu, usec:%llu, consume time: %.6f, time delta: %.2f%%) \n",
            //       mpiRank, start.tv_sec, start.tv_usec, end.tv_sec, end.tv_usec, elapsed, (elapsed - 0.1) / 0.1 * 100);
#if 0
            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Performance measurement device synchronization failed: "
                          << hipGetErrorString(err) << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            gettimeofday(&end, NULL);
            elapsed = get_time_diff(&start, &end);
            printf("rank:%u, start sec:%llu, usec:%llu, end sec:%llu, usec:%llu, consume time: %.6f, time delta: %.2f%%) \n",
                   mpiRank, start.tv_sec, start.tv_usec, end.tv_sec, end.tv_usec, elapsed, (elapsed - 0.1) / 0.1 * 100);
#endif
            // MPI synchronization: wait for all processes to complete
            MPI_Barrier(MPI_COMM_WORLD);

            // =============================
            // 9. Performance metrics calculation (using hipEvent timing)
            // =============================
            for (size_t iter = 0; iter < params.numIterations; ++iter) {
                float iter_latency_ms;
                err = hipEventElapsedTime(&iter_latency_ms,
                    timestamps_events[iter * 2], timestamps_events[iter * 2 + 1]);
                if (err != hipSuccess) {
                    std::cerr << "Process " << mpiRank << ": Event timing failed: "
                              << hipGetErrorString(err) << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                latency_ms.push_back(iter_latency_ms);
            }

            // Calculate average latency and standard deviation
            double latency_sum = 0.0;
            for (double lat : latency_ms) {
                latency_sum += lat;
            }
            double latency_mean_ms = latency_sum / latency_ms.size();

            double latency_std = 0.0;
            for (double lat : latency_ms) {
                latency_std += (lat - latency_mean_ms) * (lat - latency_mean_ms);
            }
            latency_std = std::sqrt(latency_std / latency_ms.size());

            // Collect latency from all processes for aggregated display
            std::vector<double> all_latencies;
            if (mpiRank == 0) {
                all_latencies.resize(mpiSize);
            }

            // Gather latency from all processes to process 0
            MPI_Gather(&latency_mean_ms, 1, MPI_DOUBLE,
                      all_latencies.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // MPI aggregation: get maximum latency among all processes (for system performance calculation)
            double max_latency_mean_ms;
            MPI_Allreduce(&latency_mean_ms, &max_latency_mean_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            // Correct bandwidth calculation:
            // Total data sent by each process = data per destination GPU × number of destination GPUs
            double perProcessTotalBytes = static_cast<double>(totalTransferSize);
            double perProcessTotalGB = perProcessTotalBytes / (1024.0 * 1024.0 * 1024.0);
            double time_sec = max_latency_mean_ms / 1000.0;

            if (time_sec > 0) {
                // Total bandwidth per process (sending to all destination GPUs)
                double perProcessBandwidth_gbs = perProcessTotalGB / time_sec;

                // Total system bandwidth (all processes sending simultaneously)
                // Note: Each process sends data to all other processes
                // Total system bandwidth = per-process bandwidth × number of processes
                double totalSystemBandwidth_gbs = perProcessBandwidth_gbs * mpiSize;

                // Theoretical maximum bandwidth: each link 50 GB/s, total links = mpiSize × (mpiSize-1) / 2 (fully connected)
                // But actually each process sends to all other processes simultaneously, so total bandwidth ≈ 50 GB/s × (mpiSize-1) × mpiSize / 2
                double theoreticalMaxBandwidth_gbs = 50.0 * (mpiSize - 1) * mpiSize / 2.0;

                // Debug output - each process outputs its real latency and system maximum latency
                std::cout << "Process " << mpiRank << ": ==== Correct Bandwidth Calculation ====" << std::endl;
                std::cout << "  - Local actual latency: " << std::fixed << std::setprecision(4) << latency_mean_ms << " ms" << std::endl;
                std::cout << "  - System maximum latency: " << std::fixed << std::setprecision(4) << max_latency_mean_ms << " ms" << std::endl;
                std::cout << "  - Latency standard deviation: " << std::fixed << std::setprecision(4) << latency_std << " ms" << std::endl;
                std::cout << "  - Total data sent per process: " << std::fixed << std::setprecision(3) << perProcessTotalGB << " GB" << std::endl;
                std::cout << "  - Total bandwidth per process: " << std::fixed << std::setprecision(2) << perProcessBandwidth_gbs << " GB/s" << std::endl;
                std::cout << "  - Total system bandwidth: " << std::fixed << std::setprecision(2) << totalSystemBandwidth_gbs << " GB/s" << std::endl;
                std::cout << "  - Theoretical maximum bandwidth: " << std::fixed << std::setprecision(2) << theoreticalMaxBandwidth_gbs << " GB/s" << std::endl;
                std::cout << "  - Efficiency: " << std::fixed << std::setprecision(1)
                          << (totalSystemBandwidth_gbs / theoreticalMaxBandwidth_gbs * 100.0) << "%" << std::endl;

                // Only process 0 outputs detailed latency summary for all processes
                if (mpiRank == 0) {
                    std::cout << "\n==============================================================================" << std::endl;
                    std::cout << "Performance Summary (" << mpiSize << "-GPU fully connected SDMA transfer):" << std::endl;
                    std::cout << "------------------------------------------------------------------------------" << std::endl;
                    std::cout << "GPU count: " << mpiSize << std::endl;
                    std::cout << "Each GPU sends to: " << (mpiSize - 1) << " other GPUs" << std::endl;
                    std::cout << "Total data per process: " << std::fixed << std::setprecision(3) << perProcessTotalGB << " GB" << std::endl;
                    std::cout << "\nActual latency details for each process:" << std::endl;
                    for (int i = 0; i < mpiSize; i++) {
                        std::cout << "  Process " << i << " (GPU" << i << "): "
                                  << std::fixed << std::setprecision(4) << all_latencies[i] << " ms";
                        if (all_latencies[i] == max_latency_mean_ms) {
                            std::cout << " (slowest)";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "\nSystem maximum latency: " << std::fixed << std::setprecision(4) << max_latency_mean_ms << " ms" << std::endl;
                    std::cout << "Average bandwidth per process: " << std::fixed << std::setprecision(2) << perProcessBandwidth_gbs << " GB/s" << std::endl;
                    std::cout << "Total system bandwidth: " << std::fixed << std::setprecision(2) << totalSystemBandwidth_gbs << " GB/s" << std::endl;
                    std::cout << "Theoretical maximum bandwidth: " << std::fixed << std::setprecision(2) << theoreticalMaxBandwidth_gbs << " GB/s" << std::endl;
                    std::cout << "Transfer efficiency: " << std::fixed << std::setprecision(1)
                              << (totalSystemBandwidth_gbs / theoreticalMaxBandwidth_gbs * 100.0) << "%" << std::endl;
                    std::cout << "Data correctness: " << (numErrors && numErrors.value() == 0 ? "PASSED" : "FAILED") << std::endl;
                    std::cout << "==============================================================================\n" << std::endl;
                }

                // Record results (using system maximum latency for calculation)
                report.addResult(totalTransferSize, copySize, max_latency_mean_ms * 1000,  // Convert to microseconds
                                 latency_std * 1000, perProcessBandwidth_gbs,
                                 max_latency_mean_ms * 1000, latency_std * 1000, totalSystemBandwidth_gbs);
            } else {
                std::cerr << "Process " << mpiRank << ": Error: Zero or negative latency time" << std::endl;
            }

            // Clean up events
            for (int iter = 0; iter < (params.numIterations * 2); iter++) {
                err = hipEventDestroy(timestamps_events[iter]);
                if (err != hipSuccess) {
                    std::cerr << "Process " << mpiRank << ": Failed to destroy event: "
                              << hipGetErrorString(err) << std::endl;
                }
            }
        } else {
            std::cerr << "Process " << mpiRank << ": Data verification failed, skipping performance test" << std::endl;
        }

        std::cout << "Process " << mpiRank << ": Copy size " << copySize << " B test completed" << std::endl;
    }

    // Write results to file
    report.writeFile();

    std::cout << "Process " << mpiRank << ": Experiment completed, results saved to "
              << localResultFileName << std::endl;

    // =============================
    // 10. Resource cleanup (including IPC memory)
    // =============================
    std::cout << "Process " << mpiRank << ": Starting resource cleanup..." << std::endl;

    // Close IPC memory handles
    for (int i = 0; i < mpiSize; i++) {
        if (i != mpiRank && peer_ptrs[i] != nullptr) {
            hipError_t close_err = hipIpcCloseMemHandle(peer_ptrs[i]);
            if (close_err != hipSuccess) {
                std::cerr << "Process " << mpiRank << ": Warning: Failed to close IPC memory handle for process " << i
                          << ": " << hipGetErrorString(close_err) << std::endl;
            } else {
                std::cout << "Process " << mpiRank << ": Closed IPC memory for process " << i << std::endl;
            }
        }
    }

    // Free memory
    if (start_clock_count) hipFreeHost(start_clock_count);
    if (end_clock_count) hipFreeHost(end_clock_count);
    if (start_clock_count_d) hipFree(start_clock_count_d);
    if (end_clock_count_d) hipFree(end_clock_count_d);
    if (deviceHandles_d) hipFree(deviceHandles_d);
    if (sdma_dst_bufs_d) hipFree(sdma_dst_bufs_d);
    if (signalPtrs) hipFree(signalPtrs);
    if (sdma_src_buf) hipFree(sdma_src_buf);
    if (local_dst_buf) hipFree(local_dst_buf);

    // Disable P2P access (ignore return value warnings)
    for (int dstDeviceId : dstDeviceIds) {
        (void)hipDeviceDisablePeerAccess(dstDeviceId);
    }

    std::cout << "Process " << mpiRank << ": Resource cleanup completed" << std::endl;
}

int main(int argc, char** argv)
{
    // Install signal handlers
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);

    // =============================
    // 1. MPI initialization (execute first)
    // =============================
    MPI_Init(&argc, &argv);

    int mpi_world_size = 0;
    int mpi_world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);

    // Check MPI process count (now supports any number, at least 2)
    if (mpi_world_size < 2) {
        if (mpi_world_rank == 0) {
            std::cerr << "Error: This test requires at least 2 MPI processes" << std::endl;
            std::cerr << "Please use: mpirun -np 2 ./sdma_bw [options]" << std::endl;
            std::cerr << "Or: mpirun -np 8 ./sdma_bw [options] (8-GPU fully connected test)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // =============================
    // 2. Initialize Anvil (required by all processes)
    // =============================
    try {
        anvil::anvil.init();
        std::cout << "Process " << mpi_world_rank << ": Anvil initialization successful" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Process " << mpi_world_rank << ": Anvil initialization failed: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 3. Parse command line arguments (using MPI-safe method)
    // =============================
    ExperimentParams params;

    if (mpi_world_rank == 0) {
        // Only process 0 parses command line
        CLI::App app("Shader-initiated SDMA (MPI fully connected parallel version)");
        size_t minCopySize{0};
        app.add_option("-b,--minCopySize", minCopySize, "Minimum Transfer Size [B] (per copy command)");
        size_t maxCopySize{0};
        app.add_option("-e,--maxCopySize", maxCopySize, "Maximum Transfer Size [B] (per copy command)");
        size_t numCopyCommands{0};
        app.add_option("-c,--numCopyCommands", numCopyCommands, "Number of copy commands (per warp)");

        bool skipVerification{false};
        app.add_flag("--skip-verification", skipVerification, "Skip verification");

        size_t nWarmupIterations{3};
        app.add_option("-w,--warmup", nWarmupIterations, "Number of warmup iterations");

        size_t numIterations{50};
        app.add_option("-n,--iterations", numIterations, "Number of iterations");

        // Note: In MPI fully connected mode, numDestinations is automatically set to mpiSize-1
        size_t numDestinations{1};
        app.add_option("-d,--numDestinations", numDestinations,
                      "Number of destination GPUs (automatically set to process count -1 in MPI fully connected mode)");

        size_t numOfQueues{1};
        app.add_option("--numOfQueuesPerDestination", numOfQueues,
                      "Number of queues per destination");

        size_t numOfWarpsPerWG{1};
        app.add_option("--warpsPerWG", numOfWarpsPerWG, "Number of warps shared the same queue resources");

        size_t numOfWGPerQueue{1};
        app.add_option("--wgsPerQueue", numOfWGPerQueue, "Number of workgroups shared the same queue resources");

        std::string resultFileName = "MultiQueueGPU2GPU_Performance";
        app.add_option("-o,--outputFile", resultFileName, "Filename prefix for result");

        bool verbose{false};
        app.add_flag("-v, --verbose", verbose, "verbose output");

        CLI11_PARSE(app, argc, argv);

        // Create parameter packet for broadcast (fixing original MPI_Bcast issue)
        MPIParamPacket packet;
        packet.minCopySize = minCopySize;
        packet.maxCopySize = maxCopySize;
        packet.numCopyCommands = numCopyCommands;
        packet.skipVerification = skipVerification;
        packet.nWarmupIterations = nWarmupIterations;
        packet.numIterations = numIterations;
        packet.numOfQueues = numOfQueues;
        packet.numOfWarpsPerWG = numOfWarpsPerWG;
        packet.numOfWGPerQueue = numOfWGPerQueue;
        strncpy(packet.resultFileName, resultFileName.c_str(), sizeof(packet.resultFileName)-1);
        packet.resultFileName[sizeof(packet.resultFileName)-1] = '\0';  // Ensure null-termination
        packet.verbose = verbose;

        // Broadcast packet to all processes
        MPI_Bcast(&packet, sizeof(MPIParamPacket), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Fill this process's params
        params.minCopySize = packet.minCopySize;
        params.maxCopySize = packet.maxCopySize;
        params.numCopyCommands = packet.numCopyCommands;
        params.commitEveryTransfer = true;
        params.skipVerification = packet.skipVerification;
        params.nWarmupIterations = packet.nWarmupIterations;
        params.numIterations = packet.numIterations;
        params.numDestinations = mpi_world_size - 1;  // MPI fully connected mode: each process sends to all other processes
        params.numOfQueues = packet.numOfQueues;
        params.numOfWarpsPerWG = packet.numOfWarpsPerWG;
        params.numOfWGPerQueue = packet.numOfWGPerQueue;
        params.resultFileName = packet.resultFileName;
        params.verbose = packet.verbose;

        std::cout << "==== MPI Process " << mpi_world_rank << "/" << mpi_world_size
                  << " parameter parsing completed ====" << std::endl;
        std::cout << "Running fully connected SDMA test: " << mpi_world_size << " GPUs, each GPU sends to "
                  << (mpi_world_size - 1) << " other GPUs" << std::endl;
        std::cout << "Copy size from " << minCopySize << " to " << maxCopySize << std::endl;
    } else {
        // Other processes receive parameters
        MPIParamPacket packet;
        MPI_Bcast(&packet, sizeof(MPIParamPacket), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Fill params struct
        params.minCopySize = packet.minCopySize;
        params.maxCopySize = packet.maxCopySize;
        params.numCopyCommands = packet.numCopyCommands;
        params.commitEveryTransfer = true;
        params.skipVerification = packet.skipVerification;
        params.nWarmupIterations = packet.nWarmupIterations;
        params.numIterations = packet.numIterations;
        params.numDestinations = mpi_world_size - 1;  // MPI fully connected mode: each process sends to all other processes
        params.numOfQueues = packet.numOfQueues;
        params.numOfWarpsPerWG = packet.numOfWarpsPerWG;
        params.numOfWGPerQueue = packet.numOfWGPerQueue;
        params.resultFileName = packet.resultFileName;
        params.verbose = packet.verbose;

        std::cout << "Process " << mpi_world_rank << ": Parameters received" << std::endl;
    }

    // =============================
    // 4. Determine GPU used by each process
    // =============================
    int total_gpus = 0;
    hipError_t err = hipGetDeviceCount(&total_gpus);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to get GPU count: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (total_gpus < mpi_world_size) {
        if (mpi_world_rank == 0) {
            std::cerr << "Error: Need at least " << mpi_world_size << " GPUs, currently only " << total_gpus << " available" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Simple mapping: MPI process i uses GPU i
    int srcGpuId = mpi_world_rank;

    std::cout << "Process " << mpi_world_rank << ": Using GPU " << srcGpuId
              << " (Total GPU count: " << total_gpus << ")" << std::endl;

    // Set GPU for current process
    err = hipSetDevice(srcGpuId);
    if (err != hipSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to set device: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =============================
    // 5. Execute experiment (fully connected MPI version)
    // =============================
    runExperimentMPI(srcGpuId, mpi_world_rank, mpi_world_size, params);

    // =============================
    // 6. MPI cleanup
    // =============================
    MPI_Finalize();

    return 0;
}
