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

namespace fs = std::filesystem;

#include "CLI11.hpp"

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

// Create array of template instantiation of the kernel
// typedef void (*GpuKernelFuncPtr)(size_t, void*, void**, size_t, size_t, size_t, size_t, size_t,
//                                  anvil::SdmaQueueDeviceHandle(**, HSAuint64*, HSAuint64, long long int*, long long
//                                  int*);

// std::map<bool, GpuKernelFuncPtr> gpuKernels = {
//     {false, multiQueueSDMATransferQueueMapWG<false>},
//     {true, multiQueueSDMATransferQueueMapWG<true>},
// };


void runExperiment(int srcDeviceId, const ExperimentParams& params)
{
   // =============================
   // 1. Initial Setup
   // =============================
   int hipDeviceCount = 0;
   CHECK_HIP_ERROR(hipGetDeviceCount(&hipDeviceCount));

   std::cout << "Src GPU Device Id: " << srcDeviceId << std::endl;

   std::vector<int> dstDeviceIds;

   for (int i = 0; i < hipDeviceCount; ++i)
   {
      if (srcDeviceId != i)
      {
         dstDeviceIds.push_back(i);

         if (params.verbose)
         {
            std::cout << "Device Id: " << i << std::endl;
         }
         if (dstDeviceIds.size() == params.numDestinations)
         {
            break;
         }
      }
   }
   // Print destinations
   if (params.verbose)
   {
      printList(std::cout, dstDeviceIds, "Dest GPU Ids: ");
   }

   std::vector<void*> sdmaDestBufferPtr;

   CHECK_HIP_ERROR(hipSetDevice(srcDeviceId));

   //==============================
   // 2. Resource Allocation
   //==============================
   // All destinations
   int wgSize = params.numOfWarpsPerWG * warpSize;
   int nWarpsPerWG = params.numOfWarpsPerWG;
   size_t totalNumWarps = params.numDestinations * params.numOfQueues * params.numOfWGPerQueue * params.numOfWarpsPerWG;

   // Allocate signals, one for each warp
   HSAuint64* signalPtrs;
   CHECK_HIP_ERROR(hipMalloc(&signalPtrs, sizeof(HSAuint64) * totalNumWarps));

   // single P2P transfer size
   size_t maxP2PTransferSize = params.maxCopySize * params.numCopyCommands * params.numOfWarpsPerWG *
                               params.numOfWGPerQueue * params.numOfQueues;

   void* sdma_src_buf = nullptr;
   CHECK_HIP_ERROR(hipExtMallocWithFlags(&sdma_src_buf, maxP2PTransferSize, hipDeviceMallocUncached));

   // Fill src buf with MAGIC
   size_t num_elements = maxP2PTransferSize / sizeof(uint32_t);
   std::vector<uint32_t> hostSrcBuffer(num_elements, MAGIC_VALUE);
   CHECK_HIP_ERROR(hipMemcpy(sdma_src_buf, hostSrcBuffer.data(), maxP2PTransferSize, hipMemcpyHostToDevice));

   HsaMemFlags memFlags = {};
   memFlags.ui32.NonPaged = 1;
   memFlags.ui32.HostAccess = 1;
   memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
   memFlags.ui32.NoNUMABind = 1;
   memFlags.ui32.ExecuteAccess = 1;
   memFlags.ui32.Uncached = 1;

   for (size_t d = 0; d < dstDeviceIds.size(); d++)
   {

      int dstGPUId = dstDeviceIds[d];
      void* buf;
      CHECK_HIP_ERROR(hipSetDevice(dstGPUId));
      CHECK_HIP_ERROR(hipExtMallocWithFlags(&buf, maxP2PTransferSize, hipDeviceMallocUncached));
      EnablePeerAccess(srcDeviceId, dstGPUId);
      sdmaDestBufferPtr.push_back(buf);
   }

   CHECK_HIP_ERROR(hipSetDevice(srcDeviceId));

   void** sdma_dst_bufs_d;
   CHECK_HIP_ERROR(hipMalloc((void**)&sdma_dst_bufs_d, sdmaDestBufferPtr.size() * sizeof(void*)));
   CHECK_HIP_ERROR(hipMemcpy(sdma_dst_bufs_d, sdmaDestBufferPtr.data(), sdmaDestBufferPtr.size() * sizeof(void*),
                             hipMemcpyHostToDevice));

   // ======================
   // 3. Queue Setup
   // ======================

   size_t totalNumQueues = params.numOfQueues * params.numDestinations;

   for (auto& dstDeviceId : dstDeviceIds)
   {
      // Better performance if allocating all 8 queues
      anvil::anvil.connect(srcDeviceId, dstDeviceId, 8); // params.numOfQueues);
   }

   anvil::SdmaQueueDeviceHandle** deviceHandles_d = nullptr;
   CHECK_HIP_ERROR(hipMalloc(&deviceHandles_d, totalNumQueues * sizeof(anvil::SdmaQueueDeviceHandle*)));

   size_t queueIdx = 0;
   for (auto& dstDeviceId : dstDeviceIds)
   {
      for (size_t q = 0; q < params.numOfQueues; q++)
      {
         deviceHandles_d[queueIdx] = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
         queueIdx++;
      }
   }

   // Allocate memories for timestamps
   long long int* start_clock_count;
   long long int* end_clock_count;
   long long int* start_clock_count_d;
   long long int* end_clock_count_d;

   CHECK_HIP_ERROR(hipHostMalloc(&start_clock_count, params.numIterations * totalNumWarps * sizeof(long long int)));
   CHECK_HIP_ERROR(hipHostMalloc(&end_clock_count, params.numIterations * totalNumWarps * sizeof(long long int)));
   CHECK_HIP_ERROR(hipMalloc(&start_clock_count_d, params.numIterations * totalNumWarps * sizeof(long long int)));
   CHECK_HIP_ERROR(hipMalloc(&end_clock_count_d, params.numIterations * totalNumWarps * sizeof(long long int)));

   CHECK_HIP_ERROR(hipMemset(start_clock_count_d, 0, params.numIterations * totalNumWarps * sizeof(long long int)));
   CHECK_HIP_ERROR(hipMemset(end_clock_count_d, 0, params.numIterations * totalNumWarps * sizeof(long long int)));

   // ======================
   // 4. Kernel Launch
   // ======================

   int numWgs = params.numDestinations * params.numOfQueues * params.numOfWGPerQueue;
   if (params.verbose)
   {
      std::cout << "BlockDim.x: " << wgSize << ", GridDim.x: " << numWgs << std::endl;
      std::cout << "#Warps/Q or #Warps/WG: " << params.numOfWarpsPerWG << std::endl;
   }

   dim3 grid(numWgs, 1, 1);
   dim3 block(wgSize, 1, 1);

   Reporter report(params.resultFileName);
   report.setParameters(srcDeviceId, dstDeviceIds.size(), params.numOfQueues, numWgs, wgSize, params.numCopyCommands);
   // Print header to stdout
   printHeader(std::cout, headerFields);
   std::cout << std::endl;

   for (size_t copySize = params.minCopySize; copySize <= params.maxCopySize; copySize *= 2)
   {
      // Transfer Size Per Destination
      size_t totalTransferSize =
          copySize * params.numCopyCommands * params.numOfWarpsPerWG * params.numOfWGPerQueue * params.numOfQueues;

      if (params.verbose)
      {
         std::cout << "Copy Size: " << copySize << " bytes." << std::endl;
         std::cout << "Transfer Size Per xGMI link: " << totalTransferSize << " bytes." << std::endl;
      }

      // Clear destination buffers before verification
      for (void* buf : sdmaDestBufferPtr)
      {
         CHECK_HIP_ERROR(hipMemset(buf, 0, totalTransferSize));
      }

      // std::cout<<"Verification Kernel Run Start."<<std::endl;

      // Clear signal buffers
      CHECK_HIP_ERROR(hipMemset(signalPtrs, 0, sizeof(HSAuint64) * totalNumWarps));
      HSAuint64 expectedSignal = 1;
      auto kernel = multiQueueSDMATransferQueueMapWG;

      std::optional<size_t> numErrors;
      if (!params.skipVerification)
      {
         hipLaunchKernelGGL(kernel, grid, block, 0 /*dynamicShared*/, 0 /*stream*/, 0, sdma_src_buf, sdma_dst_bufs_d,
                            copySize, params.numCopyCommands, params.numDestinations, params.numOfQueues, params.numOfWGPerQueue,
                            deviceHandles_d, signalPtrs, expectedSignal, start_clock_count_d, end_clock_count_d);
         CHECK_HIP_ERROR(hipDeviceSynchronize());
         expectedSignal++;
         numErrors = verifyData(hostSrcBuffer, sdma_dst_bufs_d, dstDeviceIds.size(), totalTransferSize);
         if (numErrors != 0)
         {
            std::cerr << "Data verification failed\n";
            exit(-1);
         }
         // std::cout<<"Verification Kernel Run Finished"<<std::endl;
      }

      // Warming Up the kernel
      for (size_t i = 0; i < params.nWarmupIterations; ++i)
      {
         hipLaunchKernelGGL(kernel, grid, block, 0 /*dynamicShared*/, 0 /*stream*/, i, sdma_src_buf, sdma_dst_bufs_d,
                            copySize, params.numCopyCommands, params.numDestinations, params.numOfQueues, params.numOfWGPerQueue,
                            deviceHandles_d, signalPtrs, expectedSignal, start_clock_count_d, end_clock_count_d);
         expectedSignal++;
      }
      CHECK_HIP_ERROR(hipDeviceSynchronize());

      // Setup hipEvents for kernel-level timestamping
      std::vector<hipEvent_t> timestamps_events(params.numIterations * 2);

      for (int iter = 0; iter < params.numIterations * 2; iter++)
      {
         CHECK_HIP_ERROR(hipEventCreate(&timestamps_events[iter]));
      }

      // Reset for measurement
      std::vector<double> latency_device;
      std::vector<double> latency_host;
      long long int* startTimestampPtr = start_clock_count_d;
      long long int* endTimestampPtr = end_clock_count_d;

      for (size_t iter = 0; iter < params.numIterations; ++iter)
      {
         hipExtLaunchKernelGGL(kernel, grid, block, 0 /*dynamicShared*/, 0 /*stream*/,
                               timestamps_events[iter * 2],     // start event
                               timestamps_events[iter * 2 + 1], // stop event
                               0,                               // flags
                               iter, sdma_src_buf, sdma_dst_bufs_d, copySize, params.numCopyCommands,
                               params.numDestinations, params.numOfQueues, params.numOfWGPerQueue, deviceHandles_d, signalPtrs, expectedSignal,
                               startTimestampPtr, endTimestampPtr);
         startTimestampPtr += totalNumWarps;
         endTimestampPtr += totalNumWarps;

         expectedSignal++;
      }
      CHECK_HIP_ERROR(hipDeviceSynchronize());

      // ======================
      // 5. Performance Metrics
      // ======================
      CHECK_HIP_ERROR(hipMemcpy(start_clock_count, start_clock_count_d,
                                params.numIterations * totalNumWarps * sizeof(long long int), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(end_clock_count, end_clock_count_d,
                                params.numIterations * totalNumWarps * sizeof(long long int), hipMemcpyDeviceToHost));

      int totalPacketsPerIteration = totalNumWarps * params.numCopyCommands;
      int totalWarpsPerIteration = totalNumWarps;

      for (size_t iter = 0; iter < params.numIterations; ++iter)
      {
         // Calculate mean elapsed time for each iterations, for nPackets or nWarps
         double device_latency_ms =
             calcMeanLatencyofGPUTransfer(start_clock_count + (iter * totalWarpsPerIteration),
                                          end_clock_count + (iter * totalWarpsPerIteration), params.numDestinations,
                                          params.numOfQueues * params.numOfWGPerQueue, params.numOfWarpsPerWG) /
             1e5;

         float host_latency_ms;
         CHECK_HIP_ERROR(
             hipEventElapsedTime(&host_latency_ms, timestamps_events[iter * 2], timestamps_events[iter * 2 + 1]));

         latency_device.push_back(device_latency_ms);
         latency_host.push_back(host_latency_ms);
      }

      auto [latency_device_mean, latency_device_std] = avg_std(latency_device);
      auto [latency_host_mean, latency_host_std] = avg_std(latency_host);

      double sizeAcrossAllLinks = (double)dstDeviceIds.size() * totalTransferSize;

      double deviceBandwidth_gbs = (sizeAcrossAllLinks / 1.0E9) / (latency_device_mean / 1000);
      double hostBandwidth_gbs = (sizeAcrossAllLinks / 1.0E9) / (latency_host_mean / 1000);

      printRowOfResults(std::cout, srcDeviceId, dstDeviceIds.size(), numWgs, wgSize, totalTransferSize, copySize,
                        params.numCopyCommands, latency_device_mean * 1000, latency_device_std, deviceBandwidth_gbs,
                        latency_host_mean * 1000, latency_host_std, hostBandwidth_gbs, numErrors);
      std::cout << std::endl;

      report.addResult(totalTransferSize, copySize, latency_device_mean * 1000, latency_device_std, deviceBandwidth_gbs,
                       latency_host_mean * 1000, latency_host_std, hostBandwidth_gbs);
      for (int iter = 0; iter < (params.numIterations * 2); iter++)
      {
         CHECK_HIP_ERROR(hipEventDestroy(timestamps_events[iter]));
      }
   } // for params.numIterations

   // Write result to file
   report.writeFile();

   //======================
   // 7. Resource Cleanup
   // ======================
   CHECK_HIP_ERROR(hipFreeHost(start_clock_count));
   CHECK_HIP_ERROR(hipFreeHost(end_clock_count));
   CHECK_HIP_ERROR(hipFree(start_clock_count_d));
   CHECK_HIP_ERROR(hipFree(end_clock_count_d));

   CHECK_HIP_ERROR(hipFree(deviceHandles_d));
}

int main(int argc, char** argv)
{
   anvil::anvil.init();

   const int srcGpuId = 6;

   //CLI::App app("Shader-initiated SDMA");
   size_t minCopySize{0};
   //app.add_option("-b,--minCopySize", minCopySize, "Minimum Transfer Size [B] (per copy command)");
   size_t maxCopySize{0};
   //app.add_option("-e,--maxCopySize", maxCopySize, "Maximum Transfer Size [B] (per copy command)");
   size_t numCopyCommands{0};
   //app.add_option("-c,--numCopyCommands", numCopyCommands, "Number of copy commands (per warp)");

   bool skipVerification{false};
   //app.add_flag("--skip-verification", skipVerification, "Skip verification");

   size_t nWarmupIterations{3};
   //app.add_option("-w,--warmup", nWarmupIterations, "Number of warmup iterations");

   size_t numIterations{50};
   //app.add_option("-n,--iterations", numIterations, "Number of iterations");

   size_t numDestinations{1};
   //app.add_option("-d,--numDestinations", numDestinations, "Number of destination GPUs");

   size_t numOfQueues{1};
   //app.add_option("--numOfQueuesPerDestination", numOfQueues,
   //               "Number of queues per destination, corresponds to number of workgroups/threadblocks");

   size_t numOfWarpsPerWG{1};
   //app.add_option("--warpsPerWG", numOfWarpsPerWG, "Number of warps shared the same queue resources");

   size_t numOfWGPerQueue{1};
   //app.add_option("--wgsPerQueue", numOfWGPerQueue, "Number of workgroups shared the same queue resources");

   std::string resultFileName = "MultiQueueGPU2GPU_Performance.csv";
   //app.add_option("-o,--outputFile", resultFileName, "Filename for result");

   bool verbose{false};
   //app.add_flag("-v, --verbose", verbose, "verbose output");

   //CLI11_PARSE(app, argc, argv);

   std::cout << "==== Running shader_bw doing " << numCopyCommands << " copies of size " << minCopySize << " to "
             << maxCopySize << " ====" << std::endl;

   ExperimentParams params{
       .minCopySize = minCopySize,
       .maxCopySize = maxCopySize,
       .numCopyCommands = numCopyCommands,
       .commitEveryTransfer = true,
       .skipVerification = skipVerification,
       .nWarmupIterations = nWarmupIterations,
       .numIterations = numIterations,
       .numDestinations = numDestinations,
       .numOfQueues = numOfQueues,
       .numOfWarpsPerWG = numOfWarpsPerWG,
       .numOfWGPerQueue = numOfWGPerQueue,
       .resultFileName = resultFileName,
       .verbose = verbose,
   };

   //runExperiment(srcGpuId, params);

   return 0;
}

