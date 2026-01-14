#include <stdio.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/application/transport/sdma/anvil.hpp"
#include "common.hpp"
#include "sdma_rate_kernel.h"
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
#include <stdexcept> // Required for std::runtime_error
#include <unistd.h>

namespace fs = std::filesystem;

#include "CLI11.hpp"

constexpr uint32_t MAGIC_VALUE = 0xDEADBEEF;

struct ExperimentParams
{
   size_t minCopySize;
   size_t maxCopySize;
   size_t numCopyCommands;
   bool skipVerification;
   size_t nWarmupIterations;
   size_t numIterations;
   size_t numOfQueues;
   std::string resultFileName;
   bool verbose;
};

void runExperiment(int srcDeviceId, int dstDeviceId, const ExperimentParams& params)
{
   // =============================
   // 1. Initial Setup
   // =============================
   int hipDeviceCount = 0;
   CHECK_HIP_ERROR(hipGetDeviceCount(&hipDeviceCount));
   if (!(srcDeviceId < hipDeviceCount && dstDeviceId < hipDeviceCount))
   {
      throw std::runtime_error("Error: not enough devices for requested device IDs.");
   }
   std::cout << "Src GPU Device Id: " << srcDeviceId << std::endl;
   std::cout << "Dst GPU Device Id: " << dstDeviceId << std::endl;
   std::vector<void*> sdmaDestBufferPtr;

   CHECK_HIP_ERROR(hipSetDevice(srcDeviceId));

   //==============================
   // 2. Resource Allocation
   //==============================

   int wgSize = params.numOfQueues * WARP_SIZE;
   size_t totalNumWarps = params.numOfQueues;

   // Allocate signals, one for each warp
   HSAuint64* signalPtrs;
   CHECK_HIP_ERROR(hipMalloc(&signalPtrs, sizeof(HSAuint64) * totalNumWarps));

   // single P2P transfer size
   size_t p2pTransferSize = params.maxCopySize * params.numCopyCommands * params.numOfQueues;

   void* sdma_src_buf = nullptr;
   CHECK_HIP_ERROR(hipExtMallocWithFlags(&sdma_src_buf, p2pTransferSize, hipDeviceMallocUncached));

   // Fill src buf with MAGIC
   size_t num_elements = p2pTransferSize / sizeof(uint32_t);
   std::vector<uint32_t> hostSrcBuffer(num_elements, MAGIC_VALUE);
   CHECK_HIP_ERROR(hipMemcpy(sdma_src_buf, hostSrcBuffer.data(), p2pTransferSize, hipMemcpyHostToDevice));

   HsaMemFlags memFlags = {};
   memFlags.ui32.NonPaged = 1;
   memFlags.ui32.HostAccess = 1;
   memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
   memFlags.ui32.NoNUMABind = 1;
   memFlags.ui32.ExecuteAccess = 1;
   memFlags.ui32.Uncached = 1;

   void* buf;
   CHECK_HIP_ERROR(hipSetDevice(dstDeviceId));
   CHECK_HIP_ERROR(hipExtMallocWithFlags(&buf, p2pTransferSize, hipDeviceMallocUncached));
   EnablePeerAccess(srcDeviceId, dstDeviceId);
   sdmaDestBufferPtr.push_back(buf);

   CHECK_HIP_ERROR(hipSetDevice(srcDeviceId));

   void** sdma_dst_bufs_d;
   CHECK_HIP_ERROR(hipMalloc((void**)&sdma_dst_bufs_d, sdmaDestBufferPtr.size() * sizeof(void*)));
   CHECK_HIP_ERROR(hipMemcpy(sdma_dst_bufs_d, sdmaDestBufferPtr.data(), sdmaDestBufferPtr.size() * sizeof(void*),
                             hipMemcpyHostToDevice));

   // ======================
   // 3. Queue Setup
   // ======================

   size_t totalNumQueues = params.numOfQueues;
   anvil::anvil.connect(srcDeviceId, dstDeviceId, params.numOfQueues);

   anvil::SdmaQueueDeviceHandle** deviceHandles_d = nullptr;
   CHECK_HIP_ERROR(hipMalloc(&deviceHandles_d, totalNumQueues * sizeof(anvil::SdmaQueueDeviceHandle*)));

   size_t queueIdx = 0;
   for (size_t q = 0; q < params.numOfQueues; q++)
   {
      deviceHandles_d[queueIdx] = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();

      // if (params.verbose)
      // {
      //    std::cout << std::dec << "Q[" << queueIdx << "]: "
      //              << "wptr: " << deviceHandles[queueIdx]->wptr << ", "
      //              << "rptr: " << deviceHandles[queueIdx]->rptr << ", "
      //              << "doorbell: " << deviceHandles[queueIdx]->doorbell << ", "
      //              << "signal: " << signalPtrs[queueIdx] << ", "
      //              << "Queue cmd buffer address: " << deviceHandles[queueIdx]->queueBuf << ", "
      //              << "committedWptr: " << *deviceHandles[queueIdx]->committedWptr << ", "
      //              << "pendingWptr: " << *deviceHandles[queueIdx]->cachedWptr << ", " << std::endl;
      // }
      queueIdx++;
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
   int numWgs = params.numOfQueues;
   dim3 grid(numWgs, 1, 1);
   dim3 block(wgSize, 1, 1);

   Reporter report(params.resultFileName);
   report.setParameters(srcDeviceId, 1, params.numOfQueues, numWgs, wgSize, params.numCopyCommands);
   // Print header to stdout
   printHeader(std::cout, headerFields);
   std::cout << std::endl;

   for (size_t copySize = params.minCopySize; copySize <= params.maxCopySize; copySize *= 2)
   {
      size_t totalTransferSize = copySize * params.numCopyCommands * params.numOfQueues;
      if (params.verbose)
      {
         std::cout << "Copy Size: " << copySize << " bytes." << std::endl;
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
      // auto kernel = gpuKernels[{useSingleProducer, params.fineGrainedLatency}];

      std::optional<size_t> numErrors;
      if (!params.skipVerification)
      {
         // hipLaunchKernelGGL(kernel, grid, block, 0/*dynamicShared*/, 0/*stream*/,
         packet_rate_kernel<<<1, wgSize, 0>>>(sdma_src_buf, sdma_dst_bufs_d[0], copySize, params.numCopyCommands,
                                              deviceHandles_d, signalPtrs, expectedSignal, start_clock_count_d,
                                              end_clock_count_d);
         CHECK_HIP_ERROR(hipDeviceSynchronize());
         expectedSignal++;
         numErrors = verifyData(hostSrcBuffer, sdma_dst_bufs_d, 1, totalTransferSize);
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
         // hipLaunchKernelGGL(kernel, grid, block, 0/*dynamicShared*/, 0/*stream*/,
         packet_rate_kernel<<<1, wgSize, 0>>>(sdma_src_buf, sdma_dst_bufs_d[0], copySize, params.numCopyCommands,
                                              deviceHandles_d, signalPtrs, expectedSignal, start_clock_count_d,
                                              end_clock_count_d);
         expectedSignal++;
      }
      CHECK_HIP_ERROR(hipDeviceSynchronize());

      // Setup hipEvents for kernel-level timestamping
      std::vector<hipEvent_t> timestamps_events(params.numIterations + 1);

      for (int iter = 0; iter < params.numIterations + 1; iter++)
      {
         CHECK_HIP_ERROR(hipEventCreate(&timestamps_events[iter]));
      }

      // Reset for measurement
      long long int* startTimestampPtr = start_clock_count_d;
      long long int* endTimestampPtr = end_clock_count_d;

      for (size_t iter = 0; iter < params.numIterations; ++iter)
      {
         CHECK_HIP_ERROR(hipEventRecord(timestamps_events[iter]));
         // hipLaunchKernelGGL(kernel, grid, block, 0/*dynamicShared*/, 0/*stream*/,
         packet_rate_kernel<<<1, wgSize, 0>>>(sdma_src_buf, sdma_dst_bufs_d[0], copySize, params.numCopyCommands,
                                              deviceHandles_d, signalPtrs, expectedSignal, startTimestampPtr,
                                              endTimestampPtr);
         startTimestampPtr += totalNumWarps;
         endTimestampPtr += totalNumWarps;

         expectedSignal++;
      }
      CHECK_HIP_ERROR(hipEventRecord(timestamps_events[params.numIterations]));
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

      std::vector<double> latency_device(params.numIterations);
      std::vector<double> latency_host(params.numIterations);

      for (size_t iter = 0; iter < params.numIterations; ++iter)
      {
         // Calculate mean elapsed time for each iterations, for nPackets or nWarps

         latency_device[iter] = calcMeanLatencyofGPUTransfer(start_clock_count + (iter * totalWarpsPerIteration),
                                                             end_clock_count + (iter * totalWarpsPerIteration), 1, 1) /
                                1e5;

         float host_latency_ms;
         CHECK_HIP_ERROR(hipEventElapsedTime(&host_latency_ms, timestamps_events[iter], timestamps_events[iter + 1]));
         latency_host[iter] = host_latency_ms;
      }

      auto [latency_device_mean, latency_device_std] = avg_std(latency_device);
      auto [latency_host_mean, latency_host_std] = avg_std(latency_host);

      double sizeAcrossAllLinks = totalTransferSize;
      double deviceBandwidth_gbs = (sizeAcrossAllLinks / 1.0E9) / (latency_device_mean / 1000);
      double hostBandwidth_gbs = (sizeAcrossAllLinks / 1.0E9) / (latency_host_mean / 1000);

      printRowOfResults(std::cout, srcDeviceId, 1, numWgs, wgSize, totalTransferSize, copySize, params.numCopyCommands,
                        latency_device_mean * 1000, latency_device_std, deviceBandwidth_gbs, latency_host_mean * 1000,
                        latency_host_std, hostBandwidth_gbs, numErrors);
      std::cout << std::endl;
      std::cout << "packet rate " << (double)totalPacketsPerIteration / (latency_device_mean * 1000) << "MPPS"
                << std::endl;

      report.addResult(totalTransferSize, copySize, latency_device_mean * 1000, latency_device_std, deviceBandwidth_gbs,
                       latency_host_mean * 1000, latency_host_std, hostBandwidth_gbs);

      for (int iter = 0; iter < (params.numIterations + 1); iter++)
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
}

int main(int argc, char** argv)
{
   anvil::anvil.init();

   int srcGpuId = 2;
   int dstGPUId = 7;

   CLI::App app("Shader-initiated SDMA");
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

   size_t numIterations{100};
   app.add_option("-n,--iterations", numIterations, "Number of iterations");

   size_t numOfQueues{1};
   app.add_option("--numOfQueues", numOfQueues, "Number of queues per");

   std::string resultFileName = "packet_rate.csv";
   app.add_option("-o,--outputFile", resultFileName, "Filename for result");

   bool verbose{false};
   app.add_flag("-v, --verbose", verbose, "verbose output");

   CLI11_PARSE(app, argc, argv);

   std::cout << "==== Running shader_sdma doing " << numCopyCommands << " copies of size " << minCopySize << " to "
             << maxCopySize << " ====" << std::endl;

   ExperimentParams params{
       .minCopySize = minCopySize,
       .maxCopySize = maxCopySize,
       .numCopyCommands = numCopyCommands,
       .skipVerification = skipVerification,
       .nWarmupIterations = nWarmupIterations,
       .numIterations = numIterations,
       .numOfQueues = numOfQueues,
       .resultFileName = resultFileName,
       .verbose = verbose,
   };

   runExperiment(srcGpuId, dstGPUId, params);

   return 0;
}
