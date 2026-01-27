/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "sdma_queue_host.hpp"

// Allow access to peerDeviceId from deviceId
inline void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
   int canAccess;
   CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
   if (!canAccess)
   {
      std::cerr << "Unable to enable peer access from GPU devices " << deviceId << " to " << peerDeviceId << "\n";
   }

   CHECK_HIP_ERROR(hipSetDevice(deviceId));
   hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
   if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
   {
      std::cerr << "Unable to enable peer to peer access from " << deviceId << "  to " << peerDeviceId << " ("
                << hipGetErrorString(error) << ")\n";
   }
}

template <typename T>
void printList(std::ostream& out, const std::vector<T>& list, const std::string& preamble = "",
               const std::string& sep = ", ", const std::string& end = "\n")
{
   out << preamble;
   for (const auto& entry : list)
   {
      out << entry;
      if (&entry != &list.back())
      {
         out << sep;
      }
   }
   out << end;
}

std::string prettyPrintSize(size_t bytes);

auto min_clock = [](long long int* first, long long int* last) {
   long long int min_clock = *first;
   for (auto p = (first + 1); p != last; ++p)
   {
      if (*p < min_clock)
      {
         min_clock = *p;
      }
   }
   return min_clock;
};

auto max_clock = [](long long int* first, long long int* last) {
   long long int max_clock = *first;
   for (auto p = (first + 1); p != last; ++p)
   {
      if (*p > max_clock)
      {
         max_clock = *p;
      }
   }
   return max_clock;
};

auto avg_std = [](const std::vector<double>& nums) -> std::pair<double, double> {
   double mean = std::accumulate(nums.begin(), nums.end(), 0.0) / nums.size();
   double variance = 0.0;
   for (double x : nums)
      variance += (x - mean) * (x - mean);
   variance /= nums.size(); // use (nums.size() - 1) for sample stddev
   double stddev = std::sqrt(variance);
   return {mean, stddev};
};

inline double calcMeanLatencyofPacket(long long int* start_t, long long int* end_t, size_t size)
{
   double total = 0;

   for (int i = 0; i < size; ++i)
   {
      total += (end_t[i] - start_t[i]);
   }
   return total / (double)size;
}

// Take Min/Max across warps and mean across threadblocks/destinations
inline double calcMeanLatencyofGPUTransfer(long long int* start_t, long long int* end_t, int nBlocks, size_t nWarps, int nBlocksPerDest = 1)
{
   const size_t stridePerDst = static_cast<size_t>(nBlocksPerDest) * nWarps;
   double totalDuration = 0.0;

   for (int b = 0; b < nBlocks; ++b)
   {
      long long* start_base = start_t + b * stridePerDst;
      long long* end_base = end_t + b * stridePerDst;

      double sumForDst = 0.0;

      for (int bd = 0; bd < nBlocksPerDest; ++bd) {
         long long* sb = start_base + static_cast<size_t>(bd) * nWarps;
         long long* eb = end_base   + static_cast<size_t>(bd) * nWarps;

         long long min_start = min_clock(sb, sb + nWarps);
         long long max_end   = max_clock(eb, eb + nWarps);

         sumForDst += static_cast<double>(max_end - min_start);
      }

      totalDuration += (sumForDst / static_cast<double>(nBlocksPerDest));
   }
   return totalDuration / nBlocks;
}

class Reporter
{
 public:
   Reporter(std::string fileName);
   ~Reporter()
   {
      if (_outputValid)
      {
         _out.close();
      }
   };
   void setParameters(const int srcGpu, const size_t numDest, const size_t numQueues, const int gridDim,
                      const int blockDim, const size_t numCopies);
   void addResult(size_t totalTransferSize, size_t copySize, double deviceLatency, double deviceLatencyStd,
                  double deviceBandwidth, double hostLatency, double hostLatencyStd, double hostBandwidth);
   void addResult(size_t totalTransferSize, size_t copySize, double deviceLatency, double deviceLatencyStd,
                  double deviceBandwidth, double hostLatency, double hostLatencyStd, double hostBandwidth,
                  double copyReserveLat, double copyReserveStd, double copyPlaceLat, double copyPlaceStd,
                  double copySubmitLat, double copySubmitStd, double fenceReserveLat, double fenceReserveStd,
                  double fencePlaceLat, double fencePlaceStd, double fenceSubmitLat, double fenceSubmitStd,
                  double sdmaTransferLat, double sdmaTransferStd);
   void writeFile();

 private:
   void addResultCommon(std::vector<std::pair<std::string, std::string>>& outputValuesKeys, size_t totalTransferSize,
                        size_t copySize, double deviceLatency, double deviceLatencyStd, double deviceBandwidth,
                        double hostLatency, double hostLatencyStd, double hostBandwidth);
   template <typename T> std::pair<std::string, std::string> makeValueKeyPair(T v, std::string k)
   {
      return std::make_pair(std::to_string(v), k);
   };
   template <> std::pair<std::string, std::string> makeValueKeyPair<std::string>(std::string v, std::string k)
   {
      return std::make_pair("\"" + v + "\"", k);
   };
   bool _outputValid = false;
   std::ofstream _out;
   int _srcGpu = 0;
   std::size_t _numDest;
   std::size_t _numQueues;
   int _gridDim;
   int _blockDim;
   std::size_t _numCopies;
   std::vector<std::vector<std::pair<std::string, std::string>>> _outputData;
};
