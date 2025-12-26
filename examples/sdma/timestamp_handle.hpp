#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "sdma_queue_host.hpp"

#include <array>
#include <vector>

struct LatencyBreakdown
{
   std::vector<double> reserveQueueSpace_et;
   std::vector<double> entailPacket_et;
   std::vector<double> submitPacket_et;
   std::vector<double> sdmaTransfer_et;

   std::vector<double> reserveQueueSpaceFence_et;
   std::vector<double> entailFencePacket_et;
   std::vector<double> submitFencePacket_et;
   std::vector<double> sdmaFenceTransfer_et;
   std::vector<double> latency_device;
   std::vector<double> latency_host;
};

enum PerIterationTimeStamps
{
   RESERVE_SPACE_FENCE_START,
   RESERVE_SPACE_FENCE_END,
   ENTAIL_FENCE_PACKET_START,
   ENTAIL_FENCE_PACKET_END,
   SUBMIT_FENCE_PACKET_START,
   SUBMIT_FENCE_PACKET_END,
   TRANSFER_START,
   TRANSFER_END,
   NUM_TIMESTAMPS
};

struct TimeStampBreakdown
{
   // Per copy packet timestamps
   long long int* reserveQueueSpace_st = nullptr;
   long long int* reserveQueueSpace_e = nullptr;
   long long int* entailPacket_st = nullptr;
   long long int* entailPacket_e = nullptr;
   long long int* submit_st = nullptr;
   long long int* submit_e = nullptr;

   // Per iteration timestamps
   std::array<long long int, PerIterationTimeStamps::NUM_TIMESTAMPS>* iterTimeStamps = nullptr;
};

struct TimeStampHandle
{
   TimeStampBreakdown t_h;
   TimeStampBreakdown* t_d = nullptr;
   size_t countNWarps = 0;
   size_t countNPackets = 0;
   size_t sizeOfPerIterTimestamps = sizeof(long long int) * PerIterationTimeStamps::NUM_TIMESTAMPS;

   TimeStampHandle() = default;

   void Alloc(size_t _countNWarps, size_t _countNPackets)
   {
      countNWarps = _countNWarps;
      countNPackets = _countNPackets;

      // std::cout<<"NPackets: "<<countNPackets<<std::endl;

      t_h.reserveQueueSpace_st = new long long int[countNPackets];
      t_h.reserveQueueSpace_e = new long long int[countNPackets];
      t_h.entailPacket_st = new long long int[countNPackets];
      t_h.entailPacket_e = new long long int[countNPackets];
      t_h.submit_st = new long long int[countNPackets];
      t_h.submit_e = new long long int[countNPackets];

      t_h.iterTimeStamps = new std::array<long long int, PerIterationTimeStamps::NUM_TIMESTAMPS>[countNWarps];

      TimeStampBreakdown temp_dt;

      // Measure Each Packet Reserve Space, Entails Packet and Submit Packet Time
      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.reserveQueueSpace_st, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.reserveQueueSpace_st, 0, countNPackets * sizeof(long long int)));

      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.reserveQueueSpace_e, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.reserveQueueSpace_e, 0, countNPackets * sizeof(long long int)));

      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.entailPacket_st, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.entailPacket_st, 0, countNPackets * sizeof(long long int)));

      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.entailPacket_e, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.entailPacket_e, 0, countNPackets * sizeof(long long int)));

      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.submit_st, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.submit_st, 0, countNPackets * sizeof(long long int)));

      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.submit_e, countNPackets * sizeof(long long int)));
      CHECK_HIP_ERROR(hipMemset(temp_dt.submit_e, 0, countNPackets * sizeof(long long int)));

      // Measure SDMA Transfer Time for Each Warp, not for for each copy packets
      CHECK_HIP_ERROR(hipMalloc((void**)&temp_dt.iterTimeStamps, countNWarps * sizeOfPerIterTimestamps));
      CHECK_HIP_ERROR(hipMemset(temp_dt.iterTimeStamps, 0, countNWarps * sizeOfPerIterTimestamps));

      CHECK_HIP_ERROR(hipMalloc((void**)&t_d, sizeof(TimeStampBreakdown)));
      CHECK_HIP_ERROR(hipMemcpy((void*)t_d, &temp_dt, sizeof(TimeStampBreakdown), hipMemcpyHostToDevice));
   }

   void CopyHtoD()
   {
      if (!t_d)
         return;

      TimeStampBreakdown temp_d;
      CHECK_HIP_ERROR(hipMemcpy((void*)&temp_d, t_d, sizeof(TimeStampBreakdown), hipMemcpyDeviceToHost));

      CHECK_HIP_ERROR(hipMemcpy(temp_d.reserveQueueSpace_st, t_h.reserveQueueSpace_st,
                                countNPackets * sizeof(long long int), hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(temp_d.reserveQueueSpace_e, t_h.reserveQueueSpace_e,
                                countNPackets * sizeof(long long int), hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(temp_d.entailPacket_st, t_h.entailPacket_st, countNPackets * sizeof(long long int),
                                hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(temp_d.entailPacket_e, t_h.entailPacket_e, countNPackets * sizeof(long long int),
                                hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(
          hipMemcpy(temp_d.submit_st, t_h.submit_st, countNPackets * sizeof(long long int), hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(
          hipMemcpy(temp_d.submit_e, t_h.submit_e, countNPackets * sizeof(long long int), hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(temp_d.iterTimeStamps, t_h.iterTimeStamps, countNWarps * sizeOfPerIterTimestamps,
                                hipMemcpyHostToDevice));
   }

   void CopyDtoH()
   {
      if (!t_d)
         return;

      TimeStampBreakdown temp_d;
      CHECK_HIP_ERROR(hipMemcpy((void*)&temp_d, t_d, sizeof(TimeStampBreakdown), hipMemcpyDeviceToHost));

      CHECK_HIP_ERROR(hipMemcpy(t_h.reserveQueueSpace_st, temp_d.reserveQueueSpace_st,
                                countNPackets * sizeof(long long int), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(t_h.reserveQueueSpace_e, temp_d.reserveQueueSpace_e,
                                countNPackets * sizeof(long long int), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(t_h.entailPacket_st, temp_d.entailPacket_st, countNPackets * sizeof(long long int),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(t_h.entailPacket_e, temp_d.entailPacket_e, countNPackets * sizeof(long long int),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(
          hipMemcpy(t_h.submit_st, temp_d.submit_st, countNPackets * sizeof(long long int), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(
          hipMemcpy(t_h.submit_e, temp_d.submit_e, countNPackets * sizeof(long long int), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(t_h.iterTimeStamps, temp_d.iterTimeStamps, countNWarps * sizeOfPerIterTimestamps,
                                hipMemcpyDeviceToHost));

      // for(int i=0; i<countNPackets; i++)
      // {
      //     // std::cout<<"Reserve Space Start Time["<<i/BLOCK_DIM<<"]"<<"["<< i%BLOCK_DIM <<"]:
      //     "<<t_h.reserveQueueSpace_st[i]<<std::endl;
      //     // std::cout<<"Reserve Space End Time["<<i/BLOCK_DIM<<"]"<<"["<< i%BLOCK_DIM <<"]:
      //     "<<t_h.reserveQueueSpace_e[i]<<std::endl; std::cout<<"Entails Packet Start Time
      //     ["<<i<<"]: "<<t_h.entailPacket_st[i]<<std::endl; std::cout<<"Entails Packet End Time
      //     ["<<i<<"]: "<<t_h.entailPacket_e[i]<<std::endl;
      // }
   }

   void Free()
   {
      delete[] t_h.reserveQueueSpace_st;
      delete[] t_h.reserveQueueSpace_e;
      delete[] t_h.entailPacket_st;
      delete[] t_h.entailPacket_e;
      delete[] t_h.submit_st;
      delete[] t_h.submit_e;
      delete[] t_h.iterTimeStamps;

      t_h = {};

      if (t_d)
      {
         TimeStampBreakdown temp_d;
         CHECK_HIP_ERROR(hipMemcpy(&temp_d, t_d, sizeof(TimeStampBreakdown), hipMemcpyDeviceToHost));

         CHECK_HIP_ERROR(hipFree(temp_d.reserveQueueSpace_st));
         CHECK_HIP_ERROR(hipFree(temp_d.reserveQueueSpace_e));
         CHECK_HIP_ERROR(hipFree(temp_d.entailPacket_st));
         CHECK_HIP_ERROR(hipFree(temp_d.entailPacket_e));
         CHECK_HIP_ERROR(hipFree(temp_d.submit_st));
         CHECK_HIP_ERROR(hipFree(temp_d.submit_e));
         CHECK_HIP_ERROR(hipFree(temp_d.iterTimeStamps));
         CHECK_HIP_ERROR(hipFree(t_d));
         t_d = nullptr;
      }

      countNPackets = 0;
      countNWarps = 0;
   }

   ~TimeStampHandle()
   {
      Free();
   }
};

template <PerIterationTimeStamps START, PerIterationTimeStamps END>
double calcMeanLatencyAcrossWarps(std::array<long long int, PerIterationTimeStamps::NUM_TIMESTAMPS>* ts, size_t nWarps)
{
   double total = 0;

   for (size_t i = 0; i < nWarps; ++i)
   {
      total += (ts[i][END] - ts[i][START]);
   }
   return total / (double)nWarps;
}
