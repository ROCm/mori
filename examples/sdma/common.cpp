/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "common.hpp"

#include <hip/hip_runtime.h>

#include "utils.hpp"

void printHeader(std::ostream& out, const std::array<std::string_view, 13>& headers)
{
   size_t deviceWidth = headerFields[6].length() + headerFields[7].length() + headerFields[8].length() + 4;
   size_t hostWidth = headerFields[9].length() + headerFields[10].length() + headerFields[11].length() + 4;

   std::cout << "                                      |" << std::setw(deviceWidth) << "Device"
             << "|" << std::setw(hostWidth) << "Host"
             << "|" << std::endl;

   for (auto& field : headerFields)
   {
      size_t width = field.length() + 1;
      out << std::setw(width) << field;
      if (&field != &headerFields.back())
      {
         out << ",";
      }
   }
}

void printRowOfResults(std::ostream& out, int srcGpuId, size_t num_dsts, int numBlocks, int numThreadsPerBlock,
                       size_t totalTransferSize, size_t copySize, size_t numCopies, double deviceLatency,
                       double deviceLatencyStd, double deviceBandwidth, double hostLatency, double hostLatencyStd,
                       double hostBandwidth, std::optional<size_t> numErrors)
{
   std::string numErrorsStr = (numErrors.has_value()) ? std::to_string(numErrors.value()) : "N/A";

   out << std::setw(headerFields[0].length() + 1) << srcGpuId << "," << std::setw(headerFields[1].length() + 1)
       << num_dsts << "," << std::setw(headerFields[2].length() + 1) << numBlocks << ", "
       << std::setw(headerFields[3].length() + 1) << numThreadsPerBlock
       << ", "
       // << std::setw(headerFields[3].length() + 1) <<  totalTransferSize<< ","
       << std::setw(headerFields[4].length()) << prettyPrintSize(copySize)
       << ", " // not sure why setw behaves not as expected
       << std::setw(headerFields[5].length() + 1) << numCopies << "," << std::setw(headerFields[6].length() + 1)
       << std::fixed << std::setprecision(1) << deviceLatency << "," << std::setw(headerFields[7].length() + 1)
       << std::fixed << std::setprecision(2) << deviceLatencyStd << "," << std::setw(headerFields[8].length() + 1)
       << std::fixed << std::setprecision(1) << deviceBandwidth << "," << std::setw(headerFields[9].length() + 1)
       << std::fixed << std::setprecision(2) << hostLatency << "," << std::setw(headerFields[10].length() + 1)
       << std::fixed << std::setprecision(1) << hostLatencyStd << "," << std::setw(headerFields[11].length() + 1)
       << std::fixed << std::setprecision(1) << hostBandwidth << "," << std::setw(headerFields[12].length() + 1)
       << numErrorsStr;
}

size_t verifyData(const std::vector<uint32_t>& hostSrcBuffer, void** dstBufs, size_t num_dsts, size_t transferSize)
{
   // ======================
   // 6. Results Verification
   // ======================
   size_t numErrors = 0;
   if (transferSize == 0)
      return numErrors;

   size_t num_elements = transferSize / sizeof(uint32_t);
   std::vector<uint32_t> hostDstData(num_elements, 0);

   for (size_t i = 0; i < num_dsts; i++)
   {
      // std::cout<<"Verify destination "<< i <<std::endl;
      // Copy destination buffer from device to host
      CHECK_HIP_ERROR(hipMemcpy(hostDstData.data(), dstBufs[i], transferSize, hipMemcpyDeviceToHost));
      if (memcmp(hostSrcBuffer.data(), hostDstData.data(), transferSize) == 0)
      {
         // std::cout << "Copy to destination " << i << " successful" << std::endl;
      }
      else
      {
         std::cout << "Copy to destination " << i << " failed" << std::endl;
         for (size_t i = 0; i < num_elements; i++)
         {
            if (hostDstData[i] != hostSrcBuffer[i])
            {
               numErrors++;
               if (numErrors < 100)
               {
                  std::cout << "Mismatch at index " << i << " : Expected 0x" << std::hex << hostSrcBuffer[i]
                            << ", Got 0x" << hostDstData[i] << std::endl;
               }
            }
         }
      }
   }
   return numErrors;
}
