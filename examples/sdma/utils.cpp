#include "utils.hpp"

std::string prettyPrintSize(size_t bytes)
{
   constexpr const char FILE_SIZE_UNITS[5][3]{"B", "KB", "MB", "GB", "TB"};

   std::stringstream ss;
   size_t index = 0;
   while (bytes >= 1024)
   {
      bytes /= 1024;
      index++;
   }

   ss << bytes << FILE_SIZE_UNITS[index];
   return ss.str();
}

// Reporter class to generate CSV result files
Reporter::Reporter(std::string fileName)
{
   if (!fileName.empty())
   {
      _out = std::ofstream(fileName, std::ios_base::out);
      _outputValid = true;
   }
}

void Reporter::setParameters(const int srcGpu, const size_t numDest, const size_t numQueues, const int gridDim,
                             const int blockDim, const size_t numCopies)
{
   if (!_outputValid)
      return;

   _srcGpu = srcGpu;
   _numDest = numDest;
   _numQueues = numQueues;
   _gridDim = gridDim;
   _blockDim = blockDim;
   _numCopies = numCopies;
}

void Reporter::addResult(size_t totalTransferSize, size_t copySize, double deviceLatency, double deviceLatencyStd,
                         double deviceBandwidth, double hostLatency, double hostLatencyStd, double hostBandwidth)
{
   if (!_outputValid)
      return;

   std::vector<std::pair<std::string, std::string>> outputValuesKeys;
   addResultCommon(outputValuesKeys, totalTransferSize, copySize, deviceLatency, deviceLatencyStd, deviceBandwidth,
                   hostLatency, hostLatencyStd, hostBandwidth);
   _outputData.push_back(outputValuesKeys);
}

void Reporter::addResult(size_t totalTransferSize, size_t copySize, double deviceLatency, double deviceLatencyStd,
                         double deviceBandwidth, double hostLatency, double hostLatencyStd, double hostBandwidth,
                         double copyReserveLat, double copyReserveStd, double copyPlaceLat, double copyPlaceStd,
                         double copySubmitLat, double copySubmitStd, double fenceReserveLat, double fenceReserveStd,
                         double fencePlaceLat, double fencePlaceStd, double fenceSubmitLat, double fenceSubmitStd,
                         double sdmaTransferLat, double sdmaTransferStd)
{
   if (!_outputValid)
      return;

   std::vector<std::pair<std::string, std::string>> outputValuesKeys;
   addResultCommon(outputValuesKeys, totalTransferSize, copySize, deviceLatency, deviceLatencyStd, deviceBandwidth,
                   hostLatency, hostLatencyStd, hostBandwidth);

   outputValuesKeys.push_back(
       makeValueKeyPair(copyReserveLat, "Reserve Space Latency (Mean)")); // Time to reserve space in the queue (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(copyReserveStd, "Reserve Space Latency (Std)")); // Standard deviation of reserve latency
   outputValuesKeys.push_back(
       makeValueKeyPair(copyPlaceLat, "Entails Packet Latency (Mean)")); // Time to place packet in queue (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(copyPlaceStd, "Entails Packet Latency (Std)")); // Standard deviation of place latency
   outputValuesKeys.push_back(
       makeValueKeyPair(copySubmitLat, "Submit Packet Latency (Mean)")); // Time to notify SDMA engine (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(copySubmitStd, "Submit Packet Latency (Std)")); // Standard deviation of submit latency
   outputValuesKeys.push_back(
       makeValueKeyPair(fenceReserveLat,
                        "Reserve Space Fence Latency (Mean)")); // Time to reserve space in the queue (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(fenceReserveStd,
                        "Reserve Space Fence Latency (Std)")); // Standard deviation of reserve latency
   outputValuesKeys.push_back(
       makeValueKeyPair(fencePlaceLat,
                        "Entails Fence Packet Latency (Mean)")); // Time to place packet in queue (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(fencePlaceStd, "Entails Fence Packet Latency (Std)")); // Standard deviation of place latency
   outputValuesKeys.push_back(
       makeValueKeyPair(fenceSubmitLat, "Submit Fence Packet Latency (Mean)")); // Time to notify SDMA engine (avg)
   outputValuesKeys.push_back(
       makeValueKeyPair(fenceSubmitStd,
                        "Submit Fence Packet Latency (Std)")); // Standard deviation of submit latency
   outputValuesKeys.push_back(
       makeValueKeyPair(sdmaTransferLat, "SDMA Transfer & Sync (Mean)")); // SDMA Engine Transfer time in us (Mean)
   outputValuesKeys.push_back(
       makeValueKeyPair(sdmaTransferStd, "SDMA Transfer & Sync (Std)")); // SDMA Engine Transfer time in us (Std)

   _outputData.push_back(outputValuesKeys);
}

void Reporter::writeFile()
{
   if (!_outputValid)
      return;

   // For CSV print header based on keys
   for (auto& valueKey : _outputData[0])
   {
      _out << valueKey.second;
      if (&valueKey != &_outputData[0].back())
      {
         _out << ",";
      }
   }
   _out << std::endl;

   // Iterate through all rows and print values
   for (auto& row : _outputData)
   {
      for (auto& valueKey : row)
      {
         _out << valueKey.first;
         if (&valueKey != &row.back())
         {
            _out << ",";
         }
      }
      _out << std::endl;
   }
}

void Reporter::addResultCommon(std::vector<std::pair<std::string, std::string>>& outputValuesKeys,
                               size_t totalTransferSize, size_t copySize, double deviceLatency, double deviceLatencyStd,
                               double deviceBandwidth, double hostLatency, double hostLatencyStd, double hostBandwidth)
{
   outputValuesKeys.push_back(makeValueKeyPair(_srcGpu, "Src"));
   outputValuesKeys.push_back(makeValueKeyPair(_numDest, "#Destinations"));
   outputValuesKeys.push_back(makeValueKeyPair(_numQueues, "#Queues"));
   outputValuesKeys.push_back(makeValueKeyPair(_gridDim, "GridDim"));
   outputValuesKeys.push_back(makeValueKeyPair(_blockDim, "BlockDim"));
   outputValuesKeys.push_back(makeValueKeyPair(totalTransferSize, "Total Transfer Size [B]"));
   outputValuesKeys.push_back(makeValueKeyPair(copySize, "Copy Size [B]"));
   outputValuesKeys.push_back(makeValueKeyPair(_numCopies, "#Copies"));
   outputValuesKeys.push_back(makeValueKeyPair(deviceLatency, "Device Latency [us] (Mean)"));
   outputValuesKeys.push_back(makeValueKeyPair(deviceLatencyStd, "Device Latency (Std)"));
   outputValuesKeys.push_back(makeValueKeyPair(deviceBandwidth, "Bandwidth [GB/s] (Device)"));
   outputValuesKeys.push_back(makeValueKeyPair(hostLatency, "Host Latency [us] (Mean)"));
   outputValuesKeys.push_back(makeValueKeyPair(hostLatencyStd, "Host Latency (Std)"));
   outputValuesKeys.push_back(makeValueKeyPair(hostBandwidth, "Bandwidth [GB/s] (Host)"));
}