#pragma once

#include <stdio.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

constexpr int WARP_SIZE = 64;

// stdout header fields
constexpr std::array<std::string_view, 13> headerFields = {
    "Src",   // Src GPU
    "#Dest", // N destinations
    "Grid",
    "Block",
    "CopySize",         // Size of each individual SDMA packet
    "#Copies",          // Number of packets used for the transfer
    "Time [us]",        // GPU-side measured transfer time (avg)
    "Time (std)",       // Std deviation of device-side latency
    "Bandwidth [GB/s]", // Average Device bandwidth for the transfer
    "Time [us] (host)", // Host-side timing (avg)
    "Time (std)",       // Std deviation of host-side latency
    "Bandwidth [GB/s]", // Average Host bandwidth for the transfer
    "#Wrong",
};

void printHeader(std::ostream& out, const std::array<std::string_view, 13>& headers);

void printRowOfResults(std::ostream& out, int srcGpuId, size_t num_dsts, int numBlocks, int numThreadsPerBlock,
                       size_t totalTransferSize, size_t copySize, size_t numCopies, double deviceLatency,
                       double deviceLatencyStd, double deviceBandwidth, double hostLatency, double hostLatencyStd,
                       double hostBandwidth, std::optional<size_t> numErrors);

size_t verifyData(const std::vector<uint32_t>& hostSrcBuffer, void** dstBufs, size_t num_dsts, size_t transferSize);
