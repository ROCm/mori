#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "anvil_device.hpp"
#include "timestamp_handle.hpp"

template <typename T, bool TIMESTAMPING_EN>
__global__ void multiQueueSDMATransfer(size_t iteration_id, void* srcBuf, void* dstBuf, size_t copySize,
                                       size_t numCopyCommands,
                                       anvil::SdmaQueueDeviceHandle* deviceHandle, // use parent type here
                                       HSAuint64* signals, HSAuint64 expectedSignal, long long int* start_clock_count,
                                       long long int* end_clock_count, TimeStampBreakdown* timestamp_breakdown);
