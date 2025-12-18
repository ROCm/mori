#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "anvil_device.hpp"

__global__ void packet_rate_kernel(void* srcBuf, void* dstBuf, size_t copySize, size_t numCopyCommands,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64 expectedSignal, long long int* start_clock_count,
                                   long long int* end_clock_count);
