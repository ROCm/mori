#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "anvil_device.hpp"

__global__ void multiQueueSDMATransferQueueMapWG(size_t iteration_id, void* srcBuf, void** dstBufs, size_t copy_size,
                                                 size_t numCopyCommands, int numOfDestinations, int numOfQueuesPerDestination, int numOfWGPerQueue,
                                                 anvil::SdmaQueueDeviceHandle** deviceHandle, HSAuint64* signals,
                                                 HSAuint64 expectedSignal, long long int* start_clock_count,
                                                 long long int* end_clock_count);
