// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/application/transport/sdma/anvil_device.hpp"

__global__ void packet_rate_kernel(void* srcBuf, void* dstBuf, size_t copySize,
                                   size_t numCopyCommands,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64 expectedSignal, long long int* start_clock_count,
                                   long long int* end_clock_count);
