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
#pragma once

// Stream-ordered barrier/put_signal/wait_signal on torch's signal pad; kernels in symm_signal.hip.
// Slot layout and CAS protocol match torch's backends; timeout_ms == 0 waits forever.

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mori {
namespace allocator {

// Signal every peer on `channel`, then wait for all of them.
hipError_t LaunchSignalPadBarrier(uint32_t** pads_dev, int channel, int rank, int world_size,
                                  size_t timeout_ms, hipStream_t stream);

// Set this rank's slot in dst_rank's pad.
hipError_t LaunchSignalPadPut(uint32_t** pads_dev, int dst_rank, int channel, int rank,
                              int world_size, size_t timeout_ms, hipStream_t stream);

// Consume src_rank's slot in this rank's pad.
hipError_t LaunchSignalPadWait(uint32_t** pads_dev, int src_rank, int channel, int rank,
                               int world_size, size_t timeout_ms, hipStream_t stream);

}  // namespace allocator
}  // namespace mori
