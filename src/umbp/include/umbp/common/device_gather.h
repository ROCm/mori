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

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mori::umbp {

struct DeviceGatherFragment {
  const void* src;
  void* dst;
  size_t bytes;
};

// Copies `count` independent fragments between a HostTierRegistration-covered
// region and one GPU in a single kernel launch, in either direction.
//
// This exists because the DRAM tier's copy path is scatter-gather by nature:
// reading one layer out of page-granular objects means dozens of small strided
// fragments per call, and hipMemcpy has no scatter-gather form. Submitting them
// one at a time costs ~5.4 us each with 8 ranks active; hipMemcpy2DAsync can
// express the uniform-stride case but collapses to ~194 us per fragment
// whenever the source pages are cold, which for a live tier is always.
// A kernel is insensitive to both: measured ~0.57 us per fragment, and flat
// across fragment sizes from 8 KiB to 221 KiB.
//
// Only worth it below roughly 128 KiB per fragment. Above that the copy engine
// wins, because one large hipMemcpyAsync already amortizes its submission.
//
// The host side must be inside a HostTierRegistration-covered range: a kernel
// cannot dereference plain mmap memory at all, it faults. Returns false before
// enqueueing any work when the launch could not be issued, so the caller can
// fall back to hipMemcpy.
bool LaunchDeviceGather(const DeviceGatherFragment* fragments, size_t count, int device_id,
                        hipStream_t stream);

// Whether the gather kernel path is compiled in and not disabled via
// UMBP_DRAM_GATHER_KERNEL=0.
bool DeviceGatherEnabled();

// Kernels launched since process start. Lets a test assert which path a batch
// actually took, rather than passing whether or not the kernel ran.
uint64_t DeviceGatherLaunchCount();

}  // namespace mori::umbp
