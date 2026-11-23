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
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"

namespace mori::umbp {
namespace {

constexpr unsigned kThreadsPerBlock = 256;

// Enough parallelism to keep the link busy on the largest fragment without
// launching blocks that immediately exit on the smallest. Fragments below one
// block's worth of work still get one block each.
constexpr unsigned kMaxBlocksPerFragment = 8;

constexpr size_t kVectorBytes = 16;

// A block copies its slice of one fragment. Fragments are independent, so the
// grid is flat: blockIdx.x / blocks_per_fragment picks the fragment and the
// remainder picks the slice.
__global__ void GatherFragmentsKernel(const DeviceGatherFragment* __restrict__ fragments,
                                      unsigned blocks_per_fragment) {
  const DeviceGatherFragment fragment = fragments[blockIdx.x / blocks_per_fragment];
  const char* __restrict__ src = static_cast<const char*>(fragment.src);
  char* __restrict__ dst = static_cast<char*>(fragment.dst);

  const size_t slice = blockIdx.x % blocks_per_fragment;
  const size_t index = slice * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blocks_per_fragment) * blockDim.x;

  // KV fragments are page-aligned in practice; the byte loop is correctness
  // insurance for pools whose element size makes them odd.
  const uintptr_t mask = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) |
                         static_cast<uintptr_t>(fragment.bytes);
  if ((mask & (kVectorBytes - 1)) == 0) {
    const uint4* __restrict__ src_vector = reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ dst_vector = reinterpret_cast<uint4*>(dst);
    const size_t count = fragment.bytes / kVectorBytes;
    for (size_t i = index; i < count; i += stride) dst_vector[i] = src_vector[i];
    return;
  }
  for (size_t i = index; i < fragment.bytes; i += stride) dst[i] = src[i];
}

// Per-stream descriptor staging.
//
// The caller holds `stream` exclusively from the upload to the synchronize
// (see LaunchDeviceGather), so a stream's buffer is never in use twice at once,
// and the count is bounded by the caller's streams rather than by however many
// threads have ever called in. Never freed, for the same reason the streams are
// never destroyed: releasing HIP resources from a static destructor races with
// runtime teardown.
struct DescriptorStaging {
  DeviceGatherFragment* host = nullptr;
  DeviceGatherFragment* device = nullptr;
  size_t capacity = 0;
};

DescriptorStaging* StagingFor(hipStream_t stream, int device_id, size_t count) {
  static std::mutex* staging_mutex = new std::mutex();
  static auto* staging_by_stream = new std::unordered_map<hipStream_t, DescriptorStaging>();

  DescriptorStaging* staging_entry = nullptr;
  {
    // References into an unordered_map stay valid across rehashing.
    std::lock_guard<std::mutex> lock(*staging_mutex);
    staging_entry = &(*staging_by_stream)[stream];
  }
  DescriptorStaging& staging = *staging_entry;
  if (staging.capacity >= count) return &staging;

  // Grow with headroom so a slowly increasing batch size does not reallocate
  // on every call.
  const size_t capacity = std::max<size_t>(count * 2, 256);
  const size_t bytes = capacity * sizeof(DeviceGatherFragment);
  DeviceGatherFragment* host = nullptr;
  DeviceGatherFragment* device = nullptr;
  hipError_t status = hipHostMalloc(reinterpret_cast<void**>(&host), bytes, hipHostMallocDefault);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] hipHostMalloc for {} gather descriptors failed on device {}: {}",
                    capacity, device_id, hipGetErrorString(status));
    (void)hipGetLastError();
    return nullptr;
  }
  status = hipMalloc(reinterpret_cast<void**>(&device), bytes);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] hipMalloc for {} gather descriptors failed on device {}: {}",
                    capacity, device_id, hipGetErrorString(status));
    (void)hipGetLastError();
    (void)hipHostFree(host);
    return nullptr;
  }

  if (staging.host != nullptr) (void)hipHostFree(staging.host);
  if (staging.device != nullptr) (void)hipFree(staging.device);
  staging.host = host;
  staging.device = device;
  staging.capacity = capacity;
  return &staging;
}

std::atomic<uint64_t>& LaunchCounter() {
  static std::atomic<uint64_t> launches{0};
  return launches;
}

}  // namespace

uint64_t DeviceGatherLaunchCount() { return LaunchCounter().load(std::memory_order_relaxed); }

bool DeviceGatherEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("UMBP_DRAM_GATHER_KERNEL");
    if (value == nullptr) return true;
    const std::string text(value);
    return text != "0" && text != "off" && text != "false";
  }();
  return enabled;
}

bool LaunchDeviceGather(const DeviceGatherFragment* fragments, size_t count, int device_id,
                        hipStream_t stream) {
  if (count == 0) return true;

  size_t max_bytes = 0;
  for (size_t i = 0; i < count; ++i) max_bytes = std::max(max_bytes, fragments[i].bytes);
  if (max_bytes == 0) return false;

  const size_t chunks = (max_bytes + kVectorBytes - 1) / kVectorBytes;
  const unsigned blocks_per_fragment = static_cast<unsigned>(std::clamp<size_t>(
      (chunks + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, kMaxBlocksPerFragment));
  if (count > std::numeric_limits<unsigned>::max() / blocks_per_fragment) return false;

  DescriptorStaging* staging = StagingFor(stream, device_id, count);
  if (staging == nullptr) return false;

  std::memcpy(staging->host, fragments, count * sizeof(DeviceGatherFragment));
  hipError_t status =
      hipMemcpyAsync(staging->device, staging->host, count * sizeof(DeviceGatherFragment),
                     hipMemcpyHostToDevice, stream);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] gather descriptor upload failed on device {}: {}", device_id,
                    hipGetErrorString(status));
    (void)hipGetLastError();
    return false;
  }

  hipLaunchKernelGGL(GatherFragmentsKernel,
                     dim3(static_cast<unsigned>(count) * blocks_per_fragment),
                     dim3(kThreadsPerBlock), 0, stream, staging->device, blocks_per_fragment);
  status = hipGetLastError();
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] gather kernel launch failed on device {} (fragments={}): {}",
                    device_id, count, hipGetErrorString(status));
    return false;
  }
  LaunchCounter().fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool PrewarmDeviceGather(int device_id, hipStream_t stream, size_t capacity) {
  DescriptorStaging* staging = StagingFor(stream, device_id, capacity);
  if (staging == nullptr) return false;

  const size_t bytes = staging->capacity * sizeof(DeviceGatherFragment);
  std::memset(staging->host, 0, bytes);
  hipError_t status =
      hipMemcpyAsync(staging->device, staging->host, bytes, hipMemcpyHostToDevice, stream);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] prewarm descriptor upload failed on device {}: {}", device_id,
                    hipGetErrorString(status));
    (void)hipGetLastError();
    return false;
  }
  // One zero-byte fragment: loads the kernel on this device, copies nothing.
  hipLaunchKernelGGL(GatherFragmentsKernel, dim3(1), dim3(kThreadsPerBlock), 0, stream,
                     staging->device, 1u);
  status = hipGetLastError();
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DeviceGather] prewarm kernel launch failed on device {}: {}", device_id,
                    hipGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace mori::umbp
