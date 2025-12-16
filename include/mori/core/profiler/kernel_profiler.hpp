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

#include <hip/hip_runtime.h>

namespace mori {
namespace core {

namespace profiler {

// Event Type Constants
enum class EventType : uint8_t { BEGIN = 0, END = 1, INSTANT = 2 };

template <typename SlotEnum, int MaxEventsPerWarp>
struct TraceProfiler {
  int64_t* warp_buffer;
  int lane_id;
  int warp_id;
  unsigned int offset;

  __device__ TraceProfiler(int64_t* warp_base_ptr, int lid, int wid)
      : warp_buffer(warp_base_ptr), lane_id(lid), warp_id(wid), offset(0) {}

  __device__ inline void log(SlotEnum slot, EventType type) {
    // Only lane 0 of each warp writes trace events
    if (lane_id == 0) {
      int64_t ts = clock64();
      // Meta encoding: [warpId:16][slot:14][type:2]
      // Bits 0-1:   EventType
      // Bits 2-15:  SlotEnum
      // Bits 16-31: WarpId
      int64_t meta = ((int64_t)warp_id << 16) | ((int64_t)slot << 2) | (int)type;

      // Circular buffer: overwrite oldest events if full
      int idx = offset % (MaxEventsPerWarp * 2);

      warp_buffer[idx] = ts;
      warp_buffer[idx + 1] = meta;
      offset += 2;
    }
  }
};

template <typename ProfilerType, typename SlotEnum>
struct ProfilerTraceScope {
  ProfilerType& profiler;
  SlotEnum slot;

  __device__ ProfilerTraceScope(ProfilerType& prof, SlotEnum target_slot)
      : profiler(prof), slot(target_slot) {
    profiler.log(slot, EventType::BEGIN);
  }

  __device__ ~ProfilerTraceScope() { profiler.log(slot, EventType::END); }
};

#define MORI_TRACE_SCOPE(profiler, slot) \
  mori::core::profiler::ProfilerTraceScope __trace_##__LINE__(profiler, slot)

}  // namespace profiler
}  // namespace core
}  // namespace mori
