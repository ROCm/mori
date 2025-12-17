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
  using slot_type = SlotEnum;

  int64_t* warp_buffer;
  int lane_id;
  int warp_id;
  unsigned int offset;

  __device__ TraceProfiler(int64_t* warp_base_ptr, int lid, int wid)
      : warp_buffer(warp_base_ptr), lane_id(lid), warp_id(wid), offset(0) {}

  __device__ inline void log(SlotEnum slot, EventType type) {
    log_with_time(slot, type, clock64());
  }

  __device__ inline void log_with_time(SlotEnum slot, EventType type, int64_t ts) {
    if (lane_id == 0) {
      int64_t meta = ((int64_t)warp_id << 16) | ((int64_t)slot << 2) | (int)type;
      int idx = offset % (MaxEventsPerWarp * 2);
      warp_buffer[idx] = ts;
      warp_buffer[idx + 1] = meta;
      offset += 2;
    }
  }
};

#ifndef PROFILER_MASK
#define PROFILER_MASK 0xFFFFFFFF
#endif

template <bool Enabled, typename ProfilerType, typename SlotEnum>
struct ProfilerSpan {
  ProfilerType& profiler;
  SlotEnum slot;

  __device__ ProfilerSpan(ProfilerType& prof, SlotEnum s) : profiler(prof), slot(s) {
    profiler.log(slot, EventType::BEGIN);
  }

  __device__ ~ProfilerSpan() { profiler.log(slot, EventType::END); }
};

template <typename ProfilerType, typename SlotEnum>
struct ProfilerSpan<false, ProfilerType, SlotEnum> {
  __device__ ProfilerSpan(ProfilerType&, SlotEnum) {}
  __device__ ~ProfilerSpan() {}
};

template <bool Enabled, typename ProfilerType, typename SlotEnum>
struct ProfilerSequential {
  ProfilerType& profiler;
  SlotEnum current_slot;
  bool has_current;

  __device__ ProfilerSequential(ProfilerType& prof)
      : profiler(prof), current_slot(), has_current(false) {}

  __device__ inline void next(SlotEnum slot) {
    if (has_current) {
      if (profiler.lane_id == 0) {
        int64_t ts = clock64();
        profiler.log_with_time(current_slot, EventType::END, ts);
        profiler.log_with_time(slot, EventType::BEGIN, ts);
      }
    } else {
      profiler.log(slot, EventType::BEGIN);
      has_current = true;
    }
    current_slot = slot;
  }

  __device__ ~ProfilerSequential() {
    if (has_current) {
      profiler.log(current_slot, EventType::END);
    }
  }
};

template <typename ProfilerType, typename SlotEnum>
struct ProfilerSequential<false, ProfilerType, SlotEnum> {
  __device__ ProfilerSequential(ProfilerType&) {}
  __device__ inline void next(SlotEnum) {}
  __device__ ~ProfilerSequential() {}
};

#ifndef ENABLE_PROFILER
#define MORI_INIT_PROFILER(name, type, ...) ((void)0)
#define MORI_TRACE_SPAN(profiler, slot, tag) ((void)0)
#define MORI_TRACE_SEQ(name, profiler, tag) ((void)0)
#define MORI_TRACE_NEXT(name, slot) ((void)0)
#define MORI_TRACE_INSTANT(profiler, slot, tag) ((void)0)
#else
#define MORI_INIT_PROFILER(name, type, ...) type name(__VA_ARGS__)

#define MORI_TRACE_SPAN(profiler, slot, tag)                                                      \
  mori::core::profiler::ProfilerSpan<((tag) & PROFILER_MASK), decltype(profiler), decltype(slot)> \
  __span_##__LINE__(profiler, slot)

#define MORI_TRACE_SEQ(name, profiler, tag)                                             \
  mori::core::profiler::ProfilerSequential<((tag) & PROFILER_MASK), decltype(profiler), \
                                           typename decltype(profiler)::slot_type>      \
  name(profiler)

#define MORI_TRACE_NEXT(name, slot) name.next(slot)

#define MORI_TRACE_INSTANT(profiler, slot, tag)                     \
  do {                                                              \
    if ((tag) & PROFILER_MASK) {                                    \
      profiler.log(slot, mori::core::profiler::EventType::INSTANT); \
    }                                                               \
  } while (0)
#endif

}  // namespace profiler
}  // namespace core
}  // namespace mori
