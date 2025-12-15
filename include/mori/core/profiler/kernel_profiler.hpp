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

template <typename SlotEnum, int MaxSlots>
struct KernelProfiler {
  int64_t* debug_buf;
  int base_offset;
  int lane_id;
  int64_t accumulators[MaxSlots];
  int flush_start_slot;
  int flush_end_slot;

  // Constructor with explicit base offset calculation (Most Flexible)
  __device__ KernelProfiler(int64_t* global_buf, int base_offset_val, int lid, int flush_start = 0,
                            int flush_end = MaxSlots)
      : debug_buf(global_buf),
        base_offset(base_offset_val),
        lane_id(lid),
        flush_start_slot(flush_start),
        flush_end_slot(flush_end) {
#pragma unroll
    for (int i = 0; i < MaxSlots; i++) {
      accumulators[i] = 0;
    }
  }

  __device__ inline void mark(SlotEnum slot, int64_t val) {
    if (lane_id == 0) {
      debug_buf[base_offset + (int)slot] = val;
    }
  }

  __device__ inline void mark(SlotEnum slot) {
    if (lane_id == 0) {
      debug_buf[base_offset + (int)slot] = clock64();
    }
  }

  __device__ inline void accum(SlotEnum slot, int64_t val) { accumulators[(int)slot] += val; }

  __device__ inline void accum_duration(SlotEnum slot, int64_t start, int64_t end) {
    accumulators[(int)slot] += (end - start);
  }

  __device__ inline void accum_duration_if_pos(SlotEnum slot, int64_t start, int64_t end) {
    int64_t dur = end - start;
    if (dur > 0) {
      accumulators[(int)slot] += dur;
    }
  }

  __device__ inline void increment(SlotEnum slot, int64_t val = 1) {
    accumulators[(int)slot] += val;
  }

  __device__ inline void flush() {
    if (lane_id == 0) {
      for (int i = flush_start_slot; i < flush_end_slot; i++) {
        debug_buf[base_offset + i] = accumulators[i];
      }
    }
  }
};

template <typename ProfilerType, typename SlotEnum>
struct ProfilerScopedTimer {
  int64_t start_time;
  ProfilerType& profiler;
  SlotEnum slot;
  bool conditional;

  __device__ ProfilerScopedTimer(ProfilerType& prof, SlotEnum target_slot, bool cond = false)
      : profiler(prof), slot(target_slot), conditional(cond) {
    start_time = clock64();
  }

  __device__ ~ProfilerScopedTimer() {
    int64_t dur = clock64() - start_time;
    if (!conditional || dur > 0) {
      profiler.accum(slot, dur);
    }
  }
};

#define MORI_PROFILE_SCOPE(profiler, slot) \
  mori::core::profiler::ProfilerScopedTimer __timer_##__LINE__(profiler, slot)

#define MORI_PROFILE_MARK_IF(profiler, slot, condition) \
  do {                                                  \
    if (condition) {                                    \
      profiler.mark(slot);                              \
    }                                                   \
  } while (0)

}  // namespace profiler
}  // namespace core
}  // namespace mori
