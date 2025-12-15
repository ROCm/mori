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

#define MAX_TRACE_EVENTS_PER_WARP 4096
#define TRACE_EVENT_SIZE_INT64 2     // Timestamp + Metadata
#define PROFILER_WARPS_PER_RANK 512  // Max warps per rank (64 blocks * 8 warps/block)

// Total buffer size per rank (in int64_t)
// 4096 * 2 * 512 = 4,194,304 int64s = 32MB per rank.
#define MAX_DEBUG_TIME_SLOTS \
  (MAX_TRACE_EVENTS_PER_WARP * TRACE_EVENT_SIZE_INT64 * PROFILER_WARPS_PER_RANK)

// Per-warp buffer stride (in int64_t elements)
// 4096 * 2 = 8192 int64s per warp
#define MAX_DEBUG_TIMESTAMP_PER_WARP (MAX_TRACE_EVENTS_PER_WARP * TRACE_EVENT_SIZE_INT64)
