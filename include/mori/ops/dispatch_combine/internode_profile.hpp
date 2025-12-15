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

// Format: X(EnumName, PyBindName)
// The PyBindName is used for automatic Python binding and visualization.

#define INTERNODE_V1_SLOTS(X)              \
  /* Timestamps */                         \
  X(Start, "start")                        \
  X(End, "end")                            \
  X(BeforeLoop, "before_loop")             \
  X(AfterLoop, "after_loop")               \
  X(BeforeWait, "before_wait")             \
  /* Counters */                           \
  X(InnerLoopCycles, "inner_loop_cycles")  \
  X(InnerLoopCount, "inner_loop_count")    \
  X(AtomicCycles, "atomic_cycles")         \
  X(AtomicCount, "atomic_count")           \
  X(ShmemPutCycles, "shmem_put_cycles")    \
  X(ShmemPutCount, "shmem_put_count")      \
  X(IterCount, "iter_count")               \
  /* Detailed Metrics */                   \
  X(PointerSetupDuration, "ptr_setup_dur") \
  X(TokenAccumDuration, "tok_accum_dur")   \
  X(WeightAccumDuration, "wgt_accum_dur")  \
  X(ProcessedTokenCount, "processed_tok_cnt")

namespace mori {
namespace moe {
namespace v1 {

enum class InterNodeSlot : int {
#define X(name, str) name,
  INTERNODE_V1_SLOTS(X)
#undef X
      MAX_SLOTS
};

}  // namespace v1
}  // namespace moe
}  // namespace mori
