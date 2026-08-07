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

// Which implementation of intra-node dispatch and combine a launch symbol gets.
//
// This is the only file that knows there is more than one. intranode.hpp holds the portable
// bodies and never mentions an architecture; intranode_1250x.hpp holds the TDM bodies and never
// tests for one. The launch symbols in ep_intranode.hip are registered with the WRAP_*_ENTRY
// macros, which forward to *_entry below instead of straight to *_body -- that indirection is the
// whole reason this file exists, and it is what keeps the choice out of the two implementations.

#include "src/ops/dispatch_combine/intranode.hpp"

// Not standalone: written against MAX_GPUS_PER_NODE and the barrier in intranode.hpp, hence the
// include order here.
#if defined(__gfx1250__) || defined(__gfx1251__)
#include "src/ops/dispatch_combine/intranode_1250x.hpp"
#endif

namespace mori {
namespace moe {

template <typename T, bool EnableStdMoE = false>
__device__ __forceinline__ void EpDispatchIntraNodeKernel_entry(EpDispatchCombineArgs<T> args) {
#if defined(__gfx1250__) || defined(__gfx1251__)
  EpDispatchIntraNodeKernel_1250x_body<T, EnableStdMoE>(args);
#else
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
#endif
}

template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, int Vec8AccumNum = 8, bool UseFp4Combine = false>
__device__ __forceinline__ void EpCombineIntraNodeKernel_entry(EpDispatchCombineArgs<T> args) {
#if defined(__gfx1250__) || defined(__gfx1251__)
  EpCombineIntraNodeKernel_1250x_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast,
                                      UseFp8BlockwiseQuant, UseWeights, Vec8Top8BlockElems,
                                      Vec8AccumNum, UseFp4Combine>(args);
#else
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems, Vec8AccumNum, UseFp4Combine>(args);
#endif
}

}  // namespace moe
}  // namespace mori
