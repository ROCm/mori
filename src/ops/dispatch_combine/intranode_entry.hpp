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
//
// The choice is not the architecture alone. Combine also asks what the instantiation transports:
// a quantizing one gets the portable body even on gfx125x, because the TDM body has nothing to
// offer it (see the comment at the combine entry).

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
  // Only the unquantized combine takes the TDM body, and the TDM body is written for that case
  // alone -- it does not take the quantization parameters at all. Vec8Top8BlockElems /
  // Vec8AccumNum only tune the scalar dequant chain, so they are not passed on either.
  //
  // What this costs, measured as a two-tree ISA diff on gfx1250 (50 symbols, s1 = before):
  //   * every unquantized symbol is instruction-for-instruction identical -- those are the ones
  //     production runs here, and they keep the whole TDM body;
  //   * the PUSH-quantized ones (_nop2p_fp8_blockwise, _nop2p_fp4_blockwise, _nop2p_fp8cast) move
  //     by -40..-111 instructions, all of it the barrier, because they really did reach no TDM
  //     instruction: their send is a per-lane quantize into the peer slot and their reduce is the
  //     scalar dequant chain;
  //   * _bf16_p2p_fp8_blockwise grows 81401 -> 104013 and gives up its one tensor_load_to_lds.
  //     That one DID use the TDM tile fold -- PULL blockwise was the case the fold's scale
  //     prefetch was written for. It is a deliberate loss: blockwise combine is 367.6us at its
  //     best tuning against 169.2us for bf16 moving twice the bytes, so the fold was buying speed
  //     for a transport nobody should pick, at the price of the scale plumbing threaded through
  //     every loop of the TDM body.
  if constexpr (!UseFp8DirectCast && !UseFp8BlockwiseQuant && !UseFp4Combine) {
    EpCombineIntraNodeKernel_1250x_body<T, UseP2PRead, EnableStdMoE, UseWeights>(args);
  } else {
    EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast,
                                  UseFp8BlockwiseQuant, UseWeights, Vec8Top8BlockElems,
                                  Vec8AccumNum, UseFp4Combine>(args);
  }
#else
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems, Vec8AccumNum, UseFp4Combine>(args);
#endif
}

}  // namespace moe
}  // namespace mori
