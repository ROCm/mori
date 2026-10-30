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

#include <cstdint>

#include "comm/common_descriptors.hip.hpp"

namespace mori::kiwi::ep {
class ep_context_t;

namespace internal {

struct post_put_cmd_desc_t {
  ep_context_t* ep_context;
  uintptr_t local_buffer;
  uint64_t size;
  uintptr_t remote_disp;
  uintptr_t readiness_flags_offset;
  int rank;
  int heap_id;
  int readiness_slot;
};

struct post_put_cmd_desc_multicast_t {
  ep_context_t* ep_context;
  uintptr_t local_buffer;
  uint64_t size;
  uintptr_t remote_disps[EP_COMM_MAX_DESTS];
  uintptr_t readiness_flags_offset;
  int heap_id;
  int n;
  int readiness_slots[EP_COMM_MAX_DESTS];
  int ranks[EP_COMM_MAX_DESTS];
};

static_assert(sizeof(post_put_cmd_desc_t) <= kiwi::queue::kMaxMessageBytes,
              "post_put_cmd_desc_t must fit in one invoke message");
static_assert(sizeof(post_put_cmd_desc_multicast_t) <= kiwi::queue::kMaxMessageBytes,
              "post_put_cmd_desc_multicast_t must fit in one invoke message");

}  // namespace internal

}  // namespace mori::kiwi::ep
