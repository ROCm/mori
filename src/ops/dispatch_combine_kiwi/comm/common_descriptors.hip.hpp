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
#include <mori/kiwi/queue/queue.hpp>

namespace mori::kiwi::ep {
class ep_context_t;

namespace internal {

// Shared fanout cap for EP transport multicast descriptors.
inline constexpr int EP_COMM_MAX_DESTS = 8;

struct ep_cmd_desc_all2all_t {
  ep_context_t* ep_context;
  uintptr_t send_buf;
  uintptr_t remote_disp;
};

static_assert(sizeof(ep_cmd_desc_all2all_t) <= kiwi::queue::kMaxMessageBytes,
              "ep_cmd_desc_all2all_t must fit in one invoke message");

struct ep_cmd_desc_active_queues_t {
  int active_count;
};

static_assert(sizeof(ep_cmd_desc_active_queues_t) <= kiwi::queue::kMaxMessageBytes,
              "ep_cmd_desc_active_queues_t must fit in one invoke message");

}  // namespace internal
}  // namespace mori::kiwi::ep
