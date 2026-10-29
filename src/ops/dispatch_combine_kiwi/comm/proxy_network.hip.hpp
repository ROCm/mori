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

#include <cstddef>
#include <cstdint>

#include "comm/rdma.hpp"

namespace mori::kiwi::ep::internal {

void proxy_network_progress();

// Every raw WRITE is signaled; record it so teardown drains its completion.
void proxy_network_record_posted_write();

// Posts a signaled RDMA WRITE on this proxy thread's device, progressing the
// CQ while the target QP's send queue is full.
void proxy_network_put(int rank, const void* local, size_t size, const rdma::mr_t& mr,
                       uint64_t remote_disp, const rdma::rmr_t& rmr);

// Inline variant: `local` may be reused as soon as this returns.
void proxy_network_put_inline(int rank, const void* local, size_t size, uint64_t remote_disp,
                              const rdma::rmr_t& rmr);

}  // namespace mori::kiwi::ep::internal
