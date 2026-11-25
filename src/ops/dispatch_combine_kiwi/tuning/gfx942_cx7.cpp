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

// MI300X with ConnectX-7 NICs.

#include "ep_tuning.hpp"

namespace mori::kiwi::ep {

namespace {

// {world_size, max_num_tokens, num_blocks, rcomp_backend}
constexpr runtime_tuning_rule_t kRules[] = {
    {8, 1, 12, rcomp_backend_t::RDMA_ONLY},  {8, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {8, 4, 32, rcomp_backend_t::RDMA_ONLY},  {8, 8, 32, rcomp_backend_t::HYBRID},
    {8, 16, 32, rcomp_backend_t::HYBRID},    {8, 32, 32, rcomp_backend_t::HYBRID},
    {8, 64, 64, rcomp_backend_t::HYBRID},    {16, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {16, 2, 20, rcomp_backend_t::RDMA_ONLY}, {16, 4, 32, rcomp_backend_t::HYBRID},
    {16, 8, 32, rcomp_backend_t::HYBRID},    {16, 16, 32, rcomp_backend_t::HYBRID},
    {16, 32, 32, rcomp_backend_t::HYBRID},   {16, 64, 64, rcomp_backend_t::HYBRID},
    {32, 1, 12, rcomp_backend_t::RDMA_ONLY}, {32, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {32, 4, 32, rcomp_backend_t::HYBRID},    {32, 8, 32, rcomp_backend_t::HYBRID},
    {32, 16, 32, rcomp_backend_t::HYBRID},   {32, 32, 32, rcomp_backend_t::HYBRID},
    {32, 64, 64, rcomp_backend_t::HYBRID},   {64, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {64, 2, 20, rcomp_backend_t::RDMA_ONLY}, {64, 4, 32, rcomp_backend_t::HYBRID},
    {64, 8, 32, rcomp_backend_t::HYBRID},    {64, 16, 32, rcomp_backend_t::HYBRID},
    {64, 32, 32, rcomp_backend_t::HYBRID},   {64, 64, 64, rcomp_backend_t::HYBRID},
};

}  // namespace

extern const runtime_tuning_table_t kGfx942Cx7Tuning = {
    "gfx942",
    "cx7",
    kRules,
    sizeof(kRules) / sizeof(kRules[0]),
};

}  // namespace mori::kiwi::ep
