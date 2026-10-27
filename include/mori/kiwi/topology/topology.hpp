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
#include <string>
#include <vector>

namespace mori::kiwi::p2p {
namespace topology {

// Return one selected CPU per proxy thread, or -1 when no usable CPU can be
// selected.  CPUs are chosen from the current HIP device's KFD-local CPU slice
// when it intersects the process affinity mask; otherwise they are partitioned
// across the currently allowed mask.
std::vector<int> choose_proxy_thread_cpus(int num_proxy_threads);

void pin_proxy_thread_to_cpu(int thread_id, int num_proxy_threads, int selected_cpu);

// Return the nearest active HCA/NIC name for a HIP GPU id.  The mapping is
// computed across all visible GPUs and balances exact ties so ranks do not all
// bind to the same NIC when several equally-near HCAs are available.
std::string get_hca_name_for_gpu(int gpu_id);
std::string get_hca_name_for_current_gpu();
std::string get_requested_or_auto_hca_name_for_current_gpu();

}  // namespace topology
}  // namespace mori::kiwi::p2p
