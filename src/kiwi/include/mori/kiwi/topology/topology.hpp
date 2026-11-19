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

// CPUs for the proxy threads serving the current HIP device: its PCI-local
// CPUs within the calling thread's affinity mask. Empty when they do not
// intersect or cannot be read; proxies then keep the mask they inherit. Call
// on the thread that selected the device, since hipSetDevice() is per thread.
// KIWI_PROXY_AFFINITY_DEBUG=1 prints the set.
std::vector<int> proxy_thread_cpu_set();

// Restrict the calling thread to cpus and leave placement within them to the
// scheduler. Pinning each proxy to a CPU chosen independently per rank lets
// ranks collide on one CPU; a shared set cannot. No-op for an empty set.
void bind_current_thread_to_cpus(const std::vector<int>& cpus);

}  // namespace topology
}  // namespace mori::kiwi::p2p
