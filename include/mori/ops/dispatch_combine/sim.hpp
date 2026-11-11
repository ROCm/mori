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

namespace mori {
namespace moe {
// GPU-free EP (MORI_EP_SIM_ENABLE=1): the EP ops are built and launched as on a GPU, eagerly or in
// CUDA graphs, on a simulated HIP runtime that executes no kernel (e.g. Sundial's virtual GPU,
// libvgpu). For the dispatch / combine / local expert count kernels the runtime calls mori's host
// handlers (sim.cpp), which draw each dispatch's routing from MORI_EP_SIM_ROUTING, exchange it
// through the runtime's all-to-all (which times the op) and write the outputs the kernels write.
namespace sim {

bool IsEnabled();
// A simulated runtime is loaded (it exports the symbols sim.cpp looks up).
bool Available();

// One op. Map entries: source = srcRank * srcStride + token; destination = destRank * destStride
// + slot, or world * destStride for a dropped pair.
struct OpSpec {
  int version;  // 1 = EpDispatchCombineOp, 2 = dispatch_combine_v2 (flydsl)
  int rank, world, gpuPerNode, maxTok, topk, expertsPerRank, maxRecv;
  int64_t srcStride, destStride;
  double dispRowBytes, combRowBytes;  // v2 (v1: from the kernel)
  uintptr_t outIdx, srcMap;           // v2 (v1: from the kernel arguments)
  std::vector<uintptr_t> keys;        // device pointers that find the op in its kernels' arguments
  std::string group;                  // the same on every rank, unique per op and job
};
void Register(const OpSpec& spec);
// Counters of every op of this process (JSON); also written to MORI_EP_SIM_STATS_DIR.
std::string Stats();

}  // namespace sim
}  // namespace moe
}  // namespace mori
