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

// Source-private run planner shared only by DRAMTier and its unit test.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace mori::umbp::detail {

enum class DeviceCopyRunKind { kSingle, kLinear, kPitched };

struct DeviceCopyRun {
  DeviceCopyRunKind kind{DeviceCopyRunKind::kSingle};
  size_t end{0};  // one past the last job
  size_t bytes{0};
  size_t width{0};
  size_t spitch{0};
  size_t dpitch{0};
};

// Off by default, and it should stay that way. Reading one layer out of
// page-sized objects gives hipMemcpy2DAsync a pitch/width ratio in the tens,
// and at that ratio it is bimodal: ~0.35 us per fragment once the source pages
// have been through it, ~194 us the first time. A live tier recycles slots
// constantly, so it only ever pays the cold price -- measured 133x slower than
// per-fragment 1D copies end to end. The gather kernel covers this shape
// instead. Set UMBP_DRAM_PITCHED_COPY=1 to measure the pitched path again.
inline bool PitchedCopyEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("UMBP_DRAM_PITCHED_COPY");
    if (value == nullptr) return false;
    const std::string text(value);
    return text == "1" || text == "on" || text == "true";
  }();
  return enabled;
}

inline uintptr_t PointerAddress(const void* ptr) { return reinterpret_cast<uintptr_t>(ptr); }

inline bool Adjacent(uintptr_t base, size_t size, uintptr_t next) {
  return size <= std::numeric_limits<uintptr_t>::max() - base && base + size == next;
}

// Job must expose `src`, `dst`, and `size`. `ordered` must be sorted by source
// address. Kept as a small pure planner so path selection can be regression-
// tested without issuing HIP work.
//
// `allow_pitched` is false for the gather-kernel path, which wants only the
// flat merges: a pitched run describes several fragments in one submission,
// which is the copy engine's vocabulary, not the kernel's. Callers pass
// PitchedCopyEnabled() rather than having it read here, so the planner stays a
// pure function of its arguments and can be tested without the environment.
template <typename Job>
DeviceCopyRun FindDeviceCopyRun(const std::vector<Job*>& ordered, size_t begin,
                                bool allow_pitched = true) {
  const Job* first = ordered[begin];
  DeviceCopyRun run{DeviceCopyRunKind::kSingle, begin + 1, first->size, first->size, 0, 0};

  // Preserve the existing fast path exactly: adjacent jobs may have different
  // sizes and still form one flat memcpy.
  size_t linear_end = begin + 1;
  size_t linear_bytes = first->size;
  while (linear_end < ordered.size()) {
    const Job* prev = ordered[linear_end - 1];
    const Job* cur = ordered[linear_end];
    if (!Adjacent(PointerAddress(prev->src), prev->size, PointerAddress(cur->src)) ||
        !Adjacent(PointerAddress(prev->dst), prev->size, PointerAddress(cur->dst)) ||
        cur->size > std::numeric_limits<size_t>::max() - linear_bytes) {
      break;
    }
    linear_bytes += cur->size;
    ++linear_end;
  }
  if (linear_end > begin + 1) {
    run.kind = DeviceCopyRunKind::kLinear;
    run.end = linear_end;
    run.bytes = linear_bytes;
    return run;
  }

  if (!allow_pitched || begin + 1 >= ordered.size() || first->size == 0 ||
      ordered[begin + 1]->size != first->size) {
    return run;
  }
  const uintptr_t src0 = PointerAddress(first->src);
  const uintptr_t src1 = PointerAddress(ordered[begin + 1]->src);
  const uintptr_t dst0 = PointerAddress(first->dst);
  const uintptr_t dst1 = PointerAddress(ordered[begin + 1]->dst);
  if (src1 <= src0 || dst1 <= dst0) return run;

  const size_t spitch = src1 - src0;
  const size_t dpitch = dst1 - dst0;
  if (spitch < first->size || dpitch < first->size) return run;

  size_t pitched_end = begin + 2;
  while (pitched_end < ordered.size()) {
    const Job* prev = ordered[pitched_end - 1];
    const Job* cur = ordered[pitched_end];
    const uintptr_t prev_src = PointerAddress(prev->src);
    const uintptr_t cur_src = PointerAddress(cur->src);
    const uintptr_t prev_dst = PointerAddress(prev->dst);
    const uintptr_t cur_dst = PointerAddress(cur->dst);
    if (cur->size != first->size || cur_src <= prev_src || cur_dst <= prev_dst ||
        cur_src - prev_src != spitch || cur_dst - prev_dst != dpitch) {
      break;
    }
    ++pitched_end;
  }

  const size_t rows = pitched_end - begin;
  run.kind = DeviceCopyRunKind::kPitched;
  run.end = pitched_end;
  run.bytes = first->size <= std::numeric_limits<size_t>::max() / rows ? first->size * rows : 0;
  run.width = first->size;
  run.spitch = spitch;
  run.dpitch = dpitch;
  return run;
}

}  // namespace mori::umbp::detail
