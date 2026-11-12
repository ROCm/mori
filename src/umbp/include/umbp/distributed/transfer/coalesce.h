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

#include <limits>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// Called after each page fragment has passed range validation. Extending the
// prior item avoids copying its heavyweight endpoint descriptors per page.
// Preserve the registration identity, per-key failure tag, and transport's
// uint32 segment-size limit; gaps and different allocations start a new item.
inline bool ExtendRegisteredGpuTransfer(TransferItem* last, const TransferRef& src,
                                        uint64_t src_offset, const TransferRef& dst,
                                        uint64_t dst_offset, uint64_t bytes, size_t tag) {
  constexpr uint64_t limit = std::numeric_limits<uint32_t>::max();
  if (!last || bytes == 0 || bytes > limit || last->size > limit - bytes || !src.HasMemoryDesc() ||
      !dst.HasMemoryDesc() || src.loc != mori::io::MemoryLocationType::GPU ||
      dst.loc != mori::io::MemoryLocationType::GPU || last->tag != tag ||
      last->src.mem.id != src.mem.id || last->src.mem.engineKey != src.mem.engineKey ||
      last->dst.mem.id != dst.mem.id || last->dst.mem.engineKey != dst.mem.engineKey ||
      src_offset < last->src_offset || src_offset - last->src_offset != last->size ||
      dst_offset < last->dst_offset || dst_offset - last->dst_offset != last->size)
    return false;
  last->size += bytes;
  return true;
}

}  // namespace mori::umbp
