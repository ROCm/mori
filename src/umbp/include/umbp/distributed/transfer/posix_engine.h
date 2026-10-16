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

#include <memory>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// File <-> host memory with preadv/pwritev, so the SSD tier moves bytes to and
// from DRAM pages with no staging.  File -> GPU stays with GdsEngine.  A write
// zero-pads its tail to a 4 KiB block when the file range has room; under
// O_DIRECT the unaligned tail goes through a small aligned bounce.
class PosixEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "PosixEngine"; }

  // Not a memory engine, and a file needs no registration here.
  TransferRef RegisterMemory(void*, size_t, mori::io::MemoryLocationType, int) override {
    return TransferRef{};
  }
  void Deregister(const TransferRef&) override {}

  // Exactly one endpoint a file, the other host memory in this process.
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // One plan per (file range, memory buffer); Submit merges file-contiguous
  // segments into one vectored call.
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items = 0) const override;

  // Synchronous; the handle comes back settled.
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;
};

}  // namespace mori::umbp
