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
#include <memory>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// Non-temporal AVX2 block copy where it pays, plain memcpy otherwise.
//
// Exposed (rather than kept file-local to the engine) because the async
// re-cache path also copies a whole KV block off the Get critical path and
// wants the same cache-bypass behavior.  Honors UMBP_DRAM_NT_COPY=0.
void HostCopyBlock(void* dst, const void* src, size_t size);

// The both-endpoints-are-local implementation of TransferEngine.
//
// This is what makes the local fast path stop being a special case (design doc
// §5 Phase 6).  Before this phase PoolClient held a raw base pointer per DRAM
// buffer and memcpy'd through it directly, which is why the local path was
// host-DRAM-only, why PoolClient had to name a concrete backend type to get
// those pointers, and why a second live medium would have needed a tier branch
// in the copy loop.  Now a local access is just a transfer whose endpoints are
// both local, and the planner picks this engine for it.
//
// §2 warned that the fold-in is only correct if it is also free: "the local
// fast path exists because a memcpy beats an engine round trip... this is a
// measurement, not an argument."  It is free here because nothing round trips —
// Submit performs the copies synchronously on the calling thread and returns an
// already-settled handle, so the cost over the old loop is one small plan
// vector per key, against a multi-MiB copy.  The cross-key parallelism
// (UMBP_DRAM_{READ,WRITE}_THREADS) stays where it was, above this engine.
class LocalCopyEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "LocalCopyEngine"; }

  // Nothing to pin: a process-local pointer is already a usable endpoint.
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return TransferRef::HostBytes(base, size, loc, device);
  }
  void Deregister(const TransferRef&) override {}

  // Both endpoints must be host-addressable in this process.  Note this is
  // deliberately narrower than "both are local": a GPU endpoint is local but
  // not memcpy-able, and falls to an engine that can reach it.
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // Group by (src base, dst base) and coalesce exactly-contiguous segments, so
  // a multi-page key whose pages happen to be adjacent collapses into one
  // memcpy instead of one per page.
  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override;

  // Performs the copies inline; the returned handle is already settled.
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;
};

}  // namespace mori::umbp
