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
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// The SSD-file-to-GPU implementation of TransferEngine, over AMD hipFile (GDS).
//
// WHY THIS EXISTS.  SsdBackend publishes a File TransferRef (an O_DIRECT fd + a
// byte range) instead of staging its bytes through host DRAM.  No memory engine
// can move a file endpoint — their CanHandle keys on HasHostPtr() /
// HasMemoryDesc(), both false for a file ref — so a (file, GPU) pair was
// unservable until this engine.  It reads the range straight into device memory
// with hipFileRead, which DMAs drive -> GPU with no host bounce (design §4/§6).
//
// READ ONLY, FOR NOW.  CanHandle claims exactly (file source, device
// destination): the prefill KV-load path.  The reverse (GPU -> file, via
// hipFileWrite) is a later change, and a host destination is left to the staged
// path, so this engine never widens what the memory engines already serve.
//
// The fd handle is registered once per fd and shared by every range on that
// segment (a segment file holds many keys): RegisterFile ref-counts, Deregister
// releases at zero.  hipFileHandle_t is kept as an opaque void* so this header
// carries no hipfile dependency; only the .cpp includes <hipfile.h>.
class GdsEngine final : public TransferEngine {
 public:
  GdsEngine() = default;
  ~GdsEngine() override;

  const char* Name() const override { return "GdsEngine"; }

  // Not a memory engine: an invalid ref tells the composite's memory fan-out to
  // skip it.
  TransferRef RegisterMemory(void*, size_t, mori::io::MemoryLocationType, int) override {
    return TransferRef{};
  }

  // Register an O_DIRECT fd as a hipFile handle (once per fd, ref-counted) and
  // return a File ref carrying it.  Invalid ref if registration fails.
  TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) override;
  void Deregister(const TransferRef& ref) override;

  // A file source into a device destination — the read path — and nothing else.
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // One plan per (file range -> device buffer).  Grouping/coalescing is left
  // out deliberately: hipFileRead already moves a contiguous range in one DMA,
  // and a batch's ranges land in disjoint GPU buffers.
  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override;

  // Performs the reads inline with hipFileRead; the returned handle is already
  // settled (same contract as the copy engines).
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

 private:
  // One registered fd: its opaque hipFileHandle_t and a ref count so a segment
  // shared by many keys registers once and releases once.
  struct HandleEntry {
    void* handle = nullptr;  // hipFileHandle_t
    int refcount = 0;
  };

  mutable std::mutex handles_mutex_;
  std::unordered_map<int, HandleEntry> handles_;
};

}  // namespace mori::umbp
