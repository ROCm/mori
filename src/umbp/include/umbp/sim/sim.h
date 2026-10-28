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

#include <array>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// Address identities only. PROT_NONE makes accidental payload access fail.
class SimPageMemorySource final : public PageMemorySource {
 public:
  SimPageMemorySource(mori::io::MemoryLocationType loc, int device) : loc_(loc), device_(device) {}
  ~SimPageMemorySource() override { Release(); }
  bool Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) override;
  void Release() override;
  mori::io::MemoryLocationType LocationType() const override { return loc_; }
  int Device() const override { return device_; }
  const char* Name() const override { return "SimPageMemorySource"; }

 private:
  mori::io::MemoryLocationType loc_;
  int device_;
  std::vector<Buffer> buffers_;
};

// Models local memory copies, completing inside Submit just like the real local engines.
class SimTransferEngine final : public TransferEngine {
 public:
  explicit SimTransferEngine(const std::string& profile_path);
  const char* Name() const override { return "SimTransferEngine"; }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return TransferRef::HostBytes(base, size, loc, device);
  }
  void Deregister(const TransferRef&) override {}
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items = 0) const override;
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

 private:
  using Clock = std::chrono::steady_clock;
  struct Path {
    std::string channel;
    double latency_ns;
    double bandwidth;
  };
  // CPU->CPU, CPU->GPU, GPU->CPU, GPU->GPU. Channels may be shared across paths.
  std::array<std::optional<Path>, 4> paths_;
  std::mutex mutex_;
  std::unordered_map<std::string, Clock::time_point> available_;
};

}  // namespace mori::umbp
