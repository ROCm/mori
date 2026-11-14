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

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
struct PoolResolvedEntry;
class BackendRegistry;

struct FabricReadPushRange {
  uint64_t source_offset = 0;
  uint64_t size = 0;
  TransferRef destination;
  uint64_t destination_offset = 0;
};
struct FabricReadPushRequest {
  // Exact wire request: retries must not change destinations or source ranges.
  std::string signature;
  std::string reader_id;
  std::string engine_desc;
  std::vector<std::vector<FabricReadPushRange>> ranges;
};
struct FabricReadPushResult {
  bool supported = false;  // false guarantees that no destination write was posted
  bool drained = true;
  uint64_t submitted_bytes = 0;
  uint64_t submitted_plans = 0;
};

// Uses the same transfer engine/completion machinery as GET. HostDeviceAddress
// returns an already registered GPU alias; this class never registers pageable
// host memory itself or changes source placement. Callers retain source pins.
class FabricReadPush {
 public:
  using HostDeviceAddress = std::function<void*(const void*, size_t, int)>;
  FabricReadPush(TransferEngine* engine, PeerDirectory* peers, BackendRegistry* backends,
                 int device, HostDeviceAddress address, std::function<void()> quarantine,
                 bool rdma = false);
  ~FabricReadPush();
  FabricReadPushResult Execute(std::vector<PoolResolvedEntry>* entries,
                               const FabricReadPushRequest& request);
  // RDMA mode: the owner posts RDMA WRITEs from its registered source buffers
  // straight into the reader's registered destination (host or GPU), so a
  // remote get never needs RDMA READ.  Otherwise the FABRIC (UALink) path.
  bool Rdma() const { return rdma_; }

 private:
  TransferEngine* engine_;
  PeerDirectory* peers_;
  BackendRegistry* backends_;
  int device_;
  HostDeviceAddress address_;
  std::function<void()> quarantine_;
  bool rdma_ = false;
  // Registration metadata only, never payload. Fixed bound; overflow is scoped
  // to the read. Backend allocations/host aliases are stable until shutdown.
  std::atomic<bool> quarantined_{false};
  std::mutex aliases_mutex_;
  std::map<std::pair<uintptr_t, size_t>, TransferRef> aliases_;
};
}  // namespace mori::umbp
