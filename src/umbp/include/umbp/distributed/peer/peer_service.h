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

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mori::umbp {

class PeerDramAllocator;
class MasterClient;

class PeerServiceServer {
 public:
  // `dram_alloc` is non-owning and may be null when the host process has no
  // DRAM/HBM tier.  When null, the AllocateSlot/CommitSlot/AbortSlot/
  // ResolveKey/EvictKey handlers respond with success=false / found=false.
  //
  // SSD read staging (PrepareSsdRead/ReleaseSsdLease) was removed in the
  // backend-agnostic refactor Phase 0 — SSD is unwired from the distributed
  // data plane (see design-backend-agnostic-refactor.md).
  PeerServiceServer(PeerDramAllocator* dram_alloc, std::vector<uint8_t> engine_desc_bytes = {},
                    MasterClient* master_client = nullptr);
  ~PeerServiceServer();

  bool Start(uint16_t port);
  void Stop();

  // Read-only access for the heartbeat shipper (lives in MasterClient
  // / PoolClient).  Never null after construction with a non-null
  // allocator argument.
  PeerDramAllocator* DramAllocator() const { return dram_alloc_; }

 private:
  PeerDramAllocator* dram_alloc_;
  MasterClient* master_client_;

  std::vector<uint8_t> engine_desc_bytes_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
