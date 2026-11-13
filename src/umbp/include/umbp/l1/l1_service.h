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
#include <string>
#include <vector>

#include "umbp/l1/l1_pool.h"

namespace grpc {
class Service;
}  // namespace grpc

namespace mori::umbp::l1 {

// The UMBPL1Pool gRPC service over a PoolManager. Returned as a grpc::Service
// so a host server registers it without seeing the generated types.
std::unique_ptr<grpc::Service> MakeL1PoolService(std::shared_ptr<PoolManager> manager);
// The same service answering Export alone, for a port other nodes can reach:
// a pool is attached and driven only by the engine on this host.
std::unique_ptr<grpc::Service> MakeL1ExportService(std::shared_ptr<PoolManager> manager);

// A gRPC server hosting only the L1 pool service. umbp_standalone_server hosts
// the service itself; this is for tests and tools.
class L1Server {
 public:
  L1Server(const std::string& address, std::shared_ptr<MemoryApi> api);
  ~L1Server();

  L1Server(const L1Server&) = delete;
  L1Server& operator=(const L1Server&) = delete;

  bool ok() const;
  // For a test to attach lower tiers before any pool exists.
  PoolManager& manager();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Engine-side handle on the service. One per process; calls are blocking and
// may be issued from any thread.
class L1Client {
 public:
  // `address` is a gRPC target, e.g. "unix:///run/umbp/standalone/n0.grpc.sock".
  explicit L1Client(const std::string& address);
  ~L1Client();

  L1Client(const L1Client&) = delete;
  L1Client& operator=(const L1Client&) = delete;

  AttachResult Attach(const std::string& pool_id, const std::string& group, int device,
                      uint64_t bytes);
  Status BindLayout(const std::string& pool_id, const Layout& layout);
  OpenResult Open(const std::string& pool_id);
  AllocateResult Allocate(const std::string& pool_id, uint32_t count, uint32_t min_count);
  Status Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp = 0);
  Status Commit(const std::string& pool_id, uint64_t committed);
  FetchResult Fetch(const std::string& pool_id, const std::vector<std::string>& keys);
  // Node to node; see PoolManager::Export.
  ExportResult Export(const std::string& group, const std::vector<std::string>& keys);
  Status Clear(const std::string& pool_id);
  Status Detach(const std::string& pool_id);
  Status Stats(std::vector<PoolStats>* out);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mori::umbp::l1
