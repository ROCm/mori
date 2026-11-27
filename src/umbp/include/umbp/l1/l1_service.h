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

#include <chrono>
#include <functional>
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
// End the service's shared-memory sessions, each of which holds an RPC open
// until its client leaves. Call before shutting the hosting server down.
void StopL1PoolService(grpc::Service* service);

// A gRPC server hosting only the L1 pool service. umbp_standalone_server hosts
// the service itself; this is for tests and tools.
class L1Server {
 public:
  L1Server(const std::string& address, std::shared_ptr<MemoryApi> api);
  ~L1Server();

  L1Server(const L1Server&) = delete;
  L1Server& operator=(const L1Server&) = delete;

  bool ok() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct L1ClientOptions {
  // Carry the per-step calls -- Apply, Commit, Allocate and a probing Fetch --
  // in shared memory instead of over the socket. Apply and Commit then return
  // as soon as they are queued: the server applies them in order, before
  // anything this client sends later by either route, and a failure is
  // reported by the client's next Apply (or Allocate / Fetch), respectively
  // Commit. Needs a server that offers ShmSession and shares the segment's
  // directory; without either the client stays on gRPC.
  bool shm = false;
  // How long a shared-memory call polls for its reply before it sleeps.
  std::chrono::microseconds shm_spin{200};
  // How long to wait for the server when opening the channel.
  std::chrono::milliseconds shm_connect{10000};

  // UMBP_L1_TRANSPORT=shm|grpc (default grpc), UMBP_L1_SHM_CLIENT_SPIN_US.
  static L1ClientOptions FromEnv();
};

// Engine-side handle on the service. One per process; calls are blocking and
// may be issued from any thread.
class L1Client {
 public:
  // `address` is a gRPC target, e.g. "unix:///run/umbp/standalone/n0.grpc.sock".
  // Options come from the environment (`L1ClientOptions::FromEnv`).
  explicit L1Client(const std::string& address);
  L1Client(const std::string& address, const L1ClientOptions& options);
  ~L1Client();

  L1Client(const L1Client&) = delete;
  L1Client& operator=(const L1Client&) = delete;

  AttachResult Attach(const std::string& pool_id, const std::string& group, int device,
                      uint64_t bytes, const ExternalMemory& external = {});
  Status BindLayout(const std::string& pool_id, const Layout& layout);
  OpenResult Open(const std::string& pool_id);
  AllocateResult Allocate(const std::string& pool_id, uint32_t count, uint32_t min_count);
  Status Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp = 0);
  Status Commit(const std::string& pool_id, uint64_t committed);
  FetchResult Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                    const FetchOptions& options = {});
  Status Clear(const std::string& pool_id);
  Status Detach(const std::string& pool_id);
  Status Stats(std::vector<PoolStats>* out);

  // Whether the per-step calls go through shared memory.
  bool shm_active() const;
  // `enter` and `leave` are called on the calling thread around every wait a
  // call makes -- an RPC, a sleep for a shared-memory reply -- and around
  // nothing else. A caller that holds a lock it must not sleep under gives it
  // up in `enter` and takes it back in `leave`: the Python binding does so
  // with the GIL, which a call queued in shared memory then never gives up.
  // Set before the client is shared between threads.
  void SetBlockingHooks(std::function<void()> enter, std::function<void()> leave);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mori::umbp::l1
