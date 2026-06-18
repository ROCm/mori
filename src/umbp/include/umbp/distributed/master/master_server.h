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

#include <grpcpp/server.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/evict_strategy.h"
#include "umbp/distributed/master/eviction_manager.h"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/external_kv_hit_index.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/routing/route_get_strategy.h"
#include "umbp/distributed/routing/route_put_strategy.h"
#include "umbp/distributed/routing/router.h"

namespace mori::umbp {

class MasterServer {
 public:
  explicit MasterServer(MasterServerConfig config);
  ~MasterServer();

  MasterServer(const MasterServer&) = delete;
  MasterServer& operator=(const MasterServer&) = delete;

  void Run();
  void Shutdown();

  // Returns the port the gRPC server is actually listening on.  Useful when
  // listen_address specifies port 0 (OS-assigned).  Returns 0 until Run()
  // has called BuildAndStart().
  uint16_t GetBoundPort() const { return bound_port_.load(); }

 private:
  MasterServerConfig config_;
  GlobalBlockIndex index_;
  ExternalKvBlockIndex external_kv_index_;
  ExternalKvHitIndex external_kv_hit_index_;
  ClientRegistry registry_;
  Router router_;

  std::unique_ptr<mori::metrics::MetricsServer> metrics_server_;
  std::unique_ptr<grpc::Server> server_;

  class UMBPMasterServiceImpl;
  std::unique_ptr<UMBPMasterServiceImpl> service_;

  // Outbound peer-stub pool used by EvictionManager to ship EvictKey
  // RPCs.  Defined in master_server.cpp's anonymous namespace; the
  // header sees only the EvictKeyDispatcher base.  Must outlive
  // eviction_manager_, which holds a non-owning pointer to it.
  std::unique_ptr<EvictKeyDispatcher> peer_stub_pool_;

  std::unique_ptr<EvictionManager> eviction_manager_;

  std::atomic<uint16_t> bound_port_{0};

  void StartHitIndexGc();
  void StopHitIndexGc();
  void HitIndexGcLoop();

  std::thread hit_index_gc_thread_;
  std::atomic<bool> hit_index_gc_running_{false};
  std::mutex hit_index_gc_cv_mutex_;
  std::condition_variable hit_index_gc_cv_;
};

}  // namespace mori::umbp
