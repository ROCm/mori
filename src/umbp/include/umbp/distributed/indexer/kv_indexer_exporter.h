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

#include <grpcpp/support/status.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_indexer_exporter.pb.h"
#include "sgl_kv_indexer.grpc.pb.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Optional soft-state DRAM exporter. The source returns non-destructive,
// committed-key snapshots; never use DrainPendingEvents/FullSync here, since
// those belong exclusively to the UMBP master's heartbeat protocol.
//
// One exporter must exclusively own its (worker_id, DRAM) Indexer bucket.
// Source lifetime must extend through Stop(). SyncOnce is serialized with the
// background loop. No data-plane lock is held across an Indexer RPC.
// Start/Stop/destruction must be externally serialized.
class KvIndexerExporter {
 public:
  using Config = indexer::ExporterConfig;
  using SnapshotSource = std::function<std::vector<KvEvent>()>;

  KvIndexerExporter(Config config, SnapshotSource source);
  ~KvIndexerExporter();
  KvIndexerExporter(const KvIndexerExporter&) = delete;
  KvIndexerExporter& operator=(const KvIndexerExporter&) = delete;

  // Strict protobuf JSON. 64-bit hashes should be decimal strings in JSON.
  static Config LoadConfig(const std::string& path);
  static void Validate(const Config& config);

  // When set, reload the manifest before every snapshot. Only blocks may be
  // changed live. Replace the file atomically; identity/options require restart.
  void Start(std::string config_path = {});
  // Joins before withdrawing the owned DRAM bucket, with the same RPC deadline.
  // A failed withdrawal is observable; this Indexer has no worker TTL.
  grpc::Status Stop();
  grpc::Status SyncOnce(bool force_full_sync = false);

 private:
  struct Placement {
    indexer::BlockMapping block;
    uint32_t mask = 0;
  };
  using Placements = std::map<int64_t, Placement>;

  grpc::Status Apply(const std::vector<kv_indexer::v1::ExternalKvAction>& actions);
  grpc::Status Clear();
  void Loop();

  Config config_;
  SnapshotSource source_;
  std::unique_ptr<kv_indexer::v1::KVIndexer::Stub> stub_;
  std::mutex sync_mutex_;
  Placements applied_;
  // Keep canonical ancestry/size across manifest removal and reintroduction.
  struct Identity {
    bool has_parent;
    int64_t parent;
    uint32_t block_size;
  };
  std::map<int64_t, Identity> identities_;
  bool needs_full_sync_ = true;
  bool contacted_ = false;
  uint64_t seq_ = 0;
  std::chrono::steady_clock::time_point last_full_sync_{};

  std::mutex lifecycle_mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
  std::string config_path_;
};

}  // namespace mori::umbp
