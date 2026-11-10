// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <hip/hip_runtime_api.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/object_view.h"
#include "umbp/engine/engine_types.h"
#include "umbp/engine/kv_allocator.h"

namespace mori::umbp::engine {

class EngineClient;

bool ValidateLoadPlanRanges(const std::vector<TransferPlan>& plans);
std::optional<uintptr_t> TranslateRegisteredHostSpan(uintptr_t host_address, size_t bytes,
                                                     uintptr_t registered_host_base,
                                                     uintptr_t registered_device_base,
                                                     size_t registered_bytes);

class EngineLoadHandle {
 public:
  ~EngineLoadHandle();
  EngineLoadHandle(const EngineLoadHandle&) = delete;
  EngineLoadHandle& operator=(const EngineLoadHandle&) = delete;

  uint64_t id() const;
  size_t layer_count() const;
  bool layer_done(size_t index) const;
  LoadOutcome layer_outcome(size_t index) const;
  std::string layer_error(size_t index) const;
  // Borrowed HIP event; valid while this LoadHandle remains alive. Remote
  // layers return 0 until/after their CPU-observed RDMA completion.
  uintptr_t layer_event(size_t index) const;
  LoadOutcome outcome() const;
  std::string error() const;
  LoadOutcome Wait();

 private:
  friend class EngineExecutor;
  struct State;
  explicit EngineLoadHandle(std::shared_ptr<State> state);
  std::shared_ptr<State> state_;
};

struct EngineExecutorConfig {
  std::string executor_id;
  int device = -1;  // -1 is the CPU test executor; >=0 enables HIP copies.
  std::string io_host;
  uint16_t io_port = 0;
  uint64_t staging_bytes = 0;
};

// Rank-side data-plane owner. It retains original framework regions and shared
// DRAM mappings until every submitted handle is drained. Unknown completion is
// reported to the caller; mappings are not silently released on that path.
class EngineExecutor {
 public:
  explicit EngineExecutor(EngineExecutorConfig config);
  ~EngineExecutor();

  bool RegisterOriginalRegion(std::string region_id, TransferRef memory);
  bool MapPeerRegion(std::string executor_id, std::string region_id, TransferRef memory);
  bool ImportPeerRegion(const PeerVmmRegion& region, int fd);
  bool MapSharedDram(const std::vector<SharedDramRegion>& regions, std::string* error = nullptr);
  bool ExecutePlans(const std::vector<TransferPlan>& plans, std::vector<TransferFailure>* failures);
  bool Hydrate(TransferRef* ref) const;
  void Close();
  std::vector<uint8_t> PackedEngineDesc() const;
  bool BootstrapRemote(std::string executor_id, const std::vector<uint8_t>& engine_desc);
  void AttachClient(EngineClient* client);
  std::shared_ptr<EngineLoadHandle> SubmitLoad(const PreparedLoad& prepared);
  std::vector<std::pair<uint64_t, LoadOutcome>> PollLoads();

  const std::string& id() const { return config_.executor_id; }
  size_t mapped_region_count() const;
#ifdef MORI_UMBP_TESTING
  void SetBeforeExecuteHookForTest(std::function<void()> hook);
  void SetExecuteOverrideForTest(
      std::function<bool(const std::vector<TransferPlan>&, std::vector<TransferFailure>*)> hook);
  void SetLoadEventCreateFailureForTest(bool fail);
  void SetLoadEventSyncFailureForTest(bool fail);
  void SetLoadEnqueueFailureForTest(bool fail);
  void SetLoadStreamSyncFailureForTest(bool fail);
  void SetImportPeerRegionFailureForTest(bool fail);
#endif

 private:
  struct Mapping {
    void* base = nullptr;
    void* device_base = nullptr;
    size_t mapped_length = 0;
    bool hip_registered = false;
    uint64_t device = 0;
    uint64_t inode = 0;
  };
  struct PeerImport {
    std::string key;
    uint64_t mapping_id = 0;
    ImportedVmmRegion allocation;
  };

  EngineExecutorConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable idle_cv_;
  size_t in_flight_ = 0;
  size_t load_in_flight_ = 0;
  bool unsafe_resources_ = false;
  std::unordered_map<std::string, TransferRef> regions_;
  std::vector<Mapping> mappings_;
  std::vector<TransferRef> owned_registrations_;
  std::vector<PeerImport> imported_regions_;
  std::unique_ptr<TransferEngine> transfer_;
  PeerDirectory* peer_directory_ = nullptr;
  EngineClient* client_ = nullptr;  // borrowed; EngineClient detaches before Close
  std::deque<std::pair<uint64_t, LoadOutcome>> completed_loads_;
  hipStream_t completion_stream_ = nullptr;
  bool closed_ = false;
#ifdef MORI_UMBP_TESTING
  std::function<void()> before_execute_hook_;
  std::function<bool(const std::vector<TransferPlan>&, std::vector<TransferFailure>*)>
      execute_override_;
  bool fail_load_event_create_ = false;
  bool fail_load_event_sync_ = false;
  bool fail_load_enqueue_ = false;
  bool fail_load_stream_sync_ = false;
  bool fail_import_peer_region_ = false;
#endif
};

}  // namespace mori::umbp::engine
