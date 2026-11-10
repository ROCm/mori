// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/engine/engine_executor.h"

#include <hip/hip_runtime_api.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>
#include <thread>

#include "umbp/common/device_gather.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/engine/engine_client.h"

namespace mori::umbp::engine {
namespace {

std::string RegionKey(std::string_view executor_id, std::string_view region_id) {
  std::string out(executor_id);
  out.push_back('\0');
  out.append(region_id);
  return out;
}

hipMemcpyKind LoadCopyKind(const TransferRef& src, const TransferRef& dst) {
  const bool source_gpu = src.loc == mori::io::MemoryLocationType::GPU;
  const bool target_gpu = dst.loc == mori::io::MemoryLocationType::GPU;
  if (source_gpu && target_gpu) return hipMemcpyDeviceToDevice;
  if (source_gpu) return hipMemcpyDeviceToHost;
  if (target_gpu) return hipMemcpyHostToDevice;
  return hipMemcpyHostToHost;
}

std::string DescribeTransferFailures(const std::vector<TransferFailure>& failures) {
  std::ostringstream out;
  for (size_t i = 0; i < failures.size(); ++i) {
    if (i != 0) out << "; ";
    out << "code=" << failures[i].code << " message='" << failures[i].message << "'";
    if (!failures[i].endpoint.empty()) out << " endpoint='" << failures[i].endpoint << "'";
    out << " submitted=" << failures[i].submitted << " quiesced=" << failures[i].quiesced;
  }
  return out.str();
}

}  // namespace

bool ValidateLoadPlanRanges(const std::vector<TransferPlan>& plans) {
  for (const auto& plan : plans) {
    if (!plan.src.HasHostPtr() || !plan.dst.HasHostPtr() ||
        plan.src_offsets.size() != plan.dst_offsets.size() ||
        plan.src_offsets.size() != plan.sizes.size()) {
      return false;
    }
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      if (plan.src_offsets[i] > plan.src.size ||
          plan.sizes[i] > plan.src.size - plan.src_offsets[i] ||
          plan.dst_offsets[i] > plan.dst.size ||
          plan.sizes[i] > plan.dst.size - plan.dst_offsets[i]) {
        return false;
      }
      const uintptr_t source = reinterpret_cast<uintptr_t>(plan.src.host_ptr);
      const uintptr_t target = reinterpret_cast<uintptr_t>(plan.dst.host_ptr);
      if (source > UINTPTR_MAX - plan.src_offsets[i] ||
          target > UINTPTR_MAX - plan.dst_offsets[i]) {
        return false;
      }
    }
  }
  return true;
}

std::optional<uintptr_t> TranslateRegisteredHostSpan(uintptr_t host_address, size_t bytes,
                                                     uintptr_t registered_host_base,
                                                     uintptr_t registered_device_base,
                                                     size_t registered_bytes) {
  if (host_address == 0 || registered_host_base == 0 || registered_device_base == 0 || bytes == 0 ||
      host_address < registered_host_base) {
    return std::nullopt;
  }
  const uintptr_t offset = host_address - registered_host_base;
  if (offset > registered_bytes || bytes > registered_bytes - offset ||
      registered_device_base > UINTPTR_MAX - offset) {
    return std::nullopt;
  }
  return registered_device_base + offset;
}

struct EngineLoadHandle::State {
  struct Layer {
    bool done = false;
    LoadOutcome outcome = LoadOutcome::kPending;
    hipEvent_t event = nullptr;
    std::string error;
  };
  mutable std::mutex mutex;
  std::condition_variable cv;
  uint64_t id = 0;
  LoadOutcome outcome = LoadOutcome::kPending;
  std::string error;
  std::vector<Layer> layers;
  ~State() {
    for (auto& layer : layers) {
      if (layer.event != nullptr) (void)hipEventDestroy(layer.event);
      layer.event = nullptr;
    }
  }
};

EngineLoadHandle::EngineLoadHandle(std::shared_ptr<State> state) : state_(std::move(state)) {}

EngineLoadHandle::~EngineLoadHandle() {
  // The shared worker retains State (and its events) until completion.
  // Dropping a Python handle therefore never joins a worker while holding GIL.
}

uint64_t EngineLoadHandle::id() const { return state_ == nullptr ? 0 : state_->id; }

size_t EngineLoadHandle::layer_count() const {
  return state_ == nullptr ? 0 : state_->layers.size();
}

bool EngineLoadHandle::layer_done(size_t index) const {
  if (state_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return index < state_->layers.size() && state_->layers[index].done;
}

LoadOutcome EngineLoadHandle::layer_outcome(size_t index) const {
  if (state_ == nullptr) return LoadOutcome::kFailed;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return index < state_->layers.size() ? state_->layers[index].outcome : LoadOutcome::kFailed;
}

std::string EngineLoadHandle::layer_error(size_t index) const {
  if (state_ == nullptr) return "load handle has no state";
  std::lock_guard<std::mutex> lock(state_->mutex);
  return index < state_->layers.size() ? state_->layers[index].error : "invalid layer index";
}

uintptr_t EngineLoadHandle::layer_event(size_t index) const {
  if (state_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return index < state_->layers.size() ? reinterpret_cast<uintptr_t>(state_->layers[index].event)
                                       : 0;
}

LoadOutcome EngineLoadHandle::outcome() const {
  if (state_ == nullptr) return LoadOutcome::kFailed;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->outcome;
}

std::string EngineLoadHandle::error() const {
  if (state_ == nullptr) return "load handle has no state";
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->error;
}

LoadOutcome EngineLoadHandle::Wait() {
  if (state_ == nullptr) return LoadOutcome::kFailed;
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->cv.wait(lock, [&] { return state_->outcome != LoadOutcome::kPending; });
  return state_->outcome;
}

EngineExecutor::EngineExecutor(EngineExecutorConfig config) : config_(std::move(config)) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  composite->AddEngine(std::make_unique<HbmCopyEngine>());
  if (!config_.io_host.empty()) {
    mori::io::IOEngineConfig io_config;
    io_config.host = config_.io_host;
    io_config.port = config_.io_port;
    auto remote = std::make_unique<MoriIoEngine>(config_.executor_id, std::move(io_config),
                                                 config_.staging_bytes);
    if (remote->Init()) {
      peer_directory_ = remote.get();
      composite->AddEngine(std::move(remote));
    }
  }
  transfer_ = std::move(composite);
  if (config_.device >= 0) {
    int old_device = -1;
    int least = 0;
    int greatest = 0;
    if (hipGetDevice(&old_device) == hipSuccess && hipSetDevice(config_.device) == hipSuccess &&
        hipDeviceGetStreamPriorityRange(&least, &greatest) == hipSuccess) {
      (void)hipStreamCreateWithPriority(&completion_stream_, hipStreamNonBlocking, greatest);
    }
    if (old_device >= 0) (void)hipSetDevice(old_device);
  }
}

EngineExecutor::~EngineExecutor() { Close(); }

bool EngineExecutor::RegisterOriginalRegion(std::string region_id, TransferRef memory) {
  if (region_id.empty() || !memory.HasHostPtr()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key = RegionKey(config_.executor_id, region_id);
  if (closed_ || regions_.count(key) != 0) return false;
  TransferRef registered =
      transfer_->RegisterMemory(memory.host_ptr, memory.size, memory.loc, memory.device);
  if (!registered.Valid() || (peer_directory_ != nullptr && !registered.HasMemoryDesc()))
    return false;
  memory = std::move(registered);
  memory.executor_id = config_.executor_id;
  memory.region_id = region_id;
  memory.executor_address = reinterpret_cast<uintptr_t>(memory.host_ptr);
  owned_registrations_.push_back(memory);
  regions_.emplace(key, std::move(memory));
  return true;
}

bool EngineExecutor::MapPeerRegion(std::string executor_id, std::string region_id,
                                   TransferRef memory) {
  if (executor_id.empty() || region_id.empty() || !memory.HasHostPtr()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key = RegionKey(executor_id, region_id);
  if (closed_ || regions_.count(key) != 0) return false;
  memory.executor_id = std::move(executor_id);
  memory.region_id = std::move(region_id);
  memory.executor_address = reinterpret_cast<uintptr_t>(memory.host_ptr);
  regions_.emplace(key, std::move(memory));
  return true;
}

bool EngineExecutor::ImportPeerRegion(const PeerVmmRegion& region, int fd) {
#ifdef MORI_UMBP_TESTING
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fail_import_peer_region_) return false;
  }
#endif
  const auto kind = static_cast<VmmHandleKind>(region.handle_kind);
  if (config_.device < 0 || region.executor_id.empty() || region.region_id.empty() ||
      region.mapped_size == 0 || region.region_size == 0 ||
      region.allocation_offset > region.mapped_size ||
      region.region_size > region.mapped_size - region.allocation_offset ||
      (kind == VmmHandleKind::kPosixFd && fd < 0) ||
      (kind == VmmHandleKind::kFabric && region.fabric.empty())) {
    return false;
  }
  const std::string key = RegionKey(region.executor_id, region.region_id);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return false;
    auto existing = std::find_if(imported_regions_.begin(), imported_regions_.end(),
                                 [&](const PeerImport& imported) { return imported.key == key; });
    if (existing != imported_regions_.end()) {
      if (existing->mapping_id == region.mapping_id) return true;
      if (load_in_flight_ != 0) return false;
      regions_.erase(key);
      (void)GlobalKvVmmAllocator().FreeImported(&existing->allocation);
      imported_regions_.erase(existing);
    } else if (regions_.count(key) != 0) {
      return false;
    }
  }
  ExportedVmmHandle exported;
  exported.kind = kind;
  exported.fd = fd;
  exported.fabric = region.fabric;
  exported.mapped_size = region.mapped_size;
  auto imported = GlobalKvVmmAllocator().Import(exported, config_.device);
  if (!imported.has_value()) return false;
  if (!MapPeerRegion(region.executor_id, region.region_id,
                     TransferRef::HostBytes(
                         static_cast<uint8_t*>(imported->base) + region.allocation_offset,
                         region.region_size, mori::io::MemoryLocationType::GPU, config_.device))) {
    (void)GlobalKvVmmAllocator().FreeImported(&*imported);
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  imported_regions_.push_back(PeerImport{key, region.mapping_id, std::move(*imported)});
  return true;
}

bool EngineExecutor::MapSharedDram(const std::vector<SharedDramRegion>& regions,
                                   std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) return false;
  const size_t old_mapping_count = mappings_.size();
  std::vector<std::string> inserted;
  auto rollback = [&]() {
    for (const auto& id : inserted) regions_.erase(id);
    while (mappings_.size() > old_mapping_count) {
      Mapping mapping = mappings_.back();
      mappings_.pop_back();
      if (mapping.hip_registered) (void)hipHostUnregister(mapping.base);
      if (mapping.base != nullptr) munmap(mapping.base, mapping.mapped_length);
    }
  };
  for (const auto& region : regions) {
    if (region.region_id.empty() || region.fd < 0 || region.mapped_length == 0 ||
        region.fd_offset > region.mapped_length ||
        region.length > region.mapped_length - region.fd_offset ||
        regions_.count(RegionKey(config_.executor_id, region.region_id)) != 0) {
      if (error) *error = "invalid or duplicate shared DRAM region";
      rollback();
      return false;
    }
    struct stat identity {};
    if (fstat(region.fd, &identity) != 0) {
      if (error) *error = std::string("fstat shared DRAM: ") + std::strerror(errno);
      rollback();
      return false;
    }
    auto existing = std::find_if(mappings_.begin(), mappings_.end(), [&](const Mapping& mapping) {
      return mapping.device == static_cast<uint64_t>(identity.st_dev) &&
             mapping.inode == static_cast<uint64_t>(identity.st_ino) &&
             mapping.mapped_length == region.mapped_length;
    });
    void* base = existing == mappings_.end() ? MAP_FAILED : existing->base;
    if (existing == mappings_.end()) {
      base = mmap(nullptr, region.mapped_length, PROT_READ | PROT_WRITE, MAP_SHARED, region.fd, 0);
      if (base == MAP_FAILED) {
        if (error) *error = std::string("mmap shared DRAM: ") + std::strerror(errno);
        rollback();
        return false;
      }
      Mapping mapping{base,
                      nullptr,
                      static_cast<size_t>(region.mapped_length),
                      false,
                      static_cast<uint64_t>(identity.st_dev),
                      static_cast<uint64_t>(identity.st_ino)};
      if (config_.device >= 0) {
        int old_device = -1;
        const bool selected =
            hipGetDevice(&old_device) == hipSuccess && hipSetDevice(config_.device) == hipSuccess;
        const hipError_t registered =
            selected ? hipHostRegister(base, region.mapped_length,
                                       hipHostRegisterMapped | hipHostRegisterPortable)
                     : hipErrorInvalidDevice;
        void* device_base = nullptr;
        const hipError_t aliased = registered == hipSuccess
                                       ? hipHostGetDevicePointer(&device_base, base, 0)
                                       : hipErrorInvalidValue;
        if (old_device >= 0) (void)hipSetDevice(old_device);
        if (!selected || registered != hipSuccess || aliased != hipSuccess ||
            device_base == nullptr) {
          if (registered == hipSuccess) (void)hipHostUnregister(base);
          munmap(base, region.mapped_length);
          if (error) *error = "shared DRAM has no device-visible host mapping";
          rollback();
          return false;
        }
        mapping.hip_registered = true;
        mapping.device_base = device_base;
      }
      mappings_.push_back(mapping);
    }
    TransferRef ref = TransferRef::HostBytes(static_cast<char*>(base) + region.fd_offset,
                                             region.length, mori::io::MemoryLocationType::CPU, -1);
    ref.executor_id = config_.executor_id;
    ref.region_id = region.region_id;
    ref.executor_address = reinterpret_cast<uintptr_t>(ref.host_ptr);
    const std::string key = RegionKey(config_.executor_id, region.region_id);
    regions_.emplace(key, std::move(ref));
    inserted.push_back(std::move(key));
  }
  return true;
}

bool EngineExecutor::Hydrate(TransferRef* ref) const {
  if (ref == nullptr) return false;
  if (ref->region_id.empty()) return ref->Valid();
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string executor_id = ref->executor_id.empty() ? config_.executor_id : ref->executor_id;
  auto it = regions_.find(RegionKey(executor_id, ref->region_id));
  if (!closed_ && it != regions_.end()) {
    *ref = it->second;
    return true;
  }
  return !closed_ && ref->HasMemoryDesc();
}

bool EngineExecutor::ExecutePlans(const std::vector<TransferPlan>& plans,
                                  std::vector<TransferFailure>* failures) {
  std::vector<size_t> all_tags;
  for (const auto& plan : plans) {
    all_tags.insert(all_tags.end(), plan.tags.begin(), plan.tags.end());
  }
  std::sort(all_tags.begin(), all_tags.end());
  all_tags.erase(std::unique(all_tags.begin(), all_tags.end()), all_tags.end());
  std::function<void()> before_execute;
#ifdef MORI_UMBP_TESTING
  std::function<bool(const std::vector<TransferPlan>&, std::vector<TransferFailure>*)>
      execute_override;
#endif
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || transfer_ == nullptr) {
      if (failures) {
        failures->push_back(
            {all_tags, 1, "executor is closed", {}, /*quiesced=*/true, /*submitted=*/false});
      }
      return false;
    }
    ++in_flight_;
#ifdef MORI_UMBP_TESTING
    before_execute = before_execute_hook_;
    execute_override = execute_override_;
#endif
  }
  auto guard = std::unique_ptr<EngineExecutor, std::function<void(EngineExecutor*)>>(
      this, [](EngineExecutor* owner) {
        std::lock_guard<std::mutex> lock(owner->mutex_);
        if (--owner->in_flight_ == 0) owner->idle_cv_.notify_all();
      });
  if (before_execute) before_execute();
#ifdef MORI_UMBP_TESTING
  if (execute_override) {
    const bool ok = execute_override(plans, failures);
    if (!ok && failures != nullptr &&
        std::any_of(failures->begin(), failures->end(),
                    [](const auto& failure) { return !failure.quiesced; })) {
      std::lock_guard<std::mutex> lock(mutex_);
      unsafe_resources_ = true;
    }
    return ok;
  }
#endif

  std::vector<TransferItem> items;
  std::vector<std::vector<size_t>> item_tags;
  for (const auto& plan : plans) {
    TransferRef src = plan.src;
    TransferRef dst = plan.dst;
    const bool source_hydrated = Hydrate(&src);
    const bool target_hydrated = Hydrate(&dst);
    if (!source_hydrated || !target_hydrated ||
        plan.src_offsets.size() != plan.dst_offsets.size() ||
        plan.src_offsets.size() != plan.sizes.size()) {
      if (failures) {
        std::ostringstream detail;
        detail << "executor could not hydrate plan source_ok=" << source_hydrated
               << " source_executor='" << plan.src.executor_id << "' source_region='"
               << plan.src.region_id << "' source_size=" << plan.src.size
               << " source_mem=" << plan.src.HasMemoryDesc() << " target_ok=" << target_hydrated
               << " target_executor='" << plan.dst.executor_id << "' target_region='"
               << plan.dst.region_id << "' target_size=" << plan.dst.size
               << " target_mem=" << plan.dst.HasMemoryDesc()
               << " source_offsets=" << plan.src_offsets.size()
               << " target_offsets=" << plan.dst_offsets.size() << " sizes=" << plan.sizes.size();
        failures->push_back({plan.tags,
                             1,
                             detail.str(),
                             {},
                             /*quiesced=*/true,
                             /*submitted=*/false});
      }
      continue;
    }
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      TransferItem item;
      item.src = src;
      item.dst = dst;
      item.src_offset = plan.src_offsets[i];
      item.dst_offset = plan.dst_offsets[i];
      item.size = plan.sizes[i];
      item.tag = item_tags.size();
      items.push_back(std::move(item));
      item_tags.push_back(plan.tags);
    }
  }
  if (items.empty()) return failures == nullptr || failures->empty();
  std::vector<size_t> failed_items;
  std::vector<TransferFailure> transfer_failures;
  bool quiesced = true;
  bool submitted = false;
  if (!transfer_->Transfer(items, &failed_items, nullptr, &quiesced, &submitted,
                           &transfer_failures)) {
    for (auto& failure : transfer_failures) {
      std::vector<size_t> mapped_tags;
      for (size_t item : failure.tags) {
        if (item >= item_tags.size()) continue;
        mapped_tags.insert(mapped_tags.end(), item_tags[item].begin(), item_tags[item].end());
      }
      std::sort(mapped_tags.begin(), mapped_tags.end());
      mapped_tags.erase(std::unique(mapped_tags.begin(), mapped_tags.end()), mapped_tags.end());
      failure.tags = mapped_tags.empty() ? all_tags : std::move(mapped_tags);
    }
    if (failures) {
      if (!transfer_failures.empty()) {
        failures->insert(failures->end(), std::make_move_iterator(transfer_failures.begin()),
                         std::make_move_iterator(transfer_failures.end()));
      } else if (failed_items.empty()) {
        failures->push_back({all_tags,
                             2,
                             "executor transfer failed without item results",
                             {},
                             quiesced,
                             submitted});
      }
      if (transfer_failures.empty()) {
        for (size_t item : failed_items) {
          failures->push_back({item < item_tags.size() ? item_tags[item] : std::vector<size_t>{},
                               2,
                               "executor transfer failed",
                               {},
                               quiesced,
                               submitted});
        }
      }
    }
    if (!quiesced) {
      std::lock_guard<std::mutex> lock(mutex_);
      unsafe_resources_ = true;
    }
    return false;
  }
  return failures == nullptr || failures->empty();
}

size_t EngineExecutor::mapped_region_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mappings_.size();
}

std::vector<uint8_t> EngineExecutor::PackedEngineDesc() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_ || peer_directory_ == nullptr ? std::vector<uint8_t>{}
                                               : peer_directory_->PackedLocalEngineDesc();
}

bool EngineExecutor::BootstrapRemote(std::string executor_id,
                                     const std::vector<uint8_t>& engine_desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  return !closed_ && peer_directory_ != nullptr && !executor_id.empty() &&
         peer_directory_->EnsureRemoteEngine(executor_id,
                                             std::string(engine_desc.begin(), engine_desc.end()));
}

void EngineExecutor::AttachClient(EngineClient* client) {
  std::lock_guard<std::mutex> lock(mutex_);
  client_ = client;
}

std::shared_ptr<EngineLoadHandle> EngineExecutor::SubmitLoad(const PreparedLoad& prepared) {
  auto state = std::make_shared<EngineLoadHandle::State>();
  state->id = prepared.id;
  state->layers.resize(prepared.layers.size());
  const auto settle_all = [state](LoadOutcome outcome, std::string error) {
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (auto& layer : state->layers) {
        layer.done = true;
        layer.outcome = outcome;
        layer.error = error;
      }
      state->outcome = outcome;
      state->error = std::move(error);
    }
    state->cv.notify_all();
  };
  EngineClient* client = nullptr;
  hipStream_t load_stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    client = client_;
    load_stream = completion_stream_;
  }
  if (client == nullptr || !prepared.success || prepared.layers.empty()) {
    if (client != nullptr && prepared.success) (void)client->CancelLoad(prepared);
    settle_all(LoadOutcome::kFailed, "load is invalid or executor has no attached client");
    return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
  }
  for (const auto& bootstrap : prepared.bootstraps) {
    if (!bootstrap.engine_desc.empty() &&
        !BootstrapRemote(bootstrap.executor_id, bootstrap.engine_desc)) {
      (void)client->CancelLoad(prepared);
      settle_all(LoadOutcome::kFailed, "remote executor bootstrap failed");
      return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
    }
  }

  std::vector<std::vector<TransferPlan>> hydrated(prepared.layers.size());
  std::vector<bool> local_layer(prepared.layers.size(), false);
  std::vector<hipEvent_t> created_events;
  bool setup_ok = true;
#ifdef MORI_UMBP_TESTING
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fail_load_event_create_) setup_ok = false;
  }
#endif
  if (setup_ok && config_.device >= 0 && load_stream != nullptr) {
    int old_device = -1;
    if (hipGetDevice(&old_device) == hipSuccess && hipSetDevice(config_.device) == hipSuccess) {
      for (size_t layer = 0; layer < prepared.layers.size(); ++layer) {
        hydrated[layer] = prepared.layers[layer].plans;
        bool local = !hydrated[layer].empty();
        for (auto& plan : hydrated[layer]) {
          local = local && Hydrate(&plan.src) && Hydrate(&plan.dst) && plan.src.HasHostPtr() &&
                  plan.dst.HasHostPtr() &&
                  (plan.src.loc == mori::io::MemoryLocationType::GPU ||
                   plan.dst.loc == mori::io::MemoryLocationType::GPU) &&
                  (plan.src.loc != mori::io::MemoryLocationType::GPU ||
                   plan.src.device == config_.device) &&
                  (plan.dst.loc != mori::io::MemoryLocationType::GPU ||
                   plan.dst.device == config_.device) &&
                  plan.src_offsets.size() == plan.dst_offsets.size() &&
                  plan.src_offsets.size() == plan.sizes.size();
        }
        local = local && ValidateLoadPlanRanges(hydrated[layer]);
        local_layer[layer] = local;
        if (!local) continue;
        hipEvent_t event = nullptr;
        if (hipEventCreateWithFlags(&event, hipEventDisableTiming) != hipSuccess) {
          setup_ok = false;
          break;
        }
        state->layers[layer].event = event;
        created_events.push_back(event);
      }
    } else {
      setup_ok = false;
    }
    if (old_device >= 0) (void)hipSetDevice(old_device);
  }

  if (!setup_ok) {
    for (hipEvent_t event : created_events) (void)hipEventDestroy(event);
    for (auto& layer : state->layers) layer.event = nullptr;
    (void)client->CancelLoad(prepared);
    settle_all(LoadOutcome::kFailed, "load event or local plan setup failed");
    return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
  }

  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_ && client_ == client) {
      ++load_in_flight_;
      accepted = true;
    }
  }
  if (!accepted) {
    for (hipEvent_t event : created_events) (void)hipEventDestroy(event);
    for (auto& layer : state->layers) layer.event = nullptr;
    (void)client->CancelLoad(prepared);
    settle_all(LoadOutcome::kFailed, "executor closed before accepting load");
    return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
  }

  if (!client->MarkLoadStarted(prepared.id)) {
    (void)client->CancelLoad(prepared);
    for (hipEvent_t event : created_events) (void)hipEventDestroy(event);
    for (auto& layer : state->layers) layer.event = nullptr;
    settle_all(LoadOutcome::kFailed, "node rejected load start");
    std::lock_guard<std::mutex> lock(mutex_);
    if (--load_in_flight_ == 0) idle_cv_.notify_all();
    return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
  }

  bool enqueue_ok = true;
  bool enqueue_quiesced = true;
  if (config_.device >= 0 && load_stream != nullptr) {
    int old_device = -1;
    enqueue_ok =
        hipGetDevice(&old_device) == hipSuccess && hipSetDevice(config_.device) == hipSuccess;
    struct AsyncCopy {
      const void* source;
      void* target;
      size_t bytes;
      hipMemcpyKind kind;
    };
    std::vector<DeviceGatherFragment> gather_fragments;
    std::vector<hipMemcpyKind> gather_kinds;
    std::vector<AsyncCopy> copy_fragments;
    std::vector<Mapping> host_mappings;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      host_mappings = mappings_;
    }
    auto device_alias = [&](const void* host, size_t bytes) -> void* {
      const uintptr_t address = reinterpret_cast<uintptr_t>(host);
      for (const auto& mapping : host_mappings) {
        auto translated = TranslateRegisteredHostSpan(
            address, bytes, reinterpret_cast<uintptr_t>(mapping.base),
            reinterpret_cast<uintptr_t>(mapping.device_base), mapping.mapped_length);
        if (translated.has_value()) return reinterpret_cast<void*>(*translated);
      }
      return nullptr;
    };
    for (size_t layer = 0; enqueue_ok && layer < hydrated.size(); ++layer) {
      if (!local_layer[layer]) continue;
      for (const auto& plan : hydrated[layer]) {
        for (size_t part = 0; part < plan.sizes.size(); ++part) {
          auto* source = static_cast<uint8_t*>(plan.src.host_ptr) + plan.src_offsets[part];
          auto* target = static_cast<uint8_t*>(plan.dst.host_ptr) + plan.dst_offsets[part];
          const bool source_gpu = plan.src.loc == mori::io::MemoryLocationType::GPU;
          const bool target_gpu = plan.dst.loc == mori::io::MemoryLocationType::GPU;
          // Imported peer VMM is validated for HIP copy/SDMA access, but that
          // does not imply shader-load access on every xGMI topology. Keep D2D
          // on hipMemcpyAsync; gather is only for registered host<->GPU spans.
          if (source_gpu != target_gpu) {
            void* gather_source = source;
            void* gather_target = target;
            if (!source_gpu) gather_source = device_alias(source, plan.sizes[part]);
            if (!target_gpu) gather_target = device_alias(target, plan.sizes[part]);
            // A gather fragment is admitted only when its host side belongs to
            // a registered mapping with an alias for this executor's device.
            if (gather_source != nullptr && gather_target != nullptr) {
              gather_fragments.push_back({gather_source, gather_target, plan.sizes[part]});
              gather_kinds.push_back(LoadCopyKind(plan.src, plan.dst));
            } else {
              copy_fragments.push_back(
                  {source, target, plan.sizes[part], LoadCopyKind(plan.src, plan.dst)});
            }
          } else {
            copy_fragments.push_back(
                {source, target, plan.sizes[part], LoadCopyKind(plan.src, plan.dst)});
          }
        }
      }
    }
    for (const auto& copy : copy_fragments) {
      if (hipMemcpyAsync(copy.target, copy.source, copy.bytes, copy.kind, load_stream) !=
          hipSuccess) {
        enqueue_ok = false;
        break;
      }
    }
    bool gathered = gather_fragments.empty();
    if (enqueue_ok && !gather_fragments.empty()) {
      gathered = LaunchDeviceGather(gather_fragments.data(), gather_fragments.size(),
                                    config_.device, load_stream);
    }
    if (enqueue_ok && !gathered) {
      for (size_t i = 0; i < gather_fragments.size(); ++i) {
        const auto& fragment = gather_fragments[i];
        if (hipMemcpyAsync(fragment.dst, fragment.src, fragment.bytes, gather_kinds[i],
                           load_stream) != hipSuccess) {
          enqueue_ok = false;
          break;
        }
      }
    }
#ifdef MORI_UMBP_TESTING
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (fail_load_enqueue_) enqueue_ok = false;
    }
#endif
    for (size_t layer = 0; enqueue_ok && layer < hydrated.size(); ++layer) {
      if (!local_layer[layer]) continue;
      if (enqueue_ok && hipEventRecord(state->layers[layer].event, load_stream) != hipSuccess) {
        enqueue_ok = false;
      }
    }
    if (!enqueue_ok) {
      bool force_sync_failure = false;
#ifdef MORI_UMBP_TESTING
      {
        std::lock_guard<std::mutex> lock(mutex_);
        force_sync_failure = fail_load_stream_sync_;
      }
#endif
      enqueue_quiesced = !force_sync_failure && hipStreamSynchronize(load_stream) == hipSuccess;
      if (!enqueue_quiesced) {
        std::lock_guard<std::mutex> lock(mutex_);
        unsafe_resources_ = true;
      }
    }
    if (old_device >= 0) (void)hipSetDevice(old_device);
  }

  if (!enqueue_ok) {
    const LoadOutcome outcome = enqueue_quiesced ? LoadOutcome::kFailed : LoadOutcome::kUnknown;
    if (!enqueue_quiesced) {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (auto& layer : state->layers) layer.event = nullptr;  // retained for process lifetime
    }
    (void)client->FinishLoad(prepared.id, outcome);
    settle_all(outcome, enqueue_quiesced ? "load enqueue failed"
                                         : "load enqueue failed and stream did not quiesce");
    std::lock_guard<std::mutex> lock(mutex_);
    completed_loads_.emplace_back(prepared.id, outcome);
    if (--load_in_flight_ == 0) idle_cv_.notify_all();
    return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
  }

  std::thread([this, client, prepared, state, load_stream, local_layer = std::move(local_layer)] {
    LoadOutcome final = LoadOutcome::kCompleted;
    const bool all_transfer_engine =
        std::none_of(local_layer.begin(), local_layer.end(), [](bool local) { return local; });
    if (all_transfer_engine) {
      std::vector<TransferPlan> combined;
      for (const auto& layer : prepared.layers) {
        combined.insert(combined.end(), layer.plans.begin(), layer.plans.end());
      }
      std::vector<TransferFailure> failures;
      const bool ok = ExecutePlans(combined, &failures);
      std::string failure_text;
      if (!ok || !failures.empty()) {
        final = std::any_of(failures.begin(), failures.end(),
                            [](const auto& failure) { return !failure.quiesced; })
                    ? LoadOutcome::kUnknown
                    : LoadOutcome::kFailed;
        failure_text = failures.empty() ? "executor returned failure without status"
                                        : DescribeTransferFailures(failures);
        MORI_UMBP_ERROR("[EngineExecutor] load {} failed: {}", prepared.id, failure_text);
      }
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        for (auto& layer : state->layers) {
          layer.done = true;
          layer.outcome = final;
          layer.error = failure_text;
        }
      }
      state->cv.notify_all();
    } else {
      size_t failed_layer = prepared.layers.size();
      std::string failed_text;
      for (size_t i = 0; i < prepared.layers.size(); ++i) {
        std::vector<TransferFailure> failures;
        bool ok = false;
        bool gpu_error = false;
        if (local_layer[i]) {
          int old_device = -1;
          ok =
              hipGetDevice(&old_device) == hipSuccess && hipSetDevice(config_.device) == hipSuccess;
          hipError_t event_status = ok && state->layers[i].event != nullptr
                                        ? hipEventSynchronize(state->layers[i].event)
                                        : hipErrorInvalidResourceHandle;
#ifdef MORI_UMBP_TESTING
          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (fail_load_event_sync_) event_status = hipErrorUnknown;
          }
#endif
          ok = ok && event_status == hipSuccess;
          gpu_error = !ok;
          if (gpu_error) {
            failures.push_back(
                {prepared.layers[i].plans.empty() ? std::vector<size_t>{}
                                                  : prepared.layers[i].plans.front().tags,
                 static_cast<uint32_t>(event_status),
                 std::string("HIP layer event failed: ") + hipGetErrorString(event_status),
                 {},
                 /*quiesced=*/false,
                 /*submitted=*/true});
            std::lock_guard<std::mutex> lock(mutex_);
            unsafe_resources_ = true;
          }
          if (old_device >= 0) (void)hipSetDevice(old_device);
        } else {
          ok = ExecutePlans(prepared.layers[i].plans, &failures);
        }
        std::string failure_text;
        LoadOutcome layer_outcome = LoadOutcome::kCompleted;
        if (!ok || !failures.empty()) {
          layer_outcome = gpu_error || std::any_of(failures.begin(), failures.end(),
                                                   [](const auto& f) { return !f.quiesced; })
                              ? LoadOutcome::kUnknown
                              : LoadOutcome::kFailed;
          failure_text = failures.empty() ? "executor returned failure without status"
                                          : DescribeTransferFailures(failures);
          MORI_UMBP_ERROR("[EngineExecutor] load {} layer {} failed: {}", prepared.id, i,
                          failure_text);
        }
        if (layer_outcome == LoadOutcome::kCompleted) {
          std::lock_guard<std::mutex> lock(state->mutex);
          state->layers[i].done = true;
          state->layers[i].outcome = layer_outcome;
          state->layers[i].error = failure_text;
          state->cv.notify_all();
          continue;
        }
        final = layer_outcome;
        failed_layer = i;
        failed_text = std::move(failure_text);
        break;
      }
      if (failed_layer != prepared.layers.size()) {
        const bool pending_local = std::any_of(local_layer.begin() + failed_layer,
                                               local_layer.end(), [](bool local) { return local; });
        if (pending_local && load_stream != nullptr) {
          bool force_sync_failure = false;
#ifdef MORI_UMBP_TESTING
          {
            std::lock_guard<std::mutex> lock(mutex_);
            force_sync_failure = fail_load_stream_sync_;
          }
#endif
          int old_device = -1;
          const bool stream_quiesced = !force_sync_failure &&
                                       hipGetDevice(&old_device) == hipSuccess &&
                                       hipSetDevice(config_.device) == hipSuccess &&
                                       hipStreamSynchronize(load_stream) == hipSuccess;
          if (old_device >= 0) (void)hipSetDevice(old_device);
          if (!stream_quiesced) {
            final = LoadOutcome::kUnknown;
            failed_text += "; HIP load stream did not quiesce";
            std::lock_guard<std::mutex> executor_lock(mutex_);
            unsafe_resources_ = true;
          }
        }
        {
          std::lock_guard<std::mutex> lock(state->mutex);
          for (size_t remaining = failed_layer; remaining < state->layers.size(); ++remaining) {
            state->layers[remaining].done = true;
            state->layers[remaining].outcome = final;
            state->layers[remaining].error =
                remaining == failed_layer
                    ? failed_text
                    : "not submitted after earlier layer failed: " + failed_text;
            if (final == LoadOutcome::kUnknown) state->layers[remaining].event = nullptr;
          }
        }
        state->cv.notify_all();
      }
    }
    if (!client->FinishLoad(prepared.id, final) && final == LoadOutcome::kCompleted) {
      final = LoadOutcome::kFailed;
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->outcome = final;
      if (final != LoadOutcome::kCompleted) {
        auto failed = std::find_if(state->layers.begin(), state->layers.end(),
                                   [](const auto& layer) { return !layer.error.empty(); });
        state->error = failed == state->layers.end() ? "load failed" : failed->error;
      }
    }
    state->cv.notify_all();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_loads_.emplace_back(prepared.id, final);
      if (--load_in_flight_ == 0) idle_cv_.notify_all();
    }
  }).detach();
  return std::shared_ptr<EngineLoadHandle>(new EngineLoadHandle(std::move(state)));
}

std::vector<std::pair<uint64_t, LoadOutcome>> EngineExecutor::PollLoads() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<uint64_t, LoadOutcome>> out(completed_loads_.begin(),
                                                    completed_loads_.end());
  completed_loads_.clear();
  return out;
}

void EngineExecutor::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (closed_) return;
  closed_ = true;
  idle_cv_.wait(lock, [&] { return in_flight_ == 0 && load_in_flight_ == 0; });
  client_ = nullptr;
  if (unsafe_resources_) {
    peer_directory_ = nullptr;
    return;  // process-lifetime retention: a DMA may still reference every region
  }
  regions_.clear();
  for (const auto& registration : owned_registrations_) transfer_->Deregister(registration);
  owned_registrations_.clear();
  for (auto& imported : imported_regions_) {
    (void)GlobalKvVmmAllocator().FreeImported(&imported.allocation);
  }
  imported_regions_.clear();
  for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
    if (it->hip_registered) (void)hipHostUnregister(it->base);
    if (it->base != nullptr) munmap(it->base, it->mapped_length);
  }
  mappings_.clear();
  peer_directory_ = nullptr;
  transfer_.reset();
  if (completion_stream_ != nullptr) {
    (void)hipStreamDestroy(completion_stream_);
    completion_stream_ = nullptr;
  }
}

#ifdef MORI_UMBP_TESTING
void EngineExecutor::SetBeforeExecuteHookForTest(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(mutex_);
  before_execute_hook_ = std::move(hook);
}

void EngineExecutor::SetExecuteOverrideForTest(
    std::function<bool(const std::vector<TransferPlan>&, std::vector<TransferFailure>*)> hook) {
  std::lock_guard<std::mutex> lock(mutex_);
  execute_override_ = std::move(hook);
}

void EngineExecutor::SetLoadEventCreateFailureForTest(bool fail) {
  std::lock_guard<std::mutex> lock(mutex_);
  fail_load_event_create_ = fail;
}

void EngineExecutor::SetLoadEventSyncFailureForTest(bool fail) {
  std::lock_guard<std::mutex> lock(mutex_);
  fail_load_event_sync_ = fail;
}

void EngineExecutor::SetLoadEnqueueFailureForTest(bool fail) {
  std::lock_guard<std::mutex> lock(mutex_);
  fail_load_enqueue_ = fail;
}

void EngineExecutor::SetLoadStreamSyncFailureForTest(bool fail) {
  std::lock_guard<std::mutex> lock(mutex_);
  fail_load_stream_sync_ = fail;
}

void EngineExecutor::SetImportPeerRegionFailureForTest(bool fail) {
  std::lock_guard<std::mutex> lock(mutex_);
  fail_import_peer_region_ = fail;
}
#endif

}  // namespace mori::umbp::engine
