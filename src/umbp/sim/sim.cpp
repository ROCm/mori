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
#include "umbp/sim/sim.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/common/host_registration.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/distributed/transfer/plan_builder.h"
#include "umbp/local/tiers/segment/segment_format.h"

namespace mori::umbp {
namespace {
using MemoryLocation = mori::io::MemoryLocationType;

int PathIndex(const TransferRef& src, const TransferRef& dst) {
  if (src.IsFile()) return dst.loc == MemoryLocation::GPU ? 12 : 10;
  if (dst.IsFile()) return src.loc == MemoryLocation::GPU ? -1 : 11;
  if (!src.HasHostPtr()) return 4;
  if (!dst.HasHostPtr()) return 5;
  return (src.loc == MemoryLocation::GPU ? 2 : 0) + (dst.loc == MemoryLocation::GPU ? 1 : 0);
}

void AppendLocal(TransferPlan* plan, const TransferItem& item) {
  const int path = PathIndex(item.src, item.dst);
  if (path == 1 || path == 2)
    AppendWithinHostPieces(plan, item, HostRegistrationChunkBytes());
  else
    AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag);
}

using Clock = std::chrono::steady_clock;

std::optional<Clock::time_point> Deadline(Clock::time_point start, long double ns) {
  const auto remaining =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - start)
          .count();
  if (!std::isfinite(ns) || ns < 0 || std::ceil(ns) >= remaining) return std::nullopt;
  return start + std::chrono::nanoseconds(static_cast<int64_t>(std::ceil(ns)));
}

void WaitUntil(Clock::time_point finish) {
  constexpr auto spin_tail = std::chrono::microseconds{100};
  if (finish - Clock::now() > spin_tail) std::this_thread::sleep_until(finish - spin_tail);
  while (Clock::now() < finish) {
  }
}

class SimHandle final : public TransferHandle {
 public:
  explicit SimHandle(Clock::time_point finish) : finish_(finish) {}
  ~SimHandle() override { Wait(nullptr); }
  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    drained_ = true;
    WaitUntil(finish_);
    if (failures) failures->insert(failures->end(), failures_.begin(), failures_.end());
  }
  std::vector<TransferFailure> failures_;

 private:
  Clock::time_point finish_;
  bool drained_ = false;
};

class SimPlanBuilder final : public TransferPlanBuilder {
 public:
  SimPlanBuilder(const SimTransferEngine& engine, std::unique_ptr<TransferPlanBuilder> remote)
      : engine_(engine),
        remote_(std::move(remote)),
        files_(MakeFilePlanBuilder(engine)),
        local_(MakePairPlanBuilder(engine, AppendLocal)) {}
  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    if (item.src.IsFile() || item.dst.IsFile()) {
      files_->Add(item);
      return;
    }
    if (!engine_.CanHandle(item.src, item.dst) || !ItemInBounds(item)) {
      rejected_.push_back(item.tag);
      return;
    }
    (PathIndex(item.src, item.dst) < 4 ? local_ : remote_)->Add(item);
  }
  TransferPlanSet Finish() override {
    auto out = local_->Finish();
    auto files = files_->Finish();
    for (auto& plan : files.plans) out.plans.push_back(std::move(plan));
    out.rejected_tags.insert(out.rejected_tags.end(), files.rejected_tags.begin(),
                             files.rejected_tags.end());
    auto remote = remote_->Finish();
    for (auto& plan : remote.plans) out.plans.push_back(std::move(plan));
    out.rejected_tags.insert(out.rejected_tags.end(), remote.rejected_tags.begin(),
                             remote.rejected_tags.end());
    out.rejected_tags.insert(out.rejected_tags.end(), rejected_.begin(), rejected_.end());
    return out;
  }

 private:
  const SimTransferEngine& engine_;
  std::unique_ptr<TransferPlanBuilder> remote_, files_, local_;
  std::vector<size_t> rejected_;
};

const google::protobuf::Value& Field(const google::protobuf::Struct& object,
                                     const std::string& name,
                                     google::protobuf::Value::KindCase kind) {
  const auto it = object.fields().find(name);
  if (it == object.fields().end() || it->second.kind_case() != kind) {
    throw std::invalid_argument("UMBP sim profile: missing or invalid " + name);
  }
  return it->second;
}

std::vector<double> Curve(const google::protobuf::Struct& object, const std::string& name,
                          bool positive) {
  const auto it = object.fields().find(name);
  if (it == object.fields().end()) throw std::invalid_argument("UMBP sim profile: missing " + name);
  std::vector<double> values;
  auto append = [&](const google::protobuf::Value& value) {
    const double n = value.number_value();
    if (value.kind_case() != google::protobuf::Value::kNumberValue || !std::isfinite(n) ||
        (positive ? n <= 0 : n < 0))
      throw std::invalid_argument("UMBP sim profile: invalid " + name);
    values.push_back(n);
  };
  if (it->second.kind_case() == google::protobuf::Value::kListValue) {
    for (const auto& value : it->second.list_value().values()) append(value);
  } else {
    append(it->second);
  }
  if (values.empty()) throw std::invalid_argument("UMBP sim profile: empty " + name);
  return values;
}

double AtThreads(const std::vector<double>& curve, size_t threads) {
  return curve[std::min(threads, curve.size()) - 1];
}
}  // namespace

HostBufferHandle AllocateSimMemory(size_t size) {
  HostBufferHandle handle;
  if (size == 0) return handle;
  void* ptr = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (ptr != MAP_FAILED) {
    handle.ptr = ptr;
    handle.requested_size = handle.mapped_size = size;
    handle.actual_alignment = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  }
  return handle;
}

bool SimPageMemorySource::Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) {
  if (out == nullptr) return false;
  std::vector<Buffer> fresh;
  fresh.reserve(sizes.size());
  buffers_.reserve(buffers_.size() + sizes.size());
  out->reserve(out->size() + sizes.size());
  for (uint64_t size : sizes) {
    if (size == 0) continue;
    const auto buffer = AllocateSimMemory(size);
    if (!buffer.valid()) {
      for (const auto& buffer : fresh) munmap(buffer.base, buffer.size);
      return false;
    }
    fresh.push_back({buffer.ptr, size,
                     loc_ == mori::io::MemoryLocationType::CPU ? buffer.actual_alignment : 0});
  }
  buffers_.insert(buffers_.end(), fresh.begin(), fresh.end());
  out->insert(out->end(), fresh.begin(), fresh.end());
  return true;
}

void SimPageMemorySource::Release() {
  for (const auto& buffer : buffers_) munmap(buffer.base, buffer.size);
  buffers_.clear();
}

SimTransferEngine::SimTransferEngine(const std::string& profile_path, std::string node_id,
                                     uint64_t bounce_bytes) {
  static std::atomic<uint64_t> generation{0};
  engine_key_ = "umbp-sim/" + node_id + "/" + std::to_string(getpid()) + "/" +
                std::to_string(Clock::now().time_since_epoch().count()) + "/" +
                std::to_string(generation.fetch_add(1));
  std::ifstream input(profile_path);
  if (!input) throw std::invalid_argument("UMBP sim: cannot read profile " + profile_path);
  const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  google::protobuf::Struct profile;
  const auto status = google::protobuf::util::JsonStringToMessage(json, &profile);
  if (!status.ok()) throw std::invalid_argument("UMBP sim profile: " + status.ToString());
  using Value = google::protobuf::Value;
  const auto& id = Field(profile, "id", Value::kStringValue).string_value();
  const auto& paths = Field(profile, "paths", Value::kStructValue).struct_value();
  const std::array<std::string, kModelCount> names = {
      "cpu_cpu",        "cpu_gpu",         "gpu_cpu",         "gpu_gpu",
      "remote_read",    "remote_write",    "cpu_staging",     "cpu_staging_read",
      "cpu_gpu_gather", "gpu_cpu_gather",  "file_cpu",        "cpu_file",
      "file_gpu",       "file_cpu_staged", "cpu_file_staged", "file_cpu_staged_no_crc"};
  if (id.empty() || paths.fields().empty() || profile.fields().size() != 2) {
    throw std::invalid_argument("UMBP sim profile requires id and nonempty paths only");
  }
  std::vector<std::string> channels;
  for (const auto& [name, value] : paths.fields()) {
    const auto at = std::find(names.begin(), names.end(), name);
    if (at == names.end() || value.kind_case() != Value::kStructValue) {
      throw std::invalid_argument("UMBP sim profile: invalid path " + name);
    }
    const auto& object = value.struct_value();
    const auto& channel = Field(object, "channel", Value::kStringValue).string_value();
    const size_t channel_id =
        std::find(channels.begin(), channels.end(), channel) - channels.begin();
    if (channel_id == channels.size()) channels.push_back(channel);
    const bool registration = object.fields().count("registration_latency_ns");
    const double registration_ns =
        registration ? Field(object, "registration_latency_ns", Value::kNumberValue).number_value()
                     : 0;
    const bool logarithmic = object.fields().count("log_size_latency_ns");
    const double log_ns =
        logarithmic ? Field(object, "log_size_latency_ns", Value::kNumberValue).number_value() : 0;
    const bool batch_logarithmic = object.fields().count("batch_log_size_latency_ns");
    const double batch_log_ns =
        batch_logarithmic
            ? Field(object, "batch_log_size_latency_ns", Value::kNumberValue).number_value()
            : 0;
    const size_t index = at - names.begin();
    const bool serial = object.fields().count("serialize_latency");
    const bool serialize_latency =
        serial && Field(object, "serialize_latency", Value::kBoolValue).bool_value();
    Path path{channel_id,
              Curve(object, "latency_ns", false),
              Curve(object, "bandwidth_bytes_per_second", true),
              object.fields().count("copy_latency_ns") ? Curve(object, "copy_latency_ns", false)
                                                       : std::vector<double>{0},
              0,
              registration_ns,
              log_ns,
              batch_log_ns,
              serialize_latency};
    if (object.fields().size() != 3 + object.fields().count("copy_latency_ns") + registration +
                                      logarithmic + batch_logarithmic + serial ||
        (registration && name != "file_gpu") || !std::isfinite(registration_ns) ||
        ((logarithmic || batch_logarithmic || serial) && index != 10 && index != 12 &&
         index != 13 && index != 15) ||
        !std::isfinite(log_ns) || log_ns < 0 || !std::isfinite(batch_log_ns) || batch_log_ns < 0 ||
        registration_ns < 0 || channel.empty() ||
        (name != "cpu_cpu" && (path.latency_ns.size() != 1 || path.bandwidth.size() != 1 ||
                               path.copy_latency_ns.size() != 1))) {
      throw std::invalid_argument("UMBP sim profile: invalid coefficients for " + name);
    }
    // Thread limits apply per call; independent callers share the measured aggregate capacity.
    path.capacity = *std::max_element(path.bandwidth.begin(), path.bandwidth.end());
    paths_[at - names.begin()] = std::move(path);
  }
  if (bounce_bytes > 0 && (paths_[4] || paths_[5])) {
    std::vector<PageMemorySource::Buffer> buffers;
    if (!bounce_memory_.Allocate({bounce_bytes}, &buffers)) throw std::bad_alloc();
    bounce_ref_ = RegisterMemory(buffers[0].base, bounce_bytes, MemoryLocation::CPU, -1);
  }
  MORI_UMBP_INFO("[UMBP sim] profile={} file={}; payload storage and copies are virtual", id,
                 profile_path);
}

bool SimTransferEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  const auto local = [](const TransferRef& ref) {
    return ref.kind == TransferRef::Kind::Memory && ref.HasHostPtr() &&
           (ref.loc == MemoryLocation::CPU || ref.loc == MemoryLocation::GPU);
  };
  if (src.kind == TransferRef::Kind::File || dst.kind == TransferRef::Kind::File) {
    const bool read = src.IsFile() && local(dst);
    const bool write = dst.IsFile() && local(src) && src.loc == MemoryLocation::CPU;
    return (read || write) && paths_[PathIndex(src, dst)].has_value();
  }
  if (local(src) && local(dst)) return paths_[PathIndex(src, dst)].has_value();
  if (local(src) == local(dst)) return false;
  const auto& remote = local(src) ? dst : src;
  const auto& caller = local(src) ? src : dst;
  if (remote.kind != TransferRef::Kind::Memory || remote.HasHostPtr() || !remote.HasMemoryDesc() ||
      remote.size != remote.mem.size ||
      (remote.loc != MemoryLocation::CPU && remote.loc != MemoryLocation::GPU) ||
      !paths_[PathIndex(src, dst)])
    return false;
  if (caller.HasMemoryDesc() ? caller.mem.engineKey != engine_key_
                             : (caller.loc != MemoryLocation::CPU || !bounce_ref_.Valid()))
    return false;
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  return std::any_of(remotes_.begin(), remotes_.end(), [&](const auto& node) {
    return node.second.engine_key == remote.mem.engineKey;
  });
}

std::unique_ptr<TransferPlanBuilder> SimTransferEngine::CreatePlanBuilder(size_t) const {
  if (!paths_[4] && !paths_[5] && !paths_[10] && !paths_[11] && !paths_[12]) {
    return MakePairPlanBuilder(*this, AppendLocal);
  }
  return std::make_unique<SimPlanBuilder>(
      *this, MakeRdmaPlanBuilder(engine_key_, bounce_ref_.size, bounce_ref_));
}

TransferRef SimTransferEngine::RegisterFile(int fd, uint64_t offset, uint64_t size) {
  if (fd < 0) return {};
  if (!paths_[12]) return TransferRef::File(fd, offset, size);
  std::lock_guard<std::mutex> lock(file_mutex_);
  auto& refs = file_refs_[fd];
  if (refs == 0) {
    const auto started = Clock::now();
    const auto ns = std::ceil(paths_[12]->registration_latency_ns);
    const auto remaining =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - started)
            .count();
    if (ns >= remaining) {
      file_refs_.erase(fd);
      return {};
    }
    WaitUntil(started + std::chrono::nanoseconds(static_cast<int64_t>(ns)));
  }
  ++refs;
  return TransferRef::File(fd, offset, size,
                           reinterpret_cast<void*>(static_cast<uintptr_t>(fd) + 1));
}

void SimTransferEngine::Deregister(const TransferRef& ref) {
  if (!ref.IsFile() || !ref.gds_handle) return;
  std::lock_guard<std::mutex> lock(file_mutex_);
  auto it = file_refs_.find(ref.file_fd);
  if (it != file_refs_.end() && --it->second == 0) file_refs_.erase(it);
}

void SimTransferEngine::AddHostGatherRegion(void* base, size_t size) {
  if (!base || size == 0 || !HostRegistrationEnabled()) return;
  std::lock_guard<std::mutex> lock(gather_mutex_);
  gather_regions_.emplace_back(reinterpret_cast<uintptr_t>(base), size);
}

void SimTransferEngine::ClearHostGatherRegions() {
  std::lock_guard<std::mutex> lock(gather_mutex_);
  gather_regions_.clear();
}

std::optional<SimTransferEngine::Reservation> SimTransferEngine::Reserve(
    const std::vector<TransferPlan>& plans, Clock::time_point submitted) {
  // Profiles measure submit-to-completion; model calculation consumes that same budget.
  Bytes bytes{};
  Copies copies{};
  struct GatherBucket {
    Bytes bytes{};
    Copies copies{};
  };
  std::map<int, GatherBucket> gather;
  const bool has_gather = paths_[8] || paths_[9];
  std::unique_lock<std::mutex> gather_lock(gather_mutex_, std::defer_lock);
  if (has_gather) gather_lock.lock();
  for (const auto& plan : plans) {
    const bool file = plan.src.IsFile() || plan.dst.IsFile();
    const auto expected =
        file ? TransferDirection::kLocal
        : plan.src.HasHostPtr()
            ? (plan.dst.HasHostPtr() ? TransferDirection::kLocal : TransferDirection::kPush)
            : TransferDirection::kPull;
    if (!CanHandle(plan.src, plan.dst) || plan.dir != expected || plan.uses_bounce ||
        plan.sizes.size() != plan.src_offsets.size() ||
        plan.sizes.size() != plan.dst_offsets.size()) {
      return std::nullopt;
    }
    const size_t path = PathIndex(plan.src, plan.dst);
    bool eligible = has_gather && (path == 1 || path == 2) && DeviceGatherEnabled();
    const auto& host = path == 1 ? plan.src : plan.dst;
    const int device = path == 1 ? plan.dst.device : plan.src.device;
    eligible = eligible && device >= 0;
    long double plan_bytes = 0;
    size_t plan_copies = 0;
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      if ((path != 12 && !RangeFits(plan.src_offsets[i], plan.sizes[i], plan.src.size)) ||
          !RangeFits(plan.dst_offsets[i], plan.sizes[i], plan.dst.size))
        return std::nullopt;
      if (file) {
        const auto& ref = plan.src.IsFile() ? plan.src : plan.dst;
        const uint64_t offset = plan.src.IsFile() ? plan.src_offsets[i] : plan.dst_offsets[i];
        constexpr uint64_t max_offset = std::numeric_limits<int64_t>::max();
        if (ref.file_offset > max_offset || offset > max_offset - ref.file_offset ||
            plan.sizes[i] > max_offset - ref.file_offset - offset ||
            (path == 12 && !ref.gds_handle))
          return std::nullopt;
      }
      plan_bytes += plan.sizes[i];
      if (plan.sizes[i] != 0) ++plan_copies;
      if (eligible) {
        const size_t offset = path == 1 ? plan.src_offsets[i] : plan.dst_offsets[i];
        const auto base = reinterpret_cast<uintptr_t>(host.host_ptr);
        eligible =
            offset <= std::numeric_limits<uintptr_t>::max() - base &&
            std::any_of(gather_regions_.begin(), gather_regions_.end(), [&](const auto& r) {
              const auto address = base + offset;
              return address >= r.first && RangeFits(address - r.first, plan.sizes[i], r.second);
            });
      }
    }
    bytes[path] += plan_bytes;
    copies[path] += plan_copies;
    if (eligible) {
      gather[device].bytes[path] += plan_bytes;
      gather[device].copies[path] += plan_copies;
    }
  }
  // Real HBM gathers eligible plans per device, including both H2D and D2H.
  for (const auto& [device, bucket] : gather) {
    const auto total_bytes = bucket.bytes[1] + bucket.bytes[2];
    const size_t total_copies = bucket.copies[1] + bucket.copies[2];
    if (total_bytes > std::numeric_limits<size_t>::max() ||
        !GatherShapeEligible(total_copies, static_cast<size_t>(total_bytes)))
      continue;
    for (size_t path : {1, 2}) {
      const size_t variant = path + 7;
      if (!paths_[variant]) continue;  // Old profiles retain their original curve.
      bytes[path] -= bucket.bytes[path];
      copies[path] -= bucket.copies[path];
      bytes[variant] += bucket.bytes[path];
      copies[variant] += bucket.copies[path];
    }
  }
  if (gather_lock.owns_lock()) gather_lock.unlock();
  // hipFileRead is synchronous per copy; only pure GDS batches share byte service.
  if (bytes[12] > 0 && !paths_[12]->serialize_latency &&
      std::count_if(bytes.begin(), bytes.end(), [](auto n) { return n > 0; }) == 1) {
    if (!SimulateGds(plans, bytes[12], copies[12], submitted)) return std::nullopt;
    const auto completed = Clock::now();
    return Reservation{completed, completed};
  }
  return ReserveCosts(bytes, copies, submitted);
}

void SimTransferEngine::AdvanceGds(Clock::time_point now) {
  long double elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - gds_updated_).count();
  while (elapsed > 0 && !gds_work_.empty()) {
    const auto first = *std::min_element(gds_work_.begin(), gds_work_.end(), [](auto a, auto b) {
      return a->remaining_ns < b->remaining_ns;
    });
    const auto step = std::min(elapsed, first->remaining_ns * gds_work_.size());
    for (auto* work : gds_work_)
      work->remaining_ns = std::max(0.0L, work->remaining_ns - step / gds_work_.size());
    elapsed -= step;
    gds_work_.erase(std::remove_if(gds_work_.begin(), gds_work_.end(),
                                   [](auto* work) {
                                     if (work->remaining_ns > 1e-6L) return false;
                                     work->done = true;
                                     return true;
                                   }),
                    gds_work_.end());
  }
  gds_updated_ = now;
}

bool SimTransferEngine::WaitGds(long double service_ns) {
  GdsWork work{service_ns};
  std::unique_lock<std::mutex> lock(mutex_);
  const auto channel = paths_[12]->channel;
  for (;;) {
    const auto now = Clock::now();
    AdvanceGds(now);
    if (available_[channel] <= now) break;  // Ordinary same-channel reservations precede us.
    const auto ready = available_[channel];
    lock.unlock();
    WaitUntil(ready);
    lock.lock();
  }
  auto total_ns = service_ns;
  for (auto* active : gds_work_) total_ns += active->remaining_ns;
  if (!Deadline(Clock::now(), total_ns)) return false;
  gds_work_.push_back(&work);
  auto remove = [&] {
    gds_work_.erase(std::remove(gds_work_.begin(), gds_work_.end(), &work), gds_work_.end());
  };
  try {
    while (!work.done) {
      const auto now = Clock::now();
      AdvanceGds(now);
      if (work.done) break;
      const auto first = *std::min_element(gds_work_.begin(), gds_work_.end(), [](auto a, auto b) {
        return a->remaining_ns < b->remaining_ns;
      });
      const auto finish = Deadline(now, first->remaining_ns * gds_work_.size());
      if (!finish) {
        remove();
        return false;
      }
      lock.unlock();
      WaitUntil(*finish);
      lock.lock();  // A new reader may have delayed this deadline; always recheck work.
    }
  } catch (...) {
    if (!lock.owns_lock()) lock.lock();
    AdvanceGds(Clock::now());
    remove();
    throw;
  }
  return true;
}

bool SimTransferEngine::SimulateGds(const std::vector<TransferPlan>& plans, long double bytes,
                                    size_t copies, Clock::time_point submitted) {
  const auto& path = *paths_[12];
  const auto log = std::log2(std::max(1.0L, bytes / copies / segment::kRecordAlign));
  const auto fixed = path.latency_ns[0] + path.batch_log_size_latency_ns * log;
  const auto latency = path.copy_latency_ns[0] + path.log_size_latency_ns * log;
  const auto begin = Deadline(submitted, fixed);
  if (!begin || !Deadline(submitted, fixed + copies * latency + bytes * 1e9L / path.capacity))
    return false;
  WaitUntil(*begin);
  auto started = *begin;
  for (const auto& plan : plans) {
    for (auto size : plan.sizes) {
      if (size == 0) continue;
      const auto ready = Deadline(started, latency);
      if (!ready) return false;
      WaitUntil(*ready);
      if (!WaitGds(size * 1e9L / path.capacity)) return false;
      started = Clock::now();  // Next hipFileRead cannot start before this copy completes.
    }
  }
  return true;
}

std::optional<SimTransferEngine::Reservation> SimTransferEngine::ReserveCosts(
    const Bytes& bytes, const Copies& copies, Clock::time_point submitted, bool serial_cpu) {
  // Charge fixed latency once per path in this submit batch, independent of page fragmentation.
  Bytes service_ns{}, local_service_ns{}, post_ns{};
  long double private_ns = 0, local_private_ns = 0;
  const bool ssd_write = bytes[11] != 0 || bytes[14] != 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (bytes[i] == 0) continue;
    if (!paths_[i]) return std::nullopt;
    const auto& path = *paths_[i];
    // One submit may fan out internally; independent submits (e.g. BatchPut keys)
    // can overlap, but all reserve service on the same aggregate-bandwidth channel.
    const size_t threads =
        i == 0 && !serial_cpu ? LocalCopyThreads(static_cast<size_t>(std::min<long double>(
                                                     bytes[i], std::numeric_limits<size_t>::max())),
                                                 copies[i])
                              : 1;
    const auto latency_ns = [&](const Path& curve) {
      return AtThreads(curve.latency_ns, threads) +
             copies[i] * AtThreads(curve.copy_latency_ns, threads) +
             (curve.log_size_latency_ns == 0 && curve.batch_log_size_latency_ns == 0
                  ? 0
                  : (copies[i] * curve.log_size_latency_ns + curve.batch_log_size_latency_ns) *
                        std::log2(std::max(1.0L, bytes[i] / copies[i] / segment::kRecordAlign)));
    };
    const auto latency = latency_ns(path);
    const auto duration = latency + bytes[i] * 1e9L / AtThreads(path.bandwidth, threads);
    private_ns = std::max(private_ns, duration);
    // Independent SSD readers overlap latency; mixed read/write batches stay serial.
    auto service =
        (!ssd_write && !path.serialize_latency && (i == 10 || i == 12 || i == 13 || i == 15)
             ? 0
             : latency) +
        bytes[i] * 1e9L / path.capacity;
    // Native SSD I/O is serialized; checksum work runs afterward without the I/O lock.
    if (i == 13 && !ssd_write && path.serialize_latency && paths_[15] &&
        paths_[15]->serialize_latency && paths_[15]->channel == path.channel) {
      const auto& no_crc = *paths_[15];
      const auto io_ns = latency_ns(no_crc) + bytes[i] * 1e9L / no_crc.capacity;
      service = std::min(service, io_ns);  // Fitted curves may cross; never charge negative CRC.
      post_ns[path.channel] += duration - service;
    }
    service_ns[path.channel] += service;
    if (i < 4 || i >= 6) {
      local_private_ns = std::max(local_private_ns, duration);
      local_service_ns[path.channel] += service;
    }
  }
  auto finish = Deadline(submitted, private_ns);
  auto local_finish = Deadline(submitted, local_private_ns);
  if (!finish || !local_finish) return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  auto next = available_;
  if (paths_[12] && service_ns[paths_[12]->channel] > 0 && !gds_work_.empty()) {
    const auto now = Clock::now();
    AdvanceGds(now);
    long double remaining = 0;
    for (auto* work : gds_work_) remaining += work->remaining_ns;
    const auto drained = Deadline(now, remaining);
    if (!drained) return std::nullopt;
    next[paths_[12]->channel] = std::max(next[paths_[12]->channel], *drained);
  }
  for (size_t i = 0; i < service_ns.size(); ++i) {
    if (service_ns[i] == 0) continue;
    const auto start = std::max(submitted, next[i]);
    const auto end = Deadline(start, service_ns[i]);
    if (!end) return std::nullopt;
    // Local paths precede remote paths on a shared channel, but only the local
    // prefix must finish inline. Reserve the entire batch atomically below.
    if (local_service_ns[i] > 0) {
      const auto local_end = Deadline(start, local_service_ns[i] + post_ns[i]);
      if (!local_end) return std::nullopt;
      local_finish = std::max(*local_finish, *local_end);
    }
    const auto completed = Deadline(start, service_ns[i] + post_ns[i]);
    if (!completed) return std::nullopt;
    next[i] = *end;  // Other I/O may start while this caller verifies its checksum.
    finish = std::max(*finish, *completed);
  }
  available_ = next;  // Commit only after every deadline passed overflow validation.
  return Reservation{*finish, *local_finish};
}

bool SimTransferEngine::SimulateStorage(bool read, uint64_t size, size_t operations, bool staged) {
  const auto started = Clock::now();
  if (size == 0) return true;
  if (operations == 0) return false;
  Bytes bytes{};
  Copies copies{};
  const size_t base = read ? 10 : 11;
  const size_t variant = read ? (staged ? 13 : 15) : 14;
  const size_t fallback = read ? 13 : 14;
  const size_t path = (staged || read) && paths_[variant] ? variant
                      : paths_[base]                      ? base
                                                          : fallback;
  bytes[path] = size;
  copies[path] = operations;
  const auto reservation = ReserveCosts(bytes, copies, started);
  if (!reservation) return false;
  WaitUntil(reservation->finish);
  return true;
}

std::unique_ptr<TransferHandle> SimTransferEngine::Submit(std::vector<TransferPlan> plans) {
  const auto submitted = Clock::now();
  if (plans.empty()) return nullptr;
  if (std::none_of(plans.begin(), plans.end(), [](const auto& plan) { return plan.uses_bounce; })) {
    const auto reservation = Reserve(plans, submitted);
    if (!reservation) return nullptr;
    WaitUntil(reservation->local_finish);
    return std::make_unique<SimHandle>(reservation->finish);
  }
  std::vector<TransferPlan> direct, bounced;
  for (auto& plan : plans) {
    if (plan.uses_bounce && plan.dir == TransferDirection::kLocal) return nullptr;
    (plan.uses_bounce ? bounced : direct).push_back(std::move(plan));
  }
  const auto reservation = Reserve(direct, submitted);
  if (!reservation) return nullptr;
  WaitUntil(reservation->local_finish);
  auto handle = std::make_unique<SimHandle>(reservation->finish);
  // Reuse mori-io's submit-direct-first, inline-staging ordering. No payload access.
  for (auto& plan : bounced) {
    std::lock_guard<std::mutex> lock(bounce_mutex_);
    const bool push = plan.dir == TransferDirection::kPush;
    plan.uses_bounce = false;
    auto stage = [&] {
      const auto started = Clock::now();
      Bytes bytes{};
      Copies copies{};
      const size_t path = !push && paths_[7] ? 7 : paths_[6] ? 6 : 0;
      for (const auto& copy : plan.bounce_copies) {
        if (!copy.host || !RangeFits(copy.bounce_offset, copy.size, bounce_ref_.size)) return false;
        bytes[path] += copy.size;
        if (copy.size != 0) ++copies[path];
      }
      // MoriIoEngine stages with serial memcpy calls, without CPU fan-out.
      const auto next = ReserveCosts(bytes, copies, started, true);
      if (next) WaitUntil(next->finish);
      return next.has_value();
    };
    bool ok = !plan.bounce_copies.empty() && (!push || stage());
    if (ok) {
      const auto started = Clock::now();
      const auto next = Reserve({plan}, started);
      if (next) WaitUntil(next->finish);
      ok = next.has_value();
    }
    if (ok && !push) ok = stage();
    if (!ok) {
      TransferFailure failure;
      failure.tags = plan.tags;
      failure.message = "simulated staged transfer failed";
      handle->failures_.push_back(std::move(failure));
    }
  }
  return handle;
}

}  // namespace mori::umbp
