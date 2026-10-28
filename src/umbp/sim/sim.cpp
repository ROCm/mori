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
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {
namespace {
using MemoryLocation = mori::io::MemoryLocationType;

int PathIndex(const TransferRef& src, const TransferRef& dst) {
  if (!src.HasHostPtr()) return 4;
  if (!dst.HasHostPtr()) return 5;
  return (src.loc == MemoryLocation::GPU ? 2 : 0) + (dst.loc == MemoryLocation::GPU ? 1 : 0);
}

using Clock = std::chrono::steady_clock;

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
        local_(MakePairPlanBuilder(engine, [](TransferPlan* plan, const TransferItem& item) {
          AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag);
        })) {}
  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    if (!engine_.CanHandle(item.src, item.dst) || !ItemInBounds(item)) {
      rejected_.push_back(item.tag);
      return;
    }
    (PathIndex(item.src, item.dst) < 4 ? local_ : remote_)->Add(item);
  }
  TransferPlanSet Finish() override {
    auto out = local_->Finish();
    auto remote = remote_->Finish();
    for (auto& plan : remote.plans) out.plans.push_back(std::move(plan));
    out.rejected_tags.insert(out.rejected_tags.end(), remote.rejected_tags.begin(),
                             remote.rejected_tags.end());
    out.rejected_tags.insert(out.rejected_tags.end(), rejected_.begin(), rejected_.end());
    return out;
  }

 private:
  const SimTransferEngine& engine_;
  std::unique_ptr<TransferPlanBuilder> remote_, local_;
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
    fresh.push_back({buffer.ptr, size});
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
  const std::array<std::string, 6> names = {"cpu_cpu", "cpu_gpu",     "gpu_cpu",
                                            "gpu_gpu", "remote_read", "remote_write"};
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
    Path path{channel_id, Curve(object, "latency_ns", false),
              Curve(object, "bandwidth_bytes_per_second", true),
              object.fields().count("copy_latency_ns") ? Curve(object, "copy_latency_ns", false)
                                                       : std::vector<double>{0},
              0};
    if (object.fields().size() != 3 + object.fields().count("copy_latency_ns") || channel.empty() ||
        (name != "cpu_cpu" && (path.latency_ns.size() != 1 || path.bandwidth.size() != 1 ||
                               path.copy_latency_ns.size() != 1))) {
      throw std::invalid_argument("UMBP sim profile: invalid coefficients for " + name);
    }
    const size_t threads = name == "cpu_cpu"
                               ? std::max(LocalCopyThreadLimit("UMBP_DRAM_COPY_THREADS"),
                                          LocalCopyThreadLimit("UMBP_DRAM_WRITE_THREADS"))
                               : 1;
    path.capacity = *std::max_element(
        path.bandwidth.begin(), path.bandwidth.begin() + std::min(threads, path.bandwidth.size()));
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
  if (!paths_[4] && !paths_[5]) {
    return MakePairPlanBuilder(*this, [](TransferPlan* plan, const TransferItem& item) {
      AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag);
    });
  }
  return std::make_unique<SimPlanBuilder>(
      *this, MakeRdmaPlanBuilder(engine_key_, bounce_ref_.size, bounce_ref_));
}

std::optional<SimTransferEngine::Clock::time_point> SimTransferEngine::Reserve(
    const std::vector<TransferPlan>& plans, Clock::time_point submitted, bool serial_cpu) {
  // Profiles measure submit-to-completion; model calculation consumes that same budget.
  std::array<long double, 6> bytes{};
  std::array<size_t, 6> copies{};
  for (const auto& plan : plans) {
    const auto expected = plan.src.HasHostPtr() ? (plan.dst.HasHostPtr() ? TransferDirection::kLocal
                                                                         : TransferDirection::kPush)
                                                : TransferDirection::kPull;
    if (!CanHandle(plan.src, plan.dst) || plan.dir != expected || plan.uses_bounce ||
        plan.sizes.size() != plan.src_offsets.size() ||
        plan.sizes.size() != plan.dst_offsets.size()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      if (!RangeFits(plan.src_offsets[i], plan.sizes[i], plan.src.size) ||
          !RangeFits(plan.dst_offsets[i], plan.sizes[i], plan.dst.size))
        return std::nullopt;
      bytes[PathIndex(plan.src, plan.dst)] += plan.sizes[i];
      if (plan.sizes[i] != 0) ++copies[PathIndex(plan.src, plan.dst)];
    }
  }
  // Charge fixed latency once per path in this submit batch, independent of page fragmentation.
  std::array<long double, 6> service_ns{};
  long double private_ns = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (bytes[i] == 0) continue;
    const auto& path = *paths_[i];
    // One submit may fan out internally; independent submits (e.g. BatchPut keys)
    // can overlap, but all reserve service on the same aggregate-bandwidth channel.
    const size_t threads =
        i == 0 && !serial_cpu ? LocalCopyThreads(static_cast<size_t>(std::min<long double>(
                                                     bytes[i], std::numeric_limits<size_t>::max())),
                                                 copies[i])
                              : 1;
    const auto latency =
        AtThreads(path.latency_ns, threads) + copies[i] * AtThreads(path.copy_latency_ns, threads);
    private_ns =
        std::max(private_ns, latency + bytes[i] * 1e9L / AtThreads(path.bandwidth, threads));
    service_ns[path.channel] += latency + bytes[i] * 1e9L / path.capacity;
  }
  auto deadline = [](Clock::time_point start, long double ns) -> std::optional<Clock::time_point> {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - start)
            .count();
    if (!std::isfinite(ns) || std::ceil(ns) >= remaining) return std::nullopt;
    return start + std::chrono::nanoseconds(static_cast<int64_t>(std::ceil(ns)));
  };
  auto finish = deadline(submitted, private_ns);
  if (!finish) return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  auto next = available_;
  for (size_t i = 0; i < service_ns.size(); ++i) {
    if (service_ns[i] == 0) continue;
    const auto end = deadline(std::max(submitted, next[i]), service_ns[i]);
    if (!end) return std::nullopt;
    next[i] = *end;
    finish = std::max(*finish, *end);
  }
  available_ = next;  // Commit only after every deadline has passed overflow validation.
  return finish;
}

std::unique_ptr<TransferHandle> SimTransferEngine::Submit(std::vector<TransferPlan> plans) {
  const auto submitted = Clock::now();
  if (plans.empty()) return nullptr;
  if (std::none_of(plans.begin(), plans.end(), [](const auto& plan) { return plan.uses_bounce; })) {
    const auto finish = Reserve(plans, submitted);
    if (!finish) return nullptr;
    if (std::all_of(plans.begin(), plans.end(),
                    [](const auto& plan) { return plan.dir == TransferDirection::kLocal; }))
      WaitUntil(*finish);
    return std::make_unique<SimHandle>(*finish);
  }
  std::vector<TransferPlan> direct, bounced;
  for (auto& plan : plans) {
    if (plan.uses_bounce && plan.dir == TransferDirection::kLocal) return nullptr;
    (plan.uses_bounce ? bounced : direct).push_back(std::move(plan));
  }
  const auto finish = Reserve(direct, submitted);
  if (!finish) return nullptr;
  auto handle = std::make_unique<SimHandle>(*finish);
  // Reuse mori-io's submit-direct-first, inline-staging ordering. No payload access.
  for (auto& plan : bounced) {
    std::lock_guard<std::mutex> lock(bounce_mutex_);
    std::vector<TransferItem> copies;
    const bool push = plan.dir == TransferDirection::kPush;
    for (const auto& copy : plan.bounce_copies) {
      const auto caller = TransferRef::HostBytes(copy.host, copy.size);
      copies.push_back(
          push ? TransferItem{caller, 0, bounce_ref_, copy.bounce_offset, copy.size, 0}
               : TransferItem{bounce_ref_, copy.bounce_offset, caller, 0, copy.size, 0});
    }
    plan.uses_bounce = false;
    auto stage = [&] {
      auto planned = Plan(copies);
      if (!planned.rejected_tags.empty()) return false;
      // MoriIoEngine stages with serial HostCopyBlock calls, without CPU fan-out.
      const auto deadline = Reserve(planned.plans, Clock::now(), true);
      if (deadline) WaitUntil(*deadline);
      return deadline.has_value();
    };
    bool ok = !copies.empty() && (!push || stage());
    if (ok) {
      const auto deadline = Reserve({plan}, Clock::now());
      ok = deadline.has_value();
      if (ok) WaitUntil(*deadline);
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
