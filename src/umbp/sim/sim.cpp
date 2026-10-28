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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {
namespace {
using MemoryLocation = mori::io::MemoryLocationType;

int PathIndex(const TransferRef& src, const TransferRef& dst) {
  return (src.loc == MemoryLocation::GPU ? 2 : 0) + (dst.loc == MemoryLocation::GPU ? 1 : 0);
}

class CompletedTransfer final : public TransferHandle {
 public:
  void Wait(std::vector<TransferFailure>*) override {}
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
}  // namespace

bool SimPageMemorySource::Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) {
  if (out == nullptr) return false;
  std::vector<Buffer> fresh;
  fresh.reserve(sizes.size());
  buffers_.reserve(buffers_.size() + sizes.size());
  out->reserve(out->size() + sizes.size());
  for (uint64_t size : sizes) {
    if (size == 0) continue;
    void* base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
      for (const auto& buffer : fresh) munmap(buffer.base, buffer.size);
      return false;
    }
    fresh.push_back({base, size});
  }
  buffers_.insert(buffers_.end(), fresh.begin(), fresh.end());
  out->insert(out->end(), fresh.begin(), fresh.end());
  return true;
}

void SimPageMemorySource::Release() {
  for (const auto& buffer : buffers_) munmap(buffer.base, buffer.size);
  buffers_.clear();
}

SimTransferEngine::SimTransferEngine(const std::string& profile_path) {
  std::ifstream input(profile_path);
  if (!input) throw std::invalid_argument("UMBP sim: cannot read profile " + profile_path);
  const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  google::protobuf::Struct profile;
  const auto status = google::protobuf::util::JsonStringToMessage(json, &profile);
  if (!status.ok()) throw std::invalid_argument("UMBP sim profile: " + status.ToString());
  using Value = google::protobuf::Value;
  const auto& id = Field(profile, "id", Value::kStringValue).string_value();
  const auto& paths = Field(profile, "paths", Value::kStructValue).struct_value();
  const std::array<std::string, 4> names = {"cpu_cpu", "cpu_gpu", "gpu_cpu", "gpu_gpu"};
  if (id.empty() || paths.fields().empty() || profile.fields().size() != 2) {
    throw std::invalid_argument("UMBP sim profile requires id and nonempty paths only");
  }
  for (const auto& [name, value] : paths.fields()) {
    const auto at = std::find(names.begin(), names.end(), name);
    if (at == names.end() || value.kind_case() != Value::kStructValue) {
      throw std::invalid_argument("UMBP sim profile: invalid path " + name);
    }
    const auto& object = value.struct_value();
    Path path{Field(object, "channel", Value::kStringValue).string_value(),
              Field(object, "latency_ns", Value::kNumberValue).number_value(),
              Field(object, "bandwidth_bytes_per_second", Value::kNumberValue).number_value()};
    if (object.fields().size() != 3 || path.channel.empty() || !std::isfinite(path.latency_ns) ||
        path.latency_ns < 0 || !std::isfinite(path.bandwidth) || path.bandwidth <= 0) {
      throw std::invalid_argument("UMBP sim profile: invalid coefficients for " + name);
    }
    available_.try_emplace(path.channel);
    paths_[at - names.begin()] = std::move(path);
  }
  MORI_UMBP_INFO("[UMBP sim] profile={} file={}; payload storage and copies are virtual", id,
                 profile_path);
}

bool SimTransferEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  const auto local = [](const TransferRef& ref) {
    return ref.kind == TransferRef::Kind::Memory && ref.HasHostPtr() &&
           (ref.loc == MemoryLocation::CPU || ref.loc == MemoryLocation::GPU);
  };
  return local(src) && local(dst) && paths_[PathIndex(src, dst)].has_value();
}

std::unique_ptr<TransferPlanBuilder> SimTransferEngine::CreatePlanBuilder(size_t) const {
  return MakePairPlanBuilder(*this, [](TransferPlan* plan, const TransferItem& item) {
    AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag);
  });
}

std::unique_ptr<TransferHandle> SimTransferEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;
  std::array<long double, 4> bytes{};
  for (const auto& plan : plans) {
    if (!CanHandle(plan.src, plan.dst) || plan.dir != TransferDirection::kLocal ||
        plan.uses_bounce || plan.sizes.size() != plan.src_offsets.size() ||
        plan.sizes.size() != plan.dst_offsets.size()) {
      return nullptr;
    }
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      if (!RangeFits(plan.src_offsets[i], plan.sizes[i], plan.src.size) ||
          !RangeFits(plan.dst_offsets[i], plan.sizes[i], plan.dst.size))
        return nullptr;
      bytes[PathIndex(plan.src, plan.dst)] += plan.sizes[i];
    }
  }
  // Charge fixed latency once per path in this submit batch, independent of page fragmentation.
  std::unordered_map<std::string, long double> service_ns;
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (bytes[i] == 0) continue;
    const auto& path = *paths_[i];
    service_ns[path.channel] += path.latency_ns + bytes[i] * 1e9L / path.bandwidth;
  }
  auto finish = Clock::now();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = Clock::now();
    // Validate all deadlines before reserving any channel.
    for (const auto& [channel, ns] : service_ns) {
      const auto start = std::max(now, available_.at(channel));
      const auto remaining =
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - start)
              .count();
      if (!std::isfinite(ns) || std::ceil(ns) >= remaining) {
        MORI_UMBP_ERROR("[UMBP sim] modeled transfer duration overflows clock");
        return nullptr;
      }
    }
    for (const auto& [channel, ns] : service_ns) {
      auto& available = available_.at(channel);
      available =
          std::max(now, available) + std::chrono::nanoseconds(static_cast<int64_t>(std::ceil(ns)));
      finish = std::max(finish, available);
    }
  }
  std::this_thread::sleep_until(finish);
  return std::make_unique<CompletedTransfer>();
}

}  // namespace mori::umbp
