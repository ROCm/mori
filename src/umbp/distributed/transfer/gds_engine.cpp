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
#include "umbp/distributed/transfer/gds_engine.h"

#include <hipfile.h>

#include <cerrno>
#include <string>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

inline bool IsGpu(const TransferRef& r) { return r.loc == mori::io::MemoryLocationType::GPU; }

// Inline-completion handle, same shape as the copy engines': every read runs
// before Submit returns, so the handle only replays the outcome.
class SettledHandle final : public TransferHandle {
 public:
  explicit SettledHandle(std::vector<TransferFailure> failures) : failures_(std::move(failures)) {}

  void Wait(std::vector<TransferFailure>* failures) override {
    if (reported_) return;
    reported_ = true;
    if (failures != nullptr) {
      for (auto& f : failures_) failures->push_back(std::move(f));
    }
    failures_.clear();
  }

 private:
  std::vector<TransferFailure> failures_;
  bool reported_ = false;
};

// Turn a hipFileRead return code into a human-readable reason.  >= 0 is a byte
// count (a short read here); -1 is a POSIX errno; any other negative is the
// negated hipFileOpError_t.
std::string ReadFailureReason(ssize_t n, size_t want) {
  if (n == -1) return std::string("errno=") + std::to_string(errno);
  if (n < 0) return hipFileGetOpErrorString(static_cast<hipFileOpError_t>(-n));
  return std::string("short read ") + std::to_string(n) + "/" + std::to_string(want);
}

}  // namespace

GdsEngine::~GdsEngine() {
  std::lock_guard<std::mutex> lock(handles_mutex_);
  for (auto& [fd, entry] : handles_) {
    if (entry.handle != nullptr) {
      hipFileHandleDeregister(static_cast<hipFileHandle_t>(entry.handle));
    }
  }
  handles_.clear();
}

TransferRef GdsEngine::RegisterFile(int fd, uint64_t offset, uint64_t size) {
  if (fd < 0) return TransferRef{};
  std::lock_guard<std::mutex> lock(handles_mutex_);

  auto it = handles_.find(fd);
  if (it != handles_.end()) {
    ++it->second.refcount;
    return TransferRef::File(fd, offset, size, it->second.handle);
  }

  hipFileDescr_t descr{};
  descr.type = hipFileHandleTypeOpaqueFD;
  descr.handle.fd = fd;
  hipFileHandle_t handle = nullptr;
  hipFileError_t err = hipFileHandleRegister(&handle, &descr);
  if (err.err != hipFileSuccess || handle == nullptr) {
    MORI_UMBP_ERROR("[GdsEngine] hipFileHandleRegister(fd={}) failed: {}", fd,
                    hipFileGetOpErrorString(err.err));
    return TransferRef{};
  }
  handles_.emplace(fd, HandleEntry{static_cast<void*>(handle), 1});
  return TransferRef::File(fd, offset, size, static_cast<void*>(handle));
}

void GdsEngine::Deregister(const TransferRef& ref) {
  if (!ref.IsFile()) return;
  std::lock_guard<std::mutex> lock(handles_mutex_);
  auto it = handles_.find(ref.file_fd);
  if (it == handles_.end()) return;
  if (--it->second.refcount <= 0) {
    if (it->second.handle != nullptr) {
      hipFileHandleDeregister(static_cast<hipFileHandle_t>(it->second.handle));
    }
    handles_.erase(it);
  }
}

bool GdsEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  // Read path only: a file source into a device buffer.  A file destination or
  // a host destination is not ours.
  return src.IsFile() && !dst.IsFile() && dst.HasHostPtr() && IsGpu(dst);
}

TransferPlanSet GdsEngine::Plan(const std::vector<TransferItem>& items) const {
  TransferPlanSet out;
  out.plans.reserve(items.size());
  for (const auto& item : items) {
    if (item.size == 0) continue;
    if (!CanHandle(item.src, item.dst)) {
      out.rejected_tags.push_back(item.tag);
      continue;
    }
    // Bound the read to the destination buffer: an overrun into a device pool
    // is silent corruption, not a segfault, so reject it here rather than let
    // hipFileRead scribble past the slot.
    if (item.size > item.dst.size || item.dst_offset > item.dst.size - item.size) {
      out.rejected_tags.push_back(item.tag);
      continue;
    }
    TransferPlan plan;
    plan.src = item.src;
    plan.dst = item.dst;
    plan.dir = TransferDirection::kLocal;
    plan.src_offsets.push_back(static_cast<size_t>(item.src_offset));
    plan.dst_offsets.push_back(static_cast<size_t>(item.dst_offset));
    plan.sizes.push_back(static_cast<size_t>(item.size));
    plan.tags.push_back(item.tag);
    out.plans.push_back(std::move(plan));
  }
  return out;
}

std::unique_ptr<TransferHandle> GdsEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;
  std::vector<TransferFailure> failures;

  for (const auto& plan : plans) {
    auto fh = static_cast<hipFileHandle_t>(plan.src.gds_handle);
    if (fh == nullptr) {
      failures.push_back(
          TransferFailure{plan.tags, 0, "GdsEngine: file ref has no hipFile handle", "gds"});
      continue;
    }
    // buffer_base is the GPU allocation base; buffer_offset selects the slot
    // inside it, and file_offset is the range's absolute position on disk.
    void* dst_base = plan.dst.host_ptr;
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      const uint64_t file_off = plan.src.file_offset + plan.src_offsets[i];
      const ssize_t n = hipFileRead(fh, dst_base, plan.sizes[i], static_cast<hoff_t>(file_off),
                                    static_cast<hoff_t>(plan.dst_offsets[i]));
      if (n < 0 || static_cast<size_t>(n) != plan.sizes[i]) {
        const std::string why = ReadFailureReason(n, plan.sizes[i]);
        MORI_UMBP_ERROR("[GdsEngine] hipFileRead file_off={} size={} failed: {}", file_off,
                        plan.sizes[i], why);
        failures.push_back(TransferFailure{plan.tags, 0, "GdsEngine: hipFileRead: " + why, "gds"});
        break;  // one failure per plan; a partially-read key is failed wholesale
      }
    }
  }
  return std::make_unique<SettledHandle>(std::move(failures));
}

}  // namespace mori::umbp
