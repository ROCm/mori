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
#include "umbp/distributed/transfer/posix_engine.h"

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/aligned_buffer.h"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {
namespace {

constexpr uint64_t kAlign = segment::kRecordAlign;

bool IsHostCpu(const TransferRef& r) {
  return !r.IsFile() && r.HasHostPtr() && r.loc == mori::io::MemoryLocationType::CPU;
}

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

struct Segment {
  uint64_t file_off;
  char* mem;
  size_t size;
};

// Moves every byte of `iov` at `offset`, retrying short and interrupted calls.
bool FullIo(int fd, std::vector<iovec> iov, uint64_t offset, bool write) {
  size_t first = 0;
  while (first < iov.size()) {
    const int count = static_cast<int>(std::min<size_t>(iov.size() - first, IOV_MAX));
    const ssize_t n = write ? ::pwritev(fd, &iov[first], count, static_cast<off_t>(offset))
                            : ::preadv(fd, &iov[first], count, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    offset += static_cast<uint64_t>(n);
    for (size_t left = static_cast<size_t>(n); left > 0;) {
      if (left >= iov[first].iov_len) {
        left -= iov[first].iov_len;
        ++first;
      } else {
        iov[first].iov_base = static_cast<char*>(iov[first].iov_base) + left;
        iov[first].iov_len -= left;
        left = 0;
      }
    }
  }
  return true;
}

bool Direct(const Segment& s) {
  return s.file_off % kAlign == 0 && IsDirectIoCompatible(s.mem, s.size);
}

// segs[from, to) through one aligned window.  A read may take bytes around it;
// a write must start aligned and zero-pads its tail, staying within `file_end`.
bool Bounce(int fd, const std::vector<Segment>& segs, size_t from, size_t to, uint64_t file_end,
            bool write) {
  const uint64_t begin = segs[from].file_off & ~(kAlign - 1);
  const uint64_t last = segs[to - 1].file_off + segs[to - 1].size;
  const uint64_t end = segment::AlignUp(last);
  if (write && (begin != segs[from].file_off || end > file_end)) {
    errno = EINVAL;
    return false;
  }
  AlignedBuffer buf(static_cast<size_t>(end - begin));
  if (write) {
    for (size_t i = from; i < to; ++i) {
      std::memcpy(buf.data() + (segs[i].file_off - begin), segs[i].mem, segs[i].size);
    }
    std::memset(buf.data() + (last - begin), 0, static_cast<size_t>(end - last));
    return FullIo(fd, {iovec{buf.data(), buf.size()}}, begin, true);
  }
  if (!FullIo(fd, {iovec{buf.data(), buf.size()}}, begin, false)) return false;
  for (size_t i = from; i < to; ++i) {
    std::memcpy(segs[i].mem, buf.data() + (segs[i].file_off - begin), segs[i].size);
  }
  return true;
}

bool RunPlan(const TransferPlan& plan) {
  const bool write = plan.dst.IsFile();
  const TransferRef& file = write ? plan.dst : plan.src;
  char* mem = static_cast<char*>(write ? plan.src.host_ptr : plan.dst.host_ptr);
  const auto& file_offs = write ? plan.dst_offsets : plan.src_offsets;
  const auto& mem_offs = write ? plan.src_offsets : plan.dst_offsets;

  std::vector<Segment> segs;
  segs.reserve(plan.sizes.size());
  for (size_t i = 0; i < plan.sizes.size(); ++i) {
    segs.push_back({file.file_offset + file_offs[i], mem + mem_offs[i], plan.sizes[i]});
  }
  std::sort(segs.begin(), segs.end(),
            [](const Segment& a, const Segment& b) { return a.file_off < b.file_off; });

  const int flags = ::fcntl(file.file_fd, F_GETFL);
  const bool direct = flags >= 0 && (flags & O_DIRECT) != 0;
  const uint64_t file_end = file.file_offset + file.size;
  for (size_t start = 0; start < segs.size();) {
    size_t end = start + 1;
    while (end < segs.size() && segs[end].file_off == segs[end - 1].file_off + segs[end - 1].size) {
      ++end;
    }
    // One vectored call for the run's O_DIRECT-legal prefix, a bounce for the rest.
    size_t split = start;
    while (split < end && (!direct || Direct(segs[split]))) ++split;
    if (split > start) {
      std::vector<iovec> iov;
      iov.reserve(split - start);
      for (size_t i = start; i < split; ++i) iov.push_back(iovec{segs[i].mem, segs[i].size});
      if (!FullIo(file.file_fd, std::move(iov), segs[start].file_off, write)) return false;
    }
    if (split < end && !Bounce(file.file_fd, segs, split, end, file_end, write)) return false;
    // A buffered write still leaves whole blocks, as the direct bounce does.
    const uint64_t last = segs[end - 1].file_off + segs[end - 1].size;
    const uint64_t padded = segment::AlignUp(last);
    if (write && !direct && padded != last && padded <= file_end) {
      static char zeros[kAlign] = {};
      if (!FullIo(file.file_fd, {iovec{zeros, padded - last}}, last, true)) return false;
    }
    start = end;
  }
  return true;
}

}  // namespace

bool PosixEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  return (src.IsFile() && IsHostCpu(dst)) || (IsHostCpu(src) && dst.IsFile());
}

namespace {

// Pairs by (memory buffer, file range, direction): a file endpoint has no
// host_ptr, so PairPlanBuilder's key would merge different files.
class PosixPlanBuilder final : public TransferPlanBuilder {
 public:
  explicit PosixPlanBuilder(const PosixEngine& engine) : engine_(engine) {}

  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    if (!engine_.CanHandle(item.src, item.dst) || !ItemInBounds(item)) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    const bool write = item.dst.IsFile();
    const TransferRef& file = write ? item.dst : item.src;
    const void* mem = write ? item.src.host_ptr : item.dst.host_ptr;
    auto [it, fresh] = plan_of_.emplace(std::make_tuple(mem, file.file_fd, file.file_offset, write),
                                        out_.plans.size());
    if (fresh) out_.plans.push_back(PlanBetween(item, TransferDirection::kLocal));
    AppendSegment(&out_.plans[it->second], item.src_offset, item.dst_offset, item.size, item.tag);
  }

  TransferPlanSet Finish() override { return std::move(out_); }

 private:
  const PosixEngine& engine_;
  TransferPlanSet out_;
  std::map<std::tuple<const void*, int, uint64_t, bool>, size_t> plan_of_;
};

}  // namespace

std::unique_ptr<TransferPlanBuilder> PosixEngine::CreatePlanBuilder(
    size_t /*expected_items*/) const {
  return std::make_unique<PosixPlanBuilder>(*this);
}

std::unique_ptr<TransferHandle> PosixEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;
  std::vector<TransferFailure> failures;
  for (const auto& plan : plans) {
    if (RunPlan(plan)) continue;
    const int err = errno;
    MORI_UMBP_ERROR("[PosixEngine] {} fd={} failed: errno={}", plan.dst.IsFile() ? "write" : "read",
                    plan.dst.IsFile() ? plan.dst.file_fd : plan.src.file_fd, err);
    failures.push_back(TransferFailure{plan.tags, static_cast<uint32_t>(err),
                                       "PosixEngine: errno=" + std::to_string(err), "posix"});
  }
  return std::make_unique<SettledHandle>(std::move(failures));
}

}  // namespace mori::umbp
