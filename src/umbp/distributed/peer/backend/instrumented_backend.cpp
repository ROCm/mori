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
#include "umbp/distributed/peer/backend/instrumented_backend.h"

#include <chrono>
#include <utility>

namespace mori::umbp {
namespace {

constexpr double kNanosPerSecond = 1e9;

// Label text for the enum indices.  These strings ARE the dashboard's contract:
// a panel groups by op= and status=, so changing one renames a series.
const char* OpName(int op) {
  switch (op) {
    case 0:
      return "allocate";
    case 1:
      return "commit";
    case 2:
      return "abort";
    case 3:
      return "resolve";
    case 4:
      return "evict";
    default:
      return "unknown";
  }
}

const char* StatusName(int status) {
  switch (status) {
    case 0:
      return "ok";
    case 1:
      return "exists";
    case 2:
      return "no_space";
    case 3:
      return "miss";
    case 4:
      return "failed";
    default:
      return "unknown";
  }
}

// Scoped timer that charges elapsed nanoseconds to an op on destruction, so an
// early return inside a forwarded call still gets measured.
class ScopedNanos {
 public:
  explicit ScopedNanos(std::atomic<uint64_t>& sink)
      : sink_(sink), start_(std::chrono::steady_clock::now()) {}
  ~ScopedNanos() {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    sink_.fetch_add(static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                    std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>& sink_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace

InstrumentedBackend::InstrumentedBackend(std::unique_ptr<MediumBackend> inner)
    : inner_(std::move(inner)) {
  for (auto& op : entries_) {
    for (auto& status : op) status.store(0, std::memory_order_relaxed);
  }
}

bool InstrumentedBackend::Init(MemoryRegistrar* registrar) { return inner_->Init(registrar); }

std::vector<AllocateResult> InstrumentedBackend::BatchAllocate(
    const std::vector<AllocateRequest>& reqs) {
  ops_[kAllocate].batches.fetch_add(1, std::memory_order_relaxed);
  std::vector<AllocateResult> out;
  {
    ScopedNanos timer(ops_[kAllocate].nanos);
    out = inner_->BatchAllocate(reqs);
  }
  // Outcome mapping is the medium-agnostic half of AllocateOutcome: kFailedNoSpace
  // stays distinct because it is the one failure a writer retries elsewhere, and
  // "is this pool full" is the first question a capacity panel has to answer.
  for (const AllocateResult& r : out) {
    switch (r.outcome) {
      case AllocateOutcome::kSuccessAllocated:
        Add(kAllocate, kOk, 1);
        break;
      case AllocateOutcome::kSuccessAlreadyExists:
        Add(kAllocate, kExists, 1);
        break;
      case AllocateOutcome::kFailedNoSpace:
        Add(kAllocate, kNoSpace, 1);
        break;
      case AllocateOutcome::kFailed:
        Add(kAllocate, kFailed, 1);
        break;
    }
  }
  return out;
}

std::vector<CommitResult> InstrumentedBackend::BatchCommit(const std::vector<CommitRequest>& reqs) {
  ops_[kCommit].batches.fetch_add(1, std::memory_order_relaxed);
  std::vector<CommitResult> out;
  {
    ScopedNanos timer(ops_[kCommit].nanos);
    out = inner_->BatchCommit(reqs);
  }
  uint64_t bytes = 0;
  for (const CommitResult& r : out) {
    if (r.success) {
      Add(kCommit, kOk, 1);
      bytes += r.bytes_committed;
    } else {
      // A slot that was reaped, aborted, or never existed.  Not "medium broken"
      // but the writer's bytes are gone either way, so it counts as failed.
      Add(kCommit, kFailed, 1);
    }
  }
  if (bytes > 0) ops_[kCommit].bytes.fetch_add(bytes, std::memory_order_relaxed);
  return out;
}

std::vector<bool> InstrumentedBackend::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  ops_[kAbort].batches.fetch_add(1, std::memory_order_relaxed);
  std::vector<bool> out;
  {
    ScopedNanos timer(ops_[kAbort].nanos);
    out = inner_->BatchAbort(slot_ids);
  }
  for (bool ok : out) Add(kAbort, ok ? kOk : kFailed, 1);
  return out;
}

std::vector<ResolvedEntry> InstrumentedBackend::BatchResolve(const std::vector<std::string>& keys,
                                                             bool include_descs) {
  ops_[kResolve].batches.fetch_add(1, std::memory_order_relaxed);
  std::vector<ResolvedEntry> out;
  {
    ScopedNanos timer(ops_[kResolve].nanos);
    out = inner_->BatchResolve(keys, include_descs);
  }
  uint64_t bytes = 0;
  for (const ResolvedEntry& r : out) {
    if (r.found) {
      Add(kResolve, kOk, 1);
      bytes += r.size;
    } else {
      Add(kResolve, kMiss, 1);
    }
  }
  if (bytes > 0) ops_[kResolve].bytes.fetch_add(bytes, std::memory_order_relaxed);
  return out;
}

std::vector<EvictResult> InstrumentedBackend::Evict(const std::vector<std::string>& keys) {
  ops_[kEvict].batches.fetch_add(1, std::memory_order_relaxed);
  std::vector<EvictResult> out;
  {
    ScopedNanos timer(ops_[kEvict].nanos);
    out = inner_->Evict(keys);
  }
  uint64_t bytes = 0;
  for (const EvictResult& r : out) {
    // bytes_freed == 0 means the key was unknown, already gone, or protected by
    // a live lease.  That is a miss, not a failure: master retries it later.
    if (r.bytes_freed > 0) {
      Add(kEvict, kOk, 1);
      bytes += r.bytes_freed;
    } else {
      Add(kEvict, kMiss, 1);
    }
  }
  if (bytes > 0) ops_[kEvict].bytes.fetch_add(bytes, std::memory_order_relaxed);
  return out;
}

std::vector<MetricSample> InstrumentedBackend::SampleMetrics() const {
  std::vector<MetricSample> out;
  out.reserve(kOpCount * 4 + 8);

  for (int op = 0; op < kOpCount; ++op) {
    const char* op_name = OpName(op);

    for (int status = 0; status < kStatusCount; ++status) {
      const uint64_t v = entries_[op][status].load(std::memory_order_relaxed);
      // Skip combinations this op never produces (an abort has no "no_space")
      // so the series set stays the shape of what actually happens.
      if (v == 0) continue;
      out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_OPS_TOTAL,
                                 MORI_UMBP_METRIC_BACKEND_OPS_TOTAL_HELP,
                                 {{"op", op_name}, {"status", StatusName(status)}},
                                 v});
    }

    const uint64_t batches = ops_[op].batches.load(std::memory_order_relaxed);
    if (batches == 0) continue;
    out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_BATCHES_TOTAL,
                               MORI_UMBP_METRIC_BACKEND_BATCHES_TOTAL_HELP,
                               {{"op", op_name}},
                               batches});

    // Accumulated in nanoseconds because that is what an atomic add can carry;
    // MetricSample::scale converts once, in the publisher, so the metric keeps
    // the seconds its name promises.
    const uint64_t nanos = ops_[op].nanos.load(std::memory_order_relaxed);
    if (nanos > 0) {
      out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_OP_SECONDS_TOTAL,
                                 MORI_UMBP_METRIC_BACKEND_OP_SECONDS_TOTAL_HELP,
                                 {{"op", op_name}},
                                 nanos,
                                 MetricKind::kCounter,
                                 1.0 / kNanosPerSecond});
    }

    const uint64_t bytes = ops_[op].bytes.load(std::memory_order_relaxed);
    if (bytes > 0) {
      out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_BACKEND_BYTES_TOTAL_HELP,
                                 {{"op", op_name}},
                                 bytes});
    }
  }

  // The medium's own view, appended unchanged.  The publisher stamps tier= and
  // backend= on both halves, so a medium counter and a decorator counter are
  // indistinguishable to a dashboard — which is the point.
  for (MetricSample& s : inner_->SampleMetrics()) out.push_back(std::move(s));
  return out;
}

std::unique_ptr<MediumBackend> MakeInstrumentedBackend(std::unique_ptr<MediumBackend> inner) {
  if (inner == nullptr) return nullptr;
  return std::make_unique<InstrumentedBackend>(std::move(inner));
}

}  // namespace mori::umbp
