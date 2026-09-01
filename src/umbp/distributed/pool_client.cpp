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
#include "umbp/distributed/pool_client.h"

#include "umbp/distributed/benchmark/workload_trace_recorder.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <msgpack.hpp>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_copy.h"
#include "umbp/common/env_time.h"
#include "umbp/common/parallel_for.h"
#include "umbp/common/range_utils.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/backend/hbm_backend.h"
#include "umbp/distributed/peer/backend/instrumented_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/backend/ssd_backend.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/range_map.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

// ---------------------------------------------------------------------------
//  Bandwidth metrics
// ---------------------------------------------------------------------------

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

const std::vector<double>& BatchBandwidthBucketsGiBps() {
  static const std::vector<double> buckets = {
      0.1,  0.2,  0.5,  1.0,   2.0,   3.0,   4.0,   6.0,   8.0,   12.0,  16.0,  24.0, 32.0,
      48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 320.0, 384.0, 448.0, 512.0, 640.0, 800.0};
  return buckets;
}

struct BatchBandwidthSplit {
  double local = 0.0;
  double remote = 0.0;
};

// Bandwidth predicate.  BatchGet uses `bool` (no dedup); BatchPut uses
// PutEntryOutcome (kAlreadyExists is success-to-caller but moves no
// bytes — excluded from bandwidth).
inline bool IsCountedForBandwidth(bool r) { return r; }
inline bool IsCountedForBandwidth(PoolClient::PutEntryOutcome r) {
  return r == PoolClient::PutEntryOutcome::kSucceeded;
}

template <typename Route, typename Result>
BatchBandwidthSplit ComputeBatchBandwidthBytes(const std::vector<Result>& results,
                                               const std::vector<size_t>& sizes,
                                               const std::vector<std::optional<Route>>& routes,
                                               std::string_view local_node_id) {
  // guard against mismatched sizes
  const size_t limit = std::min({results.size(), sizes.size(), routes.size()});
  BatchBandwidthSplit acc;
  for (size_t i = 0; i < limit; ++i) {
    if (!IsCountedForBandwidth(results[i])) continue;
    const double bytes = static_cast<double>(sizes[i]);
    // No route means the key was served from local storage (fallback path).
    const bool is_local = !routes[i].has_value() || routes[i]->node_id == local_node_id;
    (is_local ? acc.local : acc.remote) += bytes;
  }
  return acc;
}

// `master_client` is null on a node running without a master; the histogram
// simply has nowhere to go, and the caller should not have to know that.
void ObserveBatchBandwidth(MasterClient* master_client, double bytes, double seconds,
                           const char* metric_name, const char* metric_help,
                           std::string_view traffic) {
  if (master_client == nullptr || bytes <= 0.0 || seconds <= 0.0) return;
  const double gibps = (bytes / seconds) / kGiB;
  if (gibps <= 0.0) return;
  MasterClient::Labels labels = {{"traffic", std::string(traffic)}};
  master_client->Observe(metric_name, metric_help, std::move(labels), BatchBandwidthBucketsGiBps(),
                         gibps);
}

// ---------------------------------------------------------------------------
//  Ranged-call sub-timers  (UMBP_RANGED_CALL_DEBUG=1)
//
//  ScopedBatchBandwidth already times the WHOLE ranged call, and HbmCopyEngine's
//  debug mode times the copies inside it.  Neither answers the question those
//  two together raise: where does the rest of the call go?  Subtracting one
//  aggregate from the other only bounds it, because a batch fans keys across
//  executor threads, so summed copy time is not the call's copy wall-clock.
//
//  These timers close that gap by splitting a ranged call into the phases that
//  can actually stall it:
//
//    resolve  per-key BatchResolve, scanning every backend in the registry
//    route    BatchRouteGet / BatchRoutePut RPC to the master
//    xfer     transfer_engine_->Transfer (the part HbmCopyEngine reports)
//    lock     waiting on the get/put scratch mutex, which serializes arena users
//    remote   the arena loop: fetch/assemble + install
//
//  Note the paths are NOT symmetric, and the timers are what proves it: a get
//  resolves locally first and only routes on a miss, while a put issues
//  BatchRoutePut on EVERY call including fully-local ones.
//
//  Off by default; knobs mirror UMBP_HBM_COPY_DEBUG:
//    UMBP_RANGED_CALL_DEBUG=1             enable
//    UMBP_RANGED_CALL_DEBUG_SAMPLE=N      per-call line every Nth call (1)
//    UMBP_RANGED_CALL_DEBUG_SUMMARY_SEC=N cumulative rollup cadence (10; 0=off)
// ---------------------------------------------------------------------------

bool RangedDebugEnvFlag(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  const std::string text(value);
  return text != "0" && text != "off" && text != "false";
}

size_t RangedDebugEnvSize(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) return fallback;
  return static_cast<size_t>(parsed);
}

bool RangedDebugEnabled() {
  static const bool enabled = RangedDebugEnvFlag("UMBP_RANGED_CALL_DEBUG", false);
  return enabled;
}

size_t RangedDebugSampleEvery() {
  static const size_t n =
      std::max<size_t>(RangedDebugEnvSize("UMBP_RANGED_CALL_DEBUG_SAMPLE", 1), 1);
  return n;
}

size_t RangedDebugSummarySeconds() {
  static const size_t n = RangedDebugEnvSize("UMBP_RANGED_CALL_DEBUG_SUMMARY_SEC", 10);
  return n;
}

bool RangedDebugSampleThisCall() {
  static std::atomic<uint64_t> counter{0};
  return (counter.fetch_add(1, std::memory_order_relaxed) % RangedDebugSampleEvery()) == 0;
}

inline double RangedSeconds(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

struct RangedPhases {
  double resolve = 0.0;
  // Split out of `resolve`, which used to cover the whole per-key loop and so
  // could not say whether the time went to the medium's index or to describing
  // the caller's buffers.  `classify` is the latter: one pointer lookup per
  // RANGE, and a ranged call carries thousands of them.
  double classify = 0.0;
  double build = 0.0;
  double validate = 0.0;  // put only: checking the ranges tile their object
  double commit = 0.0;    // put only: slot commit
  double route = 0.0;
  double xfer = 0.0;
  // xfer split three ways.  Planning scales with items, the move with bytes, so
  // a batch of many small ranges needs them apart to say which it is paying.
  double xfer_plan = 0.0;
  double xfer_submit = 0.0;
  double xfer_wait = 0.0;
  double lock = 0.0;
  double remote = 0.0;
  size_t keys = 0;
  // TransferItems handed to the engine.  Grows with ranges, not keys, and is
  // what makes the item-per-range cost visible next to the phase times.
  size_t items = 0;
  size_t local_keys = 0;
  size_t remote_keys = 0;
  double bytes = 0.0;
  // Component identity.  A ranged call is always single-pool -- the tree
  // connector groups by PoolName and chunks inside a per-pool loop, so one
  // sample key names the component for the whole batch.  Verified empirically
  // too: 60/60 gather buckets carried a single object size.
  //
  // `object_size` is the grouping key rather than the string: the six DSv4-Pro
  // components have six distinct object sizes, so it is an exact label that
  // costs no parsing on the hot path.  `key0` is carried alongside so the
  // size->component mapping is provable from the log rather than inferred.
  const std::string* key0 = nullptr;
  size_t object_size = 0;
};

// Adds its lifetime to a sink.  A null sink makes it inert, which is how the
// hot path pays nothing when debug is off.
class PhaseTimer {
 public:
  explicit PhaseTimer(double* sink)
      : sink_(sink),
        start_(sink ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
  ~PhaseTimer() { Stop(); }
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;
  void Stop() {
    if (sink_ == nullptr) return;
    *sink_ += RangedSeconds(start_, std::chrono::steady_clock::now());
    sink_ = nullptr;
  }

 private:
  double* sink_;
  std::chrono::steady_clock::time_point start_;
};

class RangedStats {
 public:
  void Record(const char* op, const RangedPhases& p, double total) {
    const bool is_get = (op[0] == 'g');
    std::lock_guard<std::mutex> lock(mutex_);
    Totals& t = is_get ? get_ : put_;
    t.calls += 1;
    t.total += total;
    t.resolve += p.resolve;
    t.classify += p.classify;
    t.build += p.build;
    t.validate += p.validate;
    t.commit += p.commit;
    t.route += p.route;
    t.xfer += p.xfer;
    t.xfer_plan += p.xfer_plan;
    t.xfer_submit += p.xfer_submit;
    t.xfer_wait += p.xfer_wait;
    t.lock += p.lock;
    t.remote += p.remote;
    t.bytes += p.bytes;
    t.items += p.items;

    // Per-component totals, keyed by object size (one call = one pool).  The
    // sample key is stored once per (op, size) so the log proves the
    // size->component mapping instead of leaving it inferred.
    if (p.object_size != 0) {
      Component& c = components_[{is_get, p.object_size}];
      c.calls += 1;
      c.keys += p.keys;
      c.bytes += p.bytes;
      c.total += total;
      c.xfer += p.xfer;
      if (c.sample_key.empty() && p.key0 != nullptr) c.sample_key = *p.key0;
    }

    const size_t cadence = RangedDebugSummarySeconds();
    if (cadence == 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (RangedSeconds(last_report_, now) < static_cast<double>(cadence)) return;
    last_report_ = now;
    Emit("GET", get_);
    Emit("PUT", put_);
    EmitComponentsLocked();
  }

  // The cadence emit only fires when a call lands more than one window after
  // the previous one, so a measured phase shorter than the cadence reports no
  // line at all -- indistinguishable from the instrumentation being off -- and
  // the final partial window of a long run is lost the same way.
  void FlushFinal() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (get_.calls == 0 && put_.calls == 0) return;
    Emit("GET", get_);
    Emit("PUT", put_);
    EmitComponentsLocked();
  }

 private:
  struct Totals {
    uint64_t calls = 0;
    uint64_t items = 0;
    double total = 0, resolve = 0, classify = 0, build = 0, validate = 0, commit = 0, route = 0,
           xfer = 0, xfer_plan = 0, xfer_submit = 0, xfer_wait = 0, lock = 0, remote = 0, bytes = 0;
  };
  struct Component {
    uint64_t calls = 0;
    uint64_t keys = 0;
    double bytes = 0, total = 0, xfer = 0;
    std::string sample_key;
  };
  static void Emit(const char* name, const Totals& t) {
    if (t.calls == 0) return;
    const double other = t.total - (t.resolve + t.classify + t.build + t.validate + t.commit +
                                    t.route + t.xfer + t.lock + t.remote);
    auto share = [&](double v) { return t.total > 0 ? 100.0 * v / t.total : 0.0; };
    MORI_UMBP_INFO(
        "[RangedCall][dbg] SUMMARY {} calls={} total={:.3f}s mean_call={:.1f}us bytes={:.2f}GiB "
        "items_per_call={:.0f} | resolve={:.1f}% classify={:.1f}% build={:.1f}% validate={:.1f}% "
        "commit={:.1f}% route={:.1f}% xfer={:.1f}%(plan={:.1f}% submit={:.1f}% wait={:.1f}%) "
        "lock={:.1f}% remote={:.1f}% other={:.1f}% "
        "| xfer_only={:.2f}GiB/s end2end={:.2f}GiB/s",
        name, t.calls, t.total, 1e6 * t.total / t.calls, t.bytes / (1024.0 * 1024 * 1024),
        static_cast<double>(t.items) / t.calls, share(t.resolve), share(t.classify), share(t.build),
        share(t.validate), share(t.commit), share(t.route), share(t.xfer), share(t.xfer_plan),
        share(t.xfer_submit), share(t.xfer_wait), share(t.lock), share(t.remote), share(other),
        t.xfer > 0 ? (t.bytes / t.xfer) / (1024.0 * 1024 * 1024) : 0.0,
        t.total > 0 ? (t.bytes / t.total) / (1024.0 * 1024 * 1024) : 0.0);
  }
  void EmitComponentsLocked() const {
    for (const auto& [id, c] : components_) {
      if (c.calls == 0) continue;
      MORI_UMBP_INFO(
          "[RangedCall][dbg] COMPONENT {} obj={} key0='{}' calls={} keys={} bytes={:.3f}GiB "
          "total={:.3f}s mean_call={:.1f}us keys_per_call={:.1f} xfer_only={:.2f}GiB/s "
          "end2end={:.2f}GiB/s",
          id.first ? "GET" : "PUT", id.second, c.sample_key, c.calls, c.keys,
          c.bytes / (1024.0 * 1024 * 1024), c.total, 1e6 * c.total / c.calls,
          static_cast<double>(c.keys) / c.calls,
          c.xfer > 0 ? (c.bytes / c.xfer) / (1024.0 * 1024 * 1024) : 0.0,
          c.total > 0 ? (c.bytes / c.total) / (1024.0 * 1024 * 1024) : 0.0);
    }
  }

  std::mutex mutex_;
  Totals get_;
  Totals put_;
  // key = (is_get, object_size); ordered so the rollup prints deterministically.
  std::map<std::pair<bool, size_t>, Component> components_;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
};

RangedStats& RangedStatsInstance() {
  // Deliberately leaked: a destructor here would run at static-destruction
  // time, after singletons it logs through may be gone. atexit gives back the
  // final flush the leak would otherwise cost, and runs before those teardowns.
  static RangedStats* stats = [] {
    auto* created = new RangedStats();
    std::atexit([] { RangedStatsInstance().FlushFinal(); });
    return created;
  }();
  return *stats;
}

// Emits at scope exit so every early return in a ranged call is covered, the
// same reason ScopedBatchBandwidth is a destructor.
class ScopedRangedReport {
 public:
  ScopedRangedReport(const char* op, RangedPhases& phases, bool enabled, const double& local_bytes,
                     const double& remote_bytes)
      : op_(op),
        phases_(phases),
        enabled_(enabled),
        local_bytes_(local_bytes),
        remote_bytes_(remote_bytes),
        start_(std::chrono::steady_clock::now()) {}
  ~ScopedRangedReport() {
    if (!enabled_) return;
    // Read at exit: the remote phase keeps adding after any mid-call snapshot.
    phases_.bytes = local_bytes_ + remote_bytes_;
    const double total = RangedSeconds(start_, std::chrono::steady_clock::now());
    RangedStatsInstance().Record(op_, phases_, total);
    if (!RangedDebugSampleThisCall()) return;
    const double other =
        total - (phases_.resolve + phases_.classify + phases_.build + phases_.validate +
                 phases_.commit + phases_.route + phases_.xfer + phases_.lock + phases_.remote);
    MORI_UMBP_INFO(
        "[RangedCall][dbg] op={} obj={} key0='{}' keys={} items={} local={} remote={} bytes={} "
        "total={:.1f}us resolve={:.1f}us classify={:.1f}us build={:.1f}us validate={:.1f}us "
        "commit={:.1f}us route={:.1f}us xfer={:.1f}us(plan={:.1f} submit={:.1f} wait={:.1f}) "
        "lock={:.1f}us remote_phase={:.1f}us other={:.1f}us",
        op_, phases_.object_size, phases_.key0 != nullptr ? *phases_.key0 : std::string("?"),
        phases_.keys, phases_.items, phases_.local_keys, phases_.remote_keys, phases_.bytes,
        total * 1e6, phases_.resolve * 1e6, phases_.classify * 1e6, phases_.build * 1e6,
        phases_.validate * 1e6, phases_.commit * 1e6, phases_.route * 1e6, phases_.xfer * 1e6,
        phases_.xfer_plan * 1e6, phases_.xfer_submit * 1e6, phases_.xfer_wait * 1e6,
        phases_.lock * 1e6, phases_.remote * 1e6, other * 1e6);
  }

 private:
  const char* op_;
  RangedPhases& phases_;
  bool enabled_;
  const double& local_bytes_;
  const double& remote_bytes_;
  std::chrono::steady_clock::time_point start_;
};

// Runs a callable at scope exit.  Same reason the two Scoped* classes below are
// destructor-based: a ranged call has several early returns that have still
// moved bytes, and end-of-call work must not depend on each `return` having
// remembered to do it.
template <class Fn>
class ScopeExit {
 public:
  explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
  ~ScopeExit() { fn_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  Fn fn_;
};
template <class Fn>
ScopeExit(Fn) -> ScopeExit<Fn>;

// Ranged calls have several early returns that can still have moved bytes (a
// batch fully served from the local medium returns before the remote phase is
// even considered).  Emitting from a destructor keeps every exit covered,
// including ones added later, instead of relying on each `return` remembering
// to observe first.  The accumulators are held by reference and read at scope
// exit, so callers just add to them as keys succeed.
class ScopedBatchBandwidth {
 public:
  ScopedBatchBandwidth(MasterClient* master_client, const char* metric_name,
                       const char* metric_help, const double& local_bytes,
                       const double& remote_bytes)
      : master_client_(master_client),
        metric_name_(metric_name),
        metric_help_(metric_help),
        local_bytes_(local_bytes),
        remote_bytes_(remote_bytes),
        start_(std::chrono::steady_clock::now()) {}

  ScopedBatchBandwidth(const ScopedBatchBandwidth&) = delete;
  ScopedBatchBandwidth& operator=(const ScopedBatchBandwidth&) = delete;

  ~ScopedBatchBandwidth() {
    const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::steady_clock::now() - start_)
                               .count();
    ObserveBatchBandwidth(master_client_, local_bytes_, seconds, metric_name_, metric_help_,
                          "local");
    ObserveBatchBandwidth(master_client_, remote_bytes_, seconds, metric_name_, metric_help_,
                          "remote");
  }

 private:
  MasterClient* master_client_;
  const char* metric_name_;
  const char* metric_help_;
  const double& local_bytes_;
  const double& remote_bytes_;
  std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
//  Page / size math
// ---------------------------------------------------------------------------

// --- Cross-key parallelism for the self-target (local) paths. --------------
// In distributed mode 1 key == 1 page (master page_size == KV block size), so
// a self-target Put/Get copies one ~MiB-scale block per key.  The parallelism
// that pays is therefore ACROSS the many keys of one BatchPut/BatchGet
// (different threads -> different keys), not within one key's pages.  The
// per-key copy itself lives in LocalCopyEngine since Phase 6.  Threads via
// UMBP_DRAM_{READ,WRITE}_THREADS (same envs as local mode's DRAMTier).
inline int LocalCopyThreads(const char* env_name) {
  int t = 4;
  if (const char* e = std::getenv(env_name)) {
    int x = std::atoi(e);
    if (x >= 1) t = x;
  }
  unsigned hc = std::thread::hardware_concurrency();
  if (hc > 0 && t > static_cast<int>(hc)) t = static_cast<int>(hc);
  if (t < 1) t = 1;
  return t;
}

inline std::chrono::milliseconds ResolveBusyRetryTimeout() {
  uint64_t ms = 30000;
  if (const char* value = std::getenv("UMBP_RESOLVE_BUSY_TIMEOUT_MS")) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0) ms = parsed;
  }
  return std::chrono::milliseconds(std::min<uint64_t>(ms, 300000));
}

std::vector<PoolResolvedEntry> ResolveLocalBatchWithBusyRetry(
    PeerPool* pool, const std::vector<std::string>& keys, bool include_descs) {
  if (pool == nullptr) return std::vector<PoolResolvedEntry>(keys.size());
  const auto deadline = std::chrono::steady_clock::now() + ResolveBusyRetryTimeout();
  std::chrono::milliseconds backoff{1};
  size_t attempts = 0;
  while (true) {
    auto resolved = pool->BatchResolve(keys, include_descs);
    const bool busy = std::any_of(resolved.begin(), resolved.end(), [](const auto& entry) {
      return EffectiveResolveOutcome(entry.resolved) == ResolveOutcome::kBusy;
    });
    if (!busy) return resolved;
    ++attempts;
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
      MORI_UMBP_WARN("[PoolClient] local BatchResolve BUSY timeout after {} attempts", attempts);
      return resolved;
    }
    const auto sleep_for =
        std::min(backoff, std::chrono::duration_cast<std::chrono::milliseconds>(remaining));
    if (sleep_for.count() > 0) std::this_thread::sleep_for(sleep_for);
    backoff = std::min(backoff * 2, std::chrono::milliseconds{50});
  }
}

// Classify a caller's buffer so engine selection can see what it really is.
//
// An unregistered pointer used to be described unconditionally as host bytes,
// which is what CanHandle dispatches on: a device pointer would therefore be
// claimed by LocalCopyEngine and memcpy'd, or staged through MoriIoEngine's
// host bounce buffer.  Neither fails at the call site — they corrupt or fault
// somewhere else — so the classification has to happen before any engine sees
// the pair.  Registered buffers never come through here; their ref already
// carries loc/device from RegisterMemory.
TransferRef ClassifiedUserBytes(void* ptr, uint64_t size) {
  const PointerLocation location = DetectPointerLocation(ptr);
  if (!location.IsDevice()) return TransferRef::HostBytes(ptr, size);
  return TransferRef::HostBytes(ptr, size, mori::io::MemoryLocationType::GPU, location.device_id);
}

// Read-side range validity: inside the object and non-overlapping.  Weaker than
// RangesTileObject, which the write side needs — a read may take any subset of
// an object, but a write must account for every byte it commits.
bool RangesAreDisjointAndInBounds(size_t object_size, const std::vector<size_t>& sizes,
                                  const std::vector<size_t>& offsets) {
  if (object_size == 0 || sizes.empty() || sizes.size() != offsets.size()) return false;
  std::vector<size_t> order(sizes.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return offsets[a] < offsets[b]; });
  size_t previous_end = 0;
  bool first = true;
  for (size_t index : order) {
    if (sizes[index] == 0 || IsObjectRangeOverflow(offsets[index], sizes[index], object_size)) {
      return false;
    }
    if (!first && offsets[index] < previous_end) return false;
    previous_end = offsets[index] + sizes[index];
    first = false;
  }
  return true;
}

// Scratch-arena slice alignment.  Objects are packed back-to-back into the
// arena; 64 B keeps each slice on a cache line so two concurrently-filled
// slices never share one.
constexpr size_t kRangedScratchAlignment = 64;

bool AlignUpChecked(size_t value, size_t alignment, size_t* out) {
  if (!out || alignment == 0) return false;
  const size_t remainder = value % alignment;
  if (remainder == 0) {
    *out = value;
    return true;
  }
  const size_t add = alignment - remainder;
  if (value > std::numeric_limits<size_t>::max() - add) return false;
  *out = value + add;
  return true;
}

// ---------------------------------------------------------------------------
//  Lease env knobs
// ---------------------------------------------------------------------------

// Peer-side DRAM/HBM read lease: how long a single Resolve protects its key's
// pages from concurrent local Evict, covering one RDMA read.  Only needs to
// exceed one DRAM RDMA round trip (sub-ms), so 500 ms is already ~100x margin;
// exposed so operators can tighten it under eviction pressure.
std::chrono::milliseconds DramReadLeaseTtl() {
  static const auto v = GetEnvMilliseconds("UMBP_DRAM_READ_LEASE_MS",
                                           std::chrono::milliseconds(500), /*min_allowed=*/1);
  return v;
}

// The SSD equivalent, which is a different number for a different reason: it has
// to cover an SSD read plus the RDMA, not just the RDMA. It is also the window a
// staging page stays claimed, so it sets this backend's read concurrency against
// a fixed arena -- shortening it frees pages and costs read coalescing.
std::chrono::milliseconds SsdReadLeaseTtl() {
  static const auto v = GetEnvMilliseconds("UMBP_SSD_READ_LEASE_MS",
                                           std::chrono::milliseconds(3000), /*min_allowed=*/1);
  return v;
}

// ---------------------------------------------------------------------------
//  Config / proto translation
// ---------------------------------------------------------------------------

// Translate a peer-side ::umbp::AllocateSlotResponse / ResolveKeyResponse
// into the C++ shapes our code consumes.
PoolClient::SlotPlan FromAllocateSlotResponse(const ::umbp::AllocateSlotResponse& resp) {
  PoolClient::SlotPlan p;
  p.slot_id = resp.slot_id();
  p.page_size = resp.page_size();
  p.backend_id = resp.backend_id();
  p.pages.reserve(resp.pages_size());
  for (const auto& pp : resp.pages()) p.pages.push_back({pp.buffer_index(), pp.page_index()});
  p.descs.reserve(resp.descs_size());
  for (const auto& d : resp.descs()) {
    BufferMemoryDescBytes b;
    b.buffer_index = d.buffer_index();
    b.backend_id = d.backend_id();
    b.desc_bytes.assign(d.desc().begin(), d.desc().end());
    p.descs.push_back(std::move(b));
  }
  return p;
}

// ---------------------------------------------------------------------------
//  Engine outcome -> per-entry failure
//
//  A TransferItem's `tag` IS the index of the entry that produced it, so
//  mapping an engine failure back to keys is an index lookup.  A failed plan
//  fails EVERY key that contributed a segment to it (per-item AND) — the same
//  granularity the pre-refactor per-(localMR, remoteMR) group had, for the same
//  reason: the wire reports success per transfer, not per key.
//
//  Free templates rather than PoolClient members so both entry types share one
//  definition; deduction means neither ever has to be named here.
// ---------------------------------------------------------------------------

template <typename Entry>
void ApplyTransferFailures(std::vector<Entry>& entries,
                           const std::vector<TransferFailure>& failures, const char* what) {
  for (const auto& f : failures) {
    for (size_t tag : f.tags) {
      if (tag >= entries.size()) continue;
      auto& entry = entries[tag];
      MORI_UMBP_ERROR("{} transfer failed: code={} msg='{}' peer_engine='{}' key='{}'", what,
                      f.code, f.message, f.endpoint,
                      (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"});
      entry.failed = true;
    }
  }
}

// Items no engine could carry.  Distinct from a failed transfer: these never
// reached the wire, which usually means a peer descriptor was missing, the item
// was larger than the engine's bounce pool, or the caller's buffer is
// unregistered GPU memory going to a remote peer — staging that would mean a
// host memcpy from device memory, so MoriIoEngine rejects it instead.
template <typename Entry>
void ApplyRejectedTags(std::vector<Entry>& entries, const std::vector<size_t>& tags,
                       const char* what) {
  for (size_t tag : tags) {
    if (tag >= entries.size()) continue;
    auto& entry = entries[tag];
    MORI_UMBP_WARN("{} transfer unplannable (no engine claimed the endpoints), key='{}'", what,
                   (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"});
    entry.failed = true;
  }
}

std::vector<BackendInstanceConfig> EffectiveBackendConfigs(const PoolClientConfig& config) {
  if (!config.backends.empty()) return config.backends;

  BackendInstanceConfig legacy;
  legacy.name = DefaultBackendInstanceName(config.medium);
  legacy.tier = config.medium;
  legacy.dram = config.dram;
  legacy.hbm = config.hbm;
  legacy.ssd = config.ssd;
  legacy.ssd_staging_buffer_slots = config.ssd_staging_buffer_slots;
  return {std::move(legacy)};
}

std::unique_ptr<MediumBackend> MakeConfiguredBackend(const BackendInstanceConfig& config,
                                                     uint64_t page_size,
                                                     bool staging_use_hugepages,
                                                     uint64_t staging_hugepage_size) {
  switch (config.tier) {
    case TierType::DRAM: {
      PageBackend::OwnershipConfig ownership;
      ownership.buffer_sizes = config.dram.buffer_sizes;
      ownership.use_hugepages = config.dram.use_hugepages;
      ownership.hugepage_size = config.dram.hugepage_size;
      ownership.numa_node = config.dram.numa_node;
      ownership.prefault = config.dram.prefault;
      return MakePageBackend(TierType::DRAM, page_size, std::move(ownership),
                             /*pending_ttl=*/std::chrono::milliseconds{30000},
                             /*read_lease_ttl=*/DramReadLeaseTtl());
    }
    case TierType::HBM:
      return MakeHbmBackend(page_size, config.hbm.device, config.hbm.buffer_sizes,
                            /*pending_ttl=*/std::chrono::milliseconds{30000},
                            /*read_lease_ttl=*/DramReadLeaseTtl());
    case TierType::SSD: {
      SsdBackend::Config ssd_cfg;
      ssd_cfg.page_size = page_size;
      ssd_cfg.staging_pages = config.ssd_staging_buffer_slots > 0
                                  ? static_cast<uint32_t>(config.ssd_staging_buffer_slots)
                                  : 16;
      ssd_cfg.staging_use_hugepages = staging_use_hugepages;
      ssd_cfg.staging_hugepage_size = staging_hugepage_size;
      ssd_cfg.ssd = config.ssd;
      ssd_cfg.ssd.enabled = true;
      ssd_cfg.read_lease_ttl = SsdReadLeaseTtl();
      return MakeSsdBackend(std::move(ssd_cfg));
    }
    case TierType::UNKNOWN:
      return nullptr;
  }
  return nullptr;
}

std::unique_ptr<PoolPolicy> MakeConfiguredPoolPolicy(
    const PoolClientConfig& config, const std::vector<BackendInstanceConfig>& backends,
    const BackendRegistry& registry,
    std::shared_ptr<const LogicalTierGraph>* tier_graph) {
  switch (config.placement_policy) {
    case PoolPlacementPolicy::SINGLE_BACKEND:
      return MakeSingleBackendPolicy();
    case PoolPlacementPolicy::WEIGHTED: {
      std::vector<BackendPlacementWeight> weights;
      weights.reserve(backends.size());
      for (const auto& backend : backends) {
        if (backend.placement_weight == 0) {
          MORI_UMBP_ERROR("[PoolClient] weighted backend '{}' has zero placement weight",
                          backend.name);
          return nullptr;
        }
        weights.push_back(BackendPlacementWeight{backend.name, backend.placement_weight});
      }
      return MakeWeightedPlacementPolicy(std::move(weights));
    }
    case PoolPlacementPolicy::TIERED:
      if (config.logical_tiers.empty()) {
        MORI_UMBP_ERROR("[PoolClient] tiered placement has no logical tiers");
        return nullptr;
      }
      if (tier_graph == nullptr) return nullptr;
      {
        auto compiled = LogicalTierGraph::Compile(config.logical_tiers, registry);
        if (!compiled.ok()) {
          MORI_UMBP_ERROR("[PoolClient] invalid logical tier graph: {}", compiled.error);
          return nullptr;
        }
        *tier_graph = std::move(compiled.graph);
        return MakeTieredPlacementPolicy(*tier_graph);
      }
  }
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

PoolClient::PoolClient(PoolClientConfig config) : config_(std::move(config)) {}
PoolClient::~PoolClient() { Shutdown(); }

bool PoolClient::Init() {
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) return true;

  if (config_.workload_trace_path.empty()) {
    if (const char* path = std::getenv("UMBP_WORKLOAD_TRACE_PATH")) {
      config_.workload_trace_path = path;
    }
  }
  if (const char* value = std::getenv("UMBP_WORKLOAD_TRACE_CLIENT_ID")) {
    config_.workload_trace_client_id =
        static_cast<uint32_t>(std::strtoull(value, nullptr, 10));
  }
  if (const char* value = std::getenv("UMBP_WORKLOAD_TRACE_SEED")) {
    config_.workload_trace_seed = std::strtoull(value, nullptr, 10);
  }

  if (!config_.policy_config_path.empty()) {
    auto loaded = LoadBackendPolicyFile(config_.policy_config_path);
    std::string error;
    if (!loaded.ok() || !ApplyBackendPolicy(*loaded.config, &config_, &error)) {
      MORI_UMBP_ERROR("[PoolClient] failed to load backend policy '{}': {}",
                      config_.policy_config_path,
                      loaded.ok() ? error : loaded.error);
      initialized_.store(false);
      return false;
    }
  }

  // No address, no master.  Everything below -- backends, transfer engine, peer
  // service -- is built either way; only routing, registration and the
  // heartbeat are skipped.  That is what lets one binary be a single node or a
  // cluster member depending on config alone.
  if (!config_.master_config.master_address.empty()) {
    master_client_ = std::make_unique<MasterClient>(config_.master_config);
  } else {
    MORI_UMBP_INFO("[PoolClient] no master configured; running single-node");
  }

  // The one byte-moving path (design doc §4).  Order is preference order:
  // a pair both of whose endpoints are host-addressable never reaches the wire.
  //
  // This is the composition root, and the only place any concrete engine is
  // named — the same rule Phase 5 applies to backends.  Everything downstream
  // holds TransferEngine / MemoryRegistrar / PeerDirectory.
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  // Both-local pairs with a GPU endpoint.  Disjoint from LocalCopyEngine (which
  // requires both sides CPU) and from MoriIoEngine (which requires exactly one
  // side remote), so this is registration order as documentation, not as a
  // tie-break — but it must come before the wire engine on principle, since an
  // HBM pair that reached mori-io would be refused rather than served slowly.
  auto hbm = std::make_unique<HbmCopyEngine>();
  // Kept as a raw observer so the live medium's host buffers can be declared
  // kernel-addressable once it has allocated them, below.  This is the one
  // engine-specific reference PoolClient holds, and it carries no data-plane
  // call: everything that moves bytes still goes through transfer_engine_.
  hbm_engine_ = hbm.get();
  composite->AddEngine(std::move(hbm));
  if (!config_.io_engine.host.empty()) {
    mori::io::IOEngineConfig io_cfg;
    io_cfg.host = config_.io_engine.host;
    io_cfg.port = config_.io_engine.port;
    auto rdma = std::make_unique<MoriIoEngine>(config_.master_config.node_id, io_cfg,
                                               config_.staging_buffer_size);
    if (!rdma->Init()) {
      MORI_UMBP_ERROR("[PoolClient] MoriIoEngine init failed on {}:{}", config_.io_engine.host,
                      config_.io_engine.port);
      Shutdown();
      return false;
    }
    peer_directory_ = rdma.get();
    composite->AddEngine(std::move(rdma));
  }
  transfer_engine_ = std::move(composite);

  // Peer-side backend instances.  The legacy single-medium selector is
  // synthesized into one entry here; topology-aware callers may
  // supply several named instances, including several of the same TierType.
  // This composition root remains the only place concrete backend factories
  // are named. The configured PoolPolicy below owns placement among them.
  const uint64_t page_size =
      config_.dram_page_size > 0 ? config_.dram_page_size : 2ULL * 1024 * 1024;

  auto backend_configs = EffectiveBackendConfigs(config_);
  if (backend_configs.empty()) {
    MORI_UMBP_ERROR("[PoolClient] no backend instances configured");
    Shutdown();
    return false;
  }
  medium_ = backend_configs.front().tier;  // legacy default for paths without a routed tier
  for (const auto& backend_config : backend_configs) {
    if (backend_config.name.empty() || registry_.Get(backend_config.name) != nullptr) {
      MORI_UMBP_ERROR("[PoolClient] backend instance name '{}' is empty or duplicated",
                      backend_config.name);
      Shutdown();
      return false;
    }
    auto backend = MakeConfiguredBackend(backend_config, page_size,
                                         config_.ssd_staging_use_hugepages,
                                         config_.ssd_staging_hugepage_size);
    if (backend == nullptr) {
      MORI_UMBP_ERROR("[PoolClient] backend '{}' has unknown medium {}",
                      backend_config.name, static_cast<int>(backend_config.tier));
      Shutdown();
      return false;
    }

    // Apply the generic operation/byte/latency instrumentation uniformly to
    // every named backend, including multiple instances of the same medium.
    backend = MakeInstrumentedBackend(std::move(backend));

    // Narrowed to MemoryRegistrar: a backend publishes endpoints, it does not
    // move bytes, and that is a compile-time fact (design doc §5 Rule C).
    if (!backend->Init(static_cast<MemoryRegistrar*>(transfer_engine_.get()))) {
      MORI_UMBP_ERROR("[PoolClient] backend '{}' ({}) Init failed", backend_config.name,
                      TierTypeName(backend_config.tier));
      backend->Shutdown();
      Shutdown();
      return false;
    }
    if (!registry_.Register(backend_config.name, std::move(backend))) {
      Shutdown();
      return false;
    }
  }

  std::shared_ptr<const LogicalTierGraph> tier_graph;
  auto placement_policy =
      MakeConfiguredPoolPolicy(config_, backend_configs, registry_, &tier_graph);
  if (placement_policy == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] invalid placement policy configuration");
    Shutdown();
    return false;
  }
  default_pool_ = std::make_unique<PeerPool>(
      &registry_, std::move(placement_policy), transfer_engine_.get());
  const bool weighted_placement =
      config_.placement_policy == PoolPlacementPolicy::WEIGHTED ||
      config_.placement_policy == PoolPlacementPolicy::TIERED;
  if (master_client_) {
    master_client_->SetAggregateBackendCapacities(weighted_placement);
    master_client_->SetBackendRegistry(&registry_);
    master_client_->SetPeerPool(default_pool_.get());
  }

  // Declare every host region the gather kernel may dereference: every live
  // backend's buffers and the ranged scratch arenas. These are copied
  // fragment-wise against GPU callers, which is the shape the kernel exists
  // for.  Registration is best-effort and asynchronous for large pools, so
  // until it lands the engine just takes hipMemcpy — no ordering requirement
  // here beyond the buffers existing.
  if (hbm_engine_ != nullptr) {
    for (auto* live : registry_.All()) {
      if (live == nullptr) continue;
      for (uint32_t i = 0; i < live->BufferCount(); ++i) {
        const TransferRef ref = live->BufferRef(i);
        // Only host buffers: a device-resident medium is already addressable
        // from a kernel, and hipHostRegister on it would fail.
        if (ref.HasHostPtr() && ref.loc == mori::io::MemoryLocationType::CPU) {
          hbm_engine_->AddHostGatherRegion(ref.host_ptr, ref.size);
        }
      }
    }
    if (config_.ranged_get_scratch_buffer != nullptr && config_.ranged_get_scratch_size > 0) {
      hbm_engine_->AddHostGatherRegion(config_.ranged_get_scratch_buffer,
                                       config_.ranged_get_scratch_size);
    }
    if (config_.ranged_put_scratch_buffer != nullptr && config_.ranged_put_scratch_size > 0) {
      hbm_engine_->AddHostGatherRegion(config_.ranged_put_scratch_buffer,
                                       config_.ranged_put_scratch_size);
    }
  }

  if (master_client_) master_client_->SetBackendRegistry(&registry_);

  // Every instrumented component on this node rides the existing metrics tick:
  // the storage backend (generic slot-lifecycle series from the decorator, plus
  // whatever the medium says about its own internals) and the transfer engine
  // (per-engine bytes, plans and in-flight time).  Backend-agnostic by
  // construction — PoolClient forwards what each component samples and never
  // learns which medium or transport produced it.
  if (master_client_) master_client_->AddMetricsProvider([this] { PublishComponentMetrics(); });

  // Pack engine_desc for master registration.
  std::vector<uint8_t> engine_desc_bytes;
  if (peer_directory_ != nullptr) engine_desc_bytes = peer_directory_->PackedLocalEngineDesc();

  if (config_.peer_service_port > 0 || config_.auto_peer_service_port) {
    peer_service_ = std::make_unique<PeerServiceServer>(*default_pool_, engine_desc_bytes,
                                                        master_client_.get());
    if (!peer_service_->Start(config_.peer_service_port)) {
      MORI_UMBP_ERROR("[PoolClient] PeerService failed to start on port {}",
                      config_.peer_service_port);
      peer_service_.reset();
      Shutdown();
      return false;
    }
    config_.peer_service_port = peer_service_->BoundPort();
  }

  std::string peer_address;
  if (peer_service_ != nullptr) {
    std::string host = config_.master_config.node_address;
    peer_address = host + ":" + std::to_string(config_.peer_service_port);
  }

  // Register conservatively with one instance per tier. Once the Master
  // confirms max_allocatable_bytes support, weighted heartbeats advertise the
  // full aggregate without making rolling upgrades over-admit large values.
  std::map<TierType, TierCapacity> tier_caps;
  for (auto* backend : registry_.All()) {
    if (registry_.Get(backend->Tier()) != backend) continue;
    auto cap = backend->Capacity();
    if (cap.total_bytes == 0) continue;
    tier_caps[backend->Tier()] = cap;
  }
  if (master_client_) {
    const auto logical_caps =
        default_pool_ == nullptr ? std::map<std::string, LogicalTierCapacity>{}
                                 : default_pool_->LogicalTierCapacities();
    auto status = master_client_->RegisterSelf(tier_caps, peer_address, engine_desc_bytes,
                                               logical_caps);
    if (!status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] RegisterSelf failed: {}", status.error_message());
      Shutdown();
      return false;
    }
    if (config_.master_config.auto_heartbeat) master_client_->StartHeartbeat();
    if (weighted_placement && !master_client_->SupportsMaxAllocatableCapacity()) {
      MORI_UMBP_WARN(
          "[PoolClient] Master lacks max_allocatable_bytes support; weighted placement remains "
          "enabled but capacity advertisement is limited to the first instance per tier");
    }
  }

  // Start the async re-cache worker only when something can feed it and this
  // node has an exportable local medium to install into.  Two independent
  // producers share it: cache_remote_fetches (whole-object BatchGet) and
  // ranged_locality_prefetch (remote ranged reads).
  if ((config_.cache_remote_fetches || config_.ranged_locality_prefetch) &&
      registry_.Get(medium_) != nullptr) {
    {
      std::lock_guard<std::mutex> lk(recache_mutex_);
      recache_stop_ = false;
    }
    recache_worker_ = std::thread([this] { ReCacheWorkerLoop(); });
  }

  if (!config_.workload_trace_path.empty()) {
    try {
      benchmark::WorkloadTraceRecorderOptions options;
      options.path = config_.workload_trace_path;
      options.client_id = config_.workload_trace_client_id;
      options.seed = config_.workload_trace_seed;
      options.node_id = config_.master_config.node_id;
      options.backend_policy = config_.policy_config_path;
      workload_recorder_ =
          std::make_unique<benchmark::WorkloadTraceRecorder>(std::move(options));
    } catch (const std::exception& exception) {
      MORI_UMBP_ERROR("[PoolClient] failed to open workload trace '{}': {}",
                      config_.workload_trace_path, exception.what());
      Shutdown();
      return false;
    }
  }

  MORI_UMBP_INFO("[PoolClient] Initialized node_id='{}'", config_.master_config.node_id);
  return true;
}

void PoolClient::Shutdown() {
  if (!initialized_) return;
  initialized_ = false;

  // Stop the async re-cache worker first: it calls ExecuteLocalPut (which uses
  // the registry + master_client_), so it must be joined before those are torn
  // down below.
  {
    std::lock_guard<std::mutex> lk(recache_mutex_);
    recache_stop_ = true;
    recache_queue_.clear();
    prefetch_inflight_.clear();
  }
  recache_cv_.notify_all();
  if (recache_worker_.joinable()) recache_worker_.join();
  if (workload_recorder_ != nullptr) {
    try {
      workload_recorder_->Close();
    } catch (const std::exception& exception) {
      MORI_UMBP_ERROR("[PoolClient] workload trace is incomplete: {}",
                      exception.what());
    }
    workload_recorder_.reset();
  }

  if (master_client_) {
    master_client_->StopHeartbeat();
    // Idempotent with ~MasterClient.
    master_client_->StopMetricsReporting();
    auto status = master_client_->UnregisterSelf();
    if (!status.ok()) {
      MORI_UMBP_WARN("[PoolClient] UnregisterSelf failed: {}", status.error_message());
    }
  }

  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    peers_.clear();
  }

  peer_service_.reset();

  // hipHostUnregister the gather regions before their backing memory goes
  // away: the medium's buffers die with the registry below, and the scratch
  // arena is caller-owned and freed after Shutdown returns.
  if (hbm_engine_ != nullptr) hbm_engine_->ClearHostGatherRegions();

  // Every backend deregisters its memory through transfer_engine_ inside its
  // own destructor — this MUST run before transfer_engine_ is torn down below.
  // MasterClient borrows the registry, so unbind it first.
  if (master_client_) {
    master_client_->SetPeerPool(nullptr);
    master_client_->SetBackendRegistry(nullptr);
  }
  default_pool_.reset();
  registry_ = BackendRegistry{};

  if (transfer_engine_) {
    std::unique_lock<std::shared_mutex> lock(registered_mem_mutex_);
    for (auto& reg : registered_regions_) transfer_engine_->Deregister(reg.ref);
    registered_regions_.clear();
  }
  peer_directory_ = nullptr;
  hbm_engine_ = nullptr;
  transfer_engine_.reset();

  master_client_.reset();
}

bool PoolClient::Clear() {
  // Vacuously done: nothing has been initialized so there is no state to
  // clear and no master to notify.  Treat as success so callers in
  // shutdown / teardown paths do not surface a spurious error.
  if (!initialized_.load()) return true;
  // Clear physical state and the default Pool's logical placement index before
  // the full-sync empty snapshot goes out.
  if (default_pool_ != nullptr) default_pool_->ClearLocal();

  bool ok = true;
  if (master_client_) {
    ok = master_client_->ClearFullSync();
    if (!ok) MORI_UMBP_WARN("[PoolClient] Clear full-sync heartbeat failed");
  }
  return ok;
}

bool PoolClient::IsInitialized() const { return initialized_; }
MasterClient& PoolClient::Master() { return *master_client_; }

void PoolClient::RouteAllPutsLocally(size_t count,
                                     std::vector<std::optional<RoutePutResult>>* routes) const {
  RoutePutResult local;
  local.node_id = config_.master_config.node_id;
  local.tier = medium_;
  routes->assign(count, local);
}

void PoolClient::CountMetric(std::string name, std::string help, MasterClient::Labels labels,
                             double delta) {
  if (master_client_ == nullptr) return;
  master_client_->AddCounter(std::move(name), std::move(help), std::move(labels), delta);
}
BackendRegistry& PoolClient::Backends() { return registry_; }
std::map<std::string, uint64_t> PoolClient::TierReadHits() const {
  return default_pool_ == nullptr ? std::map<std::string, uint64_t>{}
                                  : default_pool_->TierReadHits();
}

std::string PoolClient::LogicalTierForBackend(uint32_t backend_id) const {
  return default_pool_ == nullptr ? std::string{}
                                  : default_pool_->LogicalTierForBackend(backend_id);
}

TierTransitionMetrics PoolClient::TransitionMetrics() const {
  return default_pool_ == nullptr ? TierTransitionMetrics{}
                                  : default_pool_->TransitionMetrics();
}

// ---------------------------------------------------------------------------
//  Memory registration
// ---------------------------------------------------------------------------

bool PoolClient::RegisterMemory(void* ptr, size_t size, mori::io::MemoryLocationType loc,
                                int device, MemoryRegistration mode) {
  if (!transfer_engine_ && mode == MemoryRegistration::kPinned) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: transfer engine not available");
    return false;
  }
  if (ptr == nullptr || size == 0) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: invalid args ptr={}, size={}", ptr, size);
    return false;
  }
  std::unique_lock<std::shared_mutex> lock(registered_mem_mutex_);
  const auto at = std::lower_bound(
      registered_regions_.begin(), registered_regions_.end(), ptr,
      [](const RegisteredRegion& r, const void* p) { return std::less<const void*>{}(r.base, p); });
  if (at != registered_regions_.end() && at->base == ptr) {
    const RegisteredRegion& reg = *at;
    // Re-registering the same base with a larger size is not idempotent: the
    // existing registration covers fewer bytes, so FindRegisteredMemory would
    // start rejecting the tail and silently fall back to unregistered bytes.
    if (size <= reg.size) return true;
    MORI_UMBP_ERROR(
        "[PoolClient] RegisterMemory: ptr={} already registered with smaller size {}<{}", ptr,
        reg.size, size);
    return false;
  }

  if (mode == MemoryRegistration::kLocalCopyOnly) {
    // Nothing to pin and nothing that can throw: the ref carries exactly what
    // local engine selection reads off it, and every engine's Deregister
    // no-ops on a ref without a descriptor.
    TransferRef ref = TransferRef::HostBytes(ptr, size, loc, device);
    registered_regions_.insert(at, RegisteredRegion{ptr, size, std::move(ref)});
    return true;
  }

  // The engine may throw (mori-io pinning can fail on a region the NIC cannot
  // map).  A throw here would propagate out through the pybind boundary; the
  // documented contract is a bool.
  try {
    TransferRef ref = transfer_engine_->RegisterMemory(ptr, size, loc, device);
    // Validate what came back rather than trusting it. A ref whose location or
    // device disagrees with what the caller allocated would send every later
    // transfer to the wrong engine, and one that covers fewer bytes than asked
    // would be used for the full range.
    if (!ref.Valid() || ref.host_ptr != ptr || ref.size < size || ref.loc != loc ||
        (loc == mori::io::MemoryLocationType::GPU && device >= 0 && ref.device != device)) {
      MORI_UMBP_ERROR(
          "[PoolClient] RegisterMemory: engine returned an inconsistent ref for ptr={}, size={}, "
          "loc={}, device={}",
          ptr, size, static_cast<uint32_t>(loc), device);
      transfer_engine_->Deregister(ref);
      return false;
    }
    // Inserted in place rather than appended: lookups binary-search this.
    registered_regions_.insert(at, RegisteredRegion{ptr, size, std::move(ref)});
  } catch (const std::exception& error) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory failed for ptr={}, size={}: {}", ptr, size,
                    error.what());
    return false;
  } catch (...) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory failed for ptr={}, size={}: unknown error", ptr,
                    size);
    return false;
  }
  return true;
}

void PoolClient::DeregisterMemory(void* ptr) {
  if (ptr == nullptr) return;
  std::unique_lock<std::shared_mutex> lock(registered_mem_mutex_);
  auto it = std::lower_bound(
      registered_regions_.begin(), registered_regions_.end(), ptr,
      [](const RegisteredRegion& r, const void* p) { return std::less<const void*>{}(r.base, p); });
  if (it != registered_regions_.end() && it->base == ptr) {
    if (transfer_engine_) transfer_engine_->Deregister(it->ref);
    registered_regions_.erase(it);
  }
}

const PoolClient::RegisteredRegion* PoolClient::FindRegisteredRegionLocked(const void* ptr,
                                                                           size_t size) const {
  // Regions never overlap, so the only candidate is the last one whose base is
  // <= ptr.  upper_bound then step back finds it in O(log n).
  auto at = std::upper_bound(
      registered_regions_.begin(), registered_regions_.end(), ptr,
      [](const void* p, const RegisteredRegion& r) { return std::less<const void*>{}(p, r.base); });
  if (at == registered_regions_.begin()) return nullptr;
  --at;
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  const auto base = reinterpret_cast<uintptr_t>(at->base);
  if (size > at->size || (addr - base) > at->size - size) return nullptr;
  return &*at;
}

std::optional<std::pair<TransferRef, size_t>> PoolClient::FindRegisteredMemory(const void* ptr,
                                                                               size_t size) const {
  std::shared_lock<std::shared_mutex> lock(registered_mem_mutex_);
  const RegisteredRegion* reg = FindRegisteredRegionLocked(ptr, size);
  if (reg == nullptr) return std::nullopt;
  const auto offset = reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(reg->base);
  return std::pair{reg->ref, static_cast<size_t>(offset)};
}

std::pair<TransferRef, uint64_t> PoolClient::UserBufferRef(void* ptr, size_t size) const {
  auto reg = FindRegisteredMemory(ptr, size);
  if (reg.has_value()) return {reg->first, reg->second};
  return {ClassifiedUserBytes(ptr, size), 0};
}

// ---------------------------------------------------------------------------
//  Self-target paths
//
//  Not a "fast path" any more, and that is the headline of Phase 6: a local
//  access is just a transfer whose endpoints are both local, planned by the
//  same engine as everything else.  What used to make it special — a raw base
//  pointer per DRAM buffer, held by PoolClient, memcpy'd through directly — is
//  what made the local path host-DRAM-only and what forced PoolClient to name a
//  concrete backend type to obtain those pointers.
// ---------------------------------------------------------------------------

// Build one TransferItem per page between a caller buffer and a backend's own
// buffers.  `to_backend` is Put (user -> pages); false is Get (pages -> user).
// Returns false when the backend publishes no endpoint for a referenced buffer,
// which means this medium cannot serve the access in-process.
bool PoolClient::BuildLocalPageTransfers(MediumBackend* backend,
                                         const std::vector<PageLocation>& pages, uint64_t page_size,
                                         void* user, size_t size, bool to_backend,
                                         std::vector<TransferItem>* items) {
  if (pages.empty() || page_size == 0) return false;
  // Classified, not assumed host: this is the self-target path, where a device
  // user buffer paired with a host DRAM page must reach HbmCopyEngine rather
  // than LocalCopyEngine.  One classification per batch, not per page.
  const TransferRef user_ref = ClassifiedUserBytes(user, size);
  items->reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    TransferRef buf = backend->BufferRef(pages[i].buffer_index);
    if (!buf.HasHostPtr()) {
      MORI_UMBP_WARN("[PoolClient] local transfer: tier={} publishes no endpoint for buffer {}",
                     static_cast<int>(backend->Tier()), pages[i].buffer_index);
      return false;
    }
    TransferItem item;
    item.size = LogicalPageBytes(i, pages.size(), page_size, size);
    item.tag = i;
    if (to_backend) {
      item.src = user_ref;
      item.src_offset = i * page_size;
      item.dst = std::move(buf);
      item.dst_offset = PageOffset(pages[i], page_size);
    } else {
      item.src = std::move(buf);
      item.src_offset = PageOffset(pages[i], page_size);
      item.dst = user_ref;
      item.dst_offset = i * page_size;
    }
    items->push_back(std::move(item));
  }
  return true;
}

PoolClient::PutAttemptOutcome PoolClient::ExecuteLocalPut(const std::string& key, const void* src,
                                                          size_t size, TierType tier,
                                                          const std::string& logical_tier) {
  if (default_pool_ == nullptr || registry_.Empty()) {
    MORI_UMBP_ERROR("[PoolClient] Local Put requested but no default pool is initialized");
    return PutAttemptOutcome::kFatal;
  }
  PoolPlacementRequest request;
  request.key = key;
  request.size = size;
  request.tier = tier;
  request.logical_tier = logical_tier;
  auto pool_alloc = default_pool_->BatchAllocate({request}).front();
  auto* backend = registry_.Get(pool_alloc.backend_id);
  // A medium that publishes no buffer endpoints cannot be reached in-process at
  // all; route the key elsewhere rather than allocating a slot we cannot fill.
  if (backend == nullptr || backend->BufferCount() == 0) {
    if (pool_alloc.allocation.outcome == AllocateOutcome::kSuccessAllocated) {
      default_pool_->BatchAbort(
          {PoolSlotRef{pool_alloc.backend_id, pool_alloc.allocation.slot_id}});
    }
    MORI_UMBP_WARN("[PoolClient] Local Put: tier={} has no in-process-addressable backend",
                   static_cast<int>(tier));
    return PutAttemptOutcome::kRetry;
  }
  auto& alloc_res = pool_alloc.allocation;
  switch (alloc_res.outcome) {
    case AllocateOutcome::kSuccessAlreadyExists:
      return PutAttemptOutcome::kSuccessAlreadyExists;
    case AllocateOutcome::kFailed:
    case AllocateOutcome::kFailedNoSpace:
      // Backend already logged the specific reason.
      return PutAttemptOutcome::kRetry;
    case AllocateOutcome::kSuccessAllocated:
      break;
  }
  std::vector<TransferItem> items;
  if (!BuildLocalPageTransfers(backend, alloc_res.pages, alloc_res.page_size,
                               const_cast<void*>(src), size, /*to_backend=*/true, &items) ||
      !transfer_engine_->Transfer(items, /*failed_tags=*/nullptr)) {
    default_pool_->BatchAbort({PoolSlotRef{pool_alloc.backend_id, alloc_res.slot_id}});
    return PutAttemptOutcome::kFatal;
  }
  PoolCommitRequest commit;
  commit.slot = PoolSlotRef{pool_alloc.backend_id, alloc_res.slot_id};
  commit.key = key;
  if (!default_pool_->BatchCommit({commit}).front().commit.success) {
    default_pool_->BatchAbort({commit.slot});
    return PutAttemptOutcome::kFatal;
  }
  CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL,
              MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP, {{"traffic", "local"}},
              static_cast<double>(size));
  CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL,
              MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP, {{"traffic", "local"}},
              static_cast<double>(size));
  return PutAttemptOutcome::kSuccess;
}

PoolClient::GetAttemptOutcome PoolClient::ExecuteLocalGet(const std::string& key, void* dst,
                                                          size_t size) {
  if (default_pool_ == nullptr || registry_.Empty()) {
    MORI_UMBP_ERROR("[PoolClient] Local Get requested but no default pool is initialized");
    return GetAttemptOutcome::kFatal;
  }
  auto pool_resolved =
      ResolveLocalBatchWithBusyRetry(default_pool_.get(), {key}, /*include_descs=*/false).front();
  auto* backend = registry_.Get(pool_resolved.backend_id);
  bool served = pool_resolved.resolved.found && backend != nullptr;
  if (served) {
    auto& resolved = pool_resolved.resolved;
    // Same guard the remote path applies after BatchResolveKeys: a stored size
    // that disagrees with the requested one is a different object, and copying
    // `size` bytes out of a slot sized for something else would read past it.
    if (resolved.size != size) {
      MORI_UMBP_WARN("[PoolClient] local Get: size mismatch for key='{}' (wanted {}, got {})", key,
                     size, resolved.size);
      return GetAttemptOutcome::kRetry;
    }
    std::vector<TransferItem> items;
    if (!BuildLocalPageTransfers(backend, resolved.pages, resolved.page_size, dst, size,
                                 /*to_backend=*/false, &items)) {
      // This medium holds the key but cannot be read in-process (no published
      // endpoint for its buffers).  Route elsewhere rather than reporting a
      // miss, which would make the client exclude a node that does hold it.
      return GetAttemptOutcome::kRetry;
    }
    if (!transfer_engine_->Transfer(items, /*failed_tags=*/nullptr)) {
      return GetAttemptOutcome::kFatal;
    }
    served = true;
  }
  if (!served) return GetAttemptOutcome::kRetry;
  CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
              MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
              static_cast<double>(size));
  CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
              MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
              static_cast<double>(size));
  return GetAttemptOutcome::kSuccess;
}

void PoolClient::ResolveLocalBatch(const std::vector<std::string>& keys,
                                   const std::vector<size_t>& candidates,
                                   std::vector<MediumBackend*>* holders,
                                   std::vector<ResolvedEntry>* resolutions) {
  holders->assign(keys.size(), nullptr);
  if (resolutions != nullptr) resolutions->assign(keys.size(), ResolvedEntry{});
  if (candidates.empty() || default_pool_ == nullptr || registry_.Empty()) return;

  // The whole-batch case -- every key a candidate, none claimed yet -- is what a
  // layer-wise restore does, and there `keys` IS the query PeerPool needs. Naming
  // it directly skips a per-key string copy of the entire set; a partial set
  // (some keys already served from elsewhere) still needs its own vector.
  const std::vector<std::string>* query = &keys;
  std::vector<std::string> batch;
  if (candidates.size() != keys.size()) {
    batch.reserve(candidates.size());
    for (size_t i : candidates) batch.push_back(keys[i]);
    query = &batch;
  }

  auto found =
      ResolveLocalBatchWithBusyRetry(default_pool_.get(), *query, /*include_descs=*/false);
  for (size_t j = 0; j < candidates.size() && j < found.size(); ++j) {
    if (!found[j].resolved.found) continue;
    auto* backend = registry_.Get(found[j].backend_id);
    if (backend == nullptr) continue;
    (*holders)[candidates[j]] = backend;
    if (resolutions != nullptr) {
      (*resolutions)[candidates[j]] = std::move(found[j].resolved);
    }
  }
}

// ---------------------------------------------------------------------------
//  Ranged I/O — object ranges onto tier pages
// ---------------------------------------------------------------------------

bool PoolClient::BuildLocalRangeTransfers(MediumBackend* backend,
                                          const std::vector<PageLocation>& pages,
                                          uint64_t page_size, uint64_t stored_size,
                                          const std::vector<ObjectRange>& ranges, bool to_backend,
                                          size_t tag, std::vector<TransferItem>* items,
                                          double* classify_sink) {
  if (pages.empty() || page_size == 0) return false;

  // Describe each range's caller buffer once, ahead of the page walk, because a
  // range can produce several page fragments and the description belongs to the
  // range.  Callers already truncate `items` when this returns false, so bailing
  // before anything is emitted is equivalent to bailing partway through.
  //
  // Prefer the registration table.  A registered ref already carries loc and
  // device, so it answers what ClassifiedUserBytes asks hipPointerGetAttributes
  // for -- and it answers it for the whole region, not per range.  That matters
  // twice over here:
  //
  //   * The HIP call is per RANGE, and a layer-wise reader carries thousands.
  //   * A registered ref names the REGION base, so every range landing in one
  //     caller buffer shares a (src, dst) pair and LocalCopyEngine::Plan folds
  //     them into a single plan.  Describing them by their own addresses made
  //     each range its own plan, with its own vectors, and defeated the
  //     coalescing that Plan exists to do.
  //
  // In standalone-process mode the server registers every client region it
  // imports, so this is the path a real deployment takes; ClassifiedUserBytes
  // stays as the answer for genuinely unregistered memory, where the pointer
  // really does have to be interrogated.
  std::vector<TransferRef> user_refs;
  std::vector<uint64_t> user_offsets;
  {
    PhaseTimer classify_timer(classify_sink);
    user_refs.reserve(ranges.size());
    user_offsets.reserve(ranges.size());
    // One shared lock for the whole batch, not one per range.
    std::shared_lock<std::shared_mutex> lock(registered_mem_mutex_);
    for (const auto& range : ranges) {
      if (range.user == nullptr) return false;
      if (const RegisteredRegion* reg = FindRegisteredRegionLocked(range.user, range.size)) {
        user_refs.push_back(reg->ref);
        user_offsets.push_back(reinterpret_cast<uintptr_t>(range.user) -
                               reinterpret_cast<uintptr_t>(reg->base));
      } else {
        user_refs.push_back(ClassifiedUserBytes(range.user, range.size));
        user_offsets.push_back(0);
      }
    }
  }

  for (size_t r = 0; r < ranges.size(); ++r) {
    const auto& range = ranges[r];
    const TransferRef& user_ref = user_refs[r];
    const uint64_t user_base = user_offsets[r];

    // The walk owns every bound: object overflow, page-list overrun, and the
    // short last page a range must not run off the end of.
    const bool walked = ForEachRangePageFragment(
        pages, page_size, stored_size, range.object_offset, range.size,
        [&](size_t page_index, uint64_t tier_offset, size_t fragment, size_t copied) {
          TransferRef buf = backend->BufferRef(pages[page_index].buffer_index);
          if (!buf.HasHostPtr()) {
            MORI_UMBP_WARN(
                "[PoolClient] ranged transfer: tier={} publishes no endpoint for buffer {}",
                static_cast<int>(backend->Tier()), pages[page_index].buffer_index);
            return false;
          }
          TransferItem item;
          item.size = fragment;
          item.tag = tag;
          // `copied` is relative to the range; the ref may describe the whole
          // registered region the range sits inside, so bias by where the range
          // starts in it.  user_base is 0 for an unregistered pointer, whose ref
          // describes exactly the range.
          const uint64_t user_offset = user_base + copied;
          if (to_backend) {
            item.src = user_ref;
            item.src_offset = user_offset;
            item.dst = std::move(buf);
            item.dst_offset = tier_offset;
          } else {
            item.src = std::move(buf);
            item.src_offset = tier_offset;
            item.dst = user_ref;
            item.dst_offset = user_offset;
          }
          items->push_back(std::move(item));
          return true;
        });
    if (!walked) return false;
  }
  return true;
}

bool PoolClient::CopyContiguousToRanges(const void* src, size_t object_size,
                                        const std::vector<ObjectRange>& ranges) {
  // The arena is plain host memory this client owns, so it is described
  // directly rather than resolved through a backend's published endpoints.
  return CopyContiguousToRanges(TransferRef::HostBytes(const_cast<void*>(src), object_size),
                                /*src_base=*/0, object_size, ranges);
}

bool PoolClient::CopyContiguousToRanges(const TransferRef& src, uint64_t src_base,
                                        size_t object_size,
                                        const std::vector<ObjectRange>& ranges) {
  std::vector<TransferItem> items;
  items.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.size == 0 || range.user == nullptr) return false;
    if (IsObjectRangeOverflow(range.object_offset, range.size, object_size)) return false;
    TransferItem item;
    item.size = range.size;
    item.tag = 0;
    item.src = src;
    item.src_offset = src_base + range.object_offset;
    item.dst = ClassifiedUserBytes(range.user, range.size);
    item.dst_offset = 0;
    items.push_back(std::move(item));
  }
  return !items.empty() && transfer_engine_->Transfer(items, /*failed_tags=*/nullptr);
}

bool PoolClient::CopyRangesToContiguous(const std::vector<ObjectRange>& ranges, void* dst,
                                        size_t object_size) {
  const TransferRef object_ref = TransferRef::HostBytes(dst, object_size);
  std::vector<TransferItem> items;
  items.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.size == 0 || range.user == nullptr) return false;
    if (IsObjectRangeOverflow(range.object_offset, range.size, object_size)) return false;
    TransferItem item;
    item.size = range.size;
    item.tag = 0;
    item.src = ClassifiedUserBytes(range.user, range.size);
    item.src_offset = 0;
    item.dst = object_ref;
    item.dst_offset = range.object_offset;
    items.push_back(std::move(item));
  }
  return !items.empty() && transfer_engine_->Transfer(items, /*failed_tags=*/nullptr);
}

void PoolClient::ExecuteLocalPutRangesBatch(const std::vector<LocalRangeWriteRequest>& requests,
                                            std::vector<bool>* results, double* committed_bytes,
                                            const RangedPhaseSinks* sinks_or_null) {
  static const RangedPhaseSinks kNoSinks;
  const RangedPhaseSinks& sinks = sinks_or_null != nullptr ? *sinks_or_null : kNoSinks;
  if (requests.empty()) return;
  if (default_pool_ == nullptr || registry_.Empty()) {
    MORI_UMBP_ERROR("[PoolClient] Local ranged Put requested but no pool is initialized");
    return;
  }

  // Keep the optimized one-batch allocation shape, but let PeerPool choose
  // each backend so weighted/logical-tier placement and pending state remain
  // authoritative.
  std::vector<PoolAllocateResult> allocations;
  std::vector<MediumBackend*> backend_of(requests.size(), nullptr);
  {
    PhaseTimer alloc_timer(sinks.resolve);
    std::vector<PoolPlacementRequest> asks;
    asks.reserve(requests.size());
    for (const auto& request : requests) {
      asks.push_back(PoolPlacementRequest{*request.key, request.object_size, request.tier,
                                          /*backend_name=*/{}, request.logical_tier});
    }
    allocations = default_pool_->BatchAllocate(asks);
    allocations.resize(requests.size());
    for (size_t r = 0; r < requests.size(); ++r) {
      backend_of[r] = registry_.Get(allocations[r].backend_id);
    }
  }

  struct PendingWrite {
    size_t request_index = 0;
    PoolSlotRef slot;
  };
  std::vector<PoolSlotRef> to_abort;
  std::vector<PendingWrite> pending;
  std::vector<TransferItem> items;
  pending.reserve(requests.size());
  {
    size_t range_total = 0;
    for (const auto& request : requests) range_total += request.ranges.size();
    items.reserve(range_total);
  }

  for (size_t r = 0; r < requests.size(); ++r) {
    MediumBackend* const backend = backend_of[r];
    const auto& request = requests[r];
    const AllocateResult& alloc_res = allocations[r].allocation;
    if (alloc_res.outcome == AllocateOutcome::kSuccessAlreadyExists) {
      (*results)[request.result_index] = true;
      continue;
    }
    if (alloc_res.outcome != AllocateOutcome::kSuccessAllocated) continue;
    const PoolSlotRef slot{allocations[r].backend_id, alloc_res.slot_id};
    if (backend == nullptr || backend->BufferCount() == 0) {
      MORI_UMBP_WARN("[PoolClient] Local ranged Put: selected backend has no local endpoint");
      to_abort.push_back(slot);
      continue;
    }

    // tag = index into `requests`, so the engine's failed tags map straight
    // back to the slot that has to be aborted.
    const size_t before = items.size();
    bool built = false;
    {
      PhaseTimer build_timer(sinks.build);
      built = BuildLocalRangeTransfers(backend, alloc_res.pages, alloc_res.page_size,
                                       request.object_size, request.ranges, /*to_backend=*/true,
                                       /*tag=*/r, &items, sinks.classify);
    }
    if (!built) {
      items.resize(before);
      to_abort.push_back(slot);
      continue;
    }
    pending.push_back({r, slot});
  }
  // Classification is timed inside the build window; subtract it so the phases
  // stay disjoint.
  if (sinks.build != nullptr && sinks.classify != nullptr) *sinks.build -= *sinks.classify;
  if (sinks.items != nullptr) *sinks.items += items.size();
  auto flush_aborts = [this, &to_abort] {
    if (to_abort.empty()) return;
    default_pool_->BatchAbort(to_abort);
    to_abort.clear();
  };

  if (pending.empty()) {
    flush_aborts();
    return;
  }

  std::vector<size_t> failed_tags;
  transfer_engine_->Transfer(items, &failed_tags, sinks.steps);
  const std::unordered_set<size_t> failed(failed_tags.begin(), failed_tags.end());

  // Commit is still strictly after the transfer, and a slot that fails either
  // step is still aborted -- only the number of calls changes.
  std::vector<PoolCommitRequest> commits;
  std::vector<size_t> commit_owner;
  commits.reserve(pending.size());
  commit_owner.reserve(pending.size());
  for (const auto& write : pending) {
    if (failed.count(write.request_index) != 0) {
      to_abort.push_back(write.slot);
      continue;
    }
    commits.push_back(PoolCommitRequest{write.slot, *requests[write.request_index].key});
    commit_owner.push_back(write.request_index);
  }

  double put_bytes = 0.0;
  {
    PhaseTimer commit_timer(sinks.commit);
    auto outcomes = default_pool_->BatchCommit(commits);
    for (size_t j = 0; j < commit_owner.size(); ++j) {
      const auto& request = requests[commit_owner[j]];
      if (j >= outcomes.size() || !outcomes[j].commit.success) {
        to_abort.push_back(commits[j].slot);
        continue;
      }
      (*results)[request.result_index] = true;
      put_bytes += static_cast<double>(request.object_size);
    }
  }
  flush_aborts();

  if (committed_bytes != nullptr) *committed_bytes += put_bytes;
  if (put_bytes > 0.0) {
    // One update for the batch rather than two per key; the totals are the
    // same.  AddCounter locks and materializes its help text every call, which
    // is not something to do once per object.
    CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                put_bytes);
    CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                put_bytes);
  }
}

void PoolClient::MaybeReCacheAfterRemote(const std::string& key, const void* src, size_t size) {
  auto* local = registry_.Get(medium_);
  if (local == nullptr || local->BufferCount() == 0) return;  // no exportable local medium here
  // Admission gate (cache_remote_fetches / size==0 / NEVER / SIZE cap): shared
  // pure predicate, unit-tested in test_cache_remote_admission.cpp.
  if (!ShouldAdmitReCache(config_.cache_remote_fetches, config_.cache_remote_admission,
                          config_.admission_max_block_bytes, size)) {
    MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: key='{}' size={} not admitted", key,
                    size);
    return;
  }
  // ALWAYS and SIZE both delegate capacity enforcement to the peer allocator:
  // Allocate returns kFailedNoSpace when the medium is full, which we treat as
  // a best-effort miss (the remote read result is unaffected).

  // Prepare the job outside the queue lock: the source buffer is valid for this
  // call, but copying a multi-MiB block while holding recache_mutex_ would
  // serialize unrelated Get finalizers behind this memcpy.
  ReCacheJob job;
  job.key = key;
  job.bytes = std::unique_ptr<char[]>(new (std::nothrow) char[size]);
  if (!job.bytes) {
    MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: allocation failed for key='{}' size={}",
                    key, size);
    return;
  }
  job.size = size;
  HostCopyBlock(job.bytes.get(), src, size);

  // Enqueue for asynchronous install. The actual DRAM Allocate + copy +
  // Commit→KvEvent::ADD publish is performed by ReCacheWorkerLoop OFF the Get
  // critical path, so it does not add latency to concurrent Gets (the tail-round
  // TTFT blowup observed with a synchronous on-path install). Bounded queue →
  // drop-on-full keeps best-effort semantics.
  {
    std::lock_guard<std::mutex> lk(recache_mutex_);
    if (recache_stop_) return;
    if (recache_queue_.size() >= recache_queue_max_) {
      MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: queue full, dropping key='{}'", key);
      return;
    }
    recache_queue_.push_back(std::move(job));
  }
  recache_cv_.notify_one();
}

// How many whole-object prefetches one worker pass may take at once.  Large
// enough that the per-batch resolve RPC is amortised over a realistic page
// batch, small enough that one batch's slot allocations and wire time stay
// bounded when objects are multi-MiB.
constexpr size_t kPrefetchBatchMax = 256;

bool PoolClient::CanInstallLocally() const {
  auto* local = registry_.Get(medium_);
  return local != nullptr && local->BufferCount() != 0;
}

bool PoolClient::LocalityPrefetchAdmits(size_t object_size) const {
  if (!config_.ranged_locality_prefetch) return false;
  if (!CanInstallLocally()) return false;
  // Same admission policy as the non-ranged re-cache, minus its enable flag:
  // ranged_locality_prefetch is this path's switch and was checked above, so
  // the predicate is asked only about the policy and the block size.
  return ShouldAdmitReCache(/*cache_remote_fetches=*/true, config_.cache_remote_admission,
                            config_.admission_max_block_bytes, object_size);
}

void PoolClient::MaybeInstallCompleteArenaObject(const std::string& key, const void* arena_slice,
                                                 size_t object_size) {
  if (!CanInstallLocally()) return;
  // Synchronous on purpose, and cheap relative to the alternative: the slice is
  // live only until the next sub-batch reuses it, and a local copy of the
  // object beats a second RDMA of the same bytes.  This is also exactly what
  // the pre-ranged code did on every remote ranged read -- the difference is
  // that it now only happens when the object really is complete here.
  const auto installed = ExecuteLocalPut(key, arena_slice, object_size, medium_);
  if (installed != PutAttemptOutcome::kSuccess &&
      installed != PutAttemptOutcome::kSuccessAlreadyExists) {
    NoteRangedInstallFailure();
  }
}

void PoolClient::MaybePrefetchWholeObject(const std::string& key, size_t object_size,
                                          const RouteGetResult& route) {
  if (!LocalityPrefetchAdmits(object_size)) return;

  ReCacheJob job;
  job.key = key;
  job.size = object_size;
  job.route = route;  // bytes stays null: the worker pulls them itself

  {
    std::lock_guard<std::mutex> lk(recache_mutex_);
    if (recache_stop_) return;
    if (recache_queue_.size() >= recache_queue_max_) {
      MORI_UMBP_DEBUG("[PoolClient] MaybePrefetchWholeObject: queue full, dropping key='{}'", key);
      return;
    }
    if (!prefetch_inflight_.insert(key).second) return;  // already queued or running
    recache_queue_.push_back(std::move(job));
  }
  recache_cv_.notify_one();
}

void PoolClient::FetchWholeObjectsIntoMedium(std::vector<ReCacheJob>& jobs) {
  auto* backend = registry_.Get(medium_);
  if (backend == nullptr || backend->BufferCount() == 0 || jobs.empty()) return;

  // Drop anything that landed while it waited in the queue — a concurrent
  // request, or the offload path writing it back.  One batched resolve, before
  // any allocation: the alternative wastes a whole object of wire per key.
  std::vector<std::string> keys;
  keys.reserve(jobs.size());
  for (const auto& job : jobs) keys.push_back(job.key);
  const auto resolved = backend->BatchResolve(keys, /*include_descs=*/false);

  std::vector<size_t> wanted;
  std::vector<AllocateRequest> requests;
  wanted.reserve(jobs.size());
  requests.reserve(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    if (i < resolved.size() && resolved[i].found) continue;
    if (!jobs[i].route.has_value()) continue;
    wanted.push_back(i);
    requests.push_back(AllocateRequest{jobs[i].key, jobs[i].size});
  }
  if (requests.empty()) return;

  auto allocs = backend->BatchAllocate(requests);

  // Peer pages and medium pages are both registered host memory, so the objects
  // move peer -> medium as plain RDMA: no arena, no bounce buffer, no memcpy,
  // and the reader's destination is never touched.
  //
  // The remote get path describes its destination as one contiguous span, so
  // the slot has to be one too.  The page allocator already prefers a
  // same-buffer continuous run for exactly this reason, and distributed mode
  // stores one key per page, so the common cases are covered; a scattered slot
  // is left uncached rather than served by a path that would misplace it.
  //
  // slot_refs is reserved up front and never grows: the plan holds pointers
  // into it.
  BatchGetPlan plan;
  std::vector<TransferRef> slot_refs;
  std::vector<size_t> planned;  // index into wanted/allocs
  slot_refs.reserve(wanted.size());
  planned.reserve(wanted.size());

  for (size_t w = 0; w < wanted.size(); ++w) {
    auto& alloc = allocs[w];
    if (alloc.outcome != AllocateOutcome::kSuccessAllocated) continue;
    const auto& job = jobs[wanted[w]];

    bool contiguous = !alloc.pages.empty();
    for (size_t p = 1; p < alloc.pages.size() && contiguous; ++p) {
      contiguous = alloc.pages[p].buffer_index == alloc.pages.front().buffer_index &&
                   alloc.pages[p].page_index == alloc.pages.front().page_index + p;
    }
    TransferRef slot_buffer =
        contiguous ? backend->BufferRef(alloc.pages.front().buffer_index) : TransferRef{};
    if (!contiguous || !slot_buffer.Valid()) {
      MORI_UMBP_DEBUG("[PoolClient] prefetch: slot for key='{}' is not one addressable run",
                      job.key);
      // Aborted here and never revisited: the commit/abort pass below walks
      // `planned`, which this key does not enter.
      backend->BatchAbort({alloc.slot_id});
      continue;
    }

    slot_refs.push_back(std::move(slot_buffer));
    // Named by the backend's own ref rather than a raw pointer: the medium pool
    // is RDMA-reachable but was never handed to RegisterMemory, so
    // UserBufferRef would not find it and the transfer would fall through to a
    // bounce the staging pool may be too small for.
    plan.remote_groups[job.route->node_id].push_back(
        BatchGetItem{.index = planned.size(),
                     .key = &job.key,
                     .dst = nullptr,
                     .size = job.size,
                     .dst_ref = &slot_refs.back(),
                     .dst_ref_offset = PageOffset(alloc.pages.front(), alloc.page_size),
                     .route = *job.route});
    planned.push_back(w);
  }
  if (planned.empty()) return;

  std::vector<bool> fetched(planned.size(), false);
  ExecuteRemoteBatchGetPlan(plan, &fetched, /*recache_remote=*/false);

  std::vector<CommitRequest> commits;
  std::vector<uint64_t> aborts;
  commits.reserve(planned.size());
  // Everything this pass wanted but could not plan -- no slot, or a slot that
  // is not one addressable run -- already failed to become local.
  size_t not_installed = wanted.size() - planned.size();
  for (size_t i = 0; i < planned.size(); ++i) {
    auto& alloc = allocs[planned[i]];
    if (!fetched[i]) {
      aborts.push_back(alloc.slot_id);
      ++not_installed;
      continue;
    }
    commits.push_back(CommitRequest{alloc.slot_id, jobs[wanted[planned[i]]].key});
  }
  if (!commits.empty()) {
    const auto results = backend->BatchCommit(commits);
    for (size_t c = 0; c < results.size(); ++c) {
      if (!results[c].success) {
        aborts.push_back(commits[c].slot_id);
        ++not_installed;
      }
    }
  }
  if (!aborts.empty()) backend->BatchAbort(aborts);
  // Reported in one go rather than per key: this is a background pass, and the
  // counter is there to say how much locality is being lost, not to trace it.
  NoteRangedInstallFailure(not_installed);
}

void PoolClient::ReCacheWorkerLoop() {
  // Nobody is blocked on this thread.  Its whole purpose is to make LATER calls
  // faster, so it must not borrow cores from the calls happening NOW -- doing
  // so cost 63% of TTFL on a remote layer-wise restore at 8 ranks.  Set once
  // here rather than around each copy: everything this loop does is background
  // by construction.
  MarkThreadBackgroundCopies();

  // Reused across iterations so a steady stream of prefetches does not
  // reallocate this vector on every batch.
  std::vector<ReCacheJob> prefetch_batch;
  for (;;) {
    ReCacheJob job;
    {
      std::unique_lock<std::mutex> lk(recache_mutex_);
      recache_cv_.wait(lk, [this] { return recache_stop_ || !recache_queue_.empty(); });
      if (recache_stop_ && recache_queue_.empty()) return;
      job = std::move(recache_queue_.front());
      recache_queue_.pop_front();
      // A prefetch is worth batching: its cost per key is a peer resolve RPC
      // plus an RDMA round trip, and this is the only thread paying it.  Take
      // every prefetch already waiting, up to a cap that keeps one batch's
      // allocation and wire time bounded.
      if (job.bytes == nullptr) {
        prefetch_batch.clear();
        prefetch_batch.push_back(std::move(job));
        for (auto it = recache_queue_.begin();
             it != recache_queue_.end() && prefetch_batch.size() < kPrefetchBatchMax;) {
          if (it->bytes != nullptr) {
            ++it;  // an install job; leave it in order
            continue;
          }
          prefetch_batch.push_back(std::move(*it));
          it = recache_queue_.erase(it);
        }
      }
    }
    if (!prefetch_batch.empty()) {
      // Whole-object locality prefetch.  The dedup slots are released
      // afterwards, so a later miss can try again for whatever did not land.
      FetchWholeObjectsIntoMedium(prefetch_batch);
      {
        std::lock_guard<std::mutex> lk(recache_mutex_);
        for (const auto& done : prefetch_batch) prefetch_inflight_.erase(done.key);
      }
      MORI_UMBP_DEBUG("[PoolClient] ReCacheWorker: prefetched {} objects", prefetch_batch.size());
      prefetch_batch.clear();
      continue;
    }
    // Install into this node's medium. ExecuteLocalPut allocates a slot on that
    // backend, copies the bytes, and Commit queues a KvEvent::ADD that reaches
    // the master via heartbeat — mirroring the local Put publish path.
    // kSuccessAlreadyExists makes this idempotent for a repeat remote read of
    // the same key.
    switch (ExecuteLocalPut(job.key, job.bytes.get(), job.size, medium_)) {
      case PutAttemptOutcome::kSuccess:
        MORI_UMBP_DEBUG("[PoolClient] ReCacheWorker: re-cached key='{}' size={}", job.key,
                        job.size);
        break;
      case PutAttemptOutcome::kSuccessAlreadyExists:
        break;
      case PutAttemptOutcome::kRetry:
      case PutAttemptOutcome::kFatal:
        MORI_UMBP_DEBUG("[PoolClient] ReCacheWorker: local install failed for key='{}'", job.key);
        break;
    }
  }
}

// ---------------------------------------------------------------------------
//  BatchPut
// ---------------------------------------------------------------------------

bool PoolClient::Put(const std::string& key, const void* src, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<const void*> srcs{src};
  std::vector<size_t> sizes{size};
  auto results = BatchPut(keys, srcs, sizes);
  return !results.empty() && results[0];
}

std::vector<bool> PoolClient::BatchPut(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& srcs,
                                       const std::vector<size_t>& sizes) {
  const auto call_start = std::chrono::steady_clock::now();
  if (keys.size() != srcs.size() || keys.size() != sizes.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: vector length mismatch");
    return std::vector<bool>(keys.size(), false);
  }
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: client not initialized");
    return std::vector<bool>(keys.size(), false);
  }

  // Tri-state pipeline; projected to vector<bool> at return.
  std::vector<PutEntryOutcome> outcomes(keys.size(), PutEntryOutcome::kFailed);

  std::vector<uint64_t> block_sizes(keys.size());
  for (size_t i = 0; i < sizes.size(); ++i) block_sizes[i] = static_cast<uint64_t>(sizes[i]);
  std::vector<std::optional<RoutePutResult>> routes;
  std::unordered_set<std::string> excludes;
  if (!HasMaster()) {
    RouteAllPutsLocally(keys.size(), &routes);
  } else {
    auto status = master_client_->BatchRoutePut(keys, block_sizes, excludes, &routes);
    if (!status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: BatchRoutePut failed: {}", status.error_message());
      return std::vector<bool>(keys.size(), false);
    }
    if (routes.size() < keys.size()) routes.resize(keys.size());
  }

  BatchPutPlan plan = PartitionBatchPutTargets(keys, srcs, sizes, routes, &outcomes);
  ExecuteBatchPutPlan(plan, &outcomes);

  const auto call_end = std::chrono::steady_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(call_end - call_start).count();
  if (seconds > 0.0) {
    auto split = ComputeBatchBandwidthBytes(outcomes, sizes, routes, config_.master_config.node_id);
    ObserveBatchBandwidth(master_client_.get(), split.local, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP, "local");
    ObserveBatchBandwidth(master_client_.get(), split.remote, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP, "remote");
  }

  std::vector<bool> results(outcomes.size());
  for (size_t i = 0; i < outcomes.size(); ++i) {
    results[i] = (outcomes[i] != PutEntryOutcome::kFailed);
  }
  if (workload_recorder_ != nullptr) {
    std::vector<benchmark::WorkloadTraceOutcome> recorded(outcomes.size());
    for (size_t i = 0; i < outcomes.size(); ++i) {
      recorded[i] = outcomes[i] != PutEntryOutcome::kFailed
                        ? benchmark::WorkloadTraceOutcome::kSuccess
                        : benchmark::WorkloadTraceOutcome::kFailure;
    }
    workload_recorder_->RecordBatchPut(keys, sizes, recorded);
  }
  return results;
}

PoolClient::BatchPutPlan PoolClient::PartitionBatchPutTargets(
    const std::vector<std::string>& keys, const std::vector<const void*>& srcs,
    const std::vector<size_t>& sizes, const std::vector<std::optional<RoutePutResult>>& routes,
    std::vector<PutEntryOutcome>* results) {
  BatchPutPlan plan;
  const size_t count = keys.size();
  for (size_t i = 0; i < count; ++i) {
    // Zero-size puts are meaningless: leave the result kFailed, never execute.
    if (sizes[i] == 0) {
      MORI_UMBP_WARN("[PoolClient] BatchPut: skipping zero-size put for key='{}'", keys[i]);
      continue;
    }
    if (i >= routes.size() || !routes[i].has_value()) continue;
    const auto& route = routes[i].value();
    // Master-side dedup hit.
    if (route.outcome == RoutePutOutcome::kAlreadyExists) {
      (*results)[i] = PutEntryOutcome::kAlreadyExists;
      continue;
    }
    if (route.node_id == config_.master_config.node_id) {
      // Self-target: deferred (with its tier) so ExecuteBatchPutPlan can run the
      // local memcpy inside the remote-DRAM submit..wait window.
      plan.local_items.push_back(BatchPutItem{
          .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route});
      continue;
    }
    // No tier filter: every medium a peer advertises publishes registered
    // pages (SSD's are its staging arena — see ssd_backend.h), so the remote
    // put path is the same for all of them.  The old DRAM/HBM allowlist here
    // silently dropped puts master routed to a peer's SSD.
    plan.remote_groups[route.node_id].push_back(BatchPutItem{
        .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route});
  }
  return plan;
}

void PoolClient::ExecuteBatchPutPlan(const BatchPutPlan& plan,
                                     std::vector<PutEntryOutcome>* results) {
  // Deferred local puts, parallel: per-key memcpy is lock-free (the allocator
  // serializes Allocate/Commit); results is not vector<bool>-bit-packed, so
  // workers write distinct indices directly. AddCounter / timing stay here.
  auto run_local_put = [&]() {
    const auto& local = plan.local_items;
    if (local.empty()) return;
    const int nthr = LocalCopyThreads("UMBP_DRAM_WRITE_THREADS");
    const auto t0 = std::chrono::steady_clock::now();
    ParallelFor(local.size(), nthr, [&](size_t k) {
      const auto& item = local[k];
      switch (ExecuteLocalPut(*item.key, item.src, item.size, item.route.tier,
                              item.route.logical_tier)) {
        case PutAttemptOutcome::kSuccess:
          (*results)[item.index] = PutEntryOutcome::kSucceeded;
          break;
        case PutAttemptOutcome::kSuccessAlreadyExists:
          (*results)[item.index] = PutEntryOutcome::kAlreadyExists;
          break;
        case PutAttemptOutcome::kRetry:
        case PutAttemptOutcome::kFatal:
          break;
      }
    });
    if (std::getenv("UMBP_LOCAL_COPY_TIMING")) {
      double sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
      size_t tot = 0;
      for (const auto& item : local) tot += item.size;
      MORI_UMBP_INFO("[LocalCopy] PUT keys={} bytes={} threads={} elapsed_ms={:.3f} GiB_s={:.2f}",
                     local.size(), tot, nthr, sec * 1000.0,
                     tot / (sec > 0 ? sec : 1e-12) / (1024.0 * 1024 * 1024));
    }
  };

  // Submit every peer (not waited) to overlap the wire across peers, run the
  // local puts in that window, then wait all + commit.  On early exit the
  // engine handle's destructor drains; the wait does mapping + commit/abort.
  //
  // There is no longer an all-zero-copy / all-staging fork here.  Staging is
  // the engine's bounce pool, and a plan that needs it settles inside Submit,
  // so submit-all is unconditionally safe — a batch that mixes registered and
  // unregistered buffers, which used to be a contract violation the client
  // failed, now just works.
  std::vector<std::unique_ptr<RemotePutInFlight>> inflights;
  inflights.reserve(plan.remote_groups.size());
  for (const auto& [node_id, items] : plan.remote_groups) {
    if (auto f = SubmitRemoteBatchPut(items, results)) inflights.push_back(std::move(f));
  }
  run_local_put();
  for (auto& f : inflights) WaitRemoteBatchPut(*f, results);
}

std::unique_ptr<PoolClient::RemotePutInFlight> PoolClient::SubmitRemoteBatchPut(
    const std::vector<BatchPutItem>& items, std::vector<PutEntryOutcome>* results) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
  };
  if (peer_directory_ == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchPut: no RDMA engine configured (items={})",
                    items.size());
    fail_all();
    return nullptr;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchPut: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
    fail_all();
    return nullptr;
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  auto inflight = std::make_unique<RemotePutInFlight>();
  inflight->peer = &peer;
  inflight->stub = stub;

  // Abort already-allocated slots on a synchronous failure that returns nullptr
  // (no WaitRemoteBatchPut/finalize will run for them).
  auto abort_now = [&](std::vector<uint64_t> slot_ids) {
    if (slot_ids.empty()) return;
    ::umbp::BatchAbortSlotsRequest abort_req;
    for (uint64_t slot_id : slot_ids) abort_req.add_slot_ids(slot_id);
    ::umbp::BatchAbortSlotsResponse abort_resp;
    grpc::ClientContext abort_ctx;
    // Best-effort: a failed abort just leaves the slots for the peer reaper to
    // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
    auto s = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    if (!s.ok()) {
      MORI_UMBP_WARN(
          "[PoolClient] SubmitRemoteBatchPut: BatchAbortSlots({} slots) failed on {}: {}",
          slot_ids.size(), first.route.node_id, s.error_message());
    }
  };

  // Allocate RPC + per-key dedup/failure mapping; malformed slots go to
  // inflight->abort_slots. On total failure results are written and the
  // malformed list already aborted inside the callee — nothing left in flight.
  if (!AllocateRemotePutEntries(items, stub, &inflight->entries, &inflight->abort_slots, results)) {
    return nullptr;
  }

  // Abort everything allocated and fail every key: used on the paths that
  // return nullptr, where no WaitRemoteBatchPut/finalize will run.
  auto abort_everything = [&] {
    std::vector<uint64_t> all = std::move(inflight->abort_slots);
    for (auto& entry : inflight->entries) {
      all.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
    }
    abort_now(std::move(all));
  };

  std::vector<TransferItem> transfer_items;
  if (!BuildRemotePutTransfers(inflight->entries, first.route.node_id, &transfer_items)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchPut: BuildRemotePutTransfers failed, node='{}' entries={} "
        "-> aborting all slots",
        first.route.node_id, inflight->entries.size());
    abort_everything();
    return nullptr;
  }

  // Drop items whose entry failed during build.  Those failed entries ride in
  // inflight->entries and are aborted by FinalizeRemotePutEntries at wait time —
  // do NOT early-abort them here (they are not a whole-batch failure).
  std::vector<TransferItem> active;
  active.reserve(transfer_items.size());
  for (auto& item : transfer_items) {
    if (!inflight->entries[item.tag].failed) active.push_back(std::move(item));
  }
  if (active.empty()) {
    abort_everything();
    return nullptr;
  }

  TransferPlanSet planned = transfer_engine_->Plan(active);
  ApplyRejectedTags(inflight->entries, planned.rejected_tags, "RemotePut");
  if (planned.plans.empty()) {
    abort_everything();
    return nullptr;
  }
  // POST; do NOT wait.  Everything the post references — including any bytes
  // staged through the engine's bounce pool — is owned by the returned handle.
  inflight->handle = transfer_engine_->Submit(std::move(planned.plans));
  if (inflight->handle == nullptr) {
    abort_everything();
    return nullptr;
  }
  return inflight;
}

void PoolClient::WaitRemoteBatchPut(RemotePutInFlight& f, std::vector<PutEntryOutcome>* results) {
  if (f.drained) return;
  f.drained = true;
  std::vector<TransferFailure> failures;
  if (f.handle != nullptr) f.handle->Wait(&failures);
  ApplyTransferFailures(f.entries, failures, "RemotePut");
  FinalizeRemotePutEntries(f.entries, f.abort_slots, results, f.stub);
}

bool PoolClient::AllocateRemotePutEntries(const std::vector<BatchPutItem>& items,
                                          ::umbp::UMBPPeer::Stub* stub,
                                          std::vector<RemotePutEntry>* entries,
                                          std::vector<uint64_t>* abort_slots,
                                          std::vector<PutEntryOutcome>* results) {
  entries->clear();
  ::umbp::BatchAllocateSlotsRequest alloc_req;
  for (const auto& item : items) {
    auto* entry = alloc_req.add_entries();
    entry->set_size(item.size);
    entry->set_tier(static_cast<::umbp::TierType>(item.route.tier));
    entry->set_key(*item.key);
    entry->set_logical_tier(item.route.logical_tier);
  }

  ::umbp::BatchAllocateSlotsResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  auto alloc_status = stub->BatchAllocateSlots(&alloc_ctx, alloc_req, &alloc_resp);
  if (!alloc_status.ok() || alloc_resp.entries_size() != static_cast<int>(items.size())) {
    MORI_UMBP_WARN("[PoolClient] BatchAllocateSlots failed on {}: {}", items.front().route.node_id,
                   alloc_status.error_message());
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
    return false;
  }

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& resp_entry = alloc_resp.entries(static_cast<int>(i));
    const auto outcome = resp_entry.outcome();

    switch (outcome) {
      case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS:
        (*results)[item.index] = PutEntryOutcome::kAlreadyExists;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED:
      case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE:
        // Peer allocator already logged the specific reason.
        (*results)[item.index] = PutEntryOutcome::kFailed;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_UNSPECIFIED:
      default:
        // Unset / unknown — proto version skew or wire corruption.
        // Must NOT fall through into slot processing below.
        MORI_UMBP_ERROR(
            "[PoolClient] BatchAllocateSlots: bad outcome={} ({}) for key='{}' on node='{}'",
            static_cast<int>(outcome), OutcomeName(outcome),
            item.key ? *item.key : std::string{"<null>"}, items.front().route.node_id);
        (*results)[item.index] = PutEntryOutcome::kFailed;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED:
        break;
    }

    PoolClient::SlotPlan plan = FromAllocateSlotResponse(resp_entry);
    if (!SizeMatchesAllocation(item.size, plan.pages.size(), plan.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: malformed slot for key='{}'", *item.key);
      abort_slots->push_back(plan.slot_id);
      (*results)[item.index] = PutEntryOutcome::kFailed;
      continue;
    }

    RemotePutEntry entry;
    entry.result_index = item.index;
    entry.item = &item;
    entry.slot_id = plan.slot_id;
    entry.plan = std::move(plan);
    entries->push_back(std::move(entry));
  }

  if (entries->empty()) {
    if (!abort_slots->empty()) {
      ::umbp::BatchAbortSlotsRequest abort_req;
      for (uint64_t slot_id : *abort_slots) abort_req.add_slot_ids(slot_id);
      ::umbp::BatchAbortSlotsResponse abort_resp;
      grpc::ClientContext abort_ctx;
      // Best-effort: a failed abort just leaves the slots for the peer reaper to
      // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
      auto abort_status = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
      if (!abort_status.ok()) {
        MORI_UMBP_WARN(
            "[PoolClient] AllocateRemotePutEntries: BatchAbortSlots({} slots) failed: {}",
            abort_slots->size(), abort_status.error_message());
      }
      abort_slots->clear();
    }
    return false;
  }
  return true;
}

bool PoolClient::BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries,
                                         const std::string& node_id,
                                         std::vector<TransferItem>* items) {
  items->clear();

  // Hydrate every entry's descs first, then snapshot the peer's buffers once
  // PER BACKEND the batch touches: the loop below indexes a snapshot instead of
  // taking the engine's remote lock per page.  A batch may span media (each
  // item carries its own route.tier), and buffer_index is backend-local, so one
  // snapshot per peer would index the wrong medium's buffers.  A concurrent
  // hydrate can only add buffers, so a snapshot taken here is never stale for
  // the indices these entries reference.
  for (const auto& entry : entries) {
    if (!entry.plan.descs.empty()) peer_directory_->CacheRemoteBuffers(node_id, entry.plan.descs);
  }
  std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> snapshots;
  std::array<bool, kMaxBackendsPerPeer> snapped{};
  auto buffers_for = [&](uint32_t backend_id) -> const std::vector<TransferRef>& {
    static const std::vector<TransferRef> kNone;
    if (backend_id >= kMaxBackendsPerPeer) return kNone;
    if (!snapped[backend_id]) {
      snapshots[backend_id] = peer_directory_->RemoteBufferSnapshot(node_id, backend_id);
      snapped[backend_id] = true;
    }
    return snapshots[backend_id];
  };

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    // Whether this goes zero-copy or through the engine's bounce pool is the
    // engine's decision; this only names the endpoint.
    const auto [src, src_base] =
        UserBufferRef(const_cast<void*>(entry.item->src), entry.item->size);
    const std::vector<TransferRef>& remote = buffers_for(entry.plan.backend_id);

    std::vector<TransferItem> entry_items;
    entry_items.reserve(entry.plan.pages.size());
    for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
      const auto& page = entry.plan.pages[p];
      if (page.buffer_index >= remote.size() || !remote[page.buffer_index].HasMemoryDesc()) {
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemotePutTransfers: peer published no buffer, "
            "key='{}' backend={} buffer_index={} peer_buffers={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.plan.backend_id, page.buffer_index, remote.size(), page.page_index);
        entry.failed = true;
        entry_items.clear();
        break;
      }
      TransferItem item;
      item.tag = idx;
      item.src = src;
      item.src_offset = src_base + static_cast<uint64_t>(p) * entry.plan.page_size;
      item.dst = remote[page.buffer_index];
      item.dst_offset = static_cast<uint64_t>(page.page_index) * entry.plan.page_size;
      item.size =
          LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
      entry_items.push_back(std::move(item));
    }

    if (!entry_items.empty()) {
      items->insert(items->end(), std::make_move_iterator(entry_items.begin()),
                    std::make_move_iterator(entry_items.end()));
    }
  }
  return true;
}

void PoolClient::FinalizeRemotePutEntries(std::vector<RemotePutEntry>& entries,
                                          std::vector<uint64_t>& abort_slots,
                                          std::vector<PutEntryOutcome>* results,
                                          ::umbp::UMBPPeer::Stub* stub) {
  ::umbp::BatchCommitSlotsRequest commit_req;
  std::vector<size_t> commit_indices;
  commit_indices.reserve(entries.size());

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    if (entry.failed) {
      abort_slots.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
      continue;
    }
    auto* commit = commit_req.add_entries();
    commit->set_slot_id(entry.slot_id);
    commit->set_key(*entry.item->key);
    commit_indices.push_back(idx);
  }

  if (!commit_indices.empty()) {
    ::umbp::BatchCommitSlotsResponse commit_resp;
    grpc::ClientContext commit_ctx;
    auto commit_status = stub->BatchCommitSlots(&commit_ctx, commit_req, &commit_resp);
    if (!commit_status.ok() ||
        commit_resp.success_size() != static_cast<int>(commit_indices.size())) {
      const std::string& node_id = entries[commit_indices.front()].item->route.node_id;
      MORI_UMBP_WARN("[PoolClient] BatchCommitSlots failed on {}: {}", node_id,
                     commit_status.error_message());
      for (size_t idx : commit_indices) {
        abort_slots.push_back(entries[idx].slot_id);
        (*results)[entries[idx].result_index] = PutEntryOutcome::kFailed;
        entries[idx].failed = true;
      }
    } else {
      for (size_t i = 0; i < commit_indices.size(); ++i) {
        auto idx = commit_indices[i];
        auto& entry = entries[idx];
        if (commit_resp.success(static_cast<int>(i))) {
          CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL,
                      MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP,
                      {{"traffic", "remote"}}, static_cast<double>(entry.item->size));
          (*results)[entry.result_index] = PutEntryOutcome::kSucceeded;
        } else {
          // Peer allocator already logged the reason (SLOT_GONE / PRE_CLEAR).
          abort_slots.push_back(entry.slot_id);
          (*results)[entry.result_index] = PutEntryOutcome::kFailed;
          entry.failed = true;
        }
      }
    }
  }

  if (!abort_slots.empty()) {
    ::umbp::BatchAbortSlotsRequest abort_req;
    for (uint64_t slot_id : abort_slots) abort_req.add_slot_ids(slot_id);
    ::umbp::BatchAbortSlotsResponse abort_resp;
    grpc::ClientContext abort_ctx;
    // Best-effort: a failed abort just leaves the slots for the peer reaper to
    // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
    auto abort_status = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    if (!abort_status.ok()) {
      MORI_UMBP_WARN("[PoolClient] FinalizeRemotePutEntries: BatchAbortSlots({} slots) failed: {}",
                     abort_slots.size(), abort_status.error_message());
    }
    abort_slots.clear();
  }
}

// ---------------------------------------------------------------------------
//  BatchGet
// ---------------------------------------------------------------------------

bool PoolClient::Get(const std::string& key, void* dst, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<void*> dsts{dst};
  std::vector<size_t> sizes{size};
  auto results = BatchGet(keys, dsts, sizes);
  return !results.empty() && results[0];
}

// Serve `indices` from this node's own media in one resolve and one transfer.
// Hits are written into *results; anything this node could not serve is
// appended to *missed when the caller wants it back (null when there is no
// fallback left to try).
//
// Batched on both axes deliberately.  One BatchResolve per BACKEND, not per
// key: that mutex is shared with Allocate / Commit / Evict, and in
// standalone-process mode every rank on the node shares this client.  Then ONE
// Transfer for every key, tagged by key index so the engine attributes a
// failure to its key instead of failing the batch.  The per-key alternative
// measured 1.4-1.6x SLOWER than route-first at batch >= 32 despite issuing no
// RPC at all; batching it made local-first win at every batch size.
void PoolClient::ServeLocalGets(const std::vector<std::string>& keys,
                                const std::vector<void*>& dsts, const std::vector<size_t>& sizes,
                                const std::vector<size_t>& indices, std::vector<bool>* results,
                                std::vector<size_t>* missed) {
  const auto miss = [&](size_t i) {
    if (missed != nullptr) missed->push_back(i);
  };
  if (indices.empty()) return;

  std::vector<MediumBackend*> holders;
  std::vector<ResolvedEntry> resolutions;
  ResolveLocalBatch(keys, indices, &holders, &resolutions);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<TransferItem> items;
  std::vector<size_t> local_keys;
  local_keys.reserve(indices.size());
  for (size_t i : indices) {
    MediumBackend* const holder = holders[i];
    if (holder == nullptr) {
      miss(i);
      continue;
    }
    const ResolvedEntry& resolved = resolutions[i];
    // A stored size that disagrees with the requested one is a different
    // object; copying sizes[i] out of a slot sized for something else would
    // read past it.  Look for it elsewhere rather than report a hit.
    if (resolved.size != sizes[i]) {
      MORI_UMBP_WARN("[PoolClient] local Get: size mismatch for key='{}' (wanted {}, got {})",
                     keys[i], sizes[i], resolved.size);
      miss(i);
      continue;
    }
    const size_t before = items.size();
    if (!BuildLocalPageTransfers(holder, resolved.pages, resolved.page_size, dsts[i], sizes[i],
                                 /*to_backend=*/false, &items)) {
      // This medium holds the key but publishes no in-process endpoint for its
      // buffers.  Drop what was appended for it -- a half-built key must not
      // ride along in the batch -- and route it instead.
      items.resize(before);
      miss(i);
      continue;
    }
    for (size_t k = before; k < items.size(); ++k) items[k].tag = i;
    local_keys.push_back(i);
  }
  if (items.empty()) return;

  std::vector<size_t> failed_tags;
  transfer_engine_->Transfer(items, &failed_tags);
  const std::unordered_set<size_t> failed(failed_tags.begin(), failed_tags.end());

  double local_get_bytes = 0.0;
  for (size_t i : local_keys) {
    if (failed.count(i) != 0) {
      // Any failed fragment fails the whole key: the destination now holds a
      // partial object, and reporting a hit would hand the caller bytes it
      // cannot tell are incomplete.
      miss(i);
      continue;
    }
    (*results)[i] = true;
    local_get_bytes += static_cast<double>(sizes[i]);
  }
  if (local_get_bytes > 0.0) {
    CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                local_get_bytes);
    CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                local_get_bytes);
  }
  if (std::getenv("UMBP_LOCAL_COPY_TIMING")) {
    const double sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    MORI_UMBP_INFO("[LocalCopy] GET keys={} items={} bytes={} elapsed_ms={:.3f} GiB_s={:.2f}",
                   local_keys.size(), items.size(), local_get_bytes, sec * 1000.0,
                   local_get_bytes / (sec > 0 ? sec : 1e-12) / (1024.0 * 1024 * 1024));
  }
}

// Whole-batch entry point for the local-first arm: filter what cannot be served
// at all, serve the rest, and hand back the keys the master still has to route.
std::vector<size_t> PoolClient::ServeLocalBatchGet(const std::vector<std::string>& keys,
                                                   const std::vector<void*>& dsts,
                                                   const std::vector<size_t>& sizes,
                                                   std::vector<bool>* results) {
  std::vector<size_t> candidates;
  candidates.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    // Silently skipped here; PartitionBatchGetTargets sees every key and owns
    // the warning, so routing it through both would log each one twice.
    if (sizes[i] == 0) continue;
    candidates.push_back(i);
  }
  std::vector<size_t> missed;
  missed.reserve(candidates.size());
  ServeLocalGets(keys, dsts, sizes, candidates, results, &missed);
  return missed;
}

// Route the keys the local phase missed and group them by peer.  `routes` is
// indexed by ORIGINAL key index and left nullopt for anything served locally,
// which is how ComputeBatchBandwidthBytes already reads a missing route.
//
// Self is excluded: ServeLocalBatchGet already asked every medium here, so a
// route back to this node could only repeat that miss.  The returned plan has
// no local half for the same reason.  False means the routing RPC failed.
bool PoolClient::RouteGetsInto(const std::vector<std::string>& keys,
                               const std::vector<size_t>& indices, bool exclude_self,
                               std::vector<std::optional<RouteGetResult>>* routes) {
  if (indices.empty()) return true;
  // No master means no peers: this node already looked, so the miss is final
  // and every slot stays nullopt.  Not a failure -- there was nothing to ask.
  if (!HasMaster()) return true;

  std::vector<std::string> route_keys;
  route_keys.reserve(indices.size());
  for (size_t i : indices) route_keys.push_back(keys[i]);

  std::unordered_set<std::string> excludes;
  if (exclude_self) excludes.insert(config_.master_config.node_id);

  std::vector<std::optional<RouteGetResult>> answers;
  auto status = master_client_->BatchRouteGet(route_keys, excludes, &answers);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: BatchRouteGet failed: {}", status.error_message());
    return false;
  }
  // A short reply is not an error; the unanswered keys stay nullopt, which the
  // partition reads as "nowhere".
  answers.resize(indices.size());
  for (size_t r = 0; r < indices.size(); ++r) (*routes)[indices[r]] = answers[r];
  return true;
}

void PoolClient::ObserveBatchGetBandwidth(const std::vector<bool>& results,
                                          const std::vector<size_t>& sizes,
                                          const std::vector<std::optional<RouteGetResult>>& routes,
                                          std::chrono::steady_clock::time_point call_start) {
  const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - call_start)
                             .count();
  if (seconds <= 0.0) return;
  auto split = ComputeBatchBandwidthBytes(results, sizes, routes, config_.master_config.node_id);
  ObserveBatchBandwidth(master_client_.get(), split.local, seconds,
                        MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                        MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP, "local");
  ObserveBatchBandwidth(master_client_.get(), split.remote, seconds,
                        MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                        MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP, "remote");
}

std::vector<bool> PoolClient::BatchGet(const std::vector<std::string>& keys,
                                       const std::vector<void*>& dsts,
                                       const std::vector<size_t>& sizes) {
  const auto call_start = std::chrono::steady_clock::now();
  std::vector<bool> results(keys.size(), false);
  if (keys.size() != dsts.size() || keys.size() != sizes.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: vector length mismatch");
    return results;
  }
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: client not initialized");
    return results;
  }

  // Both arms are route-then-partition; they differ only in WHICH keys reach
  // the master and whether an unroutable key may still fall back to a local
  // read.  Left nullopt for anything served locally, which is how
  // ComputeBatchBandwidthBytes already reads a missing route.
  std::vector<std::optional<RouteGetResult>> routes(keys.size());
  const bool local_first = config_.local_first && !registry_.Empty();

  std::vector<size_t> to_route;
  if (local_first) {
    to_route = ServeLocalBatchGet(keys, dsts, sizes, &results);
    // The whole point of the flag: a batch this node could satisfy by itself
    // never touches the master.
    if (to_route.empty()) {
      ObserveBatchGetBandwidth(results, sizes, routes, call_start);
      return results;
    }
  } else {
    to_route.resize(keys.size());
    std::iota(to_route.begin(), to_route.end(), 0);
  }

  // A routing failure is not fatal to what is already in hand: under local-first
  // the phase-1 hits stand, since they never depended on this RPC.
  if (RouteGetsInto(keys, to_route, /*exclude_self=*/local_first, &routes)) {
    const BatchGetPlan plan = PartitionBatchGetTargets(
        keys, dsts, sizes, routes, local_first ? LocalFallback::kSkip : LocalFallback::kAllow);
    ExecuteBatchGetPlan(plan, keys, dsts, sizes, &results);
  }

  ObserveBatchGetBandwidth(results, sizes, routes, call_start);
  return results;
}

// ---------------------------------------------------------------------------
//  Ranged batch entry points
// ---------------------------------------------------------------------------

namespace {

// Fills `out` rather than returning a fresh vector: the caller loops over keys
// and one allocation per key adds up on a call carrying thousands of ranges.
void FillReadRanges(const std::vector<void*>& dsts, const std::vector<size_t>& sizes,
                    const std::vector<size_t>& offsets, std::vector<PoolClient::ObjectRange>* out) {
  out->clear();
  out->reserve(dsts.size());
  for (size_t r = 0; r < dsts.size(); ++r) out->push_back({dsts[r], sizes[r], offsets[r]});
}

std::vector<PoolClient::ObjectRange> MakeWriteRanges(const std::vector<const void*>& srcs,
                                                     const std::vector<size_t>& sizes,
                                                     const std::vector<size_t>& offsets) {
  std::vector<PoolClient::ObjectRange> out;
  out.reserve(srcs.size());
  for (size_t r = 0; r < srcs.size(); ++r) {
    out.push_back({const_cast<void*>(srcs[r]), sizes[r], offsets[r]});
  }
  return out;
}

}  // namespace

// The route-first arm of BatchGetRanges: one BatchRouteGet over EVERY key,
// issued before anything is served.
//
// Turning local_first off cannot mean "do not look locally" here -- the local
// medium is the only thing that can serve a local key on this path.  What it
// means is "ask the master the way the whole-object path used to", so the
// difference between the arms is purely WHEN and over HOW MANY keys the routing
// RPC is issued: all of them up front, versus just the misses (or none at all
// when the batch is entirely local).
//
// Uses the same exclude set as the missed-key routing below, so both arms send
// an identical request shape and a self-route cannot reach the remote planner
// from either one.  False means the RPC failed.
bool PoolClient::RouteAllRangesUpFront(const std::vector<std::string>& keys, double* route_sink,
                                       std::vector<std::optional<RouteGetResult>>* preroutes) {
  if (!HasMaster()) {
    // Nothing to pre-route; phase 2 will find every miss unroutable, which on a
    // single node is the correct answer rather than a degraded one.
    preroutes->assign(keys.size(), std::nullopt);
    return true;
  }
  std::unordered_set<std::string> excludes{config_.master_config.node_id};
  PhaseTimer route_timer(route_sink);
  auto status = master_client_->BatchRouteGet(keys, excludes, preroutes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGetRanges: BatchRouteGet failed: {}",
                    status.error_message());
    return false;
  }
  preroutes->resize(keys.size());
  return true;
}

// Routes for the keys the local phase missed, parallel to `missed`.
//
// Under local_first this is the only routing RPC the call makes.  Under the
// route-first arm phase 0 already routed every key, so this projects those
// answers onto the misses instead of paying a second round trip -- that arm
// must cost ONE routing RPC, not two, or a comparison against it measures the
// harness rather than the flag.  False means the RPC failed.
bool PoolClient::RouteMissedRanges(const std::vector<std::string>& route_keys,
                                   const std::vector<size_t>& missed,
                                   const std::vector<std::optional<RouteGetResult>>& preroutes,
                                   double* route_sink,
                                   std::vector<std::optional<RouteGetResult>>* routes) {
  if (!HasMaster()) {
    routes->assign(route_keys.size(), std::nullopt);
    return true;
  }
  if (!config_.local_first) {
    routes->reserve(missed.size());
    for (size_t index : missed) routes->push_back(preroutes[index]);
  } else {
    std::unordered_set<std::string> excludes{config_.master_config.node_id};
    PhaseTimer route_timer(route_sink);
    auto status = master_client_->BatchRouteGet(route_keys, excludes, routes);
    if (!status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] BatchGetRanges: BatchRouteGet failed: {}",
                      status.error_message());
      return false;
    }
  }
  routes->resize(route_keys.size());
  return true;
}

std::vector<bool> PoolClient::BatchGetRanges(const std::vector<std::string>& keys,
                                             const std::vector<std::vector<void*>>& dsts,
                                             const std::vector<std::vector<size_t>>& sizes,
                                             const std::vector<std::vector<size_t>>& src_offsets) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (!initialized_ || n == 0 || !RangeBatchShapeValid(n, dsts, sizes, src_offsets)) {
    return results;
  }

  // Observed at scope exit for whichever phases ran; see ScopedBatchBandwidth.
  double local_bytes = 0.0;
  double remote_bytes = 0.0;
  ScopedBatchBandwidth bandwidth(
      master_client_.get(), MORI_UMBP_METRIC_CLIENT_BATCH_GET_RANGES_BANDWIDTH,
      MORI_UMBP_METRIC_CLIENT_BATCH_GET_RANGES_BANDWIDTH_HELP, local_bytes, remote_bytes);

  // Sub-timers.  `dbg` is null when disabled, which makes every PhaseTimer inert.
  const bool ranged_debug = RangedDebugEnabled();
  RangedPhases phases;
  phases.keys = n;
  if (ranged_debug && n > 0) phases.key0 = &keys[0];
  RangedPhases* dbg = ranged_debug ? &phases : nullptr;
  ScopedRangedReport ranged_report("get", phases, ranged_debug, local_bytes, remote_bytes);

  // Accumulated over the batch and reported once at the end rather than per
  // key.  AddCounter takes a lock and builds a metric key out of the name and
  // labels, and the help text is a hundred-odd bytes that has to be
  // materialized on every call -- affordable once, not a thousand times.  The
  // counter totals are identical either way; only the number of updates
  // changes.
  double local_get_bytes = 0.0;
  auto record_local_get = [&](size_t index) {
    local_get_bytes +=
        static_cast<double>(std::accumulate(sizes[index].begin(), sizes[index].end(), size_t{0}));
  };
  auto flush_local_get_bytes = [&] {
    if (local_get_bytes == 0.0) return;
    local_bytes += local_get_bytes;
    CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                local_get_bytes);
    CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                local_get_bytes);
    local_get_bytes = 0.0;
  };
  // Declared after ScopedBatchBandwidth and ScopedRangedReport, so it runs
  // BEFORE either of them reads local_bytes.
  ScopeExit flush_guard{flush_local_get_bytes};

  // ---- Phase 0: the route-first arm (local_first = false) ----
  std::vector<std::optional<RouteGetResult>> preroutes;
  if (!config_.local_first &&
      !RouteAllRangesUpFront(keys, dbg ? &dbg->route : nullptr, &preroutes)) {
    return results;
  }

  // ---- Phase 1: everything this node already holds, in one submit ----
  //
  // Every key's items carry tag = key index, so the engine reports failures per
  // key and one bad key does not fail the batch.  This is why the local half
  // needs no per-key threading the way whole-object BatchGet does: the copies
  // are already batched into a single plan set.
  std::vector<TransferItem> local_items;
  std::vector<size_t> local_keys;
  std::vector<size_t> missed;
  missed.reserve(n);
  // A TransferItem carries two TransferRefs, each with a MemoryDesc, so growing
  // this vector by doubling relocates a lot of bytes.  One range is at least one
  // item -- more only when it straddles a page -- so the range count is a tight
  // lower bound and covers the common case exactly.
  {
    size_t range_total = 0;
    for (const auto& per_key : sizes) range_total += per_key.size();
    local_items.reserve(range_total);
  }
  std::vector<ObjectRange> key_ranges;

  // A key that asked for no ranges is never resolved -- there is nothing to
  // serve it from, and it stays false either way.
  std::vector<size_t> candidates;
  candidates.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (!sizes[i].empty()) candidates.push_back(i);
  }
  std::vector<MediumBackend*> holders;
  std::vector<ResolvedEntry> resolutions;
  {
    PhaseTimer resolve_timer(dbg ? &dbg->resolve : nullptr);
    ResolveLocalBatch(keys, candidates, &holders, &resolutions);
  }

  for (size_t i = 0; i < n; ++i) {
    if (sizes[i].empty()) continue;  // asked for nothing; stays false

    MediumBackend* const holder = holders[i];
    if (holder == nullptr) {
      missed.push_back(i);
      continue;
    }
    const ResolvedEntry& resolved = resolutions[i];
    // The get API takes no object_sizes; the resolved entry is where the true
    // stored size becomes known.  First hit labels the call.
    if (dbg != nullptr && dbg->object_size == 0) {
      dbg->object_size = static_cast<size_t>(resolved.size);
    }
    if (!RangesAreDisjointAndInBounds(static_cast<size_t>(resolved.size), sizes[i],
                                      src_offsets[i])) {
      MORI_UMBP_ERROR("[PoolClient] BatchGetRanges: invalid local ranges for key='{}' size={}",
                      keys[i], resolved.size);
      continue;  // a caller bug, not a miss — do not go looking for a replica
    }
    const size_t before = local_items.size();
    bool built = false;
    {
      PhaseTimer build_timer(dbg ? &dbg->build : nullptr);
      // Refilled per key rather than returned fresh: one allocation for the
      // call instead of one per key.
      FillReadRanges(dsts[i], sizes[i], src_offsets[i], &key_ranges);
      built = BuildLocalRangeTransfers(
          holder, resolved.pages, resolved.page_size, resolved.size, key_ranges,
          /*to_backend=*/false, /*tag=*/i, &local_items, dbg ? &dbg->classify : nullptr);
    }
    if (!built) {
      // The medium holds the key but cannot be read in-process.  Same call as
      // Same disposition the local get path takes on a size mismatch: look
      // reporting a miss the caller cannot distinguish from absence.
      local_items.resize(before);
      missed.push_back(i);
      continue;
    }
    local_keys.push_back(i);
  }
  if (dbg != nullptr) {
    // Classification is timed inside the build window, so subtract it out to
    // keep the phases disjoint and `other` meaningful.
    dbg->build -= dbg->classify;
    dbg->items += local_items.size();
  }
  if (!local_items.empty()) {
    std::vector<size_t> failed_tags;
    {
      PhaseTimer xfer_timer(dbg ? &dbg->xfer : nullptr);
      TransferEngine::StepTiming steps;
      if (dbg != nullptr) steps = {&dbg->xfer_plan, &dbg->xfer_submit, &dbg->xfer_wait};
      transfer_engine_->Transfer(local_items, &failed_tags, dbg != nullptr ? &steps : nullptr);
    }
    const std::unordered_set<size_t> failed(failed_tags.begin(), failed_tags.end());
    for (size_t i : local_keys) {
      if (failed.count(i) != 0) continue;
      results[i] = true;
      record_local_get(i);
      if (dbg != nullptr) dbg->local_keys += 1;
    }
  }
  if (dbg != nullptr) dbg->remote_keys = missed.size();
  if (missed.empty()) return results;

  // ---- Phase 2: the rest, through the GET scratch arena ----
  //
  // Only the requested spans cross the wire, packed back-to-back into the
  // arena.  A layer-wise reader asks for one layer group out of many, so
  // fetching the whole object would move an order of magnitude more bytes than
  // it uses, and would do it on the first group -- the one the caller cannot
  // overlap with anything.  Packing spans instead also multiplies how many keys
  // fit in one arena round by the same factor.
  //
  // GET has its own arena and mutex, so it overlaps a concurrent remote PUT
  // (which uses the separate PUT arena/mutex).
  char* const scratch_base = static_cast<char*>(config_.ranged_get_scratch_buffer);
  const size_t scratch_size = config_.ranged_get_scratch_size;
  if (scratch_base == nullptr || scratch_size == 0 ||
      !FindRegisteredMemory(scratch_base, scratch_size).has_value()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGetRanges: ranged scratch is absent or not registered");
    return results;
  }

  // Routing is a master RPC that never touches the arena, so it runs before the
  // lock is taken rather than holding every other arena user behind it.
  std::vector<std::string> route_keys;
  route_keys.reserve(missed.size());
  for (size_t index : missed) route_keys.push_back(keys[index]);
  std::vector<std::optional<RouteGetResult>> routes;
  if (!RouteMissedRanges(route_keys, missed, preroutes, dbg ? &dbg->route : nullptr, &routes)) {
    return results;
  }

  // No tier filter here.  Upstream restricts remote ranged reads to DRAM/HBM
  // because its remote path is a hand-written DRAM RDMA read; ours goes through
  // ExecuteRemoteBatchGetPlan, which reaches any medium the owning peer
  // publishes — including SSD, whose staging page is ordinary registered host
  // memory and so is readable at range granularity like any other.
  //
  // Each eligible key becomes one fetch unit: its spans, the bytes they total,
  // and where in the arena they will land.  Splitting by span prefix rather
  // than by key means an object larger than the arena is no longer unservable —
  // only a single span larger than the arena is.
  std::vector<RangeFetchUnit> units;
  units.reserve(missed.size());

  for (size_t r = 0; r < missed.size(); ++r) {
    const size_t original = missed[r];
    if (!routes[r].has_value()) continue;  // genuinely nowhere in the cluster
    const auto& route = *routes[r];
    const size_t object_size = static_cast<size_t>(route.size);
    if (route.node_id == config_.master_config.node_id || route.size == 0 ||
        route.size > std::numeric_limits<size_t>::max() ||
        !RangesAreDisjointAndInBounds(object_size, sizes[original], src_offsets[original])) {
      MORI_UMBP_ERROR("[PoolClient] BatchGetRanges: invalid route/ranges for key='{}' size={}",
                      keys[original], route.size);
      continue;
    }

    // Built aside and only spliced in once the WHOLE key has been laid out.  A
    // key can need several units, and a later range that cannot be served must
    // discard the earlier ones too -- committing them would fetch part of the
    // key and still report it whole, which the caller cannot detect.
    std::vector<RangeFetchUnit> key_units;
    RangeFetchUnit unit;
    unit.original = original;
    unit.object_size = object_size;
    unit.route = routes[r];
    // Ascending and gapless from byte 0 is what makes the arena slice equal the
    // object; a split breaks it because no single slice then holds everything.
    bool ascending_from_zero = true;
    bool split = false;
    bool servable = true;
    size_t cursor = 0;
    for (size_t j = 0; j < sizes[original].size(); ++j) {
      const size_t span_bytes = sizes[original][j];
      if (span_bytes > scratch_size) {
        MORI_UMBP_ERROR(
            "[PoolClient] BatchGetRanges: range exceeds ranged scratch key='{}' range={} "
            "scratch={}",
            keys[original], span_bytes, scratch_size);
        servable = false;
        break;
      }
      if (src_offsets[original][j] != cursor) ascending_from_zero = false;
      cursor += span_bytes;
      if (unit.bytes + span_bytes > scratch_size) {
        split = true;
        key_units.push_back(std::move(unit));
        unit = RangeFetchUnit{};
        unit.original = original;
        unit.object_size = object_size;
        unit.route = routes[r];
      }
      // The copy-out list stays one entry per caller range -- each has its own
      // buffer -- but the FETCH list does not have to.  Consecutive layers are
      // consecutive bytes of the object, and the arena packs them in the same
      // order, so a whole layer group is one contiguous read.
      //
      // The engine's planner would coalesce these too, but only after sorting
      // and bucketing every one of them: a 61-range whole-object read would
      // build 61 TransferItems per key just to merge them back into one.  That
      // CPU is invisible next to a multi-MiB object and very visible next to a
      // small one, so the merge happens here instead.
      unit.packed.push_back(
          ObjectRange{.user = dsts[original][j], .size = span_bytes, .object_offset = unit.bytes});
      if (!unit.spans.empty() &&
          unit.spans.back().object_offset + unit.spans.back().size == src_offsets[original][j]) {
        unit.spans.back().size += span_bytes;
      } else {
        unit.spans.push_back(
            ByteSpan{.object_offset = src_offsets[original][j], .size = span_bytes});
      }
      unit.bytes += span_bytes;
    }
    if (!servable) continue;  // key stays false; none of its units are queued
    if (!unit.spans.empty()) {
      unit.holds_whole_object = !split && ascending_from_zero && unit.bytes == object_size;
      key_units.push_back(std::move(unit));
    }
    units.insert(units.end(), std::make_move_iterator(key_units.begin()),
                 std::make_move_iterator(key_units.end()));
  }
  if (units.empty()) return results;

  // A key served by several units only succeeds if every one of them does.
  std::unordered_map<size_t, bool> unit_ok;
  for (const auto& unit : units) unit_ok.emplace(unit.original, true);

  // Whole-object units skip the arena entirely when a slot can be had; what
  // comes back is everything that still needs one.
  units = ServeWholeObjectUnitsFromMedium(keys, std::move(units), &unit_ok, &remote_bytes);
  if (units.empty()) {
    for (const auto& [original, ok] : unit_ok) results[original] = ok;
    return results;
  }

  // Only the arena users serialize; the local hits above were fully concurrent,
  // and so were the routing RPC and the slot-served units.
  std::unique_lock<std::mutex> scratch_lock(ranged_get_scratch_mutex_, std::defer_lock);
  {
    PhaseTimer lock_timer(dbg ? &dbg->lock : nullptr);
    scratch_lock.lock();
  }

  // Pack as many units into the arena as fit, fetch that group, then repeat.
  // Units are 64 B apart so two concurrently-filled slices never share a cache
  // line; spans WITHIN a unit are exactly packed, which is what lets a group of
  // adjacent layer ranges coalesce into a single wire segment.
  PhaseTimer remote_timer(dbg ? &dbg->remote : nullptr);
  size_t pos = 0;
  while (pos < units.size()) {
    std::vector<size_t> scratch_offsets;
    size_t cursor = 0;
    size_t end = pos;
    for (; end < units.size(); ++end) {
      size_t aligned = 0;
      if (!AlignUpChecked(cursor, kRangedScratchAlignment, &aligned) ||
          aligned > scratch_size - units[end].bytes) {
        break;
      }
      scratch_offsets.push_back(aligned);
      cursor = aligned + units[end].bytes;
    }
    // Unit sizing already caps every unit at scratch_size, so the first one
    // always fits and this loop always advances.
    const size_t count = end - pos;

    std::vector<std::string> sub_keys;
    std::vector<void*> sub_dsts;
    std::vector<size_t> sub_object_sizes;
    std::vector<size_t> sub_bytes;
    std::vector<const std::vector<ByteSpan>*> sub_spans;
    std::vector<std::optional<RouteGetResult>> sub_routes;
    sub_keys.reserve(count);
    sub_dsts.reserve(count);
    sub_object_sizes.reserve(count);
    sub_bytes.reserve(count);
    sub_spans.reserve(count);
    sub_routes.reserve(count);
    for (size_t j = 0; j < count; ++j) {
      const auto& unit = units[pos + j];
      sub_keys.push_back(keys[unit.original]);
      sub_dsts.push_back(scratch_base + scratch_offsets[j]);
      sub_object_sizes.push_back(unit.object_size);
      sub_bytes.push_back(unit.bytes);
      sub_spans.push_back(&unit.spans);
      sub_routes.push_back(unit.route);
    }

    std::vector<bool> fetched(count, false);
    BatchGetPlan plan = PartitionBatchGetRangeTargets(sub_keys, sub_dsts, sub_object_sizes,
                                                      sub_bytes, sub_spans, sub_routes);
    // recache_remote=false: a ranged entry never holds the whole object, so
    // there is nothing the generic re-cache could legally install.  Locality is
    // restored by MaybePrefetchWholeObject below instead.
    ExecuteRemoteBatchGetPlan(plan, &fetched, /*recache_remote=*/false);

    for (size_t j = 0; j < count; ++j) {
      const auto& unit = units[pos + j];
      if (!fetched[j]) {
        unit_ok[unit.original] = false;
        continue;
      }
      if (!CopyContiguousToRanges(sub_dsts[j], unit.bytes, unit.packed)) {
        unit_ok[unit.original] = false;
        continue;
      }
      // unit.bytes is both what the caller asked for and what crossed the wire:
      // the fetch is range-granular now, so the two no longer differ.
      remote_bytes += static_cast<double>(unit.bytes);

      // Make the NEXT read of this key local.  Two ways, and exactly one of
      // them applies:
      //   * the arena slice already IS the whole object -> install it from
      //     there, which costs one local copy and nothing on the wire.  Only
      //     units the medium could not give a slot to reach this;
      //   * it is not -> ask the background worker to pull the object.
      // Installing from the arena has to happen now, before the next sub-batch
      // reuses this slice.
      if (unit.holds_whole_object) {
        MaybeInstallCompleteArenaObject(sub_keys[j], sub_dsts[j], unit.object_size);
      } else {
        MaybePrefetchWholeObject(sub_keys[j], unit.object_size, *unit.route);
      }
    }
    pos = end;
  }

  for (const auto& [original, ok] : unit_ok) results[original] = ok;
  return results;
}

std::vector<bool> PoolClient::BatchPutRanges(const std::vector<std::string>& keys,
                                             const std::vector<size_t>& object_sizes,
                                             const std::vector<std::vector<const void*>>& srcs,
                                             const std::vector<std::vector<size_t>>& sizes,
                                             const std::vector<std::vector<size_t>>& dst_offsets) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (!initialized_ || n == 0 || object_sizes.size() != n ||
      !RangeBatchShapeValid(n, srcs, sizes, dst_offsets)) {
    return results;
  }

  // Observed at scope exit for whichever phases ran; see ScopedBatchBandwidth.
  double local_bytes = 0.0;
  double remote_bytes = 0.0;
  ScopedBatchBandwidth bandwidth(
      master_client_.get(), MORI_UMBP_METRIC_CLIENT_BATCH_PUT_RANGES_BANDWIDTH,
      MORI_UMBP_METRIC_CLIENT_BATCH_PUT_RANGES_BANDWIDTH_HELP, local_bytes, remote_bytes);

  const bool ranged_debug = RangedDebugEnabled();
  RangedPhases phases;
  phases.keys = n;
  if (ranged_debug && n > 0) {
    // One call is always one pool (the tree connector groups by PoolName), so
    // key/size from the first entry label the whole batch.
    phases.key0 = &keys[0];
    phases.object_size = object_sizes[0];
  }
  RangedPhases* dbg = ranged_debug ? &phases : nullptr;
  ScopedRangedReport ranged_report("put", phases, ranged_debug, local_bytes, remote_bytes);

  // A put must account for every byte of the object it commits, so the ranges
  // have to tile it — being merely disjoint and in bounds would leave gaps of
  // whatever the slot happened to hold.
  std::vector<size_t> valid;
  std::vector<std::string> route_keys;
  std::vector<uint64_t> route_sizes;
  valid.reserve(n);
  route_keys.reserve(n);
  route_sizes.reserve(n);
  auto validate_timer = std::make_unique<PhaseTimer>(dbg ? &dbg->validate : nullptr);
  for (size_t i = 0; i < n; ++i) {
    if (!RangesTileObject(object_sizes[i], sizes[i], dst_offsets[i])) {
      MORI_UMBP_ERROR("[PoolClient] BatchPutRanges: ranges do not tile key='{}' size={}", keys[i],
                      object_sizes[i]);
      continue;
    }
    valid.push_back(i);
    route_keys.push_back(keys[i]);
    route_sizes.push_back(object_sizes[i]);
  }
  validate_timer.reset();
  if (valid.empty()) return results;

  std::vector<std::optional<RoutePutResult>> routes;
  std::unordered_set<std::string> excludes;
  // Unconditional, unlike the get path, which routes only what it missed
  // locally.  Every put pays this RPC even when every key lands on this node.
  grpc::Status route_status;
  {
    PhaseTimer route_timer(dbg ? &dbg->route : nullptr);
    if (!HasMaster()) {
      RouteAllPutsLocally(route_keys.size(), &routes);
    } else {
      route_status = master_client_->BatchRoutePut(route_keys, route_sizes, excludes, &routes);
    }
  }
  if (!route_status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPutRanges: BatchRoutePut failed: {}",
                    route_status.error_message());
    return results;
  }
  routes.resize(valid.size());

  std::vector<size_t> remote;
  std::vector<LocalRangeWriteRequest> local_writes;
  remote.reserve(valid.size());
  local_writes.reserve(valid.size());
  for (size_t r = 0; r < valid.size(); ++r) {
    const size_t original = valid[r];
    if (!routes[r].has_value()) continue;
    const auto& route = *routes[r];
    if (route.outcome == RoutePutOutcome::kAlreadyExists) {
      results[original] = true;
      continue;
    }
    if (route.node_id != config_.master_config.node_id) {
      remote.push_back(r);
      continue;
    }
    // Routed here: write the ranges straight into the slot's pages.  No
    // assembly buffer — a TransferItem is a range, so the scattered sources
    // land where they belong.
    local_writes.push_back({original, &keys[original], object_sizes[original],
                            MakeWriteRanges(srcs[original], sizes[original], dst_offsets[original]),
                            route.tier, route.logical_tier});
  }
  // One transfer for every locally-routed key in the batch, not one per key.
  if (dbg != nullptr) {
    dbg->local_keys = local_writes.size();
    dbg->remote_keys = remote.size();
  }
  {
    PhaseTimer xfer_timer(dbg ? &dbg->xfer : nullptr);
    TransferEngine::StepTiming steps;
    RangedPhaseSinks sinks;
    if (dbg != nullptr) {
      steps = {&dbg->xfer_plan, &dbg->xfer_submit, &dbg->xfer_wait};
      sinks = {&dbg->resolve, &dbg->classify, &dbg->build, &dbg->commit, &dbg->items, &steps};
    }
    ExecuteLocalPutRangesBatch(local_writes, &results, &local_bytes,
                               dbg != nullptr ? &sinks : nullptr);
  }
  if (dbg != nullptr) {
    // The sub-phases above are all timed inside the xfer window, so xfer alone
    // is left holding the engine's copy.  Subtract rather than re-time: nesting
    // is what makes each sub-phase attributable to the call it belongs to.
    dbg->xfer -= dbg->resolve + dbg->classify + dbg->build + dbg->commit;
  }
  if (remote.empty()) return results;

  // A key routed to another node has to be one contiguous object on the wire,
  // so this is the one direction that needs the arena.  PUT has its own arena
  // and mutex, so it overlaps a concurrent remote GET (which uses the separate
  // GET arena/mutex).
  char* const scratch_base = static_cast<char*>(config_.ranged_put_scratch_buffer);
  const size_t scratch_size = config_.ranged_put_scratch_size;
  if (scratch_base == nullptr || scratch_size == 0 ||
      !FindRegisteredMemory(scratch_base, scratch_size).has_value()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPutRanges: ranged scratch is absent or not registered");
    return results;
  }
  std::unique_lock<std::mutex> scratch_lock(ranged_put_scratch_mutex_, std::defer_lock);
  {
    PhaseTimer lock_timer(dbg ? &dbg->lock : nullptr);
    scratch_lock.lock();
  }

  PhaseTimer remote_timer(dbg ? &dbg->remote : nullptr);
  size_t pos = 0;
  while (pos < remote.size()) {
    std::vector<size_t> scratch_offsets;
    size_t cursor = 0;
    size_t end = pos;
    for (; end < remote.size(); ++end) {
      const size_t object_size = object_sizes[valid[remote[end]]];
      size_t aligned = 0;
      if (!AlignUpChecked(cursor, kRangedScratchAlignment, &aligned) ||
          object_size > scratch_size || aligned > scratch_size - object_size) {
        break;
      }
      scratch_offsets.push_back(aligned);
      cursor = aligned + object_size;
    }
    if (end == pos) {
      const size_t original = valid[remote[pos]];
      MORI_UMBP_ERROR(
          "[PoolClient] BatchPutRanges: object exceeds ranged scratch key='{}' size={} scratch={}",
          keys[original], object_sizes[original], scratch_size);
      ++pos;
      continue;
    }

    const size_t count = end - pos;
    std::vector<std::string> sub_keys;
    std::vector<const void*> sub_srcs;
    std::vector<size_t> sub_sizes;
    std::vector<std::optional<RoutePutResult>> sub_routes;
    std::vector<size_t> sub_originals;
    sub_keys.reserve(count);
    sub_srcs.reserve(count);
    sub_sizes.reserve(count);
    sub_routes.reserve(count);
    sub_originals.reserve(count);
    for (size_t j = 0; j < count; ++j) {
      const size_t route_index = remote[pos + j];
      const size_t original = valid[route_index];
      void* slice = scratch_base + scratch_offsets[j];
      if (!CopyRangesToContiguous(
              MakeWriteRanges(srcs[original], sizes[original], dst_offsets[original]), slice,
              object_sizes[original])) {
        MORI_UMBP_ERROR("[PoolClient] BatchPutRanges: assembly failed for key='{}'",
                        keys[original]);
        continue;
      }
      sub_keys.push_back(keys[original]);
      sub_srcs.push_back(slice);
      sub_sizes.push_back(object_sizes[original]);
      sub_routes.push_back(routes[route_index]);
      sub_originals.push_back(original);
    }

    if (!sub_keys.empty()) {
      std::vector<PutEntryOutcome> outcomes(sub_keys.size(), PutEntryOutcome::kFailed);
      BatchPutPlan plan =
          PartitionBatchPutTargets(sub_keys, sub_srcs, sub_sizes, sub_routes, &outcomes);
      ExecuteBatchPutPlan(plan, &outcomes);
      for (size_t j = 0; j < outcomes.size(); ++j) {
        results[sub_originals[j]] = outcomes[j] != PutEntryOutcome::kFailed;
        // kAlreadyExists is success to the caller but moved nothing, matching
        // IsCountedForBandwidth's rule for the whole-object BatchPut family.
        if (outcomes[j] == PutEntryOutcome::kSucceeded) {
          remote_bytes += static_cast<double>(sub_sizes[j]);
        }
      }
    }
    pos = end;
  }
  if (workload_recorder_ != nullptr) {
    std::vector<benchmark::WorkloadTraceOutcome> recorded(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      recorded[i] = results[i] ? benchmark::WorkloadTraceOutcome::kSuccess
                               : benchmark::WorkloadTraceOutcome::kFailure;
    }
    workload_recorder_->RecordBatchPut(keys, object_sizes, recorded);
  }
  return results;
}

PoolClient::BatchGetPlan PoolClient::PartitionBatchGetTargets(
    const std::vector<std::string>& keys, const std::vector<void*>& dsts,
    const std::vector<size_t>& sizes, const std::vector<std::optional<RouteGetResult>>& routes,
    LocalFallback fallback) {
  const bool allow_local = fallback == LocalFallback::kAllow && !registry_.Empty();
  BatchGetPlan plan;
  for (size_t i = 0; i < keys.size(); ++i) {
    // Zero-size gets are rejected before local fallback or remote read: an
    // explicit skip is required here because a nullopt route below would
    // otherwise fall through to a local read (result stays false).
    if (sizes[i] == 0) {
      MORI_UMBP_WARN("[PoolClient] BatchGet: skipping zero-size get for key='{}'", keys[i]);
      continue;
    }
    // No route, or one pointing back at this node.  Under kAllow that is a
    // local read the caller has not attempted yet -- deferred as an index so
    // ExecuteBatchGetPlan can run it inside the remote in-flight window.  Under
    // kSkip the local media were already asked, and the key is either served
    // (routes[i] stayed nullopt) or genuinely absent; either way it is done.
    if (i >= routes.size() || !routes[i].has_value() ||
        routes[i]->node_id == config_.master_config.node_id) {
      if (allow_local) plan.local_indices.push_back(i);
      continue;
    }
    const auto& route = *routes[i];
    plan.remote_groups[route.node_id].push_back(BatchGetItem{
        .index = i, .key = &keys[i], .dst = dsts[i], .size = sizes[i], .route = route});
  }
  return plan;
}

std::vector<PoolClient::RangeFetchUnit> PoolClient::ServeWholeObjectUnitsFromMedium(
    const std::vector<std::string>& keys, std::vector<RangeFetchUnit> units,
    std::unordered_map<size_t, bool>* unit_ok, double* remote_bytes) {
  auto* backend = registry_.Get(medium_);
  std::vector<RangeFetchUnit> leftover;
  leftover.reserve(units.size());

  // Only a unit that is the whole object can go straight into a slot.  No
  // switch and no size cap: the bytes are crossing the wire either way, so
  // landing them in a slot instead of the arena costs nothing extra and is what
  // the pre-ranged code did on every remote ranged read.
  const bool can_install = CanInstallLocally();
  std::vector<size_t> candidates;
  for (size_t i = 0; i < units.size(); ++i) {
    if (can_install && units[i].holds_whole_object) {
      candidates.push_back(i);
    } else {
      leftover.push_back(std::move(units[i]));
    }
  }
  if (candidates.empty()) return leftover;

  // Already here? Then phase 1 raced us to it; hand the unit back rather than
  // allocate a second slot for a key the medium already holds.
  std::vector<std::string> candidate_keys;
  candidate_keys.reserve(candidates.size());
  for (size_t i : candidates) candidate_keys.push_back(keys[units[i].original]);
  const auto resolved = backend->BatchResolve(candidate_keys, /*include_descs=*/false);

  std::vector<size_t> wanted;
  std::vector<AllocateRequest> requests;
  wanted.reserve(candidates.size());
  requests.reserve(candidates.size());
  for (size_t c = 0; c < candidates.size(); ++c) {
    if (c < resolved.size() && resolved[c].found) {
      leftover.push_back(std::move(units[candidates[c]]));
      continue;
    }
    wanted.push_back(candidates[c]);
    requests.push_back(AllocateRequest{candidate_keys[c], units[candidates[c]].object_size});
  }
  if (requests.empty()) return leftover;

  auto allocs = backend->BatchAllocate(requests);

  // slot_refs and slot_offsets are reserved up front and never grow: the plan
  // holds pointers into the first.
  BatchGetPlan plan;
  std::vector<TransferRef> slot_refs;
  std::vector<uint64_t> slot_offsets;
  std::vector<size_t> planned;  // index into wanted/allocs
  slot_refs.reserve(wanted.size());
  slot_offsets.reserve(wanted.size());
  planned.reserve(wanted.size());

  for (size_t w = 0; w < wanted.size(); ++w) {
    auto& alloc = allocs[w];
    RangeFetchUnit& unit = units[wanted[w]];
    if (alloc.outcome != AllocateOutcome::kSuccessAllocated) {
      leftover.push_back(std::move(unit));  // medium is full; the arena still works
      continue;
    }

    // The remote get path names its destination as one contiguous span, so the
    // slot has to be one.  The page allocator already prefers a same-buffer
    // continuous run, and distributed mode stores one key per page, so the
    // common cases are covered.
    bool contiguous = !alloc.pages.empty();
    for (size_t q = 1; q < alloc.pages.size() && contiguous; ++q) {
      contiguous = alloc.pages[q].buffer_index == alloc.pages.front().buffer_index &&
                   alloc.pages[q].page_index == alloc.pages.front().page_index + q;
    }
    TransferRef slot_buffer =
        contiguous ? backend->BufferRef(alloc.pages.front().buffer_index) : TransferRef{};
    if (!contiguous || !slot_buffer.Valid() || !slot_buffer.HasHostPtr()) {
      backend->BatchAbort({alloc.slot_id});
      NoteRangedInstallFailure();  // the arena will still serve it, but not cache it
      leftover.push_back(std::move(unit));
      continue;
    }

    const uint64_t slot_offset = PageOffset(alloc.pages.front(), alloc.page_size);
    slot_offsets.push_back(slot_offset);
    slot_refs.push_back(std::move(slot_buffer));
    // Named by the backend's own ref rather than a raw pointer: the medium pool
    // is RDMA-reachable but was never handed to RegisterMemory, so
    // UserBufferRef would not find it.
    plan.remote_groups[unit.route->node_id].push_back(BatchGetItem{.index = planned.size(),
                                                                   .key = &keys[unit.original],
                                                                   .dst = nullptr,
                                                                   .size = unit.object_size,
                                                                   .dst_bytes = unit.bytes,
                                                                   .spans = &unit.spans,
                                                                   .dst_ref = &slot_refs.back(),
                                                                   .dst_ref_offset = slot_offset,
                                                                   .route = *unit.route});
    planned.push_back(w);
  }
  if (planned.empty()) return leftover;

  std::vector<bool> fetched(planned.size(), false);
  ExecuteRemoteBatchGetPlan(plan, &fetched, /*recache_remote=*/false);

  std::vector<CommitRequest> commits;
  std::vector<uint64_t> aborts;
  commits.reserve(planned.size());
  for (size_t i = 0; i < planned.size(); ++i) {
    auto& alloc = allocs[planned[i]];
    RangeFetchUnit& unit = units[wanted[planned[i]]];
    if (!fetched[i]) {
      // The bytes never arrived, so there is nothing worth keeping.  Not a
      // fallback to the arena: see the note on the declaration.
      aborts.push_back(alloc.slot_id);
      (*unit_ok)[unit.original] = false;
      continue;
    }
    // Copy out BEFORE committing.  A pending slot cannot be reclaimed; a
    // committed one is an ordinary evictable object, and reading it after that
    // races the allocator.
    // The backend's own ref, not a bare pointer: an HBM pool hands back a
    // device pointer, and re-wrapping it as host bytes makes the engine copy
    // in the wrong direction (or memcpy device memory).
    if (!CopyContiguousToRanges(slot_refs[i], slot_offsets[i], unit.object_size, unit.packed)) {
      (*unit_ok)[unit.original] = false;
    }
    // Commit either way: the slot holds the object whether or not the caller's
    // copy-out worked, and caching it still helps the next reader.
    commits.push_back(CommitRequest{alloc.slot_id, keys[unit.original]});
  }
  if (!commits.empty()) {
    const auto committed = backend->BatchCommit(commits);
    size_t refused = 0;
    for (size_t c = 0; c < committed.size(); ++c) {
      if (!committed[c].success) {
        aborts.push_back(commits[c].slot_id);
        ++refused;
      }
    }
    NoteRangedInstallFailure(refused);
  }
  if (!aborts.empty()) backend->BatchAbort(aborts);
  return leftover;
}

PoolClient::BatchGetPlan PoolClient::PartitionBatchGetRangeTargets(
    const std::vector<std::string>& keys, const std::vector<void*>& arena_slices,
    const std::vector<size_t>& object_sizes, const std::vector<size_t>& packed_bytes,
    const std::vector<const std::vector<ByteSpan>*>& spans,
    const std::vector<std::optional<RouteGetResult>>& routes) {
  BatchGetPlan plan;
  for (size_t i = 0; i < keys.size(); ++i) {
    // Unroutable and self-routed keys were filtered by the caller, which is
    // what makes this plan remote-only.  Anything that slips through is a
    // caller bug, not a local read to fall back to.
    if (i >= routes.size() || !routes[i].has_value() || packed_bytes[i] == 0 ||
        spans[i] == nullptr || spans[i]->empty()) {
      continue;
    }
    plan.remote_groups[routes[i]->node_id].push_back(BatchGetItem{.index = i,
                                                                  .key = &keys[i],
                                                                  .dst = arena_slices[i],
                                                                  .size = object_sizes[i],
                                                                  .dst_bytes = packed_bytes[i],
                                                                  .spans = spans[i],
                                                                  .route = *routes[i]});
  }
  return plan;
}

void PoolClient::ExecuteBatchGetPlan(const BatchGetPlan& plan, const std::vector<std::string>& keys,
                                     const std::vector<void*>& dsts,
                                     const std::vector<size_t>& sizes, std::vector<bool>* results,
                                     bool recache_remote) {
  // The local half, run inside the remote submit..wait window so it overlaps
  // the wire.  One batched resolve and one Transfer -- the same core the
  // local-first arm uses; nothing left to route, so misses simply stay false.
  auto run_local = [&]() {
    ServeLocalGets(keys, dsts, sizes, plan.local_indices, results, /*missed=*/nullptr);
  };

  ExecuteRemoteBatchGetPlan(plan, results, recache_remote, run_local);
}

void PoolClient::ExecuteRemoteBatchGetPlan(const BatchGetPlan& plan, std::vector<bool>* results,
                                           bool recache_remote,
                                           const std::function<void()>& in_flight_window) {
  // Submit every peer (posted, not waited) to overlap wire time across peers,
  // run whatever the caller wants overlapped in that window, then wait all.  On
  // early/exceptional exit the engine handle's destructor drains in-flight
  // statuses (lifetime safety); the wait loop does failure mapping + backfill.
  //
  // As in Put, there is no all-zero-copy / all-staging fork any more: staging
  // lives in the engine's bounce pool and a plan that needs it settles inside
  // Submit, so submit-all is unconditionally safe.
  std::vector<std::unique_ptr<RemoteGetInFlight>> inflights;
  inflights.reserve(plan.remote_groups.size());
  for (const auto& [node_id, items] : plan.remote_groups) {
    if (auto f = SubmitRemoteBatchGet(items, results)) inflights.push_back(std::move(f));
  }
  if (in_flight_window) in_flight_window();
  for (auto& f : inflights) WaitRemoteBatchGet(*f, results, recache_remote);
}

void PoolClient::NoteRangedInstallFailure(size_t count) {
  if (count == 0) return;
  CountMetric(MORI_UMBP_METRIC_RANGED_REMOTE_INSTALL_FAILURES_TOTAL,
              MORI_UMBP_METRIC_RANGED_REMOTE_INSTALL_FAILURES_TOTAL_HELP, {},
              static_cast<double>(count));
}

std::unique_ptr<PoolClient::RemoteGetInFlight> PoolClient::SubmitRemoteBatchGet(
    const std::vector<BatchGetItem>& items, std::vector<bool>* results) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = false;
  };
  if (peer_directory_ == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchGet: no RDMA engine configured (items={})",
                    items.size());
    fail_all();
    return nullptr;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchGet: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
    fail_all();
    return nullptr;
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  auto inflight = std::make_unique<RemoteGetInFlight>();
  inflight->peer = &peer;

  // resolve RPC + per-key validation; failed keys already written to *results.
  if (!PrepareRemoteGetEntries(items, peer, stub, &inflight->entries, results)) {
    return nullptr;
  }

  std::vector<TransferItem> transfer_items;
  if (!BuildRemoteGetTransfers(inflight->entries, first.route.node_id, &transfer_items)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchGet: BuildRemoteGetTransfers failed, node='{}' entries={}",
        first.route.node_id, inflight->entries.size());
    for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
    return nullptr;
  }

  // Drop items whose entry failed during build (peer published no buffer etc.).
  std::vector<TransferItem> active;
  active.reserve(transfer_items.size());
  for (auto& item : transfer_items) {
    if (!inflight->entries[item.tag].failed) active.push_back(std::move(item));
  }
  if (active.empty()) {
    for (auto& entry : inflight->entries) {
      if (entry.failed) (*results)[entry.result_index] = false;
    }
    return nullptr;
  }

  TransferPlanSet planned = transfer_engine_->Plan(active);
  ApplyRejectedTags(inflight->entries, planned.rejected_tags, "RemoteGet");
  if (planned.plans.empty()) {
    for (auto& entry : inflight->entries) {
      if (entry.failed) (*results)[entry.result_index] = false;
    }
    return nullptr;
  }
  // POST; do NOT wait.  Everything the post references is owned by the returned
  // handle, including any bytes staged through the engine's bounce pool.
  inflight->handle = transfer_engine_->Submit(std::move(planned.plans));
  if (inflight->handle == nullptr) {
    for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
    return nullptr;
  }
  return inflight;
}

void PoolClient::WaitRemoteBatchGet(RemoteGetInFlight& f, std::vector<bool>* results,
                                    bool recache_remote) {
  if (f.drained) return;
  f.drained = true;
  std::vector<TransferFailure> failures;
  if (f.handle != nullptr) f.handle->Wait(&failures);
  ApplyTransferFailures(f.entries, failures, "RemoteGet");
  // Nothing to copy out here even for a staged read: the engine owns its bounce
  // pool and lands the bytes in the user's dst before its Wait returns.
  FinalizeRemoteGetEntries(f.entries, results, recache_remote);
}

bool PoolClient::PrepareRemoteGetEntries(const std::vector<BatchGetItem>& items,
                                         PeerConnection& peer, ::umbp::UMBPPeer::Stub* stub,
                                         std::vector<RemoteGetEntry>* entries,
                                         std::vector<bool>* results) {
  entries->clear();

  // Ask the peer to omit the buffer descriptors once we have already hydrated
  // them (from the GetPeerInfo handshake, or a prior resolve).  A wrong guess
  // is safe: a missing descriptor is caught by the transfer-build guard and the
  // entry degrades to a miss, never a corrupt read.
  const bool have_descs =
      peer_directory_ != nullptr && peer_directory_->HasRemoteBuffers(peer.node_id);

  ::umbp::BatchResolveKeysRequest resolve_req;
  for (const auto& item : items) resolve_req.add_keys(*item.key);
  resolve_req.set_omit_descs(have_descs);

  DecodedBatchResolve decoded;
  const auto retry_timeout = ResolveBusyRetryTimeout();
  const auto retry_deadline = std::chrono::steady_clock::now() + retry_timeout;
  std::chrono::milliseconds backoff{1};
  size_t busy_attempts = 0;
  while (true) {
    ::umbp::BatchResolveKeysResponse resolve_resp;
    grpc::ClientContext resolve_ctx;
    const auto remaining = retry_deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
      MORI_UMBP_WARN("[PoolClient] BatchResolveKeys BUSY timeout on {} after {} attempts",
                     items.front().route.node_id, busy_attempts);
      for (const auto& item : items) (*results)[item.index] = false;
      return false;
    }
    resolve_ctx.set_deadline(std::chrono::system_clock::now() + remaining);
    auto resolve_status = stub->BatchResolveKeys(&resolve_ctx, resolve_req, &resolve_resp);
    if (!resolve_status.ok() ||
        BatchResolveKeyCount(resolve_resp) != static_cast<int>(items.size())) {
      MORI_UMBP_WARN("[PoolClient] BatchResolveKeys failed on {}: {}", items.front().route.node_id,
                     resolve_status.error_message());
      for (const auto& item : items) (*results)[item.index] = false;
      return false;
    }

    decoded = DecodeBatchResolveResponse(resolve_resp);
    if (decoded.keys.size() != items.size()) {
      // Malformed (mismatched parallel arrays); fail the whole batch rather
      // than partially reading it.
      MORI_UMBP_WARN(
          "[PoolClient] BatchResolveKeys malformed response on {}: {} keys for {} items",
          items.front().route.node_id, decoded.keys.size(), items.size());
      for (const auto& item : items) (*results)[item.index] = false;
      return false;
    }

    const bool busy = std::any_of(decoded.keys.begin(), decoded.keys.end(), [](const auto& key) {
      return key.outcome == ResolveOutcome::kBusy;
    });
    if (!busy) break;

    // Discard the ENTIRE response.  Keeping its successful SSD entries while
    // retrying only BUSY keys would pin their leases and can make forward
    // progress impossible; it can also leave stale page locations if a retry
    // outlives the lease.
    ++busy_attempts;
    const auto sleep_for = std::min(
        backoff, std::chrono::duration_cast<std::chrono::milliseconds>(
                     retry_deadline - std::chrono::steady_clock::now()));
    if (sleep_for.count() > 0) std::this_thread::sleep_for(sleep_for);
    backoff = std::min(backoff * 2, std::chrono::milliseconds{50});
  }
  // Hydrate the batch-level descriptors once (skipped when the peer honored
  // omit_descs and sent none).
  if (!decoded.descs.empty()) peer_directory_->CacheRemoteBuffers(peer.node_id, decoded.descs);

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& key = decoded.keys[i];
    if (!key.found) {
      if (key.outcome == ResolveOutcome::kFailed) {
        MORI_UMBP_ERROR("[PoolClient] BatchGet: peer reported permanent resolve failure key='{}'",
                        *item.key);
      }
      (*results)[item.index] = false;
      continue;
    }
    if (key.size != item.size) {
      MORI_UMBP_WARN("[PoolClient] BatchGet: size mismatch for key='{}' (wanted {}, got {})",
                     *item.key, item.size, key.size);
      (*results)[item.index] = false;
      continue;
    }
    if (!SizeMatchesAllocation(item.size, key.pages.size(), decoded.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: malformed slot for key='{}'", *item.key);
      (*results)[item.index] = false;
      continue;
    }

    RemoteGetEntry entry;
    entry.result_index = item.index;
    entry.item = &item;
    entry.plan.page_size = decoded.page_size;
    // Per key, because a resolve batch may now be served from several media at
    // once; the pages below are indices into THIS backend's buffers.
    entry.plan.backend_id = key.backend_id;
    entry.plan.pages = std::move(decoded.keys[i].pages);
    // Descriptors were hydrated batch-level above; the per-entry plan carries
    // none (BuildRemoteGetTransfers' EnsureBufferDescsCached call is a no-op on
    // an empty list and the read path resolves descriptors by buffer_index).
    entries->push_back(std::move(entry));
  }

  return !entries->empty();
}

bool PoolClient::BuildRemoteGetTransfers(std::vector<RemoteGetEntry>& entries,
                                         const std::string& node_id,
                                         std::vector<TransferItem>* items) {
  items->clear();

  // Batch-level descs were already hydrated by PrepareRemoteGetEntries; any
  // per-entry ones are folded in here before the snapshots (see
  // BuildRemotePutTransfers for why one snapshot per backend beats a lock per
  // page — and why one snapshot per PEER would index the wrong medium now that
  // a resolve batch can span media).
  for (const auto& entry : entries) {
    if (!entry.plan.descs.empty()) peer_directory_->CacheRemoteBuffers(node_id, entry.plan.descs);
  }
  std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> snapshots;
  std::array<bool, kMaxBackendsPerPeer> snapped{};
  auto buffers_for = [&](uint32_t backend_id) -> const std::vector<TransferRef>& {
    static const std::vector<TransferRef> kNone;
    if (backend_id >= kMaxBackendsPerPeer) return kNone;
    if (!snapped[backend_id]) {
      snapshots[backend_id] = peer_directory_->RemoteBufferSnapshot(node_id, backend_id);
      snapped[backend_id] = true;
    }
    return snapshots[backend_id];
  };

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    // DstBytes(), not size: a ranged entry's destination is an arena slice
    // holding only the spans.  Passing the object size would make
    // FindRegisteredMemory's containment check reject a slice that sits near
    // the end of the arena, silently downgrading a zero-copy read to a staged
    // one instead of failing.
    TransferRef dst;
    uint64_t dst_base = 0;
    if (entry.item->dst_ref != nullptr) {
      dst = *entry.item->dst_ref;
      dst_base = entry.item->dst_ref_offset;
    } else {
      std::tie(dst, dst_base) = UserBufferRef(entry.item->dst, entry.item->DstBytes());
    }
    const std::vector<TransferRef>& remote = buffers_for(entry.plan.backend_id);

    // Shared by both shapes: a page the peer never published means this key
    // cannot be read at all, so the entry fails whole rather than partially.
    auto peer_buffer = [&](const PageLocation& page) -> const TransferRef* {
      if (page.buffer_index >= remote.size() || !remote[page.buffer_index].HasMemoryDesc()) {
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemoteGetTransfers: peer published no buffer, "
            "key='{}' backend={} buffer_index={} peer_buffers={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.plan.backend_id, page.buffer_index, remote.size(), page.page_index);
        return nullptr;
      }
      return &remote[page.buffer_index];
    };

    std::vector<TransferItem> entry_items;

    if (entry.item->Ranged()) {
      // Move only the requested spans, laid down back-to-back at the
      // destination in the caller's span order.  The spans are NOT assumed
      // ascending — a packed multi-component KV pool emits them interleaved —
      // so each is walked independently and any merging is left to the engine's
      // planner, which coalesces adjacent segments per MR pair anyway.
      //
      // One item per (span x page it touches), so the page count alone is not
      // the bound.
      entry_items.reserve(entry.item->spans->size() + entry.plan.pages.size());
      size_t packed_cursor = 0;
      for (const auto& span : *entry.item->spans) {
        const bool walked = ForEachRangePageFragment(
            entry.plan.pages, entry.plan.page_size, entry.item->size, span.object_offset, span.size,
            [&](size_t page_index, uint64_t tier_offset, size_t fragment, size_t copied) {
              const TransferRef* buf = peer_buffer(entry.plan.pages[page_index]);
              if (buf == nullptr) return false;
              TransferItem item;
              item.tag = idx;
              item.src = *buf;
              item.src_offset = tier_offset;
              item.dst = dst;
              item.dst_offset = dst_base + packed_cursor + copied;
              item.size = fragment;
              entry_items.push_back(std::move(item));
              return true;
            });
        if (!walked) {
          MORI_UMBP_ERROR(
              "[PoolClient] BuildRemoteGetTransfers: span [{},{}) outside object, key='{}' size={}",
              span.object_offset, span.object_offset + span.size,
              (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
              entry.item->size);
          entry.failed = true;
          entry_items.clear();
          break;
        }
        packed_cursor += span.size;
      }
    } else {
      entry_items.reserve(entry.plan.pages.size());
      for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
        const TransferRef* buf = peer_buffer(entry.plan.pages[p]);
        if (buf == nullptr) {
          entry.failed = true;
          entry_items.clear();
          break;
        }
        TransferItem item;
        item.tag = idx;
        item.src = *buf;
        item.src_offset =
            static_cast<uint64_t>(entry.plan.pages[p].page_index) * entry.plan.page_size;
        item.dst = dst;
        item.dst_offset = dst_base + static_cast<uint64_t>(p) * entry.plan.page_size;
        item.size =
            LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
        entry_items.push_back(std::move(item));
      }
    }

    if (!entry_items.empty()) {
      items->insert(items->end(), std::make_move_iterator(entry_items.begin()),
                    std::make_move_iterator(entry_items.end()));
    }
  }
  return true;
}

void PoolClient::FinalizeRemoteGetEntries(std::vector<RemoteGetEntry>& entries,
                                          std::vector<bool>* results, bool recache_remote) {
  for (auto& entry : entries) {
    if (entry.failed) {
      (*results)[entry.result_index] = false;
      continue;
    }
    // Bytes that actually crossed the wire, which for a ranged entry is the
    // span total and not the object size.  Reporting the object size here would
    // overstate remote GET bandwidth by the whole read-amplification factor the
    // ranged path exists to remove.
    const double moved = static_cast<double>(entry.item->DstBytes());
    CountMetric(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "remote"}},
                moved);
    CountMetric(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, {{"traffic", "remote"}},
                moved);
    (*results)[entry.result_index] = true;

    // Re-cache the remotely-fetched block into local DRAM (best-effort): the
    // user dst is already populated (staging copy-out and zero-copy both land
    // before Finalize runs), so subsequent reads of this key route local.
    //
    // Never for a ranged entry: the destination holds only the requested spans,
    // and a medium slot is whole-object, so installing it would publish a key
    // whose remaining bytes are undefined.  The ranged path schedules its own
    // whole-object prefetch instead (MaybePrefetchWholeObject).
    if (recache_remote && entry.item && !entry.item->Ranged()) {
      MaybeReCacheAfterRemote(*entry.item->key, entry.item->dst, entry.item->size);
    }
  }
}

// ---------------------------------------------------------------------------
//  Cluster-wide existence check
// ---------------------------------------------------------------------------

bool PoolClient::Exists(const std::string& key) {
  auto v = BatchExists({key});
  return !v.empty() && v.front();
}

std::vector<bool> PoolClient::BatchExists(const std::vector<std::string>& keys) {
  if (!initialized_ || keys.empty()) return std::vector<bool>(keys.size(), false);

  // Local-first: this node's own backends answer conclusively for the keys they
  // hold, so a batch that is entirely local needs no master round trip at all.
  // The misses are NOT conclusive -- a peer may hold them -- so they still go to
  // the master, and only they do.
  std::vector<bool> out(keys.size(), false);
  std::vector<std::string> unknown;
  std::vector<size_t> unknown_indices;
  if (config_.local_first) {
    std::vector<size_t> all(keys.size());
    std::iota(all.begin(), all.end(), 0);
    std::vector<MediumBackend*> holders;
    ResolveLocalBatch(keys, all, &holders, /*resolutions=*/nullptr);
    for (size_t i = 0; i < keys.size(); ++i) out[i] = holders[i] != nullptr;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!out[i]) {
        unknown.push_back(keys[i]);
        unknown_indices.push_back(i);
      }
    }
    if (unknown.empty()) return out;
  } else {
    unknown = keys;
    unknown_indices.resize(keys.size());
    std::iota(unknown_indices.begin(), unknown_indices.end(), 0);
  }

  // With no master this node is the whole cluster, so what it does not hold
  // does not exist.  `out` already carries that answer.
  if (!HasMaster()) return out;

  std::vector<bool> remote;
  auto status = master_client_->BatchLookup(unknown, &remote);
  // A lookup failure must not downgrade a local hit to a miss: the local half of
  // the answer did not depend on the RPC.  Only the keys we could not answer
  // ourselves fall back to false.
  if (!status.ok() || remote.size() != unknown.size()) return out;
  for (size_t j = 0; j < unknown_indices.size(); ++j) {
    if (remote[j]) out[unknown_indices[j]] = true;
  }
  return out;
}

// ---------------------------------------------------------------------------
//  External KV
// ---------------------------------------------------------------------------

bool PoolClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (hashes.empty()) return true;
  // External KV placement lives only in the master's index; with no master
  // there is no such index and nothing to report to.
  if (!HasMaster()) return true;
  return master_client_->ReportExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (hashes.empty()) return true;
  // External KV placement lives only in the master's index; with no master
  // there is no such index and nothing to report to.
  if (!HasMaster()) return true;
  return master_client_->RevokeExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  if (!initialized_) return false;
  // External KV placement lives only in the master's index; with no master
  // there is no such index and nothing to report to.
  if (!HasMaster()) return true;
  return master_client_->RevokeAllExternalKvBlocksAtTier(config_.master_config.node_id, tier).ok();
}

bool PoolClient::MatchExternalKv(const std::vector<std::string>& hashes,
                                 std::vector<MasterClient::ExternalKvNodeMatch>* out_matches,
                                 bool count_as_hit) {
  if (!initialized_) return false;
  // External KV placement lives only in the master's index; with no master
  // there is no such index and nothing to report to.
  if (!HasMaster()) return true;
  return master_client_->MatchExternalKv(hashes, out_matches, count_as_hit).ok();
}

bool PoolClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes,
    std::vector<MasterClient::ExternalKvHitCountEntry>* out_entries) {
  if (!initialized_) return false;
  // External KV placement lives only in the master's index; with no master
  // there is no such index and nothing to report to.
  if (!HasMaster()) return true;
  return master_client_->GetExternalKvHitCounts(hashes, out_entries).ok();
}

// ---------------------------------------------------------------------------
//  Peer connection cache
// ---------------------------------------------------------------------------

PoolClient::PeerConnection& PoolClient::GetOrConnectPeer(const std::string& node_id,
                                                         const std::string& peer_address) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(node_id);
  if (it != peers_.end()) return *it->second;

  auto conn = std::make_unique<PeerConnection>();
  conn->node_id = node_id;
  conn->peer_address = peer_address;
  // The peer's engine desc and buffer descriptors are hydrated lazily in
  // EnsurePeerServiceConnection, into the transfer engine rather than here.
  auto& ref = *conn;
  peers_[node_id] = std::move(conn);
  return ref;
}

// ---------------------------------------------------------------------------
//  Peer connection setup
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  std::lock_guard<std::mutex> lock(peer.conn_mutex);
  if (peer.peer_address.empty()) {
    return false;
  }

  // GetPeerInfo returns two different kinds of fact and they now go to two
  // different owners: the stub is a control-plane connection kept here, while
  // the peer's engine desc and buffer descriptors are transfer-layer facts and
  // go straight into the engine's remote cache.
  auto hydrate_from_peer = [&](::umbp::UMBPPeer::Stub* stub) -> bool {
    ::umbp::GetPeerInfoRequest req;
    ::umbp::GetPeerInfoResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->GetPeerInfo(&ctx, req, &resp);
    if (!status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] GetPeerInfo failed for '{}': {}", peer.peer_address,
                      status.error_message());
      return false;
    }

    if (peer_directory_ != nullptr) {
      if (!peer_directory_->EnsureRemoteEngine(peer.node_id, resp.engine_desc())) return false;

      // Every backend's buffers arrive in one list, each entry naming the
      // backend its buffer_index belongs to.  Before backend_id, this list held
      // one medium's buffers and the rest were unreachable — yet HasRemoteBuffers
      // still went true, so the next resolve asked the peer to omit descriptors
      // and the missing media's pages were read against the published one's
      // memory.
      std::vector<BufferMemoryDescBytes> descs;
      descs.reserve(resp.buffer_descs_size());
      for (const auto& d : resp.buffer_descs()) {
        if (d.desc().empty()) continue;
        BufferMemoryDescBytes b;
        b.buffer_index = d.buffer_index();
        b.backend_id = d.backend_id();
        b.desc_bytes.assign(d.desc().begin(), d.desc().end());
        descs.push_back(std::move(b));
      }
      peer_directory_->CacheRemoteBuffers(peer.node_id, descs);
    }
    return true;
  };

  const bool engine_known =
      peer_directory_ == nullptr || peer_directory_->HasRemoteEngine(peer.node_id);

  if (peer.peer_stub) {
    if (!engine_known) {
      auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
      if (!hydrate_from_peer(stub)) {
        peer.peer_stub.reset();
        if (peer_directory_ != nullptr) peer_directory_->ForgetRemote(peer.node_id);
        return false;
      }
    }
    return true;
  }

  auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPPeer::NewStub(channel);
  if (!hydrate_from_peer(stub.get())) {
    return false;
  }

  peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
      stub.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  return true;
}

void PoolClient::PublishComponentMetrics() {
  if (!master_client_) return;

  MetricPublisher::Sink sink{
      [this](const char* name, const char* help, const MetricLabels& labels, double delta) {
        CountMetric(name, help, labels, delta);
      },
      [this](const char* name, const char* help, const MetricLabels& labels, double value) {
        master_client_->SetGauge(name, help, labels, value);
      }};

  // Storage backends.  tier= and backend= come from the component's own
  // identity, so this loop never names a medium and a fourth one needs no edit
  // here.  Both halves of a backend's metrics — the generic series
  // InstrumentedBackend measured and whatever the medium publishes about its
  // own internals — arrive through the one SampleMetrics() call.
  for (MediumBackend* backend : registry_.All()) {
    if (backend == nullptr) continue;
    const auto* entry = registry_.GetEntry(backend);
    const std::string backend_name =
        entry == nullptr ? std::string(backend->Name()) : entry->name;
    const MetricLabels labels = {{"tier", TierTypeName(backend->Tier())},
                                 {"backend", backend_name}};
    metric_publisher_.Publish(std::string("backend:") + backend_name, labels, *backend, sink);
  }

  // The transfer layer.  Its samples already carry engine=, stamped by the
  // composite that dispatched them, so there is nothing to add at this level.
  if (transfer_engine_ != nullptr) {
    metric_publisher_.Publish("transfer", {}, *transfer_engine_, sink);
  }

  // Logical tier transitions.  PeerPool accumulates these and nothing outside
  // the tier benchmark reads them, so without this a served workload cannot
  // distinguish a tier graph that is migrating from one whose every migration
  // fails.  Read hits carry the logical tier as a label because the per-backend
  // labels above collapse every instance of a medium into one series, which
  // hides which tier actually served a read.
  if (default_pool_ != nullptr) {
    std::vector<MetricSample> samples;
    const auto sample = [&samples](const char* name, const char* help,
                                   MetricLabels labels, uint64_t value) {
      samples.push_back(MetricSample{name, help, std::move(labels), value});
    };

    const TierTransitionMetrics tiers = default_pool_->TransitionMetrics();
    sample(MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS,
           MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS_HELP,
           {{"outcome", "attempted"}}, tiers.attempted);
    sample(MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS,
           MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS_HELP,
           {{"outcome", "succeeded"}}, tiers.succeeded);
    sample(MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS,
           MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS_HELP,
           {{"outcome", "failed"}}, tiers.failed);
    sample(MORI_UMBP_METRIC_CLIENT_TIER_OFFLOADED_BYTES,
           MORI_UMBP_METRIC_CLIENT_TIER_OFFLOADED_BYTES_HELP, {}, tiers.offloaded_bytes);
    sample(MORI_UMBP_METRIC_CLIENT_TIER_PROMOTED_BYTES,
           MORI_UMBP_METRIC_CLIENT_TIER_PROMOTED_BYTES_HELP, {}, tiers.promoted_bytes);

    for (const auto& [tier, hits] : default_pool_->TierReadHits()) {
      sample(MORI_UMBP_METRIC_CLIENT_TIER_READ_HITS,
             MORI_UMBP_METRIC_CLIENT_TIER_READ_HITS_HELP, {{"tier", tier}}, hits);
    }
    metric_publisher_.Publish("pool", {}, samples, sink);
  }
}

}  // namespace mori::umbp
