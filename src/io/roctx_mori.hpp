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
// ============================================================================
// ADDITIVE, env-gated roctx markers for the MORI-IO host RDMA I/O path.
//
// TWO independent, additive instrumentations (each its own env gate; default OFF):
//
//   (1) MORI_ROCTX=1  -> SYNCHRONOUS push/pop ranges around the host ibv_post_send
//       loop (IOEngine[Session]::BatchWrite + RdmaBatchReadWrite). These measure
//       only the HOST POST cost (building WRs + ringing the NIC doorbell). They
//       are stack/same-thread ranges (MoriRoctxRange RAII) and CANNOT span the
//       asynchronous post-to-completion interval. Marker names:
//       mori.io.engine_batch_write, mori.rdma.batch_post.{write,read}.
//
//   (2) MORI_ROCTX_TRANSFER=1  -> ASYNCHRONOUS post-to-completion ranges. A range
//       starts immediately before ibv_post_send for a *signaled* WR
//       (RdmaBatchReadWrite, needSignal branch) and stops after its completion is
//       reaped and its ledger record is processed
//       (NotifManager::ProcessOneCqe -> ledger->ReleaseByCqe). This is
//       post-to-completion latency, not wire-only time: it includes submission and
//       ibv_post_send, NIC/SQ queueing, transfer, CQ availability/polling, and
//       software processing through marker stop.
//       Uses the PROCESS-WIDE async roctx API roctxRangeStartA/roctxRangeStop
//       (start on the posting thread, stop on the CQ-poll thread). Marker name:
//       mori.rdma.io_transfer (reads use mori.rdma.io_transfer.read).
//
//       RDMA uses SELECTIVE SIGNALING: only the tail WR of each post batch sets
//       IBV_SEND_SIGNALED and receives a SubmissionLedger recordId (== wr_id).
//       Only that signaled WR produces a CQE, so we start exactly ONE async range
//       per signaled WR (keyed by the ledger recordId) -> every started range has
//       a matching stop (the CQE, or the not-posted cleanup path). recordId is
//       per-EP-ledger (not globally unique), so the range map is keyed by the
//       PAIR (SubmissionLedger*, recordId), which is globally unique and identical
//       at the post site (eps[i].ledger) and the CQ site (ep.ledger) because both
//       hold the same shared SubmissionLedger instance.
//
// CRITICAL: rocprofv3 (rocprofiler-sdk) --marker-trace only intercepts the
// rocprofiler-sdk ROCTx library librocprofiler-sdk-roctx.so, NOT legacy
// libroctx64.so. We dlopen the sdk lib at runtime (RTLD_GLOBAL) and resolve the
// roctx symbols from it (no link-time dependency added to libmori_io.so).
//
// Fully gated + exception-safe: when neither gate is set the lib is never dlopen'd
// and every call is a no-op (a single bool check).
// ============================================================================
#pragma once

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace mori {
namespace io {
namespace roctx_detail {

using roctx_range_push_t = int (*)(const char*);
using roctx_range_pop_t = int (*)();
using roctx_mark_t = void (*)(const char*);
// Process-wide async range API (start on one thread, stop from any other).
using roctx_range_id_t = std::uint64_t;
using roctx_range_start_t = roctx_range_id_t (*)(const char*);
using roctx_range_stop_t = void (*)(roctx_range_id_t);

inline bool GateOn(const char* name) {
  const char* g = std::getenv(name);
  if (g == nullptr) return false;
  const char c = g[0];
  return (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y' || c == 'o' || c == 'O');
}

struct RoctxApi {
  bool enabled = false;           // MORI_ROCTX: push/pop host-post anchors
  bool transfer_enabled = false;  // MORI_ROCTX_TRANSFER: post-to-completion ranges
  roctx_range_push_t push = nullptr;
  roctx_range_pop_t pop = nullptr;
  roctx_mark_t mark = nullptr;
  roctx_range_start_t range_start = nullptr;
  roctx_range_stop_t range_stop = nullptr;

  RoctxApi() {
    const bool want_post = GateOn("MORI_ROCTX");
    const bool want_transfer = GateOn("MORI_ROCTX_TRANSFER");
    if (!want_post && !want_transfer) return;
    // sdk-roctx ONLY (the lib rocprofv3 --marker-trace intercepts).
    void* h = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) h = dlopen("librocprofiler-sdk-roctx.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) return;
    push = reinterpret_cast<roctx_range_push_t>(dlsym(h, "roctxRangePushA"));
    pop = reinterpret_cast<roctx_range_pop_t>(dlsym(h, "roctxRangePop"));
    mark = reinterpret_cast<roctx_mark_t>(dlsym(h, "roctxMarkA"));
    range_start = reinterpret_cast<roctx_range_start_t>(dlsym(h, "roctxRangeStartA"));
    range_stop = reinterpret_cast<roctx_range_stop_t>(dlsym(h, "roctxRangeStop"));
    enabled = want_post && (push != nullptr && pop != nullptr);
    transfer_enabled = want_transfer && (range_start != nullptr && range_stop != nullptr);
  }
};

inline RoctxApi& api() {
  static RoctxApi a;  // gate read + dlopen happen exactly once per process
  return a;
}

// (SubmissionLedger*, recordId) -> async roctx range id. recordId is unique only
// within one ledger, so the ledger pointer disambiguates across endpoints.
using TransferKey = std::pair<std::uintptr_t, std::uint64_t>;
struct TransferKeyHash {
  std::size_t operator()(const TransferKey& k) const {
    std::size_t h1 = std::hash<std::uintptr_t>{}(k.first);
    std::size_t h2 = std::hash<std::uint64_t>{}(k.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
struct TransferRanges {
  std::mutex mu;
  std::unordered_map<TransferKey, roctx_range_id_t, TransferKeyHash> ranges;
};
inline TransferRanges& transfer_ranges() {
  static TransferRanges t;
  return t;
}

}  // namespace roctx_detail

// Cheap, once-initialized gates for hot-path callers. When both environment
// gates are off, api() reads them once and does not attempt to load ROCTx.
inline bool MoriRoctxPostEnabled() noexcept { return roctx_detail::api().enabled; }
inline bool MoriRoctxTransferEnabled() noexcept { return roctx_detail::api().transfer_enabled; }

// RAII range: pushes on construction, pops on destruction (handles every return
// path + exception). No-op when MORI_ROCTX is off. (HOST-POST anchor only.)
class MoriRoctxRange {
 public:
  explicit MoriRoctxRange(const char* name) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      a.push(name);
      active_ = true;
    }
  }
  MoriRoctxRange(const char* name, uint64_t id) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      std::string s = std::string(name) + " id=" + std::to_string(id);
      a.push(s.c_str());
      active_ = true;
    }
  }
  // ADDITIVE: host-post anchor variant carrying the whole-call payload size.
  // Keeps id= LAST so end-anchored id= parsers stay valid: "<name> bytes=<N> id=<id>".
  MoriRoctxRange(const char* name, uint64_t id, uint64_t bytes) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      std::string s =
          std::string(name) + " bytes=" + std::to_string(bytes) + " id=" + std::to_string(id);
      a.push(s.c_str());
      active_ = true;
    }
  }
  ~MoriRoctxRange() {
    if (active_) {
      auto& a = roctx_detail::api();
      if (a.pop != nullptr) a.pop();
    }
  }
  MoriRoctxRange(const MoriRoctxRange&) = delete;
  MoriRoctxRange& operator=(const MoriRoctxRange&) = delete;

 private:
  bool active_ = false;
};

inline void MoriRoctxMark(const std::string& msg) {
  auto& a = roctx_detail::api();
  if (a.enabled && a.mark != nullptr) a.mark(msg.c_str());
}

// --- ASYNC I/O post-to-completion ranges (MORI_ROCTX_TRANSFER) ---------------
// Start an async range for a SIGNALED WR immediately before ibv_post_send.
// Keyed by (ledger,recordId).
inline void MoriRoctxTransferStart(const void* ledger, std::uint64_t recordId,
                                   std::uint64_t transferId, bool isRead, std::uint64_t bytes = 0) {
  auto& a = roctx_detail::api();
  if (!a.transfer_enabled || a.range_start == nullptr || ledger == nullptr) return;
  // bytes= placed BEFORE id= so the end-anchored id= parsers keep matching.
  std::string s = std::string(isRead ? "mori.rdma.io_transfer.read" : "mori.rdma.io_transfer") +
                  " bytes=" + std::to_string(bytes) + " id=" + std::to_string(transferId);
  roctx_detail::roctx_range_id_t rid = a.range_start(s.c_str());
  auto& t = roctx_detail::transfer_ranges();
  std::lock_guard<std::mutex> lk(t.mu);
  t.ranges[{reinterpret_cast<std::uintptr_t>(ledger), recordId}] = rid;
}

// Stop the async range for a completed/cleaned-up signaled WR. Idempotent: a
// no-op if no range was started for this (ledger,recordId) (e.g. unsignaled WRs,
// notification CQEs). The roctxRangeStop call is made OUTSIDE the map lock.
inline void MoriRoctxTransferStop(const void* ledger, std::uint64_t recordId) {
  auto& a = roctx_detail::api();
  if (!a.transfer_enabled || a.range_stop == nullptr || ledger == nullptr) return;
  roctx_detail::roctx_range_id_t rid = 0;
  bool found = false;
  {
    auto& t = roctx_detail::transfer_ranges();
    std::lock_guard<std::mutex> lk(t.mu);
    auto it = t.ranges.find({reinterpret_cast<std::uintptr_t>(ledger), recordId});
    if (it != t.ranges.end()) {
      rid = it->second;
      t.ranges.erase(it);
      found = true;
    }
  }
  if (found) a.range_stop(rid);
}

// Diagnostics: number of started-but-not-stopped transfer ranges (leak counter).
inline std::size_t MoriRoctxTransferOutstanding() {
  auto& a = roctx_detail::api();
  if (!a.transfer_enabled) return 0;
  auto& t = roctx_detail::transfer_ranges();
  std::lock_guard<std::mutex> lk(t.mu);
  return t.ranges.size();
}

}  // namespace io
}  // namespace mori
