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
#pragma once

// MORI RAS: a per-process runtime that watches the health of MORI's peers over its own
// out-of-band TCP links. Subsystems depend on this header, never the reverse.
//
// The API is built in when MORI_ENABLE_RAS is defined, which every target that links mori_ras
// inherits. Otherwise (BUILD_RAS=OFF) every call compiles to a no-op: Get() and EnsureStarted()
// return nullptr and Status() reports kBuiltOut. Call sites put everything behind the null
// check, so a built-out or disabled runtime costs one load and a branch:
//
//   if (ras::RasRuntime* rt = ras::RasRuntime::Get()) rt->Enqueue(ras::PeerAdded{...});
//
// Lifecycle:
// - Off unless MORI_RAS_ENABLE=1.
// - EnsureStarted() starts the runtime once per process, from the first IOEngine. It binds the
//   peer listener before returning, so the engine can advertise it in EngineDesc. The RAS
//   thread runs at nice 19 and on MORI_RAS_CPU_AFFINITY's CPUs, if set.
// - A runtime that fails to start, e.g. because MORI_RAS_PORT is taken, stays off for the
//   life of the process: Status() reports kFailed with kDegradedInitFailed and why, and engines
//   advertise no endpoint. A failure after start sets a degraded bit but keeps the listener
//   open, so peers learn of it from HELLO and heartbeats.
// - At exit, an atexit handler stops the RAS thread, which sends LEAVE on every link first.
//   That path doesn't log: MORI_LOG caches loggers in function-local statics, which may
//   already be destroyed. The runtime object is never freed, so a thread still calling in
//   during exit reaches a stopped runtime, not freed memory.
// - In a forked child, a pthread_atfork handler closes every RAS fd the child inherited, so a
//   killed parent's listener and links really close, and leaves the child with no runtime.
//   The child's first EnsureStarted() starts a fresh one under a new runtime UUID.
//
// Thread safety: every member may be called from any thread. Enqueue() never blocks.

#include <cstdint>
#include <string>

#include "mori/ras/events.hpp"
#include "mori/ras/types.hpp"

namespace mori {
namespace ras {

// What the first engine knows about this process.
struct StartParams {
  std::string nodeId;
  std::string hostname;
  std::string host;  // the RAS listener binds here unless MORI_RAS_IFNAME names an interface
  int32_t pid{0};
};

enum class RunState : uint8_t {
  kBuiltOut,    // BUILD_RAS=OFF
  kDisabled,    // MORI_RAS_ENABLE isn't set
  kNotStarted,  // enabled, no engine has started it yet
  kRunning,
  kFailed,   // couldn't start; see RuntimeStatus::detail
  kStopped,  // stopped at exit
};

struct RuntimeStatus {
  RunState state{RunState::kNotStarted};
  uint32_t degraded{0};  // DegradedReason bits
  std::string detail;
};

#if defined(MORI_ENABLE_RAS)

class RasRuntime {
 public:
  // The running runtime, or nullptr if it is built out, disabled, not started, or failed.
  static RasRuntime* Get();
  // Starts the runtime on the first call, then returns it. nullptr if RAS is off or failed.
  static RasRuntime* EnsureStarted(const StartParams& params);
  static RuntimeStatus Status();

  // The peer listener every local engine advertises in its EngineDesc.
  RasEndpoint Endpoint() const;
  // Hands an event to the RAS thread. Never blocks; false if the queue was full and dropped it.
  bool Enqueue(RasEvent ev);
  // The RAS thread's latest published view, at most about 50 ms old.
  RuntimeSnapshot Snapshot() const;
  // Process labels for later HELLOs. False, and nothing changes, if peers would reject them.
  bool SetLabels(const Labels& labels);
};

#else  // RAS built out

class RasRuntime {
 public:
  static RasRuntime* Get() { return nullptr; }
  static RasRuntime* EnsureStarted(const StartParams&) { return nullptr; }
  static RuntimeStatus Status() { return RuntimeStatus{RunState::kBuiltOut, 0, "built out"}; }
  RasEndpoint Endpoint() const { return RasEndpoint{}; }
  bool Enqueue(RasEvent) { return false; }
  RuntimeSnapshot Snapshot() const { return RuntimeSnapshot{}; }
  bool SetLabels(const Labels&) { return false; }
};

#endif

inline const char* ToString(RunState s) {
  switch (s) {
    case RunState::kBuiltOut:
      return "built out";
    case RunState::kDisabled:
      return "disabled";
    case RunState::kNotStarted:
      return "not started";
    case RunState::kRunning:
      return "running";
    case RunState::kFailed:
      return "failed";
    case RunState::kStopped:
      return "stopped";
  }
  return "unknown";
}

}  // namespace ras
}  // namespace mori
