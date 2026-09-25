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

// Where every per-RPC gRPC client deadline in UMBP is resolved and armed.
//
// A gRPC call with no deadline waits forever. UMBP learned that the expensive
// way: a wedged standalone-server handler blocked a scheduler rank, the rank
// never reached its next collective, and all 8 TP ranks hung with it.
//
// Arming them per call site is how UMBP ended up with deadlines on every master
// RPC that is control plane -- unregister, heartbeat, metrics -- and on none of
// the master RPCs that are data plane, nor on any peer RPC at all. So a server
// handling a request could block indefinitely on its own outbound BatchLookup
// or BatchAllocateSlots, and the only thing bounding it was the caller's
// deadline one layer up: the client covering for a server that could not free
// itself. That is the shape this header exists to prevent.
//
// THE HIERARCHY. An outbound deadline must stay below the deadline of whatever
// call it serves, so the inner call fails first and returns a real error rather
// than letting the outer one time out blind:
//
//     master/peer outbound  <  standalone client data plane
//
// There is also a hard ceiling on the peer side. A slot is reaped by the peer
// pending_ttl (30s) if it is not committed, so allocate must leave room for the
// transfer and the commit that follow it inside that window.
//
// SCOPE. "0 disables" applies to per-RPC gRPC client deadlines and nothing
// else. It does NOT apply to UMBP_RESOLVE_BUSY_TIMEOUT_MS (a retry budget --
// zero would break the retry loop, not just remove a bound), to
// UMBP_STANDALONE_STARTUP_TIMEOUT_MS (a poll-loop bound), or to the
// UMBP_*_GRPC_SHUTDOWN_DEADLINE_SEC family (server-side Shutdown(), where zero
// means "cancel immediately" rather than "wait forever").

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include "umbp/common/env_time.h"

namespace mori::umbp {

// The master switch. UMBP_RPC_DEADLINES=0 turns off every deadline this header
// resolves, for a deployment that would rather have a call hang -- where it can
// be seen with py-spy -- than have it time out and be reported as a miss.
//
// A knob set explicitly still wins, so "off except this one" is expressible.
//
// Uncached, like everything else here -- see ResolveDeadlineMs.
inline bool RpcDeadlinesEnabled() {
  const char* raw = std::getenv("UMBP_RPC_DEADLINES");
  if (raw == nullptr || raw[0] == '\0') return true;
  const std::string value(raw);
  return value != "0" && value != "off" && value != "false";
}

// Resolves one deadline knob, in milliseconds, under the master switch:
//
//   set explicitly          -> that value, whatever the switch says
//   unset, switch on        -> `def`
//   unset, switch off       -> 0, i.e. no deadline
//
// The explicit-set test has to be made here rather than left to
// GetEnvMilliseconds, which cannot tell "unset" from "set to the default".
// Parsing and validation still go through it, so a malformed or out-of-range
// value warns once and falls back exactly as everywhere else.
//
// Not cached, so that the whole header stays testable in one process: callers
// on a hot path hold the result in a function-local `static const`, which is
// the same rule env_time.h states for its own readers, and which is why the
// handful of getenv calls behind this run once per knob per process.
inline int ResolveDeadlineMs(const char* name, std::chrono::milliseconds def,
                             int64_t min_allowed = 0) {
  const char* raw = std::getenv(name);
  const bool set_explicitly = raw != nullptr && raw[0] != '\0';
  if (!set_explicitly && !RpcDeadlinesEnabled()) return 0;
  return static_cast<int>(GetEnvMilliseconds(name, def, min_allowed).count());
}

// Applies a resolved deadline. A non-positive value leaves the context alone,
// which is gRPC's "wait forever" -- so every disabled path above lands here and
// no call site needs its own branch.
inline void ArmDeadline(grpc::ClientContext& ctx, int timeout_ms) {
  if (timeout_ms <= 0) return;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
}

// Routing and lookup against the master. One round trip, issued before any slot
// exists, so nothing here can leave half-committed state behind.
inline int MasterRpcTimeoutMs() {
  static const int v =
      ResolveDeadlineMs("UMBP_MASTER_RPC_TIMEOUT_MS", std::chrono::milliseconds(30000));
  return v;
}

// Slot allocate/commit and the peer handshake.
//
// Deliberately well under the peer's 30s pending_ttl: a slot is allocated
// before this call returns, and the transfer and commit that follow it have to
// fit in what remains of that TTL. Giving this a larger budget would let a slow
// -- but successful -- allocate come back to slots the peer had already
// reclaimed, turning a working write into SLOT_GONE at commit.
inline int PeerRpcTimeoutMs() {
  static const int v =
      ResolveDeadlineMs("UMBP_PEER_RPC_TIMEOUT_MS", std::chrono::milliseconds(10000));
  return v;
}

// Abort and other peer cleanup, which must not be abandoned for being merely
// slow -- but must not pin a failure path open forever either.
//
// Derived from the peer budget rather than given a knob of its own, so the two
// cannot drift apart. Still inside pending_ttl: an abort slower than this is
// moot, because the peer's reaper is the backstop either way and is about to
// run.
inline int PeerCleanupRpcTimeoutMs() { return 2 * PeerRpcTimeoutMs(); }

}  // namespace mori::umbp
