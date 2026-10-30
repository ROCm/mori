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

// Event-path types: what subsystem threads hand to the RAS thread, without blocking.
//
// RasEvent is a closed std::variant, not an open tagged struct (subsystem, code, payload).
// Every subsystem that emits events is built in this repo, so a new event type is a new
// alternative here plus a rebuild of mori_ras, and std::visit then fails to compile in any
// dispatcher that doesn't handle it. An open struct would skip that rebuild, but the RAS
// thread would decode untyped payloads and a forgotten event would be dropped silently.
//
// Events copy their data. No pointer into subsystem memory crosses the queue, so a subsystem
// object can go away while its events are still queued.

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "mori/ras/types.hpp"

namespace mori {
namespace ras {

// A runtime's peer listener, as its EngineDesc advertises it.
struct RasEndpoint {
  std::string host;
  uint16_t port{0};
  std::string token;
  RuntimeUuid uuid;

  // An EngineDesc from an engine without RAS, or whose RAS failed to start, has none.
  bool Valid() const { return !host.empty() && port != 0 && !token.empty() && !uuid.IsZero(); }
};

// A local IOEngine was created.
struct EngineRegistered {
  std::string engineKey;
};

// A local IOEngine was destroyed. Its peer relationships end with it.
struct EngineUnregistered {
  std::string engineKey;
};

// A local engine registered a remote engine (IOEngine::RegisterRemoteEngine).
struct PeerAdded {
  std::string localEngine;
  std::string remoteEngine;
  std::string remoteNodeId;
  std::string remoteHostname;
  int32_t remotePid{0};
  RasEndpoint endpoint;  // invalid when the remote engine advertises no RAS endpoint
};

// A local engine deregistered a remote engine (IOEngine::DeregisterRemoteEngine).
struct PeerRemoved {
  std::string localEngine;
  std::string remoteEngine;
};

// A local engine uses a GPU: it registered memory there (no NICs), or an RDMA session from that
// memory picked NICs it hadn't reported yet. RAS merges these into one attachment per engine
// and GPU, so a repeat is harmless.
struct EngineGpuAttached {
  std::string engineKey;
  int32_t gpu{-1};
  std::string pciBusId;
  std::vector<NicAttachment> nics;  // device and port only; RAS fills in the rest
};

using RasEvent =
    std::variant<EngineRegistered, EngineUnregistered, PeerAdded, PeerRemoved, EngineGpuAttached>;
constexpr size_t kNumEventKinds = std::variant_size_v<RasEvent>;

// "EngineRegistered" and so on, for an index of RasEvent.
inline const char* EventKindName(size_t kind) {
  static constexpr const char* kNames[] = {"EngineRegistered", "EngineUnregistered", "PeerAdded",
                                           "PeerRemoved", "EngineGpuAttached"};
  static_assert(sizeof(kNames) / sizeof(kNames[0]) == kNumEventKinds, "one name per event");
  return kind < kNumEventKinds ? kNames[kind] : "Unknown";
}

}  // namespace ras
}  // namespace mori
