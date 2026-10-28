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

// Identity and state types shared by the RAS runtime and the subsystems that feed it. Plain
// data only: the wire mapping lives in src/ras/protocol.cpp, so this header stays usable when
// RAS is built out.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace mori {
namespace ras {

// 128-bit runtime UUID, random (version 4), one per RAS runtime. A process that forks or
// restarts gets a new one, so a UUID names one process lifetime. Wire headers carry `lo`.
struct RuntimeUuid {
  uint64_t hi{0};
  uint64_t lo{0};

  bool IsZero() const { return hi == 0 && lo == 0; }
  bool operator==(const RuntimeUuid& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const RuntimeUuid& o) const { return !(*this == o); }
  bool operator<(const RuntimeUuid& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
  std::string ToString() const {
    char buf[37];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx", static_cast<unsigned>(hi >> 32),
                  static_cast<unsigned>((hi >> 16) & 0xffff), static_cast<unsigned>(hi & 0xffff),
                  static_cast<unsigned>(lo >> 48),
                  static_cast<unsigned long long>(lo & 0xffffffffffffull));
    return buf;
  }
};

// One RDMA port this host offers, with where it attaches in the fabric when that is known.
struct NicAttachment {
  std::string device;      // RDMA device, e.g. "ionic_3"
  uint32_t port{0};        // RDMA port number
  std::string netdev;      // netdev behind the RoCE GID, e.g. "enp9s0"
  std::string address;     // RoCE GID address: IPv4 for IPv4-mapped GIDs, else IPv6
  std::string rail;        // the address's subnet, e.g. "10.3.0.0/24", or a topology-file name
  std::string switchName;  // attached switch, from LLDP data
  std::string switchPort;  // its port, from LLDP data

  bool operator==(const NicAttachment& o) const {
    return device == o.device && port == o.port && netdev == o.netdev && address == o.address &&
           rail == o.rail && switchName == o.switchName && switchPort == o.switchPort;
  }
};

struct Topology {
  std::string rack;    // from the scheduler or a topology file
  std::string leaf;    // leaf (top-of-rack) switch, same sources
  std::string source;  // comma-separated sources the values came from, e.g. "sysfs,file"
  std::vector<NicAttachment> nics;

  bool operator==(const Topology& o) const {
    return rack == o.rack && leaf == o.leaf && source == o.source && nics == o.nics;
  }
};

using Labels = std::map<std::string, std::string>;

// Who a RAS runtime is. nodeId, hostname, host, and pid come from the EngineDesc of the engine
// that started RAS; the PID namespace and start time let a peer on the same node confirm an
// exit (P3) without trusting a reused PID.
struct RasProcessId {
  RuntimeUuid uuid;
  std::string nodeId;
  std::string hostname;
  std::string host;
  int32_t pid{0};
  uint64_t pidNamespace{0};    // inode of /proc/self/ns/pid; 0 if unknown
  uint64_t startTimeTicks{0};  // field 22 of /proc/self/stat, in clock ticks since boot
  std::string bootId;          // /proc/sys/kernel/random/boot_id; scopes startTimeTicks
  Topology topology;
  Labels labels;  // process labels set by the framework

  bool operator==(const RasProcessId& o) const {
    return uuid == o.uuid && nodeId == o.nodeId && hostname == o.hostname && host == o.host &&
           pid == o.pid && pidNamespace == o.pidNamespace && startTimeTicks == o.startTimeTicks &&
           bootId == o.bootId && topology == o.topology && labels == o.labels;
  }
};

// Why a runtime is RAS_DEGRADED: it runs, but its view may be incomplete.
enum DegradedReason : uint32_t {
  kDegradedInitFailed = 1u << 0,  // the runtime couldn't start; engines advertise no endpoint
  kDegradedLoopError = 1u << 1,   // the RAS thread hit an unexpected error and carried on
  kDegradedForTest = 1u << 31,    // set by tests
};

// One RAS link, to one remote runtime. P3 replaces this with the membership, connectivity,
// and progress state machines; part 1 tracks only the link itself.
enum class LinkState : uint8_t {
  kIdle,          // registered; the first connect is pending
  kConnecting,    // TCP connect or HELLO in progress
  kConnected,     // HELLO done; heartbeats flow
  kDisconnected,  // no connection; a reconnect is scheduled
  kPeerLeft,      // the peer sent LEAVE for its exit; its runtime UUID is gone for good
};

inline const char* ToString(LinkState s) {
  switch (s) {
    case LinkState::kIdle:
      return "IDLE";
    case LinkState::kConnecting:
      return "CONNECTING";
    case LinkState::kConnected:
      return "CONNECTED";
    case LinkState::kDisconnected:
      return "DISCONNECTED";
    case LinkState::kPeerLeft:
      return "PEER_LEFT";
  }
  return "UNKNOWN";
}

enum class LinkError : uint8_t {
  kNone,
  kRasEndpointUnavailable,  // the connect was refused: nothing listens at the peer's endpoint
  kConnectTimeout,          // the connect or HELLO took longer than the handshake timeout
  kConnectFailed,           // the connect failed otherwise, e.g. the network is unreachable
  kLinkClosed,              // the peer closed or reset the link without LEAVE
  kLinkReleased,            // the peer closed the link with LEAVE: it no longer needs it
  kTokenMismatch,           // the peer rejected the token from its EngineDesc
  kEndpointReused,          // a different runtime answers at the peer's endpoint
  kVersionMismatch,         // no common protocol version or security capability
  kDuplicate,               // the peer kept a crossing connection instead of this one
  kProtocolError,           // a malformed or out-of-sequence message
};

inline const char* ToString(LinkError e) {
  switch (e) {
    case LinkError::kNone:
      return "NONE";
    case LinkError::kRasEndpointUnavailable:
      return "RAS_ENDPOINT_UNAVAILABLE";
    case LinkError::kConnectTimeout:
      return "CONNECT_TIMEOUT";
    case LinkError::kConnectFailed:
      return "CONNECT_FAILED";
    case LinkError::kLinkClosed:
      return "LINK_CLOSED";
    case LinkError::kLinkReleased:
      return "LINK_RELEASED";
    case LinkError::kTokenMismatch:
      return "TOKEN_MISMATCH";
    case LinkError::kEndpointReused:
      return "ENDPOINT_REUSED";
    case LinkError::kVersionMismatch:
      return "VERSION_MISMATCH";
    case LinkError::kDuplicate:
      return "DUPLICATE";
    case LinkError::kProtocolError:
      return "PROTOCOL_ERROR";
  }
  return "UNKNOWN";
}

struct LinkTransition {
  int64_t agoMs{0};  // before the snapshot was published
  LinkState state{LinkState::kIdle};
  LinkError error{LinkError::kNone};
};

struct LinkInfo {
  RuntimeUuid peer;
  std::string peerHost;  // the endpoint this side connects to; empty if it never registered
  uint16_t peerPort{0};
  std::string peerHostname;
  int32_t peerPid{0};
  LinkState state{LinkState::kIdle};
  LinkError lastError{LinkError::kNone};
  std::string lastErrorDetail;
  bool outbound{false};  // the current or last connection was opened by this side
  bool peerDegraded{false};
  uint16_t version{0};    // negotiated protocol version
  uint32_t relations{0};  // local-to-remote engine registrations that use this link
  uint64_t connectAttempts{0};
  uint64_t connects{0};  // HELLO handshakes completed, either direction
  uint64_t disconnects{0};
  uint64_t heartbeatsSent{0};
  uint64_t heartbeatsReceived{0};
  uint64_t acksSent{0};
  uint64_t acksReceived{0};
  int64_t lastRxAgeMs{-1};  // since the last message from the peer; -1 if none
  bool hasTiming{false};
  int64_t rttNs{0};          // latest round-trip time
  int64_t offsetNs{0};       // best clock-offset estimate: peer wall clock minus ours
  int64_t offsetBoundNs{0};  // its error bound
  std::vector<LinkTransition> transitions;  // the last few state changes, oldest first
};

struct EventStats {
  size_t capacity{0};
  uint64_t enqueued{0};
  uint64_t dropped{0};
  std::vector<uint64_t> droppedByKind;  // indexed like RasEvent's alternatives
  size_t depthHighWater{0};
};

struct RuntimeSnapshot {
  RasProcessId self;
  uint16_t listenPort{0};
  uint32_t degraded{0};  // DegradedReason bits
  std::vector<LinkInfo> links;
  EventStats events;
  uint64_t inboundAccepted{0};   // TCP connections accepted
  uint64_t inboundRejected{0};   // HELLOs refused: token, version, duplicate, ...
  uint64_t protocolErrors{0};    // connections dropped for malformed messages
  uint64_t peersWithoutRas{0};   // registrations of engines that advertise no endpoint
  uint64_t sameProcessPeers{0};  // registrations of engines in this process
};

}  // namespace ras
}  // namespace mori
