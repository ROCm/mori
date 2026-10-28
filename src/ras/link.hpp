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

// The RAS thread's event loop: the peer listener, the event queue, and the links, in one
// epoll set. Everything here runs on the RAS thread unless marked otherwise, so link state
// needs no locks; other threads see it only through published snapshots.
//
// Links. A link joins two runtimes, keyed by the remote runtime UUID that EngineDesc carries.
// Two registered engines of one remote process therefore share a link; part 2 adds reference
// counting by engine relationship on top, plus the link budget.
//
// Who connects. A side that has registered the peer opens the connection when no link
// exists. "The lower UUID connects" alone isn't enough: MORI-IO's initiator must register its
// target, but the target needn't register back, so a target with the lower UUID would never
// open the link. When both sides register at once, the two connections cross, and both sides
// keep the one the lower UUID opened: the acceptor refuses a crossing HELLO with
// kRejectedDuplicate if its own outgoing connection wins. A HELLO on a link whose current
// connection the peer itself opened replaces it, since the peer only reconnects after giving
// up on the old one.
//
// Lifetime. A link lasts while this side has a relation to the peer or the peer holds the
// connection open. Releasing the last relation on a connection this side opened sends
// LEAVE(kLinkReleased) and closes it; a peer that still has relations reconnects. A link whose
// connection drops reconnects at once if it was up, then backs off from 1 s to 30 s (both
// scaled by MORI_RAS_TIMEOUT_FACTOR). LEAVE(kProcessExit) retires the peer's UUID: no more
// reconnects. Detecting a silent peer from missing heartbeats is P3's connectivity machine.
//
// Every timer reads CLOCK_MONOTONIC, so an NTP step can't fire or suppress one.

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/ras/types.hpp"
#include "src/ras/config.hpp"
#include "src/ras/event_queue.hpp"
#include "src/ras/heartbeat.hpp"
#include "src/ras/protocol.hpp"

namespace mori {
namespace ras {

uint64_t MonoNowNs();
int64_t WallNowNs();

class LinkManager {
 public:
  struct Options {
    RasConfig config;
    RuntimeUuid uuid;
    std::string token;
    EventQueue* events{nullptr};
  };

  explicit LinkManager(Options options);
  // Closes every fd still open, without LEAVE.
  ~LinkManager();
  LinkManager(const LinkManager&) = delete;
  LinkManager& operator=(const LinkManager&) = delete;

  // On the starting thread, before the RAS thread runs. Binds and listens on `bindHost` (an
  // IPv4 or IPv6 literal; empty means every interface) at the configured port.
  bool Listen(const std::string& bindHost, std::string* error);
  uint16_t Port() const { return port_; }
  bool InitLoop(std::string* error);

  // RAS thread.
  void SetIdentity(const RasProcessId& self);
  void RunOnce(int maxWaitMs);
  // Sends LEAVE(kProcessExit) on every established link if asked, then closes everything.
  // Logs nothing: it also runs from the atexit path.
  void Shutdown(bool sendLeave);

  // Any thread.
  void RequestStop();
  bool StopRequested() const { return stop_.load(std::memory_order_acquire); }
  RuntimeSnapshot Snapshot() const;
  bool SetLabels(const Labels& labels);
  void AddDegraded(uint32_t reasons);
  uint32_t Degraded() const { return degraded_.load(std::memory_order_acquire); }
  // In a forked child, whose copies of these fds the fork handler already closed.
  void ForgetFds();

 private:
  enum class Phase : uint8_t {
    kTcpConnecting,  // outbound, connect() in progress
    kHelloSent,      // outbound, waiting for the HELLO reply
    kAwaitingHello,  // inbound, waiting for the HELLO request
    kEstablished,
    kClosing,  // half-closed after a final message; reading until the peer closes
  };

  struct Conn {
    uint64_t id{0};
    int fd{-1};
    bool outbound{false};
    bool pending{false};  // inbound before its HELLO; counts against kMaxPendingInbound
    bool broken{false};   // a send failed; closed at the end of the loop iteration
    bool inEpoll{false};
    Phase phase{Phase::kAwaitingHello};
    RuntimeUuid peer;     // expected for outbound, from the HELLO for inbound
    uint16_t version{0};  // negotiated; 0 until the HELLO exchange completes
    uint64_t txSeq{0};
    uint64_t rxSeq{0};
    uint64_t deadlineNs{0};  // the handshake deadline, or for kClosing, when to stop waiting
    uint32_t epollEvents{0};
    std::vector<uint8_t> in;
    size_t inStart{0};
    std::vector<uint8_t> out;
    size_t outStart{0};
  };

  struct Link {
    RuntimeUuid peer;
    std::set<std::pair<std::string, std::string>> relations;  // (local engine, remote engine)
    RasEndpoint endpoint;  // where to connect; invalid until a registration names it
    std::string peerHostname;
    int32_t peerPid{0};
    Conn* conn{nullptr};
    LinkState state{LinkState::kIdle};
    LinkError lastError{LinkError::kNone};
    std::string lastErrorDetail;
    bool lastOutbound{false};
    bool peerDegraded{false};
    uint16_t version{0};
    bool connectScheduled{false};
    uint64_t connectDueNs{0};
    uint64_t backoffNs{0};
    uint64_t nextHeartbeatNs{0};
    uint64_t lastRxNs{0};
    HeartbeatTracker heartbeat;
    uint64_t connectAttempts{0};
    uint64_t connects{0};
    uint64_t disconnects{0};
    uint64_t heartbeatsSent{0};
    uint64_t heartbeatsReceived{0};
    uint64_t acksSent{0};
    uint64_t acksReceived{0};
    struct Transition {
      uint64_t atNs;
      LinkState state;
      LinkError error;
    };
    std::deque<Transition> transitions;
  };

  using LinkMap = std::map<RuntimeUuid, Link>;

  // Loop plumbing.
  void AcceptAll(uint64_t now);
  void HandleEvent(RasEvent& ev, uint64_t now);
  void HandleConnEvent(Conn* c, uint32_t events, uint64_t now);
  void RunTimers(uint64_t now);
  void SweepBroken(uint64_t now);
  uint64_t NextDeadlineNs(uint64_t now) const;
  void Publish(uint64_t nowNs);
  void MarkDirty() { dirty_ = true; }

  // Connections.
  Conn* NewConn(int fd, bool outbound, uint64_t now);
  void Watch(Conn* c, uint32_t events);
  void CloseConn(Conn* c);
  void BeginClose(Conn* c, uint64_t now);
  Link* LinkOf(const Conn* c);
  bool Send(Conn* c, const proto::Message& m);
  void Flush(Conn* c);
  void SendHello(Conn* c, proto::HelloRole role, proto::HelloStatus status, const Link* l);

  // Reading and the protocol.
  void OnReadable(Conn* c, uint64_t now);
  bool ProcessFrames(Conn* c, uint64_t rxMonoNs, int64_t rxWallNs);
  bool OnHelloRequest(Conn* c, const proto::RasHeader& h, const proto::Hello& hello, uint64_t now);
  bool OnHelloReply(Conn* c, const proto::RasHeader& h, const proto::Hello& hello, uint64_t now);
  bool OnLinkMessage(Conn* c, Link& l, const proto::Message& m, uint64_t rxMonoNs,
                     int64_t rxWallNs);
  void RejectHello(Conn* c, proto::HelloStatus status, uint64_t now);
  void ProtocolError(Conn* c, const std::string& detail, uint64_t now);

  // Links.
  void AddRelation(const PeerAdded& e, uint64_t now);
  void RemoveRelation(const std::string& local, const std::string& remote, uint64_t now);
  void StartConnect(Link& l, uint64_t now);
  void OnConnected(Conn* c, uint64_t now);
  void Establish(Link& l, Conn* c, uint16_t version, const proto::Hello& hello, uint64_t now);
  // The link lost its connection. Reconnects if this side still has relations, now if
  // `retryNow`; otherwise the link is forgotten. Returns false if it was forgotten.
  bool LinkDown(LinkMap::iterator it, LinkError error, const std::string& detail, uint64_t now,
                bool retryNow);
  void ScheduleConnect(Link& l, uint64_t now, bool immediate);
  void SendHeartbeat(Link& l, uint64_t now);
  void SetState(Link& l, LinkState state, LinkError error, uint64_t now);
  LinkInfo Describe(const Link& l, uint64_t now) const;

  Options opts_;
  int listenFd_{-1};
  int epollFd_{-1};
  uint16_t port_{0};
  RasProcessId self_;
  std::set<std::string> localEngines_;
  LinkMap links_;
  std::unordered_map<uint64_t, std::unique_ptr<Conn>> conns_;
  uint64_t nextConnId_{1};
  std::map<std::pair<std::string, std::string>, RuntimeUuid> relations_;
  size_t pendingInbound_{0};
  // accept() failed for lack of fds: the listener sleeps briefly instead of spinning.
  bool acceptPaused_{false};
  uint64_t acceptResumeNs_{0};
  uint64_t inboundAccepted_{0};
  uint64_t inboundRejected_{0};
  uint64_t protocolErrors_{0};
  uint64_t peersWithoutRas_{0};
  uint64_t sameProcessPeers_{0};
  bool dirty_{true};
  bool moreEvents_{false};  // the last drain stopped at its limit
  uint64_t lastPublishNs_{0};

  std::atomic<bool> stop_{false};
  std::atomic<uint32_t> degraded_{0};
  mutable std::mutex labelsMu_;
  Labels labels_;
  std::atomic<bool> labelsChanged_{false};
  mutable std::mutex snapshotMu_;
  std::shared_ptr<const RuntimeSnapshot> published_;
};

}  // namespace ras
}  // namespace mori
