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

// Two IOEngine processes on one node, linked by their RAS runtimes. Each child is a separate
// process with its own runtime, started by its IOEngine. The harness, which never starts RAS,
// passes each child the other's packed EngineDesc the way a framework's side channel would,
// then asks each child to wait for a condition on its link. (The plan's
// tests/python/ras/test_two_process.py, in C++: the Python bindings need ops and shmem.)
#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <msgpack.hpp>
#include <string>
#include <vector>

#include "mori/io/engine.hpp"
#include "mori/ras/ras.hpp"

using namespace mori::io;
using namespace mori::ras;

namespace {

bool WriteMsg(int fd, const std::string& msg) {
  uint32_t len = static_cast<uint32_t>(msg.size());
  std::string frame(reinterpret_cast<const char*>(&len), sizeof(len));
  frame += msg;
  size_t off = 0;
  while (off < frame.size()) {
    ssize_t n = write(fd, frame.data() + off, frame.size() - off);
    if (n <= 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

bool ReadExactly(int fd, char* out, size_t n, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  size_t got = 0;
  while (got < n) {
    int left = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    deadline - std::chrono::steady_clock::now())
                                    .count());
    pollfd p{fd, POLLIN, 0};
    if (left <= 0 || poll(&p, 1, left) != 1) return false;
    ssize_t r = read(fd, out + got, n - got);
    if (r <= 0) return false;
    got += static_cast<size_t>(r);
  }
  return true;
}

bool ReadMsg(int fd, std::string* msg, int timeoutMs) {
  uint32_t len = 0;
  if (!ReadExactly(fd, reinterpret_cast<char*>(&len), sizeof(len), timeoutMs)) return false;
  msg->assign(len, '\0');
  return len == 0 || ReadExactly(fd, &(*msg)[0], len, timeoutMs);
}

std::string PackDesc(const EngineDesc& d) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, d);
  return std::string(buf.data(), buf.size());
}

EngineDesc UnpackDesc(const std::string& bytes) {
  return msgpack::unpack(bytes.data(), bytes.size()).get().as<EngineDesc>();
}

// --- child side -----------------------------------------------------------------------------

std::string CheckLink(const std::string& cond, const RuntimeUuid& peer) {
  RuntimeSnapshot s = RasRuntime::Get()->Snapshot();
  if (cond == "rejected_inbound") return s.inboundRejected >= 1 ? "" : "no HELLO was rejected";
  const LinkInfo* l = nullptr;
  for (const LinkInfo& i : s.links) {
    if (i.peer == peer) l = &i;
  }
  if (l == nullptr) return "no link to the peer";
  std::string state = std::string(ToString(l->state)) + "/" + ToString(l->lastError);
  if (cond == "heartbeats") {
    bool ok = l->state == LinkState::kConnected && l->heartbeatsReceived >= 3 &&
              l->acksReceived >= 3 && l->hasTiming;
    return ok ? "" : "link " + state;
  }
  if (cond == "refused") {
    for (size_t i = 1; i < l->transitions.size(); ++i) {
      if (l->transitions[i - 1].error == LinkError::kLinkClosed &&
          l->transitions[i].error == LinkError::kRasEndpointUnavailable) {
        return "";
      }
    }
    return "link " + state + ": no LINK_CLOSED then RAS_ENDPOINT_UNAVAILABLE";
  }
  if (cond == "token_mismatch") {
    bool ok = l->lastError == LinkError::kTokenMismatch && l->heartbeatsReceived == 0;
    return ok ? "" : "link " + state;
  }
  if (cond == "left") return l->state == LinkState::kPeerLeft ? "" : "link " + state;
  return "unknown condition " + cond;
}

int ChildMain(const std::string& name, int cmdFd, int replyFd) {
  setenv("MORI_RAS_ENABLE", "1", 1);
  setenv("MORI_RAS_HEARTBEAT_MS", "100", 1);
  IOEngine engine("engine-" + name, IOEngineConfig{"127.0.0.1", 0});
  if (!WriteMsg(replyFd, PackDesc(engine.GetEngineDesc()))) return 1;
  RuntimeUuid peer;
  for (;;) {
    std::string cmd;
    if (!ReadMsg(cmdFd, &cmd, 60000)) return 0;
    if (cmd.rfind("REGISTER ", 0) == 0) {
      EngineDesc remote = UnpackDesc(cmd.substr(9));
      peer = RuntimeUuid{remote.ras.uuidHi, remote.ras.uuidLo};
      engine.RegisterRemoteEngine(remote);
      WriteMsg(replyFd, "OK");
    } else if (cmd.rfind("WAIT ", 0) == 0) {
      std::string cond = cmd.substr(5);
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      std::string why = CheckLink(cond, peer);
      while (!why.empty() && std::chrono::steady_clock::now() < deadline) {
        usleep(5000);
        why = CheckLink(cond, peer);
      }
      WriteMsg(replyFd, why.empty() ? "OK" : "FAIL " + why);
    } else if (cmd == "EXIT") {
      std::exit(0);  // a clean exit: the atexit handler sends LEAVE
    }
  }
}

// --- harness side ---------------------------------------------------------------------------

struct Child {
  pid_t pid{-1};
  int cmd{-1};
  int reply{-1};
  std::string desc;  // its packed EngineDesc

  std::string Ask(const std::string& msg) {
    std::string answer;
    if (!WriteMsg(cmd, msg) || !ReadMsg(reply, &answer, 10000)) return "no answer";
    return answer;
  }
  void Kill() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      pid = -1;
    }
  }
  ~Child() {
    Kill();
    close(cmd);
    close(reply);
  }
};

std::unique_ptr<Child> Spawn(const std::string& name) {
  int toChild[2], fromChild[2];
  if (pipe(toChild) != 0 || pipe(fromChild) != 0) return nullptr;
  std::fflush(nullptr);
  pid_t pid = fork();
  if (pid == 0) {
    close(toChild[1]);
    close(fromChild[0]);
    _exit(ChildMain(name, toChild[0], fromChild[1]));
  }
  close(toChild[0]);
  close(fromChild[1]);
  auto c = std::make_unique<Child>();
  c->pid = pid;
  c->cmd = toChild[1];
  c->reply = fromChild[0];
  if (!ReadMsg(c->reply, &c->desc, 10000)) return nullptr;
  return c;
}

// Spawns two children and has each register the other.
void SpawnLinkedPair(std::unique_ptr<Child>* a, std::unique_ptr<Child>* b) {
  *a = Spawn("a");
  *b = Spawn("b");
  ASSERT_TRUE(*a && *b);
  ASSERT_EQ((*a)->Ask("REGISTER " + (*b)->desc), "OK");
  ASSERT_EQ((*b)->Ask("REGISTER " + (*a)->desc), "OK");
}

}  // namespace

TEST(RasTwoProcess, TwoIoEnginesExchangeHeartbeats) {
  unsetenv("MORI_RAS_ENABLE");
  std::unique_ptr<Child> a, b;
  SpawnLinkedPair(&a, &b);
  // Each EngineDesc carries its process's RAS endpoint.
  for (const Child* c : {a.get(), b.get()}) {
    EngineDesc d = UnpackDesc(c->desc);
    EXPECT_EQ(d.ras.host, "127.0.0.1");
    EXPECT_NE(d.ras.port, 0);
    EXPECT_EQ(d.ras.token.size(), 32u);
    EXPECT_NE(d.ras.uuidLo, 0u);
    EXPECT_EQ(d.pid, c->pid);
  }
  EXPECT_NE(UnpackDesc(a->desc).ras.uuidLo, UnpackDesc(b->desc).ras.uuidLo);
  EXPECT_EQ(a->Ask("WAIT heartbeats"), "OK");
  EXPECT_EQ(b->Ask("WAIT heartbeats"), "OK");
}

TEST(RasTwoProcess, SigkillClosesTheLinkAndTheSurvivorsReconnectIsRefused) {
  std::unique_ptr<Child> a, b;
  SpawnLinkedPair(&a, &b);
  ASSERT_EQ(a->Ask("WAIT heartbeats"), "OK");
  auto killed = std::chrono::steady_clock::now();
  b->Kill();
  EXPECT_EQ(a->Ask("WAIT refused"), "OK");
  // Well before any heartbeat timeout: the kernel closes the link, and the reconnect that
  // follows at once is refused.
  EXPECT_LT(std::chrono::steady_clock::now() - killed, std::chrono::seconds(2));
}

TEST(RasTwoProcess, ATokenMismatchIsRejected) {
  std::unique_ptr<Child> a = Spawn("a");
  std::unique_ptr<Child> b = Spawn("b");
  ASSERT_TRUE(a && b);
  EngineDesc tampered = UnpackDesc(b->desc);
  tampered.ras.token = "ffffffffffffffffffffffffffffffff";
  ASSERT_EQ(a->Ask("REGISTER " + PackDesc(tampered)), "OK");
  EXPECT_EQ(a->Ask("WAIT token_mismatch"), "OK");
  EXPECT_EQ(b->Ask("WAIT rejected_inbound"), "OK");
}

TEST(RasTwoProcess, ACleanExitSendsLeave) {
  std::unique_ptr<Child> a, b;
  SpawnLinkedPair(&a, &b);
  ASSERT_EQ(a->Ask("WAIT heartbeats"), "OK");
  ASSERT_TRUE(WriteMsg(b->cmd, "EXIT"));
  int status = -1;
  ASSERT_EQ(waitpid(b->pid, &status, 0), b->pid);
  b->pid = -1;
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  EXPECT_EQ(a->Ask("WAIT left"), "OK");
}
