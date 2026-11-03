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

// A process running an RCCL communicator with RCCL RAS on, and MORI RAS through an IOEngine.
// Both runtimes must keep working, each on its own thread and port. RCCL's RAS client socket
// is moved off 28028 with NCCL_RAS_ADDR, so the test doesn't collide with other jobs on the
// host.
#include <arpa/inet.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "mori/io/engine.hpp"
#include "mori/ras/ras.hpp"
#include "src/ras/fd_registry.hpp"
#include "src/ras/runtime.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::io;
using namespace mori::ras;

namespace {

// RCCL RAS's client protocol: a version line each way, then a request.
std::string QueryRcclRas(uint16_t port, const std::string& request) {
  int fd = ras_test::ConnectTcp("127.0.0.1", port, nullptr);
  if (fd < 0) return "";
  std::string out = "CLIENT PROTOCOL 2\n" + request;
  if (send(fd, out.data(), out.size(), MSG_NOSIGNAL) != static_cast<ssize_t>(out.size())) {
    close(fd);
    return "";
  }
  timeval tv{5, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  std::string reply;
  char buf[4096];
  ssize_t n;
  while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) reply.append(buf, static_cast<size_t>(n));
  close(fd);
  return reply;
}

int CountThreads(const std::string& name) {
  int count = 0;
  DIR* d = opendir("/proc/self/task");
  while (dirent* e = d != nullptr ? readdir(d) : nullptr) {
    if (e->d_name[0] == '.') continue;
    std::ifstream comm(std::string("/proc/self/task/") + e->d_name + "/comm");
    std::string line;
    std::getline(comm, line);
    if (line == name) ++count;
  }
  if (d != nullptr) closedir(d);
  return count;
}

uint16_t LocalPort(int fd) {
  sockaddr_storage a{};
  socklen_t len = sizeof(a);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0) return 0;
  if (a.ss_family == AF_INET) return ntohs(reinterpret_cast<sockaddr_in*>(&a)->sin_port);
  if (a.ss_family == AF_INET6) return ntohs(reinterpret_cast<sockaddr_in6*>(&a)->sin6_port);
  return 0;
}

// Outside CHILD_CHECK's 1-250, so it can't be mistaken for a failed check.
constexpr int kNoGpu = 251;

int Body() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) return kNoGpu;

  int holder = -1;
  uint16_t rcclPort = ras_test::HoldPort("127.0.0.1", &holder);
  close(holder);
  setenv("NCCL_RAS_ENABLE", "1", 1);
  setenv("NCCL_RAS_ADDR", ("127.0.0.1:" + std::to_string(rcclPort)).c_str(), 1);
  setenv("MORI_RAS_ENABLE", "1", 1);
  setenv("MORI_RAS_HEARTBEAT_MS", "100", 1);

  ncclComm_t comm;
  int dev = 0;
  CHILD_CHECK(ncclCommInitAll(&comm, 1, &dev) == ncclSuccess);
  // RCCL doesn't name its threads; its RAS thread shows itself by answering its client port.
  CHILD_CHECK(ras_test::WaitFor(
      [&] { return QueryRcclRas(rcclPort, "STATUS\n").find("RUNNING") != std::string::npos; },
      5000));
  CHILD_CHECK(CountThreads("mori-ras") == 0);

  // MORI RAS starts with the engine; a second runtime in this process stands in for a peer.
  IOEngine engine("engine-a", IOEngineConfig{"127.0.0.1", 0});
  RasRuntime* rt = RasRuntime::Get();
  CHILD_CHECK(rt != nullptr);
  CoreParams peerParams;
  peerParams.config.enable = true;
  peerParams.config.heartbeatMs = 100;
  peerParams.host = "127.0.0.1";
  peerParams.hostname = "peer";
  RasCore peer(std::move(peerParams));
  CHILD_CHECK(peer.Start(nullptr));
  EngineDesc remote;
  remote.key = "engine-peer";
  remote.host = "127.0.0.1";
  RasEndpoint pe = peer.Endpoint();
  remote.ras = RasDesc{pe.host, pe.port, pe.token, pe.uuid.hi, pe.uuid.lo};
  engine.RegisterRemoteEngine(remote);

  CHILD_CHECK(ras_test::WaitFor(
      [&] {
        for (const LinkInfo& l : rt->Snapshot().links) {
          if (l.peer == pe.uuid && l.state == LinkState::kConnected && l.acksReceived >= 3 &&
              l.heartbeatsReceived >= 3) {
            return true;
          }
        }
        return false;
      },
      5000));

  // RCCL RAS still answers on its own port, and sees its communicator running.
  std::string status = QueryRcclRas(rcclPort, "STATUS\n");
  CHILD_CHECK(status.rfind("SERVER PROTOCOL 2", 0) == 0);
  CHILD_CHECK(status.find("Job summary") != std::string::npos);
  CHILD_CHECK(status.find("RUNNING") != std::string::npos);

  // Separate threads and ports: none of MORI's sockets uses RCCL's port or 28028.
  CHILD_CHECK(CountThreads("mori-ras") == 2);  // the engine's runtime and the stand-in peer
  RasEndpoint mine = rt->Endpoint();
  CHILD_CHECK(mine.port != rcclPort && mine.port != 28028);
  for (int fd : fdreg::Snapshot()) {
    uint16_t p = LocalPort(fd);
    CHILD_CHECK(p != rcclPort && p != 28028);
  }

  // RCCL itself still works while both runtimes run.
  float* buf = nullptr;
  CHILD_CHECK(hipSetDevice(0) == hipSuccess);
  CHILD_CHECK(hipMalloc(&buf, 1024 * sizeof(float)) == hipSuccess);
  hipStream_t stream;
  CHILD_CHECK(hipStreamCreate(&stream) == hipSuccess);
  CHILD_CHECK(ncclAllReduce(buf, buf, 1024, ncclFloat, ncclSum, comm, stream) == ncclSuccess);
  CHILD_CHECK(hipStreamSynchronize(stream) == hipSuccess);
  CHILD_CHECK(hipStreamDestroy(stream) == hipSuccess);
  CHILD_CHECK(hipFree(buf) == hipSuccess);
  CHILD_CHECK(ncclCommDestroy(comm) == ncclSuccess);
  return 0;
}

}  // namespace

TEST(RasRcclCoexistence, BothRasRuntimesKeepWorkingInOneProcess) {
  // Everything runs in a child: HIP can't be used across fork once initialized, and the RAS
  // runtime is a per-process singleton.
  int rc = ras_test::RunInChild(Body);
  if (rc == kNoGpu) GTEST_SKIP() << "no GPU";
  EXPECT_EQ(rc, 0);
}
