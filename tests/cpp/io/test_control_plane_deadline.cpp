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
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Exercise the production epoll handler and connection cleanup without a NIC.
#include <poll.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <msgpack.hpp>
#include <stdexcept>
#include <thread>

#include "mori/application/transport/tcp/tcp.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/protocol.hpp"

using namespace mori::application;
using namespace mori::io;
using namespace std::chrono_literals;

namespace {
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

class Trickle {
 public:
  Trickle(int fd, const msgpack::sbuffer& body)
      : bytes(body.data(), body.size()), worker([this, fd] {
          for (char byte : bytes) {
            if (stop.load()) return;
            if (send(fd, &byte, 1, MSG_NOSIGNAL) != 1) return;
            std::this_thread::sleep_for(20ms);
          }
        }) {}
  ~Trickle() {
    stop.store(true);
    worker.join();
  }

 private:
  std::string bytes;
  std::atomic<bool> stop{false};
  std::thread worker;
};

void AskMissingRegion(uint16_t port) {
  TCPContext client("127.0.0.1");
  auto handle = client.Connect("127.0.0.1", port);
  Protocol protocol(handle);
  MessageAskMemoryRegion request{"healthy", 0, 123, {}};
  protocol.WriteMessageAskMemoryRegion(request);
  // The healthy request can queue behind the bad peer. Wait for readability
  // before starting its own frame deadline; bound the total wait separately.
  pollfd ready{handle.fd, POLLIN, 0};
  Require(poll(&ready, 1, 1500) == 1, "healthy peer stalled behind bad connection");
  auto header = protocol.ReadMessageHeader();
  ExpectMessage(header, MessageType::AskMemoryRegion);
  auto reply = protocol.ReadMessageAskMemoryRegion(header.len);
  Require(reply.ekey == request.ekey && reply.id == request.id,
          "healthy peer received the wrong reply");
}

void TestSlowBodyThenHealthyPeer() {
  // The absent memory ID takes the NOT_FOUND path, which does not use either
  // manager. The server's real accept/epoll/protocol/drop paths still execute.
  ControlPlaneServer server("test", "127.0.0.1", 0, {}, nullptr, nullptr);
  server.Start();
  uint16_t port = server.GetListenPort().value();
  TCPContext bad("127.0.0.1");
  auto handle = bad.Connect("127.0.0.1", port);
  msgpack::sbuffer body;
  msgpack::pack(body, MessageAskMemoryRegion{std::string(256, 'x'), 0, 456, {}});
  Protocol protocol(handle);
  protocol.WriteMessageHeader({MessageType::AskMemoryRegion, static_cast<uint32_t>(body.size())});
  {
    Trickle trickle(handle.fd, body);
    // Give the first peer a head start, so it reaches the single-threaded
    // handler before the healthy request. Its full body would take >5 seconds.
    std::this_thread::sleep_for(100ms);
    AskMissingRegion(port);
  }
  // Failure of one connection must not stop the server or poison later frames.
  AskMissingRegion(port);
  server.Shutdown();
}

void TestMalformedAndTruncatedThenHealthyPeer() {
  ControlPlaneServer server("test", "127.0.0.1", 0, {}, nullptr, nullptr);
  server.Start();
  uint16_t port = server.GetListenPort().value();
  {
    TCPContext bad("127.0.0.1");
    auto handle = bad.Connect("127.0.0.1", port);
    Protocol protocol(handle);
    protocol.WriteMessageHeader({MessageType::AskMemoryRegion, kMaxControlMessageBytes + 1});
  }
  AskMissingRegion(port);
  {
    TCPContext bad("127.0.0.1");
    auto handle = bad.Connect("127.0.0.1", port);
    TCPEndpoint endpoint(handle);
    char type = static_cast<char>(MessageType::AskMemoryRegion);
    Require(endpoint.Send(&type, 1) == 0, "could not send partial header");
  }
  AskMissingRegion(port);
  server.Shutdown();
}
}  // namespace

int main() {
  try {
    TestSlowBodyThenHealthyPeer();
    TestMalformedAndTruncatedThenHealthyPeer();
    std::cout << "PASS production control-plane recovery (slow, oversized, truncated)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
