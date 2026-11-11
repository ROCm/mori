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
// CPU-only loopback regression for the production TCP implementation.
#include <sys/socket.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"

using namespace mori::application;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace {
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

struct Pair {
  TCPContext listener{"127.0.0.1"};
  TCPContext client{"127.0.0.1"};
  TCPEndpointHandle sender;
  TCPEndpointHandle receiver;
  Pair() {
    listener.Listen();
    sender = client.Connect("127.0.0.1", listener.GetPort());
    for (int i = 0; i < 100; ++i) {
      auto accepted = listener.Accept();
      if (!accepted.empty()) {
        receiver = accepted.front();
        return;
      }
      std::this_thread::sleep_for(1ms);
    }
    throw std::runtime_error("loopback accept did not complete");
  }
};

// Deliberately send one byte per tick. MSG_NOSIGNAL makes an early deadline
// close safe for this self-owned sender. Stops after its peer closes.
std::thread Trickle(int fd, std::vector<char> bytes, std::chrono::milliseconds tick) {
  return std::thread([=] {
    for (char byte : bytes) {
      std::this_thread::sleep_for(tick);
      if (send(fd, &byte, 1, MSG_NOSIGNAL) != 1) return;
    }
  });
}

int ReadTransaction(TCPEndpoint& ep, void* header, size_t headerLen, void* body, size_t bodyLen,
                    size_t* consumed) {
#ifdef MORI_TEST_BASELINE
  int rc = ep.Recv(header, headerLen, consumed);
  if (rc != 0) return rc;
  return ep.Recv(body, bodyLen, consumed);
#else
  auto deadline = MakeControlPlaneReadDeadline();
  int rc = ep.Recv(header, headerLen, consumed, deadline);
  if (rc != 0) return rc;
  return ep.Recv(body, bodyLen, consumed, deadline);
#endif
}

void TestTrickleFrameDeadline() {
  Pair pair;
  TCPEndpoint receiver(pair.receiver);
  auto sender = Trickle(pair.sender.fd, std::vector<char>(20, 'x'), 20ms);
  char header[4]{}, body[16]{};
  size_t consumed = 0;
  auto start = Clock::now();
  int rc = ReadTransaction(receiver, header, sizeof(header), body, sizeof(body), &consumed);
  int err = errno;
  auto elapsed = Clock::now() - start;
  pair.listener.CloseEndpoint(pair.receiver);
  sender.join();
  std::cout << "trickle elapsed_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            << " rc=" << rc << " errno=" << err << " consumed=" << consumed << '\n';
  Require(rc == -1 && err == ETIMEDOUT, "trickled frame exceeded whole-message budget");
  Require(elapsed < 350ms, "transaction deadline was not bounded");
  Require(consumed < sizeof(body), "partial byte count lost");
}

void TestHeaderAndBodyShareBudget() {
  Pair pair;
  TCPEndpoint receiver(pair.receiver);
  auto sender = Trickle(pair.sender.fd, std::vector<char>(16, 'x'), 10ms);
  char header[8]{}, body[8]{};
  size_t consumed = 0;
  int rc = ReadTransaction(receiver, header, sizeof(header), body, sizeof(body), &consumed);
  int err = errno;
  pair.listener.CloseEndpoint(pair.receiver);
  sender.join();
  Require(rc == -1 && err == ETIMEDOUT, "header/body reads restarted the budget");
}

void TestEofByteCount() {
  Pair pair;
  TCPEndpoint sender(pair.sender), receiver(pair.receiver);
  Require(sender.Send("abc", 3) == 0, "send failed");
  pair.client.CloseEndpoint(pair.sender);
  char header[1]{}, body[7]{};
  size_t consumed = 99;
  int rc = ReadTransaction(receiver, header, sizeof(header), body, sizeof(body), &consumed);
  Require(rc == 1 && consumed == 2, "EOF/partial-byte diagnostics changed");
}

void TestSlowPeerAndHealthyNextClient() {
  // Serial handler pattern: first client sends a partial frame, second is
  // already ready. Exercise actual TCPContext/TCPEndpoint, not RDMA server.
  Pair bad, good;
  TCPEndpoint badReceiver(bad.receiver), goodReceiver(good.receiver), goodSender(good.sender);
  auto sender = Trickle(bad.sender.fd, std::vector<char>(20, 'x'), 20ms);
  Require(goodSender.Send("ok", 2) == 0, "healthy send failed");
  char header[4]{}, body[16]{}, healthy[2]{};
  size_t consumed = 0;
  auto start = Clock::now();
  int rc = ReadTransaction(badReceiver, header, sizeof(header), body, sizeof(body), &consumed);
  int err = errno;
  bad.listener.CloseEndpoint(bad.receiver);
  sender.join();
  Require(rc == -1 && err == ETIMEDOUT, "bad client did not expire");
  Require(goodReceiver.Recv(healthy, 2) == 0 && std::memcmp(healthy, "ok", 2) == 0,
          "healthy next client did not progress");
  Require(Clock::now() - start < 350ms, "slow client starved next client beyond deadline");
}

void TestCompletelySilentPeer() {
  Pair pair;
  TCPEndpoint receiver(pair.receiver);
  char header[1]{}, body[1]{};
  size_t consumed = 99;
  auto start = Clock::now();
  int rc = ReadTransaction(receiver, header, 1, body, 1, &consumed);
  Require(rc == -1 && errno == ETIMEDOUT && consumed == 0, "silent peer timeout changed");
  Require(Clock::now() - start < 350ms, "silent peer not bounded");
}

void TestLegacyRecvUnchanged() {
  Pair pair;
  TCPEndpoint receiver(pair.receiver);
  auto sender = Trickle(pair.sender.fd, std::vector<char>(10, 'x'), 20ms);
  char buf[10]{};
  auto start = Clock::now();
  int rc = receiver.Recv(buf, sizeof(buf));  // No transaction deadline: old API.
  auto elapsed = Clock::now() - start;
  sender.join();
  Require(rc == 0 && elapsed >= 150ms, "legacy Recv socket timeout semantics changed");
}

void TestOptOut() {
  Pair pair;
  TCPEndpoint receiver(pair.receiver);
  auto sender = Trickle(pair.sender.fd, std::vector<char>(10, 'x'), 25ms);
  char header[2]{}, body[8]{};
  size_t consumed = 0;
  auto start = Clock::now();
  int rc = ReadTransaction(receiver, header, sizeof(header), body, sizeof(body), &consumed);
  auto elapsed = Clock::now() - start;
  sender.join();
  Require(rc == 0 && elapsed >= 200ms, "timeout=0 no longer opts out");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::strcmp(argv[1], "--opt-out") == 0) {
      TestOptOut();
      std::cout << "PASS opt-out\n";
      return 0;
    }
    TestTrickleFrameDeadline();
    TestHeaderAndBodyShareBudget();
    TestEofByteCount();
    TestSlowPeerAndHealthyNextClient();
    TestCompletelySilentPeer();
    TestLegacyRecvUnchanged();
    std::cout << "PASS 6 production TCP loopback tests\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 2;
  }
}
