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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "mori/io/io.hpp"
#include "src/io/tcp/transport.hpp"

namespace mori::io {
namespace {

bool SendAll(int fd, const uint8_t* data, size_t size) {
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool WaitForSocketClose(int fd, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  uint8_t discard[256];
  while (std::chrono::steady_clock::now() < deadline) {
    pollfd pfd{fd, POLLIN | POLLHUP | POLLERR, 0};
    int remaining = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         deadline - std::chrono::steady_clock::now())
                                         .count());
    int rc = poll(&pfd, 1, std::max(1, remaining));
    if (rc < 0 && errno == EINTR) continue;
    if (rc <= 0) return false;
    ssize_t n = recv(fd, discard, sizeof(discard), 0);
    if (n == 0) return true;
    if (n < 0 && !IsWouldBlock(errno)) return true;
  }
  return false;
}

bool RecvExactWithTimeout(int fd, uint8_t* data, size_t size, int timeoutMs = 2000) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  size_t received = 0;
  while (received < size && std::chrono::steady_clock::now() < deadline) {
    pollfd pfd{fd, POLLIN | POLLHUP | POLLERR, 0};
    int remaining = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         deadline - std::chrono::steady_clock::now())
                                         .count());
    int rc = poll(&pfd, 1, std::max(1, remaining));
    if (rc < 0 && errno == EINTR) continue;
    if (rc <= 0) return false;
    ssize_t n = recv(fd, data + received, size - received, 0);
    if (n <= 0) return false;
    received += static_cast<size_t>(n);
  }
  return received == size;
}

bool RecvCtrlFrame(int fd, tcp::CtrlHeaderView* header, std::vector<uint8_t>* body) {
  std::array<uint8_t, tcp::kCtrlHeaderSize> bytes{};
  if (!RecvExactWithTimeout(fd, bytes.data(), bytes.size())) return false;
  if (tcp::TryParseCtrlHeader(bytes.data(), bytes.size(), header) != tcp::ParseError::Ok)
    return false;
  body->resize(header->bodyLen);
  return body->empty() || RecvExactWithTimeout(fd, body->data(), body->size());
}

int ConnectLoopback(uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) != 1 ||
      connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

struct TcpPair {
  std::unique_ptr<IOEngine> a;
  std::unique_ptr<IOEngine> b;

  TcpPair() {
    IOEngineConfig config{"127.0.0.1", 0};
    a = std::make_unique<IOEngine>("tcp_test_a", config);
    b = std::make_unique<IOEngine>("tcp_test_b", config);
    TcpBackendConfig tcp(32 * 1024 * 1024, 32 * 1024 * 1024, 5000, true, 30, 10, 3, true, 4, 1);
    a->CreateBackend(BackendType::TCP, tcp);
    b->CreateBackend(BackendType::TCP, tcp);
    a->RegisterRemoteEngine(b->GetEngineDesc());
    b->RegisterRemoteEngine(a->GetEngineDesc());
  }
};

TEST(TcpTransport, CpuWriteReadAndMultiSegmentStriping) {
  TcpPair pair;
  std::array<uint8_t, 4096> source{};
  std::array<uint8_t, 4096> destination{};
  for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i * 17);

  MemoryDesc sourceDesc =
      pair.a->RegisterMemory(source.data(), source.size(), -1, MemoryLocationType::CPU);
  MemoryDesc destinationDesc =
      pair.b->RegisterMemory(destination.data(), destination.size(), -1, MemoryLocationType::CPU);

  TransferStatus writeStatus;
  auto writeId = pair.a->AllocateTransferUniqueId();
  pair.a->Write(sourceDesc, 0, destinationDesc, 0, source.size(), &writeStatus, writeId);
  EXPECT_EQ(writeStatus.WaitFor(5000), StatusCode::SUCCESS) << writeStatus.Message();
  EXPECT_EQ(source, destination);

  std::fill(destination.begin(), destination.end(), 0);
  auto session = pair.a->CreateSession(sourceDesc, destinationDesc);
  ASSERT_TRUE(session.has_value());
  SizeVec offsets{0, 1024, 2048, 3072};
  SizeVec sizes{511, 509, 507, 505};
  TransferStatus batchStatus;
  session->BatchWrite(offsets, offsets, sizes, &batchStatus, session->AllocateTransferUniqueId());
  EXPECT_EQ(batchStatus.WaitFor(5000), StatusCode::SUCCESS) << batchStatus.Message();
  for (size_t i = 0; i < offsets.size(); ++i)
    EXPECT_TRUE(std::equal(source.begin() + offsets[i], source.begin() + offsets[i] + sizes[i],
                           destination.begin() + offsets[i]));

  std::fill(source.begin(), source.end(), 0);
  TransferStatus readStatus;
  pair.a->Read(sourceDesc, 0, destinationDesc, 0, destination.size(), &readStatus,
               pair.a->AllocateTransferUniqueId());
  EXPECT_EQ(readStatus.WaitFor(5000), StatusCode::SUCCESS) << readStatus.Message();
  EXPECT_EQ(source, destination);
}

TEST(TcpTransport, DuplicatePeerOpIdOnlyFailsNewSubmission) {
  TcpPair pair;
  std::array<uint8_t, 1024> source{};
  std::array<uint8_t, 1024> destination{};
  MemoryDesc sourceDesc =
      pair.a->RegisterMemory(source.data(), source.size(), -1, MemoryLocationType::CPU);
  MemoryDesc destinationDesc =
      pair.b->RegisterMemory(destination.data(), destination.size(), -1, MemoryLocationType::CPU);

  TransferStatus first;
  TransferStatus duplicate;
  constexpr TransferUniqueId kId = 91;
  pair.a->Write(sourceDesc, 0, destinationDesc, 0, source.size(), &first, kId);
  pair.a->Write(sourceDesc, 0, destinationDesc, 0, source.size(), &duplicate, kId);
  EXPECT_EQ(first.WaitFor(5000), StatusCode::SUCCESS) << first.Message();
  EXPECT_EQ(duplicate.WaitFor(5000), StatusCode::ERR_BAD_STATE);
}

TEST(TcpTransport, ReadResponseAndWritePayloadMayShareOpId) {
  TcpPair pair;
  std::array<uint8_t, 2048> readSource{};
  std::array<uint8_t, 2048> readDestination{};
  std::array<uint8_t, 2048> writeSource{};
  std::array<uint8_t, 2048> writeDestination{};
  for (size_t i = 0; i < readSource.size(); ++i) {
    readSource[i] = static_cast<uint8_t>(i * 5);
    writeSource[i] = static_cast<uint8_t>(i * 11);
  }

  MemoryDesc readSourceDesc =
      pair.b->RegisterMemory(readSource.data(), readSource.size(), -1, MemoryLocationType::CPU);
  MemoryDesc readDestinationDesc = pair.a->RegisterMemory(
      readDestination.data(), readDestination.size(), -1, MemoryLocationType::CPU);
  MemoryDesc writeSourceDesc =
      pair.b->RegisterMemory(writeSource.data(), writeSource.size(), -1, MemoryLocationType::CPU);
  MemoryDesc writeDestinationDesc = pair.a->RegisterMemory(
      writeDestination.data(), writeDestination.size(), -1, MemoryLocationType::CPU);

  constexpr TransferUniqueId kSharedId = 777;
  TransferStatus readStatus;
  TransferStatus writeStatus;
  pair.a->Read(readDestinationDesc, 0, readSourceDesc, 0, readSource.size(), &readStatus,
               kSharedId);
  pair.b->Write(writeSourceDesc, 0, writeDestinationDesc, 0, writeSource.size(), &writeStatus,
                kSharedId);

  EXPECT_EQ(readStatus.WaitFor(5000), StatusCode::SUCCESS) << readStatus.Message();
  EXPECT_EQ(writeStatus.WaitFor(5000), StatusCode::SUCCESS) << writeStatus.Message();
  EXPECT_EQ(readSource, readDestination);
  EXPECT_EQ(writeSource, writeDestination);
}

TEST(TcpTransport, PeerAbortAllowsNextOperationToReconnect) {
  IOEngineConfig config{"127.0.0.1", 0};
  auto a = std::make_unique<IOEngine>("tcp_reconnect_a", config);
  auto b = std::make_unique<IOEngine>("tcp_reconnect_b", config);
  TcpBackendConfig tcp(32 * 1024 * 1024, 32 * 1024 * 1024, 250, true, 30, 10, 3, true, 2, 1);
  a->CreateBackend(BackendType::TCP, tcp);
  b->CreateBackend(BackendType::TCP, tcp);

  EngineDesc correctB = b->GetEngineDesc();
  EngineDesc unreachableB = correctB;
  unreachableB.port = 1;
  a->RegisterRemoteEngine(unreachableB);
  b->RegisterRemoteEngine(a->GetEngineDesc());

  std::array<uint8_t, 512> source{};
  std::array<uint8_t, 512> destination{};
  source.fill(0x5a);
  MemoryDesc sourceDesc =
      a->RegisterMemory(source.data(), source.size(), -1, MemoryLocationType::CPU);
  MemoryDesc destinationDesc =
      b->RegisterMemory(destination.data(), destination.size(), -1, MemoryLocationType::CPU);

  TransferStatus failed;
  a->Write(sourceDesc, 0, destinationDesc, 0, source.size(), &failed, 1);
  EXPECT_NE(failed.WaitFor(2000), StatusCode::SUCCESS);
  EXPECT_FALSE(failed.InProgress());

  a->RegisterRemoteEngine(correctB);
  TransferStatus recovered;
  a->Write(sourceDesc, 0, destinationDesc, 0, source.size(), &recovered, 2);
  EXPECT_EQ(recovered.WaitFor(5000), StatusCode::SUCCESS) << recovered.Message();
  EXPECT_EQ(source, destination);
}

TEST(TcpTransport, RejectsInvalidLaneConfiguration) {
  EXPECT_THROW(TcpBackendConfig(1, 1, 1000, true, 1, 1, 1, true, 0, 1), std::invalid_argument);
  EXPECT_THROW(TcpBackendConfig(1, 1, 1000, true, 1, 1, 1, true, 17, 1), std::invalid_argument);
}

TEST(TcpTransportWorker, InitialPayloadWaitsForSharedTargetWithoutEarlyAllocation) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  ASSERT_EQ(SetNonBlocking(sockets[0]), 0);
  ASSERT_EQ(SetNonBlocking(sockets[1]), 0);

  constexpr TransferUniqueId kOpId = 1234;
  std::array<uint8_t, 32> payload{};
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>(i + 1);
  auto header = tcp::BuildDataHeader(tcp::DataKind::WRITE_PAYLOAD, 0, kOpId, payload.size());
  std::vector<uint8_t> initial(header.begin(), header.end());
  initial.insert(initial.end(), payload.begin(), payload.end());

  auto targets = std::make_shared<PeerRecvTargets>();
  DataConnectionWorker worker(sockets[0], "peer", targets, std::move(initial));
  worker.Start();

  bool awaiting = false;
  for (int i = 0; i < 100 && !awaiting; ++i) {
    std::deque<WorkerEvent> events;
    worker.DrainEvents(events);
    for (const auto& event : events)
      awaiting = awaiting || event.type == WorkerEventType::AWAIT_TARGET;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(awaiting);

  std::array<uint8_t, 32> destination{};
  WorkerRecvTarget target;
  target.totalLen = destination.size();
  target.cpuBase = destination.data();
  target.segs = {{0, destination.size()}};
  {
    std::lock_guard<std::mutex> lock(targets->mu);
    targets->entries.emplace(RecvTargetKey{tcp::DataKind::WRITE_PAYLOAD, kOpId}, target);
  }
  worker.WakeWorker();

  bool received = false;
  for (int i = 0; i < 100 && !received; ++i) {
    std::deque<WorkerEvent> events;
    worker.DrainEvents(events);
    for (const auto& event : events)
      received = received || event.type == WorkerEventType::RECV_DONE;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(received);
  EXPECT_EQ(payload, destination);

  worker.Stop();
  close(sockets[0]);
  close(sockets[1]);
}

TEST(TcpTransportProtocol, DuplicateHelloClosesPeer) {
  IOEngineConfig engineConfig{"127.0.0.1", 0};
  TcpBackendConfig tcpConfig(1 << 20, 1 << 20, 1000, true, 30, 10, 3, true, 1, 1);
  TcpTransport transport("hello_server", engineConfig, tcpConfig);
  transport.Start();
  auto port = transport.GetListenPort();
  ASSERT_TRUE(port.has_value());

  int fd = ConnectLoopback(*port);
  ASSERT_GE(fd, 0);

  auto hello = tcp::BuildHello(tcp::Channel::CTRL, "duplicate_peer");
  std::vector<uint8_t> duplicate = hello;
  duplicate.insert(duplicate.end(), hello.begin(), hello.end());
  ASSERT_TRUE(SendAll(fd, duplicate.data(), duplicate.size()));
  EXPECT_TRUE(WaitForSocketClose(fd, 2000));

  close(fd);
  transport.Shutdown();
}

TEST(TcpTransportProtocol, ReadResponseReusesAvailableWorkerForAllDeclaredLanes) {
  IOEngineConfig engineConfig{"127.0.0.1", 0};
  TcpBackendConfig tcpConfig(1 << 20, 1 << 20, 2000, true, 30, 10, 3, true, 1, 1);
  TcpTransport transport("read_server", engineConfig, tcpConfig);
  std::array<uint8_t, 1003> source{};
  for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i * 7);
  MemoryDesc memory;
  memory.engineKey = "read_server";
  memory.id = 42;
  memory.data = reinterpret_cast<uintptr_t>(source.data());
  memory.size = source.size();
  memory.loc = MemoryLocationType::CPU;
  transport.RegisterMemory(memory);
  transport.Start();
  auto port = transport.GetListenPort();
  ASSERT_TRUE(port.has_value());

  int ctrlFd = ConnectLoopback(*port);
  ASSERT_GE(ctrlFd, 0);
  auto ctrlHello = tcp::BuildHello(tcp::Channel::CTRL, "raw_reader");
  ASSERT_TRUE(SendAll(ctrlFd, ctrlHello.data(), ctrlHello.size()));
  tcp::CtrlHeaderView ctrlHeader;
  std::vector<uint8_t> ctrlBody;
  ASSERT_TRUE(RecvCtrlFrame(ctrlFd, &ctrlHeader, &ctrlBody));
  ASSERT_EQ(ctrlHeader.type, tcp::CtrlMsgType::HELLO);

  constexpr uint8_t kDeclaredLanes = 4;
  constexpr TransferUniqueId kOpId = 99;
  auto request = tcp::BuildLinearReq(tcp::CtrlMsgType::READ_REQ, kOpId, memory.id, 0, source.size(),
                                     kDeclaredLanes);
  // Exercise the CTRL-before-DATA handoff window: the server must queue this
  // request, then reuse the first available worker for all declared lanes.
  ASSERT_TRUE(SendAll(ctrlFd, request.data(), request.size()));

  int dataFd = ConnectLoopback(*port);
  ASSERT_GE(dataFd, 0);
  auto dataHello = tcp::BuildHello(tcp::Channel::DATA, "raw_reader");
  ASSERT_TRUE(SendAll(dataFd, dataHello.data(), dataHello.size()));
  ASSERT_TRUE(RecvCtrlFrame(dataFd, &ctrlHeader, &ctrlBody));
  ASSERT_EQ(ctrlHeader.type, tcp::CtrlMsgType::HELLO);

  std::array<uint8_t, 1003> received{};
  uint16_t lanesSeen = 0;
  for (uint8_t i = 0; i < kDeclaredLanes; ++i) {
    std::array<uint8_t, tcp::kDataHeaderSize> headerBytes{};
    ASSERT_TRUE(RecvExactWithTimeout(dataFd, headerBytes.data(), headerBytes.size()));
    tcp::ParsedDataHeader header;
    ASSERT_EQ(tcp::TryParseDataHeader(headerBytes.data(), headerBytes.size(), &header),
              tcp::ParseError::Ok);
    ASSERT_EQ(header.kind, tcp::DataKind::READ_RESPONSE);
    ASSERT_EQ(header.opId, kOpId);
    ASSERT_LT(header.lane, kDeclaredLanes);
    LaneSpan span = ComputeLaneSpan(source.size(), kDeclaredLanes, header.lane);
    ASSERT_EQ(header.payloadLen, span.len);
    ASSERT_TRUE(RecvExactWithTimeout(dataFd, received.data() + span.off, span.len));
    lanesSeen |= uint16_t(1U << header.lane);
  }
  EXPECT_EQ(lanesSeen, tcp::LanesAllMask(kDeclaredLanes));
  EXPECT_EQ(source, received);

  ASSERT_TRUE(RecvCtrlFrame(ctrlFd, &ctrlHeader, &ctrlBody));
  EXPECT_EQ(ctrlHeader.type, tcp::CtrlMsgType::COMPLETION);
  tcp::WireReader completion{ctrlBody.data(), ctrlBody.size()};
  uint64_t completionOp = 0;
  uint32_t completionCode = 0;
  uint32_t messageLen = 0;
  ASSERT_TRUE(completion.u64(&completionOp));
  ASSERT_TRUE(completion.u32(&completionCode));
  ASSERT_TRUE(completion.u32(&messageLen));
  EXPECT_EQ(completionOp, kOpId);
  EXPECT_EQ(completionCode, static_cast<uint32_t>(StatusCode::SUCCESS));

  close(dataFd);
  close(ctrlFd);
  transport.Shutdown();
}

TEST(TcpTransportLifecycle, BindFailureThrowsWithoutLeavingPartialState) {
  int occupied = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(occupied, 0);
  int reuse = 1;
  ASSERT_EQ(setsockopt(occupied, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)), 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  ASSERT_EQ(bind(occupied, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
  ASSERT_EQ(listen(occupied, 1), 0);
  socklen_t addressLen = sizeof(address);
  ASSERT_EQ(getsockname(occupied, reinterpret_cast<sockaddr*>(&address), &addressLen), 0);

  IOEngineConfig engineConfig{"127.0.0.1", ntohs(address.sin_port)};
  TcpBackendConfig tcpConfig(1 << 20, 1 << 20, 1000, true, 30, 10, 3, true, 1, 1);
  TcpTransport transport("bind_failure", engineConfig, tcpConfig);
  EXPECT_THROW(transport.Start(), std::runtime_error);
  EXPECT_FALSE(transport.GetListenPort().has_value());
  close(occupied);
}

}  // namespace
}  // namespace mori::io
