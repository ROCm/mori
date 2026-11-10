// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/engine/engine_client.h"
#include "umbp/engine/engine_dispatch.h"
#include "umbp/engine/engine_executor.h"
#include "umbp/engine/engine_session.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"
#include "umbp_engine.pb.h"

namespace mori::umbp {
namespace {

class RunningServer {
 public:
  explicit RunningServer(standalone::StandaloneServer* server)
      : server_(server), thread_([this] { server_->Run(); }) {}
  ~RunningServer() { Stop(); }
  void Stop() {
    if (server_ == nullptr) return;
    server_->Shutdown();
    if (thread_.joinable()) thread_.join();
    server_ = nullptr;
  }

 private:
  standalone::StandaloneServer* server_;
  std::thread thread_;
};

class FakeEngineControl final : public engine::EngineControl {
 public:
  bool RegisterEngine(EngineRegistration registration) override {
    registration_ = std::move(registration);
    return true;
  }
  std::optional<uint32_t> BackendId(engine::RankId rank) const override {
    return rank < 8 ? std::optional<uint32_t>{9 + rank} : std::nullopt;
  }
  std::vector<engine::SharedDramRegion> ExportSharedDram(engine::PoolEpoch) override { return {}; }
  PeerPool* Pool() const override { return nullptr; }
  engine::ReadPlan Lookup(const std::vector<std::string>&, std::string, std::string,
                          uint32_t) override {
    return {};
  }
  engine::PreparedLoad PrepareLoad(const engine::ReadPlan&, std::string,
                                   const std::vector<engine::AllocationHandle>&, uint32_t,
                                   const ReaderIdentity&, engine::OperationId) override {
    return {};
  }
  engine::CancelLoadResult CancelLoad(uint64_t, engine::OperationId) override {
    return engine::CancelLoadResult::kRejected;
  }
  bool CompleteLoad(uint64_t, engine::LoadOutcome, engine::OperationId) override { return false; }
  void SessionFailed(engine::RankId) override { ++failed_sessions; }

  EngineRegistration registration_;
  int failed_sessions = 0;
};

TEST(EngineControlTest, FramedMessageCarriesSeveralDescriptorsBothWays) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);

  HostMemAllocator allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;
  auto first = allocator.Alloc(4096, options);
  auto second = allocator.Alloc(4096, options);
  ASSERT_TRUE(first.valid());
  ASSERT_TRUE(second.valid());
  auto first_shm = HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(first.ptr),
                                                         first.mapped_size);
  auto second_shm = HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(second.ptr),
                                                          second.mapped_size);
  ASSERT_TRUE(first_shm.has_value());
  ASSERT_TRUE(second_shm.has_value());
  static_cast<uint8_t*>(first.ptr)[11] = 0x4a;
  static_cast<uint8_t*>(second.ptr)[13] = 0xb7;

  std::string error;
  ASSERT_TRUE(standalone::SendFramedMessage(sockets[0], "engine-frame",
                                            {first_shm->fd, second_shm->fd}, &error))
      << error;
  std::string payload;
  std::vector<int> received;
  ASSERT_TRUE(standalone::RecvFramedMessage(sockets[1], &payload, &received, 4096, 4, &error))
      << error;
  ASSERT_EQ(payload, "engine-frame");
  ASSERT_EQ(received.size(), 2u);
  void* first_mirror = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, received[0], 0);
  void* second_mirror = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, received[1], 0);
  ASSERT_NE(first_mirror, MAP_FAILED);
  ASSERT_NE(second_mirror, MAP_FAILED);
  EXPECT_EQ(static_cast<uint8_t*>(first_mirror)[11], 0x4a);
  EXPECT_EQ(static_cast<uint8_t*>(second_mirror)[13], 0xb7);

  munmap(first_mirror, 4096);
  munmap(second_mirror, 4096);
  for (int fd : received) close(fd);
  close(sockets[0]);
  close(sockets[1]);
  allocator.Free(first);
  allocator.Free(second);
}

TEST(EngineControlTest, PeerVmmDescriptorsArePartitionedBelowScmRightsLimit) {
  std::vector<engine::PeerVmmRegion> pending;
  for (size_t i = 0; i < 364; ++i) {
    engine::PeerVmmRegion region;
    region.executor_id = i < 182 ? "source-a" : "source-b";
    region.region_id = "region-" + std::to_string(i);
    region.fd = static_cast<int>(i + 10);
    region.mapped_size = 4096;
    region.region_size = 4096;
    region.mapping_id = i + 1;
    pending.push_back(std::move(region));
  }
  auto first = engine::TakePeerRegionBatchForTest(&pending);
  auto second = engine::TakePeerRegionBatchForTest(&pending);
  EXPECT_LE(first.size(), 192u);
  EXPECT_LE(second.size(), 192u);
  EXPECT_EQ(first.size() + second.size(), 364u);
  EXPECT_TRUE(pending.empty());
}

TEST(EngineControlTest, PublishReceiptStatusSurvivesEventOverflowUntilAcknowledged) {
  engine::EngineClient client({"unix:///tmp/unused-engine.sock", "receipt-test", 10, 100});
  client.SetEventQueueLimitForTest(2);
  for (uint64_t ticket = 1; ticket <= 5000; ++ticket) {
    client.InjectPublishReceiptForTest(ticket, engine::PublishReceiptState::kAccepted);
  }
  EXPECT_TRUE(client.ConsumeEventOverflow());
  auto oldest = client.PublishStatus(1);
  auto newest = client.PublishStatus(5000);
  ASSERT_TRUE(oldest.has_value());
  ASSERT_TRUE(newest.has_value());
  EXPECT_EQ(oldest->state, engine::PublishReceiptState::kAccepted);
  EXPECT_EQ(newest->state, engine::PublishReceiptState::kAccepted);
  EXPECT_TRUE(client.AcknowledgePublishReceipt(1));
  EXPECT_FALSE(client.PublishStatus(1).has_value());
  EXPECT_TRUE(client.AcknowledgePublishReceipt(5000));
}

TEST(EngineControlTest, PublishReceiptRetentionIsBoundedAndReportsOverflow) {
  engine::EngineClient client({"unix:///tmp/unused-engine.sock", "receipt-limit-test", 10, 100});
  client.SetPublishReceiptLimitForTest(2);
  client.InjectPublishReceiptForTest(1, engine::PublishReceiptState::kAccepted);
  client.InjectPublishReceiptForTest(2, engine::PublishReceiptState::kRejected);
  client.InjectPublishReceiptForTest(3, engine::PublishReceiptState::kAccepted);
  EXPECT_TRUE(client.ConsumeEventOverflow());
  EXPECT_TRUE(client.PublishStatus(1).has_value());
  EXPECT_TRUE(client.PublishStatus(2).has_value());
  EXPECT_FALSE(client.PublishStatus(3).has_value());
  EXPECT_TRUE(client.AcknowledgePublishReceipt(1));
  client.InjectPublishReceiptForTest(3, engine::PublishReceiptState::kAccepted);
  EXPECT_TRUE(client.PublishStatus(3).has_value());
}

TEST(EngineControlTest, SeparateNodeProcessF3RoundTripsSharedDramBytes) {
  const std::string address =
      "unix:///tmp/umbp_engine_f3_process_" + std::to_string(getpid()) + ".grpc.sock";
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());
  int ready[2];
  ASSERT_EQ(pipe2(ready, O_CLOEXEC), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(ready[0]);
    UMBPConfig config;
    config.dram.capacity_bytes = 1 << 20;
    config.dram.use_hugepages = false;
    config.dram.prefault = false;
    config.page_size = 4096;
    config = WithEmbeddedDefaults(config);
    config.distributed->engine_hbm_ranks = 8;
    standalone::StandaloneServer server(config, address);
    const uint8_t started = server.Start() ? 1 : 0;
    (void)write(ready[1], &started, sizeof(started));
    close(ready[1]);
    if (started == 0) _exit(2);
    server.Run();
    _exit(0);
  }
  close(ready[1]);
  struct ChildGuard {
    pid_t pid;
    ~ChildGuard() {
      if (pid <= 0) return;
      (void)kill(pid, SIGTERM);
      (void)waitpid(pid, nullptr, 0);
    }
  } child_guard{child};
  uint8_t started = 0;
  ASSERT_EQ(read(ready[0], &started, sizeof(started)), sizeof(started));
  close(ready[0]);
  ASSERT_EQ(started, 1u);

  constexpr size_t kBytes = 4096;
  std::vector<uint8_t> storage(2 * kBytes);
  engine::EngineExecutor executor({"separate-process-f3-rank", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(storage.data(), storage.size())));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 1;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(storage.data(), storage.size()), kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  engine::EngineClient engine_client({address, "separate-process-f3-rank", 10, 1000});
  engine_client.AttachExecutor(&executor);
  engine::PoolEpoch epoch;
  auto registration = engine_client.RegisterPool(0, "separate-process-f3-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto reservation = engine_client.Reserve("separate-process-f3", 1, 0);
  auto page = engine_client.Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(reservation.id));
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint8_t>(i * 29 + 7);
  std::memcpy(storage.data() + page.front().page_id * kBytes, expected.data(), expected.size());
  const std::string key = "separate-process-f3-key";
  ASSERT_TRUE(engine_client.Publish({ResidentRecord{key, {}, page.front()}}).success);
  ASSERT_TRUE(engine_client.ReturnPages(page));

  UMBPConfig reader_config;
  UMBPStandaloneProcessConfig standalone_config;
  standalone_config.address = address;
  standalone_config.startup_timeout_ms = 5000;
  reader_config.standalone_process = standalone_config;
  auto reader = CreateUMBPClient(reader_config);
  HostMemAllocator read_allocator;
  HostBufferOptions read_options;
  read_options.backing = HostBufferBacking::kAnonymousShm;
  read_options.prefault = false;
  auto actual = read_allocator.Alloc(kBytes, read_options);
  ASSERT_TRUE(actual.valid());
  std::memset(actual.ptr, 0, kBytes);
  ASSERT_TRUE(reader->RegisterMemory(reinterpret_cast<uintptr_t>(actual.ptr), actual.mapped_size));
  bool restored = false;
  for (int attempt = 0; attempt < 500 && !restored; ++attempt) {
    restored = reader->Get(key, reinterpret_cast<uintptr_t>(actual.ptr), kBytes);
    if (!restored) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(restored);
  EXPECT_EQ(std::memcmp(actual.ptr, expected.data(), expected.size()), 0);
  std::vector<engine::RetireTicket> released;
  for (int attempt = 0; attempt < 500 && released.empty(); ++attempt) {
    released = engine_client.PollRetires();
    if (released.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(released.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({released.front().id}));
  reader->DeregisterMemory(reinterpret_cast<uintptr_t>(actual.ptr));
  reader->Close();
  read_allocator.Free(actual);
  engine_client.Close();
  executor.Close();
}

TEST(EngineControlTest, FramedReceiveSurvivesDelayInsideHeader) {
  struct Header {
    uint32_t magic = 0x554d4246;
    uint16_t version = 1;
    uint16_t fd_count = 0;
    uint32_t payload_size = 3;
  } header;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
  timeval timeout{1, 0};
  ASSERT_EQ(setsockopt(sockets[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
  std::thread sender([&] {
    ASSERT_EQ(send(sockets[0], &header, 2, MSG_NOSIGNAL), 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    ASSERT_EQ(
        send(sockets[0], reinterpret_cast<char*>(&header) + 2, sizeof(header) - 2, MSG_NOSIGNAL),
        static_cast<ssize_t>(sizeof(header) - 2));
    ASSERT_EQ(send(sockets[0], "abc", 3, MSG_NOSIGNAL), 3);
  });
  std::string payload;
  std::vector<int> fds;
  std::string error;
  EXPECT_TRUE(standalone::RecvFramedMessage(sockets[1], &payload, &fds, 1024, 4, &error)) << error;
  EXPECT_EQ(payload, "abc");
  sender.join();
  close(sockets[0]);
  close(sockets[1]);
}

TEST(EngineControlTest, TruncatedHeaderClosesDeliveredDescriptor) {
  struct Header {
    uint32_t magic = 0x554d4246;
    uint16_t version = 1;
    uint16_t fd_count = 1;
    uint32_t payload_size = 0;
  } header;
  int sockets[2];
  int pipe_fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
  ASSERT_EQ(pipe(pipe_fds), 0);
  const auto fd_count = [] {
    return static_cast<size_t>(std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                                             std::filesystem::directory_iterator{}));
  };
  const size_t before = fd_count();
  iovec iov{&header, 2};
  char control[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &iov;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  auto* cmsg = CMSG_FIRSTHDR(&message);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &pipe_fds[0], sizeof(int));
  ASSERT_EQ(sendmsg(sockets[0], &message, MSG_NOSIGNAL), 2);
  shutdown(sockets[0], SHUT_WR);
  std::string payload;
  std::vector<int> received;
  std::string error;
  EXPECT_FALSE(standalone::RecvFramedMessage(sockets[1], &payload, &received, 1024, 4, &error));
  EXPECT_EQ(fd_count(), before);
  close(pipe_fds[0]);
  close(pipe_fds[1]);
  close(sockets[0]);
  close(sockets[1]);
}

TEST(EngineControlTest, TruncatedControlMessageClosesDeliveredDescriptors) {
  int sockets[2];
  int first_pipe[2];
  int second_pipe[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
  ASSERT_EQ(pipe(first_pipe), 0);
  ASSERT_EQ(pipe(second_pipe), 0);
  const auto fd_count = [] {
    return static_cast<size_t>(std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                                             std::filesystem::directory_iterator{}));
  };
  const size_t before = fd_count();
  std::string error;
  ASSERT_TRUE(
      standalone::SendFramedMessage(sockets[0], "x", {first_pipe[0], second_pipe[0]}, &error));
  std::string payload;
  std::vector<int> received;
  EXPECT_FALSE(standalone::RecvFramedMessage(sockets[1], &payload, &received, 1024,
                                             /*max_fds=*/1, &error));
  EXPECT_EQ(fd_count(), before);
  close(first_pipe[0]);
  close(first_pipe[1]);
  close(second_pipe[0]);
  close(second_pipe[1]);
  close(sockets[0]);
  close(sockets[1]);
}

TEST(EngineControlTest, SessionEnforcesOrderAndHeartbeatDeadline) {
  using Clock = engine::EngineSession::Clock;
  auto now = Clock::time_point{};
  FakeEngineControl control;
  {
    engine::EngineSession session(&control, [&] { return now; });
    ::umbp::engine::EngineFrame registration;
    registration.set_version(1);
    registration.mutable_op()->set_session_id(std::string(16, 's'));
    registration.mutable_op()->set_seq(1);
    registration.mutable_epoch()->set_boot_id(std::string(16, 'e'));
    auto* body = registration.mutable_register_engine();
    body->set_rank_id(0);
    body->set_pool_id("pool");
    body->set_executor_id("ordered-executor");
    body->mutable_epoch()->set_boot_id(std::string(16, 'e'));
    body->set_heartbeat_ms(5);
    body->set_heartbeat_timeout_ms(20);
    auto* geometry = body->mutable_geometry();
    geometry->set_page_tokens(1);
    geometry->set_num_pages(2);
    geometry->set_object_bytes(32);
    auto* region = geometry->add_regions();
    region->set_region_id("kv");
    region->mutable_buffer()->set_address(1);
    region->mutable_buffer()->set_size(96);
    region->mutable_buffer()->set_location(
        static_cast<uint32_t>(mori::io::MemoryLocationType::CPU));
    region->mutable_buffer()->set_executor_id("ordered-executor");
    region->mutable_buffer()->set_region_id("kv");
    region->set_row_bytes(32);
    region->set_rows_per_page(1);

    ::umbp::engine::EngineFrame response;
    std::vector<int> response_fds;
    std::string error;
    ASSERT_TRUE(session.Handle(registration, {}, &response, &response_fds, &error)) << error;
    ASSERT_TRUE(response.has_registered());
    EXPECT_FALSE(response.registered().ok());
    EXPECT_FALSE(session.Expired());
    now += std::chrono::milliseconds(21);
    EXPECT_TRUE(session.Expired());

    ::umbp::engine::EngineFrame skipped;
    skipped.set_version(1);
    skipped.mutable_op()->set_session_id(std::string(16, 's'));
    skipped.mutable_op()->set_seq(3);
    skipped.mutable_epoch()->set_boot_id(std::string(16, 'e'));
    skipped.mutable_heartbeat();
    EXPECT_FALSE(session.Handle(skipped, {}, &response, &response_fds, &error));
  }
  EXPECT_EQ(control.failed_sessions, 1);
}

TEST(EngineControlTest, RegistrationFailureClosesEveryDuplicatedVmmDescriptor) {
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  const auto fd_count = [] {
    return static_cast<size_t>(std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                                             std::filesystem::directory_iterator{}));
  };
  const size_t before = fd_count();
  FakeEngineControl control;
  engine::EngineSession session(&control);
  ::umbp::engine::EngineFrame registration;
  registration.set_version(1);
  registration.mutable_op()->set_session_id(std::string(16, 'f'));
  registration.mutable_op()->set_seq(1);
  registration.mutable_epoch()->set_boot_id(std::string(16, 'e'));
  auto* body = registration.mutable_register_engine();
  body->set_rank_id(9);  // fails only after the VMM fd has been dup'd
  body->set_pool_id("fd-cleanup-pool");
  body->set_executor_id("fd-cleanup-executor");
  body->mutable_epoch()->set_boot_id(std::string(16, 'e'));
  auto* geometry = body->mutable_geometry();
  geometry->set_page_tokens(1);
  geometry->set_num_pages(1);
  geometry->set_object_bytes(32);
  auto* page_region = geometry->add_regions();
  page_region->set_region_id("kv");
  page_region->mutable_buffer()->set_address(1);
  page_region->mutable_buffer()->set_size(64);
  page_region->mutable_buffer()->set_location(
      static_cast<uint32_t>(mori::io::MemoryLocationType::CPU));
  page_region->set_row_bytes(32);
  page_region->set_rows_per_page(1);
  auto* vmm = body->add_vmm_regions();
  vmm->set_region_id("kv");
  vmm->set_fd_index(0);
  vmm->set_mapped_size(4096);
  vmm->set_region_size(4096);
  vmm->set_mapping_id(7);
  vmm->set_handle_kind(static_cast<uint32_t>(engine::VmmHandleKind::kPosixFd));

  ::umbp::engine::EngineFrame response;
  std::vector<int> response_fds;
  std::string error;
  EXPECT_FALSE(session.Handle(registration, {pipe_fds[0]}, &response, &response_fds, &error));
  EXPECT_EQ(fd_count(), before) << "failed registration leaked its retained VMM fd";
  close(pipe_fds[0]);
  close(pipe_fds[1]);
}

TEST(EngineControlTest, UnregisteredConnectionAlsoExpires) {
  using Clock = engine::EngineSession::Clock;
  auto now = Clock::time_point{};
  FakeEngineControl control;
  engine::EngineSession session(&control, [&] { return now; });
  EXPECT_FALSE(session.Expired());
  now += std::chrono::seconds(6);
  EXPECT_TRUE(session.Expired());
}

TEST(EngineControlTest, ResetKeepsLocalQuarantineWhenNodeStillApprovesThePage) {
  const std::string suffix = std::to_string(getpid()) + "_local_quarantine";
  const std::string address = "unix:///tmp/umbp_engine_control_" + suffix + ".grpc.sock";
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());

  UMBPConfig server_config;
  server_config.dram.capacity_bytes = 1 << 20;
  server_config.dram.use_hugepages = false;
  server_config.dram.prefault = false;
  server_config.page_size = 4096;
  server_config = WithEmbeddedDefaults(server_config);
  server_config.distributed->engine_hbm_ranks = 8;
  standalone::StandaloneServer server(server_config, address);
  ASSERT_TRUE(server.Start());
  RunningServer running(&server);

  constexpr size_t kBytes = 32;
  std::vector<uint8_t> storage(2 * kBytes);
  engine::EngineExecutor executor({"local-quarantine-executor", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(storage.data(), storage.size())));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 1;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(storage.data(), storage.size()), kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  engine::EngineClient client({address, "local-quarantine-executor", 10, 1000});
  client.AttachExecutor(&executor);
  engine::PoolEpoch epoch;
  auto registration = client.RegisterPool(0, "local-quarantine-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  auto reservation = client.Reserve("local-only", 1, 0);
  ASSERT_TRUE(reservation.success);
  auto page = client.Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(page.size(), 1u);
  ASSERT_TRUE(client.QuarantineLocalForTest(page.front()));

  auto lease = client.QuiesceReset();
  ASSERT_TRUE(lease.ready);
  ASSERT_TRUE(client.FinishReset(lease, true));
  EXPECT_EQ(client.Accounting().quarantined, 1u);
  EXPECT_EQ(client.Accounting().cached, 0u)
      << "the node-approved DELEGATED page must not overwrite local quarantine";
  EXPECT_EQ(client.Reserve("must-stay-excluded", 1, 0).shortfall, 1u);
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  bool executing = false;
  bool release_execution = false;
  executor.SetBeforeExecuteHookForTest([&] {
    std::unique_lock<std::mutex> lock(completion_mutex);
    executing = true;
    completion_cv.notify_all();
    completion_cv.wait(lock, [&] { return release_execution; });
  });
  TransferPlan late_completion;
  late_completion.src = TransferRef::HostBytes(storage.data(), kBytes);
  late_completion.dst = TransferRef::HostBytes(storage.data() + kBytes, kBytes);
  late_completion.src_offsets = {0};
  late_completion.dst_offsets = {0};
  late_completion.sizes = {kBytes};
  late_completion.tags = {77};
  auto timed_out = engine::EngineDispatchRegistry::Global().Submit("local-quarantine-executor",
                                                                   {late_completion});
  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    ASSERT_TRUE(completion_cv.wait_for(lock, std::chrono::seconds(1), [&] { return executing; }));
  }
  std::vector<TransferFailure> timeout_failures;
  EXPECT_FALSE(timed_out->WaitFor(std::chrono::milliseconds(1), &timeout_failures));
  ASSERT_EQ(timeout_failures.size(), 1u);
  EXPECT_EQ(timeout_failures[0].tags, std::vector<size_t>({77}));
  {
    std::lock_guard<std::mutex> lock(completion_mutex);
    release_execution = true;
  }
  completion_cv.notify_all();
  for (int attempt = 0; attempt < 100 && client.healthy(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(client.healthy()) << "a tombstoned late completion must be ACKed and ignored";
  client.Close();
  executor.Close();
  running.Stop();
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());
}

TEST(EngineControlTest, ResetWaitsForQuarantineReportsAndUnknownReadinessPersists) {
  const std::string suffix = std::to_string(getpid()) + "_reset_quarantine";
  const std::string address = "unix:///tmp/umbp_engine_control_" + suffix + ".grpc.sock";
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());

  UMBPConfig server_config;
  server_config.dram.capacity_bytes = 1 << 20;
  server_config.dram.use_hugepages = false;
  server_config.dram.prefault = false;
  server_config.page_size = 4096;
  server_config = WithEmbeddedDefaults(server_config);
  server_config.distributed->engine_hbm_ranks = 8;
  standalone::StandaloneServer server(server_config, address);
  ASSERT_TRUE(server.Start());
  RunningServer running(&server);

  constexpr size_t kBytes = 32;
  std::vector<uint8_t> storage(4 * kBytes);
  engine::EngineExecutor executor({"reset-quarantine-executor", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(storage.data(), storage.size())));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 3;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(storage.data(), storage.size()), kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  engine::EngineClient client({address, "reset-quarantine-executor", 10, 1000});
  client.AttachExecutor(&executor);
  engine::PoolEpoch epoch;
  ASSERT_TRUE(client.RegisterPool(0, "reset-quarantine-pool", epoch, geometry).success);

  std::mutex mutex;
  std::condition_variable cv;
  bool report_entered = false;
  bool allow_report = false;
  client.SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (!request.has_quarantine()) return;
    std::unique_lock<std::mutex> lock(mutex);
    report_entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return allow_report; });
  });

  auto io_reservation = client.Reserve("unknown-io", 1, 0);
  auto io_page = client.Take(io_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(client.Refund(io_reservation.id));
  auto io_hold = client.HoldIO(io_page);
  ASSERT_TRUE(io_hold.has_value());
  ASSERT_TRUE(client.CompleteIO(*io_hold, engine::CompletionProof::kUnknown));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return report_entered; }));
  }
  auto quiesce = std::async(std::launch::async, [&] { return client.QuiesceReset(); });
  EXPECT_EQ(quiesce.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
      << "reset seal must wait for the in-flight quarantine report";
  {
    std::lock_guard<std::mutex> lock(mutex);
    allow_report = true;
  }
  cv.notify_all();
  auto first_lease = quiesce.get();
  ASSERT_TRUE(first_lease.ready);
  ASSERT_TRUE(client.FinishReset(first_lease, false));

  report_entered = false;
  allow_report = false;
  auto publish_reservation = client.Reserve("unknown-publish", 1, 0);
  auto publish_page = client.Take(publish_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(client.Refund(publish_reservation.id));
  const uint64_t ticket =
      client.QueuePublish({ResidentRecord{"unknown-publish", {}, publish_page.front()}}, false);
  ASSERT_NE(ticket, 0u);
  auto second_lease = client.QuiesceReset();
  ASSERT_TRUE(second_lease.ready);
  ASSERT_TRUE(client.MarkPublishReady(ticket, engine::CompletionProof::kUnknown));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return report_entered; }));
  }
  auto commit =
      std::async(std::launch::async, [&] { return client.FinishReset(second_lease, true); });
  EXPECT_EQ(commit.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
      << "reset commit must not overtake an Unknown publication quarantine";
  {
    std::lock_guard<std::mutex> lock(mutex);
    allow_report = true;
  }
  cv.notify_all();
  ASSERT_TRUE(commit.get());
  client.SetBeforeCallSendHookForTest({});
  EXPECT_EQ(client.Accounting().quarantined, 2u);
  EXPECT_EQ(client.Accounting().cached, 1u);
  EXPECT_EQ(second_lease.approved.size(), 1u);
  EXPECT_EQ(second_lease.new_epoch.reset_seq, second_lease.old_epoch.reset_seq + 1);
  EXPECT_TRUE(client.healthy());

  auto window_reservation = client.Reserve("reset-finish-window", 1, 0);
  auto window_page = client.Take(window_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(client.Refund(window_reservation.id));
  const uint64_t window_ticket =
      client.QueuePublish({ResidentRecord{"reset-finish-window", {}, window_page.front()}}, false);
  ASSERT_NE(window_ticket, 0u);
  auto window_lease = client.QuiesceReset();
  ASSERT_TRUE(window_lease.ready);
  std::mutex finish_mutex;
  std::condition_variable finish_cv;
  bool finish_sending = false;
  bool allow_finish = false;
  client.SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (!request.has_reset_finish()) return;
    std::unique_lock<std::mutex> lock(finish_mutex);
    finish_sending = true;
    finish_cv.notify_all();
    finish_cv.wait(lock, [&] { return allow_finish; });
  });
  auto finish =
      std::async(std::launch::async, [&] { return client.FinishReset(window_lease, true); });
  {
    std::unique_lock<std::mutex> lock(finish_mutex);
    ASSERT_TRUE(finish_cv.wait_for(lock, std::chrono::seconds(1), [&] { return finish_sending; }));
  }
  auto late_unknown = std::async(std::launch::async, [&] {
    return client.MarkPublishReady(window_ticket, engine::CompletionProof::kUnknown);
  });
  EXPECT_EQ(late_unknown.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
      << "reset finish must hold the state gate through commit";
  {
    std::lock_guard<std::mutex> lock(finish_mutex);
    allow_finish = true;
  }
  finish_cv.notify_all();
  ASSERT_TRUE(finish.get());
  EXPECT_FALSE(late_unknown.get())
      << "the committed reset cancels the old-epoch publication before it can report quarantine";
  client.SetBeforeCallSendHookForTest({});
  EXPECT_TRUE(client.healthy());

  client.Close();
  executor.Close();
  running.Stop();
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());
}

TEST(EngineControlTest, StandaloneRetirePreservesLastReplicaInDram) {
  const std::string suffix = std::to_string(getpid());
  const std::string address = "unix:///tmp/umbp_engine_control_" + suffix + ".grpc.sock";
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());

  UMBPConfig server_config;
  server_config.dram.capacity_bytes = 1 << 20;
  server_config.dram.use_hugepages = false;
  server_config.dram.prefault = false;
  server_config.page_size = 4096;
  server_config = WithEmbeddedDefaults(server_config);
  server_config.distributed->engine_hbm_ranks = 8;

  standalone::StandaloneServer server(server_config, address);
  ASSERT_TRUE(server.Start());
  RunningServer running(&server);

  constexpr size_t kObjectBytes = 32;
  std::vector<uint8_t> hbm((2 + 1) * kObjectBytes, 0);

  engine::EngineExecutor executor({"rank-0-executor", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(hbm.data(), hbm.size(), mori::io::MemoryLocationType::CPU, -1)));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 2;
  geometry.object_bytes = kObjectBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(hbm.data(), hbm.size(), mori::io::MemoryLocationType::CPU, -1),
      kObjectBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  ASSERT_TRUE(geometry.Validate());

  engine::PoolEpoch epoch;
  epoch.boot_id[0] = 0x51;
  engine::EngineClient engine_client(
      {address, "rank-0-executor", /*heartbeat_ms=*/10, /*heartbeat_timeout_ms=*/1000});
  engine_client.AttachExecutor(&executor);
  std::shared_ptr<engine::DispatchTicket> early_ticket;
  engine_client.SetAfterReceiverStartedHookForTest([&] {
    early_ticket = engine::EngineDispatchRegistry::Global().Submit("rank-0-executor", {});
  });
  auto registration = engine_client.RegisterPool(0, "engine-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  ASSERT_NE(early_ticket, nullptr);
  std::vector<TransferFailure> early_failures;
  ASSERT_TRUE(early_ticket->WaitFor(std::chrono::seconds(1), &early_failures));
  EXPECT_TRUE(early_failures.empty())
      << "a pushed work frame before initial delegation must not be mistaken for the reply";
  engine_client.SetAfterReceiverStartedHookForTest({});
  executor.SetExecuteOverrideForTest(
      [](const std::vector<TransferPlan>&, std::vector<TransferFailure>*) { return false; });
  TransferPlan fallback_plan;
  fallback_plan.src = TransferRef::HostBytes(hbm.data(), kObjectBytes);
  fallback_plan.dst = TransferRef::HostBytes(hbm.data() + kObjectBytes, kObjectBytes);
  fallback_plan.src_offsets = {0};
  fallback_plan.dst_offsets = {0};
  fallback_plan.sizes = {kObjectBytes};
  fallback_plan.tags = {41};
  auto fallback_ticket =
      engine::EngineDispatchRegistry::Global().Submit("rank-0-executor", {fallback_plan});
  std::vector<TransferFailure> fallback_failures;
  ASSERT_TRUE(fallback_ticket->WaitFor(std::chrono::seconds(1), &fallback_failures));
  ASSERT_EQ(fallback_failures.size(), 1u);
  EXPECT_EQ(fallback_failures[0].tags, std::vector<size_t>({41}));
  EXPECT_FALSE(fallback_failures[0].quiesced)
      << "false without executor details must be converted to Unknown";
  EXPECT_TRUE(fallback_failures[0].submitted);
  executor.SetExecuteOverrideForTest({});
  EXPECT_NE(registration.epoch.boot_id, epoch.boot_id) << "node must own pool boot identity";
  engine::EngineClient duplicate(
      {address, "rank-0-duplicate", /*heartbeat_ms=*/10, /*heartbeat_timeout_ms=*/1000});
  auto duplicate_registration = duplicate.RegisterPool(0, "engine-pool", epoch, geometry);
  EXPECT_FALSE(duplicate_registration.success);
  duplicate.Close();
  ASSERT_FALSE(registration.dram_regions.empty());
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  EXPECT_EQ(engine_client.Accounting().cached, 2u);

  const uint64_t reserve_checks = engine_client.ReservePageChecksForTest();
  auto held_reservation = engine_client.Reserve("held-unpublished", 1, 0);
  ASSERT_TRUE(held_reservation.success);
  EXPECT_EQ(engine_client.ReservePageChecksForTest() - reserve_checks, 1u)
      << "Reserve must pop the CACHED free list, not scan the full page table";
  auto held_pages = engine_client.Take(held_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(held_pages.size(), 1u);
  auto local_io = engine_client.HoldIO(held_pages);
  ASSERT_TRUE(local_io.has_value());
  ASSERT_TRUE(engine_client.ReturnPages(held_pages));
  EXPECT_EQ(engine_client.Accounting().pending_free, 1u);
  EXPECT_FALSE(engine_client.QuiesceReset().ready)
      << "unpublished local I/O holds must also make reset Busy";
  ASSERT_TRUE(engine_client.CompleteIO(*local_io, engine::CompletionProof::kCompleted));
  EXPECT_EQ(engine_client.Accounting().cached, 2u);
  ASSERT_TRUE(engine_client.Refund(held_reservation.id));

  auto guard_reservation = engine_client.Reserve("guarded-unpublished", 1, 0);
  ASSERT_TRUE(guard_reservation.success);
  auto guarded = engine_client.Take(guard_reservation.id, engine::ReservationPurpose::kLoad, 1);
  auto guard = engine_client.HoldReuseGuard(guarded);
  ASSERT_TRUE(guard.has_value());
  ASSERT_TRUE(engine_client.ReturnPages(guarded));
  EXPECT_EQ(engine_client.Accounting().pending_free, 1u);
  ASSERT_TRUE(engine_client.CompleteReuseGuard(*guard, engine::CompletionProof::kCompleted));
  ASSERT_TRUE(engine_client.Refund(guard_reservation.id));

  auto canceled_reservation = engine_client.Reserve("cancel-publish", 1, 0);
  ASSERT_TRUE(canceled_reservation.success);
  auto canceled = engine_client.Take(canceled_reservation.id, engine::ReservationPurpose::kLoad, 1);
  engine_client.SetEventQueueLimitForTest(1);
  const uint64_t canceled_ticket =
      engine_client.QueuePublish({ResidentRecord{"cancelled", {}, canceled.front()}}, false);
  ASSERT_NE(canceled_ticket, 0u);
  ASSERT_TRUE(engine_client.ReturnPages(canceled));
  EXPECT_FALSE(
      engine_client.MarkPublishReady(canceled_ticket, engine::CompletionProof::kCompleted));
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto receipt = engine_client.PublishStatus(canceled_ticket);
    if (receipt.has_value() && receipt->state == engine::PublishReceiptState::kCancelled) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(engine_client.PublishStatus(canceled_ticket)->state,
            engine::PublishReceiptState::kCancelled);
  EXPECT_TRUE(engine_client.ConsumeEventOverflow())
      << "event queue overflow must be observable instead of silently dropping history";
  engine_client.SetEventQueueLimitForTest(4096);
  (void)engine_client.PollPublishReceipts();
  ASSERT_TRUE(engine_client.Refund(canceled_reservation.id));
  EXPECT_EQ(engine_client.Accounting().available, 2u);

  auto abort_reservation = engine_client.Reserve("abort-resumes-publish", 1, 0);
  ASSERT_TRUE(abort_reservation.success);
  auto abort_page = engine_client.Take(abort_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(abort_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(abort_reservation.id));
  const uint64_t abort_ticket = engine_client.QueuePublish(
      {ResidentRecord{"abort-resumes-publish", {}, abort_page.front()}}, false);
  ASSERT_NE(abort_ticket, 0u);
  auto abort_lease = engine_client.QuiesceReset();
  ASSERT_TRUE(abort_lease.ready);
  ASSERT_EQ(abort_lease.approved.size(), 2u);
  ASSERT_TRUE(engine_client.MarkPublishReady(abort_ticket, engine::CompletionProof::kCompleted));
  ASSERT_TRUE(engine_client.FinishReset(abort_lease, false));
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto receipt = engine_client.PublishStatus(abort_ticket);
    if (receipt.has_value() && receipt->state == engine::PublishReceiptState::kAccepted) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto abort_receipt = engine_client.PublishStatus(abort_ticket);
  ASSERT_TRUE(abort_receipt.has_value());
  EXPECT_EQ(abort_receipt->state, engine::PublishReceiptState::kAccepted)
      << "aborting a Ready reset must resume the unsent publication";
  ASSERT_TRUE(engine_client.ReturnPages(abort_page));
  std::vector<engine::RetireTicket> abort_released;
  for (int attempt = 0; attempt < 300 && abort_released.empty(); ++attempt) {
    abort_released = engine_client.PollRetires();
    if (abort_released.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(abort_released.size(), 1u);
  for (int attempt = 0; attempt < 300 && engine_client.Accounting().cached != 2; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(engine_client.Accounting().cached, 2u)
      << "the background Released ACK must complete before the next full-pool reservation";

  auto batch_reservation = engine_client.Reserve("partial-publish-return", 2, 0);
  ASSERT_TRUE(batch_reservation.success);
  auto batch_pages = engine_client.Take(batch_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(batch_pages.size(), 2u);
  const uint64_t batch_ticket = engine_client.QueuePublish(
      {{"partial-a", {}, batch_pages[0]}, {"partial-b", {}, batch_pages[1]}}, false);
  ASSERT_NE(batch_ticket, 0u);
  ASSERT_TRUE(engine_client.ReturnPages({batch_pages[0]}));
  for (int attempt = 0; attempt < 100 && engine_client.PublishTaskCountForTest() != 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(engine_client.Accounting().cached, 1u);
  EXPECT_EQ(engine_client.Accounting().in_use, 1u)
      << "canceling one member clears pending_publish on every batch member without freeing it";
  ASSERT_TRUE(engine_client.ReturnPages({batch_pages[1]}));
  ASSERT_TRUE(engine_client.Refund(batch_reservation.id));
  EXPECT_EQ(engine_client.Accounting().cached, 2u);

  auto never_reservation = engine_client.Reserve("never-submitted-publish", 1, 0);
  ASSERT_TRUE(never_reservation.success);
  auto never_page = engine_client.Take(never_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(never_page.size(), 1u);
  const uint64_t never_ticket = engine_client.QueuePublish(
      {ResidentRecord{"never-submitted", {}, never_page.front()}}, false);
  ASSERT_NE(never_ticket, 0u);
  ASSERT_TRUE(
      engine_client.MarkPublishReady(never_ticket, engine::CompletionProof::kNeverSubmitted));
  for (int attempt = 0; attempt < 100 && engine_client.PublishTaskCountForTest() != 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(engine_client.PublishTaskCountForTest(), 0u);
  EXPECT_EQ(engine_client.Accounting().in_use, 1u)
      << "NeverSubmitted cancels publication but the tree still owns the page";
  ASSERT_TRUE(engine_client.ReturnPages(never_page));
  ASSERT_TRUE(engine_client.Refund(never_reservation.id));

  auto old_reservation = engine_client.Reserve("old-duplicate", 1, 0);
  ASSERT_TRUE(old_reservation.success);
  auto old_page = engine_client.Take(old_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(old_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(old_reservation.id));
  ASSERT_TRUE(
      engine_client.Publish({ResidentRecord{"duplicate-key", {}, old_page.front()}}).success);
  ASSERT_TRUE(engine_client.ReturnPages(old_page));
  std::vector<engine::RetireTicket> old_ready;
  for (int attempt = 0; attempt < 300 && old_ready.empty(); ++attempt) {
    old_ready = engine_client.PollRetires();
    if (old_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(old_ready.size(), 1u);
  auto replacement_reservation = engine_client.Reserve("replacement", 1, 0);
  ASSERT_TRUE(replacement_reservation.success);
  auto replacement =
      engine_client.Take(replacement_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(replacement.size(), 1u);
  ASSERT_TRUE(
      engine_client.Publish({ResidentRecord{"duplicate-key", {}, replacement.front()}}).success)
      << "the background ACK must remove the old key before exposing reusable capacity";
  ASSERT_TRUE(engine_client.ReturnPages(replacement));
  ASSERT_TRUE(engine_client.Refund(replacement_reservation.id));
  ASSERT_TRUE(engine_client.AcknowledgeReleased({old_ready.front().id}));
  std::vector<engine::RetireTicket> replacement_ready;
  for (int attempt = 0; attempt < 300 && replacement_ready.empty(); ++attempt) {
    replacement_ready = engine_client.PollRetires();
    if (replacement_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(replacement_ready.size(), 1u);
  EXPECT_EQ(engine_client.Accounting().cached, 2u);

  auto io_reservation = engine_client.Reserve("publish-io-race", 1, 0);
  ASSERT_TRUE(io_reservation.success);
  auto io_page = engine_client.Take(io_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(io_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(io_reservation.id));
  std::atomic<size_t> node_io_rpcs{0};
  engine_client.SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (request.has_io_hold() || request.has_io_complete()) ++node_io_rpcs;
  });
  auto deferred_io = engine_client.HoldIO(io_page);
  ASSERT_TRUE(deferred_io.has_value());
  ASSERT_NE(engine_client.QueuePublish({ResidentRecord{"io-race", {}, io_page.front()}}, true), 0u);
  for (int attempt = 0; attempt < 300 && engine_client.Accounting().published == 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(engine_client.Accounting().published, 1u);
  ASSERT_TRUE(engine_client.ReturnPages(io_page));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_TRUE(engine_client.PollRetires().empty());
  EXPECT_EQ(engine_client.Accounting().published, 1u)
      << "ReturnPages must not send Retire while the local PD hold is live";
  EXPECT_TRUE(engine_client.CompleteIO(*deferred_io, engine::CompletionProof::kCompleted));
  std::vector<engine::RetireTicket> io_ready;
  for (int attempt = 0; attempt < 300 && io_ready.empty(); ++attempt) {
    io_ready = engine_client.PollRetires();
    if (io_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(io_ready.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({io_ready.front().id}));
  engine_client.SetBeforeCallSendHookForTest({});
  EXPECT_EQ(node_io_rpcs.load(), 0u)
      << "owner-rank PD holds are local gates and must not add scheduler-thread RPCs";
  EXPECT_EQ(engine_client.Accounting().cached, 2u);

  auto seal_reservation = engine_client.Reserve("quiesce-publish", 1, 0);
  ASSERT_TRUE(seal_reservation.success);
  auto seal_page = engine_client.Take(seal_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(seal_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(seal_reservation.id));
  std::mutex seal_mutex;
  std::condition_variable seal_cv;
  bool at_publish_send = false;
  bool allow_sealed_publish = false;
  engine_client.SetBeforePublishSendHookForTest([&] {
    std::unique_lock<std::mutex> lock(seal_mutex);
    at_publish_send = true;
    seal_cv.notify_all();
    seal_cv.wait(lock, [&] { return allow_sealed_publish; });
  });
  ASSERT_NE(
      engine_client.QueuePublish({ResidentRecord{"quiesce-publish", {}, seal_page.front()}}, true),
      0u);
  {
    std::unique_lock<std::mutex> lock(seal_mutex);
    ASSERT_TRUE(seal_cv.wait_for(lock, std::chrono::seconds(1), [&] { return at_publish_send; }));
  }
  auto seal_reset = std::async(std::launch::async, [&] { return engine_client.QuiesceReset(); });
  EXPECT_EQ(seal_reset.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
      << "quiesce must wait for a PUBLISH_SENT result";
  {
    std::lock_guard<std::mutex> lock(seal_mutex);
    allow_sealed_publish = true;
  }
  seal_cv.notify_all();
  ASSERT_EQ(seal_reset.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  auto sealed_lease = seal_reset.get();
  ASSERT_TRUE(sealed_lease.ready);
  ASSERT_TRUE(engine_client.FinishReset(sealed_lease, false));
  engine_client.SetBeforePublishSendHookForTest({});
  ASSERT_TRUE(engine_client.ReturnPages(seal_page));
  std::vector<engine::RetireTicket> seal_ready;
  for (int attempt = 0; attempt < 300 && seal_ready.empty(); ++attempt) {
    seal_ready = engine_client.PollRetires();
    if (seal_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(seal_ready.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({seal_ready.front().id}));

  auto busy_retire_reservation = engine_client.Reserve("busy-retire-retry", 1, 0);
  ASSERT_TRUE(busy_retire_reservation.success);
  auto busy_retire_page =
      engine_client.Take(busy_retire_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(busy_retire_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(busy_retire_reservation.id));
  ASSERT_TRUE(
      engine_client.Publish({ResidentRecord{"busy-retire-retry", {}, busy_retire_page.front()}})
          .success);
  std::mutex retire_mutex;
  std::condition_variable retire_cv;
  bool retire_committed = false;
  bool allow_retire_send = false;
  bool busy_reply_received = false;
  bool allow_busy_apply = false;
  engine_client.SetBeforeRetireSendHookForTest([&] {
    std::unique_lock<std::mutex> lock(retire_mutex);
    retire_committed = true;
    retire_cv.notify_all();
    retire_cv.wait(lock, [&] { return allow_retire_send; });
  });
  engine_client.SetAfterRetireReplyHookForTest([&] {
    std::unique_lock<std::mutex> lock(retire_mutex);
    busy_reply_received = true;
    retire_cv.notify_all();
    retire_cv.wait(lock, [&] { return allow_busy_apply; });
  });
  ASSERT_TRUE(engine_client.ReturnPages(busy_retire_page));
  {
    std::unique_lock<std::mutex> lock(retire_mutex);
    ASSERT_TRUE(
        retire_cv.wait_for(lock, std::chrono::seconds(1), [&] { return retire_committed; }));
  }
  auto retire_seal = engine_client.QuiesceReset();
  ASSERT_TRUE(retire_seal.ready);
  {
    std::lock_guard<std::mutex> lock(retire_mutex);
    allow_retire_send = true;
  }
  retire_cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(retire_mutex);
    ASSERT_TRUE(
        retire_cv.wait_for(lock, std::chrono::seconds(1), [&] { return busy_reply_received; }));
  }
  ASSERT_TRUE(engine_client.FinishReset(retire_seal, false))
      << "abort must win before the Busy reply is applied to local state";
  {
    std::lock_guard<std::mutex> lock(retire_mutex);
    allow_busy_apply = true;
  }
  retire_cv.notify_all();
  engine_client.SetBeforeRetireSendHookForTest({});
  engine_client.SetAfterRetireReplyHookForTest({});
  std::vector<engine::RetireTicket> retried_retire;
  for (int attempt = 0; attempt < 300 && retried_retire.empty(); ++attempt) {
    retried_retire = engine_client.PollRetires();
    if (retried_retire.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(retried_retire.size(), 1u)
      << "ending reset must resend every returned page restored to PUBLISHED";
  ASSERT_TRUE(engine_client.AcknowledgeReleased({retried_retire.front().id}));

  auto raced_reservation = engine_client.Reserve("publish-return-race", 1, 0);
  ASSERT_TRUE(raced_reservation.success);
  auto raced = engine_client.Take(raced_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(raced.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(raced_reservation.id));
  std::mutex race_mutex;
  std::condition_variable race_cv;
  bool publish_committed = false;
  bool allow_publish_send = false;
  engine_client.SetBeforePublishSendHookForTest([&] {
    std::unique_lock<std::mutex> lock(race_mutex);
    publish_committed = true;
    race_cv.notify_all();
    race_cv.wait(lock, [&] { return allow_publish_send; });
  });
  ASSERT_NE(engine_client.QueuePublish({ResidentRecord{"raced-publish", {}, raced.front()}}, true),
            0u);
  {
    std::unique_lock<std::mutex> lock(race_mutex);
    ASSERT_TRUE(race_cv.wait_for(lock, std::chrono::seconds(1), [&] { return publish_committed; }));
  }
  EXPECT_EQ(engine_client.Accounting().publish_sent, 1u);
  ASSERT_TRUE(engine_client.ReturnPages(raced));
  {
    std::lock_guard<std::mutex> lock(race_mutex);
    allow_publish_send = true;
  }
  race_cv.notify_all();
  std::vector<engine::RetireTicket> raced_ready;
  for (int attempt = 0; attempt < 300 && raced_ready.empty(); ++attempt) {
    raced_ready = engine_client.PollRetires();
    if (raced_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(raced_ready.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({raced_ready.front().id}));
  engine_client.SetBeforePublishSendHookForTest({});
  EXPECT_EQ(engine_client.Accounting().cached, 2u);

  auto reservation = engine_client.Reserve("standalone-request", 1, 0);
  ASSERT_TRUE(reservation.success);
  auto taken = engine_client.Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(taken.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(reservation.id));
  auto first_use = taken.front();
  for (size_t i = 0; i < kObjectBytes; ++i) {
    hbm[first_use.page_id * kObjectBytes + i] = 0x40 + i;
  }
  ResidentRecord resident{"late-engine-key", {}, first_use};
  auto published = engine_client.Publish({resident});
  ASSERT_TRUE(published.success);
  ASSERT_EQ(published.accepted, std::vector<bool>({true}));

  auto published_hold = engine_client.HoldIO({first_use});
  ASSERT_TRUE(published_hold.has_value());
  auto busy_reset = engine_client.QuiesceReset();
  EXPECT_FALSE(busy_reset.ready);
  ASSERT_TRUE(engine_client.CompleteIO(*published_hold, engine::CompletionProof::kCompleted));
  auto after_busy = engine_client.Reserve("after-busy", 1, 0);
  ASSERT_TRUE(after_busy.success) << "Busy reset must reopen local reservation gate";
  ASSERT_TRUE(engine_client.Refund(after_busy.id));

  auto published_guard = engine_client.HoldReuseGuard({first_use});
  ASSERT_TRUE(published_guard.has_value());
  ASSERT_TRUE(engine_client.ReturnPages({first_use}));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_TRUE(engine_client.PollRetires().empty());
  EXPECT_EQ(engine_client.Accounting().published, 1u)
      << "the reuse guard gates Retire itself, not only post-ACK reuse";
  ASSERT_TRUE(
      engine_client.CompleteReuseGuard(*published_guard, engine::CompletionProof::kCompleted));

  UMBPConfig client_config;
  UMBPStandaloneProcessConfig standalone_config;
  standalone_config.address = address;
  standalone_config.startup_timeout_ms = 5000;
  client_config.standalone_process = standalone_config;
  auto client = CreateUMBPClient(client_config);

  HostMemAllocator allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;
  auto destination = allocator.Alloc(4096, options);
  ASSERT_TRUE(destination.valid());
  ASSERT_TRUE(client->RegisterMemory(reinterpret_cast<uintptr_t>(destination.ptr),
                                     destination.mapped_size));
  std::memset(destination.ptr, 0x5a, kObjectBytes);
  ASSERT_TRUE(
      client->Put("ordinary-put", reinterpret_cast<uintptr_t>(destination.ptr), kObjectBytes));
  std::memset(destination.ptr, 0, kObjectBytes);
  ASSERT_TRUE(
      client->Get("ordinary-put", reinterpret_cast<uintptr_t>(destination.ptr), kObjectBytes));
  for (size_t i = 0; i < kObjectBytes; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(destination.ptr)[i], 0x5a);
  }
  EXPECT_FALSE(client->Clear())
      << "legacy Clear must report rejection while any engine HBM rank is active";
  std::memset(destination.ptr, 0, kObjectBytes);
  ASSERT_TRUE(
      client->Get("ordinary-put", reinterpret_cast<uintptr_t>(destination.ptr), kObjectBytes));
  for (size_t i = 0; i < kObjectBytes; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(destination.ptr)[i], 0x5a)
        << "a rejected Clear must not discard master/local placement state";
  }
  std::memset(destination.ptr, 0, kObjectBytes);
  bool restored = false;
  for (int attempt = 0; attempt < 300 && !restored; ++attempt) {
    restored =
        client->Get(resident.key, reinterpret_cast<uintptr_t>(destination.ptr), kObjectBytes);
    if (!restored) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(restored);
  EXPECT_EQ(
      std::memcmp(destination.ptr, hbm.data() + first_use.page_id * kObjectBytes, kObjectBytes), 0);

  std::vector<engine::RetireTicket> released;
  for (int attempt = 0; attempt < 300 && released.empty(); ++attempt) {
    released = engine_client.PollRetires();
    if (released.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(released.size(), 1u);
  ASSERT_EQ(released.front().state, engine::RetireState::kReady);
  auto before_partial_ack = engine_client.Accounting();
  EXPECT_EQ(before_partial_ack.cached, 2u);
  EXPECT_EQ(before_partial_ack.retiring, 0u)
      << "Ready must be ACKed without the ineffective 10 ms client delay";
  auto invalid_ack = released.front().id;
  invalid_ack.seq += 1000;
  EXPECT_FALSE(engine_client.AcknowledgeReleased({released.front().id, invalid_ack}));
  auto after_partial_ack = engine_client.Accounting();
  EXPECT_EQ(after_partial_ack.cached, 2u);
  EXPECT_EQ(after_partial_ack.pending_free, 0u);
  EXPECT_EQ(after_partial_ack.retiring, 0u);

  auto voided_reservation = engine_client.Reserve("voided-by-reset", 1, 0);
  ASSERT_TRUE(voided_reservation.success);
  auto reset = engine_client.QuiesceReset();
  ASSERT_TRUE(reset.ready);
  EXPECT_FALSE(engine_client.Reserve("while-quiescing", 1, 0).success);
  ASSERT_TRUE(engine_client.FinishReset(reset, true));
  EXPECT_FALSE(engine_client.Refund(voided_reservation.id))
      << "reset commit must invalidate every old reservation";
  EXPECT_TRUE(engine_client.healthy());
  EXPECT_EQ(engine_client.Accounting().cached, 2u);
  std::memset(destination.ptr, 0, kObjectBytes);
  ASSERT_TRUE(
      client->Get(resident.key, reinterpret_cast<uintptr_t>(destination.ptr), kObjectBytes));
  EXPECT_EQ(
      std::memcmp(destination.ptr, hbm.data() + first_use.page_id * kObjectBytes, kObjectBytes), 0)
      << "rank reset must not remove the shared DRAM replica";

  auto quarantine_reservation = engine_client.Reserve("local-quarantine", 1, 0);
  ASSERT_TRUE(quarantine_reservation.success);
  auto quarantine_page =
      engine_client.Take(quarantine_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(quarantine_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(quarantine_reservation.id));
  ASSERT_TRUE(
      engine_client.Publish({ResidentRecord{"released-guard-unknown", {}, quarantine_page.front()}})
          .success);
  auto quarantine_guard = engine_client.HoldReuseGuard(quarantine_page);
  ASSERT_TRUE(quarantine_guard.has_value());
  ASSERT_TRUE(engine_client.ReturnPages(quarantine_page));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_TRUE(engine_client.PollRetires().empty());
  EXPECT_EQ(engine_client.Accounting().published, 1u);
  ASSERT_TRUE(
      engine_client.CompleteReuseGuard(*quarantine_guard, engine::CompletionProof::kUnknown));
  EXPECT_TRUE(engine_client.healthy())
      << "the node must accept quarantine for the still-published generation";
  EXPECT_EQ(engine_client.Accounting().quarantined, 1u);
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto events = engine_client.PollRetires();
    if (!events.empty()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto quarantine_reset = engine_client.QuiesceReset();
  ASSERT_TRUE(quarantine_reset.ready)
      << "a hold retained only for a quarantined page must not make reset Busy";
  ASSERT_TRUE(engine_client.FinishReset(quarantine_reset, true));
  EXPECT_EQ(engine_client.Accounting().quarantined, 1u);
  EXPECT_EQ(engine_client.Accounting().cached, 1u);
  EXPECT_EQ(engine_client.Reserve("quarantine-excluded", 2, 0).shortfall, 1u)
      << "logical reset must not revive the locally quarantined page";

  auto mixed_reservation = engine_client.Reserve("mixed-return", 1, 0);
  ASSERT_TRUE(mixed_reservation.success);
  auto mixed_page = engine_client.Take(mixed_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(mixed_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(mixed_reservation.id));
  ASSERT_TRUE(
      engine_client.Publish({ResidentRecord{"mixed-return-key", {}, mixed_page.front()}}).success);
  ASSERT_TRUE(engine_client.ReturnPages({mixed_page.front(), quarantine_page.front()}))
      << "a quarantined member must not strand earlier valid pages in the batch";
  std::vector<engine::RetireTicket> mixed_ready;
  for (int attempt = 0; attempt < 300 && mixed_ready.empty(); ++attempt) {
    mixed_ready = engine_client.PollRetires();
    if (mixed_ready.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(mixed_ready.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({mixed_ready.front().id}));
  EXPECT_EQ(engine_client.Accounting().cached, 1u);
  EXPECT_EQ(engine_client.Accounting().quarantined, 1u);

  client->DeregisterMemory(reinterpret_cast<uintptr_t>(destination.ptr));
  client->Close();

  auto fail_reservation = engine_client.Reserve("publish-fail-stop", 1, 0);
  ASSERT_TRUE(fail_reservation.success);
  auto fail_page = engine_client.Take(fail_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(fail_page.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(fail_reservation.id));
  std::mutex fail_mutex;
  std::condition_variable fail_cv;
  bool publish_sent = false;
  bool release_publish = false;
  engine_client.SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (!request.has_publish()) return;
    std::unique_lock<std::mutex> lock(fail_mutex);
    publish_sent = true;
    fail_cv.notify_all();
    fail_cv.wait(lock, [&] { return release_publish; });
  });
  auto synchronous_publish = std::async(std::launch::async, [&] {
    return engine_client.Publish({ResidentRecord{"unknown-publish", {}, fail_page.front()}});
  });
  {
    std::unique_lock<std::mutex> lock(fail_mutex);
    ASSERT_TRUE(fail_cv.wait_for(lock, std::chrono::seconds(1), [&] { return publish_sent; }));
  }
  ASSERT_EQ(engine_client.Accounting().publish_sent, 1u);
  running.Stop();
  for (int attempt = 0; attempt < 100 && engine_client.healthy(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  int reuse_pipe[2];
  ASSERT_EQ(pipe(reuse_pipe), 0);
  {
    std::lock_guard<std::mutex> lock(fail_mutex);
    release_publish = true;
  }
  fail_cv.notify_all();
  ASSERT_EQ(synchronous_publish.wait_for(std::chrono::seconds(1)), std::future_status::ready)
      << "fail-stop must wake a synchronous Publish waiter";
  EXPECT_FALSE(synchronous_publish.get().success);
  const uint8_t marker = 0x7d;
  uint8_t observed_marker = 0;
  ASSERT_EQ(write(reuse_pipe[1], &marker, sizeof(marker)), static_cast<ssize_t>(sizeof(marker)));
  ASSERT_EQ(read(reuse_pipe[0], &observed_marker, sizeof(observed_marker)),
            static_cast<ssize_t>(sizeof(observed_marker)));
  EXPECT_EQ(observed_marker, marker)
      << "ReceiverLoop must not close an fd number reused while CallLocked owns the socket";
  close(reuse_pipe[0]);
  close(reuse_pipe[1]);
  engine_client.SetBeforeCallSendHookForTest({});
  EXPECT_FALSE(engine_client.healthy());
  EXPECT_EQ(engine_client.Accounting().publish_sent, 1u)
      << "an already-sent publish with an unknown result must not return to cache";
  EXPECT_EQ(engine_client.Accounting().available, 0u);
  EXPECT_FALSE(engine_client.Reserve("after-fail-stop", 1, 0).success);
  engine_client.Close();
  executor.Close();
  allocator.Free(destination);
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());
}

}  // namespace
}  // namespace mori::umbp
