// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/object_view.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/transfer/engine_transfer_engine.h"
#include "umbp/engine/engine_client.h"
#include "umbp/engine/engine_control.h"
#include "umbp/engine/engine_executor.h"
#include "umbp/engine/engine_session.h"
#include "umbp/engine/kv_allocator.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {

using namespace std::chrono_literals;

int VisibleGpuCount() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  return count;
}

std::vector<uint8_t> Pattern(size_t bytes, uint8_t seed) {
  std::vector<uint8_t> out(bytes);
  for (size_t i = 0; i < bytes; ++i) out[i] = static_cast<uint8_t>(i * 17 + seed);
  return out;
}

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) old_ = old;
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_.has_value()) {
      setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> old_;
};

struct VmmBuffer {
  engine::KvVmmAllocator* allocator = nullptr;
  void* ptr = nullptr;
  size_t requested = 0;
  size_t mapped = 0;
  int device = -1;

  VmmBuffer() = default;
  VmmBuffer(engine::KvVmmAllocator* owner, size_t bytes, int gpu)
      : allocator(owner), requested(bytes), device(gpu) {
    ptr = allocator->Allocate(bytes, gpu);
    if (ptr != nullptr) {
      auto allocation = allocator->Lookup(ptr);
      if (allocation.has_value()) mapped = allocation->mapped_size;
    }
  }
  ~VmmBuffer() {
    if (allocator != nullptr && ptr != nullptr) (void)allocator->Free(ptr);
  }
  VmmBuffer(const VmmBuffer&) = delete;
  VmmBuffer& operator=(const VmmBuffer&) = delete;
  bool valid() const { return ptr != nullptr && mapped >= requested; }
};

bool SendFd(int socket_fd, int fd) {
  char byte = 'v';
  iovec iov{&byte, 1};
  std::array<char, CMSG_SPACE(sizeof(int))> control{};
  msghdr message{};
  message.msg_iov = &iov;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  cmsghdr* cmsg = CMSG_FIRSTHDR(&message);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
  return sendmsg(socket_fd, &message, MSG_NOSIGNAL) == 1;
}

int RecvFd(int socket_fd) {
  char byte = 0;
  iovec iov{&byte, 1};
  std::array<char, CMSG_SPACE(sizeof(int))> control{};
  msghdr message{};
  message.msg_iov = &iov;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  if (recvmsg(socket_fd, &message, 0) != 1 || (message.msg_flags & MSG_CTRUNC) != 0) return -1;
  for (cmsghdr* cmsg = CMSG_FIRSTHDR(&message); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&message, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
        cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
      int fd = -1;
      std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
      return fd;
    }
  }
  return -1;
}

bool WaitPid(pid_t pid, std::chrono::seconds timeout, int* status) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t result = waitpid(pid, status, WNOHANG);
    if (result == pid) return true;
    if (result < 0) return false;
    std::this_thread::sleep_for(10ms);
  }
  kill(pid, SIGKILL);
  (void)waitpid(pid, status, 0);
  return false;
}

class ChildProcessGuard {
 public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ <= 0) return;
    kill(pid_, SIGKILL);
    (void)waitpid(pid_, nullptr, 0);
  }
  void Reaped() { pid_ = -1; }

 private:
  pid_t pid_ = -1;
};

int VmmImportChild(int socket_fd, size_t mapped, size_t bytes, int device) {
  const int fd = RecvFd(socket_fd);
  close(socket_fd);
  if (fd < 0 || hipSetDevice(device) != hipSuccess) return 20;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  engine::ExportedVmmHandle exported;
  exported.kind = engine::VmmHandleKind::kPosixFd;
  exported.fd = fd;
  exported.mapped_size = mapped;
  auto imported = allocator.Import(exported, device);
  close(fd);
  if (!imported.has_value()) return 21;
  std::vector<uint8_t> actual(bytes);
  if (hipMemcpy(actual.data(), imported->base, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
      actual != Pattern(bytes, 3)) {
    return 22;
  }
  std::vector<uint8_t> changed(bytes, static_cast<uint8_t>(0xa0 + device));
  if (hipMemcpy(imported->base, changed.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess || !allocator.FreeImported(&*imported)) {
    return 23;
  }
  return 0;
}

bool RunCrossProcessImport(int access_device, std::string* error) {
  constexpr size_t kBytes = 1 << 20;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer source(&allocator, kBytes, 0);
  if (!source.valid()) {
    *error = "real POSIX-fd hipMemCreate allocation failed";
    return false;
  }
  const auto expected = Pattern(kBytes, 3);
  if (hipSetDevice(0) != hipSuccess || hipMemcpy(source.ptr, expected.data(), expected.size(),
                                                 hipMemcpyHostToDevice) != hipSuccess) {
    *error = "initial H2D failed";
    return false;
  }
  auto exported = allocator.Export(source.ptr);
  if (!exported.has_value() || exported->kind != engine::VmmHandleKind::kPosixFd ||
      exported->fd < 0) {
    *error = "POSIX fd export failed";
    return false;
  }
  if (fcntl(exported->fd, F_SETFD, FD_CLOEXEC) != 0) {
    close(exported->fd);
    *error = "failed to make the inherited export fd close-on-exec";
    return false;
  }

  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    close(exported->fd);
    *error = std::strerror(errno);
    return false;
  }
  const pid_t child = fork();
  if (child == 0) {
    close(sockets[0]);
    const std::string socket_arg = std::to_string(sockets[1]);
    const std::string mapped_arg = std::to_string(exported->mapped_size);
    const std::string bytes_arg = std::to_string(kBytes);
    const std::string device_arg = std::to_string(access_device);
    execl("/proc/self/exe", "test_engine_gpu", "--vmm-import-child", socket_arg.c_str(),
          mapped_arg.c_str(), bytes_arg.c_str(), device_arg.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  close(sockets[1]);
  if (child < 0) {
    close(exported->fd);
    close(sockets[0]);
    *error = "failed to fork import child";
    return false;
  }
  if (!SendFd(sockets[0], exported->fd)) {
    close(exported->fd);
    close(sockets[0]);
    kill(child, SIGKILL);
    (void)waitpid(child, nullptr, 0);
    *error = "failed to send exported fd";
    return false;
  }
  close(exported->fd);
  close(sockets[0]);
  int status = 0;
  if (!WaitPid(child, 30s, &status) || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    *error = "import child failed with status " + std::to_string(status);
    return false;
  }
  std::vector<uint8_t> actual(kBytes);
  if (hipSetDevice(0) != hipSuccess ||
      hipMemcpy(actual.data(), source.ptr, actual.size(), hipMemcpyDeviceToHost) != hipSuccess) {
    *error = "parent D2H after child write failed";
    return false;
  }
  const std::vector<uint8_t> changed(kBytes, static_cast<uint8_t>(0xa0 + access_device));
  if (actual != changed) {
    *error = "parent did not observe child write";
    return false;
  }
  return true;
}

TEST(UmbpEngineGpu, PosixFdCrossProcessImportSameGpuIsByteExact) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  std::string error;
  EXPECT_TRUE(RunCrossProcessImport(0, &error)) << error;
}

TEST(UmbpEngineGpu, PosixFdCrossProcessImportPeerGpuIsByteExact) {
  if (VisibleGpuCount() < 2) GTEST_SKIP() << "requires two HIP GPUs";
  int can_access = 0;
  ASSERT_EQ(hipDeviceCanAccessPeer(&can_access, 1, 0), hipSuccess);
  if (can_access == 0) GTEST_SKIP() << "GPU 1 cannot access GPU 0 VMM allocation";
  std::string error;
  EXPECT_TRUE(RunCrossProcessImport(1, &error)) << error;
}

TransferRef EngineEndpoint(std::string executor, std::string region, uint64_t size,
                           mori::io::MemoryLocationType location, int device, void* address) {
  TransferRef out;
  out.size = size;
  out.loc = location;
  out.device = device;
  out.executor_id = std::move(executor);
  out.region_id = std::move(region);
  out.executor_address = reinterpret_cast<uintptr_t>(address);
  return out;
}

std::vector<uint8_t> GatherHostView(const ObjectView& view) {
  std::vector<uint8_t> out(view.size);
  for (const auto& span : view.spans) {
    std::memcpy(out.data() + span.logical_offset,
                static_cast<const uint8_t*>(span.buffer.host_ptr) + span.offset, span.length);
  }
  return out;
}

void ScatterHostView(const ObjectView& view, const std::vector<uint8_t>& bytes) {
  for (const auto& span : view.spans) {
    std::memcpy(static_cast<uint8_t*>(span.buffer.host_ptr) + span.offset,
                bytes.data() + span.logical_offset, span.length);
  }
}

TEST(UmbpEngineGpu, ExecutorCopiesScatteredShortTailMultiRegionThroughSharedDram) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  constexpr size_t kRegion = 4096;
  constexpr size_t kObject = 777;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer gpu_a(&allocator, kRegion, 0);
  VmmBuffer gpu_b(&allocator, kRegion, 0);
  ASSERT_TRUE(gpu_a.valid());
  ASSERT_TRUE(gpu_b.valid());
  auto host_a = Pattern(kRegion, 11);
  auto host_b = Pattern(kRegion, 29);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMemcpy(gpu_a.ptr, host_a.data(), kRegion, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(gpu_b.ptr, host_b.data(), kRegion, hipMemcpyHostToDevice), hipSuccess);

  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {kRegion, kRegion};
  ownership.shared = true;
  ownership.use_hugepages = false;
  ownership.prefault = false;
  PageBackend dram(TierType::DRAM, kRegion, ownership, 30s);
  ASSERT_TRUE(dram.Init(nullptr));
  engine::PoolEpoch epoch;
  auto shared = dram.ExportSharedRegions(/*backend_id=*/7, epoch);
  ASSERT_EQ(shared.size(), 2u);

  engine::EngineExecutor executor({"span-gpu-executor", 0});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "hbm-a",
      TransferRef::HostBytes(gpu_a.ptr, gpu_a.mapped, mori::io::MemoryLocationType::GPU, 0)));
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "hbm-b",
      TransferRef::HostBytes(gpu_b.ptr, gpu_b.mapped, mori::io::MemoryLocationType::GPU, 0)));
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(shared, &map_error)) << map_error;

  auto gpu_a_ref = EngineEndpoint(executor.id(), "hbm-a", gpu_a.mapped,
                                  mori::io::MemoryLocationType::GPU, 0, gpu_a.ptr);
  auto gpu_b_ref = EngineEndpoint(executor.id(), "hbm-b", gpu_b.mapped,
                                  mori::io::MemoryLocationType::GPU, 0, gpu_b.ptr);
  auto dram_a = dram.BufferRef(0);
  auto dram_b = dram.BufferRef(1);
  dram_a.executor_id = executor.id();
  dram_b.executor_id = executor.id();
  ObjectView source{
      kObject,
      "gpu-source",
      {{gpu_a_ref, 13, 257, 0}, {gpu_b_ref, 111, 199, 257}, {gpu_a_ref, 1000, 321, 456}}};
  ObjectView target{kObject,
                    "shared-dram",
                    {{dram_a, 31, 100, 0}, {dram_b, 64, 500, 100}, {dram_a, 1000, 177, 600}}};
  ASSERT_TRUE(ValidateObjectView(source));
  ASSERT_TRUE(ValidateObjectView(target));
  std::vector<TransferItem> items;
  ASSERT_TRUE(BuildObjectViewTransfers(source, target, &items));
  ASSERT_GT(items.size(), 3u);
  EngineTransferEngine transfer(&executor);
  std::vector<size_t> failed;
  ASSERT_TRUE(transfer.Transfer(items, &failed));
  ASSERT_TRUE(failed.empty());

  ObjectView expected_source{
      kObject,
      "expected",
      {{TransferRef::HostBytes(host_a.data(), host_a.size()), 13, 257, 0},
       {TransferRef::HostBytes(host_b.data(), host_b.size()), 111, 199, 257},
       {TransferRef::HostBytes(host_a.data(), host_a.size()), 1000, 321, 456}}};
  EXPECT_EQ(GatherHostView(target), GatherHostView(expected_source));

  const auto reverse = Pattern(kObject, 73);
  ScatterHostView(target, reverse);
  ASSERT_TRUE(BuildObjectViewTransfers(target, source, &items));
  failed.clear();
  ASSERT_TRUE(transfer.Transfer(items, &failed));
  ASSERT_TRUE(failed.empty());
  ASSERT_EQ(hipMemcpy(host_a.data(), gpu_a.ptr, kRegion, hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(host_b.data(), gpu_b.ptr, kRegion, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(GatherHostView(expected_source), reverse);
  executor.Close();
  dram.Shutdown();
}

TEST(UmbpEngineGpu, OriginalVmmAllocationRegistersAndRoundTripsThroughRdma) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  if (!std::filesystem::exists("/dev/infiniband")) GTEST_SKIP() << "no RDMA device nodes";
  ScopedEnv disable_xgmi("MORI_DISABLE_AUTO_XGMI", "1");
  mori::io::IOEngineConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  auto initiator = std::make_unique<mori::io::IOEngine>("vmm-rdma-initiator", config);
  auto target = std::make_unique<mori::io::IOEngine>("vmm-rdma-target", config);
  mori::io::RdmaBackendConfig rdma;
  try {
    initiator->CreateBackend(mori::io::BackendType::RDMA, rdma);
    target->CreateBackend(mori::io::BackendType::RDMA, rdma);
  } catch (const std::exception& error) {
    GTEST_SKIP() << "RDMA backend unavailable: " << error.what();
  }
  initiator->RegisterRemoteEngine(target->GetEngineDesc());
  target->RegisterRemoteEngine(initiator->GetEngineDesc());

  constexpr size_t kBytes = 1 << 20;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer local(&allocator, kBytes, 0);
  VmmBuffer remote(&allocator, kBytes, 0);
  ASSERT_TRUE(local.valid());
  ASSERT_TRUE(remote.valid());
  auto local_desc =
      initiator->RegisterMemory(local.ptr, kBytes, 0, mori::io::MemoryLocationType::GPU);
  auto remote_desc =
      target->RegisterMemory(remote.ptr, kBytes, 0, mori::io::MemoryLocationType::GPU);

  auto first = Pattern(kBytes, 41);
  ASSERT_EQ(hipMemcpy(local.ptr, first.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(remote.ptr, 0, kBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess)
      << "the clear must finish before the NIC writes the remote VMM allocation";
  mori::io::TransferStatus write_status;
  initiator->Write(local_desc, 0, remote_desc, 0, kBytes, &write_status,
                   initiator->AllocateTransferUniqueId());
  const auto write_result = initiator->WaitAll({&write_status}, 10000);
  ASSERT_EQ(write_result, mori::io::StatusCode::SUCCESS) << write_status.Message();
  ASSERT_TRUE(write_status.Succeeded()) << write_status.Message();
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  std::vector<uint8_t> actual(kBytes);
  ASSERT_EQ(hipMemcpy(actual.data(), remote.ptr, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(actual, first);

  auto second = Pattern(kBytes, 97);
  ASSERT_EQ(hipMemcpy(remote.ptr, second.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(local.ptr, 0, kBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess)
      << "the clear must finish before the NIC reads into the local VMM allocation";
  mori::io::TransferStatus read_status;
  initiator->Read(local_desc, 0, remote_desc, 0, kBytes, &read_status,
                  initiator->AllocateTransferUniqueId());
  const auto read_result = initiator->WaitAll({&read_status}, 10000);
  ASSERT_EQ(read_result, mori::io::StatusCode::SUCCESS) << read_status.Message();
  ASSERT_TRUE(read_status.Succeeded()) << read_status.Message();
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipMemcpy(actual.data(), local.ptr, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  const auto first_mismatch = std::mismatch(actual.begin(), actual.end(), second.begin());
  EXPECT_EQ(first_mismatch.first, actual.end())
      << "RDMA Read reported SUCCESS but did not complete all " << kBytes << " bytes";
  initiator->DeregisterMemory(local_desc);
  target->DeregisterMemory(remote_desc);
}

class RunningStandalone {
 public:
  explicit RunningStandalone(standalone::StandaloneServer* server)
      : server_(server), thread_([this] { server_->Run(); }) {}
  ~RunningStandalone() { Stop(); }
  void Stop() {
    if (server_ == nullptr) return;
    server_->Shutdown();
    if (thread_.joinable()) thread_.join();
    server_ = nullptr;
  }

 private:
  standalone::StandaloneServer* server_ = nullptr;
  std::thread thread_;
};

class GpuPoolEngineControl final : public engine::EngineControl {
 public:
  explicit GpuPoolEngineControl(PoolClient* client) : client_(client) {}
  bool RegisterEngine(EngineRegistration registration) override {
    return client_ != nullptr && client_->ActivateEngine(std::move(registration));
  }
  std::optional<uint32_t> BackendId(engine::RankId rank) const override {
    return client_ == nullptr ? std::nullopt : client_->EngineBackendId(rank);
  }
  std::vector<engine::SharedDramRegion> ExportSharedDram(engine::PoolEpoch epoch) override {
    return client_ == nullptr ? std::vector<engine::SharedDramRegion>{}
                              : client_->ExportSharedDram(epoch);
  }
  PeerPool* Pool() const override { return client_ == nullptr ? nullptr : client_->EnginePool(); }
  engine::ReadPlan Lookup(const std::vector<std::string>& keys, std::string name_space,
                          std::string layout_id, uint32_t requester_backend_id) override {
    return client_ == nullptr ? engine::ReadPlan{}
                              : client_->LookupEngine(keys, std::move(name_space),
                                                      std::move(layout_id), requester_backend_id);
  }
  engine::PreparedLoad PrepareLoad(const engine::ReadPlan& plan, std::string request_id,
                                   const std::vector<engine::AllocationHandle>& targets,
                                   uint32_t n_pages, const ReaderIdentity& reader,
                                   engine::OperationId operation) override {
    return client_ == nullptr ? engine::PreparedLoad{}
                              : client_->PrepareEngineLoad(plan, std::move(request_id), targets,
                                                           n_pages, reader, operation);
  }
  engine::CancelLoadResult CancelLoad(uint64_t prepared_id,
                                      engine::OperationId operation) override {
    return client_ == nullptr ? engine::CancelLoadResult::kRejected
                              : client_->CancelEngineLoad(prepared_id, operation);
  }
  bool CompleteLoad(uint64_t prepared_id, engine::LoadOutcome outcome,
                    engine::OperationId operation) override {
    return client_ != nullptr && client_->CompleteEngineLoad(prepared_id, outcome, operation);
  }
  void SessionFailed(engine::RankId rank) override {
    if (client_ != nullptr) client_->FailEngineSession(rank);
  }

 private:
  PoolClient* client_ = nullptr;
};

class RunningMaster {
 public:
  explicit RunningMaster(MasterServer* master)
      : master_(master), thread_([this] { master_->Run(); }) {}
  ~RunningMaster() {
    if (master_ != nullptr) master_->Shutdown();
    if (thread_.joinable()) thread_.join();
  }

 private:
  MasterServer* master_ = nullptr;
  std::thread thread_;
};

std::string UniqueAddress(const char* name) {
  return "unix:///tmp/" + std::string(name) + "_" + std::to_string(getpid()) + ".grpc.sock";
}

void RemoveSockets(const std::string& address) {
  unlink(standalone::UnixPathFromGrpcAddress(address).c_str());
  unlink(standalone::DeriveFdSocketPath(address).c_str());
  unlink(standalone::DeriveEngineSocketPath(address).c_str());
}

UMBPConfig EngineServerConfig() {
  UMBPConfig config;
  config.dram.capacity_bytes = 1 << 20;
  config.dram.use_hugepages = false;
  config.dram.prefault = false;
  config.page_size = 4096;
  config = WithEmbeddedDefaults(config);
  config.distributed->engine_hbm_ranks = 8;
  return config;
}

engine::PageGeometry SingleRegionGeometry(void* base, size_t size, size_t object_bytes,
                                          std::string region_id = "kv") {
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 1;
  geometry.object_bytes = object_bytes;
  geometry.regions.push_back(engine::PageRegion{
      region_id, TransferRef::HostBytes(base, size, mori::io::MemoryLocationType::GPU, 0),
      object_bytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  return geometry;
}

TEST(UmbpEngineGpu, F3RetireCopiesVmmPageToSharedDramByteExactly) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  const std::string address = UniqueAddress("umbp_engine_gpu_f3");
  RemoveSockets(address);
  auto config = EngineServerConfig();
  standalone::StandaloneServer server(config, address);
  ASSERT_TRUE(server.Start());
  RunningStandalone running(&server);

  constexpr size_t kBytes = 4096;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer hbm(&allocator, 2 * kBytes, 0);
  ASSERT_TRUE(hbm.valid());
  engine::EngineExecutor executor({"f3-gpu-executor", 0});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(hbm.ptr, hbm.mapped, mori::io::MemoryLocationType::GPU, 0)));
  auto geometry = SingleRegionGeometry(hbm.ptr, hbm.mapped, kBytes);
  ASSERT_TRUE(geometry.Validate());
  engine::EngineClient engine_client({address, "f3-gpu-executor", 10, 1000});
  engine_client.AttachExecutor(&executor);
  engine::PoolEpoch requested_epoch;
  auto registration = engine_client.RegisterPool(0, "f3-gpu-pool", requested_epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto reservation = engine_client.Reserve("f3-gpu", 1, 0);
  ASSERT_TRUE(reservation.success);
  auto handles = engine_client.Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(handles.size(), 1u);
  ASSERT_TRUE(engine_client.Refund(reservation.id));
  const auto expected = Pattern(kBytes, 53);
  auto* page = static_cast<uint8_t*>(hbm.ptr) + handles[0].page_id * kBytes;
  ASSERT_EQ(hipMemcpy(page, expected.data(), expected.size(), hipMemcpyHostToDevice), hipSuccess);
  const std::string key = "f3-gpu-byte-key";
  ASSERT_TRUE(engine_client.Publish({ResidentRecord{key, {}, handles[0]}}).success);
  ASSERT_TRUE(engine_client.ReturnPages(handles));

  UMBPConfig client_config;
  UMBPStandaloneProcessConfig standalone_config;
  standalone_config.address = address;
  standalone_config.startup_timeout_ms = 5000;
  client_config.standalone_process = standalone_config;
  auto client = CreateUMBPClient(client_config);
  HostMemAllocator host_allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;
  auto destination = host_allocator.Alloc(kBytes, options);
  ASSERT_TRUE(destination.valid());
  ASSERT_TRUE(client->RegisterMemory(reinterpret_cast<uintptr_t>(destination.ptr),
                                     destination.mapped_size));
  bool restored = false;
  for (int attempt = 0; attempt < 500 && !restored; ++attempt) {
    restored = client->Get(key, reinterpret_cast<uintptr_t>(destination.ptr), kBytes);
    if (!restored) std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(restored);
  EXPECT_EQ(std::memcmp(destination.ptr, expected.data(), expected.size()), 0);
  std::vector<engine::RetireTicket> ready;
  for (int attempt = 0; attempt < 500 && ready.empty(); ++attempt) {
    ready = engine_client.PollRetires();
    if (ready.empty()) std::this_thread::sleep_for(10ms);
  }
  ASSERT_EQ(ready.size(), 1u);
  ASSERT_TRUE(engine_client.AcknowledgeReleased({ready[0].id}));
  client->DeregisterMemory(reinterpret_cast<uintptr_t>(destination.ptr));
  client->Close();
  engine_client.Close();
  executor.Close();
  host_allocator.Free(destination);
  running.Stop();
  RemoveSockets(address);
}

TEST(UmbpEngineGpu, LayeredLoadStitchesPeerHbmPrefixAndSharedDramTail) {
  if (VisibleGpuCount() < 2) GTEST_SKIP() << "requires two HIP GPUs";
  const std::string address = UniqueAddress("umbp_engine_gpu_layered_load");
  RemoveSockets(address);
  auto config = EngineServerConfig();
  standalone::StandaloneServer server(config, address);
  ASSERT_TRUE(server.Start());
  RunningStandalone running(&server);

  constexpr size_t kHalf = 2048;
  constexpr size_t kBytes = 2 * kHalf;
  constexpr uint32_t kPages = 3;
  auto& allocator = engine::GlobalKvVmmAllocator();
  VmmBuffer source0(&allocator, (kPages + 2) * kHalf, 0);
  VmmBuffer source1(&allocator, (kPages + 2) * kHalf, 0);
  VmmBuffer target0(&allocator, (kPages + 1) * kHalf, 1);
  VmmBuffer target1(&allocator, (kPages + 1) * kHalf, 1);
  ASSERT_TRUE(source0.valid() && source1.valid() && target0.valid() && target1.valid());
  void* source_region0 = static_cast<uint8_t*>(source0.ptr) + kHalf;
  void* source_region1 = static_cast<uint8_t*>(source1.ptr) + kHalf;
  const size_t source_region_bytes = (kPages + 1) * kHalf;

  engine::EngineExecutor source_executor({"gpu-load-source", 0});
  engine::EngineExecutor target_executor({"gpu-load-target", 1});
  ASSERT_TRUE(source_executor.RegisterOriginalRegion(
      "layer-0", TransferRef::HostBytes(source_region0, source_region_bytes,
                                        mori::io::MemoryLocationType::GPU, 0)));
  ASSERT_TRUE(source_executor.RegisterOriginalRegion(
      "layer-1", TransferRef::HostBytes(source_region1, source_region_bytes,
                                        mori::io::MemoryLocationType::GPU, 0)));
  ASSERT_TRUE(target_executor.RegisterOriginalRegion(
      "layer-0",
      TransferRef::HostBytes(target0.ptr, target0.mapped, mori::io::MemoryLocationType::GPU, 1)));
  ASSERT_TRUE(target_executor.RegisterOriginalRegion(
      "layer-1",
      TransferRef::HostBytes(target1.ptr, target1.mapped, mori::io::MemoryLocationType::GPU, 1)));
  auto geometry = [&](void* first, size_t first_size, void* second, size_t second_size,
                      int device) {
    engine::PageGeometry out;
    out.page_tokens = 1;
    out.num_pages = kPages;
    out.object_bytes = kBytes;
    out.regions = {
        {"layer-0",
         TransferRef::HostBytes(first, first_size, mori::io::MemoryLocationType::GPU, device),
         kHalf, 1, 0, 0, "kv"},
        {"layer-1",
         TransferRef::HostBytes(second, second_size, mori::io::MemoryLocationType::GPU, device),
         kHalf, 1, kHalf, 1, "kv"},
    };
    out.layout_id = engine::ComputeLayoutId(out);
    return out;
  };
  auto source_geometry =
      geometry(source_region0, source_region_bytes, source_region1, source_region_bytes, 0);
  auto target_geometry = geometry(target0.ptr, target0.mapped, target1.ptr, target1.mapped, 1);
  engine::EngineClient source_client({address, "gpu-load-source", 10, 1000});
  engine::EngineClient target_client({address, "gpu-load-target", 10, 1000});
  source_client.AttachExecutor(&source_executor);
  target_client.AttachExecutor(&target_executor);
  engine::PoolEpoch epoch;
  auto source_registration =
      source_client.RegisterPool(0, "gpu-layer-load", epoch, source_geometry);
  auto target_registration =
      target_client.RegisterPool(1, "gpu-layer-load", epoch, target_geometry);
  ASSERT_TRUE(source_registration.success) << source_registration.error;
  ASSERT_TRUE(target_registration.success) << target_registration.error;
  std::string map_error;
  ASSERT_TRUE(target_executor.MapSharedDram(target_registration.dram_regions, &map_error))
      << map_error;

  auto source_reservation = source_client.Reserve("gpu-source", 1, 0);
  auto source_page =
      source_client.Take(source_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(source_page.size(), 1u);
  ASSERT_TRUE(source_client.Refund(source_reservation.id));
  auto hbm_first = Pattern(kHalf, 21);
  auto hbm_second = Pattern(kHalf, 91);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMemcpy(static_cast<uint8_t*>(source_region0) + source_page[0].page_id * kHalf,
                      hbm_first.data(), kHalf, hipMemcpyHostToDevice),
            hipSuccess);
  ASSERT_EQ(hipMemcpy(static_cast<uint8_t*>(source_region1) + source_page[0].page_id * kHalf,
                      hbm_second.data(), kHalf, hipMemcpyHostToDevice),
            hipSuccess);
  ASSERT_TRUE(source_client.Publish({{"gpu-load-hbm", {}, source_page[0]}}).success);

  UMBPConfig client_config;
  UMBPStandaloneProcessConfig standalone_config;
  standalone_config.address = address;
  standalone_config.startup_timeout_ms = 5000;
  client_config.standalone_process = standalone_config;
  auto ordinary = CreateUMBPClient(client_config);
  auto dram = Pattern(kBytes, 47);
  HostMemAllocator dram_allocator;
  HostBufferOptions dram_options;
  dram_options.backing = HostBufferBacking::kAnonymousShm;
  dram_options.prefault = false;
  auto dram_buffer = dram_allocator.Alloc(kBytes, dram_options);
  ASSERT_TRUE(dram_buffer.valid());
  std::memcpy(dram_buffer.ptr, dram.data(), dram.size());
  ASSERT_TRUE(ordinary->RegisterMemory(reinterpret_cast<uintptr_t>(dram_buffer.ptr),
                                       dram_buffer.mapped_size));
  ASSERT_TRUE(
      ordinary->Put("gpu-load-dram", reinterpret_cast<uintptr_t>(dram_buffer.ptr), dram.size()));

  auto plan = target_client.Lookup({"gpu-load-hbm", "gpu-load-dram"}, "model@rev/gpu-layout");
  ASSERT_EQ(plan.valid_prefix, 2u);
  EXPECT_EQ(plan.sources[0].kind, engine::ReadSourceKind::kLocalHbm);
  EXPECT_EQ(plan.sources[1].kind, engine::ReadSourceKind::kLocalDram);
  auto target_reservation = target_client.Reserve("gpu-target", 2, 0);
  auto targets = target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  auto prepared = target_client.PrepareLoad(plan, "gpu-layer-request", targets, 2);
  ASSERT_TRUE(prepared.success) << prepared.error;
  const uint64_t gather_before = DeviceGatherLaunchCount();
  auto handle = target_executor.SubmitLoad(prepared);
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->layer_count(), 2u);
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  hipStream_t forward_stream = nullptr;
  ASSERT_EQ(hipStreamCreateWithFlags(&forward_stream, hipStreamNonBlocking), hipSuccess);
  uint8_t* layer_bytes = nullptr;
  ASSERT_EQ(hipHostMalloc(&layer_bytes, 2 * kHalf), hipSuccess);
  for (size_t layer = 0; layer < 2; ++layer) {
    const auto event = reinterpret_cast<hipEvent_t>(handle->layer_event(layer));
    ASSERT_NE(event, nullptr);
    ASSERT_EQ(hipStreamWaitEvent(forward_stream, event, 0), hipSuccess);
    void* first_target = layer == 0 ? target0.ptr : target1.ptr;
    ASSERT_EQ(hipMemcpyAsync(layer_bytes,
                             static_cast<uint8_t*>(first_target) + targets[0].page_id * kHalf,
                             kHalf, hipMemcpyDeviceToHost, forward_stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(layer_bytes + kHalf,
                             static_cast<uint8_t*>(first_target) + targets[1].page_id * kHalf,
                             kHalf, hipMemcpyDeviceToHost, forward_stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(forward_stream), hipSuccess);
    const auto& expected_hbm = layer == 0 ? hbm_first : hbm_second;
    EXPECT_TRUE(std::equal(layer_bytes, layer_bytes + kHalf, expected_hbm.begin()));
    EXPECT_TRUE(
        std::equal(layer_bytes + kHalf, layer_bytes + 2 * kHalf, dram.begin() + layer * kHalf));
  }
  EXPECT_EQ(handle->Wait(), engine::LoadOutcome::kCompleted);
  EXPECT_EQ(DeviceGatherLaunchCount() - gather_before, 1u)
      << "registered host fragments should use one gathered submission";
  for (size_t layer = 0; layer < 2; ++layer) {
    EXPECT_TRUE(handle->layer_done(layer));
    EXPECT_EQ(handle->layer_outcome(layer), engine::LoadOutcome::kCompleted);
  }
  ASSERT_EQ(hipFreeHost(layer_bytes), hipSuccess);
  ASSERT_EQ(hipStreamDestroy(forward_stream), hipSuccess);
  std::vector<uint8_t> actual(kHalf);
  ASSERT_EQ(
      hipMemcpy(actual.data(), static_cast<uint8_t*>(target0.ptr) + targets[0].page_id * kHalf,
                kHalf, hipMemcpyDeviceToHost),
      hipSuccess);
  EXPECT_EQ(actual, hbm_first);
  ASSERT_EQ(
      hipMemcpy(actual.data(), static_cast<uint8_t*>(target1.ptr) + targets[0].page_id * kHalf,
                kHalf, hipMemcpyDeviceToHost),
      hipSuccess);
  EXPECT_EQ(actual, hbm_second);
  ASSERT_EQ(
      hipMemcpy(actual.data(), static_cast<uint8_t*>(target0.ptr) + targets[1].page_id * kHalf,
                kHalf, hipMemcpyDeviceToHost),
      hipSuccess);
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), dram.begin()));
  ASSERT_EQ(
      hipMemcpy(actual.data(), static_cast<uint8_t*>(target1.ptr) + targets[1].page_id * kHalf,
                kHalf, hipMemcpyDeviceToHost),
      hipSuccess);
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), dram.begin() + kHalf));
  ASSERT_TRUE(target_client.ReturnPages(targets));

  auto back_to_back_a = Pattern(kBytes, 9);
  auto back_to_back_b = Pattern(kBytes, 173);
  std::memcpy(dram_buffer.ptr, back_to_back_a.data(), back_to_back_a.size());
  ASSERT_TRUE(ordinary->Put("gpu-load-dram-a", reinterpret_cast<uintptr_t>(dram_buffer.ptr),
                            back_to_back_a.size()));
  std::memcpy(dram_buffer.ptr, back_to_back_b.data(), back_to_back_b.size());
  ASSERT_TRUE(ordinary->Put("gpu-load-dram-b", reinterpret_cast<uintptr_t>(dram_buffer.ptr),
                            back_to_back_b.size()));
  const auto prepare_dram = [&](std::string key, std::string request) {
    auto one_plan = target_client.Lookup({key}, "model@rev/gpu-layout");
    EXPECT_EQ(one_plan.valid_prefix, 1u);
    auto reservation = target_client.Reserve(request, 1, 0);
    auto page = target_client.Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
    EXPECT_EQ(page.size(), 1u);
    EXPECT_TRUE(target_client.Refund(reservation.id));
    return std::make_pair(page, target_client.PrepareLoad(one_plan, request, page, 1));
  };
  auto [back_to_back_target_a, prepared_a] = prepare_dram("gpu-load-dram-a", "gpu-back-to-back-a");
  auto [back_to_back_target_b, prepared_b] = prepare_dram("gpu-load-dram-b", "gpu-back-to-back-b");
  ASSERT_TRUE(prepared_a.success) << prepared_a.error;
  ASSERT_TRUE(prepared_b.success) << prepared_b.error;
  // The first load uses two descriptors. Split the second load into 600
  // equivalent fragments so it needs a larger staging slot while the first
  // upload is still queued on the same executor stream.
  for (auto& layer : prepared_b.layers) {
    ASSERT_EQ(layer.plans.size(), 1u);
    auto& transfer = layer.plans.front();
    ASSERT_EQ(transfer.sizes.size(), 1u);
    const size_t bytes = transfer.sizes.front();
    const size_t source = transfer.src_offsets.front();
    const size_t target = transfer.dst_offsets.front();
    transfer.src_offsets.clear();
    transfer.dst_offsets.clear();
    transfer.sizes.clear();
    constexpr size_t kParts = 300;
    for (size_t part = 0, offset = 0; part < kParts; ++part) {
      const size_t chunk = bytes / kParts + (part < bytes % kParts ? 1 : 0);
      transfer.src_offsets.push_back(source + offset);
      transfer.dst_offsets.push_back(target + offset);
      transfer.sizes.push_back(chunk);
      offset += chunk;
    }
  }
  const uint64_t back_to_back_gathers = DeviceGatherLaunchCount();
  auto load_a = target_executor.SubmitLoad(prepared_a);
  auto load_b = target_executor.SubmitLoad(prepared_b);
  ASSERT_NE(load_a, nullptr);
  ASSERT_NE(load_b, nullptr);
  ASSERT_EQ(load_a->Wait(), engine::LoadOutcome::kCompleted) << load_a->error();
  ASSERT_EQ(load_b->Wait(), engine::LoadOutcome::kCompleted) << load_b->error();
  EXPECT_EQ(DeviceGatherLaunchCount() - back_to_back_gathers, 2u);
  for (size_t layer = 0; layer < 2; ++layer) {
    void* layer_target = layer == 0 ? target0.ptr : target1.ptr;
    ASSERT_EQ(hipMemcpy(actual.data(),
                        static_cast<uint8_t*>(layer_target) +
                            back_to_back_target_a.front().page_id * kHalf,
                        kHalf, hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), back_to_back_a.begin() + layer * kHalf));
    ASSERT_EQ(hipMemcpy(actual.data(),
                        static_cast<uint8_t*>(layer_target) +
                            back_to_back_target_b.front().page_id * kHalf,
                        kHalf, hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), back_to_back_b.begin() + layer * kHalf));
  }
  ASSERT_TRUE(target_client.ReturnPages(back_to_back_target_a));
  ASSERT_TRUE(target_client.ReturnPages(back_to_back_target_b));

  VmmBuffer replacement(&allocator, source_region_bytes, 0);
  ASSERT_TRUE(replacement.valid());
  auto replacement_bytes = Pattern(kHalf, 133);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMemcpy(replacement.ptr, replacement_bytes.data(), kHalf, hipMemcpyHostToDevice),
            hipSuccess);
  auto replacement_export = allocator.Export(replacement.ptr);
  ASSERT_TRUE(replacement_export.has_value());
  struct stat replacement_identity {};
  ASSERT_EQ(fstat(replacement_export->fd, &replacement_identity), 0);
  engine::PeerVmmRegion replacement_region;
  replacement_region.executor_id = "gpu-load-source";
  replacement_region.region_id = "layer-0";
  replacement_region.mapped_size = replacement_export->mapped_size;
  replacement_region.region_size = source_region_bytes;
  replacement_region.mapping_id = static_cast<uint64_t>(replacement_identity.st_dev) ^
                                  (static_cast<uint64_t>(replacement_identity.st_ino) << 1);
  replacement_region.owner_device = 0;
  replacement_region.handle_kind = static_cast<uint32_t>(engine::VmmHandleKind::kPosixFd);
  ASSERT_TRUE(target_executor.ImportPeerRegion(replacement_region, replacement_export->fd));
  close(replacement_export->fd);
  TransferRef refreshed;
  refreshed.executor_id = "gpu-load-source";
  refreshed.region_id = "layer-0";
  ASSERT_TRUE(target_executor.Hydrate(&refreshed));
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  ASSERT_EQ(hipMemcpy(actual.data(), refreshed.host_ptr, kHalf, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(actual, replacement_bytes)
      << "same region id with a new mapping identity must not reuse the stale import";

  auto [event_failure_target, event_failure_prepared] =
      prepare_dram("gpu-load-dram-a", "gpu-event-sync-failure");
  (void)event_failure_target;
  ASSERT_TRUE(event_failure_prepared.success) << event_failure_prepared.error;
  target_executor.SetLoadEventSyncFailureForTest(true);
  auto event_failure = target_executor.SubmitLoad(event_failure_prepared);
  ASSERT_NE(event_failure, nullptr);
  EXPECT_EQ(event_failure->Wait(), engine::LoadOutcome::kUnknown);
  for (size_t layer = 0; layer < event_failure->layer_count(); ++layer) {
    EXPECT_TRUE(event_failure->layer_done(layer));
    EXPECT_EQ(event_failure->layer_outcome(layer), engine::LoadOutcome::kUnknown);
  }
  target_executor.SetLoadEventSyncFailureForTest(false);

  auto [enqueue_failure_target, enqueue_failure_prepared] =
      prepare_dram("gpu-load-dram-b", "gpu-enqueue-sync-failure");
  (void)enqueue_failure_target;
  ASSERT_TRUE(enqueue_failure_prepared.success) << enqueue_failure_prepared.error;
  target_executor.SetLoadEnqueueFailureForTest(true);
  target_executor.SetLoadStreamSyncFailureForTest(true);
  auto enqueue_failure = target_executor.SubmitLoad(enqueue_failure_prepared);
  ASSERT_NE(enqueue_failure, nullptr);
  EXPECT_EQ(enqueue_failure->outcome(), engine::LoadOutcome::kUnknown);
  for (size_t layer = 0; layer < enqueue_failure->layer_count(); ++layer) {
    EXPECT_TRUE(enqueue_failure->layer_done(layer));
    EXPECT_EQ(enqueue_failure->layer_outcome(layer), engine::LoadOutcome::kUnknown);
  }
  target_executor.SetLoadEnqueueFailureForTest(false);
  target_executor.SetLoadStreamSyncFailureForTest(false);

  VmmBuffer import_failure0(&allocator, (kPages + 1) * kHalf, 1);
  VmmBuffer import_failure1(&allocator, (kPages + 1) * kHalf, 1);
  ASSERT_TRUE(import_failure0.valid() && import_failure1.valid());
  engine::EngineExecutor import_failure_executor({"gpu-load-import-failure", 1});
  ASSERT_TRUE(import_failure_executor.RegisterOriginalRegion(
      "layer-0", TransferRef::HostBytes(import_failure0.ptr, import_failure0.mapped,
                                        mori::io::MemoryLocationType::GPU, 1)));
  ASSERT_TRUE(import_failure_executor.RegisterOriginalRegion(
      "layer-1", TransferRef::HostBytes(import_failure1.ptr, import_failure1.mapped,
                                        mori::io::MemoryLocationType::GPU, 1)));
  engine::EngineClient import_failure_client({address, "gpu-load-import-failure", 10, 1000});
  import_failure_client.AttachExecutor(&import_failure_executor);
  auto import_failure_registration =
      import_failure_client.RegisterPool(2, "gpu-layer-import-failure", epoch,
                                         geometry(import_failure0.ptr, import_failure0.mapped,
                                                  import_failure1.ptr, import_failure1.mapped, 1));
  ASSERT_TRUE(import_failure_registration.success) << import_failure_registration.error;
  auto import_failure_plan = import_failure_client.Lookup({"gpu-load-hbm"}, "model@rev/gpu-layout");
  ASSERT_TRUE(import_failure_plan.success);
  auto import_failure_reservation = import_failure_client.Reserve("gpu-import-failure", 1, 0);
  auto import_failure_target = import_failure_client.Take(import_failure_reservation.id,
                                                          engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(import_failure_client.Refund(import_failure_reservation.id));
  import_failure_executor.SetImportPeerRegionFailureForTest(true);
  auto import_failure = import_failure_client.PrepareLoad(import_failure_plan, "gpu-import-failure",
                                                          import_failure_target, 1);
  EXPECT_FALSE(import_failure.success);
  EXPECT_NE(import_failure.error.find("engine session stopped"), std::string::npos);
  EXPECT_FALSE(import_failure_client.healthy())
      << "an unacknowledged peer-region import must invalidate the session";
  import_failure_client.Close();
  import_failure_executor.Close();

  ordinary->DeregisterMemory(reinterpret_cast<uintptr_t>(dram_buffer.ptr));
  ordinary->Close();
  dram_allocator.Free(dram_buffer);
  target_client.Close();
  source_client.Close();
  target_executor.Close();
  source_executor.Close();
  running.Stop();
  RemoveSockets(address);
}

TEST(UmbpEngineGpu, RemoteLayeredLoadReadsOriginalVmmMrOverRdma) {
  if (VisibleGpuCount() < 2) GTEST_SKIP() << "requires two HIP GPUs";
  if (!std::filesystem::exists("/dev/infiniband")) GTEST_SKIP() << "no RDMA device nodes";
  ScopedEnv disable_xgmi("MORI_DISABLE_AUTO_XGMI", "1");
  MasterServerConfig master_config;
  master_config.listen_address = "0.0.0.0:0";
  auto master = std::make_unique<MasterServer>(std::move(master_config));
  RunningMaster running_master(master.get());
  for (int attempt = 0; attempt < 500 && master->GetBoundPort() == 0; ++attempt) {
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_NE(master->GetBoundPort(), 0);
  const std::string master_address = "localhost:" + std::to_string(master->GetBoundPort());
  auto pool_config = [&](std::string node) {
    PoolClientConfig config;
    config.master_config.node_id = std::move(node);
    config.master_config.node_address = "127.0.0.1";
    config.master_config.master_address = master_address;
    config.io_engine.host = "127.0.0.1";
    config.io_engine.port = 0;
    config.auto_peer_service_port = true;
    config.dram.buffer_sizes = {8 << 20};
    config.dram.shared = true;
    config.dram.use_hugepages = false;
    config.dram.prefault = false;
    config.dram_page_size = 4096;
    config.engine_hbm_ranks = 8;
    return config;
  };
  auto source_pool = std::make_unique<PoolClient>(pool_config("gpu-rdma-source-node"));
  auto target_pool = std::make_unique<PoolClient>(pool_config("gpu-rdma-target-node"));
  if (!source_pool->Init() || !target_pool->Init()) {
    GTEST_SKIP() << "RDMA-backed PoolClient initialization is unavailable";
  }

  const std::string source_address = UniqueAddress("umbp_gpu_remote_source");
  const std::string target_address = UniqueAddress("umbp_gpu_remote_target");
  RemoveSockets(source_address);
  RemoveSockets(target_address);
  GpuPoolEngineControl source_control(source_pool.get());
  GpuPoolEngineControl target_control(target_pool.get());
  engine::EngineSessionServer source_sessions(&source_control);
  engine::EngineSessionServer target_sessions(&target_control);
  std::string error;
  ASSERT_TRUE(source_sessions.Start(standalone::DeriveEngineSocketPath(source_address), &error))
      << error;
  ASSERT_TRUE(target_sessions.Start(standalone::DeriveEngineSocketPath(target_address), &error))
      << error;

  constexpr size_t kHalf = 1 << 20;
  constexpr size_t kBytes = 2 * kHalf;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer source0(&allocator, 3 * kHalf, 0);
  VmmBuffer source1(&allocator, 3 * kHalf, 0);
  VmmBuffer target0(&allocator, 3 * kHalf, 1);
  VmmBuffer target1(&allocator, 3 * kHalf, 1);
  ASSERT_TRUE(source0.valid() && source1.valid() && target0.valid() && target1.valid());
  engine::EngineExecutor source_executor({"gpu-rdma-source-executor", 0, "127.0.0.1", 0, 0});
  engine::EngineExecutor target_executor({"gpu-rdma-target-executor", 1, "127.0.0.1", 0, 0});
  if (source_executor.PackedEngineDesc().empty() || target_executor.PackedEngineDesc().empty()) {
    GTEST_SKIP() << "rank Mori-I/O RDMA executor is unavailable";
  }
  auto register_regions = [&](engine::EngineExecutor* executor, VmmBuffer* first, VmmBuffer* second,
                              int device) {
    return executor->RegisterOriginalRegion(
               "rdma-layer-0", TransferRef::HostBytes(first->ptr, first->mapped,
                                                      mori::io::MemoryLocationType::GPU, device)) &&
           executor->RegisterOriginalRegion(
               "rdma-layer-1", TransferRef::HostBytes(second->ptr, second->mapped,
                                                      mori::io::MemoryLocationType::GPU, device));
  };
  ASSERT_TRUE(register_regions(&source_executor, &source0, &source1, 0));
  ASSERT_TRUE(register_regions(&target_executor, &target0, &target1, 1));
  auto geometry = [&](VmmBuffer* first, VmmBuffer* second, int device) {
    engine::PageGeometry out;
    out.page_tokens = 1;
    out.num_pages = 2;
    out.object_bytes = kBytes;
    out.regions = {
        {"rdma-layer-0",
         TransferRef::HostBytes(first->ptr, first->mapped, mori::io::MemoryLocationType::GPU,
                                device),
         kHalf, 1, 0, 0, "kv"},
        {"rdma-layer-1",
         TransferRef::HostBytes(second->ptr, second->mapped, mori::io::MemoryLocationType::GPU,
                                device),
         kHalf, 1, kHalf, 1, "kv"},
    };
    out.layout_id = engine::ComputeLayoutId(out);
    return out;
  };
  engine::EngineClient source_client({source_address, "gpu-rdma-source-executor", 20, 2000});
  engine::EngineClient target_client({target_address, "gpu-rdma-target-executor", 20, 2000});
  source_client.AttachExecutor(&source_executor);
  target_client.AttachExecutor(&target_executor);
  engine::PoolEpoch epoch;
  auto source_registration =
      source_client.RegisterPool(0, "gpu-rdma-source-pool", epoch, geometry(&source0, &source1, 0));
  auto target_registration =
      target_client.RegisterPool(0, "gpu-rdma-target-pool", epoch, geometry(&target0, &target1, 1));
  ASSERT_TRUE(source_registration.success) << source_registration.error;
  ASSERT_TRUE(target_registration.success) << target_registration.error;

  auto source_reservation = source_client.Reserve("gpu-rdma-source", 1, 0);
  auto source_page =
      source_client.Take(source_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(source_page.size(), 1u);
  ASSERT_TRUE(source_client.Refund(source_reservation.id));
  auto expected0 = Pattern(kHalf, 17);
  auto expected1 = Pattern(kHalf, 79);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMemcpy(static_cast<uint8_t*>(source0.ptr) + source_page[0].page_id * kHalf,
                      expected0.data(), kHalf, hipMemcpyHostToDevice),
            hipSuccess);
  ASSERT_EQ(hipMemcpy(static_cast<uint8_t*>(source1.ptr) + source_page[0].page_id * kHalf,
                      expected1.data(), kHalf, hipMemcpyHostToDevice),
            hipSuccess);
  const std::string key = "gpu-remote-rdma-load";
  ASSERT_TRUE(source_client.Publish({{key, {}, source_page[0]}}).success);
  const std::string dram_key = "gpu-remote-rdma-dram";
  auto dram_expected = Pattern(kBytes, 43);
  // A normal Put is free to route to either registered node. Seed the source
  // backend directly so this test deterministically exercises remote DRAM.
  auto* source_dram = source_pool->Backends().Get(TierType::DRAM);
  ASSERT_NE(source_dram, nullptr);
  auto dram_allocation = source_dram->BatchAllocate({AllocateRequest{dram_key, kBytes}}).front();
  ASSERT_EQ(dram_allocation.outcome, AllocateOutcome::kSuccessAllocated);
  size_t copied = 0;
  for (const auto& page : dram_allocation.pages) {
    TransferRef buffer = source_dram->BufferRef(page.buffer_index);
    ASSERT_TRUE(buffer.HasHostPtr());
    const size_t bytes = std::min<size_t>(source_dram->PageSize(), dram_expected.size() - copied);
    const size_t offset = static_cast<size_t>(page.page_index) * source_dram->PageSize();
    ASSERT_LE(offset + bytes, buffer.size);
    std::memcpy(static_cast<uint8_t*>(buffer.host_ptr) + offset, dram_expected.data() + copied,
                bytes);
    copied += bytes;
  }
  ASSERT_EQ(copied, dram_expected.size());
  auto dram_commit =
      source_dram->BatchCommit({CommitRequest{dram_allocation.slot_id, dram_key}}).front();
  ASSERT_TRUE(dram_commit.success);
  ASSERT_EQ(dram_commit.bytes_committed, kBytes);
  source_pool->Master().FlushHeartbeat();
  for (int attempt = 0;
       attempt < 500 && (!target_pool->Exists(key) || !target_pool->Exists(dram_key)); ++attempt) {
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(target_pool->Exists(key));
  ASSERT_TRUE(target_pool->Exists(dram_key));
  ASSERT_TRUE(source_pool->EnginePool()->BatchContains({dram_key}).front());
  ASSERT_FALSE(target_pool->EnginePool()->BatchContains({dram_key}).front())
      << "remote DRAM fixture must not seed the reader's local backend";
  auto plan = target_client.Lookup({key, dram_key}, "model@rev/rdma-layout");
  ASSERT_TRUE(plan.success) << plan.error;
  ASSERT_EQ(plan.sources[0].kind, engine::ReadSourceKind::kRemoteHbm);
  ASSERT_EQ(plan.sources[1].kind, engine::ReadSourceKind::kRemoteDram);
  auto target_reservation = target_client.Reserve("gpu-rdma-target", 2, 0);
  auto target_page =
      target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  auto prepared = target_client.PrepareLoad(plan, "gpu-rdma-request", target_page, 2);
  ASSERT_TRUE(prepared.success) << prepared.error;
  auto has_bootstrap = [&](std::string_view executor_id) {
    return std::any_of(
        prepared.bootstraps.begin(), prepared.bootstraps.end(), [&](const auto& bootstrap) {
          return bootstrap.executor_id == executor_id && !bootstrap.engine_desc.empty();
        });
  };
  ASSERT_TRUE(has_bootstrap("gpu-rdma-source-executor"));
  ASSERT_TRUE(has_bootstrap("gpu-rdma-source-node"));
  for (const auto& layer : prepared.layers) {
    ASSERT_EQ(layer.plans.size(), 2u);
    for (const auto& transfer : layer.plans) {
      ASSERT_TRUE(transfer.src.HasMemoryDesc());
      ASSERT_TRUE(transfer.dst.HasMemoryDesc());
      EXPECT_NE(transfer.src.mem.engineKey, transfer.dst.mem.engineKey);
    }
  }
  auto handle = target_executor.SubmitLoad(prepared);
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->layer_count(), 2u);
  EXPECT_EQ(handle->layer_event(0), 0u);
  EXPECT_EQ(handle->layer_event(1), 0u) << "RDMA layers must not expose an unrecorded HIP event";
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  std::vector<uint8_t> actual(kHalf);
  for (size_t layer = 0; layer < 2; ++layer) {
    for (int attempt = 0; attempt < 1000 && !handle->layer_done(layer); ++attempt) {
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(handle->layer_done(layer));
    EXPECT_EQ(handle->layer_outcome(layer), engine::LoadOutcome::kCompleted)
        << handle->layer_error(layer);
    void* layer_target = layer == 0 ? target0.ptr : target1.ptr;
    ASSERT_EQ(hipMemcpy(actual.data(),
                        static_cast<uint8_t*>(layer_target) + target_page[0].page_id * kHalf, kHalf,
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(actual, layer == 0 ? expected0 : expected1);
    ASSERT_EQ(hipMemcpy(actual.data(),
                        static_cast<uint8_t*>(layer_target) + target_page[1].page_id * kHalf, kHalf,
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), dram_expected.begin() + layer * kHalf));
  }
  EXPECT_EQ(handle->Wait(), engine::LoadOutcome::kCompleted) << handle->error();

  target_client.Close();
  source_client.Close();
  target_executor.Close();
  source_executor.Close();
  target_sessions.Shutdown();
  source_sessions.Shutdown();
  target_pool->Shutdown();
  source_pool->Shutdown();
  RemoveSockets(source_address);
  RemoveSockets(target_address);
}

std::string StandaloneBinary() {
  if (const char* configured = std::getenv("UMBP_STANDALONE_BIN")) return configured;
  std::error_code error;
  auto executable = std::filesystem::canonical("/proc/self/exe", error);
  if (error) return {};
  auto root = executable;
  for (int i = 0; i < 5; ++i) root = root.parent_path();
  return (root / "src/umbp/umbp_standalone_server").string();
}

std::string StandaloneLibraryPath(const std::string& binary) {
  std::filesystem::path cmake = std::filesystem::path(binary).parent_path();
  if (cmake.filename() == "umbp") cmake = cmake.parent_path().parent_path();
  std::vector<std::filesystem::path> candidates = {cmake / "src/application", cmake / "src/io",
                                                   cmake / "src/metrics",     cmake / "src/umbp",
                                                   cmake / "src/pybind",      cmake / "src"};
  std::string out;
  for (const auto& candidate : candidates) {
    if (!std::filesystem::exists(candidate)) continue;
    if (!out.empty()) out.push_back(':');
    out += candidate.string();
  }
  if (const char* inherited = std::getenv("LD_LIBRARY_PATH"); inherited != nullptr && *inherited) {
    if (!out.empty()) out.push_back(':');
    out += inherited;
  }
  return out;
}

TEST(UmbpEngineGpu, SeparateNodeProcessF3DemotesAndLoadsSharedDram) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  const std::string binary = StandaloneBinary();
  if (binary.empty() || !std::filesystem::exists(binary)) {
    GTEST_SKIP() << "umbp_standalone_server binary is unavailable";
  }
  constexpr size_t kRegions = 61;
  constexpr size_t kPages = 8;
  struct RegionShape {
    size_t row_bytes;
    size_t rows_per_page;
    size_t PageBytes() const { return row_bytes * rows_per_page; }
  };
  std::vector<RegionShape> shapes;
  size_t object_bytes = 0;
  for (size_t i = 0; i < kRegions; ++i) {
    const RegionShape shape = i % 3 == 0   ? RegionShape{512, 8}
                              : i % 3 == 1 ? RegionShape{1024, 2}
                                           : RegionShape{256, 12};
    shapes.push_back(shape);
    object_bytes += shape.PageBytes();
  }
  const std::string address = UniqueAddress("umbp_engine_gpu_process_f3");
  RemoveSockets(address);
  const pid_t server = fork();
  ASSERT_GE(server, 0);
  if (server == 0) {
    const std::string library_path = StandaloneLibraryPath(binary);
    if (!library_path.empty()) setenv("LD_LIBRARY_PATH", library_path.c_str(), 1);
    unsetenv("UMBP_MASTER_ADDRESS");
    setenv("UMBP_DISTRIBUTED_MEDIUM", "DRAM", 1);
    setenv("UMBP_ENGINE_HBM_RANKS", "8", 1);
    setenv("UMBP_DRAM_CAPACITY", "67108864", 1);
    setenv("UMBP_DRAM_MAX_MR_BYTES", "1048576", 1);
    setenv("UMBP_DRAM_USE_HUGEPAGES", "0", 1);
    setenv("UMBP_DRAM_PREFAULT", "0", 1);
    const std::string page_size = std::to_string(object_bytes);
    setenv("UMBP_DISTRIBUTED_DRAM_PAGE_SIZE", page_size.c_str(), 1);
    setenv("UMBP_METRICS_PORT", "0", 1);
    execl(binary.c_str(), "umbp_standalone_server", address.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  ChildProcessGuard server_guard(server);
  const std::string engine_socket = standalone::DeriveEngineSocketPath(address);
  for (int attempt = 0; attempt < 500 && !std::filesystem::exists(engine_socket); ++attempt) {
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(std::filesystem::exists(engine_socket));

  auto& allocator = engine::GlobalKvVmmAllocator();
  std::vector<std::unique_ptr<VmmBuffer>> regions;
  regions.reserve(kRegions);
  engine::EngineExecutor executor({"separate-process-gpu-f3-rank", 0});
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = kPages;
  geometry.object_bytes = object_bytes;
  std::vector<std::vector<uint8_t>> expected(kPages * kRegions);
  size_t logical_offset = 0;
  for (size_t i = 0; i < kRegions; ++i) {
    const size_t page_bytes = shapes[i].PageBytes();
    regions.push_back(std::make_unique<VmmBuffer>(&allocator, (kPages + 1) * page_bytes, 0));
    ASSERT_TRUE(regions.back()->valid());
    const std::string id = "process-f3-region-" + std::to_string(i);
    ASSERT_TRUE(executor.RegisterOriginalRegion(
        id, TransferRef::HostBytes(regions.back()->ptr, regions.back()->mapped,
                                   mori::io::MemoryLocationType::GPU, 0)));
    geometry.regions.push_back(
        engine::PageRegion{id,
                           TransferRef::HostBytes(regions.back()->ptr, regions.back()->mapped,
                                                  mori::io::MemoryLocationType::GPU, 0),
                           shapes[i].row_bytes, shapes[i].rows_per_page, logical_offset,
                           static_cast<int32_t>(i), "kv"});
    for (size_t page = 0; page < kPages; ++page) {
      expected[page * kRegions + i] =
          Pattern(page_bytes, static_cast<uint8_t>(31 + i * 37 + page * 11));
    }
    logical_offset += page_bytes;
  }
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  ASSERT_TRUE(geometry.Validate());
  engine::EngineClient client({address, "separate-process-gpu-f3-rank", 20, 2000});
  client.AttachExecutor(&executor);
  engine::PoolEpoch epoch;
  auto registration = client.RegisterPool(0, "separate-process-gpu-f3-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto reservation = client.Reserve("process-f3-source", kPages, 0);
  auto source = client.Take(reservation.id, engine::ReservationPurpose::kLoad, kPages);
  ASSERT_EQ(source.size(), kPages);
  ASSERT_TRUE(client.Refund(reservation.id));
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  for (size_t page_index = 0; page_index < kPages; ++page_index) {
    for (size_t region = 0; region < kRegions; ++region) {
      const size_t page_bytes = shapes[region].PageBytes();
      auto* page =
          static_cast<uint8_t*>(regions[region]->ptr) + source[page_index].page_id * page_bytes;
      ASSERT_EQ(hipMemcpy(page, expected[page_index * kRegions + region].data(), page_bytes,
                          hipMemcpyHostToDevice),
                hipSuccess);
    }
  }
  std::vector<std::string> keys;
  std::vector<ResidentRecord> records;
  for (size_t i = 0; i < kPages; ++i) {
    keys.push_back("separate-process-gpu-f3-key-" + std::to_string(i));
    records.push_back(ResidentRecord{keys.back(), {}, source[i]});
  }
  ASSERT_TRUE(client.Publish(records).success);
  ASSERT_TRUE(client.ReturnPages(source));
  std::vector<engine::RetireTicket> released;
  for (int attempt = 0; attempt < 1000 && released.empty(); ++attempt) {
    released = client.PollRetires();
    if (released.empty()) std::this_thread::sleep_for(5ms);
  }
  ASSERT_EQ(released.size(), 1u);
  ASSERT_TRUE(client.AcknowledgeReleased({released.front().id}));

  for (size_t page_index = 0; page_index < kPages; ++page_index) {
    for (size_t region = 0; region < kRegions; ++region) {
      const size_t page_bytes = shapes[region].PageBytes();
      auto* page =
          static_cast<uint8_t*>(regions[region]->ptr) + source[page_index].page_id * page_bytes;
      ASSERT_EQ(hipMemset(page, 0, page_bytes), hipSuccess);
    }
  }
  auto plan = client.Lookup(keys, "process-f3");
  ASSERT_TRUE(plan.success) << plan.error;
  ASSERT_EQ(plan.valid_prefix, kPages);
  for (const auto& source_hint : plan.sources) {
    ASSERT_EQ(source_hint.kind, engine::ReadSourceKind::kLocalDram);
  }
  auto target_reservation = client.Reserve("process-f3-target", kPages, 0);
  auto target = client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, kPages);
  ASSERT_TRUE(client.Refund(target_reservation.id));
  auto prepared = client.PrepareLoad(plan, "process-f3-load", target, kPages);
  ASSERT_TRUE(prepared.success) << prepared.error;
  auto load = executor.SubmitLoad(prepared);
  ASSERT_NE(load, nullptr);
  ASSERT_EQ(load->Wait(), engine::LoadOutcome::kCompleted) << load->error();
  std::vector<uint8_t> actual;
  for (size_t page_index = 0; page_index < kPages; ++page_index) {
    for (size_t region = 0; region < kRegions; ++region) {
      const size_t page_bytes = shapes[region].PageBytes();
      actual.resize(page_bytes);
      auto* page =
          static_cast<uint8_t*>(regions[region]->ptr) + target[page_index].page_id * page_bytes;
      ASSERT_EQ(hipMemcpy(actual.data(), page, page_bytes, hipMemcpyDeviceToHost), hipSuccess);
      EXPECT_EQ(actual, expected[page_index * kRegions + region])
          << "page " << page_index << " region " << region;
    }
  }
  client.Close();
  executor.Close();
  RemoveSockets(address);
}

TEST(UmbpEngineGpu, KilledNodeFailsLocalOnlyRankWithinHeartbeatDeadline) {
  if (VisibleGpuCount() < 1) GTEST_SKIP() << "requires a HIP GPU";
  const std::string binary = StandaloneBinary();
  if (binary.empty() || !std::filesystem::exists(binary)) {
    GTEST_SKIP() << "umbp_standalone_server binary is unavailable";
  }
  const std::string address = UniqueAddress("umbp_engine_gpu_kill");
  RemoveSockets(address);
  const pid_t server = fork();
  ASSERT_GE(server, 0);
  if (server == 0) {
    const std::string library_path = StandaloneLibraryPath(binary);
    if (!library_path.empty()) setenv("LD_LIBRARY_PATH", library_path.c_str(), 1);
    unsetenv("UMBP_MASTER_ADDRESS");
    setenv("UMBP_DISTRIBUTED_MEDIUM", "DRAM", 1);
    setenv("UMBP_ENGINE_HBM_RANKS", "8", 1);
    setenv("UMBP_DRAM_CAPACITY", "1048576", 1);
    setenv("UMBP_DRAM_USE_HUGEPAGES", "0", 1);
    setenv("UMBP_DRAM_PREFAULT", "0", 1);
    setenv("UMBP_DISTRIBUTED_DRAM_PAGE_SIZE", "4096", 1);
    setenv("UMBP_METRICS_PORT", "0", 1);
    execl(binary.c_str(), "umbp_standalone_server", address.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  ChildProcessGuard server_guard(server);
  const std::string engine_socket = standalone::DeriveEngineSocketPath(address);
  bool started = false;
  for (int attempt = 0; attempt < 500 && !started; ++attempt) {
    started = std::filesystem::exists(engine_socket);
    if (!started) std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(started);

  constexpr size_t kBytes = 4096;
  engine::KvVmmAllocator allocator(engine::MakeHipVmmApi(engine::VmmHandleKind::kPosixFd));
  VmmBuffer hbm(&allocator, 2 * kBytes, 0);
  ASSERT_TRUE(hbm.valid());
  engine::EngineExecutor executor({"killed-node-executor", 0});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(hbm.ptr, hbm.mapped, mori::io::MemoryLocationType::GPU, 0)));
  auto geometry = SingleRegionGeometry(hbm.ptr, hbm.mapped, kBytes);
  engine::EngineClient client({address, "killed-node-executor", 20, 250});
  client.AttachExecutor(&executor);
  engine::PoolEpoch epoch;
  auto registration = client.RegisterPool(0, "killed-node-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor.MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto before_kill = client.Reserve("before-kill", 1, 0);
  ASSERT_TRUE(before_kill.success);
  ASSERT_TRUE(client.Refund(before_kill.id));

  ASSERT_EQ(kill(server, SIGKILL), 0);
  int status = 0;
  ASSERT_EQ(waitpid(server, &status, 0), server);
  server_guard.Reaped();
  const auto failed_at = std::chrono::steady_clock::now();
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (client.healthy() && std::chrono::steady_clock::now() < deadline) {
    auto local = client.Reserve("local-only", 1, 0);
    if (local.success) (void)client.Refund(local.id);
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(client.healthy());
  EXPECT_LT(std::chrono::steady_clock::now() - failed_at, 1s);
  EXPECT_FALSE(client.Reserve("after-fail-stop", 1, 0).success);
  EXPECT_EQ(client.Accounting().available, 0u);
  client.Close();
  executor.Close();
  RemoveSockets(address);
}

}  // namespace
}  // namespace mori::umbp

int main(int argc, char** argv) {
  if (argc == 6 && std::string(argv[1]) == "--vmm-import-child") {
    return mori::umbp::VmmImportChild(std::stoi(argv[2]), std::stoull(argv[3]),
                                      std::stoull(argv[4]), std::stoi(argv[5]));
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
