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
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"
#include "umbp_standalone.grpc.pb.h"

namespace mori::umbp {
namespace {

TEST(StandaloneShmIpcTest, AnonymousShmRegistryLookupMapsSameMemory) {
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = false;

  HostBufferHandle handle = allocator.Alloc(4096, opts);
  ASSERT_TRUE(handle.valid());
  EXPECT_EQ(handle.actual_backing, HostBufferBacking::kAnonymousShm);

  auto allocation =
      HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(handle.ptr), 128);
  ASSERT_TRUE(allocation.has_value());
  EXPECT_EQ(allocation->base, handle.ptr);
  EXPECT_GE(allocation->mapped_size, handle.mapped_size);
  ASSERT_GE(allocation->fd, 0);

  int dup_fd = dup(allocation->fd);
  ASSERT_GE(dup_fd, 0);
  void* mirror =
      mmap(nullptr, allocation->mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, dup_fd, 0);
  close(dup_fd);
  ASSERT_NE(mirror, MAP_FAILED);

  static_cast<unsigned char*>(handle.ptr)[17] = 0x5a;
  EXPECT_EQ(static_cast<unsigned char*>(mirror)[17], 0x5a);
  munmap(mirror, allocation->mapped_size);

  allocator.Free(handle);
  EXPECT_FALSE(handle.valid());
  EXPECT_FALSE(
      HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(allocation->base), 128)
          .has_value());
}

TEST(StandaloneShmIpcTest, ActiveAnonymousShmFreeIsDeferredUntilRelease) {
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = false;

  HostBufferHandle handle = allocator.Alloc(4096, opts);
  ASSERT_TRUE(handle.valid());
  static_cast<unsigned char*>(handle.ptr)[9] = 0x33;

  auto held = HostMemAllocator::AcquireShmAllocation(reinterpret_cast<uintptr_t>(handle.ptr), 4096);
  ASSERT_TRUE(held.has_value());
  int dup_fd = dup(held->fd);
  ASSERT_GE(dup_fd, 0);
  uintptr_t base = reinterpret_cast<uintptr_t>(held->base);

  allocator.Free(handle);
  EXPECT_FALSE(handle.valid());
  EXPECT_FALSE(HostMemAllocator::LookupShmAllocation(base, 16).has_value());

  HostMemAllocator::ReleaseShmAllocation(base);
  void* mirror = mmap(nullptr, held->mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, dup_fd, 0);
  close(dup_fd);
  ASSERT_NE(mirror, MAP_FAILED);
  EXPECT_EQ(static_cast<unsigned char*>(mirror)[9], 0x33);
  munmap(mirror, held->mapped_size);
}

bool FillSockaddr(const std::string& path, sockaddr_un* addr, socklen_t* addr_len) {
  if (path.size() >= sizeof(addr->sun_path)) return false;
  std::memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;
  std::strncpy(addr->sun_path, path.c_str(), sizeof(addr->sun_path) - 1);
  *addr_len = static_cast<socklen_t>(sizeof(sa_family_t) + path.size() + 1);
  return true;
}

TEST(StandaloneShmIpcTest, RawUdsFdRegistrationTransfersFd) {
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = false;
  HostBufferHandle handle = allocator.Alloc(4096, opts);
  ASSERT_TRUE(handle.valid());
  static_cast<unsigned char*>(handle.ptr)[3] = 0x7b;

  auto allocation =
      HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(handle.ptr), 4096);
  ASSERT_TRUE(allocation.has_value());

  std::string path = "/tmp/umbp_standalone_ipc_test_" + std::to_string(getpid()) + ".sock";
  unlink(path.c_str());

  int listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(listen_fd, 0);
  sockaddr_un addr;
  socklen_t addr_len = 0;
  ASSERT_TRUE(FillSockaddr(path, &addr, &addr_len));
  ASSERT_EQ(bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), addr_len), 0)
      << std::strerror(errno);
  ASSERT_EQ(listen(listen_fd, 1), 0) << std::strerror(errno);

  std::atomic<bool> receiver_ok{false};
  std::thread receiver([&]() {
    int accepted = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (accepted < 0) return;
    standalone::FdRegistrationMessage msg;
    std::string error;
    int received_fd = standalone::RecvFdRegistration(accepted, &msg, &error);
    if (received_fd >= 0 && std::string(msg.client_id) == "client-a" &&
        msg.worker_base == reinterpret_cast<uintptr_t>(handle.ptr) && msg.size >= 4096) {
      void* mirror = mmap(nullptr, static_cast<size_t>(msg.size), PROT_READ | PROT_WRITE,
                          MAP_SHARED, received_fd, 0);
      close(received_fd);
      if (mirror != MAP_FAILED) {
        receiver_ok.store(static_cast<unsigned char*>(mirror)[3] == 0x7b);
        munmap(mirror, static_cast<size_t>(msg.size));
      }
      standalone::SendStatus(accepted, 0);
    } else {
      if (received_fd >= 0) close(received_fd);
      standalone::SendStatus(accepted, -1);
    }
    close(accepted);
  });

  std::string error;
  int status = standalone::SendFdRegistration(path, allocation->fd, "client-a",
                                              reinterpret_cast<uintptr_t>(handle.ptr),
                                              allocation->mapped_size, 1000, &error);
  EXPECT_EQ(status, 0) << error;
  receiver.join();
  close(listen_fd);
  unlink(path.c_str());
  allocator.Free(handle);

  EXPECT_TRUE(receiver_ok.load());
}

TEST(StandaloneShmIpcTest, GpuRegistrationRejectsSsdBackedServer) {
  const std::string suffix = std::to_string(getpid());
  const std::string address = "unix:///tmp/umbp_standalone_gpu_reject_" + suffix + ".sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  const std::string ssd_path = "/tmp/umbp_standalone_gpu_reject_ssd_" + suffix;
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
  std::filesystem::remove_all(ssd_path);

  UMBPConfig config;
  config.dram.capacity_bytes = 1 << 20;
  config.ssd.enabled = true;
  config.ssd.storage_dir = ssd_path;
  config.ssd.capacity_bytes = 1 << 20;
  standalone::StandaloneServer server(config, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPStandalone::NewStub(channel);
  grpc::ClientContext context;
  ::umbp::RegisterMemoryRequest request;
  request.set_kind(::umbp::MEMORY_KIND_GPU_IPC);
  request.set_client_id("gpu-client");
  request.set_worker_base(0x1000);
  request.set_size(4096);
  ::umbp::BoolResponse response;
  const grpc::Status status = stub->RegisterMemory(&context, request, &response);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(response.ok());
  EXPECT_NE(response.error().find("SSD"), std::string::npos);

  server.Shutdown();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
  std::filesystem::remove_all(ssd_path);
}

TEST(StandaloneShmIpcTest, StandaloneClientUsesNonZeroOffsetsAndCanReregister) {
  const std::string address =
      "unix:///tmp/umbp_standalone_e2e_" + std::to_string(getpid()) + ".grpc.sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());

  UMBPConfig server_cfg;
  server_cfg.dram.capacity_bytes = 1 << 20;
  server_cfg.ssd.enabled = false;
  UMBPStandaloneProcessConfig sp_cfg;
  sp_cfg.address = address;
  sp_cfg.startup_timeout_ms = 5000;
  server_cfg.standalone_process = sp_cfg;

  standalone::StandaloneServer server(server_cfg, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  UMBPConfig client_cfg = server_cfg;
  auto client = CreateUMBPClient(client_cfg);
  ASSERT_EQ(client->GetDeploymentMode(), UMBPDeploymentMode::StandaloneProcess);

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = false;
  HostBufferHandle handle = allocator.Alloc(4096, opts);
  ASSERT_TRUE(handle.valid());
  auto* bytes = static_cast<unsigned char*>(handle.ptr);

  ASSERT_TRUE(client->RegisterMemory(reinterpret_cast<uintptr_t>(handle.ptr), handle.mapped_size));

  for (int i = 0; i < 16; ++i) bytes[32 + i] = static_cast<unsigned char>(i + 1);
  ASSERT_TRUE(client->Put("offset-key", reinterpret_cast<uintptr_t>(bytes + 32), 16));
  std::memset(bytes + 96, 0, 16);
  ASSERT_TRUE(client->Get("offset-key", reinterpret_cast<uintptr_t>(bytes + 96), 16));
  for (int i = 0; i < 16; ++i) EXPECT_EQ(bytes[96 + i], static_cast<unsigned char>(i + 1));

  client->DeregisterMemory(reinterpret_cast<uintptr_t>(handle.ptr));
  ASSERT_TRUE(client->RegisterMemory(reinterpret_cast<uintptr_t>(handle.ptr), handle.mapped_size));
  for (int i = 0; i < 8; ++i) bytes[128 + i] = static_cast<unsigned char>(0xa0 + i);
  ASSERT_TRUE(client->Put("reregister-key", reinterpret_cast<uintptr_t>(bytes + 128), 8));
  std::memset(bytes + 192, 0, 8);
  ASSERT_TRUE(client->Get("reregister-key", reinterpret_cast<uintptr_t>(bytes + 192), 8));
  for (int i = 0; i < 8; ++i) EXPECT_EQ(bytes[192 + i], static_cast<unsigned char>(0xa0 + i));

  client->DeregisterMemory(reinterpret_cast<uintptr_t>(handle.ptr));
  client->Close();
  allocator.Free(handle);
  server.Shutdown();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
}

TEST(StandaloneShmIpcTest, StandaloneClientResolvesAcrossMultipleRegions) {
  const std::string address =
      "unix:///tmp/umbp_standalone_multiregion_" + std::to_string(getpid()) + ".grpc.sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());

  UMBPConfig server_cfg;
  server_cfg.dram.capacity_bytes = 1 << 20;
  server_cfg.ssd.enabled = false;
  UMBPStandaloneProcessConfig sp_cfg;
  sp_cfg.address = address;
  sp_cfg.startup_timeout_ms = 5000;
  server_cfg.standalone_process = sp_cfg;

  standalone::StandaloneServer server(server_cfg, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  UMBPConfig client_cfg = server_cfg;
  auto client = CreateUMBPClient(client_cfg);
  ASSERT_EQ(client->GetDeploymentMode(), UMBPDeploymentMode::StandaloneProcess);

  // Two distinct, non-contiguous host shm regions from one client, mirroring a
  // hybrid HiCache worker registering several host KV pools per rank.
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  opts.prefault = false;
  HostBufferHandle region_a = allocator.Alloc(4096, opts);
  HostBufferHandle region_b = allocator.Alloc(8192, opts);
  ASSERT_TRUE(region_a.valid());
  ASSERT_TRUE(region_b.valid());
  auto* bytes_a = static_cast<unsigned char*>(region_a.ptr);
  auto* bytes_b = static_cast<unsigned char*>(region_b.ptr);

  ASSERT_TRUE(
      client->RegisterMemory(reinterpret_cast<uintptr_t>(region_a.ptr), region_a.mapped_size));
  // Registering region B must NOT drop region A (the single-region bug).
  ASSERT_TRUE(
      client->RegisterMemory(reinterpret_cast<uintptr_t>(region_b.ptr), region_b.mapped_size));

  // Put/Get resolve correctly in region A.
  for (int i = 0; i < 16; ++i) bytes_a[32 + i] = static_cast<unsigned char>(i + 1);
  ASSERT_TRUE(client->Put("key-a", reinterpret_cast<uintptr_t>(bytes_a + 32), 16));
  std::memset(bytes_a + 96, 0, 16);
  ASSERT_TRUE(client->Get("key-a", reinterpret_cast<uintptr_t>(bytes_a + 96), 16));
  for (int i = 0; i < 16; ++i) EXPECT_EQ(bytes_a[96 + i], static_cast<unsigned char>(i + 1));

  // Put/Get resolve correctly in region B.
  for (int i = 0; i < 24; ++i) bytes_b[64 + i] = static_cast<unsigned char>(0x40 + i);
  ASSERT_TRUE(client->Put("key-b", reinterpret_cast<uintptr_t>(bytes_b + 64), 24));
  std::memset(bytes_b + 4096, 0, 24);
  ASSERT_TRUE(client->Get("key-b", reinterpret_cast<uintptr_t>(bytes_b + 4096), 24));
  for (int i = 0; i < 24; ++i) EXPECT_EQ(bytes_b[4096 + i], static_cast<unsigned char>(0x40 + i));

  // A batch spanning both regions resolves per-element via region_bases.
  for (int i = 0; i < 8; ++i) bytes_a[200 + i] = static_cast<unsigned char>(0xa0 + i);
  for (int i = 0; i < 8; ++i) bytes_b[200 + i] = static_cast<unsigned char>(0xb0 + i);
  std::vector<std::string> keys{"batch-a", "batch-b"};
  std::vector<uintptr_t> srcs{reinterpret_cast<uintptr_t>(bytes_a + 200),
                              reinterpret_cast<uintptr_t>(bytes_b + 200)};
  std::vector<size_t> sizes{8, 8};
  std::vector<bool> put_ok = client->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(put_ok.size(), 2u);
  EXPECT_TRUE(put_ok[0]);
  EXPECT_TRUE(put_ok[1]);
  std::memset(bytes_a + 300, 0, 8);
  std::memset(bytes_b + 300, 0, 8);
  std::vector<uintptr_t> dsts{reinterpret_cast<uintptr_t>(bytes_a + 300),
                              reinterpret_cast<uintptr_t>(bytes_b + 300)};
  std::vector<bool> get_ok = client->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(get_ok.size(), 2u);
  EXPECT_TRUE(get_ok[0]);
  EXPECT_TRUE(get_ok[1]);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(bytes_a[300 + i], static_cast<unsigned char>(0xa0 + i));
  for (int i = 0; i < 8; ++i) EXPECT_EQ(bytes_b[300 + i], static_cast<unsigned char>(0xb0 + i));

  // One object can be assembled from, and read back into, ranges belonging to
  // different registered regions.
  for (int i = 0; i < 8; ++i) bytes_a[400 + i] = static_cast<unsigned char>(0xc0 + i);
  for (int i = 0; i < 8; ++i) bytes_b[400 + i] = static_cast<unsigned char>(0xd0 + i);
  auto range_put = client->BatchPutRanges(
      {"range-key"}, {16},
      {{reinterpret_cast<uintptr_t>(bytes_b + 400), reinterpret_cast<uintptr_t>(bytes_a + 400)}},
      {{8, 8}}, {{8, 0}});
  ASSERT_EQ(range_put, std::vector<bool>({true}));
  EXPECT_TRUE(client->Exists("range-key"));

  std::memset(bytes_a + 500, 0, 8);
  std::memset(bytes_b + 500, 0, 8);
  auto range_get = client->BatchGetRanges(
      {"range-key"},
      {{reinterpret_cast<uintptr_t>(bytes_a + 500), reinterpret_cast<uintptr_t>(bytes_b + 500)}},
      {{8, 8}}, {{8, 0}});
  ASSERT_EQ(range_get, std::vector<bool>({true}));
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(bytes_a[500 + i], static_cast<unsigned char>(0xd0 + i));
    EXPECT_EQ(bytes_b[500 + i], static_cast<unsigned char>(0xc0 + i));
  }

  // Malformed flattened range arrays are rejected before address resolution.
  auto raw_stub = ::umbp::UMBPStandalone::NewStub(
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  grpc::ClientContext malformed_context;
  ::umbp::BatchRangeDataRequest malformed_request;
  malformed_request.add_keys("malformed");
  malformed_request.add_range_counts(1);
  malformed_request.add_shm_offsets(0);
  // region_bases is deliberately missing.
  malformed_request.add_sizes(8);
  malformed_request.add_object_offsets(0);
  malformed_request.add_object_sizes(8);
  ::umbp::BatchBoolResponse malformed_response;
  ASSERT_TRUE(
      raw_stub->BatchPutRanges(&malformed_context, malformed_request, &malformed_response).ok());
  ASSERT_EQ(malformed_response.ok_size(), 1);
  EXPECT_FALSE(malformed_response.ok(0));

  // A pointer outside every registered region fails cleanly (no crash, no hit).
  HostBufferHandle unregistered = allocator.Alloc(4096, opts);
  ASSERT_TRUE(unregistered.valid());
  EXPECT_FALSE(client->Put("key-oob", reinterpret_cast<uintptr_t>(unregistered.ptr), 16));
  EXPECT_FALSE(client->Get("key-oob", reinterpret_cast<uintptr_t>(unregistered.ptr), 16));

  client->DeregisterMemory(reinterpret_cast<uintptr_t>(region_a.ptr));
  client->Close();
  allocator.Free(region_a);
  allocator.Free(region_b);
  allocator.Free(unregistered);
  server.Shutdown();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
}

TEST(StandaloneShmIpcTest, DeregistrationCannotUnmapAnInFlightDataOperation) {
  constexpr size_t kValueSize = 32ULL << 20;
  const std::string suffix = std::to_string(getpid());
  const std::string address =
      "unix:///tmp/umbp_standalone_deregister_race_" + suffix + ".grpc.sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());

  UMBPConfig config;
  config.dram.capacity_bytes = 2 * kValueSize;
  config.ssd.enabled = false;
  standalone::StandaloneServer server(config, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  HostMemAllocator allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;
  HostBufferHandle handle = allocator.Alloc(2 * kValueSize, options);
  ASSERT_TRUE(handle.valid());
  auto* bytes = static_cast<unsigned char*>(handle.ptr);
  std::memset(bytes, 0x5a, kValueSize);
  std::memset(bytes + kValueSize, 0, kValueSize);

  auto allocation = HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(handle.ptr),
                                                          handle.mapped_size);
  ASSERT_TRUE(allocation.has_value());
  constexpr char kClientId[] = "deregister-race-client";
  std::string registration_error;
  ASSERT_EQ(standalone::SendFdRegistration(fd_path, allocation->fd, kClientId,
                                           reinterpret_cast<uintptr_t>(handle.ptr),
                                           allocation->mapped_size, 5000, &registration_error),
            0)
      << registration_error;

  auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPStandalone::NewStub(channel);
  {
    grpc::ClientContext context;
    ::umbp::RegisterMemoryRequest request;
    request.set_kind(::umbp::MEMORY_KIND_HOST_SHM);
    request.set_client_id(kClientId);
    request.set_worker_base(reinterpret_cast<uintptr_t>(handle.ptr));
    request.set_size(allocation->mapped_size);
    ::umbp::BoolResponse response;
    const grpc::Status status = stub->RegisterMemory(&context, request, &response);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(response.ok()) << response.error();
  }

  {
    grpc::ClientContext context;
    ::umbp::PutRequest request;
    request.set_key("large-value");
    request.set_client_id(kClientId);
    request.set_region_base(reinterpret_cast<uintptr_t>(handle.ptr));
    request.set_shm_offset(0);
    request.set_size(kValueSize);
    ::umbp::BoolResponse response;
    const grpc::Status status = stub->Put(&context, request, &response);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(response.ok()) << response.error();
  }

  std::atomic<bool> get_started{false};
  grpc::Status get_status;
  ::umbp::BoolResponse get_response;
  std::thread getter([&]() {
    grpc::ClientContext context;
    ::umbp::GetRequest request;
    request.set_key("large-value");
    request.set_client_id(kClientId);
    request.set_region_base(reinterpret_cast<uintptr_t>(handle.ptr));
    request.set_shm_offset(kValueSize);
    request.set_size(kValueSize);
    get_started.store(true, std::memory_order_release);
    get_status = stub->Get(&context, request, &get_response);
  });

  while (!get_started.load(std::memory_order_acquire)) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  grpc::Status deregister_status;
  {
    grpc::ClientContext context;
    ::umbp::DeregisterMemoryRequest request;
    request.set_client_id(kClientId);
    ::umbp::Empty response;
    deregister_status = stub->DeregisterMemory(&context, request, &response);
  }
  getter.join();

  ASSERT_TRUE(deregister_status.ok());
  ASSERT_TRUE(get_status.ok());
  if (get_response.ok()) {
    EXPECT_EQ(std::memcmp(bytes, bytes + kValueSize, kValueSize), 0);
  }

  // Once deregistration returns, new operations must fail resolution rather
  // than reaching a stale server-side mapping.
  {
    grpc::ClientContext context;
    ::umbp::GetRequest request;
    request.set_key("large-value");
    request.set_client_id(kClientId);
    request.set_region_base(reinterpret_cast<uintptr_t>(handle.ptr));
    request.set_shm_offset(kValueSize);
    request.set_size(kValueSize);
    ::umbp::BoolResponse response;
    const grpc::Status status = stub->Get(&context, request, &response);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(response.ok());
  }

  allocator.Free(handle);
  server.Shutdown();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
}

TEST(StandaloneShmIpcTest, WritersCompleteUnderContinuousReaderLoad) {
  constexpr int kReaderCount = 8;
  constexpr size_t kValueSize = 2ULL << 20;
  const std::string address =
      "unix:///tmp/umbp_standalone_writer_liveness_" + std::to_string(getpid()) + ".grpc.sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());

  UMBPConfig config;
  config.dram.capacity_bytes = 8 * kValueSize;
  config.ssd.enabled = false;
  UMBPStandaloneProcessConfig standalone_config;
  standalone_config.address = address;
  standalone_config.startup_timeout_ms = 5000;
  config.standalone_process = standalone_config;

  standalone::StandaloneServer server(config, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  HostMemAllocator allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;

  std::vector<std::unique_ptr<IUMBPClient>> readers;
  std::vector<HostBufferHandle> reader_buffers;
  readers.reserve(kReaderCount);
  reader_buffers.reserve(kReaderCount);
  for (int i = 0; i < kReaderCount; ++i) {
    reader_buffers.push_back(allocator.Alloc(kValueSize, options));
    ASSERT_TRUE(reader_buffers.back().valid());
    auto client = CreateUMBPClient(config);
    ASSERT_TRUE(client->RegisterMemory(reinterpret_cast<uintptr_t>(reader_buffers.back().ptr),
                                       reader_buffers.back().mapped_size));
    readers.push_back(std::move(client));
  }

  HostBufferHandle writer_buffer = allocator.Alloc(2 * kValueSize, options);
  ASSERT_TRUE(writer_buffer.valid());
  auto writer = CreateUMBPClient(config);
  ASSERT_TRUE(writer->RegisterMemory(reinterpret_cast<uintptr_t>(writer_buffer.ptr),
                                     writer_buffer.mapped_size));
  auto* writer_bytes = static_cast<unsigned char*>(writer_buffer.ptr);
  std::memset(writer_bytes, 0x31, kValueSize);
  std::memset(writer_bytes + kValueSize, 0x72, kValueSize);
  ASSERT_TRUE(writer->Put("read-hot-key", reinterpret_cast<uintptr_t>(writer_bytes), kValueSize));

  std::atomic<bool> stop{false};
  std::atomic<int> readers_started{0};
  std::vector<std::thread> reader_threads;
  reader_threads.reserve(kReaderCount);
  for (int i = 0; i < kReaderCount; ++i) {
    reader_threads.emplace_back([&, i]() {
      readers_started.fetch_add(1, std::memory_order_release);
      while (!stop.load(std::memory_order_acquire)) {
        readers[i]->Get("read-hot-key", reinterpret_cast<uintptr_t>(reader_buffers[i].ptr),
                        kValueSize);
      }
    });
  }
  while (readers_started.load(std::memory_order_acquire) != kReaderCount) {
    std::this_thread::yield();
  }
  // Let every synchronous reader enter its steady RPC loop before introducing
  // a writer; otherwise this could accidentally test an uncontended lock.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto run_with_deadline = [&](auto operation) {
    std::atomic<bool> done{false};
    std::atomic<bool> ok{false};
    std::thread operation_thread([&]() {
      ok.store(operation(), std::memory_order_relaxed);
      done.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool completed_in_time = done.load(std::memory_order_acquire);
    if (!completed_in_time) stop.store(true, std::memory_order_release);
    operation_thread.join();
    return std::pair{completed_in_time, ok.load(std::memory_order_relaxed)};
  };

  const auto put_result = run_with_deadline([&]() {
    return writer->Put("writer-liveness-key",
                       reinterpret_cast<uintptr_t>(writer_bytes + kValueSize), kValueSize);
  });
  EXPECT_TRUE(put_result.first) << "Put starved behind continuous readers";
  EXPECT_TRUE(put_result.second);

  std::pair<bool, bool> clear_result{false, false};
  if (put_result.first) clear_result = run_with_deadline([&]() { return writer->Clear(); });
  EXPECT_TRUE(clear_result.first) << "Clear starved behind continuous readers";
  EXPECT_TRUE(clear_result.second);

  stop.store(true, std::memory_order_release);
  for (auto& thread : reader_threads) thread.join();

  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i]->DeregisterMemory(reinterpret_cast<uintptr_t>(reader_buffers[i].ptr));
    readers[i]->Close();
    allocator.Free(reader_buffers[i]);
  }
  writer->DeregisterMemory(reinterpret_cast<uintptr_t>(writer_buffer.ptr));
  writer->Close();
  allocator.Free(writer_buffer);

  server.Shutdown();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
}

TEST(StandaloneShmIpcTest, ShutdownDoesNotHangOnHalfOpenFdConnection) {
  const std::string address =
      "unix:///tmp/umbp_standalone_halfopen_" + std::to_string(getpid()) + ".grpc.sock";
  const std::string grpc_path = standalone::UnixPathFromGrpcAddress(address);
  const std::string fd_path = standalone::DeriveFdSocketPath(address);
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1 << 20;
  cfg.ssd.enabled = false;
  UMBPStandaloneProcessConfig sp_cfg;
  sp_cfg.address = address;
  sp_cfg.startup_timeout_ms = 5000;
  cfg.standalone_process = sp_cfg;

  standalone::StandaloneServer server(cfg, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&]() { server.Run(); });

  int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(sock, 0);
  sockaddr_un addr;
  socklen_t addr_len = 0;
  ASSERT_TRUE(FillSockaddr(fd_path, &addr, &addr_len));
  ASSERT_EQ(connect(sock, reinterpret_cast<sockaddr*>(&addr), addr_len), 0) << std::strerror(errno);

  std::atomic<bool> done{false};
  std::thread shutdown_thread([&]() {
    server.Shutdown();
    done.store(true);
  });

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(done.load());
  close(sock);
  shutdown_thread.join();
  server_thread.join();
  unlink(grpc_path.c_str());
  unlink(fd_path.c_str());
}

}  // namespace
}  // namespace mori::umbp
