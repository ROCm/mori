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
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"

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
