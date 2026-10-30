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

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/local/host_mem_allocator.h"

namespace mori::umbp {

// Same ownership/freeing contract as HostMemAllocator, without payload pages.
HostBufferHandle AllocateSimMemory(size_t size);

// Address identities only. PROT_NONE makes accidental payload access fail.
class SimPageMemorySource final : public PageMemorySource {
 public:
  SimPageMemorySource(mori::io::MemoryLocationType loc, int device) : loc_(loc), device_(device) {}
  ~SimPageMemorySource() override { Release(); }
  bool Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) override;
  void Release() override;
  mori::io::MemoryLocationType LocationType() const override { return loc_; }
  int Device() const override { return device_; }
  const char* Name() const override { return "SimPageMemorySource"; }

 private:
  mori::io::MemoryLocationType loc_;
  int device_;
  std::vector<Buffer> buffers_;
};

// Local copies finish in Submit; remote direct copies finish in Wait, like mori-io.
class SimTransferEngine final : public TransferEngine, public PeerDirectory {
 public:
  explicit SimTransferEngine(const std::string& profile_path, std::string node_id = "",
                             uint64_t bounce_bytes = 0);
  const char* Name() const override { return "SimTransferEngine"; }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override;
  TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) override;
  void Deregister(const TransferRef& ref) override;
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items = 0) const override;
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;
  // Virtual SSD payload I/O; metadata work remains in the existing SSD controllers.
  bool SimulateStorage(bool read, uint64_t bytes, size_t operations = 1, bool staged = true,
                       size_t cpu_threads = 4);
  void AddHostGatherRegion(void* base, size_t size);
  void ClearHostGatherRegions();

  std::vector<uint8_t> PackedLocalEngineDesc() const override;
  bool EnsureRemoteEngine(const std::string& node_id, const std::string& packed) override;
  bool HasRemoteEngine(const std::string& node_id) const override;
  void ForgetRemote(const std::string& node_id) override;
  void CacheRemoteBuffers(const std::string& node_id,
                          const std::vector<BufferMemoryDescBytes>& descs) override;
  bool HasRemoteBuffers(const std::string& node_id) const override;
  std::vector<TransferRef> RemoteBufferSnapshot(const std::string& node_id,
                                                uint32_t backend_id) const override;

 private:
  using Clock = std::chrono::steady_clock;
  struct Path {
    size_t channel;
    // Scalars in old profiles become one-element curves; index = threads - 1.
    std::vector<double> latency_ns;
    std::vector<double> bandwidth;
    std::vector<double> copy_latency_ns;
    double capacity;
    double registration_latency_ns;
    double log_size_latency_ns;
    double batch_log_size_latency_ns;
    bool serialize_latency;
    double cpu_bandwidth;
    size_t cpu_threads;
    double cpu_launch_latency_ns;
    double cpu_cleanup_bandwidth;
  };
  struct Reservation {
    Clock::time_point finish;
    Clock::time_point local_finish;
  };
  std::optional<Reservation> Reserve(const std::vector<TransferPlan>& plans,
                                     Clock::time_point submitted);
  static constexpr size_t kModelCount = 16;
  using Bytes = std::array<long double, kModelCount>;
  using Copies = std::array<size_t, kModelCount>;
  std::optional<Reservation> ReserveCosts(const Bytes& bytes, const Copies& copies,
                                          Clock::time_point submitted, bool serial_cpu = false,
                                          long double completion_ns = 0);
  void AdvanceGds(Clock::time_point now);  // mutex_ held; work is full-bandwidth nanoseconds.
  bool WaitGds(long double service_ns);
  bool SimulateGds(const std::vector<TransferPlan>& plans, long double bytes, size_t copies,
                   Clock::time_point submitted);
  // Six transfer paths, staging read/write, host-device gather and file paths.
  std::array<std::optional<Path>, kModelCount> paths_;
  std::mutex file_mutex_;
  std::unordered_map<int, size_t> file_refs_;
  std::mutex mutex_;
  std::array<Clock::time_point, kModelCount> available_{};
  struct GdsWork {
    long double remaining_ns;
    bool done = false;
  };
  std::vector<GdsWork*> gds_work_;
  Clock::time_point gds_updated_{};
  std::mutex gather_mutex_;
  std::vector<std::pair<uintptr_t, size_t>> gather_regions_;
  std::string engine_key_;
  std::atomic<uint32_t> next_memory_id_{1};
  struct RemoteNode {
    std::string engine_key;
    std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> buffers;
  };
  mutable std::mutex remotes_mutex_;
  std::unordered_map<std::string, RemoteNode> remotes_;
  SimPageMemorySource bounce_memory_{mori::io::MemoryLocationType::CPU, -1};
  TransferRef bounce_ref_;
  std::mutex bounce_mutex_;
};

}  // namespace mori::umbp
