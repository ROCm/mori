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
#include <stdexcept>

#include "umbp/common/device_copy.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/pool_platform.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/sim/sim.h"
#include "umbp/sim/ssd.h"

namespace mori::umbp {
namespace {

class SimPoolPlatform final : public PoolPlatform {
 public:
  explicit SimPoolPlatform(UMBPSimConfig config) : config_(std::move(config)) {}
  Engines CreateTransferEngine(const PoolClientConfig& config,
                               const std::vector<BackendInstanceConfig>& backends) override {
    for (const auto& backend : backends) {
      if (backend.tier != TierType::DRAM && backend.tier != TierType::HBM &&
          backend.tier != TierType::SSD)
        throw std::invalid_argument("UMBP sim supports only DRAM/HBM/SSD storage");
    }
    auto engine = std::make_unique<SimTransferEngine>(
        config_.profile_path, config.master_config.node_id, config.staging_buffer_size);
    auto* peers = engine.get();
    sim_engine_ = peers;
    auto composite = std::make_unique<CompositeTransferEngine>();
    composite->AddEngine(std::move(engine));
    return {std::move(composite), peers};
  }
  std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& config,
                                               uint64_t page_size,
                                               const PoolClientConfig& pool_config) override {
    if (config.tier == TierType::SSD) {
      SsdBackend::Config ssd;
      ssd.page_size = page_size;
      ssd.staging_pages = SsdStagingPages(config.ssd_staging_buffer_slots,
                                          config.ssd.ssd.capacity_bytes, page_size);
      ssd.staging_use_hugepages = pool_config.ssd_staging_use_hugepages;
      ssd.staging_hugepage_size = pool_config.ssd_staging_hugepage_size;
      ssd.ssd = config.ssd;
      ssd.ssd.enabled = true;
      ssd.read_lease_ttl =
          GetEnvMilliseconds("UMBP_SSD_READ_LEASE_MS", std::chrono::milliseconds(3000), 1);
      return MakeSimSsdBackend(std::move(ssd), *sim_engine_);
    }
    const bool hbm = config.tier == TierType::HBM;
    if (!hbm && config.tier != TierType::DRAM) return nullptr;
    auto source = std::make_unique<SimPageMemorySource>(
        hbm ? mori::io::MemoryLocationType::GPU : mori::io::MemoryLocationType::CPU,
        hbm ? config.hbm.device : -1);
    static const auto read_lease_ttl = GetEnvMilliseconds(
        "UMBP_DRAM_READ_LEASE_MS", std::chrono::milliseconds(500), /*min_allowed=*/1);
    return std::make_unique<PageBackend>(config.tier, page_size, std::move(source),
                                         hbm ? config.hbm.buffer_sizes : config.dram.buffer_sizes,
                                         std::chrono::milliseconds{30000}, read_lease_ttl,
                                         std::chrono::milliseconds{200},
                                         hbm ? kDefaultDramMaxMrBytes : config.dram.max_mr_bytes,
                                         hbm ? std::vector<int>{} : config.dram.numa_nodes);
  }
  TransferRef LocalUserBytes(void* ptr, size_t size,
                             const TransferRef& registered_ref) const override {
    if (registered_ref.HasHostPtr())
      return TransferRef::HostBytes(ptr, size, registered_ref.loc, registered_ref.device);
    const PointerLocation location = DetectPointerLocation(ptr);
    if (!location.IsDevice()) return TransferRef::HostBytes(ptr, size);
    return TransferRef::HostBytes(ptr, size, mori::io::MemoryLocationType::GPU, location.device_id);
  }

  HostBufferHandle AllocateHost(size_t size, const HostBufferOptions&) override {
    return AllocateSimMemory(size);
  }
  void FreeHost(HostBufferHandle& buffer) override { HostMemAllocator().Free(buffer); }

 private:
  UMBPSimConfig config_;
  SimTransferEngine* sim_engine_ = nullptr;
};

}  // namespace

std::shared_ptr<PoolPlatform> MakeSimPoolPlatform(const UMBPSimConfig& sim) {
  return std::make_shared<SimPoolPlatform>(sim);
}

}  // namespace mori::umbp
