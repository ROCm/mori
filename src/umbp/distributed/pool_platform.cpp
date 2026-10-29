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
#include "umbp/distributed/pool_platform.h"

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_copy.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/peer/backend/hbm_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/backend/ssd_backend.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/distributed/transfer/posix_engine.h"
#ifdef UMBP_ENABLE_GDS
#include "umbp/distributed/transfer/gds_engine.h"
#endif

namespace mori::umbp {

std::shared_ptr<PoolPlatform> MakeSimPoolPlatform(const UMBPSimConfig& sim);

namespace {

std::chrono::milliseconds DramReadLeaseTtl() {
  static const auto v = GetEnvMilliseconds("UMBP_DRAM_READ_LEASE_MS",
                                           std::chrono::milliseconds(500), /*min_allowed=*/1);
  return v;
}

// The SSD equivalent, which is a different number for a different reason: it has
// to cover an SSD read plus the RDMA, not just the RDMA. It is also the window a
// staging page stays claimed, so it sets this backend's read concurrency against
// a fixed arena -- shortening it frees pages and costs read coalescing.
std::chrono::milliseconds SsdReadLeaseTtl() {
  static const auto v = GetEnvMilliseconds("UMBP_SSD_READ_LEASE_MS",
                                           std::chrono::milliseconds(3000), /*min_allowed=*/1);
  return v;
}

class RealPoolPlatform final : public PoolPlatform {
 public:
  Engines CreateTransferEngine(const PoolClientConfig& config,
                               const std::vector<BackendInstanceConfig>&) override {
    // Local host, local GPU, file, then remote: preserve the real dispatch order.
    auto composite = std::make_unique<CompositeTransferEngine>();
    composite->AddEngine(std::make_unique<LocalCopyEngine>());
    auto hbm = std::make_unique<HbmCopyEngine>();
    auto* hbm_observer = hbm.get();
    composite->AddEngine(std::move(hbm));
    composite->AddEngine(std::make_unique<PosixEngine>());
#ifdef UMBP_ENABLE_GDS
    if (GdsEngine::Available()) {
      composite->AddEngine(std::make_unique<GdsEngine>());
    } else {
      MORI_UMBP_INFO("[PoolClient] libhipfile unavailable — SSD reads stay host-staged");
    }
#endif
    PeerDirectory* peers = nullptr;
    if (!config.io_engine.host.empty()) {
      mori::io::IOEngineConfig io_cfg;
      io_cfg.host = config.io_engine.host;
      io_cfg.port = config.io_engine.port;
      auto rdma = std::make_unique<MoriIoEngine>(config.master_config.node_id, io_cfg,
                                                 config.staging_buffer_size);
      if (!rdma->Init()) {
        MORI_UMBP_ERROR("[PoolClient] MoriIoEngine init failed on {}:{}", config.io_engine.host,
                        config.io_engine.port);
        return {};
      }
      peers = rdma.get();
      composite->AddEngine(std::move(rdma));
    }
    hbm_engine_ = hbm_observer;
    return {std::move(composite), peers};
  }

  std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& config,
                                               uint64_t page_size,
                                               const PoolClientConfig& pool_config) override {
    switch (config.tier) {
      case TierType::DRAM: {
        PageBackend::OwnershipConfig ownership;
        ownership.buffer_sizes = config.dram.buffer_sizes;
        ownership.use_hugepages = config.dram.use_hugepages;
        ownership.hugepage_size = config.dram.hugepage_size;
        ownership.numa_nodes = config.dram.numa_nodes;
        ownership.prefault = config.dram.prefault;
        ownership.numa_strict = config.dram.numa_strict;
        ownership.prefault_threads = config.dram.prefault_threads;
        ownership.max_mr_bytes = config.dram.max_mr_bytes;
        return MakePageBackend(TierType::DRAM, page_size, std::move(ownership),
                               /*pending_ttl=*/std::chrono::milliseconds{30000},
                               /*read_lease_ttl=*/DramReadLeaseTtl());
      }
      case TierType::HBM:
        return MakeHbmBackend(page_size, config.hbm.device, config.hbm.buffer_sizes,
                              /*pending_ttl=*/std::chrono::milliseconds{30000},
                              /*read_lease_ttl=*/DramReadLeaseTtl());
      case TierType::SSD: {
        SsdBackend::Config ssd_cfg;
        ssd_cfg.page_size = page_size;
        ssd_cfg.staging_pages = SsdStagingPages(config.ssd_staging_buffer_slots,
                                                config.ssd.ssd.capacity_bytes, page_size);
        ssd_cfg.staging_use_hugepages = pool_config.ssd_staging_use_hugepages;
        ssd_cfg.staging_hugepage_size = pool_config.ssd_staging_hugepage_size;
        ssd_cfg.ssd = config.ssd;
        ssd_cfg.ssd.enabled = true;
        ssd_cfg.read_lease_ttl = SsdReadLeaseTtl();
        return MakeSsdBackend(std::move(ssd_cfg));
      }
      case TierType::UNKNOWN:
        return nullptr;
    }
    return nullptr;
  }

  TransferRef LocalUserBytes(void* ptr, size_t size, const TransferRef&) const override {
    const PointerLocation location = DetectPointerLocation(ptr);
    if (!location.IsDevice()) return TransferRef::HostBytes(ptr, size);
    return TransferRef::HostBytes(ptr, size, mori::io::MemoryLocationType::GPU, location.device_id);
  }

  HostBufferHandle AllocateHost(size_t size, const HostBufferOptions& options) override {
    return HostMemAllocator().Alloc(size, options);
  }
  void FreeHost(HostBufferHandle& buffer) override { HostMemAllocator().Free(buffer); }
  void AddHostGatherRegion(void* ptr, size_t size) override {
    if (hbm_engine_) hbm_engine_->AddHostGatherRegion(ptr, size);
  }
  void ClearHostGatherRegions() override {
    if (hbm_engine_) hbm_engine_->ClearHostGatherRegions();
    hbm_engine_ = nullptr;
  }

 private:
  HbmCopyEngine* hbm_engine_ = nullptr;
};

}  // namespace

std::shared_ptr<PoolPlatform> MakePoolPlatform(const UMBPSimConfig& sim) {
  return sim.enabled ? MakeSimPoolPlatform(sim) : std::make_shared<RealPoolPlatform>();
}

}  // namespace mori::umbp
