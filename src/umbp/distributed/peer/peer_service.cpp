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
#include "umbp/distributed/peer/peer_service.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/types.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/storage/local_storage_manager.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {
struct StagingSlot {
  bool in_use = false;
  uint64_t lease_id = 0;
  size_t allocated_size = 0;
  std::chrono::steady_clock::time_point allocated_at;
};

int AllocateSlot(std::vector<StagingSlot>& slots, std::atomic<uint64_t>& next_lease_id,
                 std::chrono::seconds lease_timeout, size_t request_size, StagingMetrics& metrics) {
  auto now = std::chrono::steady_clock::now();
  for (auto& slot : slots) {
    if (slot.in_use && now - slot.allocated_at > lease_timeout) {
      metrics.expired_reclaims.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN("[PeerService] Reclaiming expired slot (lease_id={})", slot.lease_id);
      slot.in_use = false;
    }
  }
  for (auto& slot : slots) {
    if (!slot.in_use) {
      slot.in_use = true;
      slot.lease_id = next_lease_id.fetch_add(1, std::memory_order_relaxed);
      slot.allocated_size = request_size;
      slot.allocated_at = now;
      return static_cast<int>(&slot - &slots[0]);
    }
  }
  return -1;
}

int FindSlotByLeaseId(const std::vector<StagingSlot>& slots, uint64_t lease_id) {
  for (size_t i = 0; i < slots.size(); ++i) {
    if (slots[i].in_use && slots[i].lease_id == lease_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool ReleaseSlotByLeaseId(std::vector<StagingSlot>& slots, uint64_t lease_id) {
  for (auto& slot : slots) {
    if (slot.in_use && slot.lease_id == lease_id) {
      slot.in_use = false;
      return true;
    }
  }
  return false;
}

uint64_t RemainingTtlMs(std::chrono::steady_clock::time_point alloc_time,
                        std::chrono::seconds lease_timeout) {
  auto elapsed = std::chrono::steady_clock::now() - alloc_time;
  auto remaining_ms = std::max<int64_t>(
      0, std::chrono::duration_cast<std::chrono::milliseconds>(lease_timeout - elapsed).count());
  return static_cast<uint64_t>(remaining_ms);
}
}  // namespace

class PeerServiceServer::UMBPPeerServiceImpl final : public ::umbp::UMBPPeer::Service {
 public:
  UMBPPeerServiceImpl(void* ssd_staging_base, size_t ssd_staging_size,
                      const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                      LocalStorageManager& storage, LocalBlockIndex& index, PoolClient& coordinator,
                      StagingMetrics& metrics, int num_read_slots, int num_write_slots,
                      int lease_timeout_s)
      : ssd_staging_base_(ssd_staging_base),
        ssd_staging_size_(ssd_staging_size),
        ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes),
        storage_(storage),
        index_(index),
        coordinator_(coordinator),
        metrics_(metrics),
        lease_timeout_(std::max(lease_timeout_s, 1)),
        num_read_slots_(std::max(num_read_slots, 1)),
        num_write_slots_(std::max(num_write_slots, 1)),
        read_region_base_(ssd_staging_size / 2),
        read_slot_size_((ssd_staging_size / 2) / static_cast<size_t>(std::max(num_read_slots, 1))),
        write_slot_size_((ssd_staging_size / 2) /
                         static_cast<size_t>(std::max(num_write_slots, 1))),
        read_slots_(std::max(num_read_slots, 1)),
        write_slots_(std::max(num_write_slots, 1)) {
    if (num_read_slots <= 0 || num_write_slots <= 0) {
      MORI_UMBP_ERROR("[PeerService] num_read_slots={} num_write_slots={} invalid, clamped to 1",
                      num_read_slots, num_write_slots);
    }
  }

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    response->set_ssd_staging_mem_desc(
        std::string(ssd_staging_mem_desc_bytes_.begin(), ssd_staging_mem_desc_bytes_.end()));
    response->set_ssd_staging_size(ssd_staging_size_);
    return grpc::Status::OK;
  }

  // ---- Write path: AllocateWriteSlot + CommitSsdWrite ----

  grpc::Status AllocateWriteSlot(grpc::ServerContext* /*context*/,
                                 const ::umbp::AllocateWriteSlotRequest* request,
                                 ::umbp::AllocateWriteSlotResponse* response) override {
    if (request->size() > write_slot_size_ || request->size() == 0) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    int slot_idx;
    uint64_t offset, lease_id;
    std::chrono::steady_clock::time_point alloc_time;
    {
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      slot_idx =
          AllocateSlot(write_slots_, next_lease_id_, lease_timeout_, request->size(), metrics_);
      if (slot_idx < 0) {
        metrics_.slot_full_rejects.fetch_add(1, std::memory_order_relaxed);
        response->set_success(false);
        return grpc::Status::OK;
      }
      offset = static_cast<uint64_t>(slot_idx) * write_slot_size_;
      lease_id = write_slots_[slot_idx].lease_id;
      alloc_time = write_slots_[slot_idx].allocated_at;
    }

    response->set_success(true);
    response->set_staging_offset(offset);
    response->set_lease_id(lease_id);
    response->set_lease_ttl_ms(RemainingTtlMs(alloc_time, lease_timeout_));
    return grpc::Status::OK;
  }

  grpc::Status CommitSsdWrite(grpc::ServerContext* /*context*/,
                              const ::umbp::CommitSsdWriteRequest* request,
                              ::umbp::CommitSsdWriteResponse* response) override {
    const uint64_t commit_lease_id = request->lease_id();
    uint64_t offset;
    {
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      int slot_idx = FindSlotByLeaseId(write_slots_, commit_lease_id);
      if (slot_idx < 0) {
        metrics_.invalid_lease_rejects.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: invalid/expired lease_id={}",
                        commit_lease_id);
        response->set_success(false);
        return grpc::Status::OK;
      }
      offset = static_cast<uint64_t>(slot_idx) * write_slot_size_;

      if (request->store_index() != 0) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: store_index {} != 0, rejected",
                        request->store_index());
        ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
        response->set_success(false);
        return grpc::Status::OK;
      }
      if (request->size() > write_slots_[slot_idx].allocated_size) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: size {} > allocated {}", request->size(),
                        write_slots_[slot_idx].allocated_size);
        ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
        response->set_success(false);
        return grpc::Status::OK;
      }
      if (request->staging_offset() != offset) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: staging_offset {} != slot offset {}",
                        request->staging_offset(), offset);
        ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
        response->set_success(false);
        return grpc::Status::OK;
      }
    }

    const std::string& key = request->key();
    const size_t size = request->size();

    auto existing = index_.Lookup(key);
    if (existing.has_value() && coordinator_.IsRegistered(key)) {
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
      response->set_success(true);
      return grpc::Status::OK;
    }

    const void* src = static_cast<const uint8_t*>(ssd_staging_base_) + offset;
    if (!existing.has_value()) {
      bool ok = storage_.Write(key, src, size, StorageTier::LOCAL_SSD);
      if (!ok) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: local SSD write failed for '{}'", key);
        std::lock_guard<std::mutex> lock(write_slots_mutex_);
        ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
        response->set_success(false);
        return grpc::Status::OK;
      }
      index_.Insert(key, {StorageTier::LOCAL_SSD, 0, size});
    }

    auto* ssd = storage_.GetTier(StorageTier::LOCAL_SSD);
    auto loc_id = ssd ? ssd->GetLocationId(key) : std::nullopt;
    if (!loc_id.has_value()) {
      storage_.Evict(key);
      index_.Remove(key);
      MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: GetLocationId failed for '{}'", key);
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
      response->set_success(false);
      return grpc::Status::OK;
    }
    std::string location_id = "0:" + *loc_id;

    bool finalized = coordinator_.FinalizeAllocation(key, size, location_id, TierType::SSD,
                                                     request->allocation_id());
    if (!finalized) {
      storage_.Evict(key);
      index_.Remove(key);
      MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: FinalizeAllocation failed for '{}'", key);
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
      response->set_success(false);
      return grpc::Status::OK;
    }

    {
      std::lock_guard<std::mutex> lock(write_slots_mutex_);
      ReleaseSlotByLeaseId(write_slots_, commit_lease_id);
    }
    response->set_success(true);
    response->set_ssd_location_id(location_id);
    MORI_UMBP_INFO("[PeerService] CommitSsdWrite: key={}, size={}, location={}", key, size,
                   location_id);
    return grpc::Status::OK;
  }

  // ---- Read path: PrepareSsdRead + ReleaseSsdLease ----

  grpc::Status PrepareSsdRead(grpc::ServerContext* /*context*/,
                              const ::umbp::PrepareSsdReadRequest* request,
                              ::umbp::PrepareSsdReadResponse* response) override {
    if (request->size() > read_slot_size_ || request->size() == 0) {
      MORI_UMBP_ERROR("[PeerService] PrepareSsdRead: size {} invalid (slot_size={})",
                      request->size(), read_slot_size_);
      response->set_success(false);
      return grpc::Status::OK;
    }

    int slot_idx;
    uint64_t offset, lease_id;
    std::chrono::steady_clock::time_point alloc_time;
    {
      std::lock_guard<std::mutex> lock(read_slots_mutex_);
      slot_idx =
          AllocateSlot(read_slots_, next_lease_id_, lease_timeout_, request->size(), metrics_);
      if (slot_idx < 0) {
        metrics_.slot_full_rejects.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN("[PeerService] PrepareSsdRead: no free staging slots");
        response->set_success(false);
        return grpc::Status::OK;
      }
      offset = read_region_base_ + static_cast<uint64_t>(slot_idx) * read_slot_size_;
      lease_id = read_slots_[slot_idx].lease_id;
      alloc_time = read_slots_[slot_idx].allocated_at;
    }

    void* dst = static_cast<uint8_t*>(ssd_staging_base_) + offset;
    bool ok = storage_.ReadIntoPtrNoPromote(request->key(), reinterpret_cast<uintptr_t>(dst),
                                            request->size());
    if (!ok) {
      std::lock_guard<std::mutex> lock(read_slots_mutex_);
      ReleaseSlotByLeaseId(read_slots_, lease_id);
      response->set_success(false);
      return grpc::Status::OK;
    }

    response->set_success(true);
    response->set_staging_offset(offset);
    response->set_lease_id(lease_id);
    response->set_lease_ttl_ms(RemainingTtlMs(alloc_time, lease_timeout_));
    MORI_UMBP_INFO("[PeerService] PrepareSsdRead: key={}, slot={}, offset={}, lease_id={}",
                   request->key(), slot_idx, offset, lease_id);
    return grpc::Status::OK;
  }

  grpc::Status ReleaseSsdLease(grpc::ServerContext* /*context*/,
                               const ::umbp::ReleaseSsdLeaseRequest* request,
                               ::umbp::ReleaseSsdLeaseResponse* response) override {
    std::lock_guard<std::mutex> lock(read_slots_mutex_);
    int idx = FindSlotByLeaseId(read_slots_, request->lease_id());
    if (idx >= 0) {
      read_slots_[idx].in_use = false;
      response->set_success(true);
    } else {
      response->set_success(false);
    }
    return grpc::Status::OK;
  }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  const std::vector<uint8_t>& ssd_staging_mem_desc_bytes_;
  LocalStorageManager& storage_;
  LocalBlockIndex& index_;
  PoolClient& coordinator_;
  StagingMetrics& metrics_;

  const std::chrono::seconds lease_timeout_;
  const int num_read_slots_;
  const int num_write_slots_;
  const uint64_t read_region_base_;
  const size_t read_slot_size_;
  const size_t write_slot_size_;

  std::mutex read_slots_mutex_;
  std::mutex write_slots_mutex_;
  std::vector<StagingSlot> read_slots_;
  std::vector<StagingSlot> write_slots_;
  std::atomic<uint64_t> next_lease_id_{1};
};

PeerServiceServer::PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                                     const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                                     LocalStorageManager& storage, LocalBlockIndex& index,
                                     PoolClient& coordinator, int num_read_slots,
                                     int num_write_slots, int lease_timeout_s)
    : ssd_staging_base_(ssd_staging_base),
      ssd_staging_size_(ssd_staging_size),
      storage_(storage),
      index_(index),
      coordinator_(coordinator),
      ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(
      ssd_staging_base_, ssd_staging_size_, ssd_staging_mem_desc_bytes_, storage_, index_,
      coordinator_, metrics_, num_read_slots, num_write_slots, lease_timeout_s);
}

PeerServiceServer::~PeerServiceServer() { Stop(); }

bool PeerServiceServer::Start(uint16_t port) {
  std::string address = "0.0.0.0:" + std::to_string(port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  if (!server_) {
    MORI_UMBP_ERROR("[PeerService] Failed to start on {} (port may be in use)", address);
    return false;
  }
  MORI_UMBP_INFO("[PeerService] Listening on {}", address);
  return true;
}

void PeerServiceServer::Stop() {
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
    MORI_UMBP_INFO("[PeerService] Shutting down");
    server_->Shutdown(deadline);
    server_.reset();
  }
}

}  // namespace mori::umbp
