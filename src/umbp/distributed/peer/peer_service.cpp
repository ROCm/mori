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

#include <cstring>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/types.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/storage/local_storage_manager.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

class PeerServiceServer::UMBPPeerServiceImpl final : public ::umbp::UMBPPeer::Service {
 public:
  UMBPPeerServiceImpl(void* ssd_staging_base, size_t ssd_staging_size,
                      const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                      LocalStorageManager& storage, LocalBlockIndex& index,
                      PoolClient& coordinator, std::mutex& ssd_mutex)
      : ssd_staging_base_(ssd_staging_base),
        ssd_staging_size_(ssd_staging_size),
        ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes),
        storage_(storage),
        index_(index),
        coordinator_(coordinator),
        ssd_mutex_(ssd_mutex) {}

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    // Only return SSD staging info — engine_desc and dram_memory_desc
    // are already provided by Master in RoutePut/RouteGet responses.
    response->set_ssd_staging_mem_desc(
        std::string(ssd_staging_mem_desc_bytes_.begin(), ssd_staging_mem_desc_bytes_.end()));
    response->set_ssd_staging_size(ssd_staging_size_);
    return grpc::Status::OK;
  }

  grpc::Status CommitSsdWrite(grpc::ServerContext* /*context*/,
                              const ::umbp::CommitSsdWriteRequest* request,
                              ::umbp::CommitSsdWriteResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    if (request->store_index() != 0) {
      MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: store_index {} != 0, rejected",
                      request->store_index());
      response->set_success(false);
      return grpc::Status::OK;
    }

    if (request->staging_offset() + request->size() > ssd_staging_size_ / 2) {
      MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: staging_offset + size exceeds write region");
      response->set_success(false);
      return grpc::Status::OK;
    }

    const std::string& key = request->key();
    const size_t size = request->size();
    auto existing = index_.Lookup(key);
    if (existing.has_value() && coordinator_.IsRegistered(key)) {
      response->set_success(true);
      return grpc::Status::OK;
    }

    const void* src = static_cast<const uint8_t*>(ssd_staging_base_) + request->staging_offset();
    if (!existing.has_value()) {
      bool ok = storage_.Write(key, src, size, StorageTier::LOCAL_SSD);
      if (!ok) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: local SSD write failed for '{}'", key);
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
      response->set_success(false);
      return grpc::Status::OK;
    }
    std::string location_id = "0:" + *loc_id;

    bool finalized =
        coordinator_.FinalizeAllocation(key, size, location_id, TierType::SSD, request->allocation_id());
    if (!finalized) {
      storage_.Evict(key);
      index_.Remove(key);
      MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: FinalizeAllocation failed for '{}'", key);
      response->set_success(false);
      return grpc::Status::OK;
    }

    response->set_success(true);
    response->set_ssd_location_id(location_id);
    MORI_UMBP_INFO("[PeerService] CommitSsdWrite: key={}, size={}, location={}", key, size,
                   location_id);
    return grpc::Status::OK;
  }

  grpc::Status PrepareSsdRead(grpc::ServerContext* /*context*/,
                              const ::umbp::PrepareSsdReadRequest* request,
                              ::umbp::PrepareSsdReadResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    // Read region occupies the second half of the staging buffer
    const uint64_t read_offset = ssd_staging_size_ / 2;

    if (request->size() > ssd_staging_size_ / 2) {
      MORI_UMBP_ERROR("[PeerService] PrepareSsdRead: size {} exceeds read region {}",
                      request->size(), ssd_staging_size_ / 2);
      response->set_success(false);
      return grpc::Status::OK;
    }

    void* dst = static_cast<uint8_t*>(ssd_staging_base_) + read_offset;
    bool ok = storage_.ReadIntoPtrNoPromote(request->key(), reinterpret_cast<uintptr_t>(dst),
                                            request->size());
    if (ok) {
      response->set_success(true);
      response->set_staging_offset(read_offset);
      MORI_UMBP_INFO("[PeerService] PrepareSsdRead: key={}, location={}, size={}", request->key(),
                     request->ssd_location_id(), request->size());
      return grpc::Status::OK;
    }

    response->set_success(false);
    return grpc::Status::OK;
  }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  const std::vector<uint8_t>& ssd_staging_mem_desc_bytes_;
  LocalStorageManager& storage_;
  LocalBlockIndex& index_;
  PoolClient& coordinator_;
  std::mutex& ssd_mutex_;
};

PeerServiceServer::PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                                     const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                                     LocalStorageManager& storage, LocalBlockIndex& index,
                                     PoolClient& coordinator)
    : ssd_staging_base_(ssd_staging_base),
      ssd_staging_size_(ssd_staging_size),
      storage_(storage),
      index_(index),
      coordinator_(coordinator),
      ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(
      ssd_staging_base_, ssd_staging_size_, ssd_staging_mem_desc_bytes_, storage_, index_,
      coordinator_, ssd_mutex_);
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
