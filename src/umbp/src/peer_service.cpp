#include "umbp/peer_service.h"

#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cstring>

#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

class PeerServiceServer::UMBPPeerServiceImpl final : public ::umbp::UMBPPeer::Service {
 public:
  UMBPPeerServiceImpl(void* ssd_staging_base, size_t ssd_staging_size,
                      const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                      std::vector<SsdStore>& ssd_stores,
                      std::mutex& ssd_mutex)
      : ssd_staging_base_(ssd_staging_base),
        ssd_staging_size_(ssd_staging_size),
        ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes),
        ssd_stores_(ssd_stores),
        ssd_mutex_(ssd_mutex) {}

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    // Only return SSD staging info — engine_desc and dram_memory_desc
    // are already provided by Master in RoutePut/RouteGet responses.
    response->set_ssd_staging_mem_desc(
        std::string(ssd_staging_mem_desc_bytes_.begin(),
                    ssd_staging_mem_desc_bytes_.end()));
    response->set_ssd_staging_size(ssd_staging_size_);
    return grpc::Status::OK;
  }

  grpc::Status CommitSsdWrite(grpc::ServerContext* /*context*/,
                              const ::umbp::CommitSsdWriteRequest* request,
                              ::umbp::CommitSsdWriteResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    uint32_t idx = request->store_index();
    if (idx >= ssd_stores_.size()) {
      spdlog::error("[PeerService] CommitSsdWrite: store_index {} out of range (have {})",
                    idx, ssd_stores_.size());
      response->set_success(false);
      return grpc::Status::OK;
    }
    auto& target = ssd_stores_[idx];

    if (target.used + request->size() > target.capacity) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    if (request->staging_offset() + request->size() > ssd_staging_size_ / 2) {
      spdlog::error("[PeerService] CommitSsdWrite: staging_offset + size exceeds write region");
      response->set_success(false);
      return grpc::Status::OK;
    }

    const void* src = static_cast<const uint8_t*>(ssd_staging_base_) + request->staging_offset();
    std::string filename = target.dir + "/" + request->key() + ".bin";

    int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    ssize_t written = ::write(fd, src, request->size());
    if (written < 0 || static_cast<size_t>(written) != request->size()) {
      ::close(fd);
      response->set_success(false);
      return grpc::Status::OK;
    }

    ::fsync(fd);
    ::close(fd);

    target.used += request->size();
    response->set_success(true);
    response->set_ssd_location_id(request->key() + ".bin");

    spdlog::info("[PeerService] CommitSsdWrite: key={}, size={}, store={}, file={}",
                 request->key(), request->size(), idx, filename);
    return grpc::Status::OK;
  }

  grpc::Status PrepareSsdRead(grpc::ServerContext* /*context*/,
                              const ::umbp::PrepareSsdReadRequest* request,
                              ::umbp::PrepareSsdReadResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    // Read region occupies the second half of the staging buffer
    const uint64_t read_offset = ssd_staging_size_ / 2;

    if (request->size() > ssd_staging_size_ / 2) {
      spdlog::error("[PeerService] PrepareSsdRead: size {} exceeds read region {}", 
                    request->size(), ssd_staging_size_ / 2);
      response->set_success(false);
      return grpc::Status::OK;
    }

    // Parse store_index from ssd_location_id (format: "store_index:filename" or just "filename")
    const auto& loc_id = request->ssd_location_id();
    auto colon = loc_id.find(':');
    if (colon != std::string::npos) {
      try {
        uint32_t idx = static_cast<uint32_t>(std::stoul(loc_id.substr(0, colon)));
        std::string file_part = loc_id.substr(colon + 1);
        if (idx < ssd_stores_.size()) {
          std::string filepath = ssd_stores_[idx].dir + "/" + file_part;
          int fd = ::open(filepath.c_str(), O_RDONLY);
          if (fd >= 0) {
            void* dst = static_cast<uint8_t*>(ssd_staging_base_) + read_offset;
            ssize_t bytes_read = ::pread(fd, dst, request->size(), 0);
            ::close(fd);
            if (bytes_read >= 0 && static_cast<size_t>(bytes_read) == request->size()) {
              response->set_success(true);
              response->set_staging_offset(read_offset);
              spdlog::info("[PeerService] PrepareSsdRead: key={}, store={}, file={}, size={}",
                           request->key(), idx, file_part, request->size());
              return grpc::Status::OK;
            }
          }
        }
      } catch (...) {}
    }

    // Fallback: search all stores (backward compat for plain "filename" format)
    for (const auto& s : ssd_stores_) {
      std::string filename = s.dir + "/" + loc_id;
      int fd = ::open(filename.c_str(), O_RDONLY);
      if (fd < 0) continue;

      void* dst = static_cast<uint8_t*>(ssd_staging_base_) + read_offset;
      ssize_t bytes_read = ::pread(fd, dst, request->size(), 0);
      ::close(fd);

      if (bytes_read >= 0 && static_cast<size_t>(bytes_read) == request->size()) {
        response->set_success(true);
        response->set_staging_offset(read_offset);
        spdlog::info("[PeerService] PrepareSsdRead: key={}, ssd_location={}, size={}, dir={}",
                     request->key(), loc_id, request->size(), s.dir);
        return grpc::Status::OK;
      }
    }

    response->set_success(false);
    return grpc::Status::OK;
  }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  const std::vector<uint8_t>& ssd_staging_mem_desc_bytes_;
  std::vector<SsdStore>& ssd_stores_;
  std::mutex& ssd_mutex_;
};

PeerServiceServer::PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                                     const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                                     const std::vector<std::string>& ssd_dirs,
                                     const std::vector<size_t>& ssd_capacities)
    : ssd_staging_base_(ssd_staging_base),
      ssd_staging_size_(ssd_staging_size),
      ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes) {
  for (size_t i = 0; i < ssd_dirs.size(); ++i) {
    SsdStore store;
    store.dir = ssd_dirs[i];
    store.capacity = (i < ssd_capacities.size()) ? ssd_capacities[i] : 0;
    store.used = 0;
    ssd_stores_.push_back(std::move(store));
  }
  service_ = std::make_unique<UMBPPeerServiceImpl>(
      ssd_staging_base_, ssd_staging_size_,
      ssd_staging_mem_desc_bytes_, ssd_stores_, ssd_mutex_);
}

PeerServiceServer::~PeerServiceServer() { Stop(); }

bool PeerServiceServer::Start(uint16_t port) {
  std::string address = "0.0.0.0:" + std::to_string(port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  if (!server_) {
    spdlog::error("[PeerService] Failed to start on {} (port may be in use)", address);
    return false;
  }
  spdlog::info("[PeerService] Listening on {}", address);
  return true;
}

void PeerServiceServer::Stop() {
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
    spdlog::info("[PeerService] Shutting down");
    server_->Shutdown(deadline);
    server_.reset();
  }
}

}  // namespace mori::umbp
