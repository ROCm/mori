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
                      const std::vector<uint8_t>& engine_desc_bytes,
                      const std::vector<uint8_t>& dram_memory_desc_bytes,
                      const std::string& ssd_dir, size_t ssd_capacity, size_t& ssd_used,
                      std::mutex& ssd_mutex, uint64_t staging_base_offset)
      : ssd_staging_base_(ssd_staging_base),
        ssd_staging_size_(ssd_staging_size),
        engine_desc_bytes_(engine_desc_bytes),
        dram_memory_desc_bytes_(dram_memory_desc_bytes),
        ssd_dir_(ssd_dir),
        ssd_capacity_(ssd_capacity),
        ssd_used_(ssd_used),
        ssd_mutex_(ssd_mutex),
        staging_base_offset_(staging_base_offset) {}

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    response->set_engine_desc(
        std::string(engine_desc_bytes_.begin(), engine_desc_bytes_.end()));
    response->set_dram_memory_desc(
        std::string(dram_memory_desc_bytes_.begin(), dram_memory_desc_bytes_.end()));
    response->set_ssd_capacity(ssd_capacity_);
    {
      std::lock_guard<std::mutex> lock(ssd_mutex_);
      response->set_ssd_available(ssd_capacity_ - ssd_used_);
    }
    response->set_staging_base_offset(staging_base_offset_);
    return grpc::Status::OK;
  }

  grpc::Status CommitSsdWrite(grpc::ServerContext* /*context*/,
                              const ::umbp::CommitSsdWriteRequest* request,
                              ::umbp::CommitSsdWriteResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    if (ssd_used_ + request->size() > ssd_capacity_) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    const void* src = static_cast<const uint8_t*>(ssd_staging_base_) + request->staging_offset();
    std::string filename = ssd_dir_ + "/" + request->key() + ".bin";

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

    ssd_used_ += request->size();
    response->set_success(true);
    response->set_ssd_location_id(request->key() + ".bin");

    spdlog::info("[PeerService] CommitSsdWrite: key={}, size={}, file={}",
                 request->key(), request->size(), filename);
    return grpc::Status::OK;
  }

  grpc::Status PrepareSsdRead(grpc::ServerContext* /*context*/,
                              const ::umbp::PrepareSsdReadRequest* request,
                              ::umbp::PrepareSsdReadResponse* response) override {
    std::lock_guard<std::mutex> lock(ssd_mutex_);

    std::string filename = ssd_dir_ + "/" + request->ssd_location_id();

    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    void* dst = static_cast<uint8_t*>(ssd_staging_base_) + staging_base_offset_;
    ssize_t bytes_read = ::pread(fd, dst, request->size(), 0);
    ::close(fd);

    if (bytes_read < 0 || static_cast<size_t>(bytes_read) != request->size()) {
      response->set_success(false);
      return grpc::Status::OK;
    }

    response->set_success(true);
    response->set_staging_offset(staging_base_offset_);

    spdlog::info("[PeerService] PrepareSsdRead: key={}, ssd_location={}, size={}",
                 request->key(), request->ssd_location_id(), request->size());
    return grpc::Status::OK;
  }

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  const std::vector<uint8_t>& engine_desc_bytes_;
  const std::vector<uint8_t>& dram_memory_desc_bytes_;
  const std::string& ssd_dir_;
  size_t ssd_capacity_;
  size_t& ssd_used_;
  std::mutex& ssd_mutex_;
  uint64_t staging_base_offset_;
};

PeerServiceServer::PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                                     const std::vector<uint8_t>& engine_desc_bytes,
                                     const std::vector<uint8_t>& dram_memory_desc_bytes,
                                     const std::string& ssd_dir, size_t ssd_capacity,
                                     uint64_t staging_base_offset)
    : ssd_staging_base_(ssd_staging_base),
      ssd_staging_size_(ssd_staging_size),
      ssd_dir_(ssd_dir),
      ssd_capacity_(ssd_capacity),
      engine_desc_bytes_(engine_desc_bytes),
      dram_memory_desc_bytes_(dram_memory_desc_bytes),
      staging_base_offset_(staging_base_offset),
      service_(std::make_unique<UMBPPeerServiceImpl>(
          ssd_staging_base_, ssd_staging_size_, engine_desc_bytes_, dram_memory_desc_bytes_,
          ssd_dir_, ssd_capacity_, ssd_used_, ssd_mutex_, staging_base_offset_)) {}

PeerServiceServer::~PeerServiceServer() { Stop(); }

void PeerServiceServer::Start(uint16_t port) {
  std::string address = "0.0.0.0:" + std::to_string(port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  spdlog::info("[PeerService] Listening on {}", address);
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
