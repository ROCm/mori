#pragma once

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mori::umbp {

class PeerServiceServer {
 public:
  PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                    const std::vector<uint8_t>& engine_desc_bytes,
                    const std::vector<uint8_t>& dram_memory_desc_bytes,
                    const std::string& ssd_dir, size_t ssd_capacity,
                    uint64_t staging_base_offset);
  ~PeerServiceServer();

  void Start(uint16_t port);
  void Stop();

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;

  std::string ssd_dir_;
  size_t ssd_capacity_;
  size_t ssd_used_ = 0;
  std::mutex ssd_mutex_;

  std::vector<uint8_t> engine_desc_bytes_;
  std::vector<uint8_t> dram_memory_desc_bytes_;
  uint64_t staging_base_offset_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
