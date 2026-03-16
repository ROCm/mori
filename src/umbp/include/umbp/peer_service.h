#pragma once

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mori::umbp {

struct SsdStore {
  std::string dir;
  size_t capacity = 0;
  size_t used = 0;
};

class PeerServiceServer {
 public:
  PeerServiceServer(void* ssd_staging_base, size_t ssd_staging_size,
                    const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                    const std::vector<std::string>& ssd_dirs,
                    const std::vector<size_t>& ssd_capacities);
  ~PeerServiceServer();

  void Start(uint16_t port);
  void Stop();

 private:
  void* ssd_staging_base_;
  size_t ssd_staging_size_;

  std::vector<SsdStore> ssd_stores_;
  std::mutex ssd_mutex_;

  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;

  std::unique_ptr<grpc::Server> server_;

  class UMBPPeerServiceImpl;
  std::unique_ptr<UMBPPeerServiceImpl> service_;
};

}  // namespace mori::umbp
