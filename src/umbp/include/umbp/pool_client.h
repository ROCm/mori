#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/master_client.h"
#include "umbp/peer_service.h"
#include "umbp/types.h"

namespace mori::umbp {

struct PoolClientConfig {
  MasterClientConfig master_config;

  std::string io_engine_host;
  uint16_t io_engine_port = 0;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;

  void* exportable_dram_buffer = nullptr;
  size_t exportable_dram_buffer_size = 0;

  std::string exportable_ssd_dir;
  size_t exportable_ssd_capacity = 0;

  std::map<TierType, TierCapacity> tier_capacities;

  uint16_t peer_service_port = 0;
};

class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  bool Init();
  void Shutdown();

  bool Put(const std::string& key, const void* src, size_t size);
  bool Get(const std::string& key, void* dst, size_t size);
  bool Remove(const std::string& key);

  MasterClient& Master();
  bool IsInitialized() const;

 private:
  PoolClientConfig config_;
  bool initialized_ = false;

  std::unique_ptr<MasterClient> master_client_;

  std::unique_ptr<PeerServiceServer> peer_service_;

  // IO Engine (data plane)
  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_{};
  mori::io::MemoryDesc export_dram_mem_{};
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // Peer connections (lazy init, keyed by node_id)
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    mori::io::MemoryDesc dram_memory;
    uint64_t staging_base_offset = 0;
    bool engine_registered = false;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  PeerConnection& GetOrConnectPeer(
      const std::string& node_id, const std::string& peer_address,
      const std::vector<uint8_t>& engine_desc_bytes,
      const std::vector<uint8_t>& dram_memory_desc_bytes);

  bool RemoteDramWrite(PeerConnection& peer, const void* src, size_t size,
                       uint64_t offset);
  bool RemoteDramRead(PeerConnection& peer, void* dst, size_t size,
                      uint64_t offset);
  bool RemoteSsdWrite(PeerConnection& peer, const std::string& key,
                      const void* src, size_t size);
  bool RemoteSsdRead(PeerConnection& peer, const std::string& key,
                     const std::string& location_id, void* dst, size_t size);

  std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> location_cache_;

  bool PutLocalDram(const void* src, size_t size, uint64_t offset);
  bool GetLocalDram(void* dst, size_t size, uint64_t offset);

  bool PutLocalSsd(const std::string& key, const void* src, size_t size);
  bool GetLocalSsd(const std::string& key, void* dst, size_t size);
};

}  // namespace mori::umbp
