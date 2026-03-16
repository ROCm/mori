#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/master_client.h"
#include "umbp/peer_service.h"
#include "umbp/types.h"

namespace mori::umbp {

struct ExportableDram {
  void* buffer = nullptr;
  size_t size = 0;
};

struct ExportableSsd {
  std::string dir;
  size_t capacity = 0;
};

struct PoolClientConfig {
  MasterClientConfig master_config;

  std::string io_engine_host;
  uint16_t io_engine_port = 0;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;

  std::vector<ExportableDram> dram_buffers;
  std::vector<ExportableSsd> ssd_stores;

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

  bool RegisterMemory(void* ptr, size_t size);
  void DeregisterMemory(void* ptr);

  bool Put(const std::string& key, const void* src, size_t size,
           bool zero_copy = true);
  bool Get(const std::string& key, void* dst, size_t size,
           bool zero_copy = true);
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
  std::vector<mori::io::MemoryDesc> export_dram_mems_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // SSD staging buffer — separate from DRAM exportable buffers so that
  // PeerService SSD staging traffic does not conflict with Master-managed
  // DRAM tier offset allocations.
  std::unique_ptr<char[]> ssd_staging_buffer_;
  mori::io::MemoryDesc ssd_staging_mem_{};
  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;

  // Peer connections (lazy init, keyed by node_id)
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    std::vector<mori::io::MemoryDesc> dram_memories;
    bool engine_registered = false;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;

    // Dedicated SSD staging MemoryDesc, independent of dram_memories to avoid
    // offset conflicts between DRAM tier allocations and SSD staging traffic.
    mori::io::MemoryDesc ssd_staging_mem{};
    size_t ssd_staging_size = 0;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  PeerConnection& GetOrConnectPeer(
      const std::string& node_id, const std::string& peer_address,
      const std::vector<uint8_t>& engine_desc_bytes,
      const std::vector<uint8_t>& dram_memory_desc_bytes,
      uint32_t buffer_index = 0);

  bool RemoteDramWrite(PeerConnection& peer, uint32_t buffer_index,
                       const void* src, size_t size, uint64_t offset,
                       bool zero_copy);
  bool RemoteDramRead(PeerConnection& peer, uint32_t buffer_index,
                      void* dst, size_t size, uint64_t offset,
                      bool zero_copy);
  bool RemoteSsdWrite(PeerConnection& peer, const std::string& key,
                      const void* src, size_t size, bool zero_copy,
                      uint32_t store_index = 0);
  bool RemoteSsdRead(PeerConnection& peer, const std::string& key,
                     const std::string& location_id, void* dst, size_t size,
                     bool zero_copy);

  // Zero-copy registered memory regions
  struct RegisteredRegion {
    void* base;
    size_t size;
    mori::io::MemoryDesc mem_desc;
  };
  std::mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;

  std::optional<std::pair<mori::io::MemoryDesc, size_t>>
  FindRegisteredMemory(const void* ptr, size_t size);

  std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> location_cache_;

  bool PutLocalDram(uint32_t buffer_index, const void* src, size_t size,
                    uint64_t offset);
  bool GetLocalDram(uint32_t buffer_index, void* dst, size_t size,
                    uint64_t offset);

  bool PutLocalSsd(const std::string& key, const void* src, size_t size,
                   uint32_t store_index = 0);
  bool GetLocalSsd(const std::string& filename, void* dst, size_t size,
                   uint32_t store_index = 0);
};

}  // namespace mori::umbp
