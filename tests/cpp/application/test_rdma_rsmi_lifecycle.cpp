#include <cstdio>
#include <memory>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/engine.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "rocm_smi/rocm_smi.h"

namespace {

const char* RsmiStatusString(rsmi_status_t status) {
  const char* msg = nullptr;
  rsmi_status_string(status, &msg);
  return msg ? msg : "unknown rocm smi error";
}

int RunRsmiTopologyProbe(const char* tag) {
  rsmi_status_t status = rsmi_init(0);
  if (status != RSMI_STATUS_SUCCESS) {
    std::printf("[%s] rsmi_init rc=%d %s\n", tag, static_cast<int>(status),
                RsmiStatusString(status));
    return 1;
  }

  uint32_t numGpus = 0;
  status = rsmi_num_monitor_devices(&numGpus);
  if (status != RSMI_STATUS_SUCCESS) {
    std::printf("[%s] rsmi_num_monitor_devices rc=%d %s\n", tag, static_cast<int>(status),
                RsmiStatusString(status));
    return 1;
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    uint64_t pciId = 0;
    status = rsmi_dev_pci_id_get(i, &pciId);
    if (status != RSMI_STATUS_SUCCESS) {
      std::printf("[%s] rsmi_dev_pci_id_get gpu=%u rc=%d %s\n", tag, i,
                  static_cast<int>(status), RsmiStatusString(status));
      return 1;
    }
  }

  uint32_t accessiblePairs = 0;
  for (uint32_t i = 0; i < numGpus; ++i) {
    for (uint32_t j = i + 1; j < numGpus; ++j) {
      bool accessible = false;
      status = rsmi_is_P2P_accessible(i, j, &accessible);
      if (status != RSMI_STATUS_SUCCESS) {
        std::printf("[%s] rsmi_is_P2P_accessible %u,%u rc=%d %s\n", tag, i, j,
                    static_cast<int>(status), RsmiStatusString(status));
        return 1;
      }
      if (!accessible) continue;
      ++accessiblePairs;

      uint64_t hops = 0;
      RSMI_IO_LINK_TYPE type{};
      uint64_t weight = 0;
      status = rsmi_topo_get_link_type(i, j, &hops, &type);
      if (status != RSMI_STATUS_SUCCESS) {
        std::printf("[%s] rsmi_topo_get_link_type %u,%u rc=%d %s\n", tag, i, j,
                    static_cast<int>(status), RsmiStatusString(status));
        return 1;
      }

      status = rsmi_topo_get_link_weight(i, j, &weight);
      if (status != RSMI_STATUS_SUCCESS) {
        std::printf("[%s] rsmi_topo_get_link_weight %u,%u rc=%d %s\n", tag, i, j,
                    static_cast<int>(status), RsmiStatusString(status));
        return 1;
      }
    }
  }

  status = rsmi_shut_down();
  std::printf("[%s] gpus=%u p2p_pairs=%u rsmi_shut_down rc=%d %s\n", tag, numGpus,
              accessiblePairs, static_cast<int>(status), RsmiStatusString(status));
  return status == RSMI_STATUS_SUCCESS ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;

  {
    mori::io::IOEngineConfig engineConfig{.host = "127.0.0.1", .port = 0};
    mori::io::RdmaBackendConfig rdmaConfig;
    mori::io::RdmaBackend backend("rdma_backend_probe", engineConfig, rdmaConfig);

    failures += RunRsmiTopologyProbe("rdma-backend-alive");
  }

  failures += RunRsmiTopologyProbe("rdma-backend-destroyed");

  {
    mori::io::IOEngineConfig engineConfig{.host = "127.0.0.1", .port = 0};
    mori::io::RdmaBackendConfig rdmaConfig;
    mori::io::IOEngine engine("io_engine_probe", engineConfig);
    engine.CreateBackend(mori::io::BackendType::RDMA, rdmaConfig);

    failures += RunRsmiTopologyProbe("io-engine-rdma-alive");
  }

  failures += RunRsmiTopologyProbe("io-engine-rdma-destroyed");

  return failures == 0 ? 0 : 1;
}
