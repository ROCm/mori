#include <cstdio>
#include <memory>

#include "mori/application/transport/rdma/rdma.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "rocm_smi/rocm_smi.h"

namespace {

const char* RsmiStatusString(rsmi_status_t status) {
  const char* msg = nullptr;
  rsmi_status_string(status, &msg);
  return msg ? msg : "unknown rocm smi error";
}

int RunRsmiTopologyProbe(const char* tag) {
  std::printf("[%s] rsmi_init\n", tag);
  rsmi_status_t status = rsmi_init(0);
  std::printf("[%s] rsmi_init rc=%d %s\n", tag, static_cast<int>(status),
              RsmiStatusString(status));
  if (status != RSMI_STATUS_SUCCESS) return 1;

  uint32_t numGpus = 0;
  status = rsmi_num_monitor_devices(&numGpus);
  std::printf("[%s] rsmi_num_monitor_devices rc=%d %s count=%u\n", tag, static_cast<int>(status),
              RsmiStatusString(status), numGpus);
  if (status != RSMI_STATUS_SUCCESS) return 1;

  for (uint32_t i = 0; i < numGpus; ++i) {
    uint64_t pciId = 0;
    status = rsmi_dev_pci_id_get(i, &pciId);
    std::printf("[%s] rsmi_dev_pci_id_get gpu=%u rc=%d %s pci=0x%lx\n", tag, i,
                static_cast<int>(status), RsmiStatusString(status), pciId);
    if (status != RSMI_STATUS_SUCCESS) return 1;
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    for (uint32_t j = i + 1; j < numGpus; ++j) {
      bool accessible = false;
      status = rsmi_is_P2P_accessible(i, j, &accessible);
      std::printf("[%s] rsmi_is_P2P_accessible %u,%u rc=%d %s accessible=%d\n", tag, i, j,
                  static_cast<int>(status), RsmiStatusString(status), accessible ? 1 : 0);
      if (status != RSMI_STATUS_SUCCESS) return 1;
      if (!accessible) continue;

      uint64_t hops = 0;
      RSMI_IO_LINK_TYPE type{};
      uint64_t weight = 0;
      status = rsmi_topo_get_link_type(i, j, &hops, &type);
      std::printf("[%s] rsmi_topo_get_link_type %u,%u rc=%d %s hops=%lu type=%d\n", tag, i, j,
                  static_cast<int>(status), RsmiStatusString(status), hops, static_cast<int>(type));
      if (status != RSMI_STATUS_SUCCESS) return 1;

      status = rsmi_topo_get_link_weight(i, j, &weight);
      std::printf("[%s] rsmi_topo_get_link_weight %u,%u rc=%d %s weight=%lu\n", tag, i, j,
                  static_cast<int>(status), RsmiStatusString(status), weight);
      if (status != RSMI_STATUS_SUCCESS) return 1;
    }
  }

  status = rsmi_shut_down();
  std::printf("[%s] rsmi_shut_down rc=%d %s\n", tag, static_cast<int>(status),
              RsmiStatusString(status));
  return status == RSMI_STATUS_SUCCESS ? 0 : 1;
}

void PrintRdmaDevices(const mori::application::RdmaContext& ctx) {
  const auto& devices = ctx.GetRdmaDeviceList();
  auto activePorts = mori::application::GetActiveDevicePortList(devices);
  std::printf("[rdma] devices=%zu active_ports=%zu\n", devices.size(), activePorts.size());
  for (auto* device : devices) {
    std::printf("[rdma] device=%s ports=%d active_ports=", device->Name().c_str(),
                device->GetDevicePortNum());
    for (auto port : device->GetActivePortIds()) {
      std::printf("%u ", port);
    }
    std::printf("\n");
  }
}

}  // namespace

int main() {
  int failures = 0;

  failures += RunRsmiTopologyProbe("baseline");

  {
    mori::application::RdmaContext ctx(mori::application::RdmaBackendType::IBVerbs);
    PrintRdmaDevices(ctx);
    failures += RunRsmiTopologyProbe("rdma-context-alive");
  }

  failures += RunRsmiTopologyProbe("rdma-context-destroyed");

  {
    auto ctx =
        std::make_unique<mori::application::RdmaContext>(mori::application::RdmaBackendType::IBVerbs);
    PrintRdmaDevices(*ctx);

    mori::io::RdmaBackendConfig config;
    mori::io::RdmaManager manager(config, ctx.get());
    (void)ctx.release();

    failures += RunRsmiTopologyProbe("rdma-manager-alive");
  }

  failures += RunRsmiTopologyProbe("rdma-manager-destroyed");

  return failures == 0 ? 0 : 1;
}
