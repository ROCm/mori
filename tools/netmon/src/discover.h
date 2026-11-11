// Find ionic NICs and map netdev <-> RDMA device <-> PCI function <-> firmware IPC node.
//
// On Salina the Ethernet function (e.g. 0000:09:00.0) and the tawk_ipc function
// (0000:08:00.0) sit under the same upstream switch port (0000:06:00.0, the "card" BDF that
// nicctl prints); they are matched by that grandparent PCI device.
#pragma once

#include <string>
#include <vector>

#include "util.h"

namespace netmon {

struct Nic {
  std::string netdev;
  std::string pci;
  std::string card;
  std::string ibdev;    // empty if none
  std::string ipc_dev;  // e.g. /dev/tawkipcdev002, empty if none
  std::string driver_version;
  std::string fw_version;
  long long speed_mbps = -1;
};

// sysfs_root is "" in production; tests may point it at a fake tree.
Status DiscoverNics(std::vector<Nic>* out, const std::string& sysfs_root = "");

}  // namespace netmon
