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
