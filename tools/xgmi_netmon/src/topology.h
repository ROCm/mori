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
// GPU discovery: cards with XGMI under /sys/class/drm, their gpu_metrics layout and peers.
#pragma once

#include <string>
#include <vector>

#include "gpu_metrics.h"

namespace xgmi {

struct Gpu {
  int idx = 0;               // position in card order (card0, card8, ... -> 0, 1, ...)
  std::string card;          // "card8"
  std::string pci;           // "0000:15:00.0"
  std::string metrics_path;  // .../device/gpu_metrics
  Layout layout;
  double link_max = 0;  // bytes/s per direction
  int peer[kNumLinks];  // XGMI port -> peer GPU idx, -1 if the port is not connected
};

struct PortLine {
  int src_node, src_port, dst_node;
};

// Parses xgmi_port_num lines like "01:05 ->  02:05". Returns false on a malformed line.
bool ParsePortNum(const std::string& text, std::vector<PortLine>* out, std::string* err);

// Lists GPUs below `drm_root` (normally /sys/class/drm). Returns false and sets *err on failure;
// an empty list with true means no GPU has XGMI.
bool Discover(const std::string& drm_root, std::vector<Gpu>* gpus, std::string* err);

}  // namespace xgmi
