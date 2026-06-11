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
#pragma once

#include <cctype>
#include <fstream>
#include <string>

#include "mori/utils/env_utils.hpp"

namespace mori {

// Identical for all processes on one physical node, distinct across nodes;
// empty if unavailable.
inline std::string ReadKernelBootId() {
  std::ifstream f("/proc/sys/kernel/random/boot_id");
  std::string id;
  if (f && std::getline(f, id)) {
    while (!id.empty() && std::isspace(static_cast<unsigned char>(id.back()))) id.pop_back();
    return id;
  }
  return {};
}

// Per-physical-node identity, by priority: MORI_NODE_ID override, boot_id, then
// hostname. boot_id keeps it correct when machines share one hostname.
inline std::string ResolveNodeId(const std::string& hostname) {
  if (auto nodeId = env::GetString("MORI_NODE_ID"); nodeId.has_value()) {
    return *nodeId;
  }
  std::string bootId = ReadKernelBootId();
  if (!bootId.empty()) return bootId;
  return hostname;
}

}  // namespace mori
