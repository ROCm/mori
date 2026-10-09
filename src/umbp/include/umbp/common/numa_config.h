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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mori::umbp {

// Normalize the legacy -1 spelling at the configuration boundary. An empty
// list means no binding; a nonempty list contains distinct, nonnegative nodes.
inline std::vector<int> NormalizeNumaNodes(std::vector<int> nodes) {
  if (nodes == std::vector<int>{-1}) return {};
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i] < 0 ||
        std::find(nodes.begin(), nodes.begin() + i, nodes[i]) != nodes.begin() + i) {
      throw std::invalid_argument("NUMA nodes must be distinct and nonnegative (or -1 alone)");
    }
  }
  return nodes;
}

inline std::vector<int> ParseNumaNodes(const std::string& value) {
  std::vector<int> nodes;
  std::istringstream input(value);
  std::string token;
  if (value.empty() || value.back() == ',') throw std::invalid_argument("empty NUMA node");
  while (std::getline(input, token, ',')) {
    size_t end = 0;
    const int node = std::stoi(token, &end);
    if (token.find_first_not_of(" \t\r\n", end) != std::string::npos) {
      throw std::invalid_argument("invalid NUMA node: " + token);
    }
    nodes.push_back(node);
  }
  return NormalizeNumaNodes(std::move(nodes));
}

inline std::string FormatNumaNodes(const std::vector<int>& nodes) {
  if (nodes.empty()) return "-1";
  std::string out;
  for (int node : nodes) {
    if (!out.empty()) out += ',';
    out += std::to_string(node);
  }
  return out;
}

// Both configuration producers use this, so policy and legacy tiers split the
// same way. Keep whole allocator pages; only the final buffer carries slack.
inline std::vector<uint64_t> SplitNumaCapacity(uint64_t bytes, size_t nodes, uint64_t page_size) {
  if (nodes <= 1) return {bytes};
  if (page_size == 0) page_size = 2ULL * 1024 * 1024;
  const uint64_t part = (bytes / page_size / nodes) * page_size;
  if (part == 0)
    throw std::invalid_argument("NUMA capacity must provide at least one page per node");
  std::vector<uint64_t> sizes(nodes, part);
  sizes.back() = bytes - part * (nodes - 1);
  return sizes;
}

}  // namespace mori::umbp
