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

#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mori {

namespace detail {

// Decimal digits only, at most 65535.
inline std::optional<int> ParseCpuId(const std::string& s) {
  if (s.empty() || s.size() > 5 || s.find_first_not_of("0123456789") != std::string::npos) {
    return std::nullopt;
  }
  int id = std::stoi(s);
  if (id > 65535) return std::nullopt;
  return id;
}

}  // namespace detail

// A Linux cpulist such as "0-3,8,10-11" as ascending CPU ids without duplicates. Items may have
// spaces around them. nullopt if the list is empty or any item is malformed.
inline std::optional<std::vector<int>> ParseCpuList(const std::string& spec) {
  std::set<int> cpus;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t b = item.find_first_not_of(" \t");
    item = b == std::string::npos ? "" : item.substr(b, item.find_last_not_of(" \t") - b + 1);
    size_t dash = item.find('-');
    std::optional<int> lo = detail::ParseCpuId(item.substr(0, dash));
    std::optional<int> hi =
        dash == std::string::npos ? lo : detail::ParseCpuId(item.substr(dash + 1));
    if (!lo || !hi || *lo > *hi) return std::nullopt;
    for (int c = *lo; c <= *hi; ++c) cpus.insert(c);
  }
  if (cpus.empty()) return std::nullopt;
  return std::vector<int>(cpus.begin(), cpus.end());
}

}  // namespace mori
