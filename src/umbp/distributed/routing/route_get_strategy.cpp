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
#include "umbp/distributed/routing/route_get_strategy.h"

#include <random>
#include <sstream>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

std::string SummarizeLocations(const std::vector<Location>& locations) {
  if (locations.empty()) return "<empty>";
  std::ostringstream oss;
  bool first = true;
  for (const auto& loc : locations) {
    if (!first) oss << ", ";
    first = false;
    oss << loc.node_id << ':' << TierTypeName(loc.tier) << '/' << loc.size;
  }
  return oss.str();
}

}  // namespace

Location RandomRouteGetStrategy::Select(const std::vector<Location>& locations,
                                        const std::string& /*node_id*/) {
  if (locations.empty()) {
    MORI_UMBP_WARN("[RouteGetStrategy] received empty location set; returning default Location");
    return {};
  }

  if (locations.size() == 1) {
    const auto& single = locations[0];
    MORI_UMBP_DEBUG("[RouteGetStrategy] single candidate selected node={} tier={} size={}",
                    single.node_id, TierTypeName(single.tier), single.size);
    return single;
  }

  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, locations.size() - 1);
  size_t choice = dist(rng);
  const auto& selected = locations[choice];
  MORI_UMBP_DEBUG(
      "[RouteGetStrategy] {} candidates -> choice={} node={} tier={} size={}, candidates=[{}]",
      locations.size(), choice, selected.node_id, TierTypeName(selected.tier), selected.size,
      SummarizeLocations(locations));
  return selected;
}

}  // namespace mori::umbp
