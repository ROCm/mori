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
#include "umbp/distributed/routing/route_put_strategy.h"

#include <array>

namespace mori::umbp {

std::optional<RoutePutResult> TierAwareMostAvailableStrategy::Select(
    const std::vector<ClientRecord>& alive_clients, uint64_t block_size) {
  static constexpr std::array<TierType, 3> kTierOrder = {TierType::HBM, TierType::DRAM,
                                                         TierType::SSD};

  for (TierType tier : kTierOrder) {
    const ClientRecord* best = nullptr;
    uint64_t best_available = 0;

    for (const auto& client : alive_clients) {
      auto it = client.tier_capacities.find(tier);
      if (it == client.tier_capacities.end()) {
        continue;
      }
      if (it->second.available_bytes < block_size) {
        continue;
      }
      if (best == nullptr || it->second.available_bytes > best_available) {
        best = &client;
        best_available = it->second.available_bytes;
      }
    }

    if (best != nullptr) {
      return RoutePutResult{best->node_id, best->node_address, tier};
    }
  }

  return std::nullopt;
}

}  // namespace mori::umbp
