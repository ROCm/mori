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
#include "umbp/distributed/peer/ssd_evict_strategy.h"

#include <algorithm>

namespace mori::umbp {

std::vector<std::string> LruSsdEvictStrategy::SelectVictims(
    std::vector<SsdEvictCandidate> candidates, size_t bytes_to_free) {
  std::vector<std::string> victims;
  if (bytes_to_free == 0) return victims;

  // Oldest-first within the received candidate set.  The manager already
  // enumerates candidates oldest-first, so this sort is a no-op for the default
  // LRU list ordering; we sort explicitly so the policy does not depend on the
  // caller's enumeration order.  Note this only orders the candidates passed in
  // (a recency-ordered prefix today), not the full SSD key set.
  std::sort(candidates.begin(), candidates.end(),
            [](const SsdEvictCandidate& a, const SsdEvictCandidate& b) {
              return a.lru_rank < b.lru_rank;
            });

  size_t freed = 0;
  for (auto& c : candidates) {
    victims.push_back(std::move(c.key));
    freed += c.size;
    if (freed >= bytes_to_free) break;
  }
  return victims;
}

}  // namespace mori::umbp
