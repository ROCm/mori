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

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/master/global_block_index.h"  // EvictionCandidate
#include "umbp/distributed/types.h"                      // TierType

namespace mori::umbp {

// Abstract interface for master-side victim selection.  Implement this to plug
// in a custom DRAM/HBM eviction policy.  The EvictionManager owns the rest of
// the cycle (watermark detection, candidate gathering, EvictKey dispatch) and
// delegates only the ranking + per-(node,tier) budget selection here — the same
// "manager prepares data, strategy decides" split used by Route{Put,Get}.
//
// SSD has its own peer-local policy (SsdEvictStrategy); master never evicts SSD.
class MasterEvictStrategy {
 public:
  virtual ~MasterEvictStrategy() = default;

  // Pick victims from @p candidates honouring the per-node, per-tier byte
  // budget in @p bytes_to_free (used > low_watermark slack the manager wants
  // freed).  Both arguments are taken by value so an implementation may sort /
  // decrement freely without touching master state.  @return victim keys
  // grouped by node_id (one keys[] per peer for a single EvictKey RPC); empty
  // groups are omitted.
  virtual std::unordered_map<std::string, std::vector<std::string>> SelectVictims(
      std::vector<EvictionCandidate> candidates,
      std::unordered_map<std::string, std::map<TierType, int64_t>> bytes_to_free) = 0;
};

// Default policy: pure LRU.  Evict oldest-access-first until each overloaded
// (node, tier)'s byte budget is met.
class LruMasterEvictStrategy : public MasterEvictStrategy {
 public:
  std::unordered_map<std::string, std::vector<std::string>> SelectVictims(
      std::vector<EvictionCandidate> candidates,
      std::unordered_map<std::string, std::map<TierType, int64_t>> bytes_to_free) override;
};

}  // namespace mori::umbp
