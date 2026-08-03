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
#include "umbp/local/tiers/ssd_backend_factory.h"

#include <string>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/sharded_ssd_tier.h"

namespace mori::umbp {

std::unique_ptr<TierBackend> MakeFileSsdBackend(const UMBPSsdConfig& ssd_config,
                                                SSDAccessMode access_mode) {
  const std::vector<std::string> dirs = ssd_config.StorageDirs();
  if (dirs.size() == 1) {
    return std::make_unique<SSDTier>(dirs[0], ssd_config.capacity_bytes, ssd_config, access_mode);
  }

  // Split the budget evenly; the remainder goes to the first shard so the sum
  // matches capacity_bytes exactly and the master's reported total is right.
  const size_t per_shard = ssd_config.capacity_bytes / dirs.size();
  const size_t remainder = ssd_config.capacity_bytes % dirs.size();
  if (per_shard == 0) {
    throw std::runtime_error("[MakeFileSsdBackend] ssd.capacity_bytes too small to split across " +
                             std::to_string(dirs.size()) + " directories");
  }

  std::vector<std::unique_ptr<TierBackend>> shards;
  shards.reserve(dirs.size());
  for (size_t i = 0; i < dirs.size(); ++i) {
    const size_t cap = per_shard + (i == 0 ? remainder : 0);
    MORI_UMBP_INFO("[MakeFileSsdBackend] shard {} dir={} capacity={}B", i, dirs[i], cap);
    shards.push_back(std::make_unique<SSDTier>(dirs[i], cap, ssd_config, access_mode));
  }
  return std::make_unique<ShardedSsdTier>(std::move(shards), ssd_config.shard_io_threads);
}

}  // namespace mori::umbp
