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

#include <sys/resource.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/sharded_ssd_tier.h"

namespace mori::umbp {

namespace {

// Every segment file of every shard stays open for the life of the tier, so a
// full tier holds capacity / segment_size descriptors -- 1.5 TB at the 256 MiB
// default is ~5,900 -- on top of the process's sockets, GPU IPC handles and
// io_uring rings.  The usual soft RLIMIT_NOFILE is 1,024 (docker's default),
// and the first segment open past it fails with EMFILE: every write to the tier
// then fails until something else closes a descriptor.  Raise the soft limit
// toward the hard one up front, sized for every file tier this process builds.
void EnsureSegmentFdBudget(uint64_t segments) {
  static std::mutex mu;
  static uint64_t planned = 0;
  // Sockets, GPU IPC imports, io_uring rings, model files, the rest of the app.
  constexpr rlim_t kHeadroom = 4096;
  std::lock_guard<std::mutex> lock(mu);
  planned += segments;
  struct rlimit rl{};
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return;
  const rlim_t want = static_cast<rlim_t>(planned) + kHeadroom;
  if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < want) {
    const rlim_t before = rl.rlim_cur;
    rl.rlim_cur = rl.rlim_max == RLIM_INFINITY ? want : std::min(want, rl.rlim_max);
    if (setrlimit(RLIMIT_NOFILE, &rl) == 0) {
      MORI_UMBP_INFO("[MakeFileSsdBackend] raised RLIMIT_NOFILE soft {} -> {} for {} segment files",
                     before, rl.rlim_cur, planned);
    } else {
      MORI_UMBP_WARN("[MakeFileSsdBackend] could not raise RLIMIT_NOFILE soft {} -> {}: {}", before,
                     rl.rlim_cur, std::strerror(errno));
      rl.rlim_cur = before;
    }
  }
  if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < want) {
    MORI_UMBP_ERROR(
        "[MakeFileSsdBackend] RLIMIT_NOFILE {} (hard {}) is below the {} descriptors a full SSD "
        "tier needs ({} segment files + headroom); writes will fail with EMFILE once the tier "
        "fills.  Raise the hard limit (e.g. docker --ulimit nofile) or ssd.segment_size_bytes.",
        rl.rlim_cur, rl.rlim_max, want, planned);
  }
}

}  // namespace

std::unique_ptr<TierBackend> MakeFileSsdBackend(const UMBPSsdConfig& ssd_config,
                                                SSDAccessMode access_mode) {
  const std::vector<std::string> dirs = ssd_config.StorageDirs();
  if (ssd_config.segment_size_bytes > 0) {
    // Plus each shard's partially filled active segment.
    EnsureSegmentFdBudget((ssd_config.capacity_bytes + ssd_config.segment_size_bytes - 1) /
                              ssd_config.segment_size_bytes +
                          dirs.size());
  }
  // Surface the resolved durability mode once: "strict" fdatasync()s every batch
  // write and, on a real drive, dominates write time.  UMBP_SSD_DURABILITY.
  MORI_UMBP_INFO("[MakeFileSsdBackend] drives={} durability={} io_backend={} queue_depth={}",
                 dirs.size(),
                 ssd_config.durability.mode == UMBPDurabilityMode::Strict ? "strict" : "relaxed",
                 ssd_config.io.backend == UMBPIoBackend::IoUring ? "io_uring" : "posix",
                 ssd_config.io.queue_depth);
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
