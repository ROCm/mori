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

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/sharded_ssd_tier.h"

namespace mori::umbp {

namespace {

// Every segment file of every shard stays open for the life of the tier, so a
// full tier holds about capacity / segment_size descriptors (more under churn:
// evicted space is not reclaimed, #755).  Past the soft RLIMIT_NOFILE the next
// segment open fails with EMFILE, and so does every write that needs it.  This
// library also runs inside other programs (sglang, embedded), so it leaves the
// process-wide limit alone and says what to change; UMBP's own executables
// raise it in main() (RaiseOpenFileLimit).
void CheckSegmentFdBudget(uint64_t segments) {
  // Sockets, GPU IPC imports and the like that the process opens later.
  constexpr uint64_t kSlack = 256;
  struct rlimit rl{};
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0 || rl.rlim_cur == RLIM_INFINITY) return;
  uint64_t open_now = 0;
  std::error_code ec;
  for (std::filesystem::directory_iterator it("/proc/self/fd", ec), end; !ec && it != end;
       it.increment(ec)) {
    ++open_now;
  }
  const uint64_t need = open_now + segments + kSlack;
  if (need <= rl.rlim_cur) return;
  MORI_UMBP_ERROR(
      "[MakeFileSsdBackend] RLIMIT_NOFILE soft limit {} is below the ~{} descriptors this "
      "process needs once the SSD tier fills ({} open + {} segment files + {} slack); writes "
      "will then fail with EMFILE.  Raise it (ulimit -n, docker --ulimit nofile) or "
      "ssd.segment_size_bytes.",
      rl.rlim_cur, need, open_now, segments, kSlack);
}

}  // namespace

std::unique_ptr<TierBackend> MakeFileSsdBackend(const UMBPSsdConfig& ssd_config,
                                                SSDAccessMode access_mode) {
  const std::vector<std::string> dirs = ssd_config.StorageDirs();
  if (ssd_config.segment_size_bytes > 0) {
    // Plus each shard's partially filled active segment.
    CheckSegmentFdBudget((ssd_config.capacity_bytes + ssd_config.segment_size_bytes - 1) /
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
