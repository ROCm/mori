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

// What the periodic INFO summary says, tested without a pool.
//
// The summary replaced per-key DEBUG lines, so the thing worth pinning down is
// not that it prints, but that a reader can tell the same things from it that
// the DEBUG lines told them: that work happened, how much, and -- the part a
// counter makes easy and a log line never did -- that a window in which nothing
// happened is itself reported.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "umbp/distributed/metrics/activity_summary.h"

namespace mori::umbp {
namespace {

std::string Joined(const std::vector<std::string>& lines) {
  std::string all;
  for (const auto& line : lines) {
    all += line;
    all += '\n';
  }
  return all;
}

ActivitySnapshot WithDram(uint64_t offload_bytes, uint64_t offload_keys, uint64_t load_bytes,
                          uint64_t load_keys, uint64_t evict_bytes, uint64_t evict_keys) {
  ActivitySnapshot snap;
  TierActivity& dram = snap.tiers["DRAM"];
  dram.offload_bytes = offload_bytes;
  dram.offload_keys = offload_keys;
  dram.load_bytes = load_bytes;
  dram.load_keys = load_keys;
  dram.evict_bytes = evict_bytes;
  dram.evict_keys = evict_keys;
  dram.capacity_bytes = 2ULL * 1024 * 1024 * 1024;
  dram.used_bytes = 1ULL * 1024 * 1024 * 1024;
  dram.resident_keys = 1024;
  return snap;
}

TEST(ActivitySummary, FormatsBytesTheWayAReaderSizesThem) {
  EXPECT_EQ(ActivitySummary::FormatBytes(0), "0 B");
  EXPECT_EQ(ActivitySummary::FormatBytes(512), "512 B");
  EXPECT_EQ(ActivitySummary::FormatBytes(1024), "1.00 KiB");
  EXPECT_EQ(ActivitySummary::FormatBytes(1536), "1.50 KiB");
  EXPECT_EQ(ActivitySummary::FormatBytes(1024ULL * 1024), "1.00 MiB");
  EXPECT_EQ(ActivitySummary::FormatBytes(2147483648ULL), "2.00 GiB");
}

TEST(ActivitySummary, FirstWindowReportsEverythingSoFar) {
  // No baseline exists yet. Reporting a delta against an implicit zero is the
  // only honest option; the alternative (skip the first window) loses whatever
  // a short-lived process did in its entire life.
  ActivitySummary summary;
  const auto lines = summary.Render(WithDram(1024 * 1024, 4, 2048 * 1024, 8, 0, 0), 60.0);
  const std::string all = Joined(lines);
  EXPECT_NE(all.find("offload 1.00 MiB/4 keys"), std::string::npos) << all;
  EXPECT_NE(all.find("load 2.00 MiB/8 keys"), std::string::npos) << all;
}

TEST(ActivitySummary, SecondWindowReportsOnlyWhatChanged) {
  ActivitySummary summary;
  summary.Render(WithDram(1024 * 1024, 4, 0, 0, 0, 0), 60.0);
  // Cumulative counters: 3 MiB total means 2 MiB happened in THIS window.
  const auto lines = summary.Render(WithDram(3 * 1024 * 1024, 12, 0, 0, 0, 0), 60.0);
  const std::string all = Joined(lines);
  EXPECT_NE(all.find("offload 2.00 MiB/8 keys"), std::string::npos) << all;
  EXPECT_EQ(all.find("offload 3.00 MiB"), std::string::npos) << all;
}

TEST(ActivitySummary, AnIdleWindowStillSaysWhatIsResident) {
  // The point of the idle line: a node holding 1024 keys that moved none of
  // them is the shape of a stall, and is indistinguishable from a healthy quiet
  // node unless the line says how much is sitting there.
  ActivitySummary summary;
  summary.Render(WithDram(1024 * 1024, 4, 0, 0, 0, 0), 60.0);
  const auto lines = summary.Render(WithDram(1024 * 1024, 4, 0, 0, 0, 0), 60.0);
  const std::string all = Joined(lines);
  EXPECT_NE(all.find("idle"), std::string::npos) << all;
  EXPECT_NE(all.find("1024 keys"), std::string::npos) << all;
  EXPECT_NE(all.find("1.00 GiB"), std::string::npos) << all;
}

TEST(ActivitySummary, NeverReturnsNothing) {
  // A caller logs whatever comes back; an empty result would make a quiet
  // window look like a dead summary thread.
  ActivitySummary summary;
  EXPECT_FALSE(summary.Render(ActivitySnapshot{}, 60.0).empty());
}

TEST(ActivitySummary, ReportsEvictionSeparatelyFromOffload) {
  // These two were the pair most often confused in the DEBUG era: bytes written
  // into a tier and bytes reclaimed from it both looked like "the tier is busy".
  ActivitySummary summary;
  summary.Render(WithDram(0, 0, 0, 0, 0, 0), 60.0);
  const auto lines = summary.Render(WithDram(4 * 1024 * 1024, 4, 0, 0, 1024 * 1024, 2), 60.0);
  const std::string all = Joined(lines);
  EXPECT_NE(all.find("offload 4.00 MiB/4 keys"), std::string::npos) << all;
  EXPECT_NE(all.find("evict 1.00 MiB/2 keys"), std::string::npos) << all;
}

TEST(ActivitySummary, ReportsPromotionAndDemotionAsAPairEvenWhenOneIsZero) {
  // A tier graph that only sheds is a real misconfiguration, and it is visible
  // only if the zero is printed next to the non-zero.
  ActivitySummary summary;
  ActivitySnapshot first = WithDram(0, 0, 0, 0, 0, 0);
  summary.Render(first, 60.0);

  ActivitySnapshot second = WithDram(0, 0, 0, 0, 0, 0);
  second.transitions.offloaded_bytes = 8ULL * 1024 * 1024;
  second.transitions.succeeded = 4;
  second.transitions.failed = 1;
  const std::string all = Joined(summary.Render(second, 60.0));
  EXPECT_NE(all.find("promote 0 B"), std::string::npos) << all;
  EXPECT_NE(all.find("demote 8.00 MiB"), std::string::npos) << all;
  EXPECT_NE(all.find("1 failed"), std::string::npos) << all;
}

TEST(ActivitySummary, SurfacesLoadMissesWhichBytesAloneHide) {
  // A tier being asked for keys it does not have reads as zero load bytes,
  // exactly like a tier nobody is asking. Different bugs, same byte count.
  ActivitySummary summary;
  summary.Render(WithDram(0, 0, 0, 0, 0, 0), 60.0);
  ActivitySnapshot missing = WithDram(0, 0, 0, 0, 0, 0);
  missing.tiers["DRAM"].load_misses = 77;
  const std::string all = Joined(summary.Render(missing, 60.0));
  EXPECT_NE(all.find("miss=77"), std::string::npos) << all;
}

TEST(ActivitySummary, ReCacheLineNamesWhyObjectsWereDropped) {
  // This is the line that replaced six per-key DEBUG statements. Each drop
  // reason has a different fix -- a bigger queue, a wider admission policy, a
  // larger tier -- so the counts have to stay distinguishable.
  ActivitySummary summary;
  summary.Render(ActivitySnapshot{}, 60.0);

  ActivitySnapshot snap;
  snap.recache.admitted = 100;
  snap.recache.installed = 90;
  snap.recache.queue_full = 7;
  snap.recache.rejected = 3;
  snap.recache.alloc_failed = 2;
  snap.recache.install_failed = 1;
  const std::string all = Joined(summary.Render(snap, 60.0));
  EXPECT_NE(all.find("90 installed / 100 admitted"), std::string::npos) << all;
  EXPECT_NE(all.find("queue_full=7"), std::string::npos) << all;
  EXPECT_NE(all.find("rejected=3"), std::string::npos) << all;
  EXPECT_NE(all.find("no_memory=2"), std::string::npos) << all;
  EXPECT_NE(all.find("install_failed=1"), std::string::npos) << all;
  EXPECT_NE(all.find("dropped 13"), std::string::npos) << all;
}

TEST(ActivitySummary, QuietReCachePathPrintsNoReCacheLine) {
  // A deployment with no remote peers never re-caches. A line of zeroes every
  // window would train the reader to skip the block that matters when it is
  // finally non-zero.
  ActivitySummary summary;
  summary.Render(WithDram(0, 0, 0, 0, 0, 0), 60.0);
  const std::string all = Joined(summary.Render(WithDram(1024, 1, 0, 0, 0, 0), 60.0));
  EXPECT_EQ(all.find("recache"), std::string::npos) << all;
}

TEST(ActivitySummary, BreaksDownByTierOnlyWhenThereIsMoreThanOne) {
  ActivitySummary summary;

  ActivitySnapshot two;
  two.tiers["DRAM"].offload_bytes = 1024 * 1024;
  two.tiers["DRAM"].capacity_bytes = 1024 * 1024 * 1024;
  two.tiers["SSD"].offload_bytes = 4ULL * 1024 * 1024;
  two.tiers["SSD"].capacity_bytes = 8ULL * 1024 * 1024 * 1024;
  const std::string all = Joined(summary.Render(two, 60.0));
  EXPECT_NE(all.find("DRAM"), std::string::npos) << all;
  EXPECT_NE(all.find("SSD"), std::string::npos) << all;
  // The total line must still add them up, or a two-tier node needs mental
  // arithmetic to answer "did anything move".
  EXPECT_NE(all.find("offload 5.00 MiB"), std::string::npos) << all;
}

TEST(ActivitySummary, ATierThatDisappearedDoesNotPrintAnEnormousDelta) {
  // A backend can be unregistered and re-registered, restarting its counters.
  // Without clamping, the next window subtracts a large baseline from a small
  // reading and reports something that reads like data loss.
  ActivitySummary summary;
  summary.Render(WithDram(64ULL * 1024 * 1024, 64, 0, 0, 0, 0), 60.0);
  const std::string all = Joined(summary.Render(WithDram(1024 * 1024, 1, 0, 0, 0, 0), 60.0));
  EXPECT_NE(all.find("offload 1.00 MiB/1 keys"), std::string::npos) << all;
}

TEST(ActivitySummary, ReportsTheMeasuredWindowNotTheConfiguredOne) {
  ActivitySummary summary;
  const std::string all = Joined(summary.Render(WithDram(1024 * 1024, 1, 0, 0, 0, 0), 12.5));
  EXPECT_NE(all.find("12.5s"), std::string::npos) << all;
}

}  // namespace
}  // namespace mori::umbp
