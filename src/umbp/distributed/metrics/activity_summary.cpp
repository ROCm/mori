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
#include "umbp/distributed/metrics/activity_summary.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

namespace mori::umbp {

namespace {

// Counters only ever go up, but a snapshot can still come back smaller than
// its predecessor: a backend can be unregistered and re-registered, and a tier
// that disappears and returns starts from zero.  Clamping at zero prints a
// quiet 0 for that window instead of an enormous number that reads like a
// catastrophic regression.
uint64_t Delta(uint64_t now, uint64_t before) { return now >= before ? now - before : now; }

std::string Rate(uint64_t bytes, double seconds) {
  if (seconds <= 0.0 || bytes == 0) return {};
  const double mib_per_s = (static_cast<double>(bytes) / (1024.0 * 1024.0)) / seconds;
  std::array<char, 64> buf{};
  std::snprintf(buf.data(), buf.size(), " (%.1f MiB/s)", mib_per_s);
  return std::string(buf.data());
}

// "offload 1.19 GiB/1220 keys" — bytes first because that is the number that
// decides whether a tier is doing anything, key count second because that is
// the number that decides whether the batching is sane.
std::string Movement(const char* verb, uint64_t bytes, uint64_t keys, double seconds) {
  std::array<char, 160> buf{};
  const std::string amount = ActivitySummary::FormatBytes(bytes);
  std::snprintf(buf.data(), buf.size(), "%s %s/%llu keys%s", verb, amount.c_str(),
                static_cast<unsigned long long>(keys), Rate(bytes, seconds).c_str());
  return std::string(buf.data());
}

bool Moved(const TierActivity& d) {
  return d.offload_bytes != 0 || d.offload_keys != 0 || d.load_bytes != 0 || d.load_keys != 0 ||
         d.load_misses != 0 || d.evict_bytes != 0 || d.evict_keys != 0;
}

TierActivity DiffTier(const TierActivity& now, const TierActivity& before) {
  TierActivity d;
  d.offload_keys = Delta(now.offload_keys, before.offload_keys);
  d.offload_bytes = Delta(now.offload_bytes, before.offload_bytes);
  d.load_keys = Delta(now.load_keys, before.load_keys);
  d.load_bytes = Delta(now.load_bytes, before.load_bytes);
  d.load_misses = Delta(now.load_misses, before.load_misses);
  d.evict_keys = Delta(now.evict_keys, before.evict_keys);
  d.evict_bytes = Delta(now.evict_bytes, before.evict_bytes);
  // Gauges pass through untouched: differencing "how full is the tier" would
  // turn the one reading that answers "are we wedged or just idle" into noise.
  d.capacity_bytes = now.capacity_bytes;
  d.used_bytes = now.used_bytes;
  d.resident_keys = now.resident_keys;
  return d;
}

}  // namespace

std::string ActivitySummary::FormatBytes(uint64_t bytes) {
  static constexpr std::array<const char*, 5> kUnits{"B", "KiB", "MiB", "GiB", "TiB"};
  if (bytes < 1024) {
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%llu B", static_cast<unsigned long long>(bytes));
    return std::string(buf.data());
  }
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < kUnits.size()) {
    value /= 1024.0;
    ++unit;
  }
  std::array<char, 32> buf{};
  std::snprintf(buf.data(), buf.size(), "%.2f %s", value, kUnits[unit]);
  return std::string(buf.data());
}

std::vector<std::string> ActivitySummary::Render(const ActivitySnapshot& now,
                                                 double window_seconds) {
  // A named empty baseline rather than a temporary in the conditional: the
  // ternary form materializes a full copy of previous_ every window (the two
  // operands differ in value category), which is a map copy per tick to express
  // "compare against zero".
  static const ActivitySnapshot kNothingYet;
  const ActivitySnapshot& before = has_previous_ ? previous_ : kNothingYet;

  // ---- aggregate across tiers -------------------------------------------
  TierActivity total;
  std::map<std::string, TierActivity> per_tier;
  for (const auto& [tier, sample] : now.tiers) {
    const auto it = before.tiers.find(tier);
    const TierActivity baseline = it == before.tiers.end() ? TierActivity{} : it->second;
    TierActivity d = DiffTier(sample, baseline);
    total.offload_keys += d.offload_keys;
    total.offload_bytes += d.offload_bytes;
    total.load_keys += d.load_keys;
    total.load_bytes += d.load_bytes;
    total.load_misses += d.load_misses;
    total.evict_keys += d.evict_keys;
    total.evict_bytes += d.evict_bytes;
    total.capacity_bytes += d.capacity_bytes;
    total.used_bytes += d.used_bytes;
    total.resident_keys += d.resident_keys;
    per_tier.emplace(tier, std::move(d));
  }

  const uint64_t promoted =
      Delta(now.transitions.promoted_bytes, before.transitions.promoted_bytes);
  const uint64_t demoted =
      Delta(now.transitions.offloaded_bytes, before.transitions.offloaded_bytes);
  const uint64_t moves_ok = Delta(now.transitions.succeeded, before.transitions.succeeded);
  const uint64_t moves_failed = Delta(now.transitions.failed, before.transitions.failed);

  const ReCacheActivity& rc_now = now.recache;
  const ReCacheActivity& rc_old = before.recache;
  const uint64_t rc_admitted = Delta(rc_now.admitted, rc_old.admitted);
  const uint64_t rc_installed = Delta(rc_now.installed, rc_old.installed);
  const uint64_t rc_rejected = Delta(rc_now.rejected, rc_old.rejected);
  const uint64_t rc_queue_full = Delta(rc_now.queue_full, rc_old.queue_full);
  const uint64_t rc_alloc_failed = Delta(rc_now.alloc_failed, rc_old.alloc_failed);
  const uint64_t rc_install_failed = Delta(rc_now.install_failed, rc_old.install_failed);
  const uint64_t rc_already = Delta(rc_now.already_present, rc_old.already_present);
  const uint64_t rc_prefetched = Delta(rc_now.prefetched, rc_old.prefetched);
  const uint64_t rc_fragmented = Delta(rc_now.fragmented, rc_old.fragmented);
  const uint64_t rc_dropped = rc_rejected + rc_queue_full + rc_alloc_failed + rc_install_failed;

  // Adopt the new baseline only now that every delta above has been reduced to
  // a value: `before` aliases previous_, so assigning earlier would leave the
  // remaining reads pointing at the window that just ended.
  previous_ = now;
  has_previous_ = true;

  std::array<char, 96> window{};
  std::snprintf(window.data(), window.size(), "%.1fs", window_seconds);

  const bool anything = Moved(total) || promoted != 0 || demoted != 0 || moves_failed != 0 ||
                        rc_admitted != 0 || rc_dropped != 0 || rc_prefetched != 0 ||
                        rc_fragmented != 0 || rc_already != 0;

  std::vector<std::string> lines;

  // ---- the idle case -----------------------------------------------------
  // Still a line, and still at INFO.  A node holding 40 000 keys that moved
  // none of them for a minute is the signature of the stall this summary was
  // added to catch, and it is indistinguishable from a healthy quiet node
  // unless something says how much is resident while nothing happens.
  if (!anything) {
    std::string line = std::string("[UMBP] ") + window.data() + ": idle — no offload/load/evict";
    if (total.resident_keys != 0 || total.used_bytes != 0) {
      line += ", resident " + FormatBytes(total.used_bytes);
      if (total.capacity_bytes > 0) {
        std::array<char, 32> pct{};
        std::snprintf(pct.data(), pct.size(), " (%.0f%% of ",
                      100.0 * static_cast<double>(total.used_bytes) /
                          static_cast<double>(total.capacity_bytes));
        line += std::string(pct.data()) + FormatBytes(total.capacity_bytes) + ")";
      }
      std::array<char, 48> keys{};
      std::snprintf(keys.data(), keys.size(), " in %llu keys",
                    static_cast<unsigned long long>(total.resident_keys));
      line += keys.data();
    } else {
      line += ", tiers empty";
    }
    lines.push_back(std::move(line));
    return lines;
  }

  // ---- what moved --------------------------------------------------------
  std::string head = std::string("[UMBP] ") + window.data() + ": " +
                     Movement("offload", total.offload_bytes, total.offload_keys, window_seconds) +
                     " · " + Movement("load", total.load_bytes, total.load_keys, window_seconds);
  if (total.load_misses != 0) {
    std::array<char, 48> miss{};
    std::snprintf(miss.data(), miss.size(), " miss=%llu",
                  static_cast<unsigned long long>(total.load_misses));
    head += miss.data();
  }
  head += " · " + Movement("evict", total.evict_bytes, total.evict_keys, window_seconds);

  // Promotion and demotion are the SAME mechanism in opposite directions, so
  // they are printed as a pair even when one of them is zero: "promote 0 B"
  // next to a large demote is the shape of a tier graph that only sheds.
  head += " · promote " + FormatBytes(promoted) + " · demote " + FormatBytes(demoted);
  if (moves_ok != 0 || moves_failed != 0) {
    std::array<char, 80> moves{};
    std::snprintf(moves.data(), moves.size(), " (%llu moved, %llu failed)",
                  static_cast<unsigned long long>(moves_ok),
                  static_cast<unsigned long long>(moves_failed));
    head += moves.data();
  }
  lines.push_back(std::move(head));

  // ---- the local-tier admission path -------------------------------------
  // Only when it did something, because a deployment that never re-caches
  // (no remote peers, prefetch off) would otherwise carry a line of zeroes
  // forever.
  if (rc_admitted != 0 || rc_dropped != 0 || rc_prefetched != 0 || rc_already != 0 ||
      rc_fragmented != 0) {
    std::array<char, 320> buf{};
    std::snprintf(
        buf.data(), buf.size(),
        "[UMBP]   recache: %llu installed / %llu admitted (already=%llu, "
        "prefetched=%llu); dropped %llu (rejected=%llu queue_full=%llu "
        "no_memory=%llu install_failed=%llu fragmented=%llu)",
        static_cast<unsigned long long>(rc_installed), static_cast<unsigned long long>(rc_admitted),
        static_cast<unsigned long long>(rc_already), static_cast<unsigned long long>(rc_prefetched),
        static_cast<unsigned long long>(rc_dropped), static_cast<unsigned long long>(rc_rejected),
        static_cast<unsigned long long>(rc_queue_full),
        static_cast<unsigned long long>(rc_alloc_failed),
        static_cast<unsigned long long>(rc_install_failed),
        static_cast<unsigned long long>(rc_fragmented));
    lines.emplace_back(buf.data());
  }

  // ---- per tier ----------------------------------------------------------
  // A single tier is already fully described by the total line above, so it is
  // not repeated; with two or more, which tier absorbed the traffic is the
  // whole question.
  if (per_tier.size() > 1) {
    for (const auto& [tier, d] : per_tier) {
      std::array<char, 320> buf{};
      const std::string used = FormatBytes(d.used_bytes);
      const std::string cap = FormatBytes(d.capacity_bytes);
      const double pct = d.capacity_bytes == 0 ? 0.0
                                               : 100.0 * static_cast<double>(d.used_bytes) /
                                                     static_cast<double>(d.capacity_bytes);
      std::snprintf(buf.data(), buf.size(),
                    "[UMBP]   %s: offload %s load %s evict %s | resident %s/%s (%.0f%%) %llu keys",
                    tier.c_str(), FormatBytes(d.offload_bytes).c_str(),
                    FormatBytes(d.load_bytes).c_str(), FormatBytes(d.evict_bytes).c_str(),
                    used.c_str(), cap.c_str(), pct,
                    static_cast<unsigned long long>(d.resident_keys));
      lines.emplace_back(buf.data());
    }
  } else if (per_tier.size() == 1) {
    const auto& [tier, d] = *per_tier.begin();
    std::array<char, 256> buf{};
    const double pct = d.capacity_bytes == 0 ? 0.0
                                             : 100.0 * static_cast<double>(d.used_bytes) /
                                                   static_cast<double>(d.capacity_bytes);
    std::snprintf(buf.data(), buf.size(), "[UMBP]   %s resident %s/%s (%.0f%%) %llu keys",
                  tier.c_str(), FormatBytes(d.used_bytes).c_str(),
                  FormatBytes(d.capacity_bytes).c_str(), pct,
                  static_cast<unsigned long long>(d.resident_keys));
    lines.emplace_back(buf.data());
  }

  return lines;
}

}  // namespace mori::umbp
