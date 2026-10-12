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
#include <vector>

#include "umbp/distributed/types.h"

// ---------------------------------------------------------------------------
//  The periodic INFO line a UMBP process writes about itself.
//
//  Why this exists at all: everything below is already measured and already
//  published to Prometheus.  But the deployments that are hardest to debug are
//  exactly the ones with no Prometheus attached — a unit test, a one-off
//  benchmark, a customer's node, an sglang rank someone is bisecting — and in
//  those the numbers sat in atomics that nothing ever read.  The alternative
//  was DEBUG, which is per-key: turning it on to answer "is anything being
//  evicted?" produced megabytes of lines per second and changed the timing of
//  the thing under test.  One aggregated line per window answers that question
//  at the level it is actually asked, at a volume a long run can afford.
//
//  RENDERING ONLY.  This class owns no thread, reads no clock, opens no file
//  and calls no logger: it is handed a cumulative snapshot and returns the
//  text.  That is what lets the interesting part — the deltas, the units, what
//  an idle window says — be unit-tested without standing up a pool.
// ---------------------------------------------------------------------------

namespace mori::umbp {

// One tier's cumulative counters, summed over every backend of that tier.
//
// The three verbs are the storage-backend ops seen from the caller's side, and
// they are named here the way a UMBP deployment talks about them rather than
// the way the slot lifecycle does:
//
//   offload = commit   bytes written INTO this tier  (sglang: KV leaving HBM)
//   load    = resolve  bytes handed OUT to a reader  (sglang: load-back)
//   evict   = evict    bytes reclaimed
//
// `load_misses` is the resolve calls that found nothing.  It is here because a
// tier that is loading nothing and a tier that is being asked for keys it does
// not have look identical in a bytes counter, and they are completely
// different bugs.
struct TierActivity {
  uint64_t offload_keys = 0;
  uint64_t offload_bytes = 0;
  uint64_t load_keys = 0;
  uint64_t load_bytes = 0;
  uint64_t load_misses = 0;
  uint64_t evict_keys = 0;
  uint64_t evict_bytes = 0;

  // Live state, not a counter: sampled, never differenced.
  uint64_t capacity_bytes = 0;
  uint64_t used_bytes = 0;
  uint64_t resident_keys = 0;
};

// The local-tier admission path: an object fetched from a remote node (or
// pulled ahead by locality prefetch) being installed into this node's medium.
// That is a promotion too — it just crosses a node boundary instead of a tier
// boundary — and it is the part of the system that fails silently, because
// every one of its give-up paths is a best-effort `return` that leaves the
// read it was helping perfectly correct.
struct ReCacheActivity {
  uint64_t admitted = 0;         // enqueued for install
  uint64_t rejected = 0;         // admission policy said no
  uint64_t queue_full = 0;       // dropped: install queue at its bound
  uint64_t alloc_failed = 0;     // dropped: could not stage the bytes
  uint64_t installed = 0;        // landed in the local medium
  uint64_t already_present = 0;  // a concurrent install won
  uint64_t install_failed = 0;   // local put refused it
  uint64_t prefetched = 0;       // whole objects pulled by locality prefetch
  uint64_t fragmented = 0;       // prefetch slot was not one addressable run
};

// Everything one window needs, all cumulative since process start.
struct ActivitySnapshot {
  std::map<std::string, TierActivity> tiers;  // keyed by TierTypeName
  TierTransitionMetrics transitions;          // intra-pool promote / demote
  ReCacheActivity recache;
};

class ActivitySummary {
 public:
  // Renders `now` against the previously observed snapshot and adopts it as the
  // new baseline.  The FIRST call has no baseline, so it reports `now` in full:
  // for a process that started with empty tiers that is the same number, and
  // for one that did not, a first line that undercounts would be worse than one
  // that says "this is everything so far".
  //
  // `window_seconds` is what the caller measured between the two samples, not
  // what it configured, so a tick delayed by a stalled backend says so.
  //
  // Returns one line per tier plus a leading total line, or a single line when
  // nothing moved.  Never empty: a window in which a busy node did nothing is
  // itself the finding.
  std::vector<std::string> Render(const ActivitySnapshot& now, double window_seconds);

  // Byte counts as a human reads them ("1.19 GiB").  Exposed for the tests, and
  // because every other UMBP log that prints bytes prints them raw, which is
  // how "2147483648" ends up in a bug report nobody can size at a glance.
  static std::string FormatBytes(uint64_t bytes);

 private:
  ActivitySnapshot previous_;
  bool has_previous_ = false;
};

}  // namespace mori::umbp
