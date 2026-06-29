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

#include <iterator>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// A peer-side source of owned-location mutations (ADD/REMOVE KvEvents) that the
// heartbeat shipper drains and snapshots.  Each tier-owning component on a peer
// (PeerDramAllocator for DRAM/HBM, PeerSsdManager for SSD) implements this so
// MasterClient can aggregate every source into one heartbeat bundle under a
// single monotonic sequence number — never one seq per tier (that would break
// the ack / seq-gap full-sync recovery).
//
// This interface deliberately covers ONLY events.  Per-tier capacity, owned-key
// counts and the distributed-clear write gate stay on the concrete owners
// (PeerDramAllocator / PeerSsdManager) because they are tier-specific and not
// shared across all sources.
class OwnedLocationSource {
 public:
  virtual ~OwnedLocationSource() = default;

  // Drain the events queued since the last call (delta heartbeat).  Clears the
  // source's outbox.
  virtual std::vector<KvEvent> DrainPendingEvents() = 0;

  // Full snapshot of every owned key as ADD events (full sync on seq gap or
  // master restart).  const: a snapshot must not mutate the source's state.
  virtual std::vector<KvEvent> SnapshotOwnedKeys() const = 0;

  // Snapshot for a full sync that ALSO atomically drops the source's outbox.
  // After a full sync the snapshot is the authoritative state, so any event
  // still queued at snapshot time is already reflected in it (ADDs are in the
  // snapshot; REMOVEs are implied by the master's full replace) and must not be
  // re-shipped as a redundant post-full-sync delta.  The snapshot and the outbox
  // clear MUST happen in the SAME critical section — doing it in two separate
  // locks would drop events committed in between.
  virtual std::vector<KvEvent> SnapshotOwnedKeysForFullSync() = 0;
};

// Drain every source and concat into one event list, in source order.  The
// heartbeat shipper wraps the result in a SINGLE EventBundle under one
// monotonic seq — concat here, never one bundle/seq per source.  Null sources
// are skipped.  Exposed (vs. private to MasterClient) so it can be unit-tested
// against mock sources without standing up a master RPC.
inline std::vector<KvEvent> DrainAllSources(const std::vector<OwnedLocationSource*>& sources) {
  std::vector<KvEvent> merged;
  for (auto* src : sources) {
    if (src == nullptr) continue;
    auto events = src->DrainPendingEvents();
    merged.insert(merged.end(), std::make_move_iterator(events.begin()),
                  std::make_move_iterator(events.end()));
  }
  return merged;
}

// Snapshot every source and concat into one event list, in source order (full-
// sync path).  Each source ALSO atomically clears its outbox: the snapshot is
// authoritative, so the queued delta is redundant and must not be re-shipped
// afterwards.  Null sources are skipped.  Non-const because it mutates sources.
inline std::vector<KvEvent> SnapshotAllSourcesForFullSync(
    const std::vector<OwnedLocationSource*>& sources) {
  std::vector<KvEvent> merged;
  for (auto* src : sources) {
    if (src == nullptr) continue;
    auto events = src->SnapshotOwnedKeysForFullSync();
    merged.insert(merged.end(), std::make_move_iterator(events.begin()),
                  std::make_move_iterator(events.end()));
  }
  return merged;
}

}  // namespace mori::umbp
