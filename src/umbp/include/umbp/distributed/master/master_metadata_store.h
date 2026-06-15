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

// Master-side metadata storage interface for mori::umbp.
//
// Goal: make the master stateless so multiple master replicas can serve
// traffic concurrently (HA). Today GlobalBlockIndex, ClientRegistry,
// ExternalKvBlockIndex, and ExternalKvHitIndex hold their state in
// in-process unordered_maps guarded by std::shared_mutex — that's a
// split-brain hazard the moment you run more than one master process,
// because each replica's view of locations, liveness, leases, LRU, and
// hit counts drifts independently. The fix is to pull ALL durable AND
// volatile master state out behind one abstract interface —
// IMasterMetadataStore — and ship it to a shared backend.
//
// Concrete target backend is Redis (Cluster); a SQL backend is NOT a
// target. The read path does one store round-trip per RouteGet
// (lookup + lease + access in one Lua script) which is fine against
// Redis but ruinous against an OLTP database. If a SQL backend is
// ever needed it'd be for cold archival, not the hot path served by
// this interface.
//
// =====================================================================
// Why one interface and not three (BlockLocations / ClientRecords /
// ExternalKv) per the earlier sketch:
// =====================================================================
//   - Every cross-store write today touches 2 or 3 of those stores:
//       * UnregisterClient → registry + block index + external_kv
//       * ReapExpiredClients → registry + block index + external_kv
//       * Heartbeat → registry seq-CAS + block index events
//       * RegisterExternalKvBlocks → registry alive-check + external_kv
//       * RegisterClient re-registration → registry read TTL + write
//   - Splitting forces a portable cross-store transaction abstraction.
//     In-memory implements it trivially (shared_mutex). Redis does
//     not: cross-keyspace MULTI/EXEC or Lua only works if all three
//     namespaces hash to the same slot, which leaks "must run on the
//     same cluster shard" up into the API.
//   - Collapsing to one interface lets each implementation choose its
//     own atomicity primitive instead of inventing a portable
//     transaction abstraction.
//   - Hot-path reads stay easy to identify because they're grouped in
//     the read section below; internally an in-memory impl can still
//     keep separate sub-maps for code organization. The interface just
//     stops pretending they're independently swappable.
//
// =====================================================================
// What does NOT move behind this interface:
// =====================================================================
//   - EvictionManager policy (watermark math, victim grouping, EvictKey
//     RPC dispatch). The store returns LRU-ordered candidates limited
//     by a byte budget; the manager decides how to spend that budget
//     and ships the RPCs. Stateless under HA — each tick's decision is
//     a pure function of the store's snapshot, so concurrent eviction
//     passes on different replicas converge instead of fighting.
//   - Reaper loop scheduling (timer + cv). Only the per-pass DB action
//     moves down, via ExpireStaleClients. The schedule is per-replica
//     with no shared state; the ExpireStaleClients call is idempotent
//     so multiple replicas can safely run reaper passes concurrently.
//   - Hit-index GC loop scheduling (timer + cv). Only the per-pass
//     action moves down, via GarbageCollectHits. Same per-replica /
//     idempotent reasoning as the reaper.
//
// In particular, lease_expiry / last_accessed_at / access_count and the
// per-hash hit counts DO move into the store — see hazards #4 and #7.
//
// =====================================================================
// Critical design decisions / hazards (carried forward from review):
// =====================================================================
//   1. Heartbeat is a CAS, not Get-then-Update.
//      The current in-memory ClientRegistry::Heartbeat reads
//      last_applied_seq, decides whether to accept, and writes the new
//      seq + caps + last_heartbeat under one unique_lock. If this is
//      split into Get() then UpdateHeartbeatState() across an external
//      backend, two concurrent heartbeats for the same node can both
//      observe seq N, both decide "in order," and one silently
//      corrupts last_applied_seq. ApplyHeartbeat MUST take seq as the
//      CAS value and atomically check `seq == last_applied_seq + 1`
//      inside the implementation (one Lua script on Redis;
//      shared_mutex unique_lock for the in-memory impl). See
//      HeartbeatResult below.
//
//   2. RegisterClient must accept TTL-stale ALIVE rows.
//      The current RegisterClient (client_registry.cpp:85-91) allows
//      re-registration when `(now - last_heartbeat > ExpiryDuration())`
//      even if status==ALIVE (i.e. the reaper hasn't flipped it yet).
//      A naive InsertIfNotAlive would reject. The new RegisterClient
//      therefore takes a `stale_after` duration so the implementation
//      enforces this in the same atomic step.
//
//   3. EXPIRED rows are KEPT, not erased.
//      Today's ReapExpiredClients erases rows from the map.
//      ExpireStaleClients flips status ALIVE → EXPIRED and keeps the
//      row so a re-registration can replace an EXPIRED record cleanly.
//      Behavioral change vs. today; consumers must filter status when
//      counting (use AliveClientCount, not "size of GetClient over all
//      ids", which doesn't exist anyway).
//
//   4. lease_expiry, last_accessed_at, access_count are IN the store.
//      Earlier sketches kept them in a process-local LeaseAccessTracker
//      to avoid per-RouteGet store writes; that's only safe with a
//      single master process. Under HA-stateless masters, two replicas
//      can hold conflicting in-memory leases for the same key —
//      replica A's lease doesn't block replica B's eviction loop —
//      and LRU views diverge per replica. Both are correctness bugs,
//      not stale-state inconveniences. So the tracker is gone:
//      RouteGet hits the store via LookupBlockForRouteGet which
//      atomically reads locations, sets lease_expiry, and bumps
//      last_accessed_at / access_count in one Lua script. The
//      BatchLookupBlockForRouteGet variant amortizes the cost over
//      the prefix-match path the router already uses.
//
//   5. BatchLookupBlockForRouteGet exists because router.cpp:139
//      issues N Lookup() calls in a hot loop. N round-trips against
//      Redis is unacceptable; one batched Lua call is required.
//
//   6. Sync vs async. Methods are synchronous; concurrency comes from
//      the gRPC handler threads. Cleaner than future-returning every
//      method.
//
//   7. Persisted timestamps are system_clock, not steady_clock.
//      steady_clock has a per-process arbitrary epoch — its values
//      are not meaningful to a different process or after master
//      restart, so they cannot live in a Redis row. Every timestamp
//      that crosses the IMasterMetadataStore boundary — registration
//      times, heartbeat timestamps, lease_expiry, last_accessed_at,
//      reaper cutoff, and the hit-count last_seen — is therefore
//      system_clock::time_point. No steady_clock anywhere below this
//      interface. Assumes NTP-disciplined clocks across master
//      replicas: a backward wall-clock jump effectively grants longer
//      leases until the clock recovers.
//
//   8. `last_acked_seq` is gone from the heartbeat path.
//      The current ClientRegistry::Heartbeat takes a `last_acked_seq`
//      parameter (client_registry.cpp:138) but its body ignores it —
//      the master gap-checks against its own last_applied_seq. The new
//      interface drops the parameter rather than passing through a
//      value no implementation will read. The peer wire still carries
//      last_acked_seq for the peer's own ack-on-progress logic; the
//      master_server adapter simply doesn't forward it down here.
//
// NodeTierKey, NodeMatch, ClientRegistration, HeartbeatResult, and
// EvictionCandidate are defined in umbp/distributed/types.h — they are
// part of this store's contract.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// =====================================================================
// IMasterMetadataStore — single durable-state interface for the master.
// =====================================================================
//
// All methods are thread-safe. The master's gRPC handler thread pool
// calls into a single shared instance from many threads concurrently —
// implementations must provide their own synchronization (shared_mutex
// for in-memory, single-script Lua atomicity for Redis). Callers do
// not add external locking around these calls.
//
// Every write method below is atomic in isolation. Methods that span
// what used to be multiple stores (UnregisterClient, ApplyHeartbeat,
// ExpireStaleClients, RegisterExternalKvIfAlive, and the hit-counting
// branch of MatchExternalKv) are atomic across those former
// boundaries — that is the whole reason for the merge.
//
// TODO(atomicity-contract): pin down the required isolation level
// before the Redis backend is written. The default position is
// "atomic with respect to other writes on this interface; readers
// may observe pre-state until commit" — i.e. single-script Lua for
// Redis, shared_mutex unique_lock for in-memory. Document the chosen
// level here once it's settled and add a conformance test that
// exercises a concurrent reader against an in-flight cross-store
// write.
//
// Expected implementations:
//   - InMemoryMasterMetadataStore: one std::shared_mutex over the
//     internal sub-maps. Mostly a mechanical lift of the current
//     classes; keep them as private helpers. Used for single-master
//     deployments and unit tests. The per-hash hit counts live in
//     process memory and are lost on restart, exactly as the current
//     ExternalKvHitIndex does.
//   - RedisMasterMetadataStore: cross-keyspace ops via Lua scripts.
//     All key namespaces (node:, block:, extkv:, hit:, lru:, lease:)
//     must share a hash tag (e.g. `{umbp:<deployment_id>}:node:<id>`)
//     so they hash to the same slot on Redis Cluster — that's what
//     makes cross-namespace Lua atomic, and what makes the hit counts
//     crash-durable.
class IMasterMetadataStore {
 public:
  virtual ~IMasterMetadataStore() = default;

  // ===================================================================
  // Cross-store write operations — each call is atomic.
  // ===================================================================

  // CAS-style registration. Inserts a fresh ALIVE record.
  //   - Returns true on new registration.
  //   - Returns true and replaces the record if an EXPIRED record for
  //     the same node_id exists.
  //   - Returns true and replaces the record if an ALIVE record exists
  //     whose last_heartbeat is older than `stale_after` — handles the
  //     "reaper hasn't run yet but the record is TTL-stale" case that
  //     today's RegisterClient permits (see hazard #2 in header
  //     preamble).
  //   - Returns false if an ALIVE non-stale record already exists.
  // `now` is supplied by the caller so tests can inject time. Uses
  // system_clock because the value is persisted in the backend; see
  // hazard #7 in header preamble. The store sets last_heartbeat and
  // registered_at to `now`, status to ALIVE, and last_applied_seq to 0
  // — the caller does not (and cannot) populate those fields.
  // In production, callers derive `stale_after` from
  // ClientRegistryConfig::ExpiryDuration() (heartbeat_ttl ×
  // max_missed_heartbeats); tests inject their own value.
  virtual bool RegisterClient(const ClientRegistration& registration,
                              std::chrono::system_clock::time_point now,
                              std::chrono::system_clock::duration stale_after) = 0;

  // Drop the client from the client store AND drop every block location
  // belonging to it AND drop every external-kv entry belonging to it.
  // Idempotent on missing clients.
  virtual void UnregisterClient(const std::string& node_id) = 0;

  // Heartbeat ingestion. Atomically:
  //   1. Looks up the client record; returns UNKNOWN if absent (no
  //      other state touched).
  //   2. If !is_full_sync and seq != last_applied_seq + 1, returns
  //      SEQ_GAP with acked_seq = last_applied_seq. SEQ_GAP still
  //      bumps last_heartbeat and sets status←ALIVE so the reaper
  //      doesn't kill a node that's heartbeating but mid-recovery;
  //      caps and last_applied_seq are NOT touched on SEQ_GAP. The
  //      seq check is the CAS that keeps the gap check race-free
  //      under concurrent heartbeats — DO NOT implement as a
  //      separate Get() then UpdateHeartbeatState(); two in-flight
  //      heartbeats can both observe the same seq and corrupt
  //      last_applied_seq (see hazard #1 in header preamble).
  //   3. On APPLIED, updates caps, last_heartbeat, last_applied_seq,
  //      status←ALIVE.
  //   4. Applies events to block locations:
  //        * is_full_sync=true → replace every location for node_id
  //          with the ADDs in events; REMOVE entries ignored.
  //        * is_full_sync=false → ADD with existing (node,tier)
  //          overwrites size; REMOVE for unknown (key,node,tier) is
  //          a silent no-op.
  // Returns APPLIED with acked_seq = seq on success.
  //
  // TODO(payload-sizing): the `events` vector is bounded only by what
  // the peer chooses to ship in one heartbeat batch. A full_sync from
  // a peer with millions of keys produces a single Lua script of that
  // size, which blocks every other Redis client (Redis is
  // single-threaded). The contract should be: implementations MAY
  // chunk internally for !is_full_sync, but is_full_sync MUST apply
  // atomically (a half-applied ReplaceNodeLocations would leave the
  // index in a torn state). That in turn forces an upper bound on
  // peer-side full_sync batch size — decide and document the cap
  // (e.g. 100k events) here, and have the peer fragment larger
  // resyncs into multiple full_sync calls or shift to a snapshot-
  // then-delta protocol before the Redis backend ships.
  virtual HeartbeatResult ApplyHeartbeat(const std::string& node_id, uint64_t seq,
                                         std::chrono::system_clock::time_point now,
                                         const std::map<TierType, TierCapacity>& caps,
                                         const std::vector<KvEvent>& events, bool is_full_sync) = 0;

  // Reaper pass. Atomically:
  //   - Flips status ALIVE → EXPIRED for every record with
  //     last_heartbeat < cutoff. EXPIRED records are KEPT in the store,
  //     not erased — see hazard #3 in header preamble.
  //   - Drops every block location belonging to those clients.
  //   - Drops every external-kv entry belonging to those clients.
  // Returns the affected node_ids for logging.
  virtual std::vector<std::string> ExpireStaleClients(
      std::chrono::system_clock::time_point cutoff) = 0;

  // ===================================================================
  // External-KV writes — alive-check + mutation atomic together.
  // ===================================================================

  // Add `tier` to the tier-set of every (node_id, hash). Idempotent:
  // re-registering at the same tier is a no-op; registering at a new
  // tier adds a bucket without touching existing tiers.
  // Returns true if the alive-check passed and the writes were applied
  // (even if every write was a no-op because the entries already existed).
  // Returns false if node_id was not ALIVE and nothing was written, so
  // the caller can meter the reject without the impl having to log it.
  virtual bool RegisterExternalKvIfAlive(const std::string& node_id,
                                         const std::vector<std::string>& hashes, TierType tier) = 0;

  // Remove `tier` from the tier-set of every (node_id, hash). Other
  // tiers for the same hash untouched. (node,hash) entry dropped when
  // its tier-set becomes empty. Does NOT check liveness — peers may
  // ship unregister during teardown after status has flipped.
  virtual void UnregisterExternalKv(const std::string& node_id,
                                    const std::vector<std::string>& hashes, TierType tier) = 0;

  // Remove `tier` from every hash registered by `node_id` (whole-tier
  // wipe — admin path, not heartbeat).
  virtual void UnregisterExternalKvByTier(const std::string& node_id, TierType tier) = 0;

  // Drop every per-hash hit-count entry whose last_seen < cutoff.
  // Returns the number of entries dropped. Replaces
  // ExternalKvHitIndex::GarbageCollect; the cutoff is a system_clock
  // time_point (not a uint64_t ns) because last_seen now crosses the
  // store boundary — hazard #7. Called by the master's hit-index GC
  // loop on each tick with cutoff = system_clock::now() - max_age.
  virtual std::size_t GarbageCollectHits(std::chrono::system_clock::time_point cutoff) = 0;

  // ===================================================================
  // Reads. None require cross-store atomicity; implementations SHOULD
  // make each one a single backend round-trip.
  // ===================================================================

  // --- Block locations ---

  // Plain location lookup. Returns every location for `key` without
  // granting a lease or recording an access. Pure read — no side
  // effects on lease_expiry, last_accessed_at, or access_count.
  virtual std::vector<Location> LookupBlock(const std::string& key) const = 0;

  // RouteGet primitive. Atomically reads every location for `key`,
  // filters out locations whose node_id is in `exclude_nodes`,
  // and — only if at least one location survives the filter — sets
  // lease_expiry to now + lease_duration and bumps last_accessed_at
  // to now / access_count by 1. Returns the filtered locations, or
  // empty if the key has no locations or all were excluded.
  // Filtering inside the store (not post-hoc in the caller) is
  // required so that fully-excluded keys do not receive a lease or
  // an access bump — granting those would perturb LRU ordering and
  // extend eviction protection for keys the caller explicitly chose
  // to skip. Splitting this into separate Lookup / GrantLease /
  // RecordAccess methods would be three round trips per RouteGet
  // AND would not be atomic across master replicas — see hazard #4.
  virtual std::vector<Location> LookupBlockForRouteGet(
      const std::string& key, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) = 0;

  // Vectorized RouteGet primitive. Same per-key semantics as
  // LookupBlockForRouteGet (leases granted and access recorded only
  // for keys that have at least one non-excluded location). Result
  // parallel to `keys`; absent or fully-excluded keys yield empty
  // inner vectors. One round trip for the whole batch —
  // router.cpp:139 today issues N Lookup() calls in a hot loop and
  // this is the single-RTT replacement (see hazard #5).
  virtual std::vector<std::vector<Location>> BatchLookupBlockForRouteGet(
      const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) = 0;

  // Batched existence — pure read, no lease grant, no access record.
  // Used by BatchRoutePut for dedup (router.cpp:112): a writer
  // landing on an existing key isn't a "read" and must not extend
  // the lease or perturb LRU ordering. One round trip.
  virtual std::vector<bool> BatchExistsBlock(const std::vector<std::string>& keys) const = 0;

  // LRU-prefix eviction enumeration. For each (node, tier) in
  // `bytes_to_free`, walks the store's per-bucket LRU order
  // (oldest last_accessed_at first), filters out keys whose
  // lease_expiry > now, and accumulates rows until the cumulative
  // `location.size` reaches that bucket's budget. Result map is
  // keyed by the same NodeTierKeys as the input; absent buckets had
  // no eligible candidates. Taking the whole budget map in one call
  // lets a single Lua script fan out over every overloaded bucket
  // in one round trip — important when dozens of (node, tier) pairs
  // are over watermark.
  //
  // No EraseBlock on this interface — peers ship REMOVEs on their
  // next heartbeat after EvictKey executes, so the only mutation
  // channels for block locations are ApplyHeartbeat /
  // UnregisterClient / ExpireStaleClients.
  //
  // How the LRU order is produced is an implementation detail: the
  // in-memory backend does a full entries_ scan + sort per tick (an
  // eviction tick is seconds, not a hot path), while the Redis
  // backend maintains a per-(node, tier) ZSET keyed by
  // last_accessed_at refreshed on every LookupBlockForRouteGet. The
  // contract is only "return LRU-ordered candidates within the byte
  // budget," not the index mechanism.
  virtual std::map<NodeTierKey, std::vector<EvictionCandidate>> EnumerateLruForEviction(
      const std::map<NodeTierKey, uint64_t>& bytes_to_free,
      std::chrono::system_clock::time_point now) const = 0;

  // --- Client records ---

  // Returns the record regardless of status (ALIVE or EXPIRED). Caller
  // filters when needed.
  virtual std::optional<ClientRecord> GetClient(const std::string& node_id) const = 0;

  // Hot-path liveness check. Exists as its own method so a Redis backend
  // can answer it with a single status field read instead of fetching the
  // whole ClientRecord just to filter on status.
  virtual bool IsClientAlive(const std::string& node_id) const = 0;

  // Single-node peer-address lookup. Exists as its own method so a
  // Redis backend can answer with a single HGET on the node hash
  // instead of fetching the whole ClientRecord just to read
  // peer_address. The legacy router linear-scans GetAliveClients()
  // per RouteGet for the same value; GetPeerAddress replaces that
  // with one read. Returns std::nullopt for unknown node_id;
  // EXPIRED records still surface their peer_address.
  virtual std::optional<std::string> GetPeerAddress(const std::string& node_id) const = 0;

  // ALIVE only — does not include EXPIRED records.
  virtual std::vector<ClientRecord> ListAliveClients() const = 0;
  virtual std::size_t AliveClientCount() const = 0;

  virtual std::vector<std::string> GetClientTags(const std::string& node_id) const = 0;

  // --- External KV ---

  // Returns matches grouped by node WITHOUT peer_address. Callers
  // that need peer addresses join with ListAliveClients() (snapshot
  // once per response) or GetPeerAddress(node_id) per-node;
  // embedding peer_address in NodeMatch would force every
  // implementation to read from two namespaces on every match.
  // When `count_as_hit` is true, atomically increments the per-hash
  // hit counter for every matched hash AND stamps that hash's
  // last_seen = `now`, all in one lock acquisition (lookup +
  // increment + stamp). When false, pure read — no hit counts
  // touched and `now` is ignored. `now` is system_clock because
  // last_seen is persisted / feeds GarbageCollectHits (hazard #7).
  virtual std::vector<NodeMatch> MatchExternalKv(const std::vector<std::string>& hashes,
                                                 bool count_as_hit,
                                                 std::chrono::system_clock::time_point now) = 0;

  // Sparse per-hash hit-count read. Returns an entry for each requested
  // hash that has a recorded count (hashes with no recorded hits may be
  // omitted). Replaces ExternalKvHitIndex::Lookup; backs the live
  // GetExternalKvHitCounts RPC. Pure read.
  virtual std::vector<ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes) const = 0;

  virtual std::size_t GetExternalKvCount(const std::string& node_id) const = 0;
};

}  // namespace mori::umbp
