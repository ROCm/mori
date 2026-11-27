# RFC: Role-independent KV pool lifetime and safe handoff

Status: draft for discussion; no runtime or wire change in this PR.

Baseline inspected: `23e73534431fc50637ecd7ed5ffbc7730165741a`.

## Problem and scope

An upper-layer scheduler may reassign a worker from prefill to decode, or the
reverse. Reassignment should not by itself destroy discoverable KV, invalidate
registered endpoints, or allow an old transfer to act on a newer copy.

UMBP already separates storage and transfer. The missing contract to review is
the lifetime relationship between the serving worker, the storage owner, the
published directory state, and outstanding reads during reassignment.

This proposal specifies that contract and a validation plan. It does not claim
that MORI can switch a serving engine's scheduler loop, remap model weights, or
reconfigure tensor/expert parallel groups. Those are upper-layer responsibilities.

## Existing mechanisms to reuse

- [`IUMBPClient`](../include/umbp/umbp_client.h) distinguishes direct storage owners
  from standalone-process forwarders. `Close`, `Clear`, and `Flush` already have
  different lifecycle meanings.
- [`MediumBackend`](../include/umbp/distributed/peer/backend/medium_backend.h)
  owns memory and registration through `MemoryRegistrar`; the transfer engine
  moves bytes between published endpoints.
- `ResolvedEntry::generation`, `BatchStillHolds`, `BatchFinishReads`, local holds,
  and migration pins already protect or validate particular copies. The RFC must
  build on these, not introduce a second unrelated copy-version mechanism.
- [`PeerPool::ValidateResolved`](../include/umbp/distributed/pool/peer_pool.h)
  checks the resolved backend/key/generation after a remote read.
- [`KV Indexer reporting`](kv-indexer-export.md) already supports shared-pool
  identity, committed-DRAM events, bounded repair, and forwarding clients.
  Its documentation explicitly excludes crash-time TTL/fencing and HBM/SSD
  publication; those gaps are not solved by claiming role independence.

## Three independent lifetimes

1. **Compute role:** an upper-layer admission choice, potentially changed often.
2. **Storage-owner lifetime:** the process/backend instance holding the bytes,
   registrations, service endpoints, and publication responsibilities.
3. **Physical-copy lifetime:** one committed backend/key/generation, which may
   disappear through ordinary reclamation while the owner remains alive.

A role change is neither `Clear` nor `Close`. An owner restart is not a role
change. A cache hit is not a transfer lease. Keeping these distinctions explicit
prevents accidental promises about durability or permanent residency.

## Proposed integration contract

### Role-only reassignment

The serving adapter stops admitting old-role compute work and drains or migrates
that work according to its own protocol. It keeps the storage owner alive and
does not call `Clear`, deregister buffers, reset pool identity, or withdraw
committed KV solely because the compute role changed.

An in-process owner works only while the engine process survives reassignment.
If the serving implementation restarts that process, use a separate long-lived
storage owner with a supported forwarding path, or explicitly accept a cold
cache. A standalone server does not automatically preserve engine-private HBM.

Existing registered data may remain readable from either compute role, subject
to normal eviction and endpoint reachability. No extra copy is required solely
to rename a role. Whether the new role can use that layout is a separate
compatibility check by the serving adapter.

### Copy handoff

Illustrative states, not a new runtime API in this PR:

```text
source advertised
  -> destination reserved
  -> copy completed and source generation validated
  -> destination committed
  -> destination residency acknowledged for the intended consumer
  -> source preference may be released
```

Any failure before commit aborts the destination reservation after transfers can
no longer access it. It leaves the source eligible for normal reads. Failure
after commit may leave two copies; retaining an extra copy is preferable to
claiming that a missing destination is ready. Ordinary eviction remains possible;
this is a safe optimization protocol, not guaranteed cache durability.

Destination residency acknowledgement must name the exact destination owner and
copy. A successful byte transfer, a queued index event, and an indexer acknowledgement
are distinct milestones. Select the milestone needed by the consumer; a direct
handoff reader can resolve the destination without waiting for global publication.

For same-key movement, do not infer that the old bytes survive from key equality.
Use the existing generation/validation contract. Legacy or unstamped backends
need an explicit supported fallback; a generation of zero is not evidence of
protection across an unbounded transfer.

### Owner shutdown or restart

Before graceful owner shutdown, stop new resolves, finish or safely fail in-flight
transfers, withdraw owned publications, then deregister and free endpoints in the
order required by the existing transfer contract. Specify and test that order in
the adapter rather than replacing it with a timeout.

After a crash/restart, stale descriptors and delayed directory events need
owner-incarnation fencing before this protocol can promise transparent handoff.
Define a fresh owner incarnation in addition to current per-copy generation:
delayed old-incarnation events and acknowledgements must not withdraw or validate
new data. Namespace and capability negotiation remain design decisions for a
follow-up implementation; crash-transparent handoff is not enabled first.

## External cache publication is not ownership

Engine-managed cache publication supplies scheduling metadata; it does not, on
its own, authorize or implement reads from that engine's private memory.

Phase one covers UMBP-owned committed storage only. Direct reads of engine Host
KV require a separate exporter adapter that can resolve a block, protect its
copy while a reader operates, validate completion, and withdraw or fence stale
exports. Keep byte movement in `TransferEngine`/MORI-IO. Do not make an external
index entry masquerade as a UMBP-owned allocation.

Likewise, symmetric addressing in MORI-CCO does not imply support for changing a
communicator's membership, EP group, or a model's parallel layout at runtime.

## Compatibility and rollout

- No default behavior changes or new role API inside MORI in this RFC.
- Provide one serving integration contract and an example with a persistent
  owner; older integrations may continue to restart and lose cache normally.
- Reuse existing pool identities, backend ids, descriptors, copy generations,
  and transfer validation. Add an owner incarnation only at a boundary that
  actually requires restart fencing.
- Start with graceful role changes. Gate restart-transparent behavior on explicit
  capabilities from owner, adapter, and directory protocol.
- This proposal is independent of agent-retention and readiness-estimate work:
  neither is required to test lifetime separation and safe failure.

## Validation and acceptance

First use mock backends and the existing benchmark/trace infrastructure, then
repeat critical cases on two ROCm GPU nodes with real RDMA:

- Reassign P to D and D to P while retaining the same storage owner; read every
  still-resident sentinel and verify bytes, not only directory hit counts.
- Close a forwarder while the owner serves another client; verify registrations
  and committed pool publications survive as documented.
- Race reassignment with eviction, prefetch, migration, and in-flight reads.
  A stale copy must produce a retry/failure rather than accepted corrupted data.
- Inject destination no-space, copy failure, lost commit response, and delayed
  publication. Verify bounded cleanup and no premature source-preference release.
- Exercise `Clear`, graceful owner shutdown, and owner restart separately.
  Until incarnation fencing exists, restart recovery must fail closed and rebuild
  directory/descriptor state rather than claim transparent handoff.
- Test missing, busy, and unsupported exporter cases without treating all of
  them as equivalent misses.

Measure role-switch service interruption, surviving KV reuse, re-registration
and connection-setup counts, transient duplicate bytes, handoff retries, and
post-switch TTFT in the serving integration. Use an unchanged-role control and a
restart-based baseline with equal resources. No speedup or lossless cache
guarantee is asserted by this RFC.

## Open review questions

- Which serving stack can first change role without destroying its owner?
- Is persistent host storage sufficient initially, or does a consumer require
  engine-private HBM exports in the first milestone?
- Which publication acknowledgement is necessary for that first consumer?
- Can the existing indexer protocol carry owner-incarnation fencing compatibly,
  or must that recovery capability be negotiated out of band?

## Related work

[Evolving Infra](https://arxiv.org/abs/2611.12304) motivates separating a worker's
P/D role from KV reachability. Its
[architecture](https://github.com/mit-han-lab/EvolvingInfra/blob/main/docs/architecture.md)
combines a symmetric serving engine with a shared host pool. This RFC addresses
MORI's storage/transfer contract; the symmetric engine and planner remain separate.
