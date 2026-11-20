# RFC: KV readiness estimates for UMBP routing

Status: draft for discussion; no runtime or wire change in this PR.

Baseline inspected: `23e73534431fc50637ecd7ed5ffbc7730165741a`.

## Problem and example

A prefix hit is useful only if the bytes become usable before recomputation would
finish. Replica locality alone cannot represent source congestion, destination
restore cost, staging pressure, or overlap with an engine's prefill queue.

For example, the longest-prefix worker may already have a long compute queue. A
different worker may finish sooner if it fetches the prefix while waiting for its
own compute slot. Conversely, a prefetch that misses that slot can add latency
and leave an extra replica that displaces useful objects.

## Current baseline

- [`LocalPreferringRouteGetStrategy`](../include/umbp/distributed/routing/route_get_strategy.h)
  is the default distributed read strategy: local replica first, otherwise a
  random replica. The former hard-coded HBM/DRAM/SSD rank was removed.
- [`PoolPolicy`](../include/umbp/distributed/pool/pool_policy.h) and its logical
  tier graph already order local storage choices and transitions. Do not replace
  that graph with a new hard-coded tier hierarchy.
- [`IUMBPClient::BatchPrefetch`](../include/umbp/umbp_client.h) already starts
  background promotion/pulling. Its return value is a count of copied or queued
  keys, not a completion handle or a guarantee that GPU input is ready.
- [`TransferEngine`](../include/umbp/distributed/transfer/transfer_engine.h) is
  the byte-moving boundary; backends publish endpoints. Existing engine timing
  and simulation are starting points for cost observations.
- [`RouteGetStrategy::Select`](../include/umbp/distributed/routing/route_get_strategy.h)
  currently receives locations and requester id, not compute queues, destination
  layout, or readiness deadlines.

## Ownership of the decision

MORI estimates data-path readiness. The serving router owns compute queues,
remaining prefill work, and execution placement. Neither `MediumBackend` nor
MORI-IO should inspect prompts, tokenizer state, or model schedulers.

Two decisions must remain distinguishable:

1. Which stored replica can deliver a requested range to a specified destination?
2. Which compute worker should execute the request after any restore?

The first can be a UMBP strategy. The second consumes its estimates in the
serving router. A MORI estimate must not be advertised as end-to-end TTFT.

## Proposed estimate surface

Illustrative value shape, not a new public API in this PR:

```text
EstimateRead(candidates, requested_ranges, destination, observation_snapshot)
  -> per candidate {
       estimated_ready_after_ns, uncertainty_ns, observation_age_ns,
       additional_resident_bytes, supported, path_class
     }
```

The destination includes memory location/device and the input layout needed by
the actual consumer. Metadata comes from descriptors and the adapter, not guesses
based on `TierType`. For a host destination the result is host-ready only; final
H2D, layout conversion, or rank broadcast must be added by the layer owning them.

Use observed fixed and size-dependent costs plus bounded queue estimates. Scope
observations to meaningful source/destination/path classes; do not share a single
bandwidth estimate across unrelated NICs, NUMA placements, or GPU paths. Use
local elapsed durations instead of subtracting clocks from separate machines.
Preserve uncertainty and sample age in the decision record.

An estimate is side-effect free: no lease, recency touch, staging allocation,
transfer, or new replica. An explicitly selected read still resolves at its
owner and follows the existing validation and retry protocol.

## Serving-router objective

For a fully overlappable restore, a useful starting model is:

```text
finish(worker) = max(compute_slot_ready, kv_ready) + remaining_prefill_time
```

Use durations relative to a single router observation. Charge only the transfer
tail not hidden by waiting. If the engine cannot overlap a stage, represent that
dependency explicitly rather than using `max` for the entire path.

The router may add a configurable cost for `additional_resident_bytes`, converted
to the same units. This prices pollution by a new cached copy; transient staging
bytes are a separate budget. Move away from the current choice only when the
predicted benefit exceeds model uncertainty and a measured switching margin.

## Completion and cancellation

Do not change the meaning of `BatchPrefetch`'s existing count. A future optional
prepared-read handle should separately express admitted, in-flight, data-ready,
validated, failed, and cancelled states. The first implementation may remain
observation-only until such a handle has a caller.

- A request cannot consume unvalidated bytes merely because an estimate elapsed.
- Cancelling a request stops future scheduling; it does not imply in-flight DMA
  stopped or that source/destination buffers may already be reused.
- Release transfer holds only after the existing engine/backend completion
  contract permits it. Keep per-request scratch private until that point.
- Bound prefetched bytes and in-flight work; coalesce duplicate requests where
  compatible and record when demand reads are delayed by prefetch.
- Distinguish temporarily busy from missing and permanently failed. A staging
  queue being full is not a cache miss.

## Integration and compatibility

Start with an opt-in strategy using an injected, immutable observation snapshot;
do not add a synchronous network probe per candidate or per block. Preserve
batched selection. Coalesce path observations outside the route hot path.

Keep existing strategies as explicit fallbacks when a path is unsupported,
observations are stale, or there are too few samples. Do not remove logical-tier
configuration. Invalid estimates must not turn successful reads into failures.
Support in-memory and Redis metadata stores without process-local correctness
state; a process-local performance estimate is advisory and can be discarded.

The first adapter can use the current prefetch API and instrument its actual
completion internally. Publishing a stable cross-process readiness API is a
separate implementation milestone, with mixed-version capability checks.

## Implementation sequence

1. Define estimate inputs, units, path classification, freshness, and trace fields.
2. Add an observation-only estimator and compare predictions with real reads.
3. Add an opt-in replica selector after prediction error is characterized.
4. Integrate a single serving router for execution placement and queue overlap.
5. Introduce a completion handle only when required by that integration.

## Validation and acceptance

Deterministic cases: equal estimates; missing samples; expired observations;
negative/non-finite costs; overflow; local congested versus remote idle sources;
different destination devices; empty candidate lists; per-key batch failures;
overlap disabled; busy staging; cancellation during transfer; and generation
validation failure after a successful byte transfer.

Compare local-first, configured tier policy, and readiness-aware selection on the
same hardware, placement, request trace, and offered load. Sweep concurrent
restores, object/range sizes, queue lengths, and HBM/DRAM/SSD paths actually enabled
in the deployment. Report estimate error, exposed restore time, P95/P99 demand
read latency, additional copy bytes, route CPU overhead, and serving TTFT when an
engine is present. Keep bandwidth and hit rate as explanatory metrics, not the
sole success criteria. No performance improvement is asserted by this RFC.

Simulation can rank candidate policies but does not validate switch congestion
or end-to-end GPU overlap; reproduce selected cases on real devices.

## Open review questions

- Which first consumer can supply an honest compute-slot estimate?
- Can existing transfer completion observations cover H2D and rank broadcast, or
  must the adapter report those separately?
- What freshness/error threshold provides a useful fallback under bursts?
- Should replica estimates be exposed beside selection, or only in trace output
  until a serving router commits to using them?

## Related work

[Evolving Infra](https://arxiv.org/abs/2611.12304) combines queue-aware placement,
asynchronous KV restore, and duplicate-copy cost. The proposal adapts that idea to
MORI's existing descriptor-based transfer layer and pluggable routing rather than
copying the paper's particular model or hardware coefficients.
