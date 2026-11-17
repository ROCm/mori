---
name: umbp-tuning
description: >-
  Framework for analyzing and tuning UMBP performance on any tier (HBM, DRAM,
  SSD, remote). Decomposes a throughput or TTFT gap into hit loss, load latency
  and write pressure, attributes it to one mechanism, maps that mechanism to a
  lever, and verifies the change. Use when the user asks why a UMBP pool is
  slower than expected or than another pool, wants to tune or size a UMBP tier,
  or wants to plan a tuning experiment. Tier-specific mechanics live in tiers/
  (SSD is written; the others are not yet).
---

# UMBP tuning framework

A method, not a list of results. It works on any model, engine (vLLM, SGLang) or cluster.
Follow the stages in order; do not jump to a knob before stage 2 names a mechanism.

## Ground rules

- **Analyze existing data first.** A run directory, server log or metrics dump usually answers stages 0–2.
- **A new run is a proposal.** State what it would show and ask before launching anything on a cluster.
- **Grade every claim:** measured (with the file), inferred, or unverified. Never report an inferred cause as found.
- **One variable per change.** If several settings moved together, say the split between them is inferred.

## The model

A pool tier can cost performance in only three ways.

```
                      gap vs reference
                             │
            ┌────────────────┴─────────────────┐
   engine does MORE work               engine does the same work,
   per request                         but slower or later
            │                                  │
      1. HIT LOSS                       2. LOAD LATENCY
   the pool did not serve          a pool read sits on the
   tokens it was given             request's critical path

      3. WRITE PRESSURE   (feeds 1, and wears the medium)
   data arrives faster than a tier can take it or pass it down
```

## Stage 0 — Validity

Answer both before reading any number.

| Check | How |
|---|---|
| Is the comparison fair? | Same nodes, image, workload, load level and warmup. Only the pool differs. Know the run-to-run spread on this hardware; a gap inside it is not a gap. |
| Are the settings in effect? | Read the *effective* config from the pool server's startup lines and policy file, not from the environment you set. A setting can be silently ignored (wrong process, wrong mode, unknown key, older build). |
| Was the tier exercised? | Check the tier filled and evicted during the measured window. A pool that never evicts says nothing about eviction. |
| Is the engine saturated? | If not, throughput equals offered load and hides everything; compare TTFT instead. |

## Stage 1 — Decompose

At saturation:

```
prompt throughput  =  compute rate  ÷  computed share
                      (new tok/s)      (1 − hit rate)
```

Compare each term with the reference separately.

| Observation | Branch |
|---|---|
| Computed share higher, compute rate equal | **Hit loss** → stage 2A |
| Compute rate lower | **Engine slowed** → stage 2B (blocked or starved steps, host contention) |
| Both equal, TTFT worse | **Per-request latency** → stage 2B (load wait, rescheduling) |
| Puts refused, or a tier's inflow exceeds its drain | **Write pressure** → stage 2C |

Caveat: compute rate depends on *which* tokens are computed (deep tokens cost more), so compare it only between runs with similar hit rates, or on the same node.

## Stage 2 — Attribute

### 2A. Hit loss

First get the ceiling: what an unlimited cache would have hit on this workload (replay the request log). Loss is ceiling minus measured. Then, for each lost span, ask what the pool held *at lookup time*:

| Class | Meaning | Lever family |
|---|---|---|
| Never stored | put refused, skipped, or not landed yet | store policy, entry-tier headroom |
| Evicted whole | the pool held it and dropped all of it | capacity, inflow reduction |
| Partly present, unusable | pieces remain but not an unbroken, matching run | eviction unit and order |
| Short-lived state gone | data that lives in one tier only left before reuse | residence time, eviction class |
| Present, not served | load failed, lease expired, lookup timed out, staging busy | read-path capacity, correctness |
| Present elsewhere | on another node or tier the reader did not reach | placement, routing, visibility lag |

Only the first five are tunable by the pool; tokens that were never cacheable belong to the ceiling.

### 2B. Load latency

First question: **is a pool read on the critical path at all?** That is set by the engine's load model.

| Load model | What a slow read costs |
|---|---|
| Asynchronous, before scheduling | Nothing while the request would have queued anyway. The cost is parking: load time plus the wait to be scheduled again. |
| Blocking, inside the step | Forward time for every request in the batch, including decodes sharing the engine. |

Then split one load along its stages and find the dominant one:

```
lookup → wait for a read slot → medium read → staging copy → host-to-GPU → reschedule
```

Also check the engine is not starved: steps full, scheduler time small, no idle gaps between forwards.

### 2C. Write pressure

For each tier compare inflow rate with drain rate (eviction plus demotion).

| Symptom | Meaning |
|---|---|
| Refused allocations when a tier first fills | entry tier has no headroom over its watermark |
| Demote volume far above what is ever read back | storing data nobody reuses |
| Bytes written per day far above the medium's rating | endurance problem, not a speed problem |

## Stage 3 — Choose a lever

Pick the family the class points to, then the concrete knob from the tier module.

| Family | What it changes |
|---|---|
| Capacity | tier size; pooling more nodes or drives |
| Eviction unit and order | granularity of reclaim, recency hints, watermarks |
| Residence time and classes | entry-tier size, per-class aging, demote or drop |
| Store policy | what is stored, how often, dedup, retiring dead data (cuts inflow, so acts like capacity) |
| Placement and promotion | which tier or node takes a put; whether a read moves data up |
| Read path | concurrency, staging, ranged vs whole-object reads, coalescing, prefetch, load model |
| Integrity overhead | checksums, sync-on-write |
| Host resources | hugepages, NUMA placement, worker threads |

Prefer the lever that moves the diagnosed class and nothing else. If two families could explain the class, the next experiment should separate them.

## Stage 4 — Verify

1. Write the prediction first: which class or stage should move, and roughly how far.
2. Change one variable; keep nodes, image and workload fixed.
3. Re-run the stage 2 attribution, not just the headline number.
4. Accept only if the predicted class moved. A better headline with the wrong class moving means the cause is still unknown.
5. Note the cost the lever introduced (latency, memory, writes, file count).

## Sizing rules

Derive sizes from rates measured on the workload; do not copy sizes from another model.

- **Lower tier capacity** ≥ demote rate × the longest idle gap a session must survive.
- **Entry tier capacity** ≥ inflow rate × the reuse gap of data that never moves down, plus headroom above the watermark for bursts and promotions.
- **Reclaim unit** relative to object size: the closer to one object, the less collateral eviction.
- **Read staging** scales with concurrent readers and largest object, not with data volume.

## Signals

| Where | What it shows |
|---|---|
| Engine cache counters, split by GPU vs external | hit rate and which layer served it |
| Engine per-request hit log | input to the 2A classification |
| Pool server startup lines | effective config per tier (stage 0) |
| Pool server activity summary (every `UMBP_SUMMARY_INTERVAL_MS`, default 60 s) | per tier: offload, load, evict, move-read, move-write, resident bytes and keys; promote and demote counts with failed, retried, dropped |
| Pool `/metrics` (standalone port 9092, master 9091) | `mori_umbp_client_tier_read_hits_total`, `..._tier_offloaded_bytes_total`, `..._tier_promoted_bytes_total`, `..._tier_transitions_total{outcome}`, `..._local_evict_total{event}`, `..._logical_tier_capacity_utilization_ratio`, `mori_umbp_backend_op_seconds_total{tier,op}`, `mori_umbp_backend_medium_events_total{event}`, `mori_umbp_backend_medium_state{state}` |
| `Allocate reason=NO_SPACE` warnings | refused puts (2C, and "never stored" in 2A) |
| Host: memory, hugepages, device utilization | contention and whether the medium is really being used |

Authoritative knob list and defaults: `src/umbp/doc/runtime-env-vars.md`. Read it from the commit the deployment was built from, since defaults and available knobs change between commits.

## Report format

Lead with the branch and class found, then: evidence (file and line or counter), the lever proposed, the prediction, what is still inferred, and the run that would confirm it.

## Tier modules

- `tiers/ssd.md` — SSD-specific classes, levers, signals and sizing.
- HBM, DRAM and remote modules are not written. For those tiers apply the stages above and say which tier-specific mechanics you had to work out from the source.
