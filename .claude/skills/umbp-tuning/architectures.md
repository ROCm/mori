# Tuning for the model architecture

What the model needs stored, where, and how often. Do this before tuning the pool: placement sets the hit the pool can reach at all, and no tier setting recovers a hit the placement rules out.

`SKILL.md` covers the pool; `tiers/` covers the medium. This file covers the model.

## The idea

A prefix hit is usable only if everything a resumed request needs is present at the resume point: the KV of every token before it, plus any state the model carries that cannot be derived from that KV.

```
prompt tokens  ───────────────────────────────────────►
KV             ████████████████████████████████████      dense, one object per block
state               ▲           ▲           ▲            only at saved positions

resume at L:   usable only if all KV below L AND a state at L are present
```

KV is the easy part. State is what differs per model, so the tuning problem is: at which positions is state kept, in which tier, in how many copies, and how is it kept alive exactly as long as the KV it unlocks.

## Placement variables

| Variable | Choices |
|---|---|
| Rebuild or store | recompute the state from a short window, or save it |
| Positions | prompt ends, shared-prefix junctions, a fixed grid, a denser window near the prompt end |
| Tier | GPU cache and pool, or pool only |
| Copies | one per TP rank, or one shared copy when ranks hold identical bytes |
| Eviction coupling | aged and refreshed with its KV; demoted or dropped when it leaves the entry tier |
| Object and page size | pool page close to the object size |

## Stage A — Inventory

List the cache groups the engine registers. For each one fill in:

| Field | Why it matters |
|---|---|
| Kind: per-token KV, recurrent state, sliding-window state, side buffer | decides whether it is dense or exists only at saved positions |
| Bytes per token or per saved state | sets spacing and tier (stage C) |
| Objects per restored prefix | sets lookup cost |
| Per rank or identical across ranks | sets copies |
| Needed at the resume point? Rebuildable? | sets whether it must be stored at all |
| What must be present with it | defines a usable hit |

Take sizes from the objects the pool actually stores, not from the model card.

## Stage B — Bound

Compute two numbers from the request stream, using the engine's own notion of a shared prefix:

- the hit an unlimited cache reaches with state at **prompt ends only**;
- the hit it reaches with state at **every token**.

If they are close, the workload is mostly linear and default placement is enough: stop here and go to the pool stages. If they are far apart, requests resume inside prompts (branching agents, sub-agents) and state is needed there.

Use the engine's bound, not the load generator's. A generator can credit sharing that the rendered prompts do not have.

## Stage C — Candidates

Answer in this order; each answer narrows the next.

| # | Question | It decides |
|---|---|---|
| 1 | Can the state be rebuilt from a short recompute, and does the engine support that? | Store none for that group. Fewest bytes, and usually the best hit. |
| 2 | Where do requests resume on this workload? | Positions: prompt ends only, or a grid inside prompts, or a dense window near the end. |
| 3 | How big is one state next to the KV it unlocks? | Spacing. A cheap state affords a fine grid; a costly one gets the coarsest grid the bound tolerates. |
| 4 | Does the extra state fit in the GPU cache without pushing out KV? | Tier. Pool only, unless the state is small next to the KV it covers. |
| 5 | Are the bytes right? | One copy of rank-identical objects; page size near object size. Do this before adding capacity. |
| 6 | How costly is the lookup at the largest prompt? | Many small objects make lookup, not load, the bottleneck: check key count per request, batch limits, timeouts. |

One rule on top: **evict a state together with the KV it unlocks.** Either alone is useless. Never age state faster than its KV, refresh every state along a GPU hit, and drop short-lived extras on eviction instead of demoting them.

## Stage D — Search

Replay the request stream offline under each candidate, with unlimited capacity:

1. Build the prefix structure of the stream in arrival order.
2. For each request, resume from the deepest saved state at or below its shared prefix; save the candidate's positions above that point.
3. Record hit loss against the every-token bound, and state bytes written.
4. Plot loss against bytes and pick the knee: the cheapest placement within a small distance of the bound.

Calibrate the replay on one measured run before trusting it.

| The replay is good for | It is not good for |
|---|---|
| ranking placements and choosing spacing | a finite GPU cache under high load |
| estimating state write volume | eviction: it assumes a state and its KV leave together, and a real pool evicts per key |
| | lookup failures, timeouts, load races |

## Stage E — Verify

Run one cell with per-request hit logging and compare against the engine's bound, per request.

| Loss class | Meaning | Change expected to move it |
|---|---|---|
| Branch inside a prompt | no state where the request branched | finer grid, a window near the prompt end, longer life for short-lived state |
| Extends a prompt, end state gone | the prompt-end state was evicted or stored too late | store ordering; hold looked-up keys until the load |
| Bound above the nearest saved position | spacing too coarse | finer spacing |
| State and KV not both present | they were evicted separately | couple their eviction; on SSD see `tiers/ssd.md` |
| KV evicted | the pool dropped whole blocks | not a placement problem: go to the pool stages |
| Hit cannot end inside a block | structural at this granularity | none at this block size |

Before believing any hit number, check:

- **Health:** zero failed loads, zero failed or timed-out lookups. Silent failures look like placement loss: an oversized existence check that returns all-false, a short lookup timeout counted as a miss, state read back misaligned.
- **Correctness:** whenever state handling changes, confirm the first tokens generated after a restored prefix match a cold run.

Fix the largest class, repeat. What remains once placement loss is small is a pool problem: continue with stage 0 of `SKILL.md`.

## Archetypes

| Architecture trait | Resulting placement | Example |
|---|---|---|
| Recurrent state over the whole prefix; large; cannot be rebuilt | State on the densest grid the scheduler allows, in GPU cache and pool; a pool-only window of extra states near the prompt end, short-lived and dropped on eviction | Kimi-K3 |
| Sliding-window or compressor state; about the cost of a few thousand tokens of KV; not rebuildable in the engine | State on a grid in the pool only, because grid states in the GPU cache push out KV; one copy across ranks; lookups batched for many small objects | DeepSeek-V4-Pro |
| Sliding-window state that a short recompute rebuilds | No state stored; KV only; small pool page to match small objects | DSv4.1-Flash |

A new model is usually a mix. Run stage A per cache group and apply the matching row to each group.

## mori-side names

Engine-side settings (state positions, pool-only state, replay) belong to the engine integration. On the pool side:

| Purpose | Name |
|---|---|
| Pool page size | `UMBP_DISTRIBUTED_DRAM_PAGE_SIZE` |
| Refresh pool copies on a GPU hit | `BatchTouch` |
| Existence lookup | `BatchExists` (keep each call under the RPC message limit) |
