---
name: cco-fusion-facts
description: >-
  Measured facts, each with its source and scope, for mori's fused GEMM +
  collective operators and the cco SDMA path: per-put and per-packet costs,
  fence and acquire costs, counter and wrap behaviour, compiler effects, phase
  breakdowns, measurement floors and environment traps, partitioned by GPU
  architecture, plus the claims that still have no source. Use when a design,
  a bound, a diagnosis or a review needs a number: cite the fact ID instead of
  quoting from memory. Also use when adding a newly measured fact or
  correcting one.
---

# cco-fusion-facts

Numbers that designs and bounds in the `cco-fusion-loop` skill depend on. Cite a fact
by its ID (for example `G950-SYNC-02`). A number that is not here is either measured
in the current task, with its run record, or labelled `inferred`.

## Files

| File | Holds |
|---|---|
| [gfx950.md](gfx950.md) | MI355X facts, and the claims that have no source yet |
| [gfx1250.md](gfx1250.md) | Nothing measured yet; why gfx950 numbers do not carry over |
| [corrections.md](corrections.md) | Every change to a fact or a claim, with its evidence |

## Labels

| Label | Means |
|---|---|
| `measured` | Benchmarked, with the shape, dtype, ranks, node and commit stated in the source |
| `source-reported` | Stated by code, a comment or a document in this repository, not re-measured here |
| `inferred` | Derived from other facts or from reading code; not measured |
| `unverified` | Stated somewhere (an agent prompt, a PR description) with no source in this repository |

## Rules

- **A fact does not transfer** across architectures, and often not across models or
  shapes. On DeepSeek-V4-Pro fusing is most of gemm_ar's gain; on V4.1-Flash it is the
  smallest of three effects (`G950-GAR-03`, `G950-GAR-05`).
- **Each fact states** its value, its scope (shape, dtype, ranks, node, commit, date
  where known), its source as `path:line` at a named commit, and its label.
- **Adding a fact:** give it the next ID in its group, link the run record or the
  commit that measured it, and give its scope. A fact from a task under way is added
  once the verifier has measured it, not from the implementer's development runs.
- **Changing a fact:** never edit a value in place silently. Add an entry to
  [corrections.md](corrections.md) with the old value, the new value and the evidence,
  then update the fact.
- **Searching for a fact that is not here:** a negative result names the terms and the
  paths searched and the commit.
