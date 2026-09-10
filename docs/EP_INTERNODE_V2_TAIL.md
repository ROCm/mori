# EP internode: reading a bimodal bench

Read this before running an A/B on the internode dispatch/combine path. The most
expensive mistake available here is to compare two configurations across runs
that landed in different regimes and believe the difference.

## The shape

The small-token internode bench is bimodal per RUN, not per round. Every run
lands in one of two regimes and holds it:

|                  | fast    | slow     |
|------------------|---------|----------|
| dispatch mean    | 37-40us | 50-70us  |
| combine mean     | 45-48us | 72-79us  |
| host wall/round  | ~105us  | ~150us   |

A slow run does a handful of rounds at fast-regime numbers, steps up over a
single round, and stays there. The reported worst/mean ratio is an artifact of
that step -- the mean blends the two levels -- so it is not a tail of a few bad
rounds and it does not respond to geometry tuning.

## The cause: two ranks on one physical core

Two ranks landing on the two SMT siblings of a single physical core make both
host loops run about twice as slow, and because the kernels spin on their peers,
one slow host loop drags the whole collective into the slow regime.

`application::BindCallingThreadToGpuNumaOnce()`
(`include/mori/application/utils/cpu_affinity.hpp`) is what prevents this. It is
called from `ccoCommCreateImpl` as well as from shmem init, so a CCO-only job is
bound too; before that wiring, CCO jobs ran completely unbound and any EP v2 vs
v1 comparison was measured across a bind difference.

The bind always restricts the thread to its GPU's NUMA-local CPUs. It
additionally gives the rank its OWN physical cores -- partitioning those CPUs by
`thread_siblings_list` and handing each rank a disjoint slice, so no two ranks
share a core -- but only when BOTH hold:

- `detail::DevicesOnNumaNode()` sees more than one GPU on that NUMA node. It
  enumerates `hipGetDeviceCount()`, i.e. the devices VISIBLE TO THIS PROCESS. A
  launcher that gives each rank a single GPU through `HIP_VISIBLE_DEVICES` makes
  this 1 and the split is skipped.
- the usable CPUs group into at least that many physical cores. Otherwise the
  code warns and binds the whole NUMA node.

In either case the result is the whole-NUMA-node bind, which is exactly the
placement that lets two ranks share a core. Both branches log through the
`application` module, whose default level is `ERROR`
(`include/mori/utils/mori_log.hpp`), so a user who has set nothing sees neither
the success nor the fallback.

Check that the split really happened on YOUR run rather than assuming it:

```
MORI_APP_LOG_LEVEL=info   # then look for one line per rank:
# CPU affinity: GPU <d> takes slot <s>/<n> of numa node <nn> (<k> cores)
```

No such line, or the `binding to the whole node` warning, means the ranks on a
node were never separated and a collision is still possible.

Env knobs:

- `MORI_IGNORE_CPU_AFFINITY=1` -- skip the bind entirely. Use it as the control
  when you want to reproduce the unbound behaviour.
- `MORI_CPU_AFFINITY_NO_SPLIT=1` -- force the whole-NUMA-node bind instead of
  disjoint physical cores. Collisions become unlikely rather than impossible.

## Recognising it

Run the bench with `MORI_EP_ROUND_SERIES=1` and read the per-rank host wall
(`hwal`) series it prints. In every observed case the correspondence has been
one-to-one: a run in the slow regime has at least one
rank whose host loop is almost exactly 2x the others, and a run with no doubled
rank is in the fast regime. `sched_getcpu()` on each rank identifies the
colliding pair -- compare CPU numbers WITHIN a node, since the same CPU number
on two different nodes is not a collision.

A residual offset of roughly 10-20us still appears in a minority of runs with no
doubled rank and no collision. It is smaller than the 2x effect, its mechanism is
not known, and everything measurable outside the GPU has been ruled out.

## A/B method

- Always INTERLEAVE the two configurations. Do not run all of A then all of B:
  the regime is per-run, so a block of runs can sit in one regime for reasons
  that have nothing to do with the change under test.
- Compare paired differences within a rep, not a difference of medians. The
  regime can move during a sweep, and pairing cancels that.
- Report the worst round as well as the median. A geometry that gains at the
  median and gives it back at the worst is a loss for a latency-bound collective.
- Confirm both nodes are idle before starting.
