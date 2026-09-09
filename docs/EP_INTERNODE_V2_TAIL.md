# EP internode v2: the "tail" is a bistable whole-run regime

Notes from investigating why the v2 CCO/GDA internode bench reports a large
worst/mean ratio at small token counts. Kept out of the test's module docstring
so that file stays close in shape to the examples harness it mirrors,
`examples/ops/dispatch_combine/test_dispatch_combine_internode.py`.

Read this before running another A/B on this path. The single most expensive
mistake available here is to compare two configurations across runs that landed
in different regimes and believe the difference.

WHAT IT ACTUALLY IS
-------------------
Not a tail. Every run of the 4-token EP16 bench lands in one of two regimes and
stays there:

                     fast regime        slow regime
    dispatch mean      37-40us            50-70us
    combine  mean      45-48us            72-79us
    host wall/round    ~105us             ~150us

Per-round series (`MORI_EP_ROUND_SERIES=1`) show the transition directly: a run
that is going to be slow runs 4-8 rounds at fast-regime numbers, steps up over a
single round, and holds the higher level for every remaining round. A fast run
never steps. Within a regime the per-round scatter is small.

The reported worst/mean ratio is an artifact of this: the mean is a blend of the
two levels, so a run that stepped early reports ~2.3x while the same binary that
did not step reports ~1.2x. Chasing "the tail" as if it were a few bad rounds is
chasing the wrong shape.

IT IS ENVIRONMENTAL, AND WE AMPLIFY IT
--------------------------------------
Interleaved against the #625 baseline (v1 host + the same v2 CCO kernels, so the
same transport on the same wire), four pairs back to back:

                    #625 dispatch   #625 combine   ours dispatch   ours combine
    fast pair 1        39.8            41.8           40.3            48.1
    fast pair 2        38.4            40.3           38.0            47.4
    slow pair 3        41.1            51.0           52.8            78.7
    slow pair 4        40.7            50.6           69.9            72.0

Both implementations step into the slow regime on the same runs -- so the trigger
is environmental, not ours -- but ours degrades about three times as far
(combine +60% against #625's +22%, dispatch +32% against +2%). Whatever the
trigger is, our path is more sensitive to it. That sensitivity is the thing worth
fixing; the trigger itself is not in our code.

The step is NOT caused by, all eliminated by direct measurement:

  * the caching allocator -- `segment.all.allocated`, `num_device_alloc` and
    `num_alloc_retries` are all +0 across the timed loop, in every run
  * launch-queue depth -- `--per-round-drain` (a bare `cuda.synchronize()` each
    round, no barrier) does not remove the step
  * warm-up -- `--warmup 200` steps at the same place as `--warmup 20`
  * RoCE congestion or loss -- `~/harness/_nic_snap.sh` before and after: 0 for
    np_ecn_marked_roce_packets, np_cnp_sent, rp_cnp_handled, out_of_sequence,
    packet_seq_err, roce_adp_retrans, rx/tx_roce_discards, out_of_buffer
  * PFC pause -- `~/harness/_pfc_snap.sh`: rx/tx_pfc_ena_frames_pri5 and the
    pri5 transition counters are all +0, fast runs and slow runs alike
  * GPU clocks -- sampling pp_dpm_{sclk,fclk,socclk} across the loop window puts
    sclk within 4% and fclk, if anything, HIGHER on the slow runs
  * xGMI per-link power down -- `~/harness/_plpd.sh 0` (plpd_disallow on all 16
    GPUs) against stock, four interleaved pairs: slow runs appear under both
  * PCIe -- `~/harness/_pcie_snap.sh` before and after: no link speed or width
    change on any GPU or NIC, and TOTAL_ERR_COR/NONFATAL/FATAL all +0, on slow
    runs as much as fast ones
  * the VMM/dma-buf placement problem in `.claude/skills/known-issues`
    (Issue 1) -- both nodes have CONFIG_DMABUF_MOVE_NOTIFY=y and
    CONFIG_PCI_P2PDMA=y on the running 6.8.12 kernel, and `vmm_peer_probe`
    returns `VERDICT: OK` on both. Its *family* is right (see the pass split
    below) but its specific compile-time cause is absent here
  * the host -- see below

A note on the counter probes: `_nic_snap.sh` now reads tx_pkts/tx_bytes/
rx_pkts/tx_write_req alongside the error counters, as a check on the probe
itself. A 60-round run moves +20111 packets / +49.6 MB / +12960 write requests
on each node, so the zeroes above are measured zeroes and not an unwired probe.
That check was added after noticing that a table of zeroes reads identically
either way.

`--per-round-sync` cannot be used to test any of this: its gloo barrier costs
~1.3ms a round here and injects far more rank skew than it removes (dispatch
reads ~1370us under it). `--per-round-drain` is the usable version.

ONLY THE PASSES THAT TOUCH REMOTE MEMORY DEGRADE
------------------------------------------------
The split also attributes the STEP, by first-k rounds against last-k (a
different question from which pass owns the worst round, and it need not have
the same answer). Three runs:

              copystaging   dispatch_ll   combinesync  combine_ll  combineall
    rep1        +0.0          +33.1         +0.1        -0.3        +0.2
    rep2        +0.0          +45.8         +0.1       +47.3        -0.0
    rep3        +0.0          +67.0         +0.0       +66.0        +0.0

`dispatch_ll` and `combine_ll` are the passes that read and write PEER memory.
`copystaging` (7.0us, local staging copy), `combinesync` and `combineall` touch
only local memory. The local passes are flat to under 1% in every run; the
remote ones roughly double.

So the degradation is confined to the remote-access path, and is not a general
slowdown of the GPU. That is the same family as known-issues Issue 1 -- peer
traffic getting more expensive than it should -- even though Issue 1's own
compile-time cause is ruled out here. The remaining candidates all live in that
family: NIC address-translation/MTT cache pressure, or page placement of the
symmetric window.

WHERE THE EXCESS LANDS: THE PER-PASS SPLIT
------------------------------------------
`MORI_EP_SPLIT_PASSES=1` launches the eight passes individually instead of as one
`LaunchGroup` and stamps an event after each, then prints, per rank, the round
series and the three worst rounds broken down against the median round. Splitting
roughly triples the absolute numbers -- eight ABI crossings instead of one, and
every rank's spin-wait pass absorbs the others' launch skew -- so read the
deltas, not the levels.

Dispatch, worst round against median (three ranks agreeing):

    copystaging     7.0us     median 7.0us      +0.0
    dispatch_ll   304-307us   median 114us    +190us

Combine:

    combinesync           6.4us    median 6.4us     +0.0
    combine_ll           29.6us    median 29.4us    +0.2
    combinesyncbarrier  152.3us    median 79.0us   +73.4
    combineall            6.1us    median 7.0us     -0.8

So on both legs the entire excess is in the one pass that waits on peers, and
none of it is in the passes that do local work. `combine_ll` -- combine's actual
data movement -- is flat to 1% even on the worst round: combine has no tail of
its own, it inherits one through its barrier. Any fix aimed at combine's
data path is aimed at the wrong pass.

THE HOST IS NOT THE PACER (and how that was settled)
----------------------------------------------------
`MORI_EP_ROUND_SERIES=1` prints, per rank, five aligned series: `disp`, `conv`,
`comb` (GPU, from the events), `hdis`/`hcom` (host time inside the two calls) and
`hwal` (host wall per round).

The host cost inside the measured windows is real and large -- ~15us in dispatch
and ~25us in combine, i.e. roughly 40 of the ~85us "kernel" total is the Python
wrapper. But it is not what moves:

  * rounds where `hdis` or `hcom` spike to 50-60us show completely normal `disp`
    and `comb`. The GPU is behind, so it absorbs a host hiccup entirely.
  * rounds where `disp` or `comb` spike to 150-300us show completely normal
    `hdis`/`hcom`/`hwal`.
  * summed over a 60-round loop, `hwal` (~86us/round) is far below
    disp+conv+comb (~141us/round) and `wall` -- which is measured after the final
    `cuda.synchronize()` -- matches the GPU sum. The host runs ahead and the GPU
    is the bottleneck.

Over a long run (5000 rounds) the two converge exactly: wall 94.5us/round against
disp+conv+comb 94.3us/round.

READING THE PER-RANK SERIES
---------------------------
These are spin-wait collectives, so a stall on one rank appears as a long phase
on every OTHER rank; the culprit is the rank reporting the SHORTEST time. Both
nodes must be read together (`/tmp/v2n_rank0.log` and `/tmp/v2n_rank1.log`).

Two shapes seen, and they mean different things:

  * both nodes long in the same phase on the same round -- one ~120us delivery
    delay that everyone waited out.
  * node 0 long in combine while node 1 is long in dispatch on the same round --
    the same single event, seen from the two sides of a phase-offset pipeline.
    Do not read this as two separate problems.

A step that appears on one node only (node 0's ranks step, node 1's stay flat for
the whole run) is also seen. In that case node 1 is not waiting at all: its
time lands outside the measured windows.

RIG RECIPE
----------
Ours, and the #625 baseline for comparison:

    EXTRA_ENV="-e MORI_EP_ROUND_SERIES=1" bash ~/harness/_run_v2native.sh PORT \\
      --cmd bench --kernel-type v1_ll --num-qp 1 --max-tokens 4 --rounds 60 \\
      --dtype fp8_e4m3_fnuz --combine-dtype bf16 --hidden-dim 6144 --topk 8 \\
      --experts-per-rank 16 --scale-dim 32

    bash ~/harness/_run_cust.sh PORT --cmd bench --kernel-type v1_ll --num-qp 1 \\
      --max-tokens 4 --dtype fp8_e4m3_fnuz --combine-dtype bf16 \\
      --hidden-dim 6144 --topk 8

Both need `--nproc_per_node=1` (the runners set it): each spawns its own 8
workers per node. `MORI_RDMA_TC=160 MORI_RDMA_SL=5` are set by the runners.

METHOD
------
  * Always interleave A/B. Because of the regime, the same configuration back to
    back has read 85us and 134us total. A non-interleaved sweep will hand you a
    winner that is really just a run that did not step.
  * Judge a candidate by a PAIRED comparison against a FIXED incumbent, never by
    a chain in which each winner becomes the next incumbent -- five repeats of
    the greedy shape returned five different winners. See `_tune`.
  * Prefer max/mean over the mean when comparing across sessions: the mean drifts
    with the regime, the ratio is more stable. But now that the regime is known,
    reporting which regime each run landed in is better than either.
  * Compile the kernel offline before going to the cluster. A TU instantiating
    all eight `MORI_EP_INTERNODE_CCO_ENTRY*` entries reports every template error
    at once; finding them one per cluster run costs an afternoon.

DEVICE TIMESTAMPS: NEITHER KERNEL GETS SLOWER
---------------------------------------------
`MORI_EP_DEV_TS=1` stamps `wall_clock64()` (fixed 100.008 MHz on this gfx942,
10 ns; NOT `clock64()`, which is the shader clock and moves with DVFS) at twelve
points and reports per-round spans per rank. Comparing slow rounds against fast
rounds WITHIN one run and one rank:

    span                        fast     slow    delta
    d_stag  copystaging kernel   2.6      2.6     +0.0
    d_kern  whole dispatch_ll   21.1     21.6     +0.5
      d_post  WQE + doorbell     5.6      5.4     -0.2
      d_spin  wait for peer     17.7     18.2     +0.5
      d_recv  unpack + XGMI      1.0      1.1     +0.1
      d_sync  grid barrier       1.0      1.0     +0.0
    residual (phase - kernels)  33.0     78.8    +45.9   <-- all of it
    disp    the phase           55.5    103.0    +47.5

Reproduced on every run that had slow rounds (+45.9 and +64.8 in two runs).
**Both kernels in the dispatch phase are flat to under 1 microsecond.** The
excess is entirely in the launch/scheduling path: host enqueue plus GPU-side gap
between the two kernels. Splitting that residual further, the mix varies by run
-- one run had the host flat (+1.0) with a +35 GPU-side gap, another had the
host itself +24 -- but the kernels are flat in both.

This REVERSES the per-pass split's attribution above. Under
`MORI_EP_SPLIT_PASSES` the excess appeared to be in `dispatch_ll`, but split
mode launches each pass separately and the spin passes absorb every other
launch's skew -- which is exactly the effect being measured here. Trust the
in-kernel timestamps in batched mode over the split.

Combine is different and simpler: `c_spin`, its cross-node barrier wait, takes
the whole regime shift (0.7 -> 28.5 against `comb` 48 -> 77) and its spikes
(+63, +110 on single rounds), while `c_post` is flat. Combine does no extra
work; it waits longer for peers. It is a victim of dispatch's skew, not a
source.

So the send/recv question is answered, and the answer is neither: it is not the
local GPU->NIC post (`d_post` flat) and not waiting for the peer's write
(`d_spin` flat). Nothing inside either kernel moves.

THE ANSWER: A STABLE INTER-NODE PHASE OFFSET
--------------------------------------------
The slow regime is not slower work. It is the two nodes running a fixed number
of microseconds out of phase, with the cross-node rendezvous converting that
offset into wait time, every round, forever.

Closed per-round GPU accounting (all spans from wall_clock64, and the four terms
sum to the CUDA-event round to within 1us -- 94.2 against 94.0, 107.2 against
106.5, so nothing is unaccounted):

    d_gpu    dispatch's two kernels            22-26us
    d2sync   dispatch end -> combinesync       25-35us  (holds the torch convert)
    cs_sync  combinesync kernel                   2.3us  flat always
    cs_bar   cross-device barrier              see below
    c_gpu    combine_ll + combineall           29-34us
    gap_cd   combine end -> next dispatch         4.2us  flat always

Per node, median over rounds, four runs of the same binary:

    run     total   n0 c_spin  n0 cs_bar   n1 c_spin  n1 cs_bar
    rep1    135.6      31.9        5.3         0.4       39.7
    rep2    104.0       0.4       11.9        11.9        6.2
    rep3     96.6       2.3        5.8         5.5        6.3
    rep4     90.8       4.2        6.4         3.8        6.7

`c_spin` is combine_ll's cross-node wait; `cs_bar` is the cross-device barrier
that opens the combine phase. Read the table as a phase offset:

- rep1: node 0 is ~35us BEHIND. It pays the offset inside `c_spin` (31.9us,
  and all eight of its ranks read 31-37us); node 1 pays it one rendezvous later
  at `cs_bar` (39.7us, all eight ranks). Two waits, opposite sides of the round,
  same offset.
- rep2 is the MIRROR IMAGE at smaller magnitude -- node 1 behind by ~12us, so
  node 1 waits in `c_spin` and node 0 waits in `cs_bar`. The sign flips.
- rep3 and rep4 are aligned: every wait is 2-7us.
- The total tracks the offset: 90.8 aligned, 104.0 at ~12us, 135.6 at ~35us.

Everything else is flat across all four: `d_post`, `c_post`, `cs_sync`,
`gap_cd`, `post_bar`, and both hosts (hwal 75us, hdis 16us, hcom 25-27us on BOTH
nodes in BOTH regimes). No kernel does more work in the slow regime. The host is
not the pacer and is not asymmetric.

Why it is bistable and why it never recovers: a rendezvous makes everyone wait
for the last arrival, which PRESERVES an offset rather than removing it. Aligned
and offset are both fixed points of the round loop. Which one a run falls into is
decided in the first handful of rounds and then locked -- exactly the step seen
in the per-round series.

That also explains the rest of the file: both #625 and this branch step on the
same runs because both run the same rendezvous structure, and nothing in the
fabric, the clocks, the allocator, PCIe or PFC has to move for the offset to
exist.

Two mechanisms, not one. The whole-run REGIME is the offset above. The isolated
single-round SPIKES are something else: they land in `d2sync`, the window between
the two mori calls that holds the torch convert and the launch path (+44.9,
+17.1, +51.1 on spiked rounds), with `hdis` moving in step on some runs and not
others. The convert kernel itself is flat at 9us.

THE LOOP HAS NO SLACK: HOST JITTER CONVERTS 1:1 INTO PEER WAIT
--------------------------------------------------------------
`MORI_EP_INJECT_HOST_US` busy-waits the host between dispatch and combine, on
every rank, by a known amount. Summing each node's two cross-node waits
(`c_spin` + `cs_bar`):

    injected    node0 wait   node1 wait   ratio
        0us         11.9         4.1        --
       20us         20.1        18.7       1.01
       60us         61.5        56.9       1.02
      120us        119.3       129.7       1.08

**Every microsecond of host delay inserted between the two calls comes back as a
microsecond of cross-node wait.** The round has no slack to absorb it. (An
earlier single run read ~1.6x; the four-point curve says 1:1 and supersedes it.)

That closes the causal chain and unifies the two mechanisms above: a host hiccup
in `d2sync` is a perturbation, the loop converts it one-for-one into peer wait,
and nothing damps it, so it becomes a standing offset that persists for the rest
of the run.

It also predicts the production case. In a real MoE the expert compute sits
exactly in that window, so this is not a benchmark artifact -- it is the same
window, only larger.

Periodic re-alignment does NOT fix it. `--realign-every 20` (a full
synchronize + gloo barrier every 20 rounds) still ended one of two runs in the
offset state (node0 c_spin 21.7, node1 cs_bar 32.0): the offset re-forms inside
20 rounds. Re-aligning treats the symptom and the loop walks straight back.

ROOT CAUSE FOUND: THE CCO PATH NEVER BOUND ITS THREAD
-----------------------------------------------------
The per-rank host loop is the whole story. In every run that landed in the slow
regime, a few ranks show an EXACT 2x on their host loop and the rest show
nothing:

    slow run, hwal per rank (us)
      node0  r0..r7    75 76 77 77 77 78 77 76
      node1  r8 r9 r10 75 77 75 | r11 r12 r13 r14 = 146 149 146 146 | r15 78

    fast run: all sixteen ranks 74-78, no rank doubled

`hdis` goes 16 -> 30 and `hcom` 27 -> 50 on exactly those ranks. The extra
~36us of host time per round IS the phase offset (measured 35-40us), and it
reaches the peer through the transfer function above: the affected ranks arrive
late, their node's INTRA-node barrier (`cs_bar`) makes the other seven wait, and
the peer node then waits at the cross-node rendezvous (`c_spin`).

The 2x is the giveaway. The box is 2 sockets x 96 cores with SMT (sibling of
physical core c is c+192), so two unbound processes can land on the two
hyperthreads of ONE physical core and each runs at about half speed.

**The fix already existed in the tree and was wired to the wrong path.**
`application::BindCallingThreadToGpuNumaOnce()`
(`include/mori/application/utils/cpu_affinity.hpp`) binds a thread to the CPUs
local to its GPU's NUMA node, from the same sysfs `local_cpulist` NCCL uses,
intersected with the existing cpuset. Its ONLY caller was `src/shmem/init.cpp:746`
-- "the single bind site for the shmem/EP path". A job that used CCO instead of
shmem ran completely unbound. `ccoCommCreateImpl` now calls it too.

This also means every EP v2 measurement against v1/shmem was unfair in v1's
favour: the v1 harness calls `shmem_torch_process_group_init`
(`examples/ops/dispatch_combine/test_dispatch_combine_internode.py:606`) and was
bound; ours was not. The "we degrade ~3x further than #625" result above was
measured across that difference and needs re-running before it means anything.

Measured, eight runs each, `MORI_IGNORE_CPU_AFFINITY=1` as the control:

                       unbound            bound
    median total       88.0us             84.8us
    worst total        126.0us            97.4us
    runs with a
      doubled rank     1 of 4 (and ~1 in  0 of 8
                       3 across the day)

Eight clean runs do not prove the residual rate is zero: the bind confines four
ranks to one socket's 192 CPUs, which makes an SMT collision much less likely
but not impossible. Strict per-rank disjoint pinning also gave 4 of 4 clean. If
the regime ever reappears, check the per-rank `hwal` series first -- the
correlation with a doubled rank has been perfect in every run so far.

WHAT WOULD ACTUALLY HELP
------------------------
Not geometry tuning: no kernel is slower, so no schedule can win the time back.
Not periodic re-alignment: measured, it does not hold. Three things follow from
the 1:1 transfer, in the order I would try them:

1. **Cut the host wrapper on the critical path.** It is ~42us a round (hdis 16 +
   hcom 26) inside the measured window, and by the transfer above every
   microsecond removed is a microsecond of peer wait removed. This is the
   highest-confidence lever because the transfer function is measured, not
   assumed, and it needs no semantic change. It also plausibly explains why this
   branch degrades ~3x further than #625 on the same runs -- worth checking
   whether v1's host path per round is lighter or steadier than ours.

2. **Give the round slack: decouple consecutive rounds.** The benchmark reuses
   one set of buffers every round, so round i+1 cannot start until the peer has
   drained round i -- a false dependency. Double-buffering the arena by round
   parity lets a node that is ahead keep going instead of converting its lead
   into a wait. This is the structural fix; it changes the fixed point rather
   than the symptom.

3. **Reduce the number of serialised cross-node rendezvous.** There are at least
   three a round (`d_spin`, `cs_bar`, `c_spin`), and each is another place the
   standing offset is re-converted into wait. `combine_ll` already waits on peer
   data, so whether the separate `combinesyncbarrier` in front of it is
   redundant is the first thing to read. Expect this alone to move the offset
   rather than remove it -- it is worth doing together with (2), not instead.

NEXT
----
Everything outside the GPU has been eliminated, and the per-pass split puts the
excess inside `dispatch_ll` and inside `combinesyncbarrier` -- both of which mix
posting with waiting, so the split cannot go further from the host side. The next
step is to decompose the residual: the phase window holds only two kernels, both
now proven flat, so what remains is the enqueue path and the GPU-side gaps
around them. Stamping the host at each individual pass launch (rather than
around the whole `op.dispatch()` call) would split "the host was late" from "the
command processor was late", which is the last division available from outside
the kernels.
