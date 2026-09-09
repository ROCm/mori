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

NEXT
----
Everything outside the GPU has been eliminated, and the per-pass split puts the
excess inside `dispatch_ll` and inside `combinesyncbarrier` -- both of which mix
posting with waiting, so the split cannot go further from the host side. The next
step is device-side timestamps inside `dispatch_ll` separating the send-post from
the receive spin, which would say whether the extra time is spent getting the
data out or waiting for a peer's. The step attribution above says the target is
right: whatever it is, it is on the remote-access path and not in local work.
