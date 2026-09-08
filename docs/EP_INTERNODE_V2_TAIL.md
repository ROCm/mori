# EP internode v2: where the CCO tail comes from

Notes from investigating why the v2 CCO/GDA internode path has a latency tail
that the shmem/IBGDA path does not. Kept out of the test's module docstring so
that file stays close in shape to the examples harness it mirrors,
`examples/ops/dispatch_combine/test_dispatch_combine_internode.py`.

CCO's TAIL IS NOT THE FABRIC'S: WHAT SHMEM DOES ON THE SAME WIRE
----------------------------------------------------------------
Run the v1 harness in the same session for a reference. It drives the shmem/IBGDA
transport over the SAME NICs, rails, switch and QoS, at the same shape, so it
isolates what is specific to the CCO path. Note it spawns its own 8 workers per
node, so it takes --nproc_per_node=1, not 8:

    GPU_PER_NODE=8 MORI_EP_LAUNCH_CONFIG_MODE=AUTO MORI_RDMA_TC=160 MORI_RDMA_SL=5 \\
    torchrun --nnodes=2 --node_rank=N --nproc_per_node=1 --master_addr=... \\
      examples/ops/dispatch_combine/test_dispatch_combine_internode.py \\
      --cmd bench --kernel-type v1_ll --num-qp 1 --max-tokens 4 \\
      --hidden-dim 6144 --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none

Measured, 29 kept rounds x 16 ranks = 464 samples per phase:

                       shmem/IBGDA          CCO/GDA (here)
    dispatch mean         47.4us              40-47us
    combine  mean         59.1us              46-52us
    max/mean              1.19x / 1.18x       2-6x
    rounds >1.8x base     0 of 29             11 of ~260
    cross-rank spread     median 9-13us       up to 207us (34 -> 241)

Two things follow. CCO is FASTER in the mean -- combine by ~20% -- so the port is
not simply worse. And the tail belongs to the CCO path, not to the fabric: shmem
shares every wire and shows no spiked round at all, which rules out the rail and
switch explanations that the {i, i+8} pattern below otherwise suggests.

Both harnesses do have a huge round 0 (shmem's spans 83-614us) and both drop 1.

READING THE PER-RANK SERIES (MORI_EP_ROUND_SERIES)
--------------------------------------------------
Every rank prints its own per-round series. These are spin-wait collectives, so
the rank that arrives LAST waits LEAST: on a slow round the straggler is the
MINIMUM, not the maximum, and the other fifteen are just showing what they waited
for. Threshold per NODE, not globally -- the two nodes routinely sit at different
levels in the same round (151 vs 237us was measured), and a global threshold then
misses a rank that is low within its own node.

The shape of the low set says what happened, and they are not all the same:

  {i, i+8}          one RAIL. Local rank i selects bnxt_re_bond<i> ("rank 1
                    rankInNode 1 select device [1] bnxt_re_bond1"), so the same
                    local index on both nodes is one NIC-to-NIC path.
  {i}               one rank.
  all of one node    a node-level event; the other node's eight ranks all wait.
  empty             every rank waited, including the fastest -- no straggler
                    exists and no single card can explain it.

Measured over 11 spiked rounds in 9 runs: 5 empty, 3 rail pairs (rails 1, 5 and 7,
once each), 2 single ranks, 1 other. So no one card is at fault; when there is a
straggler its identity rotates.

## Eliminated by direct measurement

Each of these was tested and came back negative, rather than argued away:
garbage collection (traced per round -- zero collections during the timed loop),
GPU clocks (identical 1650-1780MHz in fast and slow runs), host launch-queue
saturation (~50us of headroom, and the slow regime has more), RoCE traffic class
(interleaved A/B over four pairs, within noise), host load average (six runs
across loadavg 11.5-15.3, no ordering), a single bad GPU, a single bad rail, QP
count, window cacheability, and process topology (`--spawn` reproduces the
examples harness's process tree; interleaved A/B is equal in the mean and keeps
the tail under both).

## Where the time actually goes

`MORI_EP_SPLIT_PASSES=1` launches the pass sequence one plan at a time with a
timing event after each -- the only way to attribute a slow round to a kernel,
since the launch group crosses the ABI once and runs the passes back to back.
Worst round against the median round of the same phase, three runs at 4 tokens:

    copystaging          median   7.0   worst   7.1   delta  +0.1
    dispatch_ll          median  49.2   worst 109.3   delta +60.1
    combinesync          median   6.4   worst   6.5   delta  +0.1
    combinesyncbarrier   median  24.3   worst  34.9   delta +10.6
    combine_ll           median  47.4   worst  69.5   delta +22.1
    combineall           median   7.1   worst   6.5   delta  -0.6

Every microsecond of the excess is in the passes that wait on remote data; the
three purely local passes are constant to within 0.2us.

It is also not the fabric erroring. Summing the bnxt RoCE hardware counters over
all eight devices before and after each run -- packet_seq_err, out_of_sequence,
implied_nak_seq_err, local_ack_timeout_err, max_retry_exceeded,
rnr_nak_retry_err, roce_adp_retrans, roce_slow_restart, rx/tx_roce_discards,
out_of_buffer, np_ecn_marked_roce_packets, np_cnp_sent, rp_cnp_handled,
duplicate_request -- every delta was zero across four runs including the slowest
(139.1us total, worst 155/163).

Not root-caused.
