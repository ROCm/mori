---
name: cluster-network-topology
description: >-
  Discover, diagram, and diagnose the RDMA/GPU network topology of a GPU cluster
  (rails, NICs, GID/RoCE config, GPU<->NIC PCIe affinity), determine whether the fabric
  supports cross-rail communication, and localize a failure to a fabric tier
  (NIC / leaf / spine) using only unprivileged host-side probes. Picks up where
  `mori check` stops: `mori check` answers "is this host configured correctly?" and
  drives its peer over SSH, while this answers "what is the fabric, and which tier
  broke?" from inside a scheduler allocation where SSH to compute nodes is unavailable.
  Handles IPv4-mapped AND IPv6-ULA RoCE addressing and classifies at BOTH the IP and
  RDMA layers — a fabric can be IP-routable cross-rail yet rail-only for RDMA. Includes
  the addressing-plan decode that recovers rail/leaf/pod grouping without switch access,
  spine enumeration that settles shared-vs-partitioned upper tiers from an unprivileged
  shell, a confound register for the measurement tools themselves, and an sbatch harness
  for multi-node tests under Slurm/Spur. Use when asked to "map the cluster", "draw how
  NICs are connected", "cross-rail diagram", "can we enable cross-rail RDMA", "get the
  switch topology", or to diagnose RDMA/RoCE QP-setup failures (e.g. rc=110 ETIMEDOUT on
  INIT->RTR), cross-rail unreachability, rail affinity, or EP / KV-transfer connectivity
  problems on AMD (ionic/bnxt/mlx5) or NVIDIA (mlx5) fabrics.
---

# Cluster network topology

Companion to `mori check` (`tools/env_check.sh`) and the `deploy-mori` skill: those
answer "is this host configured correctly?", this answers "what is the fabric, and which
tier broke?". The scripts referenced throughout ship beside this file —
`probe_topology.sh`, `xrail_matrix.sh`, `xrail_matrix.sbatch`, `make_report.py`.

A "rail-optimized" GPU cluster gives each GPU its own NIC, and each NIC lives on its
own isolated L2/L3 domain (a "rail"). Rails may or may not be routable to each other:

- **Full-mesh fabric**: any NIC can reach any NIC (cross-rail works) at both IP and RDMA.
- **Rail-only fabric**: a NIC can only reach the *same* rail on other nodes; cross-rail
  is unroutable. Symptom: same-rail RDMA works, cross-rail QP `INIT->RTR` fails
  (`ibv_modify_qp` returns `ETIMEDOUT`/110, or `ping` shows 100% loss cross-rail).
- **IP-routable but RDMA rail-only** (seen in the wild): cross-rail *ICMP/IP* works
  (the rails are routed), but cross-rail *RoCEv2 RDMA* still fails. So **do not classify
  from ping alone** — confirm at the RDMA layer (Step 2).

The goal of this skill is to (1) discover the layout, (2) classify the fabric at both
layers, (3) **localize** a failure to a tier and say what could change it, and
(4) produce a diagram + report + guidance for RDMA workloads.

---

## Measurement discipline (read before running anything)

Three rules, each learned by getting it wrong. They cost more time to re-do than to
follow.

**1. Every negative needs a positive control from the same tool, in the same job, at
the same instant.** A cross-rail `FAIL` means nothing on its own — the peer might be
down, the server might not have bound, the GID index might be wrong. It means something
next to a same-rail `OK` from the same binary, same allocation, same minute. Sweeps
(service level, DSCP/traffic class, MTU) are the usual offenders: a sweep where *every*
value fails and no control was run is indistinguishable from a broken harness and
carries **zero** evidential weight. Build the control into the harness, not into a
follow-up run.

**2. Never let one tool be the sole source of a verdict.** Tools embed assumptions in
their address handles, and a fabric-shaped conclusion drawn from one binary is really a
conclusion about that binary. See the confound register in Step 2c — the canonical case
is `ibv_rc_pingpong`, which hardcodes `hop_limit = 1` and never uses RDMA CM. Before
reporting "the fabric cannot do X", reproduce X's failure with **at least two tools that
build their path differently** (one hand-built AH, one RDMA CM).

**3. Capture the cheap artifacts every run, whether or not you think you need them.**
`ip -6 route show table all`, `ip -6 neigh`, `ip -o addr`, `/sys/class/net/*/statistics`,
and the tool inventory cost milliseconds and are the first things you will wish you had
when the verdict is questioned weeks later. **`ip route` is the single most often
forgotten and most load-bearing artifact** — it decides whether a rail plane is on-link
(pure L2) or routed, which in turn decides whether the `hop_limit` confound is fatal or
harmless. If you get one allocation, get the routes.

Corollary: **record the confounds you did not close** in the report itself. A verdict
with a named, dated open question is usable; a verdict that quietly rests on an
unexamined assumption is a liability.

---

## Step 1 — Discover the layout

Prefer the bundled `probe_topology.sh` (auto-detects NICs, GIDs incl. IPv6-ULA rails,
GPUs, same-PCI-domain affinity; emits `topo.mmd` + `topo.dot`). Run it inside the
container/host that owns the devices:

```bash
./probe_topology.sh                 # local node
./probe_topology.sh --peer <host>   # + cross-rail reachability hints
GID_INDEX=1 ./probe_topology.sh     # force a RoCE GID index (else auto)
```

If doing it by hand, collect these facts per node:

| Fact | Where |
|---|---|
| RDMA devices | `ls /sys/class/infiniband` (or `ibv_devices`) |
| Port state / link layer / rate | `/sys/class/infiniband/<dev>/ports/1/{state,rate}`, `ibv_devinfo` |
| **RoCE GID + type + addr** | `/sys/class/infiniband/<dev>/ports/1/gids/<i>` and `.../gid_attrs/types/<i>`. Pick the **RoCEv2 global** entry — this may be IPv4-mapped (`::ffff:AABBCCDD`, tail = rail IP) **or a global IPv6, commonly a ULA `fc00::/7`** (each rail its own `/64`). **Skip `fe80::` link-local.** Note the GID **index** (RDMA apps need it: `NCCL_IB_GID_INDEX`, `MORI_IB_GID_INDEX`). |
| netdev per NIC | `.../gid_attrs/ndevs/<i>` (e.g. `enP2p0s9`) |
| Rail IP (v4 or v6) | IPv4 from the `::ffff:` tail; **for IPv6 rails** read `ip -o -6 addr show <ndev> scope global`. If a rail has **no IPv4 and only a `fe80::` GID**, look again — its routable address is usually an IPv6 **ULA at a higher GID index** (this is easy to miss). |
| **MTU per netdev** | `cat /sys/class/net/<ndev>/mtu`. Rails are usually jumbo (9000); a mgmt NIC usually is not. A 1500-MTU path still *works*, so this is easy to miss while it quietly taxes throughput. |
| **L3 shape** | `ip -6 route show table all`, `ip -4 route show table all`, `ip route show default`, `ip -6 neigh show`. See Step 3a — this is what tells you whether a rail is on-link or routed. |
| NIC PCI bus + NUMA | `readlink -f /sys/class/infiniband/<dev>/device`; `.../device/numa_node` |
| **Firmware + driver** | `cat /sys/class/infiniband/<dev>/fw_ver`; `modinfo <mod>` vs `/sys/module/<mod>/version`. Compare as a **tuple**, not lexically — a larger trailing build number can hide a smaller patch number (`1.117.1-a-63` is *older* than `1.117.5-a-58`). |
| GPU PCI bus + model | `rocm-smi --showbus` / `rocm-smi --showproductname` (AMD, verify the actual model — don't infer from PCI DID) or `nvidia-smi --query-gpu=index,pci.bus_id,name` (NVIDIA) |

**GPU↔NIC affinity.** PCI addresses are `domain:bus:dev.func` (e.g. `0002:00:01.0`).
First **isolate the rail NICs** — drop the management/front-end NIC(s): the one on the
default route (`ip route show default`) and any RDMA device with no global address. This
matters on **single-PCI-domain** boxes where mgmt NICs are interspersed among the rails
(e.g. mlx5 `eth0`/`eth1` sitting among `rdma0..7`); if you don't drop them the pairing
shifts. Then **pair within the same PCI domain by ordinal**: sort the rail NICs and the
GPUs in each domain by full PCI address and zip them (k-th GPU ↔ k-th NIC). A naive
"nearest bus number" instead collapses on multi-domain boxes (the bus field is `00` for
everything). On a rail-optimized box the two halves land on the two NUMA nodes / PCI
domains (first-half GPUs with first-half NICs, second half with second half); on
single-domain boxes take the ordinal over the rails.

Note whether the **mgmt NIC is cross-socket** for half the GPUs — it usually is, and it
matters if you later fall back to it (Step 5).

**Script portability.** The probe must run where the devices live. Two gotchas the
bundled script already handles, but watch for if you hand-roll it:
- **No `/dev/fd`** in some container / scheduler-step namespaces → bash **process
  substitution** (`while read … < <(cmd)`) fails with `/dev/fd/63: No such file`. Read
  from a temp file instead.
- Parse the **full** PCI address (domain included), per the affinity note above.

---

## Step 2 — Classify the fabric (both layers)

Pick two nodes A and B. Test **per rail**, **both layers**, and **as a full N×N matrix**
rather than a couple of spot checks — the shape of the matrix is the finding.

### 2a — The N×N matrix

For an N-rail node, run all N² source-rail × destination-rail combinations at both
layers. What you are looking for is the *pattern*:

| Matrix shape | Meaning |
|---|---|
| All N² pass | Full mesh. |
| **Perfect diagonal** (N pass, N²−N fail) | Rail-partitioned *for that tool*. The clean diagonal is itself the positive control — it proves the harness, the GID index, the ports and the peer are all fine. |
| Ragged / asymmetric | Not a fabric property. Suspect the harness, a wedged server, a per-device config difference, or a sick NIC. Re-run before interpreting. |
| Diagonal fails too | Wrong GID index, wrong port, peer down. Fix this before reading anything else. |

**IP layer (quick proxy).** Bind to the rail's netdev. `ping -I <ifname>` sets
`SO_BINDTODEVICE`, which genuinely forces egress out that device — so a cross-rail ping
success is real evidence that L3 crosses planes. **Caveat: `-I` only forces the
*outbound* path.** The return path is chosen by the far node and is not observed; a
success therefore proves "there is a route out and *some* route back", not "the reverse
traversed the same rail".

```bash
# rail-aligned: same subnet on both nodes  -> expect 0% loss
ping    -c2 -I <ethX_railN> <B_ip4_railN>
ping -6 -c2 -I <ethX_railN> <B_ip6_railN>
# cross-rail: source railN device to B's railM addr (M != N) -> loss if rail-only
ping -6 -c2 -I <ethX_railN> <B_ip6_railM>
```

Confirm the packets really left the rail NIC rather than leaking to the default route:

```bash
cat /sys/class/net/<ndev>/statistics/tx_packets     # before
ping -6 -c 20 -I <ndev> <peer cross-rail addr>
cat /sys/class/net/<ndev>/statistics/tx_packets     # after -> must rise by ~20
```

**RDMA layer (authoritative).** ICMP passing cross-rail does **not** prove RDMA works.
Server on B, client on A, using the chosen RoCEv2 GID index (`-g` / `-x`):

```bash
# same-rail:  server & client both on rail 0
B$ ibv_rc_pingpong -d <dev0> -g <gid> -p 18500          # server
A$ ibv_rc_pingpong -d <dev0> -g <gid> -p 18500 <B_host> # client  -> expect success
# cross-rail: client on rail 0, server on rail 1
B$ ibv_rc_pingpong -d <dev1> -g <gid> -p 18501          # server
A$ ibv_rc_pingpong -d <dev0> -g <gid> -p 18501 <B_host> # client  -> hangs/times out if rail-only
```

The out-of-band handshake goes over the management network by hostname; the RDMA path
follows `-d`/`-g`. Note `ibv_rc_pingpong` **exits after one connection**, so the server
side must re-listen in a loop for a matrix run; the destination rail is selected by which
server port you connect to, the source rail by the client's `-d`.

Classification:
- Same-rail RDMA OK **and** cross-rail RDMA fails ⇒ **rail-only for RDMA** (the important
  case) — regardless of whether ICMP crossed. But see 2c before writing it down.
- Cross-rail RDMA also OK ⇒ full-mesh.
- To see the real errno when a library asserts, enable the provider's debug (e.g. AMD
  ionic: `IONIC_DEBUG=1 IONIC_DEBUG_FILE=/tmp/ionic_dbg`; look for
  `modify qp ... state 1 -> 2 rc <errno>`, where `state 1->2` = INIT->RTR).

### 2b — The paired IP-vs-RDMA control

Run **both layers against the same node pair, in the same job, back to back**. This is
the single highest-value experiment in the whole skill, because the interesting fabrics
are the ones where the two layers disagree, and a disagreement measured hours apart on
different node pairs is not a disagreement — it is two unrelated observations.

Emit one row per (src rail, dst rail) with both verdicts side by side:

```
rail0 -> rail0 :  RDMA OK    IP OK
rail0 -> rail1 :  RDMA FAIL  IP OK      <- this row is the finding
```

### 2c — Confound register: the tool is part of the experiment

Before a matrix result becomes a fabric verdict, check what the tool did to the path:

| Tool | Path construction | Hop limit | Confound |
|---|---|---|---|
| `ibv_rc_pingpong` | hand-built AH, TCP out-of-band | source says **`grh.hop_limit = 1`** | 1 would permit **zero** router hops, which alone could reproduce a perfect diagonal with no fabric involvement. But the kernel overwrites it from the route, so in practice it is usually *not* in force — measured: pingpong succeeds across a 3-router path. Settle it by hop count (below), not by reading the source. |
| `ib_write_bw` / `ib_write_lat` (perftest) | hand-built AH, TCP out-of-band | sets a high hop limit | Not subject to the above, but still never consults RDMA CM. Good second opinion. |
| `ib_write_bw -R`, `rping`, `ucmatose` | **RDMA CM** (`rdma_resolve_addr`/`rdma_resolve_route`) | from the kernel route | Uses the same resolution path a real library uses. If cross-rail works here and not with a hand-built AH, the fabric is fine and your address handle was wrong. |
| `ping -I` | kernel route + `SO_BINDTODEVICE` | kernel default (64) | Egress forced, **return path unobserved**. |

**Closing the hop-limit confound cheaply.** You do not need a hop-limit sweep, and a
sweep is hard to run correctly anyway (a failed run leaves the peer's server mid-timeout,
so the next cell fails for harness reasons and the column fills with noise). Instead
**count the router hops on the path the tool already succeeded on**. If the *same-rail*
run passes and `mtr`/`traceroute` shows that same-rail crosses one or more routers, then
a hop limit of 1 was demonstrably never in force — whatever the source says — and it
cannot explain the cross-rail failures. One `mtr` run retires the whole question.

**Minimum bar for a "cross-rail is dead" verdict:** the failure reproduces under at least
two tools that build the path differently, each with a same-rail positive control in the
same job, and `ip route` has been captured so the on-link-vs-routed question is settled.
Prefer a second hand-built-AH tool over an RDMA CM tool if the CM tools cannot be made to
source-bind — CM will not pick a cross-rail path on its own anyway (see 3c).

```bash
# RDMA CM re-test — cheap, one 2-node job, and it is the test most often skipped
B$ rping -s -a <B rail1 addr>
A$ rping -c -a <B rail1 addr> -C 10
A$ ib_write_bw -R <B rail1 addr>        # and a same-rail run as the control
```

### 2d — Running multi-node tests under a scheduler (Slurm / Spur)

You need coordinated processes on **two different** nodes. Pitfalls learned the hard way:
- **Interactive `srun --overlap` from the login node lands on the first node only**
  (even with `-w` / `--nodelist` / `-N2 -n2`), and **direct SSH to compute nodes is often
  blocked** (publickey-only, key not installed). So drive it from a **batch job**, whose
  steps *do* spread across the allocation. This is also why `mori check`, which drives
  its mesh over SSH, cannot be used directly on such clusters.
- **Spur-like schedulers:** `srun` may be **non-blocking** (returns on dispatch, so the
  batch script must wait on its own sentinel files or it falls off the end and kills the
  workers); `srun` may **not propagate the submitter's environment** (pass knobs via
  files in a shared run dir); `scontrol show hostnames` may be unsupported and
  `--ntasks-per-node` rejected. Don't expand the nodelist — launch `srun -N2 -n2
  --overlap` and have the workers **self-organize by `$(hostname)`** (sort the names;
  lower = tester A, higher = target B). The batch script may run from a **spool copy**,
  so use `$SLURM_SUBMIT_DIR`, not `$BASH_SOURCE`.
- **Files on shared storage can read back empty** on the peer node. Never silently
  default on an empty read — it desynchronizes the two sides asymmetrically and produces
  a plausible-looking wrong result. Retry, then fail loudly.
- **Avoid write races:** only the tester writes results, guarded by an atomic
  `mkdir <lock>` (a scheduler may also spawn the same wrapper twice on one node); run the
  RDMA **server in a re-listen loop** and **retry the client** a few times (a first
  attempt can fire before the server is up). Retry only the "server not ready" race —
  a genuinely unreachable pair connects on TCP and then times out, and must **not** be
  retried away.
- **Scheduler flakiness:** jobs may land in `JobHoldMaxRequeue` (`scontrol release
  <jid>`) or hit transient `JobLaunchFailure`. If multi-node dispatch keeps failing, fall
  back to per-node jobs pinned with `-w <node>` rather than waiting it out. Check the
  **QoS priority** before blaming scarcity — a low-priority "burst"/scavenger QoS makes
  jobs sit `PENDING` with `Reason=None` for hours next to idle nodes, which looks exactly
  like a wedged scheduler.

Use the bundled **`xrail_matrix.sbatch`** + **`xrail_matrix.sh`** — these are
**cluster-agnostic**: the worker auto-detects RoCE rail devices, each device's global GID
index (IPv4-mapped or IPv6 ULA/GUA, skipping `fe80::`) and address, drops the management
NIC (the one on the default route), and picks `ping`/`ping6` automatically. Nothing
site-specific is baked in — pass partition/account/qos/gres on the command line:

```bash
sbatch -p <partition> -A <acct> --qos=<qos> --gres=gpu:8 xrail_matrix.sbatch
# if the job is held: scontrol release <jid>
```

All artifacts land in a per-run **output folder**: `<XRAIL_OUT>/job-<jid>/` (default
`XRAIL_OUT=<submit_dir>/xrail-output`) — containing `result.txt`, `run.log`, and the
`host.*`/`addrs.*` coordination files. Optional env overrides (export before `sbatch`):

| Env | Purpose | Default |
|---|---|---|
| `XRAIL_OUT` | base output folder | `<submit_dir>/xrail-output` |
| `WORKER` | path to `xrail_matrix.sh` | next to the sbatch script |
| `RAIL_DEV_REGEX` | only consider RDMA devices matching this ERE | `.*` |
| `EXCLUDE_DEV_REGEX` | drop RDMA devices matching this ERE | (none) |
| `INCLUDE_MGMT=1` | keep the default-route (mgmt) device as a rail | drop it |
| `GID_INDEX` | force one GID index for all devices | auto-detect |
| `SRC_RAILS` | source rail indices for the ping matrix | all rails |
| `PORT_BASE` | base TCP port for `ibv_rc_pingpong` | `18500` |

Rail index = position of the device in the `sort -V` order of `/sys/class/infiniband`
(consistent across homogeneous nodes).

**Per-job artifact checklist** — capture all of these every run, on both nodes:

```
addrs.<node>     idx dev netdev addr, all rails         (the addressing plan, Step 3a)
routes.<node>    ip -6/-4 route show table all; ip route show default; ip -6 neigh
env.<node>       ibv_devinfo, fw_ver, mtu, numa_node, PCI paths, tool inventory
stats.<node>     /sys/class/net/*/statistics before+after
matrix.txt       the N x N RDMA matrix, both directions of the pair recorded
paired.txt       the back-to-back IP-vs-RDMA rows for one pair (2b)
run.log          everything, timestamped, including the failures
```

---

## Step 3 — Localize the failure to a tier

"Cross-rail RDMA fails" is a symptom. The actionable question is *which tier decided
that*, and the host can get surprisingly far without any switch access.

### 3a — Decode the addressing plan

Rail addressing on these clusters is **structured, not random**, and the structure
usually encodes physical grouping. This is the only handle on switch topology obtainable
without root or vendor cooperation, and it is free — you already captured the addresses.

Method, in the abstract:

1. Collect all rail addresses from **every node you can reach**, tagged by rail index.
2. Split each address into its fields (IPv6 hextet groups, or IPv4 octets).
3. For each field position, ask: is it **constant per rail across nodes** (⇒ a rail
   code), **constant per node across rails** (⇒ a node id), or **constant across a
   *subset* of nodes** (⇒ a grouping: leaf, pod, or block)?
4. A field may be a **sum** of two of those — subtract the rail code out and see whether
   the remainder becomes node-invariant. That remainder is the grouping field.
5. **Validate before believing:** the derivation must hold for *all* rails of *all*
   nodes. Drop any node whose rails disagree rather than smoothing it over — an
   inconsistent node is evidence against the encoding, not noise.

A synthetic illustration of what step 4 looks like when it lands. Given ULAs of the form
`fdXX:A:B:C::/64`, one row per rail per node:

```
node   rail   A       B       C        B>>8   (B>>8) - (A>>8)
n1     0      0x0300  0x4300  0x0012   0x43   0x40
n1     1      0x0100  0x4100  0x0012   0x41   0x40
n1     2      0x0200  0x4200  0x0012   0x42   0x40     <- constant per node
n2     0      0x0300  0x4300  0x0059   0x43   0x40     <- and shared with n1
n3     0      0x0300  0x6300  0x0027   0x63   0x60     <- different group
```

Read off: `A` = rail code (a fixed permutation, not necessarily `rail << 8`), `C` = node
id, `B` = `(group + rail) << 8` — so subtracting the rail out of `B` yields the grouping
field. Here n1 and n2 share a group and n3 does not.

What you get: for each node, a `(rail, group, node_id)` triple. Then the payoff — check
whether any experiment you already ran **straddled two groups**. If a same-rail RDMA test
between two different groups passed, you have measured that a rail plane spans more than
one switch, i.e. **the upper tier already forwards RoCE**, and the block is
plane-to-plane rather than switch-to-switch. That single observation reframes the whole
problem, and it usually falls out of data you collected for another purpose.

Sanity check the grouping against any vendor documentation: if the doc says N leaves per
pod and you derive N distinct group values, the group is the pod. Do **not** assume a
group field is a small within-group index — check the value range (widely separated
values in one group rule that out).

### 3b — What the host can and cannot see

Expect to be asked "the cluster has spine switches, why can't we see them?" The answer
is structural, not a permissions problem:

| Layer | Why it can't see the fabric |
|---|---|
| **RoCEv2 itself** | Unlike InfiniBand, RoCE has **no subnet manager**, no LIDs, no fabric database. `ibnetdiscover`, `iblinkinfo`, `ibtracert`, `ibdiagnet` return nothing — there is nothing to query. Topology lives in the Ethernet control plane (BGP/ECMP on the switches); the host is not a participant. |
| **verbs** | `ibv_devinfo`, `ibv_query_port`, `rdma link` describe the **local port** only. The API's model of the network is "a destination GID"; there is no topology object. |
| **LLDP** | One hop. Even with root it names only the directly attached switch (the leaf). A host is structurally incapable of seeing the spine via LLDP. |
| **routing** | `ip route` hides the spine behind a single next-hop — but this is the one layer that *does* leak. See "Enumerating the upper tier" below. |
| **RDMA libraries** (MoRI, UCX, libfabric, RCCL/NCCL) | All stop at the host boundary for the same reason — the information is not published to anyone on the node, kernel included. `NCCL_TOPO_FILE` describes host PCI/NUMA, not the fabric. A library is not *declining* to use the spine; its connection dies at `ibv_modify_qp`/RTR because the fabric decided long before. |

The one unprivileged probe that *can* reveal a router hop is a **rail-local traceroute**,
and it is almost always the thing nobody ran. Prefer **`mtr -6 -n -c 3 -r <peer rail
addr>`**: it is more often installed than `traceroute` on minimal images, and — the part
that matters — it prints *every* next-hop it observed at a given TTL, not just the first.

**Enumerating the upper tier.** ECMP is usually described as hiding the spine. It does
hide it from `ip route`, which shows one next-hop. But each traceroute probe hashes to a
different uplink, so repeated runs *sample* the tier. Send traffic on rail `t` to the
peer's rail `t` address (no source binding needed — the route picks the matching rail),
collect the hop-2 addresses, repeat per rail, and compare the sets:

```bash
for t in 0 1 2 3 4 5 6 7; do
  printf 'r%s: ' "$t"
  for k in 1 2 3 4; do mtr -6 -n -c 3 -r "<peer rail $t addr>"; done \
    | grep -oE '<upper-tier prefix>[0-9a-f:]+' | sort -u | tr '\n' ' '
  echo
done
```

This answers, from an unprivileged shell, the question everyone assumes only the network
operator can answer:

- **Sets overlap** (a rail sees spines that other rails also see) ⇒ the upper tier is
  **shared**. The rails are not disjoint planes; cross-rail is a routing/forwarding
  **policy** decision, and asking the operator to change it is a legitimate request.
- **Sets are disjoint**, each rail seeing about `spines ÷ rails` addresses ⇒ the tier is
  **partitioned per rail**, the planes really are separate end to end, and no host-side
  change will ever bridge them.

Sanity-check the union against the vendor's stated spine count and against the address
range: a contiguous block roughly the size of the union is a good sign you sampled one
pool rather than several. Beware the converse error — a *small* sample can look disjoint
by chance, so run enough repetitions that each rail yields well more than
`spines ÷ rails` addresses before concluding "partitioned".

### 3c — The unprivileged probe set

Run all of it in one allocation. Total cost: seconds.

```bash
# --- L3 shape: is a plane on-link, or routed? ---
ip -6 route show table all | grep -E '<rail prefix>|<rail ifname pattern>'
ip -6 route get <peer rail0 addr> oif <rail0 netdev>    # same-rail
ip -6 route get <peer rail1 addr> oif <rail0 netdev>    # cross-rail: same answer?
ip -6 neigh show dev <rail0 netdev>

# --- hop count: leaf-only, or leaf-spine-leaf? ---
# traceroute is often absent on minimal images; check for mtr before assuming.
mtr -6 -n -c 3 -r <peer rail0 addr>                     # same-rail
mtr -6 -n -c 3 -r <peer rail1 addr>                     # cross-rail
traceroute -6 -n -i <rail0 netdev> -s <my rail0> <peer rail0 addr>   # if present

# --- did the cross-rail ping actually use the rail NIC? ---
cat /sys/class/net/<rail0 netdev>/statistics/tx_packets   # before
ping -6 -c 20 -I <rail0 netdev> <peer rail1 addr>
cat /sys/class/net/<rail0 netdev>/statistics/tx_packets   # after

# --- cross-rail RDMA through RDMA CM, not a hand-built address handle ---
rping -s -a <peer rail1 addr>                             # on the peer
rping -c -a <peer rail1 addr> -C 10                       # here
ib_write_bw -R <peer rail1 addr>                          # and a same-rail run as control
```

Reading it:

- **On-link plane** (a short prefix such as `fdXX:800::/32 dev <rail0>`, no `via`):
  same-rail is pure L2, cross-rail is the *first* case needing a router — and a tool with
  `hop_limit = 1` fails it for reasons that have nothing to do with the fabric.
- **Routed plane** (`via <gw>` even for same-rail): the same-rail success already proves a
  router hop works, so `hop_limit` cannot explain the cross-rail failure.
- **`route get` returns the same `oif`/`via` for cross-rail as same-rail:** routing is
  configured; the block is below L3.
- **`traceroute`/`mtr` cross-rail dies at the first hop where same-rail completes:** the
  block is at the leaf. Completes with more hops: the upper tier forwards it and the block
  is RoCE-specific (a policy/ACL/class decision, not reachability).

**Two traps that will silently invalidate the cross-rail cells:**

1. **Your "cross-rail" IP test is probably not cross-rail.** If each rail plane has a
   route of its own, then aiming an unbound tool at the peer's rail-1 address makes the
   kernel egress *rail 1* — a same-rail flow with a cross-rail-looking destination. Check
   with `ip -6 route get <peer rail1 addr> from <my rail0 addr>` and read the `dev` and
   `src` it returns. Only `-I`/`SO_BINDTODEVICE` (or an explicit source bind) forces the
   flow onto rail 0. Much of the folklore that "IP crosses rails but RoCE doesn't" is
   this artifact.
2. **Source binding often breaks the RDMA CM tools outright.** `rping -I <src>` and
   `ib_write_bw -R --bind_source_ip` may hang or fail *same-rail*. That is a tool
   limitation, not a fabric result — and with no working positive control those cells
   must be discarded, not reported as failures. Note also that `ib_write_bw -R` needs an
   explicit IPv6 flag (e.g. `--ipv6-addr`) or it fails address parsing before it ever
   touches the fabric, on *both* the client and the server.

Because RDMA CM derives its path from the routing table, it will essentially never
*choose* a cross-rail path. That makes the device-pinned test (`-d <rail0 device>`
against the peer's rail-1 GID) the one that actually matters — it is also what MoRI, UCX
and NCCL do.

### 3d — Questions only the operator can answer

Everything above narrows it; these decide it. Ask them explicitly:

1. Confirm the shared-vs-partitioned upper tier. You should already have a strong answer
   from the spine enumeration in 3b — bring it, rather than asking cold, and ask them to
   confirm or correct it. (If you skipped 3b, this is the question that decides whether
   cross-rail is policy or physics.)
2. Is there an ACL, VRF, or route-policy that permits ICMP but drops or fails to route
   RoCE (UDP/4791) between planes?
3. What is the **oversubscription ratio** — how many uplinks per leaf, at what rate? A
   plane-crossing flow that works but contends 8:1 at the spine is not a performance path
   even if enabled.
4. Is a lossless class (PFC/DSCP) configured on the plane-crossing path, or only within
   a plane?
5. What would enabling it cost, and is it supported configuration or a one-off?

Note that host-side PFC state is often **unreadable as a normal user** (`dcb pfc show`
may return "Operation not supported" if the driver has no DCB netlink support; vendor
tools require root), so question 4 genuinely cannot be answered from the node.

---

## Step 4 — Rule out the tunables (with controls)

If cross-rail fails, the reflexive hypothesis is a QoS misconfiguration — a missing
lossless class or an unmarked DSCP. Test it properly and it is usually wrong; test it
without controls and you will chase it for a week.

On a **cross-rail pair**, sweep each knob while keeping a same-rail run of the identical
command as the control:

| Knob | Flag | If all values fail (with a working control) |
|---|---|---|
| Service level | `ibv_rc_pingpong -l 0..7` | Not an SL/priority mapping problem. |
| Path MTU | `ibv_rc_pingpong -m 256..4096` | Not an MTU mismatch. |
| Traffic class / DSCP | `ib_write_bw --tclass=<tc>` | Only meaningful **with** a same-rail control at the same tclass. Without one the sweep proves nothing. |
| GID index | `-g` / `-x` over all global candidates | Rules out the commonest cause of RTR failure. |

If SL and MTU sweeps both fail flat against a working diagonal, the "missing lossless
class" hypothesis is dead and you should stop spending time on QoS environment variables
and vendor NIC-setup scripts for the cross-rail case. Say so explicitly in the report —
it is the hypothesis everyone reaches for, and killing it is a real result.

**But**: a sweep only exonerates the knob for the *tool that ran it*. If all sweeps used
one binary, Step 2c still applies to all of them at once.

---

## Step 5 — Fallback paths

When cross-rail is genuinely unavailable, there are two escape hatches. Both are worth
measuring before designing around the limitation.

### 5a — The management / default-route NIC as a full mesh

The mgmt NIC is usually on **one flat routed subnet with no rail partitioning**, and if
it is RoCE-capable it is a full node-to-node RDMA mesh. This can unblock an all-to-all
workload with **no code change** — just point the library's device selection at it.

Measure the price before recommending it:

```bash
ib_write_bw -d <mgmt dev> -x <mgmt gid> -s 65536 -D 5 <peer>   # and the same for a rail dev
```

Account for **all** of the taxes, because only the first is obvious:

- **Line rate** — mgmt is typically half the rail rate.
- **Aggregate width** — one mgmt NIC shared by all GPUs versus one rail NIC *per* GPU.
  With 8 GPUs this is often ~16x less aggregate bandwidth, which is the number that
  matters, not the per-link one.
- **MTU** — mgmt is often 1500 where rails are 9000, costing another chunk of achievable
  fraction of line rate.
- **Contention** — the mgmt network also carries NFS, the scheduler, and SSH.
- **NUMA** — the mgmt NIC is cross-socket for half the GPUs.
- **A different GID index** than the rails. Check it; do not reuse the rail's.

Cross-fabric (mgmt ↔ rail) will fail — they are separate fabrics. That is expected, not a
bug.

**How to use it:** to get a distributed workload *running* for functional or CI purposes.
Never quote its numbers as performance.

### 5b — Rail-affine peer NIC selection

This needs **nothing** from the fabric and is the durable answer to a rail-only fabric.
For each remote peer, pick the **local** NIC whose rail matches that peer's rail, so
every QP stays inside one plane. The peer's rail is readable from its GID prefix (Step
3a), so the mapping is derivable at connection setup with no new information from anyone.

The cost is structural, not incidental: a QP now belongs to a **rail** rather than to a
device, so anything the connection advertises per-device — memory registration keys in
particular — becomes **per-peer**. That is the change a library has to absorb, and it is
the honest version of "why doesn't the library just use the right NIC?".

This is in the library's control and is a real gap when it is missing. Contrast:
- **All-to-all / expert-parallel** (GPU-initiated RDMA) genuinely needs GPU*i*(rail *i*)
  ↔ GPU*j*(rail *j*), so on a rail-only fabric it hangs or asserts at QP setup unless the
  library does 5b.
- **Point-to-point / KV transfer** can usually be pinned to one rail per connection —
  look for an existing rail-affinity option before writing one.
- **Single-rail funneling is not a general workaround**: forcing every rank onto one NIC
  makes connectivity same-subnet but overloads and can deadlock all-to-all transports
  that assume one NIC per GPU. Prefer per-connection rail affinity over a global
  single-rail setting.

---

## Step 6 — Draw it

Two portable options (no AI image tools — topology must be exact):

- **Mermaid** (`topo.mmd`): paste into `https://mermaid.live`, GitHub, or any Markdown.
- **Graphviz** (`topo.dot`): `dot -Tpng topo.dot -o topo.png` (or `-Tsvg`).

Single-node diagram conventions:
- One subgraph per **NUMA / PCI domain**; GPU boxes linked to their rail-local NIC
  (label the PCIe bus). Rail-optimized boxes split first-half/second-half across the two.
- A legend documenting GPU model, RoCE GID index + address family (v4 / IPv6-ULA), and
  the fabric classification.

Two-node cross-rail diagram (example: `topology_crossrail_2node.dot`/`.png`, built from
`xrail_matrix` output; see also `topology_two_node.dot`):
- Left = Node A, right = Node B, a **middle column of rail leaves** (one per rail).
- **Green solid** = same-rail links that work (ICMP **and** RDMA OK).
- **Red dashed** = cross-rail links that fail at RDMA. If ICMP crosses but RDMA does not,
  say so explicitly in the legend (green-at-IP / red-at-RDMA) — it's the whole point.
- If Step 3a derived a grouping, draw the **upper tier** as a box above the leaves and
  mark which measured link crossed it. Draw it **dashed/greyed and labelled "inferred
  from addressing plan; not directly observed"** — never present a derived tier as a
  measured one.

---

## Step 7 — Interpret for RDMA workloads

- Map each workload to what it needs: all-to-all/EP needs full-mesh RDMA (or Step 5b);
  point-to-point/KV tolerates rail affinity.
- Always set the correct **RoCEv2 GID index** for the workload
  (`NCCL_IB_GID_INDEX`, `MORI_IB_GID_INDEX`, etc.) — the global entry (IPv4-mapped or
  IPv6-ULA), not a `fe80::` link-local; a wrong GID causes RTR failures independent of
  the fabric, and is the single most common false "the fabric is broken" report.
- **Don't hand-pin transport/device lists** in the middleware (e.g. UCX `UCX_TLS` /
  `UCX_NET_DEVICES`) as a first move — over-constraining commonly drops the verbs
  transports entirely and silently degrades to shared memory, producing "Destination is
  unreachable" errors that look like fabric problems.
- Check **device limits** before blaming the fabric for a large-transfer failure:
  `max_mr_size`, `max_mr`, `max_qp`, `max_cqe` from `ibv_devinfo -v`. A registration
  larger than `max_mr_size` fails with `ENOMEM` and looks exactly like a memlock limit.
- Check whether the provider can register **GPU/device memory** at all before designing a
  GPU-direct path on it. A provider that cannot take VRAM fails in several unrelated-
  looking ways (`ibv_reg_mr` → `EINVAL`; a parent domain whose allocator returns VRAM →
  `ibv_create_cq_ex` → `EFAULT`), and middleware that never calls the affected verb will
  not reproduce it — so "the benchmark works" does not clear the library.
- Re-read `fw_ver` before crediting or blaming firmware for any behavior change, and
  remember the tuple-compare trap from Step 1.

---

## Step 8 — Generate an HTML report (final deliverable)

Bundle everything into a single self-contained HTML page that a non-expert can read at a
glance: the fabric verdict, the diagrams, the summary, and the raw data. Use the bundled
**`make_report.py`** (no dependencies; embeds the PNGs as base64 so the file is shareable).

Collect your artifacts for the run into one **output folder** (the diagrams, the `.md`
summary, the cross-rail `result.txt`, and the `topo_report*.txt`), then:

```bash
# one report per output folder -> <folder>/report.html
python3 make_report.py <output_folder>

# optional: a comparison landing page across several clusters -> index.html
python3 make_report.py --index index.html <folder1> <folder2> <folder3>
```

The generator auto-discovers files by glob (`*node*.png`, `*crossrail*.png`,
`topology_*.md`, `*result*.txt`, `topo_report*.txt`), derives the **fabric verdict** from
the summary's `Fabric classification:` line (FULL-MESH / RDMA RAIL-ONLY / RAIL-ONLY
IP+RDMA), and renders the `.md` (headings, tables, code). Author the summary `.md` with a
top-level `#` title and a `## Fabric classification: <verdict> …` line so the report picks
them up.

It also parses `result.txt` into a **verdict evidence table** showing *both* experiments —
IP (`ping`) and RDMA — as same-rail vs cross-rail (green/red). This is important: a fabric
can pass **IP** cross-rail yet fail **RDMA** cross-rail, so always show the two side by
side rather than a single "reachable/not" line. The `--index` page carries the same two
cross-rail columns so clusters can be compared at a glance.

**Report rules, from experience:**
- **Parse the numbers out of the artifacts; never retype them into the HTML.** Every
  figure in the report should be derived at generation time from a file in the output
  folder, so regenerating cannot drift from the data.
- **Edit the generator, not the generated HTML.** Hand edits to the `.html` are lost the
  next time anyone runs the script.
- Give the report an explicit **open questions / confounds** section, dated, naming what
  was *not* closed and what one experiment would close it. This is what makes the
  document survive contact with a skeptical reader.
- Distinguish **measured** from **derived** everywhere, in the prose as well as the
  diagrams.

---

## Running `mori check` inside a Docker container

Do this **first**, before any of the steps above. `mori check` answers "is this host
configured correctly?" — wrong GID, bad firmware, missing lossless class, a dead local
NIC — and most reports of "the fabric is broken" are one of those. Only when it comes
back clean and the workload still fails is the fabric itself worth investigating.

Almost all GPU work happens in a container, and RDMA does not cross the container
boundary by default, so the flags below are not optional garnish — without them the
checks fail in ways that look like fabric problems.

### Start the container with RDMA visible

```bash
docker ps &>/dev/null || SUDO=sudo      # drop `sudo` if you are in the docker group

$SUDO docker run -d --name mori-check \
    --network=host \
    --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --device=/dev/infiniband \
    --ulimit memlock=-1:-1 \
    --ulimit nproc=100000:100000 --pids-limit=-1 \
    --privileged \
    -v /lib/modules:/lib/modules \
    -v /sys/kernel/config:/sys/kernel/config \
    -v /sys/kernel/debug:/sys/kernel/debug \
    <rocm image> sleep infinity
```

Omit any `-v` whose host path does not exist, and add `-v /home:/home` (or wherever the
MoRI source and your run directory live) if you need them inside.

Why each of the four load-bearing ones matters:

- **`--network=host`** — the container shares the host netns, so it sees the rail
  netdevs, their addresses and the host routing table. Without it every address in
  Step 1 is wrong and every peer test fails at the out-of-band handshake, not at RDMA.
- **`--device=/dev/infiniband`** — otherwise `/sys/class/infiniband` may be visible while
  the character devices are not, which produces "no RDMA devices" or an `ibv_open_device`
  failure rather than a clean error.
- **`--ulimit memlock=-1:-1`** — RDMA pins memory on QP creation. The default limit makes
  the parallel bandwidth mesh fail intermittently under load, which reads as flakiness.
  Note this is *also* the limit people wrongly blame when a registration exceeds the
  device's `max_mr_size` (Step 7) — check the device cap before raising the ulimit again.
- **configfs / debugfs mounts** — `--privileged` alone does **not** propagate them, and
  the bnxt DCQCN read/write path needs both. If absent on the host:
  `sudo mount -t configfs none /sys/kernel/config; sudo mount -t debugfs none /sys/kernel/debug`.

### Install what the checks shell out to

```bash
$SUDO docker exec mori-check bash -c "apt-get update && apt-get install -y --no-install-recommends \
    libibverbs-dev ibverbs-utils rdma-core perftest \
    iproute2 iputils-ping ethtool pciutils sudo jq"
```

Non-obvious roles: **`perftest`** provides `ib_write_bw`/`ib_write_lat` (the mesh steps
are skipped without it); **`pciutils`** is shelled out to by the ionic `nicctl` path and
its absence produces a misleading `Invalid card handle`; **`iproute2`** provides `dcb`,
used by the bnxt QoS path; **`sudo`** is invoked by the vendor paths even inside a
`--privileged` container. This same set covers the probes in Steps 2–3 — add
`traceroute` if you intend to run Step 3c.

> If `apt-get update` cannot reach `archive.ubuntu.com` (common on cloud hosts routed to
> a provider-internal mirror), copy the host's working sources in first:
> `$SUDO docker cp /etc/apt/sources.list mori-check:/etc/apt/sources.list`.

NIC userspace libraries are vendor-specific (`libionic` for AINIC, `libbnxt_re` for
Broadcom, inbox `libmlx5` for Mellanox) and **must match the host kernel driver version**
— read the host's version from `/sys/class/infiniband/<dev>/fw_ver` and
`modinfo -F version <module>` and install to match. A mismatched userspace provider fails
at QP setup and looks exactly like an unreachable fabric.

### Run it

```bash
$SUDO docker exec mori-check bash -c "mori check"              # local only
$SUDO docker exec mori-check bash -c "mori check <peer_ip>"    # + inter-node steps
```

Six steps: firmware/driver consistency; QoS (PFC + lossless TC, and it selects the SL/TC
MoRI will use); DCQCN; intra-node bandwidth mesh; inter-node bandwidth; inter-node
latency. The last two are **skipped without a peer IP** — supply one, and run it from
both nodes.

On `[FAIL]`: `mori setup` applies QoS/PFC/DCQCN for ionic/bnxt/mlx5 (use
`source $(mori setup --path)` if you also want `MORI_RDMA_SL`/`MORI_RDMA_TC` exported
into your shell), then `mori diagnose` if it still fails. On mlx5 a DCQCN fix written to
NV config needs a firmware reset or reboot before it takes effect.

### Reading its output through this skill's lens

Three of its results are fabric statements in disguise, and misreading them is what sends
people down the wrong path:

- **Unreachable cells in the intra-node mesh are usually expected.** The mesh probes
  every local device pair regardless of whether they are on the same physical network, so
  a management or otherwise incidental RDMA port shows `✗` against everything — including
  against the other incidental port. Confirm the `✗` cells are confined to those ports'
  rows and columns before treating it as a fault. For `n` local RDMA devices of which `f`
  are on the fabric, expect `n×(n−1) − f×(f−1)` unreachable cells.
- **A clean diagonal in the inter-node mesh is the rail-only signature**, not a pass. If
  same-rail pairs report bandwidth and every cross-rail pair fails, you have the Step 2a
  matrix for free — go to Step 2c before concluding anything, since `mori check` uses
  perftest and that is only *one* of the two tool families the confound register asks for.
- **A QoS `[FAIL]` on a cross-rail path is not necessarily actionable.** Step 4 exists
  because the lossless-class hypothesis is the one everyone reaches for and is frequently
  wrong; sweep SL and MTU with a same-rail control before spending time on `mori setup`
  or vendor QoS tooling for the cross-rail case.

### When `mori check` cannot run at all

Its inter-node steps drive the peer **over SSH**. On clusters where compute nodes accept
no direct SSH — common under Slurm/Spur, where the login node cannot reach a compute node
and the allocation is the only way in — those steps cannot execute at all. That is the
case Step 2d's batch harness exists for: same experiment, launched from inside an
allocation instead of over SSH. Run the local-only `mori check` on each node via the
batch job, and get the inter-node evidence from the matrix harness.

### Possible integration

A `mori topology` subcommand alongside `check`/`setup`/`diagnose` in
`python/mori/cli.py`, backed by the probe and matrix scripts, with this document as its
reference. Two conventions to match: bundled scripts live in `python/mori/tools/` and are
resolved via `_script_path()`, and every script must degrade gracefully to non-root
(which this skill's probe set already does).

---

## Quick checklist

1. **Discipline first**: positive control in every run, two tools before any verdict,
   capture `ip route` even when you think you don't need it.
2. **`mori check` first** — in a container with `--network=host`,
   `--device=/dev/infiniband` and `memlock=-1`. Most "broken fabric" reports are a wrong
   GID, a mismatched NIC userspace library, or a bad firmware version, and it finds all
   three. Read its meshes per the notes in that section before believing them.
3. `probe_topology.sh` on one node of each type → NICs, GIDs (v4/IPv6-ULA), MTU,
   firmware, GPU↔NIC affinity (same-domain ordinal, mgmt NIC excluded). Verify the GPU
   model with `rocm-smi --showproductname`.
4. Classify **both layers** between two nodes: full N×N ping matrix **and** N×N RDMA
   matrix, plus the paired back-to-back control. Under a scheduler use
   `xrail_matrix.sbatch` (Step 2d), which drives `xrail_matrix.sh` on both nodes.
5. Before believing a diagonal, run the **confound register** (Step 2c). The cheapest
   close on the `hop_limit = 1` confound is to count router hops on a *passing* path
   with `mtr` — if the pass already crossed a router, the hardcoded hop limit was never
   in force. Prefer that to sweeping `-L`, which desynchronises the peer's server.
6. Localize: decode the addressing plan for rail/group, run the unprivileged probe set
   (`ip -6 route show table all`, `route get oif`, `mtr -6 -n -c 3 -r -I <rail netdev>`,
   tx_packets delta), and check whether any passing test already straddled two groups.
7. Rule out the tunables **with controls** (SL, MTU, tclass, GID index) — then say so.
8. Price the fallbacks: mgmt-NIC mesh (measure it, and count the aggregate, not the
   link) and rail-affine peer NIC selection.
9. Enumerate the spines yourself first (Step 3b) — repeated per-rail `mtr` samples the
   upper tier and usually settles shared vs partitioned without switch access. Then take
   the remaining operator questions (Step 3d) to the vendor, bringing your answer rather
   than asking cold.
10. Render `topo.dot`/`topo.mmd` and a 2-node cross-rail diagram; mark derived tiers as
    derived.
11. `make_report.py <folder>` → self-contained `report.html` with an explicit open-
    questions section; `--index` for a multi-cluster comparison page. This is the final,
    user-facing deliverable.
