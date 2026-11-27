# MORI traffic dashboards: IO and EP in Grafana

Three Grafana dashboards for the traffic metrics a MORI process exports about itself
(`mori_io_*`, `mori_ep_*`, `mori_gpu_rdma_*`), and a Prometheus + Grafana stack that shows
them:

- **MORI traffic overview: IO and EP**: the two together, on the same links and the same time
  axis. In a serving model both run; this is the page that says whether their traffic takes
  turns or overlaps, and how close the two together come to what the links carry.

  ![MORI traffic overview: IO and EP over XGMI and over RDMA on one time axis](images/overview.png)

- **MORI EP traffic**: expert-parallel dispatch / combine. The kernel bandwidth of dispatch
  and combine (the number an EP benchmark reports), which GPU sends how much to which over
  XGMI, and what the GPUs put on each NIC over RDMA. Laid out like the IO dashboard.

  ![MORI EP traffic: tiles, kernel bandwidth of dispatch and combine, the GPU to GPU matrix, per-GPU and per-NIC charts](images/ep_traffic.png)

- **MORI IO traffic**: the IO engines. Which GPU sends to which and how fast, what the RDMA
  backend puts on each NIC, and one row of totals (throughput, failures, latency).

  ![MORI IO traffic: tiles, throughput and latency, the source to destination matrix, per-NIC charts](images/io_traffic.png)

The overview screenshot: 8 x MI300X run as two nodes of 4 GPUs, inter-node EP v1 back to back
the whole time, a minute of all-to-all IO over XGMI in the middle and IO over RDMA starting
at the end. The other two screenshots are from one run. EP: 8 ranks on the 8 GPUs of an MI308X node (`node-a`),
intra-node v1 kernel, 4096-wide bf16 tokens, a load that steps between three call rates. IO on
`node-a`: simulated transfers (`MORI_IO_SIM_ENABLE=1`). IO on `node-b`: real RDMA transfers
between host buffers through one NIC of an MI355X node.

## Start

Run the jobs with a metrics port, and a directory where each process announces itself:

```
export MORI_METRICS_PORT=9400                  # a process serves /metrics on 9400 + its local rank
export MORI_METRICS_PORT_DIR=/tmp/mori-metrics # and writes <host>-<pid>.json here for Prometheus
```

(The local rank is `LOCAL_RANK`, or what Open MPI / Slurm export. Without one, or when the port
is taken, a process takes the next free port; the file says which. The directory can be one
that all the nodes of a job share.)

All the switches, read once when the process creates its first IO engine or EP op
(`traffic.sh sampling` changes the sampling of running processes):

| Variable | Meaning |
| --- | --- |
| `MORI_METRICS_PORT` | Turns metrics on and serves `/metrics` on this port + the local rank |
| `MORI_METRICS_ENABLE=1` | Turns metrics on without a port: read them with `mori.metrics.snapshot()` and `op.get_traffic_stats()`. `0` turns them off even with a port set |
| `MORI_METRICS_PORT_DIR` | Directory for the Prometheus `file_sd` target file of each process |
| `MORI_METRICS_ADDR` | IPv4 address to listen on (default: every interface); also the address written to the target file |
| `MORI_METRICS_HOST_ID` | Value of the `host` label instead of the hostname; the same knob as `--host-id` of the netmon tools |
| `MORI_METRICS_INTERVAL_MS` | How often the counters are read into the series (default 1000; 1 with a push URL) |
| `MORI_METRICS_SAMPLE_EP`, `MORI_METRICS_SAMPLE_IO` | Count one EP dispatch + combine / one IO transfer in this many (default 1: all; 0: none). Both, and the interval above, can be changed while the process runs: see *Sampling* |
| `MORI_METRICS_PUSH_URL` | `http://host:port` of a VictoriaMetrics: the process also sends every sample it takes, with its time, for millisecond resolution (see below). Turns metrics on by itself |

Then, on the host that should show the dashboards:

```
./traffic.sh up                                                    # Prometheus :9090, Grafana :3000, data in ./data
TRAFFIC_PROM_PORT=19090 TRAFFIC_GRAFANA_PORT=13000 ./traffic.sh up # other ports
TRAFFIC_TARGETS_DIR=/shared/mori-metrics ./traffic.sh up           # the jobs' MORI_METRICS_PORT_DIR
TRAFFIC_TARGETS=node-a:9400,node-a:9401 ./traffic.sh up            # a fixed list instead, or besides
./traffic.sh down                                                  # stop; the data directory is kept
```

`traffic.sh up` starts three containers with `docker run`, `mori-traffic-<you>-prometheus`,
`mori-traffic-<you>-vm` (VictoriaMetrics, for the millisecond data below) and
`mori-traffic-<you>-grafana`, on the host network, as you, listening on 127.0.0.1 (set
`TRAFFIC_LISTEN` to an address of the host to open them to others; there is no login). Grafana
opens on the overview; the three dashboards are in the folder `MORI traffic` and link to each
other. From another machine:
`ssh -N -L 3000:127.0.0.1:3000 <host>`.

Check `http://127.0.0.1:9090/targets` first: every MORI process is one target, and a dashboard
shows only what is `UP` there.

## How the data gets to a panel

```
 one MORI process (a rank, or an IO engine)                    monitoring host
┌─────────────────────────────────────────────────────┐
│ IO: IOEngine entry / TransferStatus ─┐ host atomics │
│     RDMA backend PostSend ───────────┤              │
│ shmem / cco: RDMA issued by the GPU ─┤ counter      │
│ EP kernels: tokens per peer,         │ blocks       │
│     launches, kernel time ───────────┘ (device or   │
│              │                    host-mapped memory)│
│              ▼ every MORI_METRICS_INTERVAL_MS (1 s)  │
│ mori::metrics::Registry: reads the blocks, turns     │
│ tokens into bytes, attaches host / GPU / NIC labels  │
│              ▼                                       │        ┌────────────┐   ┌─────────┐
│ MetricsServer  :MORI_METRICS_PORT + local rank ──────┼─ 5 s ─▶│ Prometheus │──▶│ Grafana │
│ <MORI_METRICS_PORT_DIR>/<host>-<pid>.json ───────────┼ file_sd└────────────┘   └─────────┘
│ mori.metrics.snapshot()  (the same series, in Python)│
└─────────────────────────────────────────────────────┘
```

- **Sampling**: every transfer and every launch is counted unless told otherwise; see
  *Sampling* below for counting one in N, changed while the job runs.
- **Counting** happens where the traffic is: in the IO engine and its RDMA backend on the
  host, in shmem / cco for RDMA a GPU issues itself, and inside the EP kernels on the device.
  The EP kernels also time themselves with the device clock. Nothing on these paths formats a
  label or takes a lock; with metrics off the cost is one null-pointer test.
- **Publishing** is one thread per process (the registry). Once a second it reads the counter
  blocks and updates the series; the EP counters it reads are a host-mapped copy the kernel
  keeps, so reading never touches the GPU runtime (safe while a CUDA graph is being captured).
- **Serving**: each process has its own `/metrics`. There is no central MORI component:
  Prometheus scrapes every process and the dashboards add them up. A flow between two
  processes is recorded once, by the one that started it, with both ends in the labels, so a
  receive side is visible even when only the senders are scraped.
- **Delay** from a transfer to its point on a chart: up to 1 s (registry) + 5 s (scrape) + the
  dashboard refresh (5 s).

The metric names, labels and help texts are in
[`include/mori/metrics/traffic_counters.hpp`](../../../include/mori/metrics/traffic_counters.hpp)
(`namespace names`).

## Millisecond resolution

A scrape gets one value per series every few seconds. To see what happens within a second, a
process can send its samples itself, the way `ainic-netmon` and `xgmi-netmon` do:

```
export MORI_METRICS_PUSH_URL=http://127.0.0.1:8429   # the VictoriaMetrics traffic.sh up starts
```

The process then reads its counters every millisecond (`MORI_METRICS_INTERVAL_MS` to change
it), keeps a point for every sample that moved a counter or a gauge, and posts the points once
a second to `/api/v1/import`. The series, names and labels are the ones `/metrics` has, so both
dashboards work as they are: pick `mori-traffic-ms` in `Data source` and zoom the time range
to a few seconds.

- New points arrive once a second, so the last second or two are not there yet. Use
  `Average over` 5s or more for the tiles and the matrix, and end the time range a few seconds
  back (`now-10s` to `now-2s`).
- A point says that work was issued, not that bytes crossed the wire: the per-NIC counter jumps
  by a whole batch when its work requests are posted. Over a few milliseconds that reads as
  spikes; a window has to span several transfers to read as the rate of the link. A NIC's own
  counter (`ainic-netmon`) follows the wire.
- The EP intra-node counters reach the host once per kernel launch, so their resolution is a
  launch.
- Cost, measured: the sampling thread takes about 7 to 11 % of one core per process at 1 ms
  (144 to 242 series), sending under 1 %; about 0.5 to 1 MB/s of points per process. A database
  that is away or slow costs points (counted, and logged by the process), never the data path.
  Put VictoriaMetrics where the processes can reach it at that rate: a push through an ssh
  tunnel across sites did not keep up.
- `TRAFFIC_VM_LISTEN=<address>` makes VictoriaMetrics listen on an internal address for jobs
  on other nodes. `TRAFFIC_VM_URL=http://127.0.0.1:8428` uses the netmon stack's
  VictoriaMetrics instead of starting one, which puts MORI's counts in the same database as
  the NIC and XGMI hardware counters.

## Sampling: count less, while the job runs

Counting everything is the default, and what the numbers are checked against (a NIC's own
counters, a benchmark's time). Where that costs too much, the counting can be thinned and
turned back up **while the processes run**, without restarting anything:

```
./traffic.sh sampling                    # what every process Prometheus scrapes is set to
./traffic.sh sampling ep=8 io=64         # count one EP dispatch + combine in 8, one IO transfer in 64
./traffic.sh sampling ep=1 io=1          # everything again
./traffic.sh sampling ep=0               # stop counting EP
./traffic.sh sampling interval_ms=10     # the metrics thread reads the counters every 10 ms
TRAFFIC_PROM_PORT=19090 ./traffic.sh sampling ep=8                   # the Prometheus on another port
TRAFFIC_TARGETS=node-a:9400,node-a:9401 ./traffic.sh sampling ep=8   # these processes
```

**From the dashboard**: the first row of the overview, `Metrics sampling`, has five buttons
after `EP calls counted:` and five after `IO calls counted:`: `1/1` (every call), `1/8`,
`1/64`, `1/100`, `off`. The rate is one call in N (an EP dispatch with its combine, an IO put
or get), not a time: how often the counters are read is `interval_ms`, which the buttons do
not touch. A click
sets every process Prometheus scrapes at that moment. The button of the rate in effect has a
coloured ring (blue for EP, amber for IO, red for `off`); it moves at the next refresh after a
click, and it shows the rate the processes have now even when the page's time range is a fixed
one in the past. A rate that is none of the five (set with `sampling.sh`) has no ring. The
traffic charts do not move when the sampling changes: that is the point, what is counted
counts N-fold. The buttons go through a small server `traffic.sh up` starts next to Grafana
([`sampling_gateway.py`](sampling_gateway.py), container `mori-traffic-<user>-sampling`,
127.0.0.1:9390 or `TRAFFIC_GATEWAY_PORT`), which Grafana reaches as the data sources
`mori-sampling` (to set) and `mori-sampling-now` (to read the rate in effect): the browser
talks to nothing but Grafana. Two things to know:

- Open Grafana at the address it knows itself by, or a click fails: `http://localhost:<port>`
  by default (what an ssh tunnel to the same port gives), the `TRAFFIC_LISTEN` address when
  that is set, anything else with `TRAFFIC_GRAFANA_URL=http://host:port/ ./traffic.sh up`.
- Whoever can open the dashboard can change what every job counts. The gateway itself only
  listens on 127.0.0.1, and Grafana lets the buttons post to its path and no other
  (`GF_SECURITY_ACTIONS_ALLOW_POST_URL`).

`traffic.sh sampling` asks every process on its metrics port itself:

```
curl http://node-a:9400/sampling                       # {"ep":1,"io":1,"interval_ms":1000}
curl -X POST 'http://node-a:9400/sampling?ep=8&io=64'  # change; the answer is the new setting
```

From Python it is `mori.metrics.set_sampling(ep=8, io=64, interval_ms=10)` and
`mori.metrics.sampling()`. A job can start with a setting: `MORI_METRICS_SAMPLE_EP`,
`MORI_METRICS_SAMPLE_IO`, `MORI_METRICS_INTERVAL_MS`.

- **One in N counts, N-fold.** With `ep=8` one dispatch in eight, and the combine after it, is
  counted and timed, and adds eight times its tokens, calls and kernel time. With `io=64` one
  transfer in 64 is metered and counts as 64. Every series stays an estimate of the same
  total, so no dashboard query changes; the lines get noisier the larger N is and the shorter
  the window. `1` counts everything, `0` nothing (the series stop moving).
- **Which ones** is decided where the counting is done: on the device for EP (a captured graph
  replays the kernels with no host code in between), by the calling thread for IO. Never
  simply every Nth, so a job that alternates two batch sizes, or reads and writes, does not
  have only one of them counted: IO draws the gap to the next one from 1 to 2N - 1; EP takes
  one dispatch of every N in a row, at a place that changes from one N to the next.
- **Every rank counts the same dispatches**: which one of each N follows from the dispatch's
  number alone, and the ranks of a job make the same dispatches in the same order. So "what A
  sent B is what B received from A" still holds of the sampled counts.
- **A change takes effect** with the next IO transfer and the next EP dispatch, in every engine
  and op of the process. It is the process's setting and is gone when the process exits.
- **Not sampled**, always exact: the per-NIC bytes of the IO RDMA backend
  (`mori_io_nic_*`) and the RDMA a GPU issues itself (`mori_gpu_rdma_bytes_total`).
- The setting in effect is itself a series (`mori_metrics_sample_period{module="ep"|"io"}`,
  `mori_metrics_sample_interval_seconds`), for a record over time of what was counted how.

What counting costs, measured (the time of the operation with counting at 1/N against counting
paused, settings interleaved in one process):

| | every one (1/1) | 1/2 | 1/8 | 1/64 |
|---|---|---|---|---|
| EP dispatch + combine, 128 tokens a rank (MI308X, intra-node) | +3.3 % (5.7 us) | +3.0 % | +1.1 % | +0.3 % |
| EP dispatch + combine, 512 tokens a rank | +1.5 % | +1.1 % | +0.4 % | +0.1 % |
| EP dispatch + combine, 4096 tokens a rank | +0.2 % | +0.2 % | +0.05 % | not measurable |
| EP dispatch + combine between two nodes, 128 tokens a rank (2 x 8 MI300X) | +3.5 % (9 us) | | +0.65 % | +0.4 % |
| EP between two nodes, 512 tokens a rank | +1.2 % | | +0.3 % | +0.2 % |
| EP between two nodes, 4096 tokens a rank | +0.1 % | | within the noise | within the noise |
| IO between two nodes over RDMA, transfers of 4 to 64 KiB (3.3 to 4 us) | +0.75 to +6 % | +0.7 to +2.9 % | +0.3 to +0.65 % | within the noise |
| IO between two nodes over RDMA, 256 KiB and more (at the line rate) | none | none | none | none |
| IO, a 4 KiB transfer over RDMA (2 to 3.5 us) | +1.0 to 1.2 % | +0.7 % | within the noise (0.3 %) | within the noise |
| IO, a 4 KiB transfer over XGMI (9 us) | +1.8 % | +0.7 % | +0.2 % | within the noise |
| IO, transfers of 64 KiB and 1 MiB (RDMA, XGMI) | within the noise (0.2 %) | | | |

The first three EP rows are the repo's intra-node benchmark (v1 kernels, 4 ranks, graph
replays timed with events), the settings interleaved in one process; the two-node EP rows are
the v2 inter-node benchmark, 16 ranks, one setting per run and two to four runs a setting (on
that path a setting changed inside a run does not show its steady cost). The IO rows are
single transfers issued from Python; between two nodes they complete on the engine's CQ
thread, which shares the flow's counters with the submitting thread. Metering one IO transfer
takes 82 ns on the calling thread (12 ns at 1/8, 3 ns at 1/64, under 2 ns paused). Paused (`0`)
is the same as metrics off at start to within 0.3 % inside a node; between nodes it stays
0.8 % above it at 128 tokens, which is the byte counting of the RDMA the GPUs issue
themselves (not sampled). The metrics thread's interval (1 s down
to 1 ms) does not change the time of an EP kernel; what it costs is CPU (see *Millisecond
resolution*). The EPv2 kernels were checked for the estimates (one in 8 counted: the call count
comes out within 1 % of the exact one, HIP and FlyDSL), not for the cost: separate runs of
that benchmark differ by more than the cost.

With N above 1 the first half of the cost does not shrink as 1/N: a launch that counts is
timed from the end of the launch before it, so the dispatch + combine right before one that
counts still records when it ended (a combine has to wait for its blocks to know). That is
why 1/2 costs nearly as much as 1/1. And what is timed is a launch that counts, which is the
one that carries the cost: at 128 tokens the kernel time per call reads about 2 % higher at
1/64 than at 1/1; from 512 tokens on the difference is under 1 %.

## Reading the overview dashboard

The first row, `Metrics sampling`, is a switch, not a chart: buttons that set how many EP and
IO calls the metrics count, while the jobs run (see *Sampling* above). Below it, every chart
holds IO and EP against each other; IO is amber and EP blue wherever the two
appear together. `Host` and `GPU` select whose traffic is shown (what those GPUs send).
`Rate window` is the time every rate on the page is averaged over: `auto` is the data source's
own (20 s for scraped data); `1ms` to `500ms` are for the millisecond data source. There the
counters advance one batch at a time (an IO transfer, an EP launch), so pick a window no
shorter than the time between two updates of what you look at: below it the line is a train
of pulses that drop to zero in between (their height is right, the legend's mean is too low).
With scraped data a window under about 10 s is empty.

- `IO and EP on the same links, over time`: one chart for XGMI and one for RDMA, side by side, the selected
  GPUs together, bytes over wall time. Peaks that alternate: IO and EP take turns on the
  links. Peaks at the same time: both are on them at once.
  The thin grey line is the two added up. In the XGMI chart the dashed line is what XGMI can
  carry: `XGMI limit per GPU` (a text box at the top, in GB/s) times the selected GPUs that
  send. The default, 291.25 GB/s, is the payload measured on 8 x MI300X with every GPU sending
  to the 7 others, 2.33 TB/s for the node; set it for other hardware. It is a measured
  ceiling, not the line rate (7 links x 64 GB/s a GPU), which payload does not reach.
- `Per GPU`: the same per GPU, four charts in a row (XGMI by IO, XGMI by EP, RDMA by IO, RDMA
  by EP), one line for each selected GPU (what it sends; a GPU has the same colour in all
  four charts and in the IO and the EP dashboard). The dashed line in the XGMI charts is the
  limit of one GPU, which its IO and its EP traffic share. GPUs that send at the same rate
  draw on top of each other: read the legend table.

- `EP kernels`: what one call does, which does not move with how often EP is called.
  `EP bytes per call` (left) is how much data one dispatch and one combine of a rank moves
  (the bytes of the tokens its experts get, or give back; an EP benchmark's "bytes");
  `EP kernel time per call` (right) is the time one dispatch and one combine take, measured
  on the device. The first over the second is the kernel bandwidth, which the EP dashboard
  shows.
- `IO transfers`, laid out like the EP row (how much on the left, how long on the right):
  `IO bytes per transfer` is how much one put (write) and one get (read)
  carries; `IO time per transfer` is how long it takes from submission to completion (XGMI:
  until the caller looks at its status). By backend (colour); dashed lines are gets.

The RDMA numbers for EP are everything the
GPUs put on RDMA themselves (`mori_gpu_rdma_bytes_total`); for IO they are the RDMA backend's
completed bytes by source GPU. A process that exits takes its last few seconds of traffic
with it (nothing scrapes it any more), so totals over a whole run read a few percent low.

## Reading the EP dashboard

EP has two bandwidths, and they answer different questions:

| | Numerator | Denominator | Says |
| --- | --- | --- | --- |
| **Kernel bandwidth** (tiles `Dispatch bandwidth`, `Combine bandwidth`; the row `Kernel bandwidth of dispatch / combine`) | bytes of the tokens a kernel handled | time the kernel ran, measured on the device | how fast the links carry it; this is what an EP benchmark reports, and it does not move with the call rate |
| **Payload over wall time** (tiles `XGMI sent per GPU`, `RDMA sent per NIC`; the matrix; the `GPU (xgmi)` and `NIC (rdma)` rows) | bytes moved | wall time | how much EP moves; rises and falls with how often EP is called |

A low payload rate with a high kernel bandwidth means EP is called rarely, not that a link is
slow. In the screenshot the load changes several-fold (`Sent per GPU over XGMI`) while
`Kernel bandwidth by operation` stays level.

The dashboard, from the top:

- Tiles: the two kernel bandwidths (all selected ranks together), `XGMI sent per GPU` (what a
  GPU sends over XGMI to all its peers together, averaged over the GPUs that send), `RDMA
  sent per NIC` (what a NIC sends for the GPUs' own RDMA, averaged over the NICs that send),
  and `Data age` (seconds since the newest sample; growing or `no data` means nothing is
  reporting).
- `Kernel bandwidth of dispatch / combine`: one chart with a line per operation (and, for
  inter-node kernels, a dashed line per operation for the RDMA hop), and one chart per
  operation with a line per GPU. A GPU that stays below the others has a slower link or a
  slower kernel.
- `GPU to GPU`: one table per selected host, of the XGMI payload between that host's GPUs,
  dispatch and combine together. Rows are the sending GPU, columns the receiving GPU; a cell
  says its rate and its direction, `6.2 GB/s (4 -> 5)`.
- `GPU (xgmi)`: what each GPU sends and receives over XGMI.
- `NIC (rdma)`: what the GPUs themselves put on each NIC (`mori_gpu_rdma_bytes_total`, EP
  payload with its headers and signals), sent and received. Empty when EP runs inside one
  node. This is the number to hold against the NIC's own counters.

`Host` selects the GPUs and NICs of those hosts; legends start with the GPU or the NIC, then
the host, and a GPU and the NIC with the same index have the same colour in every chart, as
on the IO dashboard. Token counts, launches, kernel time per launch, traffic by path and by
RDMA operation, and the node to node view are not on the dashboard any more: the metrics are
exported as before, query them in Explore.

Kernel bandwidth is computed per rank as `tokens x bytes per token / kernel seconds`, with the
tokens that stay on the GPU included and the RDMA hop left out (dispatch: tokens the rank's
experts received; combine: tokens they sent back),
`bytes per token = XGMI bytes / XGMI tokens`, and only for kernels that are timed. Against the
benchmark's own number (`bench_dispatch_combine.py`: tokens received x hidden size / kernel
time) the kernel time agrees within 1 % (MI300X, 512 to 16384 tokens); the bytes are slightly
higher here because a token is counted with its indices and weights (0.8 % at a hidden size of
4096, 3.5 % at 512).

Everything is **payload**, not link bytes. The XGMI hardware counters (`xgmi-netmon`) read
1.25 to 4.2 times the payload depending on the traffic pattern, so the two cannot be held
against each other byte for byte. NIC counters can: for RDMA puts the software count equals
the NIC's transmit counter on the AMD AINIC (RoCE), and is about 1 % below the port counter
of an InfiniBand NIC (packet headers).

What has series: the intra-node kernels (v1, v2 HIP, v2 FlyDSL) and the inter-node kernels
(v1 `InterNodeV1` / `InterNodeV1LL`, v2 `v2` / `v2_ll`) count and time themselves. An
inter-node dispatch or combine is several launches in a row; its time is that of the whole
sequence on the stream, which is what a benchmark times. AsyncLL, the gfx125x intra-node
kernels and KIWI export no `mori_ep_*` series yet, only the RDMA bytes of the `NIC (rdma)`
row.

Inter-node, a token is counted on each physical hop: once over RDMA to the proxy rank of the
other node, once more over XGMI to the rank its expert is on. The kernel bandwidth counts it
where it ends up (what the inter-node benchmark calls the XGMI bandwidth). The dashed lines
in `Kernel bandwidth by operation` are the RDMA hop by itself: the bytes the ranks put on
RDMA for other nodes over the same kernel time. The inter-node benchmark's RDMA bandwidth
also counts a token's own node as a destination, so with N nodes and balanced routing it
reads N / (N - 1) times that line.

## Reading the IO dashboard

- A transfer is counted once, in the process that started it.
- There is one `GPU (<backend> backend)` row for every backend in use, with what each GPU
  sends and receives over that path (`cpu` = host memory): `xgmi` is traffic over the links
  between the GPUs of a host, `rdma` goes through the GPU's NIC and is the same bytes the
  `NIC (rdma backend)` row shows per NIC. `Backend` at the top selects which of these rows
  are shown, and one path for the tiles and the matrix. Legends start with the GPU or the NIC,
  then the host.
- The matrix is one table per selected host, of the traffic between that host's own GPUs:
  the host is in the title, rows are the sending GPU and columns the receiving GPU, by index.
  A cell says its rate and its direction, `48.5 GB/s (4 -> 5)`; the blue is stronger from
  10 GB/s, weaker from 1 GB/s, faint below. Traffic between two hosts has no matrix; read it
  from the per-GPU and per-NIC charts.
- The first two tiles say how fast one GPU and one NIC go: `XGMI sent per GPU` is what a GPU
  sends over XGMI to all its peers together, averaged over the GPUs that send; `RDMA sent per
  NIC` is what a NIC sends, averaged over the NICs that send.
- `Data age` is the seconds since the newest sample. A few seconds means the processes are
  reporting, and a chart at zero is no traffic; a growing age or `no data` (red) means nothing
  is reporting: the job ended, or the scrape or the push broke.
- The dashed line in the NIC charts, and in the `GPU (rdma backend)` row, is the link speed
  the system reports for the NIC (`mori_io_nic_link_speed_bytes_per_second`). XGMI has no
  such line: the system reports no speed for those links, and what a GPU can send depends on
  how many peers it sends to (about 49 GB/s a peer on MI300X).
- A GPU and the NIC with the same index have the same colour in every chart, and legends are
  in name order.
- `Host` selects by what a panel shows. `Not successful`, `In flight` and the latency tile
  show the engines running on the selected hosts. `Sent` / `Received per GPU` and `per NIC`
  show the GPUs and NICs of the selected hosts, whichever engine started the transfer: pick
  one host to see only its GPUs and NICs, `All` to see every host side by side. The list has
  every host that runs an engine or is one end of a flow, so a host whose own processes are
  not scraped can be picked too.
- The latency tile is from submission to the terminal state: for RDMA until the completion
  was processed, for XGMI and fabric until the caller looked at the status. The quantile is
  estimated from histogram buckets (1, 2.5, 5 per decade), so it is exact only to a bucket.
- `Not successful` counts failed transfers and unobserved ones (the status object was
  destroyed or reused before anything saw the transfer finish). Both should be zero.
- The `NIC` row has data only with the RDMA backend. `Sent per NIC` is the number to hold
  against the NIC's own transmit counter. Simulated IO has no NIC.
- There are no charts over time of throughput by operation, failures, latency or transfers
  in flight (the tiles give one number each). All of it is exported; query it in Explore.

## Your own Prometheus or Grafana

- Prometheus: copy the `mori_traffic` job from [`prometheus.yml`](prometheus.yml). Any scrape
  interval works; the charts use `$__rate_interval`.
- Grafana: import the two files in [`grafana/dashboards/`](grafana/dashboards/). They take the
  data source from the `Data source` variable at the top. Set the data source's scrape
  interval to match Prometheus (here 5 s, see `grafana/provisioning/datasources/traffic.yml`).
- A recording from `ainic-netmon` / `xgmi-netmon` lives in its own stack
  ([`../netmon/`](../netmon/README.md)); the series share `host`, `netdev` and GPU index, so
  the two can be compared for the same time range.

## Known limits

- A process killed by a signal leaves its `<host>-<pid>.json`; Prometheus then shows that
  target as down until the file is deleted.
- Containers on one host that have the same hostname and their own pid namespaces can still
  pick the same file name in a shared directory: give them different `MORI_METRICS_HOST_ID`s,
  or list the targets with `TRAFFIC_TARGETS`.
- The address in a target file is the `host` label (`MORI_METRICS_HOST_ID` if set), unless
  `MORI_METRICS_ADDR` is set. A host id that does not resolve needs `MORI_METRICS_ADDR`.
- A target file gives its target the labels `pid` and `local_rank`. If you write your own
  targets, do not add a label named `rank`, `gpu` or `host`: the series carry those themselves,
  and Prometheus would rename the series' own to `exported_rank` and so on.
- A series that is published only once it has counted something (`mori_gpu_rdma_bytes_total`,
  EP peers other than the ranks of the node and the proxy ranks) appears with a value already
  in it, which `rate()` and `increase()` do not see; the totals are right.
- Kernel bandwidth takes the bytes of a token from the XGMI series, so a rank that sends no
  token to another GPU of its node in the window has none.
- The per-GPU colours are fixed for GPU 0 to 7 of one host. The matrix rows sort as text, so
  a node with more than 10 GPUs shows GPU10 before GPU2.

## Editing the dashboards

Plain Grafana JSON (schema of Grafana 13). Edit in Grafana, export, and replace the file; keep
the `uid`s (`mori-traffic-overview`, `mori-ep-traffic`, `mori-io-traffic`), the dashboards
link to each other by them. Conventions: one unit per chart and no second axis; dispatch is blue and combine orange
everywhere, and on the overview IO is amber and EP blue; red and the warning yellow are kept for failures; a table legend where a chart has more than a
handful of series.
