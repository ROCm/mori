# MORI-IO Tuning Guide

This guide explains the knobs that most directly shape MORI-IO RDMA performance and
how to reason about them. It focuses on five flags exposed by the benchmark
(`tests/python/io/benchmark.py`) that map onto the `RdmaBackendConfig` used by every
transfer.

## Table of Contents

- [Where these knobs live](#where-these-knobs-live)
- [The flags](#the-flags)
  - [`--num-qp-per-transfer`](#--num-qp-per-transfer)
  - [`--post-batch-size`](#--post-batch-size)
  - [`--busy-wait`](#--busy-wait)
  - [`--disable-chunking`](#--disable-chunking)
  - [`--batch-contiguous`](#--batch-contiguous)
- [Best known config (Thor2)](#best-known-config-thor2)
- [Recommended profiles](#recommended-profiles)
- [Environment variable equivalents](#environment-variable-equivalents)
- [Tuning workflow](#tuning-workflow)

## Where these knobs live

Three of the flags are backend config (`RdmaBackendConfig`, `include/mori/io/backend.hpp`)
and are honored by every transfer regardless of how you call the engine:

| Flag | Config field | Env override |
|------|--------------|--------------|
| `--num-qp-per-transfer` | `qpPerTransfer` | `MORI_IO_QP_PER_TRANSFER` |
| `--post-batch-size` | `postBatchSize` | `MORI_IO_POST_BATCH_SIZE` |
| `--disable-chunking` | `enableTransferChunking` (inverted) | `MORI_IO_ENABLE_CHUNKING` |

Two are benchmark-side only — they change *how the benchmark drives the engine*, not the
backend config:

| Flag | Effect |
|------|--------|
| `--busy-wait` | Calls `TransferStatus::WaitBusy()` (spin) instead of `Wait()` (block on cv) |
| `--batch-contiguous` | Lays out transfer offsets contiguously so WRs can be merged |

## The flags

### `--num-qp-per-transfer`

**Config `qpPerTransfer`, default `4` in the benchmark (`1` in the raw pybind default).**

A transfer is split into per-QP post batches spread round-robin across `qpPerTransfer` queue
pairs (the start QP rotates by transfer id, so even single-WR transfers don't all land on
`eps[0]`). More QPs = more parallel send queues, which relieves single-SQ pressure under load;
fewer QPs = less state and lower overhead. `qpPerTransfer` also caps the worker-thread
executor at `min(qpPerTransfer, numWorkerThreads)` threads.

> **AINIC:** a single QP cannot saturate the link — use **at least 2 QPs**, and **4** for full
> bandwidth.

**Rule of thumb:** On Thor2, `1`/`2`/`4` are equivalent — default to `1`. Scale up only when
the single SQ is provably the bottleneck (host memory, multi-NIC, ionic), matching
`--num-worker-threads`.

### `--post-batch-size`

**Config `postBatchSize`, default `-1` (auto).**

WRs handed to a single `ibv_post_send`. Two competing effects, and message size decides which
wins (see the [Thor2 BKC](#best-known-config-thor2)):

- **`-1` (auto)** — posts each QP's whole share in one call
  (`ceil(mergedWrCount / qpCount)`, clamped to the SQ's `maxSqDepth`). Fewest doorbell rings;
  best for **small messages**, where doorbell overhead dominates.
- **`1`** — one WR per call. Overlaps posting with in-flight transfer (the CPU posts the next
  WR while the NIC DMAs the current one), so it **wins on throughput for large messages**
  where each DMA is long enough to hide the posting cost.
- On SQ-full / `ENOMEM`, **reduce** `postBatchSize` (or raise `MORI_IO_QP_MAX_SEND_WR` /
  `qpPerTransfer`); the error reports the effective value.

**Rule of thumb:** small messages → `-1`; large messages → `1`.

### `--busy-wait`

**Benchmark-side: `WaitBusy()` vs `Wait()`.**

How the waiting thread observes completion:

- **Default (blocking)** — blocks on a condition variable. Frees the core, but pays a
  cross-thread wakeup latency of ~**5–10 µs** per completion.
- **`--busy-wait`** — spins on the completion flag (`WaitBusy`), removing that latency.

Good for **latency-sensitive** cases, but spinning **burns a lot of CPU cycles** — avoid it
when many transfers are in flight or CPU is contended, since the wasted cycles are stolen from
the progress/completion path and can hurt throughput.

### `--disable-chunking`

**Config `enableTransferChunking`, chunking is ON by default.**

Chunking splits a transfer into `--chunk-bytes` pieces (default 64 KB, up to `--max-chunks`
= 64), which exposes intra-transfer parallelism across QPs and enables the GPU worker-thread
posting path (GPU local memory only; host memory falls back to inline posting).

- **On (default)** — best for large single transfers, especially GPU memory: pipelined across
  QPs/threads.
- **`--disable-chunking`** — one WR per transfer (still capped by the NIC's max message size).
  Lower overhead when messages are already small (≤ chunk size) and keeps posting on the
  simple inline path.

`--chunk-bytes` and `--max-chunks` only matter while chunking is enabled.

### `--batch-contiguous`

**Benchmark-side: transfer buffer offset layout.**

Whether batched transfers land on contiguous offsets:

- **`--batch-contiguous`** — contiguous offsets let adjacent WRs **merge**
  (`MergedWorkRequest`) into fewer, larger WRs: less SQ pressure, higher throughput.
- **Default (strided)** — each transfer is a separate WR, maximizing WR count. The stress
  path — reproduces SQ-full / `ENOMEM` and measures the per-WR overhead floor.

**Rule of thumb:** enable it for headline throughput; leave off to stress the SQ.

## Best known config (Thor2)

Best-known write-benchmark command on Thor2 (run on each node with its own `--rank`):

```bash
tools/run_internode_io_benchmark.sh \
    --rank 0 --master-addr 10.162.224.131 --master-port 29573 --ifname enp49s0f1np1 \
    -- --op-type write --enable-batch-transfer --transfer-batch-size 128 \
       --all --sweep-start-size 1024 --sweep-max-size 1048576 --iters 80 \
       --num-qp-per-transfer 1 --post-batch-size 1 --busy-wait --disable-chunking \
       --warmup-iters 64 --batch-contiguous
```

> **Choosing `--post-batch-size`:** the value flips with message size. The command above uses
> `1` because it sweeps mostly large messages (≥ 1 KB up to 1 MB), where posting one WR at a
> time overlaps posting with transfer and wins on throughput. For **small messages (≤ 16 KB)**
> use `-1` (auto) instead: transfers are too short to overlap, so doorbell/CPU overhead
> dominates and posting each QP's whole share in one ring is faster. A size-adaptive client
> should pick `post_batch_size = -1 if message_bytes <= 16 * 1024 else 1`.

Everything else is stable: **`--num-qp-per-transfer` `1`/`2`/`4` are equivalent** on Thor2, so
`1` is the default.

## Recommended profiles

**Latency-optimized (the command at the top of this guide):**

```bash
--num-qp-per-transfer 1 --post-batch-size -1 --busy-wait --disable-chunking --batch-contiguous
```

Single QP, no chunking, merged WRs, and a spinning waiter — minimal overhead. Best for small
messages at low queue depth.

**Throughput-optimized (large messages, > 16 KB):**

```bash
--num-qp-per-transfer 1 --post-batch-size 1 --batch-contiguous
# chunking left ON (default), no --busy-wait
```

`--post-batch-size 1` overlaps posting with in-flight transfer (the throughput win on large
messages per the [Thor2 BKC](#best-known-config-thor2)); contiguous merging keeps the pipeline
full; blocking waits free the CPU for the progress path. `qp=1` is sufficient on Thor2 — raise
it only if the single SQ is provably the bottleneck.

## Environment variable equivalents

The config-backed flags can also be set without touching code, which is useful when MORI-IO
is embedded in another app:

```bash
export MORI_IO_QP_PER_TRANSFER=1
export MORI_IO_POST_BATCH_SIZE=-1
export MORI_IO_ENABLE_CHUNKING=0            # disable chunking
export MORI_IO_CHUNK_BYTES=65536
export MORI_IO_MAX_CHUNKS=64
export MORI_IO_NUM_NICS_PER_TRANSFER=1
```

`--busy-wait` and `--batch-contiguous` are properties of the caller's transfer loop, not the
backend config, so they have no env equivalent — replicate them in your own client code
(`WaitBusy()` and contiguous offsets, respectively).

## Tuning workflow

1. **Start from a profile.** Latency profile for small messages, throughput profile for
   large / batched.
2. **Sweep one knob at a time.** Use `--all` (message-size sweep) and `--all-batch`
   (batch-size sweep) to find where the current setting stops scaling.
3. **Leave `--num-qp-per-transfer` at `1`** (on Thor2, 1/2/4 are equivalent); scale up only
   when the single SQ is provably the bottleneck, and match `--num-worker-threads`.
4. **Set `--post-batch-size` by message size:** `-1` for ≤ 16 KB, `1` for > 16 KB (Thor2
   BKC). Step to a small fixed N only if you hit SQ-full / `ENOMEM` (or raise
   `MORI_IO_QP_MAX_SEND_WR`).
5. **Toggle chunking by message size:** on for large single transfers, off once messages
   are at or below `--chunk-bytes`.
6. **Only spin (`--busy-wait`) with a spare core** and few outstanding transfers; otherwise
   let the blocking waiter free the CPU.
