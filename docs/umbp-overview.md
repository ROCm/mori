# MORI-UMBP Overview

UMBP (Unified Memory and Bandwidth Pool) is a key-value pool for KV cache that
spans GPU HBM, host DRAM and SSD, on one node or across many. An inference
engine puts KV blocks in by key and gets them back later from whichever node and
tier holds them. MORI-IO moves the bytes between nodes over RDMA.

SGLang uses UMBP as a HiCache storage backend (`--hicache-storage-backend umbp`),
including in prefill-decode disaggregated serving.

## Architecture

- **Clients** run inside the engine process: `mori.umbp.UMBPClient`, configured
  with a `UMBPConfig`. Data moves through `put_from_ptr` / `get_into_ptr` and
  their batch forms.
- **The master**, `umbp_master <address> [metrics_port]`, tracks which node
  holds which block, routes puts to nodes, and answers external-KV queries. It
  keeps that metadata in memory, or in Redis when MORI is built with
  `USE_REDIS_BACKEND=ON` and the master runs with `UMBP_METADATA_BACKEND=redis`
  (see the [Redis metadata store design](https://github.com/ROCm/mori/blob/main/src/umbp/doc/design-redis-metadata-store.md)).
- **Peers** are each node's storage backends (DRAM, HBM, SSD) and the transfer
  engines that move data to and from them: MORI-IO for RDMA between nodes, local
  and HBM copy engines within a node, and POSIX or GDS for SSD.
- **Eviction** is local by default: each node evicts at its own watermarks, by
  its own LRU, without asking the master. `UMBP_EVICTION_MODE` selects the
  master-assisted modes.

## Deployment modes

| Mode | When | What runs |
|---|---|---|
| Embedded | The config names no deployment | An in-process pool: one DRAM medium and the local copy engines. No master, no peers |
| Distributed | A master address is configured | Each client registers with the master, serves its own tier to peers and reads theirs over RDMA |
| Standalone server | `UMBPConfig.standalone_process` is set | The pool lives in a per-node `umbp_standalone_server` process, which clients reach over gRPC |
| Simulation | `UMBPConfig.sim.enabled` is set | Allocation, placement and routing run for real; payloads and transfers are virtual. See [Simulation](umbp-simulation.md) |

## Tiers

- A node serves a single medium (`distributed.medium`, or
  `UMBP_DISTRIBUTED_MEDIUM` for the standalone server), or several backends
  described by a tier policy (`backend_policy_path`, or `UMBP_BACKEND_POLICY`
  for the standalone server). The policy format is described on the
  [tier-management benchmark](UMBP-TIER-MANAGEMENT-BENCHMARK.md) page.
- SSD tiers, including multi-drive ones: [Pure-SSD mode](umbp-pure-ssd-mode.md).
- DRAM on multi-socket hosts: [NUMA placement](umbp-numa-tier-placement.md).

## Building

`pip install` builds UMBP by default (`BUILD_UMBP=ON`). It needs the gRPC and
Protobuf development packages; without them the build skips UMBP with a warning
and installs the rest of MORI. A plain CMake build needs `-DBUILD_UMBP=ON`, which
is off by default. The wheel ships `umbp_master` and `umbp_standalone_server`,
and `import mori.umbp` points `UMBP_MASTER_BIN` and `UMBP_STANDALONE_BIN` at them
unless those are already set.

## Configuration and observability

- Settings are `UMBPConfig` fields; `UMBPConfig.from_environment()` fills them
  from `UMBP_*` environment variables. See
  [Configuration and environment variables](umbp-configuration.md).
- The master serves Prometheus metrics on port 9091 (or its second argument). A
  standalone server without a master serves its own on 9092
  (`UMBP_STANDALONE_METRICS_PORT`). Dashboards: [Monitoring](umbp-monitoring.md).
- Each client (an engine's, or the one inside `umbp_standalone_server`) logs a
  one-line activity summary once a minute at INFO (`UMBP_SUMMARY_INTERVAL_MS`);
  `umbp_master` does not. UMBP logs at `error` by default, so set
  `MORI_UMBP_LOG_LEVEL=info` to see it.
- [KV indexer export](umbp-kv-indexer-export.md) reports committed DRAM blocks
  to SGLang's KV indexer.
- The control-plane client, `UMBPMasterClient`, is in the
  [API reference](api/umbp.rst).

## Running and measuring

- [PD disaggregation benchmark](MORI-UMBP-PD-BENCHMARK.md): SGLang
  prefill-decode serving with UMBP across nodes.
- [Tier-management benchmark](UMBP-TIER-MANAGEMENT-BENCHMARK.md):
  `umbp_tier_bench` on synthetic or recorded workloads.
- [Simulation](umbp-simulation.md) and [Calibration](umbp-calibration.md).
- [Single-node smoke test](MORI-UMBP-SINGLE-NODE-GUIDE.md): the developer
  runbook for an end-to-end SGLang check on one node.
