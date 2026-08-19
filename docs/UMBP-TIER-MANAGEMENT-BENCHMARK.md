# UMBP Distributed Tier-Management Benchmark

`umbp_tier_bench` runs deterministic synthetic or recorded workloads through an
in-process Master and one or more PoolClients. It measures the distributed UMBP
PUT/GET path, including routing, named backend placement, peer-local offload,
payload validation, throughput, latency, and backend capacity.

## Build

```bash
cmake -S . -B build -DBUILD_UMBP=ON -DBUILD_TESTS=ON
cmake --build build --target \
  umbp_tier_bench test_umbp_workload_trace test_umbp_workload_runner
```

The benchmark is a manual target and is not run by plain `ctest`.

## Run, generate, and replay

Run a weighted DRAM workload:

```bash
./build/src/umbp/umbp_tier_bench run \
  --profile mixed --clients 4 --operations 10000 --keys 2000 \
  --read-ratio 0.7 --min-value-bytes 4096 --max-value-bytes 4096 \
  --qps 20000 --placement weighted --backends-per-peer 2 \
  --placement-weights 1,3
```

Generate a reusable trace and replay it under another policy:

```bash
./build/src/umbp/umbp_tier_bench generate \
  --trace workload.umbptrace --profile hotset --seed 42 \
  --operations 100000 --keys 10000 --clients 8 --qps 50000

./build/src/umbp/umbp_tier_bench replay \
  --trace workload.umbptrace --placement weighted \
  --backends-per-peer 2 --placement-weights 1,3
```

Use `umbp_tier_bench <subcommand> --help` for the complete option list.

## JSON backend and tier policy

`--config` replaces the legacy `--tier`, `--backends-per-peer`,
`--placement`, and `--placement-weights` cluster topology:

```bash
./build/src/umbp/umbp_tier_bench run --config policy.json \
  --profile capacity-pressure --operations 1000 --page-size 2MiB
```

```json
{
  "schema_version": 1,
  "entry_tier": "hot",
  "backends": {
    "hbm":   { "type": "hbm", "capacity": "80GiB", "devices": [0, 1] },
    "dram":  { "type": "dram", "capacity": "512GiB", "numa_node": 0 },
    "ssd_a": { "type": "ssd", "capacity": "1TiB", "path": "/mnt/kvcache/hot" },
    "ssd_b": { "type": "ssd", "capacity": "3TiB", "path": "/mnt/kvcache/cold" }
  },
  "tiers": [
    {
      "name": "hot",
      "backends": { "hbm": 100 },
      "offload_to": ["warm"],
      "offload_trigger": "on_evict"
    },
    {
      "name": "warm",
      "backends": { "dram": 70, "ssd_a": 30 },
      "offload_to": ["cold"],
      "offload_trigger": "watermark",
      "high_watermark": 0.9,
      "low_watermark": 0.7,
      "candidate_policy": "lru"
    },
    {
      "name": "cold",
      "backends": { "ssd_b": 100 },
      "promote_on_read": true,
      "promotion_mode": "copy"
    }
  ]
}
```

New PUTs use `entry_tier` (the first tier when omitted) and its deterministic
backend weights. `NO_SPACE` recursively spills through `offload_to`; every
destination tier applies its own weights.
`on_evict` copies the key to the first available target before deleting its
source copy. `watermark` queues asynchronous LRU offload after aggregate tier
use reaches `high_watermark`. Migration continues toward `low_watermark` and
retries transient failures with bounded backoff. `promote_on_read` copies or
moves a cold hit into the fastest reachable upstream tier.

Backend and logical-tier names must be unique, and every backend must belong to
exactly one logical tier. Offload targets may name a backend or logical tier
and must resolve to a later tier, which makes the
topology acyclic. A multi-device HBM definition expands to one backend per
device while preserving the logical backend's aggregate weight and capacity.
The current per-peer limit is eight concrete backends.

Production distributed clients may set
`UMBPDistributedConfig::backend_policy_path`; Python exposes the same
`backend_policy_path` property. Distributed-backed standalone servers also
accept `UMBP_BACKEND_POLICY`.

Each benchmark run prints `backend_placement` and `tier_transitions` CSV
sections. Production clients can record successful PUT/GET traffic with
`UMBP_WORKLOAD_TRACE_PATH`; replay it with payload validation disabled because
production payload bytes are not embedded in the trace.

**Page size vs value size.** Each value occupies a whole page. The defaults
(`page_size=2MiB`, `backend_capacity=256MiB`) only hold 128 keys per backend.
A 4KiB value still consumes 2MiB. For small-value policy sweeps use
`--page-size 64KiB --backend-capacity 2GiB` (or size capacity as
`page_size * key_count`).

**GET affinity.** Synthetic PUT/GET pairs for one key stay on one client
stream. `--affinity none` plus `--get-strategy local` can still miss: Master
`most_available` may place the PUT on another node, and the producing client's
local GET then waits out the publication retry. For GET-heavy same-client
profiles use `--affinity local`. To exercise the remote RDMA path, PUT from
one client and GET from another (see `test_umbp_tier_benchmark`).

**RDMA QP depth.** `mori_io` defaults `max_send_wr=8192`. On some mlx5 devices
that depth times SGE/inline exceeds the per-QP work-queue budget, so
`ibv_create_qp` returns EINVAL even though `max_qp_wr` is larger. Set a smaller
depth before running multi-client benchmarks or integration tests:

```bash
export MORI_IO_QP_MAX_SEND_WR=1024
export MORI_IO_QP_MAX_CQE=4096
```

Also set `LD_LIBRARY_PATH` to the build tree (`build/src/application:build/src/io:build/src/metrics`)
when running binaries from `build/`.

**SSD.** Values must not exceed `--page-size`. `max_allocatable_bytes` for an
SSD backend is `min(available_bytes, page_size)`.

The workload controls are profile, seed, operation and key counts, minimum and
maximum value size, value-size distribution, read ratio, client count, batch
size, and QPS. Cluster controls select DRAM, HBM, or SSD; backend count,
capacity, and page size; single or weighted placement; and Master put strategy,
affinity, and get strategy.

`run` defaults to max-throughput scheduling. `replay` defaults to open-loop
scheduling using trace timestamps. `--time-scale` scales those timestamps, and
`--max-throughput` ignores them while preserving order within each client.
Different clients execute concurrently.

`--settle-ms` controls both the registration and post-run heartbeat settle.
Master eviction uses production defaults.

## Synthetic profiles

- `sequential`: write-only traversal of the key space.
- `uniform`: uniformly selected keys with the configured read ratio.
- `hotset`: Zipf-distributed access within the default hot set.
- `read-after-write`: alternating PUT and GET for the same key.
- `mixed`: deterministic key traversal with the configured read ratio.
- `capacity-pressure`: non-repeating writes intended to exceed capacity.

Random choices and values are derived from the seed. UMBP keys are immutable:
repeated logical writes create versioned keys, and reads target the latest
version. Each key's dependency chain stays on one client stream.

## Trace contract

Trace files start with the versioned `UMBPTRCE` envelope and contain
length-delimited protobuf records defined in
`src/umbp/distributed/proto/umbp_workload.proto`: one
`WorkloadTraceHeader`, followed by zero or more `WorkloadEvent` records.

The header records the schema version, nanosecond time unit, payload seed, and
all synthetic generation settings. Events record relative time, client id,
operation id, PUT/GET kind, key, value size, and batch id. Readers reject
unsupported versions and malformed records.

Payload bytes are omitted. PUT data is generated from
`(key, operation_id, seed)`, and GET data can be checked against the same
sequence. Disable checking with `--no-payload-validation` when replaying a
future externally recorded trace without payload identity.

Successful PUTs flush heartbeat publication. A GET for a key produced by the
same run retries for a fixed five seconds while that publication is pending, so
asynchronous Master visibility is not reported as a storage miss.

## Results

The `summary` CSV section reports aggregate operation and byte counts,
failures, misses, validation failures, wall time, throughput, latency
percentiles, and scheduling lag.

After the measured interval and heartbeat settle, `backend_placement` reports
each node/backend's tier, owned key count, total and available capacity, and
maximum allocatable block. Placement inspection is outside the timed region and
uses side-effect-free backend counters.

For policy comparisons, retain the trace and change only topology or routing
arguments. Keep the settle interval, heartbeat environment, and hardware
configuration identical.
