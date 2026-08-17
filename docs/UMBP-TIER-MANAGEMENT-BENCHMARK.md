# UMBP Distributed Tier-Management Benchmark

`umbp_tier_bench` drives the distributed UMBP data path directly. It does not
depend on an inference framework: an in-process Master and a configurable set
of PoolClients execute either a deterministic synthetic workload or a recorded
workload trace.

The benchmark covers:

- Master RoutePut selection and node affinity.
- local-preferring or random RouteGet selection.
- single-backend or weighted same-tier PeerPool placement.
- capacity pressure and master/peer eviction behavior.
- end-to-end PUT/GET latency, throughput, misses, payload validation, placement,
  and backend capacity.

It does not exercise the separate local/standalone DRAM-to-SSD promotion and
demotion stack.

## Build

Configure MORI with UMBP and tests enabled, then build the tool and unit tests:

```bash
cmake -S . -B build -DBUILD_UMBP=ON -DBUILD_TESTS=ON
cmake --build build --target \
  umbp_tier_bench test_umbp_workload_trace test_umbp_workload_runner
```

The benchmark is a manual target and is not run by a plain `ctest`.

## Quick start

Run a CPU-only DRAM workload:

```bash
./build/src/umbp/umbp_tier_bench run \
  --profile mixed --clients 4 --operations 10000 --keys 2000 \
  --read-ratio 0.7 --min-value-bytes 4096 --max-value-bytes 4096 --qps 20000 \
  --placement weighted --backends-per-peer 2 --placement-weights 1,3
```

Generate a reusable trace, then replay it under two policies:

```bash
./build/src/umbp/umbp_tier_bench generate \
  --trace workload.umbptrace --profile hotset --seed 42 \
  --operations 100000 --keys 10000 --clients 8 --qps 50000

./build/src/umbp/umbp_tier_bench replay \
  --trace workload.umbptrace --placement single

./build/src/umbp/umbp_tier_bench replay \
  --trace workload.umbptrace --placement weighted \
  --backends-per-peer 2 --placement-weights 1,3
```

Use `umbp_tier_bench <subcommand> --help` for the authoritative option list.
Important workload controls include the profile, seed, operation/key counts,
value-size distribution, read ratio, client and batch counts, QPS, hot-set
fraction, and Zipf exponent. Topology controls include tier, per-backend
capacity, page size, backend count and weights. Routing controls select the
Master put/get policies and node affinity.

`--duration-sec` can derive the operation count from QPS. Eviction experiments
can set the Master high/low watermarks, check interval, lease duration, and
victim batch size explicitly.

Use `--metrics-port PORT` to enable the in-process Master's Prometheus endpoint
and include Master routing, RPC, live-key, and tier-capacity series.

Replay defaults to the relative timestamps in the trace. `--time-scale` scales
the intervals; max-throughput mode ignores them while retaining operation order
within each client stream. Different client streams execute concurrently.

## Synthetic profiles

- `sequential`: write-only traversal of the key space.
- `uniform`: uniformly selected keys with the configured read ratio.
- `hotset`: Zipf-distributed access within the configured hot set.
- `read-after-write`: alternating PUT and GET for the same key.
- `mixed`: deterministic key traversal with the configured read ratio.
- `capacity-pressure`: a write-heavy traversal intended to exceed configured
  backend capacity and exercise fallback/eviction.

All random choices and generated values are derived from the trace seed.
Running the same trace against another policy therefore changes management
decisions, not traffic.

UMBP keys are immutable: a repeated logical write creates a new versioned key,
and subsequent reads of that logical key target the latest version. Every
logical key's PUT/GET chain stays on one client stream, so concurrent replay
cannot reorder a dependent GET ahead of its PUT. The capacity-pressure profile
uses a non-repeating, write-only key stream.

## Trace contract

The file starts with an explicit `UMBPTRCE` envelope and envelope version,
followed by length-delimited protobuf records defined in
`src/umbp/distributed/proto/umbp_workload.proto`:

1. one `WorkloadTraceHeader`;
2. zero or more `WorkloadEvent` records.

The header carries the schema version, nanosecond time unit, payload seed, and
generation/topology metadata. Each event carries relative time, client id,
operation id, PUT/GET kind, key, logical value size, and batch id.

Existing protobuf field numbers and meanings are immutable. Evolution appends
optional fields and increments the schema only when an older reader cannot
safely preserve semantics. Readers reject unsupported versions, malformed or
truncated records, invalid operations, and records above configured bounds.

Value bytes are deliberately absent. PUT payloads are generated
deterministically from `(key, operation_id, seed)`, and a GET can validate the
same sequence. This keeps traces compact and avoids recording model data. A
future recorder must associate a GET with the operation id of the PUT whose
value it expects when payload validation is desired; otherwise replay can
disable content validation and still preserve management traffic.

The live adapter flushes heartbeat events after successful PUT batches. A GET
for a key produced by the same run may wait up to
`--publication-timeout-ms` for Master visibility; this prevents asynchronous
heartbeat timing from being mislabeled as a storage miss while retaining that
publication delay in GET latency. Set the timeout to zero for strict trace-time
miss behavior.

## Results

The first CSV section is the workload summary and includes reproducibility
settings, wall time, operation and byte counts, success/failure counts, GET
misses, validation failures, throughput, and p50/p95/p99/max latency and
scheduling lag.

Set `--window-ms` to emit an additional time-window section with operation,
byte, failure, and throughput counters for long-running pressure tests.

After the measured window and heartbeat settle, the `backend_placement` section reports
each node/backend's tier, owned key count, total/available capacity, and maximum
single allocation. Placement inspection is intentionally outside the timed
region and uses each backend's side-effect-free counters instead of RouteGet,
which would update leases and access time.

When Master metrics are enabled, the `master_metrics` section reports baseline, final,
and measured-window delta values for every `mori_umbp_*` series. Gauge deltas
are literal; use the final value for steady-state capacity comparisons.

For policy comparisons, retain the trace and change only topology/policy
arguments. Keep the registration settle interval, heartbeat environment, and
hardware configuration identical.

## Future real-traffic recording

The recorder should wrap the public UMBP client boundary and emit the same
`WorkloadEvent` stream:

- capture request issue time relative to recorder start;
- assign a stable logical client id;
- record PUT/GET, key, value size, operation id, and batch id;
- never copy value payloads into the trace;
- write through `TraceWriter`, preferably from a bounded asynchronous queue.

No replay changes are required: the recorded file is consumed by
`TraceWorkloadSource` and executed by the same `WorkloadRunner` used for
synthetic traffic.
