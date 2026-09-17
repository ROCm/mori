# Router Prefetch Testing

This document covers validation of proactive UMBP-to-UMBP KV prefetch.
It deliberately separates single-host automated tests from multi-host
integration tests:

- The single-host tests are checked-in GoogleTest cases and are suitable for
  regular development and CI.
- A two-host run validates real NIC routing, peer discovery, and RDMA. It is an
  integration/E2E test, not a unit test.

## What must be validated

The complete behavior is:

1. A caller submits object keys with `SubmitPrefetch`.
2. The target UMBP returns a request ID and executes the request asynchronously.
3. The target checks local placement, asks Master for missing-key routes, and
   fetches directly from the source UMBP.
4. Successful objects are committed only to the hot DRAM entry tier.
5. The target applies a bounded read lease.
6. `GetPrefetchStatus` reaches `READY`, `PARTIAL`, `FAILED`, or `EXPIRED`.
7. A subsequent target-side Get is served locally and returns the original
   bytes.

## Build

Initialize the required submodules and configure a focused UMBP test build:

```bash
git submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c

cmake -S . -B build-prefetch \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_UMBP=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_MPI=OFF \
  -DBUILD_APPLICATION=ON \
  -DBUILD_SHMEM=OFF \
  -DBUILD_CCO=OFF \
  -DBUILD_OPS=OFF \
  -DBUILD_COLLECTIVE=OFF \
  -DBUILD_PYBINDS=OFF \
  -DBUILD_METRICS=ON \
  -DUSE_SPDK=OFF

cmake --build build-prefetch \
  --target test_umbp_pool_client_ranges test_standalone_shm_ipc -j
```

The build host or container must provide ROCm, gRPC C++ development headers,
Protobuf, and a working MORI IO/RDMA build environment.

## Single-host automated tests

### Focused prefetch tests

```bash
build-prefetch/tests/cpp/umbp/distributed/test_umbp_pool_client_ranges \
  --gtest_filter='PoolClientRangesTest.ExplicitBatchPrefetch*'
```

These cases validate:

- remote object materialization and byte correctness;
- expired requests do not start a remote fetch;
- duplicate keys preserve one result per input entry;
- placement goes to the hot DRAM entry tier;
- the prefetch lease protects the object and expires;
- a stale source route is excluded and rerouted once.

Expected result:

```text
[  PASSED  ] 3 tests.
```

### Async standalone RPC test

```bash
build-prefetch/tests/cpp/umbp/distributed/test_standalone_shm_ipc \
  --gtest_filter='StandaloneShmIpcTest.WorkerRegistrationUsesNonZeroOffsetsAndCanReregister'
```

In addition to shared-memory Put/Get coverage, this case validates:

- synchronous `BatchPrefetch`;
- asynchronous `SubmitPrefetch`;
- `QUEUED`/`RUNNING` polling through `GetPrefetchStatus`;
- terminal `PARTIAL` results for one local and one missing key;
- idempotent resubmission of the same request ID and payload.

Expected result:

```text
[  PASSED  ] 1 test.
```

### Full affected suites

Run both binaries without a filter:

```bash
build-prefetch/tests/cpp/umbp/distributed/test_umbp_pool_client_ranges
build-prefetch/tests/cpp/umbp/distributed/test_standalone_shm_ipc
```

At commit `8fc0523`, the expected result is:

```text
PoolClientRangesTest: 29 passed
StandaloneShmIpcTest: 11 passed, 1 conditional test skipped
```

The conditional test is intentionally skipped unless
`UMBP_KEY_HANDLE_SLOTS=0` is set.

## Two-host integration test

### Purpose

The single-host fixture uses multiple UMBP processes/services but cannot prove:

- cross-host NIC and address selection;
- peer gRPC reachability across hosts;
- remote MORI IO/RDMA connection setup;
- actual source-host to target-host byte transfer.

Use two exclusive hosts:

- Host A: Master plus source UMBP.
- Host B: target UMBP plus the prefetch test driver.

Both hosts must run the same build from the same commit.

### Network and ports

Choose routable fabric IPs, not loopback or management-only addresses:

```bash
A_IP=<source-fabric-ip>
B_IP=<target-fabric-ip>
MASTER=${A_IP}:15558
```

Use distinct ports:

```text
Master:             A_IP:15558
Source peer:        A_IP:18081
Source IO engine:   A_IP:18091
Target peer:        B_IP:18082
Target IO engine:   B_IP:18092
```

The standalone control sockets may remain host-local Unix sockets.

### Start Master on Host A

```bash
build-prefetch/src/umbp/umbp_master "${A_IP}:15558" 19091 \
  > master.log 2>&1 &
MASTER_PID=$!
```

### Start source UMBP on Host A

```bash
env \
  UMBP_MASTER_ADDRESS="${MASTER}" \
  UMBP_NODE_ID=prefetch-source \
  UMBP_NODE_ADDRESS="${A_IP}" \
  UMBP_IO_ENGINE_HOST="${A_IP}" \
  UMBP_IO_ENGINE_PORT=18091 \
  UMBP_PEER_SERVICE_PORT=18081 \
  UMBP_DISTRIBUTED_MEDIUM=DRAM \
  UMBP_DISTRIBUTED_DRAM_PAGE_SIZE=65536 \
  UMBP_DRAM_CAPACITY=$((4 * 1024 * 1024 * 1024)) \
  UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp-prefetch-source.sock \
  build-prefetch/src/umbp/umbp_standalone_server \
  > source.log 2>&1 &
SOURCE_PID=$!
```

### Start target UMBP on Host B

```bash
env \
  UMBP_MASTER_ADDRESS="${MASTER}" \
  UMBP_NODE_ID=prefetch-target \
  UMBP_NODE_ADDRESS="${B_IP}" \
  UMBP_IO_ENGINE_HOST="${B_IP}" \
  UMBP_IO_ENGINE_PORT=18092 \
  UMBP_PEER_SERVICE_PORT=18082 \
  UMBP_DISTRIBUTED_MEDIUM=DRAM \
  UMBP_DISTRIBUTED_DRAM_PAGE_SIZE=65536 \
  UMBP_DRAM_CAPACITY=$((4 * 1024 * 1024 * 1024)) \
  UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp-prefetch-target.sock \
  build-prefetch/src/umbp/umbp_standalone_server \
  > target.log 2>&1 &
TARGET_PID=$!
```

Wait until both servers report that they are ready and Master has registered
both node IDs.

### Driver operations

The driver runs these operations in order:

1. Through the source Unix socket, Put a deterministic object under
   `cross-node-prefetch-key`.
2. Force or wait for the source heartbeat so Master indexes the key.
3. Confirm the target does not contain the key locally.
4. Through the target Unix socket, call `SubmitPrefetch` with:

```text
request_id = cross-node-prefetch-1
keys = [cross-node-prefetch-key]
timeout_ms = 5000
lease_ttl_ms = 2000
```

5. Poll `GetPrefetchStatus` until terminal.
6. Require state `READY` and per-key result `true`.
7. Get the object through the target socket and compare every byte with the
   source payload.
8. Repeat target Get and confirm no additional remote-fetch bytes are recorded.

### Automated TP8 cross-host case

Build the manual two-process test binary:

```bash
cmake --build build-prefetch --target test_umbp_prefetch_tp_cross_host -j
```

Launch one copy on each host with the same shared `SYNC_DIR`. The source and
target need distinct node addresses, peer ports, and IO ports:

```bash
# Host A
UMBP_PREFETCH_TEST_ROLE=source \
UMBP_PREFETCH_TEST_SYNC_DIR="${SYNC_DIR}" \
UMBP_PREFETCH_TEST_MASTER_ADDRESS="${MASTER}" \
UMBP_PREFETCH_TEST_NODE_ADDRESS="${A_IP}" \
UMBP_PREFETCH_TEST_PEER_PORT=18081 \
UMBP_PREFETCH_TEST_IO_PORT=18091 \
build-prefetch/tests/cpp/umbp/distributed/test_umbp_prefetch_tp_cross_host \
  --gtest_filter=CrossHostPrefetch.Tp8BatchPrefetch

# Host B, launched concurrently
UMBP_PREFETCH_TEST_ROLE=target \
UMBP_PREFETCH_TEST_SYNC_DIR="${SYNC_DIR}" \
UMBP_PREFETCH_TEST_MASTER_ADDRESS="${MASTER}" \
UMBP_PREFETCH_TEST_NODE_ADDRESS="${B_IP}" \
UMBP_PREFETCH_TEST_PEER_PORT=18082 \
UMBP_PREFETCH_TEST_IO_PORT=18092 \
build-prefetch/tests/cpp/umbp/distributed/test_umbp_prefetch_tp_cross_host \
  --gtest_filter=CrossHostPrefetch.Tp8BatchPrefetch
```

The source publishes eight deterministic TP shard keys. The target prefetches
all eight into local DRAM, then asks the source process to shut down before
calling `BatchGet`. Passing therefore proves that every shard crossed hosts,
was materialized locally, and retained byte-for-byte correctness. This case
tests `PoolClient::BatchPrefetch`; the standalone async request/status RPC is
covered separately by `test_standalone_shm_ipc`.

### Pass criteria

All of the following are required:

- Master reports both live nodes and routes the key to Host A before prefetch.
- Target status reaches `READY` before its deadline.
- The target stores the key in DRAM, not SSD/cold storage.
- Source and target payloads are byte-for-byte identical.
- Logs show a peer connection from Host B to Host A.
- Transfer metrics show remote inbound bytes on Host B for the prefetch.
- The follow-up target Get is local and does not add another remote transfer.
- All processes exit cleanly and no pending slot remains.

### Negative cases

After the basic two-host case passes, run:

- missing key: terminal `FAILED`;
- deadline already exhausted: terminal `EXPIRED`;
- duplicate request ID and identical payload: idempotent response;
- duplicate request ID with different payload: rejected;
- source stopped after Master routing: failed cleanly when no other replica
  exists;
- hot DRAM full: corresponding key fails rather than spilling to cold storage.

Stale-source rerouting requires another live replica. This can be tested with a
third UMBP process, either on a third host or as a second source process with
distinct node and port identities.

### Cleanup

```bash
kill "${TARGET_PID}" "${SOURCE_PID}" "${MASTER_PID}"
wait "${TARGET_PID}" "${SOURCE_PID}" "${MASTER_PID}" || true
rm -f /tmp/umbp-prefetch-source.sock /tmp/umbp-prefetch-target.sock
```

Always release exclusive scheduler allocations after logs and metrics have
been collected.
