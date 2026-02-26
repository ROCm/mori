# MORI-IO TCP Backend Design

> **Goal**: Provide a high-reliability, high-performance **TCP fallback transport** for distributed KV-Cache transfer when RDMA/XGMI is unavailable.  
> The TCP backend must **simulate RDMA-like Read/Write semantics** on top of TCP byte streams, while keeping the public `mori::io::Backend` / `BackendSession` APIs aligned with existing backends.

---

## 1. Requirements Mapping

### 1) Semantic Mapping (RDMA-like ops on TCP stream)
TCP is a byte stream; MORI-IO upper layers are op/message oriented (Read/Write/BatchRead/BatchWrite).

**Solution**: a framed protocol:
- **Control channel** carries op metadata (`WriteReq`, `ReadReq`, `Batch*Req`) and `Completion`.
- **Data channel** carries payload with a lightweight data frame header (`op_id`, `payload_len`).

This avoids classic TCP sticky/half packet issues by always parsing `FixedHeader + VariableBody`.

### 2) Concurrency Model (non-blocking + multiplexing)
**Baseline**: `epoll`-based Reactor (single IO thread per backend instance).
- Non-blocking sockets, edge-triggered `epoll`.
- `eventfd` used to wake the reactor when new outbound ops are submitted from application threads.
- Optional `timerfd` for op timeout scanning & housekeeping.

**Evaluation**: Kernel is 5.15+ (io_uring-capable). The design keeps a clean boundary so `io_uring` Proactor can replace the Reactor later without changing public APIs.

### 3) Zero-Copy & Memory Management
#### CPU memory
- Outbound: `sendmsg` + `iovec` directly referencing user registered buffers (no extra user-space copy).
- Inbound: `recv` directly into destination user buffer.
- Optional: enable `MSG_ZEROCOPY` for large CPU payloads (best-effort).

#### GPU memory (fallback path)
TCP cannot directly read/write device VRAM. We stage through pinned host buffers:
- GPU→Host (`hipMemcpyDtoHAsync`) before sending payload.
- Host→GPU (`hipMemcpyHtoDAsync`) after receiving payload.
- Pinned host buffers are pooled to avoid per-op allocations.

### 4) Latency vs Throughput
- **Metadata**: control channel uses `TCP_NODELAY` for low latency.
- **Payload**: data channel uses large `SO_SNDBUF/SO_RCVBUF`, and can optionally enable `MSG_ZEROCOPY` for large CPU payloads.
- Backpressure: bound in-flight bytes per peer to avoid unbounded buffering.

### 5) Robustness
- NIC selection: bind outbound sockets to `IOEngineConfig.host` before connect; listen on the same host.
- Keep-alive: enable `SO_KEEPALIVE` + reasonable keepalive parameters.
- Disconnect: fail in-flight ops with `ERR_BAD_STATE` and allow lazy reconnect on next submission.
- Timeout: per-op timeout (configurable) -> fail op and clean state.

---

## 2. Architecture

### 2.1 Components

- `TcpBackend` (`mori::io::Backend`)
  - Owns one `TcpTransport` instance.
  - Implements `RegisterRemoteEngine`, `RegisterMemory`, `Read/Write/Batch*`, `CreateSession`.

- `TcpBackendSession` (`mori::io::BackendSession`)
  - Caches local/remote descriptors and routes ops to `TcpTransport` (low overhead fast path).

- `TcpTransport`
  - A single IO thread (reactor) managing:
    - Listener socket
    - Per-peer connections (control + data)
    - Framed parsing / send queue
    - Pending outbound op table
    - Inbound op handling and inbound status queue

- `PeerConn`
  - `ctrl_fd` + `data_fd`
  - control parser state, data parser state
  - outbound queues for ctrl and data

- `PinnedStagingPool`
  - Reusable pinned host buffers keyed by `(device_id, size_class)` (GPU staging).

### 2.2 Threading

Single IO thread per `TcpBackend`:
- No thread-per-connection.
- All network IO and protocol parsing happens in this thread.
- Application threads only enqueue op descriptors (cheap) and return immediately with `TransferStatus = IN_PROGRESS`.

---

## 3. Wire Protocol

### 3.1 Channels
For each peer engine we maintain **two TCP connections**:
- `CTRL`: op metadata + completions
- `DATA`: bulk payload frames

Both connections connect to the same listen port; the first message is a `HELLO` identifying:
- protocol version
- channel type (`CTRL` / `DATA`)
- `EngineKey` string

### 3.2 Control Frame
All ctrl messages are framed:

```
| CtrlHeader (fixed) | CtrlBody (variable, body_len bytes) |
```

`CtrlHeader` includes:
- magic/version
- message type
- body length

`CtrlBody` contains type-specific fields (op_id, mem_id, offsets, sizes, etc).

### 3.3 Data Frame

```
| DataHeader (fixed) | Payload (payload_len bytes) |
```

`DataHeader` includes:
- `op_id`
- `payload_len`

Ordering:
- Write: `CTRL WriteReq` then `DATA (op_id,payload)` then `CTRL Completion`
- Read: `CTRL ReadReq` then `DATA (op_id,payload)` then `CTRL Completion`

Receiver keeps op state keyed by `op_id` so ctrl/data can be processed independently.

#### Cross-channel reordering (CTRL vs DATA)
Because **CTRL** and **DATA** are separate TCP connections, the receiver may observe **DATA arriving before the corresponding CTRL request** (no global ordering across connections).

To preserve RDMA-like op semantics without adding an extra RTT handshake, the TCP backend must:
- Buffer such early DATA frames by `(peer_key, op_id)` into pinned host memory.
- Finalize the write once the CTRL request arrives (or fail/cleanup on disconnect/timeout).

For GPU destinations this maps naturally to the existing pinned-staging path; for CPU destinations this fallback path may add an extra copy only in the reordering case.

---

## 4. API Semantics

- `Write(local, remote, ...)`:
  - initiator sends payload; target writes into its local memory region.
- `Read(local, remote, ...)`:
  - initiator requests; target reads from its local memory region and sends payload back.
- `PopInboundTransferStatus(remote_key, id, ...)`:
  - target side may pop completion of inbound ops (mainly useful for write-like semantics), consistent with existing RDMA notification API.

---

## 5. Config Surface (planned)

`TcpBackendConfig`:
- `num_io_threads` (default 1)
- `max_inflight_bytes_per_peer`
- `op_timeout_ms`
- `sock_sndbuf_bytes`, `sock_rcvbuf_bytes`
- `enable_zerocopy`, `zerocopy_threshold_bytes`
- `enable_keepalive`, keepalive parameters

---

## 6. Planned Code Changes

### New files
- `src/io/tcp/backend_impl.hpp`
- `src/io/tcp/backend_impl.cpp`
- `src/io/tcp/protocol.hpp`
- `src/io/tcp/protocol.cpp`
- `src/io/tcp/poller_epoll.hpp` (reactor)
- `src/io/tcp/pinned_staging_pool.hpp`

### Modified files
- `include/mori/io/backend.hpp` (add `TcpBackendConfig`)
- `src/io/engine.cpp` (wire in `BackendType::TCP`)
- `src/io/CMakeLists.txt` (build TCP backend sources)
- `src/pybind/mori.cpp` (bind `TcpBackendConfig`)
- `python/mori/io/engine.py` (enable `BackendType.TCP`)
- `tests/python/io/benchmark.py` (support `--backend tcp` and `--op` alias)

---

## 7. Known Limits / Future Work

- `io_uring` Proactor implementation (replace epoll reactor in `TcpTransport`).
- Adaptive multi-connection striping per peer to improve throughput on high-BDP networks.
- Full `MSG_ZEROCOPY` completion accounting + buffer lifetime tracking (currently best-effort).
- Optional TLS / auth for multi-tenant environments.
