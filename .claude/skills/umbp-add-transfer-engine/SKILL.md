---
name: umbp-add-transfer-engine
description: >-
  Add a new byte-moving path to UMBP's distributed mode as a TransferEngine
  (GPU copy, a second RDMA transport, a file/GDS engine, a fake for tests).
  Covers CanHandle pair-dispatch and why engines must be disjoint, the
  Plan/Submit/Wait split, bounce buffers, PeerDirectory for transports that need
  a handshake, and how to extend TransferRef for a non-memory endpoint. Use when
  the user wants to add or debug a transfer engine in src/umbp/distributed/
  transfer, mentions TransferEngine, CanHandle, TransferRef, MoriIoEngine,
  LocalCopyEngine or CompositeTransferEngine, or asks why a transfer is
  "unplannable" / no engine claimed a pair.
---

# Adding a transfer engine to UMBP distributed mode

**There is exactly one byte-moving path in the system and it is the transfer
layer.** A backend owns bytes and publishes endpoints; an engine decides how to
move bytes between two endpoints. Admitting a second byte-moving path at the
backend level is the coupling the whole backend-agnostic refactor exists to
remove.

The companion skill `umbp-add-backend` covers the other side of that boundary.

## The one idea: selection is a function of the PAIR

Never of either endpoint alone. This is the rule that makes everything else
work, and the most common way to get an engine wrong is to write `CanHandle` as
a property of `src` or `dst` in isolation.

The engines in tree partition the `(src, dst)` space with **no overlap**:

| pair | engine |
|---|---|
| both local, both `loc == CPU` | `LocalCopyEngine` (NT-AVX2 memcpy) |
| both local, either `loc == GPU` | `HbmCopyEngine` (`hipMemcpy`) |
| exactly one endpoint remote | `MoriIoEngine` (RDMA via mori-io) |

Because they are disjoint, `CompositeTransferEngine`'s first-match registration
order is *documentation*, not a tie-break. **Keep it that way.** If two engines
can both claim a pair, registration order silently becomes a performance
decision that nothing tests. Assert disjointness:

```cpp
EXPECT_TRUE(local.CanHandle(h1, h2));
EXPECT_FALSE(hbm.CanHandle(h1, h2));   // both-CPU stays with the NT-AVX2 path
```

### Finding the gap you are filling

Before writing an engine, prove no existing one claims your pair. Read the
`CanHandle` bodies — they are short and each has a hard precondition:

- `local_copy_engine.cpp`: `src.HasHostPtr() && dst.HasHostPtr() && both loc == CPU`
- `mori_io_engine.cpp`: `if (src_remote == dst_remote) return false;` — refuses
  both-local *and* both-remote

That second line is why `HbmCopyEngine` had to exist: a local pair with a GPU
endpoint was claimed by nobody, so an HBM backend's local Put/Get could not
complete at all. Write that as your first test — it is the engine's reason for
existing, and it should be asserted against the real composite rather than by
inspection.

## `TransferRef`: what an endpoint is

```cpp
struct TransferRef {
  void* host_ptr;                    // process-local view; null for a peer's memory
  uint64_t size;
  mori::io::MemoryLocationType loc;  // CPU | GPU
  int device;
  mori::io::MemoryDesc mem;          // mori-io registration; .size == 0 if none
};
```

**`host_ptr` means "addressable in this process", not "host memory".** For an
`hipMalloc`'d buffer the *device* pointer is that view, since it is what
`hipMemcpy` accepts. So `HbmCopyEngine` needed no new field — `loc = GPU` plus
the existing pointer was enough. Check whether that is true for you before
adding anything.

It is deliberately **not** a `std::variant`. Registration fans *out*: the same
buffer is a raw pointer **and** an RDMA MR at the same time, and which one is
used is a property of the pair. A variant makes the both-local-but-also-
registered case inexpressible.

Empty handles are how an endpoint says "not reachable that way":
`host_ptr == nullptr` → not addressable here; `mem.size == 0` → not registered
with mori-io.

### If you genuinely need a non-memory endpoint

A file or object endpoint needs a `kind` tag plus a second handle set in
`transfer_engine.h` — a one-file change, and `CanHandle` already exists to route
it. But note what comes with it: `CompositeTransferEngine` does **not** implement
chaining, where no single engine spans the pair and the layer composes two hops
through a bounce buffer (a file on node A, a reader on node B). You would be
adding that too. `SsdBackend` sidesteps the whole question by staging through
registered host memory — see `umbp-add-backend`.

## The interface

```cpp
class TransferEngine : public MemoryRegistrar {
  virtual const char* Name() const = 0;
  virtual TransferRef RegisterMemory(void*, size_t, MemoryLocationType, int device) = 0;
  virtual void Deregister(const TransferRef&) = 0;
  virtual bool CanHandle(const TransferRef& src, const TransferRef& dst) const = 0;
  virtual TransferPlanSet Plan(const std::vector<TransferItem>&) const = 0;
  virtual std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan>) = 0;
  bool Transfer(const std::vector<TransferItem>&, std::vector<size_t>* failed_tags);
};
```

### `RegisterMemory`

If your engine needs no pinning, return the endpoint unchanged:

```cpp
TransferRef RegisterMemory(void* base, size_t size, MemoryLocationType loc, int dev) override {
  return TransferRef::HostBytes(base, size, loc, dev);
}
void Deregister(const TransferRef&) override {}
```

`CompositeTransferEngine` fans registration out to **every** sub-engine and
merges the handles, so a buffer can be a raw pointer and an MR at once. Your
engine seeing a buffer it cannot pin is normal, not an error.

### `Plan` — pure, and virtual for a reason

`Plan` must issue no IO and change no engine state, so a caller may plan on one
thread and submit on another. It is virtual rather than a base helper because
grouping is engine-specific: pair-grouping is an RDMA optimization (it cuts CQE
and post count), a memcpy engine gains little, and a file engine would rather
group by file and merge adjacent offset ranges.

Two things every `Plan` must do:

1. **Reject what you cannot carry**, into `rejected_tags`. This is not an error
   path the caller can ignore — those items were dropped and their keys must be
   failed. `PoolClient` logs them as "transfer unplannable (no engine claimed the
   endpoints)".

2. **Bounds-check once, here**, not per copy:

   ```cpp
   if (item.size > item.src.size || item.src_offset > item.src.size - item.size ||
       item.size > item.dst.size || item.dst_offset > item.dst.size - item.size) {
     out.rejected_tags.push_back(item.tag);
     continue;
   }
   ```

   Written that way to avoid overflow. A page index that does not fit its buffer
   is a backend bookkeeping bug, and the key must be failed rather than copied
   past the end — on a GPU endpoint an overrun is not a segfault but silent
   corruption of whatever else is in the pool.

Coalescing adjacent segments is worth it if your per-call cost is non-trivial:
`HbmCopyEngine` coalesces because each `hipMemcpy` carries launch overhead a
`memcpy` does not.

`tag` is the caller's per-entry index. It must survive grouping so a failed plan
maps back to the keys that rode in it; de-duplicate as you append.

### `Submit` — and the one blocking exception

`Submit` does **not** block on the wire, with one documented exception: a plan
that needs the engine's **bounce buffer** completes *inside* `Submit` and comes
back already settled. That keeps the staging lock from ever being held across a
return, which is what makes a submit-all-then-wait loop over several peers
deadlock-free without the caller knowing staging exists.

If your engine completes inline (any synchronous copy), return an
already-settled handle that just replays the outcome:

```cpp
class SettledHandle final : public TransferHandle {
  void Wait(std::vector<TransferFailure>* failures) override {
    if (reported_) return;              // idempotent: a second call appends nothing
    reported_ = true;
    for (auto& f : failures_) failures->push_back(std::move(f));
    failures_.clear();
  }
};
```

Return `nullptr` if nothing was posted — the caller must then treat every tag as
failed.

**Failures are per PLAN, not per segment.** The plan is the unit a tag set maps
back to, and a partially-copied key is failed wholesale anyway. Fill in
`TransferFailure{tags, code, message, endpoint}`; `endpoint` is for diagnosis.

### `Wait` lives on the handle, not the engine

Because mori-io's RDMA backend keeps raw `TransferStatus*` into the handle's
status vector: the handle must outlive the post and must never move its
statuses. Its destructor drains as a **safety net** for an early or exceptional
destroy, so a completion callback never writes freed memory. `Wait` never breaks
early — every plan is waited, or a status is left live.

## Threading and ambient state

`Submit` is called concurrently from `PoolClient`'s
`UMBP_DRAM_{READ,WRITE}_THREADS` executor threads. Two consequences:

- Shared mutable engine state needs its own synchronization. `HbmCopyEngine`
  avoids a shared HIP stream for exactly this reason — the parallelism that
  matters is *across keys* and already exists above the engine, so a stream
  would add synchronization with no matching win at KV-block sizes.
- **Restore any ambient state you touch.** Those threads are not yours to leave
  re-pointed at another GPU:

  ```cpp
  int entry_device = -1;
  if (hipGetDevice(&entry_device) != hipSuccess) entry_device = -1;
  // ... hipSetDevice(want) per plan ...
  if (touched && entry_device >= 0) (void)hipSetDevice(entry_device);
  ```

## Transports that need a handshake: `PeerDirectory`

Some transports cannot address a peer until told about it — mori-io needs the
peer's `EngineDesc` registered and its published `MemoryDesc`s unpacked before a
byte moves. That handshake is not a transfer, so it is **not** on
`TransferEngine`; it is its own interface, implemented by the engines that need
it. `PoolClient` drives peer setup through a `PeerDirectory*` obtained at
composition time, so a second remote transport implements the interface and
`pool_client.cpp` does not change. A local copy engine has no peers and
implements nothing.

Note "directory", not "cache": these calls are the authority on what this
process knows about a peer, and `ForgetRemote` is how a failed handshake is
undone.

## Wiring it in

`PoolClient::Init` is the composition root and the only place any concrete
engine is named:

```cpp
auto composite = std::make_unique<CompositeTransferEngine>();
composite->AddEngine(std::make_unique<LocalCopyEngine>());
composite->AddEngine(std::make_unique<HbmCopyEngine>());
if (!config_.io_engine.host.empty()) {
  auto rdma = std::make_unique<MoriIoEngine>(...);
  if (!rdma->Init()) { /* fail */ }
  peer_directory_ = rdma.get();
  composite->AddEngine(std::move(rdma));
}
transfer_engine_ = std::move(composite);
```

Local engines go **before** the wire engine on principle: a pair that reached
mori-io wrongly would be refused rather than served slowly. Add your source to
`src/umbp/CMakeLists.txt` under `umbp_common`.

New external dependencies are usually already transitively available —
`umbp_common` links `mori_io`, which links `hip::host`, which is why
`HbmCopyEngine` could call `hipMemcpy` with no CMake change beyond the source
listing.

That `AddEngine` call is also all the observability wiring there is.
`CompositeTransferEngine` is the dispatch point, so it is the measurement point:
it charges bytes, plan counts, failures and in-flight time to whichever engine
carried each plan and publishes them under `engine=<your Name()>,
direction=push|pull|local`. Your engine appears in the transfer panels of
`examples/monitoring/grafana/dashboards/umbp_backends.json` without a line of
metrics code.

Override `SampleMetrics()` (from `MetricSource`) only for transport-internal
state the dispatcher cannot see — bounce-pool pressure, queue depth. Name the
metric generically and put the specifics in a label, the way a storage backend
does; `engine=` is stamped for you. See
`umbp/distributed/metrics/component_metrics.h`.

## Testing

Split the suite the way the layer splits:

**Selection and planning need no hardware and should always run.** Cover:
the gap your engine fills (assert the *other* engines refuse the pair);
disjointness; composite routing via `SelectEngine`, which is exposed for exactly
this; coalescing; out-of-bounds rejection; and rejection of pairs you cannot
carry.

```cpp
TransferEngine* chosen = composite.SelectEngine(host_ref, gpu_ref);
ASSERT_NE(chosen, nullptr);
EXPECT_STREQ(chosen->Name(), "HbmCopyEngine");
```

**Byte movement needs the hardware**, so `GTEST_SKIP()` without it. Use
`Transfer()` (plan+submit+wait) for round-trip tests — it is there for callers
with nothing to overlap, and tests are such a caller.

A trap worth knowing: when building a "scattered pages" test, a stride that is
not coprime with the page count is not a permutation and silently leaves pages
uncopied — `(i * 2) % 4` gives `{0,2,0,2}`. That failure looks like an engine
bug and is not.

`MoriIoEngine` is not unit-testable (needs a real fabric); it is covered
end-to-end by `test_cross_node_smoke`, which is labeled `integration`.

## Build and run

The umbp build needs `protoc`/`grpc_cpp_plugin` from the mori Docker image, not
the bare host:

```bash
docker exec <mori-container> bash -c "cd <repo> && ninja -C build && \
  cd build && ctest -E '^cco_' -LE integration"
```

## Reference

- `transfer/transfer_engine.h` — `TransferRef`, `MemoryRegistrar`,
  `PeerDirectory`, `TransferItem/Plan/Handle`, `TransferEngine`; also records
  what was deliberately left out and why
- `transfer/local_copy_engine.{h,cpp}` — the simplest complete engine
- `transfer/hbm_copy_engine.{h,cpp}` — a second engine closing a real gap
- `transfer/mori_io_engine.{h,cpp}` — RDMA, bounce buffers, `PeerDirectory`
- `transfer/composite_transfer_engine.{h,cpp}` — fan-out registration,
  per-pair dispatch; the "no chaining" note is here
- `doc/design-backend-agnostic-refactor.md` §4
