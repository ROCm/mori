---
name: umbp-add-backend
description: >-
  Add a new storage medium to UMBP's distributed mode as a MediumBackend (HBM,
  SSD, CXL, a remote object store, a fake for tests). Covers the two shapes a
  backend can take — a PageMemorySource for anything page-addressable, or a full
  MediumBackend for a medium whose bytes are not directly addressable — plus
  registration, the heartbeat/event contract, and the failure modes that only
  show up under the peer service. Use when the user wants to add or debug a
  storage tier/medium/backend in src/umbp/distributed, mentions MediumBackend,
  PageBackend, PageMemorySource or BackendRegistry, or asks why a new tier's
  puts or gets are not working.
---

# Adding a storage medium to UMBP distributed mode

A **backend owns bytes and publishes descriptors for them. It does not move
them.** Everything in this skill follows from that one rule, which is stated at
the top of `src/umbp/include/umbp/distributed/peer/backend/medium_backend.h`
and enforced by the type system since Phase 6 (`Init` receives a
`MemoryRegistrar`, which has no `Submit` on it).

If you find yourself wanting to copy bytes inside a backend, stop — that is the
transfer layer's job, and the sibling skill `umbp-add-transfer-engine` covers
it.

## First decide which of the two shapes you need

This is the whole design decision, and getting it wrong costs a rewrite.

**Is your medium page-addressable — can you hand out a raw pointer that the
transfer layer can read and write directly?**

| | Yes (DRAM, HBM, CXL, a pmem mapping) | No (SSD, S3, a network filesystem) |
|---|---|---|
| Implement | `PageMemorySource` — **5 methods** | `MediumBackend` — 23 methods |
| Reuse | All of `PageBackend` | Nothing; but see "staging" below |
| Effort | ~100 lines | ~450 lines |
| Example | `hbm_backend.{h,cpp}` | `ssd_backend.{h,cpp}` |

Most media are the first case. `PageBackend`'s slot lifecycle, bitmap
allocator, event outbox, reaper, read leases and copy pins **never consult the
tier**, so a second paged medium needs no second copy of any of it.

---

## Shape 1: a paged medium (implement `PageMemorySource`)

Five methods, in `page_backend.h`:

```cpp
class PageMemorySource {
  virtual bool Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) = 0;
  virtual void Release() = 0;
  virtual mori::io::MemoryLocationType LocationType() const = 0;
  virtual int Device() const = 0;
  virtual const char* Name() const = 0;
};
```

`LocationType()` and `Device()` are **the facts a descriptor cannot recover**.
They are the reason this interface exists: `PageBackend` mirrors them into every
`TransferRef` it publishes, and that is what makes a local transfer against your
medium select the right engine. Get them wrong and nothing fails at build or
init time — a transfer just silently picks the wrong engine, or no engine at
all.

### The recipe

1. **Write the source.** Put it in its own file if it needs a new dependency
   (`hbm_backend.cpp` exists so HIP stays out of `page_backend.cpp`).

   ```cpp
   bool MyPageMemorySource::Allocate(const std::vector<uint64_t>& sizes,
                                     std::vector<Buffer>* out) {
     std::vector<MyHandle> taken;
     std::vector<Buffer> staged;
     for (uint64_t size : sizes) {
       if (size == 0) continue;              // skip, do not fail
       MyHandle h = MyAlloc(size);
       if (!h.valid()) {
         for (auto& t : taken) MyFree(t);    // unwind THIS call only
         return false;                        // leave `out` untouched
       }
       staged.push_back(Buffer{h.ptr, h.usable_size});
       taken.push_back(h);
     }
     handles_.insert(handles_.end(), taken.begin(), taken.end());
     out->insert(out->end(), staged.begin(), staged.end());
     return true;
   }
   ```

   Report the **usable** size in `Buffer::size`, not the requested one — host
   hugepage rounding makes the extra genuinely allocatable, and `PageBackend`
   publishes what you report.

2. **Add a factory** returning `std::unique_ptr<MediumBackend>`, next to
   `MakePageBackend` / `MakeHbmBackend`. It must return the *interface*: Phase 5
   Rule A says only `PoolClient::Init` may name a concrete backend.

   ```cpp
   std::unique_ptr<MediumBackend> MakeMyBackend(uint64_t page_size, ...) {
     return std::make_unique<PageBackend>(TierType::MY_TIER, page_size,
                                          std::make_unique<MyPageMemorySource>(...),
                                          std::move(buffer_sizes), pending_ttl,
                                          read_lease_ttl);
   }
   ```

3. **Add a config struct** in `distributed/config.h`, with `enabled = false` so
   existing deployments are bit-identical. Do **not** extend
   `DramOwnershipConfig` — hugepages/NUMA/prefault are meaningless for HBM, and
   a device ordinal is meaningless for host memory. Each medium brings its own
   knobs; that asymmetry is why the seam is a class and not an options struct.

4. **Register it in `PoolClient::Init`**, next to the DRAM block. Three lines,
   and this is the only file that changes outside your own:

   ```cpp
   if (config_.my.enabled && !config_.my.buffer_sizes.empty()) {
     auto backend = MakeMyBackend(page_size, ...);
     if (!backend->Init(static_cast<MemoryRegistrar*>(transfer_engine_.get()))) {
       MORI_UMBP_ERROR("[PoolClient] MY backend Init failed");
       initialized_ = false;
       return false;
     }
     registry_.Register(std::move(backend));
   }
   ```

5. **Add the tier to `TierType`** (`types.h`) if it is genuinely new. `HBM=1,
   DRAM=2, SSD=3` already exist. `BackendRegistry` is a `map<TierType, ...>`, so
   one instance == one medium and the enum value is the identity.

That is the whole change. Routing, the peer service, the heartbeat and the batch
executors were all written against `BackendRegistry` and need no edit.

---

## Shape 2: a medium whose bytes are not addressable

If you cannot hand out a pointer, you have two options, and the cheap one is
almost always right.

### Stage through registered host memory (what `SsdBackend` does)

Publish an ordinary registered host DRAM arena as your buffer, and move bytes
between it and your device **inside** the backend:

- `BatchAllocate` reserves a staging page and publishes it. The writer RDMAs
  into it knowing nothing about your medium.
- `BatchCommit` spills that page to your device and **returns the page to the
  arena**. The page is borrowed for the write, not the key's home.
- `BatchResolve` fills a staging page from your device and publishes it under a
  **read lease**; a reaper reclaims the page when the lease expires.

The cost is one host copy per side. The benefit is that your medium reaches the
data plane with **zero new transfer-layer concepts** — no new `TransferRef`
kind, no new engine, no chaining — and a remote peer reading from your node sees
a perfectly ordinary registered buffer and needs no code at all.

### Add a real endpoint kind (bigger; not yet done in tree)

`transfer_engine.h` reserves this: a `FileRef`/`ObjectRef` plus a `kind` tag, a
new engine, **and chaining in `CompositeTransferEngine`** for the remote reader
(device → bounce → wire), which that class explicitly does not implement. Take
this path only when you need zero-copy/GDS; staging does not block it.

### Things that bite in shape 2

- **Exhaustion has no honest encoding.** A `Resolve` that cannot get a staging
  page must report `found=false`, which makes the client exclude your node and
  retry elsewhere — wrong, because your node *does* hold the key.
  `medium_backend.h` records that the "not ready, retry here" state was proposed
  and rejected. Size the arena for read concurrency, and pin the behavior in a
  test so a future control-plane fix changes it deliberately.
- **Contiguity.** `PeerSsdManager::PrepareRead` takes one `(ptr, capacity)`, not
  a scatter list, so a staged key must fit one contiguous page. `SsdBackend`
  therefore allocates **one buffer** of `staging_pages * page_size` and refuses
  keys larger than a page. In distributed mode master's `page_size` *is* the KV
  block size, so 1 key == 1 page is the normal case.
- **Never hold the arena lock across IO.** Take the slot out under the lock,
  release it, do the spill/fill, then re-lock to return the page. Holding it
  stalls every concurrent `Allocate` and `Resolve`.

---

## The contracts that are easy to get wrong

These are the ones with no compile-time protection.

**Events go in ONE bundle under ONE seq.** The heartbeat concatenates every
backend's events (`DrainAllBackends`). Never emit one bundle or one seq per
medium — that breaks the ack / seq-gap full-sync recovery.

**`SnapshotOwnedKeysForFullSync` must clear the outbox in the SAME critical
section** as the snapshot. Two separate locks drop events committed in between.
`SnapshotOwnedKeys` is `const` and must not mutate; the full-sync variant is
not.

**`kFailedNoSpace` vs `kFailed`.** `kFailedNoSpace` means "medium exhausted,
retry on another peer". Use it only when another peer could plausibly succeed. A
key too large for your page size is `kFailed` — no peer would do better, and the
writer must not keep hunting.

**`Evict` returns one result per key, in request order.** The peer service sums
freed bytes for a key mirrored across media and relies on the positional match.
Return `bytes_freed = 0` (not an error) for unknown, already-freed, or protected
keys; master retries protected ones next round.

**`BatchAbort` is idempotent** — a slot already reaped or never seen still
reports `true`.

**`Shutdown` must tolerate a failed `Init`** and must not run concurrently with
anything else. Deregister **before** releasing memory: the registrar may still
hold an MR over those pages.

**`Init` is idempotent**, and a backend is fully live when it returns — start
your own reaper thread there, so no caller has to know you have one.

**Allocation is gated between `ClearLocal()` and `ClearFullSyncAcked()`.** No
new owned key may appear in that window.

**Threading**: every method may be called from the peer service's gRPC handler
threads and the heartbeat thread concurrently.

**`SetAutoFlushHook`'s callback runs under your lock** — it must be cheap and
must not re-enter the backend. Signal the heartbeat thread and return.

---

## Testing

Two levels, both worth having:

1. **Without the transfer layer** — drive `BatchAllocate` / `BatchCommit` /
   `BatchResolve` / `Evict` directly and assert the bookkeeping. A
   `LocalOnlyRegistrar` (returns `TransferRef::HostBytes`, counts
   register/deregister calls) is all the `MemoryRegistrar` you need; it is also
   exactly what `CompositeTransferEngine` degrades to on a node with no RDMA.

2. **Through a real `CompositeTransferEngine`** — a Put and a Get moving actual
   bytes. This is what catches a wrong `LocationType()`/`Device()`, because a
   mislabeled endpoint selects the wrong engine and the copy either fails or
   silently does nothing.

Assert the registration contract explicitly:

```cpp
EXPECT_EQ(registrar.last_loc, mori::io::MemoryLocationType::GPU);
EXPECT_EQ(backend->BufferRef(0).device, 0);
```

For a staging backend, the highest-value test is that **Commit returns the
page**: configure 2 staging pages, do 10 sequential puts, assert all succeed. A
leaked page wedges the backend after `staging_pages` puts and nothing else
catches it.

Skip rather than fail when hardware is absent (`if (!HaveGpu()) GTEST_SKIP()`),
so the suite stays runnable on a CPU-only box.

Register the test in `tests/cpp/umbp/distributed/CMakeLists.txt` with
`add_test(NAME ... COMMAND ...)`, **not** `gtest_discover_tests` — the file
explains why (discovery bakes in build-time cmake paths and goes red when the
image's ctest runs it).

## Building and running

The umbp C++ build needs `protoc` and `grpc_cpp_plugin`, which live in the mori
Docker image, not on the bare host:

```bash
docker exec <mori-container> bash -c "cd <repo> && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_UMBP=ON -DBUILD_TESTS=ON -DGPU_TARGETS=gfx950 \
    -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ && \
  ninja -C build && cd build && ctest -E '^cco_' -LE integration"
```

Exclude `^cco_` — those are unrelated GPU collective tests and some hang without
a full fabric. `-LE integration` skips the tests needing real RDMA.

## Reference

- `medium_backend.h` — the interface, and a list of things deliberately **not**
  on it (with reasons, so they are not re-proposed)
- `page_backend.h` — `PageMemorySource`, `HostPageMemorySource`, `PageBackend`
- `hbm_backend.{h,cpp}` — the minimal paged medium (shape 1)
- `ssd_backend.{h,cpp}` — the staged, non-addressable medium (shape 2)
- `doc/design-backend-agnostic-refactor.md` — §2 the descriptor/pointer rule,
  §3 the three-component split, §8 what none of it fixes
