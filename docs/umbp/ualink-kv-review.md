# UALink KV sharing review and two-host validation

Reviewed on 2026-11-09. The review covers `feat/umbp-ualink-kv-sharing`, starting
at `01b6c5e`, against upstream main `af56d9b`. It includes UMBP's public APIs,
PoolClient, peer pin/write protocols, paged storage, MORI-IO/FABRIC completion,
standalone GPU IPC, Python bindings and the hardware validation driver.

## Corrections from the review

### Async read lifetime across threads

A read state retained a `std::shared_lock<std::shared_mutex>` acquired by
`SubmitBatchGetRangesGpuDirect`. A different thread could retire it in `Wait`,
handle destruction or client shutdown. That violates the mutex's thread
ownership contract; multiple outstanding submissions on one thread also
attempted recursive shared ownership of a nonrecursive mutex.

The correction uses counted read-state lifetime leases that can retire on any
thread. The preparation lock stays within the submitting call. Shutdown and
registration changes stop admission, drain published handles, and wait for
all state leases, including handles already undergoing destruction. Admission
budget ownership is prepared before counters are changed and retires on
construction failures.

### Completion failure reporting under allocation pressure

The inherited MORI-IO transfer handle marked itself drained before waiting for
all posted groups, and could allocate an early failure report first. A reporting
allocation failure could therefore skip later waits. The handle now settles all
posted statuses before building reports.

PoolClient must also treat a thrown completion/reporting call as uncertain:
retrying an already-drained handle can otherwise lose its original failures.
GPU direct reads fail closed and retain borrowed resources when completion
cannot be proved. Quarantine retention itself must be prepared before posting
work so that emergency retention does not allocate or keep a live read-state
lease that would block shutdown forever.

The owner also prepares the retained grant before publishing a read receipt;
stopping its RPC server can retain active grants without allocating in the
pool destructor. Receipt/token publication rolls back on indexing failure.
Preparation-scope cleanup contains allocation/RPC exceptions and leaves
unresolved owner resources protected. These changes protect read drain and
emergency retention; they do not make every shutdown RPC or ROCm cleanup
allocation-free.

### Registration drain and completion callbacks

`IOEngine::DeregisterMemory` waited for transfer completion while holding a
registration ticket mutex. A concurrent FABRIC completion held the event mutex
and notified an observer needing the ticket mutex. This lock order could
deadlock. Draining must run outside ticket locks, with separate lifetime
protection for the status while a drain is using it.

Completion callback setup also needs to precede GPU submission: a host
allocation failure after posting work must not leave an in-progress status
without any progress/wait callback. Submission failures retain referenced
resources unless GPU drain is confirmed.

### GPU direct GET and remote ranged PUT compatibility

Selecting `gpu_direct` previously skipped both ranged arenas, including an
explicitly configured PUT arena. A client using direct GET and master-routed
remote `BatchPutRanges` could therefore lose its existing write path.

Direct GET skips the GET arena. An explicitly configured PUT arena remains
available, including its NUMA and hugepage settings. The intended
`fabric_strict` + `owner_local` deployment with zero scratch remains free of
host payload arenas. Remote ranged PUT still uses its existing assembly path;
this review does not introduce direct remote scatter writes.

Standalone's simulated GPU registration also needs a transport descriptor for
GPU direct GET, just as the real IPC mapping does. Its virtual-memory path uses
the simulated FABRIC-local registration without invoking HIP or IPC.

### Peer stub ownership during reconnect

The connection handshake and the later transfer setup are separate critical
sections. A successful handshake does not guarantee the stub still exists when
another thread has invalidated/reconnected the peer in between. Remote GET must
take the stub under the connection lock; GET and PUT must retain shared stub
ownership through their RPC and cleanup lifetimes and reject a missing stub
before posting work.

## Performance opportunities

The current path already avoids host payload staging for GPU direct GET, merges
contiguous transfer segments, reuses imported mappings/sessions and pools GPU
streams/events. Further changes should be driven by completed-transfer timing
on real trays, rather than enqueue timing or same-host link assumptions.

| Priority | Opportunity | Tradeoff and measurement |
| --- | --- | --- |
| First | Reuse device segment descriptors for stable KV layouts | Current staging allocations are pooled, but descriptors are uploaded for each batch. Measure small-group launch/upload cost before adding a bounded plan cache; key it by registration and owner generations. |
| First | Reduce per-group host bookkeeping allocations | Read-state leases, budget tokens and retention wrappers currently allocate per group. Profile preparation for small layer groups before combining or pooling state; preserve cross-thread retirement and allocation-free emergency retention. |
| First | Reduce control-plane overhead and profile owner acquire contention | Read pins are already batched per peer, but acquire and release remain RPCs and the owner uses one pin mutex. Measure RPC time and high fan-in before batching deferred releases or sharding receipts; unresolved pins must remain retained. |
| Workload dependent | Optional HBM partial cache and missing-range deduplication, inspired by PR #888 | One remote fetch can serve several consumers through local D2D copies. It adds HBM occupancy and a local copy, and must include semantic KV identity, owner/content epoch, completed-range visibility and quarantine. Keep direct-to-destination reads for low-reuse workloads. |
| Inference integration | Asynchronous standalone read groups | Embedded clients expose stream waits; standalone ranged RPCs currently wait for destination completion. A future async protocol needs server-side operation handles and safe local completion notification, rather than sending raw HIP pointers across RPC. |
| Inference integration | Native export of inference-owned KV | Removes the producer-to-UMBP pool copy, but requires fabric-exportable inference allocations and coordinated allocator/model lifetime. |

Source-pin admission currently charges logical object bytes as well as requested
destination bytes. This is conservative for sparse layer reads. Any refinement
must distinguish logically held objects from physical pages retained, including
shared backing allocations, and keep a bounded resource budget.

The hardware driver's inference-consumption copy intentionally copies the full
destination buffer to verify stream order and untouched bytes. Its
`consumer_ready_us` includes this copy and consumer-stream synchronization.
The driver's GET latency is end-to-end: route lookup, acquire RPC, planning,
submission, completion wait and pin-release RPC. It is not isolated kernel or
link timing. Byte-verification D2H is outside both measurements. Use a separate
inference workload for TTFT/TPOT; do not interpret the
driver's full-buffer consumer copy as an optimized inference connector.

## Validation when f01-1 is ready

Use the same branch revision and compatible ROCm build on both hosts. Build
`umbp_ualink_kv_driver` and `mori_fabric_copy_kernel`. Each GPU must be in the
same accessible fabric domain, and the master and advertised peer addresses
must be reachable over the control network. Ordinary HIP IPC is used only for
same-host worker/server mappings; the remote data path uses Fabric handles.

The launch commands and public configuration are in
[ualink-kv-sharing.md](ualink-kv-sharing.md#validation-and-measurement).
Use a new `--run-id` for each geometry and keep producer/master processes alive
until all consumers have completed.

1. Run f01-1 producer to f01-2 consumer with aligned and unaligned layouts,
   both `--async 1` and `--async 0`. Include `--cold-every 10` to separate
   first GET after client reconstruction from warm mapping reuse. Cold GET
   includes initial connection/import costs; reconstruction is outside its timer.
2. Reverse producer and consumer hosts. Verify object ownership, every requested
   byte, untouched destination regions, GPU stream dependencies,
   `selected_fabric_submitted_bytes` and absence of consumer-local replicas.
   The counter describes submitted plans, not physical link traffic; accept it
   only alongside successful completion and byte verification.
3. Sweep object/page sizes, key counts and requested fractions; record selected
   plans, requested bytes, cold/warm p50/p95/p99 and aggregate completed GET
   bandwidth. Keep sample sizes large enough for useful tail latency estimates.
   The driver fixes three ranges and requests `2 * page_bytes + 313` bytes per
   key, so object/page geometry changes the requested fraction indirectly.
   Use an application harness for arbitrary ranges or full-object reads.
4. Add concurrent consumers with distinct node IDs and destination allocations.
   Measure owner RPC contention, GPU copy/compute contention and completed
   throughput. Use an application harness to exercise multiple outstanding layer
   groups; the basic hardware driver intentionally admits one group at a time.
5. Exercise owner Clear/restart and control-RPC failures with disposable test KV.
   Confirm stale epochs fail and pins/protected writes are retained when completion
   is uncertain. Do not recycle quarantined allocations on a timeout; GPU fault
   recovery needs an external fencing policy.

Actual f01-1/f01-2 data-path validation, cross-tray bandwidth, packed-path
comparisons and model-level speedup remain outstanding until these tests run.
Paged HBM/DRAM are covered by completion protection; local SSD GPU staging fault
fencing remains separate work.

## Follow-up verification

The final review sources built with the existing ROCm `/opt/rocm` configuration
on f01-2. Relevant CTest suites passed 10/10. The selected suites cover 11 GPU
direct range cases, 132 page/pool/peer protocol cases, 27 transfer engine cases,
15 HBM cases and standalone control checks. Two NUMA cases, three completion
reporting fault cases and 11 SIM cluster/recache cases also passed separately.
The disabled-key-handle standalone case and eight transfer-wait checks passed.

FABRIC IO integration passed in CPU and real GPU modes. Its allocation sweep
targets MORI-IO C++ bookkeeping; it does not inject failures into HIP/HSA's
internal allocators. The IO simulator used by the transfer-report regression
models completion delay without copying payload. Real byte-copy verification
comes from the FABRIC GPU, GPU direct range, NUMA and process tests.

Python async GPU reads passed import/configuration, default-stream dependency,
repeat wait/close and byte checks. Three separate processes passed unaligned
async, aligned async and aligned blocking reads, including cold reconstruction.
Standalone GPU IPC passed with matching and reversed worker device ordering.
Each driver read requested 263,396 bytes and reported the same submitted FABRIC
byte count and one plan, without a consumer-local KV replica.

Only the relevant UMBP/IO targets and independent UMBP Python extension were
built. This reduced configuration disables OPS/SHMEM, so the generic MORI Python
module cannot link those libraries. Actual two-host UALink validation remains
pending; no cross-tray throughput or inference-speedup result is claimed.
