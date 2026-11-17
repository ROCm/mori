# UMBP over a UALink fabric

On a UALink scale-up fabric a GPU can read and write HBM on another tray
directly. UMBP uses that link for every DRAM/HBM pair it can reach in one
copy, through mori-io's existing FABRIC backend.

## Which pairs are one hop

Only GPU memory created fabric-exportable can be named across trays; host
memory cannot be exported. A copy is one hop when **at least one end is
exportable HBM**: the GPU on the other end's tray runs the copy, reaching its
own local memory directly and the exportable end over the fabric.

| One end | Other end, on another tray | Hops | Who copies |
| --- | --- | --- | --- |
| HBM tier | GPU memory | 1 | the GPU next to the caller's memory |
| HBM tier | registered host memory | 1 | a GPU on the caller's tray |
| DRAM tier | exportable GPU memory | 1 | a GPU on the DRAM's tray |
| DRAM tier | plain GPU memory, or host memory | none | no direct link: needs RDMA |

The HBM tier is always exportable when the fabric is enabled. "Exportable GPU
memory" on the caller's side means allocated through mori-io's fabric
allocator (`fabric_alloc`, or the fabric torch allocator) and registered.

## Enabling it

| Environment variable | Values | Meaning |
| --- | --- | --- |
| `UMBP_TRANSFER_MODE` | `rdma` (default), `auto`, `fabric_strict` | `auto` adds the fabric to RDMA and uses it for every pair it can carry. `fabric_strict` uses the fabric alone and needs no RDMA device. |
| `UMBP_FABRIC_KERNEL_PATH` | path | The compiled `src/io/kernels/fabric_copy.hip`. Required by `fabric_strict`; without it `auto` stays on RDMA. |
| `UMBP_FABRIC_HOST_ENDPOINTS` | `1` (default), `0` | Let registered host memory be the local end of a fabric copy. |
| `UMBP_FABRIC_HOST_DEVICE` | GPU ordinal, default `0` | The GPU whose copy kernel serves host memory. |

Compile the kernel once per GPU architecture:

```bash
hipcc --genco --offload-arch=<arch> -O3 src/io/kernels/fabric_copy.hip -o fabric_copy.hsaco
```

Both trays must be in the same vPOD. In `rdma` mode nothing changes.

## How each path works

**HBM tier.** The pool is allocated exportable and its descriptor carries the
fabric handle, so a reader or writer on another tray plans an ordinary
transfer and mori-io picks the fabric for it.

**Host memory on the caller's side.** A GPU reaches registered host memory
through the address `hipHostRegister` gives it. `MoriIoEngine` registers that
address with mori-io once per registered piece and its planner substitutes it
for a local host endpoint whenever the other end is exportable HBM in the
same vPOD. DRAM tier buffers, the ranged scratch arenas and caller buffers
registered with `RegisterMemory(ptr, size, CPU)` all qualify. A buffer above
2 GiB is registered piece by piece in the background; in `fabric_strict` a
transfer touching a piece that is not registered yet fails rather than waits.

**Writing into another tray's DRAM (owner pull).** The writer marks its
allocation `offer_owner_pull` when everything it sends is exportable HBM. An
owner whose slot is in host memory then publishes no pages; the writer sends
nothing and names its source ranges in `BatchCommitSlots`, and the owner's
GPU reads them straight into the tier before the key is published. The
ranges must tile the object.

**Reading another tray's DRAM (owner push).** When every destination range is
exportable HBM, the reader sends them in `BatchPushKeys` and the owner's GPU
copies its tier straight into them.

In both cases the caller's memory must stay untouched until the RPC replies;
neither call has a deadline. On the owner, an abort or a repeated commit of
a slot waits for the copy in flight before answering.

**Ranged I/O.** In `fabric_strict` a ranged read or write routed to another
node goes straight between the caller's registered ranges and the owner, with
no scratch arena. In `auto` ranged calls keep the arena, whose remote leg
rides the fabric through the host-memory path above.

## What is not covered

* DRAM on one tray against host memory or plain (`hipMalloc`) GPU memory on
  another has no direct link. `auto` carries it over RDMA; `fabric_strict`
  refuses it per key.
* SSD tiers are not reached over the fabric.
* A slot held for owner pull is reaped at the tier's pending-slot deadline
  like any other. A commit that arrives with less than five seconds, plus one
  millisecond per MiB, of that deadline left is refused rather than started.

## Validation

`umbp_ualink_direct_driver` (built with the tests, not registered with CTest)
runs a master, an owner and a reader or writer as separate processes:

```bash
umbp_ualink_direct_driver --role master --master TRAY_A:50091
umbp_ualink_direct_driver --role owner  --master TRAY_A:50091 --advertise TRAY_A \
    --node-id a --run-id t1 --kernel fabric_copy.hsaco --medium dram
umbp_ualink_direct_driver --role reader --master TRAY_A:50091 --advertise TRAY_B \
    --node-id b --owner a --run-id t1 --kernel fabric_copy.hsaco --medium dram \
    --memory fabric-hbm --expect-owner-push 1
```

`--role writer` stores objects on the owner (started with `--prefill 0`) and
reads them back; `--memory hbm|fabric-hbm|host` picks the caller's memory.
Every run compares every byte.

Measured on two MI455X trays in one vPOD, `fabric_strict`, 1 GiB per call,
end to end including the control RPCs:

| Operation | Path | p50 per call | GB/s |
| --- | --- | --- | --- |
| Read | HBM tier to GPU memory | 10.9 ms | 97 |
| Read | HBM tier to host memory | 28.2 ms | 38 |
| Read | DRAM tier to exportable GPU memory (owner push) | 20.6 ms | 52 |
| Write | GPU memory to HBM tier | 10.6 ms | 100 |
| Write | host memory to HBM tier | 22.3 ms | 48 |
| Write | exportable GPU memory to DRAM tier (owner pull) | 22.2 ms | 48 |
