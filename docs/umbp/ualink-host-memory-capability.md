# Host VMM / FABRIC capability probe

Measured locally on 2026-11-10 with HIP runtime version `71726405`, four visible
GPUs, GPU ordinal 0, and its nearest host NUMA node 1. This checks API capability
and safe 64-byte local copies; it does not run a remote host-load kernel or claim
cross-tray hardware validation.

The probe sets `hipMemAllocationProp.type = hipMemAllocationTypePinned`,
selects the location and requested handle type, then calls
`hipMemGetAllocationGranularity`, `hipMemCreate`, `hipMemAddressReserve`,
`hipMemMap`, `hipMemSetAccess` and `hipMemExportToShareableHandle`. Each case
uses a separate process with a 30-second limit. Local volatile CPU access and
64-byte H2D/D2H copies check the owner mapping. The Device control additionally
imports the exported handle and verifies a GPU copy through the imported mapping.
Unmap, free, release, imported handles and exported POSIX FDs are cleaned up.
The local probe source and detailed logs were saved under `build-ualink/`.

| Allocation / requested handle | Granularity / Create / Map | CPU access / local GPU copy | FABRIC export |
| --- | --- | --- | --- |
| Host / FABRIC | PASS | PASS | `hipErrorInvalidValue` (1) |
| HostNuma node 0 / FABRIC | PASS | PASS | `hipErrorInvalidValue` (1) |
| HostNuma node 1 / FABRIC | PASS | PASS | `hipErrorInvalidValue` (1) |
| HostNumaCurrent / FABRIC | PASS | PASS | `hipErrorInvalidValue` (1) |
| Host / FABRIC + gpuDirectRDMACapable=1 | PASS | PASS | `hipErrorInvalidValue` (1) |
| HostNuma node 1 / FABRIC + gpuDirectRDMACapable=1 | PASS | PASS | `hipErrorInvalidValue` (1) |
| Host / POSIX FD | PASS | PASS | FABRIC fails; POSIX FD export PASS |
| HostNuma node 1 / POSIX FD | PASS | PASS | FABRIC fails; POSIX FD export PASS |
| Device 0 / FABRIC | PASS | CPU PASS; local H2D/D2H PASS | PASS; import/map/GPU access PASS |

All tested allocations report minimum granularity 4096 and recommended 2097152.
Host and Device0 access grants succeed for local owner allocations; CPU volatile
write/read checks pass. `hipMemGetAccess(Host)` returns `hipErrorInvalidValue`
despite a successful access grant and verified CPU access, so it cannot be used
as evidence that local CPU access was denied. Device0 imported FABRIC mapping
allows GPU access but its Host access grant returns `hipErrorInvalidValue`.

Every executed mapping / allocation cleanup returned success; exported POSIX
FDs were closed. Each independent process was bounded by a 30-second timeout,
and all exited normally.

Detailed logs: `probe-host-fabric.log`, `probe-hostnuma0-fabric.log`,
`probe-hostnuma-near-fabric.log`, `probe-hostnumacurrent-fabric.log`,
`probe-host-fabric-rdma.log`, `probe-hostnuma-near-fabric-rdma.log`,
`probe-host-fd.log`, `probe-hostnuma-near-fd.log`, and `probe-device-fabric.log`.

The installed ROCr header `/opt/rocm/include/hsa/hsa_ext_amd.h` documents
`hsa_amd_vmem_export_fabric_handle` near line 4926 as **only supported on handles
allocated on GPU agents**. No alternative host FABRIC export API was found in
the installed HIP/ROCr headers. POSIX FD export supports handle sharing within
one OS; an FD alone does not address a remote machine's allocation.

Conclusion: this platform can allocate and locally access Host VMM memory, but
cannot export it through FABRIC. DRAM and SSD owner tiers need an explicit
exportable owner HBM staging allocation before a consumer can pull via UALink.
