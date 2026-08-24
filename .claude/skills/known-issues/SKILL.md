---
name: known-issues
description: >-
  Known environment issues that make MORI slow or behave oddly without being MORI bugs —
  currently: HIP VMM peer traffic silently falling off XGMI onto PCIe/host memory on Linux
  kernels built without CONFIG_DMABUF_MOVE_NOTIFY / CONFIG_PCI_P2PDMA, which hits mori-cco
  (EPv2) but not mori-shmem (EPv1). Use when EPv2 / dispatch_combine_v2 / cco is much slower
  than expected, when EPv2 is far slower than EPv1 on the same box, when a2a bandwidth does
  not scale with the number of peers, or when the user asks whether a perf problem is a
  MORI bug or an environment problem.
---

# MORI Known Issues

Environment-level problems that look like MORI bugs but are not. Check here **before**
debugging MORI itself when performance is unexpectedly bad on a specific machine.

---

## Issue 1 — VMM peer traffic falls off XGMI onto PCIe / host memory

**Affects:** `mori-cco` (EPv2, `dispatch_combine_v2`) and anything else built on the
HIP VMM API. **Does not affect** `mori-shmem` (EPv1), which uses IPC handles.

**Root cause:** the Linux kernel was compiled **without**
`CONFIG_DMABUF_MOVE_NOTIFY=y` (and usually also without `CONFIG_PCI_P2PDMA=y`).
This is the case for the stock Ubuntu 22.04 GA **5.15** kernel. It is a *kernel build
configuration* problem, not a kernel version bug and not an amdgpu regression — a 5.15
rebuilt with those options works fine, and the Ubuntu **6.8 HWE** kernel already has them.

**Impact is silent.** Nothing errors, correctness still passes; the a2a path is just
~9x slower.

### Symptoms

- EPv2 dispatch/combine bandwidth is a fraction of EPv1's on the same machine.
- Per-rank bandwidth plateaus at roughly a **single** XGMI link (~55 GB/s on MI355X),
  or below it — even though the box has 7 peers.
- Aggregate bandwidth **does not increase with the number of peers**: 1 peer and 7 peers
  give the same total.
- `rocm-smi --showtopotype` shows XGMI everywhere and TransferBench reports full
  aggregate bandwidth — because those use the non-VMM path and are unaffected.

Measured on an 8x MI355X box (`HIDDEN=7168 TOPK=8 EPR=32`, bf16, 4096 tok/rank, 8 ranks):

| | dispatch | combine |
|---|---|---|
| EPv1 (shmem / IPC handle) | 345 GB/s | 377 GB/s |
| EPv2 (cco / VMM) — **affected kernel** | **36 GB/s** | **54 GB/s** |
| EPv2 (cco / VMM) — after fixing the kernel | **340 GB/s** | **390 GB/s** |

### Detection (30 seconds, no GPU needed)

```bash
grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' /boot/config-$(uname -r)
```

Both must print `=y`. If either is missing (`# CONFIG_... is not set` or no output),
the machine is affected.

Also useful:

```bash
uname -r                                    # 5.15.0-*-generic (Ubuntu GA) is the usual offender
grep -E 'CONFIG_IOMMU_DEFAULT' /boot/config-$(uname -r)
```

### Runtime confirmation

`vmm_peer_probe.cpp` in this skill directory gives a direct verdict — it allocates VMM
memory on one GPU, grants peer access from another, and checks whether the memory stayed
in VRAM:

```bash
hipcc -O3 --offload-arch=$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*') \
      vmm_peer_probe.cpp -o vmm_peer_probe
./vmm_peer_probe            # defaults: owner=1 peer=0 size=8GB
```

- `VERDICT: OK` — memory stayed in VRAM, the machine is fine.
- `VERDICT: AFFECTED` — granting peer access evicted the buffer out of VRAM.

### Mechanism (what actually happens)

1. `hipMemSetAccess()` → ROCr `hsa_amd_vmem_set_access()` imports the allocation into
   **each peer** as a **dma-buf** (`ImportMemoryHandle(DMABUF_FD)` → `hsaKmtHandleImport`
   → KFD `import_obj_create()` → `amdgpu_gem_prime_import()` →
   `dma_buf_dynamic_attach()`).
2. The dma-buf core then pins the exporter's buffer, calling `amdgpu_dma_buf_pin()`
   (`amd/amdgpu/amdgpu_dma_buf.c`):

   ```c
   u32 domains = bo->allowed_domains;              /* VRAM | GTT */
   if (!IS_ENABLED(CONFIG_DMABUF_MOVE_NOTIFY)) {
           domains &= ~AMDGPU_GEM_DOMAIN_VRAM;     /* <-- VRAM stripped */
   } else {
           list_for_each_entry(attach, &dmabuf->attachments, node)
                   if (!attach->peer2peer)
                           domains &= ~AMDGPU_GEM_DOMAIN_VRAM;
   }
   return amdgpu_bo_pin(bo, domains);              /* GTT = host system memory */
   ```

   `IS_ENABLED()` is a **compile-time** constant taken from the kernel's own
   `autoconf.h`. Without move-notify support the kernel cannot ask an importer to drop
   its mapping when a buffer must move, so amdgpu conservatively refuses to pin in VRAM.

3. The buffer is therefore **pinned into host system memory**. Note this is stronger than
   "peer access falls back to PCIe": the data is not in peer VRAM at all. Every subsequent
   "peer" store is a write to host DRAM across the *writing* GPU's own PCIe link, which is
   why aggregate bandwidth is capped at ~one PCIe link no matter how many peers there are.

4. `amdgpu_dma_buf_map()` contains a `peer2peer`/VRAM placement check, but it is guarded
   by `if (!bo->tbo.pin_count)` and the pin already happened — that code never runs.

**Why EPv1 is immune:** `hsa_amd_agents_allow_access()` maps the *same* BO into every
GPU's page tables in one `hsaKmtMapMemoryToGPUNodes()` call. No dma-buf, no attach, no pin.

**Why there is no application-level workaround:** MORI runs one process per rank, so peer
memory must cross process boundaries, which requires exportable handles — and ROCr
unconditionally does the per-peer dma-buf import (it even lazily exports a dma-buf fd for
locally created handles). cco also cannot fall back to IPC handles because it needs a flat
symmetric VA at computed addresses, which only
`hipMemAddressReserve()` + `hipMemMap()` can provide.

### Fix

**Preferred — install a kernel that already has the options** (Ubuntu 22.04):

```bash
sudo apt install linux-generic-hwe-22.04       # 6.8; amdgpu DKMS rebuilds automatically
sudo reboot
```

**If the machine must stay on 5.15 — rebuild it with two options.** All their Kconfig
dependencies (`ZONE_DEVICE`, `MEMORY_HOTPLUG`, `SPARSEMEM_VMEMMAP`, `GENERIC_ALLOCATOR`,
`DMA_SHARED_BUFFER`, …) are already satisfied in the stock config:

```bash
sudo apt build-dep linux-image-unsigned-$(uname -r)
apt-get source linux-image-unsigned-$(uname -r)
cd linux-5.15.0-*/
cp /boot/config-$(uname -r) .config
scripts/config -e DMABUF_MOVE_NOTIFY -e PCI_P2PDMA
scripts/config -d DEBUG_INFO -d DEBUG_INFO_BTF     # much faster build, much smaller packages
make olddefconfig
grep -E 'CONFIG_(DMABUF_MOVE_NOTIFY|PCI_P2PDMA)=' .config   # verify before building
make -j$(nproc) bindeb-pkg
sudo dpkg -i ../linux-image-*.deb ../linux-headers-*.deb
sudo reboot                                         # amdgpu DKMS rebuilds for the new kernel
```

Check `mokutil --sb-state` first — with Secure Boot enabled a self-built kernel needs
signing. The old kernel stays in GRUB, so this is revertible. Note that a self-built
kernel leaves the distro's kernel maintenance path; prefer the HWE kernel for anything
long-lived.

`CONFIG_DMABUF_MOVE_NOTIFY` is the one that actually matters. `CONFIG_PCI_P2PDMA` is
enabled alongside it because upstream `HSA_AMD_P2P` depends on both and it costs nothing.

**Third option — one-line amdgpu DKMS patch, no kernel change.** Force the branch that
keeps VRAM (its per-attachment `peer2peer` check is already satisfied):

```bash
S=/usr/src/amdgpu-<version>/amd/amdgpu/amdgpu_dma_buf.c
sudo cp $S $S.orig
sudo sed -i 's|if (!IS_ENABLED(CONFIG_DMABUF_MOVE_NOTIFY)) {|if (0) { /* WAR: kernel lacks MOVE_NOTIFY */|' $S
sudo dkms build  amdgpu/<version> -k $(uname -r) --force
sudo dkms install amdgpu/<version> -k $(uname -r) --force
```

Then either reboot, or reload the module (`modprobe -r amdgpu && modprobe amdgpu`) if
nothing holds the GPUs (`lsmod | grep '^amdgpu'` must show `used_by=0` and
`rocm-smi --showpids` must show no KFD PIDs). Reloading invalidates every container's GPU
context, so it disturbs other users exactly as much as a reboot — it only saves time.
Roll back with `sudo cp $S.orig $S` and rebuild.

### Verification after the fix

```bash
grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' /boot/config-$(uname -r)   # both =y
./vmm_peer_probe                                                             # VERDICT: OK
```

Then re-run the EPv2 kernel bench; per-rank bandwidth should be in the same range as EPv1.

---

## About IOMMU — related but **not** the cause

`amd_iommu=on iommu=pt` does **not** fix Issue 1. This was tested directly: with
passthrough confirmed active (`dmesg` showing
`iommu: Default domain type: Passthrough`, and
`/sys/bus/pci/devices/<BDF>/iommu_group/type` reading `identity`), bandwidth and IOMMU
transaction counters were unchanged. The buffer is pinned into host memory before any
address-translation question arises.

The tempting but wrong theory: IOMMU passthrough sets `adev->ram_is_direct_mapped`, which
KFD's `reuse_dmamap()` uses to choose `KFD_MEM_ATT_SHARED` over `KFD_MEM_ATT_DMABUF`.
That choice really does flip with `iommu=pt` — but both attachment types share the same
scatter-gather BO, whose pages were already placed in host memory by the pin.
**Attachment type is not the physical route.**

**Still enable `iommu=pt` anyway.** It is the standard configuration for AMD GPU / HPC
nodes, amdgpu uses `ram_is_direct_mapped` elsewhere, and it removes IOMMU translation
overhead. Just do not let it stand in for the kernel config fix. Note that Ubuntu kernels
default to translated mode (`CONFIG_IOMMU_DEFAULT_DMA_LAZY=y`,
`CONFIG_IOMMU_DEFAULT_PASSTHROUGH` not set), so passthrough must be requested explicitly
on the kernel command line.

IOMMU is, however, an excellent **diagnostic instrument** for this issue, because host-DRAM
DMA is translated by the IOMMU while XGMI peer traffic never touches it:

```bash
EV=""; for i in 0 1 2 3 4 5 6 7; do EV="$EV -e amd_iommu_$i/mem_trans_total/"; done
sudo perf stat -a $EV -- sleep 8      # while an a2a / peer-write workload runs
```

On an affected machine one IOMMU counts billions of transactions during VMM peer traffic
(64 B each — multiply out and it matches the observed bandwidth exactly). After the fix
the same measurement drops by ~380x, down to the same noise level as non-VMM peer traffic.
`amd-smi metric --pcie` is a coarser cross-check: on an affected box exactly one card —
the one issuing the writes — shows sustained multi-GB/s PCIe traffic while all its peers
stay idle, which is itself the tell that the peers are not the destination.

---

## Also worth knowing

- **`CONFIG_HSA_AMD_P2P` cannot be relied on.** `amd/dkms/dkms-config.sh` tests for the
  misspelled `CONFIG_DMABUF_MOVENOTIFY` (missing underscore), so the amdgpu DKMS build
  derives `CONFIG_HSA_AMD_P2P=0` even on kernels where both real options are `=y`. Present
  in at least amdgpu DKMS 6.14.14 and 6.16.13. Do not use `CONFIG_HSA_AMD_P2P` as a health
  signal, and do not expect code under `#ifdef CONFIG_HSA_AMD_P2P` to be compiled in.

- **KFD may report `max_bandwidth=0` for XGMI io-links**
  (`/sys/class/kfd/kfd/topology/nodes/*/io_links/*/properties`) on some hosts while others
  report `64000`, with no functional difference. It is unexplained but unrelated to
  Issue 1 — `kfd_mem_attach()` never consults link bandwidth. Do not chase it.

- **Micro-benchmarking peer bandwidth is noisy.** Use ≥1024 blocks per stream and take the
  best of several repetitions. An under-parallelised harness produced a 1.5x spread between
  consecutive identical runs, which is more than enough to invent or hide a regression.
