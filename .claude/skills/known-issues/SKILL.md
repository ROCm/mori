---
name: known-issues
description: >-
  Known environment issues that make MORI slow, hang or behave oddly without being MORI
  bugs — currently: (1) HIP VMM peer memory pinned into host memory instead of peer VRAM
  when the kernel lacks CONFIG_DMABUF_MOVE_NOTIFY AND the loaded amdgpu build checks that
  option (most DKMS builds do), which hits mori-cco (EPv2) but not mori-shmem (EPv1) and
  shows up as a ~9x slowdown or as EPv2 hanging / never finishing startup. Use when EPv2 /
  dispatch_combine_v2 / cco is much slower than expected, or hangs on one box but not on
  another, when EPv2 is far slower than EPv1 on the same box, when a2a bandwidth does not
  scale with the number of peers, or when the user asks whether a perf problem is a MORI
  bug or an environment/toolchain problem. Also: (2) on ROCm 7.0-7.2.x a clr
  hipMemSetAccess sub-buffer bug makes mori-cco 2-node ccoDevCommCreate fail with
  "hipMemSetAccess ... invalid argument"; (3) on ROCm 7.2.x clr routes uncached VMM
  allocations to the coarse-grained pool, making cco combine ~10-15us slower.
---

# MORI Known Issues

Environment-level problems that look like MORI bugs but are not. Check here **before**
debugging MORI itself when performance is unexpectedly bad on a specific machine.

---

## Issue 1 — VMM peer traffic falls off XGMI onto PCIe / host memory

**Affects:** `mori-cco` (EPv2, `dispatch_combine_v2`) and anything else built on the
HIP VMM API. **Does not affect** `mori-shmem` (EPv1), which uses IPC handles.

**Root cause:** two conditions, and a machine is affected only when **both** hold.

1. **The kernel was compiled without `CONFIG_DMABUF_MOVE_NOTIFY=y`** (and usually also
   without `CONFIG_PCI_P2PDMA=y`).
   - Known offenders: the stock Ubuntu 22.04 GA **5.15** kernel, and the Alibaba Cloud
     Linux 3 **5.10.134-\*.al8** kernels found on MI308X hosts.
   - It is a *kernel build configuration* problem, not a kernel version bug: a 5.15
     rebuilt with those options works fine, and the Ubuntu **6.8 HWE** kernel already
     has them.
2. **The loaded amdgpu driver refuses to pin a dma-buf in VRAM when that option is off.**
   - It does so with a compile-time check in `amdgpu_dma_buf_pin()` (see Mechanism).
   - amdgpu DKMS 6.16.13 has the check (verified on three hosts).
   - A pre-release amdgpu DKMS 7.2.0 build does not. An MI308X host running it on the
     same 5.10.134-al8 kernel was **not** affected.

So the verdict belongs to the pair (kernel config, amdgpu build), not to the kernel alone.
Detection below checks both.

**Impact: slow, or stuck. Nothing reports an error.**

- **Usually slow.** Correctness still passes; the a2a path is just ~9x slower (table
  below).
- **It can also stop EPv2 outright.**
  - [#917](https://github.com/ROCm/mori/issues/917) (8x MI355X, Ubuntu 5.15.0-70, DKMS
    6.16.13): after window registration, the KFD queues of all ranks were evicted in
    90–98% of runs, and vLLM never finished starting.
  - 8x MI308X, 5.10.134-al8, DKMS 6.16.13: `test_asyncll.py` at 8 ranks hangs, on main
    as well. The same test passes on a 5.10.134-al8 host whose amdgpu build keeps the
    buffer in VRAM. The two hosts also run different firmware, so treat this as a strong
    association, not an isolated cause.

### Symptoms

- EPv2 (AsyncLL in particular) hangs or crawls in its first rounds, or a serving engine
  never finishes starting, while EPv1 on the same box is fine.
- The same EPv2 job behaves differently on two hosts with the same kernel. Compare their
  amdgpu builds (Detection, step 3).
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

### Detection

**The runtime probe (step 1) is the verdict.** Steps 2 and 3 explain the result and tell
you which side to fix; neither decides it alone. In particular, a kernel without
`CONFIG_DMABUF_MOVE_NOTIFY` does **not** by itself mean the machine is affected.

**Step 1: runtime probe.** It needs two GPUs and takes a few seconds.
`vmm_peer_probe.cpp` in this skill directory allocates VMM memory on one GPU, grants peer
access from another, and checks whether the memory stayed in VRAM:

```bash
hipcc -O3 --offload-arch=$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*') \
      vmm_peer_probe.cpp -o vmm_peer_probe
./vmm_peer_probe            # defaults: owner=1 peer=0 size=8GB
./vmm_peer_probe 7 6 2      # owner=7 peer=6 size=2GB, e.g. on GPUs other jobs leave free
```

- `VERDICT: OK`: the memory stayed in VRAM, so the machine is fine.
- `VERDICT: AFFECTED`: granting peer access evicted the buffer out of VRAM. The probe's
  message then points at the kernel config only; steps 2 and 3 show which side to change.

**Step 2: kernel config.** No GPU needed.

```bash
grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' /boot/config-$(uname -r)
```

- If both print `=y`, the kernel side is fine.
- If either is missing (`# CONFIG_... is not set`, or no output), the result depends on
  the amdgpu build (step 3).

**Step 3: the amdgpu build.** No GPU needed. On ROCm machines amdgpu normally comes from
DKMS, with its source under `/usr/src`. Print the pin function of the build that is
installed for the running kernel:

```bash
dkms status amdgpu                  # which version is installed for $(uname -r)
awk '/^static int amdgpu_dma_buf_pin\(/,/^}/' \
    /usr/src/amdgpu-<version>/amd/amdgpu/amdgpu_dma_buf.c
```

- **Pick the right tree.** Use the version that `dkms status` reports as installed for the
  running kernel. A host can carry several `/usr/src/amdgpu-*` trees, and
  `modinfo -F version amdgpu` may print the version in another format (`7.2.0.0` for an
  `amdgpu-7.2.0-...` tree).
- **Read the whole function**, not a single grep hit. DKMS sources can hold `#if` variants
  of it for different kernels.
- **No `/usr/src/amdgpu-*` at all** (the in-kernel amdgpu): rely on step 1.

What to look for in `amdgpu_dma_buf_pin()`:

| what the function does | seen in | kernel without the option | kernel with it |
|---|---|---|---|
| strips VRAM under `if (!IS_ENABLED(CONFIG_DMABUF_MOVE_NOTIFY))`, otherwise per attachment | DKMS 6.16.13 | **affected** | fine |
| strips VRAM only for attachments without `peer2peer`; no `IS_ENABLED` branch | pre-release DKMS 7.2.0 | fine | fine |
| anything else (e.g. pins straight to `AMDGPU_GEM_DOMAIN_GTT`) | — | rely on step 1 | rely on step 1 |

Hosts seen so far:

| host | kernel | amdgpu | probe |
|---|---|---|---|
| 8x MI308X (two hosts) | 5.10.134-al8, both options unset | DKMS 6.16.13 (has the check) | AFFECTED |
| 8x MI308X | 5.10.134-al8, both options unset | pre-release DKMS 7.2.0 (no check) | OK |
| 8x MI300X | Ubuntu 5.15.0-70 rebuilt with both options `=y` | DKMS 6.16.13 (has the check) | OK |

Also useful:

```bash
uname -r                                    # 5.15.0-*-generic (Ubuntu GA) and 5.10.134-*.al8 are known offenders
grep -E 'CONFIG_IOMMU_DEFAULT' /boot/config-$(uname -r)
```

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

   This is the code in the amdgpu builds that are affected. A build without the
   `IS_ENABLED` branch keeps only the per-attachment loop. A peer GPU's attachment has
   `peer2peer` set, so the buffer stays in VRAM even on a kernel without the option. That
   is why the kernel config alone cannot decide (Detection, step 3).

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
keeps VRAM (its per-attachment `peer2peer` check is already satisfied). This gives the
same behaviour as the amdgpu builds that have no check at all (Detection, step 3):

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
./vmm_peer_probe                                                             # VERDICT: OK
grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' /boot/config-$(uname -r)   # both =y,
# or the installed amdgpu_dma_buf_pin() no longer has the IS_ENABLED branch (Detection, step 3)
```

Then re-run the EPv2 kernel bench: per-rank bandwidth should be in the same range as EPv1,
and EPv2 tests that hung before should complete.

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

---

## Issue 2 — ROCm 7.0–7.2.x: `hipMemSetAccess` sub-buffer bug (cco 2-node cannot start)

**Affects:** `mori-cco` (EPv2) **multi-node**. A clr/HIP-runtime bug present in **all ROCm
7.0 → 7.2.x releases** (regression from 6.4.x; fixed on clr `develop` 2026-01-26 but not
back-ported to any 7.2.x release branch). ROCm **7.14+** already has the fix. `mori-shmem`
(EPv1) is unaffected.

**Symptom:** `ccoDevCommCreate` fails during window setup:

```
[shmem] [error] ccoMemAlloc: hipMemSetAccess failed after retries: 1 (invalid argument)
[shmem] [error] ccoDevCommCreate: resource window MemAlloc failed
RuntimeError: ccoDevCommCreate failed: -1
```

Single-node cco usually still passes; the failure shows up cross-node. **`CCO_GDR_CAPABLE=0`
does not help** — the trigger is sub-buffer geometry, not GPUDirect-RDMA.

**Root cause:** `hipMemSetAccess` validates coverage by walking the parent reservation's
sub-buffers **from sub-buffer 0, ignoring the `ptr` argument**, and returns `InvalidValue`
whenever `size` is not a prefix-sum of sub-buffer sizes starting at sub-buffer 0. cco
reserves one flat VA (`hipMemAddressReserve`) and maps several **different-sized** windows
(sub-buffers) into it, each with its own `hipMemSetAccess`, so every window after the first
fails. (SWDEV-568260 / ROCm/rocm-systems#2451, commit `be8bcd059`.)

**Fix — upgrade, or rebuild clr with the upstream fix and swap `libamdhip64.so`.**

Prefer **ROCm 7.14+**. If you must stay on 7.2.x, rebuild `clr` for your exact version and
cherry-pick the fix (this is what `docker/Dockerfile.cco` did before the 7.14 move —
`git log -S be8bcd059 -- docker/` for the original recipe):

```bash
V=$(cat /opt/rocm/.info/version | head -c5)          # e.g. 7.2.3
git clone --depth 1 --branch rocm-$V https://github.com/ROCm/clr.git /tmp/clr
git clone --depth 1 --branch rocm-$V https://github.com/ROCm/HIP.git /tmp/hip
cd /tmp/clr && git fetch origin develop && git cherry-pick be8bcd059     # clean on 7.2.x
cmake -B build -DHIP_COMMON_DIR=/tmp/hip -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
      -D__HIP_ENABLE_RTC=OFF -D__HIP_ENABLE_PCH=OFF
make -C build amdhip64 -j$(nproc)
# back up and replace the real file behind the libamdhip64.so.7 symlink, keeping its name:
T=$(readlink -f /opt/rocm/lib/libamdhip64.so.7)
cp "$T" "$T.orig" && cp build/hipamd/lib/libamdhip64.so.*-* "$T"
```

The rebuilt lib links the system `libhsa-runtime64.so.1` (ROCR is not touched). Deploy by
replacing the file (or `LD_PRELOAD`) and restarting the process; it only works on the
**exact** matching ROCm version (soname must match, e.g. `libamdhip64.so.7.2.70203` for
7.2.3).

---

## Issue 3 — ROCm 7.2.x: uncached VMM allocations land in the coarse-grained pool (cco combine slow)

**Affects:** `mori-cco` (EPv2) **combine latency** on ROCm **7.2.x** (fixed on clr
`develop`, not in 7.2.x release; 7.14+ is fine). Correctness is unaffected — this is a
pure perf issue, and it usually rides along with Issue 2 on the same 7.2.x box.

**Symptom:** cco combine is ~10–15 µs slower than on ROCm 7.14 for the same shape, and
requesting uncached memory (the default) is **no faster than forcing cached**
(`CCO_UNCACHED_WINDOW=0` gives the same time). Example: v1_ll combine, tok16, hidden 6144,
fp8/bf16, EP16 → ~65 µs on 7.2.x vs ~54 µs on 7.14; dispatch is roughly unchanged.

**Root cause:** rocclr's VMM path (`deviceVmemAlloc`) always allocates from the
coarse-grained `gpuvm_segment_`, **ignoring `hipMemAllocationTypeUncached`**. cco's
symmetric window therefore stays cached, so a GPU spinning on an RDMA "done" flag hits a
dirty L2 line and waits for coarse-grained coherence. The non-VMM path (`deviceLocalAlloc`
/ `hipMalloc`, which mori-shmem uses via `hipDeviceMallocUncached`) already routes uncached
requests to the extended fine-grained pool; only the VMM path missed it.

**Fix — upgrade to ROCm 7.14+, or add the same routing to the clr rebuild from Issue 2**
(this is the upstream `develop` behavior): in `rocclr/device/rocm/rocmemory.cpp` propagate
the uncached flag into `deviceVmemAlloc` (it is hard-coded to `0`), and in
`rocclr/device/rocm/rocdevice.cpp` `Device::deviceVmemAlloc` select
`gpu_ext_fine_grained_segment_` when the uncached flag is set and that pool exists, else
fall back to `gpuvm_segment_` (unchanged for every non-uncached allocation). Both Issue 2
and Issue 3 ship in a single rebuilt `libamdhip64.so`.
