# epcheck: reproducing the gfx1250 xGMI bandwidth measurements elsewhere

Ships three sources plus a check script that compiles them, measures, and compares against the
baseline recorded on the reference node. Exit code is non-zero if anything is out of tolerance, so it
works as a CI gate.

## What is in the image

A thin layer (4.77 kB) over a pinned ROCm base image:

```
rocm/fw-bringup:gfx1250-atom-dev-20260714-tp4_pro_flash-a8w8-fix
  HIP 7.14.60850 / AMD clang 23.0.0git / ROCm 7.14.0
```

The base tag is pinned on purpose. `tdma2a.cc` includes `<hip/amd_detail/amd_gfx1250_TDM.h>` and the
TDM builtins only exist from clang-22 (ROCm 7.2) on, so an older base cannot compile it, and a newer
one can shift codegen enough to move the numbers. Nothing is precompiled: the benchmarks are built at
check time, so a toolchain problem on the target shows up as a check result instead of being hidden.

## Choosing a transfer route

The base image is **271 GB**. That rules out `docker commit` + `docker save`: the tar would not fit
on the reference node (381 GB free), let alone transfer. Two routes work; the second avoids moving
large data entirely and is the recommended one.

### Route A - via Docker Hub (done; target pulls one image)

Already published:

```
docker.io/rocm/aigmodels-private:epcheck-gfx1250-20260731
sha256:3a7a9a6843228643e1cc514f20452ccbac0937650188553a3ed724a7c71fdfef
```

The push cost almost nothing despite the 271 GB image: `rocm/aigmodels-private` and
`rocm/fw-bringup` live in the same Docker Hub namespace, so 64 of the 75 layers were served by
cross-repo blob mount (`Mounted from rocm/fw-bringup`), 2 already existed, and only our 2 thin
layers were uploaded. The whole push finished inside the 60 s probe window.

The repo is private (an anonymous manifest GET returns HTTP 401), so the target must log in:

```bash
docker login -u aigmkt
docker pull rocm/aigmodels-private:epcheck-gfx1250-20260731
```

What the target actually downloads depends on what it already has. The base layers are shared by
digest, so a machine that has already pulled
`rocm/fw-bringup:gfx1250-atom-dev-20260714-tp4_pro_flash-a8w8-fix` only fetches the two thin
layers; a machine with nothing cached pays the full 271 GB. Prefer Route B in that case.

To republish after changing the sources:

```bash
docker build -t mori-epcheck:gfx1250-<date> tools/epimg
docker tag mori-epcheck:gfx1250-<date> rocm/aigmodels-private:epcheck-gfx1250-<date>
docker push rocm/aigmodels-private:epcheck-gfx1250-<date>
```

### Route B - target builds the thin layer itself (recommended)

Moves 44 kB instead of 271 GB. The target pulls the public base image directly and applies our layer.

Copy `Dockerfile`, `epcheck.sh`, `ualoe_bw.cpp`, `tdma2a.cc` into one directory on the target, then:

```bash
docker pull rocm/fw-bringup:gfx1250-atom-dev-20260714-tp4_pro_flash-a8w8-fix
docker build -t mori-epcheck:gfx1250-20260731 .
```

`tdma2a.cc` is generated inline by `tools/_ct_tdma2a.sh`; take it from a container that has already
run that script (`docker cp <ctr>:/tmp/tdma2a.cc .`) so it is byte-identical to what produced the
baseline.

## Running the check

The container must be started with the same device and IPC settings as the reference one. These are
not part of the image, and getting them wrong yields a container that starts but cannot use the GPUs:

```bash
docker run -d --name EPCHECK-V1 \
  --device /dev/kfd --device /dev/dri \
  --ipc=host --network host --privileged \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
  --group-add video --shm-size 64g \
  mori-epcheck:gfx1250-20260731

docker exec EPCHECK-V1 /opt/epcheck/epcheck.sh; echo "exit=$?"
docker rm -f EPCHECK-V1
```

`--ipc=host` and `--shm-size 64g` matter: `tdma2a` forks 4 ranks that share GPU memory through IPC
handles, and `ualoe_bw` rendezvous over loopback TCP between two processes.

Knobs: `TOL` tolerance percent (default 5), `ARCH` (default gfx1250), `PORT` for the rendezvous, and
`EPCHECK_XFLAGS` to override the ualoe workset. The 8 GB default needs about 24 GB of VRAM on each of
two GPUs; on a smaller card use `EPCHECK_XFLAGS='-DSWEEP_BIG -DONLY_1WAY'`.

The default `-DONLY_1WAY` also narrows the ualoe table to `CU 1way / TDM 1way / TDMdb 1w / TDMnl 1w`.
Dropping it restores the interleaved 2way columns, which cost roughly twice the runtime and measured
CU 1535.5, TDM 1528.5, TDMdb 1498.8, TDMnl 1536.8 GB/s per direction on 2026-07-31 -- about 6.4%
below the one-way figure on the same run, i.e. full duplex sustains ~3071 GB/s in total. `epcheck.sh`
picks its columns by field count, so either layout is parsed correctly.

## Baseline

Measured on ctheliosr-1b114-f01-2 (4x gfx1250, 256 CU, 432 GB each), compiled-in configuration
`BLKMUL=64 WTH=512 TWBLK=32 TWTH=256`, tile `256x8`, pipe 4:

| check | baseline GB/s | what it is |
|---|---|---|
| a2a grid=512 aggregate | 1627.6 | 4-rank TDM all-to-all, includes the self-write |
| CU copy 1way | 1641.0 | GPU0 -> GPU1, uint4 copy, 8 GB, timed on GPU0 |
| TDM copy 1way | 1637.3 | same, TDM load to LDS then store to peer |
| TDM store 1way | 1643.0 | same, store only, no staging read |

These are per-configuration numbers, not hardware limits. The grid, tile and pipe values are compiled
in; changing any of them invalidates the comparison.

### L4, low occupancy: measured but not yet gated

The four checks above all sample the saturated part of the bandwidth curve — three are two-process
point to point, and the a2a one runs grid=512. On 2026-08-04 all four passed (CU copy 1640.8 against
1641.0) on a node where mori's combine was simultaneously running 18% slow, reproducibly, on the
commit that had produced its reference figure hours earlier.

The cause was a drop in per-warp TDM throughput. It is invisible at high grid because concurrency
covers for it, and it grows as grid falls — measured with epsim, 16 KB tile, 243 MB working set:

| grid | 32 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|
| healthy | 1346 | 1748 | 1773 | 1766 | 1604 |
| that day | 896.1 | 1408.5 | 1520.3 | 1545.7 | 1541.0 |

Holding grid=64 and sweeping op size separated the two components: 512 B ops were normal (72.3 vs
74.8, within the ~3.5% run-to-run spread) while ops of 16 KB and up were uniformly 11–13% down. Fixed
per-op cost intact, asymptotic bandwidth down.

This matters because mori's combine is pinned at 64 blocks — the remaining CUs belong to the GEMM it
overlaps — so it cannot buy the loss back with concurrency the way a benchmark can.

`epcheck.sh` therefore adds an L4 section measuring grid=64 and 128 plus the ratio `sat64 =
BW(64)/BW(512)`. **Its baselines are empty on purpose.** They have to be recorded on a node known to
be good, and the node this was written on was mid-regression; epcheck had also never run grid=64, so
there was no historical figure to fall back on. While empty, L4 reports and skips, so the exit code
keeps its previous meaning. Filling in `BL_A2A_G64`, `BL_A2A_G128` and `BL_SAT64` turns it into a
gate automatically.

To calibrate, run on a node that passes L0–L3 **and** whose `epsim mode0 GRIDS=64 BLOCK=256` reads in
its healthy band (1545–1610 at default `NT`). Both conditions are required: L0–L3 passing is exactly
what failed to catch this. Note also that the `sat64` threshold cannot be carried over from epsim —
`tdma2a` is a different kernel with a different tile shape, and its healthy grid=64 may sit well
below its grid=512.

Repeatability differs sharply between the two benchmarks, which matters when reading a result. Across
three runs the copy figures moved by 0.1% (CU 1640.6/1641.0/1641.0), but a2a moved by 1.8%
(1627.6/1632.9/1657.4) -- it forks 4 ranks and synchronises them, so it carries process scheduling
noise the two-process copy test does not. Do not treat a ~2% a2a deviation as a regression signal;
the copy numbers are the sensitive ones.

The a2a figure counts 4 destinations, one of which is the rank writing to itself over local HBM rather
than the fabric. Stripping that, fabric traffic is 3/4 of it, i.e. 406.9 GB/s per remote link.

## Verified reproduction

Built and run in a freshly created container on 2026-07-31:

```
gpu0..3 arch=gfx1250 CU=256 vram=432GB
a2a grid=512 aggregate       1632.9   baseline 1627.6   PASS
CU copy 1way                 1640.6   baseline 1641.0   PASS
TDM copy 1way                1639.2   baseline 1637.3   PASS
TDM store 1way               1642.4   baseline 1643.0   PASS
pass=4 fail=0 skip=0
```

Re-verified the same day through the registry reference rather than the local build tag: manifest
resolves, digest matches what push reported, `docker pull` by digest succeeds, and a fresh container
created from `rocm/aigmodels-private:epcheck-gfx1250-20260731` gave `pass=4 fail=0` with a2a 1617.2,
CU copy 1641.1, TDM copy 1639.0, TDM store 1642.1.

Republished after the table change and re-verified the same way: `pass=4 fail=0`, CU copy 1640.7, TDM
copy 1638.5, TDM store 1642.8. Both column layouts were checked to select the same three quantities
(1way figures agreed to within 0.1% across the two builds).

Reproduce the whole build-and-verify loop from the workstation with:

```
powershell -File tools/_send_ct.ps1 -Script tools/_ct_mkimg.sh \
  -Aux tools/epimg/Dockerfile,tools/epimg/epcheck.sh,tools/07_ualoe/ualoe_bw.cpp -Tmo 900
```
