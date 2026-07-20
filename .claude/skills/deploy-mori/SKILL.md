---
name: deploy-mori
description: >-
  Deploy and set up the MORI environment in a fresh Docker container or bare host:
  start the container, install ROCm dependencies, NIC userspace libraries
  (AINIC/Broadcom/Mellanox-NVIDIA), RDMA-core, and MORI itself. Use when the user asks
  to deploy MORI, install MORI in a container, set up a fresh dev environment
  for MORI, or prepare an AINIC / Thor2 / mlx5 box for MORI.
---

# Deploy MORI

You are helping the user deploy MORI inside a Docker container.

**Locate `MORI_REPO_DIR`**: default to the current working directory if it
contains `pyproject.toml`; ask the user otherwise.

**Detect whether `docker` needs `sudo`** — every `docker` command below is
written with a `sudo` prefix, but on hosts where the calling user is already
in the `docker` group, `sudo docker ...` fails non-interactively with `sudo:
a password is required` (no TTY to prompt for a password). Check first:

```bash
docker ps &>/dev/null && echo "docker works without sudo" || echo "docker needs sudo"
```

If it works without `sudo`, drop the `sudo` prefix from **every** `docker`
command in this skill (run/exec/inspect/start/cp/ps, …) for the rest of the
session.

**Detect NIC type** — determines whether Step 3 is needed:

```bash
lspci | grep -iE "pensando|ionic|dsc|pollara"      && echo "→ ainic"
lspci | grep -iE "broadcom.*thor|bnxt"             && echo "→ thor2"
lspci | grep -iE "mellanox|connectx|nvidia.*bluefield" && echo "→ mlx5"
```

A host can have more than one type present — that's fine, `mori check`/`mori
setup` only act on the vendor with the most matching RDMA devices.

For mlx5, also check RoCE vs native InfiniBand (changes what Step 3c installs):

```bash
for d in /sys/class/infiniband/mlx5_*; do
  echo "$(basename "$d"): $(cat "$d/ports/1/link_layer" 2>/dev/null)"
done
```

---

## Step 1: Start the Docker container

Use the container name from the user's args if provided; ask if not.
Default image: `rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.10.0`
(use unless the user specifies another).

```bash
if sudo docker inspect $CONTAINER_NAME &>/dev/null; then
  sudo docker start $CONTAINER_NAME
else
  sudo docker run <flags> --name $CONTAINER_NAME $IMAGE_NAME sleep infinity
fi
```

Probe optional mount paths on the host, only include ones that exist:

```bash
for p in /shared /apps /dev/infiniband /sys/kernel/config /sys/kernel/debug; do
  [ -e "$p" ] && echo "EXISTS: $p" || echo "MISSING: $p"
done
```

Full `docker run` flags (omit missing paths):

```bash
sudo docker run \
    --group-add video \
    --network=host \
    --ulimit nproc=100000:100000 \
    --ulimit memlock=-1:-1 \
    --pids-limit=-1 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \   # only if exists
    --ipc=host \
    --privileged -d \
    -v /home/:/home/ \
    -v /root:/root \
    -v /mnt:/mnt \
    -v /shared:/shared \         # only if exists
    -v /apps:/apps \             # only if exists
    -v /lib/modules:/lib/modules \
    -v /sys/kernel/config:/sys/kernel/config \   # REQUIRED for bnxt DCQCN (configfs)
    -v /sys/kernel/debug:/sys/kernel/debug \     # REQUIRED for bnxt DCQCN (debugfs)
    --rm \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    sleep infinity
```

Notes:
- `--network=host` + `--device=/dev/infiniband` required for RDMA visibility.
- `--ulimit memlock=-1:-1` — RDMA pins memory on QP creation; needed for the
  parallel `mori check` bandwidth mesh.
- configfs/debugfs mounts are **required for bnxt** DCQCN (`mori check`/`mori
  setup` read/write congestion control through them). `--privileged` alone
  doesn't propagate them — if missing on the host: `sudo mount -t configfs
  none /sys/kernel/config; sudo mount -t debugfs none /sys/kernel/debug`.
- `--rm` — data outside mounted volumes is lost on stop.

All subsequent steps run **inside** `$CONTAINER_NAME` via `docker exec`.

---

## Step 2: Install base system packages

```bash
sudo docker exec $CONTAINER_NAME bash -c "apt-get update && apt-get install -y --no-install-recommends \
    git libpci-dev pciutils sudo libdw1 libibverbs-dev ibverbs-utils rdma-core \
    locales iputils-ping iproute2 ethtool jq perftest \
    wget unzip ca-certificates curl \
    libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler \
    libopenmpi-dev openmpi-bin"
```

> **If `apt-get update` can't reach `archive.ubuntu.com`/`security.ubuntu.com`**
> (`Network is unreachable` / `connection timed out`, often on cloud hosts
> whose network only routes to a provider-internal mirror — e.g. seen on an
> Oracle Cloud instance routed through `iad-ad-1.clouds.archive.ubuntu.com`):
> the container's default `/etc/apt/sources.list` doesn't know about that
> mirror even though the **host** does. Fix by copying the host's working
> sources into the container before retrying:
> ```bash
> sudo docker cp /etc/apt/sources.list $CONTAINER_NAME:/etc/apt/sources.list
> ```

Non-obvious package roles:
- `rdma-core` — version info `mori check` Step 1 reads. Without it, Step 1
  still passes but logs a cosmetic `[WARN] rdma-core package not found`.
- `pciutils` (`lspci`) — `nicctl` shells out to it; missing it breaks the
  ionic firmware/QoS/DCQCN checks with `Invalid card handle`.
- `sudo` — ionic/bnxt paths of `mori check`/`mori setup` invoke `nicctl`,
  `dcb`, `ethtool`, sysfs writes via `sudo`.
- `perftest` — `ib_write_bw`/`ib_write_lat` for bandwidth/latency checks.
- `iproute2` — provides `dcb`, needed by `mori setup` on bnxt.
- `libgrpc++-dev` + protobuf packages — build defaults to `BUILD_UMBP=ON`,
  whose CMake step needs gRPC headers. (`cmake`/`ninja`/`pybind11` come from
  `pyproject.toml` build isolation automatically.)
- `libopenmpi-dev openmpi-bin` — needed for `MORI_WITH_MPI=ON` /
  `BUILD_BENCHMARK=ON` (benchmarks are gated behind `WITH_MPI`).

---

## Step 3: Install NIC userspace libraries

Run the subsection matching the NIC type detected above:
- `ainic` → **Step 3a**, `thor2`/bnxt → **Step 3b**, `mlx5` → **Step 3c**

---

## Step 3a: Install AINIC userspace libraries (AINIC only)

**Skip if NIC type is not `ainic`.**

> **Recommended version**: for cross-node MORI (EP over RDMA / IBGDA), AINIC firmware
> `>= 1.117.5-a-45` is solid. The `1.117.1` major does **not** support IBGDA — if the
> host is on that branch, flag it to the user and recommend upgrading before proceeding.
> The userspace library (`libionic`) must match the kernel driver version.

### Detect host AINIC version and check against public repo

Run on the **host**:

```bash
IB_DEV=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
if [ -z "$IB_DEV" ]; then
  echo "ERROR: no InfiniBand device found under /sys/class/infiniband/"
  echo "Make sure the ionic kernel module is loaded on the host."
  exit 1
fi
HOST_AINIC_VER=$(cat /sys/class/infiniband/${IB_DEV}/fw_ver 2>/dev/null)
if [ -z "$HOST_AINIC_VER" ]; then
  echo "ERROR: cannot read fw_ver from /sys/class/infiniband/${IB_DEV}/fw_ver"
  exit 1
fi
echo "Host AINIC firmware version: $HOST_AINIC_VER (detected from $IB_DEV)"

AVAILABLE=$(curl -fsSL https://repo.radeon.com/amdainic/pensando/ubuntu/ \
  | grep -oP '(?<=href=")[^"]+(?=/)' \
  | grep -v '^\.\.' | grep -v '^https' | sort)
echo "Available AINIC versions in public repo:"
echo "$AVAILABLE"

if ! echo "$AVAILABLE" | grep -qx "$HOST_AINIC_VER"; then
  echo ""
  echo "ERROR: host AINIC version '$HOST_AINIC_VER' is not available in the public repo."
  echo "Available versions: $(echo $AVAILABLE | tr '\n' ' ')"
  echo "Please contact your AINIC vendor for a matching software bundle."
  exit 1
fi

echo "Found matching version '$HOST_AINIC_VER' in public repo — proceeding."
```

### Install inside container

```bash
sudo docker exec $CONTAINER_NAME bash -c "
set -e
AINIC_VERSION=$HOST_AINIC_VER
UBUNTU_CODENAME=\$(. /etc/os-release && echo \"\$VERSION_CODENAME\")

mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg
echo \"deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] \
    https://repo.radeon.com/amdainic/pensando/ubuntu/\${AINIC_VERSION} \
    \${UBUNTU_CODENAME} main\" > /etc/apt/sources.list.d/amdainic.list

apt-get update
apt-get install -y nicctl libionic-dev ionic-common
ldconfig
ldconfig -p | grep libionic
nicctl --version
"
```

---

## Step 3b: Install Broadcom (bnxt / thor2) userspace libraries + tools

**Skip if NIC type is not `thor2`/bnxt.**

> **Recommended version**: for cross-node MORI (EP over RDMA / IBGDA), Broadcom firmware
> is solid on `237.1.137.x` (official Broadcom release) and `235.2.86.x` (customer-specific
> build). `231.x` is too old for IBGDA — if the host is on that branch, flag it to the user
> and recommend upgrading. The userspace library (`libbnxt_re`, 3b.2) must match the kernel
> driver version detected in 3b.1.

The bnxt path of `mori check` / `mori setup` needs two NIC-specific pieces installed
here: (1) the **RoCE userspace lib** (`libbnxt_re`) and (2) a **recent `niccli`**.
(`dcb`, the third dependency, comes with `iproute2` from Step 2 — just verified in 3b.4.)

### 3b.1 — Detect host bnxt version (match the userspace lib to it)

Run on the **host**:

```bash
modinfo -F version bnxt_re 2>/dev/null || cat /sys/module/bnxt_re/version
# e.g. 235.2.88.0  → install the closest bnxt-rocelib (e.g. 235.2.86.0)
```

### 3b.2 — Install RoCE userspace lib via the Broadcom apt repo

```bash
sudo docker exec $CONTAINER_NAME bash -c '
set -e
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public \
    -o /etc/apt/keyrings/broadcom-nic.asc
chmod a+r /etc/apt/keyrings/broadcom-nic.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] \
https://packages.broadcom.com/artifactory/ethernet-nic-debian-public jammy main" \
    > /etc/apt/sources.list.d/broadcom-nic.list
apt-get update
# pin to match the host bnxt_re version from 3b.1; list options: apt-cache madison bnxt-rocelib
apt-get install -y ibverbs-utils bnxt-rocelib=235.2.86.0
# mori check looks for libbnxt_re-<ver>.so under /usr/local/lib — make it visible there
cp /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/.
ldconfig
'
```

> Replace `235.2.86.0` with the version matching the host (3b.1). If gone from
> the repo, pick the nearest from `apt-cache madison bnxt-rocelib`.

### 3b.3 — Install a recent `niccli` (must support the `qos` subcommand)

`mori check` Step 2 uses `niccli ... qos`, so install one tracking the
firmware (236.x/237.x for BCM57608):

```bash
sudo docker exec $CONTAINER_NAME bash -c '
set -e
NICCLI_VER=237.1.145.0          # deb version
NICCLI_PKG=237.1.148.0          # BRCM_<this> path segment in the URL
URL="https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/BRCM_${NICCLI_PKG}/niccli/Linux/niccli-${NICCLI_VER}_linux.zip"
cd /tmp && wget -q -O niccli.zip "$URL" && unzip -o -q niccli.zip -d niccli
dpkg -i "$(find ./niccli -name "niccli_*_x86_64.deb" | head -1)"
niccli -l | head            # must list all NICs
niccli -i 1 qos --ingress --cosq --show | head   # should print the CoSQ table (TC/State/Mode)
'
```

> Alternative (no download): host usually has a working niccli at
> `/opt/niccli`. `sudo docker cp /opt/niccli $CONTAINER_NAME:/opt/` then
> `docker exec $CONTAINER_NAME ln -sf /opt/niccli/niccli /usr/bin/niccli`.

### 3b.4 — `dcb` (iproute2), needed by `mori setup`

```bash
sudo docker exec $CONTAINER_NAME bash -c "command -v dcb"
```

> `mori setup` configures PFC/ETS via `dcb` and DCQCN via configfs, both at
> host level. With `--network=host` the container shares the host netns, so
> running it in-container is equivalent — just needs `dcb` + Step 1's mounts.

---

## Step 3c: Install Mellanox/NVIDIA (mlx5) userspace libraries + tools

**Skip if NIC type is not `mlx5`.**

The mlx5 RoCE userspace provider is **inbox** (Ubuntu's `rdma-core`/
`ibverbs-providers`, already pulled in by Step 2, includes `libmlx5`) — RDMA
works with no vendor repo at all. What's still missing for the full `mori
check`/`mori setup` flow are two small, optional, standalone tools (**neither
needs a full MLNX_OFED install**):

- **`mlnx_qos`** (package `mlnx-tools`) — reads/sets trust/PFC/DSCP state on RoCE ports.
- **`mlxconfig`/`mst`** (NVIDIA MFT) — reads/sets `ROCE_CC_PRIO_MASK_P1`/`CNP_DSCP_P1`
  from firmware NV config.

Both optional: if missing, `mori check` logs `[WARN]` and skips those checks
(and `mori setup` skips the corresponding fix). If native IB ports outnumber
RoCE ports, `mori check` treats RoCE port(s) as incidental and skips
QoS/DCQCN anyway — Step 3c becomes unnecessary.

> **`mori setup`'s mlx5 path** sets trust=dscp, maps the RoCE DSCP to its
> priority, enables PFC no-drop on it, leaves TC arbitration on `vendor`
> (no bandwidth split), and flips the `ROCE_CC_PRIO_MASK_P1` enable bit for
> DCQCN via `mlxconfig` — CNP DSCP and the CC algorithm itself are left at
> firmware defaults. A `mlxconfig` NV-config change needs a firmware reset
> (`mlxfwreset -d <pci> -y reset`) or reboot to take effect; `mori setup`
> only warns about this, it doesn't reset the NIC itself.

### 3c.1 — Detect host mlx5 driver/firmware version (informational only)

```bash
cat /sys/class/infiniband/mlx5_*/fw_ver 2>/dev/null | sort -u   # firmware version(s)
modinfo -F version mlx5_core 2>/dev/null                        # driver version
```

### 3c.2 — Install `mlnx_qos` (package: `mlnx-tools`)

**Build from source from the upstream GitHub repo** — don't pull full
MLNX_OFED just for this tiny, dependency-free, pure-Python tool. No version
coupling to the host driver/firmware.

```bash
sudo docker exec $CONTAINER_NAME bash -c '
set -e
apt-get install -y --no-install-recommends make
cd /tmp && rm -rf mlnx-tools
git clone --depth 1 https://github.com/Mellanox/mlnx-tools.git
cd mlnx-tools && make install
'
```

> `make install` places `mlnx_qos` at `/usr/bin/mlnx_qos`, helpers at
> `/usr/share/mlnx-tools/python/` (script adds this to `sys.path` itself).

**Sanity check — resolve a real netdev first.** `mlnx_qos -i <netdev>` needs
an actual interface name; there's no way to hardcode one since it depends on
which mlx5 device you pick. Resolve it from `/sys/class/infiniband/` instead
of guessing:

```bash
sudo docker exec $CONTAINER_NAME bash -c '
IB_DEV=$(ls /sys/class/infiniband/ | grep mlx5 | head -1)
NETDEV=$(ls /sys/class/infiniband/$IB_DEV/device/net/ | head -1)
echo "Using $IB_DEV -> $NETDEV"
mlnx_qos -i "$NETDEV"
'
```

### 3c.3 — Install NVIDIA MFT (`mlxconfig`/`mst`)

Proprietary/standalone, not in any apt repo — download the tarball and run
its installer. **No firmware/driver version matching needed**: MFT is broadly
backward/forward compatible across generations (verified: MFT 4.30.1 drives
ConnectX-7 firmware 28.40.1702 fine). Just match arch/package type to the
container OS; browse
https://network.nvidia.com/products/adapter-software/firmware-tools/ if the
URL below is stale.

> **Prerequisite (fails silently without it):** `mst start` shells out to
> `lsmod`/`modprobe`/`udevadm`. Minimal images (e.g. `rocm/pytorch`) lack
> these — `install.sh` still reports success and `mst start` exits 0 despite
> printing `No such file or directory`/`command not found`, but `mst status
> -v` then shows `MST` column as `NA` and `/dev/mst/` stays empty (silently
> broken `mlxconfig`/`mst`). Install first:
> ```bash
> sudo docker exec $CONTAINER_NAME bash -c "apt-get install -y --no-install-recommends kmod udev usbutils"
> ```
> Verify `mst status -v` shows real paths (e.g. `/dev/mst/mt4129_pciconf0`),
> not `NA`, before trusting `mlxconfig`/`mst` output.

```bash
sudo docker exec $CONTAINER_NAME bash -c '
set -e
MFT_VER=4.30.1-1216
ARCH=x86_64
URL="https://www.mellanox.com/downloads/MFT/mft-${MFT_VER}-${ARCH}-deb.tgz"
cd /tmp && curl -fsSL -O "$URL" && tar -xzf mft-*-deb.tgz
cd mft-*-deb && ./install.sh --without-kernel
mst start
mst status -v
'
```

> `--without-kernel` skips the optional `mst_pci` kernel module — MFT falls
> back to plain PCI-config-space access, covered by Step 1's `--privileged`.

---

## Step 4: Install MORI

`pybind11` is a required build dep missing from `pyproject.toml`:

```bash
sudo docker exec -w $MORI_REPO_DIR $CONTAINER_NAME bash -c "
pip install pybind11 -q
rm -rf build   # clear stale cmake cache — old build/ can hardcode a wrong ROCm version
pip install .
"
```

**UMBP/gRPC:** build defaults to `BUILD_UMBP=ON` (needs gRPC headers from
Step 2). To skip the storage component and its gRPC dep:

```bash
sudo docker exec -w $MORI_REPO_DIR $CONTAINER_NAME bash -c "BUILD_UMBP=OFF pip install ."
```

---

## Step 5: Verify

```bash
sudo docker exec $CONTAINER_NAME python3 -c "import mori; print('mori version:', mori.__version__)"
```

On shared-library errors (`libpci.so`, `libibverbs.so`, …):

```bash
sudo docker exec $CONTAINER_NAME bash -c "
ldd \$(python3 -c \"import mori._C; print(mori._C.__file__)\") | grep 'not found'
ldconfig
"
```

---

## Step 6: Show detected hardware

```bash
sudo docker exec $CONTAINER_NAME bash -c "
python3 -c \"
from mori.jit.config import detect_build_config, detect_nic_type
cfg = detect_build_config()
print(f'GPU arch : {cfg.arch}')
print(f'NIC type : {detect_nic_type()}')
\"
ibv_devinfo | head -20
"
```

---

## Step 7: Run `mori check` / `mori setup`

```bash
sudo docker exec $CONTAINER_NAME bash -c "mori check"
```

`mori check` validates the RDMA stack in 6 steps (vendor-specific variants,
same intent):

1. **firmware & driver** — versions consistent
2. **QoS / SL / TC** — PFC + lossless TC; selects SL/TC for MORI
3. **DCQCN** — congestion control enabled on all RoCE devices
4. **intra-node bandwidth** — `ib_write_bw` full mesh (needs `perftest`)
5. **inter-node bandwidth** — `ib_write_bw` to a peer: `mori check <peer_ip>`
6. **inter-node latency** — `ib_write_lat` to a peer: same peer IP

Steps 5/6 are skipped without a peer IP — run from both nodes to test cross-node connectivity.

**mlx5 note:** native IB ports don't use PFC/DSCP/DCQCN (IB has its own
credit-based flow control managed by the fabric SM) — steps 2/3 only run
against Ethernet/RoCE ports, and are skipped entirely if IB ports outnumber
RoCE ports (RoCE treated as an incidental management NIC). Steps 1/4/5/6 still
cover every mlx5 device.

> **Step 4 still probes incidental RoCE/management ports**, unlike 2/3 — the
> full-mesh test tries every pair regardless of link type or physical
> network, so any port not on the main RDMA fabric shows unreachable (`✗`)
> against every fabric port. This happens even with **zero native IB** —
> e.g. a host with 8 ConnectX-7 fabric ports plus 2 ConnectX-6 Dx management
> ports (`eth0`/`eth1` — a different HW generation, visible as its own
> firmware-family group in Step 1's output) showed `[WARN] intra-node BW:
> 34/90 unreachable`, with every `✗` confined to those two ports'
> rows/columns — including the mgmt↔mgmt pair itself (also unreachable,
> since the two management ports aren't on the same subnet as each other
> either). Expected, not a fabric problem — confirm the `✗` cells are only
> in incidental ports' rows/columns.
>
> General formula for the unreachable count: `N = n×(n-1) − f×(f-1)` where
> `n` = total local RDMA devices and `f` = devices actually on the fabric
> (equivalently `N = k×(k-1) + 2×k×f` for `k = n−f` incidental ports). This
> is **not** `2 × incidental_ports × fabric_ports` — that simpler formula
> only holds for exactly one incidental port; with 2+ incidental ports it
> undercounts by missing the incidental↔incidental pairs (e.g. predicts 32
> instead of the actual 34 in the example above).

If any step shows `[FAIL]`:

- `sudo docker exec $CONTAINER_NAME bash -c "mori setup"` auto-applies
  QoS/PFC/DCQCN on ionic/bnxt/mlx5. Env vars don't persist in the calling
  shell; use `source $(mori setup --path)` to export them. On mlx5, a
  DCQCN fix may need a firmware reset/reboot to take effect — see 3c.
- Still failing: `sudo docker exec $CONTAINER_NAME bash -c "mori diagnose"`.

### mlx5 hardware faults (CQ errors / firmware health)

`dmesg` repeating `cq_err_event_notifier: CQ error ..., syndrome 0x2`
(LOCAL_QP_OP_ERR) or `poll_health: device's health compromised` means the
card's *firmware* faulted. `mlxfwreset -d <mst_device> q` will likely show
only `4: Warm Reboot` supported — a host reboot is the only recovery path.

> **Caveat (hit on this box):** an OS-level `reboot` fixed the mlx5 fault but
> left every GPU on the same baseboard (AMD MI300/Aqua Vanjaram) failing to
> probe (`amdgpu: get invalid ip discovery binary signature`, all GPUs at
> once) — a warm reboot doesn't fully power-cycle the GPU baseboard's PSP.
> Neither `modprobe -r/+ amdgpu` nor per-device PCIe reset
> (`echo 1 > /sys/bus/pci/devices/0000:XX:00.0/reset`) fixed it; only a real
> BMC power cycle did: `sudo ipmitool chassis power cycle`. This is far more
> disruptive than `reboot` (drops every user's session on a shared host) —
> confirm no one else needs the box, or escalate to the hardware owner instead.

---

## Done — Report Back

- Base image and OS
- NIC library installed (`libionic` / `libbnxt_re` / mlx5 inbox `libmlx5` + optional `mlnx-tools`/MFT / none)
- Install mode: source (`pip install .`)
- GPU arch and NIC type as reported by MORI
- Kernels: JIT on first use (`~/.mori/jit/`)
- `mori check` result — include full output, highlight any `[WARN]` or `[FAIL]`
- Attach command (working directory set to the MORI source tree):

```bash
sudo docker exec -it -w "$MORI_REPO_DIR" $CONTAINER_NAME bash
```

**Built-in tools:** `mori check`, `mori setup`, `mori diagnose` — see Step 7 for usage.
