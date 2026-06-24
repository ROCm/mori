---
name: deploy-mori
description: >-
  Deploy and set up the MORI environment in a fresh Docker container or bare host:
  start the container, install ROCm dependencies, NIC userspace libraries
  (AINIC/Broadcom), RDMA-core, and MORI itself. Use when the user asks
  to deploy MORI, install MORI in a container, set up a fresh dev environment
  for MORI, or prepare an AINIC / Thor2 box for MORI.
---

# Deploy MORI

You are helping the user deploy MORI inside a Docker container.

**Locate `MORI_REPO_DIR`**: default to the current working directory if it
contains `pyproject.toml`; ask the user otherwise.

**Detect NIC type** — determines whether Step 3 is needed:

```bash
lspci | grep -iE "pensando|ionic|dsc|pollara" && echo "→ ainic"
lspci | grep -iE "broadcom.*thor|bnxt"         && echo "→ thor2"
```

---

## Step 1: Start the Docker container

Use the container name from the user's args if provided; ask if not.

Default image: `rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.10.0`
(use this unless the user specifies another).

Check if the container already exists:

```bash
if sudo docker inspect $CONTAINER_NAME &>/dev/null; then
  sudo docker start $CONTAINER_NAME
else
  sudo docker run <flags> --name $CONTAINER_NAME $IMAGE_NAME sleep infinity
fi
```

Probe optional mount paths on the host and only include ones that exist:

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
- `--ulimit memlock=-1:-1` — RDMA pins memory when creating QPs; required for the
  parallel `mori check` bandwidth mesh.
- `-v /sys/kernel/config` + `-v /sys/kernel/debug` — **required for bnxt (Broadcom)**:
  `mori check` (DCQCN) and `mori setup` read/write congestion control via configfs.
  `--privileged` does not propagate these kernel filesystems on its own. If the host
  hasn't mounted them, mount on the host first:
  `sudo mount -t configfs none /sys/kernel/config; sudo mount -t debugfs none /sys/kernel/debug`.
- `--rm` — data outside mounted volumes is lost on stop.

All subsequent steps run **inside** `$CONTAINER_NAME` via `docker exec`.

---

## Step 2: Install base system packages

```bash
sudo docker exec $CONTAINER_NAME bash -c "apt-get update && apt-get install -y --no-install-recommends \
    git libpci-dev pciutils sudo libdw1 libibverbs-dev ibverbs-utils \
    locales iputils-ping iproute2 ethtool jq perftest \
    wget unzip ca-certificates curl \
    libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler \
    libopenmpi-dev openmpi-bin"
```

Package roles:
- `jq` — required by `mori check`.
- `pciutils` — provides `lspci`, which `nicctl` shells out to in order to enumerate
  the NIC cards. Without it `nicctl` fails with `Invalid card handle` and the ionic
  firmware / QoS / DCQCN checks in `mori check` / `mori setup` break.
- `sudo` — the ionic and bnxt paths of `mori check` / `mori setup` invoke `nicctl`,
  `dcb`, `ethtool`, and sysfs writes via `sudo`.
- `perftest` — provides `ib_write_bw` / `ib_write_lat` for intra/inter-node bandwidth + latency checks.
- `iproute2` — provides `dcb`, required by `mori setup` on bnxt NICs.
- `wget unzip ca-certificates curl` — used by the NIC userspace install steps (Step 3a/3b).
- `libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler` — **mori build deps**:
  the build defaults to `BUILD_UMBP=ON`, whose CMake step needs gRPC headers
  (`grpcpp/grpcpp.h`). The remaining build tooling (`cmake`, `ninja`, `pybind11`) is
  pulled in automatically by `pyproject.toml`'s build isolation, so it doesn't need
  to be in the image.
- `libopenmpi-dev openmpi-bin` — **MPI deps**: required for `MORI_WITH_MPI=ON` and
  `BUILD_BENCHMARK=ON` (benchmark targets are gated behind `WITH_MPI`).

---

## Step 3: Install NIC userspace libraries

Run the subsection matching the NIC type detected at the top:
- `ainic` → **Step 3a** (AINIC / Pensando)
- `thor2` / bnxt → **Step 3b** (Broadcom NetXtreme-E)

---

## Step 3a: Install AINIC userspace libraries (AINIC only)

**Skip if NIC type is not `ainic`.**

### Detect host AINIC version and check against public repo

Run on the **host** (outside container):

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

**Skip if NIC type is not `thor2` / bnxt.**

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

> Replace `235.2.86.0` with the version matching the host (3b.1). If that exact
> version is gone from the repo, pick the nearest one from `apt-cache madison bnxt-rocelib`.

### 3b.3 — Install a recent `niccli` (must support the `qos` subcommand)

`mori check` Step 2 uses `niccli ... qos`, so install a niccli whose version tracks
the firmware (236.x / 237.x for BCM57608). Parameterize the version:

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

> Alternative (no download): the host usually already has a working niccli at
> `/opt/niccli` (self-contained). You can `sudo docker cp /opt/niccli $CONTAINER_NAME:/opt/`
> then `docker exec $CONTAINER_NAME ln -sf /opt/niccli/niccli /usr/bin/niccli`.

### 3b.4 — `dcb` (iproute2), needed by `mori setup`

`mori setup` configures bnxt PFC/ETS via `dcb`, which ships with `iproute2`
(installed in Step 2). Verify it's present:

```bash
sudo docker exec $CONTAINER_NAME bash -c "command -v dcb"
```

> `mori setup` does host-level NIC configuration (PFC/ETS on the netdev via `dcb`,
> DCQCN via configfs). With `--network=host` the container shares the host's
> netns, so it's equivalent to configure on the host. Either run `mori setup` on
> the host, or ensure the container has `dcb` + configfs/debugfs (Step 1 mounts).

---

## Step 4: Install MORI

`pybind11` is a required build dep missing from `pyproject.toml`. Run inside the container:

```bash
sudo docker exec -w $MORI_REPO_DIR $CONTAINER_NAME bash -c "
pip install pybind11 -q
rm -rf build   # clear stale cmake cache — old build/ can hardcode a wrong ROCm version
pip install .
"
```

**UMBP / gRPC:** the build defaults to `BUILD_UMBP=ON`, whose CMake step needs gRPC
headers (installed in Step 2). To build without the UMBP storage component instead
(no gRPC needed):

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

`mori check` validates the full RDMA stack in 6 steps (vendor-specific variants
for ionic vs bnxt, same intent):

1. **firmware & driver** — firmware/driver/userspace-lib versions consistent
2. **QoS / SL / TC** — PFC + lossless TC; selects the SL/TC for MORI to use
3. **DCQCN** — congestion control enabled on all RoCE devices
4. **intra-node bandwidth** — `ib_write_bw` between all local NIC pairs (requires `perftest`)
5. **inter-node bandwidth** — `ib_write_bw` to a peer node; pass peer IP: `mori check <peer_ip>`
6. **inter-node latency** — `ib_write_lat` to a peer node; same peer IP

Steps 5 and 6 are skipped when no peer IP is given — run `mori check <peer_ip>` from both nodes to test cross-node connectivity.

If any step shows `[FAIL]`:

- Run `mori setup` to auto-apply recommended QoS/PFC/DCQCN settings:
  ```bash
  sudo docker exec $CONTAINER_NAME bash -c "mori setup"
  ```
  Note: env vars (`MORI_RDMA_SL`/`TC`) do **not** persist in the calling shell. To export them, use `source $(mori setup --path)` instead.

- If the issue persists, run `mori diagnose` for deeper diagnostics:
  ```bash
  sudo docker exec $CONTAINER_NAME bash -c "mori diagnose"
  ```

---

## Done — Report Back

- Base image and OS
- NIC library installed (`libionic` / `libbnxt_re` / none)
- Install mode: source (`pip install .`)
- GPU arch and NIC type as reported by MORI
- Kernels: JIT on first use (`~/.mori/jit/`)
- `mori check` result — include full output, highlight any `[WARN]` or `[FAIL]`
- Attach command (working directory set to the MORI source tree):

```bash
sudo docker exec -it -w "$MORI_REPO_DIR" $CONTAINER_NAME bash
```

**Built-in tools:** `mori check`, `mori setup`, `mori diagnose` — see Step 7 for usage.
