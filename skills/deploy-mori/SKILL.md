---
name: deploy-mori
description: >-
  Deploy and set up the MORI environment in a fresh Docker container or bare host:
  start the container, install ROCm dependencies, NIC userspace libraries
  (AINIC/Mellanox/Broadcom), RDMA-core, and MORI itself. Use when the user asks
  to deploy MORI, install MORI in a container, set up a fresh dev environment
  for MORI, or prepare an AINIC / ConnectX / Thor2 box for MORI.
---

# Deploy MORI

You are helping the user deploy MORI inside a Docker container. Optional inputs:

- **`--precompile`**: precompile GPU kernels after install
- **`--with-mpi`**: enable MPI support
- **`--nic=<type>`**: `ainic` (Pensando DSC/Pollara), `cx7` (Mellanox ConnectX-7), `thor2` (Broadcom)

**Locate `MORI_REPO_DIR`**: default to the current working directory if it
contains `setup.py`; ask the user otherwise.

**Detect NIC type** (unless `--nic` was given) — determines whether Step 3 is needed:

```bash
lspci | grep -iE "pensando|ionic|dsc|pollara" && echo "→ ainic"
lspci | grep -iE "mellanox|connectx"           && echo "→ cx7"
lspci | grep -iE "broadcom.*thor|bnxt"         && echo "→ thor2"
```

---

## Step 1: Start the Docker container

**Ask the user for a container name — mandatory before proceeding.**

Default image: `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.8.0`
(use this unless the user specifies another).

Check if the container already exists:

```bash
sudo docker inspect $CONTAINER_NAME &>/dev/null \
  && sudo docker start $CONTAINER_NAME \
  || sudo docker run $(build_flags) --name $CONTAINER_NAME $IMAGE_NAME sleep infinity
```

Probe optional mount paths on the host and only include ones that exist:

```bash
for p in /shared /apps /dev/infiniband; do
  [ -e "$p" ] && echo "EXISTS: $p" || echo "MISSING: $p"
done
```

Full `docker run` flags (omit missing paths):

```bash
sudo docker run \
    --group-add video \
    --network=host \
    --ulimit nproc=100000:100000 \
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
    --rm \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    sleep infinity
```

Notes:
- `--network=host` + `--device=/dev/infiniband` required for RDMA visibility.
- `--rm` — data outside mounted volumes is lost on stop.
- `nicctl show version host-software` inside the container **will not show**
  `ionic driver` version — nicctl reads it via the `pds_core` IPC socket
  (not mounted). Use `cat /sys/module/ionic/version` instead, or add
  `-v /var/run/pds:/var/run/pds` to the `docker run` command.

All subsequent steps run **inside** `$CONTAINER_NAME` via `docker exec`.

---

## Step 2: Install base system packages

```bash
apt-get update && apt-get install -y --no-install-recommends \
    git libpci-dev libdw1 libibverbs-dev ibverbs-utils \
    locales iputils-ping iproute2 ethtool
```

---

## Step 3: Install rdma-core and nicctl (AINIC only)

**Skip if NIC type is not `ainic`.**

### Extract the AINIC bundle

First detect the firmware version from the host sysfs, then find the matching bundle:

```bash
# Read firmware version from host — pick first available IB device dynamically
IB_DEV=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
FW_VER=$(cat /sys/class/infiniband/${IB_DEV}/fw_ver 2>/dev/null)
echo "Detected IB device: $IB_DEV  firmware version: $FW_VER"

# Find bundle matching full firmware version (e.g. "1.117.5-a-77"); fall back to latest if no match
if [ -n "$FW_VER" ]; then
  BUNDLE=$(find /shared /apps /mnt /home -name "ainic_bundle_${FW_VER}*.tar.gz" 2>/dev/null | head -1)
fi
# Fallback: pick newest bundle if version-matched bundle not found
[ -z "$BUNDLE" ] && BUNDLE=$(find /shared /apps /mnt /home -name "ainic_bundle_*.tar.gz" 2>/dev/null | sort -V | tail -1)

echo "Using bundle: $BUNDLE"
WORK=$(dirname "$BUNDLE")
BUNDLE_DIR="$WORK/$(basename "$BUNDLE" .tar.gz)"

[ -d "$BUNDLE_DIR" ] || tar xf "$BUNDLE" -C "$WORK"
[ -d "$BUNDLE_DIR/host_sw_pkg" ] || tar xf "$BUNDLE_DIR/host_sw_pkg.tar.gz" -C "$BUNDLE_DIR"
[ -d "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux" ] || \
    tar xf "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux.tar.xz" \
        -C "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/"
```

### Build and install rdma-core

```bash
apt-get install -y --no-install-recommends \
    cmake ninja-build python3-docutils libudev-dev pkg-config dh-python pandoc

RDMA_SRC="$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux/rdma-core"
mkdir -p "$RDMA_SRC/build" && cd "$RDMA_SRC/build"
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 ..
ninja && ninja install && ldconfig

ldconfig -p | grep libibverbs  # verify
```

### Install nicctl

```bash
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
NICCTL_DEB=$(find "$BUNDLE_DIR/host_sw_pkg/nicctl/deb/$CODENAME" -name "nicctl_*.deb" | head -1)
dpkg -i "$NICCTL_DEB"
nicctl --version  # verify
```

---

## Step 4: Install MORI

`pybind11` is a required build dep missing from `pyproject.toml` — install it first:

```bash
pip install pybind11
```

```bash
# Clear stale cmake cache if present — old build/ can hardcode a wrong ROCm version
rm -rf "$MORI_REPO_DIR/build"
cd "$MORI_REPO_DIR" && pip install .
```

---

## Step 5: Verify

```bash
python3 -c "import mori; print('mori version:', mori.__version__)"
```

On shared-library errors (`libpci.so`, `libibverbs.so`, …):

```bash
ldd $(python3 -c "import mori._C; print(mori._C.__file__)") | grep "not found"
ldconfig
```

---

## Step 6 (optional): Precompile GPU kernels

Only if `--precompile` was passed. Recommended when baking a Docker image.

```bash
MORI_PRECOMPILE=1 python3 -c "import mori"
```

Kernels cache to `~/.mori/jit/` and are reused on every subsequent run.

---

## Step 7: Show detected hardware

```bash
python3 -c "
from mori.jit.config import detect_build_config, detect_nic_type
cfg = detect_build_config()
print(f'GPU arch : {cfg.arch}')
print(f'NIC type : {detect_nic_type()}')
"
ibv_devinfo | head -20
```

---

## Done — Report Back

- Base image and OS
- ROCm version
- NIC library installed (`libionic` / `libmlx5` / `libbnxt_re` / none)
- rdma-core: distro package or built from bundle
- Install mode (`source` / `pypi` / `nightly`)
- GPU arch and NIC type as reported by MORI
- Kernels: precompiled or JIT on first use (`~/.mori/jit/`)
- Attach command (working directory set to the MORI source tree):

```bash
sudo docker exec -it -w "$MORI_REPO_DIR" $CONTAINER_NAME bash
```
