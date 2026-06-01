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
contains `pyproject.toml`; ask the user otherwise.

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
if sudo docker inspect $CONTAINER_NAME &>/dev/null; then
  sudo docker start $CONTAINER_NAME
else
  sudo docker run <flags> --name $CONTAINER_NAME $IMAGE_NAME sleep infinity
fi
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
sudo docker exec $CONTAINER_NAME bash -c "apt-get update && apt-get install -y --no-install-recommends \
    git libpci-dev libdw1 libibverbs-dev ibverbs-utils \
    locales iputils-ping iproute2 ethtool"
```

---

## Step 3: Install rdma-core and nicctl (AINIC only)

**Skip if NIC type is not `ainic`.**

### Extract the AINIC bundle

First detect the firmware version from the **host** sysfs (outside container), then run everything else inside the container in one `docker exec` block:

```bash
# Run on HOST to detect firmware version
IB_DEV=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
FW_VER=$(cat /sys/class/infiniband/${IB_DEV}/fw_ver 2>/dev/null)
echo "Detected IB device: $IB_DEV  firmware version: $FW_VER"

# Search each root sequentially on HOST so /apps always wins over /home
for ROOT in /apps /shared /mnt /home; do
  BUNDLE=$(find "$ROOT" -maxdepth 3 -not -path "*/.snapshot/*" \
    -name "ainic_bundle_${FW_VER}*.tar.gz" 2>/dev/null | head -1)
  [ -n "$BUNDLE" ] && break
done
# Fallback: pick newest bundle if no version match found
if [ -z "$BUNDLE" ]; then
  for ROOT in /apps /shared /mnt /home; do
    BUNDLE=$(find "$ROOT" -maxdepth 3 -not -path "*/.snapshot/*" \
      -name "ainic_bundle_*.tar.gz" 2>/dev/null | sort -V | tail -1)
    [ -n "$BUNDLE" ] && break
  done
fi
echo "Using bundle: $BUNDLE"
```

Then run the full install inside the container (pass `$BUNDLE` in):

```bash
sudo docker exec $CONTAINER_NAME bash -c "
set -e
BUNDLE=$BUNDLE
BUNDLE_DIR=\"/tmp/\$(basename \"\$BUNDLE\" .tar.gz)\"

# Extract bundle into /tmp (guaranteed writable)
[ -d \"\$BUNDLE_DIR\" ] || tar xf \"\$BUNDLE\" -C /tmp
[ -d \"\$BUNDLE_DIR/host_sw_pkg\" ] || tar xf \"\$BUNDLE_DIR/host_sw_pkg.tar.gz\" -C \"\$BUNDLE_DIR\"
[ -d \"\$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux\" ] || \
    tar xf \"\$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux.tar.xz\" \
        -C \"\$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/\"

# Build deps for rdma-core
apt-get install -y --no-install-recommends \
    cmake ninja-build python3-docutils libudev-dev pkg-config dh-python pandoc

# Build and install rdma-core
RDMA_SRC=\"\$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux/rdma-core\"
mkdir -p \"\$RDMA_SRC/build\" && cd \"\$RDMA_SRC/build\"
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 ..
ninja && ninja install && ldconfig
ldconfig -p | grep libibverbs

# Install nicctl
CODENAME=\$(. /etc/os-release && echo \"\$VERSION_CODENAME\")
NICCTL_DEB=\$(find \"\$BUNDLE_DIR/host_sw_pkg/nicctl/deb/\$CODENAME\" -name 'nicctl_*.deb' | head -1)
dpkg -i \"\$NICCTL_DEB\"
nicctl --version
"
```

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

With MPI (if `--with-mpi` was passed):

```bash
sudo docker exec -w $MORI_REPO_DIR $CONTAINER_NAME bash -c "
pip install pybind11 -q
rm -rf build
MORI_WITH_MPI=ON pip install .
"
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

## Step 6 (optional): Precompile GPU kernels

Only if `--precompile` was passed. Recommended when baking a Docker image.

```bash
sudo docker exec $CONTAINER_NAME bash -c "MORI_PRECOMPILE=1 python3 -c 'import mori'"
```

Kernels cache to `~/.mori/jit/` and are reused on every subsequent run.

---

## Step 7: Show detected hardware

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

## Done — Report Back

- Base image and OS
- ROCm version
- NIC library installed (`libionic` / `libmlx5` / `libbnxt_re` / none)
- rdma-core: distro package or built from bundle
- Install mode: source (`pip install .`)
- GPU arch and NIC type as reported by MORI
- Kernels: precompiled or JIT on first use (`~/.mori/jit/`)
- Attach command (working directory set to the MORI source tree):

```bash
sudo docker exec -it -w "$MORI_REPO_DIR" $CONTAINER_NAME bash
```
