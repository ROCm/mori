---
description: Deploy and set up the MORI environment in a fresh Docker container or bare host. Installs ROCm dependencies, NIC userspace libraries (AINIC/Mellanox/Broadcom), RDMA-core, and MORI itself.
argument-hint: [source|pypi|nightly] [--precompile] [--with-mpi] [--nic=ainic|cx7|thor2]
---

You are helping the user deploy the MORI environment inside a Docker container. Follow these steps based on `$ARGUMENTS`.

## Parse arguments

- Install mode (default: `source`): `source` | `pypi` | `nightly`
- `--precompile`: precompile GPU kernels after install
- `--with-mpi`: enable MPI support
- `--nic=<type>`: NIC to prepare userspace lib for — `ainic` (Pensando DSC/Pollara), `cx7` (Mellanox ConnectX-7), `thor2` (Broadcom); omit to auto-detect

If the user has not specified a NIC type, proceed without stopping — MORI auto-detects at runtime.

---

## Step 1: Start the Docker container

**First, ask the user for a container name — this is mandatory. Do not proceed until you have it.**

> What should the Docker container be named? (e.g. `mori-dev`)

Once you have the name (`$CONTAINER_NAME`), determine which container to use:

### Case A — user provides an existing container

If the user specifies an existing container image or running container, exec into it:

```bash
# If already running:
docker exec -it $CONTAINER_NAME bash

# If stopped, restart then exec:
docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME bash
```

### Case B — no existing container provided

Ask the user for the image name (`$IMAGE_NAME`). If not provided, build from `docker/Dockerfile.dev`:

```bash
cd /home/zqz/workspace/mori
docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
# then use IMAGE_NAME=rocm/mori:dev
```

**Before running, check which optional mount paths exist on the host** and only include them if present:

```bash
for p in /shared; do
  [ -e "$p" ] && echo "EXISTS: $p" || echo "MISSING: $p"
done
```

Start the container (omit any `-v` or `--device` lines for paths that do not exist):

```bash
docker run \
    --group-add video \
    --network=host \
    --ulimit nproc=100000:100000 \
    --pids-limit=-1 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \   # omit if /dev/infiniband does not exist
    --ipc=host \
    --privileged -d \
    -v /home/:/home/ \
    -v /root:/root \
    -v /mnt:/mnt \
    -v /shared:/shared \         # omit if /shared does not exist
    -v /apps:/apps \             # omit if /apps does not exist
    -v /lib/modules:/lib/modules \
    --rm \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    sleep infinity
```

> `--privileged` + `--device=/dev/infiniband` exposes RDMA/AINIC devices.
> `--network=host` is required for RDMA/IBGDA device visibility.
> `/home/` is mounted so user home directories (including the MORI repo) are accessible at the same paths as the host.
> `--rm` means the container is removed when stopped — data outside mounted volumes will be lost.

All subsequent steps run **inside** `$CONTAINER_NAME` via `docker exec`.

---

## Step 2: Install base system packages

```bash
apt-get update && apt-get install -y --no-install-recommends \
    git \
    libpci-dev \
    libdw1 \
    libibverbs-dev \
    ibverbs-utils \
    locales \
    iputils-ping \
    iproute2 \
    ethtool
```

`libibverbs-dev` provides `libibverbs.so` — required for any RDMA/IBGDA operation.
`ibverbs-utils` provides `ibv_devinfo` for diagnosing RDMA device visibility.


---

## Step 3: Install RDMA-core and nicctl (AINIC only)

**Skip this step entirely if NIC type is not `ainic`.**

### Locate and extract the AINIC bundle

First, get the firmware version from the host to know which bundle version to look for:

```bash
# Run on HOST (before exec-ing into the container)
nicctl show version firmware
```

Note the version string (e.g. `1.117.5-a-82`), then search for the matching bundle globally inside the container:

```bash
# Step 1: find and extract the outer bundle (search globally)
BUNDLE=$(find / -name "ainic_bundle_*.tar.gz" 2>/dev/null | head -1)
echo "Found bundle: $BUNDLE"
WORK=$(dirname "$BUNDLE")
tar xf "$BUNDLE" -C "$WORK"

# The outer bundle extracts to a directory named after the tarball, e.g.:
#   ainic_bundle_1.117.5-a-82/
BUNDLE_DIR="$WORK/$(basename "$BUNDLE" .tar.gz)"

# Step 2: extract host_sw_pkg.tar.gz inside it
tar xf "$BUNDLE_DIR/host_sw_pkg.tar.gz" -C "$BUNDLE_DIR"

# Step 3: extract drivers-linux.tar.xz to get rdma-core source
tar xf "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux.tar.xz" \
    -C "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/"
```

### Install rdma-core from the bundle

```bash
RDMA_SRC="$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux/rdma-core"

# Install build dependencies
apt-get install -y --no-install-recommends cmake ninja-build python3-docutils \
    libudev-dev pkg-config dh-python pandoc

cd "$RDMA_SRC"
mkdir -p build
cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX:PATH=/usr -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 ..
ninja || exit 1
ninja install || exit 1
ldconfig
```

Verify:
```bash
ldconfig -p | grep libibverbs
```

### Install nicctl from the bundle

nicctl ships as a `.deb` under `host_sw_pkg/nicctl/deb/<distro>/`. Detect the OS codename and install the matching package:

```bash
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")  # e.g. noble, jammy
NICCTL_DEB=$(find "$BUNDLE_DIR/host_sw_pkg/nicctl/deb/$CODENAME" -name "nicctl_*.deb" | head -1)
dpkg -i "$NICCTL_DEB"
```

Verify:
```bash
nicctl --version
```

---

## Step 4: Install MORI

**source** (default, from current directory):
```bash
cd /home/zqz/workspace/mori
pip install .
```

If inside a virtualenv (avoids PEP 517 isolation issues):
```bash
pip install --no-build-isolation .
```

If `--with-mpi`:
```bash
MORI_WITH_MPI=ON pip install .
```

**pypi**:
```bash
pip install amd_mori
```

**nightly**:
```bash
pip install --pre amd-mori-nightly
# or from GitHub Pages (most recent):
pip install --no-index --force-reinstall \
    --find-links https://rocm.github.io/mori/nightly/latest/ amd_mori
```

> Note: `amd-mori` and `amd-mori-nightly` both provide the `mori` module. Do not install both — uninstall one first with `pip uninstall amd-mori` or `pip uninstall amd-mori-nightly`.

---

## Step 6: Verify install

```bash
python3 -c "import mori; print('mori version:', mori.__version__)"
```

If this fails with a shared library error (`libpci.so`, `libibverbs.so`, etc.):
```bash
# Find the missing lib
ldd $(python3 -c "import mori._C; print(mori._C.__file__)") | grep "not found"
# Then install the corresponding -dev package or re-run ldconfig
ldconfig
```

---

## Step 7 (optional): Precompile GPU kernels

Only run if `--precompile` was passed, or ask the user if they want to precompile now (avoids JIT delay on first use — recommended for Docker image builds):

```bash
MORI_PRECOMPILE=1 python3 -c "import mori"
```

Kernels compile to `~/.mori/jit/` and are reused on every subsequent run.

---

## Step 8: Show detected hardware

```bash
python3 -c "
from mori.jit.config import detect_build_config, detect_nic_type
cfg = detect_build_config()
print(f'GPU arch : {cfg.arch}')
print(f'NIC type : {detect_nic_type()}')
"
```

Confirm RDMA device visibility:
```bash
ibv_devinfo | head -20
```

---

## Done

Report:
- Base image / OS used
- ROCm version detected
- NIC library installed (`libionic` / `libmlx5` / `libbnxt_re` / none)
- RDMA-core source: distro package or built from source
- Install mode used (`source` / `pypi` / `nightly`)
- GPU arch and NIC type as reported by MORI
- Whether kernels are precompiled or will JIT on first use
- JIT cache location: `~/.mori/jit/`
