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

You are helping the user deploy MORI inside a Docker container (or on a bare
host). Walk through the steps below. Optional inputs the user can give:

- **Install mode** (default `source`): `source` | `pypi` | `nightly`
- **`--precompile`**: precompile GPU kernels after install
- **`--with-mpi`**: enable MPI support
- **`--nic=<type>`**: NIC userspace lib — `ainic` (Pensando DSC/Pollara),
  `cx7` (Mellanox ConnectX-7), `thor2` (Broadcom); omit to auto-detect at
  runtime.

If the user has not specified a NIC type, proceed without stopping — MORI
auto-detects at runtime.

Before starting, locate the MORI repo on the host (`MORI_REPO_DIR`). Default
to the current working directory if it contains `setup.py` and `mori/`; ask
the user otherwise. Do **not** hardcode another user's home path.

---

## Step 1: Start the Docker container

**Ask the user for a container name first — this is mandatory. Do not proceed
without it.**

> What should the Docker container be named? (e.g. `mori-dev`)

Once you have the name (`$CONTAINER_NAME`):

### Case A — user provides an existing container

```bash
docker start $CONTAINER_NAME 2>/dev/null || true
docker exec -it $CONTAINER_NAME bash
```

### Case B — no existing container

Ask the user for the image (`$IMAGE_NAME`). If they don't have one, build
from the repo's dev Dockerfile:

```bash
cd "$MORI_REPO_DIR"
docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
# then use IMAGE_NAME=rocm/mori:dev
```

**Probe optional mount paths and `/dev/infiniband`** on the host first and
only include the ones that exist:

```bash
for p in /shared /apps /dev/infiniband; do
  [ -e "$p" ] && echo "EXISTS: $p" || echo "MISSING: $p"
done
```

Build the `docker run` command dynamically — only add `-v <path>:<path>` /
`--device=/dev/infiniband` lines for paths that actually exist. Example
for a host where all paths exist:

```bash
docker run \
    --group-add video \
    --network=host \
    --ulimit nproc=100000:100000 \
    --pids-limit=-1 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --ipc=host \
    --privileged -d \
    -v /home/:/home/ \
    -v /root:/root \
    -v /mnt:/mnt \
    -v /shared:/shared \
    -v /apps:/apps \
    -v /lib/modules:/lib/modules \
    --rm \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    sleep infinity
```

Notes:

- `--privileged` + `--device=/dev/infiniband` exposes RDMA / AINIC devices.
- `--network=host` is required for RDMA / IBGDA device visibility.
- `/home/` is mounted so the MORI repo is accessible inside the container
  at the same path as on the host.
- `--rm` removes the container on stop — data outside mounted volumes is lost.

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

- `libibverbs-dev` provides `libibverbs.so` — required for any RDMA/IBGDA op.
- `ibverbs-utils` provides `ibv_devinfo` for diagnosing RDMA device visibility.

---

## Step 3: Install rdma-core and nicctl (AINIC only)

**Skip this whole step if NIC type is not `ainic`.**

### Locate and extract the AINIC bundle

First, get the firmware version from the host:

```bash
# Run on HOST (before exec-ing into the container)
nicctl show version firmware
```

Note the version (e.g. `1.117.5-a-82`), then find the matching bundle inside
the container:

```bash
BUNDLE=$(find / -name "ainic_bundle_*.tar.gz" 2>/dev/null | head -1)
echo "Found bundle: $BUNDLE"
WORK=$(dirname "$BUNDLE")
tar xf "$BUNDLE" -C "$WORK"

# Outer bundle extracts to a directory named after the tarball:
BUNDLE_DIR="$WORK/$(basename "$BUNDLE" .tar.gz)"

# host_sw_pkg.tar.gz contains the user-space artifacts:
tar xf "$BUNDLE_DIR/host_sw_pkg.tar.gz" -C "$BUNDLE_DIR"

# drivers-linux.tar.xz contains the rdma-core source:
tar xf "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux.tar.xz" \
    -C "$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/"
```

### Build and install rdma-core from the bundle

```bash
RDMA_SRC="$BUNDLE_DIR/host_sw_pkg/ionic_driver/src/drivers-linux/rdma-core"

apt-get install -y --no-install-recommends cmake ninja-build python3-docutils \
    libudev-dev pkg-config dh-python pandoc

cd "$RDMA_SRC"
mkdir -p build && cd build
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

**source** (default, from `$MORI_REPO_DIR`):

```bash
cd "$MORI_REPO_DIR"
pip install .
```

Inside an existing virtualenv (avoids PEP 517 build-isolation issues):

```bash
pip install --no-build-isolation .
```

For `--with-mpi`:

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
# or from GitHub Pages (latest nightly):
pip install --no-index --force-reinstall \
    --find-links https://rocm.github.io/mori/nightly/latest/ amd_mori
```

> `amd-mori` and `amd-mori-nightly` both provide the `mori` module — do not
> install both. Uninstall one first with `pip uninstall amd-mori` (or
> `amd-mori-nightly`) before installing the other.

---

## Step 5: Verify install

```bash
python3 -c "import mori; print('mori version:', mori.__version__)"
```

If this fails with a shared-library error (`libpci.so`, `libibverbs.so`, …):

```bash
ldd $(python3 -c "import mori._C; print(mori._C.__file__)") | grep "not found"
# Install the corresponding -dev package or re-run ldconfig.
ldconfig
```

---

## Step 6 (optional): Precompile GPU kernels

Only run if `--precompile` was passed, or ask the user — recommended when
baking into a Docker image, since it skips the JIT delay on first use:

```bash
MORI_PRECOMPILE=1 python3 -c "import mori"
```

Kernels compile to `~/.mori/jit/` and are reused on every subsequent run.

---

## Step 7: Show detected hardware

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

## Done — Report Back

Summarise to the user:

- Base image / OS used
- ROCm version detected
- NIC library installed (`libionic` / `libmlx5` / `libbnxt_re` / none)
- rdma-core source: distro package or built from bundle
- Install mode used (`source` / `pypi` / `nightly`)
- GPU arch and NIC type as reported by MORI
- Whether kernels are precompiled or will JIT on first use
- JIT cache location: `~/.mori/jit/`
