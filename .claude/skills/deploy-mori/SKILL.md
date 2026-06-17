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

You are helping the user deploy MORI inside a Docker container.

**Locate `MORI_REPO_DIR`**: default to the current working directory if it
contains `pyproject.toml`; ask the user otherwise.

**Detect NIC type** — determines whether Step 3 is needed:

```bash
lspci | grep -iE "pensando|ionic|dsc|pollara" && echo "→ ainic"
lspci | grep -iE "mellanox|connectx"           && echo "→ cx7"
lspci | grep -iE "broadcom.*thor|bnxt"         && echo "→ thor2"
```

---

## Step 1: Start the Docker container

Use the container name from the user's args if provided; ask if not.

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

All subsequent steps run **inside** `$CONTAINER_NAME` via `docker exec`.

---

## Step 2: Install base system packages

```bash
sudo docker exec $CONTAINER_NAME bash -c "apt-get update && apt-get install -y --no-install-recommends \
    git libpci-dev libdw1 libibverbs-dev ibverbs-utils \
    locales iputils-ping iproute2 ethtool jq perftest"
```

`jq` is required by `mori check`. `perftest` provides `ib_write_bw`/`ib_write_lat` for intra/inter-node bandwidth and latency checks.

---

## Step 3: Install AINIC userspace libraries (AINIC only)

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

## Step 4: Install MORI

`pybind11` is a required build dep missing from `pyproject.toml`. Run inside the container:

```bash
sudo docker exec -w $MORI_REPO_DIR $CONTAINER_NAME bash -c "
pip install pybind11 -q
rm -rf build   # clear stale cmake cache — old build/ can hardcode a wrong ROCm version
pip install .
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

## Step 7: Run `mori check`

```bash
sudo docker exec $CONTAINER_NAME bash -c "mori check"
```

`mori check` validates the full AINIC/RDMA stack in 6 steps:

1. **firmware & driver** — firmware, nicctl, ionic driver versions must match
2. **QoS / SL / TC** — DSCP classification, PFC, DWRR scheduling, selects the SL/TC for MORI to use
3. **DCQCN** — congestion control enabled on all RoCE devices, CNP DSCP consistent
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
- NIC library installed (`libionic` / `libmlx5` / `libbnxt_re` / none)
- Install mode: source (`pip install .`)
- GPU arch and NIC type as reported by MORI
- Kernels: JIT on first use (`~/.mori/jit/`)
- `mori check` result — include full output, highlight any `[WARN]` or `[FAIL]`
- Attach command (working directory set to the MORI source tree):

```bash
sudo docker exec -it -w "$MORI_REPO_DIR" $CONTAINER_NAME bash
```

**Built-in tools:** `mori check`, `mori setup`, `mori diagnose` — see Step 7 for usage.
