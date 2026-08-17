#!/bin/bash
set -euo pipefail

# ci_run.sh — Launch a container with automatic NIC detection and
# bind-mount of out-of-tree RDMA userspace libraries.
#
# Usage: ci_run.sh [docker-run-args...] IMAGE [cmd...]
#   e.g.: ./docker/ci_run.sh --name mori_ci rocm/mori:ci
#         ./docker/ci_run.sh --name mori_ci -v /home:/home rocm/mori:ci bash
#
# Environment:
#   MORI_NIC_TYPE        — Override auto-detection (mlx5 | bnxt | ionic)
#   CONTAINER_RUNTIME    — Override runtime (docker | podman); auto-detected
#   MORI_NIC_LIB_MOUNT   — auto (default) | always | never; see below

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── NIC detection ────────────────────────────────────────────────────────────

detect_nic_type() {
    if [[ -n "${MORI_NIC_TYPE:-}" ]]; then
        echo "$MORI_NIC_TYPE"
        return
    fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name
            name=$(basename "$dev")
            case "$name" in
                bnxt_re*) ((bnxt++)) ;;
                mlx5*)    ((mlx5++)) ;;
                ionic*)   ((ionic++)) ;;
                *)
                    local drv
                    drv=$(readlink -f "$dev/device/driver" 2>/dev/null || true)
                    drv=$(basename "$drv" 2>/dev/null || true)
                    case "$drv" in
                        bnxt*) ((bnxt++)) ;;
                        mlx5*) ((mlx5++)) ;;
                        ionic*) ((ionic++)) ;;
                    esac
                    ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then
        echo "bnxt"
    elif (( ionic >= mlx5 && ionic > 0 )); then
        echo "ionic"
    else
        echo "mlx5"
    fi
}

# ── Build bind-mount flags for OOT RDMA libs ────────────────────────────────

find_host_ibverbs() {
    local candidates=(
        /usr/lib64/libibverbs.so.1
        /lib/x86_64-linux-gnu/libibverbs.so.1
        /usr/lib/x86_64-linux-gnu/libibverbs.so.1
    )
    for c in "${candidates[@]}"; do
        local resolved
        resolved=$(readlink -f "$c" 2>/dev/null || true)
        if [[ -f "$resolved" ]]; then
            echo "$resolved"
            return
        fi
    done
}

nic_mount_flags() {
    local nic_type="$1"
    local flags=()

    case "$nic_type" in
        bnxt)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            local bnxt_dir=""
            for dir in /usr/local/lib/x86_64-linux-gnu /usr/local/lib /usr/lib/x86_64-linux-gnu; do
                if compgen -G "$dir/libbnxt_re-rdmav*.so" >/dev/null 2>&1; then
                    bnxt_dir="$dir"
                    break
                fi
            done
            if [[ -n "$bnxt_dir" ]]; then
                for lib in "$bnxt_dir"/libbnxt_re-rdmav*.so; do
                    [[ -e "$lib" ]] || continue
                    local real
                    real=$(readlink -f "$lib")
                    flags+=(-v "$real:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
                done
                if [[ -e "$bnxt_dir/libbnxt_re.so" ]]; then
                    local real
                    real=$(readlink -f "$bnxt_dir/libbnxt_re.so")
                    flags+=(-v "$real:/usr/lib/x86_64-linux-gnu/libbnxt_re.so")
                fi
            fi
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
            ;;
        ionic)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            local ionic_dirs=(/usr/local/lib /usr/lib/x86_64-linux-gnu)
            for dir in "${ionic_dirs[@]}"; do
                for lib in "$dir"/libionic*.so; do
                    if [[ -f "$lib" ]]; then
                        local real
                        real=$(readlink -f "$lib")
                        if [[ -f "$real" ]]; then
                            flags+=(-v "$real:$real")
                        fi
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local provider_dir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$provider_dir" ]]; then
                for lib in "$provider_dir"/libionic-rdmav*.so; do
                    if [[ -f "$lib" ]]; then
                        flags+=(-v "$lib:$lib")
                    fi
                done
            fi
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
            ;;
        mlx5)
            ;;
    esac

    echo "${flags[@]}"
}

# ── Should we graft the host libraries in at all? ────────────────────────────
#
# The mounts above exist for images that ship no out-of-tree provider. Once an
# image carries its own (Dockerfile.dev with BNXT_ROCELIB_VERSION), overwriting
# it with whatever the host happens to have makes the same image behave
# differently per node: tw010 carries libbnxt_re 236.1.165.0 and works, tw022
# carries 232.0.155.5 which only speaks kernel ABI 6 against a driver exposing
# 8, so every device vanishes and the internode job has no RDMA on node2.
# Probe the bare image and leave a working stack alone.

first_positional_arg() {
    local value_flags=(
        --name -v --volume -e --env --env-file -w --workdir -p --publish
        --network --net --device --entrypoint -u --user --ulimit --shm-size
        --label -l --hostname -h --cpus --memory -m --mount --add-host
        --group-add --security-opt --restart --log-driver --log-opt --pid
        --ipc --tmpfs --cap-add --cap-drop --init-path --runtime --gpus
        --pids-limit --stop-signal --stop-timeout --health-cmd
    )
    while (( $# )); do
        case "$1" in
            -*=*) shift ;;
            -*)
                local takes_value=0 flag
                for flag in "${value_flags[@]}"; do
                    if [[ "$1" == "$flag" ]]; then takes_value=1; break; fi
                done
                if (( takes_value )); then shift 2 || shift; else shift; fi
                ;;
            *) echo "$1"; return ;;
        esac
    done
}

image_enumerates_rdma_devices() {
    local image="$1"
    [[ -n "$image" && -d /dev/infiniband ]] || return 1
    timeout 120 "$RUNTIME" run --rm --privileged --network=host \
        --device=/dev/infiniband "$image" ibv_devinfo -l 2>/dev/null \
        | grep -qE '^[1-9][0-9]* HCAs found'
}

# ── Container runtime detection ───────────────────────────────────────────────

detect_runtime() {
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        echo "$CONTAINER_RUNTIME"
        return
    fi
    if docker info &>/dev/null; then
        echo "docker"
    elif command -v podman &>/dev/null; then
        echo "podman"
    elif command -v docker &>/dev/null; then
        echo "docker"
    else
        echo "docker"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

RUNTIME=$(detect_runtime)
NIC_TYPE=$(detect_nic_type)
echo "[ci_run] Runtime: $RUNTIME | NIC type: $NIC_TYPE"

NIC_MOUNTS=()
MOUNT_MODE="${MORI_NIC_LIB_MOUNT:-auto}"
if [[ "$NIC_TYPE" != "mlx5" && "$MOUNT_MODE" != "never" ]]; then
    IMAGE_ARG=$(first_positional_arg "$@")
    if [[ "$MOUNT_MODE" != "always" ]] && image_enumerates_rdma_devices "$IMAGE_ARG"; then
        echo "[ci_run] $IMAGE_ARG ships a working $NIC_TYPE provider; keeping host libs out"
    else
        echo "[ci_run] Mounting host $NIC_TYPE userspace into the container"
        read -ra NIC_MOUNTS <<< "$(nic_mount_flags "$NIC_TYPE")"
    fi
fi

# --init (tini/catatonit as PID 1) reaps exited child processes so zombies don't
# keep KFD contexts / VRAM alive, and forwards SIGTERM from `stop` to the group.
EXTRA_ARGS=()
if [[ "$RUNTIME" == "podman" ]]; then
    EXTRA_ARGS+=(--security-opt label=disable)
    # podman's --init needs catatonit; skip it if the host lacks the binary
    # (avoids "container-init binary not found: .../catatonit"). ci_stop.sh
    # still pkills/reaps on teardown.
    if [[ -x /usr/libexec/podman/catatonit ]] || command -v catatonit &>/dev/null; then
        EXTRA_ARGS+=(--init)
    elif command -v tini &>/dev/null; then
        EXTRA_ARGS+=(--init --init-path "$(command -v tini)")
    fi
else
    EXTRA_ARGS+=(--ulimit nproc=100000:100000 --pids-limit=-1 --init)
fi

# RCCL treats a missing HSA_NO_SCRATCH_RECLAIM as a fatal error on some
# runtime/firmware combinations (seen on MI300X with ROCm 7.2.x, GPU firmware
# 166): ncclCommInitRank aborts, rank 0 dies and every other rank then fails
# fetching the ncclUniqueId. tools/bench_ep_*.sh already export it; do the same
# for every CI container so the test jobs get the same environment.
exec "$RUNTIME" run \
    --group-add video \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -d --ipc=host --privileged \
    "${EXTRA_ARGS[@]}" \
    "${NIC_MOUNTS[@]}" \
    "$@"
