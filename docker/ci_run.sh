#!/bin/bash
set -euo pipefail

# ci_run.sh — Launch a Docker container with automatic NIC detection and
# bind-mount of out-of-tree RDMA userspace libraries.
#
# Usage: ci_run.sh [docker-run-args...] IMAGE [cmd...]
#   e.g.: ./docker/ci_run.sh --name mori_ci rocm/mori:ci
#         ./docker/ci_run.sh --name mori_ci -v /home:/home rocm/mori:ci bash
#
# Environment:
#   MORI_NIC_TYPE   — Override auto-detection (mlx5 | bnxt | ionic)

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
            for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
                fi
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                fi
            done
            flags+=(--tmpfs /etc/libibverbs.d:rw,size=4k)
            if [[ -f /etc/libibverbs.d/bnxt_re.driver ]]; then
                flags+=(-v /etc/libibverbs.d/bnxt_re.driver:/etc/libibverbs.d/bnxt_re.driver)
            fi
            ;;
        ionic)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            for lib in /usr/local/lib/libionic*.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                fi
            done
            flags+=(--tmpfs /etc/libibverbs.d:rw,size=4k)
            if [[ -f /etc/libibverbs.d/ionic.driver ]]; then
                flags+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver)
            fi
            ;;
        mlx5)
            ;;
    esac

    echo "${flags[@]}"
}

# ── Main ─────────────────────────────────────────────────────────────────────

NIC_TYPE=$(detect_nic_type)
echo "[ci_run] Detected NIC type: $NIC_TYPE"

read -ra NIC_MOUNTS <<< "$(nic_mount_flags "$NIC_TYPE")"

exec docker run \
    --group-add video \
    --network=host \
    --ulimit nproc=100000:100000 \
    --pids-limit=-1 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    -d --ipc=host --privileged -it \
    "${NIC_MOUNTS[@]}" \
    "$@"
