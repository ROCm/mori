#!/bin/bash
# Usage: ./tools/shmem_perf.sh <bw|lat|sweep> <output_file> [extra args...]
# Examples:
#   ./tools/shmem_perf.sh bw     bw.txt  -n 20
#   ./tools/shmem_perf.sh lat    lat.txt -n 20
#   ./tools/shmem_perf.sh sweep  gs.txt  -b 4096 -e 67108864 -G 128 -n 20

set -euo pipefail

TESTTYPE="${1:?Usage: $0 <bw|lat|sweep> <output_file> [extra args...]}"
OUTPUT="${2:?Usage: $0 <bw|lat|sweep> <output_file> [extra args...]}"
shift 2

BINDIR="$(cd "$(dirname "$0")/../build/perftest" && pwd)"
MPI="mpirun --allow-run-as-root -np 2"

# Threads per block and warp size (match binary defaults)
THREADS_PER_BLOCK=256
WARP_SIZE=64
# Minimum bytes per thread for thread scope to be valid
MIN_BYTES_PER_THREAD=32

run() {
    local bin="$1" scope="$2"; shift 2
    echo ""
    echo "### ${bin##*/} scope=${scope} ###"
    $MPI "$BINDIR/$bin" -s "$scope" "$@"
}

# ---------------------------------------------------------------------------
# Safe mpirun wrapper: fully isolates crashes and signals from the parent.
# Prints the captured stdout; returns 0 always.
# Usage: safe_mpi_bw <bin> <scope> <grid> <size> [extra_args...]
# ---------------------------------------------------------------------------
safe_mpi_bw() {
    local bin="$1" scope="$2" g="$3" sz="$4"; shift 4
    # Run in a throwaway subshell with signals ignored so mpirun crashes
    # (SIGTERM / SIGKILL from prterun) cannot propagate to our script.
    local raw
    raw=$(
        trap '' TERM INT
        set +e
        $MPI "$BINDIR/$bin" -s "$scope" -c "$g" -b "$sz" -e "$sz" "$@" 2>/dev/null
        exit 0
    ) || true
    printf '%s\n' "$raw"
}

# ---------------------------------------------------------------------------
# Check whether a (size, grid, scope) combination is valid before running.
# Returns 0 (valid) or 1 (skip).
# Rules mirror the binary's size_ok():
#   block  : size must be divisible by grid
#   warp   : size must be divisible by grid * warps_per_block
#   thread : size must be divisible by grid * threads_per_block
#            AND per-thread size >= MIN_BYTES_PER_THREAD
# ---------------------------------------------------------------------------
is_valid_combo() {
    local sz=$1 g=$2 scope=$3
    local wpb=$(( THREADS_PER_BLOCK / WARP_SIZE ))
    case "$scope" in
    block)
        [[ $(( sz % g )) -eq 0 ]] ;;
    warp)
        [[ $(( sz % (g * wpb) )) -eq 0 ]] ;;
    thread)
        local total=$(( g * THREADS_PER_BLOCK ))
        [[ $(( sz % total )) -eq 0 ]] && \
        [[ $(( sz / total )) -ge $MIN_BYTES_PER_THREAD ]] ;;
    esac
}

# ---------------------------------------------------------------------------
# Core 2D sweep: message size × grid_x.
# SCOPES_TO_RUN is a global set by the caller before invoking this function.
# Skips invalid combos with a clear reason instead of crashing.
# User extra args (-G / -b / -e / -f / -n / -w) are parsed from "$@".
# ---------------------------------------------------------------------------
SCOPES_TO_RUN=()   # set by caller

run_2d_sweep() {
    local max_grid=128
    local min_bytes=4096
    local max_bytes=67108864
    local step_factor=4
    local extra_args=()

    local i=0
    local argv=("$@")
    while [[ $i -lt ${#argv[@]} ]]; do
        case "${argv[$i]}" in
        -G) i=$(( i+1 )); max_grid="${argv[$i]}" ;;
        -b) i=$(( i+1 )); min_bytes="${argv[$i]}" ;;
        -e) i=$(( i+1 )); max_bytes="${argv[$i]}" ;;
        -f) i=$(( i+1 )); step_factor="${argv[$i]}" ;;
        *)  extra_args+=("${argv[$i]}") ;;
        esac
        i=$(( i+1 ))
    done

    echo ""
    echo "### 2D sweep scopes=(${SCOPES_TO_RUN[*]}): grid 1..${max_grid}," \
         " size ${min_bytes}..${max_bytes} step×${step_factor} ###"
    printf "%-12s  %-8s  %-6s  %-7s  %-14s  %s\n" \
        "size(B)" "grid_x" "scope" "op" "per_unit(B)" "BW_GBps"

    local scope
    for scope in "${SCOPES_TO_RUN[@]}"; do
        local sz=$min_bytes
        while [[ $sz -le $max_bytes ]]; do
            local g=1
            while [[ $g -le $max_grid ]]; do
                for op in put get; do
                    local bin="p2p_${op}_bw"
                    local wpb=$(( THREADS_PER_BLOCK / WARP_SIZE ))
                    local per_unit
                    case "$scope" in
                    block)  per_unit=$(( sz / g )) ;;
                    warp)   per_unit=$(( sz / (g * wpb) )) ;;
                    thread) per_unit=$(( sz / (g * THREADS_PER_BLOCK) )) ;;
                    *)      per_unit=0 ;;
                    esac

                    if ! is_valid_combo "$sz" "$g" "$scope"; then
                        printf "%-12s  %-8s  %-6s  %-7s  %-14s  %s\n" \
                            "$sz" "$g" "$scope" "$op" "${per_unit}B" \
                            "skip(<${MIN_BYTES_PER_THREAD}B/unit)"
                    else
                        local raw bw
                        raw=$(safe_mpi_bw "$bin" "$scope" "$g" "$sz" \
                              "${extra_args[@]}")
                        bw=$(printf '%s\n' "$raw" \
                            | grep -v '^#' | grep -v '^size' \
                            | grep -v '^$'  | grep -v 'skip' \
                            | awk '{print $NF}' | tail -1)
                        printf "%-12s  %-8s  %-6s  %-7s  %-14s  %s\n" \
                            "$sz" "$g" "$scope" "$op" "${per_unit}B" \
                            "${bw:-run_failed}"
                    fi
                done
                g=$(( g * 2 ))
            done
            sz=$(( sz * step_factor ))
        done
    done
}

: > "$OUTPUT"
echo "# testtype=$TESTTYPE  date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUTPUT"

case "$TESTTYPE" in
bw)
    for scope in block warp thread; do
        run p2p_put_bw "$scope" "$@"
        run p2p_get_bw "$scope" "$@"
    done ;;
lat)
    for scope in block warp thread; do
        run p2p_put_latency "$scope" "$@"
        run p2p_get_latency "$scope" "$@"
    done ;;
sweep)
    SCOPES_TO_RUN=(block warp thread)
    run_2d_sweep "$@" ;;
*)
    echo "ERROR: unknown test type '$TESTTYPE'" >&2; exit 1 ;;
esac 2>&1 | tee -a "$OUTPUT"

echo "# done -> $OUTPUT"
