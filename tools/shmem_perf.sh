#!/bin/bash
# Usage: ./tools/shmem_perf.sh <bw|lat> <output_file> [extra args...]
# Example: ./tools/shmem_perf.sh bw bw.txt -n 20

set -euo pipefail

TESTTYPE="${1:?Usage: $0 <bw|lat> <output_file> [extra args...]}"
OUTPUT="${2:?Usage: $0 <bw|lat> <output_file> [extra args...]}"
shift 2

BINDIR="$(cd "$(dirname "$0")/../build/perftest" && pwd)"
MPI="mpirun --allow-run-as-root -np 2"

run() {
    local bin="$1" scope="$2"; shift 2
    echo ""
    echo "### ${bin##*/} scope=${scope} ###"
    $MPI "$BINDIR/$bin" -s "$scope" "$@"
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
*)
    echo "ERROR: unknown test type '$TESTTYPE'" >&2; exit 1 ;;
esac 2>&1 | tee -a "$OUTPUT"

echo "# done -> $OUTPUT"
