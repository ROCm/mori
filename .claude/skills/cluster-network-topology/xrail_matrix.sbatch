#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --job-name=xrail-matrix
#SBATCH --time=00:35:00
# Cross-rail RDMA verification: full 8x8 matrix + SL/MTU/TC sweeps.
# Submit (Spur):
#   bash -lc "sbatch -A <account> -p <partition> --qos=<qos> --gres=gpu:8 xrail_matrix.sbatch"
set -uo pipefail
JOB=${SLURM_JOB_ID:-manual-$$}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKER="${WORKER:-$HERE/xrail_matrix.sh}"
OUT="${XRAIL_OUT:-${SLURM_SUBMIT_DIR:-$PWD}/xrail-output/job-$JOB}"
mkdir -p "$OUT"
exec >>"$OUT/run.log" 2>&1
echo "JOB=$JOB  nodelist=${SLURM_JOB_NODELIST:-?}  OUT=$OUT  start=$(date -u +%FT%TZ)"

# Spur starts the batch body before the job leaves Pending, so a bare srun here
# dies with "job is not running (state: Pending)". Wait for RUNNING, then retry.
for i in $(seq 1 60); do
  st=$(squeue -h -j "$JOB" -o "%T" 2>/dev/null | head -1)
  echo "wait[$i] state=${st:-unknown}"
  [ "$st" = "RUNNING" ] && break
  sleep 5
done

rc=1
for attempt in 1 2 3 4 5; do
  echo "--- srun attempt $attempt at $(date -u +%FT%TZ) ---"
  srun -N2 -n2 --overlap bash "$WORKER" xrail "$OUT" && { rc=0; break; }
  sleep 10
done
echo "srun rc=$rc"

echo "=== ENV (per node) ==="; cat "$OUT"/env.* 2>/dev/null
echo "=== MATRIX ==="; cat "$OUT/matrix.txt" 2>/dev/null || echo "no matrix.txt"
echo "end=$(date -u +%FT%TZ)"
