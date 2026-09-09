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

# Spur runs the batch body in a NON-LOGIN shell, so /etc/profile.d/*.sh never runs:
# srun is missing from PATH ("srun: command not found") and SPUR_CONTROLLER_ADDR is
# unset, which also makes squeue silently return nothing. Source the profile drop-ins
# and add the usual prefix before anything else.
# Drop `set -u` while sourcing: profile drop-ins are not written to be -u safe (an
# openmpi one dereferences an unset LD_LIBRARY_PATH and would abort this script).
if ! command -v srun >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/*.sh; do [ -r "$f" ] && . "$f"; done
  set -u
  case ":$PATH:" in */usr/local/bin:*) ;; *) PATH="/usr/local/bin:$PATH" ;; esac
  export PATH
fi

# Slurm runs the batch body from a SPOOL COPY (/var/spool/slurmd/job<id>/), so
# $BASH_SOURCE points at a directory that holds only this file — the worker is not
# there. Resolve against the submit dir first; $BASH_SOURCE is only the fallback for
# running this script by hand outside a scheduler.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKER="${WORKER:-${SLURM_SUBMIT_DIR:-$HERE}/xrail_matrix.sh}"
[ -r "$WORKER" ] || WORKER="$HERE/xrail_matrix.sh"
OUT="${XRAIL_OUT:-${SLURM_SUBMIT_DIR:-$PWD}/xrail-output/job-$JOB}"
mkdir -p "$OUT"

# Two different fan-out models, detected rather than configured:
#   HAVE_SRUN=1  Slurm-style — the batch body runs on the FIRST node only and srun
#                spreads the steps across the allocation.
#   HAVE_SRUN=0  Spur >= 0.10 ships no srun on compute nodes; instead the batch body
#                itself is dispatched to EVERY allocated node. The worker already
#                self-organizes by hostname, so it just runs in place.
HAVE_SRUN=0
command -v srun >/dev/null 2>&1 && HAVE_SRUN=1
ME=$(hostname)

if [ "$HAVE_SRUN" = 1 ]; then
  exec >>"$OUT/run.log" 2>&1
else
  exec >>"$OUT/run.log.$ME" 2>&1
fi
echo "JOB=$JOB  nodelist=${SLURM_JOB_NODELIST:-?}  OUT=$OUT  start=$(date -u +%FT%TZ)"

# Spur starts the batch body before the job leaves Pending, so a bare srun here
# dies with "job is not running (state: Pending)". Wait for RUNNING, then retry.
# On some Spur builds squeue cannot report the job it is running inside — the state
# reads empty forever and this loop would burn its full budget before doing any work.
# Treat a few consecutive empty reads as "squeue is unusable here" and proceed.
unknown=0
for i in $(seq 1 60); do
  st=$(squeue -h -j "$JOB" -o "%T" 2>/dev/null | head -1)
  echo "wait[$i] state=${st:-unknown}"
  [ "$st" = "RUNNING" ] && break
  if [ -z "$st" ]; then
    unknown=$((unknown + 1))
    [ "$unknown" -ge 3 ] && {
      echo "squeue cannot report our own job here; proceeding without the RUNNING check"
      break
    }
  fi
  sleep 5
done

[ -r "$WORKER" ] || {
  echo "FATAL: worker script not readable: $WORKER"
  echo "       submit from the folder holding xrail_matrix.sh, or set WORKER=/abs/path"
  exit 1
}

rc=1
if [ "$HAVE_SRUN" = 1 ]; then
  for attempt in 1 2 3 4 5; do
    echo "--- srun attempt $attempt at $(date -u +%FT%TZ) ---"
    srun -N2 -n2 --overlap bash "$WORKER" xrail "$OUT" && { rc=0; break; }
    sleep 10
  done
else
  echo "--- no srun: running worker in place on $ME at $(date -u +%FT%TZ) ---"
  bash "$WORKER" xrail "$OUT" && rc=0
fi
echo "worker rc=$rc"

echo "=== ENV (per node) ==="; cat "$OUT"/env.* 2>/dev/null
echo "=== MATRIX ==="; cat "$OUT/matrix.txt" 2>/dev/null || echo "no matrix.txt"
echo "end=$(date -u +%FT%TZ)"
