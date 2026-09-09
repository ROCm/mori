#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --job-name=xrail
#SBATCH --time=00:15:00
# Generic 2-node RDMA cross-rail test. Cluster-agnostic: the worker auto-detects
# rail devices, GID indices and address family via rail_detect.sh. Pass
# partition/account/qos/gres on the command line so nothing site-specific is
# baked in:
#
#   sbatch -A <account> -p <partition> --qos=<qos> --gres=gpu:8 xrail_worker.sbatch
#
# On Spur, wrap it so the controller address is in the environment:
#   bash -lc "sbatch -A <account> -p <partition> --qos=<qos> --gres=gpu:8 xrail_worker.sbatch"
#
# Optional env (export before sbatch, or via --export):
#   XRAIL_OUT   base output folder            (default: <submit_dir>/xrail-output)
#   WORKER      path to xrail_worker.sh       (default: next to this script)
#   plus any worker overrides: RAIL_LIB, RAIL_DEV_REGEX, EXCLUDE_DEV_REGEX,
#   INCLUDE_MGMT, GID_INDEX, SRC_RAILS, PORT_BASE
#
# All artifacts (coordination files, result, run log) land in $XRAIL_OUT/job-<jobid>/,
# which is the layout make_report.py expects.
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
WORKER="${WORKER:-${SLURM_SUBMIT_DIR:-$HERE}/xrail_worker.sh}"
[ -r "$WORKER" ] || WORKER="$HERE/xrail_worker.sh"
PROBE="${PROBE:-${SLURM_SUBMIT_DIR:-$HERE}/probe_topology.sh}"
[ -r "$PROBE" ] || PROBE="$HERE/probe_topology.sh"
BASE_OUT="${XRAIL_OUT:-${SLURM_SUBMIT_DIR:-$PWD}/xrail-output}"
RUNDIR="$BASE_OUT/job-$JOB"
mkdir -p "$RUNDIR"

# Two different fan-out models, detected rather than configured:
#   HAVE_SRUN=1  Slurm-style — the batch body runs on the FIRST node only and srun
#                spreads the steps across the allocation.
#   HAVE_SRUN=0  Spur >= 0.10 ships no srun on compute nodes; instead the batch body
#                itself is dispatched to EVERY allocated node. The worker already
#                self-organizes by hostname, so it just runs in place.
HAVE_SRUN=0
command -v srun >/dev/null 2>&1 && HAVE_SRUN=1
ME=$(hostname)

# send everything this script prints (incl. worker stdout) into the output folder.
# Without srun every node runs this file, so give each its own log or they interleave.
if [ "$HAVE_SRUN" = 1 ]; then
  exec >>"$RUNDIR/run.log" 2>&1
else
  exec >>"$RUNDIR/run.log.$ME" 2>&1
fi

echo "JOB=$JOB  nodelist=${SLURM_JOB_NODELIST:-?}"
echo "RUNDIR=$RUNDIR"
echo "WORKER=$WORKER"

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
  echo "       submit from the folder holding xrail_worker.sh, or set WORKER=/abs/path"
  exit 1
}

rc=1
if [ "$HAVE_SRUN" = 1 ]; then
  for attempt in 1 2 3 4 5; do
    echo "--- srun attempt $attempt at $(date -u +%FT%TZ) ---"
    srun -N2 -n2 --overlap bash "$WORKER" "$JOB" "$RUNDIR" && { rc=0; break; }
    sleep 10
  done
else
  echo "--- no srun: running worker in place on $ME at $(date -u +%FT%TZ) ---"
  bash "$WORKER" "$JOB" "$RUNDIR" && rc=0
fi
echo "worker rc=$rc"

# Single-node probe in the SAME allocation, so one submission leaves a folder that
# has everything make_report.py needs (topo_report.txt drives the node diagram and
# the node-summary table). Compute nodes are usually unreachable by SSH, so there is
# no second chance to collect this once the job ends.
echo ""
echo "--- probe_topology.sh at $(date -u +%FT%TZ) ---"
if [ ! -r "$PROBE" ]; then
  echo "SKIPPED: probe script not readable: $PROBE"
elif [ "$HAVE_SRUN" = 1 ]; then
  srun -N1 -n1 --overlap bash -c "OUT_DIR='$RUNDIR' bash '$PROBE'" || echo "probe rc=$?"
else
  OUT_DIR="$RUNDIR" bash "$PROBE" || echo "probe rc=$?"
fi

# Cheap artifacts the skill's Step 2d checklist asks for every run: routes decide
# on-link vs routed, which decides whether the hop_limit confound can bite.
echo ""
echo "--- routes at $(date -u +%FT%TZ) ---"
collect_routes='
  h=$(hostname)
  {
    echo "=== $h ==="
    echo "--- default route ---"; ip route show default
    echo; echo "--- ipv4 routes ---"; ip -4 route show table all
    echo; echo "--- ipv6 routes ---"; ip -6 route show table all
    echo; echo "--- neigh ---"; ip -4 neigh show; ip -6 neigh show
  } > "'"$RUNDIR"'/routes.$h" 2>&1'
if [ "$HAVE_SRUN" = 1 ]; then
  srun -N2 -n2 --overlap bash -c "$collect_routes" || echo "routes rc=$?"
else
  bash -c "$collect_routes" || echo "routes rc=$?"
fi

echo ""
echo "================ CROSS-RAIL RESULT ================"
cat "$RUNDIR/result.txt" 2>/dev/null || echo "NO RESULT FILE — tester node may not have run"
echo "=================================================="
echo "All artifacts in: $RUNDIR"
