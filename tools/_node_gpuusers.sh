#!/usr/bin/env bash
# Read-only: who (node-wide, all containers) is using the GPUs right now.
set -uo pipefail
echo "== all GPU pids node-wide (name / GPU / VRAM / CU occupancy) =="
timeout 20 rocm-smi --showpids 2>&1 | sed -n '1,40p' || echo "showpids failed"
# Every python in every container, idle ones included. A neighbour that merely HOLDS a GPU context
# -- initialised, queues created, but computing nothing -- shows 0% CPU and no VRAM worth noticing,
# yet still puts its queues in front of the hardware scheduler. That is invisible to --showpids and
# to a cpu-threshold filter, and it is the shape that would slow collectives (launch and barrier
# latency) while leaving a single-card copy_ untouched.
echo "== every python/GPU-ish process in every container, idle included =="
for c in $(docker ps --format '{{.Names}}'); do
  o=$(docker exec "$c" sh -lc 'ps -eo pid,stat,etime,pcpu,args 2>/dev/null | grep -iE "python|torch|hip|rocm" | grep -v grep | cut -c1-110 | head -8' 2>/dev/null)
  [ -n "$o" ] && { echo "  --- $c ---"; sed 's/^/    /' <<<"$o"; }
done

echo "== per-GPU utilization =="
timeout 20 rocm-smi --showuse 2>&1 | grep -iE "GPU\[|use" | head -20 || echo "showuse failed"
echo "== vram used per gpu =="
timeout 15 rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "Used" | head -8 || echo "no rocm-smi"
echo "== running containers =="
docker ps --format '{{.Names}} | {{.Status}}'
# Zombies themselves are harmless -- the kernel has already reclaimed memory, fds and the KFD
# context, leaving a task_struct and a PID. What matters is PPID: a zombie exists only because its
# parent never called wait, so the parent is still alive, and THAT is the process that might still
# be holding something. Print the parents too, or this table invites blaming the zombies.
echo "== our container: python processes with parents (Z is harmless, its parent may not be) =="
docker exec MORI-EPV2 bash -lc "ps -eo pid,ppid,stat,etime,pcpu,comm | grep -iE 'python|spawn' | grep -v grep | head -30 || echo none" 2>&1 | head -35
echo "== parents of those (are they still running, do they hold anything) =="
docker exec MORI-EPV2 bash -lc '
  for p in $(ps -eo ppid,stat --no-headers | awk "\$2 ~ /Z/ {print \$1}" | sort -u); do
    ps -o pid,stat,etime,pcpu,args -p "$p" --no-headers 2>/dev/null | cut -c1-120
  done' 2>&1 | head -15
echo "===DONE==="
