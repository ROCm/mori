#!/usr/bin/env bash
# Find out whether the bandwidth drift has an external cause. Anything else holding the GPUs, or a
# clock/power state that moved between runs, invalidates comparisons made across runs.
set -uo pipefail
echo "=== containers ==="
docker ps --format '  {{.Names}}  {{.Image}}  {{.Status}}'

echo
echo "=== processes holding /dev/kfd (host view) ==="
sudo -n fuser -v /dev/kfd 2>&1 | head -20 || ls -l /proc/*/fd 2>/dev/null | grep -c kfd || echo "  (cannot inspect without privileges)"

echo
echo "=== per-container processes that look like GPU work ==="
for c in $(docker ps --format '{{.Names}}'); do
  n=$(docker exec "$c" bash -lc 'ps -eo comm,pcpu --no-headers 2>/dev/null | awk "\$2>20" | head -5' 2>/dev/null)
  [ -n "$n" ] && { echo "  --- $c ---"; sed 's/^/    /' <<<"$n"; }
done

echo
# CPU, not GPU. dispatch/combine are collectives whose timing includes waiting on the other three
# ranks, so anything that jitters host-side kernel launch shows up as all four ranks slowing down
# together -- while a single-card op like the staging copy_ is untouched. That is exactly the
# signature seen on 2026-08-04 (dispatch +20%, combine +18%, copy_ unchanged), with rocm-smi
# reporting the GPUs idle. So look at who is eating CPU before believing the GPUs degraded.
echo "=== host load / cpu ==="
uptime
echo "  cores: $(nproc)"
echo
echo "=== per-container cpu (one shot) ==="
timeout 20 docker stats --no-stream --format '  {{.Name}}  cpu={{.CPUPerc}}  mem={{.MemPerc}}' 2>&1

echo
# etime is the column that matters, not pcpu. A neighbour eating CPU right now says nothing about
# whether it was eating CPU during the run being explained -- on 2026-08-04 a 400% container was
# briefly blamed for a slowdown measured 20 minutes before that container's processes even started.
echo "=== top cpu consumers node-wide (etime = how long it has been running) ==="
ps -eo pcpu,pid,etime,lstart,user,comm --sort=-pcpu 2>/dev/null | head -12 | sed 's/^/  /'

echo
echo "=== rocm-smi: use, power, temp, memory ==="
rocm-smi 2>/dev/null | sed 's/^/  /'

echo
echo "=== clocks (sclk/fclk/socclk) per gpu ==="
rocm-smi --showclocks 2>/dev/null | grep -iE 'GPU\[|sclk|fclk|socclk|mclk' | sed 's/^/  /'

echo
echo "=== performance level / determinism ==="
rocm-smi --showperflevel 2>/dev/null | sed 's/^/  /'

echo
echo "=== xgmi link status if exposed ==="
rocm-smi --shownodesbw 2>/dev/null | head -12 | sed 's/^/  /'
echo GPUSTATE_DONE
