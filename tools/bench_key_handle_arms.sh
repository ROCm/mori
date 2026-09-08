#!/usr/bin/env bash
# Alternating-arm measurement of what the ranged-get key handle is worth in the
# shape a chunked reader actually produces.
#
# Arm "off"  = UMBP_KEY_HANDLE_SLOTS=0, the client never remembers a handle, so
#              every call carries its keys. This is what the old eight slots
#              produced at any cycle longer than eight -- 0% hit, measured.
# Arm "on"   = the default capacity with random replacement.
#
# Arms alternate within each repeat because the host drifts; the median of the
# per-arm samples is reported.
set -u
BIN=${BIN:-./build/tests/cpp/umbp/distributed/bench_umbp_standalone_ranged_wire}
CTR=${CTR:-ctr_yw_umbpdev}
REPS=${REPS:-5}

run() {  # run <slots> <extra args...>
  local slots=$1; shift
  docker exec -e UMBP_KEY_HANDLE_SLOTS="$slots" -w /apps/yutongwu/store/mori-opt "$CTR" \
    "$BIN" "$@" 2>/dev/null | tail -1 | cut -d, -f10
}

median() { tr ' ' '\n' | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

printf '%-46s %10s %10s %8s\n' "shape" "handle_off" "handle_on" "delta"
for shape in "$@"; do
  off=""; on=""
  for _ in $(seq "$REPS"); do
    off="$off $(run 0 $shape)"
    on="$on $(run 128 $shape)"
  done
  mo=$(echo "$off" | median); mn=$(echo "$on" | median)
  d=$(awk -v a="$mo" -v b="$mn" 'BEGIN{if(a>0) printf "%+.1f%%", 100*(b-a)/a; else print "n/a"}')
  printf '%-46s %10s %10s %8s\n' "$shape" "$mo" "$mn" "$d"
done
