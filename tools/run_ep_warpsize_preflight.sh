#!/usr/bin/env bash
# Build + run the EP v1 warpSize pre-flight INSIDE the container, single GPU, with
# a hard `timeout` on top of the in-binary watchdog so it can never wedge the node.
#
# Usage (from repo root, on the 455/MI450 box or MI300 box):
#   bash tools/run_ep_warpsize_preflight.sh [CTR] [warp_num_per_block] [ept] [use_weights]
# Defaults: CTR=MORI-EPV2, wpb=8, ept=8, use_weights=1
#
# Interpreting the exit code / output:
#   RESULT: SAFE   -> device warpSize == 64; the v1 C++ IntraNode EP path is
#                     dimensionally consistent (gfx942/MI300).
#   RESULT: UNSAFE -> device warpSize < 64 (gfx1250/MI450); the hard-coded
#                     WARP_SIZE=64 launch overflows combine's dynamic LDS ->
#                     the real EP4 test will hang. Do NOT launch it.
set -uo pipefail

CTR="${1:-MORI-EPV2}"
WPB="${2:-8}"
EPT="${3:-8}"
USEW="${4:-1}"

# Path is /app/mori on the 455 box, /home/fizhang/mori-epv2 on MI300; auto-detect.
WS=$(docker exec "$CTR" bash -c 'for d in /app/mori /home/fizhang/mori-epv2; do [ -d "$d/tools" ] && { echo "$d"; break; }; done')
if [ -z "${WS:-}" ]; then echo "could not find repo (tools/) in $CTR"; exit 1; fi
echo "[pf] container=$CTR repo=$WS"

# Copy the source in fresh (in case the container repo is stale) and strip CR.
docker cp tools/ep_warpsize_preflight.hip.cpp "$CTR:/tmp/ep_warpsize_preflight.hip.cpp"
docker exec "$CTR" bash -c "sed -i 's/\r$//' /tmp/ep_warpsize_preflight.hip.cpp"

# Detect arch so we compile the matching --offload-arch (no wave64/wave32 surprise).
ARCH=$(docker exec "$CTR" bash -c 'rocminfo 2>/dev/null | grep -m1 -o "gfx[0-9a-f]*"')
ARCH="${ARCH:-gfx1250}"
echo "[pf] detected arch=$ARCH"

echo "[pf] building..."
docker exec "$CTR" bash -c "cd /tmp && hipcc -O3 --offload-arch=$ARCH ep_warpsize_preflight.hip.cpp -o ep_pf 2>&1 | tail -8 && echo BUILD_OK"

echo "[pf] running (hard 60s timeout + in-binary ${WPB}-warp watchdog)..."
docker exec "$CTR" bash -c "cd /tmp && timeout 60 ./ep_pf $WPB $EPT $USEW"
rc=$?
echo "[pf] exit_code=$rc  (0=SAFE, 2=UNSAFE, 3=watchdog-timeout, 124=hard-timeout, 1=hip-err)"
exit $rc
