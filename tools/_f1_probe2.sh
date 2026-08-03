#!/usr/bin/env bash
# Can a mori bench be stood up on f01-1 without a 100 GB image pull?
set -uo pipefail
echo "== home =="
ls -a ~ 2>/dev/null | head -20
echo "== any mori checkout on disk =="
ls -d /root/mori* ~/mori* /home/*/mori* 2>/dev/null | head -5
echo "== is the EPV2 image here =="
docker images | grep -iE 'ufb-private|atom-aiter' | head -5 || echo "(no ufb-private)"
echo "== candidate image: torch + hipcc =="
IMG="${IMG:-rocm/fw-bringup:gfx1250-atom-dev-20260729-update-compiler}"
docker run --rm --entrypoint bash "$IMG" -lc 'python3 -c "import torch;print(\"torch\",torch.__version__)" 2>&1|tail -1; which hipcc clang++ cmake 2>&1|head -3; python3 -c "import mori;print(\"mori present\")" 2>&1|tail -1' 2>&1 | head -10
echo "== registry reachable =="
timeout 60 docker pull rocm/ufb-private:atom-aiter-flydsl-triton-mi450-main-py3.12-rocm7.15.0a20260717-b5a26a95-03bb2d83 2>&1 | tail -3
echo "F1PROBE2_DONE"
