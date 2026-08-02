#!/usr/bin/env bash
# Why the container says "No CUDA GPUs are available" after a host-side GPU reset.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
echo "== host devices =="
ls -l /dev/kfd 2>&1 | head -2
ls /dev/dri 2>&1 | head -12
echo "== host rocm-smi =="
timeout 20 rocm-smi --showid 2>&1 | head -12
echo "== container devices =="
docker exec "$CTR" bash -lc 'ls -l /dev/kfd 2>&1 | head -2; ls /dev/dri 2>&1 | head -12'
echo "== container rocm-smi =="
docker exec "$CTR" bash -lc 'timeout 20 rocm-smi --showid 2>&1 | head -12'
echo "== container torch =="
docker exec "$CTR" bash -lc 'timeout 60 python3 -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())" 2>&1 | tail -3'
echo "QDEV_DONE"
