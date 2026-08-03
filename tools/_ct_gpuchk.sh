#!/usr/bin/env bash
# Are the GPUs actually there. The container came back with torch seeing 0 devices and half the
# render nodes, which is either the container holding stale device handles from before the reboot
# or the driver not up yet -- and those want opposite actions, so ask before doing either.
set +e
C=${C:-MORI-F1}
echo "HOST_RENDER=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)"
echo "HOST_KFD=$(ls -l /dev/kfd 2>/dev/null | wc -l)"
docker exec "$C" bash -lc 'echo "C_RENDER=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)"; echo "C_KFD=$(ls /dev/kfd 2>/dev/null | wc -l)"'
echo "-- host rocm-smi --"
/opt/rocm/bin/rocm-smi --showid 2>&1 | grep -iE "GPU|card|error|not" | head -12
echo "-- container rocm-smi --"
docker exec "$C" bash -lc 'rocm-smi --showid 2>&1 | grep -iE "GPU|card|error|not" | head -12'
echo "-- container hip enumeration --"
docker exec "$C" bash -lc 'python3 -c "
import torch
print(\"count\", torch.cuda.device_count())
print(\"avail\", torch.cuda.is_available())
" 2>&1 | tail -6'
echo "GPUCHK_DONE"
