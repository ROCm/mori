#!/usr/bin/env bash
# Finish the reset that got cut off, one device at a time, and say what each one did.
#
# The earlier hardreset was interrupted by the ssh timeout DURING `--gpureset -d 1`, and after that
# even the host's rocm-smi fails at amdgpu_query_gpu_info_init. Devices 1-3 were very likely never
# reset. Each call gets its own generous budget so the script cannot be cut short in the middle of
# one again.
set -uo pipefail
echo "== rocm-smi, first look =="
timeout 60 rocm-smi --showid 2>&1 | head -8
for d in 1 2 3; do
  echo "--- gpureset -d $d ---"
  timeout 180 rocm-smi --gpureset -d $d 2>&1 | grep -iE "reset|error|success|fail|initialize" | head -4 || echo "rc=$?"
done
echo "== rocm-smi, after =="
timeout 60 rocm-smi --showid 2>&1 | head -8
timeout 60 rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -4
echo "QFIX_DONE"
