#!/usr/bin/env bash
# Can the container's torch see the GPUs? Nothing below this works until it can.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
docker exec "$CTR" bash -lc 'timeout 300 python3 -c "
import torch
print(\"available\", torch.cuda.is_available())
print(\"count\", torch.cuda.device_count())
x = torch.ones(1024, device=\"cuda:0\")
print(\"sum\", float(x.sum()))
" 2>&1 | tail -12'
echo "rc=$?"
echo "QTORCH_DONE"
