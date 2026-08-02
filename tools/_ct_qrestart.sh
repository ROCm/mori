#!/usr/bin/env bash
# Restart the container and prove torch can see the four GPUs before any timing run.
#
# rocm-smi recovered on its own once the interrupted --gpureset finished; what did not recover is
# the container's view, which is why the bench came back with "No CUDA GPUs are available" on every
# rank. A restart is what re-opens /dev/kfd against the reset devices.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
timeout 180 docker restart "$CTR" 2>&1 | tail -2
sleep 8
docker exec "$CTR" bash -lc 'HN=$(hostname); grep -q "$HN" /etc/hosts || echo "127.0.0.1 $HN" >> /etc/hosts; echo hosts_ok'
docker exec "$CTR" bash -lc 'timeout 120 python3 -c "import torch;print(\"torch sees\", torch.cuda.device_count(), \"gpus\")" 2>&1 | tail -3'
echo "QRESTART_DONE"
