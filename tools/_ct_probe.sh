#!/bin/bash
# Read-only survey of a node after a reservation recycle: is the container there, is the repo there,
# do the GPUs enumerate. Everything is guarded so a missing piece reports itself instead of aborting.
set +e
C=${C:-MORI-F1}
echo "PROBE_HOST=$(hostname)"
docker start "$C" >/dev/null 2>&1
echo "PROBE_START_RC=$?"
docker ps --format '{{.Names}} {{.Status}}' | grep -i mori || echo "PROBE_NOCONTAINER"
docker exec "$C" bash -lc '
  echo "IN_C=$(hostname)"
  for d in /root/mori_tdm /root/mori /workspace/mori; do
    [ -d "$d" ] && echo "REPO=$d" && cd "$d" && git log --oneline -1 && git status -s | head -5 && break
  done
  ls /dev/dri/renderD* 2>/dev/null | wc -l | sed "s/^/RENDER_NODES=/"
  python3 -c "import torch;print(\"TORCH=\"+torch.__version__, \"GPUS=\"+str(torch.cuda.device_count()))" 2>&1 | tail -2
  python3 -c "import mori;print(\"MORI_OK\", mori.__file__)" 2>&1 | tail -2
' 2>&1
echo "PROBE_DONE"
