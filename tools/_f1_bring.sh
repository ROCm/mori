#!/usr/bin/env bash
# Stand up a MORI bench container on f01-1, since f01-2 has not answered ssh since its reboot.
#
# The image already ships a mori 1.2.2.dev20260713 wheel. That wheel must be hidden, not merely
# shadowed: it lives in dist-packages, which precedes anything a .pth appends, so leaving it in
# place would run the image's kernels and the image's JIT sources while every log line said
# tdm-dispatch. That is the same failure mode as the build-cache key -- a number attributed to
# code that never ran.
set -uo pipefail
IMG="${IMG:-rocm/fw-bringup:gfx1250-atom-dev-20260729-update-compiler}"
NC="${NC:-MORI-F1}"
SRC=/root/mori_tdm
BR="${BR:-tdm-dispatch}"

echo "== recreate $NC =="
docker rm -f "$NC" 2>/dev/null | tail -1
docker run -d --name "$NC" --privileged --ipc=host --network=host \
  --shm-size=64g --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --entrypoint sleep "$IMG" infinity 2>&1 | tail -1
sleep 4
docker ps --filter "name=$NC" --format '{{.Names}} {{.Status}}'

echo "== clone $BR =="
docker exec "$NC" bash -lc "
  git clone --quiet --branch $BR --depth 50 https://github.com/ROCm/mori.git $SRC 2>&1 | tail -2
  git -C $SRC log --oneline -1
  git config --global --add safe.directory $SRC
"

echo "== hide the image's mori wheel, point python at the checkout =="
docker exec "$NC" bash -lc '
  D=$(python3 -c "import site;print(site.getsitepackages()[0])")
  echo "site: $D"
  [ -d "$D/mori" ] && mv "$D/mori" "$D/mori.imgbak" && echo "hid $D/mori"
  echo /root/mori_tdm/python > "$D/mori_dev.pth"
  HN=$(hostname); grep -q "$HN" /etc/hosts || echo "127.0.0.1 $HN" >> /etc/hosts
  python3 -c "import mori,os;print(\"import mori ->\", os.path.dirname(mori.__file__))" 2>&1 | tail -3
'

echo "== launch build (background, /tmp/mori_build.log) =="
docker exec -d "$NC" bash -lc "
  set -uo pipefail; cd $SRC; : > /tmp/mori_build.log
  echo \">>> start \$(date +%T)\" >> /tmp/mori_build.log
  pip install -U --break-system-packages setuptools cython wheel pybind11 ninja >> /tmp/mori_build.log 2>&1
  echo \"pip_rc=\$?\" >> /tmp/mori_build.log
  git submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c >> /tmp/mori_build.log 2>&1
  echo \"submod_rc=\$?\" >> /tmp/mori_build.log
  MORI_GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace >> /tmp/mori_build.log 2>&1
  echo \"BUILD_RC=\$?\" >> /tmp/mori_build.log
  python3 -c 'import mori,os;print(\"mori:\",os.path.dirname(mori.__file__));import mori.cco;print(\"cco ok\")' >> /tmp/mori_build.log 2>&1
  echo \"IMPORT_RC=\$?\" >> /tmp/mori_build.log
  echo \">>> done \$(date +%T)\" >> /tmp/mori_build.log
"
sleep 20
docker exec "$NC" bash -lc 'tail -5 /tmp/mori_build.log'
echo "F1BRING_DONE"
