#!/usr/bin/env bash
# Stand up a mori EP bench container on a bare gfx1250 node, from an image that has no mori in it
# that can be trusted.
#
# Written after f01-2 stopped answering ssh and the work had to move to f01-1 with no MORI-EPV2
# image available there. Everything below is a fix for something this image actually got wrong, in
# the order the build hits them; none of it is precautionary.
#
#   IMG   a rocm/fw-bringup gfx1250 image (has torch + hipcc + cmake; ships an old mori wheel)
#   NC    container name          BR   branch to build
#
# Run on the NODE, not in the container:  IMG=... ./_node_bringup_gfx1250.sh
set -uo pipefail
IMG="${IMG:-rocm/fw-bringup:gfx1250-atom-dev-20260729-update-compiler}"
NC="${NC:-MORI-F1}"
BR="${BR:-tdm-dispatch}"
SRC=/root/mori_tdm

echo "== container =="
docker rm -f "$NC" 2>/dev/null | tail -1
docker run -d --name "$NC" --privileged --ipc=host --network=host \
  --shm-size=64g --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --entrypoint sleep "$IMG" infinity 2>&1 | tail -1
sleep 4
docker ps --filter "name=$NC" --format '{{.Names}} {{.Status}}'

echo "== checkout $BR =="
docker exec "$NC" bash -lc "
  git clone --quiet --branch $BR https://github.com/ROCm/mori.git $SRC 2>&1 | tail -2
  git config --global --add safe.directory $SRC
  git -C $SRC log --oneline -1
"

# The image's own mori wheel sits in dist-packages, which precedes anything a .pth appends. Leaving
# it there runs the image's python AND its _jit-sources while every log line says this branch --
# the same class of error as a build-cache key that misses a -D, and just as invisible.
echo "== hide the image's mori, point python at the checkout =="
docker exec "$NC" bash -lc "
  D=\$(python3 -c 'import site;print(site.getsitepackages()[0])')
  [ -d \"\$D/mori\" ] && mv \"\$D/mori\" \"\$D/mori.imgbak\" && echo \"hid \$D/mori\"
  echo $SRC/python > \"\$D/mori_dev.pth\"
  HN=\$(hostname); grep -q \"\$HN\" /etc/hosts || echo \"127.0.0.1 \$HN\" >> /etc/hosts
  python3 -c 'import mori,os;print(\"import mori ->\", os.path.dirname(mori.__file__))'
"

# The ROCm here is the pip SDK, split into _rocm_sdk_core (compiler, runtime, ROCM_PATH points at
# it) and _rocm_sdk_devel (headers, cmake packages). Four things follow from that split, plus one
# from the packager's own distro:
#   1. no lib/cmake under ROCM_PATH          -> CMakeDetermineHIPCompiler finds no hip-lang
#   2. AMDDeviceLibsConfig redirects to ../../llvm, which resolves to neither half
#   3. imported targets name libfoo.so.<full-version>, and only the soname link is shipped
#   4. find_library() wants bare libfoo.so development symlinks, also not shipped
#   5. hsakmtTargets.cmake hardcodes /usr/lib64/libc.so, which Debian does not have
echo "== patch the pip ROCm SDK =="
docker exec "$NC" bash -lc '
  C=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
  D=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
  [ -e "$C/lib/cmake" ] || ln -s "$D/lib/cmake" "$C/lib/cmake"
  rm -f "$C/llvm"; ln -s "$D/lib/llvm" "$C/llvm"
  # 2: the redirect resolves by hand but not from cmake, so make it absolute instead of relative.
  S="$D/lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake"
  [ -e "$S.orig" ] || cp "$S" "$S.orig"
  printf "include(\"%s/lib/llvm/lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake\")\n" "$D" > "$S"
  # 3: every IMPORTED_LOCATION basename the cmake packages name, linked to the real soname.
  n=0
  for b in $(grep -rhoE "lib[A-Za-z0-9_.+-]*\.so[0-9A-Za-z._-]*" "$C/lib/cmake" 2>/dev/null | sort -u); do
    [ -e "$C/lib/$b" ] && continue
    s="${b%%.so*}.so"
    for cand in "$C/lib/$s".[0-9] "$C/lib/$s".[0-9].[0-9] "$C/lib/$s"; do
      [ -e "$cand" ] || continue
      ln -sfn "$(basename "$cand")" "$C/lib/$b" && n=$((n+1)); break
    done
  done
  echo "versioned links: $n"
  # 4
  m=0
  for f in "$C"/lib/lib*.so.[0-9]; do
    b=$(basename "$f"); s="${b%.so.*}.so"
    [ -e "$C/lib/$s" ] && continue
    ln -sfn "$b" "$C/lib/$s" && m=$((m+1))
  done
  echo "dev links: $m"
  # 5
  [ -e /usr/lib64/libc.so ] || ln -s /usr/lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so
'

# pip refuses to upgrade the debian-installed wheel package (no RECORD), which aborts the whole
# install and leaves Cython missing -- and a missing Cython only WARNS, then skips mori.cco.
echo "== build (BUILD_UMBP=OFF: it needs gRPC headers this image lacks and the bench never enters it) =="
docker exec -d "$NC" bash -lc "
  set -uo pipefail; cd $SRC; : > /tmp/mori_build.log
  echo \">>> start \$(date +%T)\" >> /tmp/mori_build.log
  pip install -q --break-system-packages --ignore-installed cython pybind11 ninja >> /tmp/mori_build.log 2>&1
  echo \"pip_rc=\$?\" >> /tmp/mori_build.log
  git submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c >> /tmp/mori_build.log 2>&1
  BUILD_UMBP=OFF MORI_GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace -j 40 >> /tmp/mori_build.log 2>&1
  echo \"BUILD_RC=\$?\" >> /tmp/mori_build.log
  python3 -c 'import mori,os;print(\"mori:\",os.path.dirname(mori.__file__));import mori.cco;print(\"cco ok\")' >> /tmp/mori_build.log 2>&1
  echo \"IMPORT_RC=\$?\" >> /tmp/mori_build.log
  echo \">>> done \$(date +%T)\" >> /tmp/mori_build.log
"
echo "build launched; expect BUILD_RC=0 IMPORT_RC=0 within a few minutes:"
sleep 90
docker exec "$NC" bash -lc 'grep -nE "^(pip_rc|BUILD_RC|IMPORT_RC|>>>)" /tmp/mori_build.log; grep -nE "CMake Error|ninja: error|error:" /tmp/mori_build.log | head -5'
echo "BRINGUP_DONE"
