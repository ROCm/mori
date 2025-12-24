#!/bin/bash
set -e

# -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake

#pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1

full=${full:-0}

mkdir -p build
pushd build

if [[ ${full} -eq 1 ]]; then
  rm -rf *
  cmake -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Release \
      -DWARP_ACCUM_UNROLL=1 -DUSE_BNXT=OFF \
      -DGPU_TARGETS=gfx942 \
      ..
fi

make -j

cp src/pybind/libmori_pybinds.so \
   src/application/libmori_application.so \
   src/io/libmori_io.so \
   /tf/mori_deepep/mori/python/mori

popd