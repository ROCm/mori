#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake

#pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1

full=${full:-0}

mkdir -p build
pushd build

if [[ ${full} -eq 1 ]]; then

#  apt-get install -y \
#     git \
#    ibverbs-utils \
#     libpci-dev \
#     libdw1 \
#     cython3 

# NOTE this would screw up hipcc installation!!!
# better install MPI manually
    #openmpi-bin \
    #libopenmpi-dev \
    #locales

  rm -rf *
  cmake -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Debug -DMORI_MULTITHREAD_SUPPORT=ON \
      -DBUILD_EXAMPLES=ON -DWARP_ACCUM_UNROLL=1 -DUSE_BNXT=OFF \
      -DBUILD_APPLICATION=OFF -DBUILD_SHMEM=ON -DBUILD_OPS=OFF -DBUILD_IO=OFF \
      -DBUILD_PYBINDS=OFF -DBUILD_UMBP=OFF -DBUILD_TESTS=OFF -DENABLE_PROFILER=OFF \
      -DENABLE_DEBUG_PRINTF=OFF -DENABLE_STANDARD_MOE_ADAPT=OFF -DBUILD_OPS_DEVICE=OFF \
      -DGPU_TARGETS=gfx942 .. 
fi

make VERBOSE=1 -j 2>&1 | tee ../yyybuild.log

rm -f $SCRIPT_DIR/python/mori/*.so
find src -name "*.so" -exec cp {} $SCRIPT_DIR/python/mori \;

popd
