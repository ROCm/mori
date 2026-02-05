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
  cmake -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_EXAMPLES=OFF -DWARP_ACCUM_UNROLL=1 -DUSE_BNXT=OFF \
      -DGPU_TARGETS=gfx942 \
      .. 

   
fi

make VERBOSE=1 -j 2>&1 | tee ../yyybuild.log

cp src/pybind/libmori_pybinds.so \
   src/application/libmori_application.so \
   src/io/libmori_io.so \
   $SCRIPT_DIR/python/mori

popd

if [[ ${full} -eq 1 ]]; then
  pip3 install -e .
fi
