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
      -DGPU_TARGETS=gfx942 -DFORCE_CODE_OBJECT_VERSION_5=OFF \
      -DBUILD_TORCH_BOOTSTRAP=OFF -DWITH_MPI=OFF \
      -DBUILD_XLA_FFI_OPS=ON -DBUILD_OPS_DEVICE=ON \
      .. 

  # cmake -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Release \
  #     -DBUILD_EXAMPLES=OFF -DWARP_ACCUM_UNROLL=1 -DUSE_BNXT=OFF \
  #     -DGPU_TARGETS=gfx942 -DFORCE_CODE_OBJECT_VERSION_5=ON \
  #     -DBUILD_TORCH_BOOTSTRAP=ON -DWITH_MPI=ON \
  #     -DBUILD_XLA_FFI_OPS=OFF -DBUILD_OPS_DEVICE=OFF \
  #     .. 
fi

make VERBOSE=1 -j 2>&1 | tee ../yyybuild.log

rm -f $SCRIPT_DIR/python/mori/*.so
find src -name "*.so" -exec cp {} $SCRIPT_DIR/python/mori \;

popd

if [[ ${full} -eq 1 ]]; then
  pip3 install -e . --no-build-isolation
fi
