#!/bin/bash
# Run from ~/mori-ofi on the login node to configure+build.
set -e

module purge
module load rocm/6.4.1
module load libfabric/2.2.0rc1
module load cray-mpich/8.1.30
module load cmake/3.25.1

MORI_ROOT=${HOME}/mori-ofi
BUILD_DIR=${MORI_ROOT}/build-ofi

# pkg-config path for libfabric
export PKG_CONFIG_PATH=/opt/cray/libfabric/2.2.0rc1/lib64/pkgconfig:${PKG_CONFIG_PATH}

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${MORI_ROOT} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMORI_USE_LIBFABRIC=ON \
  -DCMAKE_PREFIX_PATH="/opt/rocm-6.4.1;/opt/cray/libfabric/2.2.0rc1" \
  -DCMAKE_C_COMPILER=$(which mpicc) \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  -DMORI_BUILD_TESTS=ON \
  2>&1 | tee cmake_configure.log

make -j$(nproc) mori_io 2>&1 | tee build_mori_io.log

# Build the standalone OFI test
mpicxx -std=c++17 -O2 \
  -I${MORI_ROOT}/include \
  -I/opt/cray/libfabric/2.2.0rc1/include \
  -I${MORI_ROOT}/src \
  -L${BUILD_DIR} -lmori_io \
  -L/opt/cray/libfabric/2.2.0rc1/lib64 -lfabric \
  -Wl,-rpath,${BUILD_DIR} \
  -Wl,-rpath,/opt/cray/libfabric/2.2.0rc1/lib64 \
  ${MORI_ROOT}/tests/cpp/io/ofi_slingshot_test.cpp \
  -o ${BUILD_DIR}/tests/ofi_slingshot_test \
  2>&1 | tee build_test.log

echo "Build complete: ${BUILD_DIR}/tests/ofi_slingshot_test"
