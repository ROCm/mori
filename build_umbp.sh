#!/bin/bash
# Build script for UMBP master/client and tests
# Usage: ./build_umbp.sh [build_dir]
#
# First build will be slow (~10-20 min) because FetchContent downloads
# and compiles gRPC + abseil + protobuf + c-ares + re2 + zlib.
# Subsequent builds are fast (FetchContent caches in the build dir).

set -e

BUILD_DIR="${1:-build_umbp}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DBUILD_UMBP=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . -j"$(nproc)" --target umbp_master umbp_client test_umbp_types \
  test_umbp_client_registry test_umbp_block_index \
  test_umbp_route_get_strategy test_umbp_route_put_strategy test_umbp_router

echo ""
echo "Build complete! Binaries:"
echo "  Master: ${BUILD_DIR}/src/umbp/umbp_master"
echo "  Client: ${BUILD_DIR}/src/umbp/umbp_client"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_types"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_client_registry"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_block_index"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_route_get_strategy"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_route_put_strategy"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_router"
echo ""
echo "Quick test:"
echo "  (Run from MORI top-level directory: ${ROOT_DIR})"
echo "  Terminal 1: ./${BUILD_DIR}/src/umbp/umbp_master 0.0.0.0:50051"
echo "  Terminal 2: ./${BUILD_DIR}/src/umbp/umbp_client localhost:50051 node-1 localhost:8080"
echo "  Terminal 3: ./${BUILD_DIR}/src/umbp/umbp_client localhost:50051 node-2 localhost:8081"
echo ""
echo "Run UMBP tests:"
echo "  ctest --test-dir ${BUILD_DIR} -R umbp --output-on-failure"
