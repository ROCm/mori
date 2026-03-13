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

cmake --build . -j"$(nproc)" --target umbp_master umbp_client umbp_pool_client test_umbp_types \
  test_umbp_client_registry test_umbp_block_index \
  test_umbp_route_get_strategy test_umbp_route_put_strategy test_umbp_router \
  test_umbp_pool_allocator test_umbp_peer_service test_umbp_pool_client

echo ""
echo "Build complete! Binaries:"
echo "  Master: ${BUILD_DIR}/src/umbp/umbp_master"
echo "  Client: ${BUILD_DIR}/src/umbp/umbp_client"
echo "  PoolClient: ${BUILD_DIR}/src/umbp/umbp_pool_client"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_types"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_client_registry"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_block_index"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_route_get_strategy"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_route_put_strategy"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_router"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_pool_allocator"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_peer_service"
echo "  Test:   ${BUILD_DIR}/tests/cpp/umbp/test_umbp_pool_client"
echo ""
echo "=== MasterClient demo (control plane only) ==="
echo "  Tests: RoutePut/RouteGet/Register/Unregister gRPC round-trip with simulated MORI-IO"
echo "  Terminal 1: ./${BUILD_DIR}/src/umbp/umbp_master 0.0.0.0:50051"
echo "  Terminal 2: ./${BUILD_DIR}/src/umbp/umbp_client localhost:50051 node-1 localhost:8080"
echo "  Terminal 3: ./${BUILD_DIR}/src/umbp/umbp_client localhost:50051 node-2 localhost:8081"
echo ""
echo "=== PoolClient demo (control + data plane) ==="
echo "  Tests: actual Put/Get/Remove with 4-way dispatch (local/remote × DRAM/SSD)"
echo ""
echo "  1) Local DRAM/SSD (single node, no RDMA needed):"
echo "     Terminal 1: ./${BUILD_DIR}/src/umbp/umbp_master 0.0.0.0:50051"
echo "     Terminal 2: ./${BUILD_DIR}/src/umbp/umbp_pool_client localhost:50051 node-1 localhost:8080 --provider"
echo ""
echo "  2) Remote DRAM via RDMA (provider + consumer, 2 nodes):"
echo "     Terminal 1: ./${BUILD_DIR}/src/umbp/umbp_master 0.0.0.0:50051"
echo "     Terminal 2: ./${BUILD_DIR}/src/umbp/umbp_pool_client localhost:50051 node-1 localhost:8080 \\"
echo "       --provider --tier dram --io-host <RDMA_IP> --io-port 18080 --peer-port 19080"
echo "     Terminal 3: ./${BUILD_DIR}/src/umbp/umbp_pool_client localhost:50051 node-2 localhost:8081 \\"
echo "       --consumer --io-host <RDMA_IP> --io-port 18081"
echo ""
echo "  3) Remote SSD via RDMA + PeerService (provider + consumer, 2 nodes):"
echo "     Terminal 2: ./${BUILD_DIR}/src/umbp/umbp_pool_client localhost:50051 node-1 localhost:8080 \\"
echo "       --provider --tier ssd --ssd-dir /mnt/nvme0/umbp_ssd --io-host <RDMA_IP> --io-port 18080 --peer-port 19080"
echo "     Terminal 3: ./${BUILD_DIR}/src/umbp/umbp_pool_client localhost:50051 node-2 localhost:8081 \\"
echo "       --consumer --io-host <RDMA_IP> --io-port 18081"
echo ""
echo "Run UMBP tests:"
echo "  ctest --test-dir ${BUILD_DIR} -R umbp --output-on-failure"
