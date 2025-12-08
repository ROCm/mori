#!/bin/bash
# Mori AllReduce fast build script

set -e # exit on error

# color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Mori AllReduce build script${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# check dependencies
echo -e "${YELLOW}[1/6] check dependencies...${NC}"

# check MPI
if ! command -v mpirun &>/dev/null; then
  echo -e "${RED}error: MPI not found!${NC}"
  echo "install: sudo apt-get install libopenmpi-dev"
  exit 1
fi
echo "  ✓ MPI: $(mpirun --version | head -1)"

# check HIP
if ! command -v hipcc &>/dev/null; then
  echo -e "${RED}error: HIP not found!${NC}"
  echo "install ROCm: https://rocm.docs.amd.com/"
  exit 1
fi
echo "  ✓ HIP: $(hipcc --version | grep -i hip | head -1)"

# check CMake
if ! command -v cmake &>/dev/null; then
  echo -e "${RED}error: CMake not found!${NC}"
  echo "install: sudo apt-get install cmake"
  exit 1
fi
echo "  ✓ CMake: $(cmake --version | head -1)"

echo ""

# create and enter build directory
echo -e "${YELLOW}[2/6] create build directory...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "build" ]; then
  echo "  build directory already exists"
  read -p "  clean old build? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  cleaning..."
    rm -rf build
    mkdir build
  fi
else
  mkdir build
fi

cd build
echo "  ✓ working directory: $(pwd)"
echo ""

# configure CMake
echo -e "${YELLOW}[3/6] configure CMake...${NC}"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ROCM=ON \
  -DBUILD_COLLECTIVE=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON

if [ $? -ne 0 ]; then
  echo -e "${RED}CMake configuration failed!${NC}"
  exit 1
fi
echo "  ✓ CMake configuration successful"
echo ""

# compile
echo -e "${YELLOW}[4/6] compiling...${NC}"
NPROC=$(nproc)
echo "  using ${NPROC} cores to compile"

cmake --build . -j ${NPROC}

if [ $? -ne 0 ]; then
  echo -e "${RED}compilation failed!${NC}"
  exit 1
fi
echo "  ✓ compilation successful"
echo ""

# verify compilation result
echo -e "${YELLOW}[5/6] verify compilation result...${NC}"

# check library
if [ -f "src/collective/libmori_collective.so" ]; then
  echo "  ✓ libmori_collective.so generated successfully"
  ls -lh src/collective/libmori_collective.so | awk '{print "    size:", $5}'
else
  echo -e "${RED}  ✗ libmori_collective.so not found${NC}"
  exit 1
fi

# check example program
if [ -f "examples/allreduce_example" ]; then
  echo "  ✓ allreduce_example generated successfully"
else
  echo -e "${RED}  ✗ allreduce_example not found${NC}"
  exit 1
fi

echo ""

# done
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}compilation done!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# run instructions
echo -e "${YELLOW}[6/6] run instructions:${NC}"
echo ""
echo "single node 2 GPU test:"
echo "  cd $(pwd)"
echo "  mpirun -np 2 ./examples/allreduce_example"
echo ""
echo "single node 4 GPU test:"
echo "  mpirun -np 4 ./examples/allreduce_example"
echo ""
echo "specify GPUs:"
echo "  export HIP_VISIBLE_DEVICES=0,1"
echo "  mpirun -np 2 ./examples/allreduce_example"
echo ""
echo "view more documents:"
echo "  cat $(dirname $SCRIPT_DIR)/docs/ALLREDUCE_BUILD_GUIDE.md"
echo ""

# ask whether to run immediately
read -p "run test immediately? (using 8 GPUs) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo ""
  echo -e "${GREEN}running test...${NC}"
  echo ""
  mpirun --allow-run-as-root -np 8 ./examples/allreduce_example
fi
