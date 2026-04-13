#！/bin/bash

set -x

rm -rf build && mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_PERFTEST=ON -DBUILD_PYBINDS=OFF -DWITH_MPI=ON
make -j$(nproc)

echo "Building completed"
