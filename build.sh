#!/bin/bash
echo "Removing cache..."
rm -rf ~/.mori/
rm -rf build/

echo "Rebuild examples..."
BUILD_EXAMPLES=ON pip install .
echo "Trigger compiling of kernel code"
MORI_PRECOMPILE=1 python -c "import mori"