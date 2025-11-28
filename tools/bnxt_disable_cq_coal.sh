# // Copyright © Advanced Micro Devices, Inc. All rights reserved.
# //
# // MIT License
# //
# // Permission is hereby granted, free of charge, to any person obtaining a copy
# // of this software and associated documentation files (the "Software"), to deal
# // in the Software without restriction, including without limitation the rights
# // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# // copies of the Software, and to permit persons to whom the Software is
# // furnished to do so, subject to the following conditions:
# //
# // The above copyright notice and this permission notice shall be included in all
# // copies or substantial portions of the Software.
# //
# // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# // SOFTWARE.
#!/usr/bin/env bash
#
#  Discover all bnxt_re RDMA devices
#  Automatically create the configfs directory if needed
#  Set both cq_coal_normal_maxbuf and cq_coal_during_maxbuf to 0x1
#

set -euo pipefail

CFG_ROOT="/sys/kernel/config/bnxt_re"
IB_ROOT="/sys/class/infiniband"
VALUE="0x1"

if [[ ! -d $IB_ROOT ]]; then
  echo "Error: $IB_ROOT not present, is the RDMA stack loaded?"
  exit 1
fi

mapfile -t IB_DEVS < <(find "$IB_ROOT" -maxdepth 1 -mindepth 1 -type l -printf '%f\n' | sort)

if [[ ${#IB_DEVS[@]} -eq 0 ]]; then
  echo "No RDMA devices found under $IB_ROOT"
  exit 0
fi

echo "Detected RDMA devices: ${IB_DEVS[*]}"

for dev in "${IB_DEVS[@]}"; do
  if [[ $dev != bnxt_re* && $dev != rocep* ]]; then
    echo "Skip $dev (non-bnxt device)"
    continue
  fi

  DEV_CFG_DIR="$CFG_ROOT/$dev"
  echo
  echo ">> Processing $dev"

  if [[ ! -d $DEV_CFG_DIR ]]; then
    echo "   Creating configfs directory $DEV_CFG_DIR"
    sudo mkdir -p "$DEV_CFG_DIR" || { echo "   !! mkdir failed"; continue; }
  fi

  mapfile -t PORT_DIRS < <(find "$DEV_CFG_DIR/ports" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | sort)

  if [[ ${#PORT_DIRS[@]} -eq 0 ]]; then
    echo "   No port directories found under $DEV_CFG_DIR (driver may be inactive)"
    continue
  fi

  for port in "${PORT_DIRS[@]}"; do
    TUNE_DIR="$port/tunables"
    for file in cq_coal_normal_maxbuf cq_coal_during_maxbuf; do
      path="$TUNE_DIR/$file"
      if [[ -e $path ]]; then
        echo "   set $(basename "$port")/$file -> $VALUE"
        sudo sh -c "echo $VALUE > $path" || echo "     !! write failed"
      else
        echo "   $path not found"
      fi
    done
  done
done

echo
echo "Done."