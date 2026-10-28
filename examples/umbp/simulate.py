# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Small API driver and JSONL timing summary; requires a MoRI build with UMBP."""

import argparse
import ctypes
import json
import mmap
import statistics
from collections import defaultdict
from pathlib import Path

from mori.umbp import UMBPClient, UMBPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default=str(Path(__file__).with_name("sim_profile.json"))
    )
    parser.add_argument("--trace", default="/tmp/umbp-sim.jsonl")
    parser.add_argument("--block-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.block_bytes, args.blocks, args.iterations) <= 0:
        parser.error("block size, block count and iterations must be positive")
    trace = Path(args.trace)
    # Each invocation gets its own file, so the summary cannot include earlier runs.
    with trace.open("x"):
        pass
    config = UMBPConfig.from_environment()
    config.sim.enabled = True
    config.sim.profile_path = args.profile
    config.sim.trace_path = str(trace)
    config.page_size = args.block_bytes
    config.dram.capacity_bytes = 2 * args.blocks * args.block_bytes
    config.ssd.enabled = False
    # This driver exercises one embedded DRAM owner. Frameworks keep their own config.
    config.distributed = None
    config.standalone_process = None
    config.backend_policy_path = ""
    keys = [f"block-{i}" for i in range(args.blocks)]
    sizes = [args.block_bytes] * args.blocks
    with mmap.mmap(-1, args.blocks * args.block_bytes) as buffer:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        ptrs = [ptr + i * args.block_bytes for i in range(args.blocks)]
        client = UMBPClient(config)
        try:
            assert client.register_memory(ptr, len(buffer))
            for _ in range(args.iterations):
                assert all(client.batch_put_from_ptr(keys, ptrs, sizes))
                assert all(client.batch_exists(keys))
                assert all(client.batch_get_into_ptr(keys, ptrs, sizes))
                assert client.clear()
        finally:
            del client  # The Python binding closes the client on destruction.
    elapsed = defaultdict(list)
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        elapsed[record["api"]].append(record["elapsed_ns"] / 1e6)
    print("API                         calls      mean_ms    median_ms")
    for api, values in sorted(elapsed.items()):
        print(
            f"{api:28} {len(values):5} {statistics.mean(values):12.3f} "
            f"{statistics.median(values):12.3f}"
        )
    print(f"Trace: {trace}")


if __name__ == "__main__":
    main()
