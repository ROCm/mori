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
"""Several runs of one FlyDSL benchmark in one torchrun launch.

    torchrun --nproc_per_node=8 bench_runs.py BENCH.py '[["--mode", "fused-sdma"], ...]'

BENCH.py has the benchmarks' shape: ``build_parser()`` and ``run(args)``. Each run
opens and closes a Communicator of its own, as a launch of BENCH.py would, and
prints its own RESULT_JSON, in order. Every run goes ahead even after one fails;
the exit code is the largest a run returned.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch.distributed as dist


def main() -> int:
    bench_path, runs = Path(sys.argv[1]), json.loads(sys.argv[2])
    # As if BENCH.py were the script: its siblings (timing.py) import by name.
    sys.path.insert(0, str(bench_path.parent))
    spec = importlib.util.spec_from_file_location(bench_path.stem, bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rcs = [bench.run(bench.build_parser().parse_args(args)) for args in runs]
    return max(rcs, default=0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
