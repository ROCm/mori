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
"""Several benchmark runs in one torchrun launch, each run naming its own benchmark file.

    torchrun --nproc_per_node=8 bench_driver.py '[{"bench": "A.py", "args": [...]}, ...]'

``tests/python/cco/bench_runs.py`` with one change: every run says which benchmark it
runs, so two implementations (the existing operator and a candidate, say) can be timed
interleaved in the same processes. A benchmark has the shape the FlyDSL benchmarks
share: ``build_parser()`` and ``run(args)``, a Communicator opened and closed per run,
and a ``RESULT_JSON`` line from rank 0. Each file is loaded once, with its directory on
``sys.path`` so its sibling imports resolve. Every run goes ahead even after one fails;
the exit status is the largest any run returned.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load(path, cache):
    path = Path(path).resolve()
    if path not in cache:
        sys.path.insert(0, str(path.parent))
        name = f"bench_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cache[path] = module
    return cache[path]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs, cache, codes = json.loads(argv[0]), {}, []
    for run in runs:
        bench = load(run["bench"], cache)
        codes.append(bench.run(bench.build_parser().parse_args(run["args"])) or 0)
    return max(codes, default=0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if "torch.distributed" in sys.modules:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
