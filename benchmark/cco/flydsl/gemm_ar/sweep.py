#!/usr/bin/env python3
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
"""Run a benchmark matrix, one point per process, and say so when one fails.

The matrices used to be six shell scripts that each re-implemented process
isolation, timeouts, occupancy capture and label tagging. They are presets here
instead. Process isolation is the one thing none of them could drop: a FlyDSL
compile failure takes the interpreter with it, and a multi-M process was once
caught reporting 697us and 1083us for two M that pad to the same size.

    python sweep.py --list
    python sweep.py gemm                     # the default regression
    python sweep.py gemm-full --out full.jsonl
    python sweep.py fused-wire

Exit status is the number of failed points, capped at 125. A sweep that fails
and exits 0 is how a broken run looks green.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = os.environ.get("PY") or sys.executable

#: The two shapes every default regression covers, and the ten the full sweep
#: does. Kept here rather than in the presets so a new preset cannot quietly
#: disagree with an existing one about what `wq_b` means.
MAIN = ["wq_b", "wo_b"]
ALL_SHAPES = MAIN + [
    "wq_a_tp4",
    "wkv_tp4",
    "wqkv_a_tp4",
    "wo_a_tp4",
    "shared_gate_up_tp4",
    "wq_b_tp8",
    "wo_b_tp8",
    "wo_a_tp8",
    "wq_b_tp1",
    "wo_b_tp1",
]

#: M values a default run covers. Deliberately includes both sides of the two
#: thresholds this operator has, because a coarse M grid is how the branch got
#: `NARROW_N_BELOW_M` wrong once already: 1024 -> 2048 skipped the crossover.
M_DEFAULT = [64, 256, 1024, 1280, 1536, 1792, 2048, 4096, 16384]
M_FULL = [1, 8, 32, 64, 256, 1024, 2048, 4096, 8192, 16384]
#: Either side of the 128/256 N-tile switch, which is what `--preset gemm-tile`
#: exists to keep honest.
M_TILE = [512, 1024, 1280, 1536, 1792, 2048, 4096]
#: The GEMV's token buckets and the boundaries between them.
M_GEMV = [1, 2, 3, 4, 8, 16, 17, 32]


def _grid(**axes):
    """Cartesian product of named axes, as a list of dicts."""
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


PRESETS = {
    # ---- single process, no collective ---------------------------------
    "gemm": dict(
        doc="the default regression: two shapes, both scopes, mori + SGLang",
        script="bench_gemm.py",
        grid=_grid(shape=MAIN, m=M_DEFAULT, scope=["linear"]),
        args=["--impl", "auto,sglang"],
    ),
    "gemm-full": dict(
        doc="every shape the checkpoint has, every M, every mori tile",
        script="bench_gemm.py",
        grid=_grid(shape=ALL_SHAPES, m=M_FULL, scope=["linear"]),
        args=["--impl", "auto,gemm256,gemm128,sglang"],
    ),
    "gemm-tile": dict(
        doc="the 128 vs 256 N-tile switch, kernel scope, either side of it",
        script="bench_gemm.py",
        grid=_grid(shape=ALL_SHAPES, m=M_TILE, scope=["kernel"]),
        args=["--impl", "gemm256,gemm128"],
    ),
    "gemm-blockscale": dict(
        doc="mori's other operand contract, kernel scope, no SGLang baseline",
        script="bench_gemm.py",
        grid=_grid(shape=MAIN, m=[4096, 8192, 16384], scope=["kernel"]),
        args=["--impl", "auto", "--quant", "blockscale"],
    ),
    "gemv": dict(
        doc="the skinny GEMM at its token buckets, default config",
        script="bench_gemv.py",
        grid=_grid(shape=MAIN, m=M_GEMV),
    ),
    "gemv-tune": dict(
        doc="the whole GEMV config space, per bucket -- this is what tunes the table",
        script="bench_gemv.py",
        grid=_grid(shape=MAIN, m=[1, 2, 4, 8, 16, 32]),
        args=["--sweep"],
        timeout=3600,
    ),
    # ---- multi rank, the collective ------------------------------------
    "fused": dict(
        doc="split vs fused at V4.1-Flash's wo_b, under the model's own mxfp8",
        script="bench_gemm_ar.py",
        world=4,
        grid=_grid(
            m=[4096, 8192, 16384],
            mode=["gemm-only", "split-sdma", "split-lsa", "fused-sdma", "fused-lsa"],
        ),
        args=["-n", "5120", "-k", "2048", "--quant", "mxfp8"],
        label="mxfp8",
    ),
    "fused-fp8": dict(
        doc="the same mode matrix on the winning fp8 wire, which is where the "
        "fusion is actually deployed. `split-lsa` is absent on purpose: it has "
        "no fp8 gather leg (`build_lsa_ar` takes no `gather_dtype`), so the "
        "bench refuses the pair at entry and those points can never run",
        script="bench_gemm_ar.py",
        world=4,
        grid=_grid(
            m=[4096, 8192, 16384],
            mode=["gemm-only", "split-sdma", "fused-sdma", "fused-lsa"],
        ),
        args=[
            "-n",
            "5120",
            "-k",
            "2048",
            "--quant",
            "mxfp8",
            "--gather-dtype",
            "fp8",
            "--gather-transport",
            "lsa",
            "--no-fuse-quantize",
        ],
        label="mxfp8-fp8wire",
    ),
    "fused-wire": dict(
        doc="how to move the fp8 all-gather leg: push or pull, fuse the quantise or not",
        script="bench_gemm_ar.py",
        world=4,
        grid=_grid(
            m=[4096, 16384],
            mode=["split-sdma", "fused-sdma", "fused-lsa"],
            gather_transport=["sdma", "lsa"],
            fuse_quantize=[True, False],
        ),
        args=["-n", "5120", "-k", "2048", "--quant", "mxfp8", "--gather-dtype", "fp8"],
        label="fp8gather",
    ),
    "fused-blockscale-control": dict(
        doc=(
            "CONTROL, not a V4.1-Flash benchmark: the same shapes under 1x128 "
            "blockscale, which is *not* what the checkpoint uses. Kept because "
            "the mxfp8 numbers are only interpretable against it -- see the "
            "README. Use `fused` for the real thing."
        ),
        script="bench_gemm_ar.py",
        world=4,
        grid=_grid(
            m=[4096, 8192, 16384],
            mode=["gemm-only", "split-sdma", "fused-sdma"],
        ),
        args=["-n", "5120", "-k", "2048", "--quant", "blockscale"],
        label="blockscale-control",
    ),
}


def require_sdma() -> None:
    """Refuse to run a collective benchmark against a build that has no queues.

    With `BUILD_CCO_SDMA=OFF` every put silently does nothing: the all-reduce
    returns mostly the local slice, the model still answers, and the fused path
    measures *faster* than it is because it is not moving data.
    """
    try:
        from mori.cco.device._build_flags import BUILD_CCO_SDMA
    except ImportError:
        # None, not False: the module is only written by setup.py's build_extension,
        # so a source tree on PYTHONPATH has no opinion -- and ON is the default,
        # so guessing OFF aborts a perfectly good build. Matches op.py.
        BUILD_CCO_SDMA = None
    if BUILD_CCO_SDMA is False:
        sys.exit(
            "BUILD_CCO_SDMA is OFF -- the SDMA path is compiled out, "
            "numbers would be fiction. Rebuild with BUILD_CCO_SDMA=ON."
        )


def occupancy() -> str:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "?"
    gib = [
        f"{int(p[2]) / 2**30:.0f}"
        for p in (line.split(",") for line in out.splitlines())
        if len(p) >= 3 and p[2].isdigit()
    ]
    return " ".join(gib) or "?"


def point_argv(preset, point, out_path):
    """The full argv for one point of the matrix."""
    script = str(HERE / preset["script"])
    extra = list(preset.get("args", []))
    world = preset.get("world")

    if world:
        # `python -m torch.distributed.run` rather than the `torchrun` console
        # script: the latter is only on PATH if the venv is activated, which a
        # subprocess inherits only by luck.
        cmd = [
            PY,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            script,
        ]
    else:
        cmd = [PY, script, "--json-out", str(out_path)]

    for key, val in point.items():
        if isinstance(val, bool):
            # bench_gemm_ar.py spells these as --fuse-quantize / --no-fuse-quantize
            cmd.append(
                f"--{key.replace('_', '-')}" if val else f"--no-{key.replace('_', '-')}"
            )
        elif key == "m":
            cmd += ["-m", str(val)]
        else:
            cmd += [f"--{key.replace('_', '-')}", str(val)]
    return cmd + extra


def run_point(preset, point, out_path, timeout):
    """One subprocess. Returns (ok, stdout)."""
    cmd = point_argv(preset, point, out_path)
    env = dict(os.environ)
    if preset.get("world"):
        env.setdefault("MORI_SOCKET_IFNAME", "lo")
        env["MORI_ENABLE_SDMA"] = "1"
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE, env=env
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    # Multi-rank runs print RESULT_JSON rather than writing the file themselves.
    if preset.get("world"):
        wrote = 0
        with open(out_path, "a") as f:
            for line in p.stdout.splitlines():
                if line.startswith("RESULT_JSON"):
                    d = json.loads(line.split(" ", 1)[1])
                    d["label"] = preset.get("label", "")
                    d.update({k: v for k, v in point.items() if k != "m"})
                    # What this sweep *varied*, so a reporter can tell an axis
                    # from a value the op resolved for itself. bench_gemm_ar.py
                    # echoes every knob it ran with, including ones it chose
                    # (chunks, critical_rank, tile order); keying a table on
                    # those splits one matrix into one table per point.
                    d["sweep_axes"] = sorted(point)
                    f.write(json.dumps(d) + "\n")
                    wrote += 1
        if p.returncode == 0 and wrote == 0:
            return False, "no RESULT_JSON emitted"
    return p.returncode == 0, p.stdout + p.stderr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.list or not args.preset:
        width = max(len(k) for k in PRESETS)
        for name, d in sorted(PRESETS.items()):
            print(f"  {name:<{width}}  {d['doc']}")
        return 0

    preset = PRESETS[args.preset]
    out_path = Path(args.out or f"{args.preset}.jsonl").resolve()
    timeout = args.timeout or preset.get("timeout", 1800)

    if args.dry_run:
        for point in preset["grid"]:
            print(" ".join(shlex.quote(c) for c in point_argv(preset, point, out_path)))
        return 0

    if preset.get("world"):
        require_sdma()
    out_path.write_text("")
    before = occupancy()
    print(f"# preset {args.preset}: {len(preset['grid'])} points -> {out_path}")
    print(f"# occupancy before: {before} GiB/card", file=sys.stderr)

    failed = []
    for i, point in enumerate(preset["grid"], 1):
        desc = " ".join(f"{k}={v}" for k, v in point.items())
        print(f"[{i}/{len(preset['grid'])}] {desc}", flush=True)
        ok, out = run_point(preset, point, out_path, timeout)
        if not ok:
            failed.append(desc)
            print(
                f"  FAILED: {out.strip().splitlines()[-1] if out.strip() else '?'}",
                flush=True,
            )
        else:
            for line in out.splitlines():
                if line.startswith("  ") or line.startswith("RESULT_JSON"):
                    print(line if line.startswith("  ") else "  ok", flush=True)
        if preset.get("world"):
            time.sleep(3)  # let the SDMA queues drain before the next spawn

    after = occupancy()
    print(f"# occupancy after: {after} GiB/card", file=sys.stderr)
    if before != after:
        print(
            "# NOTE: occupancy moved across the sweep; a leftover process may "
            "have shared the GPU. Re-run before trusting these.",
            file=sys.stderr,
        )
    if failed:
        print(f"\n{len(failed)} of {len(preset['grid'])} points FAILED:")
        for d in failed:
            print(f"  {d}")
    else:
        print(f"\nall {len(preset['grid'])} points ok -> {out_path}")
    return min(len(failed), 125)


if __name__ == "__main__":
    raise SystemExit(main())
