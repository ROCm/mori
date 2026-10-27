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
"""EPv2 must give its VRAM back when the process group exits.

On some machines it does not. A process that registered a cco symmetric window
exits -- cleanly, every rank reaching ``clean exit`` -- and each GPU keeps a
block of VRAM that no process owns. It is still there minutes later, and only a
reboot returns it. The residue sits on dma-bufs: one per rank, each with a
refcount equal to the number of *other* GPUs, attached to exactly those GPUs.
Every peer that imported the window left one reference behind.

This cannot be an in-process test: the residue only exists once every rank is
**gone**, so the measuring process has to outlive the measured ones. Hence the
house split already used by ``test_op.py`` /
``test_dispatch_combine_v2_intranode.py`` -- a torchrun worker plus a wrapper --
except here the wrapper does the measuring and the worker is the subject.

Five cases run as separate process groups so a failure says *which* layer:

    comm              flat-VA reservation only                    control
    window_solo       register a window, world=1, no peers        control
    window_same_gpu   peers, but all pinned to one GPU            control
    window            peers on their own GPUs                     <- the leak
    op                full EpDispatchCombineOp traffic

The three controls are clean on every machine measured so far. They are the
difference between "this box strands VRAM on cross-GPU window import" and "this
box strands VRAM on any VMM allocation", which are not the same bug. If a
control fails, stop reading the rest -- something more basic is wrong.

**Running it costs the machine.** On an affected box each failing case strands
its own copy of the residue and nothing short of a reboot returns it. That is
why it is off unless ``MORI_RUN_VRAM_LEAK_TEST=1``, and why it skips on a node
that is not idle: a delta measured against another tenant's churn is noise.

    MORI_RUN_VRAM_LEAK_TEST=1 pytest -s -v test_vram_release_on_exit.py

``-s`` is worth passing: the per-GPU table and the machine fingerprint go to
stdout, and the fingerprint is what makes two CI runs comparable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _smi  # noqa: E402  (sibling module, as bench_ep.py imports it)

_WORKER = os.path.join(_HERE, "vram_leak_worker.py")
_NPROC = 8
_GIB = 1 << 30
_MIB = 1 << 20

#: Per-GPU slack. The measured deltas are exact -- a 256 MiB window shows up as
#: 256 MiB, not 255.7 -- so this only has to cover other activity on an
#: otherwise-idle box, not measurement noise.
_TOLERANCE = int(float(os.environ.get("MORI_VRAM_LEAK_TOLERANCE_MIB", "32")) * _MIB)

#: A case that registers less than this cannot distinguish "released" from
#: "leaked" at the tolerance above, so it is an error rather than a pass.
_MIN_DETECTABLE = 4 * _TOLERANCE

#: How long to let the driver settle before calling it a leak. A healthy
#: teardown returns the memory within a second or two; the field reports show a
#: leak still present after 120 s, so waiting longer does not change the verdict.
_SETTLE_TIMEOUT_S = float(os.environ.get("MORI_VRAM_LEAK_SETTLE_S", "60"))
_SETTLE_POLL_S = 2.0

#: A loaded node cannot give a trustworthy delta.
_BASELINE_MAX = float(os.environ.get("MORI_VRAM_LEAK_BASELINE_MAX_GIB", "8")) * _GIB

_KFD_PROC_DIR = "/sys/class/kfd/kfd/proc"
_DMA_BUF_INFO = "/sys/kernel/debug/dma_buf/bufinfo"

_ENABLED = os.environ.get("MORI_RUN_VRAM_LEAK_TEST") == "1"

pytestmark = [
    pytest.mark.skipif(
        not _ENABLED,
        reason="strands VRAM on this node until it reboots; "
        "set MORI_RUN_VRAM_LEAK_TEST=1 to run",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _NPROC,
        reason=f"EPv2 VRAM-release test needs {_NPROC} GPUs",
    ),
    pytest.mark.skipif(not _smi.available(), reason="amdsmi is not importable"),
]


@dataclass(frozen=True)
class Case:
    name: str
    stage: str
    nproc: int
    #: Pin every rank to this GPU (same-device import control); None = one each.
    cco_gpu: int | None = None
    #: Controls are expected clean everywhere; a failure means a different bug.
    control: bool = False


_CASES = (
    Case("comm", stage="comm", nproc=_NPROC, control=True),
    Case("window_solo", stage="window", nproc=1, control=True),
    Case("window_same_gpu", stage="window", nproc=2, cco_gpu=0, control=True),
    Case("window", stage="window", nproc=_NPROC),
    Case("op", stage="op", nproc=_NPROC),
)


# ---- instruments -----------------------------------------------------------


def _vram_used() -> dict[int, int]:
    """Per-GPU VRAM in use, keyed by HIP ordinal.

    Via ``_smi``'s BDF match rather than indexing amdsmi by the HIP ordinal: the
    two enumerations diverge as soon as HIP_VISIBLE_DEVICES is set, and then
    every row in the report describes the wrong GPU.
    """
    out = {}
    for i in range(_NPROC):
        handle = _smi.hip_device_to_amdsmi_handle(i)
        out[i] = int(
            _smi.amdsmi.amdsmi_get_gpu_memory_usage(
                handle, _smi.amdsmi.AmdSmiMemoryType.VRAM
            )
        )
    return out


def _kfd_pids() -> set[str]:
    """PIDs the KFD driver still holds GPU state for. World-readable."""
    try:
        return set(os.listdir(_KFD_PROC_DIR))
    except OSError:
        return set()


def _dma_buf_stats():
    """(count, total_bytes) of exported dma-bufs, or None without root.

    The one instrument that sees the mechanism rather than its effect, so it is
    worth attempting even though it usually comes back None.
    """
    try:
        with open(_DMA_BUF_INFO, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    count = total = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit():
            count += 1
            total += int(fields[0])
    return count, total


# ---- machine fingerprint ---------------------------------------------------


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _fingerprint():
    """Enough to tell two CI runs apart, and no more.

    Deliberately limited to what is readable from inside an unprivileged
    container, since that is where CI runs: the kernel is the host's (containers
    share it) and ``/sys/module/amdgpu/version`` is the loaded driver, but
    ``/boot/config-*`` and ``/var/lib/dkms`` belong to the host and are not
    mounted, so anything derived from them would only ever print "unknown".
    """
    return {
        "host": platform.node(),
        "kernel": platform.release(),
        "amdgpu_module": _read("/sys/module/amdgpu/version") or "unknown",
        "rocm": _read("/opt/rocm/.info/version") or "unknown",
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_properties(0).gcnArchName,
    }


# ---- run + report ----------------------------------------------------------


def _gib(n):
    return n / _GIB


def _run_worker(case, timeout=900):
    env = os.environ.copy()
    env["STAGE"] = case.stage
    if case.cco_gpu is not None:
        env["CCO_GPU"] = str(case.cco_gpu)
    env.setdefault("MORI_ENABLE_SDMA", "1")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={case.nproc}",
        _WORKER,
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=_HERE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def _parse_probe(out):
    """Rank records and the clean-exit count.

    Scanned by marker, not by line: torchrun funnels every rank into one pipe,
    so two records regularly share a line and anything assuming one-per-line
    loses half of them. ``raw_decode`` stops at the end of the first object.
    """
    decoder = json.JSONDecoder()
    records = [decoder.raw_decode(c)[0] for c in out.split("# PROBE ")[1:]]
    return records, out.count("clean exit")


def _settle(baseline):
    """Poll until every GPU is back within tolerance, or the timeout expires.

    Returns (trace, final). A healthy teardown leaves after a poll or two; a
    leaking one spends the whole timeout, which is the price of being sure.
    """
    trace = []
    deadline = time.monotonic() + _SETTLE_TIMEOUT_S
    while True:
        elapsed = round(time.monotonic() - (deadline - _SETTLE_TIMEOUT_S), 1)
        sample = _vram_used()
        trace.append((elapsed, sample))
        if all(sample[i] - baseline[i] <= _TOLERANCE for i in baseline):
            return trace, sample
        if time.monotonic() >= deadline:
            return trace, sample
        time.sleep(_SETTLE_POLL_S)


def _report(
    case,
    fp,
    baseline,
    trace,
    final,
    probe,
    kfd_before,
    kfd_after,
    dma_before,
    dma_after,
    rc,
):
    deltas = {i: final[i] - baseline[i] for i in baseline}
    held = sorted(i for i, d in deltas.items() if d > _TOLERANCE)
    symm = probe[0]["symm_bytes"] if probe else 0

    print(
        f"\n=== EPv2 VRAM release :: {case.name} "
        f"(stage={case.stage} nproc={case.nproc}"
        f"{f' CCO_GPU={case.cco_gpu}' if case.cco_gpu is not None else ''}"
        f" rc={rc}) ==="
    )
    print("  machine: " + "  ".join(f"{k}={v}" for k, v in fp.items()))
    if symm:
        print(f"  registered per rank: {symm / _MIB:.0f} MiB")
    print(
        f"  settled after {trace[-1][0]:.0f}s "
        f"({len(trace)} samples, timeout {_SETTLE_TIMEOUT_S:.0f}s)"
    )
    for i in sorted(baseline):
        flag = "  <- HELD" if deltas[i] > _TOLERANCE else ""
        print(
            f"  GPU{i}  {_gib(baseline[i]):7.2f} -> {_gib(final[i]):7.2f} GiB"
            f"   {deltas[i] / _MIB:+9.1f} MiB{flag}"
        )
    print(
        f"  KFD procs: {len(kfd_before)} -> {len(kfd_after)}"
        + ("   (residue has no owning process)" if held and not kfd_after else "")
    )
    if dma_before is None or dma_after is None:
        print(f"  dma-buf: needs root to read {_DMA_BUF_INFO}")
    else:
        print(
            f"  dma-buf: {dma_before[0]} -> {dma_after[0]} bufs, "
            f"{_gib(dma_before[1]):.2f} -> {_gib(dma_after[1]):.2f} GiB"
        )

    print(
        "VRAM_LEAK_RESULT "
        + json.dumps(
            {
                "case": case.name,
                "stage": case.stage,
                "nproc": case.nproc,
                "cco_gpu": case.cco_gpu,
                "control": case.control,
                "machine": fp,
                "symm_bytes_per_rank": symm,
                "tolerance_bytes": _TOLERANCE,
                "baseline_bytes": baseline,
                "final_bytes": final,
                "delta_bytes": deltas,
                "held_gpus": held,
                "settle_s": trace[-1][0],
                "kfd_procs_before": len(kfd_before),
                "kfd_procs_after": len(kfd_after),
                "dma_buf_before": dma_before,
                "dma_buf_after": dma_after,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return deltas, held, symm


@pytest.fixture(scope="module")
def fingerprint():
    fp = _fingerprint()
    print("\n=== machine ===")
    for k, v in fp.items():
        print(f"  {k}: {v}")
    return fp


@pytest.fixture(scope="module")
def idle_node():
    """Refuse to measure on a node that is already in use."""
    busiest = max(_vram_used().values())
    if busiest > _BASELINE_MAX:
        pytest.skip(
            f"node is not idle: busiest GPU holds {_gib(busiest):.1f} GiB "
            f"(limit {_gib(_BASELINE_MAX):.1f} GiB, raise with "
            f"MORI_VRAM_LEAK_BASELINE_MAX_GIB). A delta measured against "
            f"another tenant's allocations is noise."
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_vram_released_after_process_group_exit(case, fingerprint, idle_node):
    """Build one cco stack in a throwaway process group; every GPU must be back
    to its baseline once that group is gone."""
    baseline = _vram_used()
    kfd_before = _kfd_pids()
    dma_before = _dma_buf_stats()

    rc, out = _run_worker(case)
    probe, clean = _parse_probe(out)
    trace, final = _settle(baseline)

    kfd_after = _kfd_pids()
    dma_after = _dma_buf_stats()
    deltas, held, symm = _report(
        case,
        fingerprint,
        baseline,
        trace,
        final,
        probe,
        kfd_before,
        kfd_after,
        dma_before,
        dma_after,
        rc,
    )

    # First: did we measure anything? A crashed worker leaves a delta too, and
    # it would mean nothing.
    assert rc == 0, f"torchrun exited {rc}; output:\n{out}"
    assert (
        len(probe) == case.nproc
    ), f"expected {case.nproc} ranks to report, got {len(probe)}; output:\n{out}"
    assert (
        clean == case.nproc
    ), f"expected {case.nproc} clean exits, got {clean}; output:\n{out}"
    # Second: could this case have seen a leak if there was one? Guards against
    # a shrunken config turning into a silent pass.
    if case.stage != "comm":
        assert symm >= _MIN_DETECTABLE, (
            f"{case.name} registered only {symm / _MIB:.0f} MiB per rank, which "
            f"cannot be told from the {_TOLERANCE / _MIB:.0f} MiB tolerance; "
            f"raise WIN_MB / the op shape in vram_leak_worker.py"
        )

    # Third: the actual claim.
    detail = "\n".join(
        f"    GPU{i}: {deltas[i] / _MIB:+.1f} MiB" for i in sorted(baseline)
    )
    hint = (
        "a control case leaked -- this is not the known cross-GPU window bug, "
        "something more basic is stranding VRAM"
        if case.control
        else f"{len(held)} GPU(s) kept memory after all {case.nproc} ranks exited "
        f"cleanly; each rank registered {symm / _MIB:.0f} MiB"
    )
    assert not held, (
        f"{case.name}: VRAM not released after the process group exited "
        f"({hint}).\n"
        f"  machine: "
        + "  ".join(f"{k}={v}" for k, v in fingerprint.items())
        + f"\n  per-GPU delta (tolerance {_TOLERANCE / _MIB:.0f} MiB):\n{detail}"
    )
