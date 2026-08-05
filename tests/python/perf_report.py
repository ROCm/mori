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
"""Lightweight, opt-in performance record emitter for the nightly perf dashboard.

Benchmarks call :func:`record_perf` to append one structured JSON record per
measured configuration. It is a no-op unless the ``MORI_PERF_OUT`` environment
variable points at an output file, so normal local/CI runs are unaffected.

Environment variables (all optional except ``MORI_PERF_OUT``):

* ``MORI_PERF_OUT``      - path to a JSONL file to append records to. When unset,
  :func:`record_perf` does nothing.
* ``MORI_PERF_RUN_ID``   - CI run id (e.g. GitHub Actions ``run_id``).
* ``MORI_PERF_COMMIT``   - git commit sha the wheel was built from.
* ``MORI_PERF_PLATFORM`` - hardware/platform label (e.g. ``MI355X_AINIC``).
* ``MORI_PERF_PYTHON``   - python version label (e.g. ``3.12``).
* ``MORI_PERF_DATE``     - ISO date/time string for the run.

Each emitted record is one JSON object per line with the shape::

    {
      "category": "intra_ep" | "internode_ep" | "io",
      "params":  { ... run configuration ... },
      "metrics": { ... measured numbers ... },
      "run_id": str, "commit": str, "platform": str,
      "python": str, "date": str, "ts": float
    }
"""

from __future__ import annotations

import json
import os
import time

__all__ = ["dtype_label", "record_perf"]


def dtype_label(dtype) -> str:
    """Short label for a torch dtype, e.g. ``torch.bfloat16`` -> ``bfloat16``."""
    return str(dtype).split(".")[-1]


def record_perf(category: str, params: dict, metrics: dict) -> None:
    """Append one perf record to ``$MORI_PERF_OUT`` as a JSON line.

    No-op unless ``MORI_PERF_OUT`` is set. Callers should typically guard this
    to rank 0 / the initiator so a record is not written once per rank. Any
    error is swallowed (with a diagnostic print) so perf reporting can never
    break a benchmark run.
    """
    out = os.environ.get("MORI_PERF_OUT")
    if not out:
        return

    record = {
        "category": category,
        "params": params,
        "metrics": metrics,
        "run_id": os.environ.get("MORI_PERF_RUN_ID", ""),
        "commit": os.environ.get("MORI_PERF_COMMIT", ""),
        "platform": os.environ.get("MORI_PERF_PLATFORM", ""),
        "python": os.environ.get("MORI_PERF_PYTHON", ""),
        "date": os.environ.get("MORI_PERF_DATE", ""),
        "ts": time.time(),
    }

    try:
        parent = os.path.dirname(os.path.abspath(out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, sort_keys=True)
        # Append with an advisory lock so concurrent ranks/processes writing to
        # the same file do not interleave partial lines.
        with open(out, "a", encoding="utf-8") as fh:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            fh.write(line + "\n")
    except Exception as exc:  # pragma: no cover - reporting must never fail a run
        print(f"[perf_report] failed to record {category} perf: {exc}")
