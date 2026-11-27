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
"""MORI's traffic metrics, read from Python.

Off unless ``MORI_METRICS_ENABLE=1`` or ``MORI_METRICS_PORT`` is set before the
first IO engine / EP op is created; with them off ``snapshot()`` is empty.
"""
from mori.cpp import (
    metrics_enabled,
    metrics_port,
    metrics_sampling,
    metrics_snapshot,
)

__all__ = ["snapshot", "series", "port", "enabled", "sampling", "set_sampling"]


def series():
    """Sample now and return every series as a list of dicts.

    Keys: ``name``, ``help``, ``kind`` (``counter`` / ``gauge`` / ``histogram``),
    ``labels`` (a dict) and ``value``; a histogram also has ``bounds``,
    ``buckets`` (cumulative over the bounds) and ``count``.
    """
    return metrics_snapshot()


def snapshot():
    """Sample now and return ``{metric name: [series, ...]}``; see ``series()``."""
    out = {}
    for s in metrics_snapshot():
        out.setdefault(s["name"], []).append(s)
    return out


def port():
    """The port ``/metrics`` is served on, or -1 when nothing is served."""
    return metrics_port()


def enabled():
    """Whether the environment turned metrics on for this process."""
    return metrics_enabled()


def sampling():
    """The sampling in effect: ``{"ep": n, "io": n, "interval_ms": ms}``.

    One EP dispatch-and-combine pair / one IO transfer in ``n`` is counted, and
    counts ``n`` times (1: everything, the default; 0: nothing).
    """
    return metrics_sampling()


def set_sampling(ep=None, io=None, interval_ms=None):
    """Change the sampling while the process runs; ``None`` leaves a setting.

    Returns the settings in effect, as ``sampling()``; raises ``ValueError`` for
    a value out of range.
    """

    def arg(value, name):
        if value is None:
            return -1
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must not be negative")
        return value

    return metrics_sampling(
        arg(ep, "ep"), arg(io, "io"), arg(interval_ms, "interval_ms")
    )
