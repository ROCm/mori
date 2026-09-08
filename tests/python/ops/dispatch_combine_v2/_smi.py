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
"""Clock/power telemetry around a measured window, via amdsmi.

A latency number means little without the clock it was measured at: a
power-limited part throttles, and the same kernel then reports a different time.
This samples gfx/soc clock, socket power, hotspot temperature, activity and VRAM
on a background thread, and summarises only the samples that fall inside a
caller-supplied window -- so warmup, compilation and input generation stay out of
the reported clocks.

The emitted record is aiter's (``op_tests/smi_monitor.py``), so
``bench_gfx1250_combo.py`` can consume mori's rows with the collector it already
has. It is a copy, not an import, for the same reason as ``_data.py``. Settings
answer to either spelling, ``MORI_SMI_*`` or ``AITER_SMI_*``.

ROCm ships the amdsmi binding without a setup.py, so the import falls back to
``/opt/rocm/share/amd_smi`` without leaving that path on ``sys.path``. When the
binding is absent ``available()`` is False and the caller skips telemetry rather
than failing the benchmark.
"""

from __future__ import annotations

import atexit
import ctypes
import glob
import importlib
import json
import os
import sys
import threading
import time
from functools import cache

SMI_RESULT_PREFIX = "AITER_SMI_RESULT "
_ROCM_AMDSMI_PATH = "/opt/rocm/share/amd_smi"


def _import_amdsmi():
    try:
        return importlib.import_module("amdsmi")
    except ImportError:
        if not os.path.isdir(_ROCM_AMDSMI_PATH):
            raise
    sys.path.insert(0, _ROCM_AMDSMI_PATH)
    try:
        return importlib.import_module("amdsmi")
    finally:
        sys.path.remove(_ROCM_AMDSMI_PATH)


try:
    amdsmi = _import_amdsmi()
    _AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure just disables telemetry
    amdsmi = None
    _AVAILABLE = False

_LOCK = threading.Lock()
_INITIALIZED = False
_BY_BDF: dict = {}
_BY_HIP_DEVICE: dict = {}


@cache
def _hip_runtime():
    names = ["libamdhip64.so"] + sorted(
        glob.glob("/opt/rocm/lib/libamdhip64.so.*"), reverse=True
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libamdhip64.so not found; is ROCm installed?")


@cache
def _hip_device_bdf(hip_device: int) -> str:
    """PCIe BDF of a HIP device, the stable key between HIP and amdsmi ordinals.

    amdsmi enumerates every GPU in the box; HIP enumerates what
    HIP_VISIBLE_DEVICES left it. Indexing amdsmi by the HIP ordinal therefore
    reads a different GPU as soon as the two lists disagree.
    """
    buf = ctypes.create_string_buffer(64)
    ret = _hip_runtime().hipDeviceGetPCIBusId(
        buf, ctypes.c_int(64), ctypes.c_int(hip_device)
    )
    if ret != 0:
        raise RuntimeError(f"hipDeviceGetPCIBusId failed: {ret}")
    return buf.value.decode().lower().strip()


def _bdf_str(handle) -> str:
    raw = amdsmi.amdsmi_get_gpu_device_bdf(handle)
    if isinstance(raw, str):
        return raw.lower().strip()
    return (
        f"{raw['domain']:04x}:{raw['bus']:02x}:"
        f"{raw['device']:02x}.{raw['function']:x}"
    )


def _ensure_initialized() -> None:
    global _INITIALIZED
    if not _AVAILABLE:
        raise ImportError("amdsmi is not importable")
    with _LOCK:
        if _INITIALIZED:
            return
        amdsmi.amdsmi_init()
        try:
            _BY_BDF.update(
                {_bdf_str(h): h for h in amdsmi.amdsmi_get_processor_handles()}
            )
        except BaseException:
            amdsmi.amdsmi_shut_down()
            raise
        _INITIALIZED = True


def _shutdown() -> None:
    global _INITIALIZED
    with _LOCK:
        if not _INITIALIZED:
            return
        try:
            amdsmi.amdsmi_shut_down()
        finally:
            _INITIALIZED = False
            _BY_BDF.clear()
            _BY_HIP_DEVICE.clear()


atexit.register(_shutdown)


def hip_device_to_amdsmi_handle(hip_device: int):
    """The amdsmi handle for a HIP device ordinal, matched by PCIe BDF."""
    _ensure_initialized()
    with _LOCK:
        cached = _BY_HIP_DEVICE.get(hip_device)
        if cached is not None:
            return cached
    bdf = _hip_device_bdf(hip_device)
    with _LOCK:
        handle = _BY_BDF.get(bdf)
        if handle is None:
            raise RuntimeError(f"no amdsmi handle with BDF {bdf!r}")
        _BY_HIP_DEVICE[hip_device] = handle
        return handle


def _collect_sample(handle) -> dict:
    sample: dict = {"timestamp_s": time.perf_counter()}
    try:
        m = amdsmi.amdsmi_get_gpu_metrics_info(handle)
        sample["gfx_clk_mhz"] = m.get("current_gfxclk")
        sample["soc_clk_mhz"] = m.get("current_socclk")
        sample["power_w"] = m.get("current_socket_power")
        sample["temp_hotspot_c"] = m.get("temperature_hotspot")
    except Exception:  # noqa: BLE001 - a missing metric must not stop sampling
        pass
    try:
        a = amdsmi.amdsmi_get_gpu_activity(handle)
        sample["gfx_activity_pct"] = a.get("gfx_activity")
        sample["umc_activity_pct"] = a.get("umc_activity")
    except Exception:  # noqa: BLE001
        pass
    try:
        used = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
        sample["vram_used_mb"] = used / 1024 / 1024
    except Exception:  # noqa: BLE001
        pass
    return sample


class GpuMonitor:
    """Poll one GPU's metrics on a background thread until stopped."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.05) -> None:
        if not _AVAILABLE:
            raise ImportError("amdsmi is not importable")
        self._device_index = device_index
        self._interval_s = interval_s
        self._samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._handle = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GpuMonitor is already running")
        self._samples = []
        self._error = None
        self._stop.clear()
        self._ready.clear()
        self._handle = hip_device_to_amdsmi_handle(self._device_index)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._stop.set()
            raise RuntimeError("timed out initializing the amdsmi monitor")
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError(f"amdsmi monitor failed to start: {error}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def samples(self) -> list[dict]:
        return list(self._samples)

    def summary(self, *, start_s=None, end_s=None) -> dict:
        """min/mean/median/max per metric over the samples inside the window.

        Windowing is the whole point: the monitor runs across the replay loop,
        but only the samples between the two timestamps the caller recorded
        describe the work that was measured.
        """
        rows = [
            s
            for s in self._samples
            if (start_s is None or s["timestamp_s"] >= start_s)
            and (end_s is None or s["timestamp_s"] <= end_s)
        ]
        if not rows:
            return {}
        out: dict = {}
        for key in sorted({k for s in rows for k in s if k != "timestamp_s"}):
            vals = sorted(
                s[key] for s in rows if s.get(key) is not None and s[key] != "N/A"
            )
            if not vals:
                continue
            n, mid = len(vals), len(vals) // 2
            out[key] = {
                "min": vals[0],
                "mean": sum(vals) / n,
                "median": vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2,
                "max": vals[-1],
                "n": n,
            }
        return out

    def __enter__(self) -> GpuMonitor:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def _poll_loop(self) -> None:
        try:
            self._ready.set()
            while not self._stop.is_set():
                t0 = time.perf_counter()
                self._samples.append(_collect_sample(self._handle))
                left = self._interval_s - (time.perf_counter() - t0)
                if left > 0:
                    self._stop.wait(timeout=left)
        except BaseException as error:  # noqa: BLE001 - reported via start()
            self._error = error
        finally:
            self._ready.set()


def available() -> bool:
    """Whether telemetry can be collected at all in this process."""
    return _AVAILABLE


def _env(name, default=None):
    """MORI_SMI_<name> if set, else AITER_SMI_<name>.

    Two spellings for one setting: MORI_ is what a mori run should have to know
    about, AITER_ is the interop contract -- a driver that already exports it for
    aiter turns mori's telemetry on with no extra plumbing.
    """
    value = os.environ.get(f"MORI_SMI_{name}")
    if value is None:
        value = os.environ.get(f"AITER_SMI_{name}")
    return default if value is None else value


def env_config():
    """(enabled, interval_s, duration_s) from {MORI,AITER}_SMI_MONITOR/INTERVAL/DURATION."""
    on = _env("MONITOR", "0") == "1"
    interval = float(_env("INTERVAL", "0.05"))
    duration = float(_env("DURATION", "1.0"))
    if on and (interval <= 0 or duration <= 0):
        raise ValueError("SMI interval and duration must be positive")
    return on, interval, duration


def emit(record: dict) -> None:
    """Write one telemetry record to the shared JSONL sink, or to stdout."""
    line = SMI_RESULT_PREFIX + json.dumps(record, sort_keys=True)
    path = _env("OUTPUT_PATH")
    if path:
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(line + "\n")
    else:
        print(line, flush=True)
