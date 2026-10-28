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
"""Fit the simulated backend's link profile (MORI_IO_SIM_<BACKEND>_*) to real hardware.

Times GPU-to-GPU batch writes between two engines in two processes, the way a
prefill and a decode engine use MORI-IO, and fits

    occupancy = segments * (segment_bytes / BW + per_segment) + per_op
    latency   = single-transfer time - occupancy

Run it on hardware to get the profile, then with MORI_IO_SIM_ENABLE=1 and the
printed variables plus --compare <hardware json> to check the simulation:

    python calibrate_sim.py --backend rdma --json rdma.json
    env MORI_IO_SIM_ENABLE=1 <printed vars> python calibrate_sim.py \
        --backend rdma --compare rdma.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

from mori.io import (
    BackendType,
    EngineDesc,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
    StatusCode,
)
from mori.cpp import io_sim_enabled

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
FAKE_PTR = 0x7F0000000000
SINGLE_SEGMENT_SIZES = [4 * KiB << (2 * i) for i in range(9)]  # 4 KiB .. 256 MiB
SEGMENT_COUNTS = [1, 4, 16, 64, 256, 1024]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=("rdma", "xgmi"), required=True)
    p.add_argument("--src-gpu", type=int, default=0)
    p.add_argument("--dst-gpu", type=int, default=1)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--buffer-mb", type=int, default=1024)
    p.add_argument(
        "--segment-kb", type=int, default=16, help="segment size of the sweep"
    )
    # sglang's mori connector defaults (SGLANG_MORI_QP_PER_TRANSFER / _NUM_WORKERS).
    p.add_argument("--qp", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", help="write the measurements and the fit here")
    p.add_argument(
        "--compare", help="a --json file to compare these measurements against"
    )
    p.add_argument("--role", choices=("initiator", "target"), default="initiator")
    return p.parse_args()


def make_engine(args, key, gpu):
    eng = IOEngine(key, IOEngineConfig(host=args.host, port=0))
    if args.backend == "rdma":
        cfg = RdmaBackendConfig(args.qp, -1, args.workers, PollCqMode.POLLING, False)
        eng.create_backend(BackendType.RDMA, cfg)
    else:
        eng.create_backend(BackendType.XGMI)
    size = args.buffer_mb * MiB
    if io_sim_enabled():
        return (
            eng,
            None,
            eng.register_memory(FAKE_PTR, size, gpu, MemoryLocationType.GPU),
        )
    import torch

    buf = torch.empty(size, dtype=torch.uint8, device=torch.device("cuda", gpu))
    return eng, buf, eng.register_torch_tensor(buf)


def run_target(args):
    eng, _buf, mem = make_engine(args, f"calib-target-{os.getpid()}", args.dst_gpu)
    # Tagged, since backend setup (e.g. the XGMI kernel load) may print to stdout too.
    print("DESC", eng.get_engine_desc().pack().hex(), mem.pack().hex(), flush=True)
    sys.stdin.read()  # until the initiator exits


class Initiator:
    def __init__(self, args, remote_engine, remote_mem):
        self.eng, self.buf, self.mem = make_engine(
            args, f"calib-initiator-{os.getpid()}", args.src_gpu
        )
        self.eng.register_remote_engine(remote_engine)
        self.remote = remote_mem
        self.size = args.buffer_mb * MiB
        self.cursor = 0

    def _segments(self, n, s):
        # Gaps between segments, so no backend can merge them into one.
        stride = s if n == 1 else 2 * s
        span = n * stride
        assert span <= self.size, (n, s)
        if self.cursor + span > self.size:
            self.cursor = 0
        offs = [self.cursor + i * stride for i in range(n)]
        self.cursor += span
        return offs

    def submit(self, n, s):
        offs = self._segments(n, s)
        uid = self.eng.allocate_transfer_uid()
        return self.eng.batch_write(
            [self.mem], [offs], [self.remote], [offs], [[s] * n], [uid]
        )[0]

    def wait(self, statuses):
        rc = self.eng.wait_all(statuses)
        if rc != StatusCode.SUCCESS:
            bad = next(st for st in statuses if not st.Succeeded())
            raise RuntimeError(f"transfer failed: {bad.Code()} {bad.Message()}")

    def single_us(self, n, s, reps):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            self.wait([self.submit(n, s)])
            times.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(times)

    def stream_us(self, n, s, calls):
        t0 = time.perf_counter()
        self.wait([self.submit(n, s) for _ in range(calls)])
        return (time.perf_counter() - t0) * 1e6

    def occupancy_us(self, n, s, repeats):
        # Slope of total time over the number of back-to-back calls: the time each call
        # holds the link, with the single-transfer latency cancelled out.
        k2 = int(min(256, max(16, 2 * GiB // (n * s))))
        k1 = k2 // 4
        slopes = []
        for _ in range(repeats):
            t1 = self.stream_us(n, s, k1)
            t2 = self.stream_us(n, s, k2)
            slopes.append((t2 - t1) / (k2 - k1))
        return statistics.median(slopes)


def measure(args, init):
    seg = args.segment_kb * KiB
    shapes = [(1, s) for s in SINGLE_SEGMENT_SIZES]
    shapes += [(n, seg) for n in SEGMENT_COUNTS if (n, seg) not in shapes]
    for n, s in shapes[:3]:  # connection setup and first-touch costs
        init.wait([init.submit(n, s) for _ in range(4)])
    points = []
    for n, s in shapes:
        reps = 5 if n * s >= 64 * MiB else 20
        single = init.single_us(n, s, reps)
        occ = init.occupancy_us(n, s, args.repeats)
        points.append(
            {
                "segments": n,
                "segment_bytes": s,
                "single_us": single,
                "occupancy_us": occ,
            }
        )
        print(
            f"  {n:5d} x {s // KiB:7d} KiB  single {single:10.1f} us  occupancy {occ:10.1f} us"
            f"  ({n * s / occ / 1e3:6.2f} GB/s)",
            flush=True,
        )
    return points


def fit(points, segment_bytes):
    """Each parameter from the points where it dominates; time in ns, so bytes/ns = GB/s.

    Small calls are bound by the host (the Python call and the submit), not the link; that
    floor overlaps the link time instead of adding to it, so per_op comes from link-bound
    points only.
    """
    single = {p["segment_bytes"]: p for p in points if p["segments"] == 1}
    occ = {(p["segments"], p["segment_bytes"]): p["occupancy_us"] * 1e3 for p in points}
    bw = statistics.median(s / occ[(1, s)] for s in sorted(single)[-2:])
    sweep = sorted(n for n, s in occ if s == segment_bytes and n >= 64)
    slope, intercept = np.polyfit(sweep, [occ[(n, segment_bytes)] for n in sweep], 1)
    per_seg = max(0.0, slope - segment_bytes / bw)
    per_op = max(0.0, intercept)

    def model_occ(n, s):
        return n * (s / bw + per_seg) + per_op

    small = [s for s in single if s <= 64 * KiB]
    latency = statistics.median(
        single[s]["single_us"] * 1e3 - model_occ(1, s) for s in small
    )
    return {
        "bw_gbps": bw,
        "latency_us": max(0.0, latency) / 1e3,
        "per_segment_ns": per_seg,
        "per_op_us": per_op / 1e3,
        "host_floor_us": statistics.median(occ[(1, s)] for s in small) / 1e3,
        "occupancy_rel_err": {
            f"{n}x{s}": model_occ(n, s) / v - 1 for (n, s), v in occ.items()
        },
    }


PARAMS = (  # fit key, env suffix, format
    ("bw_gbps", "BW_GBPS", ".2f"),
    ("latency_us", "LATENCY_US", ".2f"),
    ("per_segment_ns", "SEGMENT_NS", ".1f"),
    ("per_op_us", "OP_US", ".2f"),
)


def print_profile(title, backend, profile):
    print(f"\n{title}:")
    for key, suffix, fmt in PARAMS:
        print(f"  MORI_IO_SIM_{backend.upper()}_{suffix}={profile[key]:{fmt}}")


def compare(args, points, result, ref_path):
    ref = json.load(open(ref_path))
    ref_points = {(p["segments"], p["segment_bytes"]): p for p in ref["points"]}
    print(f"\nthis run / {ref_path}:")
    print(f"  {'shape':>18}  {'single':>7}  {'occupancy':>9}")
    for p in points:
        r = ref_points.get((p["segments"], p["segment_bytes"]))
        if r:
            shape = f"{p['segments']} x {p['segment_bytes'] // KiB} KiB"
            print(
                f"  {shape:>18}  {p['single_us'] / r['single_us']:7.2f}"
                f"  {p['occupancy_us'] / r['occupancy_us']:9.2f}"
            )
    if not io_sim_enabled():
        return
    # The simulator adds its own host costs (Python calls, thread wakeups) on top of the
    # profile, so move each parameter by the gap between the two fits.
    prefix = f"MORI_IO_SIM_{args.backend.upper()}_"
    env = {}
    for key, suffix, _ in PARAMS:
        if prefix + suffix not in os.environ:
            print(f"\nset all four {prefix}* variables to get a refined profile")
            return
        env[key] = float(os.environ[prefix + suffix])
    target, got = ref["fit"], result
    refined = {
        "bw_gbps": 1.0
        / max(
            1e-9, 1.0 / env["bw_gbps"] + 1.0 / target["bw_gbps"] - 1.0 / got["bw_gbps"]
        ),
    }
    for key in ("latency_us", "per_segment_ns", "per_op_us"):
        refined[key] = max(0.0, env[key] + target[key] - got[key])
    print_profile("refined profile (rerun with it to check)", args.backend, refined)


def main():
    args = parse_args()
    if args.role == "target":
        return run_target(args)

    cmd = [sys.executable, os.path.abspath(__file__), "--role", "target"]
    for flag in ("backend", "dst_gpu", "host", "buffer_mb", "qp", "workers"):
        cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    target = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        for line in target.stdout:
            if line.startswith("DESC "):
                break
            print(f"  target: {line.rstrip()}", flush=True)
        else:
            raise RuntimeError("target exited before printing its descriptors")
        _, ehex, mhex = line.split()
        edesc = EngineDesc.unpack(bytes.fromhex(ehex))
        mdesc = MemoryDesc.unpack(bytes.fromhex(mhex))
        mode = "sim" if io_sim_enabled() else "hardware"
        print(
            f"{args.backend} on {mode}: GPU {args.src_gpu} -> GPU {args.dst_gpu}",
            flush=True,
        )
        points = measure(args, Initiator(args, edesc, mdesc))
    finally:
        target.stdin.close()
        target.wait(timeout=30)

    result = fit(points, args.segment_kb * KiB)
    print_profile("fitted profile", args.backend, result)
    print(
        f"  (host floor: {result['host_floor_us']:.2f} us per call, not part of the profile)"
    )
    if args.compare:
        compare(args, points, result, args.compare)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "backend": args.backend,
                    "mode": mode,
                    "sim_env": {
                        k: v
                        for k, v in os.environ.items()
                        if k.startswith("MORI_IO_SIM")
                    },
                    "args": vars(args),
                    "points": points,
                    "fit": result,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
