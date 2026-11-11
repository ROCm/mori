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
"""Regression tests for report/ainic_netmon_report.py on synthetic recordings (numpy only)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
)
import ainic_netmon_report as R  # noqa: E402

MS = 1_000_000


def write_run(d, streams, host="h"):
    """streams: list of dicts(netdev, source, names, t0, rtt, real, err, values[n, k])."""
    os.makedirs(d, exist_ok=True)
    meta = {
        "format_version": 2,
        "status": "complete",
        "host_id": host,
        "tags": {},
        "streams": [],
        "nics": [{"netdev": s["netdev"], "speed_mbps": 400000} for s in streams],
        "run_stats": {},
    }
    for s in streams:
        f = f"{s['netdev']}.{s['source']}.bin"
        n = len(s["t0"])
        rec = np.zeros((n, 4 + len(s["names"])), dtype="<u8")
        rec[:, 0] = s["t0"]
        rec[:, 1] = s["t0"] + s["rtt"]
        rec[:, 2] = s["real"]
        rec[:, 3] = s["err"]
        rec[:, 4:] = s["values"]
        rec.tofile(os.path.join(d, f))
        meta["streams"].append(
            {
                "file": f,
                "netdev": s["netdev"],
                "source": s["source"],
                "names": s["names"],
                "record_words": rec.shape[1],
                "interval_ns": s.get("interval", 2 * MS),
                "snapshot_fraction": 0.5,
            }
        )
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return R.Run(d)


def steady(
    n, interval=2 * MS, rate_bytes_per_ns=50.0, rtt=1 * MS, step_at=None, step_ns=0
):
    t0 = (np.arange(n) * interval + 10**12).astype(np.uint64)
    rtts = np.full(n, rtt, dtype=np.uint64)
    real = t0.astype(np.int64) + 1_700_000_000 * 10**9
    if step_at is not None:
        real[step_at:] += step_ns
    v = (np.arange(n) * interval * rate_bytes_per_ns).astype(np.uint64)
    return t0, rtts, real.astype(np.uint64), np.zeros(n, dtype=np.uint64), v


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="netmon-report-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def one(self, t0, rtt, real, err, v, interval=2 * MS, name="x"):
        run = write_run(
            self.tmp + "/" + name,
            [
                {
                    "netdev": "n0",
                    "source": "rdma",
                    "names": ["tx_rdma_ucast_bytes"],
                    "t0": t0,
                    "rtt": rtt,
                    "real": real,
                    "err": err,
                    "values": v.reshape(-1, 1),
                    "interval": interval,
                }
            ],
        )
        return run, run.stream("n0", "rdma")

    def test_slow_sample_does_not_bridge_a_reset(self):
        t0, rtt, real, err, v = steady(40)
        v = v.copy()
        rtt = rtt.copy()
        rtt[20] = 4 * MS  # imprecise sample (> 1.5x median) ...
        v[20:] = v[20:] - v[20]  # ... that also carries a counter reset to 0,
        v[21:] += v[19] + 1000  # after which the counter passes its pre-reset value
        # (1000 -> 0 -> 1200 in miniature: without reset tracking, 19 -> 21 looks like +1000)
        _, s = self.one(t0, rtt, real, err, v)
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        # Samples 19 and 21 are adjacent after dropping 20, but a reset lies between them.
        bridged = (r.t_start == s.times()[0][19]) & (r.t_end == s.times()[0][21])
        self.assertTrue(bridged.any())
        self.assertFalse(r.valid[bridged].any())
        self.assertTrue(np.all(r.rate[r.valid] >= 0))

    def test_clock_step_segments(self):
        t0, rtt, real, err, v = steady(100, step_at=50, step_ns=10 * MS)
        _, s = self.one(t0, rtt, real, err, v)
        wall, seg = s.times()[1], s.segments()
        # Sample 49's anchor is before the step and the next anchor after it: uncertain.
        self.assertTrue(np.isnan(wall[49]))
        self.assertEqual(seg[49], -1)
        self.assertEqual(seg[48], 0)
        self.assertEqual(seg[50], 1)
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        # No displayed interval spans the step, but local rates on both sides are fine.
        self.assertFalse(r.valid[48:50].any())
        self.assertTrue(r.valid[:48].all() and r.valid[50:].all())
        self.assertTrue(np.allclose(r.rate[r.valid] * 8 / 1e9, 400.0))

    def test_peak_includes_last_window_and_needs_resolution(self):
        t0, rtt, real, err, v = steady(11)  # exactly 20 ms of 2 ms samples
        v = v.copy()
        v[6:] += 2_000_000  # extra bytes in the last 10 ms only
        _, s = self.one(t0, rtt, real, err, v)
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        m = r.valid & ~r.lowres
        peak = R.peak_over_window(r, m, 8 / 1e9)
        self.assertTrue(np.isfinite(peak))
        self.assertGreater(peak, 400.0 + 1.0)  # the last window carries the extra bytes
        # 1 s sampling cannot give a 10 ms peak.
        t0, rtt, real, err, v = steady(5, interval=1000 * MS)
        _, s = self.one(t0, rtt, real, err, v, interval=1000 * MS, name="coarse")
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        self.assertTrue(np.isnan(R.peak_over_window(r, r.valid & ~r.lowres, 8 / 1e9)))

    def test_one_coarse_interval_only_cuts_the_peak_windows(self):
        # 5 ms sampling with one 12 ms gap (not low-res: 3x baseline is 15 ms): both sides
        # still have enough data for 10 ms windows.
        steps = [5] * 6 + [12] + [5] * 6
        t0 = (np.concatenate([[0], np.cumsum(steps)]) * MS + 10**12).astype(np.uint64)
        n = len(t0)
        rtt = np.full(n, 1 * MS, dtype=np.uint64)
        real = (t0.astype(np.int64) + 1_700_000_000 * 10**9).astype(np.uint64)
        v = ((t0 - t0[0]).astype(np.float64) * 50.0).astype(np.uint64)  # 400 Gb/s
        _, s = self.one(t0, rtt, real, np.zeros(n, dtype=np.uint64), v, interval=5 * MS)
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        mask = r.valid & ~r.lowres
        self.assertTrue(mask.all())  # the 12 ms interval is not low resolution
        self.assertAlmostEqual(R.peak_over_window(r, mask, 8 / 1e9), 400.0, places=3)

    def test_budget_respects_nic_filter(self):
        t0, rtt, real, err, v = steady(10)
        streams = [
            {
                "netdev": f"n{i}",
                "source": "rdma",
                "names": ["tx_rdma_ucast_bytes"],
                "t0": t0,
                "rtt": rtt,
                "real": real,
                "err": err,
                "values": v.reshape(-1, 1),
            }
            for i in range(8)
        ]
        run = write_run(self.tmp + "/multi", streams)
        self.assertEqual(len(R.plotted_streams([run], ["n3"])), 1)
        self.assertEqual(len(R.plotted_streams([run], None)), 8)

    def test_average_skips_reset_interval(self):
        t0, rtt, real, err, v = steady(30)
        v = v.copy()
        v[15:] = v[15:] - v[15]  # reset at sample 15
        _, s = self.one(t0, rtt, real, err, v)
        r = R.Rates(s, "tx_rdma_ucast_bytes")
        w = s.times()[1]
        st = R.series_stats(r, w[0], w[-1], 8 / 1e9)
        self.assertAlmostEqual(st["avg"], 400.0, places=3)
        self.assertLess(st["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
