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
"""The live push's online interval computation (C++ LiveRates, replayed through
netmon_live_replay) against the offline report's Rates (NETMON_LIVE_PUSH_DESIGN §3, §8).

With constant rtt and sampling interval, the online medians equal the offline ones, so
every valid interval and every reset gap must be identical (end ms, resolution, rate,
increase, duration). Clock gaps are merged online (documented difference): only their
presence is compared."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "report"))
sys.path.insert(0, HERE)
import ainic_netmon_report as R  # noqa: E402
from test_report import MS, steady, write_run  # noqa: E402

REPLAY = os.environ.get("NETMON_LIVE_REPLAY")
NAMES = [R.RDMA_TX, R.RDMA_RX, "rx_rdma_ecn_pkts"]


def replay(stream, names, optional="", cap=1024):
    cols = [4 + stream.names.index(n) for n in names]
    rec = np.ascontiguousarray(stream.rec[:, [0, 1, 2, 3] + cols]).astype("<u8")
    out = subprocess.run(
        [
            REPLAY,
            str(stream.interval_ns),
            repr(stream.f),
            str(len(names)),
            optional or "0" * len(names),
            str(cap),
        ],
        input=rec.tobytes(),
        capture_output=True,
        check=True,
    ).stdout.decode()
    iv, gaps, stats = {}, {}, None
    for line in out.splitlines():
        p = line.split(",")
        if p[0] == "I":
            c = names[int(p[1])]
            iv.setdefault(c, []).append(
                (int(p[3]), bool(int(p[2])), float(p[4]), float(p[5]), float(p[6]))
            )
        elif p[0] == "G":
            gaps.setdefault((names[int(p[1])], int(p[2])), []).append(
                (int(p[3]), float(p[4]))
            )
        elif p[0] == "S":
            stats = {
                "samples": int(p[1]),
                "unconfirmed_dropped": int(p[2]),
                "clock_back": int(p[3]),
                "ms_merged": int(p[4]),
                "gaps_dropped": int(p[5]),
                "baseline_ns": float(p[6]),
            }
    return iv, gaps, stats


def offline(stream, name):
    r = R.Rates(stream, name)
    end = r.end_ms()
    m = r.valid & (end >= 0)
    iv = list(
        zip(
            end[m].tolist(),
            r.lowres[m].tolist(),
            r.rate[m].tolist(),
            r.dv[m].tolist(),
            (r.dt[m] / 1e9).tolist(),
        )
    )
    resets = [
        (int(e), float(d) / 1e9)
        for e, d, ok in zip(end, r.dt, r.reset & (end >= 0))
        if ok
    ]
    clock = bool((r.clock & (end >= 0)).any())
    return iv, resets, clock


def run_of(tmp, n=300, interval=2 * MS, mutate=None, names=NAMES):
    t0, rtt, real, err, _ = steady(n, interval=interval)
    vals = np.stack(
        [
            (np.arange(n) * interval * (10.0 + 7 * k)).astype(np.uint64)
            for k in range(len(names))
        ],
        axis=1,
    )
    if mutate:
        t0, rtt, real, err, vals = mutate(
            t0.copy(), rtt.copy(), real.copy(), err.copy(), vals.copy()
        )
    return write_run(
        tmp,
        [
            {
                "netdev": "n0",
                "source": "rdma",
                "names": names,
                "t0": t0,
                "rtt": rtt,
                "real": real,
                "err": err,
                "values": vals,
                "interval": interval,
            }
        ],
    ).stream("n0", "rdma")


@unittest.skipUnless(REPLAY, "NETMON_LIVE_REPLAY not set (run through ctest)")
class LiveVsOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="netmon-live-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def assert_same(self, s, names=NAMES, clock_expected=False):
        live, gaps, stats = replay(s, names)
        for c in names:
            off_iv, off_resets, off_clock = offline(s, c)
            self.assertEqual(live.get(c, []), off_iv, f"{c}: intervals differ")
            self.assertEqual(
                gaps.get((c, 0), []), off_resets, f"{c}: reset gaps differ"
            )
            self.assertEqual(
                bool(gaps.get((c, 1))), off_clock, f"{c}: clock gaps differ"
            )
            self.assertEqual(off_clock, clock_expected)
        return live, gaps, stats

    def test_steady(self):
        _, _, st = self.assert_same(run_of(self.tmp))
        self.assertEqual(st["baseline_ns"], 2 * MS)

    def test_counter_reset(self):
        def m(t0, rtt, real, err, v):
            v[150:, 0] -= v[150, 0]
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m))

    def test_reset_hidden_in_a_slow_sample(self):
        # 1000 -> slow 0 -> 1200: the interval between the two kept samples is a reset
        def m(t0, rtt, real, err, v):
            rtt[100] = rtt[100] * 4
            v[100, 1] = 0
            return t0, rtt, real, err, v

        live, gaps, _ = self.assert_same(run_of(self.tmp, mutate=m))
        self.assertEqual(len(gaps[(NAMES[1], 0)]), 1)

    def test_slow_samples_are_skipped(self):
        def m(t0, rtt, real, err, v):
            for i in (50, 51, 120, 250):
                rtt[i] = rtt[i] * 3
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m))

    def test_low_resolution(self):
        def m(t0, rtt, real, err, v):
            keep = np.r_[0:80, 95:300]  # 15 samples lost: one 32 ms interval
            return t0[keep], rtt[keep], real[keep], err[keep], v[keep]

        live, _, _ = self.assert_same(run_of(self.tmp, mutate=m))
        self.assertEqual(sum(1 for x in live[NAMES[0]] if x[1]), 1)

    def test_clock_step(self):
        def m(t0, rtt, real, err, v):
            real[150:] += np.uint64(5 * MS)
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m), clock_expected=True)

    def test_clock_step_with_untrusted_anchors(self):
        def m(t0, rtt, real, err, v):
            err[140:160] = 200_000  # untrusted anchors around the step
            real[150:] += np.uint64(5 * MS)
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m), clock_expected=True)

    def test_reset_and_clock_together(self):
        def m(t0, rtt, real, err, v):
            real[150:] += np.uint64(5 * MS)
            v[150:, 0] -= v[150, 0]
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m)
        live, gaps, _ = replay(s, NAMES)
        off_iv, off_resets, _ = offline(s, NAMES[0])
        self.assertEqual(live[NAMES[0]], off_iv)
        # reset wins over clock; the gap spans the dropped uncertain sample online (merged
        # span, NETMON_LIVE_PUSH_DESIGN §3.1), so only the end times are compared
        self.assertEqual(
            [e for e, _ in gaps.get((NAMES[0], 0), [])], [e for e, _ in off_resets]
        )

    def test_wall_clock_going_back_pauses(self):
        def m(t0, rtt, real, err, v):
            real[150:] -= np.uint64(100 * MS)
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m)
        live, _, st = replay(s, NAMES)
        for c in NAMES:
            ends = [x[0] for x in live[c]]
            self.assertTrue(all(b > a for a, b in zip(ends, ends[1:])), c)
        self.assertGreater(st["clock_back"], 0)
        # resumes once past the last emitted ms: the tail is emitted again
        self.assertGreater(live[NAMES[0]][-1][0], offline(s, NAMES[0])[0][148][0])

    def test_pending_overflow_never_emits_unconfirmed(self):
        def m(t0, rtt, real, err, v):
            err[100:200] = 200_000  # 100 samples without a trusted anchor
            real[150:] += np.uint64(5 * MS)  # and a step hidden in them
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m)
        live, gaps, st = replay(s, NAMES, cap=20)
        self.assertGreater(st["unconfirmed_dropped"], 0)
        off_iv, _, _ = offline(s, NAMES[0])
        # every live interval is one the offline computation also has
        self.assertTrue(set(live[NAMES[0]]) <= set(off_iv))
        self.assertTrue(gaps.get((NAMES[0], 1)))

    def test_optional_counter_from_first_change(self):
        def m(t0, rtt, real, err, v):
            v[:, 2] = 0
            v[200:, 2] = np.arange(100) * 3
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m)
        live, _, _ = replay(s, NAMES, optional="001")
        off_iv, _, _ = offline(s, NAMES[2])
        self.assertEqual(live[NAMES[2]], off_iv[-len(live[NAMES[2]]) :])
        self.assertEqual(len(live[NAMES[2]]), 99)  # first change: sample 200 -> 201

    def test_back_to_back_millisecond_merge(self):
        # Reference (design §3.4): delete the sample that ends an interval in an already used
        # millisecond, recompute the offline Rates, repeat. Bursty traffic and a reset.
        rng = np.random.default_rng(7)
        n = 400

        def m(t0, rtt, real, err, v):
            inc = rng.integers(0, 5000, size=(n, v.shape[1])).astype(np.uint64)
            v = np.cumsum(inc, axis=0, dtype=np.uint64)
            v[250:, 0] -= v[250, 0]  # reset on the first counter
            return t0, rtt, real, err, v

        s = run_of(self.tmp + "/a", n=n, interval=MS // 2, mutate=m)
        live, gaps, st = replay(s, NAMES)
        self.assertGreater(st["ms_merged"], 0)
        # Millisecond order is per series: the reference runs per counter (a reset gap does
        # not occupy a millisecond of the counter's rate series).
        k = 0
        for c in NAMES:
            rec = np.array(s.rec)
            while True:
                k += 1
                rs = write_run(
                    f"{self.tmp}/ref{k}",
                    [
                        {
                            "netdev": "n0",
                            "source": "rdma",
                            "names": s.names,
                            "t0": rec[:, 0],
                            "rtt": rec[:, 1] - rec[:, 0],
                            "real": rec[:, 2],
                            "err": rec[:, 3],
                            "values": rec[:, 4:],
                            "interval": MS // 2,
                        }
                    ],
                ).stream("n0", "rdma")
                r = R.Rates(rs, c)
                end = r.end_ms()
                last, drop = None, None
                for j in np.flatnonzero(r.valid):
                    if last is not None and end[j] <= last:
                        drop = j + 1  # the sample ending interval j
                        break
                    last = end[j]
                if drop is None:
                    break
                rec = np.delete(rec, drop, axis=0)
            off_iv, off_resets, _ = offline(rs, c)
            self.assertEqual(live[c], off_iv, c)
            self.assertEqual(gaps.get((c, 0), []), off_resets, c)


class KeyCountersInSync(unittest.TestCase):
    def test_cpp_lists_equal_the_importer(self):
        import re

        sys.path.insert(0, os.path.join(HERE, "..", "report"))
        import ainic_netmon_import as I

        with open(os.path.join(HERE, "..", "src", "exporter.cc")) as fh:
            src = fh.read()
        for source, var in (("rdma", "kKeyCountersRdma"), ("ipc", "kKeyCountersIpc")):
            body = re.search(var + r"\[\] = \{(.*?)\};", src, re.S).group(1)
            self.assertEqual(
                re.findall(r'"([a-z_0-9]+)"', body), I.KEY_COUNTERS[source]
            )
        self.assertIn(I.OPTIONAL_PATTERN.pattern.replace("\\", "\\\\"), src)


if __name__ == "__main__":
    unittest.main()
