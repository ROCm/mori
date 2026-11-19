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
            {"ipc": "1", "ethtool": "2"}.get(stream.source, "0"),
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
                "glitches": int(p[7]),
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


def run_of(tmp, n=300, interval=2 * MS, mutate=None, names=NAMES, source="rdma"):
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
                "source": source,
                "names": names,
                "t0": t0,
                "rtt": rtt,
                "real": real,
                "err": err,
                "values": vals,
                "interval": interval,
            }
        ],
    ).stream("n0", source)


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
        self.assert_baseline(s, names, stats)
        return live, gaps, stats

    def assert_baseline(self, s, names, stats):
        # The live step's baseline is the offline one: max over counters of base_ns, the
        # live value rounded up to 10 us (CUSTOMER_FIXES_DESIGN §1).
        want = max(R.Rates(s, c).base_ns for c in names)
        self.assertGreaterEqual(stats["baseline_ns"], want - 1e-6)
        self.assertLess(stats["baseline_ns"], want + 10_000)

    def test_steady(self):
        _, _, st = self.assert_same(run_of(self.tmp))
        self.assertEqual(st["baseline_ns"], 2 * MS)

    def test_counter_reset(self):
        def m(t0, rtt, real, err, v):
            v[150:, 0] -= v[150, 0]
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m))

    def test_reset_hidden_in_a_slow_sample(self):
        # The counter resets in a slow sample and stays low (X -> slow 0 -> 7 -> ...): the
        # interval between the two kept samples around it is a reset. (An isolated dip that
        # comes back, X -> 0 -> X+..., is a read glitch instead: test_read_glitches_...)
        def m(t0, rtt, real, err, v):
            rtt[100] = rtt[100] * 4
            v[100:, 1] = v[100:, 1] - v[100, 1]
            v[101:, 1] += np.uint64(7)
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

    def test_ipc_read_anomalies(self):
        # Firmware a-45 IPC: one field's upper 32 bits garbage for one sample.
        HW = np.uint64(0x25FDC << 32)

        def m(t0, rtt, real, err, v):
            v[100, 0] += HW  # high-word corruption: skipped, bridging interval invalid
            v[180, 1] = np.uint64(
                5
            )  # an ambiguous dip: skipped, bridging interval invalid
            v[230:, 0] = v[230:, 0] - v[230, 0]  # a real reset stays a reset
            v[150, 2] += np.uint64(1 << 40)  # anomaly on the third counter too
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m, source="ipc")
        live, gaps, st = self.assert_same(s)  # identical to the offline rule
        self.assertEqual(st["glitches"], 3)
        for c in NAMES:
            r = R.Rates(s, c)
            self.assertLess(np.nanmax(r.rate) * 8 / 1e9, 1e4, c)  # no 1e9 Gb/s spike
        t = s.times()[0]
        r0 = R.Rates(s, NAMES[0])
        hw_bridge = (r0.t_start == t[99]) & (r0.t_end == t[101])
        self.assertTrue(hw_bridge.any() and not r0.valid[hw_bridge].any())
        r1 = R.Rates(s, NAMES[1])
        dip_bridge = (r1.t_start == t[179]) & (r1.t_end == t[181])
        self.assertTrue(dip_bridge.any() and not r1.valid[dip_bridge].any())
        self.assertTrue(r1.reset[dip_bridge].all())

    def test_ipc_consecutive_anomalies(self):
        # 10, 1000, 500, 20, 30 (Codex N1): judged on raw neighbours, offline and live alike
        def m(t0, rtt, real, err, v):
            v[100:105, 0] = (
                np.array([10, 1000, 500, 20, 30], dtype=np.uint64) + v[99, 0]
            )
            v[105:, 0] = v[105:, 0] - v[105, 0] + v[104, 0]
            v[200:204, 1] = v[200:204, 1] + np.array(
                [0, 1 << 33, 1 << 34, 0], dtype=np.uint64
            )
            return t0, rtt, real, err, v

        self.assert_same(run_of(self.tmp, mutate=m, source="ipc"))

    def test_reset_that_looks_like_a_high_word_glitch(self):
        # Codex N2: 0x1_10000000 -> 0x0_18000000 (a real reset) -> 0x1_20000000 after a long
        # stall is possible at line rate: never a valid bridged rate, offline or live.
        def m(t0, rtt, real, err, v):
            v[:120, 0] = np.uint64(0x1_10000000)
            v[120, 0] = np.uint64(0x0_18000000)
            v[121:, 0] = np.uint64(0x1_20000000) + np.arange(
                len(v) - 121, dtype=np.uint64
            )
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m, source="ipc")
        self.assert_same(s)
        r = R.Rates(s, NAMES[0])
        t = s.times()[0]
        from_119 = r.t_start == t[119]
        self.assertTrue(from_119.any() and not r.valid[from_119].any())

    def test_small_counter_reset_is_not_bridged(self):
        # 1 -> 0 -> 2 is what a real reset of a small counter looks like: never a valid rate
        def m(t0, rtt, real, err, v):
            v[:, 2] = 1
            v[150:, 2] = 0
            v[151:, 2] = 2
            return t0, rtt, real, err, v

        for src in ("rdma", "ipc"):
            s = run_of(f"{self.tmp}/{src}", mutate=m, source=src)
            self.assert_same(s)
            r = R.Rates(s, NAMES[2])
            t = s.times()[0]
            # nothing valid leaves the pre-reset sample (no bridge over the reset); the
            # interval out of it is a reset either way
            from_149 = r.t_start == t[149]
            self.assertTrue(from_149.any(), src)
            self.assertFalse(r.valid[from_149].any(), src)
            self.assertTrue(r.reset[from_149].all(), src)

    def test_ipc_anomaly_with_slow_sample_clock_step_and_activation(self):
        def m(t0, rtt, real, err, v):
            rtt[120] = rtt[120] * 4  # slow sample next to an anomaly
            v[121, 0] += np.uint64(1 << 36)
            real[200:] += np.uint64(5 * MS)  # clock step around another
            v[199, 1] = np.uint64(3)
            v[:, 2] = 0  # optional counter activated right at an anomaly
            v[250:, 2] = np.arange(50, dtype=np.uint64) * 3 + 1
            v[251, 2] += np.uint64(1 << 40)
            return t0, rtt, real, err, v

        s = run_of(self.tmp, mutate=m, source="ipc")
        live, gaps, _ = replay(s, NAMES, optional="001")
        for c in NAMES:
            off_iv, off_resets, _ = offline(s, c)
            if c == NAMES[2]:
                self.assertEqual(
                    live.get(c, []),
                    off_iv[-len(live.get(c, [])) :] if live.get(c) else [],
                )
            else:
                self.assertEqual(live.get(c, []), off_iv, c)
            self.assertEqual(
                [e for e, _ in gaps.get((c, 0), [])], [e for e, _ in off_resets], c
            )

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

    def test_baseline_counts_each_counters_own_intervals(self):
        # Sample spacing alternates 2 and 6 ms; counter 0 decreases at the end of every 2 ms
        # interval, so its baseline intervals are the 6 ms ones only (median 6 ms) while
        # counter 1 sees both (median 4 ms): the step must follow counter 0, as offline.
        def m(t0, rtt, real, err, v):
            gaps = np.where(np.arange(len(t0)) % 2 == 0, 2 * MS, 6 * MS)
            t0[1:] = t0[0] + np.cumsum(gaps[:-1]).astype(np.uint64)
            real[:] = t0.astype(np.int64) + 1_700_000_000 * 10**9
            v[:, 0] = np.where(
                np.arange(len(t0)) % 2 == 1, 0, 10**9 + np.arange(len(t0))
            )
            return t0, rtt, real, err, v

        _, _, st = self.assert_same(run_of(self.tmp, n=401, mutate=m))
        self.assertAlmostEqual(st["baseline_ns"], 6 * MS)

    def test_long_intervals_for_a_while_do_not_raise_the_baseline(self):
        # 30 s at 2 ms, then 15 s with 6 ms between kept samples (on the customer VMs a stretch
        # of slow requests did this): the recent-window median goes to 6 ms (the old live
        # step: 18 ms for the whole run), the whole-run median stays 2 ms, as offline.
        def m(t0, rtt, real, err, v):
            gaps = np.where(np.arange(len(t0)) < 15000, 2 * MS, 6 * MS)
            t0[1:] = t0[0] + np.cumsum(gaps[:-1]).astype(np.uint64)
            real[:] = t0.astype(np.int64) + 1_700_000_000 * 10**9
            v[:] = (t0[:, None] - t0[0]) * np.array([10, 17, 24], dtype=np.uint64)
            return t0, rtt, real, err, v

        _, _, st = self.assert_same(run_of(self.tmp, n=17500, mutate=m))
        self.assertAlmostEqual(st["baseline_ns"], 2 * MS)

    def test_ethtool_rates_between_refreshes(self):
        # 1 s reads of port statistics the firmware refreshes every 5 reads: the first change
        # is only the base (the values before it may be stale), then one interval per refresh;
        # an idle stretch (nothing changes) is one long interval; a counter reset is a gap.
        names = R.ETHTOOL_PORT_COUNTERS[:4]

        def m(t0, rtt, real, err, v):
            n = len(t0)
            refreshed = (np.arange(n) // 5) * 5  # index of the last refresh
            refreshed[(np.arange(n) >= 40) & (np.arange(n) < 70)] = 40  # idle 30 s
            v[:] = (
                refreshed[:, None] * np.array([1000, 3000, 7, 11]) + 10**9
            ).astype(np.uint64)
            v[0:5] = 5  # stale until the first refresh (read 5)
            v[85:, 1] -= v[85, 1] - 1  # rx bytes reset
            return t0, rtt, real, err, v

        run = run_of(
            self.tmp, n=120, interval=1000 * MS, mutate=m, names=names, source="ethtool"
        )
        live, gaps, _ = self.assert_same(run, names)
        iv = live[names[0]]
        self.assertEqual(iv[0][0] - iv[0][0] % 1000, iv[0][0])  # ends on a refresh read
        self.assertEqual(sorted({round(d, 6) for *_, d in iv}), [5.0, 30.0])
        self.assertEqual(
            iv[0][4], 5.0
        )  # the first interval starts at the first refresh
        self.assertFalse(any(low for _, low, *_ in iv))
        self.assertTrue(gaps.get((names[1], 0)))

    def test_baseline_above_100_ms(self):
        _, _, st = self.assert_same(run_of(self.tmp, n=200, interval=150 * MS))
        self.assertAlmostEqual(st["baseline_ns"], 150 * MS)


class KeyCountersInSync(unittest.TestCase):
    def test_cpp_lists_equal_the_importer(self):
        import re

        sys.path.insert(0, os.path.join(HERE, "..", "report"))
        import ainic_netmon_import as I

        with open(os.path.join(HERE, "..", "src", "exporter.cc")) as fh:
            src = fh.read()
        for source, var in (
            ("rdma", "kKeyCountersRdma"),
            ("ipc", "kKeyCountersIpc"),
            ("ethtool", "kEthtoolPortCounters"),
        ):
            body = re.search(var + r"\[\] = \{(.*?)\};", src, re.S).group(1)
            self.assertEqual(
                re.findall(r'"([a-z_0-9]+)"', body), I.KEY_COUNTERS[source]
            )
        self.assertIn(I.OPTIONAL_PATTERN.pattern.replace("\\", "\\\\"), src)


if __name__ == "__main__":
    unittest.main()
