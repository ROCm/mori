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
"""Tests for the VictoriaMetrics importer, the dashboards and the verifier (no VM needed)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "report"))
sys.path.insert(0, HERE)
import ainic_netmon_dashboards as G  # noqa: E402
import ainic_netmon_import as I  # noqa: E402
import ainic_netmon_report as R  # noqa: E402
import ainic_netmon_verify as V  # noqa: E402
from test_report import MS, steady, write_run  # noqa: E402

DASH_DIR = os.path.join(
    HERE, "..", "..", "..", "examples", "monitoring", "netmon", "grafana", "dashboards"
)


def rdma_run(
    d,
    n=40,
    tags=None,
    tx_rate=50.0,
    rx_rate=10.0,
    interval=2 * MS,
    mutate=None,
):
    t0, rtt, real, err, _ = steady(n, interval=interval)
    tx = (np.arange(n) * interval * tx_rate).astype(np.uint64)
    rx = (np.arange(n) * interval * rx_rate).astype(np.uint64)
    vals = np.stack([tx, rx], axis=1)
    if mutate:
        t0, rtt, real, vals = mutate(t0.copy(), rtt.copy(), real.copy(), vals.copy())
    run = write_run(
        d,
        [
            {
                "netdev": "n0",
                "source": "rdma",
                "names": [R.RDMA_TX, R.RDMA_RX],
                "t0": t0,
                "rtt": rtt,
                "real": real,
                "err": err,
                "values": vals,
                "interval": interval,
            }
        ],
        host="h",
    )
    if tags:
        meta_p = os.path.join(d, "meta.json")
        meta = json.load(open(meta_p))
        meta["tags"].update(tags)
        meta["start"] = {"realtime_ns": int(real[0]) - 5 * MS}
        json.dump(meta, open(meta_p, "w"))
        run = R.Run(d)
    return run


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="netmon-import-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_series_match_rates(self):
        run = rdma_run(self.tmp + "/a")
        t = I.Target(run, "n0", "e")
        s = run.stream("n0", "rdma")
        r = R.Rates(s, R.RDMA_TX)
        out = {
            lab["__name__"] + "/" + lab.get("resolution", ""): (ts, v)
            for lab, ts, v in I.interval_series(t, s, R.RDMA_TX, r)
        }
        ts, rate = out["netmon_rate/normal"]
        m = r.valid & ~r.lowres
        self.assertEqual(len(ts), int(m.sum()))
        self.assertTrue(np.array_equal(ts, r.end_ms()[m]))
        self.assertTrue(np.allclose(rate, r.rate[m]))
        self.assertTrue(np.allclose(out["netmon_increase/normal"][1], r.dv[m]))
        self.assertTrue(
            np.allclose(out["netmon_interval_seconds/normal"][1], r.dt[m] / 1e9)
        )
        self.assertTrue(np.all(np.diff(ts) > 0))

    def test_key_counters_are_imported_even_when_zero(self):
        run = rdma_run(self.tmp + "/a", tx_rate=0.0)
        s = run.stream("n0", "rdma")
        self.assertIn(R.RDMA_TX, I.pick_counters(s, "key"))

    def test_only_nic_counters_are_imported(self):
        run = rdma_run(self.tmp + "/a")
        s = run.stream("n0", "rdma")
        names = [c for c, _ in I.rate_objects(s, "key")]
        self.assertEqual(names, I.pick_counters(s, "key"))
        self.assertEqual(set(names), {R.RDMA_TX, R.RDMA_RX})

    def test_reset_gap_series(self):
        def mutate(t0, rtt, real, vals):
            vals[20:, 0] = vals[20:, 0] - vals[20, 0]
            return t0, rtt, real, vals

        run = rdma_run(self.tmp + "/a", mutate=mutate)
        t = I.Target(run, "n0", "e")
        s = run.stream("n0", "rdma")
        gaps = [
            (lab, ts)
            for lab, ts, _ in I.interval_series(t, s, R.RDMA_TX, R.Rates(s, R.RDMA_TX))
            if lab["__name__"] == "netmon_gap_seconds"
        ]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0][0]["reason"], "reset")
        self.assertEqual(len(gaps[0][1]), 1)

    def test_clock_stepping_back_is_refused(self):
        def mutate(t0, rtt, real, vals):
            real[25:] = (
                real[25:] - 50 * MS
            )  # wall clock steps back 50 ms, anchors trusted
            return t0, rtt, real, vals

        run = rdma_run(self.tmp + "/a", n=60, mutate=mutate)
        t = I.Target(run, "n0", "e")
        s = run.stream("n0", "rdma")
        with self.assertRaises(I.NetmonImportError):
            list(I.interval_series(t, s, R.RDMA_TX, R.Rates(s, R.RDMA_TX)))

    def test_batcher_budgets(self):
        b = I.Batcher(None, dry_run=True)
        n = I.MAX_LINE_POINTS * 3 + 7
        b.add({"__name__": "x"}, np.arange(n), np.ones(n))
        b.flush()
        self.assertEqual(b.total_points, n)
        self.assertGreaterEqual(b.requests, 1)
        b2 = I.Batcher(None, dry_run=True)
        for k in range(30):
            b2.add(
                {"__name__": "x", "k": str(k)},
                np.arange(I.MAX_LINE_POINTS),
                np.ones(I.MAX_LINE_POINTS),
            )
        b2.flush()
        self.assertGreater(b2.requests, 1)  # 600k points > 500k per request

    def test_selector_escaping(self):
        sel = I.selector("netmon_.+", experiment='a"b\\c', host="h")
        self.assertEqual(
            sel, '{__name__=~"netmon_.+",experiment="a\\"b\\\\c",host="h"}'
        )

    def test_zero_valid_target_still_has_metadata(self):
        run = rdma_run(self.tmp + "/a", n=1)  # one sample: no interval at all
        t = I.Target(run, "n0", "e")
        want, summaries, infos, quality, min_step = I.plan_summary([t], "key", None)
        self.assertEqual(want, {})
        self.assertEqual(list(I.iter_series([t], "key")), [])
        self.assertEqual(len(infos), 1)
        stats = {lab["stat"] for lab, _ in summaries}
        self.assertEqual(stats, {"coverage"})
        self.assertTrue(all(v == 0.0 for _, v in summaries))

    def test_non_overlapping_recordings_are_refused(self):
        def later(t0, rtt, real, vals):
            return t0, rtt, real + np.uint64(10**12), vals  # 1000 s later

        a = rdma_run(self.tmp + "/a")
        b = rdma_run(self.tmp + "/b", mutate=later)
        with self.assertRaises(I.NetmonImportError):
            I.run_window([a, b])
        self.assertIsNotNone(I.run_window([a]))

    def test_experiment_is_imported_as_a_whole(self):
        class FakeVM:
            def __init__(self, hosts):
                self.hosts = hosts

            def flags(self):
                return {"retentionPeriod": "100y", "search.latencyOffset": "0s"}

            def label_values(self, label, match):
                return set(self.hosts)

        run = rdma_run(self.tmp + "/a")  # host "h"
        I.preflight(FakeVM([]), [run], "e", replace=False)
        with self.assertRaises(I.NetmonImportError):  # exists, no --replace
            I.preflight(FakeVM(["h"]), [run], "e", replace=False)
        I.preflight(FakeVM(["h"]), [run], "e", replace=True)
        with self.assertRaises(I.NetmonImportError):  # would leave host "g" behind
            I.preflight(FakeVM(["h", "g"]), [run], "e", replace=True)


class WindowTests(unittest.TestCase):
    def test_right_closed_windows(self):
        class FakeRates:
            valid = np.array([True, True, True])
            lowres = np.array([False, False, False])
            rate = np.array([1.0, 2.0, 3.0])
            dv = np.array([1.0, 2.0, 3.0])
            dt = np.array([1e9, 1e9, 1e9])
            w_end = np.array([100e6, 101e6, 200e6])

            def end_ms(self):
                return np.round(self.w_end / 1e6).astype(np.int64)

        grid = R.epoch_grid(1, 200, 100)
        self.assertEqual(grid.tolist(), [0, 100, 200])
        a = R.window_aggregate(FakeRates(), grid, 100)
        # 100 is in (0, 100]; 101 in (100, 200]; 200 in (100, 200]
        self.assertEqual(a["max"][1], 1.0)
        self.assertEqual(a["max"][2], 3.0)
        self.assertEqual(a["min"][2], 2.0)
        self.assertTrue(np.isnan(a["max"][0]))


class VerifierTests(unittest.TestCase):
    def test_compare_catches_every_kind_of_difference(self):
        grid = np.array([0, 10, 20, 30, 40])
        exp = np.array([np.nan, 1.0, 2.0, np.nan, 4.0])
        ok_ts, ok_v = [10, 20, 40], [1.0, 2.0, 4.0]
        self.assertEqual(V.compare_series(grid, exp, ok_ts, ok_v, "ok"), 3)
        with self.assertRaises(V.Mismatch):  # a window with data is missing
            V.compare_series(grid, exp, [10, 40], [1.0, 4.0], "missing")
        with self.assertRaises(V.Mismatch):  # an extra point where the report has none
            V.compare_series(grid, exp, [10, 20, 30, 40], [1, 2, 9, 4], "extra")
        with self.assertRaises(V.Mismatch):  # a wrong value
            V.compare_series(grid, exp, ok_ts, [1.0, 2.5, 4.0], "value")
        with self.assertRaises(V.Mismatch):  # an empty frame when the report has data
            V.compare_series(grid, exp, [], [], "empty")
        with self.assertRaises(V.Mismatch):  # off-grid timestamp
            V.compare_series(grid, exp, [10, 21, 40], [1, 2, 4], "offgrid")

    def test_uncertain_boundaries(self):
        grid, unc = V.grafana_grid(1005, 1095, 10)
        self.assertEqual(grid[0], 990)
        self.assertEqual(grid[-1], 1090)
        self.assertEqual(unc, {990, 1090})
        exp = np.full(len(grid), 1.0)
        interior = [int(x) for x in grid[1:-1]]
        # boundary windows may be absent ...
        V.compare_series(
            grid, exp, interior, [1.0] * len(interior), "no boundaries", unc
        )
        # ... but if present must still be right
        with self.assertRaises(V.Mismatch):
            V.compare_series(
                grid,
                exp,
                [990] + interior,
                [5.0] + [1.0] * len(interior),
                "bad boundary",
                unc,
            )
        # interior windows stay strict
        with self.assertRaises(V.Mismatch):
            V.compare_series(
                grid, exp, interior[1:], [1.0] * (len(interior) - 1), "interior", unc
            )

    def test_frames_must_be_the_expected_series(self):
        tmp = tempfile.mkdtemp(prefix="netmon-import-test-")
        try:
            run = rdma_run(tmp + "/a")
            t = I.Target(run, "n0", "e")
            t.counters_mode = "key"
            ex = V.Expected(I)
            spec = {"kind": "max", "source": "rdma", "counter": R.RDMA_TX}
            lab = dict(t.labels(), source="rdma", counter=R.RDMA_TX)

            def fr(labels, ts=(1, 2)):
                return (labels, 1, list(ts), [1.0] * len(ts))

            self.assertEqual(
                list(V.check_frames(I, ex, t, spec, [fr(lab)], "ok")),
                [("rdma", R.RDMA_TX)],
            )
            bad = [
                [fr(lab), fr(lab)],  # duplicate frame
                [fr(dict(lab, target="h/n1"))],  # another target
                [fr(dict(lab, experiment="x"))],  # another experiment
                [fr(dict(lab, counter=R.RDMA_RX))],  # a series not asked for
                [fr(lab, ts=(2, 2))],  # repeated timestamp
            ]
            for frames in bad:
                with self.assertRaises(V.Mismatch):
                    V.check_frames(I, ex, t, spec, frames, "bad")
        finally:
            shutil.rmtree(tmp)

    def test_query_config_sees_datasource_changes(self):
        d = json.loads(G.render()["netmon_nic_detail.json"])
        d2 = json.loads(json.dumps(d))
        for p in d2["panels"]:
            if p.get("datasource"):
                p["datasource"]["uid"] = "other"
                break
        self.assertNotEqual(V.query_config(d), V.query_config(d2))
        d3 = json.loads(json.dumps(d))
        for v in d3["templating"]["list"]:
            if v["name"] == "min_step":
                v["datasource"] = {"type": "prometheus", "uid": "other"}
        self.assertNotEqual(V.query_config(d), V.query_config(d3))


class DashboardTests(unittest.TestCase):
    def test_committed_json_is_generated(self):
        for name, text in G.render().items():
            with open(os.path.join(DASH_DIR, name)) as fh:
                self.assertEqual(
                    fh.read(),
                    text,
                    f"{name}: regenerate with ainic_netmon_dashboards.py",
                )

    def test_query_rules(self):
        for name, text in G.render().items():
            d = json.loads(text)
            self.assertNotIn("insertNulls", text)
            self.assertNotIn("spanNulls", text)
            for p in d["panels"]:
                for t in p.get("targets", []):
                    if t.get("instant"):
                        self.assertIn("last_over_time(", t["expr"])
                        self.assertIn("[3650d]", t["expr"])
                    else:
                        self.assertIn("netmon", t, f"{name}: data query without spec")
                        self.assertIn("[$__interval]", t["expr"])
                        self.assertNotIn("[$__rate_interval]", t["expr"])
                if p["type"] == "timeseries":
                    self.assertIn(p["interval"], ("${min_step}ms", "1ms"))
                    self.assertEqual(p["datasource"]["uid"], "netmon-vm")
            # Multi-valued variables are regex-escaped by Grafana: match them with =~,
            # single-valued ones (not escaped) with =.
            for v in d["templating"]["list"]:
                multi = v.get("multi") or v.get("includeAll")
                for p in d["panels"]:
                    for t in p.get("targets", []):
                        if f'"${v["name"]}"' in t["expr"]:
                            op = "=~" if multi else "="
                            self.assertIn(f'{v["name"]}{op}"${v["name"]}"', t["expr"])

    def test_interpolate_like_grafana(self):
        multi = {"templating": {"list": [{"name": "x", "multi": True}]}}
        single = {"templating": {"list": [{"name": "x"}]}}
        host = "n06-21.prov.amd.com/benic1p1"
        self.assertEqual(V.interpolate(single, "x", host), host)
        self.assertEqual(V.interpolate(multi, "x", host), host.replace(".", "\\\\."))


if __name__ == "__main__":
    unittest.main()
