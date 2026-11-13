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
"""ainic-netmon-import: import ainic-netmon recordings into VictoriaMetrics for Grafana.

Design: work-docs projects/ainic-monitor/NETMON_DASHBOARD_DESIGN.md. In short:

  * What is imported are per-interval series computed by the offline report library
    (ainic_netmon_report): trusted-sample times, slow samples skipped, resets found on the
    full series, wall-clock segments, low resolution = interval > 3x baseline. The
    dashboards only aggregate them over explicit windows, so they cannot disagree with the
    report about any of these rules.
  * Per counter: netmon_rate / netmon_increase / netmon_interval_seconds at the interval
    end (integer ms), labelled resolution=normal|low; invalid intervals as
    netmon_gap_seconds{reason=reset|clock}. Only NIC counters are imported; nothing
    about the workload (roles, traffic classes) is inferred.
  * Metadata (fixed-window summary = the offline report's numbers, target info, run
    quality, import marker) is written at the experiment anchor (earliest recording start
    - 1 ms) and read with last_over_time(...[3650d]).
  * Only failures that certainly wrote nothing are retried; an uncertain write stops the
    import without the completion marker (re-run with --replace).

Only numpy and the standard library are needed.
"""

import argparse
import datetime as _dt
import gzip
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

try:  # packaged with mori
    from . import _netmon_report as R
except ImportError:  # standalone, next to ainic_netmon_report.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ainic_netmon_report as R  # noqa: E402

KEY_COUNTERS = {
    "rdma": [
        "tx_rdma_ucast_bytes",
        "rx_rdma_ucast_bytes",
        "tx_rdma_ucast_pkts",
        "rx_rdma_ucast_pkts",
        "tx_rdma_cnp_pkts",
        "rx_rdma_cnp_pkts",
        "rx_rdma_ecn_pkts",
        "tx_rdma_retx_bytes",
        "tx_rdma_retx_pkts",
        "tx_rdma_ack_timeout",
    ],
    "ipc": ["octets_tx_ok", "octets_rx_ok", "frames_tx_ok", "frames_rx_ok"],
}
# Imported when they change during the recording (default mode).
OPTIONAL_PATTERN = re.compile(
    r"(pripause_\d+_1us_count|rx_pause_1us_count|frames_(tx|rx)_pri_\d)$"
)
# Fixed-window summaries (the offline report's table) for the byte counters.
SUMMARY_COUNTERS = {
    "rdma": ["tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes"],
    "ipc": ["octets_tx_ok", "octets_rx_ok"],
}
METRIC_PREFIX = "netmon_"
# Start of every VM search over "all" data. Not 0: VM treats start=0 as unset and the
# label / series APIs then only look at the last day (measured on v1.156.0; some other
# small epoch values also return nothing), so an older experiment would look absent.
ALL_TIME_START = "2000-01-01T00:00:00Z"
# ... and its end. Without one VM ends at the current time rounded down to the second, so a
# read-back right after a recording misses its last fraction of a second (measured).
ALL_TIME_END = "2100-01-01T00:00:00Z"
ALL_TIME = {"start": ALL_TIME_START, "end": ALL_TIME_END}
MAX_REQUEST_BYTES = 8 << 20  # uncompressed
MAX_REQUEST_POINTS = 500_000
MAX_LINE_POINTS = 20_000  # keeps one JSON line well under 1 MiB


class NetmonImportError(Exception):
    pass


def is_bytes(counter):
    return "bytes" in counter or "octets" in counter


def esc(v):
    """PromQL / MetricsQL string literal escaping for label values."""
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def selector(name_re, **labels):
    parts = [f'__name__=~"{name_re}"'] + [
        f'{k}="{esc(v)}"' for k, v in sorted(labels.items())
    ]
    return "{" + ",".join(parts) + "}"


def iso_ms(ms):
    return (
        _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3]
        + "Z"
    )


# --------------------------------------------------------------------------- VictoriaMetrics


class VM:
    def __init__(self, url, timeout=120):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _open(self, req):
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            raise NetmonImportError(
                f"VictoriaMetrics {req.full_url}: HTTP {e.code}: {body}"
            )

    def get(self, path, params=None):
        q = ("?" + urllib.parse.urlencode(params, doseq=True)) if params else ""
        with self._open(urllib.request.Request(self.url + path + q)) as r:
            return r.read()

    def get_json(self, path, params=None):
        return json.loads(self.get(path, params))

    def post(self, path, body, headers=None, retries=3):
        """POST. Retried only when the connection could not be established (nothing was
        sent); any other failure leaves the outcome unknown and is raised."""
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.url + path, data=body, headers=headers or {}, method="POST"
            )
            try:
                with self._open(req) as r:
                    return r.status
            except urllib.error.URLError as e:
                if isinstance(e.reason, ConnectionRefusedError) and attempt < retries:
                    continue
                raise NetmonImportError(
                    f"write to {self.url}{path} failed ({e.reason}); its outcome is unknown, "
                    "so nothing is retried. Re-run the import with --replace."
                )

    def flush(self):
        """Make just-imported samples searchable (VM buffers them for ~1 s)."""
        self.get("/internal/force_flush")

    def flags(self):
        """Explicitly set flags (VM lists only non-default ones)."""
        out = {}
        for line in self.get("/flags").decode().splitlines():
            m = re.match(r'-([\w.]+)="?(.*?)"?$', line.strip())
            if m:
                out[m.group(1)] = m.group(2)
        return out

    def series(self, match):
        d = self.get_json("/api/v1/series", {"match[]": match, **ALL_TIME})
        return d.get("data", [])

    def label_values(self, label, match):
        d = self.get_json(
            f"/api/v1/label/{label}/values", {"match[]": match, **ALL_TIME}
        )
        return set(d.get("data", []))

    def instant(self, expr, at_s=None):
        p = {"query": expr}
        if at_s is not None:
            p["time"] = f"{at_s:.3f}"
        return self.get_json("/api/v1/query", p)["data"]["result"]

    def delete(self, match):
        self.post(
            "/api/v1/admin/tsdb/delete_series?"
            + urllib.parse.urlencode({"match[]": match}),
            b"",
        )

    def export(self, match):
        """Stream /api/v1/export line by line (a long experiment exports hundreds of
        millions of points; never hold the whole response)."""
        q = urllib.parse.urlencode({"match[]": match, **ALL_TIME})
        with self._open(urllib.request.Request(f"{self.url}/api/v1/export?{q}")) as r:
            for line in r:
                if line.strip():
                    yield json.loads(line)

    def export_extent(self, match):
        """{label key: (points, first ts, last ts)} of every series matching `match`.
        VM may split a series over several export lines, so lines are merged."""
        out = {}
        for line in self.export(match):
            k = label_key(line["metric"])
            ts = line["timestamps"]
            n, a, b = out.get(k, (0, None, None))
            lo, hi = min(ts), max(ts)
            out[k] = (
                n + len(ts),
                lo if a is None else min(a, lo),
                hi if b is None else max(b, hi),
            )
        return out


def parse_vm_duration_ms(v, default_unit="ms"):
    """VM flag durations: 100y, 30d, 1h, 5ms, 0s; a bare number means months for
    retentionPeriod (handled by the caller) and default_unit otherwise."""
    m = re.fullmatch(r"([\d.]+)(ms|s|m|h|d|w|y)?", v.strip())
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2) or default_unit
    mult = {
        "ms": 1,
        "s": 1e3,
        "m": 60e3,
        "h": 3600e3,
        "d": 86400e3,
        "w": 7 * 86400e3,
        "y": 365 * 86400e3,
        "mo": 31 * 86400e3,
    }[unit]
    return n * mult


# --------------------------------------------------------------------------- series generation


class Target:
    def __init__(self, run, netdev, experiment):
        self.run = run
        self.host = run.host
        self.netdev = netdev
        self.target = f"{run.host}/{netdev}"
        self.experiment = experiment

    def labels(self, **extra):
        d = {
            "experiment": self.experiment,
            "host": self.host,
            "netdev": self.netdev,
            "target": self.target,
        }
        d.update(extra)
        return d

    def nic_labels(self):
        """Names for people (metadata series only; the interval series keep their identity)."""
        nic = next(
            (
                n
                for n in self.run.meta.get("nics", [])
                if n.get("netdev") == self.netdev
            ),
            {},
        )
        return display_labels(
            self.host, self.netdev, nic.get("ibdev", ""), nic.get("pci", "")
        )

    def meta_labels(self, **extra):
        return dict(self.labels(**extra), **self.nic_labels())


def display_labels(host, netdev, ibdev, pci):
    """The dashboards' NIC names (the live exporter builds the same: exporter.cc
    DisplayLabels; test_push.cc checks the same cases): "ionic_0 (benic1p1)", and for the
    multi-host overview "<host's first DNS label> · ionic_0" (short enough for a chart
    title; the netdev is in the overview table)."""
    nic = f"{ibdev} ({netdev})" if ibdev else netdev
    out = {
        "display_nic": nic,
        "display_host_nic": f"{host.split('.')[0]} · {ibdev or netdev}",
    }
    # VM drops empty label values: leave unknown ones out instead.
    if ibdev:
        out["ibdev"] = ibdev
    if pci:
        out["pci"] = pci
    return out


def pick_counters(stream, mode):
    names = stream.names
    key = [c for c in KEY_COUNTERS.get(stream.source, []) if c in names]
    out = list(key)
    for c in names:
        if c in out or not len(stream.rec):
            continue
        optional = OPTIONAL_PATTERN.search(c) is not None
        if not (optional or mode == "all"):
            continue
        col = stream.col(c)
        if int(col.max()) != int(col.min()):  # changed during the recording
            out.append(c)
    return out


def rate_objects(stream, mode):
    """(counter name, Rates) for every imported counter of one stream."""
    return [(c, R.Rates(stream, c)) for c in pick_counters(stream, mode)]


def check_increasing(ts, what):
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        bad = int(np.flatnonzero(np.diff(ts) <= 0)[0])
        raise NetmonImportError(
            f"{what}: timestamps are not strictly increasing in ms at {iso_ms(int(ts[bad]))} "
            "(wall clock stepped back?); refusing to import this recording"
        )


def interval_series(target, stream, counter, r):
    """Yield (labels, ts_ms ndarray, values ndarray) for rate/increase/interval x resolution
    and for gaps."""
    end = r.end_ms()
    base = target.labels(source=stream.source, counter=counter)
    for res, mask in (("normal", r.valid & ~r.lowres), ("low", r.valid & r.lowres)):
        if not mask.any():
            continue
        ts = end[mask]
        check_increasing(ts, f"{target.target} {stream.source}.{counter}")
        lab = dict(base, resolution=res)
        yield dict(lab, __name__="netmon_rate"), ts, r.rate[mask]
        yield dict(lab, __name__="netmon_increase"), ts, r.dv[mask]
        yield dict(lab, __name__="netmon_interval_seconds"), ts, r.dt[mask] / 1e9
    for reason, mask in (("reset", r.reset), ("clock", r.clock)):
        m = mask & (
            end >= 0
        )  # an interval with no known wall-clock end cannot be placed
        if not m.any():
            continue
        ts = end[m]
        # Gaps are informational; collapse duplicates that rounding to ms could create.
        ts, first = np.unique(ts, return_index=True)
        yield dict(base, __name__="netmon_gap_seconds", reason=reason), ts, (
            r.dt[m][first] / 1e9
        )


def run_window(runs):
    """The fixed experiment window: the offline report's common window over the runs."""
    starts, ends = [], []
    for r in runs:
        ws = [
            s.times()[1]
            for s in r.streams
            if s.source in ("rdma", "ipc") and len(s.rec) > 1
        ]
        ws = [w[np.isfinite(w)] for w in ws]
        ws = [w for w in ws if len(w) > 1]
        if ws:
            starts.append(min(w[0] for w in ws))
            ends.append(max(w[-1] for w in ws))
    if not starts:
        return None  # no samples at all: metadata only, coverage 0
    w0, w1 = max(starts), min(ends)
    if w1 <= w0:
        raise NetmonImportError(
            "the recordings do not overlap in time (no common window); import them as "
            "separate experiments"
        )
    return w0, w1


def summary_values(target, stream, counter, r, window):
    """The offline report's stats for one counter over the fixed window, in the counter's
    own unit per second. Empty stats are omitted (not zero-filled); coverage is always
    present."""
    if window is None:
        return {"coverage": 0.0}
    w0, w1 = window
    speed = target.run.speed.get(target.netdev, -1)
    threshold = (
        speed * 1e6 / 8 * 0.8 if (speed and speed > 0 and is_bytes(counter)) else None
    )
    st = R.series_stats(r, w0, w1, 1.0, threshold)
    out = {
        "avg": st["avg"],
        "peak10ms": st["peak10"],
        "p99": st["p99"],
        "above80": st["above"],
        "coverage": st["coverage"],
        "lowres_n": float(st["lowres_n"]),
        "lowres_s": st["lowres_s"],
    }
    return {k: v for k, v in out.items() if v is not None and math.isfinite(v)}


class Batcher:
    """Accumulates JSON lines and POSTs them under the request budgets."""

    def __init__(self, vm, dry_run):
        self.vm, self.dry = vm, dry_run
        self.lines, self.bytes, self.points = [], 0, 0
        self.total_points = self.total_bytes = self.requests = 0

    def add(self, labels, ts, vals):
        ts = np.asarray(ts, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        for a in range(0, len(ts), MAX_LINE_POINTS):
            t, v = ts[a : a + MAX_LINE_POINTS], vals[a : a + MAX_LINE_POINTS]
            line = json.dumps(
                {
                    "metric": labels,
                    "values": [float(x) for x in v],
                    "timestamps": [int(x) for x in t],
                },
                separators=(",", ":"),
            )
            if self.lines and (
                self.bytes + len(line) > MAX_REQUEST_BYTES
                or self.points + len(t) > MAX_REQUEST_POINTS
            ):
                self.flush()
            self.lines.append(line)
            self.bytes += len(line) + 1
            self.points += len(t)

    def flush(self):
        if not self.lines:
            return
        body = ("\n".join(self.lines) + "\n").encode()
        self.total_points += self.points
        self.total_bytes += len(body)
        self.requests += 1
        if not self.dry:
            self.vm.post(
                "/api/v1/import",
                gzip.compress(body, 3),
                {"Content-Encoding": "gzip", "Content-Type": "application/json"},
            )
        self.lines, self.bytes, self.points = [], 0, 0


def label_key(labels):
    return tuple(sorted(labels.items()))


# --------------------------------------------------------------------------- commands


def load_runs(args):
    runs = [R.Run(d) for d in args.run_dirs]
    hosts = [r.host for r in runs]
    if len(set(hosts)) != len(hosts):
        raise SystemExit(f"two recordings have the same host id: {hosts}")
    if args.experiment:
        exp = args.experiment
    else:
        parents = {
            os.path.basename(os.path.dirname(os.path.abspath(d))) for d in args.run_dirs
        }
        if len(parents) != 1:
            raise SystemExit(
                "recordings are in different parent directories: pass --experiment"
            )
        exp = parents.pop()
    return runs, exp


def targets_of(runs, exp):
    out = []
    for r in runs:
        for nd in r.netdevs():
            if any(s.netdev == nd and s.source in ("rdma", "ipc") for s in r.streams):
                out.append(Target(r, nd, exp))
    return out


def preflight(vm, runs, exp, replace):
    flags = vm.flags()
    ret = flags.get("retentionPeriod", "1")
    ret_ms = parse_vm_duration_ms(ret, default_unit="mo")
    oldest = min(
        int(r.meta.get("start", {}).get("realtime_ns", 0)) // 1_000_000 for r in runs
    )
    now_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    if ret_ms is None or now_ms - oldest > ret_ms * 0.95:
        raise NetmonImportError(
            f"VictoriaMetrics retentionPeriod={ret} does not cover recordings from {iso_ms(oldest)}; "
            "VM would silently drop them (start it with a longer -retentionPeriod)"
        )
    dedup = flags.get("dedup.minScrapeInterval", "0")
    if (parse_vm_duration_ms(dedup) or 0) > 0:
        raise NetmonImportError(
            f"VictoriaMetrics dedup.minScrapeInterval={dedup} would merge ms samples"
        )
    lat = flags.get("search.latencyOffset", "30s")
    if (parse_vm_duration_ms(lat) or 0) > 0:
        print(
            f"warning: VictoriaMetrics search.latencyOffset={lat}: the newest {lat} of data is "
            "not queryable (the netmon deployment sets it to 0s)",
            file=sys.stderr,
        )
    # An experiment is always imported as a whole: its fixed window, min_step and anchor
    # depend on every recording, so a partial re-import would mix two definitions.
    existing = vm.label_values("host", selector("netmon_.+", experiment=exp))
    if existing and not replace:
        raise NetmonImportError(
            f"experiment {exp!r} already exists (hosts {sorted(existing)}); use --replace "
            "to re-import it, or another --experiment"
        )
    left_out = existing - {r.host for r in runs}
    if left_out:
        raise NetmonImportError(
            f"--replace re-imports the whole experiment {exp!r}, but hosts {sorted(left_out)} "
            "are not in this import: pass all its recordings (or use another --experiment)"
        )


def iter_series(targets, counters_mode):
    """Every series to import, generated one stream at a time (a long recording has
    hundreds of millions of points; only one stream's arrays are alive at once)."""
    for t in targets:
        for s in t.run.streams:
            if s.netdev != t.netdev or s.source not in ("rdma", "ipc"):
                continue
            for counter, r in rate_objects(s, counters_mode):
                yield from interval_series(t, s, counter, r)


def plan_summary(targets, counters_mode, window):
    """First pass, with every check done before anything is written: the expected
    (points, first ts, last ts) per series key, the metadata series and min_step."""
    want = {}
    summaries, infos, quality = [], [], []
    min_step = 0
    for t in targets:
        st = t.run.meta.get("run_stats") or {}
        for s in t.run.streams:
            if s.netdev != t.netdev or s.source not in ("rdma", "ipc"):
                continue
            q = st.get(s.file, {})
            for k in ("samples", "missed_slots", "dropped", "errors"):
                if k in q:
                    quality.append(
                        (
                            t.meta_labels(
                                source=s.source, stat=k, row=f"{s.source} {t.target}"
                            ),
                            float(q[k]),
                        )
                    )
            for counter, r in rate_objects(s, counters_mode):
                for lab, ts, _ in interval_series(t, s, counter, r):
                    want[label_key(lab)] = (len(ts), int(ts[0]), int(ts[-1]))
                if r.valid.any():
                    min_step = max(
                        min_step, int(math.ceil(R.LOWRES_FACTOR * r.base_ns / 1e6))
                    )
                if counter in SUMMARY_COUNTERS.get(s.source, []):
                    for stat, v in summary_values(t, s, counter, r, window).items():
                        lab = t.meta_labels(
                            source=s.source,
                            counter=counter,
                            stat=stat,
                            row=f"{s.source}.{counter} {t.target}",
                        )
                        if window:
                            lab.update(
                                window_start=iso_ms(int(window[0] // 1e6)),
                                window_end=iso_ms(int(window[1] // 1e6)),
                            )
                        summaries.append((lab, v))
        infos.append(
            t.meta_labels(
                status=t.run.status,
                speed_mbps=str(t.run.speed.get(t.netdev, -1)),
            )
        )
    return want, summaries, infos, quality, max(min_step, 1)


def cmd_import(args):
    runs, exp = load_runs(args)
    targets = targets_of(runs, exp)
    window = run_window(runs)
    vm = VM(args.url)
    if not args.dry_run:
        preflight(vm, runs, exp, args.replace)
    anchor = (
        min(
            int(r.meta.get("start", {}).get("realtime_ns", 0)) // 1_000_000
            for r in runs
        )
        - 1
    )

    # ---- pass 1: every check, before anything is written ----
    expected, summaries, infos, quality, min_step = plan_summary(
        targets, args.counters, window
    )
    npoints = sum(v[0] for v in expected.values())
    print(
        f"experiment {exp}: {len(targets)} targets, {len(expected)} series, {npoints} points, "
        f"min_step {min_step} ms{'' if window else ' (no valid intervals in any recording)'}"
    )

    # ---- pass 2: write, one stream at a time ----
    if args.replace and not args.dry_run:
        vm.delete(selector("netmon_.+", experiment=exp))
    b = Batcher(vm, args.dry_run)
    for lab, ts, vals in iter_series(targets, args.counters):
        b.add(lab, ts, vals)
    b.flush()
    if args.dry_run:
        print(
            f"dry run: {b.requests} requests, {b.total_bytes / 1e6:.1f} MB uncompressed"
        )
        return 0

    # ---- read back, then publish metadata and the completion marker ----
    vm.flush()
    for r in runs:
        got = vm.export_extent(
            selector(
                "netmon_(rate|increase|interval_seconds|gap_seconds)",
                experiment=exp,
                host=r.host,
            )
        )
        want = {k: v for k, v in expected.items() if dict(k)["host"] == r.host}
        bad = [k for k, v in want.items() if got.get(k) != v]
        extra = [k for k in got if k not in want]
        if bad or extra:
            raise NetmonImportError(
                f"read-back mismatch for {r.host}: {len(bad)} series differ, {len(extra)} unexpected "
                f"(first: {dict((bad or extra)[0])}); the import is incomplete, re-run with --replace"
            )
    meta = []
    for lab, v in summaries:
        meta.append((dict(lab, __name__="netmon_summary"), v))
    for lab in infos:
        meta.append((dict(lab, __name__="netmon_target_info"), 1.0))
    for lab, v in quality:
        meta.append((dict(lab, __name__="netmon_run_quality"), v))
    for r in runs:  # the dashboards' min_step variable (max over hosts)
        meta.append(
            (
                {"__name__": "netmon_step_ms", "experiment": exp, "host": r.host},
                float(min_step),
            )
        )
    for lab, v in meta:
        b.add(lab, [anchor], [v])
    b.flush()
    for r in runs:
        b.add(
            {
                "__name__": "netmon_import_info",
                "experiment": exp,
                "host": r.host,
                "state": "complete",
                "anchor_ms": str(anchor),
                "min_step_ms": str(min_step),
            },
            [anchor],
            [1.0],
        )
    b.flush()

    if window:
        f, t = int(window[0] // 1e6), int(window[1] // 1e6)
    else:
        f, t = anchor, anchor + 1000
    q = urllib.parse.urlencode({"from": f, "to": t, "var-experiment": exp})
    print(f"imported {b.total_points} points in {b.requests} requests; read-back OK")
    print(
        f"overview:   {args.grafana.rstrip('/')}/d/netmon-overview/netmon-overview?{q}"
    )
    print(f"NIC detail: {args.grafana.rstrip('/')}/d/netmon-nic/netmon-nic-detail?{q}")
    return 0


def main(argv=None):
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    p = argparse.ArgumentParser(
        prog="ainic-netmon-import", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd")
    imp = sub.add_parser("import", help="import recordings (the default command)")
    ver = sub.add_parser(
        "verify", help="check VictoriaMetrics / Grafana against the offline report"
    )
    for sp in (imp, ver):
        sp.add_argument("run_dirs", nargs="+")
        sp.add_argument(
            "--url", default="http://127.0.0.1:8428", help="VictoriaMetrics"
        )
        sp.add_argument("--grafana", default="http://127.0.0.1:3000")
        sp.add_argument(
            "--experiment", help="default: the recordings' parent directory name"
        )
        sp.add_argument("--counters", choices=("key", "all"), default="key")
    imp.add_argument(
        "--replace",
        action="store_true",
        help="delete the whole experiment first (every host of it must be in this import)",
    )
    imp.add_argument("--dry-run", action="store_true")
    ver.add_argument(
        "--dashboards", help="dashboard JSON directory (default: fetch from Grafana)"
    )
    ver.add_argument(
        "--skip-readback", action="store_true", help=argparse.SUPPRESS
    )  # debugging
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("import", "verify", "-h", "--help"):
        argv = ["import"] + argv
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            return _verify_module().main_verify(args, sys.modules[__name__])
        return cmd_import(args)
    except NetmonImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _verify_module():
    try:
        from . import _netmon_verify as V
    except ImportError:
        import ainic_netmon_verify as V
    return V


if __name__ == "__main__":
    sys.exit(main())
