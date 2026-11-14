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
"""Verification for `ainic-netmon-import verify`: the dashboards must agree with the
offline report.

NETMON_DASHBOARD_DESIGN §8. For an imported experiment:
  1. read-back: every series VictoriaMetrics holds matches what the importer generates now;
  2. summaries: netmon_summary == the report's series_stats for the fixed window;
  3. through Grafana: every data query of the deployed dashboards is executed with
     /api/ds/query over several ranges and widths; the full window grid is enumerated
     independently (actual step from the frame, Grafana's epoch alignment), and both the
     set of windows with a point and the values must equal the report library's
     window_aggregate on that grid;
  4. metadata (target info) is found when zoomed into the earliest and latest sub-windows
     of the experiment.
"""

import json
import math
import os
import re
import urllib.request

import numpy as np

DISPLAY_LABELS = ("ibdev", "pci", "display_nic", "display_host_nic")
# Metadata formats of earlier importers, newest first: before the ethtool fallback (no
# port_stats), and before the NIC display names as well.
LEGACY_LABELS = (("port_stats",), ("port_stats",) + DISPLAY_LABELS)
REL_TOL = 1e-9
ABS_TOL = 1e-9


class Mismatch(Exception):
    pass


# --------------------------------------------------------------------------- pure checks


def grafana_grid(from_ms, to_ms, step_ms):
    """Window ends Grafana may ask VictoriaMetrics for, and which of them are uncertain.

    Measured (NETMON_DASHBOARD_DESIGN §2): the grid is the multiples of the step from the
    epoch between g0 = floor(from/step)*step and g1 = floor(to/step)*step, except that the
    seconds/ms conversion between Grafana and VM sometimes moves the start one step earlier
    and sometimes drops the end (about 13% of queries at 1 ms, each). So the grid spans
    [g0 - step, g1]; the two outer windows may or may not be requested, every window in
    between always is. Returns (grid, set of uncertain window ends)."""
    step_ms = int(step_ms)
    g0 = (from_ms // step_ms) * step_ms
    g1 = (to_ms // step_ms) * step_ms
    grid = np.arange(g0 - step_ms, g1 + 1, step_ms, dtype=np.int64)
    return grid, {int(g0 - step_ms), int(g1)}


def compare_series(grid, expected, got_ts, got_vals, what, uncertain=()):
    """expected: values on `grid` (NaN = no point). got_*: what Grafana returned.
    Raises Mismatch on any difference in the set of windows with a point or in a value.
    Windows in `uncertain` (grid boundaries Grafana may or may not request) must match in
    value when returned, but may be absent."""
    grid = np.asarray(grid, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.float64)
    got_ts = np.asarray(got_ts, dtype=np.int64)
    got_vals = np.asarray(got_vals, dtype=np.float64)
    off_grid = np.setdiff1d(got_ts, grid)
    if len(off_grid):
        raise Mismatch(
            f"{what}: {len(off_grid)} returned points are not on the grid (first {int(off_grid[0])})"
        )
    uncertain = set(uncertain)
    want = set(grid[np.isfinite(expected)].tolist()) - uncertain
    have = set(got_ts.tolist()) - uncertain
    if want != have:
        missing = sorted(want - have)
        extra = sorted(have - want)
        raise Mismatch(
            f"{what}: windows with a point differ: {len(missing)} missing (first {missing[:3]}), "
            f"{len(extra)} unexpected (first {extra[:3]})"
        )
    if not len(got_ts):
        return 0
    pos = np.searchsorted(grid, got_ts)
    exp = expected[pos]
    if not np.isfinite(exp).all():
        i = int(np.flatnonzero(~np.isfinite(exp))[0])
        raise Mismatch(
            f"{what}: point at uncertain window {int(got_ts[i])} where the report has none"
        )
    scale = max(1.0, float(np.max(np.abs(exp))))
    bad = ~np.isclose(got_vals, exp, rtol=REL_TOL, atol=ABS_TOL * scale)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise Mismatch(
            f"{what}: value differs at {int(got_ts[i])}: grafana {got_vals[i]!r}, report {exp[i]!r}"
        )
    return int(len(got_ts))


# --------------------------------------------------------------------------- expected values


class Expected:
    """Report-library Rates per (target, source, counter), computed once."""

    def __init__(self, imp):
        self.imp = imp
        self.cache = {}

    def rates(self, t, source, counter):
        key = (t.target, source, counter)
        if key not in self.cache:
            self.cache[key] = self._rates(t, source, counter)
        return self.cache[key]

    def _rates(self, t, source, counter):
        R = self.imp.R
        stream = t.run.stream(t.netdev, source)
        if stream is None or stream not in t.imported_streams():
            return None
        return (
            R.Rates(stream, counter)
            if counter in self.imp.pick_counters(stream, t.counters_mode)
            else None
        )

    def imported(self, t):
        """(source, counter, Rates) of everything imported for a target."""
        out = []
        for s in t.imported_streams():
            for counter in self.imp.pick_counters(s, t.counters_mode):
                out.append((s.source, counter, self.rates(t, s.source, counter)))
        return out


def expected_on_grid(imp, r, spec, grid, step):
    agg = imp.R.window_aggregate(r, grid, step, spec.get("scale", 1.0))
    kind = spec["kind"]
    if kind in ("max", "raw"):
        return agg["max"]
    if kind in ("min", "mean", "low", "max_all", "mean_all"):
        return agg[kind]
    raise ValueError(kind)


def expected_range_mean(r, from_ms, to_ms, scale):
    """The overview table's average over a selected range: Σ increase / Σ duration of the
    valid intervals (both resolutions) whose end lies in (to - range, to]; NaN if none.
    """
    end = r.end_ms()
    m = r.valid & (end > to_ms - (to_ms - from_ms)) & (end <= to_ms)
    dt = float(r.dt[m].sum())
    return (
        float(r.dv[m].sum()) / dt * 1e9 * scale if m.any() and dt > 0 else float("nan")
    )


def expected_gaps(r, reason, grid, step):
    mask = (r.reset if reason == "reset" else r.clock).copy()
    end = r.end_ms()
    mask &= end >= 0
    out = np.full(len(grid), np.nan)
    if not mask.any():
        return out
    idx = -((grid[0] - end[mask]) // step)
    ok = (idx >= 0) & (idx < len(grid))
    vals = np.full(len(grid), -np.inf)
    np.maximum.at(vals, idx[ok], r.dt[mask][ok] / 1e9)
    return np.where(np.isfinite(vals), vals, out)


# --------------------------------------------------------------------------- Grafana


def _post_ds_query(grafana, query, from_ms, to_ms):
    body = json.dumps(
        {"queries": [query], "from": str(int(from_ms)), "to": str(int(to_ms))}
    ).encode()
    req = urllib.request.Request(
        grafana.rstrip("/") + "/api/ds/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())["results"]["A"]
    if res.get("error"):
        raise Mismatch(f"grafana query failed: {res['error']} ({query['expr']})")
    return res


def ds_query(grafana, expr, from_ms, to_ms, max_points, min_interval, ds=None):
    """Range query as the dashboard runs it (with the panel's deployed data source);
    returns [(labels, step_ms, ts, values)]."""
    q = {
        "refId": "A",
        "datasource": ds or {"type": "prometheus", "uid": "netmon-vm"},
        "expr": expr,
        "range": True,
        "instant": False,
        "intervalMs": 1,
        "maxDataPoints": int(max_points),
        "interval": min_interval,
    }
    frames = []
    for f in _post_ds_query(grafana, q, from_ms, to_ms).get("frames", []):
        fields = f["schema"]["fields"]
        vals = f["data"]["values"]
        labels = fields[1].get("labels", {}) if len(fields) > 1 else {}
        if not vals or not vals[0]:
            frames.append((labels, None, [], []))
            continue
        step = fields[0].get("config", {}).get("interval")
        frames.append(
            (labels, step, vals[0], [float("nan") if v is None else v for v in vals[1]])
        )
    return frames


def ds_instant(grafana, expr, at_ms):
    q = {
        "refId": "A",
        "datasource": {"type": "prometheus", "uid": "netmon-vm"},
        "expr": expr,
        "range": False,
        "instant": True,
    }
    res = _post_ds_query(grafana, q, int(at_ms) - 1000, at_ms)
    return [
        f["schema"]["fields"][1].get("labels", {})
        for f in res.get("frames", [])
        if len(f["schema"]["fields"]) > 1
    ]


def ds_instant_values(grafana, expr, at_ms, ds):
    """[(labels, value, ts)] of an instant query run through Grafana on data source `ds`."""
    q = {
        "refId": "A",
        "datasource": ds,
        "expr": expr,
        "range": False,
        "instant": True,
    }
    out = []
    for f in _post_ds_query(grafana, q, int(at_ms) - 1000, at_ms).get("frames", []):
        fields = f["schema"]["fields"]
        vals = f["data"]["values"]
        if len(fields) > 1 and vals and vals[1]:
            out.append((fields[1].get("labels", {}), vals[1][0], vals[0][0]))
    return out


def resolve_step_variable(grafana, dash, exp, at_ms):
    """Value of the deployed `min_step` variable as the dashboard computes it: its
    query_result(...) expression run through Grafana, formatted like Grafana's
    query_result ('{labels} value timestamp'), then the variable's extraction regex."""
    var = next(v for v in dash["templating"]["list"] if v["name"] == "min_step")
    q = var["query"]["query"] if isinstance(var["query"], dict) else var["query"]
    q = q.replace("$experiment", interpolate(dash, "experiment", exp))
    m = re.fullmatch(r"query_result\((.*)\)", q.strip(), re.S)
    if not m:
        raise Mismatch(f"min_step variable: unexpected query {q!r}")
    rows = ds_instant_values(grafana, m.group(1), at_ms, var.get("datasource"))
    regex = re.fullmatch(r"/(.*)/", var.get("regex", "")).group(1)
    for labels, value, ts in rows:
        name = labels.get("__name__", "")
        lab = ",".join(f'{k}="{v}"' for k, v in labels.items() if k != "__name__")
        text = f"{name}{{{lab}}} {value:g} {int(ts)}"
        mm = re.search(regex, text)
        if mm:
            return mm.group(1)
    raise Mismatch(f"min_step variable resolves to nothing at {at_ms} ({rows})")


def load_dashboards(args, gen):
    """Deployed dashboards (from a directory or Grafana) must equal the generated ones."""
    generated = {name: json.loads(text) for name, text in gen.render().items()}
    deployed = {}
    for name, d in generated.items():
        if args.dashboards:
            with open(os.path.join(args.dashboards, name)) as fh:
                deployed[name] = json.load(fh)
        else:
            url = args.grafana.rstrip("/") + "/api/dashboards/uid/" + d["uid"]
            with urllib.request.urlopen(url, timeout=60) as r:
                deployed[name] = json.loads(r.read())["dashboard"]
    for name, d in generated.items():
        g, dep = query_config(d), query_config(deployed[name])
        if g != dep:
            diff = [k for k in sorted(set(g) | set(dep)) if g.get(k) != dep.get(k)]
            raise Mismatch(
                f"{name}: deployed dashboard differs from ainic_netmon_dashboards.py in "
                f"{diff[:3]} (regenerate / re-provision it)"
            )
    return deployed


# transformations / fieldConfig: the tables' joins, renames and hidden link fields decide
# which number is shown next to which NIC.
QUERY_PANEL_KEYS = (
    "type", "interval", "maxDataPoints", "datasource", "repeat", "transformations", "fieldConfig"
)  # fmt: skip
QUERY_TARGET_KEYS = (
    "expr", "datasource", "instant", "range", "format", "interval", "netmon"
)  # fmt: skip


def _canon(x):
    """JSON value with numbers compared by value (Grafana re-serializes 1.0 as 1)."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list):
        return [_canon(v) for v in x]
    return {k: _canon(v) for k, v in x.items()}


def query_config(dash):
    """Everything in a dashboard that decides which query runs and how: per panel the
    data source, min interval, repeat; per target the expression and options; and the
    variables (their multi / All flags decide how values are escaped)."""
    out = {}
    for p in all_panels(dash):
        out[("panel", p.get("id"))] = tuple(
            json.dumps(_canon(p.get(k)), sort_keys=True) for k in QUERY_PANEL_KEYS
        )
        for t in p.get("targets", []):
            out[("target", p.get("id"), t.get("refId"))] = tuple(
                json.dumps(_canon(t.get(k)), sort_keys=True) for k in QUERY_TARGET_KEYS
            )
    for v in dash.get("templating", {}).get("list", []):
        out[("var", v["name"])] = tuple(
            json.dumps(_canon(v.get(k)), sort_keys=True)
            for k in (
                "type",
                "datasource",
                "query",
                "regex",
                "multi",
                "includeAll",
                "allValue",
            )
        )
    return out


def interpolate(dash, name, value):
    """A variable's value as Grafana's Prometheus data source puts it into a query
    (public/app/plugins/datasource/prometheus: interpolateQueryExpr). Single-valued
    variables only get backslashes and quotes escaped; multi-valued or "All" ones are
    regex-escaped with doubled backslashes, which must then be matched with =~."""
    var = next(v for v in dash["templating"]["list"] if v["name"] == name)
    if var.get("multi") or var.get("includeAll"):
        value = value.replace("\\", "\\\\\\\\")
        return re.sub(r"[$^*{}\[\]'+?.()|]", lambda m: "\\\\" + m.group(0), value)
    return value.replace("\\", "\\\\").replace("'", "\\\\'")


def all_panels(dash):
    """Every panel, including those inside (collapsed) rows."""
    for p in dash["panels"]:
        yield p
        yield from p.get("panels", [])


def data_queries(dash):
    for p in all_panels(dash):
        for t in p.get("targets", []):
            if "netmon" in t:
                yield p, t


def resolve_interval(panel, min_step):
    iv = panel.get("interval", "")
    return min_step if iv == "${min_step}ms" else iv


# --------------------------------------------------------------------------- command


def cmd_verify(args, imp):
    gen = _dashboards_module()
    runs, exp = imp.load_runs(args)
    targets = imp.targets_of(runs, exp)
    for t in targets:
        t.counters_mode = args.counters
    window = imp.run_window(runs)
    vm = imp.VM(args.url)
    vm.flush()
    expected = Expected(imp)
    checks = 0

    # 1. read-back
    want, summaries, infos, quality, min_step = imp.plan_summary(
        targets, args.counters, window
    )
    # The whole experiment: series of hosts not given here are "unexpected".
    got = vm.export_extent(
        imp.selector(
            "netmon_(rate|increase|interval_seconds|gap_seconds)", experiment=exp
        )
    )
    # Imported before the ethtool fallback (NICs without IPC have no ethtool series yet):
    # accepted as a whole, and only if the metadata is of that older format too (below).
    eth = {k for k in want if dict(k).get("source") == "ethtool"}
    legacy_no_ethtool = bool(eth) and not any(
        dict(k).get("source") == "ethtool" for k in got
    )
    if legacy_no_ethtool:
        for t in targets:
            t.legacy_no_ethtool = True
        want, summaries, infos, quality, min_step = imp.plan_summary(
            targets, args.counters, window
        )
    if getattr(args, "skip_readback", False):
        print("read-back: skipped")
    else:
        if got != want:
            missing = [k for k in want if got.get(k) != want[k]]
            extra = [k for k in got if k not in want]
            raise Mismatch(
                f"read-back: {len(missing)} series differ, {len(extra)} unexpected"
            )
        print(f"read-back: {len(want)} series match")
        checks += len(want)

    # 2. the experiment's hosts and metadata: exactly what the importer writes for these
    #    recordings (a host or target with metadata only would not show up in step 1)
    hosts = vm.label_values("host", imp.selector("netmon_.+", experiment=exp))
    if hosts != {r.host for r in runs}:
        raise Mismatch(
            f"experiment {exp!r} has hosts {sorted(hosts)}, these recordings are "
            f"{sorted(r.host for r in runs)}"
        )
    markers = vm.instant(
        f'last_over_time(netmon_import_info{{experiment="{imp.esc(exp)}",state="complete"}}[3650d])'
    )
    done = {x["metric"].get("host") for x in markers}
    formats = {x["metric"].get("format") for x in markers}
    if len(formats) > 1:
        raise Mismatch(
            f"hosts imported with different importer formats {sorted(map(str, formats))}"
        )
    old_importer = formats == {
        None
    }  # before IMPORT_FORMAT existed: older formats allowed
    if not old_importer and formats != {imp.IMPORT_FORMAT}:
        raise Mismatch(
            f"import format {formats}, this verifier knows {imp.IMPORT_FORMAT}"
        )
    if legacy_no_ethtool and not old_importer:
        raise Mismatch(
            "NICs without IPC have no ethtool series in a current-format import: the import "
            "is incomplete (re-run it with --replace)"
        )
    if done != hosts:
        raise Mismatch(f"import not complete for hosts {sorted(hosts - done)}")
    want_meta = {}
    for name, items in (
        ("netmon_summary", summaries),
        ("netmon_target_info", [(lab, 1.0) for lab in infos]),
        ("netmon_run_quality", quality),
        (
            "netmon_step_ms",
            [({"experiment": exp, "host": r.host}, float(min_step)) for r in runs],
        ),
    ):
        for lab, v in items:
            want_meta[imp.label_key(dict(lab, __name__=name))] = v
    res = vm.instant(
        'last_over_time({__name__=~"netmon_(summary|target_info|run_quality|step_ms)",'
        f'experiment="{imp.esc(exp)}"}}[3650d])'
    )
    have = {imp.label_key(x["metric"]): float(x["value"][1]) for x in res}
    legacy = False
    for drop in LEGACY_LABELS if old_importer and set(have) != set(want_meta) else ():
        # An earlier importer's metadata (without port_stats, or also without the NIC
        # display names) is accepted as a whole, never mixed with the current format.
        old = {
            imp.label_key({k: v for k, v in dict(key).items() if k not in drop}): v
            for key, v in want_meta.items()
        }
        if set(have) == set(old):
            want_meta, legacy = old, drop
            break
    if legacy_no_ethtool and not legacy:
        raise Mismatch(
            "NICs without IPC have no ethtool series, but the metadata is of the current "
            "format: the import is incomplete (re-run it with --replace)"
        )
    if set(have) != set(want_meta):
        missing = sorted(set(want_meta) - set(have))
        extra = sorted(set(have) - set(want_meta))
        raise Mismatch(
            f"metadata: {len(missing)} missing (first {missing[:1]}), "
            f"{len(extra)} unexpected (first {extra[:1]})"
        )
    for k, v in want_meta.items():
        if not math.isclose(have[k], v, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            raise Mismatch(f"{dict(k)}: VM {have[k]!r}, report {v!r}")
    print(
        f"metadata: hosts, completion markers and {len(want_meta)} summary / target / "
        "quality values equal the offline report's"
        + (
            " (imported by an earlier importer: re-import with --replace to get "
            + ("the NIC display names and " if "display_nic" in legacy else "")
            + "the ethtool port statistics of NICs without IPC)"
            if legacy
            else ""
        )
    )
    checks += len(want_meta)

    # 3. dashboards through Grafana
    dashboards = load_dashboards(args, gen)
    if window is None:
        print("no valid intervals in any recording: skipping the panel comparison")
    else:
        w0, w1 = int(window[0] // 1e6), int(window[1] // 1e6)
        mid = (w0 + w1) // 2
        ranges = [
            ("whole window, 1500 px", w0, w1, 1500),
            ("whole window, 300 px", w0, w1, 300),
            ("1 s around the middle, 1000 px", mid - 500, mid + 500, 1000),
            ("100 ms at the start, 1000 px", w0, w0 + 100, 1000),
            ("40 ms at the end, 400 px", w1 - 40, w1, 400),
            # The outermost windows of a range are uncertain (may be absent); these ranges
            # put the common window's edges well inside, so they are checked strictly.
            ("whole window + 1 s each side, 1500 px", w0 - 1000, w1 + 1000, 1500),
            ("start edge ± 50 ms, 1000 px", w0 - 50, w0 + 50, 1000),
            ("end edge ± 50 ms, 1000 px", w1 - 50, w1 + 50, 1000),
        ]
        # The panels' min interval comes from the deployed min_step variable: resolve it the
        # way the dashboard does, in the earliest and latest sub-windows.
        for name, dash in dashboards.items():
            for at in (w0 + 10, w1):
                got = resolve_step_variable(args.grafana, dash, exp, at)
                if got != str(min_step):
                    raise Mismatch(
                        f"{name}: min_step variable is {got!r} at {at}, expected {min_step}"
                    )
        ms = f"{min_step}ms"
        n_points = 0
        for t in targets:
            expected.cache.clear()  # one target's Rates alive at a time
            for name, dash in dashboards.items():
                for panel, tq in data_queries(dash):
                    spec = tq["netmon"]
                    expr = (
                        tq["expr"]
                        .replace("$experiment", interpolate(dash, "experiment", exp))
                        .replace("$target", interpolate(dash, "target", t.target))
                    )
                    if spec["kind"] == "range_mean":
                        n_points += check_range_mean(
                            args.grafana,
                            expected,
                            t,
                            spec,
                            expr,
                            ranges,
                            tq.get("datasource") or panel.get("datasource"),
                            f"{name} '{panel['title'][:40]}' {tq['refId']} {t.target}",
                        )
                        continue
                    for label, f, to, width in ranges:
                        min_iv = resolve_interval(panel, ms)
                        frames = ds_query(
                            args.grafana,
                            expr,
                            f,
                            to,
                            width,
                            min_iv,
                            tq.get("datasource") or panel.get("datasource"),
                        )
                        what = f"{name} '{panel['title'][:40]}' {tq['legendFormat']} {t.target} [{label}]"
                        n_points += check_query(
                            imp,
                            expected,
                            t,
                            spec,
                            frames,
                            (f, to),
                            no_frame_margin(f, to, width, min_iv),
                            what,
                        )
        print(
            f"panels: every query matches the offline report on the full grid ({n_points} points compared)"
        )
        checks += n_points

    # 3b. the tables' metadata queries (NIC names, short host, link rate) per target, also
    #     when no recording has a valid interval (then just after the recording start)
    meta_at = (
        w1
        if window is not None
        else min(int(r.meta.get("start", {}).get("realtime_ns", 0)) for r in runs)
        // 1_000_000
        + 10
    )
    checks += check_meta_queries(
        args.grafana,
        dashboards,
        exp,
        targets,
        infos,
        meta_at,
        bool(legacy and "display_nic" in legacy),
        old_port_stats=bool(legacy),
    )
    print("tables: NIC names, hosts and link rates equal the recordings' metadata")

    # 4. through Grafana, the targets (the dashboards' variable) are found when zoomed into
    #    the earliest and latest sub-windows, also when no recording has a valid interval
    if window is not None:
        zoom = (("earliest 10 ms", w0 + 10), ("latest 10 ms", w1))
    else:
        start = min(int(r.meta.get("start", {}).get("realtime_ns", 0)) for r in runs)
        zoom = (("recording start + 10 ms", start // 1_000_000 + 10),)
    for label, at in zoom:
        labs = ds_instant(
            args.grafana,
            f'last_over_time(netmon_target_info{{experiment="{imp.esc(exp)}"}}[3650d])',
            at,
        )
        found = sorted(x.get("target") for x in labs)
        if found != sorted(t.target for t in targets):
            raise Mismatch(
                f"targets found when zoomed to the {label}: {found}, "
                f"expected {sorted(t.target for t in targets)}"
            )
    print(
        "metadata: exactly the experiment's targets are found in its first and last sub-windows"
    )
    print(f"verify OK ({checks} checks)")
    return 0


def no_frame_margin(from_ms, to_ms, max_points, min_interval):
    """Upper bound of the step Grafana could have used for a query that returned no
    frame (so its step is unknown): the panel's min interval, or the automatic step
    rounded up generously (Grafana's "nice" rounding turns anything under 15 ms into
    10 ms and otherwise rounds by less than 2x). Windows within one such step of either end may legitimately not be
    requested (the last window ends at floor(to/step)*step <= to)."""
    m = re.fullmatch(r"(\d+)ms", min_interval or "")
    floor_ms = int(m.group(1)) if m else 0
    return max(floor_ms, 10, 2 * -(-(to_ms - from_ms) // max_points))


def check_meta_queries(
    grafana, dashboards, exp, targets, infos, at_ms, legacy, old_port_stats=False
):
    """Every instant metadata query of the dashboards (no "netmon" spec), run per target:
    target_info rows carry the planned names (legacy imports: display_nic falls back to the
    netdev), host_short is the host's first DNS label, the link rate is speed_mbps * 1e6.
    """
    plan = {lab["target"]: lab for lab in infos}
    n = 0
    for name, dash in dashboards.items():
        for p in all_panels(dash):
            for tq in p.get("targets", []):
                expr = tq.get("expr", "")
                if (
                    "netmon" in tq
                    or not tq.get("instant")
                    or "netmon_target_info" not in expr
                ):
                    continue
                for t in targets:
                    e = expr.replace(
                        "$experiment", interpolate(dash, "experiment", exp)
                    ).replace("$target", interpolate(dash, "target", t.target))
                    rows = ds_instant_values(
                        grafana, e, at_ms, tq.get("datasource") or p.get("datasource")
                    )
                    what = (
                        f"{name} '{p.get('title', '')[:40]}' {tq['refId']} {t.target}"
                    )
                    want = plan[t.target]
                    if any(lab.get("target") != t.target for lab, _, _ in rows):
                        raise Mismatch(
                            f"{what}: rows of other targets {[lab for lab, _, _ in rows]}"
                        )
                    if "label_value(" in expr:
                        speed = float(want.get("speed_mbps", -1))
                        exp_rows = [speed * 1e6] if speed > 0 else []
                        got = [float(v) for _, v, _ in rows]
                        if got != exp_rows:
                            raise Mismatch(
                                f"{what}: link rate {got}, expected {exp_rows}"
                            )
                        n += 1
                        continue
                    if len(rows) != 1:
                        raise Mismatch(f"{what}: {len(rows)} rows, expected 1")
                    lab = rows[0][0]
                    nic = t.netdev if legacy else want["display_nic"]
                    if lab.get("display_nic") != nic or lab.get("host") != t.host:
                        raise Mismatch(
                            f"{what}: NIC {lab.get('display_nic')!r} host {lab.get('host')!r}, expected {nic!r} {t.host!r}"
                        )
                    if not old_port_stats and lab.get("port_stats") != want.get(
                        "port_stats"
                    ):
                        raise Mismatch(
                            f"{what}: port_stats {lab.get('port_stats')!r}, "
                            f"expected {want.get('port_stats')!r}"
                        )
                    if (
                        "host_short" in expr
                        and lab.get("host_short") != t.host.split(".")[0]
                    ):
                        raise Mismatch(f"{what}: host_short {lab.get('host_short')!r}")
                    n += 1
    return n


def ds_instant_range(grafana, expr, from_ms, to_ms, ds):
    """Instant query over a selected time range as a table panel runs it (evaluated at
    `to`, with Grafana's range variables for [from, to]); [(labels, ts, value)]."""
    q = {"refId": "A", "datasource": ds, "expr": expr, "range": False, "instant": True}
    out = []
    for f in _post_ds_query(grafana, q, from_ms, to_ms).get("frames", []):
        fields = f["schema"]["fields"]
        vals = f["data"]["values"]
        if len(fields) > 1 and vals and vals[0]:
            out.append((fields[1].get("labels", {}), vals[0][0], vals[1][0]))
    return out


def check_range_mean(grafana, expected, t, spec, expr, ranges, ds, what):
    """The overview table's per-NIC average over each selected range equals the report's
    (expected_range_mean); no row when the report has no valid interval in the range."""
    r = expected.rates(t, spec["source"], spec["counter"])
    n = 0
    for label, f, to, _ in ranges:
        got = ds_instant_range(grafana, expr, f, to, ds)
        want = (
            expected_range_mean(r, f, to, spec["scale"])
            if r is not None
            else float("nan")
        )
        w = f"{what} [{label}]"
        if any(lab.get("target") != t.target for lab, _, _ in got) or len(got) > 1:
            raise Mismatch(f"{w}: unexpected rows {[lab for lab, _, _ in got]}")
        if not got:
            if np.isfinite(want):
                raise Mismatch(f"{w}: no value, the report has {want!r}")
            continue
        _, ts, v = got[0]
        if int(ts) != int(to):
            raise Mismatch(f"{w}: evaluated at {ts}, expected the range end {to}")
        v = float("nan") if v is None else float(v)
        if not np.isfinite(want) or not np.isclose(
            v, want, rtol=REL_TOL, atol=ABS_TOL * max(1.0, abs(want))
        ):
            raise Mismatch(f"{w}: grafana {v!r}, report {want!r}")
        n += 1
    return n


def check_query(imp, expected, t, spec, frames, rng, margin, what):
    from_ms, to_ms = rng
    lo, hi = from_ms + margin, to_ms - margin  # certainly requested when no frame
    kind = spec["kind"]
    frames = check_frames(imp, expected, t, spec, frames, what)
    steps = {f[1] for f in frames.values() if f[1]}
    if len(steps) > 1:
        raise Mismatch(f"{what}: frames with different steps {steps}")
    step = int(next(iter(steps))) if steps else None
    n = 0
    if kind == "gap":
        for source, counter, r in expected.imported(t):
            if step is None:
                if _gaps_in_range(r, spec["reason"], lo, hi):
                    raise Mismatch(
                        f"{what}: no frames, but {source}.{counter} has gaps in range"
                    )
                continue
            grid, unc = grafana_grid(from_ms, to_ms, step)
            exp = expected_gaps(r, spec["reason"], grid, step)
            f = frames.get((source, counter))
            n += compare_series(
                grid,
                exp,
                f[2] if f else [],
                f[3] if f else [],
                f"{what} {source}.{counter}",
                unc,
            )
        return n
    if kind == "gap_by_source":
        # max by (source, reason): the longest gap of any imported counter of the source
        by = {}
        for source, counter, r in expected.imported(t):
            for reason in ("reset", "clock"):
                by.setdefault((source, reason), []).append(r)
        for (source, reason), rs in by.items():
            if step is None:
                if any(_gaps_in_range(r, reason, lo, hi) for r in rs):
                    raise Mismatch(
                        f"{what}: no frames, but {source} has {reason} gaps in range"
                    )
                continue
            grid, unc = grafana_grid(from_ms, to_ms, step)
            with np.errstate(invalid="ignore"):
                exp = np.fmax.reduce([expected_gaps(r, reason, grid, step) for r in rs])
            f = frames.get((source, reason))
            n += compare_series(
                grid,
                exp,
                f[2] if f else [],
                f[3] if f else [],
                f"{what} {source} {reason}",
                unc,
            )
        return n
    if "counter_re" in spec:
        stream = t.run.stream(t.netdev, spec["source"])
        imported = (
            imp.pick_counters(stream, t.counters_mode)
            if stream is not None and stream in t.imported_streams()
            else []
        )
        counters = [c for c in imported if re.fullmatch(spec["counter_re"], c)]
    else:
        counters = [spec["counter"]]
    for counter in counters:
        r = expected.rates(t, spec["source"], counter)
        f = frames.get((spec["source"], counter))
        if r is None:
            continue  # check_frames already refused any frame for it
        if step is None:
            # No frame at all: the report must have nothing away from the ends either.
            grid = imp.R.epoch_grid(from_ms, to_ms, 1)
            exp = expected_on_grid(imp, r, spec, grid, 1)
            inner = (grid > lo) & (grid <= hi)
            if np.isfinite(exp[inner]).any():
                raise Mismatch(
                    f"{what}: Grafana returned nothing, the report has data in the range"
                )
            continue
        grid, unc = grafana_grid(from_ms, to_ms, step)
        exp = expected_on_grid(imp, r, spec, grid, step)
        n += compare_series(
            grid, exp, f[2] if f else [], f[3] if f else [], f"{what} {counter}", unc
        )
    return n


def check_frames(imp, expected, t, spec, frames, what):
    """Every returned frame must belong to this experiment / target / source and to a
    series the report expects, at most once, with strictly increasing timestamps.
    Returns {(source, counter): frame}."""
    if spec["kind"] == "gap":
        allowed = {(s, c) for s, c, _ in expected.imported(t)}
    elif spec["kind"] == "gap_by_source":
        allowed = {
            (s, reason)
            for s, _, _ in expected.imported(t)
            for reason in ("reset", "clock")
        }
    elif "counter_re" in spec:
        stream = t.run.stream(t.netdev, spec["source"])
        imported = (
            imp.pick_counters(stream, t.counters_mode)
            if stream is not None and stream in t.imported_streams()
            else []
        )
        allowed = {
            (spec["source"], c) for c in imported if re.fullmatch(spec["counter_re"], c)
        }
    else:
        r = expected.rates(t, spec["source"], spec["counter"])
        allowed = {(spec["source"], spec["counter"])} if r is not None else set()
    out = {}
    for f in frames:
        lab, ts = f[0], f[2]
        if not len(ts):
            continue
        if lab.get("experiment") != t.experiment or lab.get("target") != t.target:
            raise Mismatch(f"{what}: frame of another experiment / target: {lab}")
        if spec["kind"] == "gap" and lab.get("reason") != spec["reason"]:
            raise Mismatch(f"{what}: gap frame with reason {lab.get('reason')!r}")
        key = (
            (lab.get("source"), lab.get("reason"))
            if spec["kind"] == "gap_by_source"
            else (lab.get("source"), lab.get("counter"))
        )
        if key not in allowed:
            raise Mismatch(f"{what}: unexpected series {key} (labels {lab})")
        if key in out:
            raise Mismatch(f"{what}: two frames for {key}")
        if len(ts) > 1 and not np.all(np.diff(np.asarray(ts, dtype=np.int64)) > 0):
            raise Mismatch(f"{what}: {key} has repeated or unordered timestamps")
        out[key] = f
    return out


def _gaps_in_range(r, reason, from_ms, to_ms):
    m = r.reset if reason == "reset" else r.clock
    e = r.end_ms()[m]
    return bool(((e > from_ms) & (e <= to_ms)).any())


def _dashboards_module():
    try:
        from . import _netmon_dashboards as G
    except ImportError:
        import ainic_netmon_dashboards as G
    return G


def main_verify(args, imp):
    try:
        return cmd_verify(args, imp)
    except Mismatch as e:
        print(f"verify FAILED: {e}")
        return 1
