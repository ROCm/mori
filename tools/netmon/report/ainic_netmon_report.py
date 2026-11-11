#!/usr/bin/env python3
"""ainic-netmon-report: offline report (HTML) and CSV export for ainic-netmon recordings.

Reads recording directories written by `ainic-netmon record` (format_version 2):
  meta.json + <netdev>.<source>.bin, records of little-endian u64
  [t0_mono_ns, t1_mono_ns, real_ns, real_err_ns, values...].

Rules (tools/netmon README, design §7.4 / §9):
  * Sample time (monotonic): t0 + f*rtt, f = stream "snapshot_fraction" (rdma 0.70, ipc 0.25).
    A request slower than 1.5x the median rtt has no trustworthy snapshot time: the wait can be
    before the snapshot (rdma stalls, ipc ~136 ms stalls) or after it (ipc requests of a few ms
    snapshot at the start). Such samples are skipped for rates: the rate is taken between the
    surrounding precise samples (counters are cumulative, so no bytes are lost; the interval
    is longer and may become low resolution). With 3x instead of 1.5x, 2 ms rates and even
    10 ms peaks exceeded the 400G line rate on steady traffic (n06-21, 2026-11-11).
  * Wall clock: anchor offset = real - (t0 - real_err/2), trusted only if real_err <= 50 us.
    A change of more than 1 ms between trusted anchors is a clock step: samples are split into
    segments there, and samples between the two anchors around a step are "alignment
    uncertain" (no wall time).
  * Rates: consecutive samples only, counter compared before subtracting (a decrease is a
    reset: no rate, the interval counts as uncovered). An interval longer than 3x the baseline
    (max(configured interval, median observed spacing)) is "low resolution": it only gives the
    interval average and is drawn and counted separately.
  * Stats: average = sum of valid increments / sum of their durations (low-res included);
    peak (10 ms), p99 and time above a threshold use normal-resolution intervals only,
    time-weighted, and come with their coverage.

Only numpy (and plotly for `report`) are needed; nothing here imports mori.
"""

import argparse
import datetime as _dt
import html
import json
import os
import sys

import numpy as np

HEADER = 4
SLOW_FACTOR = 1.5  # n06-21: skips ~2% of rdma and <1% of ipc samples
ANCHOR_MAX_ERR_NS = 50_000
CLOCK_STEP_NS = 1_000_000
LOWRES_FACTOR = 3.0
PEAK_WINDOW_NS = 10_000_000

RDMA_TX, RDMA_RX = "tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes"
IPC_TX, IPC_RX = "octets_tx_ok", "octets_rx_ok"
CONGESTION = [
    "rx_rdma_cnp_pkts",
    "tx_rdma_cnp_pkts",
    "rx_rdma_ecn_pkts",
    "tx_rdma_retx_pkts",
    "tx_rdma_ack_timeout",
]


# --------------------------------------------------------------------------- loading


class Stream:
    def __init__(self, run_dir, spec):
        self.file = spec["file"]
        self.netdev = spec["netdev"]
        self.source = spec["source"]
        self.names = list(spec["names"])
        self.width = int(spec["record_words"])
        self.interval_ns = int(spec["interval_ns"])
        self.f = float(spec.get("snapshot_fraction", 0.5))
        self.ipc_unverified = bool(spec.get("ipc_unverified", False))
        path = os.path.join(run_dir, self.file)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        n = size // (8 * self.width)  # drop a partial last record (killed mid-write)
        self.partial_bytes = size - n * 8 * self.width
        if n:
            self.rec = np.memmap(path, dtype="<u8", mode="r", shape=(n, self.width))
        else:
            self.rec = np.zeros((0, self.width), dtype="<u8")
        self._times = None

    def col(self, name):
        try:
            return self.rec[:, HEADER + self.names.index(name)]
        except ValueError:
            return None

    def times(self):
        """(t_mono ns float64, wall ns float64 with NaN where uncertain, slow mask)."""
        if self._times is None:
            self._times = sample_times(self.rec, self.f)
        return self._times[:3]

    def segments(self):
        """Wall-clock segment id per sample (-1: alignment uncertain)."""
        self.times()
        return self._times[3]


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        if self.meta.get("format_version") != 2:
            raise SystemExit(
                f"{run_dir}: unsupported format_version {self.meta.get('format_version')}"
            )
        self.host = (
            self.meta.get("host_id")
            or self.meta.get("hostname")
            or os.path.basename(run_dir)
        )
        self.status = self.meta.get("status", "?")
        self.streams = [Stream(run_dir, s) for s in self.meta["streams"]]
        self.speed = {
            n["netdev"]: n.get("speed_mbps", -1) for n in self.meta.get("nics", [])
        }
        self.role = (self.meta.get("tags") or {}).get("role")

    def stream(self, netdev, source):
        for s in self.streams:
            if s.netdev == netdev and s.source == source:
                return s
        return None

    def netdevs(self):
        out = []
        for s in self.streams:
            if s.netdev not in out:
                out.append(s.netdev)
        return out


# --------------------------------------------------------------------------- timestamps


def sample_times(rec, f):
    if len(rec) == 0:
        e = np.zeros(0)
        return e, e, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    t0 = rec[:, 0].astype(np.int64)
    t1 = rec[:, 1].astype(np.int64)
    rtt = (t1 - t0).astype(np.float64)
    med = float(np.median(rtt))
    slow = rtt > SLOW_FACTOR * med if med > 0 else np.zeros(len(rtt), dtype=bool)
    t = t0 + f * rtt  # for slow samples this is only a nominal time (see module doc)
    wall, seg = wall_times(rec, t)
    return t, wall, slow, seg


def wall_times(rec, t_mono):
    """Map monotonic sample times to wall clock with clock-step segmentation.

    Returns (wall ns, segment id). Record i's anchor is read just *before* its query, and the
    snapshot happens inside the query, so sample i lies between its own anchor and the next
    trusted anchor after it. If those two anchors are in different segments (a clock step in
    between), the sample's wall time is uncertain: wall = NaN, segment = -1.
    """
    n = len(rec)
    t0 = rec[:, 0].astype(np.float64)
    real = rec[:, 2].astype(np.float64)
    err = rec[:, 3].astype(np.float64)
    off = real - (t0 - err / 2.0)
    good = np.flatnonzero(err <= ANCHOR_MAX_ERR_NS)
    if len(good) == 0:
        return np.full(n, np.nan), np.full(n, -1, dtype=np.int64)
    steps = np.abs(np.diff(off[good])) > CLOCK_STEP_NS
    seg_of_good = np.concatenate([[0], np.cumsum(steps)]).astype(np.int64)
    idx = np.arange(n)
    before = np.searchsorted(good, idx, side="right") - 1  # last trusted anchor <= i
    after = before + 1  # first trusted anchor > i
    has_before = before >= 0
    has_after = after < len(good)
    seg_before = seg_of_good[np.clip(before, 0, len(good) - 1)]
    seg_after = seg_of_good[np.clip(after, 0, len(good) - 1)]
    seg = np.where(has_before, seg_before, seg_after)
    seg = np.where(has_before & has_after & (seg_before != seg_after), -1, seg)
    nseg = int(seg_of_good[-1]) + 1
    seg_off = np.array([np.median(off[good][seg_of_good == k]) for k in range(nseg)])
    wall = np.where(seg >= 0, t_mono + seg_off[np.clip(seg, 0, None)], np.nan)
    return wall, seg


# --------------------------------------------------------------------------- rates and stats


class Rates:
    """Per-interval rates of one counter. Arrays are per interval (len = samples - 1)."""

    def __init__(self, stream, name):
        t, wall, slow = stream.times()
        seg = stream.segments()
        v = stream.col(name)
        self.name = name
        resets_between = None
        if v is not None:
            v = np.asarray(v)
            # Counter resets are found on the full series first, so dropping an imprecise
            # sample can never bridge over a reset (e.g. 1000 -> [0] -> 1200 is not +200).
            dec = np.concatenate([[0], np.cumsum(v[1:] < v[:-1])])
            keep = ~slow  # imprecise (slow) samples are skipped, see module doc
            t, wall, seg, v, dec = t[keep], wall[keep], seg[keep], v[keep], dec[keep]
            resets_between = np.diff(dec) > 0
        n = 0 if v is None else len(v)
        if n < 2:
            z = np.zeros(0)
            self.t_start = self.t_end = self.w_start = self.w_end = self.rate = (
                self.dv
            ) = self.dt = z
            self.valid = self.lowres = self.local_valid = np.zeros(0, dtype=bool)
            return
        later, earlier = v[1:], v[:-1]
        ok = (later >= earlier) & ~resets_between  # compare before subtracting (u64)
        dv = np.where(ok, later - np.where(ok, earlier, later), 0).astype(np.float64)
        dt = np.diff(t)
        valid = ok & (dt > 0)
        dt_ok = dt[valid]
        base = max(
            float(stream.interval_ns), float(np.median(dt_ok)) if len(dt_ok) else 0.0
        )
        self.t_start, self.t_end = t[:-1], t[1:]
        self.w_start, self.w_end = wall[:-1], wall[1:]
        self.dv, self.dt = dv, dt
        # Local validity (monotonic rate is meaningful) vs wall-clock validity (both ends in
        # the same clock segment): display and windowed stats need both.
        self.local_valid = valid
        same_seg = (seg[:-1] == seg[1:]) & (seg[:-1] >= 0)
        self.valid = (
            valid & same_seg & np.isfinite(self.w_start) & np.isfinite(self.w_end)
        )
        self.lowres = (
            self.valid & (dt > LOWRES_FACTOR * base)
            if base > 0
            else np.zeros_like(valid)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rate = np.where(self.valid, dv / dt * 1e9, np.nan)  # per second

    def clip(self, w0, w1):
        """Mask of intervals lying inside [w0, w1] (wall ns)."""
        return self.valid & (self.w_start >= w0) & (self.w_end <= w1)


def weighted_quantile(values, weights, q):
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(cw, q), len(v) - 1)])


def series_stats(r, w0, w1, scale, threshold=None):
    """Stats of a Rates object in the window; values multiplied by `scale`."""
    m = r.clip(w0, w1)
    normal = m & ~r.lowres
    span = max(w1 - w0, 1.0)
    out = {
        "coverage": float(r.dt[normal].sum() / span) if normal.any() else 0.0,
        "lowres_n": int((m & r.lowres).sum()),
        "lowres_s": float(r.dt[m & r.lowres].sum() / 1e9),
    }
    out["avg"] = (
        float(r.dv[m].sum() / r.dt[m].sum() * 1e9 * scale)
        if r.dt[m].sum() > 0
        else float("nan")
    )
    vals = r.rate[normal] * scale
    wts = r.dt[normal]
    out["p99"] = weighted_quantile(vals, wts, 0.99)
    out["above"] = (
        float(wts[vals > threshold].sum() / wts.sum())
        if threshold and wts.sum() > 0
        else float("nan")
    )
    out["peak10"] = peak_over_window(r, normal, scale)
    return out


def peak_over_window(r, mask, scale):
    """Max rate over 10 ms windows lying entirely in normal-resolution coverage.

    Only intervals no longer than the window can resolve 10 ms: longer ones cut the runs (an
    interval average is not a 10 ms rate) without discarding the usable data around them.
    Windows step by half a window and always include the one ending at the run's end.
    """
    idx = np.flatnonzero(mask & (r.dt <= PEAK_WINDOW_NS))
    if len(idx) == 0:
        return float("nan")
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(idx)]])
    best = float("nan")
    for a, b in zip(starts, ends):
        ii = idx[a:b]
        t = np.concatenate([[r.t_start[ii[0]]], r.t_end[ii]])
        c = np.concatenate([[0.0], np.cumsum(r.dv[ii])])
        last = t[-1] - PEAK_WINDOW_NS
        if last < t[0]:
            continue
        g = np.unique(np.append(np.arange(t[0], last, PEAK_WINDOW_NS / 2), last))
        rr = (
            (np.interp(g + PEAK_WINDOW_NS, t, c) - np.interp(g, t, c))
            / PEAK_WINDOW_NS
            * 1e9
            * scale
        )
        m = float(rr.max())
        best = m if np.isnan(best) else max(best, m)
    return best


# --------------------------------------------------------------------------- report


def overview_series(r, w0, w1, bins, scale):
    """Envelope (min, max) and mean per time bin for normal-resolution intervals, plus the
    low-resolution intervals as flat segments."""
    m = r.clip(w0, w1)
    normal = m & ~r.lowres
    edges = np.linspace(w0, w1, bins + 1)
    x = r.w_end[normal]
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    vals = r.rate[normal] * scale
    cnt = np.bincount(b, minlength=bins)
    dv = np.bincount(b, weights=r.dv[normal], minlength=bins)
    dt = np.bincount(b, weights=r.dt[normal], minlength=bins)
    mn = np.full(bins, np.inf)
    mx = np.full(bins, -np.inf)
    np.minimum.at(mn, b, vals)
    np.maximum.at(mx, b, vals)
    empty = cnt == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(empty, np.nan, dv / dt * 1e9 * scale)
    mn[empty] = np.nan
    mx[empty] = np.nan
    centers = (edges[:-1] + edges[1:]) / 2
    lo = m & r.lowres
    return centers, mn, mx, mean, (r.w_start[lo], r.w_end[lo], r.rate[lo] * scale)


def detail_series(r, w0, w1, scale):
    """Raw rate points in the window, with NaN breaks at gaps (invalid / low-res)."""
    m = r.clip(w0, w1)
    normal = m & ~r.lowres
    idx = np.flatnonzero(normal)
    if len(idx) == 0:
        return np.zeros(0), np.zeros(0), (np.zeros(0), np.zeros(0), np.zeros(0))
    x = r.w_end[idx]
    y = r.rate[idx] * scale
    # Break the line at gaps: repeat the next x with a NaN y (x itself stays a valid time).
    gap = np.flatnonzero(np.diff(idx) != 1)
    x = np.insert(x, gap + 1, x[gap + 1])
    y = np.insert(y, gap + 1, np.nan)
    lo = m & r.lowres
    return x, y, (r.w_start[lo], r.w_end[lo], r.rate[lo] * scale)


class Fig:
    """Collects traces per subplot row for one NIC; renders with plotly."""

    def __init__(self, title):
        self.title = title
        self.rows = []  # (row title, [traces])

    def row(self, title):
        self.rows.append((title, []))
        return self.rows[-1][1]


def add_series(traces, r, w0, w1, scale, label, color, mode, bins):
    import plotly.graph_objects as go

    if mode == "detail":
        x, y, lo = detail_series(r, w0, w1, scale)
        traces.append(
            go.Scattergl(
                x=_dtn(x),
                y=y,
                name=label,
                mode="lines",
                line=dict(color=color, width=1.5),
                connectgaps=False,
                legendgroup=label,
            )
        )
    else:
        c, mn, mx, mean, lo = overview_series(r, w0, w1, bins, scale)
        xs = _dtn(c)
        traces.append(
            go.Scattergl(
                x=xs,
                y=mx,
                name=label + " max",
                mode="lines",
                line=dict(color=color, width=0.5),
                connectgaps=False,
                legendgroup=label,
                showlegend=False,
            )
        )
        traces.append(
            go.Scattergl(
                x=xs,
                y=mn,
                name=label + " min",
                mode="lines",
                line=dict(color=color, width=0.5),
                fill="tonexty",
                connectgaps=False,
                legendgroup=label,
                showlegend=False,
            )
        )
        traces.append(
            go.Scattergl(
                x=xs,
                y=mean,
                name=label,
                mode="lines",
                line=dict(color=color, width=1.5),
                connectgaps=False,
                legendgroup=label,
            )
        )
    s, e, v = lo
    if len(s):
        xs, ys = [], []
        for a, b, val in zip(s, e, v):
            xs += [a, b, b]
            ys += [val, val, np.nan]
        traces.append(
            go.Scattergl(
                x=_dtn(np.array(xs)),
                y=ys,
                name=label + " (low-res avg)",
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                connectgaps=False,
                legendgroup=label,
                showlegend=False,
            )
        )


def _dtn(ns):
    """Finite ns timestamps (float) -> datetime64[ns] array for a plotly date axis."""
    return (
        np.round(np.asarray(ns, dtype=np.float64))
        .astype(np.int64)
        .astype("datetime64[ns]")
    )


# Categorical palette in fixed order (validated: lightness, chroma, CVD and normal-vision
# separation pass on the light surface; slots 3-5 are below 3:1 contrast, so every series is
# also named in the legend and hover, and the summary table is the text view).
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
TX, RX = (
    PALETTE[0],
    PALETTE[1],
)  # colour follows the entity: tx is always blue, rx orange


def build_nic_fig(run, netdev, role, w0, w1, mode, bins):
    fig = Fig(f"{run.host} / {netdev}" + (f" (role {role})" if role else ""))
    gbps = 8 / 1e9
    rd = run.stream(netdev, "rdma")
    ip = run.stream(netdev, "ipc")
    if rd is not None:
        tr = fig.row("RDMA payload bytes, Gb/s")
        add_series(tr, Rates(rd, RDMA_TX), w0, w1, gbps, "tx", TX, mode, bins)
        add_series(tr, Rates(rd, RDMA_RX), w0, w1, gbps, "rx", RX, mode, bins)
        if role in ("P", "D"):
            tr = fig.row("Direction view, Gb/s (assumes EP send ≈ recv)")
            ep_only = RDMA_TX if role == "D" else RDMA_RX
            add_series(
                tr,
                Rates(rd, ep_only),
                w0,
                w1,
                gbps,
                f"EP-only ({'tx' if role == 'D' else 'rx'})",
                TX if role == "D" else RX,
                mode,
                bins,
            )
            add_series(
                tr,
                DiffRates(rd, role),
                w0,
                w1,
                gbps,
                ("rx − tx" if role == "D" else "tx − rx") + " (KV estimate)",
                PALETTE[2],
                mode,
                bins,
            )
        tr = fig.row("Congestion and retransmission, per second")
        for i, name in enumerate(CONGESTION):
            if rd.col(name) is not None:
                add_series(
                    tr, Rates(rd, name), w0, w1, 1.0, name, PALETTE[i], mode, bins
                )
    if ip is not None:
        tr = fig.row("Port wire bytes (IPC), Gb/s, all traffic incl. ACKs")
        add_series(tr, Rates(ip, IPC_TX), w0, w1, gbps, "wire tx", TX, mode, bins)
        add_series(tr, Rates(ip, IPC_RX), w0, w1, gbps, "wire rx", RX, mode, bins)
        active = [
            n
            for n in ip.names
            if (
                n.endswith("_1us_count")
                and ("pripause" in n or n == "rx_pause_1us_count")
            )
            and len(ip.col(n)) > 1
            and int(ip.col(n)[-1]) != int(ip.col(n)[0])
        ]
        if active:
            more = len(active) - len(PALETTE)
            tr = fig.row(
                "PFC pause counter increments /s (semantics unvalidated)"
                + (f"; {more} more in csv" if more > 0 else "")
            )
            for k, name in enumerate(
                active[: len(PALETTE)]
            ):  # never more hues than the palette
                add_series(
                    tr, Rates(ip, name), w0, w1, 1.0, name, PALETTE[k], mode, bins
                )
    return fig


class DiffRates(Rates):
    """Signed direction difference of RDMA bytes: D: rx - tx, P: tx - rx (same samples)."""

    def __init__(self, stream, role):
        a = Rates(stream, RDMA_RX if role == "D" else RDMA_TX)
        b = Rates(stream, RDMA_TX if role == "D" else RDMA_RX)
        self.__dict__.update(a.__dict__)
        self.name = "diff"
        self.valid = a.valid & b.valid
        self.local_valid = a.local_valid & b.local_valid
        self.lowres = a.lowres | b.lowres
        self.dv = np.where(self.valid, a.dv - b.dv, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rate = np.where(self.valid, self.dv / self.dt * 1e9, np.nan)


def to_plotly(fig):
    """One NIC: stacked panels sharing the time axis, one y-scale per panel."""
    from plotly.subplots import make_subplots

    n = len(fig.rows)
    sp = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.32 / n,
        subplot_titles=[t for t, _ in fig.rows],
    )
    legends = {}
    for i, (_, traces) in enumerate(fig.rows):
        key = "legend" if i == 0 else f"legend{i + 1}"
        for tr in traces:
            tr.legend = key
            sp.add_trace(tr, row=i + 1, col=1)
        top = sp.layout["yaxis" if i == 0 else f"yaxis{i + 1}"].domain[1]
        # Each panel has its own legend, right-aligned on the panel's title line.
        legends[key] = dict(
            orientation="h",
            x=1.0,
            xanchor="right",
            y=top,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
    sp.update_layout(
        title=dict(text=fig.title, x=0.01, xanchor="left"),
        height=250 * n + 90,
        hovermode="x unified",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#3d3d3a", size=12),
        margin=dict(l=60, r=20, t=70, b=40),
        **legends,
    )
    sp.update_annotations(font=dict(size=12, color="#5f5e5a"), x=0, xanchor="left")
    sp.update_xaxes(tickformat="%H:%M:%S.%L", gridcolor="#e8e7e2", zeroline=False)
    sp.update_yaxes(gridcolor="#e8e7e2", zeroline=False, rangemode="tozero")
    return sp


def render_html(runs, figs, summary_rows, notes, out_path, title):
    parts = [
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
        "<style>body{font-family:sans-serif;margin:16px} table{border-collapse:collapse;font-size:12px}"
        " td,th{border:1px solid #ccc;padding:2px 6px;text-align:right} th{background:#f3f3f3}"
        " td.l{text-align:left} .note{color:#a60}</style></head><body>",
        f"<h2>{html.escape(title)}</h2>",
    ]
    for n in notes:
        parts.append(f"<p class='note'>{html.escape(n)}</p>")
    parts.append(
        "<table><tr><th>host</th><th>netdev</th><th>series</th><th>avg</th><th>peak 10 ms</th>"
        "<th>p99</th><th>&gt;80% line</th><th>coverage</th><th>low-res (n / s)</th>"
        "<th>samples / missed / dropped / errors</th></tr>"
    )
    for r in summary_rows:
        parts.append(
            "<tr>"
            + "".join(
                (
                    f"<td class='l'>{html.escape(str(c))}</td>"
                    if i < 3
                    else f"<td>{html.escape(str(c))}</td>"
                )
                for i, c in enumerate(r)
            )
            + "</tr>"
        )
    parts.append("</table>")
    first = True
    for fig in figs:
        if not fig.rows:
            continue
        parts.append(to_plotly(fig).to_html(full_html=False, include_plotlyjs=first))
        first = False
    parts.append("</body></html>")
    data = "\n".join(parts)
    with open(out_path, "w") as fh:
        fh.write(data)
    return len(data)


def parse_window(spec, w0, w1):
    """START,END as seconds from the common start (e.g. 12.5,13.0) or ISO times (UTC)."""
    a, b = spec.split(",")

    def one(x):
        x = x.strip()
        try:
            return w0 + float(x) * 1e9
        except ValueError:
            t = _dt.datetime.fromisoformat(x)
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.timezone.utc)
            return t.timestamp() * 1e9

    s, e = one(a), one(b)
    if not (w0 <= s < e <= w1):
        raise SystemExit(f"--window {spec} is outside the common recording window")
    return s, e


def plotted_streams(runs, roles, nics):
    """(stream, number of curves drawn from it) for everything the report will plot."""
    out = []
    for r in runs:
        role = roles.get(r.host, r.role)
        for s in r.streams:
            if (
                (nics and s.netdev not in nics)
                or s.source not in ("rdma", "ipc")
                or not len(s.rec)
            ):
                continue
            if s.source == "rdma":
                n = 2 + len(CONGESTION) + (2 if role in ("P", "D") else 0)
            else:
                n = 2 + len(PALETTE)  # wire tx/rx + at most one palette of PFC counters
            out.append((s, n))
    return out


def cmd_report(args):
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise SystemExit("plotly is required for `report`: pip install plotly")
    runs = [Run(d) for d in args.run_dirs]
    roles = {}
    for item in args.role or []:
        host, _, role = item.partition("=")
        if role not in ("P", "D"):
            raise SystemExit(f"--role {item}: role must be P or D")
        roles[host] = role
    notes = []
    starts, ends = [], []
    for r in runs:
        # Window of one recording: span of its plotted (rdma/ipc) streams. ethtool samples once
        # per second and is not plotted, so it must not shrink the window.
        ws = [
            s.times()[1]
            for s in r.streams
            if s.source in ("rdma", "ipc") and len(s.rec) > 1
        ]
        ws = [w[np.isfinite(w)] for w in ws]
        ws = [w for w in ws if len(w) > 1]
        if not ws:
            raise SystemExit(f"{r.dir}: no rdma/ipc samples")
        starts.append(min(w[0] for w in ws))
        ends.append(max(w[-1] for w in ws))
        if r.status not in ("complete", "interrupted"):
            notes.append(
                f"{r.host}: recording status is '{r.status}' (not finished normally); data up to the last whole record is shown."
            )
        if r.meta.get("abandoned"):
            notes.append(
                f"{r.host}: streams abandoned at stop: {', '.join(r.meta['abandoned'])}"
            )
        for d in r.meta.get("degraded", []):
            notes.append(
                f"{r.host}: {d['netdev']}.{d['source']} not recorded: {d['reason']}"
            )
        for s in r.streams:
            if s.ipc_unverified:
                notes.append(f"{r.host}: {s.file} recorded with --ipc-unverified")
            unc = int(np.sum(~np.isfinite(s.times()[1])))
            if unc:
                notes.append(
                    f"{r.host}: {s.file}: {unc} samples have uncertain wall-clock alignment (clock step)"
                )
    w0, w1 = max(starts), min(ends)
    if w1 <= w0:
        raise SystemExit("the recordings do not overlap in time")
    mode = "overview"
    if args.window:
        w0, w1 = parse_window(args.window, w0, w1)
        mode = "detail"
    figs, rows = [], []
    plotted = plotted_streams(runs, roles, args.nics)
    nseries = sum(n for _, n in plotted)
    bins = args.bins
    if mode == "overview":
        # Budget: ~3 traces x bins points per series, ~25 bytes per point in the HTML.
        max_bins = int(args.max_mb * 1e6 / max(nseries, 1) / 3 / 25)
        if bins > max_bins:
            notes.append(
                f"overview reduced to {max_bins} bins per series to stay within {args.max_mb} MB"
            )
            bins = max(max_bins, 50)
    else:
        npts = 0
        for s, ncurves in plotted:  # same NIC filter and curve count as the plots
            w = s.times()[1]
            npts += int(np.sum((w >= w0) & (w <= w1))) * ncurves
        if npts > args.max_points:
            raise SystemExit(
                f"--window selects ~{npts} points (limit {args.max_points}); choose a smaller window or --nics"
            )
    for r in runs:
        role = roles.get(r.host, r.role)
        st = r.meta.get("run_stats") or {}
        for nd in r.netdevs():
            if args.nics and nd not in args.nics:
                continue
            figs.append(build_nic_fig(r, nd, role, w0, w1, mode, bins))
            line = r.speed.get(nd, -1)
            thr = line * 1e6 * 0.8 if line and line > 0 else None
            for src, keys, label in (
                ("rdma", (RDMA_TX, RDMA_RX), ("rdma tx", "rdma rx")),
                ("ipc", (IPC_TX, IPC_RX), ("wire tx", "wire rx")),
            ):
                s = r.stream(nd, src)
                if s is None:
                    continue
                q = st.get(s.file, {})
                qual = f"{q.get('samples', '?')} / {q.get('missed_slots', '?')} / {q.get('dropped', '?')} / {q.get('errors', '?')}"
                for k, lab in zip(keys, label):
                    x = series_stats(
                        Rates(s, k), w0, w1, 8 / 1e9, thr / 1e9 if thr else None
                    )
                    rows.append(
                        [
                            r.host,
                            nd,
                            lab,
                            fmt(x["avg"]),
                            fmt(x["peak10"]),
                            fmt(x["p99"]),
                            pct(x["above"]),
                            pct(x["coverage"]),
                            f"{x['lowres_n']} / {x['lowres_s']:.2f}",
                            qual,
                        ]
                    )
    span = (w1 - w0) / 1e9
    title = (
        f"ainic-netmon report: {', '.join(r.host for r in runs)} - "
        f"{_dt.datetime.fromtimestamp(w0 / 1e9, _dt.timezone.utc):%Y-%m-%d %H:%M:%S.%f} UTC, "
        f"{span:.3f} s, {mode}"
    )
    notes.insert(
        0,
        "Rates in Gb/s from RDMA payload bytes unless noted. Peak/p99/'>80% line' use "
        "normal-resolution intervals only (time-weighted); 'coverage' is their share of the "
        "window. Dashed lines are low-resolution intervals (interval averages, e.g. firmware "
        "stalls); the shaded band in the overview is the min-max envelope per time bin, "
        "the solid line its average. All times UTC.",
    )
    size = render_html(runs, figs, rows, notes, args.output, title)
    print(
        f"wrote {args.output} ({size / 1e6:.1f} MB, {len(figs)} NIC panels, window {span:.3f} s, {mode})"
    )
    print_table(rows)


def fmt(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.1f}"


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def print_table(rows):
    hdr = [
        "host",
        "netdev",
        "series",
        "avg",
        "peak10ms",
        "p99",
        ">80%",
        "coverage",
        "lowres n/s",
        "samp/miss/drop/err",
    ]
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


# --------------------------------------------------------------------------- csv


def cmd_csv(args):
    run = Run(args.run_dir)
    out = sys.stdout
    for s in run.streams:
        if args.nic and s.netdev != args.nic:
            continue
        if args.source and s.source != args.source:
            continue
        names = args.counters.split(",") if args.counters else s.names
        cols = [n for n in names if n in s.names]
        if not cols:
            continue
        t, wall, slow = s.times()
        out.write(f"# {run.host} {s.netdev}.{s.source}\n")
        if args.rate:
            rs = [Rates(s, c) for c in cols]
            out.write(
                "t_start_wall_ns,t_end_wall_ns,low_res,"
                + ",".join(c + "_per_s" for c in cols)
                + "\n"
            )
            for i in range(len(rs[0].rate)):
                if not rs[0].valid[i]:
                    continue
                vals = ",".join(
                    "" if not np.isfinite(r.rate[i]) else f"{r.rate[i]:.3f}" for r in rs
                )
                out.write(
                    f"{rs[0].w_start[i]:.0f},{rs[0].w_end[i]:.0f},{int(rs[0].lowres[i])},{vals}\n"
                )
        else:
            idx = [s.names.index(c) for c in cols]
            out.write("t_mono_ns,t_wall_ns,rtt_us,slow," + ",".join(cols) + "\n")
            for i in range(len(s.rec)):
                rec = s.rec[i]
                w = "" if not np.isfinite(wall[i]) else f"{wall[i]:.0f}"
                rtt = (int(rec[1]) - int(rec[0])) / 1e3
                out.write(
                    f"{t[i]:.0f},{w},{rtt:.1f},{int(slow[i])},"
                    + ",".join(str(int(rec[HEADER + j])) for j in idx)
                    + "\n"
                )


def main(argv=None):
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet exit when piped into head
    p = argparse.ArgumentParser(
        prog="ainic-netmon-report", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "report", help="HTML report for one or more recordings (one per node)"
    )
    r.add_argument("run_dirs", nargs="+")
    r.add_argument("-o", "--output", default="ainic-netmon-report.html")
    r.add_argument(
        "--role",
        action="append",
        metavar="HOST=P|D",
        help="node role for the direction view (default: meta tag role)",
    )
    r.add_argument(
        "--window",
        help="START,END: seconds from the common start, or ISO UTC times; "
        "draws raw points instead of the overview envelope",
    )
    r.add_argument("--nics", type=lambda s: s.split(","), help="only these netdevs")
    r.add_argument("--bins", type=int, default=2000, help="overview bins per series")
    r.add_argument(
        "--max-mb", type=float, default=50.0, help="overview HTML size budget"
    )
    r.add_argument(
        "--max-points", type=int, default=2_000_000, help="detail window point limit"
    )
    c = sub.add_parser("csv", help="raw values or rates as CSV")
    c.add_argument("run_dir")
    c.add_argument("--nic")
    c.add_argument("--source")
    c.add_argument("--counters", help="comma-separated counter names (default: all)")
    c.add_argument(
        "--rate",
        action="store_true",
        help="per-second rates between consecutive samples",
    )
    args = p.parse_args(argv)
    if args.cmd == "report":
        cmd_report(args)
    else:
        cmd_csv(args)


if __name__ == "__main__":
    main()
