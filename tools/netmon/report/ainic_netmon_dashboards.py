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
"""Generate the netmon Grafana dashboards (and the semantics of every query in them).

The dashboards are generated, not hand-edited, so that `ainic-netmon-import verify` can
check every panel query against the offline report library from the same source of truth
(SPEC below): each data query is one of a few templates over explicit windows
`[$__interval]` (NETMON_DASHBOARD_DESIGN §5).

    python3 ainic_netmon_dashboards.py OUT_DIR    # writes netmon_overview.json, netmon_nic_detail.json
"""

import json
import os
import sys

DS = {"type": "prometheus", "uid": "netmon-vm"}
GBPS = 8 / 1e9
SEL = 'experiment="$experiment",target="$target"'
META_LOOKBACK = "3650d"

RDMA_TX, RDMA_RX = "tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes"
IPC_TX, IPC_RX = "octets_tx_ok", "octets_rx_ok"
CONGESTION = [
    "rx_rdma_cnp_pkts",
    "tx_rdma_cnp_pkts",
    "rx_rdma_ecn_pkts",
    "tx_rdma_retx_pkts",
    "tx_rdma_ack_timeout",
]
# Categorical palette in fixed order (same as the offline report): tx blue, rx orange.
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


def _scale_suffix(scale):
    return " * 8 / 1e9" if scale == GBPS else ""


def q_window(fn, metric, source, counter, resolution, scale):
    sel = f'{SEL},source="{source}",counter="{counter}",resolution="{resolution}"'
    return f"{fn}({metric}{{{sel}}}[$__interval]){_scale_suffix(scale)}"


def q_mean(source, counter, scale):
    sel = f'{SEL},source="{source}",counter="{counter}",resolution="normal"'
    return (
        f"(sum_over_time(netmon_increase{{{sel}}}[$__interval]) / "
        f"sum_over_time(netmon_interval_seconds{{{sel}}}[$__interval])){_scale_suffix(scale)}"
    )


def target(ref, expr, legend, spec):
    return {
        "refId": ref,
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "range": True,
        "instant": False,
        "netmon": spec,  # ignored by Grafana; read by the verifier
    }


def meta_target(ref, expr, fmt="table"):
    return {
        "refId": ref,
        "datasource": DS,
        "expr": expr,
        "range": False,
        "instant": True,
        "format": fmt,
    }


LIVE_NOTE = (
    "Data with target status 'live' comes from ainic-netmon record --push: an online "
    "approximation of these rules; where a push failed the mean of that window may be off. "
    "Import the recording (ainic-netmon-import --replace) for the exact numbers."
)


def envelope_targets(source, counter, label, scale, color_idx, refs):
    """max / min (filled between) and mean of normal-resolution intervals, plus low-res."""
    r = iter(refs)
    spec = {"source": source, "counter": counter, "scale": scale}
    return [
        target(
            next(r),
            q_window("max_over_time", "netmon_rate", source, counter, "normal", scale),
            f"{label} max",
            dict(spec, kind="max"),
        ),
        target(
            next(r),
            q_window("min_over_time", "netmon_rate", source, counter, "normal", scale),
            f"{label} min",
            dict(spec, kind="min"),
        ),
        target(
            next(r),
            q_mean(source, counter, scale),
            f"{label} mean",
            dict(spec, kind="mean"),
        ),
        target(
            next(r),
            q_window("max_over_time", "netmon_rate", source, counter, "low", scale),
            f"{label} low-res",
            dict(spec, kind="low"),
        ),
    ], [
        _override_fill(f"{label} max", f"{label} min", PALETTE[color_idx]),
        _override_line(f"{label} min", PALETTE[color_idx], width=0),
        _override_line(f"{label} mean", PALETTE[color_idx], width=1.5),
        _override_points(f"{label} low-res", PALETTE[color_idx]),
    ]


def _override_fill(name, below, color):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [
            {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
            {"id": "custom.fillBelowTo", "value": below},
            {"id": "custom.fillOpacity", "value": 25},
            {"id": "custom.lineWidth", "value": 0},
        ],
    }


def _override_line(name, color, width):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [
            {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
            {"id": "custom.lineWidth", "value": width},
        ],
    }


def _override_points(name, color):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [
            {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
            {"id": "custom.drawStyle", "value": "points"},
            {"id": "custom.pointSize", "value": 6},
            {"id": "custom.showPoints", "value": "always"},
        ],
    }


def timeseries(
    pid,
    title,
    targets,
    overrides,
    grid,
    interval="${min_step}ms",
    unit="none",
    desc="",
    links=None,
    points=False,
):
    custom = {
        "drawStyle": "points" if points else "line",
        "lineWidth": 1,
        "fillOpacity": 0,
        "showPoints": "never" if not points else "always",
        "pointSize": 3,
    }
    p = {
        "id": pid,
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "interval": interval,
        "maxDataPoints": 2000,
        "gridPos": grid,
        "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": custom},
            "overrides": overrides,
        },
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }
    if links:
        p["links"] = links
    return p


def table(pid, title, targets, grid, transformations=None, desc=""):
    return {
        "id": pid,
        "type": "table",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": grid,
        "targets": targets,
        "transformations": transformations or [],
        "fieldConfig": {"defaults": {"decimals": 3}, "overrides": []},
        "options": {"showHeader": True},
    }


def variables(target_multi):
    v_exp = {
        "name": "experiment",
        "label": "experiment",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f"query_result(last_over_time(netmon_import_info[{META_LOOKBACK}]))",
            "refId": "V",
        },
        "regex": '/experiment="([^"]+)"/',
        "refresh": 2,
        "sort": 1,
    }
    v_tgt = {
        "name": "target",
        "label": "host/netdev",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f'query_result(last_over_time(netmon_target_info{{experiment="$experiment"}}[{META_LOOKBACK}]))',
            "refId": "V",
        },
        "regex": '/target="([^"]+)"/',
        "refresh": 2,
        "sort": 1,
        "multi": target_multi,
        "includeAll": target_multi,
    }
    # Largest safe step over the experiment's hosts (netmon_step_ms: written by the importer,
    # and by a live push, where it only grows). Panels use ${min_step}ms.
    v_step = {
        "name": "min_step",
        "label": "min step, ms (low-res threshold)",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f'query_result(max(last_over_time(netmon_step_ms{{experiment="$experiment"}}[{META_LOOKBACK}])))',
            "refId": "V",
        },
        "regex": "/\\} ([0-9]+) /",
        "refresh": 2,
        "description": "Smallest window of the envelope/mean panels: 3 x the sampling baseline, the "
        "largest over the experiment's hosts",
    }
    return {"list": [v_exp, v_tgt, v_step]}


def summary_table(pid, grid, sel):
    rate = f'last_over_time(netmon_summary{{{sel},stat=~"avg|peak10ms|p99"}}[{META_LOOKBACK}]) * 8 / 1e9'
    ratio = f'last_over_time(netmon_summary{{{sel},stat=~"coverage|above80"}}[{META_LOOKBACK}])'
    low = f'last_over_time(netmon_summary{{{sel},stat=~"lowres_n|lowres_s"}}[{META_LOOKBACK}])'
    return table(
        pid,
        "Fixed experiment window summary (offline report rules; rates in Gb/s, coverage/above80 as fractions)",
        # One query: groupingToMatrix needs a single "Value" column (several queries
        # would give "Value #A", "Value #B", ...). The stat label keeps them disjoint.
        [meta_target("A", f"({rate}) or ({ratio}) or ({low})")],
        grid,
        [
            {"id": "merge", "options": {}},
            {
                "id": "groupingToMatrix",
                "options": {
                    "columnField": "stat",
                    "rowField": "row",
                    "valueField": "Value",
                },
            },
        ],
        desc="Numbers of the offline report's summary table for the fixed window [window_start, window_end] "
        "(the experiment's common window). They do not change with zoom.",
    )


def info_table(pid, grid, sel):
    return table(
        pid,
        "Target info (recording status, link speed)",
        [
            meta_target(
                "A", f"last_over_time(netmon_target_info{{{sel}}}[{META_LOOKBACK}])"
            )
        ],
        grid,
        [
            {
                "id": "organize",
                "options": {
                    "excludeByName": {"Time": True, "Value": True, "__name__": True}
                },
            }
        ],
    )


def quality_panel(pid, grid, interval="${min_step}ms"):
    return timeseries(
        pid,
        "Data quality: gaps (no valid interval) and their duration, s",
        [
            target(
                "A",
                f'max_over_time(netmon_gap_seconds{{{SEL},reason="reset"}}[$__interval])',
                "{{source}}.{{counter}} reset",
                {"kind": "gap", "reason": "reset"},
            ),
            target(
                "B",
                f'max_over_time(netmon_gap_seconds{{{SEL},reason="clock"}}[$__interval])',
                "{{source}}.{{counter}} clock",
                {"kind": "gap", "reason": "clock"},
            ),
        ],
        [],
        grid,
        interval=interval,
        unit="s",
        points=True,
        desc="Intervals left out of every rate panel: counter resets and wall-clock uncertainty. "
        "Low-resolution intervals are shown as points in the rate panels.",
    )


DETAIL_LINK = [
    {
        "title": "NIC detail (this target, this time range)",
        "url": "/d/netmon-nic/netmon-nic-detail?var-experiment=${experiment}&var-target=${target}&${__url_time_range}",
    }
]


def multi_target(panels):
    """The overview's `target` is multi-valued (repeated rows). Grafana regex-escapes
    multi-valued variables for Prometheus (`.` -> `\\\\.` in the query text), so an equality
    matcher would never match a host name with dots: match with =~ instead (PromQL regexes
    are anchored, and the escaping makes it an exact match)."""
    for p in panels:
        for t in p.get("targets", []):
            t["expr"] = t["expr"].replace('target="$target"', 'target=~"$target"')


def overview():
    panels = []
    pid = 1
    panels.append(
        summary_table(
            pid,
            {"x": 0, "y": 0, "w": 24, "h": 8},
            'experiment="$experiment",target=~"$target"',
        )
    )
    pid += 1
    panels.append(
        {
            "id": pid,
            "type": "row",
            "title": "$target",
            "repeat": "target",
            "collapsed": False,
            "gridPos": {"x": 0, "y": 8, "w": 24, "h": 1},
            "panels": [],
        }
    )
    pid += 1
    y = 9
    t1, o1 = envelope_targets("rdma", RDMA_TX, "tx", GBPS, 0, "ABCD")
    t2, o2 = envelope_targets("rdma", RDMA_RX, "rx", GBPS, 1, "EFGH")
    panels.append(
        timeseries(
            pid,
            "RDMA payload, Gb/s (envelope = min..max per window, line = mean)",
            t1 + t2,
            o1 + o2,
            {"x": 0, "y": y, "w": 9, "h": 8},
            links=DETAIL_LINK,
            desc=LIVE_NOTE,
        )
    )
    pid += 1
    t3, o3 = envelope_targets("ipc", IPC_TX, "wire tx", GBPS, 0, "ABCD")
    t4, o4 = envelope_targets("ipc", IPC_RX, "wire rx", GBPS, 1, "EFGH")
    panels.append(
        timeseries(
            pid,
            "Port wire bytes (IPC), Gb/s, all traffic incl. ACKs",
            t3 + t4,
            o3 + o4,
            {"x": 9, "y": y, "w": 7, "h": 8},
            desc=LIVE_NOTE,
        )
    )
    pid += 1
    tc = [
        target(
            chr(65 + i),
            q_window("max_over_time", "netmon_rate", "rdma", c, "normal", 1.0),
            c,
            {"kind": "max", "source": "rdma", "counter": c, "scale": 1.0},
        )
        for i, c in enumerate(CONGESTION)
    ]
    panels.append(
        timeseries(
            pid,
            "Congestion and retransmission, max per window, /s",
            tc,
            [],
            {"x": 16, "y": y, "w": 8, "h": 8},
        )
    )
    pid += 1
    panels.append(info_table(pid, {"x": 0, "y": y + 8, "w": 12, "h": 4}, SEL))
    pid += 1
    panels.append(quality_panel(pid, {"x": 12, "y": y + 8, "w": 12, "h": 4}))
    multi_target(panels)
    return {
        "uid": "netmon-overview",
        "title": "netmon overview",
        "tags": ["netmon", "ainic"],
        "timezone": "utc",
        "schemaVersion": 39,
        "editable": True,
        "graphTooltip": 1,
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {"refresh_intervals": ["2s", "5s", "10s", "30s", "1m", "5m"]},
        "templating": variables(True),
        "panels": panels,
    }


def nic_detail():
    panels = []
    pid = 1
    panels.append(info_table(pid, {"x": 0, "y": 0, "w": 12, "h": 4}, SEL))
    pid += 1
    panels.append(summary_table(pid, {"x": 12, "y": 0, "w": 12, "h": 6}, SEL))
    pid += 1
    y = 6
    t1, o1 = envelope_targets("rdma", RDMA_TX, "tx", GBPS, 0, "ABCD")
    t2, o2 = envelope_targets("rdma", RDMA_RX, "rx", GBPS, 1, "EFGH")
    panels.append(
        timeseries(
            pid,
            "RDMA payload, Gb/s (envelope + mean per window ≥ ${min_step}ms)",
            t1 + t2,
            o1 + o2,
            {"x": 0, "y": y, "w": 24, "h": 8},
            desc=LIVE_NOTE,
        )
    )
    pid += 1
    y += 8
    raw = [
        target(
            "A",
            q_window("max_over_time", "netmon_rate", "rdma", RDMA_TX, "normal", GBPS),
            "tx",
            {"kind": "raw", "source": "rdma", "counter": RDMA_TX, "scale": GBPS},
        ),
        target(
            "B",
            q_window("max_over_time", "netmon_rate", "rdma", RDMA_RX, "normal", GBPS),
            "rx",
            {"kind": "raw", "source": "rdma", "counter": RDMA_RX, "scale": GBPS},
        ),
    ]
    panels.append(
        timeseries(
            pid,
            "Raw points, Gb/s (one point per interval when the step is 1 ms; otherwise max per step — step: $__interval)",
            raw,
            [_override_points("tx", PALETTE[0]), _override_points("rx", PALETTE[1])],
            {"x": 0, "y": y, "w": 24, "h": 8},
            interval="1ms",
            points=True,
        )
    )
    pid += 1
    y += 8
    tc = [
        target(
            chr(65 + i),
            q_window("max_over_time", "netmon_rate", "rdma", c, "normal", 1.0),
            c,
            {"kind": "max", "source": "rdma", "counter": c, "scale": 1.0},
        )
        for i, c in enumerate(CONGESTION)
    ]
    panels.append(
        timeseries(
            pid,
            "Congestion and retransmission, max per window, /s",
            tc,
            [],
            {"x": 0, "y": y, "w": 12, "h": 8},
        )
    )
    pid += 1
    t5, o5 = envelope_targets("ipc", IPC_TX, "wire tx", GBPS, 0, "ABCD")
    t6, o6 = envelope_targets("ipc", IPC_RX, "wire rx", GBPS, 1, "EFGH")
    panels.append(
        timeseries(
            pid,
            "Port wire bytes (IPC), Gb/s, all traffic incl. ACKs",
            t5 + t6,
            o5 + o6,
            {"x": 12, "y": y, "w": 12, "h": 8},
            desc=LIVE_NOTE,
        )
    )
    pid += 1
    y += 8
    pfc = [
        target(
            "A",
            f'max_over_time(netmon_rate{{{SEL},source="ipc",counter=~".*pripause.*|rx_pause_1us_count",resolution="normal"}}[$__interval])',
            "{{counter}}",
            {
                "kind": "max",
                "source": "ipc",
                "counter_re": ".*pripause.*|rx_pause_1us_count",
                "scale": 1.0,
            },
        )
    ]
    panels.append(
        timeseries(
            pid,
            "PFC pause counter increments, max per window, /s (field semantics unvalidated)",
            pfc,
            [],
            {"x": 0, "y": y, "w": 24, "h": 8},
        )
    )
    pid += 1
    y += 8
    panels.append(quality_panel(pid, {"x": 0, "y": y, "w": 24, "h": 6}))
    pid += 1
    panels.append(
        table(
            pid,
            "Run quality (samples / missed slots / dropped / errors)",
            [
                meta_target(
                    "A", f"last_over_time(netmon_run_quality{{{SEL}}}[{META_LOOKBACK}])"
                )
            ],
            {"x": 0, "y": y + 6, "w": 24, "h": 4},
            [
                {"id": "merge", "options": {}},
                {
                    "id": "groupingToMatrix",
                    "options": {
                        "columnField": "stat",
                        "rowField": "row",
                        "valueField": "Value",
                    },
                },
            ],
        )
    )
    return {
        "uid": "netmon-nic",
        "title": "netmon NIC detail",
        "tags": ["netmon", "ainic"],
        "timezone": "utc",
        "schemaVersion": 39,
        "editable": True,
        "graphTooltip": 1,
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {"refresh_intervals": ["2s", "5s", "10s", "30s", "1m", "5m"]},
        "templating": variables(False),
        "panels": panels,
    }


DASHBOARDS = {"netmon_overview.json": overview, "netmon_nic_detail.json": nic_detail}


def render():
    return {
        name: json.dumps(fn(), indent=2, ensure_ascii=False) + "\n"
        for name, fn in DASHBOARDS.items()
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    os.makedirs(argv[0], exist_ok=True)
    for name, text in render().items():
        with open(os.path.join(argv[0], name), "w") as fh:
            fh.write(text)
        print(os.path.join(argv[0], name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
