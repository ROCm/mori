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
check every panel query against the offline report library from the same source of truth:
each data query is one of a few templates over explicit windows `[$__interval]` (or
`[${__range_ms}ms]` in the overview table: Grafana's `$__range` is rounded to seconds), tagged with a "netmon" spec the verifier reads.

Design: NETMON_DASHBOARD_UX_DESIGN (v0.3). The main views show traffic only: per direction
an average line and a dashed peak line, both over every valid interval; gaps stay blank.
Collector details (gaps, sampling quality, the offline report's fixed-window summary) are
in a collapsed section at the bottom of the NIC detail page.

    python3 ainic_netmon_dashboards.py OUT_DIR    # writes netmon_overview.json, netmon_nic_detail.json
"""

import json
import os
import sys

DS = {"type": "prometheus", "uid": "netmon-vm"}
BITS = 8.0  # bytes/s -> bits/s (Grafana's "bps" unit scales to Gb/s itself)
SEL = 'experiment="$experiment",target="$target"'
META_LOOKBACK = "3650d"

RDMA_TX, RDMA_RX = "tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes"
IPC_TX, IPC_RX = "octets_tx_ok", "octets_rx_ok"
SEND_COLOR, RECV_COLOR = "#2a78d6", "#eb6834"  # fixed: blue = send, orange = receive

KEY_TEXT = (
    "**<span style='color:#2a78d6'>Blue = send</span>, "
    "<span style='color:#eb6834'>orange = receive</span>. "
    "Solid = average, dashed = observed peak. Gaps = data unavailable.**"
)
LIVE_NOTE = (
    "Data marked 'live' comes from ainic-netmon record --push: an online approximation; "
    "where a push failed the average of that time bin may be off. Import the recording "
    "(ainic-netmon-import --replace) for the exact numbers."
)
TRAFFIC_DESC = (
    "Solid line: average traffic in each time bin. Dashed line: the highest single "
    "observation in the bin (a counter read ~2 ms apart), so short spikes stay visible when "
    "zoomed out; a single observation has timing jitter and can slightly exceed the link "
    "rate. Drag across the chart to zoom in. Gaps: no valid observation (never drawn as 0). "
    + LIVE_NOTE
)
ETHTOOL_NOTE = (
    " On hosts without firmware IPC (e.g. a VM with a VF) the port statistics come from "
    "ethtool instead (lines marked '≈5 s, ethtool'): the firmware updates them about every "
    "5 s, so each point is an average over about 5 s (zoomed in: sparse points). ethtool "
    "gives a point only when the counters changed: no point means no observed change, not a "
    "measured zero."
)
ETHTOOL_SUFFIX = " (≈5 s, ethtool)"
PORT_TRAFFIC_DESC = (
    "All traffic through the port, including non-RDMA Ethernet. With firmware IPC: solid "
    "line = average per time bin, dashed line = highest observation in the bin (counters "
    "read ~2 ms apart). Without firmware IPC (e.g. a VM with a VF): only the lines marked "
    "'≈5 s, ethtool', points that each average about 5 s (the firmware updates these "
    "counters about every 5 s; zoomed in they are sparse points). ethtool gives a point only "
    "when the counters changed: no point means no observed change, not a measured zero. "
    "Gaps: no valid observation (never drawn as 0). " + LIVE_NOTE
)
DETAIL_URL = (
    "/d/netmon-nic/netmon-nic-detail?var-experiment=${experiment:percentencode}"
    "&var-host=HOST&var-target=TARGET&${__url_time_range}"
)


def detail_url(host, target):
    return DETAIL_URL.replace("HOST", host).replace("TARGET", target)


# --------------------------------------------------------------------------- queries


def _sel(source, counter):
    return f'{SEL},source="{source}",counter="{counter}"'


def q_average(source, counter, scale, window="$__interval", sel=None):
    """Σ increase / Σ duration of every valid interval ending in the window (both
    resolutions: the labels are combined before dividing)."""
    s = sel or _sel(source, counter)
    return (
        f"(sum without (resolution) (sum_over_time(netmon_increase{{{s}}}[{window}])) / "
        f"sum without (resolution) (sum_over_time(netmon_interval_seconds{{{s}}}[{window}])))"
        f" * {scale:g}"
    )


def q_peak(source, counter, scale):
    """Highest single-interval rate in the window (both resolutions)."""
    return (
        f"max without (resolution) (max_over_time(netmon_rate{{{_sel(source, counter)}}}"
        f"[$__interval])) * {scale:g}"
    )


def q_max_normal(source, counter):
    """Packet / event counters: highest rate per window (normal resolution)."""
    return (
        f'max_over_time(netmon_rate{{{_sel(source, counter)},resolution="normal"}}'
        "[$__interval])"
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


def instant_target(ref, expr, spec=None, fmt="table"):
    t = {
        "refId": ref,
        "datasource": DS,
        "expr": expr,
        "range": False,
        "instant": True,
        "format": fmt,
    }
    if spec:
        t["netmon"] = spec
    return t


def target_info(sel):
    """netmon_target_info with display_nic; experiments imported before the display names
    existed get their netdev as the name."""
    lb = f"[{META_LOOKBACK}]"
    return (
        f'last_over_time(netmon_target_info{{{sel},display_nic!=""}}{lb}) or '
        f'label_copy(last_over_time(netmon_target_info{{{sel},display_nic=""}}{lb}), '
        '"netdev", "display_nic")'
    )


def link_bps(sel):
    return (
        "sum by (target) (label_value(last_over_time(netmon_target_info"
        f'{{{sel}}}[{META_LOOKBACK}]), "speed_mbps")) * 1e6 > 0'
    )


# --------------------------------------------------------------------------- panels


def _line(name, color, width, dashed=False):
    props = [
        {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
        {"id": "custom.lineWidth", "value": width},
    ]
    if dashed:
        props.append(
            {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [4, 4]}}
        )
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


def _points(name, color):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [
            {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
            {"id": "custom.drawStyle", "value": "points"},
            {"id": "custom.pointSize", "value": 3},
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
    repeat=None,
):
    custom = {
        "drawStyle": "points" if points else "line",
        "lineWidth": 1,
        "fillOpacity": 0,
        "showPoints": "always" if points else "never",
        "pointSize": 3,
        "axisSoftMin": 0,
    }
    defaults = {"unit": unit, "custom": custom, "noValue": "No data in this time range"}
    if links:
        defaults["links"] = links
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
        "fieldConfig": {"defaults": defaults, "overrides": overrides},
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }
    if repeat:
        p.update(repeat=repeat, repeatDirection="h", maxPerRow=4)
    return p


def traffic_panel(
    pid, title, source, tx, rx, grid, links=None, repeat=None, small=False
):
    """Per direction: average (solid) and observed peak (dashed)."""
    targets, overrides = [], []
    for refs, counter, name, color in (
        ("AB", tx, "Send", SEND_COLOR),
        ("CD", rx, "Receive", RECV_COLOR),
    ):
        spec = {"source": source, "counter": counter, "scale": BITS}
        targets.append(
            target(
                refs[0],
                q_average(source, counter, BITS),
                name,
                dict(spec, kind="mean_all"),
            )
        )
        targets.append(
            target(
                refs[1],
                q_peak(source, counter, BITS),
                f"{name} peak",
                dict(spec, kind="max_all"),
            )
        )
        overrides += [
            _line(name, color, 2),
            _line(f"{name} peak", color, 1, dashed=True),
        ]
    p = timeseries(
        pid,
        title,
        targets,
        overrides,
        grid,
        unit="bps",
        desc="" if small else TRAFFIC_DESC,
        links=links,
        repeat=repeat,
    )
    if small:  # the key above the charts explains colours and line styles
        p["options"]["legend"]["showLegend"] = False
    return p


def _line_points(name, color):
    """ethtool's ~5 s averages: line and points (sparse when zoomed in, never bridged)."""
    o = _line(name, color, 2)
    o["properties"] += [
        {"id": "custom.showPoints", "value": "always"},
        {"id": "custom.pointSize", "value": 5},
    ]
    return o


def port_traffic_panel(pid, grid):
    """IPC wire bytes (2 ms), or ethtool's (~5 s) where a NIC has no IPC: average only."""
    p = traffic_panel(
        pid, "All port traffic (incl. non-RDMA)", "ipc", IPC_TX, IPC_RX, grid
    )
    for ref, counter, name, color in (
        ("E", IPC_TX, "Send", SEND_COLOR),
        ("F", IPC_RX, "Receive", RECV_COLOR),
    ):
        legend = name + ETHTOOL_SUFFIX
        p["targets"].append(
            target(
                ref,
                q_average("ethtool", counter, BITS),
                legend,
                {
                    "source": "ethtool",
                    "counter": counter,
                    "scale": BITS,
                    "kind": "mean_all",
                },
            )
        )
        p["fieldConfig"]["overrides"].append(_line_points(legend, color))
    p["description"] = PORT_TRAFFIC_DESC
    return p


def packet_panel(pid, title, counters, grid, unit="pps", desc=""):
    """counters: [(counter, legend)], highest rate per time bin."""
    targets = [
        target(
            chr(65 + i),
            q_max_normal("rdma", c),
            legend,
            {"kind": "max", "source": "rdma", "counter": c, "scale": 1.0},
        )
        for i, (c, legend) in enumerate(counters)
    ]
    return timeseries(pid, title, targets, [], grid, unit=unit, desc=desc)


def text_panel(pid, grid, content):
    return {
        "id": pid,
        "type": "text",
        "title": "",
        "transparent": True,
        "gridPos": grid,
        "options": {"mode": "markdown", "content": content},
    }


def table(pid, title, targets, grid, transformations, overrides=(), desc="", sort=None):
    p = {
        "id": pid,
        "type": "table",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": grid,
        "targets": targets,
        "transformations": transformations,
        "fieldConfig": {
            "defaults": {"noValue": "—", "custom": {"align": "auto"}},
            "overrides": list(overrides),
        },
        "options": {"showHeader": True, "cellHeight": "sm"},
    }
    if sort:
        p["options"]["sortBy"] = sort
    return p


def row(pid, title, y, collapsed=False, panels=()):
    return {
        "id": pid,
        "type": "row",
        "title": title,
        "collapsed": collapsed,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": list(panels),
    }


def _by_name(name, *props):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": k, "value": v} for k, v in props],
    }


def _hidden(name):
    return _by_name(name, ("custom.hidden", True))


PORT_STATS = {
    "type": "value",
    "options": {
        "ipc": {"text": "firmware IPC (2 ms)"},
        "ethtool": {"text": "ethtool (≈5 s)"},
        "none": {"text": "not available"},
    },
}
DATA_STATUS = {
    "type": "value",
    "options": {
        "complete": {"text": "imported"},
        # stopped by Ctrl-C / SIGTERM: the usual end of a recording without -d
        "interrupted": {"text": "imported (stopped by signal)"},
        "failed": {"text": "imported (recording stopped by an error)"},
        "running": {"text": "imported (recording was still running)"},
        "live": {"text": "live (approximate until imported)"},
        "partial": {"text": "imported (recording incomplete)"},
    },
}


# --------------------------------------------------------------------------- variables


def variables(multi):
    lb = f"[{META_LOOKBACK}]"
    info = 'netmon_target_info{experiment="$experiment"%s}'
    v_exp = {
        "name": "experiment",
        "label": "Experiment",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f"query_result(last_over_time(netmon_import_info{lb}))",
            "refId": "V",
        },
        "regex": '/experiment="([^"]+)"/',
        "refresh": 2,
        "sort": 1,
    }
    v_host = {
        "name": "host",
        "label": "Host",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f"query_result(last_over_time({info % ''}{lb}))",
            "refId": "V",
        },
        "regex": '/host="(?<value>(?<text>[^".]+)[^"]*)"/',
        "refresh": 2,
        "sort": 7,
        "multi": multi,
        "includeAll": multi,
    }
    # Value = target (host/netdev, unique); text = the RDMA device name with the netdev as a
    # note. Experiments imported before the display names existed show their target.
    display = "display_host_nic" if multi else "display_nic"
    host_m = ',host=~"$host"' if multi else ',host="$host"'
    v_nic = {
        "name": "target",
        "label": "NIC",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f"query_result(last_over_time({info % host_m}{lb}))",
            "refId": "V",
        },
        "regex": f'/{display}="(?<text>[^"]+)"|target="(?<value>[^"]+)"/g',
        "refresh": 2,
        "sort": 7,
        "multi": multi,
        "includeAll": multi,
    }
    # Smallest time bin of the traffic charts: 3 x the sampling baseline (largest over the
    # experiment's hosts), so a bin never is finer than the sampling. Not shown.
    v_step = {
        "name": "min_step",
        "label": "min step, ms",
        "type": "query",
        "datasource": DS,
        "query": {
            "query": f'query_result(max(last_over_time(netmon_step_ms{{experiment="$experiment"}}{lb})))',
            "refId": "V",
        },
        "regex": "/\\} ([0-9]+) /",
        "refresh": 2,
        "hide": 2,
    }
    return {"list": [v_exp, v_host, v_nic, v_step]}


def _multi(panels):
    """The overview's `target` is multi-valued: Grafana regex-escapes it for Prometheus
    (`.` -> `\\\\.`), so match with =~ (anchored; the escaping makes it exact)."""
    for p in panels:
        for t in p.get("targets", []):
            t["expr"] = t["expr"].replace('target="$target"', 'target=~"$target"')
        _multi(p.get("panels", []))


def dashboard(uid, title, panels, multi, links=()):
    return {
        "uid": uid,
        "title": title,
        "tags": ["netmon", "ainic"],
        "timezone": "utc",
        "schemaVersion": 39,
        "editable": True,
        "graphTooltip": 1,
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {"refresh_intervals": ["2s", "5s", "10s", "30s", "1m", "5m"]},
        "templating": variables(multi),
        "links": list(links),
        "panels": panels,
    }


# --------------------------------------------------------------------------- overview


def nic_table(pid, grid):
    sel = 'experiment="$experiment",target=~"$target"'
    avg = {"source": "rdma", "scale": BITS, "kind": "range_mean"}
    targets = [
        # host_short for display: a value mapping would also change the link's host value
        instant_target(
            "A",
            f'label_replace({target_info(sel)}, "host_short", "$1", "host", "([^.]+).*")',
        ),
        instant_target("B", link_bps(sel)),
        instant_target(
            "C",
            q_average(
                "rdma",
                RDMA_TX,
                BITS,
                "${__range_ms}ms",
                f'{sel},source="rdma",counter="{RDMA_TX}"',
            ).replace("sum without (resolution)", "sum by (target)"),
            dict(avg, counter=RDMA_TX),
        ),
        instant_target(
            "D",
            q_average(
                "rdma",
                RDMA_RX,
                BITS,
                "${__range_ms}ms",
                f'{sel},source="rdma",counter="{RDMA_RX}"',
            ).replace("sum without (resolution)", "sum by (target)"),
            dict(avg, counter=RDMA_RX),
        ),
    ]
    keep = [
        "host_short",
        "display_nic",
        "Value #B",
        "Value #C",
        "Value #D",
        "host",
        "target",
    ]
    link = detail_url(
        "${__data.fields.host:percentencode}", "${__data.fields.target:percentencode}"
    )
    return table(
        pid,
        "RDMA traffic over the selected time",
        targets,
        grid,
        [
            {"id": "joinByField", "options": {"byField": "target", "mode": "outer"}},
            {
                "id": "organize",
                "options": {
                    "includeByName": {k: True for k in keep},
                    "indexByName": {k: i for i, k in enumerate(keep)},
                    "renameByName": {
                        "host_short": "Host",
                        "display_nic": "NIC",
                        "Value #B": "Link",
                        "Value #C": "Average send",
                        "Value #D": "Average receive",
                    },
                },
            },
        ],
        [
            _by_name(
                "NIC",
                ("links", [{"title": "Open NIC detail", "url": link}]),
            ),
            _by_name("Link", ("unit", "bps"), ("decimals", 0)),
            _by_name("Average send", ("unit", "bps"), ("decimals", 1)),
            _by_name("Average receive", ("unit", "bps"), ("decimals", 1)),
            _hidden("host"),
            _hidden("target"),
        ],
        desc="One row per NIC. Average send / receive: the average of the observations that "
        "end in the selected time range (a long observation crossing the start is counted "
        "whole). Click a NIC for its detail page. " + LIVE_NOTE,
    )


def overview():
    panels = [text_panel(1, {"x": 0, "y": 0, "w": 24, "h": 2}, KEY_TEXT)]
    panels.append(nic_table(2, {"x": 0, "y": 2, "w": 24, "h": 8}))
    link = detail_url(
        "${__field.labels.host:percentencode}", "${__field.labels.target:percentencode}"
    )
    panels.append(
        traffic_panel(
            3,
            "${target:text}",
            "rdma",
            RDMA_TX,
            RDMA_RX,
            {"x": 0, "y": 10, "w": 6, "h": 7},
            links=[{"title": "Open NIC detail", "url": link}],
            repeat="target",
            small=True,
        )
    )
    _multi(panels)
    return dashboard("netmon-overview", "netmon overview", panels, True)


# --------------------------------------------------------------------------- NIC detail


def identity_strip(pid, grid):
    keep = [
        "host", "display_nic", "netdev", "pci", "Value #B", "port_stats", "status", "target"
    ]  # fmt: skip
    return table(
        pid,
        "",
        [instant_target("A", target_info(SEL)), instant_target("B", link_bps(SEL))],
        grid,
        [
            {"id": "joinByField", "options": {"byField": "target", "mode": "outer"}},
            {
                "id": "organize",
                "options": {
                    "includeByName": {k: True for k in keep},
                    "indexByName": {k: i for i, k in enumerate(keep)},
                    "renameByName": {
                        "host": "Host",
                        "display_nic": "NIC",
                        "netdev": "netdev",
                        "pci": "PCI",
                        "Value #B": "Link",
                        "port_stats": "Port stats",
                        "status": "Data",
                    },
                },
            },
        ],
        [
            _by_name("Host", ("custom.width", 480)),
            _by_name("Link", ("unit", "bps"), ("decimals", 0)),
            _by_name("Data", ("mappings", [DATA_STATUS])),
            _by_name("Port stats", ("mappings", [PORT_STATS])),
            _hidden("target"),
        ],
    )


ETHTOOL_PFC_RE = "frames_(tx|rx)_(pri)?pause"
GAP_NAMES = {
    ("rdma", "reset"): "RDMA counters: counter reset or rejected reading",
    ("rdma", "clock"): "RDMA counters: timestamp uncertain",
    ("ipc", "reset"): "Port counters: counter reset or rejected reading",
    ("ipc", "clock"): "Port counters: timestamp uncertain",
    ("ethtool", "reset"): "Port counters (ethtool): counter reset",
    ("ethtool", "clock"): "Port counters (ethtool): timestamp uncertain",
}
QUALITY_NAMES = {
    "samples": "Samples",
    "missed_slots": "Missed sampling slots",
    "dropped": "Samples lost before saving",
    "errors": "Read errors",
}
SUMMARY_NAMES = {
    "avg": "Average",
    "p99": "p99",
    "peak10ms": "10 ms peak",
    "above80": "Time above 80 % of link",
    "coverage": "Time with fine-grained observations",
    "lowres_n": "Long observations",
    "lowres_s": "Long-observation time",
}
COUNTER_NAMES = {
    "type": "value",
    "options": {
        RDMA_TX: {"text": "RDMA send"},
        RDMA_RX: {"text": "RDMA receive"},
        IPC_TX: {"text": "Port send"},
        IPC_RX: {"text": "Port receive"},
        "rdma": {"text": "RDMA counters"},
        "ipc": {"text": "Port counters"},
    },
}


def diagnostics(pid, y):
    gaps = timeseries(
        pid,
        "Time without a valid observation (each point: one gap, its length)",
        [
            target(
                "A",
                f"max by (experiment, target, source, reason) (max_over_time(netmon_gap_seconds{{{SEL}}}[$__interval]))",
                "{{source}} {{reason}}",
                {"kind": "gap_by_source"},
            )
        ],
        [
            {
                "matcher": {"id": "byName", "options": f"{s} {r}"},
                "properties": [{"id": "displayName", "value": name}],
            }
            for (s, r), name in GAP_NAMES.items()
        ],
        {"x": 0, "y": y, "w": 24, "h": 7},
        unit="s",
        points=True,
        desc="Collector quality, not network loss: times when a counter could not be turned "
        "into a rate (a counter went back, an IPC read was rejected, or the wall-clock "
        "position was uncertain). These times are the gaps in the traffic charts. Gaps of "
        "different counters are not added up.",
    )
    quality = table(
        pid + 1,
        "Recording quality",
        [
            instant_target(
                "A", f"last_over_time(netmon_run_quality{{{SEL}}}[{META_LOOKBACK}])"
            )
        ],
        {"x": 0, "y": y + 7, "w": 24, "h": 4},
        [
            {
                "id": "groupingToMatrix",
                "options": {
                    "columnField": "stat",
                    "rowField": "source",
                    "valueField": "Value",
                },
            },
            {
                "id": "organize",
                "options": {
                    "renameByName": dict(QUALITY_NAMES, **{"source\\stat": "Counters"})
                },
            },
        ],
        [
            _by_name("Counters", ("mappings", [COUNTER_NAMES])),
            {
                "matcher": {"id": "byType", "options": "number"},
                "properties": [{"id": "decimals", "value": 0}],
            },
        ],
        desc="Samples: counter reads saved. Missed sampling slots: sampling times skipped "
        "because a read was late (e.g. a firmware stall). Samples lost before saving: the "
        "writer could not keep up. Read errors: failed reads.",
    )
    rate = f'last_over_time(netmon_summary{{{SEL},stat=~"avg|peak10ms|p99"}}[{META_LOOKBACK}]) * 8'
    other = f'last_over_time(netmon_summary{{{SEL},stat!~"avg|peak10ms|p99"}}[{META_LOOKBACK}])'
    summary = table(
        pid + 2,
        "Offline report summary (fixed experiment window; does not change with zoom)",
        [instant_target("A", f"({rate}) or ({other})")],
        {"x": 0, "y": y + 11, "w": 24, "h": 5},
        [
            {
                "id": "groupingToMatrix",
                "options": {
                    "columnField": "stat",
                    "rowField": "counter",
                    "valueField": "Value",
                },
            },
            {
                "id": "organize",
                "options": {
                    "renameByName": dict(SUMMARY_NAMES, **{"counter\\stat": "Traffic"}),
                    "indexByName": {
                        k: i for i, k in enumerate(["counter\\stat", *SUMMARY_NAMES])
                    },
                },
            },
        ],
        [
            _by_name("Traffic", ("mappings", [COUNTER_NAMES])),
            *[
                _by_name(SUMMARY_NAMES[k], ("unit", "bps"), ("decimals", 1))
                for k in ("avg", "p99", "peak10ms")
            ],
            *[
                _by_name(SUMMARY_NAMES[k], ("unit", "percentunit"), ("decimals", 1))
                for k in ("above80", "coverage")
            ],
            _by_name(SUMMARY_NAMES["lowres_n"], ("decimals", 0)),
            _by_name(SUMMARY_NAMES["lowres_s"], ("unit", "s"), ("decimals", 2)),
        ],
        desc="The offline report's numbers for the experiment's common recording window "
        "(ainic-netmon-report shows the window). p99 and the time above 80 % use fine-grained "
        "observations only; the average includes long observations.",
    )
    return [gaps, quality], summary


def nic_detail():
    panels = [
        text_panel(1, {"x": 0, "y": 0, "w": 24, "h": 2}, KEY_TEXT),
        identity_strip(2, {"x": 0, "y": 2, "w": 24, "h": 3}),
        traffic_panel(
            3,
            "RDMA traffic",
            "rdma",
            RDMA_TX,
            RDMA_RX,
            {"x": 0, "y": 5, "w": 24, "h": 10},
        ),
        port_traffic_panel(4, {"x": 0, "y": 15, "w": 24, "h": 8}),
        row(5, "Retransmissions and congestion", 23),
        packet_panel(
            6,
            "Retransmitted packets",
            [("tx_rdma_retx_pkts", "Retransmitted")],
            {"x": 0, "y": 24, "w": 8, "h": 7},
            desc="Highest rate per time bin. 0 = observed zero; gaps = no observation.",
        ),
        packet_panel(
            7,
            "Congestion notifications",
            [
                ("rx_rdma_ecn_pkts", "ECN-marked received"),
                ("rx_rdma_cnp_pkts", "CNP received"),
                ("tx_rdma_cnp_pkts", "CNP sent"),
            ],
            {"x": 8, "y": 24, "w": 8, "h": 7},
            desc="Highest rate per time bin. All zero does not prove the network had no "
            "pressure: a fabric without ECN marking recovers by drops and retransmission.",
        ),
        packet_panel(
            8,
            "ACK timeouts",
            [("tx_rdma_ack_timeout", "ACK timeouts")],
            {"x": 16, "y": 24, "w": 8, "h": 7},
            unit="short",
            desc="Highest rate per time bin, per second.",
        ),
    ]
    y = 31
    samples = timeseries(
        10,
        "Individual observations (at 1 ms bins each point is one observation; otherwise the highest per $__interval)",
        [
            target(
                "A",
                q_peak("rdma", RDMA_TX, BITS),
                "Send",
                {
                    "kind": "max_all",
                    "source": "rdma",
                    "counter": RDMA_TX,
                    "scale": BITS,
                },
            ),
            target(
                "B",
                q_peak("rdma", RDMA_RX, BITS),
                "Receive",
                {
                    "kind": "max_all",
                    "source": "rdma",
                    "counter": RDMA_RX,
                    "scale": BITS,
                },
            ),
        ],
        [_points("Send", SEND_COLOR), _points("Receive", RECV_COLOR)],
        {"x": 0, "y": y + 1, "w": 24, "h": 8},
        interval="1ms",
        unit="bps",
        points=True,
        desc="RDMA traffic as individual observations. Zoom in until the bin is 1 ms to see "
        "every observation.",
    )
    panels.append(row(9, "Inspect individual observations", y, True, [samples]))
    pfc = timeseries(
        12,
        "PFC pause counters, highest rate per time bin",
        [
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
            ),
            target(
                "B",
                f'max_over_time(netmon_rate{{{SEL},source="ethtool",counter=~"{ETHTOOL_PFC_RE}",resolution="normal"}}[$__interval])',
                "{{counter}}" + ETHTOOL_SUFFIX,
                {
                    "kind": "max",
                    "source": "ethtool",
                    "counter_re": ETHTOOL_PFC_RE,
                    "scale": 1.0,
                },
            ),
        ],
        [
            {
                "matcher": {"id": "byRegexp", "options": ".*ethtool.*"},
                "properties": [
                    {"id": "custom.showPoints", "value": "always"},
                    {"id": "custom.pointSize", "value": 5},
                ],
            }
        ],
        {"x": 0, "y": y + 2, "w": 24, "h": 8},
        desc="Firmware port counters whose meaning has not been validated yet. Only counters "
        "that changed during the recording are imported (ethtool: always)."
        + ETHTOOL_NOTE,
    )
    panels.append(
        row(
            11,
            "PFC pause counters (experimental: meaning not yet validated)",
            y + 1,
            True,
            [pfc],
        )
    )
    diag, summary = diagnostics(14, y + 3)
    summary["gridPos"] = {"x": 0, "y": y + 4, "w": 24, "h": 5}
    panels.append(
        row(
            17,
            "Offline report summary (fixed experiment window)",
            y + 2,
            True,
            [summary],
        )
    )
    panels.append(
        row(
            13,
            "Recording diagnostics (for troubleshooting the monitor itself)",
            y + 3,
            True,
            diag,
        )
    )
    return dashboard(
        "netmon-nic",
        "netmon NIC detail",
        panels,
        False,
        links=[
            {
                "title": "Overview",
                "type": "link",
                "url": "/d/netmon-overview/netmon-overview?var-experiment=${experiment:percentencode}",
                "keepTime": True,
            }
        ],
    )


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
