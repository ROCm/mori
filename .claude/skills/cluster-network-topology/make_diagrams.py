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
"""Generate topology diagrams (Graphviz DOT + PNG) from a probe/cross-rail run folder.

    python3 make_diagrams.py <output_dir> [--force]

Inputs (whatever the run produced; each diagram is skipped if its input is absent):
    topo_report*.txt   -> single-node GPU <-> rail NIC diagram
    *result*.txt       -> 2-node cross-rail fabric diagram
    addrs.<host>       -> peer rail device names for the cross-rail diagram

Outputs, named to match make_report.py's globs so the report picks them up:
    topology_<host>_node.dot / .png
    topology_<A>__<B>_crossrail.dot / .png

Everything drawn here comes from the measured text — edge colours encode the
recorded pass/fail, never an assumption. Rendering shells out to the `dot`
binary; if graphviz is missing the .dot files are still written and the report
falls back to showing their source.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_report  # noqa: E402

GREEN = "#2e8b57"
RED = "#cc2b2b"
CLUSTER_FILL = ["#eef5ff", "#fff3e6", "#f1f7ee", "#f7eefb"]
CLUSTER_LINE = ["#5b8def", "#e08a2e", "#5aa469", "#9b59b6"]


# ---------- parsers ----------
def parse_probe(text):
    """Parse probe_topology.sh's topo_report.txt into structured facts."""
    out = {
        "host": "",
        "gpu_model": "",
        "mgmt_ndev": "",
        "rail_devs": [],
        "devices": [],
        "gpus": [],
        "pairs": [],
    }
    m = re.search(r"^=== node:\s*(\S+)", text, re.M)
    if m:
        out["host"] = m.group(1)
    m = re.search(r"^GPU model:\s*(.+?)\s*$", text, re.M)
    if m:
        out["gpu_model"] = m.group(1)
    m = re.search(r"^rail NICs:\s*(.*?)\s*\(mgmt.*?excluded:\s*([^)]*)\)", text, re.M)
    if m:
        out["rail_devs"] = m.group(1).split()
        out["mgmt_ndev"] = m.group(2).strip()

    for ln in text.splitlines():
        m = re.match(
            r"^(\S+)\s+state=(\S*)\s+ndev=(\S*)\s+gid\[([^\]]*)\]=(.*?)\s+ip=(\S*)\s+"
            r"pci=(\S*)\s+numa=(\S*)\s*$",
            ln,
        )
        if m:
            out["devices"].append(
                {
                    "dev": m.group(1),
                    "state": m.group(2),
                    "ndev": m.group(3),
                    "gid": m.group(4),
                    "gidtype": m.group(5).strip(),
                    "ip": m.group(6),
                    "pci": m.group(7),
                    "numa": m.group(8),
                }
            )
            continue
        m = re.match(r"^GPU(\d+)\s+pci=(\S+)\s*$", ln)
        if m:
            out["gpus"].append({"idx": int(m.group(1)), "pci": m.group(2)})
            continue
        m = re.match(
            r"^GPU(\d+)\s+\((\S+)\)\s+->\s+(\S+)\s+\((\S+)\)\s+rail-local\s*$", ln
        )
        if m:
            out["pairs"].append(
                {
                    "gpu": int(m.group(1)),
                    "gpu_pci": m.group(2),
                    "dev": m.group(3),
                    "dev_pci": m.group(4),
                }
            )
    return out


def parse_railmap(text):
    """Parse the cross-rail result.txt: hostnames, A's rail map, the IP matrix."""
    out = {"A": "", "B": "", "nr": 0, "bnr": 0, "rails": [], "ip_rows": []}
    m = re.search(r"tester=(\S+)\s+target=(\S+)", text)
    if m:
        out["A"], out["B"] = m.group(1), m.group(2)
    m = re.search(r"rails on A=(\d+)\s+rails on B=(\d+)", text)
    if m:
        out["nr"], out["bnr"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"^rail map \(A\):\s*(.+)$", text, re.M)
    if m:
        for tok in m.group(1).split():
            t = re.match(r"^r(\d+)=([^/]+)/([^/]*)/gid(\S*)$", tok)
            if t:
                out["rails"].append(
                    (int(t.group(1)), t.group(2), t.group(3), t.group(4))
                )
    for ln in text.splitlines():
        m = re.match(
            r"^rail(\d+)\(([^)]*)\)\s+rail(\d+)\s+(\S+)\s+(same-rail|cross-rail)",
            ln.strip(),
        )
        if m:
            out["ip_rows"].append(
                (int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5))
            )
    return out


def parse_addrs(path):
    """Parse an addrs.<host> file: (idx, dev, ndev, fam, gid, addr) per rail."""
    rows = []
    for ln in make_report.read(path).splitlines():
        f = ln.split()
        if len(f) >= 6 and f[0].isdigit():
            rows.append((int(f[0]), f[1], f[2], f[3], f[4], f[5]))
    return rows


# ---------- shared bits ----------
def _q(s):
    return str(s).replace('"', '\\"')


def _nid(dev):
    return "N_" + re.sub(r"\W", "_", dev)


def verdict_of(result_text):
    measured = make_report.parse_result(result_text)
    fab, desc, _cls = make_report.classify(measured)
    return fab, desc, measured


def _fam_label(ip):
    if not ip:
        return "unknown"
    return "IPv6" if ":" in ip else "IPv4"


def _legend(title, lines):
    body = "".join(_q(x) + "\\l" for x in lines)
    return (
        "  subgraph cluster_legend {\n"
        f'    label="{_q(title)}"; style="rounded,filled"; fillcolor="#fafafa"; '
        'color="#999999"; fontsize=11;\n'
        '    L [shape=note, fillcolor="#ffffff", style="filled", fontsize=10,\n'
        f'       label="{body}"];\n'
        "  }\n"
    )


# ---------- single-node diagram ----------
def node_dot(probe, result_text=""):
    host = probe["host"] or "node"
    devs = {d["dev"]: d for d in probe["devices"]}
    gpu_pci = {g["idx"]: g["pci"] for g in probe["gpus"]}
    pair_dev = {p["gpu"]: p["dev"] for p in probe["pairs"]}
    model = probe["gpu_model"] or "GPU"

    # One subgraph per (NUMA, PCI domain): that is the affinity boundary the
    # ordinal pairing in probe_topology.sh works within.
    groups = {}
    for gpu in sorted(gpu_pci):
        dev = pair_dev.get(gpu, "")
        d = devs.get(dev, {})
        key = (d.get("numa", "?"), (d.get("pci") or gpu_pci[gpu]).split(":")[0])
        groups.setdefault(key, {"gpus": [], "nics": []})["gpus"].append(gpu)
        if dev:
            groups[key]["nics"].append(dev)
    for dev in probe["rail_devs"]:
        if any(dev in g["nics"] for g in groups.values()):
            continue
        d = devs.get(dev, {})
        key = (d.get("numa", "?"), (d.get("pci") or "").split(":")[0])
        groups.setdefault(key, {"gpus": [], "nics": []})["nics"].append(dev)

    multi_domain = len({k[1] for k in groups}) > 1
    L = []
    L.append(f"// {host} — RDMA / GPU rail topology")
    L.append("// Generated by make_diagrams.py from probe_topology.sh output.")
    L.append("digraph topo {")
    L.append(
        '  rankdir=LR; compound=true; labelloc="t"; fontsize=16; fontname="Helvetica";'
    )
    L.append(
        f'  label="{_q(host)} ({_q(model)}) — RDMA / GPU rail topology '
        f'({len(gpu_pci)} GPU : {len(probe["rail_devs"])} rail NIC)";'
    )
    L.append(
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];'
    )
    L.append("")

    for n, key in enumerate(sorted(groups)):
        numa, dom = key
        g = groups[key]
        label = f"NUMA {numa}" + (f"  ·  PCI domain {dom}" if multi_domain else "")
        L.append(f"  subgraph cluster_g{n} {{")
        L.append(
            f'    label="{_q(label)}"; style="rounded,filled"; '
            f'fillcolor="{CLUSTER_FILL[n % 4]}"; color="{CLUSTER_LINE[n % 4]}";'
        )
        L.append('    node [fillcolor="#dbe9ff"];')
        for gpu in g["gpus"]:
            L.append(f'    G{gpu}[label="GPU{gpu}\\n{_q(gpu_pci[gpu])}"];')
        L.append('    node [fillcolor="#c9f2d4"];')
        for dev in g["nics"]:
            d = devs.get(dev, {})
            txt = (
                f"{dev} / {d.get('ndev', '')}\\n{d.get('pci', '')}\\n{d.get('ip', '')}"
            )
            L.append(f'    {_nid(dev)}[label="{_q(txt)}"];')
        L.append("  }")
    L.append("")

    if probe["mgmt_ndev"]:
        L.append(
            f'  MGMT [label="{_q(probe["mgmt_ndev"])} (mgmt)\\ndefault route'
            '\\n(excluded from rails)", fillcolor="#eeeeee", '
            'style="rounded,filled,dashed"];'
        )
        L.append("")

    L.append(f'  edge [dir=none, color="{GREEN}", penwidth=2.0, label="rail / PCIe"];')
    for p in probe["pairs"]:
        L.append(f"  G{p['gpu']}->{_nid(p['dev'])};")
    L.append("")

    gid = next(
        (d["gid"] for d in probe["devices"] if d["gid"] not in ("", "none")), "?"
    )
    ip = next((d["ip"] for d in probe["devices"] if d["ip"]), "")
    gidtype = next((d["gidtype"] for d in probe["devices"] if d["gidtype"]), "RoCE")
    facts = [
        f"{len(gpu_pci)}x {model}",
        f"{len(probe['rail_devs'])}x rail NIC, {gidtype}, {_fam_label(ip)}"
        f" — GID index {gid}  -> set NCCL_IB_GID_INDEX={gid}",
        f"{len(probe['pairs'])} of {len(gpu_pci)} GPUs pair rail-local (same PCI domain, ordinal)",
    ]
    if probe["mgmt_ndev"]:
        facts.append(
            f"mgmt NIC excluded from rails: {probe['mgmt_ndev']} (default route)"
        )
    excluded = [
        d["dev"]
        for d in probe["devices"]
        if d["dev"] not in probe["rail_devs"] and d["ndev"] != probe["mgmt_ndev"]
    ]
    if excluded:
        facts.append(f"also excluded (no global GID / no netdev): {' '.join(excluded)}")
    if result_text.strip():
        fab, desc, m = verdict_of(result_text)
        facts += [
            "",
            f"Fabric class: {fab} — {desc}",
            f"  - cross-rail IP (ping): {m['ip_cross'][0]}/{m['ip_cross'][1]} OK",
            f"  - cross-rail RDMA (ibv_rc_pingpong): {m['rdma_cross']}",
        ]
    L.append(_legend("Legend / facts", facts))
    L.append("}")
    return "\n".join(L) + "\n"


# ---------- 2-node cross-rail diagram ----------
def crossrail_dot(rm, b_rails, probe, result_text):
    fab, desc, m = verdict_of(result_text)
    a_devs = {i: dev for i, dev, _nd, _g in rm["rails"]}
    b_devs = {i: dev for i, dev, _nd, _f, _g, _a in b_rails} or dict(a_devs)
    b_addr = {i: a for i, _d, _nd, _f, _g, a in b_rails}
    same_ok = {
        s: st == "OK"
        for s, _nd, t, st, kind in rm["ip_rows"]
        if kind == "same-rail" and s == t
    }
    ngpu = len(probe["gpus"]) if probe["gpus"] else 0
    gpu_dev = {p["gpu"]: p["dev"] for p in probe["pairs"]}
    model = probe["gpu_model"] or "GPU"
    nr = rm["nr"] or len(a_devs)
    bnr = rm["bnr"] or len(b_devs)

    L = []
    L.append(f"// 2-node cross-rail fabric: {rm['A']} <-> {rm['B']}")
    L.append("// Generated by make_diagrams.py from the measured cross-rail result.")
    L.append("digraph xrail {")
    L.append(
        '  rankdir=LR; compound=true; labelloc="t"; fontsize=16; fontname="Helvetica";'
    )
    L.append(
        f'  label="2-node RDMA rail fabric: {_q(fab)}  ({_q(rm["A"])} <-> {_q(rm["B"])})";'
    )
    L.append(
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];'
    )
    L.append('  edge [fontname="Helvetica", fontsize=9];')
    L.append("")

    L.append("  subgraph cluster_A {")
    L.append(
        f'    label="Node A (tester): {_q(rm["A"])}"; style="rounded,filled"; '
        'fillcolor="#f2f6ff"; color="#5b8def"; fontsize=13;'
    )
    L.append('    node [fillcolor="#dbe9ff"];')
    for g in range(ngpu):
        L.append(f'    A_G{g}[label="GPU{g}"];')
    L.append('    node [fillcolor="#c9f2d4"];')
    for i in range(nr):
        L.append('    A_N%d[label="%s"];' % (i, _q(a_devs.get(i, "rail%d" % i))))
    L.append("  }")
    L.append("")

    L.append('  node [shape=hexagon, fillcolor="#ffe9a8"];')
    for i in range(max(nr, bnr)):
        sub = b_addr.get(i, "")
        tail = "\\n" + _q(sub) if sub else ""
        L.append('  R%d[label="Rail %d%s"];' % (i, i, tail))
    L.append("")

    L.append("  node [shape=box];")
    L.append("  subgraph cluster_B {")
    L.append(
        f'    label="Node B (target): {_q(rm["B"])}"; style="rounded,filled"; '
        'fillcolor="#f2f6ff"; color="#5b8def"; fontsize=13;'
    )
    L.append('    node [fillcolor="#c9f2d4"];')
    for i in range(bnr):
        L.append('    B_N%d[label="%s"];' % (i, _q(b_devs.get(i, "rail%d" % i))))
    L.append('    node [fillcolor="#dbe9ff"];')
    for g in range(ngpu):
        L.append(f'    B_G{g}[label="GPU{g}"];')
    L.append("  }")
    L.append("")

    # GPU <-> its rail NIC, on both nodes (measured on A; B is the same SKU)
    L.append(f'  edge [dir=none, color="{GREEN}", penwidth=1.5];')
    rail_of = {dev: i for i, dev in a_devs.items()}
    for g in range(ngpu):
        i = rail_of.get(gpu_dev.get(g, ""))
        if i is None:
            continue
        L.append(f"  A_G{g}->A_N{i}; B_G{g}->B_N{i};")
    L.append("")

    L.append("  // same-rail links — solid where the measured same-rail ping passed")
    L.append(f'  edge [dir=none, color="{GREEN}", penwidth=2.2, style=solid];')
    for i in range(min(nr, bnr)):
        if same_ok.get(i, True):
            L.append(f"  A_N{i}->R{i}; R{i}->B_N{i};")
    bad = [i for i in range(min(nr, bnr)) if not same_ok.get(i, True)]
    if bad:
        L.append(f'  edge [color="{RED}", penwidth=2.2, style=dashed];')
        for i in bad:
            L.append(f"  A_N{i}->R{i}; R{i}->B_N{i};")
    L.append("")

    # One representative cross-rail edge, coloured by what was actually measured.
    ip_ok, ip_tot = m["ip_cross"]
    cross_works = m["rdma_cross"] == "OK"
    colour = GREEN if cross_works else RED
    rdma_txt = {"OK": "RDMA REACHABLE", "FAIL": "RDMA UNREACHABLE"}.get(
        m["rdma_cross"], "RDMA not measured"
    )
    lbl = f"cross-rail: ICMP {ip_ok}/{ip_tot} OK, {rdma_txt}"
    L.append("  // cross-rail — colour follows the measurement, not an assumption")
    L.append(
        f'  edge [dir=forward, color="{colour}", penwidth=2.2, style=dashed, '
        f'constraint=false, fontcolor="{colour}"];'
    )
    if nr >= 1 and bnr >= 2:
        L.append(f'  A_N0->B_N1 [label="{_q(lbl)}"];')
    L.append("")

    gid = rm["rails"][0][3] if rm["rails"] else "?"
    facts = [
        f"Node = {ngpu}x {model} : {nr}x rail NIC, 1:1 rail-optimized (PCIe-local)",
        f"RoCEv2 GID index {gid}  -> set NCCL_IB_GID_INDEX={gid}",
        "",
        "GREEN solid  = same-rail (rail i on A <-> rail i on B)",
        f"{'GREEN' if cross_works else 'RED'} dashed  = cross-rail (rail i on A -> rail j on B), representative edge",
        "",
        f"ICMP matrix: same-rail {m['ip_same'][0]}/{m['ip_same'][1]} OK, "
        f"cross-rail {ip_ok}/{ip_tot} OK",
        f"RDMA ibv_rc_pingpong: same-rail {m['rdma_same']}, cross-rail {m['rdma_cross']}",
        f"=> {fab} — {desc}",
    ]
    L.append(_legend("Legend / measured result", facts))
    L.append("}")
    return "\n".join(L) + "\n"


# ---------- rendering ----------
def render(dot_path):
    """Render a .dot to .png with graphviz. Returns the PNG path, or None."""
    png = os.path.splitext(dot_path)[0] + ".png"
    try:
        p = subprocess.run(
            ["dot", "-Tpng", dot_path, "-o", png],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return None
    if p.returncode != 0:
        return None
    return png


# ---------- driver ----------
def generate(folder, force=False):
    """Write the diagrams a folder has data for. Returns the paths written."""
    probe_txt = make_report.read(make_report.find(folder, "topo_report*.txt"))
    result_txt = make_report.read(make_report.find(folder, "*result*.txt"))
    probe = parse_probe(probe_txt) if probe_txt.strip() else None
    written = []

    def emit(name, text):
        dot = os.path.join(folder, name + ".dot")
        png = os.path.join(folder, name + ".png")
        if not force and os.path.isfile(png) and os.path.isfile(dot):
            return
        with open(dot, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(dot)
        out = render(dot)
        if out:
            written.append(out)

    if probe and probe["gpus"]:
        emit(f"topology_{probe['host'] or 'node'}_node", node_dot(probe, result_txt))

    if result_txt.strip():
        rm = parse_railmap(result_txt)
        if rm["A"] and rm["B"]:
            b_rails = parse_addrs(os.path.join(folder, "addrs." + rm["B"]))
            emit(
                f"topology_{rm['A']}__{rm['B']}_crossrail",
                crossrail_dot(rm, b_rails, probe or parse_probe(""), result_txt),
            )
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument(
        "--force", action="store_true", help="regenerate even if a .png already exists"
    )
    a = ap.parse_args()
    written = generate(a.folder, a.force)
    if not written:
        print("nothing to do (no new inputs, or diagrams already present)")
    for w in written:
        print("wrote", w)
    if any(w.endswith(".dot") for w in written) and not any(
        w.endswith(".png") for w in written
    ):
        print("note: graphviz `dot` not available — wrote DOT source only")


if __name__ == "__main__":
    main()
