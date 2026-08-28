#!/usr/bin/env python3
"""Generate a self-contained HTML report from a cluster-topology output folder.

Per-folder report:
    python3 make_report.py <output_dir> [--title TITLE] [--out report.html]

Combined comparison index across several folders:
    python3 make_report.py --index <index.html> <dir1> <dir2> ...

The report embeds the diagrams (base64, so the HTML is a single shareable file),
renders the summary .md, and includes the raw cross-rail result + probe report in
collapsible sections. No third-party dependencies.

File discovery (by glob within the folder):
  *node*.png (not crossrail)  -> single-node diagram
  *crossrail*.png             -> 2-node cross-rail diagram
  topology_*.md / *.md        -> summary
  *result*.txt                -> cross-rail result
  topo_report*.txt            -> probe report
"""
import sys, os, re, glob, base64, html, argparse

# ---------- helpers ----------
def find(folder, pat, exclude=None):
    hits = sorted(glob.glob(os.path.join(folder, pat)))
    if exclude:
        hits = [h for h in hits if exclude not in os.path.basename(h)]
    return hits[0] if hits else None

def read(path):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

def img_data_uri(path):
    if not path or not os.path.isfile(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return "data:image/png;base64," + b

def detect_gpu(text):
    m = re.search(r"MI\d{3}X(?:\s*VF)?", text)
    return m.group(0) if m else "GPU"

def parse_result(text):
    """From a cross-rail result.txt, summarize both experiments as same/cross-rail status.
    Returns dict: rdma_same, rdma_cross ('OK'/'FAIL'/'-'), ip_same, ip_cross (ok,total)."""
    rdma_same = rdma_cross = "-"
    ip_same = [0, 0]; ip_cross = [0, 0]
    for ln in text.splitlines():
        s = ln.strip()
        low = s.lower()
        if low.startswith("same-rail") and ":" in s:
            rdma_same = "OK" if ("REACHABLE" in s and "UNREACHABLE" not in s) else ("FAIL" if "UNREACHABLE" in s else "-")
        elif low.startswith("cross-rail") and ":" in s:
            if "SKIPPED" in s: rdma_cross = "-"
            else: rdma_cross = "OK" if ("REACHABLE" in s and "UNREACHABLE" not in s) else ("FAIL" if "UNREACHABLE" in s else "-")
        else:
            m = re.match(r"^rail\d+\S*\s+rail\d+\s+(\S+)\s+(same-rail|cross-rail)", s)
            if m:
                ok = 1 if m.group(1) == "OK" else 0
                (ip_same if m.group(2) == "same-rail" else ip_cross)[0] += ok
                (ip_same if m.group(2) == "same-rail" else ip_cross)[1] += 1
    return {"rdma_same": rdma_same, "rdma_cross": rdma_cross, "ip_same": ip_same, "ip_cross": ip_cross}

def ip_cell(pair):
    o, t = pair
    if t == 0: return ("n/a", "muted")
    if o == t: return (f"OK ({o}/{t})", "ok")
    if o == 0: return (f"FAIL (0/{t})", "bad")
    return (f"mixed ({o}/{t})", "warn")

def rdma_cell(v):
    return {"OK": ("OK", "ok"), "FAIL": ("FAIL", "bad")}.get(v, ("n/a", "muted"))

def detect_fabric(text):
    # Prefer the explicit "Fabric classification:" heading so we don't get fooled by
    # a comparison table that mentions every fabric type.
    m = re.search(r"Fabric\s+class(?:ification)?:\s*(.+)", text, re.I)
    scope = m.group(1).lower() if m else text.lower()
    if "full-mesh" in scope:
        return ("FULL-MESH", "cross-rail RDMA works (all-to-all / EP supported)", "ok")
    if "both" in scope or "100% loss" in scope:
        return ("RAIL-ONLY (IP + RDMA)", "rails fully isolated; no cross-rail at any layer", "bad")
    if "rail-only" in scope or "rdma" in scope:
        return ("RDMA RAIL-ONLY", "IP routable cross-rail, but RDMA cross-rail fails", "warn")
    return ("UNDETERMINED", "see details", "warn")

# ---------- minimal markdown -> HTML ----------
def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s

def md_to_html(md):
    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i])); i += 1
            i += 1
            out.append("<pre class='code'>" + "\n".join(buf) + "</pre>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>")
            i += 1; continue
        if ln.strip().startswith("|") and "|" in ln.strip()[1:]:
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            out.append(render_table(tbl)); continue
        if re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append("<li>" + md_inline(re.sub(r"^\s*[-*]\s+", "", lines[i])) + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>"); continue
        if ln.strip() == "":
            i += 1; continue
        para = []
        while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith(("#", "|", "```")) \
                and not re.match(r"^\s*[-*]\s+", lines[i]):
            para.append(lines[i]); i += 1
        out.append("<p>" + md_inline(" ".join(para)) + "</p>")
    return "\n".join(out)

def render_table(rows):
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]
    if len(rows) >= 2 and set(rows[1].replace("|", "").strip()) <= set("-: "):
        head, body = cells(rows[0]), rows[2:]
    else:
        head, body = cells(rows[0]), rows[1:]
    h = "".join(f"<th>{md_inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells(r)) + "</tr>" for r in body)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

# ---------- styling ----------
CSS = """
:root{--fg:#1c2530;--muted:#5c6b7a;--line:#e2e8f0;--bg:#f7f9fc;--card:#fff;
--ok:#1f8a4c;--okbg:#e5f6ec;--warn:#b7791f;--warnbg:#fdf3e2;--bad:#c0392b;--badbg:#fbe9e7;--accent:#2f6fed;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:.2em 0}h2{font-size:20px;margin:1.4em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.25em}
h3{font-size:16px;margin:1.1em 0 .4em}h4{font-size:14px;color:var(--muted);margin:1em 0 .3em}
p{margin:.5em 0}code{background:#eef1f6;padding:.1em .35em;border-radius:4px;font:13px ui-monospace,Menlo,Consolas,monospace}
pre.code{background:#0f172a;color:#e2e8f0;padding:12px 14px;border-radius:8px;overflow:auto;font:12.5px ui-monospace,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;margin:.6em 0;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;font-size:13.5px}
th{background:#eef2f8}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
figure{margin:0 0 8px}figure img{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
.chip{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:12.5px}
.chip.ok{background:var(--okbg);color:var(--ok)}.chip.warn{background:var(--warnbg);color:var(--warn)}.chip.bad{background:var(--badbg);color:var(--bad)}
.verdict{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.verdict .sub{color:var(--muted)}
details{margin:.5em 0}summary{cursor:pointer;font-weight:600;color:var(--accent)}
pre.raw{background:#0f172a;color:#dbe4f0;padding:12px 14px;border-radius:8px;overflow:auto;font:12px ui-monospace,Menlo,Consolas,monospace;max-height:420px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.foot{color:var(--muted);font-size:12.5px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
a{color:var(--accent)}
.c-ok{color:var(--ok);font-weight:600}.c-bad{color:var(--bad);font-weight:600}.c-warn{color:var(--warn);font-weight:600}.c-muted{color:var(--muted)}
td.ok{background:var(--okbg)}td.bad{background:var(--badbg)}td.warn{background:var(--warnbg)}
"""

def page(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">{body}</div></body></html>"""

# ---------- per-folder report ----------
def build_report(folder, title=None):
    md_path = find(folder, "topology_*.md") or find(folder, "*.md")
    node_png = find(folder, "*node*.png", exclude="crossrail") or find(folder, "*node*.png")
    xr_png = find(folder, "*crossrail*.png")
    result = find(folder, "*result*.txt")
    probe = find(folder, "topo_report*.txt")

    md = read(md_path)
    res_txt = read(result)
    title = title or (re.search(r"^#\s+(.*)$", md, re.M).group(1) if re.search(r"^#\s+(.*)$", md, re.M) else os.path.basename(os.path.abspath(folder)))
    gpu = detect_gpu(md)
    fab, fab_desc, fab_cls = detect_fabric(md + "\n" + res_txt)

    b = []
    b.append(f"<h1>{html.escape(title)}</h1>")
    b.append(f"""<div class="card"><div class="verdict">
      <span class="chip {fab_cls}">Fabric: {html.escape(fab)}</span>
      <span class="chip warn" style="background:#eef2f8;color:#2f6fed">GPU: {html.escape(gpu)}</span>
      <span class="sub">{html.escape(fab_desc)}</span></div></div>""")

    # verdict evidence table — both experiments, same-rail vs cross-rail
    if res_txt.strip():
        r = parse_result(res_txt)
        def td(pair_or_val, is_ip):
            txt, cls = ip_cell(pair_or_val) if is_ip else rdma_cell(pair_or_val)
            return f"<td class='{cls}'>{txt}</td>"
        b.append(f"""<h2>Verdict — measured evidence</h2><div class="card">
          <table><thead><tr><th>Experiment</th><th>Same-rail</th><th>Cross-rail</th></tr></thead><tbody>
          <tr><td>IP reachability &mdash; ICMP <code>ping</code></td>{td(r['ip_same'],True)}{td(r['ip_cross'],True)}</tr>
          <tr><td>RDMA &mdash; <code>ibv_rc_pingpong</code> (RoCEv2)</td>{td(r['rdma_same'],False)}{td(r['rdma_cross'],False)}</tr>
          </tbody></table>
          <p class="sub">Fabric verdict: <strong>{html.escape(fab)}</strong> &mdash; {html.escape(fab_desc)}.
          Cross-rail must pass at the <em>RDMA</em> layer for all-to-all / expert-parallel to work;
          IP (ping) passing alone is not sufficient.</p></div>""")

    figs = []
    if node_png:
        figs.append(f'<figure><img src="{img_data_uri(node_png)}"><figcaption>Single-node GPU &harr; rail NIC topology</figcaption></figure>')
    if xr_png:
        figs.append(f'<figure><img src="{img_data_uri(xr_png)}"><figcaption>2-node cross-rail fabric</figcaption></figure>')
    if figs:
        b.append("<h2>Diagrams</h2>" + "".join(f'<div class="card">{f}</div>' for f in figs))

    if md:
        b.append('<h2>Summary</h2><div class="card">' + md_to_html(md) + "</div>")

    raw = []
    if result:
        raw.append(f"<details><summary>Cross-rail result ({html.escape(os.path.basename(result))})</summary><pre class='raw'>{html.escape(read(result))}</pre></details>")
    if probe:
        raw.append(f"<details><summary>Probe report ({html.escape(os.path.basename(probe))})</summary><pre class='raw'>{html.escape(read(probe))}</pre></details>")
    if raw:
        b.append("<h2>Raw data</h2><div class='card'>" + "".join(raw) + "</div>")

    b.append('<div class="foot">Generated by cluster-rdma-topology <code>make_report.py</code> — self-contained (images embedded).</div>')
    return page(title, "\n".join(b))

# ---------- combined index ----------
def build_index(folders, out_dir):
    cards, rows = [], []
    for folder in folders:
        md = read(find(folder, "topology_*.md") or find(folder, "*.md"))
        res_txt = read(find(folder, "*result*.txt"))
        title = re.search(r"^#\s+(.*)$", md, re.M)
        title = title.group(1) if title else os.path.basename(os.path.abspath(folder))
        gpu = detect_gpu(md)
        fab, fab_desc, cls = detect_fabric(md + "\n" + res_txt)
        r = parse_result(res_txt)
        ic_txt, ic_cls = ip_cell(r["ip_cross"])
        rc_txt, rc_cls = rdma_cell(r["rdma_cross"])
        xr = find(folder, "*crossrail*.png")
        rel = os.path.relpath(os.path.join(folder, "report.html"), out_dir)
        rows.append(f"<tr><td><a href='{html.escape(rel)}'>{html.escape(title)}</a></td><td>{html.escape(gpu)}</td>"
                    f"<td><span class='chip {cls}'>{html.escape(fab)}</span></td>"
                    f"<td class='{ic_cls}'>{ic_txt}</td><td class='{rc_cls}'>{rc_txt}</td></tr>")
        img = f'<img src="{img_data_uri(xr)}">' if xr else ""
        cards.append(f"<div class='card'><h3><a href='{html.escape(rel)}'>{html.escape(title)}</a></h3>"
                     f"<div class='verdict'><span class='chip {cls}'>{html.escape(fab)}</span>"
                     f"<span class='sub'>{html.escape(gpu)} &mdash; {html.escape(fab_desc)}</span></div><figure>{img}</figure></div>")
    body = ["<h1>Cluster RDMA / GPU Topology — Comparison</h1>",
            "<div class='card'><table><thead><tr><th>Cluster</th><th>GPU</th><th>Fabric verdict</th>"
            "<th>Cross-rail IP (ping)</th><th>Cross-rail RDMA</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            "<p class='sub'>Green = works, red = fails. A fabric supports all-to-all / expert-parallel"
            " only when <strong>cross-rail RDMA</strong> is green.</p></div>",
            "<h2>Clusters</h2>", "<div class='grid'>" + "".join(cards) + "</div>",
            '<div class="foot">Generated by cluster-rdma-topology <code>make_report.py --index</code>.</div>']
    return page("Cluster Topology Comparison", "\n".join(body))

# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--title")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--index", metavar="INDEX_HTML")
    a = ap.parse_args()
    if a.index:
        out_dir = os.path.dirname(os.path.abspath(a.index)) or "."
        open(a.index, "w", encoding="utf-8").write(build_index(a.paths, out_dir))
        print("wrote", a.index)
    else:
        folder = a.paths[0]
        out = a.out if os.path.isabs(a.out) else os.path.join(folder, a.out)
        open(out, "w", encoding="utf-8").write(build_report(folder, a.title))
        print("wrote", out)

if __name__ == "__main__":
    main()
