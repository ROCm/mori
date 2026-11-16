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
"""Check a FlyDSL final ISA dump for what fused compute-communication kernels get wrong.

Step 3 of the ``cco-fusion-loop`` skill: run it after every kernel change and before
any timing. It reads the ``*_final_isa.s`` that ``FLYDSL_DUMP_IR=1
FLYDSL_DUMP_DIR=<dir>`` writes and reports, per kernel:

* resources from the code-object metadata: VGPR, SGPR, spills, scratch, LDS;
* loops, found by their back edges (a branch to an earlier label), and their kind:
  a poll loop (``s_sleep`` of its own), a K loop (MFMAs of its own, possibly with a
  wait loop inside), an outer loop (only loops inside) or another loop;
* every ``buffer_wbl2`` (release) and ``buffer_inv`` (acquire) with the innermost
  loop around it, including those reached through a call made inside a loop;
* vector-memory stores by cache policy: write-through (``sc0 sc1``), non-temporal
  (``nt``) or plain;
* the main K loop: its MFMA count and mnemonics, the position of its first MFMA,
  and the mean distance, in instructions, from a vector-memory load to the
  ``s_waitcnt`` that covers it.

With ``--baseline`` (the same kernel compiled with the communication code off) it
also compares the resources and the K loop.

FAIL: scratch or VGPR spills; an acquire inside any loop; a release inside a K loop
or a poll loop; against the baseline, a different MFMA count or set of mnemonics, or
a mean load-to-wait distance that dropped by more than ``--max-distance-drop``.
WARN: SGPR spills; a release inside another loop; more VGPRs or LDS than the
baseline; a different MFMA order; stores that are not write-through (a FAIL with
``--require-write-through``); a target other than gfx950. ``--strict`` makes every
WARN a FAIL. The facts behind these rules are in
``.claude/skills/cco-fusion-facts/gfx950.md`` (G950-SYNC-01..04, G950-ISA-01).

Exit status: 0 without FAIL, 1 with at least one FAIL, 2 when the dump cannot be
read with confidence (no function body, the kernel is not found, no metadata).

The distance assumes vector-memory operations retire in issue order, the model that
``s_waitcnt vmcnt(N)`` is written against. Use it to compare two builds of one
kernel, not as a latency.

    python isa_check.py <dump-dir-or-file> [--kernel REGEX]
        [--baseline <dump-dir-or-file> [--baseline-kernel REGEX]] [--json out.json]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LABEL_RE = re.compile(r"^([A-Za-z_.$][\w.$]*):")
TYPE_FUNC_RE = re.compile(r"^\s*\.type\s+([^,\s]+)\s*,\s*@function")
COMMENT_RE = re.compile(r"\s*(;|//).*$")
TARGET_RE = re.compile(r"\.amdgcn_target\s+\"[^\"]*?(gfx\w+)")
SET_RE = re.compile(
    r"^\s*\.set\s+(?:\.L)?(\S+)\.(num_vgpr|num_agpr|numbered_sgpr|private_seg_size)\s*,\s*(\d+)"
)
META_START_RE = re.compile(r"^amdhsa\.kernels:\s*$")
META_ENTRY_RE = re.compile(r"^  - \.(\w+):\s*(.*)$")
META_KEY_RE = re.compile(r"^    \.(\w+):\s*(.*)$")
CALLEE_RE = re.compile(r"([A-Za-z_.$][\w.$]*)@(?:gotpcrel32|rel32)@(?:lo|hi)")
VMCNT_RE = re.compile(r"vmcnt\((\d+)\)")
VMEM_RE = re.compile(
    r"^(?:(?:global|buffer|flat|scratch)_(?:load|store|atomic)|buffer_(?:wbl2|inv)$)"
)
LOAD_RE = re.compile(r"^(?:global|buffer|flat)_load")
STORE_RE = re.compile(r"^(?:global|buffer|flat)_store")
MFMA_RE = re.compile(r"^v_(?:mfma|smfmac)")
CACHE_OPS = ("buffer_wbl2", "buffer_inv")
KIND_NAME = {
    "k-loop": "K loop",
    "poll": "poll loop",
    "outer": "outer loop",
    "other": "loop",
}


class DumpError(ValueError):
    """The dump cannot be read with confidence."""


@dataclass
class Instr:
    idx: int
    line: int
    op: str
    text: str

    @property
    def tokens(self):
        return set(re.split(r"[\s,]+", self.text))


@dataclass
class Loop:
    header: str
    start: int
    end: int
    line: int
    parent: "Loop | None" = None
    children: list = field(default_factory=list)
    kind: str = ""
    mfma: int = 0
    sleep: int = 0

    def contains(self, idx):
        return self.start <= idx <= self.end

    def to_json(self):
        return {
            "header": self.header,
            "line": self.line,
            "kind": self.kind,
            "instructions": self.end - self.start + 1,
            "mfma": self.mfma,
            "parent": self.parent.header if self.parent else None,
        }


@dataclass
class Function:
    name: str
    instrs: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    loops: list = field(default_factory=list)
    closed: bool = False

    def innermost_loop(self, idx):
        around = [lp for lp in self.loops if lp.contains(idx)]
        return min(around, key=lambda lp: lp.end - lp.start) if around else None


@dataclass
class Dump:
    path: Path
    target: str | None
    functions: dict
    metadata: dict
    symbols: dict


def resolve_path(path):
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("*final_isa.s"))
        if not found:
            raise DumpError(f"{path}: no *final_isa.s inside")
        if len(found) > 1:
            listing = "\n  ".join(str(p) for p in found)
            raise DumpError(
                f"{path}: more than one *final_isa.s, pass one:\n  {listing}"
            )
        return found[0]
    if not path.is_file():
        raise DumpError(f"{path}: no such file or directory")
    return path


def _value(text):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return text


def parse_dump(path):
    path = resolve_path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    declared, functions, entries, symbols = set(), {}, [], {}
    target, current, entry, in_meta = None, None, None, False

    for lineno, raw in enumerate(lines, 1):
        if target is None:
            m = TARGET_RE.search(raw)
            if m:
                target = m.group(1)
        m = TYPE_FUNC_RE.match(raw)
        if m:
            declared.add(m.group(1))
            continue
        m = SET_RE.match(raw)
        if m:
            symbols.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
            continue
        if META_START_RE.match(raw):
            in_meta = True
            continue
        if in_meta:
            if raw and not raw[0].isspace():
                in_meta = False
            else:
                m = META_ENTRY_RE.match(raw)
                if m:
                    entry = {m.group(1): _value(m.group(2))}
                    entries.append(entry)
                    continue
                m = META_KEY_RE.match(raw)
                if m and entry is not None:
                    entry[m.group(1)] = _value(m.group(2))
                continue

        stripped = raw.strip()
        if not stripped or stripped.startswith((";", "//")):
            continue
        m = LABEL_RE.match(raw)
        if m and not raw[0].isspace():
            label = m.group(1)
            if current is None and label in declared:
                current = Function(label)
                functions[label] = current
            elif current is not None:
                if label.startswith(".Lfunc_end"):
                    current.closed = True
                    current = None
                else:
                    current.labels[label] = len(current.instrs)
            continue
        if current is None or stripped.startswith("."):
            continue
        text = COMMENT_RE.sub("", stripped)
        if text:
            current.instrs.append(
                Instr(len(current.instrs), lineno, text.split(None, 1)[0], text)
            )

    kernels = {e["name"]: e for e in entries if "name" in e}
    for func in functions.values():
        func.loops = find_loops(func)
    return Dump(path, target, functions, kernels, symbols)


def find_loops(func):
    spans = {}
    for ins in func.instrs:
        if not ins.op.startswith(("s_branch", "s_cbranch")):
            continue
        parts = ins.text.split()
        if len(parts) < 2:
            continue
        target = parts[1].rstrip(",")
        start = func.labels.get(target)
        if start is not None and start <= ins.idx:
            spans[target] = (
                start,
                max(ins.idx, spans.get(target, (start, ins.idx))[1]),
            )
    loops = []
    for header, (start, end) in spans.items():
        line = func.instrs[start].line - 1 if start < len(func.instrs) else 0
        loops.append(Loop(header, start, end, line))
    loops.sort(key=lambda lp: (lp.start, -lp.end))
    for lp in loops:
        around = [
            p for p in loops if p is not lp and p.start <= lp.start and lp.end <= p.end
        ]
        if around:
            lp.parent = min(around, key=lambda p: p.end - p.start)
            lp.parent.children.append(lp)
    for lp in loops:
        nested = set()
        for child in lp.children:
            nested.update(range(child.start, child.end + 1))
        for i in range(lp.start, lp.end + 1):
            if i in nested:
                continue
            op = func.instrs[i].op
            lp.mfma += bool(MFMA_RE.match(op))
            lp.sleep += op == "s_sleep"
        if lp.sleep:
            lp.kind = "poll"
        elif lp.mfma:
            lp.kind = "k-loop"
        elif lp.children:
            lp.kind = "outer"
        else:
            lp.kind = "other"
    return loops


def waitcnt_vmcnt(ins):
    if ins.op != "s_waitcnt":
        return None
    m = VMCNT_RE.search(ins.text)
    if m:
        return int(m.group(1))
    parts = ins.text.split(None, 1)
    if len(parts) == 2 and parts[1].strip() in ("0", "0x0"):
        return 0
    return None


def kloop_metrics(func, loop):
    body = func.instrs[loop.start : loop.end + 1]
    size = len(body)
    mfma_ops = [ins.op for ins in body if MFMA_RE.match(ins.op)]
    first = next((k for k, ins in enumerate(body) if MFMA_RE.match(ins.op)), None)
    distances, unresolved = [], 0
    for k, ins in enumerate(body):
        if not LOAD_RE.match(ins.op):
            continue
        issued_after, found = 0, None
        for step in range(1, 2 * size + 1):
            nxt = body[(k + step) % size]
            if VMEM_RE.match(nxt.op):
                issued_after += 1
                continue
            vm = waitcnt_vmcnt(nxt)
            if vm is not None and vm <= issued_after:
                found = step
                break
        if found is None:
            unresolved += 1
        else:
            distances.append(found)
    return {
        "header": loop.header,
        "line": loop.line,
        "instructions": size,
        "mfma": len(mfma_ops),
        "mfma_ops": mfma_ops,
        "first_mfma": first,
        "loads": len(distances) + unresolved,
        "mean_load_wait": sum(distances) / len(distances) if distances else None,
        "loads_not_waited_in_loop": unresolved,
    }


def resources(dump, name):
    meta = dump.metadata.get(name)
    if meta is None:
        raise DumpError(f"{dump.path}: kernel {name} has no amdhsa.kernels metadata")
    sym = dump.symbols.get(name, {})
    return {
        "vgpr": meta.get("vgpr_count"),
        "arch_vgpr": sym.get("num_vgpr"),
        "agpr": meta.get("agpr_count", sym.get("num_agpr")),
        "sgpr": meta.get("sgpr_count"),
        "vgpr_spill": meta.get("vgpr_spill_count"),
        "sgpr_spill": meta.get("sgpr_spill_count"),
        "scratch_bytes": meta.get("private_segment_fixed_size"),
        "lds_bytes": meta.get("group_segment_fixed_size"),
        "max_workgroup": meta.get("max_flat_workgroup_size"),
        "wavefront": meta.get("wavefront_size"),
    }


def callee_of(func, ins):
    for back in range(ins.idx - 1, max(-1, ins.idx - 12), -1):
        m = CALLEE_RE.search(func.instrs[back].text)
        if m:
            return m.group(1)
    return None


class Report:
    def __init__(self, strict):
        self.strict = strict
        self.findings = []

    def add(self, level, code, message, where=None):
        if level == "WARN" and self.strict:
            level = "FAIL"
        self.findings.append(
            {"level": level, "code": code, "message": message, "where": where}
        )

    @property
    def failed(self):
        return any(f["level"] == "FAIL" for f in self.findings)


def _cache_op_level(op, loop):
    if op == "buffer_inv":
        return "FAIL", "an acquire inside a loop runs once per iteration (G950-SYNC-03)"
    if loop.kind in ("k-loop", "poll") or loop.mfma:
        return (
            "FAIL",
            f"a release inside a {KIND_NAME[loop.kind]} writes back the whole L2 every iteration",
        )
    return (
        "WARN",
        "a release inside a loop runs once per iteration: fine once per tile, not per poll",
    )


def direct_cache_ops(func):
    out = []
    for ins in func.instrs:
        if ins.op in CACHE_OPS:
            loop = func.innermost_loop(ins.idx)
            out.append(
                {
                    "op": ins.op,
                    "text": ins.text,
                    "line": ins.line,
                    "function": func.name,
                    "loop": loop.header if loop else None,
                    "loop_kind": loop.kind if loop else None,
                }
            )
    return out


def call_sites(func):
    for ins in func.instrs:
        if ins.op.startswith("s_swappc"):
            yield ins, callee_of(func, ins)


def reachable(dump, func, found=None):
    """``func`` and every function it reaches through calls this dump resolves."""
    found = {} if found is None else found
    if func.name not in found:
        found[func.name] = func
        for _, callee in call_sites(func):
            if callee in dump.functions:
                reachable(dump, dump.functions[callee], found)
    return found


def check_cache_ops(dump, kernel, report):
    """Every cache operation the kernel can execute, judged by the loop around it."""
    funcs = reachable(dump, kernel)
    ops = {name: direct_cache_ops(f) for name, f in funcs.items()}
    for name, func in funcs.items():
        for op in ops[name]:
            if op["loop"]:
                loop = next(lp for lp in func.loops if lp.header == op["loop"])
                level, why = _cache_op_level(op["op"], loop)
                report.add(
                    level,
                    "cache-op-in-loop",
                    f"{op['text']} inside {KIND_NAME[loop.kind]} {loop.header}: {why}",
                    f"{dump.path.name}:{op['line']} ({name})",
                )
        for ins, callee in call_sites(func):
            loop = func.innermost_loop(ins.idx)
            if loop is None:
                continue
            where = f"{dump.path.name}:{ins.line} ({name})"
            if callee not in dump.functions:
                report.add(
                    "WARN",
                    "call-in-loop",
                    f"call to {callee or 'an unresolved target'} inside {KIND_NAME[loop.kind]} {loop.header} "
                    "cannot be followed in this dump",
                    where,
                )
                continue
            kinds = sorted(
                {
                    op["op"]
                    for g in reachable(dump, dump.functions[callee])
                    for op in ops[g]
                }
            )
            for kind in kinds:
                level, why = _cache_op_level(kind, loop)
                report.add(
                    level,
                    "cache-op-via-call-in-loop",
                    f"{callee} reaches {kind} and is called inside {KIND_NAME[loop.kind]} {loop.header}: {why}",
                    where,
                )
    return [op for name in funcs for op in ops[name]]


def store_policy(funcs):
    counts = {"write_through": 0, "non_temporal": 0, "plain": 0, "scratch": 0}
    plain_lines = []
    for ins in (i for f in funcs for i in f.instrs):
        if ins.op.startswith("scratch_store"):
            counts["scratch"] += 1
        elif STORE_RE.match(ins.op):
            tokens = ins.tokens
            if "sc0" in tokens and "sc1" in tokens:
                counts["write_through"] += 1
            elif "nt" in tokens:
                counts["non_temporal"] += 1
                plain_lines.append(ins.line)
            else:
                counts["plain"] += 1
                plain_lines.append(ins.line)
    return counts, plain_lines


def main_kloop(func):
    kloops = [lp for lp in func.loops if lp.kind == "k-loop"]
    return max(kloops, key=lambda lp: lp.mfma) if kloops else None


def select_kernel(dump, pattern):
    names = [n for n in dump.metadata if n in dump.functions]
    if not names:
        raise DumpError(f"{dump.path}: no kernel has both metadata and a function body")
    if pattern:
        names = [n for n in names if re.search(pattern, n)]
        if not names:
            raise DumpError(f"{dump.path}: no kernel matches {pattern!r}")
    return names


def analyze_kernel(dump, name, report, require_write_through):
    func = dump.functions[name]
    res = resources(dump, name)
    where = f"{dump.path.name} ({name})"
    if not func.closed:
        report.add(
            "WARN",
            "truncated",
            "the function has no .Lfunc_end label; the dump may be cut short",
            where,
        )
    if (res["scratch_bytes"] or 0) > 0 or (res["vgpr_spill"] or 0) > 0:
        report.add(
            "FAIL",
            "scratch",
            f"{res['vgpr_spill']} VGPR spills, {res['scratch_bytes']} B scratch per lane: scratch must stay 0",
            where,
        )
    if (res["sgpr_spill"] or 0) > 0:
        report.add(
            "WARN",
            "sgpr-spill",
            f"{res['sgpr_spill']} SGPR spills: check they are outside the loops",
            where,
        )

    cache_ops = check_cache_ops(dump, func, report)
    stores, plain_lines = store_policy(reachable(dump, func).values())
    not_wt = stores["non_temporal"] + stores["plain"]
    if not_wt and stores["write_through"]:
        level = "FAIL" if require_write_through else "WARN"
        report.add(
            level,
            "store-policy",
            f"{not_wt} vector stores are not write-through (sc0 sc1), first at line {min(plain_lines)}; "
            "every store into a cco window must be",
            where,
        )
    elif not_wt and require_write_through:
        report.add(
            "FAIL",
            "store-policy",
            f"no store is write-through (sc0 sc1); {not_wt} are not",
            where,
        )

    kloop = main_kloop(func)
    calls = []
    for f in reachable(dump, func).values():
        for ins, callee in call_sites(f):
            loop = f.innermost_loop(ins.idx)
            calls.append(
                {
                    "line": ins.line,
                    "function": f.name,
                    "callee": callee,
                    "loop": loop.header if loop else None,
                }
            )
    return {
        "name": name,
        "resources": res,
        "loops": [lp.to_json() for lp in func.loops],
        "calls": calls,
        "cache_ops": cache_ops,
        "stores": stores,
        "k_loop": kloop_metrics(func, kloop) if kloop else None,
    }


def compare(cur, base, report, max_drop):
    where = f"{cur['name']} vs {base['name']}"
    for key, label in (("vgpr", "VGPRs"), ("lds_bytes", "LDS bytes")):
        a, b = cur["resources"].get(key), base["resources"].get(key)
        if isinstance(a, int) and isinstance(b, int) and a > b:
            report.add(
                "WARN",
                f"more-{key}",
                f"{label} {b} -> {a}: blocks per CU may drop; check grid residency",
                where,
            )
    k, kb = cur["k_loop"], base["k_loop"]
    delta = {}
    if k is None or kb is None:
        report.add(
            "WARN",
            "k-loop-missing",
            "no K loop on one side; the K-loop comparison is skipped",
            where,
        )
        return delta
    if k["mfma"] != kb["mfma"]:
        report.add(
            "FAIL",
            "k-loop-mfma-count",
            f"K-loop MFMA count {kb['mfma']} -> {k['mfma']}",
            where,
        )
    elif sorted(k["mfma_ops"]) != sorted(kb["mfma_ops"]):
        report.add("FAIL", "k-loop-mfma-ops", "K-loop MFMA mnemonics differ", where)
    elif k["mfma_ops"] != kb["mfma_ops"]:
        report.add("WARN", "k-loop-mfma-order", "K-loop MFMA order differs", where)
    delta["first_mfma"] = None
    if k["first_mfma"] is not None and kb["first_mfma"] is not None:
        delta["first_mfma"] = k["first_mfma"] - kb["first_mfma"]
    delta["mean_load_wait"] = None
    if k["mean_load_wait"] and kb["mean_load_wait"]:
        drop = (kb["mean_load_wait"] - k["mean_load_wait"]) / kb["mean_load_wait"]
        delta["mean_load_wait"] = k["mean_load_wait"] - kb["mean_load_wait"]
        if drop > max_drop:
            report.add(
                "FAIL",
                "k-loop-latency-hiding",
                f"mean load-to-wait distance {kb['mean_load_wait']:.1f} -> {k['mean_load_wait']:.1f} "
                f"({100 * drop:.0f}% shorter; limit {100 * max_drop:.0f}%): the K loop hides less "
                f"latency (G950-ISA-01)"
                + (
                    f"; first MFMA moved {delta['first_mfma']:+d} instructions"
                    if delta["first_mfma"] is not None
                    else ""
                ),
                where,
            )
    return delta


def _text_report(dump, kernels, base_dump, deltas, report):
    out = [f"isa_check: {dump.path}  target {dump.target or '?'}"]
    for k in kernels:
        r = {key: ("?" if v is None else v) for key, v in k["resources"].items()}
        out.append(f"kernel {k['name']}")
        out.append(
            f"  resources  vgpr {r['vgpr']} (arch {r['arch_vgpr']}, agpr {r['agpr']})  sgpr {r['sgpr']}  "
            f"spills v{r['vgpr_spill']}/s{r['sgpr_spill']}  scratch {r['scratch_bytes']} B  lds {r['lds_bytes']} B"
        )
        kinds = {}
        for lp in k["loops"]:
            kinds[lp["kind"]] = kinds.get(lp["kind"], 0) + 1
        out.append(
            f"  loops      {len(k['loops'])}: "
            + (", ".join(f"{v} {n}" for n, v in sorted(kinds.items())) or "none")
        )
        calls_in_loop = sum(1 for c in k["calls"] if c["loop"])
        out.append(f"  calls      {len(k['calls'])} ({calls_in_loop} inside a loop)")
        in_loop = sum(1 for c in k["cache_ops"] if c["loop"])
        out.append(f"  cache ops  {len(k['cache_ops'])} ({in_loop} inside a loop)")
        s = k["stores"]
        out.append(
            f"  stores     write-through {s['write_through']}, non-temporal {s['non_temporal']}, "
            f"plain {s['plain']}, scratch {s['scratch']}"
        )
        kl = k["k_loop"]
        if kl:
            mean = (
                f"{kl['mean_load_wait']:.1f}"
                if kl["mean_load_wait"] is not None
                else "n/a"
            )
            out.append(
                f"  K loop     {kl['header']} (line {kl['line']}): {kl['instructions']} instructions, "
                f"{kl['mfma']} MFMA, first MFMA at +{kl['first_mfma']}, {kl['loads']} loads, "
                f"mean load-to-wait {mean} ({kl['loads_not_waited_in_loop']} waited after the loop)"
            )
        else:
            out.append("  K loop     none found")
    if base_dump is not None:
        out.append(f"baseline: {base_dump.path}")
        if deltas:
            out.append(
                f"  delta      first MFMA {deltas.get('first_mfma')}, mean load-to-wait {deltas.get('mean_load_wait')}"
            )
    for f in report.findings:
        out.append(
            f"{f['level']:4}  {f['code']}: {f['message']}"
            + (f"  [{f['where']}]" if f["where"] else "")
        )
    fails = sum(f["level"] == "FAIL" for f in report.findings)
    warns = sum(f["level"] == "WARN" for f in report.findings)
    out.append(f"result: {'FAIL' if fails else 'PASS'} ({fails} FAIL, {warns} WARN)")
    return "\n".join(out) + "\n"


def run(args):
    report = Report(args.strict)
    dump = parse_dump(args.dump)
    if dump.target and dump.target != "gfx950":
        report.add(
            "WARN",
            "target",
            f"target {dump.target}: these rules and facts are for gfx950",
            str(dump.path.name),
        )
    names = select_kernel(dump, args.kernel)
    kernels = [
        analyze_kernel(dump, n, report, args.require_write_through) for n in names
    ]

    base_dump, deltas = None, {}
    if args.baseline:
        base_dump = parse_dump(args.baseline)
        base_names = select_kernel(base_dump, args.baseline_kernel)
        if len(names) != 1 or len(base_names) != 1:
            raise DumpError(
                "--baseline compares one kernel with one kernel; use --kernel / --baseline-kernel"
            )
        base = analyze_kernel(base_dump, base_names[0], Report(False), False)
        deltas = compare(kernels[0], base, report, args.max_distance_drop)

    payload = {
        "dump": str(dump.path),
        "target": dump.target,
        "kernels": kernels,
        "baseline": str(base_dump.path) if base_dump else None,
        "delta": deltas,
        "findings": report.findings,
        "result": "FAIL" if report.failed else "PASS",
    }
    return payload, _text_report(dump, kernels, base_dump, deltas, report), report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check a FlyDSL final ISA dump of a fused kernel."
    )
    parser.add_argument(
        "dump", help="a *final_isa.s file, or a dump directory holding one"
    )
    parser.add_argument("--kernel", help="regex selecting the kernel(s) to check")
    parser.add_argument(
        "--baseline", help="the same kernel compiled with communication off"
    )
    parser.add_argument("--baseline-kernel", help="regex selecting the baseline kernel")
    parser.add_argument(
        "--max-distance-drop",
        type=float,
        default=0.10,
        help="largest allowed drop of the mean load-to-wait distance vs the baseline (default 0.10)",
    )
    parser.add_argument(
        "--require-write-through",
        action="store_true",
        help="FAIL on stores without sc0 sc1",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat every WARN as a FAIL"
    )
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args(argv)

    try:
        payload, text, report = run(args)
    except (DumpError, OSError) as e:
        print(f"isa_check.py: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
