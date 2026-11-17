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
"""Run PyTorch's own symm_mem test file against a symm_mem backend (MORI by default).

Each test runs in its own pytest process and process session: torch's
MultiProcContinuousTest poisons a whole class after one failure, and a hung peer must
not take the next test down with it. Every result is matched against expected.tsv, the
documented list of tests that cannot pass on this backend or platform; anything else
that fails is reported as unexpected and makes the exit status non-zero.

Typical use (inside an environment with torch, mori and pytest-timeout installed):

    python run_conformance.py --fetch --out /tmp/conf --gpus 0,1,2,3

See README.md for the views, the expectation format and how to read the report.
"""

import argparse
import fnmatch
import json
import os
import queue
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_FILE = "test_symmetric_memory.py"
RAW_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/{ref}/test/distributed/"
    + TEST_FILE
)
FAILING = ("FAILED", "ERROR", "TIMEOUT", "NO_RESULT")


def torch_info(python):
    code = (
        "import json, sys, torch; print(json.dumps({'version': torch.__version__, "
        "'git': torch.version.git_version, 'hip': torch.version.hip, 'prefix': sys.prefix}))"
    )
    out = subprocess.run(
        [python, "-c", code], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetch_test_file(git_ref, dest):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / TEST_FILE
    with urllib.request.urlopen(RAW_URL.format(ref=git_ref), timeout=60) as r:
        path.write_bytes(r.read())
    return path


def base_env(args, lane=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(HERE / "site"), env.get("PYTHONPATH")) if p
    )
    env.update(
        MORI_CONFORMANCE="1",
        MORI_CONFORMANCE_VIEW=args.view,
        MORI_CONFORMANCE_BACKEND=args.backend,
        MORI_CONFORMANCE_AS_CUDA="1" if args.as_cuda else "0",
        MORI_CONFORMANCE_PREFIX=args.prefix,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.setdefault("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "90")
    if lane is not None:
        env["HIP_VISIBLE_DEVICES"] = lane
    return env


def collect(args, test_file):
    cmd = [
        args.python,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    out = subprocess.run(
        cmd + [test_file.name],
        cwd=test_file.parent,
        env=base_env(args, args.gpus[0]),
        capture_output=True,
        text=True,
    )
    ids = [x.strip() for x in out.stdout.splitlines() if x.startswith(TEST_FILE + "::")]
    if not ids:
        sys.exit(
            f"collection found no tests:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
        )
    return ids


def warm_up(args, lane):
    """Select the backend once up front: a broken build fails here, not in every test."""
    code = (
        "import torch.distributed._symmetric_memory as s; "
        "print('backend', s.get_backend('cuda'))"
    )
    env = base_env(args, lane)
    env["MORI_CONFORMANCE_AS_CUDA"] = "0"
    out = subprocess.run(
        [args.python, "-c", code], env=env, capture_output=True, text=True
    )
    if out.returncode != 0 or f"backend {args.backend}" not in out.stdout:
        sys.exit(
            f"cannot select backend {args.backend}:\n{out.stdout}\n{out.stderr[-4000:]}"
        )


def gpus_idle(lane, max_use):
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True
    use = {
        int(g): int(u)
        for g, u in re.findall(r"GPU\[(\d+)\]\s*: GPU use \(%\): (\d+)", out)
    }
    return all(use.get(int(g), 0) < max_use for g in lane.split(","))


def session_pids(sid):
    """Live (non-zombie) processes of session `sid`."""
    pids = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        # After "comm)": state ppid pgrp session ...
        if int(fields[3]) == sid and fields[0] != "Z":
            pids.append(int(stat.parent.name))
    return pids


def reap_session(sid, grace):
    """Give the test's session `grace` seconds to empty, then kill what is left."""
    deadline = time.time() + grace
    while (left := session_pids(sid)) and time.time() < deadline:
        time.sleep(0.5)
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return left


def safe_name(test_id):
    cls, name = test_id.split("::")[1:3]
    return f"{cls}__{re.sub(r'[^A-Za-z0-9_]', '_', name)}"


SKIP_LINES = (
    "RuntimeError: Process ",
    "RuntimeError: Exception in worker",
    "RuntimeError: Exited with",
    "AssertionError: Scalars are not equal",
    "AssertionError: Tensor-likes",
)
ERR_LINE = re.compile(
    r"^(?:E\s+)?(?:\[rank\d+\]:\s*)?((?:[\w.]+\.)?\w*(?:Error|Exception|Timeout|Failed)\b:?[^\n]*)$",
    re.M,
)


def root_cause(text):
    """The most telling error line of a log: a worker's own exception over the harness's."""
    worker = "\n".join(
        re.findall(r"\[conformance worker[^\n]*\n(?:[^\n]*\n){0,80}", text)
    )
    for source in (worker, text):
        hits = [m.group(1).strip() for m in ERR_LINE.finditer(source)]
        good = [h for h in hits if not h.startswith(SKIP_LINES) and len(h) > 12]
        pref = [
            h
            for h in good
            if re.search(
                r"mori|hip|NYI|backend|not implemented|cco|channel|signal|group",
                h,
                re.I,
            )
        ]
        pick = (pref or good or hits or [""])[-1]
        if pick:
            return pick[:300]
    return ""


def result_of(xml_path, log_text):
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return "NO_RESULT", root_cause(log_text)
    for case in root.iter("testcase"):
        for tag, status in (("failure", "FAILED"), ("error", "ERROR")):
            e = case.find(tag)
            if e is not None:
                msg = root_cause(log_text) or root_cause(
                    (e.text or "") + "\n" + (e.get("message") or "")
                )
                return status, msg
        s = case.find("skipped")
        if s is not None:
            msg = s.get("message") or ""
            # MultiProcessTestCase reports a worker's skip generically; its log has the reason.
            sub = re.search(
                r"skipping test \S+ for following reason: ([^\n]*)", log_text
            )
            if "skipped at subprocess level" in msg and sub:
                msg = sub.group(1)
            return "SKIPPED", msg.replace("\n", " ")[:200]
        return "PASSED", ""
    return "NO_RESULT", root_cause(log_text)


def run_one(args, test_id, lane, out, has_timeout_plugin):
    name = safe_name(test_id)
    log, xml = out / "logs" / f"{name}.log", out / "logs" / f"{name}.xml"
    lifted = out / "lifted" / f"{name}.txt"
    for p in (xml, lifted):
        p.unlink(missing_ok=True)
    env = base_env(args, lane)
    env["MORI_CONFORMANCE_LIFTED"] = str(lifted)
    cmd = [args.python, "-m", "pytest", "-s", "-rA", "-p", "no:cacheprovider"]
    if has_timeout_plugin:
        cmd.append(f"--timeout={args.per_test_timeout}")
    cmd += [f"--junitxml={xml}", test_id]
    start = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=args.test_dir,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        killed = False
        try:
            proc.wait(timeout=args.per_test_timeout + 60)
        except subprocess.TimeoutExpired:
            killed = True
            for pid in [proc.pid] + session_pids(proc.pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
    # Anything still in the test's session after a grace period (multiprocessing's
    # resource tracker needs a few seconds) is a leftover: a hung worker, typically.
    left = reap_session(proc.pid, grace=10)
    status, cause = result_of(xml, log.read_text(errors="replace"))
    if killed:
        status = "TIMEOUT"
    return {
        "id": test_id,
        "status": status,
        "lifted": int(lifted.exists() and lifted.stat().st_size > 0),
        "seconds": int(time.time() - start),
        "leftovers": len(left),
        "cause": cause.replace("\t", " ").replace("\n", " "),
    }


COLUMNS = ("id", "status", "lifted", "seconds", "leftovers", "cause")


def read_summary(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == len(COLUMNS) and parts[0] != "id":
                rows[parts[0]] = dict(zip(COLUMNS, parts))
    return rows


def run_all(args, ids, out):
    summary = out / "summary.tsv"
    done = read_summary(summary) if args.resume else {}
    if not done:
        summary.write_text("\t".join(COLUMNS) + "\n")
    todo = queue.Queue()
    for i in ids:
        if i not in done:
            todo.put(i)
    probe = subprocess.run(
        [args.python, "-c", "import pytest_timeout"], capture_output=True
    )
    has_timeout_plugin = probe.returncode == 0
    lock = threading.Lock()

    def lane_worker(lane):
        while True:
            try:
                test_id = todo.get_nowait()
            except queue.Empty:
                return
            while args.wait_idle and not gpus_idle(lane, args.wait_idle):
                time.sleep(30)
            row = run_one(args, test_id, lane, out, has_timeout_plugin)
            with lock:
                with open(summary, "a") as f:
                    f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
                print(f"{row['status']:8s} {row['seconds']:4d}s {test_id}", flush=True)

    threads = [threading.Thread(target=lane_worker, args=(lane,)) for lane in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class Rule:
    """One line of expected.tsv: status, torch, view, test, cause, category, note."""

    def __init__(self, fields):
        self.status, self.torch, self.view, self.test, cause = fields[:5]
        self.category, self.note = fields[5:]
        self.cause = re.compile(cause)

    def applies(self, view):
        return self.view == "*" or view in self.view

    def matches(self, row):
        test = row["id"].split("::", 1)[1]
        status = row["status"]
        if self.status != "ANY" and not (
            (self.status == "FAIL" and status in ("FAILED", "ERROR"))
            or (self.status == "SKIP" and status == "SKIPPED")
            or self.status == status
        ):
            return False
        return fnmatch.fnmatchcase(test, self.test) and bool(
            self.cause.search(row["cause"])
        )


def load_expected(path, torch_version):
    rules = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            sys.exit(f"{path}:{n}: expected 7 tab-separated fields, got {len(fields)}")
        if fields[0] not in ("FAIL", "SKIP", "TIMEOUT", "ANY"):
            sys.exit(f"{path}:{n}: unknown status {fields[0]!r}")
        if fields[2] != "*" and (not fields[2] or set(fields[2]) - set("ABC")):
            sys.exit(
                f"{path}:{n}: view must be '*' or letters of ABC, got {fields[2]!r}"
            )
        if fnmatch.fnmatch(torch_version, fields[1]):
            rules.append(Rule(fields))
    return rules


def classify(row, rules):
    return next((r.category for r in rules if r.matches(row)), None)


def short(test_id):
    return test_id.split("::", 1)[1]


def reparse(out, rows):
    """Re-read each test's log and XML, so a better parser also fixes older runs."""
    for test_id, r in rows.items():
        name = safe_name(test_id)
        log = out / "logs" / f"{name}.log"
        if r["status"] == "TIMEOUT" or not log.exists():
            continue
        status, cause = result_of(
            out / "logs" / f"{name}.xml", log.read_text(errors="replace")
        )
        r["status"], r["cause"] = status, cause.replace("\t", " ").replace("\n", " ")


def report(args, out, info):
    rows = read_summary(out / "summary.tsv")
    if args.report_only:
        reparse(out, rows)
    # A rule scoped to other views must not excuse a skip in this one.
    rules = [
        r for r in load_expected(args.expected, info["version"]) if r.applies(args.view)
    ]
    views = {args.view: {i: r["status"] for i, r in rows.items()}}
    if args.view == "C":
        # View B differs from C only by the skips C lifts, which the tests recorded.
        views["B"] = {
            i: ("SKIPPED" if r["lifted"] == "1" else r["status"])
            for i, r in rows.items()
        }
    order = ("PASSED", "FAILED", "ERROR", "TIMEOUT", "NO_RESULT", "SKIPPED")
    lines = [
        f"torch {info['version']} (git {info['git']}, hip {info['hip']}), "
        f"backend {args.backend}{' (as CUDA)' if args.as_cuda else ''}, view {args.view}, "
        f"{len(rows)} tests",
        "",
    ]
    for view in sorted(views):
        c = Counter(views[view].values())
        lines.append(
            f"view {view}: " + ", ".join(f"{c[s]} {s.lower()}" for s in order if c[s])
        )
    expected, unexpected, unexpected_skips, stale = defaultdict(list), [], [], []
    for test_id, r in sorted(rows.items()):
        cat = classify(r, rules)
        if r["status"] in FAILING:
            (expected[cat].append(r) if cat else unexpected.append(r))
        elif r["status"] == "SKIPPED":
            (expected[cat].append(r) if cat else unexpected_skips.append(r))
        elif r["status"] == "PASSED":
            hit = next(
                (
                    x
                    for x in rules
                    if x.status == "FAIL"
                    and x.test != "*"
                    and fnmatch.fnmatchcase(short(test_id), x.test)
                ),
                None,
            )
            if hit:
                stale.append((r, hit.category))
    lines += ["", f"expected failures and skips ({sum(map(len, expected.values()))}):"]
    for cat, rs in sorted(expected.items(), key=lambda x: (-len(x[1]), x[0])):
        st = Counter(r["status"] for r in rs)
        lines.append(
            f"  [{len(rs)}] {cat} ({', '.join(f'{n} {s.lower()}' for s, n in st.items())})"
        )
        if any(r["status"] != "SKIPPED" for r in rs) or args.verbose:
            lines += [f"      {r['status']:8s} {short(r['id'])}" for r in rs]
    lines += ["", f"UNEXPECTED failures ({len(unexpected)}):"]
    lines += [
        f"  {r['status']:8s} {short(r['id'])}  -- {r['cause']}" for r in unexpected
    ]
    lines += ["", f"UNEXPECTED skips ({len(unexpected_skips)}):"]
    lines += [f"  {short(r['id'])}  -- {r['cause']}" for r in unexpected_skips]
    lines += ["", f"expected to fail but passed ({len(stale)}; update expected.tsv):"]
    lines += [f"  {short(r['id'])}  [{cat}]" for r, cat in stale]
    left = [r for r in rows.values() if r["leftovers"] not in ("0", 0)]
    lines += ["", f"tests that left processes behind ({len(left)}):"]
    lines += [f"  {r['leftovers']:>3s}  {short(r['id'])}" for r in left]
    if args.baseline:
        base = read_baseline(args.baseline)
        changed = [
            (i, base[i], views[args.view][i])
            for i in sorted(views[args.view])
            if i in base and base[i] != views[args.view][i]
        ]
        lines += ["", f"changes against {args.baseline} ({len(changed)}):"]
        lines += [f"  {a:8s} -> {b:8s} {short(i)}" for i, a, b in changed]
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)
    return 1 if unexpected or unexpected_skips else 0


def read_baseline(path):
    """id -> status from an earlier summary.tsv (first two columns)."""
    base = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and "::" in parts[0]:
            base[parts[0]] = parts[1]
    return base


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter of the environment under test",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--test-file", type=Path, help=f"a local copy of torch's {TEST_FILE}"
    )
    src.add_argument(
        "--fetch",
        action="store_true",
        help="download it for the installed torch's git commit",
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--view",
        choices=("A", "B", "C"),
        default="C",
        help="ROCm skips to lift (README)",
    )
    p.add_argument("--backend", default="MORI")
    p.add_argument(
        "--as-cuda",
        action="store_true",
        help='get_backend() reports "CUDA": runs tests gated on that name; use with -k (README)',
    )
    p.add_argument(
        "--gpus",
        action="append",
        help="GPU list of one lane, e.g. 0,1,2,3; repeat for parallel lanes",
    )
    p.add_argument("--ids", type=Path, help="run only the test ids listed in this file")
    p.add_argument(
        "-k",
        dest="match",
        action="append",
        help="run only test ids containing this substring (repeat for any of several)",
    )
    p.add_argument("--per-test-timeout", type=int, default=300)
    p.add_argument(
        "--wait-idle",
        type=int,
        metavar="PCT",
        help="before each test, wait until the lane's GPUs are below PCT%% use",
    )
    p.add_argument("--expected", type=Path, default=HERE / "expected.tsv")
    p.add_argument(
        "--baseline", help="earlier summary.tsv to list status changes against"
    )
    p.add_argument(
        "--resume", action="store_true", help="keep results already in OUT/summary.tsv"
    )
    p.add_argument(
        "--report-only", action="store_true", help="rebuild the report from OUT"
    )
    p.add_argument(
        "-v", dest="verbose", action="store_true", help="list skipped tests too"
    )
    args = p.parse_args()
    args.gpus = args.gpus or ["0,1,2,3"]

    out = args.out.resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "lifted").mkdir(exist_ok=True)
    info = torch_info(args.python)
    args.prefix = info["prefix"]
    # A trapped kernel makes ROCm write a GPU core file of GBs and abort; without core files
    # it surfaces as the catchable error torch's timeout tests expect.
    resource.setrlimit(
        resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1])
    )
    (out / "torch.json").write_text(json.dumps(info) + "\n")
    if args.report_only:
        sys.exit(report(args, out, info))

    if args.fetch:
        test_file = fetch_test_file(info["git"], out / "src")
    elif args.test_file:
        test_file = args.test_file.resolve()
    else:
        p.error("give --test-file or --fetch")
    args.test_dir = test_file.parent
    ids = collect(args, test_file)
    if args.ids:
        wanted = {x.strip() for x in args.ids.read_text().splitlines() if x.strip()}
        ids = [i for i in ids if i in wanted]
    if args.match:
        ids = [i for i in ids if any(m in i for m in args.match)]
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    warm_up(args, args.gpus[0])
    run_all(args, ids, out)
    sys.exit(report(args, out, info))


if __name__ == "__main__":
    main()
