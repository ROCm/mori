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
"""One measurement entry for fused-operator benchmarks: one shape, interleaved modes.

Used in steps 1 and 3 of the ``cco-fusion-loop`` skill and by the verifier in step 4:

    python run.py --op gemm_ar --modes split-sdma,fused-sdma --rounds 3 --aa \\
        --bench-args "-m 16384 -n 7168 -k 2048 --quant blockscale" --out <run-dir>

It launches ``torchrun`` once, through ``tests/python/cco/bench_runs.py``, so every mode
runs in the same processes, interleaved round by round: A, B, A, B, ... Every mode is
compared with ``--baseline`` (default: the first mode); with ``--aa`` the baseline runs
again at the end of each round and the A/A difference is the noise band. One invocation is one shape (G950-MEAS-03): ``--bench-args`` sets it,
and the record keeps the benchmark's own report of it.

Before the launch it runs ``preflight.py`` and stops on a refusal (busy GPUs, mori
without SDMA, another architecture). For multi-rank runs it waits until 10 s have
passed since the previous one it started on this host (G950-ENV-05). The launch runs in
a process group of its own, under coreutils ``timeout`` on the executing side and a
timeout here; on timeout only that group and the processes it started are terminated.

It never allocates GPUs: it runs where it is started, or behind ``--launcher`` (a
prefix such as ``docker exec <container>``; the repository must then sit at the same
path on both sides, so the command names it by absolute path, and cleanup on timeout is
left to the ``timeout`` it runs there).

With ``--dump-isa`` FlyDSL writes every kernel's final ISA into ``<run-dir>/isa``, using
a fresh runtime cache so that every kernel is compiled and dumped, and the record lists
each with its sha256 and the result of ``isa_check.py``.

Writes ``<run-dir>/result.json`` (``schemas/result.schema.json``), ``env.json``,
``stdout.log`` and ``stderr.log``. Exit status: 0 when every run completed and validated,
1 when a run failed validation, 2 for a usage error or a preflight refusal, 3 for a
timeout or missing results.
"""

import argparse
import hashlib
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import isa_check  # noqa: E402
import preflight  # noqa: E402

SCHEMA = "cco-fusion/result/v1"
DRIVER = "tests/python/cco/bench_runs.py"
GAP_FILE = Path(os.environ.get("TMPDIR", "/tmp")) / "cco_fusion_last_multirank_run"
GAP_S = 10.0
ENV_DEFAULTS = {"MORI_ENABLE_SDMA": "1", "MORI_SOCKET_IFNAME": "lo"}

OPS = {
    "gemm_ar": {
        "bench": "benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py",
        "time_field": "max_rank_time_us",
        "per_rank_field": "per_rank_time_us",
        "modes": (
            "gemm-only",
            "split-sdma",
            "fused-sdma",
            "fused-lsa",
            "split-lsa",
            "split-rccl",
        ),
        "sdma_modes": ("split-sdma", "fused-sdma"),
    },
}


class UsageError(ValueError):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan_runs(modes, rounds, aa, bench_args, baseline=None):
    """The runs in launch order: every round runs every mode, plus the baseline again with ``aa``."""
    baseline = baseline or modes[0]
    plan = []
    for r in range(1, rounds + 1):
        order = [(m, False) for m in modes] + ([(baseline, True)] if aa else [])
        for mode, repeat in order:
            plan.append(
                {
                    "round": r,
                    "mode": mode,
                    "repeat": repeat,
                    "args": ["--mode", mode, *bench_args],
                }
            )
    return plan


def build_command(spec, plan, nproc, timeout_s, launcher, python, env):
    runs = json.dumps([p["args"] for p in plan])
    return [
        *launcher,
        "timeout",
        "--signal=TERM",
        "--kill-after=30",
        str(int(timeout_s)),
        "env",
        *[f"{k}={v}" for k, v in sorted(env.items())],
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(REPO / DRIVER),
        str(REPO / spec["bench"]),
        runs,
    ]


def _children():
    """{ppid: [pid, ...]} for every process this host shows us."""
    tree = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        tree.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    return tree


def descendants(pid):
    tree, out, todo = _children(), [], [pid]
    while todo:
        for child in tree.get(todo.pop(), []):
            out.append(child)
            todo.append(child)
    return out


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def terminate(proc, grace=15.0):
    """SIGTERM the run's process group and everything it started, then SIGKILL what is left."""
    pids = [proc.pid, *descendants(proc.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + grace
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.2)
        if not any(_alive(p) for p in pids):
            break
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return [p for p in pids if _alive(p)]


def execute(cmd, cwd, timeout_s, log_dir):
    """Run ``cmd`` in its own process group with output logged; (status, returncode, leftover pids)."""
    with (
        open(log_dir / "stdout.log", "w") as out,
        open(log_dir / "stderr.log", "w") as err,
    ):
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=out, stderr=err, start_new_session=True
        )
        try:
            rc = proc.wait(timeout=timeout_s + 60)
        except subprocess.TimeoutExpired:
            return "timeout", None, terminate(proc)
    # coreutils timeout: 124 after TERM, 137 after the follow-up KILL
    return ("timeout" if rc in (124, 137) else "done"), rc, []


def parse_results(text):
    records = []
    for line in text.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                records.append(json.loads(line[len("RESULT_JSON ") :]))
            except json.JSONDecodeError:
                records.append(None)
    return records


def aggregate(spec, plan, records, aa, baseline=None):
    rows = []
    for i, p in enumerate(plan):
        rec = (
            records[i]
            if i < len(records) and records[i] and records[i].get("mode") == p["mode"]
            else None
        )
        rows.append(
            {
                "round": p["round"],
                "mode": p["mode"],
                "repeat": p["repeat"],
                "time_us": rec.get(spec["time_field"]) if rec else None,
                "per_rank_us": rec.get(spec["per_rank_field"]) if rec else None,
                "critical_rank": rec.get("critical_rank") if rec else None,
                "validated": rec.get("validated") if rec else None,
                "rel_l2": rec.get("rel_l2") if rec else None,
                "record": rec,
            }
        )
    modes = list(dict.fromkeys(p["mode"] for p in plan))
    summary = {}
    for mode in modes:
        mine = [r for r in rows if r["mode"] == mode and not r["repeat"]]
        times = [r["time_us"] for r in mine if isinstance(r["time_us"], (int, float))]
        if not times:
            summary[mode] = {"rounds": 0, "validated": False}
            continue
        median = statistics.median(times)
        summary[mode] = {
            "rounds": len(times),
            "median_us": median,
            "min_us": min(times),
            "max_us": max(times),
            "spread_pct": 100.0 * (max(times) - min(times)) / median,
            "validated": all(r["validated"] is True for r in mine),
        }

    noise = {"source": None, "pct": None}
    base = baseline or modes[0]
    if aa:
        diffs = []
        for r in sorted({p["round"] for p in plan}):
            a = next(
                (
                    x["time_us"]
                    for x in rows
                    if x["round"] == r and x["mode"] == base and not x["repeat"]
                ),
                None,
            )
            b = next(
                (x["time_us"] for x in rows if x["round"] == r and x["repeat"]), None
            )
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diffs.append(100.0 * abs(a - b) / ((a + b) / 2))
        if diffs:
            noise = {
                "source": "A/A difference, median over rounds",
                "pct": statistics.median(diffs),
            }
    if noise["pct"] is None and summary[base].get("rounds", 0) > 1:
        noise = {
            "source": f"spread of {base} across rounds",
            "pct": summary[base]["spread_pct"],
        }

    comparisons = []
    for mode in (m for m in modes if m != base):
        a, b = summary[base].get("median_us"), summary[mode].get("median_us")
        if a and b:
            delta = 100.0 * (b - a) / a
            comparisons.append(
                {
                    "baseline": base,
                    "candidate": mode,
                    "speedup": a / b,
                    "delta_pct": delta,
                    "outside_noise": (
                        None if noise["pct"] is None else abs(delta) > noise["pct"]
                    ),
                }
            )
    return rows, summary, noise, comparisons


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_isa(dump_dir):
    items = []
    for path in sorted(Path(dump_dir).rglob("*final_isa.s")):
        item = {"kernel": path.parent.name, "path": str(path), "sha256": sha256(path)}
        args = argparse.Namespace(
            dump=str(path),
            kernel=None,
            baseline=None,
            baseline_kernel=None,
            max_distance_drop=0.10,
            require_write_through=False,
            strict=False,
        )
        try:
            payload, _, _ = isa_check.run(args)
            item["isa_check"] = payload["result"]
            item["findings"] = [
                f"{f['level']} {f['code']}" for f in payload["findings"]
            ]
        except (isa_check.DumpError, OSError) as e:
            item["isa_check"] = "unreadable"
            item["findings"] = [str(e)]
        items.append(item)
    return items


def wait_gap(nproc):
    if nproc <= 1:
        return 0.0
    try:
        last = float(GAP_FILE.read_text())
    except (OSError, ValueError):
        return 0.0
    wait = GAP_S - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
        return wait
    return 0.0


def mark_gap(nproc):
    if nproc > 1:
        try:
            GAP_FILE.write_text(f"{time.time()}\n")
        except OSError:
            pass


def run_preflight(args, launcher, require_sdma, out_dir):
    flags = ["--repo", str(REPO)]
    if args.allow_busy:
        flags.append("--allow-busy")
    if not require_sdma:
        flags.append("--no-require-sdma")
    if launcher:
        cmd = [*launcher, args.python, str(HERE / "preflight.py"), *flags]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        record = (
            json.loads(done.stdout) if done.stdout.strip().startswith("{") else None
        )
        if record is None:
            raise UsageError(
                f"preflight through the launcher printed no record: {done.stderr.strip()[-400:]}"
            )
    else:
        info = preflight.collect(REPO)
        findings = preflight.evaluate(
            info, allow_busy=args.allow_busy, require_sdma=require_sdma
        )
        failed = any(f["level"] == "FAIL" for f in findings)
        record = {
            "info": info,
            "findings": findings,
            "result": "FAIL" if failed else "PASS",
        }
    (out_dir / "env.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run one shape of a fused-operator benchmark, modes interleaved."
    )
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    parser.add_argument("--modes", required=True, help="comma-separated modes")
    parser.add_argument(
        "--baseline",
        help="the mode every other mode is compared with (default: the first)",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--aa",
        action="store_true",
        help="run the baseline twice per round for the noise band",
    )
    parser.add_argument(
        "--bench-args", default="", help="arguments for every run, e.g. the shape"
    )
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds for the whole launch"
    )
    parser.add_argument(
        "--launcher",
        default="",
        help="prefix that reaches the GPUs, e.g. 'docker exec ctr'",
    )
    parser.add_argument(
        "--python", default=None, help="interpreter on the executing side"
    )
    parser.add_argument(
        "--dump-isa",
        action="store_true",
        help="dump and check every kernel's final ISA",
    )
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing result.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the command, run nothing",
    )
    parser.add_argument("--out", required=True, help="run directory")
    args = parser.parse_args(argv)

    spec = OPS[args.op]
    args.mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in args.mode_list if m not in spec["modes"]]
    if not args.mode_list or unknown:
        raise UsageError(
            f"modes for {args.op}: {', '.join(spec['modes'])}; unknown: {', '.join(unknown) or 'none given'}"
        )
    if len(set(args.mode_list)) != len(args.mode_list):
        raise UsageError("a mode is listed twice; use --aa for an A/A pair")
    if args.baseline is None:
        args.baseline = args.mode_list[0]
    elif args.baseline not in args.mode_list:
        raise UsageError(f"--baseline {args.baseline} is not one of --modes")
    if args.rounds < 1 or args.nproc < 1 or args.timeout <= 0:
        raise UsageError("--rounds, --nproc and --timeout must be positive")
    if args.python is None:
        args.python = "python3" if args.launcher else sys.executable
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
    except UsageError as e:
        print(f"run.py: {e}", file=sys.stderr)
        return 2
    spec = OPS[args.op]
    out_dir = Path(args.out).resolve()
    launcher = shlex.split(args.launcher)
    bench_args = shlex.split(args.bench_args)
    plan = plan_runs(args.mode_list, args.rounds, args.aa, bench_args, args.baseline)
    env = dict(ENV_DEFAULTS)
    if args.dump_isa:
        env.update(
            {
                "FLYDSL_DUMP_IR": "1",
                "FLYDSL_DUMP_DIR": str(out_dir / "isa"),
                "FLYDSL_RUNTIME_CACHE_DIR": str(out_dir / "flydsl-cache"),
            }
        )
    cmd = build_command(
        spec, plan, args.nproc, args.timeout, launcher, args.python, env
    )
    if args.dry_run:
        print(json.dumps({"plan": plan, "command": cmd}, indent=2))
        return 0

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".write-test").write_text("")
        (out_dir / ".write-test").unlink()
    except OSError as e:
        print(f"run.py: cannot write the run directory: {e}", file=sys.stderr)
        return 2
    if (out_dir / "result.json").exists() and not args.force:
        print(
            f"run.py: {out_dir / 'result.json'} exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 2

    result = {
        "schema": SCHEMA,
        "op": args.op,
        "modes": args.mode_list,
        "baseline": args.baseline,
        "rounds": args.rounds,
        "aa": args.aa,
        "bench": spec["bench"],
        "bench_args": bench_args,
        "nproc": args.nproc,
        "launcher": launcher,
        "env_overrides": env,
        "command": cmd,
        "timeout_s": args.timeout,
        "started_at": now_iso(),
    }
    try:
        if args.no_preflight:
            result["preflight"] = "skipped"
        else:
            require_sdma = any(m in spec["sdma_modes"] for m in args.mode_list)
            record = run_preflight(args, launcher, require_sdma, out_dir)
            result["preflight"] = record["result"]
            info = record.get("info", {})
            result["scope"] = {
                "host": info.get("host"),
                "repo": info.get("repo"),
                "mori": info.get("mori"),
                "torch": info.get("torch"),
                "flydsl": info.get("flydsl"),
                "rocm": info.get("rocm"),
                "gpus": len(info.get("gpus") or []),
                "env": "env.json",
            }
            if record["result"] != "PASS":
                result.update(status="refused", finished_at=now_iso())
                (out_dir / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                for f in record.get("findings", []):
                    print(
                        f"preflight {f['level']} {f['code']}: {f['message']}",
                        file=sys.stderr,
                    )
                return 2
    except (UsageError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"run.py: preflight failed: {e}", file=sys.stderr)
        return 2

    result["gap_wait_s"] = round(wait_gap(args.nproc), 3)
    t0 = time.time()
    status, rc, leftover = execute(cmd, REPO, args.timeout, out_dir)
    mark_gap(args.nproc)
    result["duration_s"] = round(time.time() - t0, 3)
    result["returncode"] = rc
    result["leftover_pids"] = leftover

    records = parse_results((out_dir / "stdout.log").read_text(errors="replace"))
    rows, summary, noise, comparisons = aggregate(
        spec, plan, records, args.aa, args.baseline
    )
    complete = all(r["record"] is not None for r in rows)
    if status == "timeout":
        result["status"] = "timeout"
    elif not complete:
        result["status"] = "incomplete"
    elif any(r["validated"] is not True for r in rows):
        result["status"] = "validation-failed"
    else:
        result["status"] = "ok"
    result.update(runs=rows, summary=summary, noise=noise, comparisons=comparisons)
    if args.dump_isa:
        result["isa"] = collect_isa(out_dir / "isa")
    result["finished_at"] = now_iso()
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    for mode, s in summary.items():
        if s.get("rounds"):
            print(
                f"{mode:12} median {s['median_us']:.1f} us  spread {s['spread_pct']:.1f}%  "
                f"rounds {s['rounds']}  validated {s['validated']}"
            )
    for c in comparisons:
        print(
            f"{c['candidate']} vs {c['baseline']}: {c['delta_pct']:+.1f}%  speedup {c['speedup']:.3f}"
        )
    if noise["pct"] is not None:
        print(f"noise band {noise['pct']:.2f}% ({noise['source']})")
    print(f"status {result['status']}  ->  {out_dir / 'result.json'}")
    return {"ok": 0, "validation-failed": 1}.get(result["status"], 3)


if __name__ == "__main__":
    sys.exit(main())
