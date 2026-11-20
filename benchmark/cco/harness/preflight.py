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
"""Record the environment of a GPU run and refuse a run that cannot be valid.

Used by ``run.py`` before every launch, and on its own:

    python preflight.py [--out env.json] [--repo <mori checkout>] [--allow-busy]

It never initialises HIP: it does not import torch or mori, only finds them, and asks
the GPUs through ``rocminfo`` and ``rocm-smi``. It records the host, the repository
commit, where ``mori``, ``torch`` and ``flydsl`` load from and their versions, the
``BUILD_CCO_SDMA`` flag baked into the installed mori, the ROCm version, the GPUs and
their architecture, the processes on them, the link types between them, the kernel's
``CONFIG_DMABUF_MOVE_NOTIFY`` and the environment variables that steer mori.

Refuses (exit 1): no GPU of the required architecture; a GPU with a process on it
(``--allow-busy`` to override); SDMA not built into mori or ``MORI_ENABLE_SDMA`` not
``1`` (``--no-require-sdma`` to override; without both, every SDMA put silently
does nothing, G950-ENV-01); mori not importable. Warns: ROCm 7.2.x (G950-ENV-04);
``CONFIG_DMABUF_MOVE_NOTIFY`` off or unknown (G950-ENV-03); links that are not xGMI;
a mori loaded from outside ``--repo``; a dirty repository.
"""

import argparse
import gzip
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

ENV_KEYS = (
    "MORI_ENABLE_SDMA",
    "BUILD_CCO_SDMA",
    "MORI_SOCKET_IFNAME",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_USERPTR_FOR_PAGED_MEM",
    "HSA_USE_SVM",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_DUMP_IR",
    "FLYDSL_DUMP_DIR",
)


def sh(cmd, timeout=60):
    """(returncode, stdout+stderr), or (None, reason) when the command cannot run."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]}: timed out after {timeout}s"
    return done.returncode, done.stdout + done.stderr


def parse_rocminfo(text):
    """GPU agents as [{"name": "gfx950", "marketing_name": ...}], in agent order."""
    agents, current = [], None
    for line in text.splitlines():
        if re.match(r"^\s*Agent \d+\s*$", line):
            current = {}
            agents.append(current)
            continue
        if current is None:
            continue
        m = re.match(r"^\s{2}(Name|Marketing Name|Device Type):\s*(.*?)\s*$", line)
        if m and m.group(1) not in current:
            current[m.group(1)] = m.group(2)
    return [
        {"name": a.get("Name"), "marketing_name": a.get("Marketing Name")}
        for a in agents
        if a.get("Device Type") == "GPU"
    ]


def parse_showpids(text):
    """Processes on the GPUs from ``rocm-smi --showpids``: [] when idle, None when unreadable."""
    if "No KFD PIDs currently running" in text:
        return []
    pids = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)", line)
        if m:
            pids.append({"pid": int(m.group(1)), "name": m.group(2)})
    return pids if pids else None


def parse_topotype(text):
    """Link types between GPUs from ``rocm-smi --showtopotype``: {"GPU0-GPU1": "XGMI", ...}."""
    header, links = None, {}
    for line in text.splitlines():
        cells = line.split()
        if cells and all(re.fullmatch(r"GPU\d+", c) for c in cells):
            header = cells
            continue
        if (
            header
            and cells
            and re.fullmatch(r"GPU\d+", cells[0])
            and len(cells) == len(header) + 1
        ):
            for other, kind in zip(header, cells[1:]):
                if other != cells[0]:
                    links[f"{cells[0]}-{other}"] = kind
    return links


def parse_build_flags(text):
    m = re.search(r"^BUILD_CCO_SDMA\s*=\s*(\S+)", text, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip("\"'").upper() in ("TRUE", "1", "ON", "YES")


def parse_kconfig(text, key="CONFIG_DMABUF_MOVE_NOTIFY"):
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
        if line.strip() == f"# {key} is not set":
            return "n"
    return None


def _find(module):
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def _dist_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _git(repo, *args):
    # safe.directory: in a container git runs as root over a checkout someone else owns,
    # and without it refuses even read-only queries ("dubious ownership").
    rc, out = sh(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args])
    return out.strip() if rc == 0 else None


def _rocm_version(run, torch_dir):
    info = Path("/opt/rocm/.info/version")
    if info.is_file():
        return info.read_text().strip(), str(info)
    rc, out = run(["amd-smi", "version"])
    if rc == 0:
        m = re.search(r"ROCm version:\s*([\w.\-]+)", out)
        if m:
            return m.group(1), "amd-smi version"
    if torch_dir is not None and (torch_dir / "version.py").is_file():
        m = re.search(
            r"^hip\s*(?::\s*\S+)?\s*=\s*['\"]([^'\"]+)",
            (torch_dir / "version.py").read_text(),
            re.MULTILINE,
        )
        if m:
            return (
                m.group(1),
                "torch.version.hip (the HIP runtime torch was built with)",
            )
    return None, None


def _kernel_config(release):
    for path in (Path(f"/boot/config-{release}"), Path("/proc/config.gz")):
        try:
            text = (
                gzip.decompress(path.read_bytes()).decode()
                if path.suffix == ".gz"
                else path.read_text()
            )
        except OSError:
            continue
        return parse_kconfig(text), str(path)
    return None, None


def collect(repo=None, run=sh):
    """Everything the run's record needs; touches no GPU state."""
    release = platform.release()
    info = {
        "host": socket.gethostname(),
        "kernel": release,
        "in_container": Path("/.dockerenv").exists(),
        "python": sys.version.split()[0],
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
    }
    if repo is not None:
        repo = Path(repo).resolve()
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
        info["repo"] = {
            "path": str(repo),
            "commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
        }

    mori_dir = _find("mori")
    flags = mori_dir / "cco" / "device" / "_build_flags.py" if mori_dir else None
    info["mori"] = {
        "path": str(mori_dir) if mori_dir else None,
        "version": _dist_version("amd_mori", "amd-mori", "mori"),
        "build_cco_sdma": (
            parse_build_flags(flags.read_text()) if flags and flags.is_file() else None
        ),
    }
    torch_dir = _find("torch")
    info["torch"] = {
        "path": str(torch_dir) if torch_dir else None,
        "version": _dist_version("torch"),
    }
    flydsl_dir = _find("flydsl")
    info["flydsl"] = {
        "path": str(flydsl_dir) if flydsl_dir else None,
        "version": _dist_version("flydsl"),
    }
    version, source = _rocm_version(run, torch_dir)
    info["rocm"] = {"version": version, "source": source}

    rc, out = run(["rocminfo"])
    info["gpus"] = parse_rocminfo(out) if rc == 0 else None
    rc, out = run(["rocm-smi", "--showpids"])
    info["gpu_processes"] = parse_showpids(out) if rc == 0 else None
    rc, out = run(["rocm-smi", "--showtopotype"])
    info["links"] = parse_topotype(out) if rc == 0 else None
    value, source = _kernel_config(release)
    info["dmabuf_move_notify"] = {"value": value, "source": source}
    return info


def evaluate(info, require_arch="gfx950", allow_busy=False, require_sdma=True):
    """[{"level": "FAIL" | "WARN", "code", "message"}] for a collected environment."""
    findings = []

    def add(level, code, message):
        findings.append({"level": level, "code": code, "message": message})

    gpus = info.get("gpus")
    if gpus is None:
        add("FAIL", "gpus", "rocminfo did not run: no GPU information")
    elif not gpus:
        add("FAIL", "gpus", "rocminfo lists no GPU agent")
    elif require_arch and any(g["name"] != require_arch for g in gpus):
        names = sorted({g["name"] for g in gpus})
        add(
            "FAIL",
            "arch",
            f"GPUs are {', '.join(names)}, not {require_arch}: the facts and rules do not carry over",
        )

    procs = info.get("gpu_processes")
    if procs is None:
        add(
            "WARN" if allow_busy else "FAIL",
            "busy",
            "rocm-smi --showpids could not be read: unknown is not idle",
        )
    elif procs:
        listing = ", ".join(f"{p['pid']} {p['name']}" for p in procs[:6])
        add(
            "WARN" if allow_busy else "FAIL",
            "busy",
            f"{len(procs)} process(es) on the GPUs: {listing}",
        )

    mori = info.get("mori", {})
    if mori.get("path") is None:
        add("FAIL", "mori", "mori is not importable")
    env = info.get("env", {})
    if require_sdma:
        baked = mori.get("build_cco_sdma")
        override = env.get("BUILD_CCO_SDMA")
        built = (
            override.strip().upper() in ("1", "ON", "TRUE", "YES")
            if override is not None
            else baked
        )
        if not built:
            add(
                "FAIL",
                "sdma-build",
                "mori was not built with BUILD_CCO_SDMA: every SDMA put does nothing (G950-ENV-01)",
            )
        if env.get("MORI_ENABLE_SDMA") != "1":
            add(
                "FAIL",
                "sdma-env",
                "MORI_ENABLE_SDMA is not 1: cco sets up no SDMA queue (G950-ENV-01)",
            )

    repo = info.get("repo")
    if repo:
        if repo.get("dirty"):
            add(
                "WARN",
                "repo-dirty",
                "the repository has uncommitted changes: the commit does not identify the code",
            )
        if mori.get("path") and not str(mori["path"]).startswith(repo["path"] + os.sep):
            add(
                "WARN",
                "mori-source",
                f"mori loads from {mori['path']}, not from the repository at {repo['path']}",
            )

    version = (info.get("rocm") or {}).get("version") or ""
    if re.match(r"7\.2\.", version):
        add(
            "WARN",
            "rocm-7.2",
            f"ROCm {version}: uncached VMM allocations land in the coarse-grained pool (G950-ENV-04)",
        )
    dmabuf = (info.get("dmabuf_move_notify") or {}).get("value")
    if dmabuf != "y":
        state = "unknown" if dmabuf is None else "not set"
        add(
            "WARN",
            "dmabuf-move-notify",
            f"CONFIG_DMABUF_MOVE_NOTIFY {state}: peer memory may land in host memory (G950-ENV-03)",
        )
    links = info.get("links")
    if links:
        other = sorted({k for k in links.values() if k.upper() != "XGMI"})
        if other:
            add("WARN", "links", f"links that are not xGMI: {', '.join(other)}")
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record and check the environment of a GPU run."
    )
    parser.add_argument("--out", help="write the record here (default: stdout)")
    parser.add_argument("--repo", help="the mori checkout the run is meant to use")
    parser.add_argument(
        "--require-arch",
        default="gfx950",
        help="GPU architecture to require ('' for any)",
    )
    parser.add_argument(
        "--allow-busy", action="store_true", help="run even with processes on the GPUs"
    )
    parser.add_argument(
        "--no-require-sdma",
        action="store_true",
        help="do not require SDMA (non-SDMA modes)",
    )
    args = parser.parse_args(argv)

    info = collect(args.repo)
    findings = evaluate(
        info, args.require_arch, args.allow_busy, not args.no_require_sdma
    )
    failed = any(f["level"] == "FAIL" for f in findings)
    record = {
        "info": info,
        "findings": findings,
        "result": "FAIL" if failed else "PASS",
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for f in findings:
        print(f"{f['level']:4}  {f['code']}: {f['message']}", file=sys.stderr)
    print(f"preflight: {record['result']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
