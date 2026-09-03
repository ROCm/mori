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
"""CI gate over the five UMBP primitives, swept over concurrency.

Brings up standalone-process mode -- `umbp_standalone_server` plus, for the
second arm, a `umbp_master` in front of it -- runs `umbp_prim_bench.py`
against each, and turns the rows into a pass/fail verdict.

    python3 umbp_prim_ci.py --arms nomaster master

Two things are gated, and the second is the reason this exists.

**Correctness.**  Every one of `put`, `get`, `exists`, `put_ranges` and
`get_ranges` must report zero failed operations and zero read-back
mismatches at every rank count.  Eight concurrent clients writing and
reading disjoint keyspaces through one server is a real functional test on
its own.

It is not, however, where a control-plane regression shows up: a build whose
resolve path serializes still returns every correct answer, just slowly.  Do
not read a correctness pass as evidence about performance, or the reverse --
and treat a correctness failure on ONE arm of an A/B as a claim about the
harness until the pool has been ruled out (see `Pass.resident_bytes`).

**Concurrency scaling of `exists`.**  Aggregate `batch_exists` throughput
must grow by at least `--exists-scaling-floor` from the smallest rank count
to the largest.  This is the axis a whole benchmark suite missed once: an
exclusive lock over the resolve path is invisible at one rank and fatal at
eight, so an absolute latency number taken at a single concurrency cannot
see it, and comparing absolute numbers across CI runs on a shared machine
is noise.  A ratio measured inside one run is not.

The floor is 2.0x, and it was set by measuring rather than by borrowing a
number from an idle machine.  On a busy node, three identical runs of one
unmodified build scored 3.50x, 5.39x and 4.33x, with the one-rank baseline
steady to within 3%; the regression this is aimed at scored 0.75x.  So the
floor sits ~43% under the worst healthy run seen and 2.7x over the
regressed one.

That margin exists because of the window, not because the threshold is
generous.  Measured through the correctness pass's geometry -- a sub-
millisecond window -- the same three runs scattered 1.59x, 1.88x and 3.49x
and were not even monotonic in rank count.  If this number ever reads
noisy, the fix is a longer window (`--scaling-passes`), not a lower floor.

`exists` is the gated op deliberately -- it is the only one of the five
that moves no bytes, so its scaling is a property of the control plane and
not of how much DRAM bandwidth the runner had left.  The other four are
measured and reported, but not gated: reads scale with the memory system,
and both a good and a bad build of the write path have been observed to
*anti*-scale under this geometry, so a floor there would be either
vacuous or permanently red.  If a write-path scaling gate is ever wanted,
it needs its own baseline, not this one's.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "umbp_prim_bench.py"

#: Aggregate `exists` throughput at the largest rank count, over the smallest.
#: See SCALING_* below for why this is measured in its own pass, and the
#: module docstring for how the floor was picked.
DEFAULT_EXISTS_SCALING_FLOOR = 2.0

#: Geometry for the scaling pass, which measures `exists` and nothing else.
#:
#: It is deliberately not the correctness pass's geometry.  A `batch_exists`
#: call costs tens of microseconds, so the correctness pass's few dozen calls
#: add up to a timed window under a millisecond -- short enough that process
#: scheduling jitter, not the control plane, decides the throughput.  Measured
#: on one loaded node, three identical runs of a single build scattered across
#: 1.59x, 1.88x and 3.49x on such a window; no floor survives that.  So this
#: pass buys a window in the tens of milliseconds by issuing thousands of
#: calls, and buys it cheaply by shrinking the object to 4 KiB -- `exists`
#: moves no bytes, so object size costs only seed time and pool space.
SCALING_KEYS_PER_RANK = 8192
SCALING_BATCH = 256
SCALING_PASSES = 100
SCALING_RANGE_BYTES = 4096

#: Absent-key fractions the scaling sweep runs at.  0.0 is the calibrated
#: gate (see the floor above); the rest are measured and reported so the
#: master-answered half of the probe is under the same concurrency sweep.
EXISTS_ABSENT_FRACS = (0.0, 0.5, 1.0)

#: Storage media a backend can be.  `hbm` is GPU memory -- the server's, not
#: the client buffer's, which is what --loc selects and is a separate axis.
MEDIA_KINDS = ("dram", "hbm", "ssd")

#: Ops whose read set has to be written before the timed window, and so
#: occupy the pool for the whole run.
SEEDED_OPS = ("get", "exists", "get_ranges")
#: Ops that insert a fresh generation of keys on every pass.
WRITING_OPS = ("put", "put_ranges")


# ---------------------------------------------------------------------------
#  Gate
#
#  Pure: it takes parsed result payloads and returns findings.  Nothing here
#  imports mori or touches a process, so it is unit-testable against recorded
#  numbers -- see test_umbp_prim_gate.py, which feeds it both arms of the
#  regression this floor was calibrated on.
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    arm: str
    check: str
    value: str
    threshold: str
    ok: bool
    detail: str = ""
    #: False for a number that is measured and printed but does not decide
    #: the build.  Used for the absent-key fractions, which have no
    #: calibrated baseline yet -- shipping a threshold nobody has measured is
    #: how a gate becomes a flake.
    gated: bool = True


def _rows_for(payload: dict, op: str) -> list[dict]:
    return sorted(
        (r for r in payload["rows"] if r["op"] == op), key=lambda r: r["ranks"]
    )


def check_correctness(arm: str, payload: dict, label: str = "") -> list[Finding]:
    rows = payload["rows"]
    bad = [r for r in rows if r["failures"] or r["mismatches"]]
    detail = "; ".join(
        f"{r['op']}@{r['ranks']}: {r['failures']} failed, {r['mismatches']} mismatched"
        for r in bad
    )
    return [
        Finding(
            arm=arm,
            check=f"correctness {label} ({len(rows)} rows)".replace("  ", " "),
            value=f"{len(bad)} bad",
            threshold="0 bad",
            ok=not bad,
            detail=detail,
        )
    ]


def check_exists_scaling(
    arm: str, payload: dict, floor: float, label: str = "", gated: bool = True
) -> list[Finding]:
    rows = _rows_for(payload, "exists")
    if len(rows) < 2:
        return [
            Finding(
                arm=arm,
                check=f"exists scaling {label}",
                gated=gated,
                value="n/a",
                threshold=f"{floor:.2f}x",
                ok=False,
                detail="needs at least two rank counts; the gate is a ratio",
            )
        ]
    low, high = rows[0], rows[-1]
    # A zero at one rank is not a scaling result, it is a broken run, and
    # dividing by it would hand back either inf or a crash.
    if low["keys_per_s"] <= 0:
        return [
            Finding(
                arm=arm,
                check=f"exists scaling {label} {low['ranks']}->{high['ranks']}",
                value="0 keys/s at the base rank count",
                threshold=f"{floor:.2f}x",
                ok=False,
                gated=gated,
            )
        ]
    scale = high["keys_per_s"] / low["keys_per_s"]
    return [
        Finding(
            arm=arm,
            check=f"exists scaling {label} {low['ranks']}->{high['ranks']}",
            value=f"{scale:.2f}x",
            threshold=(f">= {floor:.2f}x" if gated else "(not gated)"),
            ok=(scale >= floor) if gated else True,
            gated=gated,
            detail=(
                f"{low['keys_per_s']:,.0f} -> {high['keys_per_s']:,.0f} keys/s"
                if scale < floor
                else ""
            ),
        )
    ]


def _absent_frac(payload: dict) -> float:
    return float(payload.get("geometry", {}).get("exists_absent_frac", 0.0))


def gate(results: dict[str, dict], floor: float) -> list[Finding]:
    """Correctness over every pass; the scaling floor over the calibrated one.

    Correctness is checked on the absent-key passes too, and that is not
    ceremony: those passes assert the probe answers False for a key that was
    never written, so a false POSITIVE lands here.  For a prefix-cache probe
    that is the more dangerous direction -- it makes the engine read a key
    that does not exist.

    The scaling floor applies only at absent_frac = 0, which is the one
    measured against both arms of a known regression.  The other fractions
    put the master-answered half of the probe under the same sweep and are
    reported, but they gate nothing until someone baselines them.
    """
    findings: list[Finding] = []
    for arm, arm_results in results.items():
        for name, payload in sorted(arm_results.items()):
            findings += check_correctness(arm, payload, label=name)
        for name, payload in sorted(arm_results.items()):
            if not name.startswith("scaling"):
                continue
            frac = _absent_frac(payload)
            findings += check_exists_scaling(
                arm,
                payload,
                floor,
                label=f"absent={frac:g}",
                gated=(frac == 0.0),
            )
    return findings


def scaling_table(results: dict[str, dict]) -> list[str]:
    """Every op at every rank count, including the four that are not gated.

    Reported rather than gated: reads scale with the memory system and both a
    good and a regressed build have been seen to anti-scale on the writes, so
    a floor there would be either vacuous or permanently red.  Printing them
    still makes a change visible to whoever reads the log.
    """
    lines = [
        f"\n{'arm':<12}{'pass':<12}{'op':<12}{'ranks':>7}{'keys/s':>14}{'scale':>8}"
    ]
    for arm, arm_results in results.items():
        for which, payload in arm_results.items():
            for op in ("put", "get", "exists", "put_ranges", "get_ranges"):
                rows = _rows_for(payload, op)
                base = rows[0]["keys_per_s"] if rows else 0.0
                for row in rows:
                    scale = row["keys_per_s"] / base if base else 0.0
                    lines.append(
                        f"{arm:<12}{which:<12}{op:<12}{row['ranks']:>7}"
                        f"{row['keys_per_s']:>14,.0f}{scale:>7.2f}x"
                    )
    return lines


def report_findings(findings: list[Finding]) -> bool:
    print(f"\n{'arm':<10}{'check':<42}{'value':>16}{'threshold':>14}{'':>8}")
    for f in findings:
        verdict = "  PASS" if f.ok else "  FAIL"
        if not f.gated:
            verdict = "  info"
        print(f"{f.arm:<10}{f.check:<42}{f.value:>16}{f.threshold:>14}{verdict:>8}")
        if f.detail:
            print(f"{'':<10}  {f.detail}")
    return all(f.ok for f in findings if f.gated)


# ---------------------------------------------------------------------------
#  Media
#
#  What the SERVER stores in.  Three ways to say it, and which one applies is
#  forced by what the server can actually be told:
#
#  * one DRAM backend    -- the implicit embedded deployment, no selector env.
#    Kept as the default because it is the configuration the scaling floor was
#    calibrated against; switching the default would silently re-baseline it.
#  * one SSD backend     -- UMBP_DISTRIBUTED_MEDIUM=SSD plus the SSD dir and
#    capacity.  Naming a medium makes the server build an explicit distributed
#    config, which is also why the page size stops being optional there.
#  * anything else       -- a backend-policy file.  HBM has NO capacity env
#    var at all (Validate() demands hbm.capacity_bytes > 0 and nothing parses
#    one), so GPU memory is reachable ONLY this way; and mixtures need
#    per-backend weights, which only the policy expresses.
#
#  STATUS, 2026-09-03.  Only `--media dram` is verified end to end against a
#  live server.  The rest is wired and unit-tested but does NOT yet work:
#
#    ssd   every put fails with "[SsdBackend] Resolve: staging arena busy;
#          rolled back 0 reservations".  Rolled back ZERO means the very first
#          AcquireStagingSpanLocked failed, and it failed with 256 free pages
#          against a 16-key batch -- so this is not arena sizing, which is
#          already computed from the widest call.  AcquireStagingSpanLocked
#          returns empty for page_count == 0, so the likely cause is a
#          per-key page count of zero; that is inside the backend, not here.
#    combo the policy file now loads (its capacity needs a unit suffix, which
#          cost a round), but the server then reports
#          `medium=[DRAM pool=...]` with no SSD backend at all, so the policy
#          is not visibly taking effect on the media.  Unresolved.
#    hbm   untested.
#
#  Do not read a passing run at another --media as evidence until those are
#  settled: `dram` is the only one that has produced a real measurement.
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """One medium in the server's storage, and its share of the pool."""

    kind: str
    weight: float
    path: str = ""  # ssd only
    device: int = 0  # hbm only

    @property
    def name(self) -> str:
        return f"{self.kind}{self.device}" if self.kind == "hbm" else self.kind


def parse_media(
    spec: str, ssd_dirs: list[str], hbm_devices: list[int]
) -> list[Backend]:
    """`dram`, `ssd`, `hbm`, `dram:70,ssd:30`, `dram:50,ssd:30,hbm:20`.

    Weights are relative and normalised to 100; omitted weights split what is
    left evenly, so `dram,ssd` is 50/50 without having to say so.
    """
    entries = [part.strip() for part in spec.split(",") if part.strip()]
    if not entries:
        raise ValueError("--media is empty")
    parsed: list[tuple[str, float | None]] = []
    for entry in entries:
        kind, _, weight = entry.partition(":")
        kind = kind.strip().lower()
        if kind not in MEDIA_KINDS:
            raise ValueError(
                f"unknown medium {kind!r}; pick from {', '.join(MEDIA_KINDS)}"
            )
        parsed.append((kind, float(weight) if weight else None))
    if len({k for k, _ in parsed}) != len(parsed):
        raise ValueError("--media names a medium twice; give one weight instead")

    named = sum(w for _, w in parsed if w is not None)
    unweighted = [i for i, (_, w) in enumerate(parsed) if w is None]
    if unweighted:
        if named >= 100:
            raise ValueError(f"weights already total {named:g}; nothing left to split")
        share = (100.0 - named) / len(unweighted)
        parsed = [(k, share if w is None else w) for k, w in parsed]
    total = sum(w for _, w in parsed)
    if total <= 0:
        raise ValueError("--media weights must be positive")

    out, ssd_i = [], 0
    for kind, weight in parsed:
        weight = weight * 100.0 / total
        if kind == "ssd":
            if ssd_i >= len(ssd_dirs):
                raise ValueError("more ssd backends than --ssd-dirs entries")
            out.append(Backend(kind, weight, path=ssd_dirs[ssd_i]))
            ssd_i += 1
        elif kind == "hbm":
            out.append(Backend(kind, weight, device=hbm_devices[0]))
        else:
            out.append(Backend(kind, weight))
    return out


def backend_policy(backends: list[Backend], capacity: int) -> dict:
    """One logical tier over every backend, split by weight.

    A single tier on purpose: this measures placement across media on one
    node, not promotion and demotion between tiers, which would put an
    eviction policy inside a latency measurement.
    """
    entry = "bench"
    spec: dict = {}
    for b in backends:
        share = max(1 << 20, int(capacity * b.weight / 100.0))
        # The unit is not optional: the loader rejects a bare byte count with
        # "capacity unit must be B, KiB, MiB, GiB, or TiB".
        one: dict = {"type": b.kind, "capacity": f"{share}B"}
        if b.kind == "ssd":
            one["path"] = b.path
        elif b.kind == "hbm":
            one["devices"] = [b.device]
        spec[b.name] = one
    return {
        "schema_version": 1,
        "entry_tier": entry,
        "backends": spec,
        "tiers": [
            {"name": entry, "backends": {b.name: round(b.weight, 3) for b in backends}}
        ],
    }


# ---------------------------------------------------------------------------
#  Deployment
# ---------------------------------------------------------------------------


def _outbound_ip() -> str:
    """This host's address on the route out, which is what the master hands
    peers and what the IO engine binds.  127.0.0.1 does not work: the address
    is published, not merely dialled."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 53))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _require_gpu() -> None:
    """The master arm needs a GPU; say so before the server dies proving it.

    A master-backed server builds a MoRI IO engine, which initialises ROCm.
    In a container without /dev/kfd that fails inside engine construction and
    the process exits 255 after printing only "rsmi_num_monitor_devices
    reported 0 GPUs" -- which reads like an unrelated telemetry warning.  The
    masterless arm has no engine and does not care.
    """
    if not os.path.exists("/dev/kfd"):
        raise RuntimeError(
            "the master arm needs a GPU (/dev/kfd is not present): a "
            "master-backed standalone server builds a MoRI IO engine, which "
            "initialises ROCm. Run the container with --device=/dev/kfd "
            "--device=/dev/dri (docker/ci_run.sh already does), or use "
            "--arms nomaster."
        )


def _binary(env_name: str, override: str) -> str:
    """Resolve a UMBP binary.

    Importing mori.umbp is what publishes UMBP_STANDALONE_BIN and
    UMBP_MASTER_BIN, pointing at the binaries packaged beside the extension
    module -- so the server under test is always the one built from the tree
    that produced the installed wheel, never a stray one on PATH.
    """
    if override:
        return override
    try:
        import mori.umbp  # noqa: F401  (imported for its import-time side effect)
    except ImportError as err:
        raise RuntimeError(
            f"cannot import mori.umbp to locate {env_name}: {err}"
        ) from err
    path = os.environ.get(env_name, "")
    if not path or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"{env_name} is unset or not executable ({path!r}). Build mori with "
            "BUILD_UMBP=ON, or pass the path explicitly."
        )
    return path


class Deployment:
    """A standalone server, with or without a master in front of it.

    Context manager: everything it starts is torn down on the way out,
    including on an exception, because a leaked server holds both the unix
    socket and the pool and the next arm would fail to bind.
    """

    def __init__(
        self,
        arm: str,
        run_dir: Path,
        dram_bytes: int,
        page_bytes: int,
        server_bin: str,
        master_bin: str,
        ready_timeout: float,
        backends: list[Backend] | None = None,
        ssd_staging_slots: int = 0,
        ssd_staging_bytes: int = 0,
    ):
        self.ssd_staging_slots = ssd_staging_slots
        self.ssd_staging_bytes = ssd_staging_bytes
        self.backends = backends or [Backend("dram", 100.0)]
        self.arm = arm
        self.with_master = arm == "master"
        self.run_dir = run_dir
        self.dram_bytes = dram_bytes
        self.page_bytes = page_bytes
        self.server_bin = server_bin
        self.master_bin = master_bin
        self.ready_timeout = ready_timeout
        self.procs: list[tuple[str, subprocess.Popen]] = []
        # The socket lives in a short path on purpose: an AF_UNIX path is
        # capped at 108 bytes, and a workspace-relative one silently exceeds
        # it on a CI checkout.  The server derives the fd socket by replacing
        # the .grpc.sock suffix, so the name has to end in exactly that.
        self.sock_dir = Path(tempfile.mkdtemp(prefix=f"umbp-ci-{arm}-", dir="/tmp"))
        self.address = f"unix://{self.sock_dir}/node.grpc.sock"

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Deployment":
        try:
            master_address = self._start_master() if self.with_master else ""
            self._start_server(master_address)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *_exc) -> None:
        for name, proc in reversed(self.procs):
            if proc.poll() is not None:
                continue
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"[ci] {name} ignored SIGTERM, killing", flush=True)
                proc.kill()
                proc.wait(timeout=10)
        self.procs.clear()
        shutil.rmtree(self.sock_dir, ignore_errors=True)

    # -- processes ---------------------------------------------------------

    def _spawn(self, name: str, argv: list[str], env: dict[str, str]) -> Path:
        log = self.run_dir / f"{self.arm}_{name}.log"
        handle = open(log, "wb")
        print(f"[ci] starting {name}: {' '.join(argv)}", flush=True)
        proc = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, **env}
        )
        self.procs.append((name, proc))
        return log

    def _start_master(self) -> str:
        ip = _outbound_ip()
        port = _free_port()
        log = self._spawn(
            "master",
            [self.master_bin, f"0.0.0.0:{port}", str(_free_port())],
            {"MORI_UMBP_LOG_LEVEL": "info", "MORI_GLOBAL_LOG_LEVEL": "info"},
        )
        address = f"{ip}:{port}"
        # The master logs its address before it serves, so the log is not the
        # readiness signal here -- a completed TCP connect is.
        self._await(
            lambda: self._can_connect(ip, port),
            what=f"master on {address}",
            log=log,
            proc_name="master",
        )
        return address

    def _staging_env(self) -> dict:
        """Size the SSD read-staging arena for the batches actually issued.

        A Resolve reserves one arena page per key IN THE WHOLE CALL, and if it
        cannot get them all it rolls the reservation back and fails every key
        with "staging arena busy" -- which surfaces as `seed put failed`, not
        as anything mentioning staging.  The default arena is 16 slots, so a
        default 128-key batch cannot ever succeed.  Slots must therefore scale
        with the batch, and each slot must be at least one whole object (the
        buffer is divided evenly among them).
        """
        if not self.ssd_staging_slots:
            return {}
        return {
            "UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS": str(self.ssd_staging_slots),
            "UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SIZE": str(self.ssd_staging_bytes),
        }

    def _media_env(self) -> dict:
        """Translate the requested media into what the server can be told."""
        kinds = {b.kind for b in self.backends}
        if len(self.backends) == 1 and kinds == {"dram"}:
            # Implicit embedded DRAM: no selector, so the factory fills in the
            # deployment.  This is the shape the scaling floor was measured
            # against; do not "tidy" it into an explicit medium without
            # re-baselining, because they are not the same code path.
            return {"UMBP_SSD_ENABLED": "0", "UMBP_DRAM_CAPACITY": str(self.dram_bytes)}

        if len(self.backends) == 1 and kinds == {"ssd"}:
            ssd = self.backends[0]
            os.makedirs(ssd.path, exist_ok=True)
            return {
                "UMBP_DISTRIBUTED_MEDIUM": "SSD",
                "UMBP_SSD_ENABLED": "1",
                "UMBP_SSD_DIR": ssd.path,
                "UMBP_SSD_CAPACITY": str(self.dram_bytes),
                # Explicit distributed config: the page size stops being
                # optional, and 0 is rejected outright.
                "UMBP_DISTRIBUTED_DRAM_PAGE_SIZE": str(self.page_bytes),
                **self._staging_env(),
            }

        # Everything else -- any mixture, and any HBM at all -- is a policy.
        for b in self.backends:
            if b.kind == "ssd":
                os.makedirs(b.path, exist_ok=True)
        path = self.run_dir / f"{self.arm}_backend_policy.json"
        path.write_text(
            json.dumps(backend_policy(self.backends, self.dram_bytes), indent=2)
        )
        return {
            "UMBP_BACKEND_POLICY": str(path),
            # A medium must still be named and must still be one of the three
            # words the server accepts -- it reads this with getenv() and an
            # EMPTY string is a value, not an absence, so leaving it blank is
            # rejected outright.  The policy supplies the real backends; this
            # only has to validate.  It cannot be HBM: Validate() then demands
            # hbm.capacity_bytes > 0 and no env var sets it.
            "UMBP_DISTRIBUTED_MEDIUM": ("SSD" if kinds == {"ssd"} else "DRAM"),
            # UMBP_BACKEND_POLICY is NOT one of the env vars that make the
            # server build a distributed config, so on its own it is read and
            # then thrown away with the rest of the deployment.  Naming the
            # node is the selector with the least meaning of its own -- the
            # media all come from the policy.
            "UMBP_NODE_ID": f"prim-ci-{self.arm}",
            "UMBP_SSD_ENABLED": "1" if "ssd" in kinds else "0",
            "UMBP_DRAM_CAPACITY": str(self.dram_bytes),
            "UMBP_DISTRIBUTED_DRAM_PAGE_SIZE": str(self.page_bytes),
            **(self._staging_env() if "ssd" in kinds else {}),
        }

    def _start_server(self, master_address: str) -> None:
        env = {
            "UMBP_ROLE": "standalone",
            # Not hugepages: CI has no reservation, and falling back is silent.
            "UMBP_DRAM_USE_HUGEPAGES": "0",
            # Pin the paged medium's page size instead of taking the 2 MiB
            # default, so pool_bytes() computes against the size the server
            # actually allocates in.  Left implicit, a 64 KiB object pads out
            # to a 2 MiB page and the pool runs 32x short.  The masterless arm
            # reads this name, the master-backed one reads
            # UMBP_DISTRIBUTED_DRAM_PAGE_SIZE below, and they must agree or
            # the two arms are not measuring the same store.
            "UMBP_EMBEDDED_DRAM_PAGE_SIZE": str(self.page_bytes),
            # The readiness line below is INFO, and the default level is WARN,
            # so without this the wait can only ever time out.
            "MORI_UMBP_LOG_LEVEL": "info",
            "MORI_GLOBAL_LOG_LEVEL": "info",
        }
        env.update(self._media_env())
        if master_address:
            _require_gpu()
            ip = _outbound_ip()
            env.update(
                {
                    "UMBP_MASTER_ADDRESS": master_address,
                    # With a master all four are required: the node is a
                    # cluster member, so it must state where peers reach it.
                    "UMBP_NODE_ADDRESS": ip,
                    "UMBP_NODE_ID": f"{ip}:prim-ci",
                    "UMBP_IO_ENGINE_HOST": ip,
                    "UMBP_IO_ENGINE_PORT": str(_free_port()),
                    "UMBP_PEER_SERVICE_PORT": str(_free_port()),
                    # Defaults to 0, which the server rejects outright rather
                    # than delegating to the master's registry default.
                    "UMBP_DISTRIBUTED_DRAM_PAGE_SIZE": str(self.page_bytes),
                }
            )
        else:
            # Naming any of the selector vars is what switches the server off
            # its embedded backend, so a stale one in the environment would
            # silently turn this arm into the other one.
            for name in (
                "UMBP_MASTER_ADDRESS",
                "UMBP_NODE_ADDRESS",
                "UMBP_NODE_ID",
                "UMBP_IO_ENGINE_HOST",
                "UMBP_DISTRIBUTED_MEDIUM",
            ):
                # setdefault, not assignment: the media config above may have
                # set one of these deliberately (the policy path needs a
                # selector), and blanking it would drop the whole deployment.
                env.setdefault(name, "")

        log = self._spawn("server", [self.server_bin, self.address], env)
        # "data plane" is logged at the end of Start(), after both the gRPC
        # port and the fd-handoff listener are up.  The socket file appears
        # well before that, so waiting on the path yields a client that
        # connects and is then told the server is not ready.
        self._await(
            lambda: "data plane" in log.read_text(errors="replace"),
            what=f"standalone server on {self.address}",
            log=log,
            proc_name="server",
        )

    # -- waiting -----------------------------------------------------------

    @staticmethod
    def _can_connect(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((host, port)) == 0

    def _await(self, ready, what: str, log: Path, proc_name: str) -> None:
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            for name, proc in self.procs:
                if name == proc_name and proc.poll() is not None:
                    raise RuntimeError(
                        f"{what} exited with {proc.returncode} before becoming "
                        f"ready:\n{self._tail(log)}"
                    )
            if ready():
                print(f"[ci] {what} ready", flush=True)
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"{what} not ready after {self.ready_timeout:.0f}s:\n{self._tail(log)}"
        )

    @staticmethod
    def _tail(log: Path, lines: int = 30) -> str:
        try:
            return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
        except OSError as err:
            return f"(could not read {log}: {err})"


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


@dataclass
class Pass:
    """One bench invocation: which primitives, over what geometry."""

    name: str
    ops: list[str]
    keys_per_rank: int
    batch: int
    passes: int
    ranges: int
    range_bytes: int
    #: Fraction of each `exists` call naming a never-written key.  Only the
    #: scaling passes vary it; the correctness pass leaves it at 0.
    exists_absent_frac: float = 0.0
    #: Key namespace.  The scaling passes deliberately SHARE one: they probe
    #: the same dataset and differ only in how much of each call is absent,
    #: so seeding it once makes them comparable (identical store contents)
    #: and stops the pool being charged for the same keys per fraction.
    key_prefix: str = ""

    @property
    def residency_key(self) -> tuple:
        """Passes with the same value occupy the same pages, not more."""
        return (
            self.key_prefix,
            tuple(self.ops),
            self.keys_per_rank,
            self.object_bytes,
            self.passes,
        )

    @property
    def object_bytes(self) -> int:
        return self.ranges * self.range_bytes

    def resident_bytes(self, ranks: list[int], page_bytes: int) -> int:
        """What this pass leaves in the pool for the rest of the run.

        Charged in PAGES, not in bytes, and that distinction is the whole
        reason this method takes a page size.  The DRAM medium is paged and
        the allocator hands out whole pages, so a 4 KiB value in a 2 MiB page
        occupies 2 MiB.  Sizing this by `keys * object_bytes` understates
        demand by the padding ratio -- 32x for a 64 KiB object in a 2 MiB
        page -- and the pool then runs at a permanent eviction deficit.  What
        that looks like from the outside is NOT an out-of-space error: the
        just-seeded keys are the least recently used, so they are evicted
        first, and the run dies in the visibility wait with "seeded keys
        never became visible".  Measured, not deduced: it cost a whole
        false A/B result.

        Every rank count also writes into its own key namespace and nothing
        is deleted between them, so this sums over the whole sweep rather
        than taking the largest rank count.
        """
        seeded = sum(1 for op in SEEDED_OPS if op in self.ops)
        written = sum(1 for op in WRITING_OPS if op in self.ops)
        generations = seeded + written * self.passes
        keys = sum(ranks) * self.keys_per_rank * generations
        pages_per_key = max(1, -(-self.object_bytes // page_bytes))
        return keys * pages_per_key * page_bytes


def passes_for(args) -> list[Pass]:
    """The measurements, in the order they run against one server.

    Split because they want opposite geometries: correctness wants all five
    primitives over objects big enough for a transfer to mean something,
    scaling wants `exists` alone over a window long enough to measure.

    Scaling is then one pass per absent-key fraction.  The fraction decides
    how much of the probe the master has to answer -- a resident key is
    resolved locally and never reaches it, an absent one always does -- so
    each fraction is a different amount of control plane under the same
    concurrency sweep.  Separate invocations rather than one compound run,
    so each stays a single-purpose measurement.
    """
    out = [
        Pass(
            "correctness",
            list(args.ops),
            args.keys_per_rank,
            args.batch,
            args.passes,
            args.ranges,
            args.range_bytes,
        )
    ]
    for frac in args.exists_absent_fracs:
        out.append(
            Pass(
                f"scaling@{frac:g}",
                ["exists"],
                args.scaling_keys_per_rank,
                args.scaling_batch,
                args.scaling_passes,
                1,
                args.scaling_range_bytes,
                exists_absent_frac=frac,
                key_prefix=args.scaling_key_prefix,
            )
        )
    return out


def ssd_staging(args) -> tuple[int, int]:
    """(slots, bytes) for the SSD staging arena, from the widest call issued."""
    specs = passes_for(args)
    widest = max(p.batch for p in specs)
    biggest = max(p.object_bytes for p in specs)
    slot = max(args.page_bytes, biggest)
    # 4x the widest call, so the ranks running concurrently are not each
    # waiting on the last one to release before their own batch can reserve.
    slots = max(64, widest * 4)
    return slots, slots * slot


def pool_bytes(args, headroom: float) -> int:
    """Size the server pool from the geometry rather than a constant.

    Deduplicated by residency_key: the absent-fraction sweep is several
    passes over ONE dataset, so charging the pool once per fraction would
    triple a figure the run never occupies.
    """
    by_keyspace = {
        p.residency_key: p.resident_bytes(args.ranks, args.page_bytes)
        for p in passes_for(args)
    }
    return int(sum(by_keyspace.values()) * headroom)


def run_arm(arm: str, args, run_dir: Path, dram: int) -> dict:
    """Bring the arm up once and run both passes against it."""
    out: dict[str, dict] = {}
    with Deployment(
        arm=arm,
        run_dir=run_dir,
        dram_bytes=dram,
        page_bytes=args.page_bytes,
        server_bin=_binary("UMBP_STANDALONE_BIN", args.server_bin),
        # Resolved only for the arm that runs one: the nomaster arm must not
        # be blocked by a build that shipped no master binary.
        master_bin=(
            _binary("UMBP_MASTER_BIN", args.master_bin) if arm == "master" else ""
        ),
        ready_timeout=args.ready_timeout,
        backends=args.backends,
        **dict(zip(("ssd_staging_slots", "ssd_staging_bytes"), ssd_staging(args))),
    ) as deploy:
        for spec in passes_for(args):
            path = run_dir / f"{arm}_{spec.name.replace('@', '_')}.json"
            argv = [
                sys.executable,
                str(BENCH),
                "umbp-server",
                deploy.address,
                "--ops",
                *spec.ops,
                "--ranks",
                *[str(r) for r in args.ranks],
                "--keys-per-rank",
                str(spec.keys_per_rank),
                "--batch",
                str(spec.batch),
                "--passes",
                str(spec.passes),
                "--ranges",
                str(spec.ranges),
                "--range-bytes",
                str(spec.range_bytes),
                "--exists-absent-frac",
                str(spec.exists_absent_frac),
                *(("--key-prefix", spec.key_prefix) if spec.key_prefix else ()),
                "--loc",
                args.loc,
                "--json",
                str(path),
            ]
            print(
                f"\n[ci] === arm {arm}, pass {spec.name} ===\n[ci] {' '.join(argv)}",
                flush=True,
            )
            rc = subprocess.run(
                argv, env={**os.environ, "UMBP_STANDALONE_TIMEOUT_MS": "60000"}
            ).returncode
            if not path.exists():
                raise RuntimeError(
                    f"arm {arm}: the {spec.name} pass exited {rc} without writing {path}"
                )
            payload = json.loads(path.read_text())
            payload["bench_returncode"] = rc
            out[spec.name] = payload
    return out


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=["nomaster", "master"],
        default=["nomaster", "master"],
        help="standalone server without a master, with one, or both (default: both)",
    )
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument(
        "--ops",
        nargs="+",
        choices=["put", "get", "exists", "put_ranges", "get_ranges"],
        default=["put", "get", "exists", "put_ranges", "get_ranges"],
    )
    p.add_argument("--keys-per-rank", type=int, default=512)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--ranges", type=int, default=4)
    p.add_argument("--range-bytes", type=int, default=16384)
    p.add_argument("--loc", choices=["host", "gpu"], default="host")
    p.add_argument(
        "--exists-absent-fracs",
        nargs="+",
        type=float,
        default=list(EXISTS_ABSENT_FRACS),
        help="absent-key fractions to sweep the exists probe over. 0 never "
        "reaches the master, 1 always does.",
    )
    p.add_argument("--scaling-keys-per-rank", type=int, default=SCALING_KEYS_PER_RANK)
    p.add_argument("--scaling-batch", type=int, default=SCALING_BATCH)
    p.add_argument(
        "--scaling-passes",
        type=int,
        default=SCALING_PASSES,
        help="raise this, not the floor, if the scaling number is noisy",
    )
    p.add_argument("--scaling-range-bytes", type=int, default=SCALING_RANGE_BYTES)
    p.add_argument(
        "--page-bytes",
        type=int,
        default=65536,
        help="page size of the server's DRAM medium, on both arms. The pool is "
        "charged in whole pages per key, so this and the object size together "
        "decide how much pool the sweep needs.",
    )
    p.add_argument(
        "--pool-headroom",
        type=float,
        default=3.0,
        help="multiple of the run's resident set to give the pool",
    )
    p.add_argument(
        "--exists-scaling-floor", type=float, default=DEFAULT_EXISTS_SCALING_FLOOR
    )
    p.add_argument(
        "--media",
        default="dram",
        help="what the SERVER stores in: dram, hbm, ssd, or a weighted "
        "mixture such as 'dram:70,ssd:30'. Weights are relative; omitted ones "
        "split the remainder evenly. Note this is the server's storage, not "
        "the client buffer's location, which is --loc.",
    )
    p.add_argument(
        "--ssd-dirs",
        default="",
        help="comma-separated directories, one per ssd backend in --media "
        "(default: a per-run dir under $UMBP_PRIM_CI_SSD_ROOT or /data)",
    )
    p.add_argument(
        "--hbm-devices",
        default="0",
        help="comma-separated GPU ordinals for hbm backends",
    )
    p.add_argument("--ready-timeout", type=float, default=300.0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--server-bin", default="", help="override UMBP_STANDALONE_BIN")
    p.add_argument("--master-bin", default="", help="override UMBP_MASTER_BIN")
    a = p.parse_args(argv)
    if len(a.ranks) < 2:
        p.error("--ranks needs at least two counts: the gate is a ratio between them")
    if "exists" not in a.ops:
        p.error("--ops must include exists: it is the gated primitive")
    if any(not 0.0 <= f <= 1.0 for f in a.exists_absent_fracs):
        p.error("--exists-absent-fracs must all be between 0.0 and 1.0")
    if 0.0 not in a.exists_absent_fracs:
        p.error("--exists-absent-fracs must include 0.0: it is the gated point")
    # Computed once, so every scaling pass in this run names the same keys.
    a.scaling_key_prefix = f"s{os.getpid() % 100000:05d}"

    root = os.environ.get("UMBP_PRIM_CI_SSD_ROOT", "/data")
    ssd_dirs = (
        [d.strip() for d in a.ssd_dirs.split(",") if d.strip()]
        if a.ssd_dirs
        else [f"{root}/umbp_prim_ci_{os.getpid()}_{i}" for i in range(4)]
    )
    try:
        a.backends = parse_media(
            a.media, ssd_dirs, [int(d) for d in a.hbm_devices.split(",") if d.strip()]
        )
    except ValueError as err:
        p.error(f"--media: {err}")
    return a


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="umbp-prim-ci-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    dram = pool_bytes(args, args.pool_headroom)
    media = " + ".join(
        f"{b.kind}{'@' + b.path if b.kind == 'ssd' else ''} {b.weight:.0f}%"
        for b in args.backends
    )
    lines = [
        f"[ci] arms={','.join(args.arms)} ranks={args.ranks} loc={args.loc}",
        f"[ci] server media: {media}",
    ]
    for spec in passes_for(args):
        lines.append(
            f"[ci] pass {spec.name:<12} ops={','.join(spec.ops)} "
            f"object={spec.object_bytes}B keys/rank={spec.keys_per_rank} "
            f"batch={spec.batch} passes={spec.passes}"
        )
    lines.append(
        f"[ci] server pool: {dram / 2**30:.1f} GiB "
        f"({args.pool_headroom:g}x both passes' resident set)"
    )
    lines.append(f"[ci] logs and results: {run_dir}")
    print("\n".join(lines), flush=True)

    results: dict[str, dict] = {}
    failed_arms: list[str] = []
    for arm in args.arms:
        try:
            results[arm] = run_arm(arm, args, run_dir, dram)
        except Exception as err:  # noqa: BLE001 - one arm must not hide the other
            print(f"\n[ci] arm {arm} FAILED TO RUN: {err}", file=sys.stderr, flush=True)
            failed_arms.append(arm)

    if results:
        print("\n".join(scaling_table(results)), flush=True)
        ok = report_findings(gate(results, args.exists_scaling_floor))
    else:
        ok = False

    if failed_arms:
        print(
            f"\n[ci] arms that never produced a result: {', '.join(failed_arms)}",
            file=sys.stderr,
        )
    verdict = ok and not failed_arms
    print(f"\n[ci] {'PASS' if verdict else 'FAIL'}", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
